from __future__ import annotations

import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np

from . import _nd2file

if TYPE_CHECKING:
    import dask.array as da
    import xarray as xr

Index = Union[int, slice]
_AXMAP = {
    "Unknown": "U",
    "TimeLoop": "T",
    "XYPosLoop": "S",
    "ZStackLoop": "Z",
    "NETimeLoop": "T",
}

if TYPE_CHECKING:
    from typing import List, Sequence

    from typing_extensions import Protocol

    from .structures import Attributes, Coordinate, ExpLoop, Metadata

    class _ND2Reader(Protocol):
        path: str
        _is_legacy: bool

        def __init__(self, path: str) -> None:
            ...

        def open(self) -> None:
            ...

        def close(self) -> None:
            ...

        def is_open(self) -> bool:
            ...

        # def __enter__(self) -> _ND2Reader: ...
        # def __exit__(self, *args: Any) -> bool: ...
        def attributes(self) -> Attributes:
            ...

        def _voxel_size(self) -> Tuple[float, float, float]:
            ...

        def data(self, seq_index: int = 0) -> np.ndarray:
            ...

        def seq_count(self) -> int:
            ...

        def coord_info(self) -> List[Coordinate]:
            ...

        def coord_size(self) -> int:
            ...  # == len(coord_info())

        def seq_index_from_coords(self, coords: Sequence[int]) -> int:
            ...

        def text_info(self) -> dict:
            ...

        # needed for _expand_coords
        def experiment(self) -> List[ExpLoop]:
            ...

        def metadata(self) -> Union[dict, Metadata]:
            ...

        # optional
        def coords_from_seq_index(self, seq_index: int) -> Tuple[int, ...]:
            ...


class ND2File:
    _rdr: _ND2Reader

    def __init__(self, path) -> None:
        from ._chunkmap import read_chunkmap

        try:
            self._chunkmap = read_chunkmap(path)
            self._frame_chunks: Dict[int, Tuple[int, int]] = self._chunkmap.get(
                "images", {}
            )
            self._bad_frames = self._chunkmap.get("bad_frames", set())
            self._safe_frames = self._chunkmap.get("safe_frames", dict())
        except AssertionError:
            self._frame_chunks = {}
            self._bad_frames = set()
        try:
            self._rdr = _nd2file.ND2Reader(str(path))
        except OSError:
            from ._util import is_old_format

            if is_old_format(path):
                raise NotImplementedError("Legacy file not yet supported")
            # try:
            #     from . import _nd2file_legacy

            #     self._rdr = _nd2file_legacy.ND2Reader(str(path))
            # except OSError:
            #     raise
        self._fh = open(path, "rb")
        self.mem = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def ndim(self) -> int:
        # TODO: this depends on whether we squeeze or not
        # XXX: also... this needs to agree with shape and axes
        # return self.coord_size() + 2 + int(self.attributes().componentCount > 1)
        return self._rdr.coord_size() + 3

    @property
    def shape(self) -> Tuple[int, ...]:
        attr = self._rdr.attributes()
        _shape = [c.size for c in self._rdr.coord_info()]
        # TODO: widthPx may be None?
        _shape += [attr.componentCount, attr.heightPx, attr.widthPx or -1]
        return tuple(_shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def dtype(self) -> np.dtype:
        attrs = self._rdr.attributes()
        b = attrs.bitsPerComponentInMemory // 8
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{b}")  # TODO: check is it always uint?

    @property
    def axes(self) -> str:
        a = [_AXMAP[c.type] for c in self._rdr.coord_info()] + list("CYX")
        return "".join(a)

    def voxel_size(self, channel=0) -> Tuple[float, float, float]:
        return self._rdr._voxel_size()

    def asarray(self) -> np.ndarray:
        arr = np.stack([self._data(i) for i in range(self._rdr.seq_count())])
        return arr.reshape(self.shape)

    def to_dask(self) -> da.Array:
        from dask.array import map_blocks

        *rest, nc, ny, nx = self.shape
        darr = map_blocks(
            self._get_chunk,
            chunks=[(1,) * i for i in rest] + [(nc,), (ny,), (nx,)],
            dtype=self.dtype,
        )
        darr._ctx = self  # XXX: ok?  or will we leak refs
        return darr

    def _get_chunk(self, block_id):
        # primarily for to_dask
        if isinstance(block_id, np.ndarray):
            return
        idx = self._rdr.seq_index_from_coords(block_id[:-3])
        if idx == -1:
            if any(block_id):
                raise ValueError(f"Cannot get chunk {block_id} for single frame image.")
            idx = 0
        return self._data(idx)[(np.newaxis,) * self._rdr.coord_size()]

    def to_xarray(self, delayed: bool = True) -> xr.DataArray:
        import xarray as xr

        return xr.DataArray(
            self.to_dask() if delayed else self.asarray(),
            dims=list(self.axes),
            coords=self._expand_coords(),
            attrs={
                "metadata": {
                    "metadata": self._rdr.metadata(),
                    "experiment": self._rdr.experiment(),
                    "attributes": self._rdr.attributes(),
                    "text_info": self._rdr.text_info(),
                }
            },
        )

    def _expand_coords(self) -> dict:
        if self._rdr._is_legacy:
            return {}
        attrs = self._rdr.attributes()
        _channels = self._rdr.metadata().channels  # type: ignore  # FIXME
        dx, dy, dz = self._rdr._voxel_size()
        coords = {
            "C": [c.channel.name for c in _channels],
            "Y": np.arange(attrs.heightPx) * dy,
            "X": np.arange(attrs.widthPx) * dx,
        }
        for c in self._rdr.experiment():
            if c.type == "ZStackLoop":
                coords[_AXMAP["ZStackLoop"]] = np.arange(c.count) * c.parameters.stepUm
            elif c.type == "TimeLoop":
                coords[_AXMAP["TimeLoop"]] = np.arange(c.count) * c.parameters.periodMs
            elif c.type == "NETimeLoop":
                pers = [np.arange(p.count) * p.periodMs for p in c.parameters.periods]
                coords[_AXMAP["NETimeLoop"]] = np.hstack(pers)
            elif c.type == "XYPosLoop":
                coords[_AXMAP["XYPosLoop"]] = [
                    p.name or repr(tuple(p.stagePositionUm))
                    for p in c.parameters.points
                ]
        return coords

    def __repr__(self):
        details = (
            f" {self.dtype}: {self.shape!r}" if self._rdr.is_open() else " (closed)"
        )
        return f"<ND2File at {hex(id(self))}: {Path(self._rdr.path).name!r}{details}>"

    def __enter__(self) -> ND2File:
        self._rdr.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _data(self, index: int = 0):
        # this seems to be much faster than using the SDK to read
        return self._read_chunk(index)
        # if index in self._bad_frames:
        #     return self._empty_frame()
        # elif index in self._chunkmap.get("fixed_frames", set()):
        #     return self._read_chunk(index)
        # return self._rdr.data(index)

    def open(self):
        return self._rdr.open()

    def close(self):
        self._fh.close()
        self._rdr.close()

    def is_open(self):
        return self._rdr.is_open()

    def attributes(self):
        return self._rdr.attributes()

    def text_info(self):
        return self._rdr.text_info()

    def experiment(self):
        return self._rdr.experiment()

    def metadata(self, *args, **kwargs):
        return self._rdr.metadata(*args, **kwargs)  # type: ignore  # FIXME

    def _coord_info(self):
        return self._rdr.coord_info()

    @property
    def path(self) -> str:
        return self._rdr.path

    def _read_chunk(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK"""
        if index not in self._safe_frames:
            raise IndexError(f"Frame out of range: {index}")
        offset = self._safe_frames[index]
        if offset is None:
            return self._empty_frame()
        a = self.attributes()
        array_kwargs = dict(
            shape=(a.heightPx, a.widthPx, a.componentCount),
            dtype=self.dtype,
            buffer=self.mem,
            offset=offset,
        )

        bypc = a.bitsPerComponentInMemory // 8
        stride = a.widthBytes - (bypc * a.widthPx * a.componentCount)
        if stride != 0:
            array_kwargs["strides"] = (
                stride + a.widthPx * bypc * a.componentCount,
                a.componentCount * bypc,
                bypc,
            )
        return np.ndarray(**array_kwargs).transpose((2, 0, 1))

    def _empty_frame(self):
        a = self.attributes()
        return np.ndarray((a.componentCount, a.heightPx, a.widthPx), self.dtype)


def imread(file: str = None) -> np.ndarray:
    with ND2File(str(file)) as nd2:
        arr = nd2.asarray()
    return arr
