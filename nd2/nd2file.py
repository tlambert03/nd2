from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from . import _nd2file, _nd2file_legacy

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
        try:
            self._rdr = _nd2file.ND2Reader(str(path))
        except OSError:
            try:
                self._rdr = _nd2file_legacy.ND2Reader(str(path))
            except OSError:
                raise

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
        arr = np.stack([self._rdr.data(i) for i in range(self._rdr.seq_count())])
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
        return self._rdr.data(idx)[(np.newaxis,) * self._rdr.coord_size()]

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
        self._rdr.close()

    def _data(self, index: int = 0):
        return self._rdr.data(index)

    def open(self):
        return self._rdr.open()

    def close(self):
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


def imread(file: str = None) -> np.ndarray:
    with ND2File(str(file)) as nd2:
        arr = nd2.asarray()
    return arr
