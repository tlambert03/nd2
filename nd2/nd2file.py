from __future__ import annotations

import mmap
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Tuple, Union

from ._chunkmap import read_chunkmap

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore

import numpy as np

from . import _nd2file

if TYPE_CHECKING:
    from typing import List

    import dask.array as da
    import xarray as xr

    from .structures import Metadata

Index = Union[int, slice]
_AXMAP = {
    "Unknown": "U",
    "TimeLoop": "T",
    "XYPosLoop": "S",
    "ZStackLoop": "Z",
    "NETimeLoop": "T",
}


class ND2File:
    _rdr: _nd2file.ND2Reader
    _fh: BinaryIO
    _memmap: mmap.mmap

    def __init__(self, path) -> None:
        self._closed = True
        self._path = str(path)

        try:
            self._rdr = _nd2file.ND2Reader(str(path))
        except OSError:
            from ._util import is_old_format

            if is_old_format(path):
                raise NotImplementedError("Legacy file not yet supported")

        self.open()
        self._frame_map, self._meta_map = read_chunkmap(self._fh, fixup=True)
        if "safe" in self._frame_map:
            self._safe_frames = self._frame_map["safe"]
            self._frame_mode = "mmap"
        else:
            self._safe_frames = {}
            self._frame_mode = "sdk"

        a = self.attributes
        self._bypc = a.bitsPerComponentInMemory // 8
        self._stride = a.widthBytes - (self._bypc * a.widthPx * a.componentCount)

    def _get_frame(self, index, mode=None):
        mode = mode or self._frame_mode  # TODO: protect this value
        get_frame = self._sdk_data if mode == "sdk" else self._mmap_chunk
        frame = get_frame(index)
        frame.shape = (
            self.attributes.heightPx,
            self.attributes.widthPx or -1,
            self._n_true_channels,
            self._components_per_channel,
        )
        return frame.transpose((2, 0, 1, 3)).squeeze()

    @property
    def path(self):
        return self._path

    @cached_property
    def ndim(self) -> int:
        return len(self.shape)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple([c.size for c in self._coord_info] + self._final_frame_shape)

    @cached_property
    def _final_frame_shape(self) -> List[int]:
        """after reshaping, transposing, and squeezing a raw frame chunk"""
        attr = self.attributes
        _shape = [attr.heightPx, attr.widthPx or -1]
        if self._n_true_channels > 1:
            _shape.insert(0, self._n_true_channels)
        if self._components_per_channel > 1:
            _shape.append(self._components_per_channel)
        return _shape

    @cached_property
    def _ndim_frame(self):
        return len(self._final_frame_shape)

    @cached_property
    def is_rgb(self) -> bool:
        return self._components_per_channel >= 3

    @cached_property
    def _components_per_channel(self) -> int:
        return self.attributes.componentCount // self.metadata().contents.channelCount

    @cached_property
    def size(self) -> int:
        return np.prod(self.shape)

    @cached_property
    def dtype(self) -> np.dtype:
        attrs = self.attributes
        b = attrs.bitsPerComponentInMemory // 8
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{b}")  # TODO: check is it always uint?

    @cached_property
    def axes(self) -> str:
        a = [_AXMAP[c.type] for c in self._coord_info]
        if self._n_true_channels > 1:
            a.append("C")
        a += ["Y", "X"]
        if self._components_per_channel > 1:
            a.append("c")
        return "".join(a)

    @cached_property
    def _n_true_channels(self) -> int:
        return self.attributes.componentCount // self._components_per_channel

    def voxel_size(self, channel=0) -> Tuple[float, float, float]:
        return self._rdr._voxel_size()

    def asarray(self) -> np.ndarray:
        arr = np.stack([self._get_frame(i) for i in range(self._rdr.seq_count())])
        return arr.reshape(self.shape)

    def to_dask(self) -> da.Array:
        from dask.array import map_blocks

        chunkshape = [(1,) * c.size for c in self._coord_info]
        chunkshape += [(x,) for x in self._final_frame_shape]
        darr = map_blocks(self._dask_block, chunks=chunkshape, dtype=self.dtype)
        darr._ctx = self  # XXX: ok?  or will we leak refs
        return darr

    def _dask_block(self, block_id: Tuple[int]):
        if isinstance(block_id, np.ndarray):
            return
        idx = self._rdr.seq_index_from_coords(block_id[: -self._ndim_frame])
        if idx == -1:
            if any(block_id):
                raise ValueError(f"Cannot get chunk {block_id} for single frame image.")
            idx = 0
        return self._get_frame(idx)[(np.newaxis,) * len(self._coord_info)]

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
        """Return a dict that can be used as the coords argument to xr.DataArray"""
        if self._rdr._is_legacy:
            return {}
        attrs = self._rdr.attributes()
        dx, dy, _ = self._rdr._voxel_size()

        # TODO: make these coordinate labels an enum
        coords = {
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
        if self._n_true_channels > 1:
            _channels = self._rdr.metadata().channels  # type: ignore  # FIXME
            coords["C"] = [c.channel.name for c in _channels]
        if self._components_per_channel > 1:
            coords["c"] = ["Red", "Green", "Blue", "alpha"][
                : self._components_per_channel
            ]
        return coords

    def __repr__(self):
        details = (
            f" {self.dtype}: {self.shape!r}" if self._rdr.is_open() else " (closed)"
        )
        return f"<ND2File at {hex(id(self))}: {Path(self._rdr.path).name!r}{details}>"

    def __enter__(self) -> ND2File:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self):
        if not self.is_open():
            self._fh = open(self._path, "rb")
            self._memmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._rdr.open()
            self._closed = False

    def close(self):
        if self.is_open():
            self._memmap = None  # type: ignore
            self._fh.close()
            self._rdr.close()
            self._closed = True

    def is_open(self):
        return not self._closed

    @cached_property
    def attributes(self):
        return self._rdr.attributes()

    @cached_property
    def text_info(self):
        return self._rdr.text_info()

    @cached_property
    def experiment(self):
        return self._rdr.experiment()

    def metadata(self, *args, **kwargs) -> Metadata:
        return self._rdr.metadata(*args, **kwargs)  # type: ignore  # FIXME

    @cached_property
    def _coord_info(self):
        return self._rdr.coord_info()

    def _sdk_data(self, index: int) -> np.ndarray:
        """Read a chunk using the SDK"""
        return self._rdr.data(index)

    def _mmap_chunk(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK"""
        if index not in self._safe_frames:
            raise IndexError(f"Frame out of range: {index}")
        offset = self._safe_frames[index]
        if offset is None:
            return self._missing_frame(index)

        a = self.attributes
        array_kwargs = dict(
            shape=(a.heightPx, a.widthPx, a.componentCount),
            dtype=self.dtype,
            buffer=self._memmap,
            offset=offset,
        )

        if self._stride != 0:
            array_kwargs["strides"] = (
                self._stride + a.widthPx * self._bypc * a.componentCount,
                a.componentCount * self._bypc,
                self._bypc,
            )
        return np.ndarray(**array_kwargs)

    def _missing_frame(self, index: int = 0):
        a = self.attributes
        return np.ndarray((a.heightPx, a.widthPx, a.componentCount), self.dtype)

    @cached_property
    def custom_data(self):
        from ._xml import parse_xml_block

        return {
            k[14:]: parse_xml_block(self._get_meta_chunk(k))
            for k, v in self._meta_map.items()
            if k.startswith("CustomDataVar|")
        }

    def _get_meta_chunk(self, key):
        from ._chunkmap import read_chunk

        return read_chunk(self._fh, self._meta_map[key])


def imread(file: str = None) -> np.ndarray:
    with ND2File(str(file)) as nd2:
        arr = nd2.asarray()
    return arr
