from __future__ import annotations

import mmap
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from . import _nd2file
from ._chunkmap import read_chunkmap

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    from typing import Any, BinaryIO, Dict, List, Tuple

    import dask.array as da
    import xarray as xr

    from .structures import Attributes, ExpLoop, Metadata


Index = Union[int, slice]


class ReadMode(str, Enum):
    MMAP = "mmap"
    SDK = "sdk"


class AXIS:
    X = "X"
    Y = "Y"
    Z = "Z"
    CHANNEL = "C"
    RGB = "c"
    TIME = "T"
    POSITION = "S"
    UNKNOWN = "U"

    _MAP = {
        "Unknown": UNKNOWN,
        "TimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "NETimeLoop": TIME,
    }


class ND2File:
    _rdr: _nd2file.ND2Reader
    _fh: BinaryIO
    _memmap: mmap.mmap

    def __init__(
        self, path: Union[Path, str], read_mode: Union[str, ReadMode] = ReadMode.MMAP
    ) -> None:
        self._closed = True
        self._path = str(path)
        try:
            self._rdr = _nd2file.ND2Reader(self._path)
        except OSError:
            from ._util import is_old_format

            if is_old_format(path):
                raise NotImplementedError("Legacy file not yet supported")

        self.open()
        self._frame_map, self._meta_map = read_chunkmap(self._fh, fixup=True)
        self._safe_frames = self._frame_map["safe"]
        self.read_mode = read_mode  # type: ignore

    @property
    def path(self):
        return self._path

    @cached_property
    def ndim(self) -> int:
        return len(self.shape)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(c.size for c in self._coord_info) + self._final_frame_shape

    @cached_property
    def _coord_info(self):
        """simple info on each non-XYC coordinate"""
        return self._rdr.coord_info()

    @cached_property
    def _final_frame_shape(self) -> Tuple[int, ...]:
        """after reshaping, transposing, and squeezing a raw frame chunk

        [C]YX[c]  (where big C is a true channel, and little c is RGB channel.
        """
        attr = self.attributes
        _shape = [attr.heightPx, attr.widthPx or -1]
        if self._n_true_channels > 1:
            _shape.insert(0, self._n_true_channels)
        if self.components_per_channel > 1:
            _shape.append(self.components_per_channel)
        return tuple(_shape)

    @cached_property
    def is_rgb(self) -> bool:
        return self.components_per_channel in (3, 4)

    @cached_property
    def components_per_channel(self) -> int:
        return self.attributes.componentCount // self.metadata().contents.channelCount

    @cached_property
    def size(self) -> int:
        return np.prod(self.shape)

    @cached_property
    def dtype(self) -> np.dtype:
        attrs = self.attributes
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{attrs.bitsPerComponentInMemory // 8}")

    @cached_property
    def axes(self) -> str:
        a = [AXIS._MAP[c.type] for c in self._coord_info]
        if self._n_true_channels > 1:
            a.append(AXIS.CHANNEL)
        a += [AXIS.Y, AXIS.X]
        if self.components_per_channel > 1:
            a.append(AXIS.RGB)
        return "".join(a)

    @cached_property
    def _n_true_channels(self) -> int:
        return self.attributes.componentCount // self.components_per_channel

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

    @property
    def read_mode(self) -> ReadMode:
        return self._read_mode

    @read_mode.setter
    def read_mode(self, value: Union[str, ReadMode]) -> None:
        self._read_mode = ReadMode(value)

    def _get_frame(self, index, mode=None):
        mode = mode or self._read_mode  # TODO: protect this value
        get_frame = self._sdk_data if mode == "sdk" else self._mmap_chunk
        frame = get_frame(index)
        frame.shape = (
            self.attributes.heightPx,
            self.attributes.widthPx or -1,
            self._n_true_channels,
            self.components_per_channel,
        )
        return frame.transpose((2, 0, 1, 3)).squeeze()

    def _dask_block(self, block_id: Tuple[int]):
        if isinstance(block_id, np.ndarray):
            return
        idx = self._rdr.seq_index_from_coords(block_id[: -len(self._final_frame_shape)])
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
                    "metadata": self.metadata(),
                    "experiment": self.experiment,
                    "attributes": self.attributes,
                    "text_info": self.text_info,
                }
            },
        )

    def _expand_coords(self) -> dict:
        """Return a dict that can be used as the coords argument to xr.DataArray"""
        if self._rdr._is_legacy:
            return {}

        dx, dy, _ = self.voxel_size()
        coords = {
            AXIS.Y: np.arange(self.attributes.heightPx) * dy,
            AXIS.X: np.arange(self.attributes.widthPx) * dx,
        }
        for c in self.experiment:
            if c.type == "ZStackLoop":
                coords[AXIS._MAP["ZStackLoop"]] = (
                    np.arange(c.count) * c.parameters.stepUm
                )
            elif c.type == "TimeLoop":
                coords[AXIS._MAP["TimeLoop"]] = (
                    np.arange(c.count) * c.parameters.periodMs
                )
            elif c.type == "NETimeLoop":
                pers = [np.arange(p.count) * p.periodMs for p in c.parameters.periods]
                coords[AXIS._MAP["NETimeLoop"]] = np.hstack(pers)
            elif c.type == "XYPosLoop":
                coords[AXIS._MAP["XYPosLoop"]] = [
                    p.name or repr(tuple(p.stagePositionUm))
                    for p in c.parameters.points
                ]
        if self._n_true_channels > 1:
            _channels = self.metadata().channels
            coords[AXIS.CHANNEL] = [c.channel.name for c in _channels]
        if self.components_per_channel > 1:
            coords[AXIS.RGB] = ["Red", "Green", "Blue", "alpha"][
                : self.components_per_channel
            ]
        return coords

    def __repr__(self) -> str:
        try:
            shp_dict = dict(zip(self.axes, self.shape))
            details = f" {self.dtype}: {shp_dict!r}" if self.is_open() else " (closed)"
            extra = f": {Path(self.path).name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"

    def __enter__(self) -> ND2File:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def open(self) -> None:
        if not self.is_open():
            self._fh = open(self._path, "rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._rdr.open()
            self._closed = False

    def close(self) -> None:
        if self.is_open():
            self._fh.close()
            self._rdr.close()
            self._closed = True

    def is_open(self) -> bool:
        return not self._closed

    @cached_property
    def attributes(self) -> Attributes:
        return self._rdr.attributes()

    @property
    def text_info(self) -> Dict[str, Any]:
        return self._rdr.text_info()

    @cached_property
    def custom_data(self) -> Dict[str, Any]:
        from ._xml import parse_xml_block

        return {
            k[14:]: parse_xml_block(self._get_meta_chunk(k))
            for k, v in self._meta_map.items()
            if k.startswith("CustomDataVar|")
        }

    @cached_property
    def experiment(self) -> List[ExpLoop]:
        return self._rdr.experiment()

    def metadata(self, *args, **kwargs) -> Metadata:
        return self._rdr.metadata(*args, **kwargs)  # type: ignore  # FIXME

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
            buffer=self._mmap,
            offset=offset,
        )

        if self._array_stride != 0:
            bypc = a.bitsPerComponentInMemory // 8
            array_kwargs["strides"] = (
                self._array_stride + a.widthPx * bypc * a.componentCount,  # type:ignore
                a.componentCount * bypc,
                bypc,
            )
        return np.ndarray(**array_kwargs)

    @cached_property
    def _array_stride(self) -> int:
        a = self.attributes
        bypc = a.bitsPerComponentInMemory // 8
        return a.widthBytes - (bypc * a.widthPx * a.componentCount)  # type: ignore

    def _missing_frame(self, index: int = 0):
        a = self.attributes
        return np.ndarray((a.heightPx, a.widthPx, a.componentCount), self.dtype)

    def _get_meta_chunk(self, key):
        from ._chunkmap import read_chunk

        try:
            pos = self._meta_map[key]
        except KeyError:
            raise KeyError(
                f"No metdata chunk with key {key}. "
                f"Options include {set(self._meta_map)}"
            )
        return read_chunk(self._fh, pos)


def imread(file: str = None) -> np.ndarray:
    with ND2File(str(file)) as nd2:
        arr = nd2.asarray()
    return arr
