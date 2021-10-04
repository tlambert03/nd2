from __future__ import annotations

import mmap
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Union, overload

import numpy as np

from ._chunkmap import read_chunkmap
from ._util import AXIS, open_nd2
from .structures import Attributes, Coordinate, ExpLoop, Metadata

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    from typing import Any, BinaryIO, Dict, List, Tuple

    import dask.array as da
    import xarray as xr
    from typing_extensions import Literal


Index = Union[int, slice]


class ReadMode(str, Enum):
    MMAP = "mmap"
    SDK = "sdk"


class ND2File:
    _fh: BinaryIO
    _memmap: mmap.mmap

    def __init__(
        self, path: Union[Path, str], read_mode: Union[str, ReadMode] = ReadMode.MMAP
    ) -> None:
        self._closed = True
        self._path = str(path)

        self.open()
        self._frame_map, self._meta_map = read_chunkmap(self._fh, fixup=True)
        self._safe_frames = self._frame_map["safe"]
        self._max_safe = max(self._safe_frames)
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
        return [Coordinate(*c) for c in self._rdr._coord_info()]

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
        return self.attributes.componentCount // self.metadata.contents.channelCount

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
        meta = self.metadata
        if meta:
            ch = meta.channels
            if ch:
                vol = ch[0].volume
                return tuple(vol.axesCalibration)
        return (None, None, None)

    def asarray(self) -> np.ndarray:
        arr = np.stack([self._get_frame(i) for i in range(self._seq_count())])
        return arr.reshape(self.shape)

    def _seq_count(self) -> int:
        return self._rdr._seq_count()

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
        get_frame = self._image_from_sdk if mode == "sdk" else self._image_from_mmap
        frame = get_frame(index)
        frame.shape = (
            self.attributes.heightPx,
            self.attributes.widthPx or -1,
            self._n_true_channels,
            self.components_per_channel,
        )
        return frame.transpose((2, 0, 1, 3)).squeeze()

    def _dask_block(self, block_id: Tuple[int]) -> np.ndarray:
        if isinstance(block_id, np.ndarray):
            return
        idx = self._rdr._seq_index_from_coords(
            block_id[: -len(self._final_frame_shape)]
        )
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
                    "metadata": self.metadata,
                    "experiment": self.experiment,
                    "attributes": self.attributes,
                    "text_info": self.text_info,
                }
            },
        )

    def _expand_coords(self) -> dict:
        """Return a dict that can be used as the coords argument to xr.DataArray"""
        # if self._rdr._is_legacy:
        #     return {}

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
            _channels = self.metadata.channels
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
            self._fh, self._rdr = open_nd2(self._path)
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
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
        return Attributes(**self._rdr._attributes())

    @property
    def text_info(self) -> Dict[str, Any]:
        return self._rdr._text_info()

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
        from .structures import parse_experiment

        return parse_experiment(self._rdr._experiment())

    @cached_property
    def metadata(self) -> Metadata:
        return Metadata(**self._rdr._metadata())

    def _image_from_sdk(self, index: int) -> np.ndarray:
        """Read a chunk using the SDK"""
        return self._rdr._image(index)

    def _image_from_mmap(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK"""
        if index > self._max_safe:
            raise IndexError(f"Frame out of range: {index}")
        offset = self._safe_frames.get(index, None)
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
        try:
            return np.ndarray(**array_kwargs)  # type: ignore
        except TypeError:
            # If the chunkmap is wrong, and the mmap isn't long enough
            # for the requested offset & size, a type error is raised.
            return self._missing_frame(index)

    @cached_property
    def _array_stride(self) -> int:
        a = self.attributes
        bypc = a.bitsPerComponentInMemory // 8
        return a.widthBytes - (bypc * a.widthPx * a.componentCount)  # type: ignore

    def _missing_frame(self, index: int = 0) -> np.ndarray:
        # TODO: add other modes for filling missing data
        a = self.attributes
        return np.ndarray((a.heightPx, a.widthPx, a.componentCount), self.dtype)

    def _get_meta_chunk(self, key: str) -> bytes:
        from ._chunkmap import read_chunk

        try:
            pos = self._meta_map[key]
        except KeyError:
            raise KeyError(
                f"No metdata chunk with key {key}. "
                f"Options include {set(self._meta_map)}"
            )
        return read_chunk(self._fh, pos)


@overload
def imread(  # type: ignore
    file: str, dask: Literal[False] = False, xarray: Literal[False] = False
) -> np.ndarray:
    ...


@overload
def imread(file: str, dask: bool = ..., xarray: Literal[True] = True) -> xr.DataArray:
    ...


@overload
def imread(file: str, dask: Literal[True] = ..., xarray=False) -> da.Array:
    ...


def imread(file: str, dask: bool = False, xarray: bool = False):
    with ND2File(file) as nd2:
        if xarray:
            return nd2.to_xarray(delayed=dask)
        elif dask:
            return nd2.to_dask()
        else:
            return nd2.asarray()
