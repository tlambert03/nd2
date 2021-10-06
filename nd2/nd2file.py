from __future__ import annotations

import mmap
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
    overload,
)

import numpy as np

from ._chunkmap import read_chunkmap
from ._util import AXIS, open_nd2
from .structures import Attributes, ExpLoop, Metadata, Volume, parse_experiment

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
    __read_frame: Callable[[ND2File, int], np.ndarray] = None
    __ravel_coords: Callable[[ND2File, Sequence[int]], int]

    def __init__(
        self, path: Union[Path, str], read_mode: Union[str, ReadMode] = ReadMode.MMAP
    ) -> None:
        self._closed = True
        self._path = str(path)

        self.open()
        self._frame_map, self._meta_map = read_chunkmap(
            self._fh, fixup=True, legacy=self._is_legacy
        )
        self._max_safe = max(self._frame_map["safe"])

        self._read_mode = ReadMode(read_mode)
        if self._read_mode == ReadMode.MMAP:
            self.__read_frame = self._image_from_mmap  # type: ignore[assignment]
            self.__ravel_coords = self._seq_index_from_coords  # type: ignore
        else:
            self.__read_frame = self._image_from_sdk  # type: ignore[assignment]
            self.__ravel_coords = self._rdr._seq_index_from_coords  # type: ignore
        if self._is_legacy:
            self.__read_frame = self._rdr._read_image

    # PUBLIC API:

    @property
    def path(self):
        return self._path

    def open(self) -> None:
        if self.closed:
            self._fh, self._rdr, self._is_legacy = open_nd2(self._path)
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._closed = False

    def close(self) -> None:
        if not self.closed:
            self._fh.close()
            self._rdr.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @cached_property
    def attributes(self) -> Attributes:
        if self._is_legacy:
            return self._rdr.attributes
        cont = self.metadata.contents
        attrs = self._rdr._attributes()
        nC = cont.channelCount if cont else attrs.get("componentCount", 1)
        return Attributes(**attrs, channelCount=nC)

    @cached_property
    def text_info(self) -> Dict[str, Any]:
        if self._is_legacy:
            return self._rdr.text_info
        return self._rdr._text_info()

    @cached_property
    def experiment(self) -> List[ExpLoop]:
        if self._is_legacy:
            return self._rdr.experiment
        return parse_experiment(self._rdr._experiment())

    @cached_property
    def metadata(self) -> Metadata:
        if self._is_legacy:
            return self._rdr.metadata
        return Metadata(**self._rdr._metadata())

    @cached_property
    def ndim(self) -> int:
        return len(self.shape)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self._coord_shape + self._frame_shape

    @cached_property
    def sizes(self) -> Dict[str, int]:
        attrs = cast(Attributes, self.attributes)
        from ._util import AXIS, dims_from_description

        # often, the 'Description' field in textinfo is the best source of dimension
        # (dims are strangely missing from coord_info sometimes)
        # so we start there, and fall back to coord_info if ddims are empty
        ddims = dims_from_description(self.text_info.get("description"))
        # dims from coord info
        cdims = {AXIS._MAP[c[1]]: c[2] for c in self._rdr._coord_info()}

        # prefer the value in coord info if it exists. (it's usually more accurate)
        dims = {k: cdims.get(k, v) for k, v in ddims.items()} if ddims else cdims
        dims[AXIS.CHANNEL] = (
            dims.pop(AXIS.CHANNEL)
            if AXIS.CHANNEL in dims
            else (attrs.channelCount or 1)
        )
        dims[AXIS.Y] = attrs.heightPx
        dims[AXIS.X] = attrs.widthPx or -1
        if self.components_per_channel == 3:  # rgb
            dims[AXIS.RGB] = self.components_per_channel
        else:
            # if not exactly 3 channels, throw them all into monochrome channels
            dims[AXIS.CHANNEL] = attrs.componentCount
        return {k: v for k, v in dims.items() if v != 1}

    @property
    def is_rgb(self) -> bool:
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        attrs = cast(Attributes, self.attributes)
        return attrs.componentCount // (attrs.channelCount or 1)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @cached_property
    def dtype(self) -> np.dtype:
        attrs = self.attributes
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{attrs.bitsPerComponentInMemory // 8}")

    def voxel_size(self, channel: int = 0) -> Tuple[float, float, float]:
        if self._is_legacy:
            return self._rdr.voxel_size()
        meta = self.metadata
        if meta:
            ch = meta.channels
            if ch:
                vol = cast(Volume, ch[0].volume)
                return vol.axesCalibration
        return (1, 1, 1)

    # ARRAY OUTPUT

    def asarray(self) -> np.ndarray:
        arr = np.stack([self._get_frame(i) for i in range(self._frame_count)])
        return arr.reshape(self.shape)

    def to_dask(self) -> da.Array:
        from dask.array import map_blocks

        chunks = [(1,) * x for x in self._coord_shape]
        chunks += [(x,) for x in self._frame_shape]
        darr = map_blocks(self._dask_block, chunks=chunks, dtype=self.dtype)
        darr._ctx = self  # XXX: ok?  or will we leak refs
        return darr

    NO_IDX = -1

    def _seq_index_from_coords(self, coords: Sequence) -> int:
        if not self._coord_shape:
            return self.NO_IDX
        return np.ravel_multi_index(coords, self._coord_shape)

    def _dask_block(self, block_id: Tuple[int]) -> np.ndarray:
        if isinstance(block_id, np.ndarray):
            return

        ncoords = len(self._coord_shape)
        idx = self.__ravel_coords(block_id[:ncoords])

        if idx == self.NO_IDX:
            if any(block_id):
                raise ValueError(f"Cannot get chunk {block_id} for single frame image.")
            idx = 0
        return self._get_frame(idx)[(np.newaxis,) * ncoords]

    def to_xarray(self, delayed: bool = True) -> xr.DataArray:
        import xarray as xr

        return xr.DataArray(
            self.to_dask() if delayed else self.asarray(),
            dims=list(self.sizes),
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

    @property
    def _frame_coords(self) -> Set[str]:
        return {AXIS.X, AXIS.Y, AXIS.CHANNEL, AXIS.RGB}

    @property
    def _raw_frame_shape(self) -> Tuple[int, int, int, int]:
        """sizes of each frame coordinate, prior to reshape"""
        attr = self.attributes
        return (
            attr.heightPx,
            attr.widthPx or -1,
            attr.channelCount or 1,
            self.components_per_channel,
        )

    @property
    def _frame_shape(self) -> Tuple[int, ...]:
        """sizes of each frame coordinate, after reshape & squeeze"""
        return tuple(v for k, v in self.sizes.items() if k in self._frame_coords)

    @cached_property
    def _coord_shape(self) -> Tuple[int, ...]:
        """sizes of each *non-frame* coordinate"""
        if self.read_mode == ReadMode.SDK:
            return tuple(v[2] for v in self._rdr._coord_info())
        else:
            return tuple(
                v for k, v in self.sizes.items() if k not in self._frame_coords
            )

    @property
    def _frame_count(self) -> int:
        return int(np.prod(self._coord_shape))

    def _get_frame(self, index):
        frame: np.ndarray = self.__read_frame(index)
        frame.shape = self._raw_frame_shape
        return frame.transpose((2, 0, 1, 3)).squeeze()

    def _expand_coords(self) -> dict:
        """Return a dict that can be used as the coords argument to xr.DataArray"""
        dx, dy, dz = self.voxel_size()

        coords = {
            AXIS.Y: np.arange(self.attributes.heightPx) * dy,
            AXIS.X: np.arange(self.attributes.widthPx) * dx,
        }
        for c in self.experiment:
            if getattr(c, "count") == 1:  # squeeze
                continue
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
        if self.attributes.channelCount and self.attributes.channelCount > 1:
            if self._is_legacy:
                coords[AXIS.CHANNEL] = self._rdr.channel_names()
            else:
                _channels = self.metadata.channels or []
                coords[AXIS.CHANNEL] = [c.channel.name for c in _channels]
        if self.components_per_channel > 1:
            coords[AXIS.RGB] = ["Red", "Green", "Blue", "alpha"][
                : self.components_per_channel
            ]

        # fix for Z axis missing from experiment:
        if AXIS.Z in self.sizes and AXIS.Z not in coords:
            coords[AXIS.Z] = np.arange(self.sizes[AXIS.Z]) * dz

        return coords

    def __repr__(self) -> str:
        try:
            details = " (closed)" if self.closed else f" {self.dtype}: {self.sizes!r}"
            extra = f": {Path(self.path).name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"

    def __enter__(self) -> ND2File:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def read_mode(self) -> ReadMode:
        return self._read_mode

    def _image_from_sdk(self, index: int) -> np.ndarray:
        """Read a chunk using the SDK"""
        return self._rdr._image(index)

    def _image_from_mmap(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK"""
        if index > self._max_safe:
            raise IndexError(f"Frame out of range: {index}")
        offset = self._frame_map["safe"].get(index, None)
        if offset is None:
            return self._missing_frame(index)

        try:
            return np.ndarray(
                shape=self._raw_frame_shape,
                dtype=self.dtype,
                buffer=self._mmap,  # type: ignore
                offset=offset,
                strides=self._strides,
            )
        except TypeError:
            # If the chunkmap is wrong, and the mmap isn't long enough
            # for the requested offset & size, a type error is raised.
            return self._missing_frame(index)

    @cached_property
    def _strides(self) -> Optional[Tuple[int, ...]]:
        a = cast(Attributes, self.attributes)
        width = a.widthPx
        widthB = a.widthBytes
        if not (width and widthB):
            return None
        bypc = a.bitsPerComponentInMemory // 8
        array_stride = widthB - (bypc * width * a.componentCount)
        if array_stride == 0:
            return None
        return (
            array_stride + width * bypc * a.componentCount,
            a.componentCount * bypc,
            self.components_per_channel * bypc,
            bypc,
        )

    def _missing_frame(self, index: int = 0) -> np.ndarray:
        # TODO: add other modes for filling missing data
        return np.zeros(self._raw_frame_shape, self.dtype)

    @cached_property
    def custom_data(self) -> Dict[str, Any]:
        from ._xml import parse_xml_block

        return {
            k[14:]: parse_xml_block(self._get_meta_chunk(k))
            for k, v in self._meta_map.items()
            if k.startswith("CustomDataVar|")
        }

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
def imread(
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
