from __future__ import annotations

import mmap
import threading
from enum import Enum
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
    Sequence,
    Set,
    Sized,
    SupportsInt,
    Union,
    cast,
    overload,
)

import numpy as np

from ._util import AXIS, VoxelSize, get_reader, is_supported_file
from .structures import Attributes, ExpLoop, FrameMetadata, Metadata, XYPosLoop

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple

    import dask.array as da
    import xarray as xr
    from typing_extensions import Literal


Index = Union[int, slice]


class ReadMode(str, Enum):
    MMAP = "mmap"
    SDK = "sdk"


class ND2File:
    _memmap: mmap.mmap
    _is_legacy: bool

    def __init__(self, path: Union[Path, str]) -> None:
        self._path = str(path)
        self._rdr = get_reader(self._path)
        self._closed = False
        self._is_legacy = "Legacy" in type(self._rdr).__name__
        self._lock = threading.RLock()

    @staticmethod
    def is_supported_file(path) -> bool:
        return is_supported_file(path)

    @property
    def path(self):
        """Path of the image."""
        return self._path

    @property
    def is_legacy(self) -> bool:
        """Whether file is a legacy nd2 (JPEG2000) file."""
        return self._is_legacy

    def open(self) -> None:
        """open file for reading."""
        if self.closed:
            self._rdr.open()
            self._closed = False

    def close(self) -> None:
        """Close file (may cause segfault if read when closed in some cases)."""
        if not self.closed:
            self._rdr.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        """Whether the file is closed."""
        return self._closed

    def __enter__(self) -> ND2File:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_rdr"]
        del state["_lock"]
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self._lock = threading.RLock()
        self._rdr = get_reader(self._path)
        if self._closed:
            self._rdr.close()

    @cached_property
    def attributes(self) -> Attributes:
        """Core image attributes"""
        return self._rdr.attributes

    @cached_property
    def text_info(self) -> Dict[str, Any]:
        """Misc text info."""
        return self._rdr.text_info()

    @cached_property
    def experiment(self) -> List[ExpLoop]:
        """Loop information for each nd axis"""
        return self._rdr.experiment()

    @cached_property
    def metadata(self) -> Union[Metadata, dict]:
        """Various metadata (will be dict if legacy format)."""
        return self._rdr.metadata()

    @lru_cache(maxsize=1024)
    def frame_metadata(
        self, seq_index: Union[int, tuple]
    ) -> Union[FrameMetadata, dict]:
        """Metadata for specific frame.

        This includes the global metadata from the metadata function.
        (will be dict if legacy format).
        """
        idx = cast(
            int,
            self._seq_index_from_coords(seq_index)
            if isinstance(seq_index, tuple)
            else seq_index,
        )
        return self._rdr.frame_metadata(idx)

    @cached_property
    def custom_data(self) -> Dict[str, Any]:
        """Dict of various unstructured custom metadata."""
        return self._rdr._custom_data()

    @cached_property
    def ndim(self) -> int:
        """number of dimensions"""
        return len(self.shape)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        """size of each axis"""
        return self._coord_shape + self._frame_shape

    @cached_property
    def sizes(self) -> Dict[str, int]:
        """names and sizes for each axis"""
        attrs = self.attributes
        dims = {AXIS._MAP[c[1]]: c[2] for c in self._rdr._coord_info()}
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
        """Whether the image is rgb"""
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        """Number of components per channel (e.g. 3 for rgb)"""
        attrs = cast(Attributes, self.attributes)
        return attrs.componentCount // (attrs.channelCount or 1)

    @property
    def size(self) -> int:
        """Total number of pixels in the volume."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Total bytes of image data."""
        return self.size * self.dtype.itemsize

    @cached_property
    def dtype(self) -> np.dtype:
        """Image data type"""
        attrs = self.attributes
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{attrs.bitsPerComponentInMemory // 8}")

    def voxel_size(self, channel: int = 0) -> VoxelSize:
        """XYZ voxel size."""
        return VoxelSize(*self._rdr.voxel_size())

    def asarray(self, position: Optional[int] = None) -> np.ndarray:
        """Read image into numpy array.

        Parameters
        ----------
        position : int, optional
            A specific XY position to extract, by default (None) reads all.

        Returns
        -------
        np.ndarray

        Raises
        ------
        IndexError
            if position is provided and is out of range
        """
        final_shape = list(self.shape)
        if position is None:
            seqs: Sequence[int] = range(self._frame_count)
        else:
            if isinstance(position, str):
                try:
                    position = self._position_names().index(position)
                except ValueError as e:
                    raise ValueError(
                        f"{position!r} is not a valid position name"
                    ) from e
            try:
                pidx = list(self.sizes).index(AXIS.POSITION)
            except ValueError as exc:
                if position > 0:
                    raise IndexError(
                        f"Position {position} is out of range. "
                        f"Only 1 position available"
                    ) from exc
                seqs = range(self._frame_count)
            else:
                if position >= self.sizes[AXIS.POSITION]:
                    raise IndexError(
                        f"Position {position} is out of range. "
                        f"Only {self.sizes[AXIS.POSITION]} positions available"
                    )

                ranges: List[Union[range, tuple]] = [
                    range(x) for x in self._coord_shape
                ]
                ranges[pidx] = (position,)
                coords = list(zip(*product(*ranges)))
                seqs = self._seq_index_from_coords(coords)  # type: ignore
                final_shape[pidx] = 1

        arr: np.ndarray = np.stack([self._get_frame(i) for i in seqs])
        return arr.reshape(final_shape)

    def __array__(self) -> np.ndarray:
        """array protocol"""
        return self.asarray()

    def to_dask(self, wrapper=True, copy=True) -> da.Array:
        """Create dask array (delayed reader) representing image.

        This generally works well, but it remains to be seen whether performance
        is optimized, or if we're duplicating safety mechanisms. You may try
        various combinations of `wrapper` and `copy`, setting both to `False`
        will very likely cause segmentation faults in many cases.  But setting
        one of them to `False`, may slightly improve read speed in certain
        cases.

        Parameters
        ----------
        wrapper : bool
            If True (the default), the returned obect will be a thin subclass of
            a :class:`dask.array.Array` (an
            `ResourceBackedDaskArray`) that manages the opening
            and closing of this file when getting chunks via compute(). If `wrapper`
            is `False`, then a pure `da.Array` will be returned. However, when that
            array is computed, it will incur a file open/close on *every* chunk
            that is read (in the `_dask_block` method).  As such `wrapper`
            will generally be much faster, however, it *may* fail (i.e. result in
            segmentation faults) with certain dask schedulers.
        copy : bool
            If `True` (the default), the dask chunk-reading function will return
            an array copy. This can avoid segfaults in certain cases, though it
            may also add overhead.

        Returns
        -------
        da.Array
        """
        from dask.array import map_blocks

        chunks = [(1,) * x for x in self._coord_shape]
        chunks += [(x,) for x in self._frame_shape]
        dask_arr = map_blocks(
            self._dask_block,
            copy=copy,
            chunks=chunks,
            dtype=self.dtype,
        )
        if wrapper:
            from resource_backed_dask_array import ResourceBackedDaskArray

            # this subtype allows the dask array to re-open the underlying
            # nd2 file on compute.
            return ResourceBackedDaskArray.from_array(dask_arr, self)
        return dask_arr

    _NO_IDX = -1

    def _seq_index_from_coords(
        self, coords: Sequence
    ) -> Union[Sequence[int], SupportsInt]:
        if not self._coord_shape:
            return self._NO_IDX
        return np.ravel_multi_index(coords, self._coord_shape)

    def _dask_block(self, copy: bool, block_id: Tuple[int]) -> np.ndarray:
        if isinstance(block_id, np.ndarray):
            return
        with self._lock:
            was_closed = self.closed
            if self.closed:
                self.open()
            try:
                ncoords = len(self._coord_shape)
                idx = self._seq_index_from_coords(block_id[:ncoords])

                if idx == self._NO_IDX:
                    if any(block_id):
                        raise ValueError(
                            f"Cannot get chunk {block_id} for single frame image."
                        )
                    idx = 0
                data = self._get_frame(cast(int, idx))[(np.newaxis,) * ncoords]
                return data.copy() if copy else data
            finally:
                if was_closed:
                    self.close()

    def to_xarray(
        self,
        delayed: bool = True,
        squeeze: bool = True,
        position: Optional[int] = None,
        copy: bool = True,
    ) -> xr.DataArray:
        """Create labeled xarray representing image.

        Parameters
        ----------
        delayed : bool
            Whether the DataArray should be backed by dask array or numpy array,
            by default True (dask).
        squeeze : bool
            Whether to squeeze singleton dimensions, by default True
        position : int, optional
            A specific XY position to extract, by default (None) reads all.
        copy : bool
            Only applies when `delayed==True`.  See `to_dask` for details.

        Returns
        -------
        xr.DataArray
            xarray with all axes labeled.
        """
        import xarray as xr

        data = self.to_dask(copy=copy) if delayed else self.asarray(position)
        dims = list(self.sizes)
        coords = self._expand_coords(squeeze)
        if not squeeze:
            for missing_dim in set(coords).difference(dims):
                dims.insert(0, missing_dim)
            missing_axes = len(dims) - data.ndim
            if missing_axes > 0:
                data = data[(np.newaxis,) * missing_axes]

        if position is not None and not delayed and AXIS.POSITION in coords:
            # if it's delayed, we do this using isel below instead.
            coords[AXIS.POSITION] = [coords[AXIS.POSITION][position]]

        x = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs={
                "metadata": {
                    "metadata": self.metadata,
                    "experiment": self.experiment,
                    "attributes": self.attributes,
                    "text_info": self.text_info,
                }
            },
        )
        if delayed and position is not None and AXIS.POSITION in coords:
            x = x.isel({AXIS.POSITION: [position]})
        return x.squeeze() if squeeze else x

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
        return tuple(v for k, v in self.sizes.items() if k not in self._frame_coords)

    @property
    def _frame_count(self) -> int:
        return int(np.prod(self._coord_shape))

    def _get_frame(self, index: int) -> np.ndarray:
        frame = self._rdr._read_image(index)
        frame.shape = self._raw_frame_shape
        return frame.transpose((2, 0, 1, 3)).squeeze()

    def _expand_coords(self, squeeze=True) -> dict:
        """Return a dict that can be used as the coords argument to xr.DataArray"""
        dx, dy, dz = self.voxel_size()

        coords: Dict[str, Sized] = {
            AXIS.Y: np.arange(self.attributes.heightPx) * dy,
            AXIS.X: np.arange(self.attributes.widthPx) * dx,
            AXIS.CHANNEL: self._channel_names,
            AXIS.POSITION: ["XYPos:0"],  # maybe overwritten below
        }

        for c in self.experiment:
            if squeeze and getattr(c, "count") <= 1:
                continue
            if c.type == "ZStackLoop":
                coords[AXIS.Z] = np.arange(c.count) * c.parameters.stepUm
            elif c.type == "TimeLoop":
                coords[AXIS.TIME] = np.arange(c.count) * c.parameters.periodMs
            elif c.type == "NETimeLoop":
                pers = [np.arange(p.count) * p.periodMs for p in c.parameters.periods]
                coords[AXIS.TIME] = np.hstack(pers)
            elif c.type == "XYPosLoop":
                coords[AXIS._MAP["XYPosLoop"]] = self._position_names(c)

        if self.components_per_channel > 1:
            coords[AXIS.RGB] = ["Red", "Green", "Blue", "alpha"][
                : self.components_per_channel
            ]

        # fix for Z axis missing from experiment:
        if AXIS.Z in self.sizes and AXIS.Z not in coords:
            coords[AXIS.Z] = np.arange(self.sizes[AXIS.Z]) * dz

        if squeeze:
            return {k: v for k, v in coords.items() if len(v) > 1}
        return coords

    def _position_names(self, loop: Optional[XYPosLoop] = None) -> List[str]:
        if loop is None:
            for c in self.experiment:
                if c.type == "XYPosLoop":
                    loop = c
                    break
        if loop is None:
            return ["XYPos:0"]
        return [p.name or f"XYPos:{i}" for i, p in enumerate(loop.parameters.points)]

    @property
    def _channel_names(self) -> List[str]:
        return self._rdr.channel_names()

    def __repr__(self) -> str:
        try:
            details = " (closed)" if self.closed else f" {self.dtype}: {self.sizes!r}"
            extra = f": {Path(self.path).name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"


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
