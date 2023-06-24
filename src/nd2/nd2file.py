from __future__ import annotations

import threading
import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from nd2 import _util

from ._pysdk._chunk_decode import ND2_FILE_SIGNATURE, get_version
from ._util import AXIS, TIME_KEY, is_supported_file
from .structures import ROI

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    import mmap
    from typing import Any, Sequence, Sized, SupportsInt

    import dask.array.core
    import xarray as xr
    from typing_extensions import Literal

    from ._binary import BinaryLayers
    from ._pysdk._pysdk import ND2Reader as LatestSDKReader
    from ._util import DictOfDicts, DictOfLists, ListOfDicts, StrOrBytesPath
    from .structures import (
        Attributes,
        ExpLoop,
        FrameMetadata,
        Metadata,
        TextInfo,
        XYPosLoop,
    )

__all__ = ["ND2File", "imread"]


class ND2File:
    """Main objecting for opening and extracting data from an nd2 file.

    Parameters
    ----------
    path : Path | str
        Filename of an nd2 file.
    validate_frames : bool
        Whether to verify (and attempt to fix) frames whose positions have been
        shifted relative to the predicted offset (i.e. in a corrupted file).
        This comes at a slight performance penalty at file open, but may "rescue"
        some corrupt files. by default False.
    search_window : int
        When validate_frames is true, this is the search window (in KB) that will
        be used to try to find the actual chunk position. by default 100 KB
    read_using_sdk : Optional[bool]
        DEPRECATED.  No longer does anything.
        If `True`, use the SDK to read the file. If `False`, inspects the chunkmap
        and reads from a `numpy.memmap`. If `None` (the default), uses the SDK if
        the file is compressed, otherwise uses the memmap. Note: using
        `read_using_sdk=False` on a compressed file will result in a ValueError.
    """

    _memmap: mmap.mmap
    _is_legacy: bool

    def __init__(
        self,
        path: Path | str,
        *,
        validate_frames: bool = False,
        search_window: int = 100,
        read_using_sdk: bool | None = None,
    ) -> None:
        if read_using_sdk is not None:
            warnings.warn(
                "The `read_using_sdk` argument is deprecated and will be removed in "
                "a future version.",
                FutureWarning,
                stacklevel=2,
            )
        self._path = str(path)
        self._rdr = _util.get_reader(
            self._path,
            validate_frames=validate_frames,
            search_window=search_window,
        )
        self._closed = False
        self._is_legacy = "Legacy" in type(self._rdr).__name__
        self._lock = threading.RLock()
        self._version: tuple[int, ...] | None = None

    @staticmethod
    def is_supported_file(path: StrOrBytesPath) -> bool:
        """Return True if the file is supported by this reader."""
        return is_supported_file(path)

    @property
    def version(self) -> tuple[int, ...]:
        """Return the file format version as a tuple of ints."""
        if self._version is None:
            try:
                self._version = get_version(self._rdr._fh or self._rdr._path)
            except Exception:
                self._version = (-1, -1)
                raise
        return self._version

    @property
    def path(self) -> str:
        """Path of the image."""
        return self._path

    @property
    def is_legacy(self) -> bool:
        """Whether file is a legacy nd2 (JPEG2000) file."""
        return self._is_legacy

    def open(self) -> None:
        """Open file for reading."""
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
        """Open file for reading."""
        self.open()
        return self

    def __del__(self) -> None:
        """Delete file handle on garbage collection."""
        if not getattr(self, "_closed", True):
            warnings.warn(
                "ND2File file not closed before garbage collection. "
                "Please use `with ND2File(...):` context or call `.close()`.",
                stacklevel=2,
            )
            self._rdr.close()

    def __exit__(self, *_: Any) -> None:
        """Exit context manager and close file."""
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Return state for pickling."""
        state = self.__dict__.copy()
        del state["_rdr"]
        del state["_lock"]
        return state

    def __setstate__(self, d: dict[str, Any]) -> None:
        """Load state from pickling."""
        self.__dict__ = d
        self._lock = threading.RLock()
        self._rdr = _util.get_reader(self._path)
        if self._closed:
            self._rdr.close()

    @cached_property
    def attributes(self) -> Attributes:
        """Core image attributes."""
        return self._rdr.attributes

    @cached_property
    def text_info(self) -> TextInfo | dict:
        """Misc text info."""
        return self._rdr.text_info()

    @cached_property
    def rois(self) -> dict[int, ROI]:
        """Return dict of {id: ROI} for all ROIs found in the metadata."""
        key = b"CustomData|RoiMetadata_v1!"
        if self.is_legacy or key not in self._rdr.chunkmap:  # type: ignore
            return {}  # pragma: no cover

        data = cast("LatestSDKReader", self._rdr)._decode_chunk(key)
        data = data.get("RoiMetadata_v1", {}).copy()
        data.pop("Global_Size", None)
        try:
            _rois = [ROI._from_meta_dict(d) for d in data.values()]
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Could not parse ROI metadata: {e}") from e
        return {r.id: r for r in _rois}

    @cached_property
    def experiment(self) -> list[ExpLoop]:
        """Loop information for each nd axis."""
        return self._rdr.experiment()

    @overload
    def events(
        self, *, orient: Literal["records"] = ..., null_value: Any = ...
    ) -> ListOfDicts:
        ...

    @overload
    def events(self, *, orient: Literal["list"], null_value: Any = ...) -> DictOfLists:
        ...

    @overload
    def events(self, *, orient: Literal["dict"], null_value: Any = ...) -> DictOfDicts:
        ...

    def events(
        self,
        *,
        orient: Literal["records", "list", "dict"] = "records",
        null_value: Any = float("nan"),
    ) -> ListOfDicts | DictOfLists | DictOfDicts:
        """Return tabular data recorded for each frame and/or event of the experiment.

        This method returns tabular data in the format specified by the `orient`
        argument:
            - 'records' : list of dict - `[{column -> value}, ...]` (default)
            - 'dict' :    dict of dict - `{column -> {index -> value}, ...}`
            - 'list' :    dict of list - `{column -> [value, ...]}`

        All return types are passable to pd.DataFrame(). It matches the tabular data
        reported in the Image Properties > Recorded Data tab of the NIS Viewer.

        There will be a column for each tag in the `CustomDataV2_0` section of
        `ND2File.custom_data`, as well columns for any events recorded in the
        data.  Not all cells will be populated, and empty cells will be filled
        with `null_value` (default `float('nan')`).

        Legacy ND2 files are not supported.

        Parameters
        ----------
        orient : {'records', 'dict', 'list'}, default 'records'
            The format of the returned data. See `pandas.DataFrame
                - 'records' : list of dict - `[{column -> value}, ...]` (default)
                - 'dict' :    dict of dict - `{column -> {index -> value}, ...}`
                - 'list' :    dict of list - `{column -> [value, ...]}`
        null_value : Any, default float('nan')
            The value to use for missing data.
        """
        if orient not in ("records", "dict", "list"):  # pragma: no cover
            raise ValueError("orient must be one of 'records', 'dict', or 'list'")

        if self.is_legacy:  # pragma: no cover
            warnings.warn(
                "`recorded_data` is not implemented for legacy ND2 files",
                UserWarning,
                stacklevel=2,
            )
            return [] if orient == "records" else {}  # type: ignore[return-value]

        rdr = cast("LatestSDKReader", self._rdr)
        acq_data = rdr._acquisition_data()  # comes back as a dict of lists
        acq_data.update(rdr._custom_tags())

        img_events = rdr._img_exp_events()
        if not img_events and orient == "list":
            # by default, acq_data is already oriented as a dict of lists,
            # so if we don't have any image events, we can just return it
            return acq_data

        # re-orient acq_data as a list of dicts, to combine with events
        records = _util.convert_dict_of_lists_to_records(acq_data)
        for e in img_events:
            records.append({TIME_KEY: e.time / 1000, "Events": e.description})

        # sort by time
        records.sort(key=lambda x: x.get(TIME_KEY, 0))

        if orient == "dict":
            return _util.convert_records_to_dict_of_dicts(records, null_val=null_value)
        elif orient == "list":
            return _util.convert_records_to_dict_of_lists(records, null_val=null_value)
        return records

    def unstructured_metadata(
        self,
        *,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        unnest: bool | None = None,
    ) -> dict[str, Any]:
        """Exposes, and attempts to decode, each metadata chunk in the file.

        This is provided as a *experimental* fallback in the event that
        `ND2File.experiment` does not contain all of the information you need. No
        attempt is made to parse or validate the metadata, and the format of various
        sections, *may* change in future versions of nd2. Consumption of this metadata
        should use appropriate exception handling!

        The 'ImageMetadataLV' chunk is the most likely to contain useful information,
        but if you're generally looking for "hidden" metadata, it may be helpful to
        look at the full output.

        Parameters
        ----------
        strip_prefix : bool, optional
            Whether to strip the type information from the front of the keys in the
            dict. For example, if `True`: `uiModeFQ` becomes `ModeFQ` and `bUsePFS`
            becomes `UsePFS`, etc... by default `True`
        include : Optional[Set[str]], optional
            If provided, only include the specified keys in the output. by default,
            all metadata sections found in the file are included.
        exclude : Optional[Set[str]], optional
            If provided, exclude the specified keys from the output. by default `None`
        unnest : bool, optional
            DEPRECATED.  No longer does anything.

        Returns
        -------
        Dict[str, Any]
            A dict of the unstructured metadata, with keys that are the type of the
            metadata chunk (things like 'CustomData|RoiMetadata_v1' or
            'ImageMetadataLV'), and values that are associated metadata chunk.
        """
        if self.is_legacy:  # pragma: no cover
            raise NotImplementedError(
                "unstructured_metadata not available for legacy files"
            )

        if unnest is not None:
            warnings.warn(
                "The unnest parameter is deprecated, and no longer has any effect.",
                FutureWarning,
                stacklevel=2,
            )

        rdr = cast("LatestSDKReader", self._rdr)
        keys = {
            k.decode()[:-1]
            for k in rdr.chunkmap
            if not k.startswith((b"ImageDataSeq", b"CustomData", ND2_FILE_SIGNATURE))
        }

        if include:
            _keys: set[str] = set()
            for i in include:
                if i not in keys:
                    warnings.warn(f"Key {i!r} not found in metadata", stacklevel=2)
                else:
                    _keys.add(i)
            keys = _keys
        if exclude:
            keys = {k for k in keys if k not in exclude}

        output: dict[str, Any] = {}
        for key in sorted(keys):
            name = f"{key}!".encode()
            try:
                output[key] = rdr._decode_chunk(name, strip_prefix=strip_prefix)
            except Exception:  # pragma: no cover
                output[key] = rdr._load_chunk(name)
        return output

    @cached_property
    def metadata(self) -> Metadata | dict:
        """Various metadata (will be dict if legacy format)."""
        return self._rdr.metadata()

    def frame_metadata(self, seq_index: int | tuple) -> FrameMetadata | dict:
        """Metadata for specific frame.

        This includes the global metadata from the metadata function.
        (will be dict if legacy format).

        Parameters
        ----------
        seq_index : Union[int, tuple]
            frame index

        Returns
        -------
        Union[FrameMetadata, dict]
            dict if legacy format, else FrameMetadata
        """
        idx = cast(
            int,
            self._seq_index_from_coords(seq_index)
            if isinstance(seq_index, tuple)
            else seq_index,
        )
        return self._rdr.frame_metadata(idx)

    @cached_property
    def custom_data(self) -> dict[str, Any]:
        """Dict of various unstructured custom metadata."""
        return self._rdr._custom_data()

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Size of each axis."""
        return self._coord_shape + self._frame_shape

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Names and sizes for each axis."""
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
        """Whether the image is rgb."""
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        """Number of components per channel (e.g. 3 for rgb)."""
        attrs = self.attributes
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
        """Image data type."""
        attrs = self.attributes
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{attrs.bitsPerComponentInMemory // 8}")

    def voxel_size(self, channel: int = 0) -> _util.VoxelSize:
        """XYZ voxel size.

        Parameters
        ----------
        channel : int
            Channel for which to retrieve voxel info, by default 0

        Returns
        -------
        VoxelSize
            Named tuple with attrs `x`, `y`, and `z`.
        """
        return _util.VoxelSize(*self._rdr.voxel_size())

    def asarray(self, position: int | None = None) -> np.ndarray:
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
        ValueError
            if `position` is a string and is not a valid position name
        IndexError
            if `position` is provided and is out of range
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
                if position > 0:  # pragma: no cover
                    raise IndexError(
                        f"Position {position} is out of range. "
                        f"Only 1 position available"
                    ) from exc
                seqs = range(self._frame_count)
            else:
                if position >= self.sizes[AXIS.POSITION]:
                    raise IndexError(  # pragma: no cover
                        f"Position {position} is out of range. "
                        f"Only {self.sizes[AXIS.POSITION]} positions available"
                    )

                ranges: list[range | tuple] = [range(x) for x in self._coord_shape]
                ranges[pidx] = (position,)
                coords = list(zip(*product(*ranges)))
                seqs = self._seq_index_from_coords(coords)  # type: ignore
                final_shape[pidx] = 1

        arr: np.ndarray = np.stack([self._get_frame(i) for i in seqs])
        return arr.reshape(final_shape)

    def __array__(self) -> np.ndarray:
        """Array protocol."""
        return self.asarray()

    def to_dask(self, wrapper: bool = True, copy: bool = True) -> dask.array.core.Array:
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
            `ResourceBackedDaskArray`) that manages the opening and closing of this file
            when getting chunks via compute(). If `wrapper` is `False`, then a pure
            `dask.array.core.Array` will be returned. However, when that array is
            computed, it will incur a file open/close on *every* chunk that is read (in
            the `_dask_block` method).  As such `wrapper` will generally be much faster,
            however, it *may* fail (i.e. result in segmentation faults) with certain
            dask schedulers.
        copy : bool
            If `True` (the default), the dask chunk-reading function will return
            an array copy. This can avoid segfaults in certain cases, though it
            may also add overhead.

        Returns
        -------
        dask.array.core.Array
        """
        from dask.array.core import map_blocks

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

    def _seq_index_from_coords(self, coords: Sequence) -> Sequence[int] | SupportsInt:
        if not self._coord_shape:
            return self._NO_IDX
        return np.ravel_multi_index(coords, self._coord_shape)  # type: ignore

    def _dask_block(self, copy: bool, block_id: tuple[int]) -> np.ndarray:
        if isinstance(block_id, np.ndarray):
            return None
        with self._lock:
            was_closed = self.closed
            if self.closed:
                self.open()
            try:
                ncoords = len(self._coord_shape)
                idx = self._seq_index_from_coords(block_id[:ncoords])

                if idx == self._NO_IDX:
                    if any(block_id):  # pragma: no cover
                        raise ValueError(
                            f"Cannot get chunk {block_id} for single frame image."
                        )
                    idx = 0
                data = self._get_frame(int(idx))  # type: ignore
                data = data.copy() if copy else data
                return data[(np.newaxis,) * ncoords]
            finally:
                if was_closed:
                    self.close()

    def to_xarray(
        self,
        delayed: bool = True,
        squeeze: bool = True,
        position: int | None = None,
        copy: bool = True,
    ) -> xr.DataArray:
        """Create labeled xarray representing image.

        `array.dims` will be populated according to image metadata, and coordinates
        will be populated based on pixel spacings. Additional metadata is available
        in `array.attrs['metadata']`.

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
    def _frame_coords(self) -> set[str]:
        return {AXIS.X, AXIS.Y, AXIS.CHANNEL, AXIS.RGB}

    @property
    def _raw_frame_shape(self) -> tuple[int, int, int, int]:
        """Sizes of each frame coordinate, prior to reshape."""
        attr = self.attributes
        return (
            attr.heightPx,
            attr.widthPx or -1,
            attr.channelCount or 1,
            self.components_per_channel,
        )

    @property
    def _frame_shape(self) -> tuple[int, ...]:
        """Sizes of each frame coordinate, after reshape & squeeze."""
        return tuple(v for k, v in self.sizes.items() if k in self._frame_coords)

    @cached_property
    def _coord_shape(self) -> tuple[int, ...]:
        """Sizes of each *non-frame* coordinate."""
        return tuple(v for k, v in self.sizes.items() if k not in self._frame_coords)

    @property
    def _frame_count(self) -> int:
        return int(np.prod(self._coord_shape))

    def _get_frame(self, index: SupportsInt) -> np.ndarray:
        frame = self._rdr._read_image(int(index))
        frame.shape = self._raw_frame_shape
        return frame.transpose((2, 0, 1, 3)).squeeze()

    def _expand_coords(self, squeeze: bool = True) -> dict:
        """Return a dict that can be used as the coords argument to xr.DataArray.

        Parameters
        ----------
        squeeze : bool
            whether to squeeze axes with length < 2, by default True

        Returns
        -------
        dict
            dict of axis name -> coordinates
        """
        dx, dy, dz = self.voxel_size()

        coords: dict[str, Sized] = {
            AXIS.Y: np.arange(self.attributes.heightPx) * dy,
            AXIS.X: np.arange(self.attributes.widthPx or 1) * dx,
            AXIS.CHANNEL: self._channel_names,
            AXIS.POSITION: ["XYPos:0"],  # maybe overwritten below
        }

        for c in self.experiment:
            if squeeze and c.count <= 1:
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
        # TODO: this isn't hit by coverage... maybe it's not needed?
        if AXIS.Z in self.sizes and AXIS.Z not in coords:
            coords[AXIS.Z] = np.arange(self.sizes[AXIS.Z]) * dz

        if squeeze:
            coords = {k: v for k, v in coords.items() if len(v) > 1}
        return coords

    def _position_names(self, loop: XYPosLoop | None = None) -> list[str]:
        if loop is None:
            for c in self.experiment:
                if c.type == "XYPosLoop":
                    loop = c
                    break
        if loop is None:
            return ["XYPos:0"]
        return [p.name or f"XYPos:{i}" for i, p in enumerate(loop.parameters.points)]

    @property
    def _channel_names(self) -> list[str]:
        return self._rdr.channel_names()

    def __repr__(self) -> str:
        """Return a string representation of the ND2File."""
        try:
            details = " (closed)" if self.closed else f" {self.dtype}: {self.sizes!r}"
            extra = f": {Path(self.path).name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"

    @property
    def recorded_data(
        self,
    ) -> DictOfLists:
        """Return tabular data recorded for each frame of the experiment.

        This method returns a dict of equal-length sequences (passable to
        pd.DataFrame()). It matches the tabular data reported in the Image Properties >
        Recorded Data tab of the NIS Viewer.

        (There will be a column for each tag in the `CustomDataV2_0` section of
        `ND2File.custom_data`)

        Legacy ND2 files are not supported.
        """
        warnings.warn(
            "recorded_data is deprecated and will be removed in a future version."
            "Please use the `events` method instead. To get the same dict-of-lists "
            "output, use `events(orient='list')`",
            FutureWarning,
            stacklevel=2,
        )

        return self.events(orient="list")

    @cached_property
    def binary_data(self) -> BinaryLayers | None:
        """Return binary layers embedded in the file.

        The returned `BinaryLayers` object is an immutable sequence of `BinaryLayer`
        objects, one for each binary layer in the file.  Each `BinaryLayer` object in
        the sequence has a `name` attribute, and a `data` attribute which is list of
        numpy arrays (or `None` if there was no binary mask for that frame).  The length
        of the list will be the same as the number of sequence frames in this file
        (i.e. `self.attributes.sequenceCount`).

        Both the `BinaryLayers` and individual `BinaryLayer` objects can be cast to a
        numpy array with `np.asarray()`, or by using the `.asarray()` method

        Returns
        -------
        BinaryLayers | None
            The binary layers embedded in the file, or None if there are no binary
            layers.

        Examples
        --------
        >>> f = ND2File("path/to/file.nd2")
        >>> f.binary_data
        <BinaryLayers with 4 layers>
        >>> f.binary_data[0]  # the first binary layer
        BinaryLayer(name='attached Widefield green (green color)',
        comp_name='Widefield Green', comp_order=2, color=65280, color_mode=0,
        state=524288, file_tag='RleZipBinarySequence_1_v1', layer_id=2)
        >>> f.binary_data[0].data  # list of arrays
        >>> np.asarray(f.binary_data[0])  # just the first binary mask
        >>> np.asarray(f.binary_data).shape  # cast all layers to array
        (4, 3, 4, 5, 32, 32)
        """
        from ._binary import BinaryLayers

        return BinaryLayers.from_nd2file(self)


@overload
def imread(
    file: Path | str,
    *,
    dask: Literal[False],
    xarray: Literal[False],
    validate_frames: bool = False,
    read_using_sdk: bool | None = None,
) -> np.ndarray:
    ...


@overload
def imread(
    file: Path | str,
    *,
    dask: bool = ...,
    xarray: Literal[True],
    validate_frames: bool = False,
    read_using_sdk: bool | None = None,
) -> xr.DataArray:
    ...


@overload
def imread(
    file: Path | str,
    *,
    dask: Literal[True],
    xarray: Literal[False],
    validate_frames: bool = False,
    read_using_sdk: bool | None = None,
) -> dask.array.core.Array:
    ...


def imread(
    file: Path | str,
    *,
    dask: bool = False,
    xarray: bool = False,
    validate_frames: bool = False,
    read_using_sdk: bool | None = None,
) -> np.ndarray | xr.DataArray | dask.array.core.Array:
    """Open `file`, return requested array type, and close `file`.

    Parameters
    ----------
    file : Union[Path, str]
        Filepath (`str`) or `Path` object to ND2 file.
    dask : bool
        If `True`, returns a (delayed) `dask.array.Array`. This will avoid reading
        any data from disk until specifically requested by using `.compute()` or
        casting to a numpy array with `np.asarray()`. By default `False`.
    xarray : bool
        If `True`, returns an `xarray.DataArray`, `array.dims` will be populated
        according to image metadata, and coordinates will be populated based on pixel
        spacings. Additional metadata is available in `array.attrs['metadata']`.
        If `dask` is also `True`, will return an xarray backed by a delayed dask array.
        By default `False`.
    validate_frames : bool
        Whether to verify (and attempt to fix) frames whose positions have been
        shifted relative to the predicted offset (i.e. in a corrupted file).
        This comes at a slight performance penalty at file open, but may "rescue"
        some corrupt files. by default False.
    read_using_sdk : Optional[bool]
        DEPRECATED: no longer used.
        If `True`, use the SDK to read the file. If `False`, inspects the chunkmap and
        reads from a `numpy.memmap`. If `None` (the default), uses the SDK if the file
        is compressed, otherwise uses the memmap.
        Note: using `read_using_sdk=False` on a compressed file will result in a
        ValueError.

    Returns
    -------
    Union[np.ndarray, dask.array.Array, xarray.DataArray]
        Array subclass, depending on arguments used.
    """
    with ND2File(
        file, validate_frames=validate_frames, read_using_sdk=read_using_sdk
    ) as nd2:
        if xarray:
            return nd2.to_xarray(delayed=dask)
        elif dask:
            return nd2.to_dask()
        else:
            return nd2.asarray()
