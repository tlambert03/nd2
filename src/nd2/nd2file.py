from __future__ import annotations

import threading
import warnings
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, cast, overload

import numpy as np

from nd2 import _util

from ._util import AXIS, is_supported_file
from .readers.protocol import ND2Reader

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path
    from typing import Any, Literal, Sequence, Sized, SupportsInt

    import dask.array
    import dask.array.core
    import ome_types
    import xarray as xr
    from ome_types import OME

    from ._binary import BinaryLayers
    from ._util import (
        DictOfDicts,
        DictOfLists,
        FileOrBinaryIO,
        ListOfDicts,
        StrOrPath,
    )
    from .structures import (
        ROI,
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

    ```python
    with nd2.ND2File("path/to/file.nd2") as nd2_file:
        ...
    ```

    The key metadata outputs are:

    - [attributes][nd2.ND2File.attributes]
    - [metadata][nd2.ND2File.metadata] / [frame_metadata][nd2.ND2File.frame_metadata]
    - [experiment][nd2.ND2File.experiment]
    - [text_info][nd2.ND2File.text_info]

    Some files may also have:

    - [binary_data][nd2.ND2File.binary_data]
    - [rois][nd2.ND2File.rois]

    !!! tip

        For a simple way to read nd2 file data into an array, see [nd2.imread][].

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
    """

    def __init__(
        self,
        path: FileOrBinaryIO,
        *,
        validate_frames: bool = False,
        search_window: int = 100,
    ) -> None:
        self._error_radius: int | None = (
            search_window * 1000 if validate_frames else None
        )
        self._rdr = ND2Reader.create(path, self._error_radius)
        self._path = self._rdr._path
        self._lock = threading.RLock()

    @staticmethod
    def is_supported_file(path: StrOrPath) -> bool:
        """Return `True` if the file is supported by this reader."""
        return is_supported_file(path)

    @cached_property
    def version(self) -> tuple[int, ...]:
        """Return the file format version as a tuple of ints.

        Likely values are:

        - `(1, 0)` = a legacy nd2 file (JPEG2000)
        - `(2, 0)`, `(2, 1)` = non-JPEG2000 nd2 with xml metadata
        - `(3, 0)` = new format nd2 file with lite variant metadata
        - `(-1, -1)` =

        Returns
        -------
        tuple[int, ...]
            The file format version as a tuple of ints.

        Raises
        ------
        ValueError
            If the file is not a valid nd2 file.
        """
        return self._rdr.version()

    @property
    def path(self) -> str:
        """Path of the image."""
        return str(self._path)

    @property
    def is_legacy(self) -> bool:
        """Whether file is a legacy nd2 (JPEG2000) file."""
        return self._rdr.is_legacy()

    def open(self) -> None:
        """Open file for reading.

        !!! note

            Files are best opened using a context manager:

            ```python
            with nd2.ND2File("path/to/file.nd2") as nd2_file:
                ...
            ```

            This will automatically close the file when the context exits.
        """
        if self.closed:
            self._rdr.open()

    def close(self) -> None:
        """Close file.

        !!! note

            Files are best opened using a context manager:

            ```python
            with nd2.ND2File("path/to/file.nd2") as nd2_file:
                ...
            ```

            This will automatically close the file when the context exits.
        """
        if not self.closed:
            self._rdr.close()

    @property
    def closed(self) -> bool:
        """Return `True` if the file is closed."""
        return self._rdr._closed

    def __enter__(self) -> ND2File:
        """Open file for reading."""
        self.open()
        return self

    def __del__(self) -> None:
        """Delete file handle on garbage collection."""
        # if it came in as an open file handle, it's ok to remain open after deletion
        if not getattr(self, "closed", True) and not self._rdr._was_open:
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
        state.pop("sizes", None)  # cannot pickle MappingProxyType, we can make it again
        state["_closed"] = self.closed
        return state

    def __setstate__(self, d: dict[str, Any]) -> None:
        """Load state from pickling."""
        _was_closed = d.pop("_closed", False)
        self.__dict__ = d
        self._lock = threading.RLock()
        self._rdr = ND2Reader.create(self._path, self._error_radius)

        if _was_closed:
            self._rdr.close()

    @cached_property
    def attributes(self) -> Attributes:
        """Core image attributes.

        !!! example "Example Output"

            ```python
            Attributes(
                bitsPerComponentInMemory=16,
                bitsPerComponentSignificant=16,
                componentCount=2,
                heightPx=32,
                pixelDataType="unsigned",
                sequenceCount=60,
                widthBytes=128,
                widthPx=32,
                compressionLevel=None,
                compressionType=None,
                tileHeightPx=None,
                tileWidthPx=None,
                channelCount=2,
            )
            ```


        Returns
        -------
        attrs : Attributes
            Core image attributes
        """
        return self._rdr.attributes()

    @cached_property
    def text_info(self) -> TextInfo:
        r"""Miscellaneous text info.

        ??? example "Example Output"

            ```python
            {
                'description': 'Metadata:\r\nDimensions: T(3) x XY(4) x λ(2) x Z(5)...'
                'capturing': 'Flash4.0, SN:101412\r\nSample 1:\r\n  Exposure: 100 ms...'
                'date': '9/28/2021  9:41:27 AM',
                'optics': 'Plan Fluor 10x Ph1 DLL'
            }
            ```

        Returns
        -------
        TextInfo | dict
            If the file is a legacy nd2 file, a dict is returned. Otherwise, a
            `TextInfo` object is returned.
        """
        return self._rdr.text_info()

    @cached_property
    def rois(self) -> dict[int, ROI]:
        """Return dict of `{id: ROI}` for all ROIs found in the metadata.

        Returns
        -------
        dict[int, ROI]
            The dict of ROIs is keyed by the ROI ID.
        """
        return {r.id: r for r in self._rdr.rois()}

    @cached_property
    def experiment(self) -> list[ExpLoop]:
        """Loop information for each axis of an nD acquisition.

        ??? example "Example Output"

            ```python
            [
                TimeLoop(
                    count=3,
                    nestingLevel=0,
                    parameters=TimeLoopParams(
                        startMs=0.0,
                        periodMs=1.0,
                        durationMs=0.0,
                        periodDiff=PeriodDiff(
                            avg=3674.199951171875,
                            max=3701.219970703125,
                            min=3647.179931640625,
                        ),
                    ),
                    type="TimeLoop",
                ),
                ZStackLoop(
                    count=5,
                    nestingLevel=1,
                    parameters=ZStackLoopParams(
                        homeIndex=2,
                        stepUm=1.0,
                        bottomToTop=True,
                        deviceName="Ti2 ZDrive",
                    ),
                    type="ZStackLoop",
                ),
            ]
            ```

        Returns
        -------
        list[ExpLoop]
        """
        return self._rdr.experiment()

    @overload
    def events(
        self, *, orient: Literal["records"] = ..., null_value: Any = ...
    ) -> ListOfDicts: ...

    @overload
    def events(
        self, *, orient: Literal["list"], null_value: Any = ...
    ) -> DictOfLists: ...

    @overload
    def events(
        self, *, orient: Literal["dict"], null_value: Any = ...
    ) -> DictOfDicts: ...

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


        Returns
        -------
        ListOfDicts | DictOfLists | DictOfDicts
            Tabular data in the format specified by `orient`.
        """
        if orient not in ("records", "dict", "list"):  # pragma: no cover
            raise ValueError("orient must be one of 'records', 'dict', or 'list'")

        return self._rdr.events(orient=orient, null_value=null_value)

    def unstructured_metadata(
        self,
        *,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
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
        include : set[str] | None, optional
            If provided, only include the specified keys in the output. by default,
            all metadata sections found in the file are included.
        exclude : set[str] | None, optional
            If provided, exclude the specified keys from the output. by default `None`

        Returns
        -------
        dict[str, Any]
            A dict of the unstructured metadata, with keys that are the type of the
            metadata chunk (things like 'CustomData|RoiMetadata_v1' or
            'ImageMetadataLV'), and values that are associated metadata chunk.
        """
        return self._rdr.unstructured_metadata(strip_prefix, include, exclude)

    @cached_property
    def metadata(self) -> Metadata:
        """Various metadata (will be `dict` only if legacy format).

        ??? example "Example output"

            ```python
            Metadata(
                contents=Contents(channelCount=2, frameCount=15),
                channels=[
                    Channel(
                        channel=ChannelMeta(
                            name="Widefield Green",
                            index=0,
                            color=Color(r=91, g=255, b=0, a=1.0),
                            emissionLambdaNm=535.0,
                            excitationLambdaNm=None,
                        ),
                        loops=LoopIndices(
                            NETimeLoop=None, TimeLoop=0, XYPosLoop=None, ZStackLoop=1
                        ),
                        microscope=Microscope(
                            objectiveMagnification=10.0,
                            objectiveName="Plan Fluor 10x Ph1 DLL",
                            objectiveNumericalAperture=0.3,
                            zoomMagnification=1.0,
                            immersionRefractiveIndex=1.0,
                            projectiveMagnification=None,
                            pinholeDiameterUm=None,
                            modalityFlags=["fluorescence"],
                        ),
                        volume=Volume(
                            axesCalibrated=[True, True, True],
                            axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                            axesInterpretation=["distance", "distance", "distance"],
                            bitsPerComponentInMemory=16,
                            bitsPerComponentSignificant=16,
                            cameraTransformationMatrix=[
                                -0.9998932296054086,
                                -0.014612644841559427,
                                0.014612644841559427,
                                -0.9998932296054086,
                            ],
                            componentCount=1,
                            componentDataType="unsigned",
                            voxelCount=[32, 32, 5],
                            componentMaxima=[0.0],
                            componentMinima=[0.0],
                            pixelToStageTransformationMatrix=None,
                        ),
                    ),
                    Channel(
                        channel=ChannelMeta(
                            name="Widefield Red",
                            index=1,
                            color=Color(r=255, g=85, b=0, a=1.0),
                            emissionLambdaNm=620.0,
                            excitationLambdaNm=None,
                        ),
                        loops=LoopIndices(
                            NETimeLoop=None, TimeLoop=0, XYPosLoop=None, ZStackLoop=1
                        ),
                        microscope=Microscope(
                            objectiveMagnification=10.0,
                            objectiveName="Plan Fluor 10x Ph1 DLL",
                            objectiveNumericalAperture=0.3,
                            zoomMagnification=1.0,
                            immersionRefractiveIndex=1.0,
                            projectiveMagnification=None,
                            pinholeDiameterUm=None,
                            modalityFlags=["fluorescence"],
                        ),
                        volume=Volume(
                            axesCalibrated=[True, True, True],
                            axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                            axesInterpretation=["distance", "distance", "distance"],
                            bitsPerComponentInMemory=16,
                            bitsPerComponentSignificant=16,
                            cameraTransformationMatrix=[
                                -0.9998932296054086,
                                -0.014612644841559427,
                                0.014612644841559427,
                                -0.9998932296054086,
                            ],
                            componentCount=1,
                            componentDataType="unsigned",
                            voxelCount=[32, 32, 5],
                            componentMaxima=[0.0],
                            componentMinima=[0.0],
                            pixelToStageTransformationMatrix=None,
                        ),
                    ),
                ],
            )
            ```

        Returns
        -------
        Metadata | dict
            dict if legacy format, else `Metadata`
        """
        return self._rdr.metadata()

    def frame_metadata(self, seq_index: int | tuple) -> FrameMetadata | dict:
        """Metadata for specific frame.

        :eyes: **See also:** [metadata][nd2.ND2File.metadata]

        This includes the global metadata from the metadata function.
        (will be dict if legacy format).

        ??? example "Example output"

            ```python
            FrameMetadata(
                contents=Contents(channelCount=2, frameCount=15),
                channels=[
                    FrameChannel(
                        channel=ChannelMeta(
                            name="Widefield Green",
                            index=0,
                            color=Color(r=91, g=255, b=0, a=1.0),
                            emissionLambdaNm=535.0,
                            excitationLambdaNm=None,
                        ),
                        loops=LoopIndices(
                            NETimeLoop=None, TimeLoop=0, XYPosLoop=None, ZStackLoop=1
                        ),
                        microscope=Microscope(
                            objectiveMagnification=10.0,
                            objectiveName="Plan Fluor 10x Ph1 DLL",
                            objectiveNumericalAperture=0.3,
                            zoomMagnification=1.0,
                            immersionRefractiveIndex=1.0,
                            projectiveMagnification=None,
                            pinholeDiameterUm=None,
                            modalityFlags=["fluorescence"],
                        ),
                        volume=Volume(
                            axesCalibrated=[True, True, True],
                            axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                            axesInterpretation=["distance", "distance", "distance"],
                            bitsPerComponentInMemory=16,
                            bitsPerComponentSignificant=16,
                            cameraTransformationMatrix=[
                                -0.9998932296054086,
                                -0.014612644841559427,
                                0.014612644841559427,
                                -0.9998932296054086,
                            ],
                            componentCount=1,
                            componentDataType="unsigned",
                            voxelCount=[32, 32, 5],
                            componentMaxima=[0.0],
                            componentMinima=[0.0],
                            pixelToStageTransformationMatrix=None,
                        ),
                        position=Position(
                            stagePositionUm=StagePosition(
                                x=26950.2, y=-1801.6000000000001, z=494.3
                            ),
                            pfsOffset=None,
                            name=None,
                        ),
                        time=TimeStamp(
                            absoluteJulianDayNumber=2459486.0682717753,
                            relativeTimeMs=580.3582921028137,
                        ),
                    ),
                    FrameChannel(
                        channel=ChannelMeta(
                            name="Widefield Red",
                            index=1,
                            color=Color(r=255, g=85, b=0, a=1.0),
                            emissionLambdaNm=620.0,
                            excitationLambdaNm=None,
                        ),
                        loops=LoopIndices(
                            NETimeLoop=None, TimeLoop=0, XYPosLoop=None, ZStackLoop=1
                        ),
                        microscope=Microscope(
                            objectiveMagnification=10.0,
                            objectiveName="Plan Fluor 10x Ph1 DLL",
                            objectiveNumericalAperture=0.3,
                            zoomMagnification=1.0,
                            immersionRefractiveIndex=1.0,
                            projectiveMagnification=None,
                            pinholeDiameterUm=None,
                            modalityFlags=["fluorescence"],
                        ),
                        volume=Volume(
                            axesCalibrated=[True, True, True],
                            axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                            axesInterpretation=["distance", "distance", "distance"],
                            bitsPerComponentInMemory=16,
                            bitsPerComponentSignificant=16,
                            cameraTransformationMatrix=[
                                -0.9998932296054086,
                                -0.014612644841559427,
                                0.014612644841559427,
                                -0.9998932296054086,
                            ],
                            componentCount=1,
                            componentDataType="unsigned",
                            voxelCount=[32, 32, 5],
                            componentMaxima=[0.0],
                            componentMinima=[0.0],
                            pixelToStageTransformationMatrix=None,
                        ),
                        position=Position(
                            stagePositionUm=StagePosition(
                                x=26950.2, y=-1801.6000000000001, z=494.3
                            ),
                            pfsOffset=None,
                            name=None,
                        ),
                        time=TimeStamp(
                            absoluteJulianDayNumber=2459486.0682717753,
                            relativeTimeMs=580.3582921028137,
                        ),
                    ),
                ],
            )
            ```

        Parameters
        ----------
        seq_index : Union[int, tuple]
            frame index

        Returns
        -------
        FrameMetadata | dict
            dict if legacy format, else FrameMetadata
        """
        idx = cast(
            int,
            (
                self._seq_index_from_coords(seq_index)
                if isinstance(seq_index, tuple)
                else seq_index
            ),
        )
        return self._rdr.frame_metadata(idx)

    @cached_property
    def custom_data(self) -> dict[str, Any]:
        """Dict of various unstructured custom metadata."""
        return self._rdr.custom_data()

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions (i.e. `len(`[`self.shape`][nd2.ND2File.shape]`)`)."""
        return len(self.shape)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Size of each axis.

        Examples
        --------
        >>> ndfile.shape
        (3, 5, 2, 512, 512)
        """
        return self._coord_shape + self._frame_shape

    def _coord_info(self) -> list[tuple[int, str, int]]:
        return [(i, x.type, x.count) for i, x in enumerate(self.experiment)]

    @cached_property
    def sizes(self) -> Mapping[str, int]:
        """Names and sizes for each axis.

        This is an ordered dict, with the same order
        as the corresponding [shape][nd2.ND2File.shape]

        Examples
        --------
        >>> ndfile.sizes
        {'T': 3, 'Z': 5, 'C': 2, 'Y': 512, 'X': 512}
        >>> ndfile.shape
        (3, 5, 2, 512, 512)
        """
        attrs = self.attributes
        dims = {AXIS._MAP[c[1]]: c[2] for c in self._coord_info()}
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
        return MappingProxyType({k: v for k, v in dims.items() if v != 1})

    @property
    def is_rgb(self) -> bool:
        """Whether the image is rgb (i.e. it has 3 or 4 components per channel)."""
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        """Number of components per channel (e.g. 3 for rgb)."""
        attrs = self.attributes
        return attrs.componentCount // (attrs.channelCount or 1)

    @property
    def size(self) -> int:
        """Total number of voxels in the volume (the product of the shape)."""
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
        """XYZ voxel size in microns.

        Parameters
        ----------
        channel : int
            Channel for which to retrieve voxel info, by default 0.
            (Not yet implemented.)

        Returns
        -------
        VoxelSize
            Named tuple with attrs `x`, `y`, and `z`.
        """
        return _util.VoxelSize(*self._rdr.voxel_size())

    def asarray(self, position: int | None = None) -> np.ndarray:
        """Read image into a [numpy.ndarray][].

        For a simple way to read a file into a numpy array, see [nd2.imread][].

        Parameters
        ----------
        position : int, optional
            A specific XY position to extract, by default (None) reads all.

        Returns
        -------
        array : np.ndarray

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

        arr: np.ndarray = np.stack([self.read_frame(i) for i in seqs])
        return arr.reshape(final_shape)

    def __array__(self) -> np.ndarray:
        """Array protocol."""
        return self.asarray()

    def write_tiff(
        self,
        dest: str | PathLike,
        *,
        include_unstructured_metadata: bool = True,
        progress: bool = False,
        on_frame: Callable[[int, int, dict[str, int]], None] | None | None = None,
        modify_ome: Callable[[ome_types.OME], None] | None = None,
    ) -> None:
        """Export to an (OME)-TIFF file.

        To include OME-XML metadata, use extension `.ome.tif` or `.ome.tiff`.

        Parameters
        ----------
        dest : str  | PathLike
            The destination TIFF file.
        include_unstructured_metadata :  bool
            Whether to include unstructured metadata in the OME-XML.
            This includes all of the metadata that we can find in the ND2 file in the
            StructuredAnnotations section of the OME-XML (as mapping of
            metadata chunk name to JSON-encoded string). By default `True`.
        progress : bool
            Whether to display progress bar.  If `True` and `tqdm` is installed, it will
            be used. Otherwise, a simple text counter will be printed to the console.
            By default `False`.
        on_frame : Callable[[int, int, dict[str, int]], None] | None
            A function to call after each frame is written. The function should accept
            three arguments: the current frame number, the total number of frames, and
            a dictionary of the current frame's indices (e.g. `{"T": 0, "Z": 1}`)
            (Useful for integrating custom progress bars or logging.)
        modify_ome : Callable[[ome_types.OME], None]
            A function to modify the OME metadata before writing it to the file.
            Accepts an `ome_types.OME` object and should modify it in place.
            (reminder: OME-XML is only written if the file extension is `.ome.tif` or
            `.ome.tiff`)
        """
        from .tiff import nd2_to_tiff

        return nd2_to_tiff(
            self,
            dest,
            include_unstructured_metadata=include_unstructured_metadata,
            progress=progress,
            on_frame=on_frame,
            modify_ome=modify_ome,
        )

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
            If `True` (the default), the returned object will be a thin subclass of a
            [`dask.array.Array`][] (a `ResourceBackedDaskArray`) that manages the
            opening and closing of this file when getting chunks via compute(). If
            `wrapper` is `False`, then a pure `dask.array.core.Array` will be returned.
            However, when that array is computed, it will incur a file open/close on
            *every* chunk that is read (in the `_dask_block` method).  As such `wrapper`
            will generally be much faster, however, it *may* fail (i.e. result in
            segmentation faults) with certain dask schedulers.
        copy : bool
            If `True` (the default), the dask chunk-reading function will return
            an array copy. This can avoid segfaults in certain cases, though it
            may also add overhead.

        Returns
        -------
        dask_array: dask.array.Array
            A dask array representing the image data.
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
                data = self.read_frame(int(idx))  # type: ignore
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
        """Return a labeled [xarray.DataArray][] representing image.

        Xarrays are a powerful way to label and manipulate n-dimensional data with
        axis-associated coordinates.

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
        return tuple(v for k, v in self.sizes.items() if k in AXIS.frame_coords())

    @cached_property
    def _coord_shape(self) -> tuple[int, ...]:
        """Sizes of each *non-frame* coordinate."""
        return tuple(v for k, v in self.sizes.items() if k not in AXIS.frame_coords())

    @property
    def _frame_count(self) -> int:
        if hasattr(self._rdr, "_seq_count"):
            return cast(int, self._rdr._seq_count())
        return int(np.prod(self._coord_shape))

    def _get_frame(self, index: SupportsInt) -> np.ndarray:  # pragma: no cover
        warnings.warn(
            'Use of "_get_frame" is deprecated, use the public "read_frame" instead.',
            stacklevel=2,
        )
        return self.read_frame(index)

    def read_frame(self, frame_index: SupportsInt) -> np.ndarray:
        """Read a single frame from the file, indexed by frame number."""
        frame = self._rdr.read_frame(int(frame_index))
        frame.shape = self._raw_frame_shape
        return frame.transpose((2, 0, 1, 3)).squeeze()

    @cached_property
    def loop_indices(self) -> tuple[dict[str, int], ...]:
        """Return a tuple of dicts of loop indices for each frame.

        Examples
        --------
        >>> with nd2.ND2File("path/to/file.nd2") as f:
        ...     f.loop_indices
        (
            {'Z': 0, 'T': 0, 'C': 0},
            {'Z': 0, 'T': 0, 'C': 1},
            {'Z': 0, 'T': 0, 'C': 2},
            ...
        )
        """
        return _util.loop_indices(self.experiment)

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
        return [c.channel.name for c in self.metadata.channels or []]

    def __repr__(self) -> str:
        """Return a string representation of the ND2File."""
        try:
            details = " (closed)" if self.closed else f" {self.dtype}: {self.sizes!r}"
            extra = f": {self._path.name!r}{details}"
        except Exception:
            extra = ""
        return f"<ND2File at {hex(id(self))}{extra}>"

    @cached_property
    def binary_data(self) -> BinaryLayers | None:
        """Return binary layers embedded in the file.

        The returned `BinaryLayers` object is an immutable sequence of `BinaryLayer`
        objects, one for each binary layer in the file (there will usually be a binary
        layer associated with each channel in the dataset).

        Each `BinaryLayer` object in the sequence has a `name` attribute, and a `data`
        attribute which is list of numpy arrays (or `None` if there was no binary mask
        for that frame).  The length of the list will be the same as the number of
        sequence frames in this file (i.e. `self.attributes.sequenceCount`).
        `BinaryLayers` can be indexed directly with an integer corresponding to the
        *frame* index.

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
        >>> first_layer = f.binary_data[0]  # the first binary layer
        >>> first_layer
        BinaryLayer(name='attached Widefield green (green color)',
        comp_name='Widefield Green', comp_order=2, color=65280, color_mode=0,
        state=524288, file_tag='RleZipBinarySequence_1_v1', layer_id=2)
        >>> first_layer.data  # list of arrays
        # you can also index in to the BinaryLayers object itself
        >>> first_layer[0]  # get binary data for first frame (or None if missing)
        >>> np.asarray(first_layer)  # cast to array matching shape of full sequence
        >>> np.asarray(f.binary_data).shape  # cast all layers to array
        (4, 3, 4, 5, 32, 32)
        """
        return self._rdr.binary_data()

    def ome_metadata(
        self, *, include_unstructured: bool = True, tiff_file_name: str | None = None
    ) -> OME:
        """Return `ome_types.OME` metadata object for this file.

        See the [`ome_types.OME`][] documentation for details on this object.

        Parameters
        ----------
        include_unstructured : bool
            Whether to include all available metadata in the OME file. If `True`,
            (the default), the `unstructured_metadata` method is used to fetch
            all retrievable metadata, and the output is added to
            OME.structured_annotations, where each key is the chunk key, and the
            value is a JSON-serialized dict of the metadata. If `False`, only metadata
            which can be directly added to the OME data model are included.
        tiff_file_name : str | None
            If provided, [`ome_types.model.TiffData`][] block entries are added for
            each [`ome_types.model.Plane`][] in the OME object, with the
            `TiffData.uuid.file_name` set to this value. (Useful for exporting to
            tiff.)

        Examples
        --------
        ```python
        import nd2

        with nd2.ND2File("path/to/file.nd2") as f:
            ome = f.ome_metadata()
            xml = ome.to_xml()
        ```
        """
        from ._ome import nd2_ome_metadata

        return nd2_ome_metadata(
            self,
            include_unstructured=include_unstructured,
            tiff_file_name=tiff_file_name,
        )


@overload
def imread(
    file: Path | str,
    *,
    dask: Literal[False] = ...,
    xarray: Literal[False] = ...,
    validate_frames: bool = ...,
) -> np.ndarray: ...


@overload
def imread(
    file: Path | str,
    *,
    dask: bool = ...,
    xarray: Literal[True],
    validate_frames: bool = ...,
) -> xr.DataArray: ...


@overload
def imread(
    file: Path | str,
    *,
    dask: Literal[True],
    xarray: Literal[False] = ...,
    validate_frames: bool = ...,
) -> dask.array.core.Array: ...


def imread(
    file: Path | str,
    *,
    dask: bool = False,
    xarray: bool = False,
    validate_frames: bool = False,
) -> np.ndarray | xr.DataArray | dask.array.core.Array:
    """Open `file`, return requested array type, and close `file`.

    Parameters
    ----------
    file : Path | str
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

    Returns
    -------
    Union[np.ndarray, dask.array.Array, xarray.DataArray]
        Array subclass, depending on arguments used.
    """
    with ND2File(file, validate_frames=validate_frames) as nd2:
        if xarray:
            return nd2.to_xarray(delayed=dask)
        elif dask:
            return nd2.to_dask()
        else:
            return nd2.asarray()
