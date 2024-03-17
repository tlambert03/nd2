from __future__ import annotations

import os
import warnings
import zlib
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, cast

import numpy as np

from nd2 import _util, structures
from nd2._parse._chunk_decode import (
    ND2_FILE_SIGNATURE,
    _robustly_read_named_chunk,
    get_chunkmap,
    read_nd2_chunk,
)
from nd2._parse._clx_lite import json_from_clx_lite_variant
from nd2._parse._clx_xml import json_from_clx_variant
from nd2._parse._parse import (
    load_attributes,
    load_events,
    load_experiment,
    load_frame_metadata,
    load_global_metadata,
    load_metadata,
    load_text_info,
)
from nd2.readers.protocol import ND2Reader
from nd2.structures import ROI

if TYPE_CHECKING:
    import datetime
    from os import PathLike
    from typing import Literal

    from typing_extensions import TypeAlias

    from nd2._binary import BinaryLayers
    from nd2._parse._chunk_decode import ChunkMap
    from nd2._sdk_types import (
        BinaryMetaDict,
        GlobalMetadata,
        RawAttributesDict,
        RawExperimentDict,
        RawExperimentRecordDict,
        RawMetaDict,
        RawTagDict,
        RawTextInfoDict,
    )
    from nd2._util import FileOrBinaryIO

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class ModernReader(ND2Reader):
    HEADER_MAGIC = _util.NEW_HEADER_MAGIC

    def __init__(self, path: FileOrBinaryIO, error_radius: int | None = None) -> None:
        super().__init__(path, error_radius)

        self._cached_decoded_chunks: dict[bytes, Any] = {}

        self._attributes: structures.Attributes | None = None
        self._experiment: list[structures.ExpLoop] | None = None
        self._text_info: structures.TextInfo | None = None
        self._metadata: structures.Metadata | None = None
        self._events: list[structures.ExperimentEvent] | None = None

        self._global_metadata: GlobalMetadata | None = None
        self._cached_frame_offsets: dict[int, int] | None = None
        self._raw_frame_shape_: tuple[int, ...] | None = None
        self._dtype_: np.dtype | None = None
        self._strides_: tuple[int, ...] | None = None
        self._frame_times: list[float] | None = None
        # these caches could be removed... they aren't really used
        self._raw_attributes: RawAttributesDict | None = None
        self._raw_experiment: RawExperimentDict | None = None
        self._raw_text_info: RawTextInfoDict | None = None
        self._raw_image_metadata: RawMetaDict | None = None

        self._loop_indices: tuple[dict[str, int], ...] | None = None

    @property
    def chunkmap(self) -> ChunkMap:
        """Load and return the chunkmap.

        a Chunkmap is mapping of chunk names (bytes) to (offset, size) pairs.
        {
            b'ImageTextInfoLV!': (13041664, 2128),
            b'ImageTextInfo!': (13037568, 1884),
            b'ImageMetadataSeq|0!': (237568, 33412),
            ...
        }
        """
        if not self._chunkmap:
            if self._fh is None:  # pragma: no cover
                raise OSError("File not open")
            self._chunkmap = get_chunkmap(self._fh, error_radius=self._error_radius)
        return cast("ChunkMap", self._chunkmap)

    def attributes(self) -> structures.Attributes:
        """Load and return the image attributes."""
        if self._attributes is None:
            k = (
                b"ImageAttributesLV!"
                if self.version() >= (3, 0)
                else b"ImageAttributes!"
            )
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            raw_meta = self._cached_raw_metadata()  # ugly
            n_channels = raw_meta.get("sPicturePlanes", {}).get("uiCount", 1)
            self._raw_attributes = cast("RawAttributesDict", attrs)
            self._attributes = load_attributes(self._raw_attributes, n_channels)
        return self._attributes

    def _load_chunk(self, name: bytes) -> bytes:
        """Load raw bytes from a specific chunk in the chunkmap.

        `name` must be a valid key in the chunkmap.
        """
        if self._fh is None:  # pragma: no cover
            raise OSError("File not open")

        try:
            offset = self.chunkmap[name][0]
        except KeyError as e:
            raise KeyError(
                f"Chunk key {name!r} not found in chunkmap: {set(self.chunkmap)}"
            ) from e

        if self._error_radius is None:
            return read_nd2_chunk(self._fh, offset)
        return _robustly_read_named_chunk(
            self._fh, offset, expect_name=name, search_radius=self._error_radius
        )

    def _decode_chunk(self, name: bytes, strip_prefix: bool = True) -> dict | Any:
        """Convert raw chunk bytes to a Python object.

        Parameters
        ----------
        name : bytes
            The name of the chunk to load.  Must be a valid key in the chunkmap.
        strip_prefix : bool, optional
            If True, strip the lowercase "type" prefix from the tag names, by default
            False.
        """
        if name not in self._cached_decoded_chunks:
            data = self._load_chunk(name)
            if data.startswith(b"<"):
                decoded: Any = json_from_clx_variant(data, strip_prefix=strip_prefix)
            else:
                decoded = json_from_clx_lite_variant(data, strip_prefix=strip_prefix)
            self._cached_decoded_chunks[name] = decoded
        return self._cached_decoded_chunks[name]

    def _cached_raw_metadata(self) -> RawMetaDict:
        if self._raw_image_metadata is None:
            k = (
                b"ImageMetadataSeqLV|0!"
                if self.version() >= (3, 0)
                else b"ImageMetadataSeq|0!"
            )
            meta = self._decode_chunk(k, strip_prefix=False)
            meta = meta.get("SLxPictureMetadata", meta)  # for v3 only
            self._raw_image_metadata = cast("RawMetaDict", meta)
        return self._raw_image_metadata

    def _cached_global_metadata(self) -> GlobalMetadata:
        if not self._global_metadata:
            self._global_metadata = load_global_metadata(
                attrs=self.attributes(),
                raw_meta=self._cached_raw_metadata(),
                exp_loops=self.experiment(),
                text_info=self.text_info(),
            )
            if self._global_metadata["time"]["absoluteJulianDayNumber"] < 1:
                julian_day = os.stat(self._path).st_ctime / 86400.0 + 2440587.5
                self._global_metadata["time"]["absoluteJulianDayNumber"] = julian_day

        return self._global_metadata

    def metadata(self) -> structures.Metadata:
        if not self._metadata:
            self._metadata = load_metadata(
                raw_meta=self._cached_raw_metadata(),
                global_meta=self._cached_global_metadata(),
            )
        return self._metadata

    def frame_metadata(self, seq_index: int) -> structures.FrameMetadata:
        frame_time = self._cached_frame_times()[seq_index]
        global_meta = self._cached_global_metadata()
        loop_indices = self.loop_indices()[seq_index]
        return load_frame_metadata(
            global_meta, self.metadata(), self.experiment(), frame_time, loop_indices
        )

    def text_info(self) -> structures.TextInfo:
        if self._text_info is None:
            k = b"ImageTextInfoLV!" if self.version() >= (3, 0) else b"ImageTextInfo!"
            if k not in self.chunkmap:
                self._text_info = {}
            else:
                info = self._decode_chunk(k, strip_prefix=False)
                info = info.get("SLxImageTextInfo", info)  # for v3 only
                self._raw_text_info = cast("RawTextInfoDict", info)
                self._text_info = load_text_info(self._raw_text_info)
        return self._text_info

    def experiment(self) -> list[structures.ExpLoop]:
        if not self._experiment:
            k = b"ImageMetadataLV!" if self.version() >= (3, 0) else b"ImageMetadata!"
            if k not in self.chunkmap:
                self._experiment = []
            else:
                exp = self._decode_chunk(k, strip_prefix=False)
                exp = exp.get("SLxExperiment", exp)  # for v3 only
                self._raw_experiment = cast("RawExperimentDict", exp)
                self._experiment = load_experiment(self._raw_experiment)
        return self._experiment

    def loop_indices(self) -> tuple[dict[str, int], ...]:
        """Return a tuple of dicts of loop indices for each frame.

        Examples
        --------
        >>> with nd2.ND2File("path/to/file.nd2") as f:
        ...     f.loop_indices()
        (
            {'Z': 0, 'T': 0, 'C': 0},
            {'Z': 0, 'T': 0, 'C': 1},
            {'Z': 0, 'T': 0, 'C': 2},
            ...
        )
        """
        if self._loop_indices is None:
            self._loop_indices = _util.loop_indices(self.experiment())
        return self._loop_indices

    def _img_exp_events(self) -> list[structures.ExperimentEvent]:
        """Parse and return all Image and Experiment events."""
        if not self._events:
            events = []
            for key in (
                b"ImageEvents",
                b"ImageEventsLV!",
                b"CustomData|ExperimentEventsV1_0!",
            ):
                if key in self.chunkmap:
                    e = self._decode_chunk(key, strip_prefix=False)
                    e = e.get("RLxExperimentRecord", e)
                    events.extend(load_events(cast("RawExperimentRecordDict", e)))
            self._events = events
        return self._events

    def _cached_frame_times(self) -> list[float]:
        """Returns frame times in milliseconds."""
        if self._frame_times is None:
            try:
                acq_times = self._load_chunk(b"CustomData|AcqTimesCache!")
                times = np.frombuffer(acq_times, dtype=np.float64).tolist()
                self._frame_times = times[: self._seq_count()]  # limit to valid frames
            except Exception as e:
                warnings.warn(f"Failed to load frame times: {e}", stacklevel=2)
                self._frame_times = []

        return self._frame_times

    def voxel_size(self) -> tuple[float, float, float]:
        meta = self.metadata()
        if meta:
            ch = meta.channels
            if ch:
                return ch[0].volume.axesCalibration
        return (1, 1, 1)

    def _coords_from_seq_index(self, seq_index: int) -> tuple[int, ...]:
        """Convert a sequence index to a coordinate tuple."""
        coords: list[int] = []
        for loop in self.experiment():
            coords.append(seq_index % loop.count)
            seq_index //= loop.count

        return tuple(coords)

    def _coord_size(self) -> int:
        return len(self.experiment())

    def _seq_count(self) -> int:
        # this differs from self.attributes().SequenceCount in that it
        # includes the actual number of frames in the experiment,
        # excluding "invalid" frames.
        return int(np.prod([x.count for x in self.experiment()]))

    @property
    def _frame_offsets(self) -> dict[int, int]:
        """Return map of frame number to offset in the file."""
        if self._cached_frame_offsets is None:
            # image frames are stored in the chunkmap as "ImageDataSeq|<frame>!"
            # and their data is stored 24 + 4072 bytes after the chunkmap offset
            data_offset = 24 + 4072
            self._cached_frame_offsets = {
                int(chunk_key[13:-1]): int(offset + data_offset)
                for chunk_key, (offset, _) in sorted(self.chunkmap.items())
                if chunk_key.startswith(b"ImageDataSeq|")
            }
        return self._cached_frame_offsets

    def read_frame(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK."""
        if not self._fh:  # pragma: no cover
            raise ValueError("Attempt to read from closed nd2 file")

        # sometimes a frame index has a valid offset even if it's greater than
        # _seq_count() (for example, if experiment parsing misses stuff)
        # so, it should still be accessible.
        offset = self._frame_offsets.get(index, None)
        if offset is None:
            if index > self._seq_count():  # pragma: no cover
                raise IndexError(f"Frame out of range: {index}")
            return self._missing_frame(index)

        if self.attributes().compressionType == "lossless":
            return self._read_compressed_frame(index)

        try:
            return np.ndarray(
                shape=self._actual_frame_shape(),
                dtype=self._dtype(),
                buffer=self._mmap,
                offset=offset,
                strides=self._strides,
            )
        except TypeError:
            # If the chunkmap is wrong, and the mmap isn't long enough
            # for the requested offset & size, a TypeError is raised.
            return self._missing_frame(index)

    def _read_compressed_frame(self, index: int) -> np.ndarray:
        ch = self._load_chunk(f"ImageDataSeq|{index}!".encode())
        return np.ndarray(
            shape=self._actual_frame_shape(),
            dtype=self._dtype(),
            buffer=zlib.decompress(ch[8:]),
            strides=self._strides,
        )

    def _missing_frame(self, index: int = 0) -> np.ndarray:
        # TODO: add other modes for filling missing data
        return np.zeros(self._raw_frame_shape(), self._dtype())

    def _raw_frame_shape(self) -> tuple[int, ...]:
        if self._raw_frame_shape_ is None:
            attr = self.attributes()
            ncomp = attr.componentCount
            self._raw_frame_shape_ = (
                attr.heightPx,
                (attr.widthBytes or 0) // self._bytes_per_pixel() // ncomp,
                attr.channelCount or 1,
                ncomp // (attr.channelCount or 1),
            )
        return self._raw_frame_shape_

    def _bytes_per_pixel(self) -> int:
        return self.attributes().bitsPerComponentInMemory // 8

    def _dtype(self) -> np.dtype:
        if self._dtype_ is None:
            a = self.attributes()
            d = a.pixelDataType[0] if a.pixelDataType else "u"
            self._dtype_ = np.dtype(f"{d}{self._bytes_per_pixel()}")
        return self._dtype_

    @property
    def _strides(self) -> tuple[int, ...] | None:
        if self._strides_ is None:
            a = self.attributes()
            widthP = a.widthPx
            widthB = a.widthBytes
            if not (widthP and widthB):
                self._strides_ = None
            else:
                n_components = a.componentCount
                bypc = a.bitsPerComponentInMemory // 8
                if widthB == (widthP * bypc * n_components):
                    self._strides_ = None
                else:
                    # the extra bypc is because we shape this as
                    # (width, height, channels, RGBcompents)
                    # see _actual_frame_shape() below
                    self._strides_ = (widthB, n_components * bypc, bypc, bypc)
        return self._strides_

    def _actual_frame_shape(self) -> tuple[int, ...]:
        attr = self.attributes()
        nC = attr.channelCount or 1
        return (attr.heightPx, attr.widthPx or 1, nC, attr.componentCount // nC)

    def custom_data(self) -> dict[str, Any]:
        return {
            k.decode()[14:-1]: self._decode_chunk(k)
            for k in self.chunkmap
            if k.startswith(b"CustomDataVar|")
        }

    def _acquisition_data(self) -> dict[str, Sequence[Any]]:
        """Return a dict of acquisition times and z-series indices for each image.

        {
            "Time [s]": [0.0, 0.0, 0.0, ...],
            "Z-Series": [-1.0, 0., 1.0, ...],
            "Index": [0, 1, 2, ...],
            "T Index": [0, 0, 0, ...],
            "Z Index": [0, 1, 2, ...],
        }
        """
        data: dict[str, list] = {}
        frame_times = self._cached_frame_times()
        if frame_times:
            data[_util.TIME_KEY] = [x / 1000 for x in frame_times]

        loop_indices = self.loop_indices()
        for frame_idx, loop_idx in enumerate(loop_indices):
            data.setdefault("Index", []).append(frame_idx)
            for axis, value in loop_idx.items():
                data.setdefault(f"{axis} Index", []).append(value)

        for loop in self.experiment():
            if isinstance(loop, structures.ZStackLoop):
                # zpos is a list of actual z positions at each z-index in a single stack
                # e.g. [-1, -.5, 0, 0.5, 1]
                params = loop.parameters
                zpos = [
                    params.stepUm * (i - params.homeIndex) for i in range(loop.count)
                ]
                if not params.bottomToTop:
                    zpos.reverse()
                data[_util.Z_SERIES_KEY] = [
                    zpos[frame_index[_util.AXIS.Z]] for frame_index in loop_indices
                ]
            elif isinstance(loop, structures.XYPosLoop):
                names = [p.name or "" for p in loop.parameters.points]
                data[_util.POSITION_NAME] = [
                    names[frame_index[_util.AXIS.POSITION]]
                    for frame_index in loop_indices
                ]

        return data  # type: ignore [return-value]

    def _custom_tags(self) -> dict[str, Any]:
        """Return tags mentioned in in CustomDataVar|CustomDataV2_0.

        This is a dict of {header: [values]}, where
        len([values]) == self.attributes().sequenceCount

        {
            "Camera_ExposureTime1": [0.0, 0.0, 0.0, ...],
            "PFS_OFFSET": [-1.0, 0., 1.0, ...],
            "PFS_STATUS": [0, 0, 0, ...],
        }
        """
        data: dict[str, Any] = {}
        try:
            cd = self._decode_chunk(b"CustomDataVar|CustomDataV2_0!")
        except KeyError:
            return data

        if not cd:  # pragma: no cover
            return data

        if "CustomTagDescription_v1.0" not in cd:  # pragma: no cover
            warnings.warn(
                "Could not find 'CustomTagDescription_v1' tag, please open an issue "
                "with this nd2 file at https://github.com/tlambert03/nd2/issues/new",
                stacklevel=2,
            )
            return {}

        # tags will be a dict of dicts: eg:
        # {
        #     'Tag0': {'ID': 'Camera_ExposureTime1', 'Type': 3, ... },
        #     'Tag1': {'ID': 'PFS_OFFSET', 'Type': 2, 'Group': 0, 'Size': 1, ...},
        #     'Tag2': {'ID': 'PFS_STATUS', 'Type': 2, 'Group': 0, 'Size': 1, ...},
        # }
        tags = cast("Iterable[RawTagDict]", cd["CustomTagDescription_v1.0"].values())
        for tag in tags:
            col_header = tag["Desc"]
            if col_header in data:  # pragma: no cover
                # sourcery skip: hoist-if-from-if
                col_header = tag["ID"]
                if col_header in data:
                    col_header = f"{tag['Desc']} ({tag['ID']})"

            if tag["Unit"].strip():
                col_header += f" [{tag['Unit']}]"

            buffer_ = self._load_chunk(f"CustomData|{tag['ID']}!".encode())
            count = tag["Size"]
            if tag["Type"] == 1:
                # string data, so far I've only seen this as wide-strings of utf-16
                # in 512 byte chunks. (I'm not sure if this is always the case)
                try:
                    chunk_size = len(buffer_) // count
                    rows: Any = []
                    for i in range(count):
                        word = buffer_[i * chunk_size : (i + 1) * chunk_size]
                        rows.append(word.decode("utf-16").split("\x00", 1)[0])
                except Exception as e:  # pragma: no cover
                    warnings.warn(
                        f"Failed to parse {tag['Desc']!r} column: {e}. Please open an "
                        "issue with this nd2 file at "
                        "https://github.com/tlambert03/nd2/issues/new",
                        stacklevel=2,
                    )
                    continue
            else:
                dtype = {3: np.float64, 2: np.int32}[tag["Type"]]
                rows = np.frombuffer(buffer_, dtype=dtype, count=count)
            data[col_header] = rows

        return data

    def _app_info(self) -> dict:
        """Return a dict of app info."""
        k = b"CustomDataVar|AppInfo_V1_0!"
        return self._decode_chunk(k) if k in self.chunkmap else {}

    def _acquisition_date(self) -> datetime.datetime | str | None:
        """Try to extract acquisition date.

        A best effort is made to extract a datetime object from the date string,
        but if that fails, the raw string is returned.  Use isinstance() to
        be safe.
        """
        date = self.text_info().get("date")
        if date:
            try:
                return _util.parse_time(date)
            except ValueError:
                return date

        time = self._cached_global_metadata().get("time", {})
        jdn = time.get("absoluteJulianDayNumber")
        return _util.jdn_to_datetime(jdn) if jdn else None

    def binary_data(self) -> BinaryLayers | None:
        from nd2._binary import BinaryLayer, BinaryLayers, decode_binary_mask

        chunk_key = b"CustomDataVar|BinaryMetadata_v1!"
        if chunk_key not in self.chunkmap:
            return None
        binary_meta = self._decode_chunk(chunk_key, strip_prefix=True)

        try:
            items = cast("dict[str, BinaryMetaDict]", binary_meta["BinaryMetadata_v1"])
        except KeyError:  # pragma: no cover
            warnings.warn(
                "Could not find 'BinaryMetadata_v1' tag, please open an "
                "issue with this file at https://github.com/tlambert03/nd2/issues/new",
                stacklevel=2,
            )
            return None

        mask_items = []
        coord_shape = tuple(x.count for x in self.experiment())
        for _, item in sorted(items.items()):
            # something like: RleZipBinarySequence_1bd900c
            key = item["FileTag"]
            _masks: list[np.ndarray | None] = []
            for plane in range(self._seq_count()):
                # this will be something like
                # b'CustomDataSeq|RleZipBinarySequence_1bd900c|1153!
                chunk_key = f"CustomDataSeq|{key}|{plane}!".encode()
                if chunk_key in self.chunkmap:
                    data = self._load_chunk(chunk_key)[4:]
                else:
                    # it's conceivable that some frames don't have binary
                    # sequence masks written, so we'll just fill in None
                    data = None  # pragma: no cover
                _masks.append(decode_binary_mask(data) if data else None)

            mask_items.append(
                BinaryLayer(
                    data=_masks,
                    file_tag=key,
                    name=item["Name"],
                    comp_name=item.get("CompName"),
                    comp_order=item.get("CompOrder"),
                    color_mode=item.get("ColorMode"),
                    state=item.get("State"),
                    color=item.get("Color"),
                    layer_id=item.get("BinLayerID"),
                    coordinate_shape=coord_shape,
                )
            )

        return BinaryLayers(mask_items)

    def events(
        self, orient: Literal["records", "list", "dict"], null_value: Any
    ) -> list | Mapping:
        acq_data = self._acquisition_data()  # comes back as a dict of lists
        acq_data.update(self._custom_tags())

        img_events = self._img_exp_events()
        if not img_events and orient == "list":
            # by default, acq_data is already oriented as a dict of lists,
            # so if we don't have any image events, we can just return it
            return acq_data

        # re-orient acq_data as a list of dicts, to combine with events
        records = _util.convert_dict_of_lists_to_records(acq_data)
        for e in img_events:
            records.append({_util.TIME_KEY: e.time / 1000, "Events": e.description})

        # sort by time
        records.sort(key=lambda x: x.get(_util.TIME_KEY, 0))

        if orient == "dict":
            return _util.convert_records_to_dict_of_dicts(records, null_val=null_value)
        elif orient == "list":
            return _util.convert_records_to_dict_of_lists(records, null_val=null_value)
        return records

    def rois(self) -> list[ROI]:
        key = b"CustomData|RoiMetadata_v1!"
        if key not in self.chunkmap:
            return []  # pragma: no cover

        data = self._decode_chunk(key)
        data = data.get("RoiMetadata_v1", {}).copy()
        dicts: list[dict] = []
        if "Global_Size" in data:
            dicts.extend(data[f"Global_{i}"] for i in range(data["Global_Size"]))
        if "2PerMPoint_Size" in data:
            for i in range(data.get("2PerMPoint_Size", 0)):
                item: dict = data[f"2PerMPoint_{i}"]
                dicts.extend(item[str(idx)] for idx in range(item.get("Size", 0)))

        try:
            return [ROI._from_meta_dict(d) for d in dicts]
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Could not parse ROI metadata: {e}") from e

    def unstructured_metadata(
        self,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        keys = {
            k.decode()[:-1]
            for k in self.chunkmap
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
                output[key] = self._decode_chunk(name, strip_prefix=strip_prefix)
            except Exception:  # pragma: no cover
                output[key] = self._load_chunk(name)
        return output
