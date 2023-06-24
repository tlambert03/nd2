from __future__ import annotations

import mmap
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence, cast

import numpy as np

from nd2 import structures
from nd2._clx_lite import json_from_clx_lite_variant
from nd2._clx_xml import json_from_clx_variant
from nd2._pysdk._chunk_decode import (
    _robustly_read_named_chunk,
    get_chunkmap,
    get_version,
    read_nd2_chunk,
)
from nd2._pysdk._parse import (
    load_attributes,
    load_events,
    load_experiment,
    load_frame_metadata,
    load_global_metadata,
    load_metadata,
    load_text_info,
)
from nd2._util import TIME_KEY, Z_SERIES_KEY

if TYPE_CHECKING:
    import datetime
    from io import BufferedReader
    from os import PathLike
    from typing import Any

    from typing_extensions import TypeAlias

    from ._chunk_decode import ChunkMap
    from ._sdk_types import (
        GlobalMetadata,
        RawAttributesDict,
        RawExperimentDict,
        RawExperimentRecordDict,
        RawMetaDict,
        RawTagDict,
        RawTextInfoDict,
    )

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class ND2Reader:
    def __init__(
        self, path: str | Path, validate_frames: bool = False, search_window: int = 100
    ) -> None:
        self._path = Path(path)
        self._fh: BufferedReader | None = None
        self._mmap: mmap.mmap | None = None
        self._chunkmap: ChunkMap = {}
        self._cached_decoded_chunks: dict[bytes, Any] = {}
        self._error_radius: int | None = (
            search_window * 1000 if validate_frames else None
        )

        self._version: tuple[int, int] | None = None
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

        self.open()

    def open(self) -> None:
        if self._fh is None:
            self._fh = open(self._path, "rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self) -> ND2Reader:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

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
            if self._fh is None:
                raise OSError("File not open")
            self._chunkmap = get_chunkmap(self._fh, error_radius=self._error_radius)
        return self._chunkmap

    @property
    def attributes(self) -> structures.Attributes:
        """Load and return the image attributes."""
        if self._attributes is None:
            k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            self._raw_attributes = cast("RawAttributesDict", attrs)
            raw_meta = self._get_raw_image_metadata()  # ugly
            n_channels = raw_meta.get("sPicturePlanes", {}).get("uiCount", 1)
            self._attributes = load_attributes(self._raw_attributes, n_channels)
        return self._attributes

    def _load_chunk(self, name: bytes) -> bytes:
        """Load raw bytes from a specific chunk in the chunkmap.

        `name` must be a valid key in the chunkmap.
        """
        if self._fh is None:
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
            elif self.version < (3, 0):
                decoded = json_from_clx_variant(data, strip_prefix=strip_prefix)
            else:
                decoded = json_from_clx_lite_variant(data, strip_prefix=strip_prefix)
            self._cached_decoded_chunks[name] = decoded
        return self._cached_decoded_chunks[name]

    @property
    def version(self) -> tuple[int, int]:
        """Return the file format version as a tuple of ints."""
        if self._version is None:
            try:
                self._version = get_version(self._fh or self._path)
            except Exception:
                self._version = (-1, -1)
                raise
        return self._version

    def _get_raw_image_metadata(self) -> RawMetaDict:
        if not self._raw_image_metadata:
            k = (
                b"ImageMetadataSeqLV|0!"
                if self.version >= (3, 0)
                else b"ImageMetadataSeq|0!"
            )
            if k not in self.chunkmap:
                self._raw_image_metadata = {}
            else:
                meta = self._decode_chunk(k, strip_prefix=False)
                meta = meta.get("SLxPictureMetadata", meta)  # for v3 only
                self._raw_image_metadata = cast("RawMetaDict", meta)
        return self._raw_image_metadata

    def _cached_global_metadata(self) -> GlobalMetadata:
        if not self._global_metadata:
            self._global_metadata = load_global_metadata(
                attrs=self.attributes,
                raw_meta=self._get_raw_image_metadata(),
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
                raw_meta=self._get_raw_image_metadata(),
                global_meta=self._cached_global_metadata(),
            )
        return self._metadata

    def frame_metadata(self, seq_index: int) -> structures.FrameMetadata:
        frame_time = self._cached_frame_times()[seq_index]
        global_meta = self._cached_global_metadata()
        return load_frame_metadata(
            global_meta, self.metadata(), self.experiment(), frame_time, seq_index
        )

    def text_info(self) -> structures.TextInfo:
        if self._text_info is None:
            k = b"ImageTextInfoLV!" if self.version >= (3, 0) else b"ImageTextInfo!"
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
            k = b"ImageMetadataLV!" if self.version >= (3, 0) else b"ImageMetadata!"
            if k not in self.chunkmap:
                self._experiment = []
            else:
                exp = self._decode_chunk(k, strip_prefix=False)
                exp = exp.get("SLxExperiment", exp)  # for v3 only
                self._raw_experiment = cast("RawExperimentDict", exp)
                self._experiment = load_experiment(0, self._raw_experiment)
        return self._experiment

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

    def channel_names(self) -> list[str]:
        return [c.channel.name for c in self.metadata().channels or []]

    def _coords_from_seq_index(self, seq_index: int) -> tuple[int, ...]:
        """Convert a sequence index to a coordinate tuple."""
        coords: list[int] = []
        for loop in self.experiment():
            coords.append(seq_index % loop.count)
            seq_index //= loop.count

        return tuple(coords)

    def _coord_size(self) -> int:
        return len(self.experiment())

    def _coord_info(self) -> list[tuple[int, str, int]]:
        return [(i, x.type, x.count) for i, x in enumerate(self.experiment())]

    def _seq_count(self) -> int:
        # this differs from self.attributes.SequenceCount in that it
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

    def _read_image(self, index: int) -> np.ndarray:
        """Read a chunk directly without using SDK."""
        if index > self._seq_count():
            raise IndexError(f"Frame out of range: {index}")
        if not self._fh:
            raise ValueError("Attempt to read from closed nd2 file")
        offset = self._frame_offsets.get(index, None)
        if offset is None:
            return self._missing_frame(index)

        if self.attributes.compressionType == "lossless":
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
        import zlib

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
            attr = self.attributes
            ncomp = attr.componentCount
            self._raw_frame_shape_ = (
                attr.heightPx,
                (attr.widthBytes or 0) // self._bytes_per_pixel() // ncomp,
                attr.channelCount or 1,
                ncomp // (attr.channelCount or 1),
            )
        return self._raw_frame_shape_

    def _bytes_per_pixel(self) -> int:
        return self.attributes.bitsPerComponentInMemory // 8

    def _dtype(self) -> np.dtype:
        if self._dtype_ is None:
            a = self.attributes
            d = a.pixelDataType[0] if a.pixelDataType else "u"
            self._dtype_ = np.dtype(f"{d}{self._bytes_per_pixel()}")
        return self._dtype_

    @property
    def _strides(self) -> tuple[int, ...] | None:
        if self._strides_ is None:
            a = self.attributes
            widthP = a.widthPx
            widthB = a.widthBytes
            if not (widthP and widthB):
                self._strides_ = None
            else:
                bypc = self._bytes_per_pixel()
                compCount = a.componentCount
                array_stride = widthB - (bypc * widthP * compCount)
                if array_stride == 0:
                    self._strides_ = None
                else:
                    self._strides_ = (
                        array_stride + widthP * bypc * compCount,
                        compCount * bypc,
                        compCount // (a.channelCount or 1) * bypc,
                        bypc,
                    )
        return self._strides_

    def _actual_frame_shape(self) -> tuple[int, ...]:
        attr = self.attributes
        return (
            attr.heightPx,
            attr.widthPx or 1,
            attr.channelCount or 1,
            attr.componentCount // (attr.channelCount or 1),
        )

    def _custom_data(self) -> dict[str, Any]:
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
        }
        """
        data: dict[str, np.ndarray | Sequence] = {}
        frame_times = self._cached_frame_times()
        if frame_times:
            data[TIME_KEY] = [x / 1000 for x in frame_times]

        # FIXME: this whole thing is dumb... must be a better way
        experiment = self.experiment()
        for i, z_loop in enumerate(experiment):
            if not isinstance(z_loop, structures.ZStackLoop):
                continue

            z_positions = [
                z_loop.parameters.stepUm * (i - z_loop.parameters.homeIndex)
                for i in range(z_loop.count)
            ]
            if not z_loop.parameters.bottomToTop:
                z_positions.reverse()

            def _seq_z_pos(
                seq_index: int, z_idx: int = i, _zp: list[float] = z_positions
            ) -> float:
                """Convert a sequence index to a coordinate tuple."""
                for n, _loop in enumerate(experiment):
                    if n == z_idx:
                        return _zp[seq_index % _loop.count]
                    seq_index //= _loop.count
                raise ValueError("Invalid sequence index or z_idx")

            seq_count = self._seq_count()
            data[Z_SERIES_KEY] = np.array([_seq_z_pos(i) for i in range(seq_count)])

        return data  # type: ignore [return-value]

    def _custom_tags(self) -> dict[str, Any]:
        """Return tags mentioned in in CustomDataVar|CustomDataV2_0.

        This is a dict of {header: [values]}, where
        len([values]) == self.attributes.sequenceCount

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
            if tag["Type"] == 1:
                warnings.warn(
                    f"{tag['Desc']!r} column skipped: "
                    "(parsing string data is not yet implemented).  Please open an "
                    "issue with this nd2 file at "
                    "https://github.com/tlambert03/nd2/issues/new",
                    stacklevel=2,
                )
                continue

            col_header = tag["Desc"]
            if col_header in data:  # pragma: no cover
                # sourcery skip: hoist-if-from-if
                col_header = tag["ID"]
                if col_header in data:
                    col_header = f"{tag['Desc']} ({tag['ID']})"

            if tag["Unit"].strip():
                col_header += f" [{tag['Unit']}]"

            buffer_ = self._load_chunk(f"CustomData|{tag['ID']}!".encode())
            dtype = {3: np.float64, 2: np.int32}[tag["Type"]]
            data[col_header] = np.frombuffer(buffer_, dtype=dtype, count=tag["Size"])

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
        from nd2._util import parse_time

        date = self.text_info().get("date")
        if date:
            try:
                return parse_time(date)
            except ValueError:
                return date

        time = self._cached_global_metadata().get("time", {})
        jdn = time.get("absoluteJulianDayNumber")
        if jdn:
            from nd2._util import jdn_to_datetime_utc

            return jdn_to_datetime_utc(jdn)
        return None
