from __future__ import annotations

import mmap
import os
import warnings
from io import BufferedReader
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from nd2 import structures
from nd2._clx_lite import json_from_clx_lite_variant
from nd2._clx_xml import json_from_clx_variant
from nd2._pysdk._chunk_decode import (
    _read_nd2_chunk,
    get_chunkmap,
    get_version,
)
from nd2._pysdk._parse import (
    load_attributes,
    load_exp_loop,
    load_frame_metadata,
    load_global_metadata,
    load_metadata,
    load_text_info,
)

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path
    from typing import Any

    from typing_extensions import TypeAlias

    from ._chunk_decode import ChunkMap
    from ._parse import GlobalMetadata

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class ND2Reader:
    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self._filename = path
        self._fh: BufferedReader | None = None
        self._mmap: mmap.mmap | None = None
        self._chunkmap: ChunkMap = {}

        self._version: tuple[int, int] | None = None
        self._attributes: structures.Attributes | None = None
        self._experiment: list[structures.ExpLoop] | None = None
        self._text_info: structures.TextInfo | None = None
        self._metadata: structures.Metadata | None = None
        self._raw_attributes: dict | None = None
        self._raw_experiment: dict | None = None
        self._raw_text_info: dict | None = None
        self._raw_image_metadata: dict | None = None
        self._global_metadata: GlobalMetadata | None = None
        self._frame_offsets_: dict[int, int] | None = None
        self._raw_frame_shape_: tuple[int, ...] | None = None
        self._dtype_: np.dtype | None = None
        self._strides_: tuple[int, ...] | None = None
        self._frame_times: list[float] | None = None

        self.open()

    def open(self) -> None:
        if self._fh is None:
            self._fh = open(self._filename, "rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def chunkmap(self) -> ChunkMap:
        if not self._chunkmap:
            if self._fh is None:
                raise OSError("File not open")
            self._chunkmap = get_chunkmap(self._fh)
        return self._chunkmap

    @property
    def _frame_offsets(self) -> dict[int, int]:
        if self._frame_offsets_ is None:
            DEFAULT_SHIFT = 4072
            offsets = [
                (int(key[13:-1]), pos)
                for key, (pos, _) in sorted(self.chunkmap.items())
                if key.startswith(b"ImageDataSeq|")
            ]
            # if validate_frames:
            #     return _validate_frames(fh, image_map, kbrange=search_window), meta_ma
            self._frame_offsets_ = {f: int(o + 24 + DEFAULT_SHIFT) for f, o in offsets}
        return self._frame_offsets_

    @property
    def attributes(self) -> structures.Attributes:
        if self._attributes is None:
            k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            self._raw_attributes = attrs
            raw_meta = self._get_raw_image_metadata()  # ugly
            n_channels = cast(int, raw_meta.get("sPicturePlanes", {}).get("uiCount", 1))
            self._attributes = load_attributes(attrs, n_channels)
        return self._attributes

    def _load_chunk(self, name: bytes) -> bytes:
        if self._fh is None:
            raise OSError("File not open")
        offset, _ = self.chunkmap[name]
        # TODO: there's a possibility of speed up here since we're rereading the header
        return _read_nd2_chunk(self._fh, offset)

    def _decode_chunk(self, name: bytes, strip_prefix: bool = True) -> dict:
        data = self._load_chunk(name)
        if self.version < (3, 0):
            return cast("dict", json_from_clx_variant(data, strip_prefix=strip_prefix))
        return json_from_clx_lite_variant(data, strip_prefix=strip_prefix)

    @property
    def version(self) -> tuple[int, int]:
        if self._version is None:
            try:
                self._version = get_version(self._fh or self._filename)
            except Exception:
                self._version = (-1, -1)
                raise
        return self._version

    def _get_raw_image_metadata(self) -> dict:
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
                self._raw_image_metadata = meta
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
                julian_day = os.stat(self._filename).st_ctime / 86400.0 + 2440587.5
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
                self._raw_text_info = info
                self._text_info = load_text_info(info)
        return self._text_info

    def experiment(self) -> list[structures.ExpLoop]:
        if not self._experiment:
            k = b"ImageMetadataLV!" if self.version >= (3, 0) else b"ImageMetadata!"
            if k not in self.chunkmap:
                self._experiment = []
            else:
                exp = self._decode_chunk(k, strip_prefix=False)
                exp = exp.get("SLxExperiment", exp)  # for v3 only
                self._raw_experiment = exp
                loops = load_exp_loop(0, exp)
                self._experiment = [structures._Loop.create(x) for x in loops]
        return self._experiment

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

    # -----------

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
        return int(np.prod([x.count for x in self.experiment()]))

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

    def _dtype(self):
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

    def _actual_frame_shape(self):
        attr = self.attributes
        return (
            attr.heightPx,
            attr.widthPx,
            attr.channelCount or 1,
            attr.componentCount // (attr.channelCount or 1),
        )

    def _get_meta_chunk(self, key: str) -> bytes:
        # deprecated
        return self._load_chunk(f"{key}!".encode())

    @property
    def _meta_map(self) -> dict[str, int]:
        # deprecated
        return {k.decode()[:-1]: v for k, (v, _) in self.chunkmap.items()}

    def _custom_data(self) -> dict[str, Any]:
        return {
            k.decode()[14:-1]: json_from_clx_variant(self._load_chunk(k))
            for k in self.chunkmap
            if k.startswith(b"CustomDataVar|")
        }

    # probably a temporary method, for testing
    def _raw_meta(self) -> dict:
        k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
        attrs = self._decode_chunk(k, strip_prefix=False) if k in self.chunkmap else {}
        attrs = attrs.get("SLxImageAttributes", attrs)

        k = b"ImageTextInfoLV!" if self.version >= (3, 0) else b"ImageTextInfo!"
        ti = self._decode_chunk(k, strip_prefix=False) if k in self.chunkmap else {}
        ti = ti.get("SLxImageTextInfo", ti)

        k = b"ImageMetadataLV!" if self.version >= (3, 0) else b"ImageMetadata!"
        exp = self._decode_chunk(k, strip_prefix=False) if k in self.chunkmap else {}
        exp = exp.get("SLxExperiment", exp)

        k = (
            b"ImageMetadataSeqLV|0!"
            if self.version >= (3, 0)
            else b"ImageMetadataSeq|0!"
        )
        meta = self._decode_chunk(k, strip_prefix=False) if k in self.chunkmap else {}
        meta = meta.get("SLxPictureMetadata", meta)

        return {
            "Attributes": attrs,
            "Experiment": exp,
            "Metadata": meta,
            "TextInfo": ti,
        }

    # TODO: merge with decode_chunk
    def _decoded_custom_data_chunk(
        self, key: bytes, default: Any = None, strip_prefix: bool = False
    ) -> dict:
        k = b"CustomDataVar|" + key
        if k not in self.chunkmap:
            return default

        bytes_ = self._load_chunk(k)
        return cast("dict", json_from_clx_variant(bytes_, strip_prefix=strip_prefix))

    def recorded_data(self) -> dict[str, np.ndarray | Sequence]:
        """Return tabular data recorded for each frame of the experiment.

        This method returns a dict of equal-length sequences (passable to
        pd.DataFrame()). It matches the tabular data reported in the Image Properties >
        Recorded Data tab of the NIS Viewer.

        (There will be a column for each tag in the `CustomDataV2_0` section of
        `ND2File.custom_data`)

        Legacy ND2 files are not supported.
        """
        cd = self._decoded_custom_data_chunk(b"CustomDataV2_0!", {})
        if not cd:
            return {}

        if "CustomTagDescription_v1.0" not in cd:
            warnings.warn(
                "Could not find 'CustomTagDescription_v1' tag, please open an issue "
                "with this nd2 file at https://github.com/tlambert03/nd2/issues/new",
                stacklevel=2,
            )
            return {}

        data: dict[str, np.ndarray | Sequence] = {}
        frame_times = self._cached_frame_times()
        if frame_times:
            data["Time [s]"] = [x / 1000 for x in frame_times]

        experiment = self.experiment()
        for i, loop in enumerate(experiment):
            if not isinstance(loop, structures.ZStackLoop):
                continue

            z_loop = loop
            z_positions = [
                z_loop.parameters.stepUm * (i - z_loop.parameters.homeIndex)
                for i in range(z_loop.count)
            ]
            if not z_loop.parameters.bottomToTop:
                z_positions = list(reversed(z_positions))

            def _seq_z_pos(seq_index: int, z_idx=i, z_positions=z_positions) -> float:
                """Convert a sequence index to a coordinate tuple."""
                for n, _loop in enumerate(experiment):
                    if n == z_idx:
                        return z_positions[seq_index % _loop.count]
                    seq_index //= _loop.count
                raise ValueError("Invalid sequence index or z_idx")

            seq_count = self.attributes.sequenceCount
            data["Z-Series"] = np.array([_seq_z_pos(i) for i in range(seq_count)])

        # tags will be a dict of dicts: eg:
        # {
        #     'Tag0': {'ID': 'Camera_ExposureTime1', 'Type': 3, ... },
        #     'Tag1': {'ID': 'PFS_OFFSET', 'Type': 2, 'Group': 0, 'Size': 1, ...},
        #     'Tag2': {'ID': 'PFS_STATUS', 'Type': 2, 'Group': 0, 'Size': 1, ...},
        # }
        # FIXME: technically, it is possible to have multiple tags with the same Desc
        # (e.g. for IDs PFS_OFFSET and Z2). In the current implementation, the
        # 2nd tag will overwrite the first one.
        for tag in cd["CustomTagDescription_v1.0"].values():
            if tag["Type"] == 1:
                warnings.warn(
                    f"{tag['Desc']!r} column skipped: "
                    "(parsing string data is not yet implemented)",
                    stacklevel=2,
                )
                continue
            col_header = f"{tag['Desc']}"
            if tag["Unit"]:
                col_header += f" [{tag['Unit']}]"

            buffer = self._load_chunk(f"CustomData|{tag['ID']}!".encode())
            dtype = {3: np.float64, 2: np.int32}[tag["Type"]]
            data[col_header] = np.frombuffer(buffer, dtype=dtype, count=tag["Size"])

        return data
