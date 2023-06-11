from __future__ import annotations

from io import BufferedReader
from typing import TYPE_CHECKING

from nd2 import structures
from nd2._pysdk._chunk_decode import (
    _read_nd2_chunk,
    get_version,
    json_from_clx_lite_variant,
    load_chunkmap,
)
from nd2._pysdk._parse import (
    load_attributes,
    load_exp_loop,
    load_global_metadata,
    load_metadata,
    load_text_info,
)

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any

    from typing_extensions import TypeAlias

    from ._chunk_decode import ChunkMap
    from ._parse import GlobalMetadata

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class LimFile:
    _filename: str
    _fh: BufferedReader | None
    _version: tuple[int, int] | None = None
    _chunkmap: ChunkMap
    _attributes: structures.Attributes | None = None
    _experiment: list[structures.ExpLoop] | None = None
    _text_info: structures.TextInfo | None = None
    _metadata: structures.Metadata | None = None
    _raw_attributes: dict | None = None
    _raw_experiment: dict | None = None
    _raw_text_info: dict | None = None
    _raw_image_metadata: dict | None = None
    _global_metadata: GlobalMetadata | None = None

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._fh: BufferedReader | None = None
        self._chunkmap = {}

    def open(self) -> None:
        self._fh = open(self._filename, "rb")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> LimFile:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def version(self) -> tuple[int, int]:
        if self._version is None:
            try:
                self._version = get_version(self._fh or self._filename)
            except Exception:
                self._version = (-1, -1)
                raise
        return self._version

    @property
    def chunkmap(self) -> ChunkMap:
        if not self._chunkmap:
            if self._fh is None:
                raise OSError("File not open")
            self._chunkmap = load_chunkmap(self._fh)
        return self._chunkmap

    def _load_chunk(self, name: bytes) -> bytes:
        if self._fh is None:
            raise OSError("File not open")
        offset, _ = self.chunkmap[name]
        # TODO: there's a possibility of speed up here since we're rereading the header
        return _read_nd2_chunk(self._fh, offset)

    def _decode_chunk(self, name: bytes, strip_prefix: bool = True) -> dict:
        data = self._load_chunk(name)
        if self.version < (3, 0):
            from nd2._clx_xml import json_from_clx_variant

            return json_from_clx_variant(data)
        return json_from_clx_lite_variant(data, strip_prefix=strip_prefix)

    @property
    def attributes(self) -> structures.Attributes:
        if not self._attributes:
            k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            self._raw_attributes = attrs
            self._attributes = load_attributes(attrs, 1)
        return self._attributes

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
                breakpoint()
        return self._experiment

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

    def metadata(self) -> structures.Metadata:
        if not self._metadata:
            self._metadata = load_metadata(
                raw_meta=self._get_raw_image_metadata(),
                global_meta=self.global_metadata(),
            )
        return self._metadata

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

    def global_metadata(self) -> GlobalMetadata:
        if not self._global_metadata:
            self._global_metadata = load_global_metadata(
                attrs=self.attributes,
                raw_meta=self._get_raw_image_metadata(),
                exp_loops=self.experiment(),
                text_info=self.text_info(),
            )
        return self._global_metadata
