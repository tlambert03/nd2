from __future__ import annotations

from io import BufferedReader
from typing import TYPE_CHECKING, cast

from nd2 import structures
from nd2._pysdk._decode import (
    _read_nd2_chunk,
    decode_CLxLiteVariant_json,
    decode_xml,
    get_version,
    load_chunkmap,
)
from nd2._pysdk._parse import load_exp_loop, load_attributes, load_metadata
from typing_extensions import Literal

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any

    from typing_extensions import TypeAlias

    from ._decode import ChunkMap

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class LimFile:
    _filename: str
    _fh: BufferedReader | None
    _version: tuple[int, int] | None = None
    _chunkmap: ChunkMap
    _attributes: structures.Attributes | None = None
    _experiment: list[structures.ExpLoop] | None = None
    _metadata: structures.Metadata | None = None

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
            return decode_xml(data)
        return decode_CLxLiteVariant_json(data, strip_prefix=strip_prefix)

    @property
    def attributes(self) -> structures.Attributes:
        if not self._attributes:
            k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            self._attributes = load_attributes(attrs)
        return self._attributes

    def experiment(self) -> list[structures.ExpLoop]:
        if not self._experiment:
            k = b"ImageMetadataLV!" if self.version >= (3, 0) else b"ImageMetadata!"
            if k not in self.chunkmap:
                self._experiment = []
            else:
                exp = self._decode_chunk(k, strip_prefix=False)
                exp = exp.get("SLxExperiment", exp)
                loops = load_exp_loop(0, exp)
                self._experiment = [structures._Loop.create(x) for x in loops]
        return self._experiment

    def metadata(self) -> structures.Metadata:
        if not self._metadata:

            k = (
                b'ImageMetadataSeqLV|0!'
                if self.version >= (3, 0)
                else b"ImageMetadataSeq|0!"
            )
            meta = self._decode_chunk(k, strip_prefix=False)
            meta = meta.get("SLxPictureMetadata", meta)  # for v3 only
            self._metadata = load_metadata(meta)
            breakpoint()
        return self._metadata
