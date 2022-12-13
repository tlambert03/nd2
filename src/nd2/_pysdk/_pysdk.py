from __future__ import annotations

from io import BufferedReader
from typing import TYPE_CHECKING, cast

from nd2 import structures
from nd2._pysdk._structures import loadExperiment
from nd2._pysdk._util import (
    _read_nd2_chunk,
    decode_CLxLiteVariant_json,
    get_version,
    load_chunkmap,
)
from typing_extensions import Literal

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any

    from typing_extensions import TypeAlias

    from ._util import ChunkMap

    StrOrBytesPath: TypeAlias = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk: TypeAlias = tuple[int, int, int, bytes, bytes]


class LimFile:
    _filename: str
    _fh: BufferedReader | None
    _attributes: structures.Attributes | None = None
    _metadata: structures.Metadata | None = None
    _version: tuple[int, int] | None = None
    _chunkmap: ChunkMap = {}

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
            self._version = get_version(self._fh or self._filename)
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
        return decode_CLxLiteVariant_json(data, strip_prefix=strip_prefix)

    @property
    def attributes(self) -> structures.Attributes:
        if not self._attributes:
            # FIXME: this key will depend on version
            attrs = self._decode_chunk(b"ImageAttributesLV!", strip_prefix=False)
            attrs = attrs["SLxImageAttributes"]
            bpc = attrs["uiBpcInMemory"]
            _ecomp: int = attrs.get("eCompression", 2)
            comp_type: Literal["lossless", "lossy", "none"] | None
            if 0 <= _ecomp < 2:
                comp_type = cast(
                    'Literal["lossless", "lossy", "none"]',
                    [
                        "lossless",
                        "lossy",
                        "none",
                    ][_ecomp],
                )
                comp_level = attrs.get("dCompressionParam")
            else:
                comp_type = None
                comp_level = None

            tile_width = attrs.get("uiTileWidth", 0)
            tile_height = attrs.get("uiTileHeight", 0)
            if (tile_width <= 0 or tile_width == attrs["uiWidth"]) and (
                tile_height <= 0 or tile_height == attrs["uiHeight"]
            ):
                tile_width = tile_height = None

            self._attributes = structures.Attributes(
                bitsPerComponentInMemory=bpc,
                bitsPerComponentSignificant=attrs["uiBpcSignificant"],
                componentCount=attrs["uiComp"],
                heightPx=attrs["uiHeight"],
                pixelDataType="float" if bpc == 32 else "unsigned",
                sequenceCount=attrs["uiSequenceCount"],
                widthBytes=attrs["uiWidthBytes"],
                widthPx=attrs["uiWidth"],
                compressionLevel=comp_level,
                compressionType=comp_type,
                tileHeightPx=tile_height,
                tileWidthPx=tile_width,
                # channelCount=attrs[""],  # this comes from metadata
            )
        return self._attributes

    def experiment(self) -> list[structures.ExpLoop]:
        if not self._metadata:
            exp = self._decode_chunk(b"ImageMetadataLV!", strip_prefix=False)
            exp = exp["SLxExperiment"]
            sequence_count = self.attributes.sequenceCount
            self._metadata = loadExperiment(exp, sequence_count)
        return self._metadata


if __name__ == "__main__":
    import sys
    import time

    from rich import print

    fname = sys.argv[1]

    t = time.perf_counter()
    with LimFile(fname) as lim:
        print("version", lim.version)
        print(lim.attributes)
        print(lim.experiment())

    # t2 = time.perf_counter()
    # print(f"Time: {t2 - t:.4f}s")
    # print("----------")
    # t = time.perf_counter()
    # with nd2.ND2File(fname) as f:
    #     print(f.attributes)
    #     print(f.experiment)
    #     ...

    t2 = time.perf_counter()
    print(f"Time: {t2 - t:.4f}s")
