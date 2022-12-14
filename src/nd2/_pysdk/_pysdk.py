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
from nd2._pysdk._parse import load_exp_loop
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
    _attributes: structures.Attributes | None = None
    _experiment: list[structures.ExpLoop] | None = None
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
        if self.version < (3, 0):
            return decode_xml(data)
        return decode_CLxLiteVariant_json(data, strip_prefix=strip_prefix)

    @property
    def attributes(self) -> structures.Attributes:
        if not self._attributes:
            # FIXME: this key will depend on version
            k = b"ImageAttributesLV!" if self.version >= (3, 0) else b"ImageAttributes!"
            attrs = self._decode_chunk(k, strip_prefix=False)
            attrs = attrs.get("SLxImageAttributes", attrs)  # for v3 only
            try:
                bpc = attrs["uiBpcInMemory"]
            except KeyError:
                breakpoint()
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


def sort_nested_dict(raw: dict) -> dict:
    if isinstance(raw, dict):
        return {k: sort_nested_dict(v) for k, v in sorted(raw.items())}
    elif isinstance(raw, list):
        return [sort_nested_dict(v) for v in raw]
    else:
        return raw


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import nd2
    from rich import print

    DATA = Path(__file__).parent.parent.parent.parent / "tests" / "data"

    if len(sys.argv) > 1:
        files = sys.argv[1:]
        verbose = True
    else:
        OK = {
            "compressed_lossless.nd2",
            "dims_rgb_t3p2c2z3x64y64.nd2",
            "karl_sample_image.nd2",
        }
        files = [str(p) for p in DATA.glob("*.nd2") if p.name not in OK]
        verbose = True

    for p in files:
        with LimFile(p) as lim:
            try:
                lim.version
            except Exception:
                continue
            with nd2.ND2File(p) as ndf:
                lima = lim.attributes
                nda = ndf.attributes
                lima = lima._replace(channelCount=nda.channelCount)
                nde = ndf.experiment
                lime = lim.experiment()
                nda = ndf.attributes
                if lime != nde or lima != nda:
                    print("---------------------")
                    print(f"{lim.version} {p}")
                    if verbose:
                        print("lime", lime)
                        print("nde", nde)
                        print(ndf.sizes)
                    else:
                        print(
                            f"mismatch {lim.version}",
                            p,
                        )
                else:
                    print(f"ok {lim.version}", p)
