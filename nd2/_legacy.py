import io
import struct
from pathlib import Path
from typing import BinaryIO, DefaultDict, Dict, List, Union

from imagecodecs import jpeg2k_decode

from ._xml import parse_xml_block
from .structures import Attributes

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


I4s = struct.Struct(">I4s")
JP2_MAP_CHUNK = struct.Struct("<4s4sI")
IHDR = struct.Struct(">iihBB")  # yxc-dtype in jpeg 2000


class LegacyND2Reader:
    _fh: BinaryIO

    def __init__(self, file: Union[Path, str, io.BufferedReader]):
        self._fh = (
            file if isinstance(file, io.BufferedReader) else open(file, mode="rb")
        )
        length, box_type = I4s.unpack(self._fh.read(I4s.size))
        if length != 12 and box_type == b"jP  ":
            raise ValueError("File not recognized as Legacy ND2 (JPEG2000) format.")

        self._chunkmap = legacy_nd2_chunkmap(self._fh)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def experiment(self):
        # nT = self.metadata["LoopPars"]["uLoopPars"]["Count"]
        ...

    @property
    def attributes(self):
        fm = self.frame_meta(0)
        pics = fm.get("MetadataSeq", {}).get("PicturePlanes", {}).get("sPicturePlanes")
        head = self.header
        bpcim = head["bits_per_component"]
        bpcs = self._advanced_image_attributes.get("SignificantBits", bpcim)
        widthPx = head.get("columns")
        compCount = pics.get("CompCount") if pics else 1
        return Attributes(
            bitsPerComponentInMemory=bpcim,
            bitsPerComponentSignificant=bpcs,
            componentCount=compCount,
            heightPx=head.get("rows", -1),
            widthPx=widthPx,
            widthBytes=widthPx * bpcim // 8 * compCount,
            pixelDataType="unsigned",
            sequenceCount=len(self._chunkmap[b"VCAL"]),
            compressionLevel=head.get("compression"),
        )

    def _get_xml_dict(self, key: bytes, index=0) -> dict:
        try:
            bxml = self._read_chunk(self._chunkmap[key][index])
            return parse_xml_block(bxml)
        except KeyError:
            return {}

    @property
    def events(self):
        return self._get_xml_dict(b"IEVE")

    @property
    def text_info(self):
        return self._get_xml_dict(b"TINF")

    @property
    def _advanced_image_attributes(self):
        return self._get_xml_dict(b"ARTT").get("AdvancedImageAttributes", {})

    @property
    def metadata(self):
        meta = self._get_xml_dict(b"AIM1") or self._get_xml_dict(b"AIMD")
        version = ""
        meta.pop("UnknownData", None)
        while len(meta) == 1:
            key, val = meta.popitem()
            if "_V" in key:
                version = key.split("_V")[1]
            meta = val
            meta.pop("UnknownData", None)
        meta["Version"] = version
        return meta

    @property
    def calibration(self) -> dict:
        return self._get_xml_dict(b"ACAL")

    def _read_chunk(self, pos) -> bytes:
        self._fh.seek(pos)
        length, box_type = I4s.unpack(self._fh.read(I4s.size))
        return self._fh.read(length - I4s.size)

    def _read_image(self, index: int):
        data = self._read_chunk(self._chunkmap[b"LUNK"][index])
        return jpeg2k_decode(data)

    def frame_meta(self, index: int) -> dict:
        return {
            **self._get_xml_dict(b"VCAL", index),
            **self._get_xml_dict(b"VIMD", index),
        }

    @cached_property
    def header(self):
        try:
            pos = self._chunkmap[b"jp2h"][0]
        except (KeyError, IndexError):
            raise KeyError("No valid jp2h header found in file")
        self._fh.seek(pos + I4s.size + 4)  # 4 bytes for "label"
        if self._fh.read(4) != b"ihdr":
            raise KeyError("No valid ihdr header found in jp2h header")

        w, h, c, t, z = IHDR.unpack(self._fh.read(IHDR.size))
        return {
            "rows": w,
            "columns": h,
            "channels": c,
            "bits_per_component": t + 1,
            "compression": z,
        }


def legacy_nd2_chunkmap(fh: BinaryIO) -> Dict[bytes, List[int]]:
    fh.seek(-40, 2)
    sig, map_start = struct.unpack("<32sQ", fh.read())
    assert sig == b"LABORATORY IMAGING ND BOX MAP 00", "Not a legacy ND2 file"
    fh.seek(-map_start, 2)
    n_chunks = int.from_bytes(fh.read(4), "big")
    data = fh.read()
    d: DefaultDict[bytes, List[int]] = DefaultDict(list)
    for i in range(n_chunks):
        box_type, lim_type, offset = JP2_MAP_CHUNK.unpack_from(data, i * 16)
        if box_type in {b"jP  ", b"ftyp", b"jp2h"}:
            d[box_type].append(offset)
        else:
            d[lim_type].append(offset)
    return dict(d)


# def jp2_chunkmap(fh: io.BufferedReader) -> Dict[bytes, List[Tuple[int, int]]]:
#     """Retrieve chunk positions and shape from jpeg 2000 (legacy ND2) format

#     https://www.file-recovery.com/jp2-signature-format.htm

#     ISO/IEC 15444

#     The JPEG 2000 file format specification was based on the QuickTime
#     container format specification. A JPEG 2000 file is always big-endian.

#     JPEG 2000 files consist of consecutive chunks. Each chunk has 8 byte header:
#     4-byte chunk size (big-endian, high byte first) and 4-byte chunk type -
#     one of pre-defined signatures: "jP " or "jP2 ".

#     """
#     out = DefaultDict(list)
#     for box_type, pos, length in iter_jp2_chunks(fh):
#         out[box_type].append((pos, length))
#     return dict(out)


# def iter_jp2_chunks(fh: io.BufferedReader) -> Iterator[Tuple[bytes, int, int]]:
#     file_size = fh.seek(0, 2)
#     fh.seek(0)
#     pos = 0
#     while True:
#         length, box_type = I4s.unpack(fh.read(I4s.size))
#         yield box_type, pos, length
#         pos += length
#         if pos >= file_size:
#             break
#         fh.seek(pos)  # jump to next box
