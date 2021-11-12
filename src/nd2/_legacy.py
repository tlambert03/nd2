import re
import struct
import threading
from pathlib import Path
from typing import BinaryIO, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np

from . import structures as strct
from ._util import VoxelSize, dims_from_description
from ._xml import parse_xml_block

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore


I4s = struct.Struct(">I4s")
JP2_MAP_CHUNK = struct.Struct("<4s4sI")
IHDR = struct.Struct(">iihBB")  # yxc-dtype in jpeg 2000


class LegacyND2Reader:
    _fh: BinaryIO

    def __init__(self, path: Union[Path, str]):
        self._path = str(path)
        self._fh = open(self._path, mode="rb")
        length, box_type = I4s.unpack(self._fh.read(I4s.size))
        if length != 12 and box_type == b"jP  ":
            raise ValueError("File not recognized as Legacy ND2 (JPEG2000) format.")
        self._chunkmap = legacy_nd2_chunkmap(self._fh)
        self.lock = threading.RLock()

    def open(self) -> None:
        if self.closed:
            self._fh = open(self._path, mode="rb")

    def close(self) -> None:
        if not self.closed:
            self._fh.close()

    @property
    def closed(self) -> bool:
        return self._fh.closed

    def __enter__(self) -> "LegacyND2Reader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @cached_property
    def ddim(self) -> dict:
        return dims_from_description(self.text_info().get("description"))

    def experiment(self) -> List[strct.ExpLoop]:
        meta = self._metadata
        exp = []
        if "LoopNo00" in meta:
            # old style:
            for i, (k, v) in enumerate(meta.items()):
                if k != "Version":
                    loop = self._make_loop(v, i)
                    if loop:
                        exp.append(loop)
        else:
            i = 0
            while meta:
                # ugh... another hack for weird metadata
                if len(meta) == 1:
                    meta = list(meta.values())[0]
                    if meta and isinstance(meta, list):
                        meta = meta[0]
                loop = self._make_loop(meta, i)
                if loop:
                    exp.append(loop)
                meta = meta.get("NextLevelEx")  # type: ignore
                i += 1
        return exp

    def _coord_info(self) -> List[Tuple[int, str, int]]:
        return [(i, l.type, l.count) for i, l in enumerate(self.experiment())]

    def _make_loop(
        self, meta_level: dict, nest_level: int = 0
    ) -> Optional[strct.ExpLoop]:
        """converts an old style metadata loop dict to a new ExpLoop structure."""
        type_ = meta_level.get("Type")
        params: dict = meta_level["LoopPars"]
        if type_ == 2:  # XYPosLoop
            poscount = len(params["PosX"])
            # empirically, it appears that some files list more positions in the
            # metadata than are actually recorded in the textinfo -> description.
            # NIS viewer seems to agree more with the description
            count = self.ddim.get("S") or params["Count"]
            points = []
            for i in range(count):
                idx = f"{poscount-i-1:05}"
                points.append(
                    strct.Position(
                        pfsOffset=params["PFSOffset"][idx],
                        stagePositionUm=strct.StagePosition(
                            x=params["PosX"][idx],
                            y=params["PosY"][idx],
                            z=params["PosZ"][idx],
                        ),
                    )
                )
            return strct.XYPosLoop(
                nestingLevel=nest_level,
                count=count,
                parameters=strct.XYPosLoopParams(
                    isSettingZ=params["UseZ"], points=points
                ),
            )
        if type_ == 8:  # TimeLoop
            # XXX strangely, I've seen files that seem to have a mult-phase
            # NETimeLoop, but use one of the Period...
            # try to find one that matches the dims in the description
            count = self.ddim.get("T")
            per = None
            if count is not None:
                for p in params["Period"].values():
                    if p["Count"] == count:
                        per = p
            # otherwise take the first period
            # XXX: this is definitely error prone.
            if per is None:
                per = next(iter(params["Period"].values()))
            return strct.TimeLoop(
                count=per["Count"],
                nestingLevel=nest_level,
                parameters=strct.TimeLoopParams(
                    startMs=per["Start"],
                    periodMs=per["Period"],
                    durationMs=per["Duration"],
                    periodDiff=strct.PeriodDiff(
                        avg=per.get("AvgPeriodDiff"),
                        max=per.get("MaxPeriodDiff"),
                        min=per.get("MinPeriodDiff"),
                    ),
                ),
            )
        if type_ == 4:
            return strct.ZStackLoop(
                count=params["Count"],
                nestingLevel=nest_level,
                parameters=strct.ZStackLoopParams(
                    homeIndex=params["ZHome"],
                    stepUm=params["ZStep"],
                    bottomToTop=params["ZLow"] < params["ZHigh"],
                ),
            )
        if type_ == 6:  # channel
            return None

        raise ValueError(f"unrecognized type: {type_}")

    @cached_property
    def attributes(self) -> strct.Attributes:
        head = self.header
        bpcim = head["bits_per_component"]
        bpcs = self._advanced_image_attributes.get("SignificantBits", bpcim)
        widthPx = head.get("columns")

        try:
            picplanes = self._get_xml_dict(b"VIMD", 0)["PicturePlanes"]
            nC = picplanes["Count"]
            compCount = picplanes["CompCount"]
        except Exception:
            compCount = nC = 1

        return strct.Attributes(
            bitsPerComponentInMemory=bpcim,
            bitsPerComponentSignificant=bpcs,
            componentCount=compCount,
            heightPx=head.get("rows", -1),
            widthPx=widthPx,
            widthBytes=widthPx * bpcim // 8 * compCount,
            pixelDataType="unsigned",
            sequenceCount=len(self._chunkmap[b"VCAL"]),
            compressionLevel=head.get("compression"),
            channelCount=nC,
        )

    def _get_xml_dict(self, key: bytes, index=0) -> dict:
        try:
            bxml = self._read_chunk(self._chunkmap[key][index])
            return parse_xml_block(bxml)
        except KeyError:
            return {}

    @cached_property
    def events(self) -> dict:
        return self._get_xml_dict(b"IEVE")

    # def sizes(self):
    #     attrs = cast(Attributes, self.attributes)

    def text_info(self) -> dict:
        d = self._get_xml_dict(b"TINF")
        for i in d.get("TextInfoItem", []):
            txt = i.get("Text", "")
            if txt.startswith("Metadata:"):
                return {"description": txt}
        return {}

    @cached_property
    def _advanced_image_attributes(self) -> dict:
        return self._get_xml_dict(b"ARTT").get("AdvancedImageAttributes", {})

    @cached_property
    def _metadata(self) -> dict:
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

    def metadata(self) -> dict:
        return self._metadata

    @property
    def calibration(self) -> dict:
        return self._get_xml_dict(b"ACAL")

    def _read_chunk(self, pos) -> bytes:
        with self.lock:
            self._fh.seek(pos)
            length, box_type = I4s.unpack(self._fh.read(I4s.size))
            return self._fh.read(length - I4s.size)

    def _read_image(self, index: int) -> np.ndarray:
        try:
            from imagecodecs import jpeg2k_decode
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n"
                f"Reading legacy format nd2 {self._fh.name!r} requires imagecodecs.\n"
                "Please install with `pip install imagecodecs`."
            )

        data = []
        cc = self.attributes.channelCount or 1
        for i in range(cc):
            d = self._read_chunk(self._chunkmap[b"LUNK"][index * cc + i])
            data.append(jpeg2k_decode(d))
        return np.stack(data, axis=-1)

    def frame_metadata(self, index: int) -> dict:
        return {
            **self._get_xml_dict(b"VCAL", index),
            **self._get_xml_dict(b"VIMD", index),
        }

    def _scan_vimd(self):
        zs = set()
        xys = set()
        ts = set()
        cs = set()
        for i in range(len(self._chunkmap[b"VIMD"])):
            xml = self._get_xml_dict(b"VIMD", i)
            ts.add(xml["TimeMSec"])
            xys.add((xml["XPos"], xml["YPos"]))
            for p in xml["PicturePlanes"]["Plane"].values():
                cs.add(p["OpticalConfigName"])
                zs.add(p["OpticalConfigFull"]["ZPosition0"])
        return (zs, xys, ts, cs)

    def voxel_size(self) -> VoxelSize:

        z: Optional[float] = None
        d = self.text_info().get("description") or ""
        _z = re.search(r"Z Stack Loop: 5\s+-\s+Step\s+([.\d]+)", d)
        if _z:
            try:
                z = float(_z.groups()[0])
            except Exception:
                pass
        if z is None:
            for e in self.experiment():
                if e.type == "ZStackLoop":
                    z = e.parameters.stepUm
                    break
        xy = self._get_xml_dict(b"VIMD", 0).get("Calibration") or 1
        return VoxelSize(xy, xy, z or 1)

    def channel_names(self) -> List[str]:
        xml = self._get_xml_dict(b"VIMD", 0)
        return [p["OpticalConfigName"] for p in xml["PicturePlanes"]["Plane"].values()]

    def time_stamps(self) -> List[str]:
        xml = self._get_xml_dict(b"VIMD", 0)
        return [p["OpticalConfigName"] for p in xml["PicturePlanes"]["Plane"].values()]

    @cached_property
    def header(self) -> dict:
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

    def _custom_data(self) -> dict:
        return {}


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
