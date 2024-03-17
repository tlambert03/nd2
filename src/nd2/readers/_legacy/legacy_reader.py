from __future__ import annotations

import contextlib
import re
import struct
import threading
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, DefaultDict, cast

import numpy as np

from nd2 import _util
from nd2 import structures as strct
from nd2._parse._legacy_xml import parse_xml_block
from nd2.readers.protocol import ND2Reader

try:
    from functools import cached_property
except ImportError:
    cached_property = property  # type: ignore

if TYPE_CHECKING:
    from collections import defaultdict
    from typing import Any, BinaryIO, Mapping, TypedDict

    from nd2._util import FileOrBinaryIO

    class RawExperimentLoop(TypedDict, total=False):
        Type: int
        ApplicationDesc: str
        UserDesc: str
        MeasProbesBase64: bytes
        LoopPars: LoopPars2 | LoopPars4 | LoopPars6 | LoopPars8
        ItemValid: dict
        AutoFocusBeforeLoop: dict
        CommandBeforeLoop: str
        CommandBeforeCapture: str
        CommandAfterCapture: str
        CommandAfterLoop: str
        ControlShutter: bool
        UsePFS: bool
        RepeatCount: int
        NextLevelEx: RawExperimentLoop
        ControlLight: bool
        Version: str

    class LoopPars8(TypedDict, total=False):
        Count: int
        PeriodCount: int
        Period: dict
        SubLoops: dict
        AutoFocusBeforePeriod: dict
        AutoFocusBeforeCapture: dict
        CommandBeforePeriod: dict
        CommandAfterPeriod: dict
        PeriodValid: dict

    class LoopPars2(TypedDict, total=False):
        Count: int
        PosX: dict
        PosY: dict
        UseZ: bool
        PosZ: dict
        PFSOffset: dict
        LargeImageRows: int
        LargeImageCols: int
        AutoFocusBeforeCapture: dict

    class LoopPars4(TypedDict, total=False):
        Count: int
        ZLow: float
        ZLowPFSOffset: float
        ZHigh: float
        ZHighPFSOffset: float
        ZHome: float
        ZStep: float
        Absolute: bool
        TriggeredPiezo: bool

    class LoopPars6(TypedDict, total=False):
        Count: int
        PlaneDesc: dict
        AutoFocus: dict
        CommandBeforeCapture: dict
        CommandAfterCapture: dict

    class FrameMetaDict(TypedDict, total=False):
        TimeMSec: float
        TimeAbsolute: float
        XPos: float
        YPos: float
        Row: int
        Col: int
        ZPos: float
        ZPosAbsolute: bool
        Angle: float
        PicturePlanes: PPlanesDict
        CameraSetting: dict
        TemperK: float
        Calibration: float
        Aspect: float
        Calibrated: bool
        ObjectiveName: str
        ObjectiveMag: float
        ObjectiveNA: float
        RefractIndex1: float
        RefractIndex2: float
        PinholeRadius: float
        Zoom: float
        ProjectiveMag: float
        PhysicalVar: dict
        PhysicalVarCount: int
        CustomData: str

    class PPlanesDict(TypedDict, total=False):
        Count: int
        CompCount: int
        Plane: dict[str, PlaneDict]
        Description: str

    class PlaneDict(TypedDict, total=False):
        CompCount: int
        OpticalConfigName: str
        OpticalConfigFull: dict
        Modality: int
        FluorescentProbe: dict
        FilterPath: dict
        CameraSetting: dict
        LampVoltage: float
        FadingCorr: float
        Color: int
        Description: str
        AcqTime: float


I4s = struct.Struct(">I4s")
JP2_MAP_CHUNK = struct.Struct("<4s4sI")
IHDR = struct.Struct(">iihBB")  # yxc-dtype in jpeg 2000


class LegacyReader(ND2Reader):
    HEADER_MAGIC = _util.OLD_HEADER_MAGIC

    def __init__(self, path: FileOrBinaryIO, error_radius: int | None = None) -> None:
        super().__init__(path, error_radius)
        self._attributes: strct.Attributes | None = None
        # super().__init__ called open()
        length, box_type = I4s.unpack(self._fh.read(I4s.size))  # type: ignore
        if length != 12 and box_type == b"jP  ":  # pragma: no cover
            raise ValueError("File not recognized as Legacy ND2 (JPEG2000) format.")
        self.lock = threading.RLock()
        self._frame0_meta_cache: FrameMetaDict | None = None

    def is_legacy(self) -> bool:
        return True

    @property
    def chunkmap(self) -> dict[bytes, list[int]]:
        """Return the chunkmap for the file."""
        if not self._chunkmap:
            if self._fh is None:  # pragma: no cover
                raise OSError("File not open")
            self._chunkmap = legacy_nd2_chunkmap(self._fh)
        return self._chunkmap

    @cached_property
    def ddim(self) -> dict:
        return _dims_from_description(self.text_info().get("description"))

    def experiment(self) -> list[strct.ExpLoop]:
        meta = self._raw_exp_loops
        exp = []
        if "LoopNo00" in meta:
            # old style:
            for i, (k, v) in enumerate(meta.items()):
                if k != "Version":
                    loop = self._make_loop(v, i)  # type: ignore
                    if loop:
                        exp.append(loop)
        else:
            i = 0
            while meta:
                # ugh... another hack for weird metadata
                if len(meta) == 1:
                    meta = cast("RawExperimentLoop", next(iter(meta.values())))
                    if meta and isinstance(meta, list):
                        meta = meta[0]
                loop = self._make_loop(meta, i)
                if loop:
                    exp.append(loop)
                meta = meta.get("NextLevelEx")  # type: ignore
                i += 1
        return exp

    def _make_loop(
        self, meta_level: RawExperimentLoop, nest_level: int = 0
    ) -> strct.ExpLoop | None:
        """Converts an old style metadata loop dict to a new ExpLoop structure."""
        type_ = meta_level.get("Type")
        params = meta_level["LoopPars"]
        if type_ == 2:  # XYPosLoop
            params = cast("LoopPars2", params)
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
            params = cast("LoopPars8", params)
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
            params = cast("LoopPars4", params)
            return strct.ZStackLoop(
                count=params["Count"],
                nestingLevel=nest_level,
                parameters=strct.ZStackLoopParams(
                    homeIndex=int(params["ZHome"]),
                    stepUm=params["ZStep"],
                    bottomToTop=params["ZLow"] < params["ZHigh"],
                ),
            )
        if type_ == 6:  # channel
            params = cast("LoopPars6", params)
            return None

        raise ValueError(f"unrecognized type: {type_}")  # pragma: no cover

    def attributes(self) -> strct.Attributes:
        """Load and return the image attributes."""
        if self._attributes is None:
            head = self.header
            bpcim = head["bits_per_component"]
            bpcs = self._advanced_image_attributes.get("SignificantBits", bpcim)
            widthPx = head.get("columns")

            try:
                picplanes = self._frame0_meta()["PicturePlanes"]
                nC = picplanes["Count"]
                compCount = picplanes["CompCount"]
            except Exception:
                compCount = nC = 1

            self._attributes = strct.Attributes(
                bitsPerComponentInMemory=bpcim,
                bitsPerComponentSignificant=bpcs,
                componentCount=compCount,
                heightPx=head.get("rows", -1),
                widthPx=widthPx,
                widthBytes=widthPx * bpcim // 8 * compCount,
                pixelDataType="unsigned",
                sequenceCount=len(self.chunkmap[b"VCAL"]),
                compressionLevel=head.get("compression"),
                channelCount=nC,
            )
        return self._attributes

    def _decode_chunk(self, key: bytes, index: int = 0) -> dict:
        try:
            bxml = self._load_chunk(key, index)
            return parse_xml_block(bxml)
        except KeyError:
            return {}

    def _img_exp_events(self) -> list[strct.ExperimentEvent]:
        from nd2._parse._parse import load_legacy_events

        _events = self._decode_chunk(b"IEVE")
        events: list[dict] = _events.get("FirstEvent", {}).get("no_name", [])
        return load_legacy_events(events)

    def text_info(self) -> strct.TextInfo:
        d = self._decode_chunk(b"TINF")
        for i in d.get("TextInfoItem", []):
            txt = i.get("Text", "")
            if txt.startswith("Metadata:"):
                return {"description": txt}
        return {}

    @cached_property
    def _advanced_image_attributes(self) -> dict:
        return self._decode_chunk(b"ARTT").get("AdvancedImageAttributes") or {}

    @cached_property
    def _raw_exp_loops(self) -> RawExperimentLoop:
        meta = self._decode_chunk(b"AIM1") or self._decode_chunk(b"AIMD")
        version = ""
        meta.pop("UnknownData", None)
        while len(meta) == 1:
            key, val = meta.popitem()
            if "_V" in key:
                version = key.split("_V")[1]
            meta = val
            meta.pop("UnknownData", None)
        meta["Version"] = version
        return meta  # type: ignore

    def metadata(self) -> strct.Metadata:
        return _load_metadata(self._frame0_meta(), self.attributes(), self.experiment())

    @property
    def calibration(self) -> dict:
        return self._decode_chunk(b"ACAL")

    def _load_chunk(self, key: bytes, index: int = 0) -> bytes:
        if not self._fh:  # pragma: no cover
            raise ValueError("Attempt to read from closed nd2 file")
        pos = self.chunkmap[key][index]
        with self.lock:
            self._fh.seek(pos)
            length, box_type = I4s.unpack(self._fh.read(I4s.size))
            return self._fh.read(length - I4s.size)

    def read_frame(self, index: int) -> np.ndarray:
        if not self._fh:  # pragma: no cover
            raise ValueError("Attempt to read from closed nd2 file")

        try:
            from imagecodecs import jpeg2k_decode
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                f"{e}\n"
                f"Reading legacy format nd2 {self._fh.name!r} requires imagecodecs.\n"
                "Please install with `pip install imagecodecs`."
            ) from e

        data = []
        cc = self.attributes().channelCount or 1
        for i in range(cc):
            d = self._load_chunk(b"LUNK", index * cc + i)
            data.append(jpeg2k_decode(d))
        return np.stack(data, axis=-1)

    def frame_metadata(self, index: int) -> dict:
        return {
            **self._decode_chunk(b"VCAL", index),
            **self._decode_chunk(b"VIMD", index),
        }

    def voxel_size(self) -> _util.VoxelSize:
        z: float | None = None
        d = self.text_info().get("description") or ""
        _z = re.search(r"Z Stack Loop: 5\s+-\s+Step\s+([.\d]+)", d)
        if _z:
            with contextlib.suppress(Exception):
                z = float(_z.groups()[0])
        if z is None:
            for e in self.experiment():
                if e.type == "ZStackLoop":
                    z = e.parameters.stepUm
                    break
        xy = self._frame0_meta().get("Calibration") or 1
        return _util.VoxelSize(xy, xy, z or 1)

    def time_stamps(self) -> list[str]:
        planes = self._frame0_meta()["PicturePlanes"]["Plane"]
        return [p["OpticalConfigName"] for p in planes.values()]

    def _frame0_meta(self) -> FrameMetaDict:
        if self._frame0_meta_cache is None:
            meta = self._decode_chunk(b"VIMD", 0)
            self._frame0_meta_cache = cast("FrameMetaDict", meta)
        return self._frame0_meta_cache

    @cached_property
    def header(self) -> dict:
        try:
            pos = self.chunkmap[b"jp2h"][0]
        except (KeyError, IndexError) as e:  # pragma: no cover
            raise KeyError("No valid jp2h header found in file") from e
        fh = cast("BinaryIO", self._fh)
        fh.seek(pos + I4s.size + 4)  # 4 bytes for "label"
        if fh.read(4) != b"ihdr":
            raise KeyError("No valid ihdr header found in jp2h header")

        w, h, c, t, z = IHDR.unpack(fh.read(IHDR.size))
        return {
            "rows": w,
            "columns": h,
            "channels": c,
            "bits_per_component": t + 1,
            "compression": z,
        }

    def events(self, orient: str, null_value: Any) -> list | Mapping:
        warnings.warn(
            "`events` is not implemented for legacy ND2 files",
            UserWarning,
            stacklevel=2,
        )
        return [] if orient == "records" else {}


def legacy_nd2_chunkmap(fh: BinaryIO) -> dict[bytes, list[int]]:
    fh.seek(-40, 2)
    sig, map_start = struct.unpack("<32sQ", fh.read())
    if sig != b"LABORATORY IMAGING ND BOX MAP 00":  # pragma: no cover
        raise ValueError("Not a legacy ND2 file")
    fh.seek(-map_start, 2)
    n_chunks = int.from_bytes(fh.read(4), "big")
    data = fh.read()
    d: defaultdict[bytes, list[int]] = DefaultDict(list)
    for i in range(n_chunks):
        box_type, lim_type, offset = JP2_MAP_CHUNK.unpack_from(data, i * 16)
        if box_type in {b"jP  ", b"ftyp", b"jp2h"}:
            d[box_type].append(offset)
        else:
            d[lim_type].append(offset)
    return dict(d)


DIMSIZE = re.compile(r"(\w+)'?\((\d+)\)")


def _dims_from_description(desc: str | None) -> dict:
    if not desc:
        return {}
    match = re.search(r"Dimensions:\s?([^\r]+)\r?\n", desc)
    if not match:
        return {}
    dims = match.groups()[0]
    dims = dims.replace("λ", _util.AXIS.CHANNEL)
    dims = dims.replace("XY", _util.AXIS.POSITION)
    return {k: int(v) for k, v in DIMSIZE.findall(dims)}


def _load_metadata(
    meta: FrameMetaDict, attrs: strct.Attributes, exp: list[strct.ExpLoop]
) -> strct.Metadata:
    channels = []
    mag = meta.get("ObjectiveMag", -1)
    na = meta.get("ObjectiveNA", -1)
    ri = meta.get("RefractIndex1", -1)
    proj_mag = meta.get("ProjectiveMag", -1)
    pin_rad = meta.get("PinholeRadius", -1)
    microscope = strct.Microscope(
        objectiveMagnification=mag if mag > 0 else None,
        objectiveName=meta.get("ObjectiveName"),
        objectiveNumericalAperture=na if na > 0 else None,
        zoomMagnification=None,
        immersionRefractiveIndex=ri if ri > 0 else None,
        projectiveMagnification=proj_mag if proj_mag > 0 else None,
        pinholeDiameterUm=2 * pin_rad if pin_rad > 0 else None,
    )
    _xy = meta.get("Calibration", -1)
    xy = float(_xy) if _xy > 0 else 1.0
    calib = bool(meta.get("Calibrated"))
    dtype = "float" if attrs.bitsPerComponentSignificant == 32 else "unsigned"

    voxel_count: list[int] = [attrs.widthPx or 0, attrs.heightPx or 0, 1]
    _loops: dict[str, int] = {}
    for i, loop in enumerate(exp):
        _loops[loop.type] = i
        if loop.type == "ZStackLoop":
            voxel_count[2] = loop.count

    if _loops:
        loops = strct.LoopIndices(
            NETimeLoop=_loops.get("NETimeLoop"),
            TimeLoop=_loops.get("TimeLoop"),
            XYPosLoop=_loops.get("XYPosLoop"),
            ZStackLoop=_loops.get("ZStackLoop"),
            CustomLoop=_loops.get("CustomLoop"),
        )
    else:
        loops = None

    volume = strct.Volume(
        axesCalibrated=(calib, calib, False),
        axesCalibration=(xy, xy, 1.0),  # TODO: z step size
        axesInterpretation=("distance",) * 3,
        bitsPerComponentInMemory=attrs.bitsPerComponentInMemory,
        bitsPerComponentSignificant=attrs.bitsPerComponentSignificant,
        cameraTransformationMatrix=(-1, 0, 0, -1),
        componentCount=attrs.componentCount,
        componentDataType=dtype,  # type: ignore [arg-type]
        voxelCount=tuple(voxel_count),  # type: ignore [arg-type]
    )
    planes = meta["PicturePlanes"]["Plane"]
    for idx, plane in planes.items():
        channel_meta = strct.ChannelMeta(
            name=plane["OpticalConfigName"],
            index=int(idx),
            color=strct.Color.from_abgr_u4(plane["Color"]),
            emissionLambdaNm=None,
            excitationLambdaNm=None,
        )
        vol = replace(volume, componentCount=plane["CompCount"])
        ch = strct.Channel(
            channel=channel_meta, microscope=microscope, volume=vol, loops=loops
        )
        channels.append(ch)
    contents = strct.Contents(
        channelCount=len(channels), frameCount=attrs.sequenceCount
    )
    return strct.Metadata(channels=channels, contents=contents)
