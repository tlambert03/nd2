from enum import Enum
from typing import TYPE_CHECKING, Sequence, cast
import re

from nd2.structures import (
    Attributes,
    Channel,
    ChannelMeta,
    ExpLoop,
    LoopIndices,
    LoopParams,
    LoopType,
    Metadata,
    Microscope,
    NETimeLoopParams,
    Period,
    PeriodDiff,
    Position,
    StagePosition,
    TextInfo,
    TimeLoopParams,
    Volume,
    XYPosLoopParams,
    ZStackLoop,
    ZStackLoopParams,
    AxisInterpretation,
)

if TYPE_CHECKING:
    from typing_extensions import Literal

    CompressionType = Literal["lossless", "lossy", "none"]


#   enum ELxModalityMask : uint64_t
#   {
#      LX_ModFluorescence = 0x0000000000000001ULL,
#      LX_ModBrightfield = 0x0000000000000002ULL,
#      LX_ModMaskLight = (LX_ModFluorescence | LX_ModBrightfield),

#      LX_ModPhaseContrast = 0x0000000000000010ULL,
#      LX_ModDIContrast = 0x0000000000000020ULL,
#      LX_ModMaskContrast = (LX_ModPhaseContrast | LX_ModDIContrast),

#      LX_ModCamera = 0x0000000000000100ULL,
#      LX_ModLaserScanConfocal = 0x0000000000000200ULL,
#      LX_ModSpinDiskConfocal = 0x0000000000000400ULL,
#      LX_ModSweptFieldConfocalSlit = 0x0000000000000800ULL,
#      LX_ModSweptFieldConfocalPinholes = 0x0000000000001000ULL,
#      LX_ModDSDConfocal = 0x0000000000002000ULL,
#      LX_ModSIM = 0x0000000000004000ULL,
#      LX_ModISIM = 0x0000000000008000ULL,
#      LX_ModRCM = 0x0000000000000040ULL, // point scan, detected by camera (multiple detections)
#      LX_ModVCS = 0x0000000000000080ULL, // VideoConfocal super-resolution
#      LX_ModSora = 0x0000000040000000ULL, // Yokogawa in Super-resolution mode
#      LX_ModLiveSR = 0x0000000000040000ULL,
#      LX_ModMaskAcqHWType = (LX_ModCamera | LX_ModLaserScanConfocal | LX_ModSpinDiskConfocal | LX_ModSweptFieldConfocalSlit | LX_ModSweptFieldConfocalPinholes | LX_ModDSDConfocal | LX_ModRCM | LX_ModVCS | LX_ModISIM),

#      LX_ModMultiPhotonFluo = 0x0000000000010000ULL,
#      LX_ModTIRF = 0x0000000000020000ULL,
#      LX_ModPMT = 0x0000000000100000ULL,
#      LX_ModSpectral = 0x0000000000200000ULL,
#      LX_ModVAAS_IF = 0x0000000000400000ULL,
#      LX_ModVAAS_NF = 0x0000000000800000ULL,
#      LX_ModTransmitDetector = 0x0000000001000000ULL,
#      LX_ModNonDescannedDetector = 0x0000000002000000ULL,
#      LX_ModVirtualFilter = 0x0000000004000000ULL,
#      LX_ModGaAsP = 0x0000000008000000ULL,
#      LX_ModRemainder = 0x0000000010000000ULL,
#      LX_ModAUX = 0x0000000020000000ULL,
#      LX_ModMaskDetector = (LX_ModSpectral | LX_ModVAAS_IF | LX_ModVAAS_NF | LX_ModTransmitDetector | LX_ModNonDescannedDetector | LX_ModVirtualFilter | LX_ModAUX),

#      EXCLUDE_LIGHT = 0x00000000000ff000ULL
#   };


class ELxModalityMask(Enum):
    ...

    
def _parse_xy_pos_loop(
    item: dict, valid: Sequence[int] = ()
) -> tuple[int, XYPosLoopParams]:
    useZ = item.get("bUseZ", False)
    relXY = item.get("bRelativeXY", False)
    refX = item.get("dReferenceX", 0) if relXY else 0
    refY = item.get("dReferenceY", 0) if relXY else 0
    it_points: dict | list[dict] = item["Points"]
    out_points: list[Position] = []

    _points = it_points if isinstance(it_points, list) else [it_points]
    for it in _points:
        _offset = it.get("dPFSOffset", 0)
        out_points.append(
            Position(
                stagePositionUm=StagePosition(
                    refX + it.get("dPosX", 0.0),
                    refY + it.get("dPosY", 0.0),
                    it.get("dPosZ", 0.0) if useZ else 0.0,
                ),
                pfsOffset=_offset if _offset >= 0 else None,
                # note: the SDK only checks for pPosName
                name=it.get("pPosName") or it.get("dPosName"),
            )
        )
    if valid:
        out_points = [p for p, is_valid in zip(out_points, valid) if is_valid]

    params = XYPosLoopParams(isSettingZ=useZ, points=out_points)
    return len(out_points), params


def _parse_z_stack_loop(item: dict) -> tuple[int, ZStackLoopParams]:
    count = item.get("uiCount", 0)

    low = item.get("dZLow", 0)
    hi = item.get("dZHigh", 0)
    step = item.get("dZStep", 0)
    home = item.get("dZHome", 0)
    inv = item.get("bZInverted", False)
    type_ = item.get("iType", 0)
    if step == 0 and count > 1:
        step = abs(hi - low) / (count - 1)
    home_index = _calc_zstack_home_index(inv, count, type_, home, low, hi, step)

    params = ZStackLoopParams(
        homeIndex=home_index,
        stepUm=step,
        bottomToTop=bool(type_ < 4),
        deviceName=item.get("wsZDevice"),
    )
    return count, params


def _calc_zstack_home_index(
    inverted: bool,
    count: int,
    type_: int,
    home_um: float,
    low_um: float,
    high_um: float,
    step_um: float,
    tol: float = 0.05,
) -> int:
    from math import ceil

    home_range_f = abs(low_um - home_um)
    home_range_i = abs(high_um - home_um)

    if type_ in {2, 3}:
        hrange = inverted and home_range_i or home_range_f
    elif type_ in {6, 7}:
        hrange = inverted and home_range_f or home_range_i
    else:
        return (count - 1) // 2

    if step_um <= 0:
        return min(int((count - 1) * hrange / abs(high_um - low_um)), count - 1)
    else:
        return min(int(abs(ceil((hrange - tol * step_um) / step_um))), count - 1)


def _parse_time_loop(item: dict) -> tuple[int, TimeLoopParams | None]:
    count = item.get("uiCount", 0)
    if not count:
        return (0, None)

    params = TimeLoopParams(
        startMs=item["dStart"],
        periodMs=item["dPeriod"],
        durationMs=item["dDuration"],
        periodDiff=PeriodDiff(
            avg=item.get("dAvgPeriodDiff", 0),
            max=item.get("dMaxPeriodDiff", 0),
            min=item.get("dMinPeriodDiff", 0),
        ),
    )
    return count, params


def _parse_ne_time_loop(item: dict) -> tuple[int, NETimeLoopParams]:
    out_periods: list[Period] = []
    _per: dict | list = item["pPeriod"]
    periods: list[dict] = _per if isinstance(_per, list) else [_per]
    period_valid = [bool(x) for x in item.get("pPeriodValid", [])]

    count = 0
    for it, is_valid in zip(periods, period_valid):
        if not is_valid:
            continue

        c, period_params = _parse_time_loop(it)
        if period_params:
            out_periods.append(
                Period(
                    count=c,
                    startMs=period_params.startMs,
                    periodMs=period_params.periodMs,
                    durationMs=period_params.durationMs,
                    periodDiff=period_params.periodDiff,
                )
            )
            count += c

    params = NETimeLoopParams(periods=out_periods)
    return count, params


def load_exp_loop(level: int, src: dict, dest: list[dict] | None = None) -> list[dict]:
    """Parse the "ImageMetadata[LV]!" section of an nd2 file."""
    loop = _load_single_exp_loop(src)
    loop_type = loop.get("type")
    loop_count = loop.get("count")
    dest = dest or []
    if not loop or loop_count == 0 or loop_type == LoopType.Unknown:
        return dest

    if loop_type == LoopType.SpectLoop:
        level -= 1
    elif not dest or dest[-1]["nestingLevel"] < level:
        loop["nestingLevel"] = level
        dest.append(loop)
    else:
        prev = dest[-1]
        if prev["nestingLevel"] == level and prev["type"] == loop_type:
            loop["nestingLevel"] = level
            if prev["count"] < loop_count:
                dest[-1] = loop

    next_level_src = src.get("ppNextLevelEx")
    if next_level_src:
        items = [next_level_src] if isinstance(next_level_src, dict) else next_level_src
        for item in items:
            dest = load_exp_loop(level + 1, item, dest)
    return dest


def _load_single_exp_loop(exp: dict) -> dict:
    loop_type = exp.get("eType", 0)
    loop_params: dict = exp.get("uLoopPars", {})
    if not loop_params or loop_type > max(LoopType):
        return {}

    count = loop_params.get("uiCount", 0)
    params: LoopParams | None = None
    if loop_type == LoopType.TimeLoop:
        count, params = _parse_time_loop(loop_params)
    elif loop_type == LoopType.XYPosLoop:
        valid = exp.get("pItemValid", ())
        count, params = _parse_xy_pos_loop(loop_params, valid)
    elif loop_type == LoopType.ZStackLoop:
        count, params = _parse_z_stack_loop(loop_params)
    elif loop_type == LoopType.SpectLoop:
        count = loop_params.get("pPlanes", {}).get("uiCount", count)
    elif loop_type == LoopType.NETimeLoop:
        count, params = _parse_ne_time_loop(loop_params)

    # TODO: loop_type to string
    return {"type": loop_type, "count": count, "parameters": params}


def load_attributes(src: dict) -> Attributes:
    """Parse the ImageAttributes[LV]! portion of an nd2 file."""
    bpc = src["uiBpcInMemory"]
    _ecomp: int = src.get("eCompression", 2)
    comp_type: "CompressionType" | None
    if 0 <= _ecomp < 2:
        comp_type = cast("CompressionType", ["lossless", "lossy", "none"][_ecomp])
        comp_level = src.get("dCompressionParam")
    else:
        comp_type = None
        comp_level = None

    tile_width = src.get("uiTileWidth", 0)
    tile_height = src.get("uiTileHeight", 0)
    if (tile_width <= 0 or tile_width == src["uiWidth"]) and (
        tile_height <= 0 or tile_height == src["uiHeight"]
    ):
        tile_width = tile_height = None

    return Attributes(
        bitsPerComponentInMemory=bpc,
        bitsPerComponentSignificant=src["uiBpcSignificant"],
        componentCount=src["uiComp"],
        heightPx=src["uiHeight"],
        pixelDataType="float" if bpc == 32 else "unsigned",
        sequenceCount=src["uiSequenceCount"],
        widthBytes=src["uiWidthBytes"],
        widthPx=src["uiWidth"],
        compressionLevel=comp_level,
        compressionType=comp_type,
        tileHeightPx=tile_height,
        tileWidthPx=tile_width,
        # channelCount=attrs[""],  # this comes from metadata
    )


def _read_wavelengths(plane: dict, compIndex: int) -> tuple[float, float]:
    comp_count = plane.get("uiCompCount")
    emission = 0.0
    excitation = 0.0

    if comp_count == 3:  # RGB
        if compIndex == 0:
            emission = 420.0
        elif compIndex == 1:
            emission = 515.0
        elif compIndex == 2:
            emission = 590.0

        probe = plane.get("pFluorescentProbe")
        if probe:
            excitation_spectrum = probe.get("m_ExcitationSpectrum")
            if excitation_spectrum:
                excitation = _get_single_wavelength(excitation_spectrum)

        if excitation == 0.0:
            filter_path = plane.get("pFilterPath")
            if filter_path:
                excitation = _closest_excitation_wavelength(emission, filter_path)

        if excitation == 0.0:
            excitation = emission

    return emission, excitation


def _closest_excitation_wavelength(emission: float, filter_path: dict) -> float:
    closest: float = 0
    if "m_pFilter" in filter_path:
        breakpoint()
    return closest

    # Tval   Wavel  type


Spectrum = list[tuple[float, float, int]]
# types:
#    eSptInvalid = 0,
#    eSptPoint = 1,
#    eSptRaisingEdge = 2,
#    eSptFallingEdge = 3,
#    eSptPeak = 4,
#    eSptRange = 5


def _get_spectrum(item: dict) -> Spectrum:
    return [
        (p.get("dTValue", 0.0), p.get("dWavelength", 0.0), p.get("eType", 0))
        for p in cast("dict[str, dict]", item.get("pPoint", {})).values()
    ]


def _get_single_wavelength(item: dict) -> float:
    # return the wavelength associated with the max value, or 0.0 if no spectrum
    spectrum = _get_spectrum(item)
    return max(spectrum, key=lambda x: x[0])[1] if spectrum else 0.0


def load_text_info(src: dict) -> TextInfo:
    return {
        key: src[lookup]
        for key, lookup in (
            ("appVersion", "TextInfoItem_14"),
            ("author", "TextInfoItem_4"),
            ("capturing", "TextInfoItem_6"),
            ("conclusion", "TextInfoItem_10"),
            ("date", "TextInfoItem_9"),
            ("description", "TextInfoItem_5"),
            ("group", "TextInfoItem_2"),
            ("imageId", "TextInfoItem_0"),
            ("info1", "TextInfoItem_11"),
            ("info2", "TextInfoItem_12"),
            ("location", "TextInfoItem_8"),
            ("optics", "TextInfoItem_13"),
            ("sampleId", "TextInfoItem_3"),
            ("sampling", "TextInfoItem_7"),
            ("type", "TextInfoItem_1"),
        )
        if src.get(lookup)
    }


def load_global_metadata(
    attrs: Attributes, raw_meta: dict, exp_loops: list[ExpLoop], text_info: TextInfo
):
    loops = LoopIndices(**{str(loop.type): i for i, loop in enumerate(exp_loops)})

    axesInterpretation = ["distance", "distance", "distance"]
    axesCalibrated = [False, False, False]
    axesCalibration = [1.0, 1.0, 1.0]
    voxel_count = [attrs.widthPx, attrs.heightPx, 1]  # needs loops
    if not raw_meta:
        StagePositionUm = [0.0, 0.0, 0.0]
        relativeTimeMs = 0.0
    else:
        if raw_meta["ePictureXAxis"] == 2:
            axesInterpretation[0] = "time"
        if raw_meta["ePictureYAxis"] == 2:
            axesInterpretation[1] = "time"
        axesCalibrated[:2] = [raw_meta["bCalibrated"]] * 2
        axesCalibration[:2] = [raw_meta["dCalibration"]] * 2

        if loops.ZStackLoop is not None:
            z_loop = cast(ZStackLoop, exp_loops[loops.ZStackLoop])
            axesCalibration[2] = abs(z_loop.parameters.stepUm)
            voxel_count[2] = z_loop.count
        axesCalibrated[2] = bool(axesCalibration[2] > 0)

    axesCalibration = [i if i > 0 else 1.0 for i in axesCalibration]

    cam_matrix = tuple(
        raw_meta.get(n, 0.0)
        for n in ("dStgLgCT11", "dStgLgCT12", "dStgLgCT21", "dStgLgCT22")
    )

    volume = Volume(
        axesCalibrated=cast("tuple[bool,bool,bool]", tuple(axesCalibrated)),
        axesCalibration=cast("tuple[float,float,float]", (axesCalibration)),
        axesInterpretation=cast(
            "tuple[AxisInterpretation, AxisInterpretation, AxisInterpretation]",
            tuple(axesInterpretation),
        ),
        bitsPerComponentInMemory=attrs.bitsPerComponentInMemory,
        bitsPerComponentSignificant=attrs.bitsPerComponentSignificant,
        cameraTransformationMatrix=cast(tuple[float, float, float, float], cam_matrix),
        componentCount=None,
        componentDataType=(
            "float" if attrs.bitsPerComponentSignificant == 32 else "unsigned"
        ),
        voxelCount=tuple(voxel_count),
        componentMaxima=None,
        componentMinima=None,
        pixelToStageTransformationMatrix=None,
    )

    mag = raw_meta.get("dObjectiveMag", 0.0)
    if mag < 0:
        optics = text_info.get("optics")
        if optics:
            match = re.search(r"\s?(\d+)?x", optics, re.IGNORECASE)
            if match:
                mag = float(match[1])

    if "dPinholeRadius" in raw_meta:
        pinhole_diam: float | None = raw_meta["dPinholeRadius"] * 2
    else:
        pinhole_diam = None

    planes = raw_meta.get("sPicturePlanes", {})
    raw_planes = planes.get("sPlaneNew", {})
    plane = next(iter(raw_planes.values()), {})
    if plane:

    microscope = Microscope(
        objectiveMagnification=mag,
        objectiveName=raw_meta.get("wsObjectiveName"),
        objectiveNumericalAperture=raw_meta.get("dObjectiveNA"),
        zoomMagnification=raw_meta.get("dZoom"),
        immersionRefractiveIndex=(
            raw_meta.get("dRefractIndex1") or raw_meta.get("dRefractIndex2")
        ),
        pinholeDiameterUm=pinhole_diam,
    )
    breakpoint()


def load_metadata(src: dict) -> Metadata:
    it: dict = src.get("sPicturePlanes")
    channel_count = it.get("uiCount")
    sample_count = it.get("uiSampleCount")
    channels: list[dict] = []
    raw_planes: dict = it.get("sPlaneNew") or it.get("sPlane", {})
    raw_sample_settings: dict[str, dict] = it.get("sSampleSetting")

    if len(raw_planes) != channel_count:
        raise ValueError("Channel count does not match number of planes")

    for i, (key, plane) in enumerate(raw_planes.items()):
        k = plane.get("uiSampleIndex")
        sample_settings = raw_sample_settings[key]
        em, ex = _read_wavelengths(plane, i)
        channel_meta = ChannelMeta(
            index=k,
            name=plane.get("sDescription"),
            colorRGB=plane.get("uiColor"),
            emissionLambdaNm=em,
            excitationLambdaNm=ex,
        )
        breakpoint()
        channel = Channel(
            channel=channel_meta,
            loops=...,
            microscope=...,
            volume=...,
        )

    contents = ...
    channels = ...
    return Metadata(contents=contents, channels=channels)
