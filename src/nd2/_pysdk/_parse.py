from __future__ import annotations

import re
from dataclasses import asdict
from enum import IntEnum
from struct import Struct
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from nd2.structures import (
    Attributes,
    AxisInterpretation,
    Channel,
    ChannelMeta,
    Contents,
    ExpLoop,
    FrameChannel,
    FrameMetadata,
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
    TimeStamp,
    Volume,
    XYPosLoopParams,
    ZStackLoop,
    ZStackLoopParams,
)

if TYPE_CHECKING:
    from typing_extensions import Literal, TypedDict

    CompressionType = Literal["lossless", "lossy", "none"]

    class GlobalMetadata(TypedDict):
        contents: dict
        loops: dict
        microscope: dict
        position: dict
        time: dict
        volume: dict


strctd = Struct("d")


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


def load_attributes(src: dict, channel_count: int) -> Attributes:
    """Parse the ImageAttributes[LV]! portion of an nd2 file."""
    bpc = src["uiBpcInMemory"]
    _ecomp: int = src.get("eCompression", 2)
    comp_type: CompressionType | None
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
        channelCount=channel_count,
    )


RGB_COLORS: tuple[float, float, float] = (420.0, 515.0, 590.0)


def _get_excitation(probe: dict, filter_: dict, plane: dict, compIndex: int) -> float:
    """Get the excitation wavelength from the probe or filter."""
    if probe:
        excitation = _get_spectrum_max(probe.get("m_ExcitationSpectrum", {}))
    if not excitation and filter_:
        fspectrum = filter_.get("m_ExcitationSpectrum", {})
        ppoint = fspectrum.get("pPoint", {})
        if fspectrum.get("uiCount", 0) > 1 and all(
            i.get("eType") == 4 for i in ppoint.values()
        ):
            excitation = ppoint.get(f"Point{compIndex}", {}).get("dWavelength", 0)
        if not excitation:
            excitation = _get_spectrum_max(fspectrum)
    if not excitation and plane.get("uiCompCount") == 3:
        excitation = RGB_COLORS[compIndex]
    return excitation


def _get_emission(probe: dict, filter_: dict, plane: dict, compIndex: int) -> float:
    """Get the emission wavelength from the probe or filter."""
    if plane.get("uiCompCount") == 3:
        return RGB_COLORS[compIndex]

    if probe:
        emission = _get_spectrum_max(probe.get("m_EmissionSpectrum", {}))
    if not emission and filter_:
        emission = _get_spectrum_max(filter_.get("m_EmissionSpectrum", {}))
    return emission


def _read_wavelengths(plane: dict, compIndex: int) -> tuple[float, float]:
    probe: dict = plane.get("pFluorescentProbe", {})
    filter_: dict = plane.get("pFilterPath", {}).get("m_pFilter", {})
    while isinstance(filter_, list):
        filter_ = filter_[0] if filter_ else {}

    excitation = _get_excitation(probe, filter_, plane, compIndex)
    emission = _get_emission(probe, filter_, plane, compIndex)

    return excitation, emission


def _closest_excitation_wavelength(emission: float, filter_: dict) -> float:
    closest: float = 0
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


def _get_spectrum_max(item: dict | None) -> float:
    # return the wavelength associated with the max value, or 0.0 if no spectrum
    if not item:
        return 0.0
    spectrum = _get_spectrum(item)
    return max(spectrum, key=lambda x: x[0])[1] if spectrum else 0.0


def load_text_info(src: dict) -> TextInfo:
    # we only want keys that are present in the src
    out = {
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
    return cast(TextInfo, out)


def load_global_metadata(
    attrs: Attributes, raw_meta: dict, exp_loops: list[ExpLoop], text_info: TextInfo
) -> GlobalMetadata:
    axesInterpretation: list[AxisInterpretation] = ["distance", "distance", "distance"]
    axesCalibrated: list[bool] = [False, False, False]
    axesCalibration: list[float] = [1.0, 1.0, 1.0]
    if raw_meta.get("ePictureXAxis") == 2:
        axesInterpretation[0] = "time"
    if raw_meta.get("ePictureYAxis") == 2:
        axesInterpretation[1] = "time"
    axesCalibrated[:2] = [raw_meta["bCalibrated"]] * 2
    axesCalibration[:2] = [raw_meta["dCalibration"]] * 2

    voxel_count: list[int] = [attrs.widthPx or 0, attrs.heightPx or 0, 1]
    loops: dict[str, int] = {}
    for i, loop in enumerate(exp_loops):
        loops[str(loop.type)] = i
        if loop.type == "ZStackLoop":
            voxel_count[2] = loop.count
            axesCalibration[2] = abs(loop.parameters.stepUm)
            axesCalibrated[2] = bool(axesCalibration[2] > 0)

    dtype = "float" if attrs.bitsPerComponentSignificant == 32 else "unsigned"
    volume = {
        "axesCalibrated": axesCalibrated,
        "axesCalibration": [i if i > 0 else 1.0 for i in axesCalibration],
        "axesInterpretation": axesInterpretation,
        "bitsPerComponentInMemory": attrs.bitsPerComponentInMemory,
        "bitsPerComponentSignificant": attrs.bitsPerComponentSignificant,
        "cameraTransformationMatrix": [
            raw_meta.get("dStgLgCT11", 1.0),
            raw_meta.get("dStgLgCT12", 0.0),
            raw_meta.get("dStgLgCT21", 0.0),
            raw_meta.get("dStgLgCT22", 1.0),
        ],
        "componentDataType": dtype,
        "voxelCount": voxel_count,
    }
    mag = raw_meta.get("dObjectiveMag", 0.0)
    if mag <= 0 and "optics" in text_info:
        match = re.search(r"\s?(\d+)?x", text_info["optics"], re.IGNORECASE)
        if match:
            mag = float(match[1])
    projectiveMagnification = raw_meta.get("dProjectiveMag")
    if projectiveMagnification and projectiveMagnification < 0:
        projectiveMagnification = None
    pinhole = raw_meta.get("dPinholeRadius", 0) * 2
    na = raw_meta.get("dObjectiveNA", -1)
    zoom = raw_meta.get("dZoom", -1)
    imm = raw_meta.get("dRefractIndex1") or raw_meta.get("dRefractIndex2")
    microscope = {
        "objectiveMagnification": mag if mag > 0 else None,
        "objectiveName": raw_meta.get("wsObjectiveName") or None,
        "objectiveNumericalAperture": na if na > 0 else None,
        "projectiveMagnification": projectiveMagnification,
        "zoomMagnification": zoom if zoom > 0 else None,
        "immersionRefractiveIndex": imm if imm and imm > 0 else None,
        "pinholeDiameterUm": pinhole if pinhole > 0 else None,
    }

    return {
        "contents": {"frameCount": attrs.sequenceCount},
        "loops": loops,
        "microscope": microscope,
        "position": {
            "stagePositionUm": [
                raw_meta.get("dXPos", 0.0),
                raw_meta.get("dYPos", 0.0),
                raw_meta.get("dZPos", 0.0),
            ]
        },
        "time": {
            "relativeTimeMs": raw_meta.get("dTimeMSec", 0.0),
            "absoluteJulianDayNumber": raw_meta.get("dTimeAbsolute", 0.0),
        },
        "volume": volume,
    }


def load_metadata(raw_meta: dict, global_meta: GlobalMetadata) -> Metadata:
    it: dict = raw_meta.get("sPicturePlanes", {})
    raw_planes: dict = it.get("sPlaneNew", None) or it.get("sPlane", {})
    raw_sample_settings: dict[str, dict] = it.get("sSampleSetting", {})
    if len(raw_planes) != it.get("uiCount"):
        raise ValueError("Channel count does not match number of planes")

    pixel_to_stage: list[float] | None = None
    channels: list[Channel] = []
    for i, plane in enumerate(raw_planes.values()):
        k = plane.get("uiSampleIndex") or i
        ex, em = _read_wavelengths(plane, i)
        srcSampleSettings: dict = raw_sample_settings.get(f"a{k}", {})
        channel_meta = ChannelMeta(
            index=k,
            name=plane.get("sDescription"),
            colorRGB=plane.get("uiColor"),
            emissionLambdaNm=em or None,
            excitationLambdaNm=ex or None,
        )
        compCount = plane.get("uiCompCount", 1)
        mask = plane.get("uiModalityMask", None)
        if mask is None:
            modality = plane.get("eModality", None)
            mask = _MODALITY_MASK_MAP.get(
                modality, ELxModalityMask.fluorescence | ELxModalityMask.camera
            )
        flags = _get_modality_flags(mask, compCount)
        volume = global_meta["volume"].copy()
        microscope = global_meta["microscope"].copy()
        camera_matrix = volume.get("cameraTransformationMatrix")
        matrix = srcSampleSettings.get("matCameraToStage")
        if matrix:
            cols = matrix.get("Columns")
            rows = matrix.get("Rows")
            # matrix["Data"] is a list of int64, we need to recast to float
            matrix_data = [i[0] for i in strctd.iter_unpack(bytearray(matrix["Data"]))]

            if cols == 2 and rows == 2:
                volume["cameraTransformationMatrix"] = matrix_data

        _pixel_to_stage = pixel_to_stage
        if camera_matrix:
            # XXX: Not sure if this is correct, specifically with regards to keeping
            # pixel_to_stage of previous channels, vs clearing it for each channel.
            m11, m12, m21, m22 = camera_matrix
            devSettings: dict | None = srcSampleSettings.get("pDeviceSetting")
            if devSettings is not None:
                validStageInversions = devSettings.get("m_iXYUse0")
                if not validStageInversions:
                    _pixel_to_stage = None
                else:
                    calX, calY = volume["axesCalibration"][:2]
                    width, height = volume["voxelCount"][:2]
                    if validStageInversions and all(volume["axesCalibrated"][:2]):
                        invX = devSettings.get("m_iXOrientation0", 0)
                        invY = devSettings.get("m_iYOrientation0", 0)
                        _pixel_to_stage = [
                            invX * calX * m11,
                            invX * calY * m12,
                            -0.5
                            * (invX * calX * m11 * width + invX * calY * m12 * height),
                            invY * calX * m21,
                            invY * calY * m22,
                            -0.5
                            * (invY * calX * m21 * width + invY * calY * m22 * height),
                        ]
                        if _pixel_to_stage is not None:
                            pixel_to_stage = _pixel_to_stage

        volume["pixelToStageTransformationMatrix"] = _pixel_to_stage
        if plane.get("dPinholeDiameter", -1) > 0:
            microscope["pinholeDiameterUm"] = plane["dPinholeDiameter"]

        loops = LoopIndices(**global_meta["loops"]) if global_meta["loops"] else None
        channel = Channel(
            channel=channel_meta,
            loops=loops,
            microscope=Microscope(**microscope, modalityFlags=flags),
            volume=Volume(
                **volume,
                componentCount=compCount,
                componentMinima=[0.0] * compCount,  # FIXME
                componentMaxima=[0.0] * compCount,  # FIXME
            ),
        )
        channels.append(channel)

    contents = Contents(**global_meta["contents"], channelCount=len(channels))
    return Metadata(contents=contents, channels=channels)


def load_frame_metadata(
    global_meta: GlobalMetadata,
    meta: Metadata,
    exp_loops: list[ExpLoop],
    frame_time: float,
    seq_index: int,
) -> FrameMetadata:  # sourcery skip: extract-method
    xy_loop_idx = global_meta["loops"].get("XYPosLoop", -1)
    z_loop_idx = global_meta["loops"].get("ZStackLoop", -1)
    if 0 <= xy_loop_idx < len(exp_loops):
        params = cast("XYPosLoopParams", exp_loops[xy_loop_idx].parameters)
        point: Position = params.points[seq_index]
        name = point.name
        x, y, z = point.stagePositionUm
        if not params.isSettingZ:
            z = global_meta["position"]["stagePositionUm"][2]
    else:
        name = None
        x, y, z = global_meta["position"]["stagePositionUm"]

    if np.isfinite(frame_time):
        julian_day = global_meta["time"].get("absoluteJulianDayNumber", 0)
        time = TimeStamp(
            absoluteJulianDayNumber=julian_day + frame_time / 86_400_000,
            relativeTimeMs=frame_time,
        )
    else:
        time = TimeStamp(**global_meta["time"])

    if 0 <= z_loop_idx < len(exp_loops):
        # Not sure about this, but it's matching the output of the SDK
        zparams = cast(ZStackLoop, exp_loops[z_loop_idx])
        home = zparams.parameters.homeIndex or 0
        step = zparams.parameters.stepUm or 1
        z -= home * step

    frame_channels = []
    for channel in meta.channels or []:
        position = Position(name=name, stagePositionUm=StagePosition(x, y, z))
        frame_channel = FrameChannel(**asdict(channel), time=time, position=position)
        frame_channels.append(frame_channel)

    contents = cast(Contents, meta.contents)
    return FrameMetadata(contents=contents, channels=frame_channels)


def _get_modality_flags(modality_mask: int, component_count: int) -> list[str]:
    if not _modality_mask_valid(modality_mask):
        return ["brightfield"] if component_count == 3 else ["fluorescence"]
    return [e.name for e in ELxModalityMask if e & modality_mask]


def _modality_mask_valid(mask) -> bool:
    return bool(mask & MaskCombo.LX_ModMaskLight != 0)


class ELxModalityMask(IntEnum):
    fluorescence = 0x0000000000000001
    brightfield = 0x0000000000000002
    phaseContrast = 0x0000000000000010
    diContrast = 0x0000000000000020
    camera = 0x0000000000000100
    laserScanConfocal = 0x0000000000000200
    spinningDiskConfocal = 0x0000000000000400
    sweptFieldConfocalSlit = 0x0000000000000800
    sweptFieldConfocalPinhole = 0x0000000000001000
    dsdConfocal = 0x0000000000002000
    SIM = 0x0000000000004000
    iSIM = 0x0000000000008000
    RCM = 0x0000000000000040  # point scan detected by camera (multiple detection)
    VCS = 0x0000000000000080  # VideoConfocal super-resolution
    sora = 0x0000000040000000  # Yokogawa in Super-resolution mode
    liveSR = 0x0000000000040000
    multiphoton = 0x0000000000010000
    TIRF = 0x0000000000020000
    pmt = 0x0000000000100000
    spectral = 0x0000000000200000
    vaasIF = 0x0000000000400000
    vaasNF = 0x0000000000800000
    transmitDetector = 0x0000000001000000
    nonDescannedDetector = 0x0000000002000000
    virtualFilter = 0x0000000004000000
    gaasp = 0x0000000008000000
    remainder = 0x0000000010000000
    aux = 0x0000000020000000


class MaskCombo(IntEnum):
    EXCLUDE_LIGHT = 0x00000000000FF000
    LX_ModMaskLight = ELxModalityMask.fluorescence | ELxModalityMask.brightfield
    LX_ModMaskContrast = ELxModalityMask.phaseContrast | ELxModalityMask.diContrast
    LX_ModMaskAcqHWType = (
        ELxModalityMask.camera
        | ELxModalityMask.laserScanConfocal
        | ELxModalityMask.spinningDiskConfocal
        | ELxModalityMask.sweptFieldConfocalSlit
        | ELxModalityMask.sweptFieldConfocalPinhole
        | ELxModalityMask.dsdConfocal
        | ELxModalityMask.RCM
        | ELxModalityMask.VCS
        | ELxModalityMask.iSIM
    )
    LX_ModMaskDetector = (
        ELxModalityMask.spectral
        | ELxModalityMask.vaasIF
        | ELxModalityMask.vaasNF
        | ELxModalityMask.transmitDetector
        | ELxModalityMask.nonDescannedDetector
        | ELxModalityMask.virtualFilter
        | ELxModalityMask.aux
    )


class ELxModality(IntEnum):
    eModWidefieldFluo = 0
    eModBrightfield = 1
    eModLaserScanConfocal = 2
    eModSpinDiskConfocal = 3
    eModSweptFieldConfocal = 4
    eModMultiPhotonFluo = 5
    eModPhaseContrast = 6
    eModDIContrast = 7
    eModSpectralConfocal = 8
    eModVAASConfocal = 9
    eModVAASConfocalIF = 10
    eModVAASConfocalNF = 11
    eModDSDConfocal = 12
    eModMaxValue = 12


_MODALITY_MASK_MAP: dict[int, int] = {
    ELxModality.eModWidefieldFluo: (
        ELxModalityMask.fluorescence | ELxModalityMask.camera
    ),
    ELxModality.eModBrightfield: ELxModalityMask.brightfield | ELxModalityMask.camera,
    ELxModality.eModLaserScanConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModSpinDiskConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.spinningDiskConfocal
    ),
    ELxModality.eModSweptFieldConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.sweptFieldConfocalSlit
    ),
    ELxModality.eModMultiPhotonFluo: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.multiphoton
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModPhaseContrast: (
        ELxModalityMask.brightfield | ELxModalityMask.phaseContrast
    ),
    ELxModality.eModDIContrast: (
        ELxModalityMask.brightfield | ELxModalityMask.diContrast
    ),
    ELxModality.eModSpectralConfocal: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.spectral
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocal: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasNF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocalIF: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasIF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocalNF: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasNF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModDSDConfocal: (ELxModalityMask.dsdConfocal),
}
