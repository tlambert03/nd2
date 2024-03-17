from __future__ import annotations

import re
import warnings
from dataclasses import asdict
from math import ceil
from struct import Struct
from typing import TYPE_CHECKING, Iterable, cast

import numpy as np

from nd2 import _util
from nd2 import structures as strct
from nd2._sdk_types import ELxModalityMask, EventMeaning, StimulationType

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

    from nd2._sdk_types import (
        AxisInterpretation,
        CompressionType,
        FilterDict,
        FluorescentProbeDict,
        GlobalMetadata,
        LoopTypeString,
        NETimeLoopPars,
        PicturePlanesDict,
        PlaneDict,
        RawAttributesDict,
        RawExperimentDict,
        RawExperimentRecordDict,
        RawLiteEventDict,
        RawMetaDict,
        RawTextInfoDict,
        SpectLoopPars,
        SpectrumDict,
        TimeLoopPars,
        XYPosLoopPars,
        ZStackLoopPars,
    )
    from nd2.structures import ExpLoop, XYPosLoopParams


strctd = Struct("d")


def _parse_xy_pos_loop(
    item: XYPosLoopPars, valid: list[int] | dict[str, bool]
) -> strct.XYPosLoop:
    useZ = item.get("bUseZ", False)
    relXY = item.get("bRelativeXY", False)
    refX = item.get("dReferenceX", 0) if relXY else 0
    refY = item.get("dReferenceY", 0) if relXY else 0
    out_points: list[strct.Position] = []

    if "Points" in item:
        it_points = item["Points"].values()
    else:
        # legacy
        # FIXME: can we move this?  does this ever hit?
        it_points = [
            {
                "dPosX": item["dPosX"][key],
                "dPosY": item["dPosY"][key],
                "dPosZ": item["dPosZ"][key],
                # 'pPosName': pos_names[key],
            }
            for key in item["dPosX"]
        ]

    for it in it_points:
        _offset = it.get("dPFSOffset", 0)
        out_points.append(
            strct.Position(
                stagePositionUm=strct.StagePosition(
                    refX + it.get("dPosX", 0.0),
                    refY + it.get("dPosY", 0.0),
                    it.get("dPosZ", 0.0) if useZ else 0.0,
                ),
                pfsOffset=_offset if _offset >= 0 else None,
                # note: the SDK only checks for pPosName
                name=it.get("dPosName") or it.get("pPosName"),
            )
        )
    if valid:
        if isinstance(valid, dict):
            valid = [v for k, v in sorted(valid.items())]
        out_points = [p for p, is_valid in zip(out_points, valid) if is_valid]

    params = strct.XYPosLoopParams(isSettingZ=useZ, points=out_points)
    return strct.XYPosLoop(count=len(out_points), nestingLevel=0, parameters=params)


def _parse_z_stack_loop(item: ZStackLoopPars) -> strct.ZStackLoop:
    count = item.get("uiCount", 0)

    low = item.get("dZLow", 0.0)
    hi = item.get("dZHigh", 0.0)
    step = item.get("dZStep", 0.0)
    home = item.get("dZHome", 0.0)
    inv = item.get("bZInverted", False)
    type_ = item.get("iType", 0)
    if step == 0 and count > 1:
        step = abs(hi - low) / (count - 1)

    params = strct.ZStackLoopParams(
        homeIndex=_calc_zstack_home_index(inv, count, type_, home, low, hi, step),
        stepUm=step,
        bottomToTop=bool(type_ < 4),
        deviceName=item.get("wsZDevice"),
    )
    return strct.ZStackLoop(count=count, nestingLevel=0, parameters=params)


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


def _parse_time_loop(item: TimeLoopPars) -> strct.TimeLoop | None:
    count = item.get("uiCount", 0)
    if not count:
        return None

    params = strct.TimeLoopParams(
        startMs=item["dStart"],
        periodMs=item["dPeriod"],
        durationMs=item["dDuration"],
        periodDiff=strct.PeriodDiff(
            avg=item.get("dAvgPeriodDiff", 0),
            max=item.get("dMaxPeriodDiff", 0),
            min=item.get("dMinPeriodDiff", 0),
        ),
    )
    return strct.TimeLoop(count=count, nestingLevel=0, parameters=params)


def _parse_ne_time_loop(item: NETimeLoopPars) -> strct.NETimeLoop:
    period_valid = item.get("pPeriodValid", [])
    if isinstance(period_valid, dict):
        period_valid = [period_valid[k] for k in sorted(period_valid)]
    elif not isinstance(period_valid, list):
        raise TypeError(f"invalid type for pPeriodValid: {type(period_valid)}")

    count = 0
    out_periods: list[strct.Period] = []
    for it, is_valid in zip(item["pPeriod"].values(), period_valid):
        if not is_valid:
            continue

        time_loop = _parse_time_loop(it)
        if time_loop:
            out_periods.append(
                strct.Period(
                    count=time_loop.count,
                    startMs=time_loop.parameters.startMs,
                    periodMs=time_loop.parameters.periodMs,
                    durationMs=time_loop.parameters.durationMs,
                    periodDiff=time_loop.parameters.periodDiff,
                )
            )
            count += time_loop.count

    params = strct.NETimeLoopParams(periods=out_periods)
    return strct.NETimeLoop(count=count, nestingLevel=0, parameters=params)


def load_experiment(
    src: RawExperimentDict, level: int = 0, dest: list[ExpLoop] | None = None
) -> list[ExpLoop]:
    """Parse the "ImageMetadata[LV]!" section of an nd2 file."""
    dest = dest or []
    loop = _load_single_experiment_loop(src)

    if not loop or loop.count == 0:
        return dest

    if isinstance(loop, strct.SpectLoop):
        level -= 1
    elif not dest or dest[-1].nestingLevel < level:
        loop.nestingLevel = level
        dest.append(loop)
    else:
        prev = dest[-1]
        if prev.nestingLevel == level and prev.type == loop.type:
            loop.nestingLevel = level
            if prev.count < loop.count:
                dest[-1] = loop

    # FIXME:
    # hack for file in https://github.com/tlambert03/nd2/issues/190
    # there is a better fix, but this is a very rare case
    loop_params = src.get("uLoopPars", {})
    if "pSubLoops" in loop_params:
        loop_params = cast("NETimeLoopPars", loop_params)
        subloops = loop_params["pSubLoops"]
        i0 = "i0000000000"
        if i0 in subloops:
            subnext = subloops[i0]["ppNextLevelEx"]
            if i0 in subnext:
                experiment = subnext[i0]["SLxExperiment"]
                dest.extend(load_experiment(experiment))

    next_level_src = src.get("ppNextLevelEx")
    if next_level_src:
        for item in next_level_src.values():
            dest = load_experiment(item, level + 1, dest)

    return dest


def _load_single_experiment_loop(
    exp: RawExperimentDict,
) -> ExpLoop | strct.SpectLoop | strct.CustomLoop | None:
    loop_type = exp.get("eType", 0)
    loop_params = exp.get("uLoopPars", {})
    if not loop_params or loop_type > max(strct.LoopType):
        return None

    # FIXME: sometimes it's a dict with a single i000000 key?
    # this only happens with version < (3, 0)
    if list(loop_params) == ["i0000000000"]:
        loop_params = next(iter(loop_params.values()))  # type: ignore

    if loop_type == strct.LoopType.TimeLoop:  # 1
        return _parse_time_loop(cast("TimeLoopPars", loop_params))
    elif loop_type == strct.LoopType.XYPosLoop:  # 2
        valid = exp.get("pItemValid", [])
        return _parse_xy_pos_loop(cast("XYPosLoopPars", loop_params), valid)
    elif loop_type == strct.LoopType.ZStackLoop:  # 4
        return _parse_z_stack_loop(cast("ZStackLoopPars", loop_params))
    elif loop_type == strct.LoopType.NETimeLoop:  # 8
        return _parse_ne_time_loop(cast("NETimeLoopPars", loop_params))
    elif loop_type == strct.LoopType.SpectLoop:  # 6
        loop_params = cast("SpectLoopPars", loop_params)
        count = loop_params.get("uiCount", 0)
        count = loop_params.get("pPlanes", {}).get("uiCount", count)
        return strct.SpectLoop(count=count)
    elif loop_type == strct.LoopType.CustomLoop:  # 7
        count = cast("int", loop_params.get("uiCount", 0))
        return strct.CustomLoop(count=count)

    raise NotImplementedError(  # pragma: no cover
        f"We've never seen a file like this! (loop_type={loop_type!r}). We'd "
        "appreciate it if you would submit this file at "
        "https://github.com/tlambert03/nd2/issues/new",
    )


def load_attributes(
    raw_attrs: RawAttributesDict, channel_count: int
) -> strct.Attributes:
    """Parse the ImageAttributes[LV]! portion of an nd2 file."""
    bpc = raw_attrs["uiBpcInMemory"]
    _ecomp = raw_attrs.get("eCompression", 2)
    comp_type: CompressionType | None
    if 0 <= _ecomp < 2:
        comp_type = cast("CompressionType", ["lossless", "lossy", "none"][_ecomp])
        comp_level = raw_attrs.get("dCompressionParam")
    else:
        comp_type = None
        comp_level = None

    tile_width = raw_attrs.get("uiTileWidth", 0)
    tile_height = raw_attrs.get("uiTileHeight", 0)
    if (tile_width <= 0 or tile_width == raw_attrs["uiWidth"]) and (
        tile_height <= 0 or tile_height == raw_attrs["uiHeight"]
    ):
        tile_width = tile_height = None  # type: ignore

    return strct.Attributes(
        bitsPerComponentInMemory=bpc,
        bitsPerComponentSignificant=raw_attrs["uiBpcSignificant"],
        componentCount=raw_attrs["uiComp"],
        heightPx=raw_attrs["uiHeight"],
        pixelDataType="float" if bpc == 32 else "unsigned",
        sequenceCount=raw_attrs["uiSequenceCount"],
        widthBytes=raw_attrs["uiWidthBytes"],
        widthPx=raw_attrs["uiWidth"],
        compressionLevel=comp_level,
        compressionType=comp_type,
        tileHeightPx=tile_height,
        tileWidthPx=tile_width,
        channelCount=channel_count,
    )


RGB_COLORS: tuple[float, float, float] = (420.0, 515.0, 590.0)


def _get_excitation(
    probe: FluorescentProbeDict, filter_: FilterDict, plane: PlaneDict, compIndex: int
) -> float:
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


def _get_emission(
    probe: FluorescentProbeDict, filter_: FilterDict, plane: PlaneDict, compIndex: int
) -> float:
    """Get the emission wavelength from the probe or filter."""
    if plane.get("uiCompCount") == 3:
        return RGB_COLORS[compIndex]

    if probe:
        emission = _get_spectrum_max(probe.get("m_EmissionSpectrum", {}))
    if not emission and filter_:
        emission = _get_spectrum_max(filter_.get("m_EmissionSpectrum", {}))
    return emission


def _read_wavelengths(plane: PlaneDict, compIndex: int) -> tuple[float, float]:
    probe: FluorescentProbeDict = plane.get("pFluorescentProbe", {})
    filters = plane.get("pFilterPath", {}).get("m_pFilter", {})

    # FIXME: always taking the first value?
    filter_: FilterDict = next(iter(filters.values()), {})

    excitation = _get_excitation(probe, filter_, plane, compIndex)
    emission = _get_emission(probe, filter_, plane, compIndex)

    return excitation, emission


def _get_spectrum(item: SpectrumDict) -> list[tuple[float, float, int]]:
    return [
        (p.get("dTValue", 0.0), p.get("dWavelength", 0.0), p.get("eType", 0))
        for p in item.get("pPoint", {}).values()
    ]


def _get_spectrum_max(item: SpectrumDict | None) -> float:
    # return the wavelength associated with the max value, or 0.0 if no spectrum
    if not item:
        return 0.0
    spectrum = _get_spectrum(item)
    return max(spectrum, key=lambda x: x[0])[1] if spectrum else 0.0


LITE_EVENT_KEYS = {"T", "T2", "M", "D", "A", "I", "S"}


def _is_lite_events(events: dict) -> TypeGuard[dict[str, RawLiteEventDict]]:
    event_keys = set.union(*(set(x) for x in events.values()))
    return event_keys.issubset(LITE_EVENT_KEYS)


def load_events(events: RawExperimentRecordDict) -> list[strct.ExperimentEvent]:
    # found in b'CustomData|ExperimentEventsV1_0!'
    count = events.get("uiCount", 0)
    if count == 0:
        return []
    p_events = events.get("pEvents", {})
    if _is_lite_events(p_events):
        return [_load_lite_event(x[1]) for x in sorted(p_events.items())]
    warnings.warn(  # pragma: no cover
        "We haven't seen this event type before, we'd appreciate if you submit this "
        "file at https://github.com/tlambert03/nd2/issues/new",
        stacklevel=2,
    )
    return []


def load_legacy_events(events: Iterable[dict]) -> list[strct.ExperimentEvent]:
    return [_load_legacy_event(*ie) for ie in enumerate(events)]


def _load_legacy_event(id: int, event: dict) -> strct.ExperimentEvent:
    # event will have keys: 'Time', 'Meaning', 'Description', 'Data',
    # meaning seems to almost always be 7

    meaning = EventMeaning(event.get("Meaning", 0))
    description = event.get("Description", "") or meaning.description()
    data = event.get("Data", "")
    if data:
        description += f" - {data}"

    return strct.ExperimentEvent(
        id=id,
        time=event.get("Time", 0.0),
        meaning=meaning,
        description=description,
        data=data,
    )


def _load_lite_event(event: RawLiteEventDict) -> strct.ExperimentEvent:
    stim_event = event.get("S", {})
    if stim_event:
        stim_struct = strct.StimulationEvent(
            type=StimulationType(stim_event.get("T", 0)),
            loop_index=stim_event.get("L", 0),
            position=stim_event.get("P", 0),
            description=stim_event.get("D", ""),
        )
    else:
        stim_struct = None

    meaning = EventMeaning(event.get("M", 0))
    description = event.get("D", "") or meaning.description()
    if stim_struct:
        description += f" Phase {stim_struct.type.name}"
        if stim_struct.description:
            description += f" - {stim_struct.description}"
    return strct.ExperimentEvent(
        id=event.get("I", 0),
        time=event.get("T", 0.0),
        time2=event.get("T2", 0.0),
        meaning=meaning,
        description=description,
        data=event.get("A", ""),
        stimulation=stim_struct,
    )


def load_text_info(raw_txt_info: RawTextInfoDict) -> strct.TextInfo:
    # we only want keys that are present in the raw_txt_info

    out = {
        key: raw_txt_info.get(lookup)
        for key, lookup in (
            ("imageId", "TextInfoItem_0"),
            ("type", "TextInfoItem_1"),
            ("group", "TextInfoItem_2"),
            ("sampleId", "TextInfoItem_3"),
            ("author", "TextInfoItem_4"),
            ("description", "TextInfoItem_5"),
            ("capturing", "TextInfoItem_6"),
            ("sampling", "TextInfoItem_7"),
            ("location", "TextInfoItem_8"),
            ("date", "TextInfoItem_9"),
            ("conclusion", "TextInfoItem_10"),
            ("info1", "TextInfoItem_11"),
            ("info2", "TextInfoItem_12"),
            ("optics", "TextInfoItem_13"),
            ("appVersion", "TextInfoItem_14"),
        )
        if raw_txt_info.get(lookup)
    }
    return cast(strct.TextInfo, out)


def load_global_metadata(
    attrs: strct.Attributes,
    raw_meta: RawMetaDict,
    exp_loops: list[ExpLoop],
    text_info: strct.TextInfo,
) -> GlobalMetadata:
    axesCalibrated: tuple[bool, bool, bool] = (raw_meta["bCalibrated"],) * 2 + (False,)
    axesCalibration: list[float] = [raw_meta["dCalibration"]] * 2 + [1.0]
    axInterp: tuple[AxisInterpretation, AxisInterpretation, AxisInterpretation] = (
        "time" if raw_meta.get("ePictureXAxis") == 2 else "distance",
        "time" if raw_meta.get("ePictureYAxis") == 2 else "distance",
        "distance",
    )

    voxel_count: list[int] = [attrs.widthPx or 0, attrs.heightPx or 0, 1]
    loops: dict[LoopTypeString, int] = {}
    for i, loop in enumerate(exp_loops):
        loops[loop.type] = i
        if loop.type == "ZStackLoop":
            voxel_count[2] = loop.count
            axesCalibration[2] = abs(loop.parameters.stepUm)
            axesCalibrated = axesCalibrated[:2] + (bool(axesCalibration[2] > 0),)

    dtype = "float" if attrs.bitsPerComponentSignificant == 32 else "unsigned"
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

    return {
        "contents": {"frameCount": attrs.sequenceCount},
        "loops": loops,
        "microscope": {
            "objectiveMagnification": mag if mag > 0 else None,
            "objectiveName": raw_meta.get("wsObjectiveName") or None,
            "objectiveNumericalAperture": na if na > 0 else None,
            "projectiveMagnification": projectiveMagnification,
            "zoomMagnification": zoom if zoom > 0 else None,
            "immersionRefractiveIndex": imm if imm and imm > 0 else None,
            "pinholeDiameterUm": pinhole if pinhole > 0 else None,
        },
        "position": {
            "stagePositionUm": (
                raw_meta.get("dXPos", 0.0),
                raw_meta.get("dYPos", 0.0),
                raw_meta.get("dZPos", 0.0),
            )
        },
        "time": {
            "relativeTimeMs": raw_meta.get("dTimeMSec", 0.0),
            "absoluteJulianDayNumber": raw_meta.get("dTimeAbsolute", 0.0),
        },
        "volume": {
            "axesCalibrated": axesCalibrated,
            "axesCalibration": tuple(i if i > 0 else 1.0 for i in axesCalibration),  # type: ignore
            "axesInterpretation": axInterp,
            "bitsPerComponentInMemory": attrs.bitsPerComponentInMemory,
            "bitsPerComponentSignificant": attrs.bitsPerComponentSignificant,
            "cameraTransformationMatrix": (
                raw_meta.get("dStgLgCT11", 1.0),
                raw_meta.get("dStgLgCT12", 0.0),
                raw_meta.get("dStgLgCT21", 0.0),
                raw_meta.get("dStgLgCT22", 1.0),
            ),
            "componentDataType": dtype,  # type: ignore
            "voxelCount": tuple(voxel_count),  # type: ignore
        },
    }


def load_metadata(raw_meta: RawMetaDict, global_meta: GlobalMetadata) -> strct.Metadata:
    pplanes: PicturePlanesDict = raw_meta.get("sPicturePlanes", {})
    raw_planes = pplanes.get("sPlaneNew", None) or pplanes.get("sPlane", {})
    raw_sample_settings = pplanes.get("sSampleSetting", {})
    if len(raw_planes) != pplanes.get("uiCount"):
        raise ValueError("Channel count does not match number of planes")

    pixel_to_stage: list[float] | None = None
    channels: list[strct.Channel] = []
    for i, plane in enumerate(raw_planes.values()):
        k = plane.get("uiSampleIndex") or i
        ex, em = _read_wavelengths(plane, i)
        srcSampleSettings = raw_sample_settings.get(f"a{k}", {})
        channel_meta = strct.ChannelMeta(
            index=k,
            name=plane.get("sDescription", ""),
            color=strct.Color.from_abgr_u4(plane.get("uiColor", 0)),
            emissionLambdaNm=em or None,
            excitationLambdaNm=ex or None,
        )
        compCount = plane.get("uiCompCount", 1)
        mask = plane.get("uiModalityMask", None)
        if mask is None:
            modality = plane.get("eModality", -1)
            default_modality = ELxModalityMask.fluorescence | ELxModalityMask.camera
            mask = ELxModalityMask.get(modality, default_modality)

        flags = ELxModalityMask.flags(mask, compCount)
        volume = global_meta["volume"].copy()
        microscope = global_meta["microscope"].copy()
        camera_matrix = volume.get("cameraTransformationMatrix")
        matrix = srcSampleSettings.get("matCameraToStage")
        if matrix and (matrix.get("Columns") == 2 and matrix.get("Rows") == 2):
            # matrix["Data"] is a list of int64, we need to recast to float
            data = bytearray(matrix["Data"])
            matrix_data: tuple[float, float, float, float] = tuple(
                i[0] for i in strctd.iter_unpack(data)
            )
            volume["cameraTransformationMatrix"] = matrix_data

        _pixel_to_stage = pixel_to_stage
        if camera_matrix:
            # XXX: Not sure if this is correct, specifically with regards to keeping
            # pixel_to_stage of previous channels, vs clearing it for each channel.
            m11, m12, m21, m22 = camera_matrix
            devSettings: dict | None = srcSampleSettings.get("pDeviceSetting")
            if devSettings is not None:
                if not devSettings.get("m_iXYUse0"):
                    _pixel_to_stage = None
                else:
                    calX, calY = volume["axesCalibration"][:2]
                    width, height = volume["voxelCount"][:2]

                    if all(volume["axesCalibrated"][:2]):
                        invX = cast(int, devSettings.get("m_iXOrientation0", 0))
                        invY = cast(int, devSettings.get("m_iYOrientation0", 0))
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

        if plane.get("dPinholeDiameter", -1) > 0:
            microscope["pinholeDiameterUm"] = plane["dPinholeDiameter"]

        glb_loops = global_meta["loops"]
        if glb_loops:
            loops = strct.LoopIndices(
                NETimeLoop=glb_loops.get("NETimeLoop"),
                TimeLoop=glb_loops.get("TimeLoop"),
                XYPosLoop=glb_loops.get("XYPosLoop"),
                ZStackLoop=glb_loops.get("ZStackLoop"),
                CustomLoop=glb_loops.get("CustomLoop"),
            )
        else:
            loops = None

        channel = strct.Channel(
            channel=channel_meta,
            loops=loops,
            microscope=strct.Microscope(
                **microscope,
                modalityFlags=flags,  # type: ignore
            ),
            volume=strct.Volume(
                **volume,
                pixelToStageTransformationMatrix=(
                    None if _pixel_to_stage is None else tuple(_pixel_to_stage)  # type: ignore
                ),
                componentCount=compCount,
                componentMinima=[0.0] * compCount,  # FIXME
                componentMaxima=[0.0] * compCount,  # FIXME
            ),
        )
        channels.append(channel)

    contents = strct.Contents(**global_meta["contents"], channelCount=len(channels))
    return strct.Metadata(contents=contents, channels=channels)


def load_frame_metadata(
    global_meta: GlobalMetadata,
    meta: strct.Metadata,
    exp_loops: list[ExpLoop],
    frame_time: float,
    loop_indices: dict[str, int],
) -> strct.FrameMetadata:
    xy_loop_idx = global_meta["loops"].get("XYPosLoop", -1)
    z_loop_idx = global_meta["loops"].get("ZStackLoop", -1)
    if 0 <= xy_loop_idx < len(exp_loops):
        xy_params = cast("XYPosLoopParams", exp_loops[xy_loop_idx].parameters)
        point = xy_params.points[loop_indices[_util.AXIS.POSITION]]
        name = point.name
        x, y, z = point.stagePositionUm
        if not xy_params.isSettingZ:
            z = global_meta["position"]["stagePositionUm"][2]
    else:
        name = None
        x, y, z = global_meta["position"]["stagePositionUm"]

    if np.isfinite(frame_time):
        julian_day = global_meta["time"].get("absoluteJulianDayNumber", 0)
        time = strct.TimeStamp(
            absoluteJulianDayNumber=julian_day + frame_time / 86_400_000,
            relativeTimeMs=frame_time,
        )
    else:
        time = strct.TimeStamp(**global_meta["time"])

    if 0 <= z_loop_idx < len(exp_loops):
        # Not sure about this, but it's matching the output of the SDK
        zparams = cast(strct.ZStackLoop, exp_loops[z_loop_idx])
        home = zparams.parameters.homeIndex or 0
        step = zparams.parameters.stepUm or 1
        z -= home * step

    P = strct.StagePosition(x, y, z)
    frame_channels = [
        strct.FrameChannel(
            **asdict(channel),
            time=time,
            position=strct.Position(name=name, stagePositionUm=P),
        )
        for channel in meta.channels or ()
    ]
    contents = cast(strct.Contents, meta.contents)
    return strct.FrameMetadata(contents=contents, channels=frame_channels)
