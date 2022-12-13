from typing import Callable, List

from nd2.structures import (
    LoopParams,
    LoopTypeString,
    NETimeLoopParams,
    Period,
    PeriodDiff,
    Position,
    StagePosition,
    TimeLoopParams,
    XYPosLoopParams,
    ZStackLoopParams,
)

LOOP_TYPES: List[LoopTypeString] = [
    "Unknown",
    "TimeLoop",
    "XYPosLoop",
    "XYDiscrLoop",
    "ZStackLoop",
    "PolarLoop",
    "SpectLoop",
    "CustomLoop",
    "NETimeLoop",
    "ManTimeLoop",
    "ZStackLoopAccurate",
]


def _parse_xy_pos_loop(item: dict, count: int) -> tuple[int, XYPosLoopParams]:
    useZ = item.get("bUseZ", False)
    relXY = item.get("bRelativeXY", False)
    refX = item.get("dReferenceX", 0) if relXY else 0
    refY = item.get("dReferenceY", 0) if relXY else 0
    it_points: dict | list[dict] = item["Points"]
    out_points: list[Position] = []

    _points = it_points if isinstance(it_points, list) else [it_points]
    for _n, it in enumerate(_points, 1):
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

    params = XYPosLoopParams(isSettingZ=useZ, points=out_points)
    return len(out_points), params


def _parse_z_stack_loop(item: dict, count: int) -> tuple[int, ZStackLoopParams]:
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


def _parse_time_loop(item: dict, count: int) -> tuple[int, TimeLoopParams | None]:
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


def _parse_ne_time_loop(item: dict, count: int) -> tuple[int, NETimeLoopParams]:
    out_periods: list[Period] = []
    _per: dict | list = item["pPeriod"]
    periods: list[dict] = _per if isinstance(_per, list) else [_per]
    period_valid = [bool(x) for x in item.get("pPeriodValid", [])]

    for it, is_valid in zip(periods, period_valid):
        if not is_valid:
            continue

        c, period_params = _parse_time_loop(it, 0)
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


_EXP_PARSERS: dict[int, Callable[[dict, int], tuple[int, LoopParams]]] = {
    1: _parse_time_loop,
    2: _parse_xy_pos_loop,
    4: _parse_z_stack_loop,
    8: _parse_ne_time_loop,
}


def load_exp_loop(level: int, src: dict, dest: list[dict] | None = None) -> list[dict]:
    loop = load_single_exp_loop(src)
    loop_type = loop.get("type")
    loop_count = loop.get("count")
    dest = dest or []
    if not loop or loop_count == 0 or loop_type == 0:
        return dest

    if loop_type == 6:
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
            load_exp_loop(level + 1, item, dest)

    return dest


def load_single_exp_loop(exp: dict) -> dict:
    if not isinstance(exp, dict):
        breakpoint()
    loop_type = exp.get("eType", 0)
    loop_params: dict = exp.get("uLoopPars", {})
    if not loop_params or loop_type > len(LOOP_TYPES):
        return {}

    count = 0
    params: LoopParams = {}
    if loop_type == 1:  # time loop
        count, params = _parse_time_loop(loop_params, count)
    elif loop_type == 2:
        count, params = _parse_xy_pos_loop(loop_params, count)
    elif loop_type == 4:  # z stack loop
        count, params = _parse_z_stack_loop(loop_params, count)
    elif loop_type == 6:
        # spect loop
        count = loop_params.get("pPlanes", {}).get("uiCount", 0)
        if not count:
            count = loop_params.get("uiCount", 0)
    elif loop_type == 8:  # ne time loop
        count, params = _parse_ne_time_loop(loop_params, count)

    # TODO: loop_type to string
    return {"type": loop_type, "count": count, "parameters": params}


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
