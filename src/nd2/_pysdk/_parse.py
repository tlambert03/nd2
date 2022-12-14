from typing import Sequence

from nd2.structures import (
    LoopParams,
    LoopType,
    NETimeLoopParams,
    Period,
    PeriodDiff,
    Position,
    StagePosition,
    TimeLoopParams,
    XYPosLoopParams,
    ZStackLoopParams,
)


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
    loop = load_single_exp_loop(src)
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


def load_single_exp_loop(exp: dict) -> dict:
    if not isinstance(exp, dict):
        breakpoint()
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
