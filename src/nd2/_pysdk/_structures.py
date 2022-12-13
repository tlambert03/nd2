from typing import Any, Callable, Iterator, List

from nd2.structures import (
    ExpLoop,
    LoopTypeString,
    PeriodDiff,
    TimeLoop,
    TimeLoopParams,
    _Loop,
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


def unnest_experiments(metadata: dict) -> Iterator[dict[str, Any]]:
    """Unnest the experiments from the metadata.

    Parameters
    ----------
    metadata : dict
        The metadata to unnest.

    Returns
    -------
    List[Dict[str, Any]]
        A list of the experiments.
    """
    if "SLxExperiment" in metadata:
        metadata = metadata["SLxExperiment"]

    next_level = metadata.copy()
    while True:
        if isinstance(next_level, dict) and len(next_level) == 1 and "" in next_level:
            next_level = next_level[""]
        if isinstance(next_level, list):
            yield [unnest_experiments(i)[0] for i in next_level]
            break
        next_level.pop("NextLevelCount", None)
        next_level.pop("ppNextLevelCount", None)
        yield next_level
        next_level = next_level.pop("NextLevelEx", None) or next_level.pop(
            "ppNextLevelEx", None
        )
        if next_level is None:
            break


def _parse_time_loop(level: int, item: dict) -> TimeLoop:
    count = item.get("uiCount", 0)
    if not count:
        raise ValueError("Invalid time loop count")

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
    return TimeLoop(
        nestingLevel=level,
        count=count,
        parameters=params,
    )


def _parse_default_loop(level: int, item: dict) -> _Loop:
    count = item.get("uiCount", 0)


def _parse_xy_pos_loop(level: int, item: dict) -> _Loop:
    breakpoint()


def _parse_z_stack_loop(level: int, item: dict) -> _Loop:
    breakpoint()


def _parse_spect_loop(level: int, item: dict) -> _Loop:
    breakpoint()


def _parse_ne_time_loop(level: int, item: dict) -> _Loop:
    breakpoint()


_EXP_PARSERS: dict[int, Callable[[int, dict], _Loop]] = {
    1: _parse_time_loop,
    2: _parse_xy_pos_loop,
    4: _parse_z_stack_loop,
    6: _parse_spect_loop,
    8: _parse_ne_time_loop,
}


def loadExperiment(exp: dict[str, Any], sequence_count: int) -> List[ExpLoop]:
    return [
        loadExperimentLoopOld(level, exp)
        for level, exp in enumerate(unnest_experiments(exp))
    ]


def loadExperimentLoopOld(level: int, exp: dict[str, Any]) -> ExpLoop:
    """Load a single experiment loop from exp.

    `exp` is a dict as would be found in the `SLxExperiment` portion of
    the `ImageMetadataLV!` section of the ND2 file.
    """
    loopType = exp.get("eType", 0)
    if loopType > len(LOOP_TYPES):
        raise ValueError(f"Invalid loop type: {loopType!r}")

    item: dict = exp.get("uLoopPars", {})
    return _EXP_PARSERS.get(loopType, _parse_default_loop)(level, item)
