import io
import re
import struct
from enum import IntEnum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

lower = re.compile("^[a-z]*")


def decode_metadata(data: bytes, count: int = 1, strip_prefix=True) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if not data:
        return output

    stream = io.BytesIO(data)
    for _ in range(count):

        curs = stream.tell()
        header = stream.read(2)
        if not header:
            break

        data_type, name_length = strctBB.unpack(header)
        name = stream.read(name_length * 2).decode("utf16")[:-1]
        if strip_prefix:
            name = lower.sub("", name)
        if data_type == 11:
            new_count, length = strctIQ.unpack(stream.read(strctIQ.size))
            next_data_length = stream.read(length - (stream.tell() - curs))
            value = decode_metadata(next_data_length, new_count)
            stream.seek(new_count * 8, 1)
        elif data_type in _PARSER:
            value = _PARSER[data_type](stream)
        else:
            value = None

        if name in output:
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast(list, output[name]).append(value)
        else:
            output[name] = value

    return output


def _unpack_one(strct: struct.Struct, data: io.BytesIO):
    return strct.unpack(data.read(strct.size))[0]


strctBB = struct.Struct("BB")
strctIQ = struct.Struct("<IQ")
strctB = struct.Struct("B")
unpack_B = partial(_unpack_one, strctB)
unpack_I = partial(_unpack_one, struct.Struct("I"))
unpack_Q = partial(_unpack_one, struct.Struct("Q"))
unpack_d = partial(_unpack_one, struct.Struct("d"))


def _unpack_list(data: io.BytesIO):
    return [i[0] for i in strctB.iter_unpack(data.read(unpack_Q(data)))]


def _unpack_string(data: io.BytesIO):
    value = data.read(2)
    # the string ends at the first instance of \x00\x00
    while not value.endswith(b"\x00\x00"):
        next_data = data.read(2)
        if len(next_data) == 0:
            break
        value += next_data

    try:
        return value.decode("utf16")[:-1]
    except UnicodeDecodeError:
        return value.decode("utf8")


_PARSER: Dict[int, Callable] = {
    1: unpack_B,
    2: unpack_I,
    3: unpack_I,
    5: unpack_Q,
    6: unpack_d,
    8: _unpack_string,
    9: _unpack_list,
}

from pydantic import BaseModel


class AutoFocusBeforeLoop(BaseModel):
    Type: int = 0
    Step: float = 0.0
    Range: float = 0.0
    Precision: float = 0.0
    Speed: float = 0.0
    Offset: float = 0.0
    FocusPlane: Optional[dict] = None
    Flags: int = 0
    FocusCriterion: int = 0
    ZDrive: str = ""
    PiezoZTriggered: int = 0
    OptConf: str = ""
    PreferedAFChannel: str = ""


class LoopType(IntEnum):
    NETimeLoop = 8
    XYPosLoop = 2
    ZStackLoop = 6


class ExperimentLoop(BaseModel):
    Type: LoopType
    ApplicationDesc: str = ""
    UserDesc: str = ""
    MeasProbesBase64: str = ""
    CameraName: str = ""
    ItemValid: Optional[list] = None
    LoopPars: Union["TimeLoopPars", "XYPosLoopPars", "ZStackLoopPars", dict]
    AutoFocusBeforeLoop: Optional[dict] = None
    ParallelExperiment: Optional[dict] = None
    CommandBeforeLoop: str = ""
    CommandBeforeCapture: str = ""
    CommandAfterCapture: str = ""
    CommandAfterLoop: str = ""
    ControlShutter: Optional[int] = None
    ControlLight: Optional[int] = None
    UsePFS: Optional[int] = None
    UseHWSequencer: Optional[int] = None
    UseTiRecipe: Optional[int] = None
    RecipeDSCPort: Optional[int] = None
    UseIntenzityCorrection: Optional[int] = None
    KeepObject: Optional[int] = None
    RepeatCount: Optional[int] = None
    NextLevelCount: Optional[int] = None
    NextLevelEx: Union["ExperimentLoop", List["ExperimentLoop"], None] = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> "ExperimentLoop":
        if isinstance(value, dict):
            if "" in value:
                value = value[""]
            if "NextLevelEx" in value:
                nlex = value["NextLevelEx"]
                if isinstance(nlex, dict) and len(nlex) == 1 and "" in nlex:
                    value["NextLevelEx"] = nlex[""]
            return cls(**value)

    class Config:
        extra = "forbid"


class TimeLoopPars(BaseModel):
    Count: int
    PeriodCount: int
    Period: dict
    CommandAfterPeriod: dict
    CommandBeforePeriod: dict
    PeriodValid: list
    AutoFocusBeforeCapture: dict
    AutoFocusBeforePeriod: dict


class TimeLoop(ExperimentLoop):
    Type: Literal[LoopType.NETimeLoop]
    LoopPars: TimeLoopPars


class XYPosLoopPars(BaseModel):
    Count: int
    RelativeXY: int
    ReferenceX: float
    ReferenceY: float
    RedefineAfterPFS: int
    RedefineAfterAutoFocus: int
    KeepPFSOn: int
    SplitMultipoints: int
    UseAFPlane: int
    UseZ: int
    ZDevice: str
    AFBefore: dict
    Points: dict


class XYPosLoop(ExperimentLoop):
    Type: Literal[LoopType.XYPosLoop]
    LoopPars: XYPosLoopPars


class ZStackLoopPars(BaseModel):
    Planes: dict
    MergeCameras: int
    AskForFilter: int
    OffsetReference: int
    Points: dict


class ZStackLoop(ExperimentLoop):
    Type: Literal[LoopType.ZStackLoop]
    LoopPars: ZStackLoopPars


class Experiment(BaseModel):
    SLxExperiment: Union[TimeLoop, XYPosLoopPars, ZStackLoopPars, ExperimentLoop]


ExperimentLoop.update_forward_refs()
