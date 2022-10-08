from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, NamedTuple, Optional, Tuple, Union

from typing_extensions import Literal

# enums


class LoopType(IntEnum):
    NETimeLoop = 8
    XYPosLoop = 2
    ZStackLoop = 6
    TimeLoop = 1  # not sure about this
    Unknown = 4  # not sure about this


class AxisInterpretation(str, Enum):
    distance = "distance"
    time = "time"


# tuples


class Attributes(NamedTuple):
    bitsPerComponentInMemory: int
    bitsPerComponentSignificant: int
    componentCount: int
    heightPx: int
    pixelDataType: str
    sequenceCount: int
    widthBytes: Optional[int] = None
    widthPx: Optional[int] = None
    compressionLevel: Optional[int] = None
    compressionType: Optional[str] = None
    tileHeightPx: Optional[int] = None
    tileWidthPx: Optional[int] = None
    channelCount: Optional[int] = None


class ImageInfo(NamedTuple):
    width: int
    height: int
    components: int
    bits_per_pixel: int


# experiment #################


LoopTypeString = Union[
    Literal["TimeLoop"],
    Literal["NETimeLoop"],
    Literal["XYPosLoop"],
    Literal["ZStackLoop"],
]


@dataclass
class _Loop:
    count: int
    nestingLevel: int
    parameters: LoopParams
    type: LoopTypeString

    @classmethod
    def create(cls, obj: dict) -> ExpLoop:
        return globals()[obj["type"]](**obj)


#####


@dataclass
class TimeLoop(_Loop):
    parameters: TimeLoopParams
    type: Literal["TimeLoop"] = "TimeLoop"

    def __post_init__(self):
        if isinstance(self.parameters, dict):
            self.parameters = TimeLoopParams(**self.parameters)


@dataclass
class TimeLoopParams:
    startMs: float
    periodMs: float
    durationMs: float
    periodDiff: PeriodDiff

    def __post_init__(self):
        if isinstance(self.periodDiff, dict):
            self.periodDiff = PeriodDiff(**self.periodDiff)


@dataclass
class PeriodDiff:
    avg: float = 0
    max: float = 0
    min: float = 0


######


@dataclass
class NETimeLoop(_Loop):
    parameters: NETimeLoopParams
    type: Literal["NETimeLoop"] = "NETimeLoop"

    def __post_init__(self):
        if isinstance(self.parameters, dict):
            self.parameters = NETimeLoopParams(**self.parameters)


@dataclass
class NETimeLoopParams:
    periods: List[Period]

    def __post_init__(self):
        self.periods = [Period(**i) if isinstance(i, dict) else i for i in self.periods]


@dataclass
class Period(TimeLoopParams):
    count: int


#########


@dataclass
class XYPosLoop(_Loop):
    parameters: XYPosLoopParams
    type: Literal["XYPosLoop"] = "XYPosLoop"

    def __post_init__(self):
        if isinstance(self.parameters, dict):
            self.parameters = XYPosLoopParams(**self.parameters)


@dataclass
class XYPosLoopParams:
    isSettingZ: bool
    points: List[Position]

    def __post_init__(self):
        self.points = [Position(**i) if isinstance(i, dict) else i for i in self.points]


@dataclass
class Position:
    stagePositionUm: StagePosition
    pfsOffset: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.stagePositionUm, dict):
            self.stagePositionUm = StagePosition(*self.stagePositionUm)


class StagePosition(NamedTuple):
    x: float
    y: float
    z: float


# ######


@dataclass
class ZStackLoop(_Loop):
    parameters: ZStackLoopParams
    type: Literal["ZStackLoop"] = "ZStackLoop"

    def __post_init__(self):
        if isinstance(self.parameters, dict):
            self.parameters = ZStackLoopParams(**self.parameters)


@dataclass
class ZStackLoopParams:
    homeIndex: int
    stepUm: float
    bottomToTop: bool
    deviceName: Optional[str] = None


###

ExpLoop = Union[TimeLoop, NETimeLoop, XYPosLoop, ZStackLoop]
LoopParams = Union[TimeLoopParams, NETimeLoopParams, XYPosLoopParams, ZStackLoopParams]

# metadata #################


@dataclass
class Metadata:
    contents: Optional[Contents] = None
    channels: Optional[List[Channel]] = None

    def __post_init__(self):
        if isinstance(self.contents, dict):
            self.contents = Contents(**self.contents)
        if self.channels:
            self.channels = [
                Channel(**i) if isinstance(i, dict) else i for i in self.channels
            ]


@dataclass
class Contents:
    channelCount: int
    frameCount: int


@dataclass
class Channel:
    channel: ChannelMeta
    loops: Optional[LoopIndices]
    microscope: Microscope
    volume: Volume

    def __post_init__(self):
        self.channel = ChannelMeta(**self.channel)
        self.microscope = Microscope(**self.microscope)
        self.volume = Volume(**self.volume)
        if self.loops:
            self.loops = LoopIndices(**self.loops)


@dataclass
class ChannelMeta:
    name: str
    index: int
    colorRGB: int  # probably 0xBBGGRR
    emissionLambdaNm: Optional[float] = None
    excitationLambdaNm: Optional[float] = None


@dataclass
class LoopIndices:
    NETimeLoop: Optional[int] = None
    TimeLoop: Optional[int] = None
    XYPosLoop: Optional[int] = None
    ZStackLoop: Optional[int] = None


@dataclass
class Microscope:
    objectiveMagnification: Optional[float] = None
    objectiveName: Optional[str] = None
    objectiveNumericalAperture: Optional[float] = None
    zoomMagnification: Optional[float] = None
    immersionRefractiveIndex: Optional[float] = None
    projectiveMagnification: Optional[float] = None
    pinholeDiameterUm: Optional[float] = None
    modalityFlags: List[str] = field(default_factory=list)


@dataclass
class Volume:
    axesCalibrated: Tuple[bool, bool, bool]
    axesCalibration: Tuple[float, float, float]
    axesInterpretation: Tuple[
        AxisInterpretation, AxisInterpretation, AxisInterpretation
    ]
    bitsPerComponentInMemory: int
    bitsPerComponentSignificant: int
    cameraTransformationMatrix: Tuple[float, float, float, float]
    componentCount: int
    componentDataType: Union[Literal["unsigned"], Literal["float"]]
    voxelCount: Tuple[int, int, int]
    componentMaxima: Optional[List[float]] = None
    componentMinima: Optional[List[float]] = None
    pixelToStageTransformationMatrix: Optional[
        Tuple[float, float, float, float, float, float]
    ] = None

    # NIS Microscope Absolute frame in um =
    # pixelToStageTransformationMatrix * (X_in_px,  Y_in_px,  1) + stagePositionUm

    def __post_init__(self):
        self.axesInterpretation = tuple(  # type: ignore
            AxisInterpretation(i) for i in self.axesInterpretation
        )


@dataclass
class TimeStamp:
    absoluteJulianDayNumber: float
    relativeTimeMs: float


@dataclass
class FrameChannel(Channel):
    position: Position
    time: TimeStamp

    def __post_init__(self):
        super().__post_init__()
        self.position = Position(**self.position)
        self.time = TimeStamp(**self.time)


@dataclass
class FrameMetadata:
    contents: Contents
    channels: List[FrameChannel]

    def __post_init__(self):
        self.contents = Contents(**self.contents)
        self.channels = [FrameChannel(**i) for i in self.channels]


class Coordinate(NamedTuple):
    index: int
    type: str
    size: int


def _lower0(x: str) -> str:
    return x[0].lower() + x[1:]


class BoxShape(NamedTuple):
    sizeX: float = 0
    sizeY: float = 0
    sizeZ: float = 0


class XYPoint(NamedTuple):
    x: float = 0
    y: float = 0


class XYZPoint(NamedTuple):
    x: float = 0
    y: float = 0
    z: float = 0


class ExtrudedShape(NamedTuple):
    sizeZ: float = 0
    basePoints: List[XYPoint] = []

    @classmethod
    def _from_meta_dict(cls, val: dict) -> ExtrudedShape:
        return cls(
            sizeZ=val.get("SizeZ") or val.get("sizeZ") or 0,
            basePoints=[
                XYPoint(*val[f"BasePoints_{i}"].get("", []))
                for i in range(val.get("BasePoints_Size", 0))
            ],
        )


@dataclass
class ROI:
    """ROI object from NIS Elements."""

    id: int
    info: RoiInfo
    guid: str
    animParams: List[AnimParam] = field(default_factory=list)

    def __post_init__(self):
        self.info = RoiInfo(**self.info)
        self.animParams = [AnimParam(**i) for i in self.animParams]

    @classmethod
    def _from_meta_dict(cls, val: dict) -> ROI:
        anim_params = [
            {_lower0(k): v for k, v in val[f"AnimParams_{i}"].items()}
            for i in range(val.pop("AnimParams_Size", 0))
        ]
        return cls(
            id=val["Id"],
            info={_lower0(k): v for k, v in val["Info"].items()},
            guid=val.get("GUID", ""),
            animParams=anim_params,
        )


@dataclass
class AnimParam:
    """Parameters of ROI position/shape."""

    timeMs: float = 0
    enabled: bool = True
    centerX: float = 0
    centerY: float = 0
    centerZ: float = 0
    rotationZ: float = 0
    boxShape: BoxShape = BoxShape()
    extrudedShape: ExtrudedShape = ExtrudedShape()

    def __post_init__(self):
        if isinstance(self.boxShape, dict):
            self.boxShape = BoxShape(
                **{_lower0(k): v for k, v in self.boxShape.items()}
            )
        if isinstance(self.extrudedShape, dict):
            self.extrudedShape = ExtrudedShape._from_meta_dict(self.extrudedShape)

    @property
    def center(self) -> XYZPoint:
        """Center point as a named tuple (x, y, z)."""
        return XYZPoint(self.centerX, self.centerY, self.centerZ)


class RoiShapeType(IntEnum):
    """The type of ROI shape."""

    Raster = 1
    Unknown2 = 2
    Rectangle = 3
    Ellipse = 4
    Polygon = 5
    Bezier = 6
    Unknown7 = 7
    Unknown8 = 8
    Circle = 9
    Square = 10


class InterpType(IntEnum):
    """The role that the ROI plays."""

    StandardROI = 1
    BackgroundROI = 2
    ReferenceROI = 3
    StimulationROI = 4


@dataclass
class RoiInfo:
    """Info associated with an ROI."""

    shapeType: RoiShapeType
    interpType: InterpType
    cookie: int = 0
    color: int = 255
    label: str = ""
    # everything will default to zero, EVEN if "use as stimulation" is not checked
    # use interpType to determine if it's a stimulation ROI
    stimulationGroup: int = 0
    scope: int = 1
    appData: int = 0
    multiFrame: bool = False
    locked: bool = False
    compCount: int = 2
    bpc: int = 16
    autodetected: bool = False
    gradientStimulation: bool = False
    gradientStimulationBitDepth: int = 0
    gradientStimulationLo: float = 0.0
    gradientStimulationHi: float = 0.0

    def __post_init__(self):
        # coerce types
        for key, anno in self.__annotations__.items():
            if key == "shapeType":
                self.shapeType = RoiShapeType(self.shapeType)
            elif key == "interpType":
                self.interpType = InterpType(self.interpType)
            else:
                type_ = getattr(builtins, anno)
                setattr(self, key, type_(getattr(self, key)))
