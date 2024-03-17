from __future__ import annotations

import builtins
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, Union

from ._sdk_types import EventMeaning, StimulationType

if TYPE_CHECKING:
    from ._sdk_types import AxisInterpretation, LoopTypeString


class TextInfo(TypedDict, total=False):
    imageId: str
    type: str
    group: str
    sampleId: str
    author: str
    description: str
    capturing: str
    sampling: str
    location: str
    date: str
    conclusion: str
    info1: str
    info2: str
    optics: str
    appVersion: str


class LoopType(IntEnum):
    Unknown = 0
    TimeLoop = 1
    XYPosLoop = 2
    XYDiscrLoop = 3
    ZStackLoop = 4
    PolarLoop = 5
    SpectLoop = 6
    CustomLoop = 7
    NETimeLoop = 8
    ManTimeLoop = 9
    ZStackLoopAccurate = 10


# tuples


class Attributes(NamedTuple):
    bitsPerComponentInMemory: int
    bitsPerComponentSignificant: int
    componentCount: int
    heightPx: int
    pixelDataType: Literal["float", "unsigned"]
    sequenceCount: int
    widthBytes: int | None = None
    widthPx: int | None = None
    compressionLevel: float | None = None
    compressionType: Literal["lossless", "lossy", "none"] | None = None
    tileHeightPx: int | None = None
    tileWidthPx: int | None = None
    channelCount: int | None = None


class ImageInfo(NamedTuple):
    width: int
    height: int
    components: int
    bits_per_pixel: int


# experiment #################


@dataclass
class _Loop:
    count: int
    nestingLevel: int
    parameters: LoopParams | None
    type: LoopTypeString


@dataclass
class SpectLoop:
    count: int
    type: Literal["SpectLoop"] = "SpectLoop"


@dataclass
class CustomLoop(_Loop):
    count: int
    nestingLevel: int = 0
    parameters: None = None
    type: Literal["CustomLoop"] = "CustomLoop"


#####


@dataclass
class TimeLoop(_Loop):
    """The time dimension of an experiment."""

    parameters: TimeLoopParams
    type: Literal["TimeLoop"] = "TimeLoop"

    def __post_init__(self) -> None:
        # TODO: make superclass do this
        if isinstance(self.parameters, dict):
            if "periodDiff" not in self.parameters:
                self.parameters["periodDiff"] = None
            self.parameters = TimeLoopParams(**self.parameters)


@dataclass
class TimeLoopParams:
    startMs: float
    periodMs: float
    durationMs: float
    periodDiff: PeriodDiff

    def __post_init__(self) -> None:
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
    """The time dimension of an nD experiment."""

    parameters: NETimeLoopParams
    type: Literal["NETimeLoop"] = "NETimeLoop"

    def __post_init__(self) -> None:
        if isinstance(self.parameters, dict):
            self.parameters = NETimeLoopParams(**self.parameters)


@dataclass
class NETimeLoopParams:
    periods: list[Period]

    def __post_init__(self) -> None:
        self.periods = [Period(**i) if isinstance(i, dict) else i for i in self.periods]


@dataclass
class Period(TimeLoopParams):
    count: int


#########


@dataclass
class XYPosLoop(_Loop):
    parameters: XYPosLoopParams
    type: Literal["XYPosLoop"] = "XYPosLoop"

    def __post_init__(self) -> None:
        if isinstance(self.parameters, dict):
            self.parameters = XYPosLoopParams(**self.parameters)


@dataclass
class XYPosLoopParams:
    isSettingZ: bool
    points: list[Position]

    def __post_init__(self) -> None:
        self.points = [Position(**i) if isinstance(i, dict) else i for i in self.points]


@dataclass
class Position:
    stagePositionUm: StagePosition
    pfsOffset: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stagePositionUm, dict):
            self.stagePositionUm = StagePosition(*self.stagePositionUm)
        elif isinstance(self.stagePositionUm, (tuple, list)):
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

    def __post_init__(self) -> None:
        if isinstance(self.parameters, dict):
            self.parameters = ZStackLoopParams(**self.parameters)


@dataclass
class ZStackLoopParams:
    homeIndex: int
    stepUm: float
    bottomToTop: bool
    deviceName: str | None = None


###

ExpLoop = Union[TimeLoop, NETimeLoop, XYPosLoop, ZStackLoop, CustomLoop]
LoopParams = Union[TimeLoopParams, NETimeLoopParams, XYPosLoopParams, ZStackLoopParams]

# metadata #################


@dataclass
class Metadata:
    contents: Contents | None = None
    channels: list[Channel] | None = None

    def __post_init__(self) -> None:
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
    loops: LoopIndices | None
    microscope: Microscope
    volume: Volume

    def __post_init__(self) -> None:
        if isinstance(self.channel, dict):
            self.channel = ChannelMeta(**self.channel)
        if isinstance(self.microscope, dict):
            self.microscope = Microscope(**self.microscope)
        if isinstance(self.volume, dict):
            self.volume = Volume(**self.volume)
        if isinstance(self.loops, dict):
            self.loops = LoopIndices(**self.loops)


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: float = 1.0

    def as_hex(self) -> str:  # pragma: no cover
        """Return color as a hex string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @classmethod
    def from_abgr_u4(cls, val: int) -> Color:
        """Create a color from an unsigned 4-byte (32-bit) integer in ABGR format."""
        return cls(
            r=val & 255,
            g=val >> 8 & 255,
            b=val >> 16 & 255,
            # it's not clear if the alpha channel is used in NIS Elements
            # so we default to 1.0 if it comes in as 0
            a=((val >> 24 & 255) / 255) or 1.0,
        )

    def as_abgr_u4(self) -> int:
        """Return color as an unsigned 4-byte (32-bit) integer in ABGR format.

        This is the native format of NIS Elements.
        """
        # for the sake of round-tripping, we'll assume that 1.0 alpha is 0
        alpha = 0 if self.a == 1.0 else int(self.a * 255)
        return (alpha << 24) + (self.b << 16) + (self.g << 8) + self.r


@dataclass
class ChannelMeta:
    name: str
    index: int
    color: Color
    emissionLambdaNm: float | None = None
    excitationLambdaNm: float | None = None

    @property
    def colorRGBA(self) -> int:
        """Return color as unsigned 4-byte (32-bit) integer in ABGR format."""
        warnings.warn(
            "`.colorRGBA` is deprecated, use `.color..as_abgr_u4()` instead "
            "if you want the color in the original 32-bit ABGR format.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.color.as_abgr_u4()


@dataclass
class LoopIndices:
    NETimeLoop: int | None = None
    TimeLoop: int | None = None
    XYPosLoop: int | None = None
    ZStackLoop: int | None = None
    CustomLoop: int | None = field(default=None, repr=False, compare=False)


ModalityFlags = Literal[
    "aux",
    "brightfield",
    "camera",
    "diContrast",
    "dsdConfocal",
    "fluorescence",
    "gaasp",
    "iSIM",
    "laserScanConfocal",
    "liveSR",
    "multiphoton",
    "nonDescannedDetector",
    "phaseContrast",
    "pmt",
    "RCM",
    "remainder",
    "SIM",
    "sora",
    "spectral",
    "spinningDiskConfocal",
    "sweptFieldConfocalPinhole",
    "sweptFieldConfocalSlit",
    "TIRF",
    "transmitDetector",
    "vaasIF",
    "vaasNF",
    "VCS",
    "virtualFilter",
]


@dataclass
class Microscope:
    objectiveMagnification: float | None = None
    objectiveName: str | None = None
    objectiveNumericalAperture: float | None = None
    zoomMagnification: float | None = None
    immersionRefractiveIndex: float | None = None
    projectiveMagnification: float | None = None
    pinholeDiameterUm: float | None = None
    modalityFlags: list[ModalityFlags] = field(default_factory=list)


@dataclass
class Volume:
    axesCalibrated: tuple[bool, bool, bool]
    axesCalibration: tuple[float, float, float]
    axesInterpretation: tuple[
        AxisInterpretation, AxisInterpretation, AxisInterpretation
    ]
    bitsPerComponentInMemory: int
    bitsPerComponentSignificant: int
    cameraTransformationMatrix: tuple[float, float, float, float]
    componentCount: int
    componentDataType: Literal["unsigned", "float"]
    voxelCount: tuple[int, int, int]
    componentMaxima: list[float] | None = None
    componentMinima: list[float] | None = None
    pixelToStageTransformationMatrix: (
        tuple[float, float, float, float, float, float] | None
    ) = None

    # NIS Microscope Absolute frame in um =
    # pixelToStageTransformationMatrix * (X_in_px,  Y_in_px,  1) + stagePositionUm


@dataclass
class TimeStamp:
    absoluteJulianDayNumber: float
    relativeTimeMs: float


@dataclass
class FrameChannel(Channel):
    position: Position
    time: TimeStamp

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.position, dict):
            self.position = Position(**self.position)
        if isinstance(self.time, dict):
            self.time = TimeStamp(**self.time)


@dataclass
class FrameMetadata:
    contents: Contents
    channels: list[FrameChannel]

    def __post_init__(self) -> None:
        if isinstance(self.contents, dict):
            self.contents = Contents(**self.contents)
        self.channels = [
            FrameChannel(**i) if isinstance(i, dict) else i for i in self.channels
        ]


class Coordinate(NamedTuple):
    index: int  # type: ignore
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
    basePoints: list[XYPoint] = field(default_factory=list)

    @classmethod
    def _from_meta_dict(cls, val: dict) -> ExtrudedShape:
        return cls(
            sizeZ=val.get("SizeZ") or val.get("sizeZ") or 0,
            basePoints=[
                XYPoint(*val[f"BasePoints_{i}"].values())
                for i in range(val.get("BasePoints_Size", 0))
            ],
        )


@dataclass
class ROI:
    """ROI object from NIS Elements."""

    id: int
    info: RoiInfo
    guid: str
    animParams: list[AnimParam] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.info, dict):
            self.info = RoiInfo(**self.info)
        self.animParams = [
            AnimParam(**i) if isinstance(i, dict) else i for i in self.animParams
        ]

    @classmethod
    def _from_meta_dict(cls, val: dict) -> ROI:
        # val has keys:
        # 'Id', 'Info', 'GUID', 'AnimParams_Size', 'AnimParams_{i}'
        # where GUID and AnimParams keys are optional
        anim_params = [
            AnimParam(
                **{
                    _lower0(k): v
                    for k, v in val[f"AnimParams_{i}"].items()
                    if _lower0(k) in AnimParam.__annotations__
                }
            )
            for i in range(val.get("AnimParams_Size", 0))
        ]
        info = RoiInfo(
            **{
                _lower0(k): v
                for k, v in val["Info"].items()
                if _lower0(k) in RoiInfo.__annotations__
            }
        )
        return cls(
            id=val["Id"],
            info=info,
            guid=val.get("GUID", ""),
            animParams=anim_params,
        )


class T(TypedDict):
    Id: int
    Info: dict
    GUID: str
    AnimParams_Size: int
    # AnimParams_{i}: dict


@dataclass
class AnimParam:
    """Parameters of ROI position/shape."""

    timeMs: float = 0
    enabled: bool = True
    centerX: float = 0
    centerY: float = 0
    centerZ: float = 0
    rotationZ: float = 0
    boxShape: BoxShape = field(default_factory=BoxShape)
    extrudedShape: ExtrudedShape = field(default_factory=ExtrudedShape)

    def __post_init__(self) -> None:
        if isinstance(self.boxShape, dict):
            self.boxShape = BoxShape(
                **{
                    _lower0(k): v
                    for k, v in self.boxShape.items()
                    if _lower0(k) in BoxShape.__annotations__
                }
            )
        if isinstance(self.extrudedShape, dict):
            self.extrudedShape = ExtrudedShape._from_meta_dict(self.extrudedShape)

    @property
    def center(self) -> XYZPoint:
        """Center point as a named tuple (x, y, z)."""
        return XYZPoint(self.centerX, self.centerY, self.centerZ)


class RoiShapeType(IntEnum):
    """The type of ROI shape."""

    Any = 0
    Raster = 1
    Point = 2
    Rectangle = 3
    Ellipse = 4
    Polygon = 5
    Bezier = 6
    Line = 7
    PolyLine = 8
    Circle = 9
    Square = 10
    Ring = 11
    Spiral = 12


class InterpType(IntEnum):
    """The role that the ROI plays."""

    AnyROI = 0
    StandardROI = 1
    BackgroundROI = 2
    ReferenceROI = 3
    StimulationROI = 4


class ScopeType(IntEnum):
    Any = 0
    Global = 1
    MPoint = 2


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
    scope: ScopeType = ScopeType.Global
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

    def __post_init__(self) -> None:
        # coerce types
        for key, anno in self.__annotations__.items():
            if key == "shapeType":
                self.shapeType = RoiShapeType(self.shapeType)
            elif key == "interpType":
                self.interpType = InterpType(self.interpType)
            elif key == "scope":
                self.scope = ScopeType(self.scope)
            else:
                type_ = getattr(builtins, anno)
                setattr(self, key, type_(getattr(self, key)))


@dataclass
class ExperimentEvent:
    id: int = 0  #  ID of the event
    # meaning of the time/time2 could be found in the globalmetadata-eTimeSource
    time: float = 0.0  # time in msec, in the same axis as time in image metaformat
    time2: float = 0.0  # time in msec, similar to acqtime2
    meaning: EventMeaning = EventMeaning.Unspecified
    description: str = ""  # description that the user typed in (if any)
    data: str = ""  # the additional data (command code, macro file path etc.)
    stimulation: StimulationEvent | None = None


@dataclass
class StimulationEvent:
    type: StimulationType = StimulationType.NoStimulation
    loop_index: int = 0
    position: int = 0
    description: str = ""
