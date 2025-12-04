"""Task type definitions for JOBS workflow tasks.

The JOBS task structure is generic - tasks share a common structure with
Data/Parameters fields containing nested dictionaries whose structure varies
based on the specific task configuration. The Name field is a user-provided
label, not a structural type indicator.
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypeAlias, TypedDict

class SlotConnection(TypedDict, total=False):
    """Connection from one task's output to another's input."""

    SlotType: int
    """Type identifier for the slot.  Likely a bit flag enum."""
    ParameterName: str
    """Name of the parameter being connected (e.g., 'Plate.Wellplate').

    Likely follow a TaskName.PropertyPath pattern.
    """
    ParameterType: int
    """Type of the parameter. Usually just 0 or 1."""

class SlotConnectionsWrapper(TypedDict, total=False):
    """Wrapper for slot connections."""

    SlotConnections: dict[str, SlotConnection]
    """Dictionary mapping slot names (Slot0, Slot1, etc.) to connections."""

# this is a loose collection of observations, not a definitive list
# sometimes keys will appear with commas, like "Point, SampleArea, RegionDefinition"
LockableParamName: TypeAlias = (
    Literal[
        "Action",
        "AdvancedZ-StackSettings",
        "AutoFocus",
        "BinaryList",
        "CaptureDefinition",
        "Capturedata",
        "Experiment",
        "Field Size",
        "FinishWhen",
        "FramesOnBorder",
        "JobTaskCaptureStorage",
        "JobTaskForLoop",
        "LabelIdList",
        "LabelList",
        "LargeImageEnable",
        "MessageBody",
        "ObjectivesOffsetCorrection",
        "OptConfig",
        "OrderingType",
        "PFSPosition",
        "PerfusionTreatmentList",
        "Point",
        "PointPlacement",
        "PointSet",
        "RegionDefinition",
        "RegionList",
        "RunWhileWaiting",
        "SampleArea",
        "SaveOptions",
        "ScanDirection",
        "Settings.AutoMode",
        "Settings.Criterion",
        "Settings.OptConfig",
        "Settings.ZDevice",
        "ShadingOptions",
        "Show Plate Preview",
        "Show Task Name",
        "Slide",
        "SlideList, ",
        "SpecifyObjective",
        "Stimulation Area",
        "Storage",
        "TTLSignal",
        "WellLabelList",
        "WellPlate",
        "WellPlateList",
        "WellPlateList, ",
        "WellSelection",
        "ZPosition",
        "ZStack",
        "ZStack.Absolute",
    ]
    | str
)

class LockableParamsWrapper(TypedDict, total=False):
    """Wrapper for lockable parameters."""

    LockableParams: dict[LockableParamName, bool]
    """Dictionary mapping parameter names to locked state."""

class Parameter(TypedDict):
    Name: str
    Type: int  # 0, 1, 2, 3, 4, or 7
    Value: int  # 0, 1, 2, or 3

class ParametersWrapper(TypedDict, total=False):
    """Wrapper for task parameters."""

    Parameters: dict[str, Any]
    """Task-specific parameters structure.

    Keys are 'Parameter0', 'Parameter1', etc., each mapping to a Parameter dict.
    """

class Task(TypedDict, total=False):
    """JOBS workflow task structure.

    All JOBS tasks share this common structure. The Data and Parameters fields
    contain nested structures that vary based on the task configuration, but
    follow generic patterns (dicts with keys like 'CLxJobTask_*', 'Settings',
    'CaptureLambda', etc.).

    The Name field is a user-provided label for the task, not a type indicator.
    Task dict keys in the Tasks dict are also user-provided identifiers.
    """

    Key: int
    """Unique key identifying this task within the job."""
    JobtaskKeyRef: int
    """Reference to parent task key (0 if none)."""
    BlockNumber: int
    """Block number for task grouping."""
    Order: int
    """Execution order within the block."""
    Version: int
    """Task version number."""
    Name: str
    """User-provided label for this task."""
    Data: dict[str, DataDictUnion | dict] | list[Any]
    """Task-specific data structure (varies by task configuration).

    Generally, the dict will have a *single* top-level key like 'CLxJobTask_*',
    'Settings', 'CaptureLambda', etc., whose value is another dict or list
    containing the actual task configuration.
    """
    Parameters: ParametersWrapper
    """Task-specific parameters."""
    SlotConnections: SlotConnectionsWrapper | list[Any]  # list is always empty []
    """Connections to other tasks' outputs."""
    Notes: str
    """User notes for this task."""
    NotesDetailed: str
    """Detailed description of what this task does."""
    StateFlags: int
    """State flags for the task."""
    LockableParams: LockableParamsWrapper
    """Parameters that can be locked/unlocked."""

# ##########################################################
# Task.Data
# ##########################################################

# These dicts are provided just as a convenient list of observed task data structures.
# They are not exhaustive or definitive; many variations likely exist.
# But as you parse a Task.Data field, you may find that your payload matches one
# of these structures, and you can cast(...) if desired.

class AssignExposure(TypedDict, total=False):
    Exposure: float
    ExposureRel: float
    OptConf: str
    Relative: int
    Variable: str
    VariableRel: str

class CLxAnalysis_Ga3(TypedDict, total=False):
    # NOTE: empty string key observed
    # "": list
    ExportParametersModel: dict
    Info: str
    IsAfterRun: bool
    ListModels: str
    Refresh: bool
    Settings: list | str
    StimROIParametersModel: dict

class CLxJobTask_AlignLayout(TypedDict, total=False):
    NumPoints: int
    Point0_x: float
    Point0_y: float
    Point1_x: float
    Point1_y: float

class CLxJobTask_AlignWellplate_v2(TypedDict, total=False):
    AlignedUUID: str
    CLxJobTaskParamArray: dict
    Type: str

class CLxJobTask_AssayWrapper(TypedDict, total=False):
    AfterAcquisition: bool
    AssayDefinition: str
    Description: str
    ModuleName: str

class CLxJobTask_AssignCaptureDefinition(TypedDict, total=False):
    Label: str

class CLxJobTask_AssignZStack(TypedDict, total=False):
    Offset: float
    WellRowColumn: int

class CLxJobTask_CaptureLambdaAssayLI(TypedDict, total=False):
    CLxJobTask_CaptureLambda: dict
    ImageRegistration: int
    LargeImageEnable: int
    LargeImageXFields: int
    LargeImageYFields: int
    ShadingPostprocess: int

class CLxJobTask_CaptureLambdaNSim(TypedDict, total=False):
    CLxJobTask_CaptureLambda: dict
    CaptureDefinition: dict
    NSimMode: int

class CLxJobTask_DefineFocusSurface(TypedDict, total=False):
    InterpolationMethod: int

class CLxJobTask_DefinePFSSurface(TypedDict, total=False):
    InterpolationMethod: int

class CLxJobTask_DrawRegion(TypedDict, total=False):
    RegionDivision: int
    WindowPos: int
    WindowSize: int

class CLxJobTask_EveryNth(TypedDict, total=False):
    BottomCount: int
    BottomEveryNth: int
    BottomUse: int
    ColumnOffset: int
    ColumnRadioSelection: int
    EveryNth: int
    EveryNthColumn: int
    HomeCount: int
    HomeEveryNth: int
    HomeUse: int
    IncludeFirst: int
    Nodes: dict
    NthWellType: int
    OneWellIndex: int
    RadioSelection: int
    RowOffset: int
    TopCount: int
    TopEveryNth: int
    TopUse: int

class CLxJobTask_FastTimelapse(TypedDict, total=False):
    Count: int
    MemoryOnly: int
    OptConfig: str
    TTLLine: int
    TTLSignal: int

class CLxJobTask_FocusSurface(TypedDict, total=False):
    ZDeviceDesc: dict

class CLxJobTask_KeepObjectInview(TypedDict, total=False):
    Channel: int
    PointSource: int

class CLxJobTask_LargeImageScan(TypedDict, total=False):
    AFAfterDistance: float
    AFAfterDistanceUnit: int
    AFEveryField: int
    AutoCorrectionSplines: int
    AutoCorrectionType: int
    AutoFocusPeriodicityType: int
    AutomaticPostprocessing: int
    BorderRestriction: float
    CenterRestriction: float
    ClipToRegion: int
    CorrectObjectiveOffset: int
    FieldsPlacement: int
    FieldsX: int
    FieldsY: int
    FluorescenceOffset: int
    ImageRegistration: int
    Objective: str
    OverlapPercent: float
    RegionDilatation: float
    RemoveDust: int
    SaveOption: int
    ShadingCorrection: int
    StitchUsingChannel: str
    StitchingType: int
    UseAutoFocus: int
    UseBorderRestriction: int
    UseCenterRestriction: int
    UseFocusSurface: int
    UseRegionDilatation: int

class CLxJobTask_LoopPoints(TypedDict, total=False):
    AdvancedSettings: bool
    ForClass: str
    ForLabel: str
    MoveXY: int
    MoveZ: int
    Storage: dict
    UsePFSOffset: int

class CLxJobTask_LoopRegions(TypedDict, total=False):
    Storage: dict

class CLxJobTask_LoopTime(TypedDict, total=False):
    Count: int
    DurationCount: int
    DurationDisplayUnit: int
    DurationMs: float
    Expression: str
    IdleDisplayUnit: int
    IdleInterval: float
    IntervalDisplayUnit: int
    IntervalMs: float
    RunOnIdle: bool
    Speed: int
    Storage: dict
    UseExpression: int
    WaitDisplayUnit: int
    WaitMs: float

class CLxJobTask_LoopTimeSequence(TypedDict, total=False):
    Loop0: dict

class CLxJobTask_LoopWells(TypedDict, total=False):
    AdvancedSettings: bool
    ForClass: str
    ForLabel: str
    Storage: dict

class CLxJobTask_LoopZStack(TypedDict, total=False):
    MoveZ: int

class CLxJobTask_ManualPointSet_V2(TypedDict, total=False):
    ClearROIs: bool
    IncludeCamera: bool
    IncludeLabels: bool
    IncludePFS: bool
    IncludeROI: bool
    IncludeZ: bool
    IncludeZStack: bool
    Positions: list
    ShowAdvanced: bool
    StimDevice: str
    UseLaserPower: bool

class CLxJobTask_MoveToPoint(TypedDict, total=False):
    MoveXY: int
    MoveZ: int
    UsePFSOffset: int

class CLxJobTask_MoveToWellCenter(TypedDict, total=False):
    MoveZ: int
    UsePFSOffset: int

class CLxJobTask_MoveToZ(TypedDict, total=False):
    DeviceVarinatVersion: int
    ZDeviceDesc: dict

class CLxJobTask_OffsetFocusSurface(TypedDict, total=False):
    Offset: float

class CLxJobTask_RegionAppend(TypedDict, total=False):
    PtSet_From: str
    PtSet_To: str

class CLxJobTask_RegisterRegion(TypedDict, total=False):
    OptConf: str

class CLxJobTask_STORMAcq(TypedDict, total=False):
    CLxJobTaskParamSTORMAcqSettings: dict
    StoreToND2: bool

class CLxJobTask_SetClass(TypedDict, total=False):
    Class: int

class CLxJobTask_SetLabel(TypedDict, total=False):
    Label: str
    ResetLabels: int

class CLxJobTask_StimROIFromAnalysis(TypedDict, total=False):
    CLxJobTaskParamStimArea: dict
    S1: str
    S2: str
    S3: str

class CLxJobTask_Storage(TypedDict, total=False):
    FileType: str
    Multichannel: int
    SaveAsMCH: int
    SaveOMEMetadata: int
    Split: int

class CLxJobTask_StoreToFsOnly(TypedDict, total=False):
    CreateSubfolderForEachRun: int
    Folder: str
    OpenFilesAfterRun: int

class CLxJobTask_StoreToFsOnlyEx(TypedDict, total=False):
    Filename: str
    Folder: str
    SaveIntoFolder: int

class CLxJobTask_TTLOut(TypedDict, total=False):
    NDAcquisitionIO: dict

class CLxJobTask_Variables(TypedDict, total=False):
    VariablesList: list

class CLxJobTask_Wait(TypedDict, total=False):
    DurationDisplayUnit: int
    DurationMs: float

class CLxJobTask_WellplateListManual(TypedDict, total=False):
    FirstOnStage: int
    UseHolder: int
    Wellplates: dict

class CLxJobTask_ZStackDefinition(TypedDict, total=False):
    AutoDevice: bool
    DeviceVarinatVersion: int
    ZCombined: bool
    ZDeviceDesc: dict
    ZFullName: str
    ZStack: dict
    ZTriggered: bool

class CaptureLambda(TypedDict, total=False):
    CalculatorZStep: float
    FocusOffsetSetupOpened: int
    IgnoreObjectives: int
    LambdaList: dict
    OpticalConfigsLite: dict
    ZReference: int

class Comment(TypedDict, total=False):
    CommentText: str

class DishDefinition(TypedDict, total=False):
    Alignment: dict
    CLxSingleElementHolder: dict

class Expression(TypedDict, total=False):
    ExpressionText: str

class JobTask_NDStimulationSim(TypedDict, total=False):
    OptConf: str
    SLxExperiment: dict
    Settings: dict
    StimDevice: str
    StimType: int

class Labels(TypedDict, total=False):
    Label0: dict
    Label1: dict

class Live(TypedDict, total=False):
    Action: int
    Camera: str
    WindowPos: int
    WindowSize: int

class LoopForCount_V2(TypedDict, total=False):
    Count: int
    Expression: str
    Method: int
    UseExpression: int

class Macro(TypedDict, total=False):
    ExecutedOnMainThread: bool
    MacroText: str

class OptConf(TypedDict, total=False):
    OptConf: str

class PFS(TypedDict, total=False):
    FailMethod: int
    FailMoveZDirection: int
    FailMoveZDown: float
    FailMoveZUp: float
    FocusInterval: int
    OnFail: int

class Question(TypedDict, total=False):
    AlwaysVisible: bool
    BtnIsUsed1: bool
    BtnIsUsed10: bool
    BtnIsUsed11: bool
    BtnIsUsed12: bool
    BtnIsUsed13: bool
    BtnIsUsed14: bool
    BtnIsUsed15: bool
    BtnIsUsed16: bool
    BtnIsUsed17: bool
    BtnIsUsed18: bool
    BtnIsUsed19: bool
    BtnIsUsed2: bool
    BtnIsUsed20: bool
    BtnIsUsed21: bool
    BtnIsUsed22: bool
    BtnIsUsed23: bool
    BtnIsUsed24: bool
    BtnIsUsed25: bool
    BtnIsUsed26: bool
    BtnIsUsed27: bool
    BtnIsUsed28: bool
    BtnIsUsed29: bool
    BtnIsUsed3: bool
    BtnIsUsed30: bool
    BtnIsUsed31: bool
    BtnIsUsed32: bool
    BtnIsUsed4: bool
    BtnIsUsed5: bool
    BtnIsUsed6: bool
    BtnIsUsed7: bool
    BtnIsUsed8: bool
    BtnIsUsed9: bool
    Button10Pressed: str
    Button11Pressed: str
    Button12Pressed: str
    Button13Pressed: str
    Button14Pressed: str
    Button15Pressed: str
    Button16Pressed: str
    Button17Pressed: str
    Button18Pressed: str
    Button19Pressed: str
    Button1Pressed: str
    Button20Pressed: str
    Button21Pressed: str
    Button22Pressed: str
    Button23Pressed: str
    Button24Pressed: str
    Button25Pressed: str
    Button26Pressed: str
    Button27Pressed: str
    Button28Pressed: str
    Button29Pressed: str
    Button2Pressed: str
    Button30Pressed: str
    Button31Pressed: str
    Button32Pressed: str
    Button3Pressed: str
    Button4Pressed: str
    Button5Pressed: str
    Button6Pressed: str
    Button7Pressed: str
    Button8Pressed: str
    Button9Pressed: str
    Caption: str
    Edit1: str
    Edit2: str
    Edit3: str
    Edit4: str
    EditValue1: str
    EditValue2: str
    EditValue3: str
    EditValue4: str
    Embedded: bool
    Enter1: bool
    Enter10: bool
    Enter11: bool
    Enter12: bool
    Enter13: bool
    Enter14: bool
    Enter15: bool
    Enter16: bool
    Enter17: bool
    Enter18: bool
    Enter19: bool
    Enter2: bool
    Enter20: bool
    Enter21: bool
    Enter22: bool
    Enter23: bool
    Enter24: bool
    Enter25: bool
    Enter26: bool
    Enter27: bool
    Enter28: bool
    Enter29: bool
    Enter3: bool
    Enter30: bool
    Enter31: bool
    Enter32: bool
    Enter4: bool
    Enter5: bool
    Enter6: bool
    Enter7: bool
    Enter8: bool
    Enter9: bool
    Esc1: bool
    Esc10: bool
    Esc11: bool
    Esc12: bool
    Esc13: bool
    Esc14: bool
    Esc15: bool
    Esc16: bool
    Esc17: bool
    Esc18: bool
    Esc19: bool
    Esc2: bool
    Esc20: bool
    Esc21: bool
    Esc22: bool
    Esc23: bool
    Esc24: bool
    Esc25: bool
    Esc26: bool
    Esc27: bool
    Esc28: bool
    Esc29: bool
    Esc3: bool
    Esc30: bool
    Esc31: bool
    Esc32: bool
    Esc4: bool
    Esc5: bool
    Esc6: bool
    Esc7: bool
    Esc8: bool
    Esc9: bool
    Icon: int
    IndexBtnGroup: int
    KeepVisible: bool
    Text: str
    UseEdit: bool

class Settings(TypedDict, total=False):
    AutoMode: int
    BorderExclusionPercent: float
    Camera: int
    CenterRestriction: float
    Channel: str
    CompensatingZDriveID: str
    ControlActiveShutter: int
    CorrectXYObjectivesOffset: int
    Criterion: int
    CustomFieldSize: int
    Definition: dict
    EdgeRestriction: float
    FineStep: float
    GeneratePointsOnStage: int
    LastAreaRestrictionShape: dict
    LastDisplacementX: float
    LastDisplacementY: float
    LastElementAreaRestrictionType: int
    LastElementWorkingAreaType: int
    LastOverlapValue: float
    LastStageAreaRestrictionType: int
    LastStageWorkingAreaType: int
    LastWorkingAreaShape: dict
    ManualPoints: dict
    Method: int
    Objective: str
    Offset: float
    OptConfig: str
    OrigZOnFail: int
    Range: float
    SinglePointDisplaced: int
    SkipBackground: int
    Speed: float
    Step: float
    Triggered: int
    TwoPasses: int
    UseAFPlane: int
    UseBorderExclusion: int
    UseCenterRestriction: int
    UseEdgeRestriction: int
    UseOverlap: bool
    ZDriveID: str

class SlideDefinition(TypedDict, total=False):
    CLxSingleElementHolder: dict
    NumPoints: int
    Points: list
    SlideLimitWorkingArea: int
    SlideOrientation: int
    WorkingAreaDefineBy: int
    WorkingAreaShape: int

class SlideLoader(TypedDict, total=False):
    Hotel0: dict

class SlideLoader_ver2(TypedDict, total=False):
    List: dict
    SlideDef: dict

class SmsNotification(TypedDict, total=False):
    Carrier: str
    Country: str
    MessageBody: str
    PhoneNumber: str

class StageArea(TypedDict, total=False):
    CaptureOnDefine: int
    CircleCenterX: float
    CircleCenterY: float
    CircleRadius: float
    DefinitionType: int
    Dilation: float
    PointsCircle: dict
    PointsConvexHull: dict
    PointsPolygon: dict
    PointsRect2: dict
    PointsRect3: dict
    PointsRect4: dict
    UiState: dict

class TaskBreakState(TypedDict, total=False):
    BreakJob: int
    BreakLoopName: str

class UNDO(TypedDict, total=False):
    Camera: dict
    Sequence: dict

class WellSelectionSettings(TypedDict, total=False):
    Color: int
    OrderingType: int
    SelectionMask: dict

class WellplateLabeling(TypedDict, total=False):
    PresetLinkType: int
    Presets0: dict

class WellplateUUID(TypedDict, total=False):
    BARCODE: str
    NAME: str
    UUID: str

DataDictUnion: TypeAlias = """AssignExposure |
    CLxAnalysis_Ga3 |
    CLxJobTask_AlignLayout |
    CLxJobTask_AlignWellplate_v2 |
    CLxJobTask_AssayWrapper |
    CLxJobTask_AssignCaptureDefinition |
    CLxJobTask_AssignZStack |
    CLxJobTask_CaptureLambdaAssayLI |
    CLxJobTask_CaptureLambdaNSim |
    CLxJobTask_DefineFocusSurface |
    CLxJobTask_DefinePFSSurface |
    CLxJobTask_DrawRegion |
    CLxJobTask_EveryNth |
    CLxJobTask_FastTimelapse |
    CLxJobTask_FocusSurface |
    CLxJobTask_KeepObjectInview |
    CLxJobTask_LargeImageScan |
    CLxJobTask_LoopPoints |
    CLxJobTask_LoopRegions |
    CLxJobTask_LoopTime |
    CLxJobTask_LoopTimeSequence |
    CLxJobTask_LoopWells |
    CLxJobTask_LoopZStack |
    CLxJobTask_ManualPointSet_V2 |
    CLxJobTask_MoveToPoint |
    CLxJobTask_MoveToWellCenter |
    CLxJobTask_MoveToZ |
    CLxJobTask_OffsetFocusSurface |
    CLxJobTask_RegionAppend |
    CLxJobTask_RegisterRegion |
    CLxJobTask_STORMAcq |
    CLxJobTask_SetClass |
    CLxJobTask_SetLabel |
    CLxJobTask_StimROIFromAnalysis |
    CLxJobTask_Storage |
    CLxJobTask_StoreToFsOnly |
    CLxJobTask_StoreToFsOnlyEx |
    CLxJobTask_TTLOut |
    CLxJobTask_Variables |
    CLxJobTask_Wait |
    CLxJobTask_WellplateListManual |
    CLxJobTask_ZStackDefinition |
    CaptureLambda |
    Comment |
    DishDefinition |
    Expression |
    JobTask_NDStimulationSim |
    Labels |
    Live |
    LoopForCount_V2 |
    Macro |
    OptConf |
    PFS |
    Question |
    Settings |
    SlideDefinition |
    SlideLoader |
    SlideLoader_ver2 |
    SmsNotification |
    StageArea |
    TaskBreakState |
    UNDO |
    WellSelectionSettings |
    WellplateLabeling |
    WellplateUUID
"""
