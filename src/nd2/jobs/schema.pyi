"""TypedDict definitions for JOBS metadata schema.

This module provides type definitions for the JOBS workflow metadata found in ND2 files.
JOBS is Nikon's workflow automation system for microscopy acquisition.

The schema was inferred from analyzing 42 .bin job definition files and
job metadata embedded in ND2 files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Required, TypedDict

if TYPE_CHECKING:
    from ._tasks import Task

class JobsDict(TypedDict, total=False):
    """Top-level JOBS metadata from ND2 files.

    This is what `ND2File.jobs` returns. It wraps the Job definition
    with additional runtime metadata.
    """

    JobRunGUID: str
    """Unique identifier for this specific job run."""
    ProgramDesc: ProgramDesc
    """Program description metadata."""
    Job: Job
    """The actual job definition."""
    ProtectedJob: ProtectedJob | None
    """Encryption info if job is protected, None otherwise."""
    Custom_wellplate: Custom_wellplate
    """Custom wellplate definition at top level (for standalone .bin files)."""

class ProgramDesc(TypedDict, total=False):
    """Program description metadata."""

    JobDefType: int

class ProtectedJob(TypedDict):
    """Encryption information for protected jobs."""

    EncryptedJobDefinition: list[int]
    """Encrypted job data as byte array."""
    HashCheck: list[int]
    """Hash for verification."""

class Job(TypedDict):
    """Main job definition structure.

    This is the core structure that defines a JOBS workflow.
    It appears both in ND2 files (inside JobsDict) and in standalone
    .bin job definition files.
    """

    CustomDefinitions: CustomDefinitions
    """Custom definitions like wellplate configurations."""
    Tasks: TasksDict
    """Dictionary mapping task names to task definitions."""
    PropertyDefinitions: PropertyDefinitions
    """User-configurable property definitions."""
    Property: JobProperty
    """Job metadata like name and description."""
    ProgramParameters: ProgramParameters
    """Program-level parameters."""

class JobProperty(TypedDict, total=False):
    """Job metadata properties."""

    Name: Required[str]
    """Name of the job."""
    Desc: Required[str]
    """Description of the job."""
    State: Required[int]
    """Job state flag."""
    Usability: Required[list[int]]
    """Usability flags."""
    ProgressProperties: int
    """Progress properties version 1."""
    ProgressProperties_v2: int
    """Progress properties version 2."""

class ProgramParameters(TypedDict):
    """Program-level parameters."""

    Summary: str
    """Summary text for the program."""
    Properties: dict[str, Any] | list[Any]
    """Additional properties."""

class CustomDefinitions(TypedDict, total=False):
    """Custom definitions container.

    Contains custom configurations like wellplate definitions.
    """

    CustomWellplate: CustomWellplate

class Custom_wellplate(TypedDict):
    """Top-level custom wellplate definition wrapper.

    This appears at the top level in standalone .bin files,
    separate from the Job structure.
    """

    CLxWellplate: CLxWellplate

class CustomWellplate(TypedDict):
    """Custom wellplate definition wrapper (inside CustomDefinitions)."""

    CLxWellplate: CLxWellplate

class CLxWellplate(TypedDict, total=False):
    """Wellplate configuration.

    Defines the geometry and naming of a wellplate.
    """

    Uuid: Required[str]
    """Unique identifier for this wellplate definition."""
    CompanyName: Required[str]
    """Company/manufacturer name."""
    TypeName: Required[str]
    """Type name (may be empty)."""
    ModelName: Required[str]
    """Model name of the wellplate."""
    NamingMode: Required[int]
    """Naming mode for wells."""
    XNamingType: Required[int]
    """Column naming type (0=numeric, 1=alphabetic)."""
    YNamingType: Required[int]
    """Row naming type (0=numeric, 1=alphabetic)."""
    ColsAscending: bool
    """Whether columns are numbered ascending."""
    RowsAscending: bool
    """Whether rows are numbered ascending."""
    CLxLayoutElement: CLxLayoutElement
    """Layout element defining well active area."""
    SLxShape: SLxShape
    """Shape defining overall plate bounds."""
    XWellOffset: Required[float]
    """X offset to first well center in micrometers."""
    YWellOffset: Required[float]
    """Y offset to first well center in micrometers."""
    XCount: Required[int]
    """Number of columns."""
    YCount: Required[int]
    """Number of rows."""
    XDistance: Required[float]
    """Distance between well centers in X (micrometers)."""
    YDistance: Required[float]
    """Distance between well centers in Y (micrometers)."""

class CLxLayoutElement(TypedDict):
    """Layout element for well configuration."""

    ActiveArea: ActiveArea

class ActiveArea(TypedDict):
    """Active area within a well."""

    SLxShape: SLxShape

class SLxShape(TypedDict, total=False):
    """Shape definition.

    Can represent either a circle (Type=1) or rectangle (Type=2).
    """

    Type: Required[int]
    """Shape type: 1=circle, 2=rectangle."""
    # Circle parameters
    Radius: float
    """Radius in micrometers (for circular shapes)."""
    CenterX: float
    """Center X coordinate (for circular shapes)."""
    CenterY: float
    """Center Y coordinate (for circular shapes)."""
    # Rectangle parameters
    RectL: float
    """Left edge in micrometers (for rectangular shapes)."""
    RectT: float
    """Top edge in micrometers (for rectangular shapes)."""
    RectR: float
    """Right edge in micrometers (for rectangular shapes)."""
    RectB: float
    """Bottom edge in micrometers (for rectangular shapes)."""

PropertyDefinitions = dict[str, "PropertyDefinitionItem"]
"""Dictionary of property definitions, keyed by index (i0000000000, etc.)."""

class PropertyDefinitionItem(TypedDict, total=False):
    """A single property definition.

    Defines a user-configurable parameter for the job.
    """

    Key: Required[str]
    """Property name/key."""
    Value: Required[float]
    """Current value."""
    Type: Required[int]
    """Property type."""
    Visibility: Required[int]
    """Visibility flag."""
    Value1Desc: str
    """Description for value 1 (e.g., 'Yes')."""
    Value2Desc: str
    """Description for value 2 (e.g., 'No')."""
    Value3Desc: str
    """Description for value 3."""
    Value4Desc: str
    """Description for value 4."""
    Flags: int
    """Additional flags."""
    DependencyExpression: dict[str, Any]
    """Expression for conditional visibility."""

# all the TaskNames we have observed in the wild... but not necessarily all that exist.
TaskName = Literal[
    "CaptureLambdaDefinition",
    "CaptureLambdaAssayLI",
    "LoopPoint",
    "PointSet",
    "WellplateSelection",
    "WellplateDefinition",
    "LoopWells",
    "Condition",
    "ZStackDefinition",
    "LoopZStack",
    "LoopRegions",
    "Question",
    "OCSelect",
    "LoopTime",
    "LargeImageScan",
    "Comment",
    "AutoFocus",
    "MoveToWellCenter",
    "LargeImageScanRegion",
    "LargeImageScanHolder",
    "Expression",
    "EmptyPointSet",
    "DrawRegions",
    "AutoFocusPerform",
    "Variables",
    "EveryNth",
    "AssayWrapperLimCellCountVer1",
    "AddPoint",
    "ManualPointSet",
    "Live",
    "AlignWellplate",
    "Regions",
    "RedefineZ",
    "PFSON",
    "MoveToRegionCenter",
    "BypassDatabase",
    "AssayWrapperLimCellCountVer2",
    "StageArea",
    "SlideDefinitionNew",
    "Repeating",
    "Macro",
    "WellplateLabeling",
    "Wait",
    "STORAGE",
    "StageAreaNew",
    "SetClass",
    "RegisterRegion",
    "PFSOFF",
    "LoopSlides",
    "KeepObjectInview",
    "Ga3",
    "FocusSurface",
    "DefineFocusSurface",
    "ConditionIfElse",
    "Break",
    "AssignExposure",
    "AlignSlide",
    "WellplateListManual",
    "UsePFSSurface",
    "TTLOut",
    "STORMAcqClass",
    "StimROIFromAnalysis",
    "SmsNotification",
    "SlideLoaderNew",
    "SlideLoader",
    "SetLabel",
    "Replenish Water",
    "RedefinePfs",
    "OffsetFocusSurface",
    "NdStimulationSim",
    "MoveToZ",
    "MoveToPoint",
    "LoopWellplates",
    "LoopTimeSequence",
    "LoadCaptureDefinition",
    "LabelDefinition",
    "IlluminationSequence",
    "FastTimelapse",
    "DishDefinition",
    "DefinePFSSurface",
    "CaptureNSim",
    "BypassDatabaseEx",
    "AssignZStack",
    "AssignCaptureDefinition",
    "AssayWrapperLimOutProcVer1",
    "AppendRegions",
]

TasksDict = dict[TaskName | str, "Task"]
"""Dictionary mapping user-provided task identifiers to task definitions.

The keys are arbitrary user-provided identifiers (like variable names in a workflow).
The values are Task objects with a common structure defined in nd2.jobs._tasks.
"""
