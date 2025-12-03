"""TypedDict definitions for JOBS metadata schema.

This module provides type definitions for the JOBS workflow metadata found in ND2 files.
JOBS is Nikon's workflow automation system for microscopy acquisition.

The schema was inferred from analyzing 42 .bin job definition files and
job metadata embedded in ND2 files.
"""

from __future__ import annotations

from typing import Any, Union

from typing_extensions import Required, TypedDict

# ============================================================
# Top-level structures
# ============================================================


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


class ProgramDesc(TypedDict, total=False):
    """Program description metadata."""

    JobDefType: int


class ProtectedJob(TypedDict):
    """Encryption information for protected jobs."""

    EncryptedJobDefinition: list[int]
    """Encrypted job data as byte array."""
    HashCheck: list[int]
    """Hash for verification."""


# ============================================================
# Job structure
# ============================================================


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


# ============================================================
# Custom Definitions
# ============================================================


class CustomDefinitions(TypedDict, total=False):
    """Custom definitions container.

    Contains custom configurations like wellplate definitions.
    """

    CustomWellplate: CustomWellplate


class CustomWellplate(TypedDict):
    """Custom wellplate definition wrapper."""

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


# ============================================================
# Property Definitions
# ============================================================

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
    Flags: int
    """Additional flags."""
    DependencyExpression: dict[str, Any]
    """Expression for conditional visibility."""


# ============================================================
# Tasks
# ============================================================

TasksDict = dict[str, "BaseTask"]
"""Dictionary mapping task names to task definitions.

Task names include: WellplateDefinition, WellplateSelection, LoopWells,
CaptureLambdaDefinition, LoopPoint, LoopTime, AutoFocus, etc.
"""

TaskData = Union[dict[str, Any], list[Any], list[int]]
"""Task-specific data. Structure varies by task type."""


class BaseTask(TypedDict):
    """Base task structure common to all task types.

    All 89+ task types share these common fields. The `Data` and
    `Parameters` fields contain task-specific information.
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
    """Display name for the task."""
    Data: TaskData
    """Task-specific data (varies by task type)."""
    Parameters: dict[str, Any]
    """Task-specific parameters."""
    SlotConnections: SlotConnectionsWrapper | list[Any]
    """Connections to other tasks' outputs."""
    Notes: str
    """User notes for this task."""
    NotesDetailed: str
    """Detailed description of what this task does."""
    StateFlags: int
    """State flags for the task."""
    LockableParams: LockableParamsWrapper
    """Parameters that can be locked/unlocked."""


class SlotConnectionsWrapper(TypedDict):
    """Wrapper for slot connections."""

    SlotConnections: dict[str, SlotConnection]
    """Dictionary mapping slot names (Slot0, Slot1, etc.) to connections."""


class SlotConnection(TypedDict):
    """Connection from one task's output to another's input."""

    SlotType: int
    """Type identifier for the slot."""
    ParameterName: str
    """Name of the parameter being connected (e.g., 'Plate.Wellplate')."""
    ParameterType: int
    """Type of the parameter."""


class LockableParamsWrapper(TypedDict):
    """Wrapper for lockable parameters."""

    LockableParams: dict[str, bool]
    """Dictionary mapping parameter names to locked state."""


# ============================================================
# Well Selection (specific task data)
# ============================================================


class WellSelectionData(TypedDict):
    """Data for WellplateSelection task."""

    WellSelectionSettings: WellSelectionSettings


class WellSelectionSettings(TypedDict):
    """Settings for well selection."""

    Color: int
    """Color code for selection display."""
    OrderingType: int
    """Ordering type for well iteration."""
    SelectionMask: SelectionMask
    """Bitmask defining selected wells."""


class SelectionMask(TypedDict):
    """Bitmask for well selection.

    The `Ma` field is a byte array where each bit represents a well.
    Wells are numbered row-by-row: well 0 = A1, well 1 = A2, etc.
    """

    Si: int
    """Total number of wells (e.g., 96 for 96-well plate)."""
    St: int
    """Start index."""
    Co: int
    """Column count."""
    Sp: int
    """Spacing."""
    Ma: list[int]
    """Bitmask as byte array. Each bit indicates if well is selected."""


# ============================================================
# Wellplate Definition (specific task data)
# ============================================================


class WellplateDefinitionData(TypedDict):
    """Data for WellplateDefinition task."""

    WellplateUUID: WellplateUUID


class WellplateUUID(TypedDict):
    """UUID reference to a wellplate definition."""

    UUID: str
    """UUID of the wellplate definition."""


# ============================================================
# Capture Definition (specific task data)
# ============================================================


class CaptureLambdaData(TypedDict, total=False):
    """Data for CaptureLambdaDefinition task."""

    CaptureLambda: CaptureLambda


class CaptureLambda(TypedDict, total=False):
    """Capture/acquisition settings."""

    UUID: str
    """Unique identifier."""
    Channels: dict[str, ChannelConfig]
    """Channel configurations."""


class ChannelConfig(TypedDict, total=False):
    """Configuration for a single channel."""

    Name: str
    """Channel name."""
    ExposureMs: float
    """Exposure time in milliseconds."""
    OpticalConfigName: str
    """Name of the optical configuration."""
