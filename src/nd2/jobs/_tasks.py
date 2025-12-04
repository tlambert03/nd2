"""Task type definitions for JOBS workflow tasks.

The JOBS task structure is generic - tasks share a common structure with
Data/Parameters fields containing nested dictionaries whose structure varies
based on the specific task configuration. The Name field is a user-provided
label, not a structural type indicator.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class SlotConnection(TypedDict, total=False):
    """Connection from one task's output to another's input."""

    SlotType: int
    """Type identifier for the slot."""
    ParameterName: str
    """Name of the parameter being connected (e.g., 'Plate.Wellplate')."""
    ParameterType: int
    """Type of the parameter."""


class SlotConnectionsWrapper(TypedDict, total=False):
    """Wrapper for slot connections."""

    SlotConnections: dict[str, SlotConnection]
    """Dictionary mapping slot names (Slot0, Slot1, etc.) to connections."""


class LockableParamsWrapper(TypedDict, total=False):
    """Wrapper for lockable parameters."""

    LockableParams: dict[str, bool]
    """Dictionary mapping parameter names to locked state."""


class ParametersWrapper(TypedDict, total=False):
    """Wrapper for task parameters."""

    Parameters: dict[str, Any]
    """Task-specific parameters structure."""


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
    Data: dict[str, Any] | list[Any]
    """Task-specific data structure (varies by task configuration)."""
    Parameters: ParametersWrapper
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
