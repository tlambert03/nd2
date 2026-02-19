from __future__ import annotations

import math
import re
from contextlib import suppress
from datetime import datetime, timezone
from itertools import product
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike
    from typing import Any, ClassVar, Final, Protocol, TypeAlias, Union

    from typing_extensions import TypeGuard

    from nd2.structures import ExpLoop

    class ReadSeekBinary(Protocol):
        @property
        def closed(self) -> bool: ...
        def read(self, size: int | None = -1, /) -> bytes: ...
        def seek(self, offset: int, whence: int = 0, /) -> int: ...
        def close(self) -> None: ...

    StrOrPath: TypeAlias = Union[str, PathLike]
    FileOrBinaryIO: TypeAlias = Union[StrOrPath, ReadSeekBinary]

    ListOfDicts: TypeAlias = list[dict[str, Any]]
    DictOfLists: TypeAlias = Mapping[str, Sequence[Any]]
    DictOfDicts: TypeAlias = Mapping[str, dict[int, Any]]

NEW_HEADER_MAGIC = b"\xda\xce\xbe\n"
OLD_HEADER_MAGIC = b"\x00\x00\x00\x0c"
VERSION = re.compile(r"^ND2 FILE SIGNATURE CHUNK NAME01!Ver([\d\.]+)$")


def is_fsspec_url(path: Any) -> TypeGuard[str]:
    """True if `path` is a string with a remote URL scheme (e.g. 's3://')."""
    if not isinstance(path, str):
        return False
    idx = path.find("://")
    # idx > 1 excludes Windows drive letters like C:/
    return idx > 1


def open_fsspec_url(url: str, storage_options: dict | None = None) -> ReadSeekBinary:
    """Open a remote nd2 URL with a pre-warmed metadata cache.

    For smaller files, pre-fetches the full object and opens with
    cache_type='parts'. For larger files, opens directly via filesystem
    defaults to avoid non-contiguous cache composition artifacts.
    """
    try:
        import fsspec
    except ImportError as e:
        raise ImportError(
            "fsspec is required to open remote URLs. Install with: pip install fsspec"
        ) from e

    sopts = storage_options or {}
    fs, fpath = fsspec.url_to_fs(url, **sopts)
    size: int = fs.info(fpath)["size"]

    START_BYTES = 512 * 1024  # 512 KB — covers magic + ImageMetadataSeqLV|0!
    END_BYTES = 5 * 1024 * 1024  # 5 MB — covers chunkmap + all end metadata

    if size <= START_BYTES + END_BYTES:
        raw = fs.cat_file(fpath)
        known: dict[tuple[int, int], bytes] = {(0, size): raw}
        return cast(
            "ReadSeekBinary",
            fs.open(
                fpath,
                "rb",
                cache_type="parts",
                cache_options={"data": known, "strict": False},
            ),
        )

    return cast("ReadSeekBinary", fs.open(fpath, "rb"))


def is_read_seek_binary(obj: object) -> TypeGuard[ReadSeekBinary]:
    return (
        hasattr(obj, "read")
        and hasattr(obj, "seek")
        and hasattr(obj, "close")
        and hasattr(obj, "closed")
    )


def is_supported_file(path: FileOrBinaryIO) -> bool:
    """Return `True` if `path` can be opened as an nd2 file.

    Parameters
    ----------
    path : Union[str, bytes, PathLike]
        A path to query

    Returns
    -------
    bool
        Whether the can be opened.
    """
    if is_read_seek_binary(path):
        path.seek(0)
        magic = path.read(4)
    else:
        with open(cast("StrOrPath", path), "rb") as fh:
            magic = fh.read(4)
    return magic in (NEW_HEADER_MAGIC, OLD_HEADER_MAGIC)


def is_legacy(path: StrOrPath) -> bool:
    """Return `True` if `path` is a legacy ND2 file.

    Parameters
    ----------
    path : Union[str, bytes, PathLike]
        A path to query

    Returns
    -------
    bool
        Whether the file is a legacy ND2 file.
    """
    with open(path, "rb") as fh:
        return fh.read(4) == OLD_HEADER_MAGIC


def is_new_format(path: str) -> bool:
    # TODO: this is just for dealing with missing test data
    with open(path, "rb") as fh:
        return fh.read(4) == NEW_HEADER_MAGIC


JDN_UNIX_EPOCH = 2440587.5
SECONDS_PER_DAY = 86400


def jdn_to_datetime(jdn: float, tz: timezone = timezone.utc) -> datetime:
    seconds_since_epoch = (jdn - JDN_UNIX_EPOCH) * SECONDS_PER_DAY
    # very negative values can cause OverflowError on Windows, and are meaningless
    dt = datetime.fromtimestamp(max(seconds_since_epoch, 0), tz)
    with suppress(ValueError, OSError):
        # astimezone() without arguments will use the system's local timezone
        return dt.astimezone()
    return dt


# these are used has headers in the events() table
TIME_KEY = "Time [s]"
Z_SERIES_KEY = "Z-Series"
POSITION_NAME = "Position Name"


class AXIS:
    X: Final = "X"
    Y: Final = "Y"
    Z: Final = "Z"
    CHANNEL: Final = "C"
    RGB: Final = "S"
    TIME: Final = "T"
    POSITION: Final = "P"
    UNKNOWN: Final = "U"

    _MAP: ClassVar[dict[str, str]] = {
        "Unknown": UNKNOWN,
        "TimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "NETimeLoop": TIME,
        "CustomLoop": UNKNOWN,
    }

    @classmethod
    def frame_coords(cls) -> set[str]:
        return {cls.X, cls.Y, cls.CHANNEL, cls.RGB}


class VoxelSize(NamedTuple):
    x: float
    y: float
    z: float


TIME_FMT_STRINGS = [
    "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d-%b-%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
]


def convert_records_to_dict_of_lists(
    records: ListOfDicts, null_val: Any = float("nan")
) -> DictOfLists:
    """Convert a list of records (dicts) to a dict of lists.

    Examples
    --------
    >>> records = [
    ...     {"a": 1, "c": 3},
    ...     {"a": 4, "b": 5, "c": 6},
    ...     {"b": 8, "c": 9},
    ... ]
    >>> convert_records_to_dict(records)
    {'a': [1, 4, nan], 'b': [nan, 5, 8], 'c': [3, 6, 9]}
    """
    # get the column names in the order they appear in the records
    col_names: dict[str, None] = {column: None for r in records for column in r}
    output: Mapping[str, list[Any]] = {col_name: [] for col_name in col_names}

    for record, col_name in product(records, col_names):
        output[col_name].append(record.get(col_name, null_val))

    return output


def convert_records_to_dict_of_dicts(
    records: ListOfDicts, null_val: Any = float("nan")
) -> DictOfDicts:
    """Convert a list of records (dicts) to a dict of dicts.

    Examples
    --------
    >>> records = [
    ...     {"a": 1, "c": 3},
    ...     {"a": 4, "b": 5, "c": 6},
    ...     {"b": 8, "c": 9},
    ... ]
    >>> convert_records_to_dict_of_dicts(records)
    {'b': {0: nan, 1: 5, 2: 8}, 'a': {0: 1, 1: 4, 2: nan}, 'c': {0: 3, 1: 6, 2: 9}}
    """
    # get the column names in the order they appear in the records
    col_names: dict[str, None] = {column: None for r in records for column in r}
    output: DictOfDicts = {col_name: {} for col_name in col_names}

    for (idx, record), col_name in product(enumerate(records), col_names):
        output[col_name][idx] = record.get(col_name, null_val)

    return output


def convert_dict_of_lists_to_records(
    columns: DictOfLists, strip_nan: bool = False
) -> ListOfDicts:
    """Convert a dict of column lists to a list of records (dicts).

    Examples
    --------
    >>> lists = {"a": [1, 4, float("nan")], "b": [float("nan"), 5, 8], "c": [3, 6, 9]}
    >>> convert_dict_of_lists_to_records(records)
    [
        {"a": 1, "c": 3},
        {"a": 4, "b": 5, "c": 6},
        {"b": 8, "c": 9},
    ]
    """
    return [
        {
            col_name: value
            for col_name, value in zip(columns, row_data)
            if not strip_nan or not math.isnan(value)
        }
        for row_data in zip(*columns.values())
    ]


def loop_indices(experiment: list[ExpLoop]) -> tuple[dict[str, int], ...]:
    """Return a tuple of dicts of loop indices for each frame.

    Examples
    --------
    >>> with nd2.ND2File("path/to/file.nd2") as f:
    ...     f.loop_indices()
    (
        {'Z': 0, 'T': 0, 'C': 0},
        {'Z': 0, 'T': 0, 'C': 1},
        {'Z': 0, 'T': 0, 'C': 2},
        ...
    )
    """
    axes = [AXIS._MAP[x.type] for x in experiment]
    indices = product(*(range(x.count) for x in experiment))
    return tuple(dict(zip(axes, x)) for x in indices)
