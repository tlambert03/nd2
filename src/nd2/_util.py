from __future__ import annotations

import math
import re
from datetime import datetime
from itertools import product
from typing import TYPE_CHECKING, Mapping, NamedTuple

if TYPE_CHECKING:
    from os import PathLike
    from typing import IO, Any, Callable, ClassVar, Sequence, Union

    from ._legacy import LegacyND2Reader
    from ._pysdk._pysdk import ND2Reader

    StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

    ListOfDicts = list[dict[str, Any]]
    DictOfLists = Mapping[str, Sequence[Any]]
    DictOfDicts = Mapping[str, dict[int, Any]]

NEW_HEADER_MAGIC = b"\xda\xce\xbe\n"
OLD_HEADER_MAGIC = b"\x00\x00\x00\x0c"
VERSION = re.compile(r"^ND2 FILE SIGNATURE CHUNK NAME01!Ver([\d\.]+)$")


def is_supported_file(
    path: StrOrBytesPath, open_: Callable[[StrOrBytesPath, str], IO[Any]] = open
) -> bool:
    """Return `True` if `path` can be opened as an nd2 file.

    Parameters
    ----------
    path : Union[str, bytes, PathLike]
        A path to query
    open_ : Callable[[StrOrBytesPath, str], IO[Any]]
        Filesystem opener, by default `builtins.open`

    Returns
    -------
    bool
        Whether the can be opened.
    """
    with open_(path, "rb") as fh:
        return fh.read(4) in (NEW_HEADER_MAGIC, OLD_HEADER_MAGIC)


def is_legacy(path: StrOrBytesPath) -> bool:
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


def get_reader(
    path: str,
    validate_frames: bool = False,
    search_window: int = 100,
) -> ND2Reader | LegacyND2Reader:
    with open(path, "rb") as fh:
        magic_num = fh.read(4)
        if magic_num == NEW_HEADER_MAGIC:
            from ._pysdk._pysdk import ND2Reader

            return ND2Reader(
                path,
                validate_frames=validate_frames,
                search_window=search_window,
            )
        elif magic_num == OLD_HEADER_MAGIC:
            from ._legacy._legacy import LegacyND2Reader

            return LegacyND2Reader(path)
        raise OSError(
            f"file {path} not recognized as ND2.  First 4 bytes: {magic_num!r}"
        )


def is_new_format(path: str) -> bool:
    # TODO: this is just for dealing with missing test data
    with open(path, "rb") as fh:
        return fh.read(4) == NEW_HEADER_MAGIC


def jdn_to_datetime_local(jdn: float) -> datetime:
    return datetime.fromtimestamp((jdn - 2440587.5) * 86400.0)


def jdn_to_datetime_utc(jdn: float) -> datetime:
    return datetime.utcfromtimestamp((jdn - 2440587.5) * 86400.0)


def rgb_int_to_tuple(rgb: int) -> tuple[int, int, int]:
    return ((rgb & 255), (rgb >> 8 & 255), (rgb >> 16 & 255))


# these are used has headers in the events() table
TIME_KEY = "Time [s]"
Z_SERIES_KEY = "Z-Series"


class AXIS:
    X = "X"
    Y = "Y"
    Z = "Z"
    CHANNEL = "C"
    RGB = "S"
    TIME = "T"
    POSITION = "P"
    UNKNOWN = "U"

    _MAP: ClassVar[dict[str, str]] = {
        "Unknown": UNKNOWN,
        "TimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "NETimeLoop": TIME,
        "CustomLoop": UNKNOWN,
    }


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


def parse_time(time_str: str) -> datetime:
    for fmt_str in TIME_FMT_STRINGS:
        try:
            return datetime.strptime(time_str, fmt_str)
        except ValueError:
            continue
    raise ValueError(f"Could not parse {time_str}")


# utils for converting records to dicts, in recorded_data method


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
    >>> lists = {'a': [1, 4, float('nan')], 'b': [float('nan'), 5, 8], 'c': [3, 6, 9]}
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
