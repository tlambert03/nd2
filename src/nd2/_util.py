from __future__ import annotations

import re
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Callable, NamedTuple, Union

if TYPE_CHECKING:
    from os import PathLike

    from ._legacy import LegacyND2Reader
    from ._pysdk._pysdk import ND2Reader

    StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes]]


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


class AXIS:
    X = "X"
    Y = "Y"
    Z = "Z"
    CHANNEL = "C"
    RGB = "S"
    TIME = "T"
    POSITION = "P"
    UNKNOWN = "U"

    _MAP = {
        "Unknown": UNKNOWN,
        "TimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "NETimeLoop": TIME,
    }


class VoxelSize(NamedTuple):
    x: float
    y: float
    z: float
