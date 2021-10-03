import io
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from ._sdk import FileHandle, SDKModule

NEW_HEADER_MAGIC_NUM = 0x0ABECEDA
OLD_HEADER_MAGIC_NUM = 0x0C000000
VERSION = re.compile(r"^ND2 FILE SIGNATURE CHUNK NAME01!Ver([\d\.]+)$")


def open_nd2(path: str) -> Tuple[io.BufferedReader, "FileHandle", "SDKModule"]:
    fh = open(path, "rb")
    magic_num = fh.read(4)
    try:
        if magic_num == b"\xda\xce\xbe\n":
            from ._sdk import latest

            lim_fh = latest.open(path)
            return fh, lim_fh, latest  # type: ignore
        elif magic_num == b"\x00\x00\x00\x0c":
            from ._sdk import v9

            lim_fh = v9.open(path)
            return fh, lim_fh, v9  # type: ignore
    except Exception:
        fh.close()
    raise OSError(f"file {path} not recognized as ND2.  First 4 bytes: {magic_num!r}")


def is_new_format(path: str) -> bool:
    # TODO: this is just for dealing with missing test data
    try:
        return magic_num(path) == NEW_HEADER_MAGIC_NUM
    except Exception:
        return False


def is_old_format(path: Union[str, Path]) -> bool:
    return magic_num(path) == OLD_HEADER_MAGIC_NUM


def magic_num(path: Union[str, Path]) -> int:
    with open(path, "rb") as fh:
        return int.from_bytes(fh.read(4), "little")


def jdn_to_datetime_local(jdn):
    return datetime.fromtimestamp((jdn - 2440587.5) * 86400.0)


def jdn_to_datetime_utc(jdn):
    return datetime.utcfromtimestamp((jdn - 2440587.5) * 86400.0)


def rgb_int_to_tuple(rgb):
    return ((rgb & 255), (rgb >> 8 & 255), (rgb >> 16 & 255))
