import re
from datetime import datetime

NEW_HEADER_MAGIC_NUM = 180276954
OLD_HEADER_MAGIC_NUM = 201326592
VERSION = re.compile(r"^ND2 FILE SIGNATURE CHUNK NAME01!Ver([\d\.]+)$")


def is_new_format(path: str) -> bool:
    # TODO: this is just for dealing with missing test data
    try:
        return magic_num(path) == NEW_HEADER_MAGIC_NUM
    except Exception:
        return False


def is_old_format(path: str) -> bool:
    return magic_num(path) == OLD_HEADER_MAGIC_NUM


def magic_num(path: str) -> int:
    with open(path, "rb") as fh:
        return int.from_bytes(fh.read(4), "little")


def file_chunk_version(file: str) -> str:
    try:
        with open(file, "rb") as fh:
            fh.seek(16)
            match = VERSION.search(fh.read(38).decode("utf8"))
            if match:
                return match.groups()[0]
    except Exception:
        pass
    return ""


def jdn_to_datetime_local(jdn):
    return datetime.fromtimestamp((jdn - 2440587.5) * 86400.0)


def jdn_to_datetime_utc(jdn):
    return datetime.utcfromtimestamp((jdn - 2440587.5) * 86400.0)


def rgb_int_to_tuple(rgb):
    return ((rgb & 255), (rgb >> 8 & 255), (rgb >> 16 & 255))
