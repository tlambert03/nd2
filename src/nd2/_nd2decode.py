import io
import re
import struct
from functools import partial
from typing import Any, Callable, Dict, List, cast

from .structures import LoopType

__all__ = ["decode_metadata", "unnest_experiments"]

lower = re.compile("^[_a-z]*")


def decode_metadata(data: bytes, strip_prefix=True, _count: int = 1) -> Dict[str, Any]:
    """Decode the `ImageMetadataLV` metadata block from an ND2 file.

    Parameters
    ----------
    data : bytes
        The bytes of the metadata.
    strip_prefix : bool, optional
        Whether to strip the prefix from the metadata keys, by default True
    _count : int, optional
        The number of metadata entries to decode, by default 1

    Returns
    -------
    Dict[str, Any]
        A dictionary of the metadata.
    """
    output: Dict[str, Any] = {}
    if not data:
        return output

    stream = io.BytesIO(data)
    for _ in range(_count):

        curs = stream.tell()
        header = stream.read(2)
        if not header:
            break

        data_type, name_length = strctBB.unpack(header)
        name = stream.read(name_length * 2).decode("utf16")[:-1]
        if strip_prefix:
            name = lower.sub("", name)
        if data_type == 11:
            new_count, length = strctIQ.unpack(stream.read(strctIQ.size))
            next_data_length = stream.read(length - (stream.tell() - curs))
            value = decode_metadata(next_data_length, strip_prefix, new_count)
            stream.seek(new_count * 8, 1)
        elif data_type in _PARSER:
            value = _PARSER[data_type](stream)
        else:
            value = None

        if isinstance(value, dict):
            t = "Type" if strip_prefix else "eType"
            a = "ApplicationDesc" if strip_prefix else "wsApplicationDesc"
            if t in value and a in value:
                value[t] = LoopType(value[t])

        if name in output:
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast(list, output[name]).append(value)
        else:
            output[name] = value

    return output


def unnest_experiments(metadata: dict) -> List[Dict[str, Any]]:
    """Unnest the experiments from the metadata.

    Parameters
    ----------
    metadata : dict
        The metadata to unnest.

    Returns
    -------
    List[Dict[str, Any]]
        A list of the experiments.
    """
    if "SLxExperiment" in metadata:
        metadata = metadata["SLxExperiment"]

    loops = []
    next_level = metadata.copy()
    while True:
        if isinstance(next_level, dict) and len(next_level) == 1 and "" in next_level:
            next_level = next_level[""]
        if isinstance(next_level, list):
            loops.append([unnest_experiments(i)[0] for i in next_level])
            break
        next_level.pop("NextLevelCount", None)
        loops.append(next_level)
        next_level = next_level.pop("NextLevelEx", None)
        if next_level is None:
            break
    return loops


def _unpack_one(strct: struct.Struct, data: io.BytesIO):
    return strct.unpack(data.read(strct.size))[0]


strctBB = struct.Struct("BB")
strctIQ = struct.Struct("<IQ")
strctB = struct.Struct("B")
unpack_B = partial(_unpack_one, strctB)
unpack_I = partial(_unpack_one, struct.Struct("I"))
unpack_Q = partial(_unpack_one, struct.Struct("Q"))
unpack_d = partial(_unpack_one, struct.Struct("d"))


def _unpack_list(data: io.BytesIO):
    return [i[0] for i in strctB.iter_unpack(data.read(unpack_Q(data)))]


def _unpack_string(data: io.BytesIO):
    value = data.read(2)
    # the string ends at the first instance of \x00\x00
    while not value.endswith(b"\x00\x00"):
        next_data = data.read(2)
        if len(next_data) == 0:
            break
        value += next_data

    try:
        return value.decode("utf16")[:-1]
    except UnicodeDecodeError:
        return value.decode("utf8")


_PARSER: Dict[int, Callable] = {
    1: unpack_B,
    2: unpack_I,
    3: unpack_I,
    5: unpack_Q,
    6: unpack_d,
    8: _unpack_string,
    9: _unpack_list,
}
