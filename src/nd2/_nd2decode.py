import io
import struct
from functools import partial
from typing import Any, Callable, Dict, cast


def decode_metadata(data: bytes, count: int = 1) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if not data:
        return output

    stream = io.BytesIO(data)
    for _ in range(count):

        curs = stream.tell()
        header = stream.read(2)
        if not header:
            break

        data_type, name_length = strctBB.unpack(header)
        name = stream.read(name_length * 2).decode("utf16")[:-1]

        if data_type == 11:
            value = _parse_metadata_item(stream, curs)
        elif data_type in _PARSER:
            value = _PARSER[data_type](stream)
        else:
            value = None

        if name in output:
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast(list, output[name]).append(value)
        else:
            output[name] = value

    return output


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


def _parse_metadata_item(data: io.BytesIO, curs: int):
    new_count, length = strctIQ.unpack(data.read(strctIQ.size))
    next_data_length = data.read(length - (data.tell() - curs))
    value = decode_metadata(next_data_length, new_count)
    data.read(new_count * 8)
    return value


_PARSER: Dict[int, Callable] = {
    1: unpack_B,
    2: unpack_I,
    3: unpack_I,
    5: unpack_Q,
    6: unpack_d,
    8: _unpack_string,
    9: _unpack_list,
    11: _parse_metadata_item,
}
