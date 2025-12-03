from __future__ import annotations

import io
import re
import struct
import zlib
from typing import TYPE_CHECKING, Any, Callable, Union, cast

if TYPE_CHECKING:
    from typing import Final

    JsonValueType = Union[dict[str, "JsonValueType"], int, str, float, None, bool, list]


strctBB = struct.Struct("BB")  # 2x uint8_t
strctIQ = struct.Struct("<IQ")  # 1x uint32_t, 1x uint64_t
strctB = struct.Struct("B")  # uint8_t
strctI = struct.Struct("I")  # uint32_t
strcti = struct.Struct("i")  # int32_t
strctq = struct.Struct("q")  # int64_t
strctQ = struct.Struct("Q")  # uint64_t
strctd = struct.Struct("d")  # float64_t
strctf = struct.Struct("f")  # float32_t
lower = re.compile("^[_a-z]*")


def _unpack_bool(stream: io.BytesIO) -> bool:
    data = stream.read(strctB.size)
    return bool(strctB.unpack(data)[0])
    # strangely enough, sometimes this value is something other than 0 or 1
    # `dims_p1z5t3c2y32x32.nd2` for example has a value of 116
    # this results in a case where readlimfile dumps a boolean of False
    # but LIMFILE_EXPORT json experiment (in JsonBridge.cpp) dumps a boolean of True
    # return strctB.unpack(data)[0] == 1


def _unpack_int32(stream: io.BytesIO) -> int:
    return int(strcti.unpack(stream.read(strcti.size))[0])


def _unpack_uint32(stream: io.BytesIO) -> int:
    return int(strctI.unpack(stream.read(strctI.size))[0])


def _unpack_int64(stream: io.BytesIO) -> int:
    return int(strctq.unpack(stream.read(strctq.size))[0])


def _unpack_uint64(stream: io.BytesIO) -> int:
    return int(strctQ.unpack(stream.read(strctQ.size))[0])


def _unpack_double(stream: io.BytesIO) -> float:
    return float(strctd.unpack(stream.read(strctd.size))[0])


def _unpack_void_pointer(stream: io.BytesIO) -> int:
    # TODO: i think nd2 will actually return a encodeBase64 string
    return strctQ.unpack(stream.read(strctQ.size))[0]  # type: ignore


def _unpack_string(data: io.BytesIO) -> str:
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


class ELxLiteVariantType:
    UNKNOWN: Final = 0
    BOOL: Final = 1
    INT32: Final = 2
    UINT32: Final = 3
    INT64: Final = 4
    UINT64: Final = 5
    DOUBLE: Final = 6
    VOIDPOINTER: Final = 7
    STRING: Final = 8
    BYTEARRAY: Final = 9
    DEPRECATED: Final = 10
    LEVEL: Final = 11
    COMPRESS: Final = 76  # 'L'


_PARSERS: dict[int, Callable[[io.BytesIO], Any]] = {
    ELxLiteVariantType.BOOL: _unpack_bool,  # 1
    ELxLiteVariantType.INT32: _unpack_int32,  # 2
    ELxLiteVariantType.UINT32: _unpack_uint32,  # 3
    ELxLiteVariantType.INT64: _unpack_int64,  # 4
    ELxLiteVariantType.UINT64: _unpack_uint64,  # 5
    ELxLiteVariantType.DOUBLE: _unpack_double,  # 6
    ELxLiteVariantType.VOIDPOINTER: _unpack_void_pointer,  # 7
    ELxLiteVariantType.STRING: _unpack_string,  # 8
    # BYTEARRAY (9) is handled specially in json_from_clx_lite_variant
    # to allow recursive decoding of nested CLX Lite data
}

DTYPE_SIZES = {
    ELxLiteVariantType.BOOL: 1,
    ELxLiteVariantType.INT32: 4,
    ELxLiteVariantType.UINT32: 4,
    ELxLiteVariantType.INT64: 8,
    ELxLiteVariantType.UINT64: 8,
    ELxLiteVariantType.DOUBLE: 8,
    ELxLiteVariantType.VOIDPOINTER: 8,
    ELxLiteVariantType.STRING: 2,
    ELxLiteVariantType.BYTEARRAY: 8,
    ELxLiteVariantType.DEPRECATED: 0,
    ELxLiteVariantType.LEVEL: 12,  # item_count (4) + length (8)
}


def _looks_like_clx_lite(data: bytes) -> bool:
    """Check if data looks like valid CLX Lite encoded data.

    CLX Lite format starts with:
    - byte 0: data type (1-11 for normal types, 76 for compressed)
    - byte 1: name length (in UTF-16 chars, so name is name_length * 2 bytes)
    - bytes 2 to 2+name_length*2: UTF-16 encoded name (null-terminated)

    We detect invalid data by checking size requirements and UTF-16 patterns.
    """
    if not data or len(data) < 2:
        return False

    data_type = data[0]
    name_length = data[1]

    # Valid data types: 1-11 (normal) or 76 (compressed 'L')
    if data_type == 76:  # COMPRESS
        return True
    if not (1 <= data_type <= 11):
        return False

    # Calculate minimum size based on type
    # header (2) + name (name_length * 2) + value data
    name_bytes = name_length * 2
    header_and_name = 2 + name_bytes

    # Each type has minimum value size requirements
    value_size = DTYPE_SIZES.get(data_type, 0)
    min_size = header_and_name + value_size
    if len(data) < min_size:
        return False

    # For non-empty names, verify UTF-16 structure: names should end with null char
    # UTF-16 null terminator is \x00\x00, and it's included in name_length
    if name_length > 0:
        name_end = 2 + name_bytes
        # Last 2 bytes of name should be null terminator
        if data[name_end - 2 : name_end] != b"\x00\x00":
            return False

    return True


def _chunk_name_and_dtype(
    stream: io.BytesIO, strip_prefix: bool = True
) -> tuple[str, int]:
    header = stream.read(strctBB.size)
    if not header:
        return ("", -1)

    data_type, name_length = strctBB.unpack(header)
    if data_type in (ELxLiteVariantType.DEPRECATED, ELxLiteVariantType.UNKNOWN):
        raise ValueError(  # pragma: no cover
            f"Unknown data type in metadata header: {data_type}"
        )
    elif data_type == ELxLiteVariantType.COMPRESS:
        name = ""
    else:
        # name of the section is a utf16 string of length `name_length * 2`
        name = stream.read(name_length * 2).decode("utf16")[:-1]
        if strip_prefix:
            name = lower.sub("", name)
    return (name, data_type)


# lite variant
def json_from_clx_lite_variant(
    data: bytes | io.BytesIO, strip_prefix: bool = True, _count: int = 1
) -> dict[str, JsonValueType]:
    output: dict[str, JsonValueType] = {}
    if not data:
        return output

    stream = data if isinstance(data, io.BytesIO) else io.BytesIO(data)

    for _ in range(_count):
        curs = stream.tell()

        name, data_type = _chunk_name_and_dtype(stream, strip_prefix)

        if data_type == ELxLiteVariantType.COMPRESS:
            stream.seek(10, 1)
            deflated = zlib.decompress(stream.read())
            return json_from_clx_lite_variant(deflated, strip_prefix)

        if data_type == -1:
            # never seen this, but it's in the sdk
            break  # pragma: no cover

        value: JsonValueType
        if data_type == ELxLiteVariantType.LEVEL:
            item_count, length = strctIQ.unpack(stream.read(strctIQ.size))
            next_data_length = stream.read(length - (stream.tell() - curs))
            val: dict = json_from_clx_lite_variant(
                next_data_length, strip_prefix, item_count
            )
            stream.seek(item_count * 8, 1)
            # levels with a single "" key are actually lists
            if len(val) == 1 and "" in val:
                value = val[""]
                if not isinstance(value, list):
                    value = [value]
                value = {f"i{n:010}": x for n, x in enumerate(value)}
            else:
                value = val

        elif data_type == ELxLiteVariantType.BYTEARRAY:
            # Read size, then check if it looks like nested CLX Lite data
            size = _unpack_uint64(stream)
            raw_bytes = stream.read(size)
            if _looks_like_clx_lite(raw_bytes):
                try:
                    decoded = json_from_clx_lite_variant(raw_bytes, strip_prefix)
                    value = decoded if decoded else list(raw_bytes)
                except Exception:
                    value = list(raw_bytes)
            else:
                value = list(raw_bytes)

        elif data_type in _PARSERS:
            value = _PARSERS[data_type](stream)
        else:
            # also never seen this
            value = None  # pragma: no cover
        if name == "" and name in output:
            # nd2 uses empty strings as keys for lists
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast("list", output[name]).append(value)
        else:
            output[name] = value

    return output
