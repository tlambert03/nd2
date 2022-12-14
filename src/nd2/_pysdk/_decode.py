from __future__ import annotations

import io
import re
import struct
from io import BufferedReader
from typing import TYPE_CHECKING, Any, Callable, Union, cast

if TYPE_CHECKING:
    from os import PathLike
    from typing import Final

    StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]
    StartFileChunk = tuple[int, int, int, bytes, bytes]
    ChunkMapItem = tuple[int, int]  # (offset, size)
    ChunkMap = dict[bytes, ChunkMapItem]
    JsonValueType = Union[dict[str, "JsonValueType"], int, str, float, None, bool, list]

# fmt: off
ND2_FILE_SIGNATURE:     Final = b"ND2 FILE SIGNATURE CHUNK NAME01!"  # len 32
ND2_FILEMAP_SIGNATURE:  Final = b"ND2 FILEMAP SIGNATURE NAME 0001!"
ND2_CHUNKMAP_SIGNATURE: Final = b"ND2 CHUNK MAP SIGNATURE 0000001!"
CHUNK_ALIGNMENT:        Final = 4096
CHUNK_NAME_RESERVE:     Final = 20
ND2_CHUNK_MAGIC:        Final = 0x0ABECEDA
JP2_MAGIC:              Final = 0x0C000000
# fmt: on

QQ = struct.Struct("QQ")
# 2 x uint64_t
# used for (offset, size) in chunkmap

CHUNK_HEADER = struct.Struct("IIQ")  # beginning of every chunk an ND2 file
# uint32_t magic
# uint32_t nameLen
# uint64_t dataLen

START_FILE_CHUNK = struct.Struct(f"{CHUNK_HEADER.format}32s64s")
# ChunkHeader header
# char name[32]
# char data[64]

# the last 40 bytes of the file, containing the signature and locatio of chunkmap
SIG_CHUNKMAP_LOC = struct.Struct("32sQ")
# char name[32]
# uint64_t offset


def get_version(fh: BufferedReader | StrOrBytesPath) -> tuple[int, int]:
    """Get the version of the ND2 file or raise an exception.

    Parameters
    ----------
    fh : BufferedReader | str | bytes | Path
        The file handle or path to the ND2 file.

    Returns
    -------
    tuple[int, int]
        (major, minor) version of the ND2 file

    Raises
    ------
    ValueError
        If the file is not a valid ND2 file or the header chunk is corrupt.
    """
    if not isinstance(fh, BufferedReader):
        with open(fh, "rb") as fh:
            chunk = START_FILE_CHUNK.unpack(fh.read(START_FILE_CHUNK.size))
    else:
        # leave it open if it came in open
        chunk = START_FILE_CHUNK.unpack(fh.read(START_FILE_CHUNK.size))

    magic, name_length, data_length, name, data = cast("StartFileChunk", chunk)

    # sanity checks
    if magic != ND2_CHUNK_MAGIC:
        if magic == JP2_MAGIC:
            raise NotImplementedError(f"Legacy ND2 file not yet supported: {fh.name}")
        raise ValueError(f"Not a valid ND2 file: {fh.name}. (magic: {magic!r})")
    if name_length != 32 or data_length != 64 or name != ND2_FILE_SIGNATURE:
        raise ValueError(f"Corrupt ND2 file header chunk: {fh.name}")

    # data will now be something like Ver2.0, Ver3.0, etc.
    return (int(chr(data[3])), int(chr(data[5])))


def load_chunkmap(fh: BufferedReader) -> ChunkMap:
    """Read the map of the chunks at the end of an ND2 file.

    chunk rules:
    - each data chunk starts with
      - 4 bytes: CHUNK_MAGIC -> 0x0ABECEDA (big endian: 0xDACEBE0A)
      - 4 bytes: length of the chunk header (this section contains the chunk name...)
      - 8 bytes: length of chunk following the header, up to the next CHUNK_MAGIC

    Parameters
    ----------
    fh : BufferedReader
        An open nd2 file.  File is assumed to be a valid ND2 file.  (use `get_version`)

    Returns
    -------
    ChunkMap
        A dictionary mapping chunk names to (offset, size) pairs.

    Raises
    ------
    ValueError
        If the file is not a valid ND2 file or the chunkmap is corrupt.
    """
    fh.seek(-40, 2)
    sig, chunkmap_location = SIG_CHUNKMAP_LOC.unpack(fh.read(SIG_CHUNKMAP_LOC.size))
    if sig != ND2_CHUNKMAP_SIGNATURE:
        raise ValueError(f"Invalid ChunkMap signature {sig!r} in file {fh.name!r}")

    # then we get all of the data in the chunkmap
    # this asserts that the chunkmap begins with CHUNK_MAGIC
    chunkmap_data = _read_nd2_chunk(fh, chunkmap_location)
    current_position = 0
    chunk_map: ChunkMap = {}
    while True:
        # find the first "!", starting at pos, then go to next byte
        p = chunkmap_data.index(b"!", current_position) + 1
        # read the chunk name
        chunk_name = chunkmap_data[current_position:p]
        if chunk_name == ND2_CHUNKMAP_SIGNATURE:
            # break when we find the end
            break

        # the next 16 bytes contain...
        # (8) -> offset of this key in the file  (@ the chunk magic)
        # (8) -> size of this chunk in the file (not including the chunk header)
        # Note: one still needs to go to `position` to read the CHUNK_INFO to know
        # the absolute position of the data (excluding the chunk header).
        # This can be done using `_read_chunk(fh, position)``
        offset, size = QQ.unpack(chunkmap_data[p : p + QQ.size])
        if size == offset:
            size = -1
        chunk_map[chunk_name] = (offset, size)
        current_position = p + QQ.size
    return chunk_map


def _read_nd2_chunk(fh: BufferedReader, start_position: int) -> bytes:
    """Read a single chunk in an ND2 file at `start_position`.

    Parameters
    ----------
    fh : BufferedReader
        An open nd2 file.  File is assumed to be a valid ND2 file.  (use `get_version`)
    start_position : int
        The position in the file to start reading the chunk.

    Returns
    -------
    bytes
        The data in the chunk.

    Raises
    ------
    ValueError
        If the file is not a valid ND2 file or the chunk is corrupt.
    """
    fh.seek(start_position)
    magic, name_length, data_length = CHUNK_HEADER.unpack(fh.read(CHUNK_HEADER.size))
    # confirm chunk magic, seek to shift, read for length
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(f"Invalid chunk header magic: {magic:x}")
    fh.seek(name_length, 1)
    return fh.read(data_length)


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
    return bool(strctB.unpack(stream.read(strctB.size))[0])


def _unpack_int32(stream: io.BytesIO) -> int:
    return strcti.unpack(stream.read(strcti.size))[0]


def _unpack_uint32(stream: io.BytesIO) -> int:
    return strctI.unpack(stream.read(strctI.size))[0]


def _unpack_int64(stream: io.BytesIO) -> int:
    return strctq.unpack(stream.read(strctq.size))[0]


def _unpack_uint64(stream: io.BytesIO) -> int:
    return strctQ.unpack(stream.read(strctQ.size))[0]


def _unpack_double(stream: io.BytesIO) -> float:
    return float(strctd.unpack(stream.read(strctd.size))[0])


def _unpack_void_pointer(stream: io.BytesIO) -> int:
    # TODO: i think nd2 will actually return a encodeBase64 string
    return strctQ.unpack(stream.read(strctQ.size))[0]


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


def _unpack_bytearray(data: io.BytesIO) -> list:
    return [i[0] for i in strctB.iter_unpack(data.read(_unpack_uint64(data)))]


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
    ELxLiteVariantType.BYTEARRAY: _unpack_bytearray,  # 9
}


def _chunk_name_and_dtype(
    stream: io.BytesIO, strip_prefix: bool = True
) -> tuple[str, int]:
    header = stream.read(strctBB.size)
    if not header:
        return ("", -1)

    data_type, name_length = strctBB.unpack(header)
    if data_type == ELxLiteVariantType.COMPRESS:
        raise NotImplementedError("Compressed metadata not yet implemented.")
    if data_type in (ELxLiteVariantType.DEPRECATED, ELxLiteVariantType.UNKNOWN):
        raise ValueError(f"Unknown data type in metadata header: {data_type}")

    # name of the section is a utf16 string of length `name_length * 2`
    name = stream.read(name_length * 2).decode("utf16")[:-1]
    if strip_prefix:
        name = lower.sub("", name)
    return (name, data_type)


_XMLCAST: dict[str | None, Callable[[str], Any]] = {
    "lx_uint32": int,
    "lx_int32": int,
    "lx_uint64": int,
    "lx_int64": int,
    "double": float,
    "float": float,
    "CLxStringW": str,
    "CLxByteArray": lambda x: bytearray(x, "utf8"),
    "bool": lambda x: x.lower() in {"true", "1"},
    None: str,
}


def _variant_to_dict(node: dict[str, Any]) -> Any:
    _node: dict | list = node
    if isinstance(node, dict):
        v = node.pop("@version", "1.0")
        if v != "1.0":
            raise ValueError(f"Unknown version of metadata: {v}")
        if "no_name" in node:
            _node = node["no_name"]

    if isinstance(_node, list):
        return [_variant_to_dict(i) for i in _node]
    if isinstance(_node, dict):
        runtype = _node.pop("@runtype", None)
        if runtype == "CLxListVariant" and all(k.startswith("_") for k in list(_node)):
            return [_variant_to_dict(_node[k]) for k in list(_node)]
        if "@value" in _node:
            return _XMLCAST.get(runtype, str)(_node["@value"])

        return {k: _variant_to_dict(v) for k, v in _node.items()}
    return _node


def decode_xml(data: bytes):
    import xmltodict

    d = xmltodict.parse(data)
    return _variant_to_dict(d["variant"])


# lite variant
def decode_CLxLiteVariant_json(
    data: bytes | io.BytesIO, strip_prefix: bool = True, _count: int = 1
) -> dict[str, JsonValueType]:
    output: dict[str, JsonValueType] = {}
    if not data:
        return output

    stream = data if isinstance(data, io.BytesIO) else io.BytesIO(data)

    for _ in range(_count):

        curs = stream.tell()

        name, data_type = _chunk_name_and_dtype(stream, strip_prefix)
        if data_type == -1:
            break

        value: JsonValueType
        if data_type == ELxLiteVariantType.LEVEL:
            item_count, length = strctIQ.unpack(stream.read(strctIQ.size))
            next_data_length = stream.read(length - (stream.tell() - curs))
            val: dict = decode_CLxLiteVariant_json(
                next_data_length, strip_prefix, item_count
            )
            stream.seek(item_count * 8, 1)
            # levels with a single "" key are actually lists
            value = val[""] if len(val) == 1 and "" in val else val

        elif data_type in _PARSERS:
            value = _PARSERS[data_type](stream)
        else:
            value = None

        if name == "" and name in output:
            # nd2 uses empty strings as keys for lists
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast(list, output[name]).append(value)
        else:
            output[name] = value

    return output
