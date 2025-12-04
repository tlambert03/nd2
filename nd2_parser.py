"""
ND2 File Parser - Extract metadata to JSON based on ND2_FILE_FORMAT.md spec.

This parser handles modern ND2 files (versions 2.x and 3.x).

Usage:
    python nd2_parser.py <file.nd2> [output.json]

The parser extracts all metadata chunks (excluding raw image data) and outputs
them as a JSON document. It handles:
- CLX Lite binary format (Version 3.x)
- CLX XML format (Version 2.x)
- Raw CustomData chunks (arrays of primitives)
"""

from __future__ import annotations

import base64
import contextlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

# Constants from spec
ND2_FILE_SIGNATURE = b"ND2 FILE SIGNATURE CHUNK NAME01!"
ND2_CHUNKMAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"
ND2_FILEMAP_SIGNATURE = b"ND2 FILEMAP SIGNATURE NAME 0001!"
ND2_CHUNK_MAGIC = 0x0ABECEDA
CHUNK_ALIGNMENT = 4096


def read_file_header(f) -> dict[str, Any]:
    """Read the 112-byte file header chunk."""
    f.seek(0)
    data = f.read(112)

    magic = struct.unpack("<I", data[0:4])[0]
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(f"Invalid magic number: {hex(magic)}")

    name_len = struct.unpack("<I", data[4:8])[0]
    data_len = struct.unpack("<Q", data[8:16])[0]
    chunk_name = data[16:48]
    version_bytes = data[48:112]

    # Extract version string (null-terminated)
    version_str = version_bytes.split(b"\x00")[0].decode("utf-8")

    return {
        "magic": hex(magic),
        "name_len": name_len,
        "data_len": data_len,
        "chunk_name": chunk_name.decode("utf-8").rstrip("\x00"),
        "version": version_str,
    }


def read_chunk_map(f) -> dict[str, tuple[int, int]]:
    """Read the chunk map from the end of the file."""
    # Read footer signature (last 40 bytes)
    f.seek(-40, 2)
    footer = f.read(40)

    signature = footer[0:32]
    if signature != ND2_CHUNKMAP_SIGNATURE:
        raise ValueError(f"Invalid chunk map signature: {signature}")

    chunk_map_offset = struct.unpack("<Q", footer[32:40])[0]

    # Seek to chunk map and read its header
    f.seek(chunk_map_offset)
    header = f.read(16)

    magic = struct.unpack("<I", header[0:4])[0]
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(f"Invalid chunk map magic: {hex(magic)}")

    name_len = struct.unpack("<I", header[4:8])[0]
    data_len = struct.unpack("<Q", header[8:16])[0]

    # Read chunk name
    f.read(name_len).rstrip(b"\x00")

    # Read chunk map data
    map_data = f.read(data_len)

    # Parse chunk map entries
    chunks = {}
    pos = 0
    while pos < len(map_data):
        # Find end of chunk name (ends with '!')
        name_end = map_data.find(b"!", pos)
        if name_end == -1:
            break

        name = map_data[pos : name_end + 1].decode("utf-8")
        pos = name_end + 1

        # Check if we've hit the signature
        if name == "ND2 CHUNK MAP SIGNATURE 0000001!":
            break

        # Read offset and size (8 bytes each)
        if pos + 16 > len(map_data):
            break

        offset = struct.unpack("<Q", map_data[pos : pos + 8])[0]
        size = struct.unpack("<Q", map_data[pos + 8 : pos + 16])[0]
        pos += 16

        chunks[name] = (offset, size)

    return chunks


def read_chunk_data(f, offset: int) -> bytes:
    """Read raw chunk data from the file."""
    f.seek(offset)
    header = f.read(16)

    magic = struct.unpack("<I", header[0:4])[0]
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(f"Invalid chunk magic at offset {offset}: {hex(magic)}")

    name_len = struct.unpack("<I", header[4:8])[0]
    data_len = struct.unpack("<Q", header[8:16])[0]

    # Skip chunk name
    f.read(name_len)

    # Read chunk data
    return f.read(data_len)


# CLX Lite decoder (Version 3.x)
CLX_TYPE_BOOL = 1
CLX_TYPE_INT32 = 2
CLX_TYPE_UINT32 = 3
CLX_TYPE_INT64 = 4
CLX_TYPE_UINT64 = 5
CLX_TYPE_DOUBLE = 6
CLX_TYPE_VOID_PTR = 7
CLX_TYPE_STRING = 8
CLX_TYPE_BYTE_ARRAY = 9
CLX_TYPE_DEPRECATED = 10
CLX_TYPE_LEVEL = 11
CLX_TYPE_COMPRESSED = 76  # 'L'


def decode_utf16_string(data: bytes, pos: int) -> tuple[str, int]:
    """Decode a null-terminated UTF-16 string."""
    chars = []
    while pos + 2 <= len(data):
        code_unit = struct.unpack("<H", data[pos : pos + 2])[0]
        pos += 2
        if code_unit == 0:
            break
        chars.append(chr(code_unit))
    return "".join(chars), pos


def decode_clx_lite_value(data: bytes, pos: int) -> tuple[Any, int]:
    """Decode a single CLX Lite value. Returns (value, new_position)."""
    if pos >= len(data):
        return None, pos

    # Read type code
    type_code = data[pos]
    pos += 1

    if pos >= len(data):
        return None, pos

    # Read name length (number of UTF-16 code units)
    name_len = data[pos]
    pos += 1

    # Read name (UTF-16, name_len * 2 bytes)
    name_bytes = name_len * 2
    if pos + name_bytes > len(data):
        return None, pos

    if name_len > 0:
        name = data[pos : pos + name_bytes].decode("utf-16-le").rstrip("\x00")
    else:
        name = ""  # Empty name = list item

    pos += name_bytes

    # Decode value based on type
    value: Any = None

    if type_code == CLX_TYPE_BOOL:
        if pos >= len(data):
            return (name, None), pos
        value = bool(data[pos])
        pos += 1

    elif type_code == CLX_TYPE_INT32:
        if pos + 4 > len(data):
            return (name, None), pos
        value = struct.unpack("<i", data[pos : pos + 4])[0]
        pos += 4

    elif type_code == CLX_TYPE_UINT32:
        if pos + 4 > len(data):
            return (name, None), pos
        value = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4

    elif type_code == CLX_TYPE_INT64:
        if pos + 8 > len(data):
            return (name, None), pos
        value = struct.unpack("<q", data[pos : pos + 8])[0]
        pos += 8

    elif type_code == CLX_TYPE_UINT64:
        if pos + 8 > len(data):
            return (name, None), pos
        value = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8

    elif type_code == CLX_TYPE_DOUBLE:
        if pos + 8 > len(data):
            return (name, None), pos
        value = struct.unpack("<d", data[pos : pos + 8])[0]
        pos += 8

    elif type_code == CLX_TYPE_VOID_PTR:
        if pos + 8 > len(data):
            return (name, None), pos
        value = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8

    elif type_code == CLX_TYPE_STRING:
        value, pos = decode_utf16_string(data, pos)

    elif type_code == CLX_TYPE_BYTE_ARRAY:
        if pos + 8 > len(data):
            return (name, None), pos
        arr_len = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        if pos + arr_len > len(data):
            arr_len = len(data) - pos
        # Store as base64 for JSON compatibility
        value = base64.b64encode(data[pos : pos + arr_len]).decode("ascii")
        pos += arr_len

    elif type_code == CLX_TYPE_LEVEL:
        if pos + 12 > len(data):
            return (name, None), pos
        item_count = struct.unpack("<I", data[pos : pos + 4])[0]
        level_len = struct.unpack("<Q", data[pos + 4 : pos + 12])[0]
        pos += 12

        # level_len is the total byte length of nested data + index array
        # index array is at the end: item_count * 8 bytes
        index_size = item_count * 8
        nested_data_len = level_len - index_size

        # Parse nested items
        nested_end = pos + nested_data_len
        if nested_end > len(data):
            nested_end = len(data)

        value = {}
        list_index = 0
        while pos < nested_end:
            item, pos = decode_clx_lite_value(data, pos)
            if item is None:
                break
            item_name, item_value = item
            if item_name == "":
                item_name = f"i{list_index:010d}"
                list_index += 1
            value[item_name] = item_value

        # Skip index array
        pos = pos + index_size
        if pos > len(data):
            pos = len(data)

    elif type_code == CLX_TYPE_COMPRESSED:
        # Skip 10 bytes after type/name header, then decompress
        if pos + 10 > len(data):
            return (name, None), pos
        pos += 10
        # Rest is zlib compressed
        try:
            decompressed = zlib.decompress(data[pos:])
            value = decode_clx_lite(decompressed)
            pos = len(data)  # Consumed all remaining data
        except zlib.error:
            value = {"_error": "decompression failed"}
            pos = len(data)

    else:
        # Unknown type - likely trailing garbage, stop parsing
        return None, len(data)

    return (name, value), pos


def decode_clx_lite(data: bytes, pos: int = 0) -> dict[str, Any]:
    """Decode CLX Lite binary format into a dictionary."""
    result = {}
    list_index = 0

    while pos < len(data):
        item, pos = decode_clx_lite_value(data, pos)
        if item is None:
            break
        name, value = item
        if name == "":
            name = f"i{list_index:010d}"
            list_index += 1
        if name:
            result[name] = value

    return result


def decode_clx_xml(data: bytes) -> dict[str, Any]:
    """Decode CLX XML format (Version 2.x) into a dictionary."""
    import xml.etree.ElementTree as ET

    # Try various encodings
    xml_str = None

    # Check for UTF-8 BOM or XML declaration
    if data.startswith(b"<?xml") or data.startswith(b"\xef\xbb\xbf"):
        with contextlib.suppress(UnicodeDecodeError):
            xml_str = data.decode("utf-8")

    # Try UTF-16 with BOM
    if xml_str is None:
        if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
            with contextlib.suppress(UnicodeDecodeError):
                xml_str = data.decode("utf-16")

    # Try UTF-16-LE without BOM
    if xml_str is None:
        try:
            xml_str = data.decode("utf-16-le")
            # Validate it looks like XML
            if not xml_str.strip().startswith("<"):
                xml_str = None
        except UnicodeDecodeError:
            pass

    if xml_str is None:
        raise ValueError("Could not decode XML data")

    # Parse XML
    root = ET.fromstring(xml_str)

    def parse_element(elem) -> Any:
        runtype = elem.get("runtype")
        value_str = elem.get("value")

        if runtype and value_str is not None:
            # Leaf node with value
            if runtype in ("lx_int32", "lx_uint32", "lx_int64", "lx_uint64"):
                return int(value_str)
            elif runtype == "double":
                return float(value_str)
            elif runtype == "bool":
                return value_str.lower() == "true"
            elif runtype in ("CLxStringW", "string"):
                return value_str
            elif runtype == "CLxByteArray":
                return value_str  # Keep as string
            else:
                return value_str
        else:
            # Container node
            result = {}
            for child in elem:
                child_name = child.tag
                child_value = parse_element(child)
                result[child_name] = child_value
            return result if result else None

    return parse_element(root)


def parse_raw_custom_data(data: bytes) -> dict[str, Any]:
    """Parse raw CustomData chunks which are typically arrays of primitives."""
    # Determine data type based on content
    # Most are float64 arrays (positions, times, etc.)
    if len(data) == 0:
        return {"_empty": True}

    # Try to interpret as float64 array if size is multiple of 8
    if len(data) % 8 == 0:
        count = len(data) // 8
        if count > 0 and count <= 10000:  # Reasonable array size
            try:
                values = list(struct.unpack(f"<{count}d", data))
                # Check if values look reasonable (not NaN/Inf garbage)
                if all(-1e20 < v < 1e20 for v in values[: min(10, len(values))]):
                    return {"_type": "float64_array", "_count": count, "values": values}
            except struct.error:
                pass

    # Try to interpret as uint64 array
    if len(data) % 8 == 0:
        count = len(data) // 8
        if count > 0 and count <= 10000:
            try:
                values = list(struct.unpack(f"<{count}Q", data))
                return {"_type": "uint64_array", "_count": count, "values": values}
            except struct.error:
                pass

    # Fall back to base64-encoded raw bytes
    return {
        "_type": "raw_bytes",
        "_size": len(data),
        "_base64": base64.b64encode(data).decode("ascii"),
    }


def parse_metadata_chunk(
    data: bytes, version: str, chunk_name: str = ""
) -> dict[str, Any]:
    """Parse a metadata chunk based on version and chunk type."""
    # Raw CustomData chunks (not CustomDataVar/CustomDataSeq) are primitive arrays
    if (
        chunk_name.startswith("CustomData|")
        and "Var" not in chunk_name
        and "Seq" not in chunk_name
    ):
        return parse_raw_custom_data(data)

    # Check if data looks like XML (starts with <)
    is_xml = (
        data.startswith(b"<?xml")
        or data.startswith(b"\xef\xbb\xbf")
        or (len(data) > 2 and data[:2] in (b"\xff\xfe", b"\xfe\xff"))
    )

    if is_xml:
        # Try XML first
        try:
            result = decode_clx_xml(data)
            if result:
                return result
        except Exception:
            pass

    if version.startswith("Ver3") or version.startswith("Ver2"):
        # Try CLX Lite first (more common in modern Ver3 files)
        try:
            result = decode_clx_lite(data)
            if result:  # Only return if we got something
                return result
        except Exception:
            pass

        # Try CLX XML as fallback
        if not is_xml:
            try:
                result = decode_clx_xml(data)
                if result:
                    return result
            except Exception:
                pass

    # Return raw bytes as fallback
    return {
        "_raw_size": len(data),
        "_raw_base64": base64.b64encode(data[:1000]).decode("ascii")
        + ("..." if len(data) > 1000 else ""),
    }


def is_metadata_chunk(name: str) -> bool:
    """Check if a chunk contains metadata (not image data)."""
    # Skip image data chunks
    if name.startswith("ImageDataSeq|"):
        return False
    return True


def parse_nd2_metadata(file_path: str | Path) -> dict[str, Any]:
    """Parse an ND2 file and extract all metadata as a dictionary."""
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        # Read file header
        header = read_file_header(f)
        version = header["version"]

        # Read chunk map
        chunk_map = read_chunk_map(f)

        # Parse all metadata chunks
        metadata = {
            "_file_header": header,
            "_chunk_names": list(chunk_map.keys()),
        }

        for chunk_name, (offset, _size) in chunk_map.items():
            if not is_metadata_chunk(chunk_name):
                continue

            try:
                chunk_data = read_chunk_data(f, offset)
                if len(chunk_data) > 0:
                    parsed = parse_metadata_chunk(chunk_data, version, chunk_name)
                    metadata[chunk_name] = parsed
                else:
                    metadata[chunk_name] = None
            except Exception as e:
                metadata[chunk_name] = {"_error": str(e)}

        return metadata


def main() -> None:
    """CLI entry point for the ND2 metadata parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nd2_parser.py <file.nd2> [output.json]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    metadata = parse_nd2_metadata(file_path)

    # Output as JSON
    json_output = json.dumps(metadata, indent=2, default=str)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_output)
        print(f"Metadata written to {output_path}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
