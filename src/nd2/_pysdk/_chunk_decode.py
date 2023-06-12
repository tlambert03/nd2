"""FIXME: this has a lot of code duplication with _chunkmap.py."""
from __future__ import annotations

import struct
from io import BufferedReader
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from os import PathLike
    from typing import Final

    StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]

    StartFileChunk = tuple[int, int, int, bytes, bytes]
    #              = (magic, name_length, data_length, name, data)

    ChunkMapItem = tuple[int, int]  # (offset, size)
    ChunkMap = dict[bytes, ChunkMapItem]
    # a Chunkmap is mapping of chunk names (bytes) to (offset, size) pairs.
    # {
    #   b'ImageTextInfoLV!': (13041664, 2128),
    #   b'ImageTextInfo!': (13037568, 1884),
    #   b'ImageMetadataSeq|0!': (237568, 33412),
    #   ...
    # }

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
# used for (offset, size) in chunkmap
# 2 x uint64_t

CHUNK_HEADER = struct.Struct("IIQ")
# beginning of every chunk an ND2 file
# uint32_t magic
# uint32_t nameLen
# uint64_t dataLen

START_FILE_CHUNK = struct.Struct(f"{CHUNK_HEADER.format}32s64s")
# ChunkHeader header
# char name[32]
# char data[64]

SIG_CHUNKMAP_LOC = struct.Struct("32sQ")
# the last 40 bytes of the file, containing the signature and locatio of chunkmap
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
        fh.seek(0)
        chunk = START_FILE_CHUNK.unpack(fh.read(START_FILE_CHUNK.size))

    magic, name_length, data_length, name, data = cast("StartFileChunk", chunk)

    # sanity checks
    if magic != ND2_CHUNK_MAGIC:
        if magic == JP2_MAGIC:
            return (1, 0)  # legacy JP2 files are version 1.0
        raise ValueError(f"Not a valid ND2 file: {fh.name}. (magic: {magic!r})")
    if name_length != 32 or data_length != 64 or name != ND2_FILE_SIGNATURE:
        raise ValueError(f"Corrupt ND2 file header chunk: {fh.name}")

    # data will now be something like Ver2.0, Ver3.0, etc.
    return (int(chr(data[3])), int(chr(data[5])))


def get_chunkmap(fh: BufferedReader) -> ChunkMap:
    """Read the map of the chunks at the end of an ND2 file.

    A Chunkmap is mapping of chunk names (bytes) to (offset, size) pairs.

    ```python
    {
      b'ImageTextInfoLV!': (13041664, 2128),
      b'ImageTextInfo!': (13037568, 1884),
      b'ImageMetadataSeq|0!': (237568, 33412),
      ...
    }
    ```

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

    chunk rules:
    - each data chunk starts with
      - 4 bytes: CHUNK_MAGIC -> 0x0ABECEDA (big endian: 0xDACEBE0A)
      - 4 bytes: length of the chunk header (this section contains the chunk name...)
      - 8 bytes: length of chunk following the header, up to the next CHUNK_MAGIC

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
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(f"Invalid chunk header magic: {magic:x}")

    fh.seek(name_length, 1)  # seek over name_length
    return fh.read(data_length)  # then read data_length bytes
