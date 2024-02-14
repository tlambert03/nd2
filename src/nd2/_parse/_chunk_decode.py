"""FIXME: this has a lot of code duplication with _chunkmap.py."""

from __future__ import annotations

import mmap
import struct
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, ContextManager, cast

import numpy as np

if TYPE_CHECKING:
    from os import PathLike
    from typing import Final, Iterator

    from numpy.typing import DTypeLike

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
# this appears at the beginning of the file, as the "name" in the StartFileChunk
ND2_FILE_SIGNATURE:     Final = b"ND2 FILE SIGNATURE CHUNK NAME01!"  # len 32
# should appear at the very end of the file
ND2_CHUNKMAP_SIGNATURE: Final = b"ND2 CHUNK MAP SIGNATURE 0000001!"
# should be the name at the beginning of the chunkmap section
ND2_FILEMAP_SIGNATURE:  Final = b"ND2 FILEMAP SIGNATURE NAME 0001!"
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
# the last 40 bytes of the file, containing the signature and location of chunkmap
# char name[32]
# uint64_t offset


def get_version(fh: BinaryIO | StrOrBytesPath) -> tuple[int, int]:
    """Get the version of the ND2 file or raise an exception.

    Parameters
    ----------
    fh : BinaryIO | str | bytes | Path
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
    if hasattr(fh, "read"):
        ctx: ContextManager[BinaryIO] = nullcontext(cast("BinaryIO", fh))
    else:
        ctx = open(fh, "rb")

    with ctx as fh:
        fh.seek(0)
        fname = str(fh.name)
        chunk = START_FILE_CHUNK.unpack(fh.read(START_FILE_CHUNK.size))

    magic, name_length, data_length, name, data = cast("StartFileChunk", chunk)

    # sanity checks
    if magic != ND2_CHUNK_MAGIC:
        if magic == JP2_MAGIC:
            return (1, 0)  # legacy JP2 files are version 1.0
        raise ValueError(  # pragma: no cover
            f"Not a valid ND2 file: {fname}. (magic: {magic!r})"
        )
    if name_length != 32 or data_length != 64 or name != ND2_FILE_SIGNATURE:
        raise ValueError(f"Corrupt ND2 file header chunk: {fname}")  # pragma: no cover

    # data will now be something like Ver2.0, Ver3.0, etc.
    return (int(chr(data[3])), int(chr(data[5])))


def get_chunkmap(fh: BinaryIO, error_radius: int | None = None) -> ChunkMap:
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
    fh : BinaryIO
        An open nd2 file.  File is assumed to be a valid ND2 file.  (use `get_version`)
    error_radius : int, optional
        If b"ND2 FILEMAP SIGNATURE NAME 0001!" is not found at expected location and
        `error_radius` is not None, then an area of +/- `error_radius` bytes will be
        searched for the signature.

    Returns
    -------
    ChunkMap
        A dictionary mapping chunk names to (offset, size) pairs.

    Raises
    ------
    ValueError
        If the file is not a valid ND2 file or the chunkmap is corrupt.
    """
    # the last (32,8) bytes of the file contain the (signature, location) of chunkmap
    fh.seek(-40, 2)
    sig, location = SIG_CHUNKMAP_LOC.unpack(fh.read(SIG_CHUNKMAP_LOC.size))
    if sig != ND2_CHUNKMAP_SIGNATURE:  # pragma: no cover
        raise ValueError(f"Invalid ChunkMap signature {sig!r} in file {fh.name!r}")

    # get all of the data in the chunkmap
    chunkmap_data = _robustly_read_named_chunk(
        fh, location, search_radius=error_radius, expect_name=ND2_FILEMAP_SIGNATURE
    )

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


def read_nd2_chunk(
    fh: BinaryIO, start_position: int, expect_name: bytes | None = None
) -> bytes:
    """Read a single chunk in an ND2 file at `start_position`.

    Each data chunk starts with:
      - 4 bytes: CHUNK_MAGIC -> 0x0ABECEDA (big endian: 0xDACEBE0A)
      - 4 bytes: length of the chunk header (this section contains the chunk name...)
      - 8 bytes: length of chunk following the header, up to the next CHUNK_MAGIC

    For example:
        magic    name_len <-  data_len   -> <- NameChunk (name_len long) ...
        DACEBE0A 20100000 D00F0000 00000000 004E4432 20434855 4E4B204D 41...

    Parameters
    ----------
    fh : BinaryIO
        An open nd2 file.  File is assumed to be a valid ND2 file.  (use `get_version`)
    start_position : int
        The position in the file to start reading the chunk.
    expect_name : bytes | None
        If not None, the chunk name must match this value.

    Returns
    -------
    bytes
        The data in the chunk.

    Raises
    ------
    ValueError
        If the first 4 bytes are not 0x0ABECEDA or if `expect_name` is provided and
        the chunk name does not match.
    """
    fh.seek(start_position)
    magic, name_length, data_length = CHUNK_HEADER.unpack(fh.read(CHUNK_HEADER.size))
    if magic != ND2_CHUNK_MAGIC:
        raise ValueError(
            f"Invalid nd2 chunk header '{magic:x}' at pos {start_position}"
        )
    if expect_name is None:
        fh.seek(name_length, 1)  # seek over name_length
    else:
        name = fh.read(name_length)
        if not name.startswith(expect_name):
            _name = name.decode("utf-8", "replace").replace("\x00", "")
            raise ValueError(
                f"Expected chunk name {expect_name!r} at {start_position}"
                f" but found {_name!r}"
            )
    return fh.read(data_length)  # then read data_length bytes


def _robustly_read_named_chunk(
    fh: BinaryIO,
    start_position: int,
    expect_name: bytes = ND2_FILEMAP_SIGNATURE,
    search_radius: int | None = None,
) -> bytes:
    """Read nd2 chunk at start_position ND2 file, with error robustness.

    Same logic as _read_nd2_chunk, but allows for a search radius around the
    expected location of the chunkmap if a chunk named `expect_name` is not found at
    the expected location.

    Parameters
    ----------
    fh : BinaryIO
        An open nd2 file.  File is assumed to be a valid ND2 file.
    start_position : int
        The position in the file to start reading the chunk.
    expect_name : bytes
        The name of the chunk to expect at `start_position` + 16 bytes.
    search_radius : int | None
        If not None, and `expect_name` is not found at `start_position` + 16 bytes,
        search for `expect_name` in the surrounding `search_radius` bytes.
    """
    try:
        return read_nd2_chunk(fh, start_position, expect_name=expect_name)
    except ValueError as e:
        err_msg = (
            f"File {fh.name!r} appears to be corrupt. Expected "
            f"{expect_name!r} at position "
            f"{start_position} but did not find it."
        )
        if search_radius is not None:
            # if we didn't find the expect_name name, look in surrounding area
            new_start = start_position - search_radius
            fh.seek(new_start)
            data = fh.read(search_radius * 2)
            try:
                idx = data.index(expect_name)
            except ValueError:
                err_msg += f" (Also looked in the surrounding {search_radius} bytes)"
            else:
                fixed_position = new_start + idx - CHUNK_HEADER.size
                return read_nd2_chunk(fh, fixed_position)

        raise ValueError(err_msg) from e


def iter_chunks(handle: BinaryIO) -> Iterator[tuple[str, int, int]]:
    file_size = handle.seek(0, 2)
    handle.seek(0)
    pos = 0
    while True:
        magic, shift, length = CHUNK_HEADER.unpack(handle.read(CHUNK_HEADER.size))
        if magic:
            try:
                name = handle.read(shift).split(b"\x00", 1)[0].decode("utf-8")
            except UnicodeDecodeError:  # pragma: no cover
                name = "?"
            yield (name, pos + +CHUNK_HEADER.size + shift, length)
        pos += CHUNK_HEADER.size + shift + length
        if pos >= file_size:
            break
        handle.seek(pos)


_default_chunk_start = ND2_CHUNK_MAGIC.to_bytes(4, "little")


def rescue_nd2(
    handle: BinaryIO | str,
    frame_shape: tuple[int, ...] = (),
    dtype: DTypeLike = "uint16",
    max_iters: int | None = None,
    verbose: bool = True,
    chunk_start: bytes = _default_chunk_start,
) -> Iterator[np.ndarray]:
    """Iterator that yields all discovered frames in a file handle.

    In nd2 files, each "frame" contains XY and all channel info (both true
    channels as well as RGB components).  Frames are laid out as (Y, X, C),
    and the `frame_shape` should match the expected frame size.  If
    `frame_shape` is not provided, a guess will be made about the vector shape
    of each frame, but it may be incorrect.

    Parameters
    ----------
    handle : BinaryIO | str
        Filepath string, or binary file handle (For example
        `handle = open('some.nd2', 'rb')`)
    frame_shape : Tuple[int, ...], optional
        expected shape of each frame, by default a 1 dimensional array will
        be yielded for each frame, which can be reshaped later if desired.
        NOTE: nd2 frames are generally ordered as
        (height, width, true_channels, rgbcomponents).
        So unlike numpy, which would use (channels, Y, X), you should use
        (Y, X, channels)
    dtype : np.dtype, optional
        Data type, by default np.uint16
    max_iters : Optional[int], optional
        A maximum number of frames to yield, by default will yield until the
        end of the file is reached
    verbose : bool
        whether to print info
    chunk_start : bytes, optional
        The bytes that start each chunk, by default 0x0ABECEDA.to_bytes(4, "little")

    Yields
    ------
    np.ndarray
        each discovered frame in the file

    Examples
    --------
    >>> with open('some_bad.nd2', 'rb') as fh:
    >>>     frames = rescue_nd2(fh, (512, 512, 4), 'uint16')
    >>>     ary = np.stack(frames)

    You will likely want to reshape `ary` after that.
    """
    dtype = np.dtype(dtype)
    with ensure_handle(handle) as _fh:
        mm = mmap.mmap(_fh.fileno(), 0, access=mmap.ACCESS_READ)

        offset = 0
        iters = 0
        while True:
            # search for the next part of the file starting with CHUNK_START
            offset = mm.find(chunk_start, offset)
            if offset < 0:
                if verbose:
                    print("End of file.")
                return

            # location at the end of the chunk header
            end_hdr = offset + CHUNK_HEADER.size

            # find the next "!"
            # In nd2 files, each data chunk starts with the
            # string "ImageDataSeq|N" ... where N is the frame index
            next_bang = mm.find(b"!", end_hdr)
            if next_bang > 0 and (0 < next_bang - end_hdr < 128):
                # if we find the "!"... make sure we have an ImageDataSeq
                chunk_name = mm[end_hdr:next_bang]
                if chunk_name.startswith(b"ImageDataSeq|"):
                    if verbose:
                        print(f"Found image {iters} at offset {offset}")
                    # Now, read the actual data
                    _, shift, length = CHUNK_HEADER.unpack(mm[offset:end_hdr])
                    # convert to numpy array and yield
                    # (can't remember why the extra 8 bytes)
                    try:
                        shape = frame_shape or ((length - 8) // dtype.itemsize,)
                        yield np.ndarray(
                            shape=shape,
                            dtype=dtype,
                            buffer=mm,
                            offset=end_hdr + shift + 8,
                        )
                    except TypeError as e:  # pragma: no cover
                        # buffer is likely too small
                        if verbose:
                            print(f"Error at offset {offset}: {e}")
                    iters += 1
            elif verbose:
                print(f"Found chunk at offset {offset} with no image data")

            offset += 1
            if max_iters and iters >= max_iters:
                return


@contextmanager
def ensure_handle(obj: str | BinaryIO) -> Iterator[BinaryIO]:
    fh = open(obj, "rb") if isinstance(obj, (str, bytes, Path)) else obj
    try:
        yield fh
    finally:
        # close it if we were the one to open it
        if not hasattr(obj, "fileno"):
            fh.close()
