from __future__ import annotations

import mmap
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, NewType, overload

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import BinaryIO, Iterator, Literal, Optional, Set, Tuple, Union

    from numpy.typing import DTypeLike

# h = short              (2)
# i = int                (4)
# I = unsigned int       (4)
# Q = unsigned long long (8)
CHUNK_INFO = struct.Struct("IIQ")  # chunk_magic, shift, length
QQ = struct.Struct("QQ")
CHUNK_MAGIC = 0x0ABECEDA
CHUNK_MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"
DEFAULT_SHIFT = 4072


@contextmanager
def ensure_handle(obj: Union[str, BinaryIO]) -> Iterator[BinaryIO]:
    fh = open(obj, "rb") if isinstance(obj, (str, bytes, Path)) else obj
    try:
        yield fh
    finally:
        # close it if we were the one to open it
        if not hasattr(obj, "fileno"):
            fh.close()


FrameIndex = NewType("FrameIndex", int)
FrameOffset = NewType("FrameOffset", int)
ImageMap = Dict[FrameIndex, FrameOffset]


class FixedImageMap(TypedDict):
    bad: Set[FrameIndex]  # frames that could not be found
    fixed: Set[FrameIndex]  # frames that were bad but fixed
    # final mapping of frame number to absolute byte offset starting the chunk
    # or None, if the chunk could not be verified
    good: ImageMap


@overload
def read_chunkmap(
    file: Union[str, BinaryIO],
    *,
    validate_frames: Literal[True] = True,
    legacy: bool = False,
    search_window: int = ...,
) -> Tuple[FixedImageMap, Dict[str, int]]:
    ...


@overload
def read_chunkmap(
    file: Union[str, BinaryIO],
    *,
    validate_frames: Literal[False],
    legacy: bool = False,
    search_window: int = ...,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    ...


def read_chunkmap(
    file: Union[str, BinaryIO],
    *,
    validate_frames=False,
    legacy: bool = False,
    search_window: int = 100,
):
    """Read chunkmap of nd2 `file`.

    Parameters
    ----------
    file : Union[str, BinaryIO]
        Filename or file handle to nd2 file.
    validate_frames : bool, optional
        Whether to verify (and attempt to fix) frames whose positions have been
        shifted relative to the predicted offset (i.e. in a corrupted file),
        by default False.
    legacy : bool, optional
        Treat file as legacy nd2 format, by default False
    search_window : int, optional
        When validate_frames is true, this is the search window (in KB) that will
        be used to try to find the actual chunk position. by default 100 KB

    Returns
    -------
    tuple
        (image chunk positions, metadata chunk positions).  If `validate_frames` is
        true, the image chunk dict will have three keys:
        `bad`: estimated frame positions that were invalid.
        `fixed`: estimated frame positions that were invalid, but corrected.
        `good`: estimated frame positions that were already valid.
    """
    with ensure_handle(file) as fh:
        if not legacy:
            return read_new_chunkmap(
                fh, validate_frames=validate_frames, search_window=search_window
            )
        from ._legacy import legacy_nd2_chunkmap

        d = legacy_nd2_chunkmap(fh)
        if validate_frames:
            f = {"bad": [], "fixed": [], "good": dict(enumerate(d.pop(b"LUNK")))}
            return f, d


def read_new_chunkmap(
    fh: BinaryIO, validate_frames: bool = False, search_window: int = 100
) -> Tuple[Union[ImageMap, FixedImageMap], Dict[str, int]]:
    """read the map of the chunks at the end of the file

    chunk rules:
    - each data chunk starts with
      - 4 bytes: CHUNK_MAGIC -> 0x0ABECEDA (big endian: 0xDACEBE0A)
      - 4 bytes: length of the chunk header (this section contains the chunk name...)
      - 8 bytes: length of chunk following the header, up to the next CHUNK_MAGIC
    """
    # the last 8 bytes contain the location of the beginning
    # of the chunkamp (~FILEMAP SIGNATURE NAME)
    # but we grab -40 to confirm that the CHUNK_MAP_SIGNATURE
    # string appears before the last 8 bytes.
    fh.seek(-40, 2)
    name, chunk = struct.unpack("32sQ", fh.read(40))
    assert name == CHUNK_MAP_SIGNATURE, f"Not a valid ND2 file: {fh.name}"

    # then we get all of the data in the chunkmap
    # this asserts that the chunkmap begins with CHUNK_MAGIC
    chunkmap_data = read_chunk(fh, chunk)

    # now look for each "!" in the chunkmap
    # and record the position associated with each chunkname
    pos = 0
    image_map: ImageMap = {}
    meta_map: Dict[str, int] = {}
    while True:
        # find the first "!", starting at pos, then go to next byte
        p = chunkmap_data.index(b"!", pos) + 1
        name = chunkmap_data[pos:p]  # name of the chunk
        if name == CHUNK_MAP_SIGNATURE:
            # break when we find the end
            break
        # the next 16 bytes contain...
        # (8) -> position of this key in the file  (@ the chunk magic)
        # (8) -> length of this chunk in the file (not including the chunk header)
        # Note: one still needs to go to `position` to read the CHUNK_INFO to know
        # the absolute position of the data (excluding the chunk header).  This can
        # be done using `read_chunk(..., position)``
        position, _ = QQ.unpack(chunkmap_data[p : p + 16])  # noqa
        if name[:13] == b"ImageDataSeq|":
            image_map[FrameIndex(int(name[13:-1]))] = position
        else:
            meta_map[name[:-1].decode("ascii")] = position
        pos = p + 16
    if validate_frames:
        return _validate_frames(fh, image_map, kbrange=search_window), meta_map
    image_map = {f: FrameOffset(o + 24 + DEFAULT_SHIFT) for f, o in image_map.items()}
    return image_map, meta_map


def _validate_frames(
    fh: BinaryIO, images: ImageMap, kbrange: int = 100
) -> FixedImageMap:
    """Look for invalid frames, and try to find their actual positions."""
    bad: Set[FrameIndex] = set()
    fixed: Set[FrameIndex] = set()
    good: ImageMap = {}
    _lengths = set()
    for fnum, _p in images.items():
        fh.seek(_p)
        magic, shift, length = CHUNK_INFO.unpack(fh.read(16))
        _lengths.add(length)
        if magic != CHUNK_MAGIC:  # corrupt frame
            correct_pos = _search(
                fh, b"ImageDataSeq|%a!" % fnum, images[fnum], kbrange=kbrange
            )
            if correct_pos is not None:
                fixed.add(fnum)
                good[fnum] = correct_pos + 24 + int(shift)
                images[fnum] = correct_pos
            else:
                bad.add(fnum)
        else:
            good[fnum] = FrameOffset(_p + 24 + int(shift))
    return {"bad": bad, "fixed": fixed, "good": good}


def _search(fh: BinaryIO, string: bytes, guess: int, kbrange: int = 100):
    """Search for `string`, in the `kbrange` bytes around position `guess`."""
    fh.seek(max(guess - ((1000 * kbrange) // 2), 0))
    try:
        p = fh.tell() + fh.read(1000 * kbrange).index(string) - 16
        fh.seek(p)
        if CHUNK_INFO.unpack(fh.read(CHUNK_INFO.size))[0] == CHUNK_MAGIC:
            return p
    except ValueError:
        return None


def read_chunk(handle: BinaryIO, position: int):
    handle.seek(position)
    # confirm chunk magic, seek to shift, read for length
    magic, shift, length = CHUNK_INFO.unpack(handle.read(CHUNK_INFO.size))
    assert magic == CHUNK_MAGIC, "invalid magic %x" % magic
    handle.seek(shift, 1)
    return handle.read(length)


def iter_chunks(handle) -> Iterator[Tuple[str, int, int]]:
    file_size = handle.seek(0, 2)
    handle.seek(0)
    pos = 0
    while True:
        magic, shift, length = CHUNK_INFO.unpack(handle.read(CHUNK_INFO.size))
        if magic:
            try:
                name = handle.read(shift).split(b"\x00", 1)[0].decode("utf-8")
            except UnicodeDecodeError:
                name = "?"
            yield (name, pos + +CHUNK_INFO.size + shift, length)
        pos += CHUNK_INFO.size + shift + length
        if pos >= file_size:
            break
        handle.seek(pos)


_default_chunk_start = CHUNK_MAGIC.to_bytes(4, "little")


def rescue_nd2(
    handle: Union[BinaryIO, str],
    frame_shape: Tuple[int, ...] = (),
    dtype: DTypeLike = "uint16",
    max_iters: Optional[int] = None,
    verbose=True,
    chunk_start: bytes = _default_chunk_start,
):
    """Iterator that yields all discovered frames in a file handle

    In nd2 files, each "frame" contains XY and all channel info (both true
    channels as well as RGB components).  Frames are laid out as (Y, X, C),
    and the `frame_shape` should match the expected frame size.  If
    `frame_shape` is not provided, a guess will be made about the vector shape
    of each frame, but it may be incorrect.

    Parameters
    ----------
    handle : Union[BinaryIO,str]
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
    chunk_start

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
            end_hdr = offset + CHUNK_INFO.size

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
                    _, shift, length = CHUNK_INFO.unpack(mm[offset:end_hdr])
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
                    except TypeError as e:
                        # buffer is likely too small
                        if verbose:
                            print(f"Error at offset {offset}: {e}")
                    iters += 1
            elif verbose:
                print(f"Found chunk at offset {offset} with no image data")

            offset += 1
            if max_iters and iters >= max_iters:
                return
