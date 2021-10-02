from __future__ import annotations

import io
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, overload

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import BinaryIO, Dict, Iterator, Literal, Optional, Set, Tuple, Union

# h = short              (2)
# i = int                (4)
# I = unsigned int       (4)
# Q = unsigned long long (8)
CHUNK_INFO = struct.Struct("IIQ")
QQ = struct.Struct("QQ")
CHUNK_MAGIC = 0x0ABECEDA
CHUNK_MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"


@contextmanager
def ensure_handle(obj: Union[str, io.BytesIO]) -> Iterator[BinaryIO]:
    fh = obj if isinstance(obj, io.IOBase) else open(obj, "rb")
    try:
        yield fh
    finally:
        # close it if we were the one to open it
        if not hasattr(obj, "fileno"):
            fh.close()


class FixedImageMap(TypedDict):
    bad: Set[int]  # frames that could not be found
    fixed: Set[int]  # frames that were bad but fixed
    # final mapping of frame number to absolute byte offset starting the chunk
    # or None, if the chunk could not be verified
    safe: Dict[int, Optional[int]]


@overload
def read_chunkmap(
    file, fixup: Literal[True] = True
) -> Tuple[FixedImageMap, Dict[str, int]]:
    ...


@overload
def read_chunkmap(file, fixup: Literal[False]) -> Tuple[Dict[int, int], Dict[str, int]]:
    ...


def read_chunkmap(file, fixup=True):
    """read the map of the chunks at the end of the file

    chunk rules:
    - each data chunk starts with
      - 4 bytes: CHUNK_MAGIC -> 0x0ABECEDA (big endian: 0xDACEBE0A)
      - 4 bytes: length of the chunk header (this section contains the chunk name...)
      - 8 bytes: length of chunk following the header, up to the next CHUNK_MAGIC
    """
    with ensure_handle(file) as fh:
        # the last 8 bytes contain the location of the beginning
        # of the chunkamp (~FILEMAP SIGNATURE NAME)
        # but we grab -40 to confirm that the CHUNK_MAP_SIGNATURE
        # string appears before the last 8 bytes.
        fh.seek(-40, 2)
        name, chunk = struct.unpack("32sQ", fh.read(40))
        assert name == CHUNK_MAP_SIGNATURE, "Not a valid ND2 file"

        # then we get all of the data in the chunkmap
        # this asserts that the chunkmap begins with CHUNK_MAGIC
        chunkmap_data = read_chunk(fh, chunk)

        # now look for each "!" in the chunkmap
        # and record the position associated with each chunkname
        pos = 0
        image_map: dict = {}
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
                image_map[int(name[13:-1])] = position
            else:
                meta_map[name[:-1].decode("ascii")] = position
            pos = p + 16
    if fixup:
        return _fix_frames(fh, image_map), meta_map
    return image_map, meta_map


def _fix_frames(fh: BinaryIO, images: Dict[int, int]) -> FixedImageMap:
    """Look for corrupt frames, and try to find their actual positions."""
    bad: Set[int] = set()
    fixed: Set[int] = set()
    safe: Dict[int, Optional[int]] = {}
    _lengths = set()
    for fnum, _p in images.items():
        fh.seek(_p)
        magic, shift, length = CHUNK_INFO.unpack(fh.read(16))
        _lengths.add(length)
        if magic != CHUNK_MAGIC:  # corrupt frame
            correct_pos = _search(fh, b"ImageDataSeq|%a!" % fnum, images[fnum])
            if correct_pos is not None:
                fixed.add(fnum)
                safe[fnum] = correct_pos + 24 + int(shift)
                images[fnum] = correct_pos
            else:
                safe[fnum] = None
        else:
            safe[fnum] = _p + 24 + int(shift)
    return {"bad": bad, "fixed": fixed, "safe": safe}


def _search(fh: BinaryIO, string: bytes, guess: int, kbrange=100):
    """Search for `string`, in the `kbrange` bytes around position `guess`."""
    fh.seek(max(guess - ((1000 * kbrange) // 2), 0))
    try:
        p = fh.tell() + fh.read(1000 * kbrange).index(string) - 16
        fh.seek(p)
        if CHUNK_INFO.unpack(fh.read(16))[0] == CHUNK_MAGIC:
            return p
    except ValueError:
        return None


def read_chunk(handle: BinaryIO, position: int):
    handle.seek(position)
    # confirm chunk magic, seek to shift, read for length
    magic, shift, length = CHUNK_INFO.unpack(handle.read(16))
    assert magic == CHUNK_MAGIC, "invalid magic %x" % magic
    handle.seek(shift, 1)
    return handle.read(length)
