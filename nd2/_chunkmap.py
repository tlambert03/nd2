import io
import re
import struct
from contextlib import contextmanager
from typing import Any, BinaryIO, Dict, Iterator, Optional, Set, Tuple, Union

import numpy as np

ImageSeqPtrn = re.compile(br"ImageDataSeq\|(\d+)!?")

CHUNK_MAGIC = 0x0ABECEDA
CHUNK_MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"

# h = short              (2)
# i = int                (4)
# I = unsigned int       (4)
# Q = unsigned long long (8)
big_iihi = struct.Struct(">iihi")
IIQ = struct.Struct("IIQ")
QQ = struct.Struct("QQ")


@contextmanager
def ensure_handle(obj: Union[str, io.BytesIO]) -> Iterator[BinaryIO]:
    fh = obj if isinstance(obj, io.IOBase) else open(obj, "rb")
    try:
        yield fh
    finally:
        if not hasattr(obj, "fileno"):
            fh.close()


def read_chunkmap(file, fixup=True) -> dict:
    "read the map of the chunks at the end of the file"
    with ensure_handle(file) as fh:
        # the last 8 bytes contain the location of the beginning
        # of the chunkamp (~FILEMAP SIGNATURE NAME)
        # but we grab -40 to confirm that the CHUNK_MAP_SIGNATURE
        # string appears before the last 8 bytes.
        fh.seek(-40, 2)
        name, chunk = struct.unpack("32sQ", fh.read(40))
        assert name == CHUNK_MAP_SIGNATURE

        # then we get all of the data in the chunkmap
        # this asserts that the chunkmap begins with CHUNK_MAGIC
        data = read_chunk(fh, chunk)

        # now look for each "!" in the chunkmap
        # and record the offset associated with each chunkname
        pos = 0
        map: Dict[str, Any] = {"images": {}}
        while True:
            # find the first "!", starting at p, go to next byte
            p = data.index(b"!", pos) + 1
            name = data[pos:p]  # name of the chunk
            if name == CHUNK_MAP_SIGNATURE:
                # break when we find the end
                break
            # the next 8 bytes contain the position of the chunk magic
            # corresponding to this key
            result = QQ.unpack(data[p : p + 16])  # noqa
            if name[:13] == b"ImageDataSeq|":
                map["images"][int(name[13:-1])] = result
            else:
                map[name[:-1].decode("ascii")] = result
            pos = p + 16
        if fixup:
            bad, fixed, safe = _fix_frames(fh, map["images"])
            map["bad_frames"] = bad
            map["fixed_frames"] = fixed
            map["safe_frames"] = safe
    return map


def _fix_frames(
    fh: BinaryIO, images: Dict[int, Tuple[int, int]]
) -> Tuple[Set[int], Set[int], Dict[int, Optional[int]]]:
    bad: Set[int] = set()
    fixed: Set[int] = set()
    safe: Dict[int, Optional[int]] = {}
    _lengths = set()
    for fnum, (_p, _) in images.items():
        fh.seek(_p)
        magic, shift, length = IIQ.unpack(fh.read(16))
        _lengths.add(length)
        if magic != CHUNK_MAGIC:
            correct_pos = _search(fh, b"ImageDataSeq|%a!" % fnum, images[fnum][0])
            if correct_pos is not None:
                fixed.add(fnum)
                safe[fnum] = correct_pos + 24 + int(shift)
                images[fnum] = (correct_pos, correct_pos)
            else:
                safe[fnum] = None
        else:
            safe[fnum] = _p + 24 + int(shift)
    return bad, fixed, safe


def _search(fh: BinaryIO, string: bytes, guess: int, kbrange=100):
    fh.seek(max(guess - ((1000 * kbrange) // 2), 0))
    try:
        p = fh.tell() + fh.read(1000 * kbrange).index(string) - 16
        fh.seek(p)
        if IIQ.unpack(fh.read(16))[0] == CHUNK_MAGIC:
            return p
    except ValueError:
        return None


def read_chunk(handle: BinaryIO, position: int):
    handle.seek(position)
    # confirm chunk magic, seek to shift, read for length
    magic, shift, length = IIQ.unpack(handle.read(16))
    assert magic == CHUNK_MAGIC, "invalid magic %x" % magic
    handle.seek(position + 16 + shift)
    return handle.read(length)


def jpeg_chunkmap(file):
    """Retrieve chunk offsets and shape from old jpeg format"""
    with ensure_handle(file) as f:
        f.seek(0)
        assert f.read(4) == b"\x00\x00\x00\x0c", "Not a JPEG image!"
        size = f.seek(0, 2)
        f.seek(0)

        vs = []
        x = y = x = type_ = None

        while True:
            pos = f.tell()
            length = int.from_bytes(f.read(4), "big")
            box = f.read(4)
            if box == b"jp2c":
                vs.append(f.tell())
            elif box == b"jp2h":
                f.seek(4, 1)
                if f.read(4) == b"ihdr":
                    y, x, c, t = big_iihi.unpack(f.read(14))
                    type_ = np.uint16 if t in (252117248, 252116992) else np.uint8
                continue

            nextPos = pos + length
            if nextPos < 0 or nextPos >= size or length < 8:
                break
            f.seek(length - 8, 1)  # skip bytes

    return vs, (c, y, x, type_)
