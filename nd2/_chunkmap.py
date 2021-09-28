import io
import mmap
import re
import struct
from contextlib import contextmanager
from typing import Iterator, Union

import numpy as np

ImageSeqPtrn = re.compile(br"ImageDataSeq\|(\d+)!?")

CHUNK_MAGIC = 0x0ABECEDA
CHUNK_MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"


@contextmanager
def ensure_handle(obj: Union[str, io.BytesIO], mode="rb") -> Iterator[io.BytesIO]:
    _handle = obj if isinstance(obj, io.BytesIO) else open(obj, mode)
    try:
        yield _handle
    finally:
        if not hasattr(obj, "fileno"):
            _handle.close()


def read_chunkmap(file):
    "read the map of the chunks at the end of the file"
    with ensure_handle(file, "rb") as fh:
        fh.seek(-40, 2)
        name, chunk = struct.unpack("32sQ", fh.read(40))
        assert name == CHUNK_MAP_SIGNATURE

        data = read_chunk(fh, chunk)
        pos = 0
        images = {}
        meta = {}
        while True:
            p = data.index(b"!", pos) + 1
            chunk = data[p : p + 16]  # noqa
            try:
                result = struct.unpack("QQ", chunk)
            except struct.error:
                break

            name = data[pos:p].rstrip(b"!")
            if name.startswith(b"ImageDataSeq|"):
                images[int(name[13:])] = result[0]
            else:
                meta[name.decode("ascii")] = result

            # abort if we found the magic end
            if name == CHUNK_MAP_SIGNATURE:
                break
            pos = p + 16
    return meta, images


def read_chunk(handle: io.BytesIO, position: int):
    handle.seek(position)
    magic, shift, length = struct.unpack("IIQ", handle.read(16))
    assert magic == CHUNK_MAGIC, "invalid magic %x" % magic
    handle.seek(position + 16 + shift)
    return handle.read(length)


def good_and_bad_frames(file):
    with ensure_handle(file) as fh:
        mem = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        images = read_chunkmap(fh)[1]
        return (
            images,
            tuple(
                f
                for f, pos in images.items()
                if _sfp(pos, np.uint32, mem) != CHUNK_MAGIC
            ),
        )


def _sfp(offset, dtype, mem):
    return np.ndarray((1,), dtype, mem, offset=offset).item()


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
                    y, x, c, t = struct.unpack(">iihi", f.read(14))
                    type_ = np.uint16 if t in (252117248, 252116992) else np.uint8
                continue

            nextPos = pos + length
            if nextPos < 0 or nextPos >= size or length < 8:
                break
            f.seek(length - 8, 1)  # skip bytes

    return vs, (c, y, x, type_)
