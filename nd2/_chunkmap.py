import mmap
import re
import struct

import numpy as np

ImageSeqPtrn = re.compile(br"ImageDataSeq\|(\d+)!?")

CHUNK_MAGIC = 0x0ABECEDA
CHUNK_MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"


def read_chunkmap(file):
    "read the map of the chunks at the end of the file"
    _handle = file if hasattr(file, "fileno") else open(file, "rb")
    _handle.seek(-40, 2)
    mapptr = struct.unpack("32sQ", _handle.read(40))
    assert mapptr[0] == CHUNK_MAP_SIGNATURE

    data = read_chunk(_handle, mapptr[1:])
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
    if not hasattr(file, "fileno"):
        _handle.close()
    return meta, images


def read_chunk(handle, chunk):
    handle.seek(chunk[0])
    h = struct.unpack("IIQ", handle.read(16))
    assert h[0] == CHUNK_MAGIC, "invalid magic %x" % h[0]
    handle.seek(chunk[0] + 16 + h[1])
    return handle.read(h[2])


def good_and_bad_frames(file):
    _handle = file if hasattr(file, "fileno") else open(file, "rb")
    mem = mmap.mmap(_handle.fileno(), 0, access=mmap.ACCESS_READ)
    images = read_chunkmap(_handle)[1]
    return (
        images,
        tuple(
            f for f, pos in images.items() if _sfp(pos, np.uint32, mem) != CHUNK_MAGIC
        ),
    )


def safe_chunk(self, index):
    a = self.attributes()
    bypc = a.bitsPerComponentInMemory // 8
    stride = a.widthBytes - (bypc * a.widthPx * a.componentCount)

    with open(self.path, "rb") as handle:
        mem = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        bpos, _ = chunk_position(self._chunk_pos[index], mem)

        kwargs = dict(
            shape=(a.heightPx, a.widthPx, a.componentCount),
            dtype=self.dtype,
            buffer=mem,
            offset=bpos + 8,
        )
        if stride != 0:
            kwargs["strides"] = (
                stride + a.widthPx * bypc * a.componentCount,
                a.componentCount * bypc,
                bypc,
            )


def chunk_position(offset, mem):
    magic, shift = _dfp(offset, np.uint32, mem, count=2)
    if magic != CHUNK_MAGIC:
        raise RuntimeError("Error")
    real_bpos = offset + 16 + int(shift)
    thelen = _sfp(offset + 8, np.uint64, mem)
    return real_bpos, real_bpos + thelen


def _sfp(offset, dtype, mem):
    return np.ndarray((1,), dtype, mem, offset=offset).item()


def _dfp(offset, dtype, mem, count=1):
    if offset < 0:
        print("y")
        offset += len(mem)
    return np.ndarray((count,), dtype, mem, offset=offset)


def jpeg_chunkmap(file):
    """Retrieve chunk offsets and shape from old jpeg format"""
    f = file if hasattr(file, "fileno") else open(file, "rb")
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
