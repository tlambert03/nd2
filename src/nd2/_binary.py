"""Utilities for binary layers in ND2 files."""
import io
import struct
import warnings
from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Tuple

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import SupportsIndex

I7 = struct.Struct("<" + "I" * 7)
I9 = struct.Struct("<" + "I" * 9)
I2 = struct.Struct("<" + "I" * 2)


class BinaryData(NamedTuple):
    data: List[np.ndarray | None]
    name: str
    comp_name: str
    comp_order: int
    color: int
    color_mode: int
    state: int
    file_tag: str
    layer_id: int

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        """Shape of the frame."""
        return next((s.shape for s in self.data if s is not None), (0, 0))

    def asarray(self) -> np.ndarray | None:
        import numpy as np

        frame_shape = self.frame_shape
        if frame_shape == (0, 0):
            return None

        # TODO: this is a bit of a hack (takes up memory), but it works for now
        # could do something with dask
        return np.stack(
            [
                i if i is not None else np.zeros(frame_shape, dtype="uint16")
                for i in self.data
            ]
        )

    def __repr__(self) -> str:
        """Return a nicely formatted string"""
        field_names = (f for f in self._fields if f != "data")
        repr_fmt = "(" + ", ".join(f"{name}=%r" for name in field_names) + ")"
        return self.__class__.__name__ + repr_fmt % self[1:]


class BinaryWrapper:
    def __init__(self, data: list[BinaryData], shape: Tuple[int, ...]) -> None:
        self._data = data
        self._shape = shape

    def __getitem__(self, key: SupportsIndex) -> BinaryData:
        return self._data[key]

    def __iter__(self) -> Iterator[BinaryData]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} layers)"

    def __array__(self) -> np.ndarray:
        return self.asarray()

    def asarray(self) -> np.ndarray:
        import numpy as np

        out = []
        for bin_layer in self._data:
            d = bin_layer.asarray()
            if d is not None:
                out.append(d)
        _out = np.stack(out)
        return _out.reshape((-1, *self._shape, *_out.shape[-2:]))


def _unpack(stream: io.BufferedIOBase, strct: struct.Struct):
    return strct.unpack(stream.read(strct.size))


def _decode_binary_mask(data: bytes, dtype="uint16") -> np.ndarray:
    # this receives data as would be extracted from a
    # `CustomDataSeq|RleZipBinarySequence...` section in the metadata
    # data = f._rdr._get_meta_chunk('CustomDataSeq|RleZipBinarySequence_1_v1|0')[:4]

    # NOTE it is up to ND2File to strip the first 4 bytes... and not call this if there
    # is no data (i.e. if the chunk is just '\x00')
    import zlib

    decomp = zlib.decompress(data)
    stream = io.BytesIO(decomp)

    # still not sure what _q is
    # tot_bytes should be length of the stream remaining after this
    (v, ncols, nrows, nmasks, tot_bytes, _q, _zero) = _unpack(stream, I7)
    if v != 3:
        warnings.warn(
            f"Expected first byte to be 3 but got {v}. "
            "Please submit this file :) https://github.com/tlambert03/nd2/issues/."
        )

    output = np.zeros((nrows, ncols), dtype=dtype)
    for _m in range(nmasks):
        # (1,     1,  0, 15, 11,       412,      12, 396, 0)
        (roi_id, c0, r0, c1, r1, roi_bytes, maskrows, _y, _zero) = _unpack(stream, I9)
        for _r in range(maskrows):
            (row, nruns) = _unpack(stream, I2)
            for _s in range(nruns):
                (col, n) = _unpack(stream, I2)
                output[row, col : col + n] = roi_id  # noqa: E203

    return output
