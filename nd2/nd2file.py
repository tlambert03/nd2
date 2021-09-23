from typing import TYPE_CHECKING, Tuple, Union
from dask.array.rechunk import rechunk
from dask.delayed import delayed
import numpy as np
from numpy.core.shape_base import block
from . import _nd2file
from pathlib import Path


if TYPE_CHECKING:
    import dask.array as da

Index = Union[int, slice]
_AXMAP = {
    "Unknown": "U",
    "TimeLoop": "T",
    "XYPosLoop": "S",
    "ZStackLoop": "Z",
    "NETimeLoop": "T",
}


class ND2File(_nd2file.CND2File):
    @property
    def ndim(self) -> int:
        # TODO: this depends on whether we squeeze or not
        # XXX: also... this needs to agree with shape and axes
        # return self.coord_size() + 2 + int(self.attributes().componentCount > 1)
        return self.coord_size() + 3

    @property
    def shape(self) -> Tuple[int, ...]:
        attr = self.attributes()
        _shape = [c.size for c in self.coord_info()]
        # TODO: widthPx may be None?
        _shape += [attr.componentCount, attr.heightPx, attr.widthPx]
        return tuple(_shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def dtype(self) -> np.dtype:
        attrs = self.attributes()
        b = attrs.bitsPerComponentInMemory // 8
        d = attrs.pixelDataType[0] if attrs.pixelDataType else "u"
        return np.dtype(f"{d}{b}")  # TODO: check is it always uint?

    @property
    def axes(self) -> str:
        a = [_AXMAP[c.type] for c in self.coord_info()] + list("CYZ")
        return "".join(a)

    def asarray(self) -> np.ndarray:
        arr = np.stack([self.data(i) for i in range(self.seq_count())])
        return arr.reshape(self.shape)

    def to_dask(self) -> "da.Array":
        from dask.array import map_blocks

        *rest, nc, ny, nx = self.shape
        return map_blocks(
            self._get_chunk,
            chunks=[(1,) * i for i in rest] + [(nc,), (ny,), (nx,)],
            dtype=self.dtype,
        )

    def _get_chunk(self, block_id):
        # primarily for to_dask
        if not block_id:
            return
        idx = self.seq_index_from_coords(block_id[:-3])
        return self.data(idx)[(np.newaxis,) * self.coord_size()]

    def __repr__(self):
        path = Path(self.path).name
        return f"<ND2File at {hex(id(self))}: {path!r} {self.dtype}: {self.shape!r}>"

    def __enter__(self) -> 'ND2File':
        return super().__enter__()
