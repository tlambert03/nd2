from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from . import _nd2file

if TYPE_CHECKING:
    import dask.array as da
    import xarray as xr

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
        _shape += [attr.componentCount, attr.heightPx, attr.widthPx or -1]
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
        a = [_AXMAP[c.type] for c in self.coord_info()] + list("CYX")
        return "".join(a)

    def pixel_size(self, channel=0) -> Tuple[float, float, float]:
        return self.metadata().channels[channel].volume.axesCalibration

    def asarray(self) -> np.ndarray:
        arr = np.stack([self.data(i) for i in range(self.seq_count())])
        return arr.reshape(self.shape)

    def to_dask(self) -> "da.Array":
        from dask.array import map_blocks

        *rest, nc, ny, nx = self.shape
        darr = map_blocks(
            self._get_chunk,
            chunks=[(1,) * i for i in rest] + [(nc,), (ny,), (nx,)],
            dtype=self.dtype,
        )
        darr._ctx = self  # XXX: ok?  or will we leak refs
        return darr

    def _get_chunk(self, block_id):
        # primarily for to_dask
        if isinstance(block_id, np.ndarray):
            return
        idx = self.seq_index_from_coords(block_id[:-3])
        if idx == -1:
            if any(block_id):
                raise ValueError(f"Cannot get chunk {block_id} for single frame image.")
            idx = 0
        return self.data(idx)[(np.newaxis,) * self.coord_size()]

    def to_xarray(self, delayed: bool = True) -> "xr.DataArray":
        import xarray as xr

        return xr.DataArray(
            self.to_dask() if delayed else self.asarray(),
            dims=list(self.axes),
            coords=self._expand_coords(),
            attrs={
                "metadata": {
                    "metadata": self.metadata(),
                    "experiment": self.experiment(),
                    "attributes": self.attributes(),
                    "text_info": self.text_info(),
                }
            },
        )

    def _expand_coords(self) -> dict:
        attrs = self.attributes()
        dx, dy, dz = self.metadata().channels[0].volume.axesCalibration
        coords = {
            "C": [c.channel.name for c in self.metadata().channels],
            "Y": np.arange(attrs.heightPx) * dy,
            "X": np.arange(attrs.widthPx) * dx,
        }
        for c in self.experiment():
            if c.type == "ZStackLoop":
                coords[_AXMAP["ZStackLoop"]] = np.arange(c.count) * c.parameters.stepUm
            elif c.type == "TimeLoop":
                coords[_AXMAP["TimeLoop"]] = np.arange(c.count) * c.parameters.periodMs
            elif c.type == "NETimeLoop":
                pers = [np.arange(p.count) * p.periodMs for p in c.parameters.periods]
                coords[_AXMAP["NETimeLoop"]] = np.hstack(pers)
            elif c.type == "XYPosLoop":
                coords[_AXMAP["XYPosLoop"]] = [
                    p.name or repr(tuple(p.stagePositionUm))
                    for p in c.parameters.points
                ]
        return coords

    def __repr__(self):
        details = f" {self.dtype}: {self.shape!r}" if self.is_open() else " (closed)"
        return f"<ND2File at {hex(id(self))}: {Path(self.path).name!r}{details}>"

    def __enter__(self) -> "ND2File":
        # just for the type hint
        return super().__enter__()  # type: ignore


def imread(file: str = None) -> np.ndarray:
    with ND2File(str(file)) as nd2:
        arr = nd2.asarray()
    return arr
