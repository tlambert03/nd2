"""Utilities for binary layers in ND2 files."""
from __future__ import annotations

import io
import struct
import warnings
from typing import TYPE_CHECKING, Iterator, NamedTuple, Sequence, cast, overload

import numpy as np

if TYPE_CHECKING:
    from ._sdk.latest import ND2Reader as LatestSDKReader
    from .nd2file import ND2File

I7 = struct.Struct("<" + "I" * 7)
I9 = struct.Struct("<" + "I" * 9)
I2 = struct.Struct("<" + "I" * 2)


class BinaryLayer(NamedTuple):
    """Wrapper for data from a single binary layer in an ND2 file.

    `data` will have length of num_sequences, with `None` for any frames
    that lack binary data.

    Parameters
    ----------
    data : list of numpy.ndarray or None
        The data for each frame. If a frame has no binary data, the value
        will be None.  Data will have the same length as the number of sequences
        in the file.
    name: str
        The name of the binary layer.
    comp_name: str
        The name of the associated component, if Any.
    comp_order: int
        The order of the associated component, if Any.
    color: int
        The color of the binary layer.
    color_mode: int
        The color mode of the binary layer.  I believe this is related to how colors
        are chosen in NIS-Elements software.  Where "0" is direct color (i.e. use,
        the color value), "8" is color by 3D ... and I'm not sure about the rest :)
    state: int
        The state of the binary layer. (meaning still unclear)
    file_tag: str
        The key for the binary layer in the CustomData metadata,
        e.g. `RleZipBinarySequence_1_v1`
    layer_id: int
        The ID of the binary layer.
    coordinate_shape: tuple of int
        The shape of the coordinates for the associated nd2 file.  This is used
        to reshape the data into a 3D array in `asarray`.
    """

    data: list[np.ndarray | None]
    name: str
    comp_name: str
    comp_order: int
    color: int
    color_mode: int
    state: int
    file_tag: str
    layer_id: int
    coordinate_shape: tuple[int, ...]

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Shape (Y, X) of each mask in `data`."""
        return next((s.shape for s in self.data if s is not None), (0, 0))

    def __array__(self) -> np.ndarray:
        """Return the data as a numpy array."""
        ary = self.asarray()
        return ary if ary is not None else np.ndarray([])

    def asarray(self) -> np.ndarray | None:
        """Stack all the frames into a single array.

        If there are no frames, returns None.
        """
        frame_shape = self.frame_shape
        if frame_shape == (0, 0):
            return None

        # TODO: this is a bit of a hack (takes up memory), but it works for now
        # could do something with dask
        d = [
            i if i is not None else np.zeros(frame_shape, dtype="uint16")
            for i in self.data
        ]
        return np.stack(d).reshape(self.coordinate_shape + frame_shape)

    def __repr__(self) -> str:
        """Return a nicely formatted string."""
        field_names = (f for f in self._fields if f != "data")
        repr_fmt = "(" + ", ".join(f"{name}=%r" for name in field_names) + ")"
        return self.__class__.__name__ + repr_fmt % self[1:]


class BinaryLayers(Sequence[BinaryLayer]):
    """Sequence of Binary Layers found in an ND2 file.

    This object is a sequence of `BinaryLayer` objects, one for each binary layer in the
    file.  Each layer has a `name` attribute, and a `data` attribute that is list of
    numpy arrays - one for each frame in the experiment - or None if the layer was not
    present in that frame.

    The wrapper can be cast to a numpy array (with `BinaryLayers.asarray()` or
    np.asarray(BinaryLayers)) to stack all the layers into a single array.  The output
    array will have shape (n_layers, *coord_shape, *frame_shape).
    """

    def __init__(self, data: list[BinaryLayer]) -> None:
        self._data = data

    @overload
    def __getitem__(self, key: int) -> BinaryLayer:
        ...

    @overload
    def __getitem__(self, key: slice) -> list[BinaryLayer]:
        ...

    def __getitem__(self, key: int | slice) -> BinaryLayer | list[BinaryLayer]:
        return self._data[key]

    def __iter__(self) -> Iterator[BinaryLayer]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} with {len(self)} layers>"

    def __array__(self) -> np.ndarray:
        """Compatibility with np.asarray(BinaryLayers)."""
        return self.asarray()

    def asarray(self) -> np.ndarray:
        """Stack all the layers/frames into a single array.

        The output array will have shape (n_layers, *coord_shape, *frame_shape).
        """
        out = []
        for bin_layer in self._data:
            d = bin_layer.asarray()
            if d is not None:
                out.append(d)
        return np.stack(out)

    @classmethod
    def from_nd2file(cls, nd2file: ND2File) -> BinaryLayers | None:
        """Extract binary layers from an ND2 file."""
        if nd2file.is_legacy:
            warnings.warn(
                "`binary_data` is not supported for legacy ND2 files", UserWarning
            )
            return None
        rdr = cast("LatestSDKReader", nd2file._rdr)

        binary_meta = nd2file.custom_data.get("BinaryMetadata_v1")
        if binary_meta is None:
            return None
        try:
            items: list[dict] = binary_meta["BinaryMetadata_v1"]["BinaryItem"]
        except KeyError:
            warnings.warn(
                "Could not find 'BinaryMetadata_v1->BinaryItem' tag, please open an "
                "issue with this file at https://github.com/tlambert03/nd2/issues/new",
            )
            return None
        if isinstance(items, dict):
            items = [items]

        binseqs = sorted(x for x in rdr._meta_map if "RleZipBinarySequence" in x)
        mask_items = []
        for item in items:
            key = item["FileTag"]
            _masks: list[np.ndarray | None] = []
            for bs in binseqs:
                if key in bs:
                    data = rdr._get_meta_chunk(bs)[4:]
                    _masks.append(_decode_binary_mask(data) if data else None)
            mask_items.append(
                BinaryLayer(
                    data=_masks,
                    name=item["Name"],
                    comp_name=item["CompName"],
                    comp_order=item["CompOrder"],
                    color_mode=item["ColorMode"],
                    state=item["State"],
                    color=item["Color"],
                    file_tag=key,
                    layer_id=item["BinLayerID"],
                    coordinate_shape=nd2file._coord_shape,
                )
            )

        return cls(mask_items)


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
                output[row, col : col + n] = roi_id

    return output
