"""Utilities for binary layers in ND2 files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, overload

import numpy as np


    from ._pysdk._pysdk import ND2Reader as LatestSDKReader
    from .nd2file import ND2File

I7 = struct.Struct("<" + "I" * 7)
I9 = struct.Struct("<" + "I" * 9)
I2 = struct.Struct("<" + "I" * 2)


class BinaryLayer(NamedTuple):
    """Wrapper for data from a single binary layer in an [`nd2.ND2File`][].

    `data` will have length of num_sequences, with `None` for any frames
    that lack binary data.

    Attributes
    ----------
    data : list[numpy.ndarray] | None
        The data for each frame. If a frame has no binary data, the value
        will be None.  Data will have the same length as the number of sequences
        in the file.
    name : str
        The name of the binary layer.
    comp_name : str
        The name of the associated component, if Any.
    comp_order : int
        The order of the associated component, if Any.
    color : int
        The color of the binary layer.
    color_mode : int
        The color mode of the binary layer.  I believe this is related to how colors
        are chosen in NIS-Elements software.  Where "0" is direct color (i.e. use,
        the color value), "8" is color by 3D ... and I'm not sure about the rest :)
    state : int
        The state of the binary layer. (meaning still unclear)
    file_tag : str
        The key for the binary layer in the CustomData metadata,
        e.g. `RleZipBinarySequence_1_v1`
    layer_id : int
        The ID of the binary layer.
    coordinate_shape : tuple[int, ...]
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
        return np.stack(d).reshape(self.coordinate_shape + frame_shape)  # type: ignore


class BinaryLayers(Sequence[BinaryLayer]):
    """Sequence of Binary Layers found in an ND2 file.

    This object is a sequence of `BinaryLayer` objects, one for each binary layer in the
    file.  Each layer has a `name` attribute, and a `data` attribute that is list of
    numpy arrays - one for each frame in the experiment - or None if the layer was not
    present in that frame.

    The wrapper can be cast to a numpy array (with `BinaryLayers.asarray()` or
    `np.asarray(BinaryLayers)`) to stack all the layers into a single array.  The output
    array will have shape `(n_layers, *coord_shape, *frame_shape)`.
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
        """Compatibility with `np.asarray(BinaryLayers)`."""
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
