from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable

import nd2
import zarr
import zarr.core
import zarr.storage
from zarr.util import json_dumps

AX_TYPES = {"T": "time", "C": "channel", "Z": "space", "X": "space", "Y": "space"}
AX_UNITS = {"time": "millisecond", "space": "micrometer"}


def create_meta_store(path: Path, tilesize: int | None) -> dict[str, bytes]:
    """Creates a dict containing the zarr metadata for an nd2 image."""
    store: dict[str, bytes] = {}
    with nd2.ND2File(path) as f:
        axes = [
            {"name": ax, "type": (t := AX_TYPES.get(ax)), "unit": AX_UNITS.get(t, "")}
            for ax in f.sizes
        ]
        scale = [1] * len(axes)
        shape = f.shape
        chunks = [1] * len(axes)
        chunks[-2:] = f.shape[-2:]

    root_attrs = {
        "multiscales": [
            {
                "version": "0.5-dev",
                "name": Path(path).name,
                "axes": axes,
                "datasets": [
                    {
                        "path": "",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale}
                        ],
                    }
                ],
            }
        ]
    }
    zarr.storage.init_group(store)
    store[zarr.storage.attrs_key] = json_dumps(root_attrs)

    path = "0"
    zarr.storage.init_array(
        store,
        path=path,
        shape=shape,
        chunks=chunks,
        dtype="<u2",
        compressor=None,
    )
    # xarray support
    key = f"{path}/{zarr.storage.attrs_key}"
    store[key] = json_dumps({"_ARRAY_DIMENSIONS": [ax["name"] for ax in axes]})
    return store


class ND2Store(zarr.storage.Store):
    """Zarr store for nd2 files."""

    def __init__(self, path: str | Path, tilesize: int | None = None):
        super().__init__()
        self._path = Path(path)
        self._store = create_meta_store(self._path, tilesize)

    def __getitem__(self, key: str) -> bytes:
        """Retrieves a chunk from the store."""
        if key in self._store:
            # key is for metadata
            return self._store[key]

        try:
            level, ckey = key.split("/")
            chunk_idx = tuple(map(int, ckey.split(".")))
        except ValueError as e:
            raise KeyError(key) from e
        if level != "0":
            raise KeyError(key)

        # key should now be a path to an array chunk
        # e.g '0/0.0.1.0.0' -> '<level>/<chunk_key>'
        with nd2.ND2File(self._path) as f:
            dd = f._dask_block(copy=True, block_id=chunk_idx)
            with contextlib.suppress(ValueError):
                channel_idx = list(f.sizes).index("C")
                dd = dd.take(chunk_idx[channel_idx], axis=channel_idx)
            return dd.tobytes()  # tobytes doesn't appear to be strictly necessary

    def __iter__(self) -> Iterable[str]:
        """Iterates over the keys in the store."""
        yield from self._store

    def __len__(self) -> int:
        """Returns the number of keys in the store."""
        return len(self._store)

    def __setitem__(self, key, val):
        """Not implemented."""
        raise RuntimeError("__setitem__ not implemented")

    def __delitem__(self, key):
        """Not implemented."""
        raise RuntimeError("__delitem__ not implemented")


store = ND2Store("tests/data/10ms_2xbin_100xmag.nd2")
root = zarr.open(store=store, mode="r")
ary = root["0"]
