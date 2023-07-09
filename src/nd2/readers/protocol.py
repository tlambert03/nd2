from __future__ import annotations

import abc
import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from nd2._parse._chunk_decode import get_version  # FIXME

if TYPE_CHECKING:
    from io import BufferedReader

    import numpy as np
    from typing_extensions import Literal

    from nd2._binary import BinaryLayers
    from nd2._parse._chunk_decode import ChunkMap
    from nd2.structures import (
        ROI,
        Attributes,
        ExpLoop,
        FrameMetadata,
        Metadata,
        TextInfo,
    )


class ND2Reader(abc.ABC):
    """Abstract Base class for ND2 file readers."""

    HEADER_MAGIC: bytes

    def __init__(self, path: str | Path, error_radius: int | None = None) -> None:
        self._chunkmap: ChunkMap | None = None
        self._path: Path = Path(path)
        self._fh: BufferedReader | None = None
        self._mmap: mmap.mmap | None = None
        self._error_radius: int | None = error_radius
        self.open()

    @property
    def chunkmap(self) -> ChunkMap:
        """Return the chunkmap for the file."""
        if not self._chunkmap:
            if self._fh is None:
                raise OSError("File not open")
            self._chunkmap = self._load_chunkmap(
                self._fh, error_radius=self._error_radius
            )
        return self._chunkmap

    @classmethod
    @abc.abstractmethod
    def _load_chunkmap(cls, fh: BufferedReader, error_radius: int | None) -> ChunkMap:
        ...

    def is_legacy(self) -> bool:
        return False

    def open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._fh = open(self._path, "rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self) -> ND2Reader:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def version(self) -> tuple[int, int]:
        """Return the file format version as a tuple of ints."""
        return get_version(self._fh or self._path)

    @abc.abstractmethod
    def attributes(self) -> Attributes:
        ...

    @abc.abstractmethod
    def metadata(self) -> Metadata:
        ...

    @abc.abstractmethod
    def read_frame(self, seq_index: int | tuple) -> np.ndarray:
        ...

    @abc.abstractmethod
    def frame_metadata(self, seq_index: int) -> FrameMetadata | dict:
        ...

    @abc.abstractmethod
    def text_info(self) -> TextInfo:
        ...

    @abc.abstractmethod
    def experiment(self) -> list[ExpLoop]:
        ...

    def rois(self) -> list[ROI]:
        # not implemented for legacy files
        return []

    @abc.abstractmethod
    def events(
        self, orient: Literal["records", "list", "dict"], null_value: Any
    ) -> dict[str, Sequence[Any]]:
        ...

    def unstructured_metadata(self) -> dict[str, Any]:
        raise NotImplementedError(
            "unstructured_metadata not available for legacy files"
        )

    @abc.abstractmethod
    def voxel_size(self) -> tuple[float, float, float]:
        ...

    @abc.abstractmethod
    def channel_names(self) -> list[str]:
        ...

    def binary_data(self) -> BinaryLayers | None:
        raise NotImplementedError("binary_data not available for legacy files")

    @classmethod
    def create(
        cls,
        path: str,
        error_radius: int | None = None,
    ) -> ND2Reader:
        from nd2.readers import LegacyReader, ModernReader

        with open(path, "rb") as fh:
            magic_num = fh.read(4)

        for subcls in (ModernReader, LegacyReader):
            if magic_num == subcls.HEADER_MAGIC:
                return subcls(path, error_radius=error_radius)
        raise OSError(
            f"file {path} not recognized as ND2.  First 4 bytes: {magic_num!r}"
        )

    def custom_data(self) -> dict:
        return {}
