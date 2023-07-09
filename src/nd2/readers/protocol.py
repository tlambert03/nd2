from __future__ import annotations

import abc
import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from nd2._parse._chunk_decode import get_version  # FIXME

if TYPE_CHECKING:
    from io import BufferedReader

    import numpy as np
    from typing_extensions import Literal

    from nd2._binary import BinaryLayers
    from nd2.structures import (
        ROI,
        Attributes,
        ExpLoop,
        FrameMetadata,
        Metadata,
        TextInfo,
    )

    ChunkMap = dict[bytes, Sequence[int]]


class ND2Reader(abc.ABC):
    """Abstract Base class for ND2 file readers."""

    HEADER_MAGIC: bytes

    @classmethod
    def create(
        cls,
        path: str,
        error_radius: int | None = None,
    ) -> ND2Reader:
        """Create an ND2Reader for the given path, using the appropriate subclass."""
        from nd2.readers import LegacyReader, ModernReader

        with open(path, "rb") as fh:
            magic_num = fh.read(4)

        for subcls in (ModernReader, LegacyReader):
            if magic_num == subcls.HEADER_MAGIC:
                return subcls(path, error_radius=error_radius)
        raise OSError(
            f"file {path} not recognized as ND2.  First 4 bytes: {magic_num!r}"
        )

    def __init__(self, path: str | Path, error_radius: int | None = None) -> None:
        self._chunkmap: dict | None = None
        self._path: Path = Path(path)
        self._fh: BufferedReader | None = None
        self._mmap: mmap.mmap | None = None
        self._error_radius: int | None = error_radius
        self.open()

    def is_legacy(self) -> bool:
        """Return True if the file is a legacy file."""
        return False

    def open(self) -> None:
        """Open the file handle."""
        if self._fh is None or self._fh.closed:
            self._fh = open(self._path, "rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        """Close the file handle."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self) -> ND2Reader:
        """Context manager enter method."""
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        """Context manager exit method."""
        self.close()

    def version(self) -> tuple[int, int]:
        """Return the file format version as a tuple of ints."""
        return get_version(self._fh or self._path)

    def rois(self) -> list[ROI]:
        """Return ROIs in the file."""
        # not implemented for legacy files
        return []

    def binary_data(self) -> BinaryLayers | None:
        """Return BinaryLayers in the file."""
        raise NotImplementedError("binary_data not available for legacy files")

    @abc.abstractmethod
    def attributes(self) -> Attributes:
        """Return the attributes of the file."""

    @abc.abstractmethod
    def metadata(self) -> Metadata:
        """Return the metadata of the file."""

    @abc.abstractmethod
    def read_frame(self, seq_index: int) -> np.ndarray:
        """Read a single frame at the given index."""

    @abc.abstractmethod
    def frame_metadata(self, seq_index: int) -> FrameMetadata | dict:
        """Load the metadata for a single frame."""

    @abc.abstractmethod
    def text_info(self) -> TextInfo:
        """Return the text info of the file."""

    @abc.abstractmethod
    def experiment(self) -> list[ExpLoop]:
        """Return the experiment loops of the file."""

    @abc.abstractmethod
    def events(
        self, orient: Literal["records", "list", "dict"], null_value: Any
    ) -> list | Mapping:
        """Return events in the file."""

    def unstructured_metadata(
        self,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return unstructured metadata from the file."""
        raise NotImplementedError(
            "unstructured_metadata not available for legacy files"
        )

    @abc.abstractmethod
    def voxel_size(self) -> tuple[float, float, float]:
        """Return tuple of (x, y, z) voxel size in microns."""

    @abc.abstractmethod
    def channel_names(self) -> list[str]:
        """Return list of channel names."""

    def custom_data(self) -> dict:
        """Return all data from CustomData chunks in the file."""
        return {}
