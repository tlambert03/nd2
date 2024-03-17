from __future__ import annotations

import abc
import mmap
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

from nd2._parse._chunk_decode import get_version

if TYPE_CHECKING:
    from typing import Any, ContextManager, Literal, Mapping, Sequence

    import numpy as np

    from nd2._binary import BinaryLayers
    from nd2._util import FileOrBinaryIO
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
        path: FileOrBinaryIO,
        error_radius: int | None = None,
    ) -> ND2Reader:
        """Create an ND2Reader for the given path, using the appropriate subclass.

        Parameters
        ----------
        path : str
            Path to the ND2 file.
        error_radius : int, optional
            If b"ND2 FILEMAP SIGNATURE NAME 0001!" is not found at expected location and
            `error_radius` is not None, then an area of +/- `error_radius` bytes will be
            searched for the signature.
        """
        from nd2.readers import LegacyReader, ModernReader

        if hasattr(path, "read"):
            path = cast(BinaryIO, path)
            if "b" not in path.mode:
                raise ValueError(
                    "File handles passed to ND2File must be in binary mode"
                )
            ctx: ContextManager[BinaryIO] = nullcontext(path)
        else:
            path = Path(path).expanduser().resolve()
            ctx = open(path, "rb")

        with ctx as fh:
            fname = fh.name
            fh.seek(0)
            magic_num = fh.read(4)

        for subcls in (ModernReader, LegacyReader):
            if magic_num == subcls.HEADER_MAGIC:
                return subcls(path, error_radius=error_radius)
        raise OSError(
            f"file {fname} not recognized as ND2.  First 4 bytes: {magic_num!r}"
        )

    def __init__(self, path: FileOrBinaryIO, error_radius: int | None = None) -> None:
        self._chunkmap: dict | None = None

        self._mmap: mmap.mmap | None = None
        if hasattr(path, "read"):
            self._fh: BinaryIO | None = cast("BinaryIO", path)
            self._was_open = not self._fh.closed
            self._path: Path = Path(self._fh.name)
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self._was_open = False
            self._path = Path(path)
            self._fh = None
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

    @property
    def _closed(self) -> bool:
        return self._fh is None or self._fh.closed

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
        warnings.warn("ROI extraction not implemented for legacy files", stacklevel=2)
        return []

    def binary_data(self) -> BinaryLayers | None:
        """Return BinaryLayers in the file."""
        warnings.warn("binary_data not implemented for legacy files", stacklevel=2)
        return None

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

    def custom_data(self) -> dict:
        """Return all data from CustomData chunks in the file."""
        warnings.warn("CustomData is not relevant for legacy files", stacklevel=2)
        return {}
