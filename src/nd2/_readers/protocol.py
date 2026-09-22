from __future__ import annotations

import abc
import io
import mmap
import threading
import warnings
from contextlib import AbstractContextManager, nullcontext, suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nd2._parse._chunk_decode import get_version
from nd2._util import is_fsspec_url, is_read_seek_binary, open_fsspec_url

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Literal

    import numpy as np

    from nd2._binary import BinaryLayers
    from nd2._util import FileOrBinaryIO, ReadSeekBinary
    from nd2.jobs.types import JobsDict
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
        storage_options: dict | None = None,
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
        storage_options : dict, optional
            Extra kwargs passed to fsspec when opening remote URLs.
        """
        from nd2._readers import LegacyReader, ModernReader

        is_url = is_fsspec_url(path)
        opened_here = False
        reopen: Callable[[], ReadSeekBinary] | None = None
        ctx: AbstractContextManager
        if is_file_handle := is_read_seek_binary(path):
            mode = getattr(path, "mode", "b")
            if isinstance(mode, str) and "b" not in mode:
                raise ValueError(
                    "File handles passed to ND2File must be in binary mode"
                )
            ctx = nullcontext(path)
        elif is_url:
            reopen = partial(
                open_fsspec_url, str(path), storage_options=storage_options
            )
            fh = reopen()
            opened_here = True
            ctx = nullcontext(fh)
        else:
            path = Path(cast("str | Path", path)).expanduser().absolute()
            ctx = open(path, "rb")

        with ctx as fh:
            fname = getattr(fh, "name", "")
            fh.seek(0)
            magic_num = fh.read(4)

        for subcls in (ModernReader, LegacyReader):
            if magic_num == subcls.HEADER_MAGIC:
                # For URL/file-like cases pass the open handle; for local paths
                # pass the Path so the reader can reopen it as needed.
                effective_path = fh if (is_url or is_file_handle) else path
                try:
                    return subcls(
                        effective_path, error_radius=error_radius, reopen=reopen
                    )
                except Exception:
                    if opened_here:
                        fh.close()
                    raise
        if opened_here:
            fh.close()
        raise OSError(
            f"file {fname!r} not recognized as ND2.  First 4 bytes: {magic_num!r}"
        )

    def __init__(
        self,
        obj: FileOrBinaryIO,
        error_radius: int | None = None,
        reopen: Callable[[], ReadSeekBinary] | None = None,
    ) -> None:
        self._chunkmap: dict | None = None
        # guards seek+read pairs on the shared file handle
        self._fh_lock = threading.RLock()
        self._version: tuple[int, int] | None = None

        self._mmap: mmap.mmap | None = None
        self._fh: ReadSeekBinary | None
        self._path: str | Path | None
        if is_read_seek_binary(obj):
            self._fh = obj
            # if `reopen` was provided, we opened the handle and must close it.
            self._was_open = reopen is None and not obj.closed
            fs, fs_path = getattr(obj, "fs", None), getattr(obj, "path", None)
            full_name = getattr(obj, "full_name", None)
            name = getattr(obj, "name", None)
            if isinstance(full_name, str) and is_fsspec_url(full_name):
                # remote fsspec file: can be reopened from its own filesystem
                self._path = full_name
                if reopen is None and fs is not None and fs_path is not None:
                    reopen = partial(fs.open, fs_path, "rb")
            elif isinstance(name, str):
                self._path = Path(name)  # local file, may be reopened by name
            else:
                self._path = None
            self._try_mmap()
        else:
            self._was_open = False
            self._path = Path(cast("str | Path", obj))
            self._fh = None
        self._reopen = reopen
        self._error_radius: int | None = error_radius
        self.open()

    def is_legacy(self) -> bool:
        """Return True if the file is a legacy file."""
        return False

    def open(self) -> None:
        """Open the file handle."""
        if self._fh is None or self._fh.closed:
            if self._reopen is not None:
                self._fh = self._reopen()
                self._try_mmap()
            elif isinstance(self._path, Path):
                self._fh = open(self._path, "rb")
                fh = cast("Any", self._fh)
                self._mmap = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                raise RuntimeError(
                    "Cannot reopen an unnamed file-like object after it has been "
                    "closed. Keep the file open, or pass a path or URL instead."
                )

    def _try_mmap(self) -> None:
        """Memory-map the file handle, if it is a plain local file object.

        Wrappers such as `gzip.GzipFile` proxy `fileno()` to the *underlying* file,
        so mmap would expose the wrong bytes.  Anything that isn't a real file
        object falls back to seek/read.
        """
        self._mmap = None
        if isinstance(self._fh, (io.FileIO, io.BufferedReader)):
            with suppress(Exception):
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
        with self._fh_lock:
            if self._version is None:
                if self._fh is not None:
                    self._version = get_version(self._fh)
                elif self._path is not None:
                    self._version = get_version(self._path)
                else:
                    raise RuntimeError(
                        "Cannot determine version without an open file handle"
                    )
            return self._version

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

    def jobs(self) -> JobsDict | None:
        """Return JOBS metadata if the file was acquired using JOBS, else None."""
        return None
