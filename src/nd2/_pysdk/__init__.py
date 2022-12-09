from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from io import BufferedReader


def Lim_FileGetAttributes(hFile: BufferedReader) -> str:
    """Get the attributes of a file."""


class LimFile:
    _filename: str
    _fh: BufferedReader | None
    _attributes: dict
    _raw_metadata: dict

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._fh: BufferedReader | None = None
        self._attributes = {}
        self._raw_metadata = {}

    def open(self) -> None:
        self._fh = open(self._filename, "rb")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def attributes(self) -> dict:
        if not self._attributes:
            if self._fh is None:
                raise OSError("File not open")
            self._attributes = Lim_FileGetAttributes(self._fh)
        return self._attributes

    def _cached_raw_metadata(self) -> dict:
        if not self._raw_metadata:
            self._raw_metadata = self._raw_metadata()
        return self._raw_metadata
    