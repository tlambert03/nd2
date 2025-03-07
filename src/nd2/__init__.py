"""nd2: A Python library for reading and writing ND2 files."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # uses optional tifffile dependency
    from .tiff import nd2_to_tiff

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
__all__ = [
    "AXIS",
    "BinaryLayer",
    "BinaryLayers",
    "ND2File",
    "__version__",
    "imread",
    "is_legacy",
    "is_supported_file",
    "nd2_to_tiff",
    "rescue_nd2",
    "structures",
]


from . import structures
from ._binary import BinaryLayer, BinaryLayers
from ._nd2file import ND2File, imread
from ._parse._chunk_decode import rescue_nd2
from ._util import AXIS, is_legacy, is_supported_file


def __getattr__(name: str) -> Any:
    if name == "nd2_to_tiff":
        from .tiff import nd2_to_tiff

        return nd2_to_tiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
