try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
__all__ = [
    "AXIS",
    "BinaryLayer",
    "BinaryLayers",
    "imread",
    "is_supported_file",
    "is_legacy",
    "ND2File",
    "read_chunkmap",
    "rescue_nd2",
    "structures",
]


from . import structures
from ._binary import BinaryLayer, BinaryLayers
from ._chunkmap import read_chunkmap, rescue_nd2
from ._util import AXIS, is_legacy, is_supported_file
from .nd2file import ND2File, imread
