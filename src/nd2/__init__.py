try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
__all__ = ["ND2File", "imread", "structures"]


from . import structures
from .nd2file import ND2File, imread
