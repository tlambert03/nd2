try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"

from .nd2file import ND2File, imread

__all__ = ["ND2File", "imread"]
