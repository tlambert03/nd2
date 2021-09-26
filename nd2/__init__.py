try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
__all__ = ["ND2File", "imread", "structures", "LegacyND2File"]


from . import structures
from .nd2file import ND2File, imread


def __getattr__(name: str):
    if name == "LegacyND2File":
        from ._nd2file_legacy import CLegacyND2File as LegacyND2File

        return LegacyND2File
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
