try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"

from typing import TYPE_CHECKING

from ._nd2file import ND2File as ND2File

if TYPE_CHECKING:
    import numpy as np


def imread(file: str = None, sequence: int = 0) -> "np.ndarray":
    with ND2File(str(file)) as nd2:
        return nd2.data(sequence)
