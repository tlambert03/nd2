# this module exists just to deprecate the public module access
from typing import Any


def __getattr__(name: str) -> Any:
    import warnings

    warnings.warn(
        "Importing directly from nd2.readers is deprecated. "
        "Please import objects from nd2 instead. If the object you were "
        "looking for is not available at the top level, please open an issue.",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import _readers

    return getattr(_readers, name)
