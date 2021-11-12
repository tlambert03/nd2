"""experimental Dask array that opens/closes a resource when computing"""
from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np

if TYPE_CHECKING:
    from typing import Protocol

    # fmt: off
    class CheckableContext(Protocol):
        @property
        def closed(self) -> bool: ...  # noqa: E704
        def __enter__(self): ...  # noqa: E704
        def __exit__(self, *a): ...  # noqa: E704
    # fmt: on


def _copy_doc(method):
    extra: str = getattr(method, "__doc__", None) or ""
    original_method = getattr(da.Array, method.__name__)
    doc = original_method.__doc__ or ""
    if extra:
        doc += extra.rstrip("\n") + "\n\n"

    method.__doc__ = doc
    return method


class ResourceBackedDaskArray(da.Array):
    _file_ctx: CheckableContext

    def __new__(
        cls,
        dask,
        name,
        chunks,
        dtype=None,
        meta=None,
        shape=None,
        *,
        _file_ctx: CheckableContext
    ):
        arr = super().__new__(
            cls, dask, name, chunks, dtype=dtype, meta=meta, shape=shape
        )
        arr._file_ctx = _file_ctx
        return arr

    @classmethod
    def from_array(cls, arr, ctx: CheckableContext) -> ResourceBackedDaskArray:
        """Create an OpeningDaskArray with a checkable context.

        `ctx` must be a context manager that opens/closes some underlying resource (like
        a file), and has a `closed` attribute that returns the current state of the
        resource.  This subclass will take care of opening and closing the resource on
        compute.
        """
        if isinstance(arr, ResourceBackedDaskArray):
            return arr
        _a = arr if isinstance(arr, da.Array) else da.from_array(arr)
        arr = cls(
            _a.dask,
            _a.name,
            _a.chunks,
            dtype=_a.dtype,
            meta=_a._meta,
            shape=_a.shape,
            _file_ctx=ctx,
        )
        return arr

    @_copy_doc
    def compute(self, **kwargs: Any) -> np.ndarray:
        """
        Notes
        -----
        This subclass of da.Array will re-open the underlying file before compute."""
        with self._file_ctx if self._file_ctx.closed else nullcontext():
            return super().compute(**kwargs)

    def __getitem__(self, index):
        # indexing should also return an Opening Array
        return ResourceBackedDaskArray.from_array(super().__getitem__(index), self._file_ctx)

    def __getattribute__(self, name: Any) -> Any:
        # allows methods like `array.mean()` to also return an OpeningDaskArray
        attr = object.__getattribute__(self, name)
        if (
            not name.startswith("_")
            and name not in ResourceBackedDaskArray.__dict__
            and callable(attr)
        ):
            return _ArrayMethodProxy(attr, self._file_ctx)
        return attr

    def __array_function__(self, func, types, args, kwargs):
        # obey NEP18
        types = tuple(da.Array if x is ResourceBackedDaskArray else x for x in types)
        arr = super().__array_function__(func, types, args, kwargs)
        if isinstance(arr, da.Array):
            return ResourceBackedDaskArray.from_array(arr, self._file_ctx)
        return arr


class _ArrayMethodProxy:
    """Wraps method on a dask array and returns a OpeningDaskArray if the result of the
    method is a dask array.  see details in OpeningDaskArray docstring."""

    def __init__(self, method: MethodType, file_ctx: CheckableContext) -> None:
        self.method = method
        self._file_ctx = file_ctx

    def __repr__(self) -> str:
        return repr(self.method)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with self._file_ctx if self._file_ctx.closed else nullcontext():
            result = self.method(*args, **kwds)
        if isinstance(result, da.Array):
            return ResourceBackedDaskArray.from_array(result, self._file_ctx)
        return result
