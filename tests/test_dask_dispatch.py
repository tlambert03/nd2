import dask.array as da
import numpy as np
import pytest
from nd2 import ND2File
from resource_backed_dask_array import ResourceBackedDaskArray


@pytest.mark.parametrize("leave_open", [True, False])
@pytest.mark.parametrize("wrapper", [True, False])
def test_nd2_dask_closed_wrapper(single_nd2, wrapper, leave_open):
    f = ND2File(single_nd2)
    arr = f.to_dask(wrapper=wrapper)
    if not leave_open:
        f.close()

    is_wrapped = isinstance(arr, ResourceBackedDaskArray)
    assert is_wrapped if wrapper else not is_wrapped
    assert isinstance(arr, da.Array)
    assert isinstance(arr.compute(), np.ndarray)

    if leave_open:
        f.close()


def test_nd2_dask_einsum(single_nd2):
    with ND2File(single_nd2) as f:
        arr = f.to_dask()
    assert isinstance(arr, ResourceBackedDaskArray)
    assert arr.shape == (3, 2, 32, 32)
    reordered_dask = da.einsum("abcd->abcd", arr)
    assert isinstance(reordered_dask[:1, :1, :1, :1].compute(), np.ndarray)


def test_nd2_dask_einsum_via_nep18(single_nd2):
    with ND2File(single_nd2) as f:
        arr = f.to_dask()
    assert isinstance(arr, ResourceBackedDaskArray)
    reordered_nep18 = np.einsum("abcd->abcd", arr)
    assert isinstance(reordered_nep18, ResourceBackedDaskArray)
    assert isinstance(reordered_nep18[:1, :1, :1, :1].compute(), np.ndarray)


def test_synthetic_dask_einsum_via_nep18():
    arr = da.zeros([1000, 1000, 100, 100])
    reordered_nep18 = np.einsum("abcd->abcd", arr)
    assert isinstance(reordered_nep18, da.Array)
    assert isinstance(reordered_nep18[:1, :1, :1, :1].compute(), np.ndarray)


def test_nd2_dask_einsum_via_nep18_small(single_nd2):
    with ND2File(single_nd2) as f:
        arr = f.to_dask()
    assert isinstance(arr, ResourceBackedDaskArray)
    arr = arr[:10, :10, :10, :10]
    assert isinstance(arr, ResourceBackedDaskArray)
    reordered_nep18 = np.einsum("abcd->abcd", arr)
    assert isinstance(reordered_nep18, da.Array)
    assert isinstance(reordered_nep18[:1, :1, :1, :1].compute(), np.ndarray)
