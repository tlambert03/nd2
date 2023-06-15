import nd2
import pytest


def test_read_using_sdk(single_nd2):
    with pytest.warns(FutureWarning, match="read_using_sdk"):
        f = nd2.ND2File(single_nd2, read_using_sdk=True)
    f.close()


def test_unnest_param(single_nd2):
    with nd2.ND2File(single_nd2) as f:
        with pytest.warns(FutureWarning, match="unnest"):
            f.unstructured_metadata(unnest=True)
