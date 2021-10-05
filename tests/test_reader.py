import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nd2 import ND2File, imread, structures

DATA = Path(__file__).parent / "data"

SDK_MISSES_COORDS = {
    "jonas_100217_OD122_001.nd2",
    "jonas_512c_nikonTest_two.nd2",
    "jonas_512c_cag_p5_simgc_2511_70ms22s_crop.nd2",
    "jonas_2112-2265.nd2",
}


def test_metadata_extraction(new_nd2):
    with ND2File(new_nd2) as nd:
        assert nd.path == str(new_nd2)
        assert not nd.closed

        assert isinstance(nd._rdr._seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.sizes, dict)

        assert isinstance(nd.custom_data, dict)

        n_coords = nd._rdr._coord_size()
        assert isinstance(n_coords, int)
        if n_coords:
            assert isinstance(nd._rdr._seq_index_from_coords([0] * n_coords), int)
            zcoord = nd._rdr._coords_from_seq_index(0)
            assert isinstance(zcoord, tuple)
            assert len(zcoord) == n_coords

    assert nd.closed


def test_read_safety(new_nd2: Path):
    with ND2File(new_nd2) as nd:
        for i in range(nd._frame_count):
            nd._image_from_mmap(i)


def test_dask(new_nd2):
    if new_nd2.stat().st_size > 1_000_000_000 and not os.getenv("CI"):
        pytest.skip("use CI=1 to include big files in dask test")
    with ND2File(new_nd2) as nd:
        dsk = nd.to_dask()
        assert isinstance(dsk, da.Array)
        arr = np.asarray(dsk)
        assert arr.shape == nd.shape
        np.testing.assert_allclose(arr, nd.asarray())


def test_get_frame(new_nd2):
    with ND2File(new_nd2) as nd:
        assert isinstance(nd._image_from_sdk(0), np.ndarray)
        assert isinstance(nd._image_from_mmap(0), np.ndarray)


def test_xarray(new_nd2):
    with ND2File(new_nd2) as nd:
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)


# def test_read_mode(new_nd2):
#     with ND2File(new_nd2, read_mode='mmap') as nd:
#         a1 =  nd.asarray()
#     with ND2File(new_nd2, read_mode='sdk') as nd:
#         a2 =  nd.asarray()
#     np.testing.assert_allclose(a1, a2)


@pytest.mark.skip
def test_metadata_extraction_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        assert nd.path == str(old_nd2)
        assert not nd.closed

        assert isinstance(nd._rdr._seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        n_coords = nd._rdr._coord_size()
        assert isinstance(n_coords, int)
        if n_coords:
            assert isinstance(nd._rdr._seq_index_from_coords([0] * n_coords), int)
            # FIXME: currently causing strange intermittent segfault
            # zcoord = nd._rdr.coords_from_seq_index(0)
            # assert isinstance(zcoord, tuple)
            # assert len(zcoord) == n_coords

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)

    assert nd.closed


@pytest.mark.skip
def test_get_data_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        assert isinstance(nd._image_from_sdk(0), np.ndarray)
        assert isinstance(nd._image_from_mmap(0), np.ndarray)
        assert isinstance(nd.to_dask(), da.Array)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        result = xarr[nd._rdr._coords_from_seq_index(0)].compute()
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, np.ndarray)


def test_missing():
    with pytest.raises(OSError):
        ND2File("asdfs")


def test_imread():
    d = imread(str(DATA / "jonas_header_test2.nd2"))
    assert isinstance(d, np.ndarray)
    assert d.shape == (4, 5, 520, 696)


# import json

# with open(DATA / "shapes.json") as f:
#     BFINO = json.load(f)


# @pytest.mark.parametrize("fname", NEW_FORMATS, ids=lambda x: x.name)
# def test_bioformats_parit(fname: Path):
#     if fname.name in {
#         "dims_rgb_t3p2c2z3x64y64.nd2",
#         "dims_rgb_c2x64y64.nd2",
#         "dims_rgb.nd2",
#     }:
#         pytest.xfail()
#     info = BFINO[fname.name]["shape"]
#     for k, v in list(info.items()):
#         if v == 1:
#             info.pop(k)
#     with ND2File(fname) as nd:
#         d = dict(zip(nd.axes, nd.shape))
#         for k, v in list(d.items()):
#             if v == 1:
#                 d.pop(k)

#         assert d == info
