from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nd2 import ND2File, imread, structures

DATA = Path(__file__).parent / "data"


def test_metadata_extraction(new_nd2):
    with ND2File(new_nd2) as nd:
        assert nd.path == str(new_nd2)
        assert nd.is_open()

        assert isinstance(nd._rdr._seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd._coord_info, list)
        assert all(isinstance(x, structures.Coordinate) for x in nd._coord_info)

        assert isinstance(nd.custom_data, dict)

        n_coords = nd._rdr._coord_size()
        assert isinstance(n_coords, int)
        if n_coords:
            assert isinstance(nd._rdr._seq_index_from_coords([0] * n_coords), int)
            zcoord = nd._rdr._coords_from_seq_index(0)
            assert isinstance(zcoord, tuple)
            assert len(zcoord) == n_coords

    assert not nd.is_open()


def test_read_safety(new_nd2):
    with ND2File(new_nd2) as nd:
        sz = np.prod([v for k, v in nd._rdr.sizes().items() if k not in "YXCc"])
        for i in range(int(sz)):
            nd._image_from_mmap(i)


@pytest.mark.skip
def test_metadata_extraction_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        assert nd.path == str(old_nd2)
        assert nd.is_open()

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
        assert isinstance(nd._coord_info, list)
        assert all(isinstance(x, structures.Coordinate) for x in nd._coord_info)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)

    assert not nd.is_open()


def test_get_data(new_nd2):
    with ND2File(new_nd2) as nd:
        assert isinstance(nd._image_from_sdk(0), np.ndarray)
        assert isinstance(nd._image_from_mmap(0), np.ndarray)
        assert isinstance(nd.to_dask(), da.Array)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        result = xarr[nd._rdr._coords_from_seq_index(0)].compute()
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, np.ndarray)


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


def test_data():
    with ND2File(str(DATA / "jonas_header_test1.nd2")) as nd:
        assert nd._rdr._coord_size() == 2
        assert nd._rdr._seq_index_from_coords([0, 1]) == 1
        assert nd._rdr._coords_from_seq_index(0) == (0, 0)

        sdk_data = nd._image_from_sdk(0)
        mmep_data = nd._image_from_mmap(0)
        assert isinstance(sdk_data, np.ndarray)
        assert isinstance(mmep_data, np.ndarray)
        np.testing.assert_array_equal(sdk_data, mmep_data)
        assert sdk_data.shape == (520, 696, 2)
        np.testing.assert_array_equal(sdk_data[0, :3, 0], [200, 199, 213])
        np.testing.assert_array_equal(sdk_data[0, :3, 1], [204, 205, 199])

        with pytest.raises(IndexError):
            nd._image_from_sdk(23)

        with pytest.raises(IndexError):
            nd._get_frame(23)

        with pytest.raises(IndexError):
            nd._image_from_mmap(23)


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
