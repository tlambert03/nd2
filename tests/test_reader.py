from pathlib import Path
from typing import List

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nd2 import ND2File, _nd2file, imread, structures
from nd2._util import is_new_format

DATA = Path(__file__).parent / "data"
NEW_FORMATS: List[Path] = []
OLD_FORMATS: List[Path] = []
MAX_FILES = 45
ND2 = sorted(DATA.glob("*.nd2"), key=lambda x: x.stat().st_size)[:MAX_FILES]
for x in ND2:
    lst = NEW_FORMATS if is_new_format(str(x)) else OLD_FORMATS
    lst.append(x)


@pytest.mark.parametrize("fname", NEW_FORMATS, ids=lambda x: x.name)
def test_metadata_extraction(fname):
    with ND2File(fname) as nd:
        assert nd.path == str(fname)
        assert nd.is_open()

        assert isinstance(nd._rdr.seq_count(), int)
        # assert isinstance(nd.image_info(), structures.ImageInfo)
        # assert isinstance(nd.image_info(0), structures.ImageInfo)
        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata(), (structures.Metadata, dict))
        assert isinstance(nd.metadata(frame=0), (structures.FrameMetadata, dict))
        assert isinstance(nd.metadata(format=False), dict)
        assert isinstance(nd.metadata(frame=0, format=False), dict)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd._coord_info, list)
        assert all(isinstance(x, structures.Coordinate) for x in nd._coord_info)

        assert isinstance(nd.custom_data, dict)

        n_coords = nd._rdr.coord_size()
        assert isinstance(n_coords, int)
        if n_coords:
            assert isinstance(nd._rdr.seq_index_from_coords([0] * n_coords), int)
            zcoord = nd._rdr.coords_from_seq_index(0)
            assert isinstance(zcoord, tuple)
            assert len(zcoord) == n_coords

    assert not nd.is_open()


@pytest.mark.skip
@pytest.mark.parametrize("fname", OLD_FORMATS, ids=lambda x: x.name)
def test_metadata_extraction_legacy(fname):
    with ND2File(fname) as nd:
        assert nd.path == str(fname)
        assert nd.is_open()

        assert isinstance(nd._rdr.seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        n_coords = nd._rdr.coord_size()
        assert isinstance(n_coords, int)
        if n_coords:
            assert isinstance(nd._rdr.seq_index_from_coords([0] * n_coords), int)
            # FIXME: currently causing strange intermittent segfault
            # zcoord = nd._rdr.coords_from_seq_index(0)
            # assert isinstance(zcoord, tuple)
            # assert len(zcoord) == n_coords

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata(), (structures.Metadata, dict))
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd._coord_info, list)
        assert all(isinstance(x, structures.Coordinate) for x in nd._coord_info)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)

    assert not nd.is_open()
    # doesn't work with new SDK
    with pytest.raises(OSError):
        _nd2file.ND2Reader(str(fname))


@pytest.mark.parametrize("fname", NEW_FORMATS, ids=lambda x: x.name)
def test_get_data(fname):
    if "divisionByZero" in str(fname):
        pytest.skip()
    with ND2File(fname) as nd:
        assert isinstance(nd._sdk_data(0), np.ndarray)
        assert isinstance(nd._mmap_chunk(0), np.ndarray)
        assert isinstance(nd.to_dask(), da.Array)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        result = xarr[nd._rdr.coords_from_seq_index(0)].compute()
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, np.ndarray)


@pytest.mark.skip
@pytest.mark.parametrize("fname", OLD_FORMATS, ids=lambda x: x.name)
def test_get_data_legacy(fname):
    if "divisionByZero" in str(fname):
        pytest.skip()
    with ND2File(fname) as nd:
        assert isinstance(nd._sdk_data(0), np.ndarray)
        assert isinstance(nd._mmap_chunk(0), np.ndarray)
        assert isinstance(nd.to_dask(), da.Array)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        result = xarr[nd._rdr.coords_from_seq_index(0)].compute()
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, np.ndarray)


def test_data():
    with ND2File(str(DATA / "jonas_header_test1.nd2")) as nd:
        assert nd._rdr.coord_size() == 2
        assert nd._rdr.seq_index_from_coords([0, 1]) == 1
        assert nd._rdr.coords_from_seq_index(0) == (0, 0)

        sdk_data = nd._sdk_data(0)
        mmep_data = nd._mmap_chunk(0)
        assert isinstance(sdk_data, np.ndarray)
        assert isinstance(mmep_data, np.ndarray)
        np.testing.assert_array_equal(sdk_data, mmep_data)
        assert sdk_data.shape == (520, 696, 2)
        np.testing.assert_array_equal(sdk_data[0, :3, 0], [200, 199, 213])
        np.testing.assert_array_equal(sdk_data[0, :3, 1], [204, 205, 199])

        with pytest.raises(IndexError):
            nd._sdk_data(23)

        with pytest.raises(IndexError):
            nd._get_frame(23)

        with pytest.raises(IndexError):
            nd._mmap_chunk(23)


def test_missing():
    with pytest.raises(OSError):
        ND2File("asdfs")


def test_imread():
    d = imread(str(DATA / "jonas_header_test2.nd2"))
    assert isinstance(d, np.ndarray)
    assert d.shape == (4, 5, 520, 696)
