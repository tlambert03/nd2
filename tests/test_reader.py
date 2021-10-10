import json
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nd2 import ND2File, imread, structures

DATA = Path(__file__).parent / "data"

# SDK_MISSES_COORDS = {
#     "jonas_100217_OD122_001.nd2",
#     "jonas_512c_nikonTest_two.nd2",
#     "jonas_512c_cag_p5_simgc_2511_70ms22s_crop.nd2",
#     "jonas_2112-2265.nd2",
# }


def test_metadata_extraction(new_nd2):
    assert ND2File.is_supported_file(new_nd2)
    with ND2File(new_nd2) as nd:
        assert nd.path == str(new_nd2)
        assert not nd.closed

        # assert isinstance(nd._rdr._seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.sizes, dict)
        assert isinstance(nd.custom_data, dict)
        assert isinstance(nd.shape, tuple)
        assert isinstance(nd.size, int)
        assert isinstance(nd.closed, bool)
        assert isinstance(nd.ndim, int)

    assert nd.closed


def test_read_safety(new_nd2: Path):
    with ND2File(new_nd2) as nd:
        for i in range(nd._frame_count):
            nd._rdr._read_image(i)


def test_dask(new_nd2):
    with ND2File(new_nd2) as nd:
        dsk = nd.to_dask()
        assert isinstance(dsk, da.Array)
        assert dsk.shape == nd.shape
        arr = np.asarray(dsk[(0,) * (len(nd.shape) - 2)])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == nd.shape[-2:]


def test_full_read(new_nd2):
    with ND2File(new_nd2) as nd:
        if (new_nd2.stat().st_size > 500_000_000) and "--runslow" not in sys.argv:
            pytest.skip("use --runslow to test full read")
        delayed_xarray = np.asarray(nd.to_xarray(delayed=True))
        assert delayed_xarray.shape == nd.shape
        np.testing.assert_allclose(delayed_xarray, nd.asarray())


def test_dask_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        dsk = nd.to_dask()
        assert isinstance(dsk, da.Array)
        assert dsk.shape == nd.shape
        arr = np.asarray(dsk[(0,) * (len(nd.shape) - 2)])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == nd.shape[-2:]


def test_full_read_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        if (old_nd2.stat().st_size > 500_000) and "--runslow" not in sys.argv:
            pytest.skip("use --runslow to test full read")
        delayed_xarray = np.asarray(nd.to_xarray(delayed=True))
        assert delayed_xarray.shape == nd.shape
        np.testing.assert_allclose(delayed_xarray, nd.asarray())


def test_xarray(new_nd2):
    with ND2File(new_nd2) as nd:
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        assert isinstance(nd.to_xarray(squeeze=False), xr.DataArray)


def test_xarray_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        assert isinstance(nd.to_xarray(squeeze=False), xr.DataArray)


def test_metadata_extraction_legacy(old_nd2):
    assert ND2File.is_supported_file(old_nd2)
    with ND2File(old_nd2) as nd:
        assert nd.path == str(old_nd2)
        assert not nd.closed

        assert isinstance(nd.attributes, structures.Attributes)

        # # TODO: deal with typing when metadata is completely missing
        # assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)

    assert nd.closed


def test_missing():
    with pytest.raises(FileNotFoundError):
        ND2File("asdfs")


def test_imread():
    d = imread(str(DATA / "jonas_header_test2.nd2"))
    assert isinstance(d, np.ndarray)
    assert d.shape == (4, 5, 520, 696)


@pytest.fixture
def bfshapes():
    with open(DATA / "bf_shapes.json") as f:
        return json.load(f)


def test_bioformats_parity(new_nd2: Path, bfshapes: dict):
    """Testing that match bioformats shapes (or better when bioformats misses it)."""
    if new_nd2.name in {
        "dims_rgb_t3p2c2z3x64y64.nd2",  # bioformats seems to miss the RGB
        "dims_rgb_c2x64y64.nd2",  # bioformats seems to miss the RGB
        "dims_t3y32x32.nd2",  # bioformats misses T
        "jonas_3.nd2",  # bioformats misses Z
        "cluster.nd2",  # bioformats misses both Z and T
    }:
        pytest.xfail()
    if new_nd2.name.startswith("JOBS_"):
        pytest.xfail()  # bioformats misses XY position info in JOBS files
    try:
        bf_info = {k: v for k, v in bfshapes[new_nd2.name]["shape"].items() if v > 1}
    except KeyError:
        pytest.skip(f"{new_nd2.name} not in stats")
    with ND2File(new_nd2) as nd:
        # doing these weird checks/asserts for better error messages
        if len(bf_info) != len(nd.sizes):
            assert bf_info == nd.sizes
        # allowing for off-by-one errors for now
        if bf_info != nd.sizes and any(
            a < b - 1 or a > b + 1 for a, b in _common_entries(bf_info, nd.sizes)
        ):
            assert bf_info == nd.sizes


def _common_entries(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield tuple(d[i] for d in dcts)
