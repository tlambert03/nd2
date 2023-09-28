import json
import pickle
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from nd2 import ND2File, imread
from nd2._parse._chunk_decode import get_version
from nd2._util import AXIS, is_supported_file
from resource_backed_dask_array import ResourceBackedDaskArray

DATA = Path(__file__).parent / "data"


def test_read_safety(new_nd2: Path):
    with ND2File(new_nd2) as nd:
        for i in range(nd._frame_count):
            nd._rdr.read_frame(i)


def test_position(new_nd2: Path):
    """use position to extract a single stage position with asarray."""
    pytest.importorskip("xarray")
    if new_nd2.stat().st_size > 250_000_000:
        pytest.skip("skipping read on big files")
    with ND2File(new_nd2) as nd:
        dx = nd.to_xarray(delayed=True, position=0, squeeze=False)
        nx = nd.to_xarray(delayed=False, position=0, squeeze=False)
        assert dx.sizes[AXIS.POSITION] == 1
        assert nx.sizes[AXIS.POSITION] == 1
        dx = nd.to_xarray(delayed=True, position=0, squeeze=True)
        nx = nd.to_xarray(delayed=False, position=0, squeeze=True)
        assert AXIS.POSITION not in dx.sizes
        assert AXIS.POSITION not in nx.sizes


def test_dask(new_nd2):
    with ND2File(new_nd2) as nd:
        dsk = nd.to_dask()
        assert isinstance(dsk, da.Array)
        assert dsk.shape == nd.shape
        arr: np.ndarray = np.asarray(dsk[(0,) * (len(nd.shape) - 2)])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == nd.shape[-2:]


def test_dask_closed(single_nd2):
    with ND2File(single_nd2) as nd:
        dsk = nd.to_dask()
    assert isinstance(dsk.compute(), np.ndarray)


def test_full_read(new_nd2):
    pytest.importorskip("xarray")
    with ND2File(new_nd2) as nd:
        if new_nd2.stat().st_size > 500_000_000:
            pytest.skip("skipping full read on big files")
        delayed_xarray: np.ndarray = np.asarray(nd.to_xarray(delayed=True))
        assert delayed_xarray.shape == nd.shape
        np.testing.assert_allclose(delayed_xarray, nd.asarray())


def test_dask_legacy(old_nd2):
    pytest.importorskip("imagecodecs")
    with ND2File(old_nd2) as nd:
        dsk = nd.to_dask()
        assert isinstance(dsk, da.Array)
        assert dsk.shape == nd.shape
        arr: np.ndarray = np.asarray(dsk[(0,) * (len(nd.shape) - 2)])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == nd.shape[-2:]


def test_full_read_legacy(old_nd2):
    with ND2File(old_nd2) as nd:
        if (old_nd2.stat().st_size > 500_000) and "--runslow" not in sys.argv:
            pytest.skip("use --runslow to test full read")
        delayed_xarray: np.ndarray = np.asarray(nd.to_xarray(delayed=True))
        assert delayed_xarray.shape == nd.shape
        np.testing.assert_allclose(delayed_xarray, nd.asarray())


def test_xarray(new_nd2):
    xr = pytest.importorskip("xarray")
    with ND2File(new_nd2) as nd:
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        assert isinstance(nd.to_xarray(squeeze=False), xr.DataArray)


def test_xarray_legacy(old_nd2):
    xr = pytest.importorskip("xarray")
    with ND2File(old_nd2) as nd:
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)
        assert isinstance(nd.to_xarray(squeeze=False), xr.DataArray)


def test_missing():
    with pytest.raises(FileNotFoundError):
        ND2File("asdfs")


def test_imread():
    d = imread(str(DATA / "jonas_header_test2.nd2"))
    assert isinstance(d, np.ndarray)
    assert d.shape == (4, 5, 520, 696)


@pytest.fixture()
def bfshapes():
    with open(DATA / "bf_shapes.json") as f:
        return json.load(f)


def test_bioformats_parity(new_nd2: Path, bfshapes: dict):
    """Testing that match bioformats shapes (or better when bioformats misses it)."""
    if new_nd2.name.startswith("JOBS_") or new_nd2.name in {
        "dims_rgb_t3p2c2z3x64y64.nd2",  # bioformats seems to miss the RGB
        "dims_rgb_c2x64y64.nd2",  # bioformats seems to miss the RGB
        "dims_t3y32x32.nd2",  # bioformats misses T
        "jonas_3.nd2",  # bioformats misses Z
        "cluster.nd2",  # bioformats misses both Z and T
    }:
        return  # bioformats misses XY position info in JOBS files
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


def test_pickle_open_reader(single_nd2):
    """test that we can pickle and restore an ND2File"""
    f = ND2File(single_nd2)
    pf = pickle.dumps(f)
    assert isinstance(pf, bytes)
    f2: ND2File = pickle.loads(pf)
    np.testing.assert_array_equal(f, f2)
    f.close()
    f2.close()


def test_pickle_closed_reader(single_nd2):
    """test that we can pickle and restore an ND2File"""
    f = ND2File(single_nd2)
    f.close()
    pf = pickle.dumps(f)
    assert isinstance(pf, bytes)
    f2: ND2File = pickle.loads(pf)
    assert f.closed
    assert f2.closed


def test_pickle_dask_wrapper(single_nd2):
    """test that we can pickle and restore just the dask wrapper"""

    # test that we can pickle and restore a file
    with ND2File(single_nd2) as f:
        d = f.to_dask()

    pd = pickle.dumps(d)
    assert isinstance(pd, bytes)
    d2 = pickle.loads(pd)
    assert isinstance(d2, ResourceBackedDaskArray)
    np.testing.assert_array_equal(d, d2)


# in v1.7.0.0, the sdk missed z coords for these
OLD_SDK_MISSES_COORDS = (
    (
        "jonas_100217_OD122_001.nd2",
        {"T": 25, "Z": 29, "C": 2, "Y": 311, "X": 277},
    ),
    (
        "jonas_512c_nikonTest_two.nd2",
        {"T": 16, "Z": 11, "C": 2, "Y": 520, "X": 696},
    ),
)


@pytest.mark.parametrize(("fname", "sizes"), OLD_SDK_MISSES_COORDS)
def test_sizes(fname, sizes):
    with ND2File(DATA / fname) as f:
        assert f.sizes == sizes


@pytest.mark.parametrize("validate", [True, False])
def test_chunkmap(validate):
    d = imread(str(DATA / "1.2audrosophila.nd2"), validate_frames=validate)
    expected: np.ndarray = np.array(
        [
            [57, 11, 51, 60, 92],
            [9, 19, 63, 80, 90],
            [17, 4, 47, 104, 62],
            [33, 28, 48, 36, 53],
            [67, 73, 86, 64, 69],
        ],
        dtype="uint16",
    )

    assert isinstance(d, np.ndarray)
    assert d.shape == (512, 512)
    assert np.array_equal(d[250:255, 250:255], expected)


def test_compressed():
    data = imread(DATA / "compressed_lossless.nd2")
    expected = np.asarray(
        [
            [
                [151, 131, 130, 120],
                [124, 125, 131, 126],
                [123, 121, 121, 138],
                [119, 123, 141, 114],
            ],
            [
                [133, 121, 128, 128],
                [136, 130, 141, 137],
                [115, 125, 115, 101],
                [132, 141, 131, 120],
            ],
        ]
    ).astype("uint16")
    assert np.array_equal(data[:, 100:104, 100:104], expected)


def test_extra_width_bytes():
    expected = [
        [203, 195, 193, 197],
        [203, 195, 195, 197],
        [205, 191, 192, 190],
        [204, 201, 196, 206],
    ]

    im = imread(str(DATA / "jonas_JJ1473_control_24h_JJ1473_control_24h_03.nd2"))
    np.testing.assert_array_equal(im[0, 0, :4, :4], expected)


def test_gc_triggers_cleanup(single_nd2):
    # this test takes advantage of the `no_files_left_open``
    # fixture in conftest to ensure that the file is closed
    import gc

    f: ND2File | None = ND2File(single_nd2)

    with pytest.warns(UserWarning, match="ND2File file not closed"):
        f = None  # noqa: F841
        gc.collect()


def test_file_handles(single_nd2: Path) -> None:
    """Test that we can open a file with a file handle also"""
    # just for coverage, since usually it will use the filehandle
    assert get_version(single_nd2) == (3, 0)

    with open(single_nd2, "rb") as fh:
        assert is_supported_file(fh)
        f = ND2File(fh)
        assert f.path == str(single_nd2)
        assert f.version == (3, 0)
        assert isinstance(f.asarray(), np.ndarray)
    assert fh.closed
    assert f.closed
