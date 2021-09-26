from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nd2 import ND2File, structures
from nd2._nd2file import ND2Reader
from nd2._nd2file_legacy import ND2Reader as LegacyND2Reader

DATA = Path(__file__).parent / "data"

OLD_FORMATS = [
    "tests/data/aryeh_but3_cont200-1.nd2",
    "tests/data/aryeh_b16_14_12.nd2",
    "tests/data/aryeh_Time_sequence_24.nd2",
    "tests/data/aryeh_por003.nd2",
    "tests/data/aryeh_weekend002.nd2",
    "tests/data/aryeh_multipoint.nd2",
    "tests/data/aryeh_4_2_1_cont_NoMR001.nd2",
    "tests/data/aryeh_4_con_2_1_cot002.nd2",
    # 'tests/data/aryeh_but3_cont2002.nd',
]


@pytest.mark.parametrize("fname", OLD_FORMATS)
def test_metadata_extraction_legacy(fname):
    with ND2File(fname) as nd:
        assert nd.path == fname
        assert nd.is_open()

        assert isinstance(nd._rdr.seq_count(), int)
        assert isinstance(nd.attributes(), structures.Attributes)

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
        assert isinstance(nd.experiment(), list)
        assert isinstance(nd.text_info(), dict)
        assert isinstance(nd._coord_info(), list)
        assert all(isinstance(x, structures.Coordinate) for x in nd._coord_info())
        xarr = nd.to_xarray()
        assert isinstance(xarr, xr.DataArray)
        assert isinstance(xarr.data, da.Array)

    assert not nd.is_open()


def test_file_open():
    path = str(DATA / "aryeh_4_2_1_cont_NoMR001.nd2")
    with LegacyND2Reader(path) as nd:
        assert nd.is_open()
        assert nd.path == path

        assert isinstance(nd.seq_count(), int)
        assert nd.seq_count() == 405

        assert isinstance(nd.attributes(), structures.Attributes)

        data = nd.data(0)
        assert isinstance(data, np.ndarray)
        assert data.shape == (2, 520, 696)
    assert not nd.is_open()

    # doesn't work with new SDK
    with pytest.raises(OSError):
        ND2Reader(path)
