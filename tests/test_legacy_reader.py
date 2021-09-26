from pathlib import Path

import numpy as np
import pytest

from nd2 import LegacyND2File, ND2File, structures

DATA = Path(__file__).parent / "data"


def test_file_open():
    path = str(DATA / "aryeh_4_2_1_cont_NoMR001.nd2")
    with LegacyND2File(path) as nd:
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
        ND2File(path)
