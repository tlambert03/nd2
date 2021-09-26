from pathlib import Path

import numpy as np
import pytest

from nd2 import structures
from nd2._nd2file import ND2Reader
from nd2._nd2file_legacy import ND2Reader as LegacyND2Reader

DATA = Path(__file__).parent / "data"


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
