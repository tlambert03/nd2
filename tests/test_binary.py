from pathlib import Path

import nd2
import numpy as np

DATA = Path(__file__).parent / "data"


def test_binary():
    with nd2.ND2File(DATA / "with_binary_and_rois.nd2") as f:
        binlayers = f.binary_data
        assert binlayers is not None
        assert len(binlayers) == 4
        assert binlayers[0].name == "attached Widefield green (green color)"
        assert len(binlayers[0].data) == f.attributes.sequenceCount
        ary = np.asarray(binlayers)
        assert ary.shape == (4, 3, 4, 5, 32, 32)
        assert ary.sum() == 172947
