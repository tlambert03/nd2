from pathlib import Path

import nd2
import numpy as np
import numpy.testing as npt

DATA = Path(__file__).parent / "data"

# fmt: off
ROW0 = [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,0,0,0,3,0,0,0,0,0,0,0]
# fmt: on


def test_binary():
    with nd2.ND2File(DATA / "with_binary_and_rois.nd2") as f:
        binlayers = f.binary_data
        repr(binlayers)
        repr(binlayers[0])
        assert binlayers is not None
        assert len(binlayers) == 4
        assert binlayers[0].name == "attached Widefield green (green color)"
        assert len(binlayers[0]) == f.attributes.sequenceCount
        # you can index into the data
        npt.assert_array_equal(binlayers[0].data[2][0], ROW0)
        # you can also index a BinaryLayer directly
        assert isinstance(binlayers[0][2], np.ndarray)
        assert binlayers[0][3] is None
        npt.assert_array_equal(binlayers[0][2][0], ROW0)
        ary = np.asarray(binlayers)
        assert ary.shape == (4, 3, 4, 5, 32, 32)
        assert ary.sum() == 172947
