from pathlib import Path

import nd2
import numpy as np

DATA = Path(__file__).parent / "data"


def test_seg():
    with nd2.ND2File(str(DATA / "jonas_header_test2.nd2")) as f:
        img = f.to_xarray(delayed=True, squeeze=False, position=0)

    a = img.compute()
    b = img.compute()

    assert np.array_equal(a, b)
