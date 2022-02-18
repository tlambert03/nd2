from pathlib import Path

import nd2

DATA = Path(__file__).parent / "data"


def test_seg():
    with nd2.ND2File(str(DATA / "jonas_header_test2.nd2")) as f:
        img = f.to_xarray(delayed=True, squeeze=False, position=0)

    # this was segfaulting when nullcontext was used instead of self._lock:
    img.compute()
