from pathlib import Path

import nd2

DATA = Path(__file__).parent / "data"


def test_rois():
    with nd2.ND2File(DATA / "rois.nd2") as f:
        rois = f.rois.values()
        assert len(rois) == 18
        assert [r.id for r in rois] == list(range(1, 19))
