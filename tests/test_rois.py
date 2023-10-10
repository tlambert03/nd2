from pathlib import Path

import nd2
from nd2.structures import InterpType

DATA = Path(__file__).parent / "data"


def test_rois():
    with nd2.ND2File(DATA / "rois.nd2") as f:
        rois = f.rois.values()
        assert len(rois) == 18
        assert [r.id for r in rois] == list(range(1, 19))

        roi1 = f.rois[1]
        assert roi1.info.label == "rect bgrd"

        roi16 = f.rois[16]
        assert roi16.info.label == "S3:16 stim 3 poly"
        assert roi16.info.interpType == InterpType.StimulationROI
        assert roi16.animParams[0].extrudedShape.basePoints == [
            (-0.05231780847932399, -0.10247210748706266),
            (0.09780325689597874, 0.04522765038218665),
            (0.030006646726487105, 0.11302426055167814),
            (-0.0959013435882828, -0.05646726487205058),
        ]
