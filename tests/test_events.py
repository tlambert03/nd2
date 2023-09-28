from pathlib import Path

import nd2
import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


EXPECTED_COORDS = ([0] * 5 + [1000] * 5 + [2000] * 5) * 3


@pytest.mark.parametrize("fname", ["t3p3z5c3.nd2", "t3p3c3z5.nd2", "t1t1t1p3c3z5.nd2"])
def test_events(fname: str) -> None:
    with nd2.ND2File(DATA / fname) as f:
        events = f.events(orient="list")
        assert np.array_equal(events["X Coord [µm]"], EXPECTED_COORDS)
        assert np.array_equal(events["Y Coord [µm]"], EXPECTED_COORDS)


@pytest.mark.parametrize("fname", ["t3p3z5c3.nd2", "t3p3c3z5.nd2", "t1t1t1p3c3z5.nd2"])
def test_events_pandas(fname: str) -> None:
    pd = pytest.importorskip("pandas")
    with nd2.ND2File(DATA / fname) as f:
        df = pd.DataFrame(f.events())
        assert all(df["X Coord [µm]"] == EXPECTED_COORDS)
        assert all(df["Y Coord [µm]"] == EXPECTED_COORDS)
        assert all(df["Z-Series"] == [-1, -0.5, 0, 0.5, 1] * 9)
        assert all(df["T Index"] == [0] * 15 + [1] * 15 + [2] * 15)
        assert all(df["Z Index"] == [0, 1, 2, 3, 4] * 9)
        assert all(df["Position Name"] == (["p1"] * 5 + [""] * 5 + ["p3"] * 5) * 3)
