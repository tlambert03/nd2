from pathlib import Path

import nd2
import pytest


def test_ome_meta(new_nd2: Path) -> None:
    ome = pytest.importorskip("ome_types")

    with nd2.ND2File(new_nd2) as f:
        assert isinstance(f.ome_metadata(), ome.OME)
