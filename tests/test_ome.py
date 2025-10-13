from pathlib import Path

import nd2
import pytest


def test_ome_meta(new_nd2: Path) -> None:
    ome = pytest.importorskip("ome_types")

    with nd2.ND2File(new_nd2) as f:
        meta = f.ome_metadata()
    assert isinstance(meta, ome.OME)

    # test naming
    if new_nd2.name == "dims_p4z5t3c2y32x32":
        names = [img.name for img in meta.images]
        assert names == ["point name 1", "point name 2", "point name 3", "point name 4"]
