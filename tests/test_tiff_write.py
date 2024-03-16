from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import pytest

if TYPE_CHECKING:
    import tifffile as tf
else:
    tf = pytest.importorskip("tifffile")


def test_write_to_tiff(small_nd2s: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out.ome.tif"
    with nd2.ND2File(small_nd2s) as f:
        f.write_tiff(dest)

    real_shape = tf.imread(dest).shape
    assert real_shape == f.shape
