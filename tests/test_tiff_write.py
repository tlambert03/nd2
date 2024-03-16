from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import ome_types
import pytest

if TYPE_CHECKING:
    import tifffile as tf
else:
    tf = pytest.importorskip("tifffile")


@pytest.mark.parametrize("fname", ["out.tif", "out.ome.tif"])
def test_write_to_tiff(fname: str, small_nd2s: Path, tmp_path: Path) -> None:
    dest = tmp_path / fname

    def _mod_ome(ome: ome_types.OME) -> None:
        ome.creator = "ME"

    # semi-randomly choose whether to use the ND2File or the nd2_to_tiff function
    # and whether to test progress or OME-XML modification
    event_path = len(small_nd2s.stem) % 2 == 0
    if event_path:
        expected_creator = "ME"
        with nd2.ND2File(small_nd2s) as f:
            f.write_tiff(dest, progress=True, modify_ome=_mod_ome)
    else:
        expected_creator = f"nd2 v{nd2.__version__}"
        nd2.nd2_to_tiff(small_nd2s, dest, progress=False, modify_ome=None)

    with nd2.ND2File(small_nd2s) as f:
        shape = f.shape

    real_shape = tf.imread(dest).shape
    assert real_shape == shape

    if fname.endswith(".ome.tif"):
        assert ome_types.from_tiff(dest).creator == expected_creator
