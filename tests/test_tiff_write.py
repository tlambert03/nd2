from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import nd2
import ome_types
import pytest
from nd2._util import AXIS

if TYPE_CHECKING:
    import tifffile as tf
else:
    tf = pytest.importorskip("tifffile")


@pytest.mark.parametrize("fname", ["out.tif", "out.ome.tif"])
def test_write_to_tiff(fname: str, small_nd2s: Path, tmp_path: Path) -> None:
    nd2f = small_nd2s
    dest = tmp_path / fname
    on_frame = Mock()
    ME = "ME"

    def _mod_ome(ome: ome_types.OME) -> None:
        ome.creator = ME

    # semi-randomly choose whether to use the ND2File or the nd2_to_tiff function
    # and whether to test progress or OME-XML modification
    if len(nd2f.stem) % 2 == 0:
        expected_creator = ME
        with nd2.ND2File(nd2f) as f:
            f.write_tiff(dest, progress=True, modify_ome=_mod_ome, on_frame=on_frame)
    else:
        expected_creator = f"nd2 v{nd2.__version__}"
        nd2.nd2_to_tiff(nd2f, dest, progress=False, modify_ome=None, on_frame=on_frame)

    on_frame.assert_called()
    assert [type(x) for x in on_frame.call_args.args] == [int, int, dict]

    with nd2.ND2File(nd2f) as f, tf.TiffFile(dest) as tif:
        sizes = dict(f.sizes)
        assert len(tif.series) == sizes.pop(AXIS.POSITION, 1)
        expected_shape = tuple(x for k, x in sizes.items())
        assert all(s.shape == expected_shape for s in tif.series)

    if fname.endswith(".ome.tif"):
        assert ome_types.from_tiff(dest).creator == expected_creator
