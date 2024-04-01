from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import nd2
import pytest
import some_types
from nd2._util import AXIS

if TYPE_CHECKING:
    import tifffile as tf
else:
    tf = pytest.importorskip("tifffile")


@pytest.mark.parametrize("fname", ["out.tif", "out.some.tif"])
def test_write_to_tiff(fname: str, small_nd2s: Path, tmp_path: Path) -> None:
    nd2f = small_nd2s
    dest = tmp_path / fname
    on_frame = Mock()
    ME = "ME"

    def _mod_some(some: some_types.SOME) -> None:
        some.creator = ME

    # semi-randomly choose whether to use the ND2File or the nd2_to_tiff function
    # and whether to test progress or SOME-XML modification
    if len(nd2f.stem) % 2 == 0:
        expected_creator = ME
        with nd2.ND2File(nd2f) as f:
            f.write_tiff(dest, progress=True, modify_some=_mod_some, on_frame=on_frame)
    else:
        expected_creator = f"nd2 v{nd2.__version__}"
        nd2.nd2_to_tiff(nd2f, dest, progress=False, modify_some=None, on_frame=on_frame)

    on_frame.assert_called()
    assert [type(x) for x in on_frame.call_args.args] == [int, int, dict]

    with nd2.ND2File(nd2f) as f, tf.TiffFile(dest) as tif:
        sizes = dict(f.sizes)
        assert len(tif.series) == sizes.pop(AXIS.POSITION, 1)
        expected_shape = tuple(x for k, x in sizes.items())
        assert all(s.shape == expected_shape for s in tif.series)

    if fname.endswith(".some.tif"):
        assert some_types.from_tiff(dest).creator == expected_creator
