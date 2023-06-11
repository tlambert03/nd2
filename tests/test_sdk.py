from pathlib import Path

import numpy as np
import pytest

try:
    from nd2._sdk import latest
except ImportError:
    pytest.skip("No SDK found", allow_module_level=True)


def test_new_sdk(new_nd2: Path):
    with latest.ND2Reader(new_nd2, read_using_sdk=True) as nd:
        a = nd._attributes()
        assert isinstance(a, dict)
        assert isinstance(nd._metadata(), dict)
        assert isinstance(nd.text_info(), dict)
        assert isinstance(nd._experiment(), list)

        csize = nd._coord_size()
        scount = nd._seq_count()
        assert isinstance(scount, int)
        assert isinstance(csize, int)

        # sometimes _seq_count is lower than attrs.sequenceCount
        # if it is, _seq_count provides the highest "good" frame you can retrieve.
        if scount != a.get("sequenceCount"):
            nd._read_image(scount - 1)
            with pytest.raises(IndexError):
                nd._read_image(scount)

        midframe = scount // 2
        if midframe > 1:
            coords = nd._coords_from_seq_index(midframe)
            assert isinstance(coords, tuple)
            assert nd._seq_index_from_coords(coords) == midframe

        assert isinstance(nd._seq_index_from_coords((0,) * csize), int)
        assert isinstance(nd._coord_info(), list)
        frame = nd._read_image(midframe)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (a["heightPx"], a["widthPx"], a["componentCount"])

