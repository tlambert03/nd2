from pathlib import Path

import numpy as np
import pytest

from nd2._sdk import latest


def test_new_sdk(new_nd2: Path):
    with latest.ND2Reader(new_nd2) as nd:
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
        # if it is, _seq_count provides the highest "safe" frame you can retrieve.
        if scount != a.get("sequenceCount"):
            nd._image(scount - 1)
            with pytest.raises(IndexError):
                nd._image(scount)

        midframe = scount // 2
        if midframe > 1:
            coords = nd._coords_from_seq_index(midframe)
            assert isinstance(coords, tuple)
            assert nd._seq_index_from_coords(coords) == midframe

        assert isinstance(nd._seq_index_from_coords((0,) * csize), int)
        assert isinstance(nd._coord_info(), list)
        frame = nd._image(midframe)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (a["heightPx"], a["widthPx"], a["componentCount"])


# def test_old_sdk():
#     ...
#     fh = v9.open("tests/data/aryeh_4_2_1_cont_NoMR001.nd2")
#     assert fh

#     assert isinstance(v9.get_attributes(fh), dict)
#     assert isinstance(v9.get_metadata(fh), dict)
#     # assert isinstance(v9.get_frame_metadata(fh, 0), str)
#     assert isinstance(v9.get_text_info(fh), dict)
#     assert isinstance(v9.get_experiment(fh), dict)
#     assert isinstance(v9.get_stage_coords(fh), tuple)
#     assert isinstance(v9.get_seq_count(fh), int)
#     assert isinstance(v9.get_seq_index_from_coords(fh, (0, 1)), int)

#     # # SEGFAULTS sometimes
#     # assert isinstance(v9.get_coords_from_seq_index(fh, 7), tuple)
#     assert isinstance(v9.get_coord_info(fh), list)
#     assert isinstance(v9.get_custom_data_count(fh), int)
#     assert isinstance(v9.get_zstack_home(fh), int)
#     d = v9.get_image(fh, 1)
#     assert d.shape
#     assert d.mean()
#     v9.close(fh)
