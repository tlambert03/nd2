from pathlib import Path

import numpy as np
import pytest

from nd2._sdk import latest

SDK_MISSES_COORDS = {
    "jonas_100217_OD122_001.nd2",
    "jonas_512c_nikonTest_two.nd2",
    "jonas_512c_cag_p5_simgc_2511_70ms22s_crop.nd2",
    "jonas_2112-2265.nd2",
}


def test_new_sdk(new_nd2: Path):
    with latest.ND2Reader(new_nd2) as nd:
        a = nd._attributes()
        assert isinstance(a, dict)
        assert isinstance(nd._metadata(), dict)
        assert isinstance(nd._text_info(), dict)
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

            pycoords = nd._pycoords_from_seq_index(midframe)
            assert isinstance(pycoords, tuple)
            assert nd._seq_index_from_pycoords(pycoords) == midframe

            if new_nd2.name not in SDK_MISSES_COORDS:
                assert coords == pycoords

        assert isinstance(nd._seq_index_from_coords((0,) * csize), int)
        assert isinstance(nd._coord_info(), list)
        frame = nd._image(midframe)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (a["heightPx"], a["widthPx"], a["componentCount"])
        assert isinstance(nd.sizes(), dict)


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
