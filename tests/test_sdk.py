from nd2._sdk import latest, v9


def test_new_sdk():

    fh = latest.open("tests/data/dims_p2z5t3-2c4y32x32.nd2")
    assert fh

    assert isinstance(latest.get_attributes(fh), dict)
    assert isinstance(latest.get_metadata(fh), dict)
    assert isinstance(latest.get_frame_metadata(fh, 0), dict)
    assert isinstance(latest.get_text_info(fh), dict)
    assert isinstance(latest.get_experiment(fh), list)

    assert isinstance(latest.get_seq_count(fh), int)
    assert isinstance(latest.get_seq_index_from_coords(fh, (0, 1, 2)), int)
    assert isinstance(latest.get_coords_from_seq_index(fh, 7), tuple)
    assert isinstance(latest.get_coord_info(fh), list)
    d = latest.get_image(fh, 1)
    assert d.shape
    assert d.mean()

    latest.close(fh)


def test_old_sdk():
    ...
    fh = v9.open("tests/data/aryeh_4_2_1_cont_NoMR001.nd2")
    assert fh

    assert isinstance(v9.get_attributes(fh), dict)
    assert isinstance(v9.get_metadata(fh), dict)
    # assert isinstance(v9.get_frame_metadata(fh, 0), str)
    assert isinstance(v9.get_text_info(fh), dict)
    assert isinstance(v9.get_experiment(fh), dict)
    assert isinstance(v9.get_stage_coords(fh), tuple)
    assert isinstance(v9.get_seq_count(fh), int)
    assert isinstance(v9.get_seq_index_from_coords(fh, (0, 1)), int)

    # # SEGFAULTS sometimes
    # assert isinstance(v9.get_coords_from_seq_index(fh, 7), tuple)
    assert isinstance(v9.get_coord_info(fh), list)
    assert isinstance(v9.get_custom_data_count(fh), int)
    assert isinstance(v9.get_zstack_home(fh), int)
    d = v9.get_image(fh, 1)
    assert d.shape
    assert d.mean()
    v9.close(fh)
