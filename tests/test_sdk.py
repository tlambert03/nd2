from nd2._sdk import latest


def test_new_sdk():

    fh = latest.open("tests/data/dims_p2z5t3-2c4y32x32.nd2")

    assert isinstance(latest.get_attributes(fh), str)
    assert isinstance(latest.get_metadata(fh), str)
    assert isinstance(latest.get_frame_metadata(fh, 0), str)
    assert isinstance(latest.get_textinfo(fh), str)
    assert isinstance(latest.get_experiment(fh), str)

    assert latest.get_seq_count(fh)
    assert latest.get_seq_index_from_coords(fh, (0, 1, 2))
    assert latest.get_coords_from_seq_index(fh, 7)
    assert latest.get_coord_info(fh)
    d = latest.get_image(fh, 1)
    assert d.shape
    assert d.mean()

    latest.close(fh)
