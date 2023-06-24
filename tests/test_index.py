from nd2.index import _index_file


def test_index_new(new_nd2):
    assert isinstance(_index_file(new_nd2), dict)


def test_index_legacy(old_nd2):
    assert isinstance(_index_file(old_nd2), dict)
