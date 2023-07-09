import nd2
import numpy as np
import pytest
from nd2._parse._chunk_decode import get_chunkmap


@pytest.fixture()
def broken_nd2(tmp_path, single_nd2):
    with open(single_nd2, "rb") as f:
        data = f.read()

    # single_nd2 has 352256 bytes
    # break the file by removing N bytes at offset Q
    Q = 287000
    N = 2000
    new_data = data[:Q] + data[(Q + N) :]
    assert len(new_data) < len(data)

    # write the broken file
    broken = tmp_path / "broken.nd2"
    with open(broken, "wb") as f:
        f.write(new_data)
    return broken


def test_rescue(broken_nd2, single_nd2, capsys):
    # TODO: we could potentially put more of this logic into convenience functions
    # we can't do too much magic about guessing shape and dtype since some files
    # may not have that information intact

    with open(broken_nd2, "rb") as f2:
        with pytest.raises(ValueError, match="appears to be corrupt. Expected "):
            cm2 = get_chunkmap(f2)

    with open(broken_nd2, "rb") as f2:
        with pytest.raises(ValueError, match="Also looked in the surrounding 1000"):
            # where 1000 is less than N above in broken_nd2
            cm2 = get_chunkmap(f2, error_radius=1000)

    with open(single_nd2, "rb") as f1, open(broken_nd2, "rb") as f2:
        cm1 = get_chunkmap(f1)
        cm2 = get_chunkmap(f2, error_radius=100_000)
    assert cm1 == cm2

    frame_shape = (32, 32, 2, 1)
    final_shape = (3, 2, 32, 32)
    rescued = nd2.rescue_nd2(broken_nd2, frame_shape, "uint16")
    raw_frames = [f.transpose((2, 0, 1, 3)).squeeze() for f in rescued]
    raw_read = np.stack(raw_frames).reshape(final_shape)
    assert "Found image 1" in capsys.readouterr().out

    with nd2.ND2File(single_nd2, validate_frames=True) as rdr:
        real_read = rdr.asarray()

    with nd2.ND2File(broken_nd2, validate_frames=True) as rdr:
        _ = rdr._rdr.chunkmap  # should not raise
        broken_read = rdr.asarray()

    # test that broken file is the same as the real file
    np.testing.assert_array_equal(real_read, raw_read)
    np.testing.assert_array_equal(broken_read, raw_read)

    crop = raw_read[:2, :2, 10:12, 10:12].flatten()
    expect = [99, 98, 102, 100, 99, 96, 97, 98, 100, 99, 100, 100, 94, 99, 98, 98]
    np.testing.assert_array_equal(crop, expect)
