import nd2
import numpy as np


def test_rescue(single_nd2):
    # TODO: we could potentially put more of this logic into convenience functions
    # we can't do too much magic about guessing shape and dtype since some files
    # may not have that information intact
    with nd2.ND2File(single_nd2) as rdr:
        real_read = rdr.asarray()
        raw_frames = [
            f.transpose((2, 0, 1, 3)).squeeze()
            for f in nd2.rescue_nd2(
                single_nd2, frame_shape=rdr._raw_frame_shape, dtype=rdr.dtype
            )
        ]
        raw_read: np.ndarray = np.stack(raw_frames).reshape(rdr.shape)
        np.testing.assert_array_equal(real_read, raw_read)
