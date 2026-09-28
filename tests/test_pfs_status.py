from __future__ import annotations

from pathlib import Path

import nd2
from nd2.structures import PFSStatus

DATA = Path(__file__).parent / "data"
DOCUMENTED = {int(m) for m in PFSStatus}


def test_pfs_status_all_files(any_nd2: Path) -> None:
    """Every file yields either None or a documented int on each frame."""
    with nd2.ND2File(any_nd2) as f:
        if f.is_legacy:
            assert isinstance(f.frame_metadata(0), dict)
            return
        has_chunk = b"CustomData|PFS_STATUS!" in f._rdr.chunkmap
        # rows for user events have no PFS value and are filled with null_value
        events = f.events(orient="list", null_value=None)
        column = [v for v in events.get("PFS Status", []) if v is not None]
        # some files have more frames than loop_indices (e.g. jonas_3.nd2), and
        # frame_metadata() cannot be called on those trailing frames
        n = min(f.attributes.sequenceCount, len(f.loop_indices))
        for i in (0, n // 2, n - 1):
            fm = f.frame_metadata(i)
            assert isinstance(fm, nd2.structures.FrameMetadata)
            for channel in fm.channels:
                status = channel.pfs_status
                if not has_chunk:
                    assert status is None
                    continue
                assert isinstance(status, int)
                # all values in the test suite are documented by Nikon
                assert status in DOCUMENTED
                # agrees with the "PFS Status" column of events()
                assert status == column[i]


def test_pfs_status_values() -> None:
    with nd2.ND2File(DATA / "compressed_lossless.nd2") as f:
        assert (
            PFSStatus(f.frame_metadata(0).channels[0].pfs_status) is PFSStatus.SEARCHING
        )
    with nd2.ND2File(DATA / "cluster.nd2") as f:
        assert (
            PFSStatus(f.frame_metadata(29).channels[0].pfs_status) is PFSStatus.DISABLED
        )
    with nd2.ND2File(DATA / "t3p3c3z5.nd2") as f:
        assert (
            PFSStatus(f.frame_metadata((1, 2, 3)).channels[2].pfs_status)
            is PFSStatus.OUT_OF_RANGE
        )


def test_pfs_status_absent() -> None:
    with nd2.ND2File(DATA / "rois.nd2") as f:
        assert not f.is_legacy
        assert f.frame_metadata(0).channels[0].pfs_status is None
