from __future__ import annotations

from pathlib import Path

import nd2
import pytest

DATA = Path(__file__).parent / "data"

JOBS_FILES = [
    DATA / "JOBS_Platename_WellA01_ChannelWidefield_Green_Seq0000.nd2",
    DATA / "JOBS_Platename_WellA02_ChannelWidefield_Green_Seq0001.nd2",
    DATA / "JOBS_Platename_WellB01_ChannelWidefield_Green_Seq0003.nd2",
    DATA / "JOBS_Platename_WellB02_ChannelWidefield_Green_Seq0002.nd2",
    DATA / "wellplate96_4_wells_with_jobs.nd2",
]


@pytest.mark.parametrize("path", JOBS_FILES)
def test_jobs_returns_dict_for_jobs_files(path: Path) -> None:
    with nd2.ND2File(path) as f:
        jobs = f.jobs()
        assert jobs is not None, f"Expected jobs() to return dict for {path.name}"
        assert "JobRunGUID" in jobs
        assert "ProgramDesc" in jobs
        if jobs["ProtectedJob"] is None:
            assert jobs["Job"]
        else:
            assert jobs["Job"] is None
