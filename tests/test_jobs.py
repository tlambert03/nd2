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


def test_jobs_nested_bytearrays_decoded() -> None:
    """JOBS task Data fields contain nested CLX Lite that must be parsed as dicts."""
    path = DATA / "wellplate96_4_wells_with_jobs.nd2"
    with nd2.ND2File(path) as f:
        jobs = f.jobs()
        assert jobs is not None
        job = jobs["Job"]
        assert isinstance(job, dict)
        tasks = job["Tasks"]
        for task in tasks.values():
            data = task.get("Data")
            # Data fields should be recursively decoded dicts, not raw list[int]
            assert isinstance(data, dict), f"Task {task['Name']!r} Data not decoded"
