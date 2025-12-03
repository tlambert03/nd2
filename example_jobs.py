"""Example script for exploring JOBS metadata in ND2 files."""

import sys

import nd2
from rich import print

FILE = (
    sys.argv[1] if len(sys.argv) > 1 else "tests/data/wellplate96_4_wells_with_jobs.nd"
)
with nd2.ND2File(FILE) as f:
    jobs = f.jobs

    print(jobs)
