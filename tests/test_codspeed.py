from __future__ import annotations

import sys
from pathlib import Path

import nd2
import pytest

if all(x not in {"--codspeed", "tests/test_codspeed.py"} for x in sys.argv):
    pytest.skip("use --codspeed to run benchmarks", allow_module_level=True)

# fmt: off
DATA = Path(__file__).parent / "data"
FILES: list[Path] = [
    DATA / "jonas_control002.nd2",          # v2.1 -  48.05 MB
    DATA / "dims_p2z5t3-2c4y32x32.nd2",     # v3.0 -   1.10 MB
    DATA / "train_TR67_Inj7_fr50.nd2",      # v3.0 -  69.55 MB
    DATA / "karl_sample_image.nd2",         # v3.0 - 218.65 MB
]
# super slow on codspeed...
LEGACY = DATA / "aryeh_but3_cont200-1.nd2"  # v1.0 -  16.14 MB
# fmt: on


@pytest.mark.benchmark
@pytest.mark.parametrize("file", FILES, ids=lambda x: x.stem)
def test_time_imread(file: Path) -> None:
    """Test time to read a file."""
    _ = nd2.imread(file)


@pytest.mark.benchmark
@pytest.mark.parametrize("file", FILES, ids=lambda x: x.stem)
def test_time_imread_dask(file: Path) -> None:
    """Test time to read a file."""
    _ = nd2.imread(file, dask=True).compute()


@pytest.mark.benchmark
@pytest.mark.parametrize("file", [*FILES, LEGACY], ids=lambda x: x.stem)
def test_time_all_metadata(file: Path) -> None:
    """Test time to read all metadata."""
    with nd2.ND2File(file) as nd:
        _ = nd.metadata
        _ = nd.frame_metadata(0)
        _ = nd.attributes
        _ = nd.experiment
        _ = nd.text_info
