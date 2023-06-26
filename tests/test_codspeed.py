import sys
from pathlib import Path

import nd2
import pytest

if all(x not in {"--codspeed", "--benchmark", "tests/test_bench.py"} for x in sys.argv):
    pytest.skip("use --benchmark to run benchmark", allow_module_level=True)


DATA = Path(__file__).parent / "data"
TEST_FILE = DATA / "train_TR67_Inj7_fr50.nd2"


@pytest.mark.benchmark
@pytest.mark.parametrize("file", [TEST_FILE], ids=lambda x: x.stem)
def test_time_imread(file: Path) -> None:
    """Test time to read a file."""
    _ = nd2.imread(file)


@pytest.mark.benchmark
@pytest.mark.parametrize("file", [TEST_FILE], ids=lambda x: x.stem)
def test_time_all_metadata(file: Path) -> None:
    """Test time to read all metadata."""
    with nd2.ND2File(file) as nd:
        _ = nd.metadata
        _ = nd.frame_metadata(0)
        _ = nd.attributes
        _ = nd.experiment
        _ = nd.text_info
