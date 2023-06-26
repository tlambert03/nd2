import sys
from pathlib import Path
from typing import Callable

import nd2
import pytest

if all(x not in {"--codspeed", "--benchmark", "tests/test_bench.py"} for x in sys.argv):
    pytest.skip("use --benchmark to run benchmark", allow_module_level=True)


DATA = Path(__file__).parent.parent / "tests" / "data"
TEST_FILE = DATA / "train_TR67_Inj7_fr50.nd2"


def test_time_imread(benchmark: Callable) -> None:
    """Test time to read a file."""
    _ = nd2.imread(TEST_FILE)


def test_time_all_metadata(benchmark: Callable) -> None:
    """Test time to read all metadata."""
    with nd2.ND2File(TEST_FILE) as nd:
        _ = nd.metadata
        _ = nd.frame_metadata(0)
        _ = nd.attributes
        _ = nd.experiment
        _ = nd.text_info
