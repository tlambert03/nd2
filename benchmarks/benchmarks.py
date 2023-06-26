# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from pathlib import Path

import nd2

DATA = Path(__file__).parent.parent / "tests" / "data"
TEST_FILE = DATA / "train_TR67_Inj7_fr50.nd2"


class TimeSuite:
    """Test time to do things."""

    def time_imread(self) -> None:
        """Test time to read a file."""
        _x = nd2.imread(TEST_FILE)

    def time_all_metadata(self) -> None:
        """Test time to read a file."""
        with nd2.ND2File(TEST_FILE) as nd:
            _ = nd.metadata
            _ = nd.frame_metadata(0)
            _ = nd.attributes
            _ = nd.experiment
            _ = nd.text_info
