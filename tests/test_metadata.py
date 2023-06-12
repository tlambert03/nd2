import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from nd2_describe import get_nd2_stats  # noqa: E402

with open("tests/samples_metadata.json") as f:
    EXPECTED = json.load(f)


@pytest.mark.parametrize("path", EXPECTED, ids=lambda x: f'{x}_{EXPECTED[x]["ver"]}')
def test_metadata_integrity(path: str):
    """Test that the current API matches the expected output for sample data."""
    target = Path("tests/data") / path
    name, stats = get_nd2_stats(target)

    # normalize serizalized stuff
    stats = json.loads(json.dumps(stats, default=str))

    for key in stats:
        # The SDK has a bug in position name fetching... we do it better, so just clear
        if key == "experiment" and stats["ver"] >= "Ver3.0":
            _clear_names(stats[key], EXPECTED[name][key])
        assert stats[key] == EXPECTED[name][key], f"{key} mismatch"


def _clear_names(*exps):
    for exp in exps:
        for item in exp:
            if item["type"] == "XYPosLoop":
                for point in item["parameters"]["points"]:
                    point.pop("name", None)
