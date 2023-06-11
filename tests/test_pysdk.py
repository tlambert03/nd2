import json
import time
from dataclasses import asdict
from pathlib import Path

from nd2._pysdk._pysdk import ND2Reader

DATA = Path(__file__).parent / "data"
JSON = DATA / "json"
JSON.mkdir(exist_ok=True)


def test_new_pysdk(new_nd2: Path):
    EXPECT = json.loads((JSON / f"{new_nd2.stem}.json").read_text())

    start = time.perf_counter()
    nd = ND2Reader(new_nd2)
    nd.open()
    if nd.version < (3, 0):
        nd.close()
        return

    d = {
        "attributes": nd.attributes._asdict(),
        "metadata": asdict(nd.metadata()),
        "frame_metadata": asdict(nd.frame_metadata(0)),
        "text_info": nd.text_info(),
        "experiment": [asdict(x) for x in nd.experiment()],
        "coord_info": nd._coord_info(),
        "coords_from_seq_index": nd._coords_from_seq_index(0),
        "seq_count": nd._seq_count(),
        "coord_size": nd._coord_size(),
        # "data": nd._read_image(0)[..., :3, :3].squeeze().tolist(),
    }
    mytime = time.perf_counter() - start
    nd.close()

    print("\n    time", mytime, EXPECT["time"])
    d = json.loads(json.dumps(d))

    for k in d:
        if d[k] != EXPECT[k] and k in ("experiment", "frame_metadata"):
            print("name mismatch in file", new_nd2)
            _clear_names(d[k])
            _clear_names(EXPECT[k])
        assert d[k] == EXPECT[k], f"Key {k} does not match"


def _clear_names(exp):
    # The SDK has a bug in position name fetching... we do it better, so don't check it
    if isinstance(exp, list):
        for item in exp:
            if item["type"] == "XYPosLoop":
                for point in item["parameters"]["points"]:
                    point.pop("name", None)
    elif isinstance(exp, dict):
        for ch in exp["channels"]:
            ch["position"].pop("name", None)
            dn = ch["time"].get("absoluteJulianDayNumber")
            if dn is not None:
                ch["time"]["absoluteJulianDayNumber"] = round(dn, 4)
