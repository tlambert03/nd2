from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import dask.array as da
import pytest
from nd2 import ND2File, _util, structures
from nd2._parse._chunk_decode import ND2_FILE_SIGNATURE

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from nd2_describe import get_nd2_stats

try:
    import xarray as xr
except ImportError:
    xr = None


with open("tests/samples_metadata.json") as f:
    EXPECTED = json.load(f)

DATA = Path(__file__).parent / "data"
EXPECTED = {k: v for k, v in EXPECTED.items() if not _util.is_legacy(DATA / k)}


@pytest.mark.parametrize("path", EXPECTED, ids=lambda x: f'{x}_{EXPECTED[x]["ver"]}')
def test_metadata_integrity(path: str) -> None:
    """Test that the current API matches the expected output for sample data."""
    name, stats = get_nd2_stats(DATA / path)

    # normalize serizalized stuff
    stats = json.loads(json.dumps(stats, default=str))

    for key in stats:
        # The SDK has a bug in position name fetching... we do it better, so just clear
        if key == "experiment" and stats["ver"] >= "Ver3.0":
            _clear_names(stats[key], EXPECTED[name][key])
        assert stats[key] == EXPECTED[name][key], f"{key} mismatch"


def _clear_names(*exps: Any) -> None:
    for exp in exps:
        for item in exp:
            if item["type"] == "XYPosLoop":
                for point in item["parameters"]["points"]:
                    point.pop("name", None)


def test_decode_all_chunks(new_nd2: Path) -> None:
    with ND2File(new_nd2) as f:
        for key in f._rdr.chunkmap:
            if not key.startswith((b"ImageDataSeq", b"CustomData", ND2_FILE_SIGNATURE)):
                f._rdr._decode_chunk(key)


def test_metadata_extraction(new_nd2: Path) -> None:
    assert ND2File.is_supported_file(new_nd2)
    with ND2File(new_nd2) as nd:
        assert repr(nd)
        assert nd.path == str(new_nd2)
        assert not nd.closed

        assert isinstance(nd._rdr._seq_count(), int)
        assert isinstance(nd.attributes, structures.Attributes)

        # TODO: deal with typing when metadata is completely missing
        assert isinstance(nd.metadata, structures.Metadata)
        for i in range(nd._rdr._seq_count()):
            assert isinstance(nd.frame_metadata(i), structures.FrameMetadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.loop_indices, tuple)
        assert all(isinstance(x, dict) for x in nd.loop_indices)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.sizes, MappingProxyType)
        assert isinstance(nd.custom_data, dict)
        assert isinstance(nd.shape, tuple)
        assert isinstance(nd.size, int)
        assert isinstance(nd.closed, bool)
        assert isinstance(nd.ndim, int)
        _bd = nd.binary_data
        assert all(isinstance(x, structures.ROI) for x in nd.rois.values())
        assert isinstance(nd.is_rgb, bool)
        assert isinstance(nd.nbytes, int)

        assert isinstance(nd.unstructured_metadata(), dict)
        assert isinstance(nd.events(), list)

    assert nd.closed


def test_metadata_extraction_legacy(old_nd2: Path) -> None:
    assert ND2File.is_supported_file(old_nd2)
    with ND2File(old_nd2) as nd:
        assert repr(nd)
        assert nd.path == str(old_nd2)
        assert not nd.closed

        assert isinstance(nd.attributes, structures.Attributes)

        # # TODO: deal with typing when metadata is completely missing
        # assert isinstance(nd.metadata, structures.Metadata)
        assert isinstance(nd.experiment, list)
        assert isinstance(nd.text_info, dict)
        assert isinstance(nd.metadata, structures.Metadata)
        if xr is not None:
            xarr = nd.to_xarray()
            assert isinstance(xarr, xr.DataArray)
            assert isinstance(xarr.data, da.Array)

        with pytest.warns(UserWarning, match="not implemented"):
            nd.events()

    assert nd.closed


def test_events() -> None:
    # this method is smoke-tested for every file above...
    # but specific values are asserted here:
    with ND2File(DATA / "cluster.nd2") as f:
        rd = f.events(orient="list")

        headers = list(rd)
        row_0 = [rd[h][0] for h in headers]
        assert headers == [
            _util.TIME_KEY,
            "Index",
            "T Index",
            "Z Index",
            "Z-Series",
            "Camera 1 Temperature [°C]",
            "Laser Power; 1.channel [%]",
            "High Voltage; 1.channel",
            "Laser Power; 2.channel [%]",
            "High Voltage; 2.channel",
            "Laser Power; 3.channel [%]",
            "High Voltage; 3.channel",
            "Laser Power; 4.channel [%]",
            "High Voltage; 4.channel",
            "Camera 1 Exposure Time [ms]",
            "High Voltage; TD",
            "PFS Offset",
            "PFS Status",
            "X Coord [µm]",
            "Y Coord [µm]",
            "Ti ZDrive [µm]",
        ]
        assert row_0 == [
            0.44508349828422067,
            0,
            0,
            0,
            -2.0,
            -5.0,
            0.0,
            0,
            0.5,
            37,
            10.758400000000002,
            137,
            9.0,
            75,
            8.1,
            0,
            -1,
            7,
            -26056.951209195162,
            -4155.462732842248,
            3916.7250000000004,
        ]


@pytest.mark.parametrize("orient", ["records", "dict", "list"])
def test_events2(new_nd2: Path, orient: Literal["records", "dict", "list"]) -> None:
    with ND2File(new_nd2) as f:
        events = f.events(orient=orient)

    assert isinstance(events, list if orient == "records" else dict)
    if events and isinstance(events, dict):
        assert _util.TIME_KEY in events

    pd = pytest.importorskip("pandas")
    print(pd.DataFrame(events))


def test_compressed_metadata() -> None:
    with ND2File(DATA / "rois.nd2") as f:
        chunk = f._rdr._decode_chunk(b"CustomData|CustomDescriptionV1_0!")
        assert "CLxCustomDescription" in chunk
        assert "Name" in chunk["CLxCustomDescription"]
