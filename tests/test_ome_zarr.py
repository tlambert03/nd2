"""Tests for OME-Zarr export functionality."""

from __future__ import annotations

import json
from pathlib import Path

import nd2
import pytest
from nd2._util import AXIS

zarr = pytest.importorskip("zarr", minversion="3.0.0")
yaozarrs = pytest.importorskip("yaozarrs")


@pytest.fixture
def ome_zarr_nd2s(request: pytest.FixtureRequest) -> Path:
    """Fixture providing various ND2 files for OME-Zarr testing."""
    return request.param


# Test files with different dimension combinations
OME_ZARR_TEST_FILES = [
    "dims_t3c2y32x32.nd2",  # TCY
    "dims_c2y32x32.nd2",  # CYX
    "dims_z5t3c2y32x32.nd2",  # TZCYX (needs transpose)
    "cluster.nd2",  # TZCYX
]

# Files with positions
POSITION_TEST_FILES = [
    "dims_p4z5t3c2y32x32.nd2",  # TPZCYX
    "JOBS_Platename_WellA01_ChannelWidefield_Green_Seq0000.nd2",  # PYX
]


@pytest.mark.parametrize(
    "ome_zarr_nd2s",
    OME_ZARR_TEST_FILES,
    indirect=True,
    ids=lambda x: x,
)
def test_to_ome_zarr_basic(ome_zarr_nd2s: Path, tmp_path: Path) -> None:
    """Test basic OME-Zarr export without positions."""
    data_path = Path(__file__).parent / "data" / ome_zarr_nd2s
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "test.zarr"

    with nd2.ND2File(data_path) as f:
        result = f.to_ome_zarr(dest)

        assert result == dest
        assert dest.exists()

        # Check zarr.json exists and has OME metadata
        zarr_json_path = dest / "zarr.json"
        assert zarr_json_path.exists()

        with open(zarr_json_path) as fh:
            root_meta = json.load(fh)

        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"
        assert "ome" in root_meta["attributes"]
        assert "multiscales" in root_meta["attributes"]["ome"]

        multiscales = root_meta["attributes"]["ome"]["multiscales"]
        assert len(multiscales) == 1

        # Check axes are in OME-NGFF order: time, channel, spatial
        axes = multiscales[0]["axes"]
        axis_names = [ax["name"] for ax in axes]

        # Verify ordering: t before c, c before z, z before y, y before x
        if "t" in axis_names and "c" in axis_names:
            assert axis_names.index("t") < axis_names.index("c")
        if "c" in axis_names and "z" in axis_names:
            assert axis_names.index("c") < axis_names.index("z")
        if "z" in axis_names:
            assert axis_names.index("z") < axis_names.index("y")
        assert axis_names.index("y") < axis_names.index("x")

        # Check array exists
        array_path = dest / "0"
        assert array_path.exists()
        assert (array_path / "zarr.json").exists()

        # Validate with yaozarrs
        issues = yaozarrs.validate_zarr_store(str(dest))
        assert not issues, f"Validation failed: {issues}"


@pytest.mark.parametrize(
    "ome_zarr_nd2s",
    POSITION_TEST_FILES,
    indirect=True,
    ids=lambda x: x,
)
def test_to_ome_zarr_with_positions(ome_zarr_nd2s: Path, tmp_path: Path) -> None:
    """Test OME-Zarr export with multiple positions."""
    data_path = Path(__file__).parent / "data" / ome_zarr_nd2s
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "test.zarr"

    with nd2.ND2File(data_path) as f:
        n_positions = f.sizes.get(AXIS.POSITION, 1)
        result = f.to_ome_zarr(dest)

        assert result == dest

        # Root should be a plain group
        with open(dest / "zarr.json") as fh:
            root_meta = json.load(fh)
        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"

        # Each position should have its own group
        for i in range(n_positions):
            pos_path = dest / f"p{i}"
            assert pos_path.exists(), f"Position {i} not found"

            with open(pos_path / "zarr.json") as fh:
                pos_meta = json.load(fh)

            assert "ome" in pos_meta["attributes"]
            assert pos_meta["attributes"]["ome"]["multiscales"][0]["name"] == f"p{i}"

            # Validate each position
            issues = yaozarrs.validate_zarr_store(str(pos_path))
            assert not issues, f"Validation failed for p{i}: {issues}"


@pytest.mark.parametrize(
    "ome_zarr_nd2s",
    POSITION_TEST_FILES[:1],
    indirect=True,
    ids=lambda x: x,
)
def test_to_ome_zarr_single_position(ome_zarr_nd2s: Path, tmp_path: Path) -> None:
    """Test exporting a single position from a multi-position file."""
    data_path = Path(__file__).parent / "data" / ome_zarr_nd2s
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "single_pos.zarr"

    with nd2.ND2File(data_path) as f:
        n_positions = f.sizes.get(AXIS.POSITION, 1)
        if n_positions < 2:
            pytest.skip("Need multi-position file for this test")

        # Export position 1
        result = f.to_ome_zarr(dest, position=1)

        assert result == dest

        # Should have OME metadata at root (not under p1)
        with open(dest / "zarr.json") as fh:
            root_meta = json.load(fh)

        assert "ome" in root_meta["attributes"]

        # Array should be directly under root/0
        array_path = dest / "0"
        assert array_path.exists()

        # Validate
        issues = yaozarrs.validate_zarr_store(str(dest))
        assert not issues, f"Validation failed: {issues}"


def test_to_ome_zarr_axis_transposition(tmp_path: Path) -> None:
    """Test that axes are properly transposed to OME-NGFF order."""
    # Use a file with TZCYX order (Z before C in nd2)
    data_path = Path(__file__).parent / "data" / "dims_z5t3c2y32x32.nd2"
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "transposed.zarr"

    with nd2.ND2File(data_path) as f:
        # ND2 order: T=3, Z=5, C=2, Y=32, X=32
        nd2_sizes = dict(f.sizes)
        assert list(nd2_sizes.keys()) == ["T", "Z", "C", "Y", "X"]

        f.to_ome_zarr(dest)

    # Read back and verify shape is TCZYX (C and Z swapped)
    store = zarr.storage.LocalStore(str(dest / "0"))
    arr = zarr.open_array(store)

    # Expected OME order: T=3, C=2, Z=5, Y=32, X=32
    assert arr.shape == (3, 2, 5, 32, 32)

    # Verify axes metadata is correct
    with open(dest / "zarr.json") as fh:
        meta = json.load(fh)
    axes = meta["attributes"]["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]
    assert axis_names == ["t", "c", "z", "y", "x"]


def test_to_ome_zarr_coordinate_transforms(tmp_path: Path) -> None:
    """Test that coordinate transformations include proper scale values."""
    data_path = Path(__file__).parent / "data" / "dims_c2y32x32.nd2"
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "coords.zarr"

    with nd2.ND2File(data_path) as f:
        voxel = f.voxel_size()
        f.to_ome_zarr(dest)

    with open(dest / "zarr.json") as fh:
        meta = json.load(fh)

    transforms = meta["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert len(transforms) == 1
    assert transforms[0]["type"] == "scale"

    # Scale values should include voxel sizes for spatial dimensions
    scale = transforms[0]["scale"]
    # For CYX file: scales are [c_scale, y_scale, x_scale]
    assert scale[-1] == voxel.x  # X scale
    assert scale[-2] == voxel.y  # Y scale


def test_to_ome_zarr_custom_chunks(tmp_path: Path) -> None:
    """Test custom chunk specification."""
    data_path = Path(__file__).parent / "data" / "dims_c2y32x32.nd2"
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "chunked.zarr"
    custom_chunks = (1, 16, 16)

    with nd2.ND2File(data_path) as f:
        f.to_ome_zarr(dest, chunk_shape=custom_chunks)

    store = zarr.storage.LocalStore(str(dest / "0"))
    arr = zarr.open_array(store)

    # Chunks should match (or be smaller if shape is smaller)
    assert arr.chunks == custom_chunks


def test_to_ome_zarr_invalid_position(tmp_path: Path) -> None:
    """Test that invalid position index raises error."""
    data_path = Path(__file__).parent / "data" / "dims_p4z5t3c2y32x32.nd2"
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "invalid.zarr"

    with nd2.ND2File(data_path) as f:
        with pytest.raises(IndexError, match="out of range"):
            f.to_ome_zarr(dest, position=100)


def test_to_ome_zarr_invalid_backend(tmp_path: Path) -> None:
    """Test that invalid backend raises error."""
    data_path = Path(__file__).parent / "data" / "dims_c2y32x32.nd2"
    if not data_path.exists():
        pytest.skip(f"Test file not found: {data_path}")

    dest = tmp_path / "invalid_backend.zarr"

    with nd2.ND2File(data_path) as f:
        with pytest.raises(ValueError, match="Unknown backend"):
            f.to_ome_zarr(dest, backend="invalid")  # type: ignore[arg-type]
