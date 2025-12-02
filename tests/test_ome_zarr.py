"""Tests for OME-Zarr export functionality."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import nd2
import pytest
from nd2._util import AXIS

if TYPE_CHECKING:
    from nd2._ome_zarr import ZarrBackend

try:
    import yaozarrs
except ImportError:
    pytest.skip(
        "yaozarrs and zarr is required for OME-Zarr tests", allow_module_level=True
    )

BACKENDS: list[ZarrBackend] = []
if importlib.util.find_spec("zarr") is not None:
    BACKENDS.append("zarr")
if importlib.util.find_spec("tensorstore") is not None:
    BACKENDS.append("tensorstore")

if not BACKENDS:
    pytest.skip(
        "No supported Zarr backend (zarr or tensorstore) found", allow_module_level=True
    )

TEST_DATA = Path(__file__).parent / "data"
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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nd2_path", OME_ZARR_TEST_FILES)
def test_to_ome_zarr_basic(
    nd2_path: Path, tmp_path: Path, backend: ZarrBackend
) -> None:
    """Test basic OME-Zarr export without positions."""
    data_path = TEST_DATA / nd2_path
    dest = tmp_path / "test.zarr"

    with nd2.ND2File(data_path) as f:
        result = f.to_ome_zarr(dest, backend=backend)
        assert result == dest
        yaozarrs.validate_zarr_store(str(dest))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nd2_path", POSITION_TEST_FILES)
def test_to_ome_zarr_with_positions(
    nd2_path: Path, tmp_path: Path, backend: ZarrBackend
) -> None:
    """Test OME-Zarr export with multiple positions using bioformats2raw layout."""
    data_path = TEST_DATA / nd2_path
    dest = tmp_path / "test.zarr"

    with nd2.ND2File(data_path) as f:
        n_positions = f.sizes.get(AXIS.POSITION, 1)
        result = f.to_ome_zarr(dest, backend=backend)
        assert result == dest

        # Root should have bioformats2raw.layout attribute under ome
        with open(dest / "zarr.json") as fh:
            root_meta = json.load(fh)
        assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

        # OME directory should exist with series metadata and METADATA.ome.xml
        ome_path = dest / "OME"
        assert ome_path.exists()
        assert (ome_path / "METADATA.ome.xml").exists()

        with open(ome_path / "zarr.json") as fh:
            ome_meta = json.load(fh)
        assert ome_meta["attributes"]["ome"]["series"] == [
            str(i) for i in range(n_positions)
        ]

        # Each position should have its own group with valid OME metadata
        for i in range(n_positions):
            pos_path = dest / str(i)
            assert pos_path.exists(), f"Position {i} not found"

            # Validate each position with yaozarrs
            yaozarrs.validate_zarr_store(str(pos_path))

            # Check position name in metadata
            with open(pos_path / "zarr.json") as fh:
                pos_meta = json.load(fh)
            assert pos_meta["attributes"]["ome"]["multiscales"][0]["name"] == str(i)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nd2_path", POSITION_TEST_FILES)
def test_to_ome_zarr_single_position(
    nd2_path: Path, tmp_path: Path, backend: ZarrBackend
) -> None:
    """Test exporting a single position from a multi-position file."""
    data_path = TEST_DATA / nd2_path
    dest = tmp_path / "single_pos.zarr"

    with nd2.ND2File(data_path) as f:
        assert f.sizes.get(AXIS.POSITION, 1) > 1, "Test requires a multi-position file"

        # Export position 1
        result = f.to_ome_zarr(dest, position=1, backend=backend)
        assert result == dest

        # Should have OME metadata at root (not under p1)
        with open(dest / "zarr.json") as fh:
            root_meta = json.load(fh)

        assert "ome" in root_meta["attributes"]

        # Array should be directly under root/0
        array_path = dest / "0"
        assert array_path.exists()

        # Validate
        yaozarrs.validate_zarr_store(str(dest))


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_ome_zarr_force_series(tmp_path: Path, backend: ZarrBackend) -> None:
    """Test that force_series creates bioformats2raw layout for single-pos files."""
    # Use a single-position file
    data_path = TEST_DATA / "dims_c2y32x32.nd2"
    dest = tmp_path / "series.zarr"

    with nd2.ND2File(data_path) as f:
        assert AXIS.POSITION not in f.sizes, "Test requires a single-position file"

        result = f.to_ome_zarr(dest, force_series=True, backend=backend)
        assert result == dest

        # Should have bioformats2raw.layout in root zarr.json
        with open(dest / "zarr.json") as fh:
            root_meta = json.load(fh)
        assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

        # OME directory should exist with series metadata and METADATA.ome.xml
        ome_path = dest / "OME"
        assert ome_path.exists()
        assert (ome_path / "METADATA.ome.xml").exists()

        with open(ome_path / "zarr.json") as fh:
            ome_meta = json.load(fh)
        assert ome_meta["attributes"]["ome"]["series"] == ["0"]

        # Image should be under 0/ directory
        pos_path = dest / "0"
        assert pos_path.exists()

        # Validate position with yaozarrs
        yaozarrs.validate_zarr_store(str(pos_path))


def test_to_ome_zarr_axis_transposition(tmp_path: Path) -> None:
    """Test that axes are properly transposed to OME-NGFF order."""
    zarr = pytest.importorskip("zarr")

    # Use a file with TZCYX order (Z before C in nd2)
    data_path = TEST_DATA / "dims_z5t3c2y32x32.nd2"

    dest = tmp_path / "transposed.zarr"

    with nd2.ND2File(data_path) as f:
        # ND2 order: T=3, Z=5, C=2, Y=32, X=32
        nd2_sizes = dict(f.sizes)
        assert list(nd2_sizes.keys()) == ["T", "Z", "C", "Y", "X"]

        f.to_ome_zarr(dest, backend=BACKENDS[0])  # Use first available backend

    # Read back and verify shape is TCZYX (C and Z swapped)
    arr = zarr.open_array(dest / "0")

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
    data_path = TEST_DATA / "dims_c2y32x32.nd2"

    dest = tmp_path / "coords.zarr"

    with nd2.ND2File(data_path) as f:
        voxel = f.voxel_size()
        f.to_ome_zarr(dest, backend=BACKENDS[0])

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
    zarr = pytest.importorskip("zarr")
    data_path = TEST_DATA / "dims_c2y32x32.nd2"

    dest = tmp_path / "chunked.zarr"
    custom_chunks = (1, 16, 16)

    with nd2.ND2File(data_path) as f:
        f.to_ome_zarr(dest, chunk_shape=custom_chunks, backend=BACKENDS[0])

    arr = zarr.open_array(dest / "0")

    # Chunks should match (or be smaller if shape is smaller)
    assert arr.chunks == custom_chunks


def test_to_ome_zarr_invalid_position(tmp_path: Path) -> None:
    """Test that invalid position index raises error."""
    data_path = TEST_DATA / "dims_p4z5t3c2y32x32.nd2"
    dest = tmp_path / "invalid.zarr"
    with nd2.ND2File(data_path) as f:
        with pytest.raises(IndexError, match="out of range"):
            f.to_ome_zarr(dest, position=100, backend=BACKENDS[0])


def test_to_ome_zarr_invalid_backend(tmp_path: Path) -> None:
    """Test that invalid backend raises error."""
    data_path = TEST_DATA / "dims_c2y32x32.nd2"

    dest = tmp_path / "invalid_backend.zarr"

    with nd2.ND2File(data_path) as f:
        with pytest.raises(ValueError, match="Unknown backend"):
            f.to_ome_zarr(dest, backend="invalid")  # type: ignore[arg-type]
