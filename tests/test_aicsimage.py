"""This test module is largely duplicated from aicsimageio. but edited a bit
to make it work here without the full aicsimageio test suite.

It may need updating if it changes upstream
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
from ome_types import OME

pytest.importorskip("aicsimageio")
from aicsimageio.readers.nd2_reader import ND2Reader  # noqa

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    (
        "filename",
        "set_scene",
        "expected_scenes",
        "expected_shape",
        "expected_dtype",
        "expected_dims_order",
        "expected_channel_names",
        "expected_physical_pixel_sizes",
        "expected_metadata_type",
    ),
    [
        pytest.param(
            "ND2_aryeh_but3_cont200-1.nd2",
            "XYPos:0",
            ("XYPos:0", "XYPos:1", "XYPos:2", "XYPos:3", "XYPos:4"),
            (1, 2, 1040, 1392),
            np.uint16,
            "TCYX",
            ["20phase", "20xDiO"],
            (1, 50, 50),
            dict,
        ),
        (
            "ND2_jonas_header_test2.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 4, 5, 520, 696),
            np.uint16,
            "CTZYX",
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
            OME,
        ),
        (
            "ND2_maxime_BF007.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 156, 164),
            np.uint16,
            "CYX",
            ["405/488/561/633nm"],
            (1.0, 0.158389678930686, 0.158389678930686),
            OME,
        ),
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            "point name 1",
            ("point name 1", "point name 2", "point name 3", "point name 4"),
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (2, 32, 32),
            np.uint16,
            "CYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_p1z5t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            "point name 1",
            ("point name 1", "point name 2"),
            (5, 5, 4, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red", "Widefield Far-Red", "Brightfield"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 2, 32, 32),
            np.uint16,
            "TCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            "XYPos:1",
            ("XYPos:0", "XYPos:1"),
            (3, 3, 2, 32, 32, 3),
            np.uint8,
            "TZCYXS",
            ["Brightfield", "Brightfield"],
            (0.01, 0.34285714285714286, 0.34285714285714286),
            OME,
        ),
        (
            "ND2_dims_rgb.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 64, 64, 3),
            np.uint8,
            "CYXS",
            ["Brightfield"],
            (1.0, 0.34285714285714286, 0.34285714285714286),
            OME,
        ),
    ],
)
def test_nd2_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_metadata_type: type,
) -> None:
    if filename.startswith("ND2_"):
        filename = filename[4:]
    image_container = ND2Reader(DATA / filename)

    # Run checks
    run_image_container_checks(
        image_container=image_container,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


def run_image_container_checks(
    image_container: ND2Reader,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_current_scene: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[
        Optional[float], Optional[float], Optional[float]
    ],
    expected_metadata_type: type,
) -> ND2Reader:
    """
    A general suite of tests to run against image containers (Reader and AICSImage).
    """

    # Set scene
    image_container.set_scene(set_scene)

    # Check scene info
    assert image_container.scenes == expected_scenes
    assert image_container.current_scene == expected_current_scene

    # Check basics
    assert image_container.shape == expected_shape
    assert image_container.dtype == expected_dtype
    assert image_container.dims.order == expected_dims_order
    assert image_container.dims.shape == expected_shape
    assert image_container.channel_names == expected_channel_names
    assert image_container.physical_pixel_sizes == expected_physical_pixel_sizes
    assert isinstance(image_container.metadata, expected_metadata_type)

    # Read different chunks
    zyx_chunk_from_delayed = image_container.get_image_dask_data("ZYX").compute()
    cyx_chunk_from_delayed = image_container.get_image_dask_data("CYX").compute()

    # Check image still not fully in memory
    assert image_container._xarray_data is None

    # Read in mem then pull chunks
    zyx_chunk_from_mem = image_container.get_image_data("ZYX")
    cyz_chunk_from_mem = image_container.get_image_data("CYX")

    # Compare chunk reads
    np.testing.assert_array_equal(
        zyx_chunk_from_delayed,
        zyx_chunk_from_mem,
    )
    np.testing.assert_array_equal(
        cyx_chunk_from_delayed,
        cyz_chunk_from_mem,
    )

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == expected_shape
    assert image_container.data.dtype == expected_dtype

    return image_container
