from __future__ import annotations

from pathlib import Path

import nd2
import pytest

DATA = Path(__file__).parent / "data"


def test_ome_meta(new_nd2: Path) -> None:
    ome = pytest.importorskip("ome_types")

    with nd2.ND2File(new_nd2) as f:
        meta = f.ome_metadata()
    assert isinstance(meta, ome.OME)

    # test naming
    if new_nd2.name == "dims_p4z5t3c2y32x32.nd2":
        names = [img.name for img in meta.images]
        assert names == ["point name 1", "point name 2", "point name 3", "point name 4"]


def test_ome_exposure_invariants(new_nd2: Path) -> None:
    pytest.importorskip("ome_types")
    with nd2.ND2File(new_nd2) as f:
        meta = f.ome_metadata(include_unstructured=False)

    for img in meta.images:
        for plane in img.pixels.planes:
            assert plane.exposure_time_unit.value == "ms"
            assert plane.exposure_time is None or plane.exposure_time > 0


@pytest.mark.parametrize(
    "name, expected",
    [
        ("10ms_2xbin_100xmag.nd2", [10.0]),
        ("Exp3_9.8.21_Mouse1_DiI_4x_2x2-Slide1-1_B12.nd2", [150.0, 100.0]),
        ("cluster.nd2", [996.9975547008216] * 2),
        ("ML_06_72_ni_all8-MaxIP.nd2", [None] * 5),
    ],
)
def test_ome_exposure_vals(name: str, expected: list[float | None]) -> None:
    pytest.importorskip("ome_types")
    with nd2.ND2File(DATA / name) as f:
        meta = f.ome_metadata(include_unstructured=False)

    by_channel = {p.the_c: p.exposure_time for p in meta.images[0].pixels.planes}
    assert [by_channel[channel] for channel in range(len(expected))] == expected


@pytest.mark.parametrize(
    "dtype, expected", [("float32", "float"), ("float64", "double")]
)
def test_ome_float_pixel_type(
    monkeypatch: pytest.MonkeyPatch, dtype: str, expected: str
) -> None:
    pytest.importorskip("ome_types")
    import numpy as np

    # ND2 files can store float32 pixels (pixelDataType 'float'), but the
    # numpy dtype name is not a valid OME PixelType.
    monkeypatch.setattr(nd2.ND2File, "dtype", property(lambda self: np.dtype(dtype)))
    with nd2.ND2File(DATA / "dims_t3c2y32x32.nd2") as f:
        meta = f.ome_metadata(include_unstructured=False)

    assert meta.images[0].pixels.type.value == expected
