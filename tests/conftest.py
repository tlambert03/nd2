from pathlib import Path
from typing import List

import psutil
import pytest
from nd2._util import is_new_format

DATA = Path(__file__).parent / "data"
MAX_FILES = None
ALL = sorted(
    (x for x in DATA.glob("*.nd2") if not x.name.startswith(".")),
    key=lambda x: x.stat().st_size,
)[:MAX_FILES]
NEW: List[Path] = []
OLD: List[Path] = []

for x in ALL:
    NEW.append(x) if is_new_format(str(x)) else OLD.append(x)

SINGLE = DATA / "dims_t3c2y32x32.nd2"


@pytest.fixture()
def single_nd2():
    return SINGLE


@pytest.fixture(params=ALL[:40] + [DATA / "jonas_control002.nd2"], ids=lambda x: x.name)
def small_nd2s(request) -> Path:
    return request.param


@pytest.fixture(params=ALL, ids=lambda x: x.name)
def any_nd2(request):
    return request.param


@pytest.fixture(params=NEW, ids=lambda x: f"{x.name}")
def new_nd2(request):
    return request.param


@pytest.fixture(params=OLD, ids=lambda x: x.name)
def old_nd2(request):
    return request.param


@pytest.fixture(autouse=True)
def _assert_no_files_left_open():
    files_before = {p for p in psutil.Process().open_files() if p.path.endswith("nd2")}
    yield
    files_after = {p for p in psutil.Process().open_files() if p.path.endswith("nd2")}
    assert files_before == files_after == set()
