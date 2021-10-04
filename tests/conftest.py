from pathlib import Path
from typing import List

import pytest

from nd2._util import is_new_format

DATA = Path(__file__).parent / "data"
MAX_FILES = None
ALL = sorted(DATA.glob("*.nd2"), key=lambda x: x.stat().st_size)[:MAX_FILES]
NEW: List[Path] = []
OLD: List[Path] = []

for x in ALL:
    NEW.append(x) if is_new_format(str(x)) else OLD.append(x)


@pytest.fixture(params=ALL, ids=lambda x: x.name)
def any_nd2(request):
    return request.param


@pytest.fixture(params=NEW, ids=lambda x: x.name)
def new_nd2(request):
    return request.param


@pytest.fixture(params=OLD, ids=lambda x: x.name)
def old_nd2(request):
    return request.param
