import json
from pathlib import Path
from typing import Any

from nd2._pysdk._pysdk import ND2Reader

TESTS = Path(__file__).parent
limdump = json.loads((TESTS / "readlim_output.json").read_text())


def assert_lim_close_enough(a: Any, lim_data: Any, key=()):
    # sourcery skip: assign-if-exp, reintroduce-else
    if isinstance(a, dict) and isinstance(lim_data, dict):
        if a == lim_data:
            return

        # clean type suffix from lim_data keys
        lim_data = {k.rsplit("_", 1)[0]: v for k, v in lim_data.items()}

        for k in a:
            av = a[k]
            if k not in lim_data:
                if bool(av) and av != [[]]:
                    raise AssertionError(
                        f"in key={key}: non-falsey key {k} not in limdump"
                    )
                continue
            bv = lim_data[k]
            if bv is None and bool(av):
                raise AssertionError(f"in key={key}: key {k} is None in limdump")
            assert_lim_close_enough(av, bv, (*key, k))
    elif a != lim_data:
        if lim_data is None and not bool(a):
            # lim may set {} or [] to None
            return
        if isinstance(lim_data, str) and isinstance(a, list):
            # FIXME: bytearrays
            return
        raise AssertionError(f"in key={key}: {a} != {lim_data}")


def test_parse_raw_metadata(new_nd2):
    with ND2Reader(new_nd2) as f:
        if f.version != (3, 0):
            return
        meta = f._raw_meta()
        lim_meta = limdump[new_nd2.name]["raw_metadata"]
        assert_lim_close_enough(meta, lim_meta)
