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
                    if key and "sAutoFocus" in key[-1]:
                        # jonas_jonas_nd2Test_Exception9_e3.nd2 has a strange case
                        # in Experiment.uLoopPars.i0000000000.sAutoFocusBeforePeriod
                        # where readlim is able to recover data that doesn't appear to
                        # be in the XML
                        continue
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
        # FIXME: bytearrays
        if (
            isinstance(lim_data, str)
            and isinstance(a, list)
            or isinstance(a, bytearray)
        ):
            return
        if key and key[-1] == "bUseZ":
            # bUseZ has a bug where Truthy values are set to 116 rather than 1
            # TODO talk to lim folks about this
            return
        raise AssertionError(f"in key={key}: {a} != {lim_data}")


def test_parse_raw_metadata(new_nd2):
    with ND2Reader(new_nd2) as f:
        meta = f._raw_meta()
        lim_meta = limdump[new_nd2.name]["raw_metadata"]
        assert_lim_close_enough(meta, lim_meta)
