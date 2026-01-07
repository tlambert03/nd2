import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from nd2 import structures
from nd2._parse import _parse
from nd2._readers import ModernReader


@cache
def readlim_output():
    TESTS = Path(__file__).parent
    return json.loads((TESTS / "readlim_output.json").read_text())


def test_parse_raw_metadata(new_nd2: Path):
    expected = readlim_output()
    if new_nd2.name not in expected:
        pytest.skip(f"{new_nd2.name} not in readlim_output.json")
    with ModernReader(new_nd2) as rdr:
        rdr._cached_global_metadata()  # force metadata to be read
        meta = {
            "Attributes": rdr._raw_attributes,
            "Experiment": rdr._raw_experiment,
            "Metadata": rdr._raw_image_metadata,
            "TextInfo": rdr._raw_text_info,
        }
        lim_meta = expected[new_nd2.name]["raw_metadata"]
        _assert_lim_close_enough(meta, lim_meta)


def _assert_lim_close_enough(a: Any, lim_data: Any, key=()):
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
            _assert_lim_close_enough(av, bv, (*key, k))
    elif a != lim_data:
        if lim_data is None and not bool(a):
            # lim may set {} or [] to None
            return
        # FIXME: bytearrays - LIM returns base64 strings for byte arrays,
        # but we decode nested CLX Lite data as dicts/lists
        if isinstance(lim_data, str) and isinstance(a, (list, dict, bytearray)):
            return
        if key and key[-1] == "bUseZ":
            # bUseZ has a bug where Truthy values are set to 116 rather than 1
            # TODO talk to lim folks about this
            return
        raise AssertionError(f"in key={key}: {a} != {lim_data}")


def test_load_events():
    # this is the output of
    # f._rdr._decode_chunk(b'CustomData|ExperimentEventsV1_0!', strip_prefix=False)
    # e = f['RLxExperimentRecord']
    # need to find a small file for this
    e = {
        "uiCount": 8,
        "pEvents": {
            "i0000000000": {
                "T": 30919.564199984074,
                "M": 15,
                "I": 1,
                "S": {
                    "T": 4,
                    "L": 0,
                    "P": 0,
                    "D": "DMD:S1 = (365 nm : 0.0%, 440 nm : 0.0%, 488 nm : 3.0%)",
                },
            },
            "i0000000001": {
                "T": 31128.348900020123,
                "M": 15,
                "I": 2,
                "S": {"T": 5, "L": 0, "P": 0, "D": ""},
            },
            "i0000000002": {
                "T": 61436.26100003719,
                "M": 15,
                "I": 3,
                "S": {
                    "T": 4,
                    "L": 1,
                    "P": 0,
                    "D": "DMD:S1 = (365 nm : 0.0%, 440 nm : 0.0%, 488 nm : 3.0%)",
                },
            },
            "i0000000003": {
                "T": 61649.4361000061,
                "M": 15,
                "I": 4,
                "S": {"T": 5, "L": 1, "P": 0, "D": ""},
            },
        },
    }
    events = _parse.load_events(e)
    assert isinstance(events[0], structures.ExperimentEvent)


def test_looks_like_clx_lite_rejects_false_positives():
    """Test _looks_like_clx_lite rejects byte patterns that aren't CLX Lite.

    Regression test for https://github.com/tlambert03/nd2/issues/288
    Byte arrays like pItemValid=[1, 1, 0, 0, ...] could be falsely detected as
    CLX Lite because byte 0=1 looks like BOOL type, byte 1=1 looks like
    name_length=1, and bytes 2-3=\\x00\\x00 look like a null-terminated empty name.
    """
    from nd2._parse._clx_lite import _looks_like_clx_lite

    # Pattern [type, 1, 0, 0, ...] should NOT be detected as CLX Lite
    # because name_length=1 is just the null terminator (empty name)
    # and empty-name entries are only valid inside a LEVEL container

    # [1, 1, 0, 0, 1] - byte 0=BOOL, byte 1=name_length=1, bytes 2-3=null term
    assert not _looks_like_clx_lite(bytes([1, 1, 0, 0, 1]))

    # [2, 1, 0, 0, 0, 0, 0, 0] - byte 0=INT32, byte 1=name_length=1
    assert not _looks_like_clx_lite(bytes([2, 1, 0, 0, 0, 0, 0, 0]))

    # [3, 1, 0, 0, 0, 0, 0, 0] - byte 0=UINT32, byte 1=name_length=1
    assert not _looks_like_clx_lite(bytes([3, 1, 0, 0, 0, 0, 0, 0]))

    # Valid CLX Lite with name_length >= 2 should still be detected
    # type=2 (INT32), name_length=2, name='A\0' in UTF-16, value=42
    import struct

    name = "A\x00".encode("utf-16-le")  # 4 bytes: 'A' + null
    value = struct.pack("<i", 42)  # 4 bytes
    valid_data = bytes([2, 2]) + name + value
    assert _looks_like_clx_lite(valid_data)
