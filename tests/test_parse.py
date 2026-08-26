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


def test_large_bytearray_stays_as_list():
    """Large BYTEARRAY fields must decode as list[int], not base64 strings.

    Regression test for https://github.com/tlambert03/nd2/issues/293
    Files with many XY positions have pItemValid bytearrays > 256 bytes.
    These must remain as list[int] so _parse_xy_pos_loop can filter positions.
    """
    import struct

    from nd2._parse._clx_lite import (
        ELxLiteVariantType,
        json_from_clx_lite_variant,
    )

    def _clx_bytearray_field(name: str, data: bytes) -> bytes:
        """Build a CLX Lite BYTEARRAY record."""
        name_utf16 = (name + "\x00").encode("utf-16-le")
        name_len = len(name_utf16) // 2
        return (
            struct.pack("BB", ELxLiteVariantType.BYTEARRAY, name_len)
            + name_utf16
            + struct.pack("<Q", len(data))
            + data
        )

    def _clx_int32_field(name: str, val: int) -> bytes:
        """Build a CLX Lite INT32 record."""
        name_utf16 = (name + "\x00").encode("utf-16-le")
        name_len = len(name_utf16) // 2
        return (
            struct.pack("BB", ELxLiteVariantType.INT32, name_len)
            + name_utf16
            + struct.pack("<i", val)
        )

    # --- plain byte arrays (pItemValid) must stay as list[int] ---
    flags = bytes([1, 0] * 250)  # 500 bytes, alternating valid/invalid
    chunk = _clx_bytearray_field("pItemValid", flags)

    result = json_from_clx_lite_variant(chunk, strip_prefix=False)
    item_valid = result["pItemValid"]
    assert isinstance(item_valid, list)
    assert len(item_valid) == 500
    assert item_valid == list(flags)

    # --- bytearrays containing nested CLX Lite must be recursively decoded ---
    # This is how JOBS task Data/SlotConnections fields are stored.
    nested_clx = _clx_int32_field("Answer", 42)
    chunk2 = _clx_bytearray_field("Data", nested_clx)

    result2 = json_from_clx_lite_variant(chunk2, strip_prefix=False)
    assert isinstance(result2["Data"], dict)
    assert result2["Data"]["Answer"] == 42


def test_degenerate_zstack_loop():
    """A 1-plane ZStackLoop with dZStep=0 and dZLow == dZHigh must not divide by zero.

    Regression test for https://github.com/tlambert03/nd2/issues/305
    NIS-Elements writes such a loop for single-plane acquisitions on systems with
    a piezo Z device configured.
    """
    loop = _parse._parse_z_stack_loop(
        {
            "uiCount": 1,
            "iType": 3,
            "bZInverted": True,
            "dZHome": 100.0,
            "dZLow": 100.0,
            "dZStep": 0.0,
            "dZHigh": 100.0,
            "wsZDevice": "NIDAQ Piezo Z (name: Piezo Z)",
        }
    )
    assert loop.count == 1
    assert loop.parameters.homeIndex == 0
    assert loop.parameters.stepUm == 0.0


def test_zstack_home_index_zero_range():
    """A multi-plane loop with dZLow == dZHigh has no meaningful home index."""
    assert (
        _parse._calc_zstack_home_index(
            False, 3, 3, home_um=100.0, low_um=100.0, high_um=100.0, step_um=0.0
        )
        == 0
    )


@pytest.mark.parametrize("type_", [2, 3])
def test_zstack_home_index_inverted_at_high(type_: int):
    """For types 2/3 an inverted loop uses the distance from dZHigh...

    ...even when that distance is exactly zero.
    """
    assert (
        _parse._calc_zstack_home_index(
            True, 5, type_, home_um=4.0, low_um=0.0, high_um=4.0, step_um=1.0
        )
        == 0
    )


@pytest.mark.parametrize("type_", [6, 7])
def test_zstack_home_index_inverted_at_low(type_: int):
    """For types 6/7 an inverted loop uses the distance from dZLow...

    ...even when that distance is exactly zero.
    """
    assert (
        _parse._calc_zstack_home_index(
            True, 5, type_, home_um=0.0, low_um=0.0, high_um=4.0, step_um=1.0
        )
        == 0
    )
