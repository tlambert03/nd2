from pathlib import Path
from typing import cast

from nd2._parse._clx_xml import json_from_clx_variant

XML = (Path(__file__).parent / "variant.xml").read_bytes()


def test_parse_xml() -> None:
    result = cast(dict, json_from_clx_variant(XML))
    assert list(result) == [
        "eType",
        "wsApplicationDesc",
        "wsUserDesc",
        "aMeasProbesBase64",
        "uLoopPars",
        "pItemValid",
        "sAutoFocusBeforeLoop",
        "wsCommandBeforeLoop",
        "wsCommandBeforeCapture",
        "wsCommandAfterCapture",
        "wsCommandAfterLoop",
        "bControlShutter",
        "bUsePFS",
        "uiRepeatCount",
        "ppNextLevelEx",
        "bControlLight",
        "pLargeImage",
        "sParallelExperiment",
    ]


def test_parse_variant_xml():
    variant = Path(__file__).parent / "variant_CustomDataV2_0.xml"
    xml = variant.read_bytes()
    data = json_from_clx_variant(xml, strip_variant=False)
    assert "variant" in data
    variant_dict = data["variant"]
    assert isinstance(variant_dict, dict)
    assert "CustomTagDescription_v1.0" in variant_dict
