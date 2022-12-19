from nd2._xml import parse_variant_xml
from pathlib import Path
from nd2 import ND2File

XML = (Path(__file__).parent / "data" / "variant.xml").read_bytes()


# def test_parse_xml() -> None:
#     result = parse_variant_xml(XML)
#     assert list(result) == [
#         "eType",
#         "wsApplicationDesc",
#         "wsUserDesc",
#         "aMeasProbesBase64",
#         "uLoopPars",
#         "pItemValid",
#         "sAutoFocusBeforeLoop",
#         "wsCommandBeforeLoop",
#         "wsCommandBeforeCapture",
#         "wsCommandAfterCapture",
#         "wsCommandAfterLoop",
#         "bControlShutter",
#         "bUsePFS",
#         "uiRepeatCount",
#         "ppNextLevelEx",
#         "bControlLight",
#         "pLargeImage",
#         "sParallelExperiment",
#     ]


from nd2._pysdk._decode import decode_xml
from nd2._xml import parse_variant_xml
from rich import print

def test_metadata_extraction(new_nd2: Path):
    with ND2File(new_nd2) as f:
        if f._rdr.version >= (3, 0):
            return

        data = f._rdr._load_chunk(b"ImageMetadataSeq|0!")
        good = decode_xml(data)
        bad = parse_variant_xml(data)
        gp = good['sPicturePlanes']['sPlane']['a0']['pFilterPath']['m_pFilter']
        bp = bad['sPicturePlanes']['sPlane']['a0']['pFilterPath']['m_pFilter']
        if good != bad:
            import dictdiffer
            diffs = list(dictdiffer.diff(good, bad))
            assert good == bad