from nd2._xml import parse_xml_block


def test_parse_xml() -> None:
    bXML = b"""
    <?xml version="1.0" encoding="UTF-8"?>
    <variant version="1.0">
        <CustomTagDescription_v1.0 runtype="CLxListVariant">
            <Tag0 runtype="CLxListVariant">
                <ID runtype="CLxStringW" value="CameraTemp1"/>
                <Type runtype="lx_int32" value="3"/>
                <Group runtype="lx_int32" value="0"/>
                <Size runtype="lx_int32" value="21"/>
                <Desc runtype="CLxStringW" value="Camera Temperature"/>
                <Unit runtype="CLxStringW" value="\xc2\xb0C"/>
            </Tag0>
            <Tag1 runtype="CLxListVariant">
                <ID runtype="CLxStringW" value="Camera_ExposureTime1"/>
                <Type runtype="lx_int32" value="3"/>
                <Group runtype="lx_int32" value="0"/>
                <Size runtype="lx_int32" value="21"/>
                <Desc runtype="CLxStringW" value="Exposure Time"/>
                <Unit runtype="CLxStringW" value="ms"/>
            </Tag1>
        </CustomTagDescription_v1.0>
    </variant>
    """

    EXPECT = {
        "CustomTagDescription_v1.0": {
            "Tag0": {
                "ID": "CameraTemp1",
                "Type": 3,
                "Group": 0,
                "Size": 21,
                "Desc": "Camera Temperature",
                "Unit": "°C",
            },
            "Tag1": {
                "ID": "Camera_ExposureTime1",
                "Type": 3,
                "Group": 0,
                "Size": 21,
                "Desc": "Exposure Time",
                "Unit": "ms",
            },
        }
    }

    assert parse_xml_block(bXML) == EXPECT
