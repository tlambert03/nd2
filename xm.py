raw = b'<?xml version="1.0" encoding="UTF-8"?><variant version="1.0"><no_name runtype="CLxListVariant"><uiWidth runtype="lx_uint32" value="696"/><uiWidthBytes runtype="lx_uint32" value="1392"/><uiHeight runtype="lx_uint32" value="520"/><uiComp runtype="lx_uint32" value="1"/><uiBpcInMemory runtype="lx_uint32" value="16"/><uiBpcSignificant runtype="lx_uint32" value="14"/><uiSequenceCount runtype="lx_uint32" value="31"/><uiTileWidth runtype="lx_uint32" value="696"/><uiTileHeight runtype="lx_uint32" value="520"/><eCompression runtype="lx_int32" value="0"/><dCompressionParam runtype="double" value="1"/><uiVirtualComponents runtype="lx_uint32" value="1"/></no_name></variant>'
expect = {
    "variant": {
        "@version": "1.0",
        "no_name": {
            "@runtype": "CLxListVariant",
            "uiWidth": {"@runtype": "lx_uint32", "@value": "696"},
            "uiWidthBytes": {"@runtype": "lx_uint32", "@value": "1392"},
            "uiHeight": {"@runtype": "lx_uint32", "@value": "520"},
            "uiComp": {"@runtype": "lx_uint32", "@value": "1"},
            "uiBpcInMemory": {"@runtype": "lx_uint32", "@value": "16"},
            "uiBpcSignificant": {"@runtype": "lx_uint32", "@value": "14"},
            "uiSequenceCount": {"@runtype": "lx_uint32", "@value": "31"},
            "uiTileWidth": {"@runtype": "lx_uint32", "@value": "696"},
            "uiTileHeight": {"@runtype": "lx_uint32", "@value": "520"},
            "eCompression": {"@runtype": "lx_int32", "@value": "0"},
            "dCompressionParam": {"@runtype": "double", "@value": "1"},
            "uiVirtualComponents": {"@runtype": "lx_uint32", "@value": "1"},
        },
    }
}

from
