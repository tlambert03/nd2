import re
from typing import Any, Dict, Optional

import lxml.etree


def parse_xml_block(bxml: bytes) -> Dict[str, Any]:
    node = lxml.etree.XML(bxml.split(b"?>", 1)[-1])
    return elem2dict(node)


lower = re.compile("^[a-z_]+")
_TYPEMAP: Dict[Optional[str], type] = {
    "bool": bool,
    "lx_uint32": int,
    "lx_uint64": int,
    "lx_int32": int,
    "lx_int64": int,
    "double": float,
    "CLxStringW": str,
    "CLxByteArray": str,
    "unknown": str,
    None: str,
}


def elem2dict(node: lxml.etree._Element) -> Dict[str, Any]:
    """
    Convert an lxml.etree node tree into a dict.
    """
    result: Dict[str, Any] = {}

    if "value" in node.attrib:
        return _TYPEMAP[node.attrib.get("runtype")](node.attrib["value"])

    for element in node.iterchildren():
        # Remove namespace prefix
        key = element.tag.split("}")[1] if "}" in element.tag else element.tag
        key = lower.sub("", key) or key

        # Process element as tree element if the inner XML contains non-whitespace
        # content
        if element.text and element.text.strip():
            value = element.text
        else:
            value = elem2dict(element)
        if key in result:
            if type(result[key]) is list:
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value
    return result["no_name"] if set(result.keys()) == {"no_name"} else result
