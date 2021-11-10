import re
from functools import partial
from typing import Any, Callable, Dict, Optional

import lxml.etree


def parse_xml_block(bxml: bytes) -> Dict[str, Any]:
    node = lxml.etree.XML(bxml.split(b"?>", 1)[-1])
    return elem2dict(node)


lower = re.compile("^[a-z_]+")
_TYPEMAP: Dict[Optional[str], Callable] = {
    "bool": bool,
    "lx_uint32": int,
    "lx_uint64": int,
    "lx_int32": int,
    "lx_int64": int,
    "double": float,
    "CLxStringW": str,
    "CLxByteArray": partial(bytes, encoding="utf-8"),
    "unknown": str,
    None: str,
}


def elem2dict(node: lxml.etree._Element) -> Dict[str, Any]:
    """
    Convert an lxml.etree node tree into a dict.
    """
    result: Dict[str, Any] = {}

    if "value" in node.attrib:
        type_ = _TYPEMAP[node.attrib.get("runtype")]
        try:
            return type_(node.attrib["value"])
        except ValueError:
            return node.attrib["value"]

    attrs = node.attrib
    attrs.pop("runtype", None)
    attrs.pop("version", None)
    result.update(node.attrib)

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
        # elif key in {"no_name", "variant"}:
        #     result.update(value)
        else:
            result[key] = value

    if isinstance(result, dict):
        if "variant" in result and not isinstance(result["variant"], list):
            result = result["variant"]
        if set(result.keys()) == {"no_name"} and not isinstance(
            result["no_name"], list
        ):
            result = result["no_name"]
    # if node.tag not in {"no_name", "variant"}:
    #     result = {node.tag: result}
    return result
