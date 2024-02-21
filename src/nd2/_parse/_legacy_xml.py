"""XML parsing utilities for legacy ND2 files.

Todo:
----
all of this logic is duplicated in _clx_xml.py.
_legacy.py just needs some slight updates to deal with different parsing results.
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any, Callable

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree  # type: ignore


def parse_xml_block(bxml: bytes) -> dict[str, Any]:
    node = etree.XML(bxml.split(b"?>", 1)[-1])
    return elem2dict(node)  # type: ignore


lower = re.compile("^[a-z_]+")
_TYPEMAP: dict[str | bytes | None, Callable] = {
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


def elem2dict(node: etree._Element) -> Any:
    """Convert an lxml.etree node tree into a dict."""
    result: dict[str, Any] = {}

    if "value" in node.attrib:
        type_ = _TYPEMAP[node.attrib.get("runtype")]
        try:
            return type_(node.attrib["value"])
        except ValueError:
            return node.attrib["value"]

    attrs = node.attrib
    attrs.pop("runtype", "")
    attrs.pop("version", "")
    result.update(node.attrib)

    # [<Element CustomTagDescription_v1.0 at 0x12a29ac40>]
    for element in node:
        # Remove namespace prefix
        key = element.tag.split("}")[1] if "}" in element.tag else element.tag
        key = lower.sub("", key) or key

        # Process element as tree element if the inner XML contains non-whitespace
        # content
        if element.text and element.text.strip():
            value: str | dict = element.text
        else:
            value = elem2dict(element)
        if key in result:
            if isinstance(result[key], list):
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
