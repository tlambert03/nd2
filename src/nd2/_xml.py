from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Union, cast

if TYPE_CHECKING:
    import xml.etree.ElementTree

    import lxml.etree

    Element = Union[xml.etree.ElementTree.Element, lxml.etree._Element]
    Parser = Callable[[bytes], Element]
    Value = Union[float, str, int, bytearray, bool, dict[str, "Value"], list["Value"]]
    XML: Parser

else:
    try:
        from lxml.etree import XML  # faster if it's available
    except ImportError:
        from xml.etree.ElementTree import XML

LOWER = re.compile("^[a-z_]+")


def _float_or_nan(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


# functions to cast CLxvariants to python types
_XMLCAST: dict[str | None, Callable[[str], float | str | int | bytearray | bool]] = {
    "bool": lambda x: x.lower() in {"true", "1"},
    "CLxByteArray": lambda x: bytearray(x, "utf8"),
    "CLxStringW": str,
    "double": _float_or_nan,
    "float": _float_or_nan,
    "lx_int32": int,
    "lx_int64": int,
    "lx_uint32": int,
    "lx_uint64": int,
    "unknown": str,
    None: str,
}


def parse_variant_xml(
    bxml: bytes,
    parser: Parser = XML,
    strip_variant: bool = True,
    strip_prefix: bool = False,
) -> dict[str, Value]:
    """Return a dict of the xml data.

    If strip_variant is True, the top level variant tag will be stripped if present.
    """
    node = parser(bxml.split(b"?>", 1)[-1])  # strip xml header
    variant_dict = elem2dict(node, strip_prefix)

    if strip_variant and len(variant_dict) == 1:
        if "variant" in variant_dict:
            return cast("dict[str, Value]", variant_dict["variant"])

        k = next(iter(variant_dict))
        inner = variant_dict[k]
        if isinstance(inner, dict) and len(inner) == 1 and "variant" in inner:
            variant_dict = {k: inner["variant"]}
    return variant_dict


def _list_variant(node: Element, key: str, strip_prefix: bool) -> dict[str, Value]:
    if len(node) == 1 and node[0].tag == "no_name":
        node = node[0]

    if all(getattr(child, "tag", "").startswith("_") for child in node):
        # when all children are prefixed with '_' ('_00', '_01'):
        # return a list of the inner values
        _l: list[dict[str, Value]] = [elem2dict(c, strip_prefix) for c in node]
        return {key: [i.popitem()[1] for i in _l]}

    _resultd = {}
    for child in node:
        _resultd.update(elem2dict(child, strip_prefix))
    return {key: _resultd}


def elem2dict(node: Element, strip_prefix: bool = False) -> dict[str, Value]:
    """Convert an lxml.etree or ElementTree.node into a dict."""
    runtype = node.attrib.get("runtype")

    if strip_prefix and node.tag != "no_name":
        key = LOWER.sub("", node.tag) or node.tag
    else:
        key = node.tag

    if runtype == "CLxListVariant":
        return _list_variant(node, key, strip_prefix)
    elif len(node) == 0:
        if node.tag == "TextInfoItem":  # legacy nd2s
            idx = node.attrib["Index"]
            return {f"TextInfoItem_{idx}": node.attrib["Text"]}
        return {key: _XMLCAST[runtype](node.attrib["value"])}
    else:
        result: dict[str, Any] = {}
        for element in node:
            item = elem2dict(element, strip_prefix)
            duplicates = [i for i in item if i in result]
            # if any(duplicates):
            #     warnings.warn("duplicate keys in xml: " + ", ".join(duplicates))
            result.update(item)

    if len(result) == 1 and "no_name" in result:
        result = result["no_name"]

    return result if key == "no_name" else {key: result}
