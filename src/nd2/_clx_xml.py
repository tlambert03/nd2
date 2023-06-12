from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    import xml.etree.ElementTree

    import lxml.etree

    Element = Union[xml.etree.ElementTree.Element, lxml.etree._Element]
    Parser = Callable[[bytes], Element]
    Scalar = Union[float, str, int, bytearray, bool]
    JsonValue = Union[Scalar, dict[str, "JsonValue"]]
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
_XMLCAST: dict[str | None, Callable[[str], Scalar]] = {
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
}


def json_from_clx_variant(
    bxml: bytes,
    strip_variant: bool = True,
    strip_prefix: bool = False,
    parser: Parser = XML,
) -> JsonValue:
    """Parse a CLxVariant XML bytes into a python object.

    Parameters
    ----------
    bxml : bytes
        The CLxVariant XML bytes.
    strip_variant : bool, optional
        If True, strip the outermost <variant> tag if present, by default True.
    strip_prefix : bool, optional
        If True, strip the lowercase "type" prefix from the tag names, by default False.
    parser : Callable[[bytes], Element], optional
        The parser to use, by default will use lxml.etree.XML if available,
        and fall back to xml.etree.ElementTree.XML.

    Returns
    -------
    value : JsonValue
        The parsed CLxVariant XML bytes. Will either be a scalar or a dict, depending
        on the XML structure. (A <variant><no_name>...</no_name></variant> is the most
        likely case where a scalar is returned.)
    """
    node = parser(bxml.split(b"?>", 1)[-1])  # strip xml header
    name, val = _node_name_value(node, strip_prefix)

    # the special case of a single <variant><no_name>...</no_name></variant>
    # this is mostly here for Attributes, Experiment, Metadata, and TextInfo
    # LIM handles these special cases in JsonMetadata::composeRawMetadata
    if isinstance(val, dict) and list(val) == [f"i{0:010}"]:
        val = val["i0000000000"]

    return val if strip_variant and name == "variant" else {name: val}


def _node_name_value(
    node: Element, strip_prefix: bool = False
) -> tuple[str, JsonValue]:
    """Return the name and value of an XML node.

    Parameters
    ----------
    node : Element
        The XML node.
    strip_prefix : bool, optional
        If True, strip the lowercase "type" prefix from the tag names, by default False.

    Returns
    -------
    tuple[str, JsonValue]
        The name and value of the XML node.
    """
    name = node.tag
    if strip_prefix and name != "no_name":
        name = LOWER.sub("", name) or name

    runtype = node.attrib.get("runtype")
    if runtype in _XMLCAST:
        value: JsonValue = _XMLCAST[runtype](node.attrib["value"])
    else:
        value = {}
        for i, child in enumerate(node):
            cname, cval = _node_name_value(child, strip_prefix)
            # NOTE: "no_name" is the standard name for a list-type node
            # "BinaryItem" is a special case found in the BinaryMetadata_v1 tag...
            # without special handling, you would only get the last item in the list
            if cname in ("no_name", None, "", "BinaryItem", "TextInfoItem"):
                if not cval:
                    # skip empty nodes ... the sdk does this too
                    continue
                cname = f"i{i:010}"
            if cname in value:
                warnings.warn(f"Duplicate key {cname} in {name}", stacklevel=2)
            value[cname] = cval

    return name, value
