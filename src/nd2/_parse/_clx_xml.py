from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    import xml.etree.ElementTree

    import lxml.etree

    Element = Union[xml.etree.ElementTree.Element, lxml.etree._Element]
    Parser = Callable[[bytes | str], Element]
    Scalar = Union[float, str, int, bytearray, bool]
    JsonValue = Union[Scalar, dict[str, "JsonValue"]]
    XML: Parser
    ParseError: Exception

else:
    try:
        from lxml.etree import XML  # faster if it's available
    except ImportError:
        from xml.etree.ElementTree import XML

LOWER = re.compile("^[a-z_]+")


def _float_or_nan(x: str) -> float:
    try:
        return float(x)
    except ValueError:  # pragma: no cover
        return float("nan")


# functions to cast CLxvariants to python types
_XMLCAST: dict[str | None | bytes, Callable[[Any], Scalar]] = {
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
    if bxml.startswith(b"<?xml"):
        bxml = bxml.split(b"?>", 1)[-1]  # strip xml header

    try:
        node = parser(bxml)
    except SyntaxError:  # when there are invalid characters in the XML
        # could go straight to this ... not sure if it's slower
        try:
            node = parser(bxml.decode(encoding="utf-8", errors="ignore"))
        except Exception:
            node = parser(bxml.decode(encoding="utf-16", errors="ignore"))

    is_legacy = node.attrib.get("_VERSION") == "1.000000"
    name, val = _node_name_value(node, strip_prefix, include_attrs=is_legacy)

    # the special case of a single <variant><no_name>...</no_name></variant>
    # this is mostly here for Attributes, Experiment, Metadata, and TextInfo
    # LIM handles these special cases in JsonMetadata::composeRawMetadata
    if isinstance(val, dict) and list(val) == [f"i{0:010}"]:
        val = val["i0000000000"]

    return val if strip_variant and name == "variant" else {name: val}


def _node_name_value(
    node: Element, strip_prefix: bool = False, include_attrs: bool = False
) -> tuple[str, JsonValue]:
    """Return the name and value of an XML node.

    Parameters
    ----------
    node : Element
        The XML node.
    strip_prefix : bool, optional
        If True, strip the lowercase "type" prefix from the tag names, by default False.
    include_attrs: bool, optional
        If True, include the node attributes in the value, by default False.
        (This is only used for legacy XML.)

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
        val = node.attrib.get("value")
        value: JsonValue = _XMLCAST[runtype](val)
    else:
        value = dict(node.attrib) if include_attrs else {}
        for i, child in enumerate(node):
            cname, cval = _node_name_value(
                child,  # type: ignore
                strip_prefix,
                include_attrs,
            )
            # NOTE: "no_name" is the standard name for a list-type node
            # "BinaryItem" is a special case found in the BinaryMetadata_v1 tag...
            # without special handling, you would only get the last item in the list
            # FIXME: handle the special cases below "" better.
            if cname in (
                "no_name",
                None,
                "",
                "BinaryItem",
                "TextInfoItem",
                "Wavelength",
                "MinSrc",
                "MaxSrc",
            ):
                if not cval:
                    # skip empty nodes ... the sdk does this too
                    continue
                cname = f"i{i:010}"
            if cname in value:  # pragma: no cover
                # don't see this in tests anymore. but just in case...
                warnings.warn(f"Duplicate key {cname} in {name}", stacklevel=2)
            value[cname] = cval

    return name, value
