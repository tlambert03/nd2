"""Dumps info about nd2 files in tests directory.

Run using:

    python scripts/nd2_describe.py > tests/samples_metadata.json
"""

import struct
from dataclasses import asdict
from pathlib import Path

import nd2
from nd2 import structures
from nd2._parse._chunk_decode import iter_chunks


def _get_version(path):
    with open(path, "rb") as fh:
        if fh.read(4) == b"\xda\xce\xbe\n":
            for name, start, length in iter_chunks(fh):
                fh.seek(start)
                if name == "ND2 FILE SIGNATURE CHUNK NAME01!":
                    return fh.read(length).split(b"\x00", 1)[0].decode()
        else:
            fh.seek(-40, 2)
            sig, _ = struct.unpack("<32sQ", fh.read())
            if sig == b"LABORATORY IMAGING ND BOX MAP 00":
                return "legacy"
    raise RuntimeError("Not an ND2 file")


def get_nd2_stats(path: Path) -> "tuple[str, dict]":
    data = {"ver": _get_version(path)}

    with nd2.ND2File(path) as nd:
        meta = nd.metadata if isinstance(nd.metadata, dict) else asdict(nd.metadata)
        for channel in meta.get("channels", []):
            # we changed colorRGB to color inb v0.10.0
            if color := channel["channel"].pop("color", None):
                if isinstance(color, structures.Color):
                    channel["channel"]["colorRGB"] = color.as_abgr_u4()
            # Remove custom loops if null... they're super rare, and
            # readlimfile.json doesn't include them
            if channel.get("loops") and not channel["loops"].get("CustomLoop"):
                channel["loops"].pop("CustomLoop", None)
        data["attributes"] = nd.attributes._asdict()
        data["experiment"] = [asdict(x) for x in nd.experiment]
        data["metadata"] = meta
        data["textinfo"] = nd.text_info

    return path.name, data


if __name__ == "__main__":
    import json
    import sys
    from concurrent.futures import ThreadPoolExecutor

    DATA = Path(__file__).parent.parent / "tests" / "data"

    _paths = sys.argv[1:]
    if _paths:
        paths = [Path(p) for p in _paths]
    elif DATA.exists():
        paths = list(DATA.glob("*.nd2"))
    else:
        raise RuntimeError(f"Could not find test data: {DATA}")

    with ThreadPoolExecutor() as exc:
        results = dict(exc.map(get_nd2_stats, paths))

    print(json.dumps(results, default=str, indent=2))
