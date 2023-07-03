"""example usage:
python scripts/mine_json.py raw_metadata.Metadata_dic.sPicturePlanes_dic. \
    sSampleSetting_dic.a0_dic.pCameraSetting_dic.PropertiesQuality_dic
"""
import json
from pathlib import Path
from typing import Any, Iterator


def iter_members(
    obj: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_members(value, (*path, key))
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from iter_members(item, (*path, f"[{i}]"))


def get_key_sets(file: str | Path, path: str, show_all: bool = False) -> set[str]:
    with open(file) as f:
        data = json.load(f)

    allpaths = set()
    seen_members = set()
    for _path, _value in iter_members(data):
        joined = ".".join(_path).split(".nd2.", 1)[-1]
        allpaths.add(joined)
        if path == joined:
            seen_members.add(frozenset(_value))

    if seen_members:
        union = set()
        for i in seen_members:
            print(sorted(i))
            union |= i
        print(sorted(union))

    if show_all:
        for i in sorted(allpaths):
            print(i)
    return set()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Get key sets from a JSON file.")
    parser.add_argument("path", type=str, help="Path to the object in the JSON file.")
    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        default=Path(__file__).parent.parent / "tests" / "readlim_output.json",
        help="Path to the JSON file.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all key sets in the JSON file.",
    )
    args = parser.parse_args()

    get_key_sets(args.file_path, args.path, args.show_all)
