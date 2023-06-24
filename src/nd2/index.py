from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import nd2


def _index_file(path: Path) -> dict:
    """Return a dict with the index file data."""
    with nd2.ND2File(path) as nd:
        stat = path.stat()

        software = {} if nd.is_legacy else nd._rdr._app_info()  # type: ignore
        acquired = "" if nd.is_legacy else nd._rdr._acquisition_date()  # type: ignore
        exp = [(x.type, x.count) for x in nd.experiment]
        axes, shape = zip(*nd.sizes.items())
        return {
            # "path": str(path.resolve()),
            "name": path.name,
            "nd2v": ".".join(map(str, nd.version)),
            "size": stat.st_size,
            # "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "acquired": acquired,
            "experiment": ";".join([f"{t}:{c}" for t, c in exp]),
            "dtype": str(nd.dtype),
            "shape": shape,
            "axes": "".join(axes),
            # "software_name": software.get("SWNameString", ""),
            "software_version": software.get("VersionString", ""),
            "grabber": software.get("GrabberString", ""),
        }


def _gather_files(
    paths: Iterable[Path], recurse: bool = False, glob: str = "*.nd2"
) -> Iterator[Path]:
    """Return a generator of all files in the given path."""
    for p in paths:
        if p.is_dir():
            yield from p.rglob(glob) if recurse else p.glob(glob)
        else:
            yield p


def _index_files(
    paths: Iterable[Path], recurse: bool = False, glob: str = "*.nd2"
) -> list[dict]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_index_file, _gather_files(paths, recurse, glob)))

    return results


def _pretty_print_table(data: list[dict[str, Any]]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold")

    # Extract the keys from the first dictionary to create table headers
    for header in data[0]:
        table.add_column(header)

    # Add data rows
    for row in data:
        table.add_row(*[str(value) for value in row.values()])

    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Path to ND2 file or directory containing ND2 files.",
    )
    parser.add_argument("--recurse", "-r", action="store_true")
    parser.add_argument("--glob-pattern", "-g", default="*.nd2")
    args = parser.parse_args()

    data = _index_files(args.paths, args.recurse, args.glob_pattern)

    _pretty_print_table(data)
