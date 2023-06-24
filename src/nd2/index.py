from __future__ import annotations

import argparse
import json
import sys
from argparse import RawTextHelpFormatter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence, cast, no_type_check

from typing_extensions import TypedDict

import nd2

try:
    import rich

    print = rich.print  # noqa: A001
except ImportError:
    rich = None


class Record(TypedDict):
    """Dict returned by `index_file`."""

    path: str
    name: str
    version: str
    kb: float
    acquired: str
    experiment: str
    dtype: str
    shape: list[int]
    axes: str
    software_name: str
    software_version: str
    grabber: str


HEADERS = list(Record.__annotations__)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # YYYY-MM-DD HH:MM:SS


def index_file(path: Path) -> Record:
    """Return a dict with the index file data."""
    with nd2.ND2File(path) as nd:
        if nd.is_legacy:
            software: dict = {}
            acquired: str | None = ""
        else:
            software = nd._rdr._app_info()  # type: ignore
            acquired = nd._rdr._acquisition_date()  # type: ignore

        stat = path.stat()
        exp = [(x.type, x.count) for x in nd.experiment]
        axes, shape = zip(*nd.sizes.items())
        if isinstance(acquired, datetime):
            acquired = acquired.strftime(TIME_FORMAT)

        return Record(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "version": ".".join(map(str, nd.version)),
                "kb": round(stat.st_size / 1000, 2),
                "acquired": acquired or "",
                "experiment": ";".join([f"{t}:{c}" for t, c in exp]),
                "dtype": str(nd.dtype),
                "shape": list(shape),
                "axes": "".join(axes),
                "software_name": software.get("SWNameString", ""),
                "software_version": software.get("VersionString", ""),
                "grabber": software.get("GrabberString", ""),
            }
        )


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
) -> list[Record]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(index_file, _gather_files(paths, recurse, glob)))

    return results


def _pretty_print_table(data: list[Record]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        raise sys.exit(
            "rich is required to print a pretty table. "
            "Install it with `pip install rich`."
        ) from None

    table = Table(show_header=True, header_style="bold")

    for header in data[0]:
        table.add_column(header)
    for row in data:
        table.add_row(*[str(value) for value in row.values()])

    Console().print(table)


def _print_csv(records: list[Record], skip_header: bool = False) -> None:
    import csv
    import sys

    writer = csv.DictWriter(sys.stdout, fieldnames=records[0].keys())
    if not skip_header:
        writer.writeheader()
    writer.writerows(records)


def _print_json(records: list[Record]) -> None:
    print(json.dumps(records, indent=2))


def _parse_args(argv: Sequence[str] = ()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Create an index of important metadata in ND2 files."
        f"\n\nValid column names are:\n{HEADERS!r}",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Path to ND2 file or directory containing ND2 files.",
    )
    parser.add_argument(
        "--recurse",
        "-r",
        action="store_true",
        default=False,
        help="Recursively search directories",
    )
    parser.add_argument(
        "--glob-pattern",
        "-g",
        type=str,
        default="*.nd2",
        help="Glob pattern to search for",
    )
    parser.add_argument(
        "--sort-by",
        "-s",
        default="",
        type=str,
        choices=[*HEADERS, "", *(f"{x}-" for x in HEADERS)],
        metavar="COLUMN_NAME",
        help="Column to sort by. If not specified, the order is not guaranteed. "
        "\nTo sort in reverse, append a hyphen.",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="table" if rich is not None else "json",
        type=str,
        choices=["table", "csv", "json"],
    )
    parser.add_argument(
        "--include",
        "-i",
        type=str,
        help="Comma-separated columns to include in the output",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        help="Comma-separated columns to exclude in the output",
    )
    parser.add_argument(
        "--no-header",
        default=False,
        action="store_true",
        help="Don't write the CSV header",
    )

    return parser.parse_args(argv or sys.argv[1:])


@no_type_check
def _filter_data(
    data: list[Record],
    to_include: Sequence[str] = (),
    sort_by: str | None = None,
    exclude: str | None = None,
) -> list[Record]:
    unrecognized = set(to_include) - set(HEADERS)
    if unrecognized:  # pragma: no cover
        print(f"Unrecognized columns: {', '.join(unrecognized)}", file=sys.stderr)
        to_include = [x for x in to_include if x not in unrecognized]

    if to_include:
        # preserve order of to_include
        data = [{h: row[h] for h in to_include} for row in data]

    to_exclude = cast("list[str]", exclude.split(",") if exclude else [])

    if to_exclude:
        data = [{h: row[h] for h in HEADERS if h not in to_exclude} for row in data]

    if sort_by:
        if sort_by.endswith("-"):
            data.sort(key=lambda x: x[sort_by[:-1]], reverse=True)
        else:
            data.sort(key=lambda x: x[sort_by])

    return data


def main(argv: Sequence[str] = ()) -> None:
    """Index ND2 files and print the results as a table."""
    args = _parse_args(argv)

    to_include = cast("list[str]", args.include.split(",") if args.include else [])
    if args.sort_by and to_include and args.sort_by not in to_include:
        raise sys.exit(  # pragma: no cover
            f"The sort column {args.sort_by!r} must be in the "
            f"included columns: {to_include!r}."
        )

    data = _index_files(paths=args.paths, recurse=args.recurse, glob=args.glob_pattern)
    data = _filter_data(
        data, to_include=to_include, sort_by=args.sort_by, exclude=args.exclude
    )

    if args.format == "table":
        _pretty_print_table(data)
    elif args.format == "csv":
        _print_csv(data, args.no_header)
    elif args.format == "json":
        _print_json(data)


if __name__ == "__main__":
    main()
