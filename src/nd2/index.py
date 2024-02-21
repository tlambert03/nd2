from __future__ import annotations

import argparse
import json
import sys
from argparse import RawTextHelpFormatter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypedDict, cast, no_type_check

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
    binary: bool
    rois: bool
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
            binary = False
        else:
            software = nd._rdr._app_info()  # type: ignore
            acquired = nd._rdr._acquisition_date()  # type: ignore
            binary = nd.binary_data is not None

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
                "binary": binary,
                "rois": False if nd.is_legacy else bool(nd.rois),
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


def _pretty_print_table(data: list[Record], sort_column: str | None = None) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

    except ImportError:
        raise sys.exit(
            "rich is required to print a pretty table. "
            "Install it with `pip install rich`."
        ) from None

    table = Table(show_header=True, header_style="bold")
    headers = list(data[0])

    # add headers, and highlight any sorted columns
    sort_col = ""
    if sort_column:
        sort_col = (sort_column or "").rstrip("-")
        direction = " ↓" if sort_column.endswith("-") else " ↑"
    for header in headers:
        if header == sort_col:
            table.add_column(header + direction, style="green")
        else:
            table.add_column(header)

    for row in data:
        table.add_row(*[_strify(value) for value in row.values()])

    Console().print(table)


def _strify(val: Any) -> str:
    if isinstance(val, bool):
        return "✅" if val else ""
    return str(val)


def _print_csv(records: list[Record], skip_header: bool = False) -> None:
    import csv

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
    parser.add_argument(
        "--filter",
        "-F",
        type=str,
        action="append",
        help="Filter the output. Each filter "
        "should be a python expression (string)\nthat evaluates to True or False. "
        "It will be evaluated in the context\nof each row. You can use any of the "
        "column names as variables.\ne.g.: \"acquired > '2020' and kb < 500\". (May "
        "be used multiple times).",
    )

    return parser.parse_args(argv or sys.argv[1:])


@no_type_check
def _filter_data(
    data: list[Record],
    sort_by: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    filters: Sequence[str] = (),
) -> list[Record]:
    """Filter and sort the data.

    Parameters
    ----------
    data : list[Record]
        the data to filter
    sort_by : str | None, optional
        Name of column to sort by, by default None
    include : str | None, optional
        Comma-separated list of columns to include, by default None
    exclude : str | None, optional
        Comma-separated list of columns to exclude, by default None
    filters : Sequence[str], optional
        Sequence of python expression strings to filter the data, by default ()

    Returns
    -------
    list[Record]
        _description_
    """
    includes = include.split(",") if include else []
    unrecognized = set(includes) - set(HEADERS)
    if unrecognized:  # pragma: no cover
        print(f"Unrecognized columns: {', '.join(unrecognized)}", file=sys.stderr)
        includes = [x for x in includes if x not in unrecognized]

    if sort_by:
        if sort_by.endswith("-"):
            data.sort(key=lambda x: x[sort_by[:-1]], reverse=True)
        else:
            data.sort(key=lambda x: x[sort_by])

    if includes:
        # preserve order of to_include
        data = [{h: row[h] for h in includes} for row in data]

    to_exclude = cast("list[str]", exclude.split(",") if exclude else [])

    if to_exclude:
        data = [{h: row[h] for h in HEADERS if h not in to_exclude} for row in data]

    if filters:
        # filters are in the form of a string expression, to be evaluated
        # against each row. For example, "'TimeLoop' in experiment"
        for f in filters:
            try:
                data = [row for row in data if bool(eval(f, None, row))]  # noqa: S307
            except Exception as e:  # pragma: no cover
                print(f"Error evaluating filter {f!r}: {e}", file=sys.stderr)
                sys.exit(1)

    return data


def main(argv: Sequence[str] = ()) -> None:
    """Index ND2 files and print the results as a table."""
    args = _parse_args(argv)

    data = _index_files(paths=args.paths, recurse=args.recurse, glob=args.glob_pattern)
    data = _filter_data(
        data,
        sort_by=args.sort_by,
        include=args.include,
        exclude=args.exclude,
        filters=args.filter,
    )

    if args.format == "table":
        _pretty_print_table(data, args.sort_by)
    elif args.format == "csv":
        _print_csv(data, args.no_header)
    elif args.format == "json":
        _print_json(data)


if __name__ == "__main__":
    main()
