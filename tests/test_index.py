from pathlib import Path

import nd2.index
import pytest

DATA = (Path(__file__).parent / "data").resolve()


@pytest.fixture(scope="module")
def records():
    return list(nd2.index._index_files([DATA, DATA / "cluster.nd2"]))


@pytest.mark.parametrize("fmt", ["csv", "json", "table"])
def test_format(records, fmt, capsys):
    filtered = nd2.index._filter_data(records)

    if fmt == "table":
        nd2.index._pretty_print_table(filtered, sort_column="name")
    elif fmt == "csv":
        nd2.index._print_csv(filtered)
    elif fmt == "json":
        nd2.index._print_json(filtered)
    captured = capsys.readouterr()
    assert captured.out
    assert not captured.err


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"include": "path,name,version"},
        {"sort_by": "version"},
        {"sort_by": "version-"},
        {"exclude": "path"},
        {"filters": ("'TimeLoop' in experiment",)},
        {"filters": ["acquired > '2020' and kb < 500"], "sort_by": "kb-"},
    ],
)
def test_filter_data(records, filters: dict) -> None:
    filtered = nd2.index._filter_data(records, **filters)
    assert isinstance(filtered, list)
    if filters.get("to_include"):
        assert len(filtered[0]) == len(filters["to_include"])
    sb = filters.get("sort_by")
    if sb:
        first_version = filtered[0]["version"]
        # ascending / descending
        assert first_version == "3.0" if sb.endswith("-") else "1.0"
    if filters.get("exclude"):
        assert "path" not in filtered[0]
    if filters.get("filters"):
        assert len(filtered) < len(records)
    else:
        assert len(filtered) == len(records)


def test_index(capsys):
    nd2.index.main([str(DATA), "--format", "csv"])
    captured = capsys.readouterr()
    assert "path" in captured.out
