"""Debug script to explore JOBS nested byte arrays and find decoding issues."""

from pathlib import Path

from nd2._parse._chunk_decode import get_chunkmap, read_nd2_chunk
from nd2._parse._clx_lite import json_from_clx_lite_variant
from rich import print


def try_decode(data_bytes: bytes) -> tuple[bool, any, str]:
    """Try to decode bytes and return (success, result, error)."""
    try:
        result = json_from_clx_lite_variant(data_bytes)
        return True, result, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def decode_recursive_with_tracking(obj, path: str = "", failures: list | None = None):
    """Recursively decode, tracking all failures."""
    if failures is None:
        failures = []

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            result[k] = decode_recursive_with_tracking(v, new_path, failures)
        return result
    elif isinstance(obj, list):
        # Check if this looks like a byte array
        if obj and all(isinstance(x, int) and 0 <= x <= 255 for x in obj):
            data_bytes = bytes(obj)
            success, decoded, error = try_decode(data_bytes)
            if success and decoded:
                # Recursively decode the result
                return decode_recursive_with_tracking(decoded, path, failures)
            elif not success:
                failures.append(
                    {
                        "path": path,
                        "length": len(obj),
                        "error": error,
                        "first_bytes": data_bytes[:30],
                    }
                )
            # Return original if decode failed or empty
            return obj
        else:
            return [
                decode_recursive_with_tracking(x, f"{path}[{i}]", failures)
                for i, x in enumerate(obj)
            ]
    return obj


def analyze_file(fpath: Path) -> list[dict]:
    """Analyze a single file for JOBS decode failures."""
    failures = []

    try:
        with open(fpath, "rb") as fh:
            chunkmap = get_chunkmap(fh)

            def_key = b"CustomData|JobDefinitionV1_0!"
            if def_key not in chunkmap:
                return []  # Not a JOBS file

            offset, _ = chunkmap[def_key]
            data = read_nd2_chunk(fh, offset)
            job_def = json_from_clx_lite_variant(data)

            # Check for encrypted
            if "ProtectedJob" in job_def:
                return []  # Encrypted, can't analyze

            job = job_def.get("Job", {})
            decode_recursive_with_tracking(job, "", failures)
    except Exception as e:
        return [{"path": "FILE_ERROR", "error": str(e)}]

    return failures


def main():
    test_data = Path("tests/data")
    files = list(test_data.glob("*.nd2")) + list(test_data.glob("*.nd"))

    print(f"[bold]Analyzing {len(files)} files for JOBS decode failures...[/bold]\n")

    all_failures = {}

    for fpath in sorted(files):
        failures = analyze_file(fpath)
        if failures:
            all_failures[fpath.name] = failures

    if not all_failures:
        print("[green]No decode failures found![/green]")
        return

    print(f"[red]Found failures in {len(all_failures)} files:[/red]\n")

    # Group by failure path to see patterns
    failure_paths = {}
    for fname, failures in all_failures.items():
        for f in failures:
            path = f["path"]
            if path not in failure_paths:
                failure_paths[path] = []
            failure_paths[path].append(
                {
                    "file": fname,
                    "length": f.get("length"),
                    "error": f.get("error"),
                    "first_bytes": f.get("first_bytes"),
                }
            )

    print("[bold]Failures grouped by path:[/bold]\n")
    for path, occurrences in sorted(failure_paths.items()):
        print(f"[yellow]{path}[/yellow] ({len(occurrences)} occurrences)")
        for occ in occurrences[:3]:  # Show first 3
            print(f"  File: {occ['file']}")
            print(f"  Length: {occ['length']} bytes")
            if occ["first_bytes"]:
                print(f"  First bytes: {occ['first_bytes'][:20].hex()}")
        if len(occurrences) > 3:
            print(f"  ... and {len(occurrences) - 3} more")
        print()


if __name__ == "__main__":
    main()
