# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Download the sample files used by the test suite into `tests/data`.

Samples are assets of the `test-data` release on GitHub, and `samples.sha256` lists
the name and SHA-256 of each one.  Files already present with a matching checksum
are skipped.

To add a sample, upload it to the release and add its checksum to the manifest:

    gh release upload test-data new.nd2
    shasum -a 256 new.nd2 >> scripts/samples.sha256  # run next to the file
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).parent
TEST_DATA = HERE.parent / "tests" / "data"
MANIFEST = HERE / "samples.sha256"
BASE_URL = "https://github.com/tlambert03/nd2/releases/download/test-data/"
# assets that are zipped folders, extracted into TEST_DATA after download
ARCHIVES = {"JOBS.zip"}
RETRIES = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, sha: str) -> bool:
    """Download `name` unless an up-to-date copy exists. Return True if fetched."""
    dest = TEST_DATA / name
    if dest.exists() and _sha256(dest) == sha:
        return False

    part = dest.with_name(f"{name}.part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(BASE_URL + name) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError:
            if attempt == RETRIES:
                raise
    if (actual := _sha256(part)) != sha:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {name}: got {actual}")
    part.replace(dest)
    return True


def main() -> None:
    lines = MANIFEST.read_text().splitlines()
    manifest = {name: sha for sha, name in (ln.split(maxsplit=1) for ln in lines if ln)}
    TEST_DATA.mkdir(parents=True, exist_ok=True)

    errors = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            name: pool.submit(_fetch, name, sha) for name, sha in manifest.items()
        }
        for name, future in futures.items():
            try:
                fetched = future.result()
            except Exception as e:
                errors += 1
                print(f"FAILED  {name}: {e}", file=sys.stderr)
                continue
            if name in ARCHIVES and (
                fetched or not (TEST_DATA / Path(name).stem).exists()
            ):
                with ZipFile(TEST_DATA / name) as zf:
                    zf.extractall(TEST_DATA)
            print(f"{'fetched' if fetched else 'ok':8}{name}")

    if errors:
        sys.exit(f"{errors} of {len(manifest)} samples failed to download")


if __name__ == "__main__":
    main()
