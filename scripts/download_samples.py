import shutil
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests

TEST_DATA = Path(__file__).parent.parent / "tests" / "data"
URL = "https://www.dropbox.com/sh/pg9my6hnjj918x8/AACiKLlcDsljRgjJOec-9PQwa?dl=1"

# this is just here to invalidate the github actions cache
# change it when a new file is added to the test data in the dropbox folder
__HASH__ = "a1b2c3d4-e5f6-g7h8-i9j0-j1l2m3n4o5p6"


def main() -> None:
    response = requests.get(URL, stream=True)
    total_length = response.headers.get("content-length")

    f = BytesIO()
    if total_length is None:  # no content length header
        f.write(response.content)
    else:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=total_length // 100):
            dl += len(data)
            f.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write(f'\r[{"=" * done}{" " * (50 - done)}]')
            sys.stdout.flush()
    with ZipFile(f) as zf:
        zf.extractall(str(TEST_DATA))
    shutil.rmtree(TEST_DATA / "__MACOSX", ignore_errors=True)


if __name__ == "__main__":
    main()
