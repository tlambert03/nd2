# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests>=2.31",
# ]
# ///

import shutil
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests

TEST_DATA = Path(__file__).parent.parent / "tests" / "data"
URL = "https://www.dropbox.com/scl/fi/behxmt6ps2s5lp3k5qpjp/nd2_test_data.zip?rlkey=u8ra0s99xxuyan73669jwoq7f&dl=1"

# this is just here to invalidate the github actions cache
# change it when a new file is added to the test data in the dropbox folder
__HASH__ = "a1b2c3d4-e5f6-g7h8-i9j0-j1l2m3n4o5p7"


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
            sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}]")
            sys.stdout.flush()
    with ZipFile(f) as zf:
        zf.extractall(str(TEST_DATA))
    shutil.rmtree(TEST_DATA / "__MACOSX", ignore_errors=True)


if __name__ == "__main__":
    main()
