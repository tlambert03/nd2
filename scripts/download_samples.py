import shutil
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests

TEST_DATA = Path(__file__).parent.parent / "tests" / "data"
URL = "https://www.dropbox.com/s/heo9ss4tcsi15x5/nd2_test_data.zip?dl=1"


def main():
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
    shutil.rmtree(TEST_DATA / "__MACOSX")


if __name__ == "__main__":
    main()
