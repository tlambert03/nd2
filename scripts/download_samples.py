import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests

TEST_DATA = str(Path(__file__).parent.parent / "tests" / "data")
URL = "https://www.dropbox.com/s/shbuvnkheudt7d7/nd2_test_data.zip?dl=1"


def main():
    response = requests.get(URL, stream=True)
    total_length = response.headers.get("content-length")

    f = BytesIO()
    if total_length is None:  # no content length header
        f.write(response.content)
    else:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=4096):
            dl += len(data)
            f.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write("\r[{}{}]".format("=" * done, " " * (50 - done)))
            sys.stdout.flush()
    with ZipFile(f) as zf:
        zf.extractall(TEST_DATA)


# def main(dest: str = TEST_DATA):
#     with request.urlopen(URL) as resp:
#         with ZipFile(BytesIO(resp.read())) as zf:
#             zf.extractall(dest)


if __name__ == "__main__":
    main()
