import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import dropbox

if TYPE_CHECKING:
    from dropbox.files import FileMetadata, ListFolderResult

TEST_DATA = Path(__file__).parent.parent / "tests" / "data"

REMOTE_SAMPLES = "/nd2_samples/"
TOKEN = os.getenv("DROPBOX_TOKEN")
assert TOKEN, "must set DROPBOX_TOKEN to download files"


with dropbox.Dropbox(TOKEN) as dbx:
    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except Exception:
        sys.exit("ERROR: Invalid access token")

    file_list: "ListFolderResult" = dbx.files_list_folder(REMOTE_SAMPLES)

    def fetch(entry: "FileMetadata"):
        if not entry.is_downloadable:
            return
        try:
            print(f"downloading {entry.path_display} ...")
            dbx.files_download_to_file(TEST_DATA / entry.name, entry.path_display)
        except Exception as e:
            print(f"ERROR {entry.path_display}, {e}")

    with ThreadPoolExecutor() as exc:
        TEST_DATA.mkdir(exist_ok=True)
        list(exc.map(fetch, file_list.entries))
