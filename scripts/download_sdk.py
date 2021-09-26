import os
import shutil
import sys
from zipfile import ZipFile

import dropbox

TOKEN = os.getenv("DROPBOX_TOKEN")
assert TOKEN, "must set DROPBOX_TOKEN to download files"
DEST = "nd2sdk"

with dropbox.Dropbox(TOKEN) as dbx:
    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except Exception:
        sys.exit("ERROR: Invalid access token")

    dbx.files_download_zip_to_file("_sdk.zip", "/nd2sdk")
    with ZipFile("_sdk.zip", "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.unlink("_sdk.zip")
    if os.path.exists("__MACOSX"):
        shutil.rmtree("__MACOSX")
