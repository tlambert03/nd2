import os
import shutil
import sys
from zipfile import ZipFile

import dropbox

TOKEN = os.getenv("DROPBOX_TOKEN")
VERSION = "v9" if "--legacy" in sys.argv else "1.7.0.0"
assert TOKEN, "must set DROPBOX_TOKEN to download files"
DEST = "sdk_legacy" if VERSION == "v9" else "sdk"

with dropbox.Dropbox(TOKEN) as dbx:
    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except Exception:
        sys.exit("ERROR: Invalid access token")

    dbx.files_download_zip_to_file("_sdk.zip", f"/nd2sdk/{VERSION}/")
    with ZipFile("_sdk.zip", "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.rename(VERSION, DEST)
    os.unlink("_sdk.zip")
