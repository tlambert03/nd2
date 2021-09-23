import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dropbox import Dropbox

TEST_DATA = str(Path(__file__).parent.parent / "tests" / "data")
REMOTE_SAMPLES = "/nd2_samples/"
SUBSET = [
    "1.2audrosophila.nd2",
    "10audrosophila.nd2",
    "10ms_2xbin_100xmag.nd2",
    "5audrosophila.nd2",
    "5ms_2xbin_40xmag.nd2",
    "accPB_specUNMX.nd2",
    "aryeh_b16_14_12.nd2",
    "aryeh_but3_cont200-1.nd2",
    "aryeh_MeOh_high_fluo_003.nd2",
    "aryeh_MeOh_high_fluo_007.nd2",
    "aryeh_MeOh_high_fluo_011.nd2",
    "aryeh_por003.nd2",
    "aryeh_Time_sequence_24.nd2",
    "cluster.nd2",
    "FRETposUNMX.nd2",
    "gain1.nd2",
    "gain64.nd2",
    "gain6p2.nd2",
    "high_OA_1.nd2",
    "jonas_3.nd2",
    "jonas_control002.nd2",
    "jonas_divisionByZero_290110.tranMgc005cr.nd2",
    "jonas_header_test1.nd2",
    "jonas_header_test2.nd2",
    "jonas_jonas_nd2Test_Exception_2.nd2",
    "jonas_jonas_nd2Test_Exception61.nd2",
    "jonas_jonas_nd2Test_Exception9_e3.nd2",
    "jonas_movieSize_JJ_RNAi_hmr-1_001_CropCyan.nd2",
    "maxime_BF007.nd2",
    "rylie_SIM_A6L2m67_Reconstructed.nd2",
    "train_TR67_Inj7_fr50.nd2",
]

TOKEN = os.getenv("DROPBOX_TOKEN")
assert TOKEN, "must set DROPBOX_TOKEN to download files"


def fetch(dbx: Dropbox, remote_path: str, local_dest: str):
    try:
        dbx.files_download_to_file(
            Path(local_dest) / Path(remote_path).name, remote_path
        )
        print(f"success: {remote_path}")
    except Exception as e:
        print(f"ERROR: {remote_path} ({e})")


def download_folder(dbx: Dropbox, remote_folder: str, local_dest: str):
    files = [
        (dbx, x.path_display, local_dest)
        for x in dbx.files_list_folder(remote_folder).entries
        if x.is_downloadable
    ]
    if not files:
        return
    Path(local_dest).mkdir(exist_ok=True)
    with ThreadPoolExecutor() as exc:
        print(f"downloading {remote_folder} ...")
        list(exc.map(lambda _: fetch(*_), files))


def download_subset(dbx: Dropbox, local_dest: str):
    _files = [(dbx, REMOTE_SAMPLES + f, local_dest) for f in SUBSET]
    Path(local_dest).mkdir(exist_ok=True)
    with ThreadPoolExecutor() as exc:
        print("downloading subset ...")
        list(exc.map(lambda _: fetch(*_), _files))


def main(full=False, dest: str = TEST_DATA):
    with Dropbox(TOKEN) as dbx:
        # Check that the access token is valid
        try:
            dbx.users_get_current_account()
        except Exception:
            sys.exit("ERROR: Invalid access token")
        if full:
            download_folder(dbx, REMOTE_SAMPLES, dest)
        else:
            download_subset(dbx, dest)


if __name__ == "__main__":
    full = "--full" in sys.argv
    dest = sys.argv[sys.argv.index("--dest") + 1] if "--dest" in sys.argv else TEST_DATA
    main(full=full, dest=dest)
