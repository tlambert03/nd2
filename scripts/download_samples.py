import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dropbox import Dropbox

TEST_DATA = str(Path(__file__).parent.parent / "tests" / "data")
REMOTE_SAMPLES = "/nd2_samples/"
SUBSET = [
    "jonas_headerTest2_nikon.mat",
    "jonas_headerTest1_nikon.mat",
    "jonas_512c_wtf_alignment_header.mat",
    "jonas_512c_wtf_DIC_header.mat",
    "jonas_512c_nikonTest_2_header.mat",
    "jonas_512c_nikonTest_1_header.mat",
    "jonas_512c_cag_p5_simgc_2511_70ms22s_crop_header.mat",
    "dims_p4z5t3c2y32x32-tile2x3.txt",
    "bf_shapes.json",
    "shapes.json",
    "dims_p4z5t3c2y32x32-Zlambda.txt",
    "jonas_512c_cag_p5_simgc_2511_70ms22s_header.mat",
    "ndchunks.json",
    "dims_rgb.nd2",
    "dims_snap.nd2",
    "JOBS_Platename_WellA02_ChannelWidefield_Green_Seq0001.nd2",
    "JOBS_Platename_WellB01_ChannelWidefield_Green_Seq0003.nd2",
    "JOBS_Platename_WellA01_ChannelWidefield_Green_Seq0000.nd2",
    "JOBS_Platename_WellB02_ChannelWidefield_Green_Seq0002.nd2",
    "dims_c1y32x32.nd2",
    "dims_t3y32x32.nd2",
    "dims_y32x32_custom_meta.nd2",
    "maxime_BF007.nd2",
    "dims_c2y32x32.nd2",
    "dims_t3c2y32x32.nd2",
    "dims_rgb_c2x64y64.nd2",
    "dims_z5t3c2y32x32.nd2",
    "dims_p1z5t3c2y32x32.nd2",
    "dims_5z3t2c-jagged.nd2",
    "cluster.nd2",
    "dims_rgb_t3p2c2z3x64y64.nd2",
    "accPB_specUNMX.nd2",
    "10audrosophila.nd2",
    "1.2audrosophila.nd2",
    "5audrosophila.nd2",
    "dims_p4z5t3c2y32x32.nd2",
    "dims_p4z5t3c2y32x32-Zlambda.nd2",
    "dims_z5t3c4_0all_1firstt_2homez_3every2thomez_y32x32.nd2",
    "dims_p2z5t3-2c4y32x32.nd2",
    "dims_p4z5t3c2y32x32-tile2x3.nd2",
    "high_OA_1.nd2",
    "jonas_jonas_nd2Test_Exception_2.nd2",
    "jonas_jonas_nd2Test_Exception9_e3.nd2",
    "gain6p2.nd2",
    "gain1.nd2",
    "gain64.nd2",
    "aryeh_MeOh_high_fluo_003.nd2",
    "aryeh_MeOh_high_fluo_011.nd2",
    "aryeh_MeOh_high_fluo_007.nd2",
    "10ms_2xbin_100xmag.nd2",
    "5ms_2xbin_40xmag.nd2",
    "jonas_header_test2.nd2",
    "aryeh_but3_cont200-1.nd2",
    "mrna_rgb.tif",
    "1_noadd_020.nd2",
    "Exp3_9.8.21_Mouse1_DiI_4x_2x2-Slide1-1_B12.nd2",
    "aryeh_b16_14_12.nd2",
    "FRETposUNMX.nd2",
    "E12g-1_cy3b-P1-E05g-2-THG-20xd_1_50-RH_2_P1-5n_survey002.nd2",
    "jonas_header_test1.nd2",
    "jonas_movieSize_JJ_RNAi_hmr-1_001_CropCyan.nd2",
    "jonas_jonas_nd2Test_Exception61.nd2",
    "jonas_control002.nd2",
    "rylie_SIM_A6L2m67_Reconstructed.nd2",
    "train_TR67_Inj7_fr50.nd2",
    "aryeh_Time_sequence_24.nd2",
    "E12g-2_cy3b-THG-20xd-50_1_E05g-1-LF01_1_A-GS_survey003.nd2",
    "aryeh_por003.nd2",
    "jonas_3.nd2",
    "jonas_JJ1473_control_24h_JJ1473_control_24h_03.nd2",
    "aryeh_weekend002.nd2",
    "jonas_JJ1473_control_24h_JJ1473_conrol_24h_02.nd2",
    "jonas_movieSize_JJ_RNAi_hmr-1_001.nd2",
    "karl_sample_image.nd2",
    "jonas_qa-7534_130606_JCC719_003.nd2",
    "jonas_100217_OD122_001.nd2",
    "jonas_512c_nikonTest_two.nd2",
    "rylie_SIM_A6L2m67.nd2",
    "aryeh_multipoint.nd2",
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
