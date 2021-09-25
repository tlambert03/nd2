import json

with open("tests/samples_meta.json") as fh:
    d = json.load(fh)

new = {}
for fname, val in d.items():
    bf = val["bioformats"]
    nd2 = val["nd2"]
    pim = val["pims"]
    nd2r = val["nd2reader"]

    nd2shape = (
        dict(zip(nd2["axes"], nd2["shape"]))
        if "axes" in nd2 and "shape" in nd2
        else None
    )
    new[fname] = {
        "shape": {
            "bioformats": dict(zip("TCZYX", bf["shape"][:-1])),
            "nd2": nd2shape,
            "nd2reader": nd2r.get("sizes"),
            "pims_nd2": pim.get("sizes"),
        },
        "dtype": {
            "bioformats": bf["dtype"],
            "nd2": nd2.get("dtype"),
            "nd2reader": nd2r.get("dtype"),
            "pims_nd2": pim.get("dtype"),
        },
    }

with open("scripts/comparison.json", "w") as f:
    json.dump(new, f)


# shapes different between bioformats and nd2
# jonas_512c_cag_p5_simgc_2511_70ms22s_crop.nd2
# aryeh_b16_pdtB+y50__crop.nd2
# cluster.nd2
# jonas_512c_wtf_alignment.nd2
# jonas_2112-2265.nd2
# jonas_512c_nikonTest_two.nd2
# jonas_512c_wtf_DIC.nd2
# aryeh_qa-9507_control003.nd2
# jonas_3.nd2
# aryeh_rfp_h2a_cells_02.nd2
# jonas_jonas_nd2Test_Exception_2.nd2
# jonas_divisionByZero_290110.tranMgc005cr.nd2
# 20180427 JenlinkerCompounds002.nd2
# jonas_100217_OD122_001.nd2
# aryeh_old_cells_02.nd2
