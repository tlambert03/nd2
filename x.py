
import sys
from pathlib import Path

import nd2
from nd2._pysdk._pysdk import LimFile
from rich import print

DATA = Path(__file__).parent / "tests" / "data"

if len(sys.argv) > 1:
    files = sys.argv[1:]
    verbose = True
else:
    OK = {
        "compressed_lossless.nd2",
        "dims_rgb_t3p2c2z3x64y64.nd2",
        "karl_sample_image.nd2",
    }
    files = [str(p) for p in DATA.glob("*.nd2") if p.name not in OK]
    verbose = True

for p in files:
    with LimFile(p) as lim:
        try:
            v = lim.version
        except Exception:
            continue
        lima = lim.attributes
        with nd2.ND2File(p) as ndf:
            nda = ndf.attributes
            lima = lima._replace(channelCount=nda.channelCount)
            nde = ndf.experiment
            lime = lim.experiment()
            nda = ndf.attributes
            limt = lim.text_info()
            ndt = ndf.text_info
            limm = lim.global_metadata()
            if lime != nde or lima != nda or limt != ndt:
                print("---------------------")
                print(f"{lim.version} {p}")
                if verbose:
                    print("lime", lime)
                    print("nde", nde)
                    print(ndf.sizes)
                else:
                    print(
                        f"mismatch {lim.version}",
                        p,
                    )
            else:
                print(f"ok {lim.version}", p)
