import sys
from pathlib import Path

import nd2
from nd2._pysdk import _pysdk 
from nd2._sdk.latest import ND2Reader
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
    verbose = False

if __name__ == "__main__":
    for p in files:
        with _pysdk.ND2Reader(p) as lim:
            try:
                v = lim.version
            except Exception:
                continue
            lima = lim.attributes
            with ND2Reader(p) as ndf:
                nda = ndf.attributes
                lima = lima._replace(channelCount=nda.channelCount)
                if lima != nda:
                    print(f"   {lim.version} mismatch attributes {p}")
                    continue
                limt = lim.text_info()
                ndt = ndf.text_info()
                if limt != ndt:
                    print(f"   {lim.version} mismatch text_info {p}")
                    continue
                lime = lim.experiment()
                nde = ndf.experiment()
                if lime != nde:
                    print(f"   {lim.version} mismatch experiment {p}")
                    breakpoint()
                    continue
                limm = lim.metadata()
                ndm = ndf.metadata()
                if limm != ndm:
                    print(f"   {lim.version} mismatch metadata {p}")
                    continue

                print(f"ok {lim.version}", p)
