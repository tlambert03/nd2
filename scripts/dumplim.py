import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
readlim = str(ROOT / "src" / "sdk" / "Darwin" / "x86_64" / "bin" / "readlimfile")
TEST_DATA = ROOT / "tests" / "data"

out: dict = {}
for f in TEST_DATA.glob("*.nd2"):
    subd = out.setdefault(f.name, {})

    try:
        if raw_meta := subprocess.check_output([readlim, "-x", str(f)]):
            subd["raw_metadata"] = json.loads(raw_meta[13:])
    except Exception:
        continue

    try:
        if exp := subprocess.check_output([readlim, "-e", str(f)]):
            subd["experiment"] = json.loads(exp[12:])
    except Exception:
        continue

    try:
        if global_meta := subprocess.check_output([readlim, "-m", str(f)]):
            subd["global_metadata"] = json.loads(global_meta.split(b"\n", 1)[1])
    except Exception:
        continue

    try:
        if frame0 := subprocess.check_output([readlim, "-f", "0", str(f)]):
            subd["frame0_metadata"] = json.loads(frame0.split(b"\n", 1)[1])
    except Exception:
        continue

    try:
        subd["indexes"] = subprocess.check_output([readlim, "-i", str(f)]).decode()
    except Exception:
        continue


with open(ROOT / "tests" / "readlim_output.json", "w") as dump:
    json.dump(out, dump, indent=2, sort_keys=True)
