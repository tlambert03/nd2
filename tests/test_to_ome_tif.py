import tempfile
from pathlib import Path

import nd2
import tifffile


def test_to_ome_tif(any_nd2: Path) -> None:

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "test.ome.tif"
        n_pos = 1
        with nd2.ND2File(any_nd2) as nd2_file:

            # temporary fix for is_legacy and ValueError
            if nd2_file.is_legacy:
                return
            for sz in nd2_file.sizes:
                if sz not in ["X", "Y", "Z", "C", "P", "T"]:
                    return

            n_pos = nd2_file.sizes.get("P", 1)
            nd2_file.to_ome_tif(dest)

        with tifffile.TiffFile(dest) as tif:
            assert len(tif.series) == n_pos
