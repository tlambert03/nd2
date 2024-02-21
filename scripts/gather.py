"""gather metadata from all files in test/data with all nd readers."""

import contextlib
import json
from pathlib import Path

import numpy as np


def get_nd2_stats(file) -> dict:
    from nd2 import ND2File

    try:
        fh = ND2File(file)
    except Exception as e:
        return {"err": str(e)}
    d = fh.attributes()._asdict()
    d["coords"] = [c._asdict() for c in fh._coord_info()]
    m = fh.metadata()
    if m and m.channels:
        d["pixel_size"] = m.channels[0].volume.axesCalibration
    d["shape"] = fh.shape
    d["axes"] = fh.axes
    with contextlib.suppress(Exception):
        d["dtype"] = str(fh.dtype)

    fh.close()
    return d


def get_bf_stats(file) -> dict:
    from aicsimageio.readers.bioformats_reader import BioFile

    d = {}
    with BioFile(file) as fh:
        meta = fh.core_meta
        d.update(meta._asdict())
        d["dtype"] = str(d["dtype"])
        d["series_count"] = int(d["series_count"])
        d["dimension_order"] = str(d["dimension_order"])
        d["resolution_count"] = int(d["resolution_count"])
        md = fh._r.getMetadataStore()
        xyz = (
            md.getPixelsPhysicalSizeX(0),
            md.getPixelsPhysicalSizeY(0),
            md.getPixelsPhysicalSizeZ(0),
        )
        d["pixel_size"] = tuple(i.value() if i else None for i in xyz)
    return d


def get_nd2reader_stats(file) -> dict:
    from nd2reader import ND2Reader

    try:
        fh = ND2Reader(file)
    except Exception as e:
        return {"err": str(e)}

    d = fh.metadata
    d["sizes"] = fh.sizes
    d["dtype"] = str(np.dtype(fh.pixel_type))
    d.pop("date")
    d.pop("z_coordinates", None)
    d.pop("rois", None)
    d.update(
        {
            "axes": fh.axes,
            "frame_rate": fh.frame_rate,
        }
    )
    for k, v in d.items():
        if isinstance(v, (range, np.ndarray)):
            d[k] = list(v)[-1]
    fh.close()
    return d


def get_pims_stats(file) -> dict:
    from pims_nd2 import ND2_Reader

    try:
        fh = ND2_Reader(file)
    except Exception as e:
        return {"err": str(e)}

    try:
        d = fh.metadata
    except Exception as e:
        fh.close()
        return {"err": str(e)}
    d["dtype"] = str(np.dtype(fh.pixel_type))

    d.pop("time_start_utc", None)
    d.pop("time_start", None)
    d["sizes"] = fh.sizes
    import json

    json.dumps(d)
    fh.close()
    return d


if __name__ == "__main__":
    D = {}
    for _f in Path("tests/data").glob("*.nd2"):
        f = str(_f)
        print(f)
        D[f] = {"bioformats": get_bf_stats(f)}
        D[f]["nd2"] = get_nd2_stats(f)
        D[f]["nd2reader"] = get_nd2reader_stats(f)
        D[f]["pims"] = get_pims_stats(f)

    with open("samples_meta.json", "w") as fh:
        json.dump(D, fh)
