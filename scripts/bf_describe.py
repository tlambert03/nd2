from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aicsimageio.readers.bioformats_reader import BioFile, BioformatsReader

DATA = Path(__file__).parent.parent / "tests" / "data"


def getinfo(path: Path):
    print(path)
    r = BioformatsReader(path)
    shp = dict(r.xarray_dask_data.sizes)
    if len(r.scenes) > 1:
        shp["S"] = len(r.scenes)
    with BioFile(path) as rdr:
        if rdr.core_meta.is_rgb:
            shp["c"] = 3
    return (path.name, {"shape": shp, "dtype": str(r.dtype)})


def get_bioformats_info():
    with ProcessPoolExecutor() as exc:
        data = {a: b for a, b in exc.map(getinfo, DATA.glob("*nd2"))}

    with open(DATA / "bf_shapes.json", "w") as fh:
        import json

        json.dump(data, fh)


if __name__ == "__main__":
    get_bioformats_info()
