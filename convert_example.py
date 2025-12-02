"""Example script to convert an ND2 file to OME-Zarr format."""

import shutil
import time

import nd2

nd2_path = "tests/data/dims_p4z5t3c2y32x32.nd2"
zarr_path = "output.zarr"

shutil.rmtree(zarr_path, ignore_errors=True)

with nd2.ND2File(nd2_path) as f:
    start = time.time()
    result = f.to_ome_zarr(zarr_path, backend="tensorstore")
    end = time.time()
    print(f"Converted {nd2_path} to {zarr_path} in {end - start:.2f} seconds.")
    print(f"Result: {result}")
