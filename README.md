# nd2

[![License](https://img.shields.io/pypi/l/nd2.svg?color=green)](https://github.com/tlambert03/nd2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nd2.svg?color=green)](https://pypi.org/project/nd2)
[![Python Version](https://img.shields.io/pypi/pyversions/nd2.svg?color=green)](https://python.org)
[![Test](https://github.com/tlambert03/nd2/actions/workflows/test.yml/badge.svg)](https://github.com/tlambert03/nd2/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/tlambert03/nd2/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/nd2)

Yet another `.nd2` (Nikon NIS Elements) file reader.

This reader provides a Cython wrapper for the official Nikon SDK (currently version 1.7.0.0, released Jun 23, 2021).

Features good metadata retrieval, and direct `to_dask` and `to_xarray` options for lazy and/or annotated arrays.

It does not currently support legacy format nd2 files ([JPEG_XL](https://en.wikipedia.org/wiki/JPEG_XL) files starting with bytes `0x0000000c`)

## install

```sh
pip install nd2
```

*(linux support coming but not yet ready)*

## usage

```python
import nd2

# directly read file to numpy array:
my_array = nd2.imread('some_file.nd2')


# or open a file with ND2File
f = nd2.ND2File('some_file.nd2')

# attributes:   # example output
f.shape         # (10, 2, 256, 256)
f.ndim          # 4
f.dtype         # np.dtype('uint16')
f.axes          # 'TCYX'

# methods
f.pixel_size()  # (0.1, 0.1, 0.5) (x,y,z)
f.asarray()     # np.ndarray, greedy reader
f.to_dask()     # dask.array.Array, lazy reader
f.to_xarray()   # xr.DataArray, with labeled axes/coords

# metadata           # returns instance of ...
f.attributes()       # nd2.structures.Attributes
f.metadata()         # nd2.structures.Metadata
f.metadata(frame=3)  # nd2.structures.FrameMetadata
f.experiment()       # List[nd2.structures.ExpLoop]
f.text_info()        # dict of misc info

f.close()  # don't forget to close when done!

# ... or you can use it as a context manager
with nd2.ND2File('some_file.nd2') as ndfile:
    print(ndfile.metadata())
    xarr = ndfile.to_xarray(delayed=False)
```

## issues

This is a work in progress (though working pretty well for most files!). If you have an nd2 file that errors or otherwise provides unexpected results, please [open an issue](https://github.com/tlambert03/nd2/issues/new) and link to the file.  I'd love to add it to the growing list of test files.  (Note, if `nd2._util.is_old_format('your.nd2')` returns `True`, this is a legacy format that is not yet supported.)

## alternatives

- [pims_nd2](https://github.com/soft-matter/pims_nd2) - *pims-based reader. also uses the SDK, (but uses version v9 from Apr 2016)*
- [nd2reader](https://github.com/rbnvrw/nd2reader) - *pims-based reader, using reverse-engineered file headers. mostly tested on NIS Elements 4.30.02*
- [nd2file](https://github.com/csachs/nd2file) - *another pure-python, chunk map reader, developed 2017*
