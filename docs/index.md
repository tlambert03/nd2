# Quickstart

[![License](https://img.shields.io/pypi/l/nd2.svg?style=flat-square&color=yellow)](https://github.com/tlambert03/nd2/raw/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nd2?style=flat-square&color=yellow)](https://pypi.org/project/nd2)
[![PyPI](https://img.shields.io/pypi/v/nd2.svg?style=flat-square&color=yellow)](https://pypi.org/project/nd2)
[![Conda](https://img.shields.io/conda/v/conda-forge/nd2?style=flat-square&color=yellow)](https://anaconda.org/conda-forge/nd2)

.nd2 (Nikon NIS Elements) file reader.

Features complete metadata retrieval, and many array outputs, including
[`to_dask()`][nd2.ND2File.to_dask] and [`to_xarray()`][nd2.ND2File.to_xarray]
options for lazy and/or annotated arrays (in addition to numpy arrays).

This library is thoroughly tested against many nd2 files with the goal of
maximizing compatibility and data extraction. (If you find an nd2 file that
fails in any way, please [open an
issue](https://github.com/tlambert03/nd2/issues/new) with the file!)

!!! Note
    This library is not affiliated with Nikon in any way, but we are
    grateful for assistance from the SDK developers at [Laboratory
    Imaging](https://www.lim.cz).

## Installation

From pip:

```sh
pip install nd2
```

From conda:

```sh
conda install -c conda-forge nd2
```

### With legacy nd2 file support

Legacy nd2 (JPEG2000) files are also supported, but require `imagecodecs`.  To
install with support for these files use the `legacy` extra:

```sh
pip install nd2[legacy]
```

### Faster XML parsing

Much of the metadata in the file stored as XML.  If found in the environment,
`nd2` will use [`lxml`](https://pypi.org/project/lxml/) which is faster than the
built-in `xml` module.  To install with support for `lxml` use:

```sh
pip install nd2 lxml
```

## Usage overview

For complete usage details, see the [API](API/nd2.md)

### Reading nd2 files into arrays

To quickly read an nd2 file into a numpy, dask, or xarray array,
use `nd2.imread()`:

```python
import nd2

# read to numpy array
my_array = nd2.imread('some_file.nd2')

# read to dask array
my_array = nd2.imread('some_file.nd2', dask=True)

# read to xarray
my_array = nd2.imread('some_file.nd2', xarray=True)

# read file to dask-xarray
my_array = nd2.imread('some_file.nd2', xarray=True, dask=True)
```

### Extracting metadata

If you want to get metadata, then use the [`nd2.ND2File`][] class directly:

```python
myfile = nd2.ND2File('some_file.nd2')
```

!!! tip
    It's best to use it as a context manager, so that the file is closed
    automatically when you're done with it.

    ```python
    with nd2.ND2File('some_file.nd2') as myfile:
        print(myfile.metadata)
        ...
    ```

The primary metadata is available as attributes on the file object:

The key metadata outputs are:

- [`ND2File.attributes`][nd2.ND2File.attributes]
- [`ND2File.metadata`][nd2.ND2File.metadata] / [`ND2File.frame_metadata()`][nd2.ND2File.frame_metadata]
- [`ND2File.experiment`][nd2.ND2File.experiment]
- [`ND2File.text_info`][nd2.ND2File.text_info]
- [`ND2File.events()`][nd2.ND2File.events]

Other attributes of note include:

| ATTRIBUTE          | EXAMPLE OUTPUT                           |
|--------------------|------------------------------------------|
| `myfile.shape`     | `(10, 2, 256, 256)`                      |
| `myfile.ndim`      | `4`                                      |
| `myfile.dtype`     | `np.dtype('uint16')`                     |
| `myfile.size`      | `1310720` (total voxel elements)         |
| `myfile.sizes`     | `{'T': 10, 'C': 2, 'Y': 256, 'X': 256}`  |
| `myfile.voxel_size()` | `VoxelSize(x=0.65, y=0.65, z=1.0)`    |
| `myfile.is_rgb`    | `False` (whether the file is rgb)        |

### Binary masks and ROIs

Binary masks, if present, can be accessed at
[`ND2File.binary_data`][nd2.ND2File.binary_data].

ROIs, if present, can be accessed at [`ND2File.rois`][nd2.ND2File.rois].

### There's more in there!

If you're still looking for something that you don't see in the above
properties and methods, try looking through:

- [`ND2File.custom_data`][nd2.ND2File.custom_data]
- [`ND2File.unstructured_metadata()`][nd2.ND2File.unstructured_metadata]

These methods parse and return more of the metadata found in the file,
but no attempt is made to extract it into a more useful form.

## Export nd2 to OME-TIFF

To convert an nd2 file to an OME-TIFF file, use [`nd2.ND2File.write_tiff`][] or
the convenience function `nd2.nd2_to_tiff`:

```python
import nd2


nd2.nd2_to_tiff('some_file.nd2', 'new_file.ome.tiff', progress=True)

# or with an ND2File object

with nd2.ND2File('some_file.nd2') as myfile:
    myfile.write_tiff('my_file.ome.tiff', progress=True)
```

Note that if you simply want the OME metadata, you can use the
[`ome_metadata()`][nd2.ND2File.ome_metadata] method to retrieve an instance of
[`ome_types.OME`][]:

```python
with nd2.ND2File('some_file.nd2') as myfile:
    ome_metadata = myfile.ome_metadata()
```
