# nd2

[![License](https://img.shields.io/pypi/l/nd2.svg?color=green)](https://github.com/tlambert03/nd2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nd2.svg?color=green)](https://pypi.org/project/nd2)
[![Python Version](https://img.shields.io/pypi/pyversions/nd2.svg?color=green)](https://python.org)
[![Tests](https://github.com/tlambert03/nd2/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/nd2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/nd2/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/nd2)
[![Benchmarks](https://img.shields.io/badge/⏱-codspeed-%23FF7B53)](https://codspeed.io/tlambert03/nd2)

`.nd2` (Nikon NIS Elements) file reader.

This reader provides a pure python implementation of the Nikon ND2 SDK.

> It _used_ to wrap the official SDK with Cython, but has since been completely
> rewritten to be pure python (for performance, ease of distribution, and
> maintenance) while retaining complete API parity with the official SDK.
>
> **Note:** This library is not affiliated with Nikon in any way, but we are
> grateful for assistance from the SDK developers at Laboratory Imaging.

Features good metadata retrieval, direct `to_dask` and `to_xarray` options
for lazy and/or annotated arrays, and output to OME-TIFF.

This library is tested against many nd2 files with the goal of maximizing
compatibility and data extraction. (If you find an nd2 file that fails in some
way, please [open an issue](https://github.com/tlambert03/nd2/issues/new) with
the file!)

### :book: [Documentation](https://tlambert03.github.io/nd2)

## install

```sh
pip install nd2
```

or from conda:

```sh
conda install -c conda-forge nd2
```

### Legacy nd2 file support

Legacy nd2 (JPEG2000) files are also supported, but require `imagecodecs`. To
install with support for these files use the `legacy` extra:

```sh
pip install nd2[legacy]
```

### Faster XML parsing

Much of the metadata in the file stored as XML. If found in the environment,
`nd2` will use [`lxml`](https://pypi.org/project/lxml/) which is much faster
than the built-in `xml` module. To install with support for `lxml` use:

```sh
pip install nd2 lxml
```

## Usage and API

Full API documentation is available at
[https://tlambert03.github.io/nd2](https://tlambert03.github.io/nd2)

Quick summary below:

```python
import nd2
import numpy as np

my_array = nd2.imread('some_file.nd2')                          # read to numpy array
my_array = nd2.imread('some_file.nd2', dask=True)               # read to dask array
my_array = nd2.imread('some_file.nd2', xarray=True)             # read to xarray
my_array = nd2.imread('some_file.nd2', xarray=True, dask=True)  # read to dask-xarray

# or open a file with nd2.ND2File
f = nd2.ND2File('some_file.nd2')

# (you can also use nd2.ND2File() as a context manager)
with nd2.ND2File('some_file.nd2') as ndfile:
    print(ndfile.metadata)
    ...


# ATTRIBUTES:   # example output
f.path          # 'some_file.nd2'
f.shape         # (10, 2, 256, 256)
f.ndim          # 4
f.dtype         # np.dtype('uint16')
f.size          # 1310720  (total voxel elements)
f.sizes         # {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
f.is_rgb        # False (whether the file is rgb)
                # if the file is RGB, `f.sizes` will have
                # an additional {'S': 3} component

# ARRAY OUTPUTS
f.asarray()         # in-memory np.ndarray - or use np.asarray(f)
f.to_dask()         # delayed dask.array.Array
f.to_xarray()       # in-memory xarray.DataArray, with labeled axes/coords
f.to_xarray(delayed=True)   # delayed xarray.DataArray

# OME-TIFF OUTPUT (new in v0.10.0)
f.write_tiff('output.ome.tif')  # write to ome-tiff file

                    # see below for examples of these structures
# METADATA          # returns instance of ...
f.attributes        # nd2.structures.Attributes
f.metadata          # nd2.structures.Metadata
f.frame_metadata(0) # nd2.structures.FrameMetadata (frame-specific meta)
f.experiment        # List[nd2.structures.ExpLoop]
f.text_info         # dict of misc info
f.voxel_size()      # VoxelSize(x=0.65, y=0.65, z=1.0)

f.rois              # Dict[int, nd2.structures.ROI]
f.binary_data       # any binary masks stored in the file.  See below.
f.events()          # returns tabular "Recorded Data" view from in NIS Elements/Viewer
                    # with info for each frame in the experiment.
                    # output is passabled to pandas.DataFrame

f.ome_metadata()    # returns metadata as an ome_types.OME object
                    # (requires ome-types package)

# allll the metadata we can find...
# no attempt made to standardize or parse it
# look in here if you're searching for metadata that isn't exposed in the above
# but try not to rely on it, as it's not guaranteed to be stable
f.unstructured_metadata()

f.close()           # don't forget to close when not using a context manager!
f.closed            # boolean, whether the file is closed
```

## Metadata structures

These follow the structure of the nikon SDK outputs (where relevant).
Here are some example outputs

<details>

<summary><code>attributes</code></summary>

```python
Attributes(
    bitsPerComponentInMemory=16,
    bitsPerComponentSignificant=16,
    componentCount=2,
    heightPx=32,
    pixelDataType='unsigned',
    sequenceCount=60,
    widthBytes=128,
    widthPx=32,
    compressionLevel=None,
    compressionType=None,
    tileHeightPx=None,
    tileWidthPx=None,
    channelCount=2
)
```

</details>

<details>

<summary><code>metadata</code></summary>

_Note: the `metadata` for legacy (JPEG2000) files will be a plain unstructured dict._

```python
Metadata(
    contents=Contents(channelCount=2, frameCount=60),
    channels=[
        Channel(
            channel=ChannelMeta(
                name='Widefield Green',
                index=0,
                color=Color(r=91, g=255, b=0, a=1.0),
                emissionLambdaNm=535.0,
                excitationLambdaNm=None
            ),
            loops=LoopIndices(NETimeLoop=None, TimeLoop=0, XYPosLoop=1, ZStackLoop=2),
            microscope=Microscope(
                objectiveMagnification=10.0,
                objectiveName='Plan Fluor 10x Ph1 DLL',
                objectiveNumericalAperture=0.3,
                zoomMagnification=1.0,
                immersionRefractiveIndex=1.0,
                projectiveMagnification=None,
                pinholeDiameterUm=None,
                modalityFlags=['fluorescence']
            ),
            volume=Volume(
                axesCalibrated=[True, True, True],
                axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                axesInterpretation=(
                    <AxisInterpretation.distance: 'distance'>,
                    <AxisInterpretation.distance: 'distance'>,
                    <AxisInterpretation.distance: 'distance'>
                ),
                bitsPerComponentInMemory=16,
                bitsPerComponentSignificant=16,
                cameraTransformationMatrix=[-0.9998932296054086, -0.014612644841559427, 0.014612644841559427, -0.9998932296054086],
                componentCount=1,
                componentDataType='unsigned',
                voxelCount=[32, 32, 5],
                componentMaxima=[0.0],
                componentMinima=[0.0],
                pixelToStageTransformationMatrix=None
            )
        ),
        Channel(
            channel=ChannelMeta(
                name='Widefield Red',
                index=1,
                color=Color(r=255, g=85, b=0, a=1.0),
                emissionLambdaNm=620.0,
                excitationLambdaNm=None
            ),
            loops=LoopIndices(NETimeLoop=None, TimeLoop=0, XYPosLoop=1, ZStackLoop=2),
            microscope=Microscope(
                objectiveMagnification=10.0,
                objectiveName='Plan Fluor 10x Ph1 DLL',
                objectiveNumericalAperture=0.3,
                zoomMagnification=1.0,
                immersionRefractiveIndex=1.0,
                projectiveMagnification=None,
                pinholeDiameterUm=None,
                modalityFlags=['fluorescence']
            ),
            volume=Volume(
                axesCalibrated=[True, True, True],
                axesCalibration=[0.652452890023035, 0.652452890023035, 1.0],
                axesInterpretation=(
                    <AxisInterpretation.distance: 'distance'>,
                    <AxisInterpretation.distance: 'distance'>,
                    <AxisInterpretation.distance: 'distance'>
                ),
                bitsPerComponentInMemory=16,
                bitsPerComponentSignificant=16,
                cameraTransformationMatrix=[-0.9998932296054086, -0.014612644841559427, 0.014612644841559427, -0.9998932296054086],
                componentCount=1,
                componentDataType='unsigned',
                voxelCount=[32, 32, 5],
                componentMaxima=[0.0],
                componentMinima=[0.0],
                pixelToStageTransformationMatrix=None
            )
        )
    ]
)
```

</details>

<details>

<summary><code>experiment</code></summary>

```python
[
    TimeLoop(
        count=3,
        nestingLevel=0,
        parameters=TimeLoopParams(
            startMs=0.0,
            periodMs=1.0,
            durationMs=0.0,
            periodDiff=PeriodDiff(avg=16278.339965820312, max=16411.849853515625, min=16144.830078125)
        ),
        type='TimeLoop'
    ),
    XYPosLoop(
        count=4,
        nestingLevel=1,
        parameters=XYPosLoopParams(
            isSettingZ=True,
            points=[
                Position(stagePositionUm=[26950.2, -1801.6000000000001, 498.46000000000004], pfsOffset=None, name=None),
                Position(stagePositionUm=[31452.2, -1801.6000000000001, 670.7], pfsOffset=None, name=None),
                Position(stagePositionUm=[35234.3, 2116.4, 664.08], pfsOffset=None, name=None),
                Position(stagePositionUm=[40642.9, -3585.1000000000004, 555.12], pfsOffset=None, name=None)
            ]
        ),
        type='XYPosLoop'
    ),
    ZStackLoop(count=5, nestingLevel=2, parameters=ZStackLoopParams(homeIndex=2, stepUm=1.0, bottomToTop=True, deviceName='Ti2 ZDrive'), type='ZStackLoop')
]
```

</details>

<details>

<summary><code>rois</code></summary>

ROIs found in the metadata are available at `ND2File.rois`, which is a
`dict` of `nd2.structures.ROI` objects, keyed by the ROI ID:

```python
{
    1: ROI(
        id=1,
        info=RoiInfo(
            shapeType=<RoiShapeType.Rectangle: 3>,
            interpType=<InterpType.StimulationROI: 4>,
            cookie=1,
            color=255,
            label='',
            stimulationGroup=0,
            scope=1,
            appData=0,
            multiFrame=False,
            locked=False,
            compCount=2,
            bpc=16,
            autodetected=False,
            gradientStimulation=False,
            gradientStimulationBitDepth=0,
            gradientStimulationLo=0.0,
            gradientStimulationHi=0.0
        ),
        guid='{87190352-9B32-46E4-8297-C46621C1E1EF}',
        animParams=[
            AnimParam(
                timeMs=0.0,
                enabled=1,
                centerX=-0.4228425369685782,
                centerY=-0.5194951478743071,
                centerZ=0.0,
                rotationZ=0.0,
                boxShape=BoxShape(
                    sizeX=0.21256931608133062,
                    sizeY=0.21441774491682075,
                    sizeZ=0.0
                ),
                extrudedShape=ExtrudedShape(sizeZ=0, basePoints=[])
            )
        ]
    ),
    ...
}
```

</details>

<details>

<summary><code>text_info</code></summary>

```python
{
    'capturing': 'Flash4.0, SN:101412\r\nSample 1:\r\n  Exposure: 100 ms\r\n  Binning: 1x1\r\n  Scan Mode: Fast\r\nSample 2:\r\n  Exposure: 100 ms\r\n  Binning: 1x1\r\n  Scan Mode: Fast',
    'date': '9/28/2021  9:41:27 AM',
    'description': 'Metadata:\r\nDimensions: T(3) x XY(4) x λ(2) x Z(5)\r\nCamera Name: Flash4.0, SN:101412\r\nNumerical Aperture: 0.3\r\nRefractive Index: 1\r\nNumber of Picture Planes: 2\r\nPlane #1:\r\n Name: Widefield Green\r\n Component Count: 1\r\n Modality: Widefield Fluorescence\r\n Camera Settings:   Exposure: 100 ms\r\n  Binning: 1x1\r\n  Scan Mode: Fast\r\n Microscope Settings:   Nikon Ti2, FilterChanger(Turret-Lo): 3 (FITC)\r\n  Nikon Ti2, Shutter(FL-Lo): Open\r\n  Nikon Ti2, Shutter(DIA LED): Closed\r\n  Nikon Ti2, Illuminator(DIA): Off\r\n  Nikon Ti2, Illuminator(DIA) Iris intensity: 3.0\r\n  Analyzer Slider: Extracted\r\n  Analyzer Cube: Extracted\r\n  Condenser: 1 (Shutter)\r\n  PFS, state: On\r\n  PFS, offset: 7959\r\n  PFS, mirror: Inserted\r\n  PFS, Dish Type: Glass\r\n  Zoom: 1.00x\r\n  Sola, Shutter(Sola): Active\r\n  Sola, Illuminator(Sola) Voltage: 100.0\r\nPlane #2:\r\n Name: Widefield Red\r\n Component Count: 1\r\n Modality: Widefield Fluorescence\r\n Camera Settings:   Exposure: 100 ms\r\n  Binning: 1x1\r\n  Scan Mode: Fast\r\n Microscope Settings:   Nikon Ti2, FilterChanger(Turret-Lo): 4 (TRITC)\r\n  Nikon Ti2, Shutter(FL-Lo): Open\r\n  Nikon Ti2, Shutter(DIA LED): Closed\r\n  Nikon Ti2, Illuminator(DIA): Off\r\n  Nikon Ti2, Illuminator(DIA) Iris intensity: 1.5\r\n  Analyzer Slider: Extracted\r\n  Analyzer Cube: Extracted\r\n  Condenser: 1 (Shutter)\r\n  PFS, state: On\r\n  PFS, offset: 7959\r\n  PFS, mirror: Inserted\r\n  PFS, Dish Type: Glass\r\n  Zoom: 1.00x\r\n  Sola, Shutter(Sola): Active\r\n  Sola, Illuminator(Sola) Voltage: 100.0\r\nTime Loop: 3\r\n- Equidistant (Period 1 ms)\r\nZ Stack Loop: 5\r\n- Step: 1 µm\r\n- Device: Ti2 ZDrive',
    'optics': 'Plan Fluor 10x Ph1 DLL'
}
```

</details>

<details>

<summary><code>binary_data</code></summary>

This property returns an `nd2.BinaryLayers` object representing all of the
binary masks in the nd2 file.

A `nd2.BinaryLayers` object is a sequence of individual `nd2.BinaryLayer`
objects (one for each binary layer found in the file). Each `BinaryLayer` in
the sequence is a named tuple that has, among other things, a `name` attribute,
and a `data` attribute that is list of numpy arrays (one for each frame in the
experiment) or `None` if the binary layer had no data in that frame.

The most common use case will be to cast either the entire `BinaryLayers` object
or an individual `BinaryLayer` to a `numpy.ndarray`:

```python
>>> import nd2
>>> nd2file = nd2.ND2File('path/to/file.nd2')
>>> binary_layers = nd2file.binary_data

# The output array will have shape
# (n_binary_layers, *coord_shape, *frame_shape).
>>> np.asarray(binary_layers)
```

For example, if the data in the nd2 file has shape `(nT, nZ, nC, nY, nX)`, and
there are 4 binary layers, then the output of `np.asarray(nd2file.binary_data)` will
have shape `(4, nT, nZ, nY, nX)`. (Note that the `nC` dimension is not present
in the output array, and the binary layers are always in the first axis).

You can also cast an individual `BinaryLayer` to a numpy array:

```python
>>> binary_layer = binary_layers[0]
>>> np.asarray(binary_layer)
```

</details>

<details>

<summary><code>events()</code></summary>

This property returns the tabular data reported in the `Image Properties >
Recorded Data` tab of the NIS Viewer.

(There will be a column for each tag in the `CustomDataV2_0` section of
`custom_data` above, as well as any additional events found in the metadata)

The format of the return type data is controlled by the `orient` argument:

- `'records'` : list of dicts - `[{column -> value}, ...]` (default)
- `'dict'` : dict of dicts - `{column -> {index -> value}, ...}`
- `'list'` : dict of lists - `{column -> [value, ...]}`

Not every column header appears in every event, so when `orient` is either
`'dict'` or `'list'`, `float('nan')` will be inserted to maintain a consistent
length for each column.

```python

# with `orient='records'` (DEFAULT)
[
    {
        'Time [s]': 1.32686654,
        'Z-Series': -2.0,
        'Exposure Time [ms]': 100.0,
        'PFS Offset': 0,
        'PFS Status': 0,
        'X Coord [µm]': 31452.2,
        'Y Coord [µm]': -1801.6,
        'Z Coord [µm]': 552.74,
        'Ti2 ZDrive [µm]': 552.74
    },
    {
        'Time [s]': 1.69089657,
        'Z-Series': -1.0,
        'Exposure Time [ms]': 100.0,
        'PFS Offset': 0,
        'PFS Status': 0,
        'X Coord [µm]': 31452.2,
        'Y Coord [µm]': -1801.6,
        'Z Coord [µm]': 553.74,
        'Ti2 ZDrive [µm]': 553.74
    },
    {
        'Time [s]': 2.04194662,
        'Z-Series': 0.0,
        'Exposure Time [ms]': 100.0,
        'PFS Offset': 0,
        'PFS Status': 0,
        'X Coord [µm]': 31452.2,
        'Y Coord [µm]': -1801.6,
        'Z Coord [µm]': 554.74,
        'Ti2 ZDrive [µm]': 554.74
    },
    {
        'Time [s]': 2.38194662,
        'Z-Series': 1.0,
        'Exposure Time [ms]': 100.0,
        'PFS Offset': 0,
        'PFS Status': 0,
        'X Coord [µm]': 31452.2,
        'Y Coord [µm]': -1801.6,
        'Z Coord [µm]': 555.74,
        'Ti2 ZDrive [µm]': 555.74
    },
    {
        'Time [s]': 2.63795663,
        'Z-Series': 2.0,
        'Exposure Time [ms]': 100.0,
        'PFS Offset': 0,
        'PFS Status': 0,
        'X Coord [µm]': 31452.2,
        'Y Coord [µm]': -1801.6,
        'Z Coord [µm]': 556.74,
        'Ti2 ZDrive [µm]': 556.74
    }
]

# with `orient='list'`
{
    'Time [s]': array([1.32686654, 1.69089657, 2.04194662, 2.38194662, 2.63795663]),
    'Z-Series': array([-2., -1.,  0.,  1.,  2.]),
    'Exposure Time [ms]': array([100., 100., 100., 100., 100.]),
    'PFS Offset': array([0, 0, 0, 0, 0], dtype=int32),
    'PFS Status': array([0, 0, 0, 0, 0], dtype=int32),
    'X Coord [µm]': array([31452.2, 31452.2, 31452.2, 31452.2, 31452.2]),
    'Y Coord [µm]': array([-1801.6, -1801.6, -1801.6, -1801.6, -1801.6]),
    'Z Coord [µm]': array([552.74, 553.74, 554.74, 555.74, 556.74]),
    'Ti2 ZDrive [µm]': array([552.74, 553.74, 554.74, 555.74, 556.74])
}

# with `orient='dict'`
{
    'Time [s]': {0: 1.32686654, 1: 1.69089657, 2: 2.04194662, 3: 2.38194662, 4: 2.63795663},
    'Z-Series': {0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0},
    'Exposure Time [ms]': {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
    'PFS Offset []': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'PFS Status []': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'X Coord [µm]': {0: 31452.2, 1: 31452.2, 2: 31452.2, 3: 31452.2, 4: 31452.2},
    'Y Coord [µm]': {0: -1801.6, 1: -1801.6, 2: -1801.6, 3: -1801.6, 4: -1801.6},
    'Z Coord [µm]': {0: 552.74, 1: 553.74, 2: 554.74, 3: 555.74, 4: 556.74},
    'Ti2 ZDrive [µm]': {0: 552.74, 1: 553.74, 2: 554.74, 3: 555.74, 4: 556.74}
}


```

You can pass the output of `events()` to `pandas.DataFrame`:

```python
In [1]: pd.DataFrame(nd2file.events())
Out[1]:
     Time [s]  Z-Series  Exposure Time [ms]  PFS Offset  PFS Status []  X Coord [µm]  Y Coord [µm]  Z Coord [µm]  Ti2 ZDrive [µm]
0    1.326867      -2.0               100.0              0              0       31452.2       -1801.6        552.74           552.74
1    1.690897      -1.0               100.0              0              0       31452.2       -1801.6        553.74           553.74
2    2.041947       0.0               100.0              0              0       31452.2       -1801.6        554.74           554.74
3    2.381947       1.0               100.0              0              0       31452.2       -1801.6        555.74           555.74
4    2.637957       2.0               100.0              0              0       31452.2       -1801.6        556.74           556.74
5    8.702229      -2.0               100.0              0              0       31452.2       -1801.6        552.70           552.70
6    9.036269      -1.0               100.0              0              0       31452.2       -1801.6        553.70           553.70
7    9.330319       0.0               100.0              0              0       31452.2       -1801.6        554.68           554.68
8    9.639349       1.0               100.0              0              0       31452.2       -1801.6        555.70           555.70
9    9.906369       2.0               100.0              0              0       31452.2       -1801.6        556.64           556.64
10  11.481439      -2.0               100.0              0              0       31452.2       -1801.6        552.68           552.68
11  11.796479      -1.0               100.0              0              0       31452.2       -1801.6        553.68           553.68
12  12.089479       0.0               100.0              0              0       31452.2       -1801.6        554.68           554.68
13  12.371539       1.0               100.0              0              0       31452.2       -1801.6        555.68           555.68
14  12.665469       2.0               100.0              0              0       31452.2       -1801.6        556.68           556.68

```

</details>

<details>

<summary><code>ome_metadata()</code></summary>

See the [ome-types documentation](https://ome-types.readthedocs.io/) for details on
the `OME` type returned by this method.

```python
In [1]: ome = nd2file.ome_metadata()

In [2]: print(ome)
OME(
    instruments=[<1 Instrument>],
    images=[<1 Image>],
    creator='nd2 v0.7.1'
)

In [3]: print(ome.to_xml())
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
     Creator="nd2 v0.7.1.dev2+g4ea166e.d20230709">
  <Instrument ID="Instrument:0">
    <Detector Model="Hamamatsu Dual C14440-20UP" SerialNumber="Hamamatsu Dual C14440-20UP" ID="Detector:0"/>
  </Instrument>
  <Image ID="Image:0" Name="test39">
    <AcquisitionDate>2023-07-08T09:30:55</AcquisitionDate>
    ...
```

</details>

## Contributing / Development

To test locally and contribute. Clone this repo, then:

```
pip install -e .[dev]
```

To download sample data:

```
pip install requests
python scripts/download_samples.py
```

then run tests:

```
pytest
```

(and feel free to open an issue if that doesn't work!)

## Fsspec Remote Reader

This fork adds high-performance streaming ND2 file access with support for remote files (HTTP, S3, GCS, Azure) and optimized 3D bounding box crop reading.

### Features

- **Streaming access**: Only downloads frames you request (no full file download)
- **Parallel I/O**: 2x+ speedup with ThreadPoolExecutor for Z-stack reading
- **Cloud support**: S3, GCS, Azure, HTTP via fsspec
- **3D Bounding Box Crop**: Read specific regions without downloading full frames
- **File List Optimization**: Pre-compute chunk offsets for repeated reads
- **Full Metadata Extraction**: Voxel sizes, time intervals, scene positions

### Installation

```bash
pip install nd2 fsspec aiohttp requests

# For S3 support
pip install s3fs

# For Google Cloud Storage
pip install gcsfs

# For Azure Blob Storage
pip install adlfs
```

### Quick Start

```python
from nd2 import ND2FsspecReader

# Local file (default: 64 parallel workers)
with ND2FsspecReader("/path/to/file.nd2") as reader:
    print(f"Shape: {reader.shape}")  # (T, C, Z, Y, X)
    print(f"Channels: {reader.channels}")
    zstack = reader.read_zstack(t=0, c=0)  # Uses 64 workers by default

# Remote file (HTTP/S3)
with ND2FsspecReader("https://example.com/data/file.nd2") as reader:
    zstack = reader.read_zstack(t=0, c=0)  # Parallel HTTP requests

# S3 file
with ND2FsspecReader("s3://bucket/path/file.nd2") as reader:
    zstack = reader.read_zstack(t=0, c=0)
```

### Full Metadata Extraction

```python
with ND2FsspecReader("file.nd2") as reader:
    # Dimensions
    print(f"Shape: {reader.shape}")        # (T, C, Z, Y, X)
    print(f"Sizes: {reader.sizes}")        # {'T': 10, 'C': 2, 'Z': 50, ...}

    # Physical metadata
    print(f"Voxel sizes (Z,Y,X): {reader.voxel_sizes} µm")  # (0.5, 0.1625, 0.1625)
    print(f"Time interval: {reader.time_interval} s")       # 60.0

    # Channel information
    print(f"Channels: {reader.channels}")      # ['DAPI', 'GFP']
    print(f"Channel map: {reader.channel_map}") # {'DAPI': 0, 'GFP': 1}

    # Scene/position information
    print(f"Number of scenes: {reader.n_scenes}")
    print(f"Scene positions: {reader.scene_positions}")

    # Full metadata object
    metadata = reader.metadata  # ImageMetadata dataclass
```

### 3D Bounding Box Crop

Read specific Z-slice ranges efficiently:

```python
with ND2FsspecReader("file.nd2") as reader:
    # Define crop region: (z_min, z_max, y_min, y_max, x_min, x_max)
    crop = (10, 40, 500, 1500, 500, 1500)

    # Read cropped Z-stack (only fetches Z slices 10-40)
    cropped = reader.read_zstack(t=0, c=0, crop=crop)
    print(f"Cropped shape: {cropped.shape}")  # (30, 1000, 1000)
```

**Note on cropping behavior:**
- **Z cropping**: Efficient - only the specified Z slices are fetched
- **XY cropping**: Full frames are fetched and cropped in memory (ND2 frames are compressed, so partial byte-range reads aren't possible)

### File List Optimization

Pre-compute chunk offsets for fast repeated reads of the same region:

```python
from nd2 import ND2FsspecReader, ND2FileList

# Generate file list
with ND2FsspecReader("s3://bucket/large_file.nd2") as reader:
    file_list = reader.generate_file_list(
        crop=(10, 40, 500, 1500, 500, 1500),
        t_range=(0, 5),
        c_range=(0, 2),
        s_range=(0, 1),
    )
    # Save for later use
    file_list.save("file_list.json")
    print(f"Output shape: {file_list.output_shape}")

# Later: read using pre-computed offsets
file_list = ND2FileList.load("file_list.json")
with ND2FsspecReader(file_list.path) as reader:
    data = reader.read_from_file_list(file_list, max_workers=16)
```

### Dask Integration

```python
with ND2FsspecReader("large_file.nd2") as reader:
    darr = reader.to_dask()
    print(f"Dask array: {darr.shape}, chunks={darr.chunks}")

    # Compute only what you need
    max_proj = darr[0, 0].max(axis=0).compute()
```

### ND2FsspecReader API

```python
class ND2FsspecReader:
    # Properties
    path: str                              # File path/URL
    shape: tuple[int, int, int, int, int]  # (T, C, Z, Y, X)
    sizes: dict[str, int]                  # {'T': ..., 'C': ..., ...}
    dtype: np.dtype                        # Data type (e.g., uint16)
    channels: list[str]                    # Channel names
    channel_map: dict[str, int]            # Channel name -> index
    voxel_sizes: tuple[float, float, float] | None  # (Z, Y, X) in µm
    time_interval: float | None            # Time between frames in seconds
    n_scenes: int                          # Number of scenes/positions
    scene_positions: list[tuple[float, float]]  # Stage positions
    metadata: ImageMetadata                # Full metadata object
    is_remote: bool                        # Whether file is remote

    # Methods
    def read_frame(t=0, c=0, z=0, s=0) -> np.ndarray
    def read_zstack(t=0, c=0, s=0, crop=None, max_workers=64) -> np.ndarray
    def generate_file_list(crop=None, t_range=None, c_range=None, s_range=None) -> ND2FileList
    def read_from_file_list(file_list, max_workers=64) -> np.ndarray
    def to_dask(chunks=None) -> dask.array.Array
    def asarray(max_workers=64) -> np.ndarray
```

### Performance Optimizations

**Metadata Caching:**
- The ND2 chunk map is read once during initialization and cached
- No repeated metadata requests when reading multiple frames/Z-stacks
- For local files, uses `ND2File` for reliable metadata extraction

**Parallel I/O:**
- Default: 64 parallel workers for maximum throughput
- Each Z-slice is fetched in a separate thread
- For remote files, uses HTTP connection pooling with keep-alive

**Efficient Z Cropping:**
- Only fetches the Z-slices within the specified range
- Skips frames outside the crop region entirely

### Benchmark Results

From testing with 35GB ND2 files on a 10 Gbit network:

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| nd2.ND2File (local) | 0.5 Gbit/s | Sequential reads |
| ND2FsspecReader (1 worker) | 0.5 Gbit/s | Same as ND2File |
| ND2FsspecReader (8 workers) | 1.3 Gbit/s | 2.6x speedup |
| ND2FsspecReader (64 workers) | 2.5+ Gbit/s | Default, max throughput |
| ND2FsspecReader (64 workers, S3) | 3.0+ Gbit/s | Cloud optimized |

## alternatives

Here are some other nd2 readers that I know of, though many
of them are unmaintained:

- [pims_nd2](https://github.com/soft-matter/pims_nd2) - _pims-based reader.
  ctypes wrapper around the v9.00 (2015) SDK_
- [nd2reader](https://github.com/rbnvrw/nd2reader) - _pims-based reader, using
  reverse-engineered file headers. mostly tested on files from NIS Elements
  4.30.02_
- [nd2file](https://github.com/csachs/nd2file) - _another pure-python, chunk map
  reader, unmaintained?_
- [pyND2SDK](https://github.com/aarpon/pyND2SDK) - _windows-only cython wrapper
  around the v9.00 (2015) SDK. not on PyPI_

The motivating factors for this library were:

- support for as many nd2 files as possible, with a large test suite
  an and emphasis on correctness
- pims-independent delayed reader based on dask
- axis-associated metadata via xarray
