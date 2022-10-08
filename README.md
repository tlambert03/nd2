# nd2

[![License](https://img.shields.io/pypi/l/nd2.svg?color=green)](https://github.com/tlambert03/nd2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nd2.svg?color=green)](https://pypi.org/project/nd2)
[![Python Version](https://img.shields.io/pypi/pyversions/nd2.svg?color=green)](https://python.org)
[![Tests](https://github.com/tlambert03/nd2/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/nd2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/nd2/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/nd2)

Yet another `.nd2` (Nikon NIS Elements) file reader.

This reader provides a Cython wrapper for the official Nikon SDK.
(The actual reading of image frames, however, uses a direct memmap approach, instead of the SDK,
for performance reasons and to avoid occasional segfaults from the SDK.)

Features good metadata retrieval, and direct `to_dask` and `to_xarray` options for lazy and/or annotated arrays.

This library is tested against many nd2 files with the goal of maximizing compatibility and data extraction.
(If you find an nd2 file that fails in some way, please open an issue with the file!)

## install

```sh
pip install nd2
```

Legacy nd2 (JPEG2000) files are also supported, but require `imagecodecs`.  To install with support for these files use:

```sh
pip install nd2[legacy]
```

## usage and API

```python
import nd2
import numpy as np

my_array = nd2.imread('some_file.nd2')                          # read to numpy array
my_array = nd2.imread('some_file.nd2', dask=True)               # read to dask array
my_array = nd2.imread('some_file.nd2', xarray=True)             # read to xarray
my_array = nd2.imread('some_file.nd2', xarray=True, dask=True)  # read file to dask-xarray

# or open a file with nd2.ND2File
f = nd2.ND2File('some_file.nd2')

# attributes:   # example output
f.path          # 'some_file.nd2'
f.shape         # (10, 2, 256, 256)
f.ndim          # 4
f.dtype         # np.dtype('uint16')
f.size          # 1310720  (total voxel elements)
f.sizes         # {'T': 10, 'C': 2, 'Y': 256, 'X': 256}
f.is_rgb        # False (whether the file is rgb)
# if RGB, sizes will have an additional {'S': 3} component

# array output
f.asarray()         # in-memory np.ndarray
np.asarray(f)       # alternative to f.asarray()
f.to_dask()         # delayed dask.array.Array
f.to_xarray()       # in-memory xarray.DataArray, with labeled axes/coords
f.to_xarray(delayed=True)   # delayed xarray.DataArray

                    # see below for examples of these structures
# metadata          # returns instance of ...
f.attributes        # nd2.structures.Attributes
f.metadata          # nd2.structures.Metadata
f.frame_metadata(0) # nd2.structures.FrameMetadata (frame-specific meta)
f.experiment        # List[nd2.structures.ExpLoop]
f.rois              # Dict[int, nd2.structures.ROI]
f.voxel_size()      # VoxelSize(x=0.65, y=0.65, z=1.0)
f.text_info         # dict of misc info

# allll the metadata we can find...
# no attempt made to standardize or parse it
# look in here if you're searching for metdata that isn't exposed in the above
f.unstructured_metadata()
f.custom_data       # bits of unstructured metadata that start with CustomData

f.close()           # don't forget to close when done!
f.closed            # boolean, whether the file is closed

# ... or you can use it as a context manager
with nd2.ND2File('some_file.nd2') as ndfile:
    print(ndfile.metadata)
    xarr = ndfile.to_xarray()
```

## Metadata structures

These follow the structure of the nikon SDK outputs.  Here are some example outputs

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

*Note: the `metadata` for legacy (JPEG2000) files will be a plain unstructured dict.*

```python
Metadata(
    contents=Contents(channelCount=2, frameCount=60),
    channels=[
        Channel(
            channel=ChannelMeta(name='Widefield Green', index=0, colorRGB=65371, emissionLambdaNm=535.0, excitationLambdaNm=None),
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
            channel=ChannelMeta(name='Widefield Red', index=1, colorRGB=22015, emissionLambdaNm=620.0, excitationLambdaNm=None),
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

<summary><code>custom_data</code></summary>

No attempt is made to parse this data.  It will vary from file to file, but you may find something useful here:

```python
{
    'StreamDataV1_0': {
        'Vector_StreamAnalogIn': '',
        'Vector_StreamDigitalIn': '',
        'Vector_AnalogIn': '',
        'Vector_DigitalIn': '',
        'Vector_Other': '',
        'Vector_StreamAnalogOut': '',
        'Vector_StreamDigitalOut': '',
        'Vector_AnalogOut': '',
        'Vector_DigitalOut': ''
    },
    'NDControlV1_0': {
        'NDControl': {
            'LoopState': {'no_name': [529, 529, 529, 529, 529]},
            'PlayFPS': {'no_name': [20.0, 20.0, 0.0, 20.0, 0.0]},
            'LoopSize': {'no_name': [3, 4, 0, 5, 0]},
            'LoopPosition': {'no_name': [2, 3, 0, 4, 0]},
            'LoopSelection': {'no_name': [b'AAAA', b'AAAAAA==', b'', b'AAAAAAA=', b'']},
            'LoopRangeSelection': {'no_name': [b'AQEB', b'AQEBAQ==', b'', b'AQEBAQE=', b'']},
            'LoopEventSelection': {'no_name': [b'AAAA', b'AAAAAA==', b'', b'AAAAAAA=', b'']},
            'FramesInRange': '',
            'LoopStep': {'no_name': [0, 0, 0, 0, 0]},
            'UserEventType': 2,
            'SelectionStyle': 0,
            'FramesBefore': 2,
            'FramesAfter': 1,
            'TimeBefore': 1.0,
            'TimeAfter': 1.0
        }
    },
    'LUTDataV1_0': {
        'ViewLut': True,
        'LutParam': {
            'Gradient': 0,
            'GradientBrightField': 0,
            'MinSrc': 0,
            'MaxSrc': 65535,
            'GammaSrc': 1.0,
            'MinDst': 0,
            'MaxDst': 65535,
            'ColorSpace': 4,
            'Representation': 0,
            'LutComponentCount': 2,
            'GroupCount': 1,
            'CompLutParam': {
                '00': {'MinSrc': [82, 0.0], 'MaxSrc': [113, 1.0], 'GammaSrc': 1.0, 'MinDst': 0, 'MaxDst': 65535, 'Group': 0},
                '01': {'MinSrc': [82, 0.0], 'MaxSrc': [114, 1.0], 'GammaSrc': 1.0, 'MinDst': 0, 'MaxDst': 65535, 'Group': 0},
                '02': {'MinSrc': [0, 0.0], 'MaxSrc': [65535, 1.0], 'GammaSrc': 1.0, 'MinDst': 0, 'MaxDst': 65535, 'Group': 0}
            },
            'LutDataSpectral': {
                'GainTrueColor': 1.0,
                'OffsetTrueColor': 0.0,
                'GainGrayScale': 1.0,
                'OffsetGrayScal': 0.0,
                'SpectralColorMode': 0,
                'Group00': {
                    'ColorGroup': 16711680,
                    'ColorCustom': 16711680,
                    'GainCustom': 1.0,
                    'OffsetCustom': 0.0,
                    'GainGrouped': 1.0,
                    'OffsetGrouped': 0.0
                }
            }
        },
        'EnableAutoContrast': True,
        'EnableAutoWhite': True,
        'AutoWhiteColor': 16777215,
        'RatioDesc': {
            'Numer': 0,
            'Denom': 1,
            'NumOffset': 0,
            'DenOffset': 0,
            'Min': 0.0,
            'Max': 2.0,
            'BkgndSize': 0,
            'Calibrated': True,
            'Cal.dKd': 224.0,
            'Cal.dVisc': 1.0,
            'Cal.dFmin': 255.0,
            'Cal.dFmax': 1.0,
            'Cal.dRmin': 0.0,
            'Cal.dRmax': 2.0,
            'Cal.dTMeasCalMin': 0.0,
            'Cal.dTMeasCalMax': 0.0,
            'PickFromGraph': True,
            'RatioViewEnabled': True
        },
        'GraphSelected': -1,
        'GraphVerticalSplit': True,
        'GrayGraph': True,
        'ShowAllComp': True,
        'ShowSpectralGraph': True,
        'GraphScale': 0,
        'GraphZoom00': 1.0,
        'GraphOffset00': 0.0,
        'GraphZoom01': 1.0,
        'GraphOffset01': 0.0,
        'GraphZoom02': 1.0,
        'GraphOffset02': 0.0
    },
    'GrabberCameraSettingsV1_0': {
        'GrabberCameraSettings': {
            'CameraUniqueName': 'Hamamatsu C11440-22C SN:101412',
            'CameraUserName': 'Flash4.0, SN:101412',
            'CameraFamilyName': 'ecmC11440_22C',
            'OverloadedUniqueName': '',
            'ModifiedAtJDN': 2459486.07103009,
            'FormatFast': {
                'Desc': {
                    'UniqueName': 'FMT 1x1 16',
                    'Interpretation': 1,
                    'FQModeUsage': 15,
                    'CanExecAsyncSampleGet': True,
                    'Fps': 30.00300030003,
                    'Sensitivity': 1.0,
                    'SensorPixels': {'cx': 2048, 'cy': 2044},
                    'SensorMicrons': {'cx': 13312, 'cy': 13286},
                    'SensorMin': {'cx': 4, 'cy': 4},
                    'SensorStep': {'cx': 2, 'cy': 2},
                    'BinningX': 1.0,
                    'BinningY': 1.0,
                    'SensorSource': {'left': 0, 'top': 0, 'right': 2048, 'bottom': 2044},
                    'FormatText': '16-bit - No Binning',
                    'FormatDesc': '16-bit - No Binning (30.0 FPS)',
                    'CamCorrReq': True,
                    'Comp': 1,
                    'Bpc': 16,
                    'UsageFlags': 1
                },
                'SensorUser': {'left': 512, 'top': 512, 'right': 544, 'bottom': 544}
            },
            'FormatQuality': {
                'Desc': {
                    'UniqueName': 'FMT 1x1 16',
                    'Interpretation': 1,
                    'FQModeUsage': 15,
                    'CanExecAsyncSampleGet': True,
                    'Fps': 30.00300030003,
                    'Sensitivity': 1.0,
                    'SensorPixels': {'cx': 2048, 'cy': 2044},
                    'SensorMicrons': {'cx': 13312, 'cy': 13286},
                    'SensorMin': {'cx': 4, 'cy': 4},
                    'SensorStep': {'cx': 2, 'cy': 2},
                    'BinningX': 1.0,
                    'BinningY': 1.0,
                    'SensorSource': {'left': 0, 'top': 0, 'right': 2048, 'bottom': 2044},
                    'FormatText': '16-bit - No Binning',
                    'FormatDesc': '16-bit - No Binning (30.0 FPS)',
                    'CamCorrReq': True,
                    'Comp': 1,
                    'Bpc': 16,
                    'UsageFlags': 1
                },
                'SensorUser': {'left': 512, 'top': 512, 'right': 544, 'bottom': 544}
            },
            'PropertiesFast': {
                'Exposure': 100.0,
                'LiveSpeedUp': 1,
                'CaptureQuality': 75,
                'CaptureMaxExposure': 10000.0,
                'QuantilRelative': True,
                'QuantilPromile': 0.1,
                'QuantilPixels': 100,
                'EnableAutoExposure': True,
                'ScanMode': 2,
                'Average': 1,
                'Integrate': 1,
                'AverageToQuality': 0.0,
                'AverageCH': '',
                'IntegrateCH': '',
                'AverageToQualityCH': '',
                'IntegrateToQualityCH': '',
                'FlexibleHeight': -1,
                'Negate': 0,
                'MultiExcitation': ''
            },
            'PropertiesFast_Extra': {'PropGroupCount': 0, 'PropGroupUsageArray': {}, 'PropGroupNameArray': {}},
            'PropertiesQuality': {
                'Exposure': 100.0,
                'LiveSpeedUp': 1,
                'CaptureQuality': 75,
                'CaptureMaxExposure': 10000.0,
                'QuantilRelative': True,
                'QuantilPromile': 0.1,
                'QuantilPixels': 100,
                'EnableAutoExposure': True,
                'ScanMode': 2,
                'Average': 1,
                'Integrate': 1,
                'AverageToQuality': 0.0,
                'AverageCH': '',
                'IntegrateCH': '',
                'AverageToQualityCH': '',
                'IntegrateToQualityCH': '',
                'FlexibleHeight': -1,
                'Negate': 0,
                'MultiExcitation': ''
            },
            'PropertiesQuality_Extra': {
                'PropGroupCount': 1,
                'PropGroupUsageArray': {'0': 0},
                'PropGroupNameArray': {'0': 'Use Stored ROI'}
            },
            'Metadata': {
                'Key': 'MV=0,TA=0,CH=1',
                'ChannelCount': 1,
                'Channels': {
                    'Channel_0': {
                        'Color': 22015,
                        'Name': 'Widefield Red',
                        'EmWavelength': 620.0,
                        'ChannelIsActive': True,
                        'ExWavelength': 540.5,
                        'MaxSaturatedValue': 4294967295
                    }
                }
            },
            'LightPath': {
                'TypeID': 0,
                'ExcitationSourceKey': 'LIGHT-EPI',
                'ExcitationSourceName': '',
                'EPIAdditionalFilterKey': '',
                'EPIAdditionalFilterName': '',
                'DIAAdditionalFilterKey': '',
                'DIAAdditionalFilterName': '',
                'LastEmissionFilterKey1': 'Turret-Lo',
                'LastEmissionFilterName1': 'Nikon Ti2, FilterChanger(Turret-Lo)',
                'SetColorManually': True,
                'MultiViewEnabled': True,
                'UpdateLPAutomatically': True
            },
            'ROI': {'Left': 512, 'Top': 512, 'Right': 544, 'Bottom': 544}
        },
        'GrabberCameraSettingsFQMode': 1
    },
    'CustomDataV2_0': {
        'CustomTagDescription_v1.0': {
            'Tag0': {'ID': 'Camera_ExposureTime1', 'Type': 3, 'Group': 2, 'Size': 60, 'Desc': 'Exposure Time', 'Unit': 'ms'},
            'Tag1': {'ID': 'PFS_OFFSET', 'Type': 2, 'Group': 1, 'Size': 60, 'Desc': 'PFS Offset', 'Unit': ''},
            'Tag2': {'ID': 'PFS_STATUS', 'Type': 2, 'Group': 1, 'Size': 60, 'Desc': 'PFS Status', 'Unit': ''},
            'Tag3': {'ID': 'X', 'Type': 3, 'Group': 1, 'Size': 60, 'Desc': 'X Coord', 'Unit': 'µm'},
            'Tag4': {'ID': 'Y', 'Type': 3, 'Group': 1, 'Size': 60, 'Desc': 'Y Coord', 'Unit': 'µm'},
            'Tag5': {'ID': 'Z', 'Type': 3, 'Group': 1, 'Size': 60, 'Desc': 'Z Coord', 'Unit': 'µm'},
            'Tag6': {'ID': 'Z1', 'Type': 3, 'Group': 1, 'Size': 60, 'Desc': 'Ti2 ZDrive', 'Unit': 'µm'}
        }
    },
    'AppInfo_V1_0': {
        'SWNameString': 'NIS-Elements AR',
        'GrabberString': 'Hamamatsu',
        'VersionString': '5.20.02 (Build 1453)',
        'CopyrightString': 'Copyright © 1991-2019  Laboratory Imaging,  http://www.lim.cz',
        'CompanyString': 'NIKON Corporation',
        'NFRString': ''
    },
    'AcqTimeV1_0': 2459486.07044662
}
```

</details>

## alternatives

- [pims_nd2](https://github.com/soft-matter/pims_nd2) - *pims-based reader. ctypes wrapper around the v9.00 (2015) SDK*
- [nd2reader](https://github.com/rbnvrw/nd2reader) - *pims-based reader, using reverse-engineered file headers. mostly tested on NIS Elements 4.30.02*
- [nd2file](https://github.com/csachs/nd2file) - *another pure-python, chunk map reader, unmaintained?*
- [pyND2SDK](https://github.com/aarpon/pyND2SDK) - *windows-only cython wrapper around the v9.00 (2015) SDK. not on PyPI*

The motivating factors for this library were:

- support for as many nd2 files as possible, with a large test suite
- pims-independent delayed reader based on dask
- axis-associated metadata via xarray
- combined approach of SDK and direct binary reads

## Contributing / Development

To test locally and contribute.  Clone this repo, then:

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
