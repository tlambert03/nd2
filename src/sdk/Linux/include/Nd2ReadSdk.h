#pragma once

#include <stdlib.h>

/*! \mainpage

## Overview

The Nd2ReadSdk is a set of C functions (and structures) declared in Nd2ReadSdk.h.

Nd2ReadSdk is a thin wrapper around proprietary limfile library. The SDK can access propritary ND2 files
as well as TIFFs with proprietary tags (equivalent to ND2 file). It uses pointers and JSON strings
to pass out image data and metadata. Json string can be easily parsed using 3rd party [libraries](https://github.com/nlohmann/json).

The SDK is available for following platforms:
- Windows 32 and 64 bit:
  - Nd2ReadSdkShared-A.B.C.D-win32.exe
  - Nd2ReadSdkStatic-A.B.C.D-win32.exe
  - Nd2ReadSdkShared-A.B.C.D-win64.exe
  - Nd2ReadSdkStatic-A.B.C.D-win64.exe
- Linux 64 bit:
  - Nd2ReadSdkStatic-A.B.C.D-Linux.rpm (built on Centos 7)
  - Nd2ReadSdkShared-A.B.C.D-Linux.deb (built on Ubuntu 18.04)
  - Nd2ReadSdkStatic-A.B.C.D-Linux.deb (built on Ubuntu 18.04)
- MacOS X is in development

A.B.C.D - refers to the SDK version.

The SDK requires (links with) following packages:
- limfile: Proprietary library to access Lim ND2 and TIFF files.
- libtiff: Provides support for the TIFF (<http://www.libtiff.org/>).
- zlib: Lossless data-compression library (<https://www.zlib.net/>).

The SDK and the above libraries are provided either as shared or static library.
If the target application using Nd2ReadSdk is built with static libs it is not necessary to install
the above libraries. The shared variant may be useful on systems where the libtiff and zlib is installed anyway 
(most Linux distributions). In that case nd2readsdk and limfile shared libraries must be installed together with
the target application.

Beside above mentioned libraries the SDK needs c++ run-time libraries which are part of the system or can be 
easily installed.


| operating system   | linking       | c++ run-time library                                      |
| ------------------ | ------------- | --------------------------------------------------------- |
| Windows 32bit      | static/shared | msvcp140.dll <sup>1</sup>                                 |
| Windows 64bit      | static/shared | msvcp140.dll <sup>1</sup>                                 |
| Centos 7 64bit     | static        | libstdc++.so.6<sup>2</sup>, GLIBCXX_3.4.19, CXXABI_1.3.7  |
| Ubuntu 18.04 64bit | static/shared | libstdc++.so.6<sup>2</sup>, GLIBCXX_3.4.25, CXXABI_1.3.11 |

1. The msvcp140.dll is part of [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

2. The libstdc++.so.6 is part of all recent Linux distributions. However, they differ in the version.
The SDK built Centos 7 requires lower version of c++ run-time and is suitable for older Linux systems.

## Opening file

The file must be first opened using Lim_FileOpenForRead() or Lim_FileOpenForReadUtf8() whereby a handle to the file is obtained. The handle is used in subsequent
SDK calls. When done the file must be closed with Lim_FileClose().

## Accessing image data

The main purpose of the SDK is to access image data. ND2 files can store many 2D images (frames) which 
form an experiment. A single frame is held and described using LIMPICTURE data structure.
Arbitrary frames can be read using Lim_FileGetImageData().

The range of sequence indexes goes from 0 to N which can be queried using Lim_FileGetSeqCount().

Every frame has a experiment coordinate represented as a vector of loop indexes. The dimension of loop indexes (size of 
the coordinate vector) depends on the number of loops in the experiment. For example if the experiment has only one loop
(e.g. time-lapse or z-stack) the coordinate dimension is one. On the other hand if the experiment is a z-stack inside a
time-lapse, the dimension is 2. Furthermore if both loops have 2 iterations (2 times and 2 z position) the total number
of frames is 4 and the coordinates are like so:

```
Seq   T, Z
0 -> [0, 0]
1 -> [0, 1]
2 -> [1, 0]
3 -> [1, 1]
```

To convert between sequence index (seqIndex) used by the SDK and experiment coordinates use
Lim_FileGetSeqIndexFromCoords() and Lim_FileGetCoordsFromSeqIndex() functions.

Lim_FileGetCoordSize() returns coordinate dimension, Lim_FileGetCoordInfo() 
can be used to get information about the loop at given coordinate index.

## Accessing image metadata

Image metadata are obtained as a JSON string as they are structured. It is convenient to use a JSON library 
(e.g. [this one](https://github.com/nlohmann/json)) to parse it. The string returned by following functions must 
be freed by Lim_FileFreeString().

There are four type of metadata in the ND2 file:
- attributes is the only one present in all files. It provides information about the image data similar to LIMPICTURE. It can be read with Lim_FileGetAttributes().

- experiment is an optional information describing the experiment loops in detail. It can be read with Lim_FileGetExperiment().

- metadata is an optional information describing global pixel calibration, axis interpretation as well as frame time and stage position. It can be read with Lim_FileGetMetadata() and Lim_FileGetFrameMetadata().

- textinfo is an optional information with catalogization fields like author, sample id etc. It can be read with Lim_FileGetTextinfo().


   ## nd2info example program

   The program reads ND2 and TIFFs. It outputs given metadata as JSON string to stdout.

   Following metadata can be read:
   - attributes (see Lim_FileGetAttributes())
   - dimensions (see Lim_FileGetCoordSize() and Lim_FileGetCoordInfo())
   - coordinates (see Lim_FileGetSeqCount() and Lim_FileGetCoordsFromSeqIndex())
   - experiment (see Lim_FileGetExperiment())
   - metadata global or per frame if sequence index is provided (see Lim_FileGetMetadata())
   - textinfo (see Lim_FileGetTextinfo())

   All image data can be copied line after line and frame after frame into given file (see Lim_FileGetImageData()).

   ~~~~~
   Usage: nd2info cmd [options] file.nd2
   cmd:
      allinfo <no options> - prints all metadata
      attributes <no options> - prints attributes
      coordinates <no options> - prints all frame coordinates
      dimensions <no options> - prints coordinate dimensions
      dumpallimages filename - makes superstack of all images
      experiment <no options> - prints experiment
      imageinfo <no options> - prints image size
      metadata <no options> - prints global metadata
      metadata seqIndex - prints global metadata merged with frame metadata
      textinfo <no options> - prints textinfo
   ~~~~~
*/

/*! \file Nd2ReadSdk.h

   \brief C Interface to access images and metadata stored in ND2 files.

   ND2 file contains image data, metadata and other assets.

   There are four metadata types that can be accessed using this SDK: attributes, experiment, metadata and textinfo.
   Metadata are return by the SDK as JSON string (the encoding is utf-8).

   Attributes Lim_FileGetAttributes() describe 2D image frames (width, height, frameCount, etc.) in the file.

   Experiment Lim_FileGetExperiment() describes dimensions (also referred to as acquisition loops) of the ND2 file and related information.
   There can be zero (single frame) or more dimensions (e.g. TimeLoop and ZStackLoop).

   Metadata Lim_FileGetMetadata and() Lim_FileGetFrameMetadata() describe global or per frame properties of the acquisition system.

   Image data are stored as 2D frames or images containing one or more color components (channels).
   Frames are accessed by Lim_FileGetImageData() and specifying frame sequential index.

   Sequential index can be conferted from and to coordinates (or logical loop indexes).
   It is useful to get 3rd Time index and 4th ZStack index [2, 3] (indexes are zero based).
*/

typedef char                     LIMCHAR;    //!< Multi-byte char for UTF-8 strings
typedef LIMCHAR*                 LIMSTR;     //!< Pointer to null-terminated multi-byte char array
typedef LIMCHAR const*           LIMCSTR;    //!< Pointer to null-terminated const multi-byte char array
typedef wchar_t                  LIMWCHAR;   //!< Wide-char (platform specific)
typedef LIMWCHAR*                LIMWSTR;    //!< Pointer to null-terminated wide-char array
typedef LIMWCHAR const*          LIMCWSTR;   //!< Pointer to null-terminated const wide-char array
typedef unsigned int             LIMUINT;    //!< Unsigned integer 32-bit
typedef unsigned long long       LIMUINT64;  //!< Unsigned integer 64-bit
typedef size_t                   LIMSIZE;    //!< Memory Size type
typedef int                      LIMINT;     //!< Integer 32-bit
typedef int                      LIMBOOL;    //!< Integer boolean value {0, 1}
typedef int                      LIMRESULT;  //!< Integer result codes

#define LIM_OK                    0
#define LIM_ERR_UNEXPECTED       -1
#define LIM_ERR_NOTIMPL          -2
#define LIM_ERR_OUTOFMEMORY      -3
#define LIM_ERR_INVALIDARG       -4
#define LIM_ERR_NOINTERFACE      -5
#define LIM_ERR_POINTER          -6
#define LIM_ERR_HANDLE           -7
#define LIM_ERR_ABORT            -8
#define LIM_ERR_FAIL             -9
#define LIM_ERR_ACCESSDENIED     -10
#define LIM_ERR_OS_FAIL          -11
#define LIM_ERR_NOTINITIALIZED   -12
#define LIM_ERR_NOTFOUND         -13
#define LIM_ERR_IMPL_FAILED      -14
#define LIM_ERR_DLG_CANCELED     -15
#define LIM_ERR_DB_PROC_FAILED   -16
#define LIM_ERR_OUTOFRANGE       -17
#define LIM_ERR_PRIVILEGES       -18
#define LIM_ERR_VERSION          -19
#define LIM_SUCCESS(res)         (0 <= (res))

/*!
\brief holds the pointer to the image data together with description of image data layout in memory

The interpretation is following:
- The smallest part of an image in LIMPICTURE is a component (e.g. Red in case of RGB image).
- The size of component is given by uiBitsPerComp (in bytes: (uiBitsPerComp + 7) / 8).<br>
  In practice it is either:
  - 8 bits in memory is a BYTE,
  - 9 ...16 bits in memory is a WORD; valid data are in the lower bits of the word if when less than 16 or
  - 32 bits in memory is a float or int; interpretation depends on pixelDataType member of attributes.
- The number of components uiComponents defines how many components make a pixel.
- Pixels are laid out in memory in lines (first line is first until uiHeight) and first pixel is first until uiWidth.
- Each line is aligned to 4 bytes. The offset between lines is uiWidthBytes bytes long. 

```
uiWidthBytes = (uiComponents * (uiBitsPerComp + 7) / 8 * uiWidth + 3) / 4 * 4 
pLine1 = (uint8_t*)pLine0 + uiWidthBytes;
```

For example:
In case of three components c0, c1, c2 each one 16bit WORD long the image data is in the memory as follows:
```
      pixel1 pixel2 ... pixelN 
Line0 c0c1c2 c0c1c2 ... c0c1c2
Line1 c0c1c2 c0c1c2 ... c0c1c2
                    ...
LineM c0c1c2 c0c1c2 ... c0c1c2
```

It is a good idea to always initialize the structure to zeros like so:
```
LIMPICTURE pic = { 0 }; // fills the whole struct with zeros
```

LIMPICTURE is initialized and allocated using Lim_InitPicture(). It must be always freed with Lim_DestroyPicture().

A frame is retrieved from a file with Lim_FileGetImageData().
*/
struct _LIMPICTURE
{
   LIMUINT     uiWidth;             //!< Width (in pixels) of the picture
   LIMUINT     uiHeight;            //!< Height (in pixels) of the picture
   LIMUINT     uiBitsPerComp;       //!< Number of bits for each component
   LIMUINT     uiComponents;        //!< Number of components in each pixel
   LIMSIZE     uiWidthBytes;        //!< Number of bytes for each pixel line (stride); aligned to 4bytes
   LIMSIZE     uiSize;              //!< Number of bytes the image occupies
   void*       pImageData;          //!< Image data
};

typedef struct _LIMPICTURE LIMPICTURE; //!< Picture description and data pointer
typedef void*  LIMFILEHANDLE;          //!< Opaque type representing an opened ND2 file

#if defined(_WIN32) && defined(_DLL) && !defined(LX_STATIC_LINKING)
#  define DLLEXPORT __declspec(dllexport)
#  define DLLIMPORT __declspec(dllimport)
#else
#  define DLLEXPORT
#  define DLLIMPORT
#endif

#if defined(__cplusplus)
#  define EXTERN extern "C"
#else
#  define EXTERN extern
#endif

#if defined(GNR_ND2_SDK_EXPORTS)
#  define LIMFILEAPI EXTERN DLLEXPORT
#else
#  define LIMFILEAPI EXTERN DLLIMPORT
#endif

/*!
\brief Opens an ND2 file for reading. This is wchar_t version.
\param[in] wszFileName The filename (system wchar_t) to be used.

Returns \c nullptr if the file does not exist or cannot be opened for read or is corrupted.
On success returns (non-null) \c LIMFILEHANDLE which must be closed with \c Lim_FileClose to deallocate resources.

\sa Lim_FileClose(), Lim_FileOpenForReadUtf8(LIMCSTR szFileNameUtf8)
*/
LIMFILEAPI LIMFILEHANDLE   Lim_FileOpenForRead(LIMCWSTR wszFileName);

/*!
\brief Opens an ND2 file for reading. This is multi-byte version (the encoding is UTF8).
\param[in] szFileNameUtf8 The filename (multi-byte UTF8 encoding) to be used.

Returns \c nullptr if the file does not exist or cannot be opened for read or is corrupted.
On succes returns (non-null) \c LIMFILEHANDLE which must be closed with \c Lim_FileClose to deallocate resources.
\sa Lim_FileClose(), Lim_FileOpenForRead(LIMCWSTR wszFilename)
*/
LIMFILEAPI LIMFILEHANDLE   Lim_FileOpenForReadUtf8(LIMCSTR szFileNameUtf8);

/*!
\brief Closes a file previously opened by this SDK.
\param[in] hFile The handle to an opened file.

If \a hFile is nullptr the function does nothing.

\sa Lim_FileOpenForReadUtf8(LIMCSTR szFileNameUtf8), Lim_FileOpenForRead(LIMCWSTR wszFilename)
*/
LIMFILEAPI void            Lim_FileClose(LIMFILEHANDLE hFile);

/*!
\brief Returns the dimension of the frame coordinate vector.
\param[in] hFile The handle to an opened file.

The dimension of the coordinate vector is equal to the number of experiment
loop in the experiment.

Zero means the file contains only one frame (not an ND document).
*/
LIMFILEAPI LIMSIZE         Lim_FileGetCoordSize(LIMFILEHANDLE hFile);

/*!
\brief Returns the loop size of a coordinate.
\param[in] hFile The handle to an opened file.
\param[in] coord The index of the coordinate.
\param[out] type Pointer to string buffer which receives the type.
\param[in] maxTypeSize Maximum number of chars the buffer can hold.

The \c coord must be lower than \c Lim_FileGetCoordSize().
If \a type is not nullptr it is filled by the name of the loop type: "Unknown", "TimeLoop", "XYPosLoop", "ZStackLoop", "NETimeLoop".
*/
LIMFILEAPI LIMUINT         Lim_FileGetCoordInfo(LIMFILEHANDLE hFile, LIMUINT coord, LIMSTR type, LIMSIZE maxTypeSize);

/*!
\brief Returns the number of frames in the file.
\param[in] hFile The handle to an opened file.
*/
LIMFILEAPI LIMUINT         Lim_FileGetSeqCount(LIMFILEHANDLE hFile);

/*!
\brief Converts coordinates into sequence index.
\param[in] hFile The handle to an opened file.
\param[in] coords The array of experiment coordinates.
\param[in] coordCount The number of experiment coordinates.
\param[out] seqIdx The pointer that is filled with corresponding sequence index.

If wrong argument is passed or the coordinate is not present in the file the function fails and returns 0.
On success it returns nonzero value.
*/
LIMFILEAPI LIMBOOL         Lim_FileGetSeqIndexFromCoords(LIMFILEHANDLE hFile, const LIMUINT * coords, LIMSIZE coordCount, LIMUINT* seqIdx);

/*!
\brief Converts sequence index into coordinates.
\param[in] hFile The handle to an opened file.
\param[in] seqIdx The sequence index.
\param[out] coords The array that is filled with experiment coordinates.
\param[in] maxCoordCount The maximum number of coordinates the array can hold.

On success it returns the number of coordinates filled.
If \c coords is \c nullptr the function only returns the dimension of coordinate required to store the result.
*/
LIMFILEAPI LIMSIZE         Lim_FileGetCoordsFromSeqIndex(LIMFILEHANDLE hFile, LIMUINT seqIdx, LIMUINT* coords, LIMSIZE maxCoordCount);

/*!
\brief Returns attributes as JSON (object) string.
\param[in] hFile The handle to an opened file.

Attributes are always present in the file and contain following members:

member                      | type               | description
--------------------------- | ------------------ | ---------------
bitsPerComponentInMemory    | number             | bits allocated to hold each component
bitsPerComponentSignificant | number             | bits effectively used by each component (not used bits must be zero)
componentCount              | number             | number of components in a pixel
compressionLevel            | number, optional   | if compression is used the level of compression
compressionType             | string, optional   | type of compression: "lossless" or "lossy"
heightPx                    | number             | height of the image
pixelDataType               | string             | underlying data type "unsigned" or "float"
sequenceCount               | number             | number of image frames in the file
tileHeightPx                | number, optional   | suggested tile height if saved as tiled
tileWidthPx                 | number, optional   | suggested tile width if saved as tiled
widthBytes                  | number             | number of bytes from the beginning of one line to the next one
widthPx                     | number             | width of the image

The memory size for image buffer is calculated as widthBytes * heightPx.

Returned string must be deleted using Lim_FileFreeString().

\sa Lim_FileFreeString()
*/
LIMFILEAPI LIMSTR          Lim_FileGetAttributes(LIMFILEHANDLE hFile);

/*!
\brief Returns metadata as JSON (object) string.
\param[in] hFile The handle to an opened file.

A JSON Object containing following fields:

JSON path to item     | where / different | description
--------------------- | ----------------- | ------------
/contents             | root (global)     | assets in the file (e.g. number of frames and channels)
/channels             | root (global)     | array of channels
/channel/C/channel    | per channel C     | channel related info (e.g. name, color)
/channel/C/loops      | per channel C     | loopname to loopindex map
/channel/C/microscope | per channel C     | relevant microscope settings (magnifications)
/channel/C/volume     | per channel C     | image data volume related information

_contents_ list the number of global assets in the file:
- channelCount determines the number of channels across all frames and
- frameCount determines the number of frames in the file.

_channels_ contains the array of channels where each contains:
- channel containing:
  - name of the channel,
  - index of the channel which uniquely identifies the channel in the file and
  - colorRGB defining the RGB color to show the channel in.
- loops containing a mapping from loop name into loop index
- microscope containing:
  - objectiveName
  - objectiveMagnification
  - objectiveNumericalAperture
  - projectiveMagnification
  - zoomMagnification
  - immersionRefractiveIndex
  - pinholeDiameterUm
- volume containing:
  - axesCalibrated contains 3 bools (XYZ) indicating which axes are calibrated
  - axesCalibration contains 3 doubles (XYZ) with calibration
  - axesInterpretation contains 3 strings (XYZ) defining the physical interpretation:
    - distance (default) axis is in microns (um) and calibration is in um/px
    - time axis is in milliseconds (ms) and calibration is in ms/px
  - bitsPerComponentInMemory
  - bitsPerComponentSignificant
  - componentCount
  - componentDataType is either unsigned or float
  - voxelCount contains 3 integers (XYZ) indicating the number of voxels in each direction
  - cameraTransformationMatrix a 2x2 matrix mapping camera space 
    (origin is the image center, X going right, Y down) to normalized stage space 
    (X going left, Y going up). It does not convert between pixels and um.
  - pixelToStageTransformationMatrix a 2x3 matrix which transforms pixel coordinates 
    (origin is the image top-left corner) to the actual device coordinates in um. 
    It does not add the image position to the coordinates.

NIS Microscope Absolute frame in um = pixelToStageTransformationMatrix * (X_in_px,  Y_in_px,  1) + stagePositionUm

NOTE: The whole object as well as all the member fields are optional.

Returned string must be deleted using Lim_FileFreeString().

\sa Lim_FileFreeString()
*/
LIMFILEAPI LIMSTR          Lim_FileGetMetadata(LIMFILEHANDLE hFile);

/*!
\brief Returns frame metadata as JSON (object) string.
\param[in] hFile The handle to an opened file.
\param[in] uiSeqIndex The frame sequence index.

A JSON Object containing the above Global metadata plus following fields:

JSON path to item     | where / different                 | description
--------------------- | --------------------------------- | ------------
/channel/C/position   | per channel (C) and current frame | frame position in :micro:m
/channel/C/time       | per channel (C) and current frame | frame time

_position_ holds position of the frame
- stagePositionUm contains 3 numbers (XYZ) indicating absolute position

_time_ holds information about the frame time
- relativeTimeMs relative time (to the beginning of the experiment) of the frame
- absoluteJulianDayNumber absolute time of the frame (see https://en.wikipedia.org/wiki/Julian_day)
- timerSourceHardware (if present) indicates the hardware used to capture the time (otherwise it is the software)

NOTE: The whole object as well as all the member fields are optional.

Returned string must be deleted using Lim_FileFreeString().

\sa Lim_FileFreeString()
*/
LIMFILEAPI LIMSTR          Lim_FileGetFrameMetadata(LIMFILEHANDLE hFile, LIMUINT uiSeqIndex);

/*!
\brief Returns text info as JSON (object) string.
\param[in] hFile The handle to an opened file.

Presence of the textinfo in the file as well as any field is optional.

Following fields are available:
- imageId
- type
- group
- sampleId
- author
- description
- capturing
- sampling
- location
- date
- conclusion
- info1
- info2
- optics

Returned string must be deleted using Lim_FileFreeString().

\sa Lim_FileFreeString()
*/
LIMFILEAPI LIMSTR          Lim_FileGetTextinfo(LIMFILEHANDLE hFile);

/*!
\brief Returns experiment as JSON (array) string.
\param hFile The handle to an opened file.

Presence of the experiment in the file as well as any field is optional.

Experiment is an array of loop objects. Each loop object contains info about the loop.
Each loop object contains:
- type defining the loop type (either "TimeLoop", "XYPosLoop", "ZStackLoop", "NETimeLoop"),
- count defining the number of iterations in the loop,
- nestingLevel defining the loop level and
- parameters describing the relevant experiment parameters.

_TimeLoop_ contains following items in parameters:
- startMs defining requested start of the sequence,
- periodMs defining requested period,
- durationMs defining requested duration and
- periodDiff which contains frame-to-frane statistics (average, maximum and minimum).

_NETimeLoop_ contains following items in parameters:
- periods which is a array of period information where each item contains:
   - count with the number of frames and
   - startMs, periodMs, durationMs and periodDiff as in TimeLoop

_XYPosLoop_ contains following items in parameters:
- isSettingZ defines if the Z position was set when visiting each point (otherwise only XY was set) and
- points which is an array of objects containing following members:
   - stagePositionUm defining the position of the point,
   - pfsOffset defining the PGFS offset and
   - name (optionally) contains the name of the point.

_ZStackLoop_ contains following items in parameters:
- homeIndex defines which index is home position,
- stepUm defines the distance between slices
- bottomToTop defines the acquisition direction
- deviceName (optionally) contains the name of the device used to acquire the z-stack.

Returned string must be deleted using Lim_FileFreeString().

\sa Lim_FileFreeString()
*/
LIMFILEAPI LIMSTR          Lim_FileGetExperiment(LIMFILEHANDLE hFile);

/*!
\brief Fills the \a pPicture with the frame indicated by the \a uiSeqIndex from the file.
\param hFile The handle to an opened file.
\param uiSeqIndex The the sequence index of the frame.
\param pPicture The pointer to \c LIMPICTURE structure that is filled with picture data.

If the \a pPicture is nullptr the function fails.

If the \c LIMPICTURE::pImageData and \c LIMPICTURE::uiSize members are zero the \c LIMPICTURE is properly initialized and allocated to correct size using \c Lim_InitPicture.

If the \a pPicture is already initialized but the size doesn't match the function fails.

The \a pPicture must be deleted using Lim_DestroyPicture().

\sa Lim_InitPicture(), Lim_DestroyPicture()
*/
LIMFILEAPI LIMRESULT       Lim_FileGetImageData(LIMFILEHANDLE hFile, LIMUINT uiSeqIndex, LIMPICTURE* pPicture);

/*!
\brief Deallocates the string returned by any metadata retrieving SDK function.
\param str The pointer to the string to be deallocated.

\sa Lim_FileGetAttributes(), Lim_FileGetExperiment(), Lim_FileGetMetadata(), Lim_FileGetFrameMetadata(), Lim_FileGetTextinfo()
*/
LIMFILEAPI void            Lim_FileFreeString(LIMSTR str);

/*!
\brief Initializes and allocates \a pPicture buffer to hold the image with given parameters.
\param pPicture The pointer `LIMPICTURE` structure to be initialized.
\param width The width (in pixels) of the picture.
\param height The height (in pixels) of the picture.
\param bpc The number of bits per each component (integer: 8-16 and floating: 32).
\param components The number of components in each pixel.

The parameters \a width, \a height \a bpc (bits per component) and \a components (number of color components in each pixel) are taken from attributes (Lim_FileGetAttributes()).

\sa Lim_DestroyPicture()
*/
LIMFILEAPI LIMSIZE         Lim_InitPicture(LIMPICTURE* pPicture, LIMUINT width, LIMUINT height, LIMUINT bpc, LIMUINT components);

/*!
\brief Deallocates resources allocated by \c Lim_InitPicture().
\param pPicture The pointer `LIMPICTURE` structure to be deallocated.

\sa Lim_InitPicture()
*/
LIMFILEAPI void            Lim_DestroyPicture(LIMPICTURE* pPicture);
