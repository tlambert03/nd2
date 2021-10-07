from libc.stddef cimport wchar_t

from .picture cimport LIMPICTURE


cdef extern from "Python.h":
    wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)
    object PyUnicode_FromWideChar(wchar_t*, Py_ssize_t size)

cdef inline object _wchar2uni(wchar_t* buffer):
    return PyUnicode_FromWideChar(buffer, -1)


cdef extern from "nd2ReadSDK.h":

    enum: LIMMAXPICTUREPLANES
    enum: LIMMAXEXPERIMENTLEVEL
    enum: LIMMAXBINARIES
    enum: LIM_OK
    enum: LIMLOOP_TIME
    enum: LIMLOOP_MULTIPOINT
    enum: LIMLOOP_Z
    enum: LIMLOOP_OTHER
    enum: LIMSTRETCH_QUICK
    enum: LIMSTRETCH_SPLINES
    enum: LIMSTRETCH_LINEAR

    ctypedef int            LIMFILEHANDLE  # int (not pointer) in legacy SDK
    ctypedef int            LIMRESULT # Result of operation
    ctypedef unsigned int   LIMUINT
    ctypedef int            LIMINT
    ctypedef bint           LIMBOOL
    ctypedef wchar_t        LIMWCHAR
    ctypedef wchar_t*       LIMWSTR
    ctypedef const wchar_t* LIMCWSTR
    ctypedef size_t         LIMSIZE

    ctypedef struct LIMATTRIBUTES:
        LIMUINT  uiWidth            # Width of images
        LIMUINT  uiWidthBytes       # Line length 4-byte aligned
        LIMUINT  uiHeight           # Height if images
        LIMUINT  uiComp             # Number of components
        LIMUINT  uiBpcInMemory      # Bits per component 8, 16 or 32 (for float image)
        LIMUINT  uiBpcSignificant   # Bits per component used 8 .. 16 or 32 (for float image)
        LIMUINT  uiSequenceCount    # Number of images in the sequence
        LIMUINT  uiTileWidth        # If an image is tiled size of the tile/strip
        LIMUINT  uiTileHeight       # otherwise both zero
        LIMUINT  uiCompression      # 0 (lossless), 1 (lossy), 2 (None)
        LIMUINT  uiQuality          # 0 (worst) - 100 (best)

    ctypedef struct LIMPICTUREPLANE_DESC:
        LIMUINT     uiCompCount     # Number of physical components
        LIMUINT     uiColorRGB      # RGB color for display
        LIMWCHAR    wszName[256]    # Name for display
        LIMWCHAR    wszOCName[256]  # Name of the Optical Configuration
        double      dEmissionWL

    ctypedef struct LIMMETADATA_DESC:
        double   dTimeStart          # Absolute Time in JDN
        double   dAngle              # Camera Angle
        double   dCalibration        # um/px (0.0 = uncalibrated)
        double   dAspect             # pixel aspect (always 1.0)
        LIMWCHAR wszObjectiveName[256]
        double   dObjectiveMag       # Optional additional information
        double   dObjectiveNA        # dCalibration takes into accont all these
        double   dRefractIndex1
        double   dRefractIndex2
        double   dPinholeRadius
        double   dZoom
        double   dProjectiveMag
        LIMUINT  uiImageType         # 0 (normal), 1 (spectral)
        LIMUINT  uiPlaneCount        # Number of logical planes (uiPlaneCount <= uiComponentCount)
        LIMUINT  uiComponentCount    # Number of physical components (same as uiComp in LIMFILEATTRIBUTES)
        LIMPICTUREPLANE_DESC pPlanes[LIMMAXPICTUREPLANES]

    ctypedef struct LIMTEXTINFO:
        LIMWCHAR wszImageID[256]
        LIMWCHAR wszType[256]
        LIMWCHAR wszGroup[256]
        LIMWCHAR wszSampleID[256]
        LIMWCHAR wszAuthor[256]
        LIMWCHAR wszDescription[4096]
        LIMWCHAR wszCapturing[4096]
        LIMWCHAR wszSampling[256]
        LIMWCHAR wszLocation[256]
        LIMWCHAR wszDate[256]
        LIMWCHAR wszConclusion[256]
        LIMWCHAR wszInfo1[256]
        LIMWCHAR wszInfo2[256]
        LIMWCHAR wszOptics[256]

    ctypedef struct LIMEXPERIMENTLEVEL:
        LIMUINT  uiExpType    # see LIMLOOP_TIME etc.
        LIMUINT  uiLoopSize   # Number of images in the loop
        double   dInterval    # ms (for Time), um (for ZStack), -1.0 (for Multipoint)

    ctypedef struct LIMEXPERIMENT:
        LIMUINT  uiLevelCount  # Number of dimensions excl. Lambda
        LIMEXPERIMENTLEVEL pAllocatedLevels[LIMMAXEXPERIMENTLEVEL]



    ctypedef struct LIMLOCALMETADATA:
        double   dTimeMSec             # Relative time msec from the first
        double   dXPos                 # Stage XPos
        double   dYPos                 # Stage YPos
        double   dZPos                 # Stage ZPos

    ctypedef struct LIMBINARYDESCRIPTOR:
        LIMWCHAR wszName[256]      # name of binary layer
        LIMWCHAR wszCompName[256]  # name of component to which it is bound (or empty string)
        LIMUINT uiColorRGB         # color of the layer

    ctypedef struct LIMBINARIES:
        LIMUINT             uiCount # number of binary layers
        LIMBINARYDESCRIPTOR pDescriptors[LIMMAXBINARIES]

    ctypedef struct LIMFILEUSEREVENT:
        LIMUINT    uiID
        double     dTime
        LIMWCHAR   wsType[128]
        LIMWCHAR   wsDescription[256]

    # This function reads the ND2 file.
    # It must be used before using the other functions.
    # When finished working with the ND2 file, Lim_FileClose should be called.
    LIMFILEHANDLE Lim_FileOpenForRead(LIMCWSTR wszFileName)

    # Use this function to close the current ND2 file when finished.
    LIMRESULT Lim_FileClose(LIMFILEHANDLE file_handle)

    # returns an array of attributes of the ND2 file. See LIMATTRIBUTES.
    LIMRESULT Lim_FileGetAttributes(LIMFILEHANDLE hFile,
                                    LIMATTRIBUTES* attr)

    # returns meta-data of the current ND2 file.
    # Not all members of the structure are necessarily filled.
    LIMRESULT Lim_FileGetMetadata(LIMFILEHANDLE hFile,
                                  LIMMETADATA_DESC* meta)

    # reads additional text info from the ND2 file.
    LIMRESULT Lim_FileGetTextinfo(LIMFILEHANDLE hFile,
                                  LIMTEXTINFO* info)

    # returns structure of the ND2 file or, if you like, the original ND experiment.
    LIMRESULT Lim_FileGetExperiment(LIMFILEHANDLE hFile,
                                    LIMEXPERIMENT* exp)

    # fills the LIMPICTURE structure.
    # Use the Lim_FileGetAttributes function to get correct values for the parameters.
    LIMSIZE Lim_InitPicture(LIMPICTURE* pPicture,
                              LIMUINT width,
                              LIMUINT height,
                              LIMUINT bpc,
                              LIMUINT components)

    # reads the image data of the current picture.
    # The Lim_InitPicture function must be called beforehand
    # to fill the LIMPICTURE structure.
    LIMRESULT Lim_FileGetImageData(LIMFILEHANDLE hFile,
                                   LIMUINT uiSeqIndex,
                                   LIMPICTURE* pPicture,
                                   LIMLOCALMETADATA* pImgInfo)

    # reads the current image and stores its image data
    # to the prepared memory buffer. It enables to resize the image to fit
    # the specified rectangle. The rectangle position (uiDstX, uiDstY) and
    # size (uiDstW, uiDstH) specifies placement of the resized image data
    # within the destination image (which has the size of uiDstTotalW, uiDstTotalH).
    # The Destination buffer must have the size of uiDstLineSize x uiDstTotalH.
    LIMRESULT Lim_FileGetImageRectData(LIMFILEHANDLE hFile,
                                       LIMUINT uiSeqIndex,
                                       LIMUINT uiDstTotalW,
                                       LIMUINT uiDstTotalH,
                                       LIMUINT uiDstX,
                                       LIMUINT uiDstY,
                                       LIMUINT uiDstW,
                                       LIMUINT uiDstH,
                                       void* pBuffer,
                                       LIMUINT uiDstLineSize,
                                       LIMINT iStretchMode,
                                       LIMLOCALMETADATA* pImgInfo)

    # deallocates memory of used to store information about the current
    # image. Use this function when finished working with the current ND2 file.
    void Lim_DestroyPicture(LIMPICTURE* pPicture)

    # returns sequence index of a frame based on the given coordinates
    # within the ND experiment structure.
    # inverse to the Lim_GetCoordsFromSeqIndex function
    LIMUINT Lim_GetSeqIndexFromCoords(LIMEXPERIMENT* pExperiment,
                                        LIMUINT* pExpCoords)

    # This function returns coordinates of a frame within the
    # ND experiment structure based on the sequence index.
    # inverse to Lim_GetSeqIndexFromCoords
    void Lim_GetCoordsFromSeqIndex(LIMEXPERIMENT* pExperiment,
                                        LIMUINT uiSeqIdx,
                                        LIMUINT* pExpCoords)

    # Returns an array of stage coordinates of the specified image pixels.
    # Stage coordinates are computed either from stage settings stored to
    # the file during acquisition or from the result of the Lim_SetStageAlignment function.
    LIMRESULT Lim_GetStageCoordinates(LIMFILEHANDLE hFile,
                                      LIMUINT uiPosCount,
                                      LIMUINT* puiSeqIdx,
                                      LIMUINT* puiXPos,
                                      LIMUINT* puiYPos,
                                      double* pdXPos,
                                      double *pdYPos,
                                      double *pdZPos,
                                      LIMINT iUseAlignment)


    # returns Z home position of the Z sequence, -1 if there is no Z dimension.
    LIMINT Lim_GetZStackHome(LIMFILEHANDLE hFile)


    # Get recorded integer data
    LIMRESULT Lim_GetRecordedDataInt(LIMFILEHANDLE hFile,
                                             LIMCWSTR wszName,
                                             LIMINT uiSeqIndex,
                                             LIMINT *piData)

    # Get recorded double data
    LIMRESULT Lim_GetRecordedDataDouble(LIMFILEHANDLE hFile,
                                                LIMCWSTR wszName,
                                                LIMINT uiSeqIndex,
                                                double* pdData)

    # Get recorded string data
    LIMRESULT Lim_GetRecordedDataString(LIMFILEHANDLE hFile,
                                                LIMCWSTR wszName,
                                                LIMINT uiSeqIndex,
                                                LIMWSTR wszData)

    # The function returns the number of Custom Metadata fields
    # defined in a file specified by hFile parameter.
    LIMINT Lim_GetCustomDataCount(LIMFILEHANDLE hFile)

    # returns information about the Custom Data field at index uiCustomDataIndex.
    # Index value is from zero to the number of fields returned by the
    # Lim_GetCustomDataCount function.
    LIMRESULT Lim_GetCustomDataInfo(LIMFILEHANDLE hFile,
                                            LIMINT uiCustomDataIndex,
                                            LIMWSTR wszName,
                                            LIMWSTR wszDescription,
                                            LIMINT *piType,
                                            LIMINT *piFlags)

    # returns numerical representation of the Custom Metadata field value.
    LIMRESULT Lim_GetCustomDataDouble(LIMFILEHANDLE hFile,
                                              LIMINT uiCustomDataIndex, # index of the Custom Metadata field
                                              double* pdData)           # will be filled by a double value

    # returns text representation of the Custom Metadata field value.
    # Maximal length of wszData buffer can be specified by the piLength parameter.
    # The piLength parameter will contain the desired length of the buffer on return.
    LIMRESULT Lim_GetCustomDataString(LIMFILEHANDLE hFile,
                                              LIMINT uiCustomDataIndex,
                                              LIMWSTR wszData,
                                              LIMINT *piLength)

    # retrieves user names of multi-point phases of the ND experiment.
    # (Default values are “#1”, “#2”, etc.)
    LIMRESULT Lim_GetMultipointName(LIMFILEHANDLE hFile,
                                            LIMUINT uiPointIdx,
                                            LIMWSTR wstrPointName)

    # reads information about binary layers contained within the ND2 file.
    LIMRESULT Lim_FileGetBinaryDescriptors(LIMFILEHANDLE hFile,
                                                   LIMBINARIES* pBinaries)

    # reads the contents of the binary layer.
    # The Lim_InitPicture function must be called beforehand
    # to fill the LIMPICTURE structure.
    LIMRESULT Lim_FileGetBinary(LIMFILEHANDLE hFile,
                                        LIMUINT uiSequenceIndex,
                                        LIMUINT uiBinaryIndex,
                                        LIMPICTURE* pPicture)

    # retrieves information about the large-image loop.
    LIMRESULT Lim_GetLargeImageDimensions(LIMFILEHANDLE hFile,
                                                  LIMUINT* puiXFields,
                                                  LIMUINT* puiYFields,
                                                  double* pdOverlap)

    # Retrieves an array of significant points from the image.
    # These points can be set to the image metadata within NIS-Elements
    # by calling the SetAlignmentPoints function.
    # If puiXPos, puiYPos, pdXPos, pdYPos are 0, the function can be
    # used just for retrieving the number of points (uiPosCount).
    LIMRESULT Lim_GetAlignmentPoints(LIMFILEHANDLE hFile,
                                     LIMUINT* puiPosCount,
                                     LIMUINT* puiSeqIdx,
                                     LIMUINT* puiXPos,
                                     LIMUINT* puiYPos,
                                     double *pdXPos,
                                     double *pdYPos)

    # Get next user event
    LIMRESULT Lim_GetNextUserEvent(LIMFILEHANDLE hFile,
                                           LIMUINT *puiNextID,
                                           LIMFILEUSEREVENT* pEventInfo)

    # Writes user stage settings for the current image to memory.
    # The settings are defined by several XY point pairs - original
    # stage coordinates (from the image) and destination (user) stage coordinates.
    # Stage coordinates of the current image set by this function can be retrieved
    # by the Lim_GetStageCoordinates function.
    LIMRESULT Lim_SetStageAlignment(LIMFILEHANDLE hFile,
                                            LIMUINT uiPosCount,
                                            double* pdXSrc,
                                            double* pdYSrc,
                                            double* pdXDst,
                                            double *pdYDst)
