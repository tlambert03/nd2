# Wrapper for nd2ReadSDK.h
#
# Uses the Nikon SDK for accessing data and metadata from ND2 files.

from libc.stddef cimport wchar_t
from libc.stdint cimport uint16_t, uint8_t

cdef extern from "Python.h":
    wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)

cdef extern from "wchar.h":
    int wprintf(const wchar_t *, ...)

cdef extern from "nd2Reader_helper.h":

    ctypedef int LIMFILEHANDLE
    ctypedef unsigned int LIMUINT
    ctypedef int LIMINT

    # DEBUG functions
    void c_dump_LIMATTRIBUTES_struct(LIMATTRIBUTES *s)
    void c_dump_LIMMETADATA_DESC_struct(LIMMETADATA_DESC *s)
    void c_dump_LIMTEXTINFO_struct(LIMTEXTINFO *s)
    void c_dump_LIMPICTUREPLANE_DESC_struct(LIMPICTUREPLANE_DESC *s)
    void c_dump_LIMEXPERIMENTLEVEL_struct(LIMEXPERIMENTLEVEL *s)
    void c_dump_LIMEXPERIMENT_struct(LIMEXPERIMENT *s)
    void c_dump_LIMPICTURE_struct(LIMPICTURE *s)
    void c_dump_LIMLOCALMETADATA_struct(const LIMLOCALMETADATA *p)

    # Structure to  dictionary functions
    object c_LIMATTRIBUTES_to_dict(LIMATTRIBUTES *s)
    object c_LIMMETADATA_DESC_to_dict(LIMMETADATA_DESC *s)
    object c_LIMTEXTINFO_to_dict(LIMTEXTINFO * s)
    object c_LIMPICTUREPLANE_DESC_to_dict(LIMPICTUREPLANE_DESC *s)
    object c_LIMEXPERIMENTLEVEL_to_dict(LIMEXPERIMENTLEVEL * s)
    object c_LIMEXPERIMENT_to_dict(LIMEXPERIMENT * s)
    object c_LIMLOCALMETADATA_to_dict(const LIMLOCALMETADATA * s)
    object c_LIMBINARIES_to_dict(const LIMBINARIES * s)
    object c_LIMBINARYDESCRIPTOR_to_dict(const LIMBINARYDESCRIPTOR * s)
    object c_LIMFILEUSEREVENT_to_dict(const LIMFILEUSEREVENT * s)

    # Data functions
    float * c_get_float_pointer_to_picture_data(LIMPICTURE * p)
    uint16_t * c_get_uint16_pointer_to_picture_data(LIMPICTURE * p)
    uint8_t * c_get_uint8_pointer_to_picture_data(LIMPICTURE * p)
    void c_load_image_data(LIMFILEHANDLE hFile, LIMPICTURE *p, LIMLOCALMETADATA *m, LIMUINT uiSeqIndex, LIMINT iStretchMode)
    void c_to_rgb(LIMPICTURE *d, const LIMPICTURE *s)

    # Metadata functions
    object c_index_to_subscripts(LIMUINT seq_index, LIMEXPERIMENT *exp, LIMUINT *coords)
    LIMUINT c_subscripts_to_index(LIMEXPERIMENT *exp, LIMUINT *coords)
    object c_parse_stage_coords(LIMFILEHANDLE f, LIMATTRIBUTES a, int iUseAlignment)
    object c_get_recorded_data_int(LIMFILEHANDLE f, LIMATTRIBUTES a)
    object c_get_recorded_data_double(LIMFILEHANDLE f, LIMATTRIBUTES a)
    object c_get_recorded_data_string(LIMFILEHANDLE f, LIMATTRIBUTES a)
    object c_get_custom_data(LIMFILEHANDLE f_handle)
    object c_get_multi_point_names(LIMFILEHANDLE f, LIMUINT n)
    LIMUINT c_get_num_binary_descriptors(LIMFILEHANDLE f)
    object c_get_binary_descr(LIMFILEHANDLE f)
    object c_get_large_image_dimensions(LIMFILEHANDLE f)
    object c_get_alignment_points(LIMFILEHANDLE f)
    object c_get_user_events(LIMFILEHANDLE f)

cdef extern from "nd2ReadSDK.h":

    # Constants (i.e. #DEFINEs)
    #
    # Defining them as enum allows declarations such as array[number]
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

    # File handle
    ctypedef int LIMFILEHANDLE

    # Result of operation
    ctypedef int LIMRESULT

    # Unsigned integer
    ctypedef unsigned int LIMUINT

    # Signed integer
    ctypedef int LIMINT

    # "Bool"
    ctypedef int LIMBOOL

    # Wide char
    ctypedef wchar_t LIMWCHAR "wchar_t"

    # Wide char string (python unicode will be cast to it)
    ctypedef wchar_t* LIMWSTR "wchar_t *"

    # Constant pointer to wide char string (python unicode will be cast to it)
    ctypedef const wchar_t* LIMCWSTR "const wchar_t *"

    # Size (unsigned)
    ctypedef size_t LIMSIZE "size_t"

    # Struct LIMATTRIBUTES
    ctypedef struct LIMATTRIBUTES:
        pass

    # Struct LIMMETADATA_DESC
    ctypedef struct LIMMETADATA_DESC:
        pass

    # Struct LIMTEXTINFO
    ctypedef struct LIMTEXTINFO:
        pass

    # Struct LIMPICTUREPLANE_DESC
    ctypedef struct LIMPICTUREPLANE_DESC:
        pass

    # Struct LIMEXPERIMENTLEVEL
    ctypedef struct LIMEXPERIMENTLEVEL:
        pass

    # Struct LIMEXPERIMENT
    ctypedef struct LIMEXPERIMENT:
        pass

    # Struct LIMPICTURE
    ctypedef struct LIMPICTURE:
        pass

    # Struct LIMLOCALMETADATA
    ctypedef struct LIMLOCALMETADATA:
        pass

    # Struct LIMBINARIES
    ctypedef struct LIMBINARIES:
        pass

    # Struct LIMBINARYDESCRIPTOR
    ctypedef struct LIMBINARYDESCRIPTOR:
        pass

    # Struct LIMFILEUSEREVENT
    ctypedef struct LIMFILEUSEREVENT:
        pass

    # Open file for reading (and return file handle)
    LIMRESULT _Lim_FileOpenForRead \
            "Lim_FileOpenForRead"(LIMCWSTR wszFileName)

    # Close the file with given handle
    LIMRESULT _Lim_FileClose \
            "Lim_FileClose"(LIMFILEHANDLE file_handle)

    # Get the attributes
    LIMRESULT _Lim_FileGetAttributes \
            "Lim_FileGetAttributes"(LIMFILEHANDLE hFile,
                                    LIMATTRIBUTES* attr)

    # Get the metadata
    LIMRESULT _Lim_FileGetMetadata \
            "Lim_FileGetMetadata"(LIMFILEHANDLE hFile,
                                  LIMMETADATA_DESC* meta)

    # Get the the text info
    LIMRESULT _Lim_FileGetTextinfo \
            "Lim_FileGetTextinfo"(LIMFILEHANDLE hFile,
                                  LIMTEXTINFO* info)

    # Get the experiment
    LIMRESULT _Lim_FileGetExperiment \
            "Lim_FileGetExperiment"(LIMFILEHANDLE hFile,
                                    LIMEXPERIMENT* exp)

    # Initialize a picture
    LIMSIZE _Lim_InitPicture \
            "Lim_InitPicture"(LIMPICTURE* pPicture,
                              LIMUINT width,
                              LIMUINT height,
                              LIMUINT bpc,
                              LIMUINT components)

    # Get image data
    LIMRESULT _Lim_FileGetImageData \
            "Lim_FileGetImageData"(LIMFILEHANDLE hFile,
                                   LIMUINT uiSeqIndex,
                                   LIMPICTURE* pPicture,
                                   LIMLOCALMETADATA* pImgInfo)

    # Destroy a picture
    void _Lim_DestroyPicture \
            "Lim_DestroyPicture"(LIMPICTURE* pPicture)

    # Get sequence index from coordinates
    LIMUINT _Lim_GetSeqIndexFromCoords \
            "Lim_GetSeqIndexFromCoords"(LIMEXPERIMENT* pExperiment,
                                        LIMUINT* pExpCoords)

    # Get coordinates from sequence index
    void _Lim_GetCoordsFromSeqIndex \
            "Lim_GetCoordsFromSeqIndex"(LIMEXPERIMENT* pExperiment,
                                        LIMUINT uiSeqIdx,
                                        LIMUINT* pExpCoords)

    # Read the stage coordinates
    LIMRESULT _Lim_GetStageCoordinates \
            "Lim_GetStageCoordinates"(LIMFILEHANDLE hFile,
                                      LIMUINT uiPosCount,
                                      LIMUINT* puiSeqIdx,
                                      LIMUINT* puiXPos,
                                      LIMUINT* puiYPos,
                                      double* pdXPos,
                                      double *pdYPos,
                                      double *pdZPos,
                                      LIMINT iUseAlignment)


    # Get the Z stack home
    LIMINT _Lim_GetZStackHome \
            "Lim_GetZStackHome"(LIMFILEHANDLE hFile)


    # Get recorded integer data
    # @TODO This is currently unused. How to use it?
    LIMRESULT _Lim_GetRecordedDataInt \
                    "Lim_GetRecordedDataInt"(LIMFILEHANDLE hFile,
                                             LIMCWSTR wszName,
                                             LIMINT uiSeqIndex,
                                             LIMINT *piData)

    # Get recorded double data
    LIMRESULT _Lim_GetRecordedDataDouble \
                    "Lim_GetRecordedDataDouble"(LIMFILEHANDLE hFile,
                                                LIMCWSTR wszName,
                                                LIMINT uiSeqIndex,
                                                double* pdData)

    # Get recorded string data
    # @TODO This is currently unused. How to use it?
    LIMRESULT _Lim_GetRecordedDataString \
                    "Lim_GetRecordedDataString"(LIMFILEHANDLE hFile,
                                                LIMCWSTR wszName,
                                                LIMINT uiSeqIndex,
                                                LIMWSTR wszData)

    # Get count of custom data
    LIMINT _Lim_GetCustomDataCount \
                    "Lim_GetCustomDataCount"(LIMFILEHANDLE hFile)

    # Get custom data info
    LIMRESULT _Lim_GetCustomDataInfo \
                    "Lim_GetCustomDataInfo"(LIMFILEHANDLE hFile,
                                            LIMINT uiCustomDataIndex,
                                            LIMWSTR wszName,
                                            LIMWSTR wszDescription,
                                            LIMINT *piType,
                                            LIMINT *piFlags)

    # Get custom data value as double
    # @TODO This is currently unused. How to use it?
    LIMRESULT _Lim_GetCustomDataDouble \
                    "Lim_GetCustomDataDouble"(LIMFILEHANDLE hFile,
                                              LIMINT uiCustomDataIndex,
                                              double* pdData)

    # Get custom data value as string
    LIMRESULT _Lim_GetCustomDataString \
                    "Lim_GetCustomDataString"(LIMFILEHANDLE hFile,
                                              LIMINT uiCustomDataIndex,
                                              LIMWSTR wszData,
                                              LIMINT *piLength)

    # Get multi-point name
    LIMRESULT _Lim_GetMultipointName \
                    "Lim_GetMultipointName"(LIMFILEHANDLE hFile,
                                            LIMUINT uiPointIdx,
                                            LIMWSTR wstrPointName)

    # Get binary descriptors
    LIMRESULT _Lim_FileGetBinaryDescriptors \
                    "Lim_FileGetBinaryDescriptors"(LIMFILEHANDLE hFile,
                                                   LIMBINARIES* pBinaries)

    # Get binary image
    LIMRESULT _Lim_FileGetBinary \
                    "Lim_FileGetBinary"(LIMFILEHANDLE hFile,
                                        LIMUINT uiSequenceIndex,
                                        LIMUINT uiBinaryIndex,
                                        LIMPICTURE* pPicture)

    # Get large image dimensions
    LIMRESULT _Lim_GetLargeImageDimensions \
                    "Lim_GetLargeImageDimensions"(LIMFILEHANDLE hFile,
                                                  LIMUINT* puiXFields,
                                                  LIMUINT* puiYFields,
                                                  double* pdOverlap)

    # Read the alignment points
    LIMRESULT _Lim_GetAlignmentPoints \
            "Lim_GetAlignmentPoints"(LIMFILEHANDLE hFile,
                                     LIMUINT* puiPosCount,
                                     LIMUINT* puiSeqIdx,
                                     LIMUINT* puiXPos,
                                     LIMUINT* puiYPos,
                                     double *pdXPos,
                                     double *pdYPos)

    # Get next user event
    LIMRESULT _Lim_GetNextUserEvent \
                    "Lim_GetNextUserEvent"(LIMFILEHANDLE hFile,
                                           LIMUINT *puiNextID,
                                           LIMFILEUSEREVENT* pEventInfo)

    # Set stage alignment
    LIMRESULT _Lim_SetStageAlignment \
                    "Lim_SetStageAlignment"(LIMFILEHANDLE hFile,
                                            LIMUINT uiPosCount,
                                            double* pdXSrc,
                                            double* pdYSrc,
                                            double* pdXDst,
                                            double *pdYDst)
