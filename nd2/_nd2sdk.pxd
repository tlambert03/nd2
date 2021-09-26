import json

from libc.stddef cimport wchar_t


cdef extern from "Python.h":
    wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)


cdef extern from "nd2sdk.h":
    ctypedef void*           LIMFILEHANDLE
    ctypedef char            LIMCHAR
    ctypedef char*           LIMSTR
    ctypedef unsigned int    LIMUINT
    ctypedef int             LIMRESULT
    ctypedef size_t          LIMSIZE
    ctypedef int             LIMBOOL
    ctypedef char*           LIMCSTR
    ctypedef const wchar_t*  LIMCWSTR

    ctypedef struct LIMPICTURE:
        LIMUINT     uiWidth;             # !< Width (in pixels) of the picture
        LIMUINT     uiHeight;            # !< Height (in pixels) of the picture
        LIMUINT     uiBitsPerComp;       # !< Number of bits for each component
        LIMUINT     uiComponents;        # !< Number of components in each pixel
        LIMSIZE     uiWidthBytes;        # !< Number of bytes for each pixel line (stride); aligned to 4bytes
        LIMSIZE     uiSize;              # !< Number of bytes the image occupies
        void*       pImageData;          # !< Image data

    LIMFILEHANDLE Lim_FileOpenForReadUtf8(LIMCSTR wszFileName)
    LIMFILEHANDLE Lim_FileOpenForRead(LIMCWSTR wszFileName)

    void      Lim_FileClose(LIMFILEHANDLE hFile)

    LIMSTR    Lim_FileGetAttributes(LIMFILEHANDLE hFile)
    LIMSTR    Lim_FileGetMetadata(LIMFILEHANDLE hFile)
    LIMSTR    Lim_FileGetFrameMetadata(LIMFILEHANDLE hFile, LIMUINT uiSeqIndex)
    LIMSTR    Lim_FileGetTextinfo(LIMFILEHANDLE hFile)
    LIMSTR    Lim_FileGetExperiment(LIMFILEHANDLE hFile)
    LIMUINT   Lim_FileGetSeqCount(LIMFILEHANDLE hFile)

    LIMSIZE   Lim_FileGetCoordSize(LIMFILEHANDLE hFile)
    LIMBOOL   Lim_FileGetSeqIndexFromCoords(LIMFILEHANDLE hFile, const LIMUINT * coords, LIMSIZE coordCount, LIMUINT* seqIdx)
    LIMSIZE   Lim_FileGetCoordsFromSeqIndex(LIMFILEHANDLE hFile, LIMUINT seqIdx, LIMUINT* coords, LIMSIZE maxCoordCount)
    LIMUINT   Lim_FileGetCoordInfo(LIMFILEHANDLE hFile, LIMUINT coord, LIMSTR type, LIMSIZE maxTypeSize)

    LIMRESULT Lim_FileGetImageData(LIMFILEHANDLE hFile, LIMUINT uiSeqIndex, LIMPICTURE* pPicture)

    LIMSIZE   Lim_InitPicture(LIMPICTURE* pPicture, LIMUINT width, LIMUINT height, LIMUINT bpc, LIMUINT components)
    void      Lim_DestroyPicture(LIMPICTURE* pPicture)

    void      Lim_FileFreeString(LIMSTR str)


LIM_ERR_CODE = {
    0: 'LIM_OK',
    -1: 'LIM_ERR_UNEXPECTED',
    -2: 'LIM_ERR_NOTIMPL',  # NotImplementedError
    -3: 'LIM_ERR_OUTOFMEMORY',  # MemoryError
    -4: 'LIM_ERR_INVALIDARG',
    -5: 'LIM_ERR_NOINTERFACE',
    -6: 'LIM_ERR_POINTER',
    -7: 'LIM_ERR_HANDLE',
    -8: 'LIM_ERR_ABORT',
    -9: 'LIM_ERR_FAIL',
    -10: 'LIM_ERR_ACCESSDENIED',
    -11: 'LIM_ERR_OS_FAIL',  # OSError
    -12: 'LIM_ERR_NOTINITIALIZED',
    -13: 'LIM_ERR_NOTFOUND',
    -14: 'LIM_ERR_IMPL_FAILED',
    -15: 'LIM_ERR_DLG_CANCELED',
    -16: 'LIM_ERR_DB_PROC_FAILED',
    -17: 'LIM_ERR_OUTOFRANGE',  # IndexError
    -18: 'LIM_ERR_PRIVILEGES',
    -19: 'LIM_ERR_VERSION',
}

cdef inline LIMPICTURE nullpic():
    cdef LIMPICTURE p
    p.uiWidth = 0
    p.uiHeight = 0
    p.uiBitsPerComp = 0
    p.uiComponents = 0
    p.uiWidthBytes = 0
    p.uiSize = 0
    p.pImageData = NULL
    return p


cdef inline dict _loads(LIMSTR string):
    if not string:
        return {}
    try:
        return json.loads(string)
    finally:
        Lim_FileFreeString(string)


# cdef _open(str path):
#     cdef Py_ssize_t length
#     cdef wchar_t *w_filename = PyUnicode_AsWideCharString(path, &length)
#     return Lim_FileOpenForRead(w_filename)
