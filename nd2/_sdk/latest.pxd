from libc.stddef cimport wchar_t

from .picture cimport LIMPICTURE


cdef extern from "Nd2ReadSdk.h":
    ctypedef void*           LIMFILEHANDLE
    ctypedef char            LIMCHAR
    ctypedef char*           LIMSTR
    ctypedef unsigned int    LIMUINT
    ctypedef int             LIMRESULT
    ctypedef size_t          LIMSIZE
    ctypedef bint            LIMBOOL
    ctypedef char*           LIMCSTR
    ctypedef const wchar_t*  LIMCWSTR

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
