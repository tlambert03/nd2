cimport numpy as np
from cpython cimport Py_INCREF, PyObject, bool
from libc.stddef cimport wchar_t
from libc.stdlib cimport free, malloc

import numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

from ._sdk.v9 cimport *

from .structures import Attributes, Coordinate


cdef class ND2Reader:

    cdef LIMFILEHANDLE hFile
    cdef public path
    cdef public bint _is_legacy

    def __cinit__(self, path):
        self._is_legacy = 1
        self.hFile = 0
        self.path = path
        self.open()

    def __dealloc__(self):
        if self.hFile != 0:
            self.close()

    cpdef int open(ND2Reader self) except -1:
        if self.hFile != 0:
            return 1

        cdef Py_ssize_t length
        cdef wchar_t *w_filename = PyUnicode_AsWideCharString(self.path, &length)
        self.hFile = Lim_FileOpenForRead(w_filename)
        if not self.hFile:
            raise OSError("Could not open file: %s" % self.path)
        return 1

    cpdef void close(ND2Reader self):
        if self.hFile == 0:
            return
        self.hFile = 0
        Lim_FileClose(self.hFile)

    def is_open(self) -> bool:
        return bool(self.hFile)

    cpdef ND2Reader __enter__(ND2Reader self):
        self.open()
        return self

    cpdef void __exit__(ND2Reader self, exc_type, exc_val, exc_tb):
        self.close()

    cpdef attributes(ND2Reader self):
        cdef LIMATTRIBUTES attrs
        _rescheck(Lim_FileGetAttributes(self.hFile, &attrs))
        if attrs.uiCompression == 0:
            ctype = 'lossless'
            clevel = attrs.uiQuality
        elif attrs.uiCompression == 1:
            ctype = 'lossy'
            clevel = attrs.uiQuality
        else:
            ctype = None
            clevel = None

        # converting to new API
        return Attributes(
            bitsPerComponentInMemory = attrs.uiBpcInMemory,
            bitsPerComponentSignificant = attrs.uiBpcSignificant,
            componentCount = attrs.uiComp,
            heightPx = attrs.uiHeight,
            pixelDataType = "unsigned",
            sequenceCount = attrs.uiSequenceCount,
            widthBytes = attrs.uiWidthBytes,
            widthPx = attrs.uiWidth,
            compressionLevel = clevel,
            compressionType = ctype,
            tileHeightPx = attrs.uiTileHeight,
            tileWidthPx = attrs.uiTileWidth,
        )

    cpdef dict metadata(ND2Reader self):
        cdef LIMMETADATA_DESC meta
        _rescheck(Lim_FileGetMetadata(self.hFile, &meta))
        out = dict(meta)
        # TODO: remove all the type prefixes
        out['wszObjectiveName'] = _wchar2uni(meta.wszObjectiveName)
        out['pPlanes'] = []
        for desc in meta.pPlanes:
            if not desc.uiCompCount:
                break
            plane = dict(desc)
            plane['wszName'] = _wchar2uni(desc.wszName)
            plane['wszOCName'] = _wchar2uni(desc.wszOCName)
            out['pPlanes'].append(plane)

        return out

    cpdef list experiment(ND2Reader self):
        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        d = dict(levelCount=exp.uiLevelCount)
        out = []
        for i in range(exp.uiLevelCount):
            e = exp.pAllocatedLevels[i]
            out.append({
                'type': _EXP_TYPE[e.uiExpType],
                'count': e.uiLoopSize,
                'interval': e.dInterval,
            })
        return out


    cpdef text_info(ND2Reader self):
        cdef LIMTEXTINFO info
        _rescheck(Lim_FileGetTextinfo(self.hFile, &info))
        out = dict(
            imageID=_wchar2uni(info.wszImageID),
            type=_wchar2uni(info.wszType),
            group=_wchar2uni(info.wszGroup),
            sampleID=_wchar2uni(info.wszSampleID),
            author=_wchar2uni(info.wszAuthor),
            description=_wchar2uni(info.wszDescription),
            capturing=_wchar2uni(info.wszCapturing),
            sampling=_wchar2uni(info.wszSampling),
            location=_wchar2uni(info.wszLocation),
            date=_wchar2uni(info.wszDate),
            conclusion=_wchar2uni(info.wszConclusion),
            info1=_wchar2uni(info.wszInfo1),
            info2=_wchar2uni(info.wszInfo2),
            optics=_wchar2uni(info.wszOptics)
        )
        return out

    cpdef LIMUINT seq_count(ND2Reader self):
        cdef LIMATTRIBUTES attrs
        _rescheck(Lim_FileGetAttributes(self.hFile, &attrs))
        return attrs.uiSequenceCount

    cpdef LIMSIZE coord_size(self):
        """
        Returns the dimension of the frame coordinate vector.

        The dimension of the coordinate vector is equal to the number of experiment
        loop in the experiment.

        Zero means the file contains only one frame (not an ND document).
        """
        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        return exp.uiLevelCount

    cpdef list coord_info(ND2Reader self):
        """can be used to get information about the experiment loop.

        list of tuple of (coord_index, dim_type, dim_size)
        e.g. [(0, 'NETimeLoop', 4), (1, 'ZStackLoop', 5)]
        """

        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        d = dict(levelCount=exp.uiLevelCount)
        out = []
        for i in range(exp.uiLevelCount):
            e = exp.pAllocatedLevels[i]
            out.append(Coordinate(i, _EXP_TYPE[e.uiExpType], e.uiLoopSize))
        return out

    cpdef int seq_index_from_coords(ND2Reader self, coords) except -99:
        """Convert experiment coords to sequence index."""
        cdef LIMSIZE n = len(coords)
        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        if n != exp.uiLevelCount:
            raise ValueError("Coords must have length %s" % exp.uiLevelCount)

        cdef LIMUINT *_coords
        _coords = <LIMUINT *>malloc(n * sizeof(LIMUINT))
        try:
            for i in range(n):
                _coords[i] = coords[i]

            return Lim_GetSeqIndexFromCoords(&exp, _coords)
        finally:
            free(_coords)


    cpdef tuple coords_from_seq_index(self, LIMUINT seq_index):
        """Convert sequence index to experiment coords."""

        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        if exp.uiLevelCount == 0:
            return ()

        cdef LIMUINT *tmp = <LIMUINT *> malloc(exp.uiLevelCount * sizeof(LIMUINT))
        if not tmp:
            raise MemoryError()
        Lim_GetCoordsFromSeqIndex(&exp, seq_index, tmp)
        out = tuple([x for x in tmp[:exp.uiLevelCount]])
        free(tmp)
        return out

    cpdef np.ndarray data(ND2Reader self, LIMUINT seq_index=0):
        cdef LIMATTRIBUTES attrs
        _rescheck(Lim_FileGetAttributes(self.hFile, &attrs))

        if seq_index >= attrs.uiSequenceCount:
            raise IndexError("Sequence %d out of range (sequence count: %d)"
                             % (seq_index, attrs.uiSequenceCount))

        cdef LIMPICTURE pic = nullpic()
        cdef LIMSIZE size
        cdef LIMLOCALMETADATA pImgInfo

        size = Lim_InitPicture(&pic, attrs.uiWidth, attrs.uiHeight, attrs.uiBpcInMemory, attrs.uiComp)
        _rescheck(Lim_FileGetImageData(self.hFile, seq_index, &pic, &pImgInfo))

        array_wrapper = PicWrapper()
        array_wrapper.set_pic(pic)
        return array_wrapper.to_ndarray()

    cpdef tuple _voxel_size(self):
        cdef LIMMETADATA_DESC meta
        cdef LIMEXPERIMENT exp

        if not Lim_FileGetMetadata(self.hFile, &meta):
            xy = meta.dCalibration
        else:
            xy = 1.0

        z = 1.0
        if Lim_FileGetExperiment(self.hFile, &exp):
            for e in exp.pAllocatedLevels:
                if e.uiExpType == 2:  # Z stack loop
                    z = e.uiLoopSize
                    break
        return (xy, xy, z)

    cpdef get_stage_coords(self):
        cdef unsigned int n = 0
        cdef LIMEXPERIMENT exp
        _rescheck(Lim_FileGetExperiment(self.hFile, &exp))
        for i in range(exp.uiLevelCount):
            e = exp.pAllocatedLevels[i]
            if e.uiExpType == 1:  # XY loop
                n = e.uiLoopSize
                break
        if n == 0:
            return ()

        cdef LIMUINT *puiSeqIdx = <LIMUINT *> malloc(n * sizeof(LIMUINT))
        cdef LIMUINT *puiXPos = <LIMUINT *> malloc(n * sizeof(LIMUINT))
        cdef LIMUINT *puiYPos = <LIMUINT *> malloc(n * sizeof(LIMUINT))
        cdef double *pdXPos = <double *> malloc(n * sizeof(double))
        cdef double *pdYPos = <double *> malloc(n * sizeof(double))
        cdef double *pdZPos = <double *> malloc(n * sizeof(double))
        if not puiSeqIdx and puiXPos and puiYPos and pdXPos and pdYPos and pdZPos:
            raise MemoryError()

        for i in range(n):
            puiSeqIdx[i] = i
            puiXPos[i] = 0
            puiYPos[i] = 0
            pdXPos[i] = 0
            pdYPos[i] = 0
            pdZPos[i] = 0

        try:
            _rescheck(Lim_GetStageCoordinates(
                self.hFile, n,
                puiSeqIdx, puiXPos, puiYPos,
                pdXPos, pdYPos, pdZPos, 0
            ))

            out = []
            for i in range(n):
                # TODO: make StagePosition object
                out.append((pdXPos[i], pdYPos[i], pdZPos[i]))
            return tuple(out)

        finally:
            free(puiSeqIdx)
            free(puiXPos)
            free(puiYPos)
            free(pdXPos)
            free(pdYPos)
            free(pdZPos)

    # LIMRESULT Lim_GetMultipointName(LIMFILEHANDLE hFile,
    #                                         LIMUINT uiPointIdx,
    #                                         LIMWSTR wstrPointName)




    cpdef LIMINT _zstack_home(self):
        return Lim_GetZStackHome(self.hFile)

    cpdef LIMINT _custom_data_count(self):
        return Lim_GetCustomDataCount(self.hFile)


    # LIMRESULT Lim_GetRecordedDataInt(LIMFILEHANDLE hFile,
    #                                          LIMCWSTR wszName,
    #                                          LIMINT uiSeqIndex,
    #                                          LIMINT *piData)

    # LIMRESULT Lim_GetRecordedDataDouble(LIMFILEHANDLE hFile,
    #                                             LIMCWSTR wszName,
    #                                             LIMINT uiSeqIndex,
    #                                             double* pdData)

    # LIMRESULT Lim_GetRecordedDataString(LIMFILEHANDLE hFile,
    #                                             LIMCWSTR wszName,
    #                                             LIMINT uiSeqIndex,
    #                                             LIMWSTR wszData)



    # LIMRESULT Lim_GetCustomDataInfo(LIMFILEHANDLE hFile,
    #                                         LIMINT uiCustomDataIndex,
    #                                         LIMWSTR wszName,
    #                                         LIMWSTR wszDescription,
    #                                         LIMINT *piType,
    #                                         LIMINT *piFlags)

    # LIMRESULT Lim_GetCustomDataDouble(LIMFILEHANDLE hFile,
    #                                           LIMINT uiCustomDataIndex, # index of the Custom Metadata field
    #                                           double* pdData)           # will be filled by a double value

    # LIMRESULT Lim_GetCustomDataString(LIMFILEHANDLE hFile,
    #                                           LIMINT uiCustomDataIndex,
    #                                           LIMWSTR wszData,
    #                                           LIMINT *piLength)



    # LIMRESULT Lim_FileGetBinaryDescriptors(LIMFILEHANDLE hFile,
    #                                                LIMBINARIES* pBinaries)

    # LIMRESULT Lim_FileGetBinary(LIMFILEHANDLE hFile,
    #                                     LIMUINT uiSequenceIndex,
    #                                     LIMUINT uiBinaryIndex,
    #                                     LIMPICTURE* pPicture)

    # LIMRESULT Lim_GetLargeImageDimensions(LIMFILEHANDLE hFile,
    #                                               LIMUINT* puiXFields,
    #                                               LIMUINT* puiYFields,
    #                                               double* pdOverlap)

    # LIMRESULT Lim_GetAlignmentPoints(LIMFILEHANDLE hFile,
    #                                  LIMUINT* puiPosCount,
    #                                  LIMUINT* puiSeqIdx,
    #                                  LIMUINT* puiXPos,
    #                                  LIMUINT* puiYPos,
    #                                  double *pdXPos,
    #                                  double *pdYPos)

    # LIMRESULT Lim_GetNextUserEvent(LIMFILEHANDLE hFile,
    #                                        LIMUINT *puiNextID,
    #                                        LIMFILEUSEREVENT* pEventInfo)

    # LIMRESULT Lim_SetStageAlignment(LIMFILEHANDLE hFile,
    #                                         LIMUINT uiPosCount,
    #                                         double* pdXSrc,
    #                                         double* pdYSrc,
    #                                         double* pdXDst,
    #                                         double *pdYDst)

    # LIMRESULT Lim_FileGetImageRectData(LIMFILEHANDLE hFile,
    #                                    LIMUINT uiSeqIndex,
    #                                    LIMUINT uiDstTotalW,
    #                                    LIMUINT uiDstTotalH,
    #                                    LIMUINT uiDstX,
    #                                    LIMUINT uiDstY,
    #                                    LIMUINT uiDstW,
    #                                    LIMUINT uiDstH,
    #                                    void* pBuffer,
    #                                    LIMUINT uiDstLineSize,
    #                                    LIMINT iStretchMode,
    #                                    LIMLOCALMETADATA* pImgInfo)





_EXP_TYPE = {
    0: 'TimeLoop',
    1: 'XYPosLoop',
    2: 'ZStackLoop',
    3: 'Unknown',
}


_LIM_ERR_CODE = {
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

cdef void _rescheck(LIMRESULT result):
    if result != 0:
        # TODO: make error message editable
        msg = 'An error occured.'
        code = _LIM_ERR_CODE.get(result, str(result))
        raise RuntimeError('%s %s' % (msg, code))

####################################################
# (Duplicated from nd2file ... but de-dupping seemed to require mixing the sdks... which was confusing)

# based on https://gist.github.com/GaelVaroquaux/1249305
# from Gael Varoquaux License: BSD 3 clause
cdef class PicWrapper:
    cdef int dtype
    cdef LIMPICTURE pic

    cdef void set_pic(self, LIMPICTURE pic):
        if pic.uiBitsPerComp == 8:
            self.dtype = np.NPY_UINT8
        elif 8 < pic.uiBitsPerComp <= 16:
            self.dtype = np.NPY_UINT16
        elif pic.uiBitsPerComp == 32:
            self.dtype = np.NPY_UINT32
        else:
            raise ValueError("Unexpected bits per component: %d" % pic.uiBitsPerComp)

        self.pic = pic

    def __array__(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.pic.uiHeight
        shape[1] = <np.npy_intp> self.pic.uiWidth
        shape[2] = <np.npy_intp> self.pic.uiComponents
        array = np.PyArray_SimpleNewFromData(3, shape, self.dtype, self.pic.pImageData)
        return array.transpose((2, 0, 1))

    def __dealloc__(self):
        # free(<void*>self.data_ptr)
        Lim_DestroyPicture(&self.pic)

    cdef np.ndarray to_ndarray(self):
        cdef np.ndarray ndarray = np.array(self, copy=False)

        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> self

        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)
        return ndarray
