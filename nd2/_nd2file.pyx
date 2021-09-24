import json

from cpython cimport Py_INCREF, PyObject, bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc

import numpy as np

cimport numpy as np

from .structures import (
    Attributes,
    Coordinate,
    FrameMetadata,
    ImageInfo,
    Metadata,
    parse_experiment,
)

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# ########### Externs ################

cdef extern from "Nd2ReadSdk.h":
    ctypedef void*          LIMFILEHANDLE
    ctypedef char           LIMCHAR
    ctypedef char*          LIMSTR
    ctypedef unsigned int   LIMUINT
    ctypedef int            LIMRESULT
    ctypedef size_t         LIMSIZE
    ctypedef int            LIMBOOL
    ctypedef char*          LIMCSTR

    ctypedef struct LIMPICTURE:
        LIMUINT     uiWidth;             # !< Width (in pixels) of the picture
        LIMUINT     uiHeight;            # !< Height (in pixels) of the picture
        LIMUINT     uiBitsPerComp;       # !< Number of bits for each component
        LIMUINT     uiComponents;        # !< Number of components in each pixel
        LIMSIZE     uiWidthBytes;        # !< Number of bytes for each pixel line (stride); aligned to 4bytes
        LIMSIZE     uiSize;              # !< Number of bytes the image occupies
        void*       pImageData;          # !< Image data

    LIMFILEHANDLE Lim_FileOpenForReadUtf8(LIMCSTR wszFileName)

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

    # LIMFILEHANDLE Lim_FileOpenForRead(LIMCWSTR wszFileName)

# from libc.stddef cimport wchar_t

# cdef extern from "Python.h":
    # wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)

# cdef _open(str path):
#     cdef Py_ssize_t length
#     cdef wchar_t *w_filename = PyUnicode_AsWideCharString(path, &length)
#     return Lim_FileOpenForRead(w_filename)

# ########### structures ################

cdef LIMPICTURE nullpic():
    cdef LIMPICTURE p
    p.uiWidth = 0
    p.uiHeight = 0
    p.uiBitsPerComp = 0
    p.uiComponents = 0
    p.uiWidthBytes = 0
    p.uiSize = 0
    p.pImageData = <void*> 0
    return p

ERR_CODE ={
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

# ########### classes ################

# based on https://gist.github.com/GaelVaroquaux/1249305
# from Gael Varoquaux License: BSD 3 clause
cdef class PicWrapper:

    cdef int dtype
    cdef LIMPICTURE pic

    cdef set_pic(self, LIMPICTURE pic):
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

cdef dict _todict(LIMSTR string):
    if not string:
        return {}
    try:
        return json.loads(string)
    finally:
        Lim_FileFreeString(string)



cdef class CND2File:

    cdef LIMFILEHANDLE hFile
    cdef public path

    def __cinit__(self, path):
        self.hFile = <void*> 0
        self.path = path
        self.open()

    def __dealloc__(self):
        self.close()

    cpdef open(CND2File self):
        if not self.is_open():
            self.hFile = Lim_FileOpenForReadUtf8(self.path)
            if <uintptr_t> self.hFile == 0:
                raise OSError("Could not open file: %s" % self.path)
            self._is_open = 1

    cpdef void close(CND2File self):
        if self.is_open():
            Lim_FileClose(self.hFile)
            self.hFile = <void*> 0

    cpdef bool is_open(CND2File self):
        return bool(<uintptr_t> self.hFile != 0)

    cpdef CND2File __enter__(CND2File self):
        self.open()
        return self

    cpdef void __exit__(CND2File self, exc_type, exc_val, exc_tb):
        self.close()

    cpdef attributes(CND2File self):
        d = _todict(Lim_FileGetAttributes(self.hFile))
        return Attributes(**d)

    # TODO: decide on "frame" vs "seq_index"
    cpdef metadata(CND2File self, int frame=-1, int format=1):
        if frame >=0:
            return self._frame_metadata(frame, format)
        d = _todict(Lim_FileGetMetadata(self.hFile))
        if not d:
            return {}
        if format:
            return Metadata(**d)
        return d

    cdef _frame_metadata(CND2File self, LIMUINT frame, int format=1):
        self._validate_seq(frame)
        d = _todict(Lim_FileGetFrameMetadata(self.hFile, frame))
        if not d:
            return {}
        if format:
            return FrameMetadata(**d)
        return d

    # cpdef image_info(CND2File self, LIMUINT seq_index=0):
    #     """named tuple with (width, heigh, components, bits)
    #     e.g. ImageInfo(width=696, height=520, components=1, bits_per_pixel=14)
    #     """
    #     self._validate_seq(seq_index)
    #     cdef LIMPICTURE pic = nullpic()
    #     result = Lim_FileGetImageData(self.hFile, seq_index, &pic)
    #     out = ImageInfo(pic.uiWidth, pic.uiHeight, pic.uiComponents, pic.uiBitsPerComp)
    #     Lim_DestroyPicture(&pic)
    #     return out

    cpdef list experiment(CND2File self, int format=1):
        cdef LIMSTR s = Lim_FileGetExperiment(self.hFile)
        if not s:
            out = []
        else:
            out = json.loads(s)
            Lim_FileFreeString(s)
            if format:
                out = parse_experiment(out)
        return out

    cpdef dict text_info(CND2File self):
        return _todict(Lim_FileGetTextinfo(self.hFile))

    cdef _validate_seq(CND2File self, LIMUINT seq_index):
        if seq_index >= self.seq_count():
            raise IndexError("Sequence %d out of range (sequence count: %d)"
                             % (seq_index, self.seq_count()))

    cpdef LIMUINT seq_count(CND2File self):
        return Lim_FileGetSeqCount(self.hFile)

    # coords
    cpdef LIMSIZE coord_size(self):
        """
        Returns the dimension of the frame coordinate vector.

        The dimension of the coordinate vector is equal to the number of experiment
        loop in the experiment.

        Zero means the file contains only one frame (not an ND document).
        """
        return Lim_FileGetCoordSize(self.hFile)

    cpdef int seq_index_from_coords(CND2File self, coords):
        """Convert experiment coords to sequence index.

        Returns -1 if coord_size == 0.

        e.g.
         T, Z     Seq
        (0, 0) -> 0
        (0, 1) -> 1
        (1, 0) -> 2
        (1, 1) -> 3
        """
        cdef LIMSIZE size = Lim_FileGetCoordSize(self.hFile)
        if size == 0:
            return -1

        cdef LIMSIZE n = len(coords)

        cdef LIMUINT *_coords
        cdef LIMUINT seq_index
        _coords = <LIMUINT *>malloc(n * sizeof(unsigned int))
        try:
            for i in range(n):
                _coords[i] = coords[i]

            if not Lim_FileGetSeqIndexFromCoords(self.hFile, _coords, n, &seq_index):
                raise ValueError("Coordinate %r has no sequence index" % coords)
            return seq_index
        finally:
            free(_coords)

    cpdef tuple coords_from_seq_index(self, LIMUINT seq_index):
        """Convert sequence index to experiment coords.

        e.g.
        Seq   T, Z
        0 -> (0, 0)
        1 -> (0, 1)
        2 -> (1, 0)
        3 -> (1, 1)
        """
        cdef LIMSIZE size = Lim_FileGetCoordSize(self.hFile)
        if size == 0:
            return ()

        cdef LIMUINT *output = <LIMUINT *> malloc(size * sizeof(LIMUINT))
        if not output:
            raise MemoryError()
        try:
            Lim_FileGetCoordsFromSeqIndex(self.hFile, seq_index, output, size)
            return tuple([x for x in output[:size]])
        finally:
            free(output)

    cpdef list coord_info(CND2File self):
        """can be used to get information about the experiment loop.

        list of tuple of (coord_index, dim_type, dim_size)
        e.g. [(0, 'NETimeLoop', 4), (1, 'ZStackLoop', 5)]
        """
        cdef LIMSIZE size = Lim_FileGetCoordSize(self.hFile)
        if size == 0:
            return []
        out = []
        cdef LIMCHAR buffer[1024]
        for i in range(size):
            count = Lim_FileGetCoordInfo(self.hFile, i, buffer, 1024)
            out.append(Coordinate(i, buffer, count))
        return out

    cpdef np.ndarray data(CND2File self, LIMUINT seq_index=0):
        # Load the data into the LIMPicture structure
        self._validate_seq(seq_index)

        cdef LIMPICTURE pic = nullpic()
        cdef LIMRESULT result = Lim_FileGetImageData(self.hFile, seq_index, &pic)

        if result != 0:
            error = ERR_CODE[result]
            raise RuntimeError('Error retrieving image data: %s' % error)

        array_wrapper = PicWrapper()
        array_wrapper.set_pic(pic)

        cdef np.ndarray ndarray = np.array(array_wrapper, copy=False)

        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> array_wrapper

        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(array_wrapper)
        return ndarray
