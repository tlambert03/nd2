import json

cimport numpy as np
from cpython cimport Py_INCREF, PyObject, bool
from libc.stdlib cimport free, malloc

import numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

from ._sdk.latest cimport *


cdef class CND2File:

    cdef LIMFILEHANDLE hFile
    cdef public path

    def __cinit__(self, path):
        self.hFile = NULL
        self.path = path
        self.open()

    def __dealloc__(self):
        self.close()

    cpdef open(CND2File self):
        if not self.is_open():
            self.hFile = Lim_FileOpenForReadUtf8(self.path)
            if not self.hFile:
                raise OSError("Could not open file: %s" % self.path)

    cpdef void close(CND2File self):
        if self.is_open():
            Lim_FileClose(self.hFile)
            self.hFile = NULL

    cpdef bool is_open(CND2File self):
        return bool(<int> self.hFile)

    cpdef CND2File __enter__(CND2File self):
        self.open()
        return self

    cpdef void __exit__(CND2File self, exc_type, exc_val, exc_tb):
        self.close()

    cpdef attributes(CND2File self):
        d = _loads(Lim_FileGetAttributes(self.hFile))
        return Attributes(**d)

    # TODO: decide on "frame" vs "seq_index"
    cpdef metadata(CND2File self, int frame=-1, int format=1):
        if frame >=0:
            return self._frame_metadata(frame, format)
        d = _loads(Lim_FileGetMetadata(self.hFile))
        if not d:
            return {}
        if format:
            return Metadata(**d)
        return d

    cdef _frame_metadata(CND2File self, LIMUINT frame, int format=1):
        self._validate_seq(frame)
        d = _loads(Lim_FileGetFrameMetadata(self.hFile, frame))
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
    cpdef tuple _voxel_size(self):
        meta = _loads(Lim_FileGetMetadata(self.hFile))
        if meta:
            ch = meta.get("channels")
            if ch:
                vol = ch[0].get('volume')
                if vol and 'axesCalibration' in vol:
                    return tuple(vol['axesCalibration'])
        return (None, None, None)

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
        return _loads(Lim_FileGetTextinfo(self.hFile))

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
        cdef LIMUINT seq_index = -1
        _coords = <LIMUINT *>malloc(n * sizeof(LIMUINT))
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
            error = LIM_ERR_CODE[result]
            raise RuntimeError('Error retrieving image data: %s' % error)

        array_wrapper = PicWrapper()
        array_wrapper.set_pic(pic)
        return array_wrapper.to_ndarray()


####################################################


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
