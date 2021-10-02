from cpython cimport Py_INCREF, PyObject

import numpy as np

cimport numpy as np

np.import_array()


# based on https://gist.github.com/GaelVaroquaux/1249305
# from Gael Varoquaux License: BSD 3 clause
cdef class PicWrapper:

    cdef void set_pic(PicWrapper self, LIMPICTURE pic, destroyer_t finalizer):
        if pic.uiBitsPerComp == 8:
            self.dtype = np.NPY_UINT8
        elif 8 < pic.uiBitsPerComp <= 16:
            self.dtype = np.NPY_UINT16
        elif pic.uiBitsPerComp == 32:
            self.dtype = np.NPY_UINT32
        else:
            raise ValueError("Unexpected bits per component: %d" % pic.uiBitsPerComp)

        self.pic = pic
        self.finalizer = finalizer

    def __array__(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.pic.uiHeight
        shape[1] = <np.npy_intp> self.pic.uiWidth
        shape[2] = <np.npy_intp> self.pic.uiComponents
        return np.PyArray_SimpleNewFromData(3, shape, self.dtype, self.pic.pImageData)

    def __dealloc__(self):
        self.finalizer(&self.pic)

    cdef np.ndarray to_ndarray(self):
        cdef np.ndarray ndarray = np.array(self, copy=False)

        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> self

        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)
        return ndarray
