cimport numpy as np

ctypedef struct LIMPICTURE:
    unsigned int  uiWidth             # !< Width (in pixels) of the picture
    unsigned int  uiHeight            # !< Height (in pixels) of the picture
    unsigned int  uiBitsPerComp       # !< Number of bits for each component
    unsigned int  uiComponents        # !< Number of components in each pixel
    size_t        uiWidthBytes        # !< Number of bytes for each pixel line (stride); aligned to 4bytes
    size_t        uiSize              # !< Number of bytes the image occupies
    void*         pImageData          # !< Image data


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


ctypedef void (*destroyer_t)(LIMPICTURE*)


cdef class PicWrapper:
    cdef int dtype
    cdef LIMPICTURE pic
    cdef destroyer_t finalizer
    cdef void set_pic(PicWrapper self, LIMPICTURE pic, destroyer_t finalizer)
    cdef np.ndarray to_ndarray(self)
