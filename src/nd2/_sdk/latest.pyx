import json

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc


import mmap
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .._chunkmap import read_new_chunkmap

cimport numpy as np

from .. import structures
from cpython cimport Py_INCREF, PyObject

np.import_array()

cdef class ND2Reader:

    cdef LIMFILEHANDLE _fh
    cdef public str path
    cdef bint _is_open
    cdef public dict _frame_map
    cdef public dict _meta_map
    cdef int _max_safe
    cdef _mmap
    cdef __strides
    cdef __attributes
    cdef __dtype
    cdef __raw_frame_shape

    def __cinit__(self, path: str | Path):
        self._is_open = 0
        self.__raw_frame_shape = None
        self._fh = NULL
        self.path = str(path)

        with open(path, 'rb') as pyfh:
            self._frame_map, self._meta_map = read_new_chunkmap(pyfh)

        self._max_safe = max(self._frame_map["safe"])
        self.open()

    cpdef open(self):
        if not self._is_open:
            self._fh = Lim_FileOpenForReadUtf8(self.path)
            if not self._fh:
                raise OSError("Could not open file: %s" % self.path)
            with open(self.path, 'rb') as fh:
                self._mmap = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._is_open = 1

    cpdef close(self):
        if self._is_open:
            Lim_FileClose(self._fh)
            self._mmap.close()
            self._is_open = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    cpdef dict _attributes(self):
        return _loads(Lim_FileGetAttributes(self._fh))

    @property
    def attributes(self) -> structures.Attributes:
        if not hasattr(self, '__attributes'):
            if not self._is_open:
                raise ValueError("Attempt to get attributes from closed nd2 file")
            cont = self._metadata().get('contents')
            attrs = self._attributes()
            nC = cont.get('channelCount') if cont else attrs.get("componentCount", 1)
            self.__attributes = structures.Attributes(**attrs, channelCount=nC)
        return self.__attributes

    def voxel_size(self) -> tuple[float, float, float]:
        meta = self.metadata()
        if meta:
            ch = meta.channels
            if ch:
                return ch[0].volume.axesCalibration
        return (1, 1, 1)

    def _metadata(self) -> dict:
        if not self._is_open:
            raise ValueError("Attempt to get metadata from closed nd2 file")
        return _loads(Lim_FileGetMetadata(self._fh))

    def metadata(self) -> structures.Metadata:
        return structures.Metadata(**self._metadata())

    def _frame_metadata(self, seq_index: int) -> dict:
        if not self._is_open:
            raise ValueError("Attempt to get frame_metadata from closed nd2 file")
        return _loads(Lim_FileGetFrameMetadata(self._fh, seq_index))

    def frame_metadata(self, seq_index: int) -> structures.Metadata:
        return structures.FrameMetadata(**self._frame_metadata(seq_index))

    def text_info(self) -> dict:
        if not self._is_open:
            raise ValueError("Attempt to get text_info from closed nd2 file")
        return _loads(Lim_FileGetTextinfo(self._fh))

    def _description(self) -> str:
        return self.text_info().get("description", '')

    def _experiment(self) -> list:
        if not self._is_open:
            raise ValueError("Attempt to get experiment from closed nd2 file")
        return _loads(Lim_FileGetExperiment(self._fh), list)

    def experiment(self) -> List[structures.ExpLoop]:
        from ..structures import _Loop
        return [_Loop.create(i) for i in self._experiment()]

    cpdef LIMUINT _seq_count(self):
        return Lim_FileGetSeqCount(self._fh)

    cpdef LIMSIZE _coord_size(self):
        return Lim_FileGetCoordSize(self._fh)

    def _seq_index_from_coords(self, coords: Sequence) -> int:
        if not self._is_open:
            raise ValueError("Attempt to seq_index from closed nd2 file")
        cdef LIMSIZE size = self._coord_size()
        if size == 0:
            return -1

        cdef LIMSIZE n = len(coords)
        if n != size:
            raise ValueError("Coords must be length: %d" % size)

        cdef LIMUINT seq_index = -1
        cdef LIMUINT *_coords
        _coords = <LIMUINT *>malloc(n * sizeof(LIMUINT))
        if not _coords:
            raise MemoryError()
        for i in range(n):
            _coords[i] = coords[i]

        try:
            if not Lim_FileGetSeqIndexFromCoords(self._fh, &_coords[0], n, &seq_index):
                raise ValueError("Coordinate %r has no sequence index" % (coords,))
            return seq_index
        finally:
            free(_coords)

    def _coords_from_seq_index(self, seq_index: int) -> tuple:
        cdef LIMSIZE size = self._coord_size()
        if size == 0:
            return ()

        cdef LIMUINT *output = <LIMUINT *> malloc(size * sizeof(LIMUINT))
        if not output:
            raise MemoryError()

        try:
            Lim_FileGetCoordsFromSeqIndex(self._fh, seq_index, output, size)
            return tuple([x for x in output[:size]])
        finally:
            free(output)

    def _coord_info(self) -> List[Tuple[int, str, int]]:
        cdef LIMCHAR loop_type[256]
        cdef LIMSIZE size = self._coord_size()
        if size == 0:
            return []

        out = []
        for i in range(size):
            loop_size = Lim_FileGetCoordInfo(self._fh, i, loop_type, 256)
            out.append((i, loop_type, loop_size))
        return out

    cdef _validate_seq(self, LIMUINT seq_index):
        cdef LIMUINT seq_count = self._seq_count()
        if seq_index >= seq_count:
            raise IndexError(
                "Sequence %d out of range (sequence count: %d)" % (seq_index, seq_count)
            )

    def _image(self, LIMUINT seq_index):
        self._validate_seq(seq_index)

        cdef LIMPICTURE pic = nullpic()
        cdef LIMRESULT result = Lim_FileGetImageData(self._fh, seq_index, &pic)

        if result != 0:
            error = LIM_ERR_CODE[result]
            raise RuntimeError('Error retrieving image data: %s' % error)

        array_wrapper = PicWrapper()
        array_wrapper.set_pic(pic, Lim_DestroyPicture)
        return array_wrapper.to_ndarray()

    def _custom_data(self) -> dict:
        from .._xml import parse_xml_block

        return {
            k[14:]: parse_xml_block(self._get_meta_chunk(k))
            for k, v in self._meta_map.items()
            if k.startswith("CustomDataVar|")
        }

    def _get_meta_chunk(self, key: str) -> bytes:
        from .._chunkmap import read_chunk

        try:
            pos = self._meta_map[key]
        except KeyError:
            raise KeyError(
                f"No metdata chunk with key {key}. "
                f"Options include {set(self._meta_map)}"
            )
        with open(self.path, 'rb') as fh:
            return read_chunk(fh, pos)

    cdef _raw_frame_shape(self):
        if self.__raw_frame_shape is None:
            attr = self.attributes
            self.__raw_frame_shape = (
                    attr.heightPx,
                    attr.widthPx or -1,
                    attr.channelCount or 1,
                    attr.componentCount // (attr.channelCount or 1),
                )
        return self.__raw_frame_shape

    cdef _dtype(self):
        if self.__dtype is None:
            a = self.attributes
            d = a.pixelDataType[0] if a.pixelDataType else "u"
            self.__dtype = np.dtype(f"{d}{a.bitsPerComponentInMemory // 8}")
        return self.__dtype

    cpdef np.ndarray _read_image(self, index: int):
        """Read a chunk directly without using SDK"""
        if index > self._max_safe:
            raise IndexError(f"Frame out of range: {index}")
        if not self._is_open:
            raise ValueError("Attempt to read from closed nd2 file")
        offset = self._frame_map["safe"].get(index, None)
        if offset is None:
            return self._missing_frame(index)

        # try:
        #     return np.ndarray(
        #         shape=self._raw_frame_shape(),
        #         dtype=self._dtype(),
        #         buffer=self._mmap,
        #         offset=offset,
        #         strides=self._strides,
        #     )
        # except TypeError:
        #     # If the chunkmap is wrong, and the mmap isn't long enough
        #     # for the requested offset & size, a TypeError is raised.
        #     return self._missing_frame(index)

        try:
            return np.frombuffer(
                self._mmap,
                dtype=self._dtype(),
                count=np.prod(self._raw_frame_shape()),
                offset=offset
            )  # this will be reshaped in nd2file.py
        except ValueError:
            # If the chunkmap is wrong, and the mmap isn't long enough
            # for the requested offset & size, a ValueError is raised.
            return self._missing_frame(index)

    @property
    def _strides(self):
        if not hasattr(self, '__strides'):
            a = self.attributes
            width = a.widthPx
            widthB = a.widthBytes
            if not (width and widthB):
                self.__strides = None
            else:
                bypc = a.bitsPerComponentInMemory // 8
                array_stride = widthB - (bypc * width * a.componentCount)
                if array_stride == 0:
                    self.__strides = None
                else:
                    self.__strides = (
                        array_stride + width * bypc * a.componentCount,
                        a.componentCount * bypc,
                        a.componentCount // a.channelCount * bypc,
                        bypc,
                    )
        return self.__strides

    cdef np.ndarray _missing_frame(self, int index = 0):
        # TODO: add other modes for filling missing data
        return np.zeros(self._raw_frame_shape(), self._dtype())

    cpdef channel_names(self):
        return [c.channel.name for c in self.metadata().channels or []]

    cpdef time_stamps(self):
        stamps = []
        for s in range(self._seq_count()):
            chs = self._frame_metadata(s).get('channels')
            if not chs:
                break
            stamps.append(chs[0]['time']['relativeTimeMs'])
        return stamps



cdef _loads(LIMSTR string, default=dict):
    if not string:
        return default()
    try:
        return json.loads(string)
    finally:
        Lim_FileFreeString(string)



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
