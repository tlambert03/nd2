import json

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc

from .picture cimport PicWrapper, nullpic

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .. import structures


cdef class ND2Reader:

    cdef LIMFILEHANDLE _fh
    cdef public str path
    cdef bint _is_open

    def __cinit__(self, path: str | Path):
        self._is_open = 0
        self._fh = NULL
        self.path = str(path)
        self.open()

    cpdef open(self):
        if not self._is_open:
            self._fh = Lim_FileOpenForReadUtf8(self.path)
            if not self._fh:
                raise OSError("Could not open file: %s" % self.path)
            self._is_open = 1

    cpdef close(self):
        if self._is_open:
            Lim_FileClose(self._fh)
            self._is_open = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    cpdef dict _attributes(self):
        return _loads(Lim_FileGetAttributes(self._fh))

    @property
    def attributes(self) -> structures.Attributes:
        cont = self._metadata().get('contents')
        attrs = self._attributes()
        nC = cont.get('channelCount') if cont else attrs.get("componentCount", 1)
        return structures.Attributes(**attrs, channelCount=nC)

    def voxel_size(self) -> tuple[float, float, float]:
        meta = self.metadata()
        if meta:
            ch = meta.channels
            if ch:
                return ch[0].volume.axesCalibration
        return (1, 1, 1)

    def _metadata(self) -> dict:
        return _loads(Lim_FileGetMetadata(self._fh))

    def metadata(self) -> structures.Metadata:
        return structures.Metadata(**self._metadata())

    def _frame_metadata(self, seq_index: int) -> dict:
        return _loads(Lim_FileGetFrameMetadata(self._fh, seq_index))

    def text_info(self) -> dict:
        return _loads(Lim_FileGetTextinfo(self._fh))

    def _description(self) -> str:
        return self.text_info().get("description", '')

    def _experiment(self) -> list:
        return _loads(Lim_FileGetExperiment(self._fh), list)

    def experiment(self) -> List[structures.ExpLoop]:
        from ..structures import _Loop
        return [_Loop.create(i) for i in self._experiment()]

    cpdef LIMUINT _seq_count(self):
        return Lim_FileGetSeqCount(self._fh)

    cpdef LIMSIZE _coord_size(self):
        return Lim_FileGetCoordSize(self._fh)

    def _seq_index_from_coords(self, coords: Sequence) -> int:
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
