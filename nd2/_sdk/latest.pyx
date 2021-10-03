import json

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc

from .picture cimport PicWrapper, nullpic


def open(file_name: str):
    return <uintptr_t> Lim_FileOpenForReadUtf8(file_name)


def close(fh: uintptr_t):
    Lim_FileClose(<void *>fh)


cdef _loads(LIMSTR string, default=dict):
    if not string:
        return default()
    try:
        return json.loads(string)
    finally:
        Lim_FileFreeString(string)


def get_attributes(fh: uintptr_t):
    return _loads(Lim_FileGetAttributes(<void *>fh))


def get_metadata(fh: uintptr_t):
    return _loads(Lim_FileGetMetadata(<void *>fh))


def get_frame_metadata(fh: uintptr_t, seq_index: LIMUINT):
    return _loads(Lim_FileGetFrameMetadata(<void *>fh, seq_index))


def get_text_info(fh: uintptr_t):
    return _loads(Lim_FileGetTextinfo(<void *>fh))


def get_experiment(fh: uintptr_t):
    return _loads(Lim_FileGetExperiment(<void *>fh), list)


def get_seq_count(fh: uintptr_t):
    return Lim_FileGetSeqCount(<void *>fh)


def get_coord_size(fh: uintptr_t):
    return Lim_FileGetCoordSize(<void *>fh)


def get_seq_index_from_coords(fh: uintptr_t, coords: list | tuple):

    cdef LIMSIZE size = get_coord_size(fh)
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
        if not Lim_FileGetSeqIndexFromCoords(<void *>fh, &_coords[0], n, &seq_index):
            raise ValueError("Coordinate %r has no sequence index" % coords)
        return seq_index
    finally:
        free(_coords)


def get_coords_from_seq_index(fh: uintptr_t, seq_index: LIMUINT):
    cdef LIMSIZE size = get_coord_size(fh)
    if size == 0:
        return ()

    cdef LIMUINT *output = <LIMUINT *> malloc(size * sizeof(LIMUINT))
    if not output:
        raise MemoryError()

    try:
        Lim_FileGetCoordsFromSeqIndex(<void *>fh, seq_index, output, size)
        return tuple([x for x in output[:size]])
    finally:
        free(output)


def get_coord_info(fh: uintptr_t, coord=None):
    cdef LIMCHAR loop_type[256]
    cdef LIMSIZE size = get_coord_size(fh)
    if size == 0:
        return []
    if coord is None or coord < 0:
        out = []
        for i in range(size):
            loop_size = Lim_FileGetCoordInfo(<void *>fh, i, loop_type, 256)
            out.append((i, loop_type, loop_size))
        return out

    if coord >= size:
        raise IndexError(
            "coord index %r too large for file with %d coords" % (coord, size)
        )

    loop_size = Lim_FileGetCoordInfo(<void *>fh, coord, loop_type, 256)
    return (coord, loop_type, loop_size)


cdef _validate_seq(fh: uintptr_t, LIMUINT seq_index):
    cdef LIMUINT seq_count = get_seq_count(fh)
    if seq_index >= seq_count:
        raise IndexError(
            "Sequence %d out of range (sequence count: %d)" % (seq_index, seq_count)
        )


def get_image(fh: uintptr_t, LIMUINT seq_index=0):
    _validate_seq(fh, seq_index)

    cdef LIMPICTURE pic = nullpic()
    cdef LIMRESULT result = Lim_FileGetImageData(<void *>fh, seq_index, &pic)

    if result != 0:
        error = LIM_ERR_CODE[result]
        raise RuntimeError('Error retrieving image data: %s' % error)

    array_wrapper = PicWrapper()
    array_wrapper.set_pic(pic, Lim_DestroyPicture)
    return array_wrapper.to_ndarray()


# put this in python land
# cpdef tuple _voxel_size(self):
#     meta = _loads(Lim_FileGetMetadata(self.hFile))
#     if meta:
#         ch = meta.get("channels")
#         if ch:
#             vol = ch[0].get('volume')
#             if vol and 'axesCalibration' in vol:
#                 return tuple(vol['axesCalibration'])
#     return (None, None, None)
