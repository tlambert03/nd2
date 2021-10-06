from libc.stdlib cimport free, malloc

from .picture cimport PicWrapper, nullpic

try:
    # pipes lets us hide some annoying C warnings I can't otherwise get to
    from wurlitzer import pipes
except ImportError:
    from contextlib import nullcontext
    pipex = nullcontext


def open(file_name: str) -> int:
    cdef Py_ssize_t length
    cdef wchar_t *w_filename = PyUnicode_AsWideCharString(file_name, &length)
    with pipes():
        hFile = Lim_FileOpenForRead(w_filename)
    if not hFile:
        raise OSError("Could not open file: %s" % file_name)
    return hFile



def close(fh: int):
    Lim_FileClose(fh)


def get_attributes(fh: int) -> dict:
    cdef LIMATTRIBUTES attrs
    _rescheck(Lim_FileGetAttributes(fh, &attrs))
    return attrs

def get_metadata(fh: int):
    cdef LIMMETADATA_DESC meta
    _rescheck(Lim_FileGetMetadata(fh, &meta))
    return meta

# def get_frame_metadata(fh: int, seq_index: LIMUINT):
#     out = Lim_FileGetFrameMetadata(fh, seq_index)
#     if not out:
#         return ''
#     try:
#         return out
#     finally:
#         Lim_FileFreeString(out)


def get_text_info(fh: int):
    cdef LIMTEXTINFO info
    _rescheck(Lim_FileGetTextinfo(fh, &info))
    return info

def get_experiment(fh: int):
    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
    return exp


def get_seq_count(fh: int):
    cdef LIMATTRIBUTES attrs
    _rescheck(Lim_FileGetAttributes(fh, &attrs))
    return attrs.uiSequenceCount


def get_coord_size(fh: int):
    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
    return exp.uiLevelCount

def get_seq_index_from_coords(fh: int, coords: list | tuple):

    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
    size = exp.uiLevelCount
    if size == 0:
        return -1

    cdef LIMSIZE n = len(coords)
    if n != size:
        raise ValueError("Coords must have length %s" % size)

    cdef LIMUINT *_coords
    _coords = <LIMUINT *>malloc(n * sizeof(LIMUINT))
    if not _coords:
        raise MemoryError()
    for i in range(n):
        _coords[i] = coords[i]

    try:
        return Lim_GetSeqIndexFromCoords(&exp, _coords)
    finally:
        free(_coords)


def get_coords_from_seq_index(fh: int, seq_index: LIMUINT):
    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
    size = exp.uiLevelCount
    if size == 0:
        return ()

    cdef LIMUINT *coords
    coords = <LIMUINT *> malloc(size * sizeof(LIMUINT))
    if not coords:
        raise MemoryError()

    # FIXME: for some reason, this segfaults from time to time
    try:
        Lim_GetCoordsFromSeqIndex(&exp, seq_index, coords)
        return tuple([x for x in coords[:size]])
    finally:
        free(coords)

def get_coord_info(fh: int, coord: int=-1):
    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
    size = exp.uiLevelCount
    if size == 0:
        return []
    if coord is None or coord < 0:
        out = []
        for i in range(size):
            e = exp.pAllocatedLevels[i]
            out.append((i, _EXP_TYPE[e['uiExpType']], e['uiLoopSize']))
        return out

    if coord >= size:
        raise IndexError(
            "coord index %r too large for file with %d coords" % (coord, size)
        )

    e = exp.pAllocatedLevels[coord]
    return (coord, _EXP_TYPE[e['uiExpType']], e['uiLoopSize'])



cdef _validate_seq(fh: int,  seq_index: LIMUINT):
    cdef LIMUINT seq_count = get_seq_count(fh)
    if seq_index >= seq_count:
        raise IndexError(
            "Sequence %d out of range (sequence count: %d)" % (seq_index, seq_count)
        )


def get_image(fh: int, seq_index: LIMUINT=0):
    cdef LIMATTRIBUTES attrs
    _rescheck(Lim_FileGetAttributes(fh, &attrs))

    if seq_index >= attrs.uiSequenceCount:
        raise IndexError("Sequence %d out of range (sequence count: %d)"
                            % (seq_index, attrs.uiSequenceCount))


    cdef LIMPICTURE pic = nullpic()
    cdef LIMLOCALMETADATA pImgInfo

    Lim_InitPicture(&pic, attrs.uiWidth, attrs.uiHeight, attrs.uiBpcInMemory, attrs.uiComp)
    with pipes():
        _rescheck(Lim_FileGetImageData(fh, seq_index, &pic, &pImgInfo))

    array_wrapper = PicWrapper()
    array_wrapper.set_pic(pic, Lim_DestroyPicture)
    return array_wrapper.to_ndarray()


def get_stage_coords(fh):
    cdef unsigned int n = 0
    cdef LIMEXPERIMENT exp
    _rescheck(Lim_FileGetExperiment(fh, &exp))
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
            fh, n,
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

def get_zstack_home(fh: int) -> int:
    return Lim_GetZStackHome(fh)

def get_custom_data_count(fh: int) -> int:
    return Lim_GetCustomDataCount(fh)


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

_EXP_TYPE = {
    0: 'TimeLoop',
    1: 'XYPosLoop',
    2: 'ZStackLoop',
    3: 'Unknown',
}


cdef void _rescheck(LIMRESULT result):
    if result != 0:
        # TODO: make error message editable
        msg = 'An error occured.'
        code = _LIM_ERR_CODE.get(result, str(result))
        raise RuntimeError('%s %s' % (msg, code))

# put this in python land
# cpdef tuple _voxel_size(self):
#     cdef LIMMETADATA_DESC meta
#     cdef LIMEXPERIMENT exp

#     if not Lim_FileGetMetadata(self.hFile, &meta):
#         xy = meta.dCalibration
#     else:
#         xy = 1.0

#     z = 1.0
#     if Lim_FileGetExperiment(self.hFile, &exp):
#         for e in exp.pAllocatedLevels:
#             if e.uiExpType == 2:  # Z stack loop
#                 z = e.uiLoopSize
#                 break
#     return (xy, xy, z)
