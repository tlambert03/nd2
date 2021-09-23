#include "Python.h"
#include "nd2ReadSDK.h"
#include <inttypes.h>
#include <stdint.h>

#ifndef __C_HELPER_H__
#define __C_HELPER_H__

/* -----------------------------------------------------------------------------

    Debug functions: dump the LIM structures for control

    These functions are only compiled if a DEBUG build is asked.

----------------------------------------------------------------------------- */

#ifdef DEBUG

double min(double a, double b);
double max(double a, double b);

void c_dump_LIMATTRIBUTES_struct(const LIMATTRIBUTES *s);
void c_dump_LIMMETADATA_DESC_struct(const LIMMETADATA_DESC *s);
void c_dump_LIMTEXTINFO_struct(const LIMTEXTINFO *s);
void c_dump_LIMPICTUREPLANE_DESC_struct(const LIMPICTUREPLANE_DESC *s);
void c_dump_LIMEXPERIMENTLEVEL_struct(const LIMEXPERIMENTLEVEL *s);
void c_dump_LIMEXPERIMENT_struct(const LIMEXPERIMENT *s);
void c_dump_LIMPICTURE_struct(const LIMPICTURE *p);
void c_dump_LIMLOCALMETADATA_struct(const LIMLOCALMETADATA *s);
void c_dump_LIMBINARIES_struct(const LIMBINARIES *s);
void c_dump_LIMBINARYDESCRIPTOR_struct(const LIMBINARYDESCRIPTOR *s);
void c_dump_LIMFILEUSEREVENT_struct(const LIMFILEUSEREVENT *s);

#endif

/* -----------------------------------------------------------------------------

    Conversion functions: map LIM structures to python dictionaries;
    and other mappings

----------------------------------------------------------------------------- */

PyObject* c_LIMATTRIBUTES_to_dict(const LIMATTRIBUTES *s);
PyObject* c_LIMMETADATA_DESC_to_dict(const LIMMETADATA_DESC *s);
PyObject* c_LIMTEXTINFO_to_dict(const LIMTEXTINFO * s);
PyObject* c_LIMPICTUREPLANE_DESC_to_dict(const LIMPICTUREPLANE_DESC * s);
PyObject* c_LIMEXPERIMENTLEVEL_to_dict(const LIMEXPERIMENTLEVEL * s);
PyObject* c_LIMEXPERIMENT_to_dict(const LIMEXPERIMENT * s);
PyObject* c_LIMLOCALMETADATA_to_dict(const LIMLOCALMETADATA * s);
PyObject* c_LIMBINARIES_to_dict(const LIMBINARIES * s);
PyObject* c_LIMBINARYDESCRIPTOR_to_dict(const LIMBINARYDESCRIPTOR * s);
PyObject* c_LIMFILEUSEREVENT_to_dict(const LIMFILEUSEREVENT * s);

/* -----------------------------------------------------------------------------

    Data access/conversion functions

----------------------------------------------------------------------------- */

float *c_get_float_pointer_to_picture_data(const LIMPICTURE * p);
uint16_t *c_get_uint16_pointer_to_picture_data(const LIMPICTURE * p);
uint8_t *c_get_uint8_pointer_to_picture_data(const LIMPICTURE * p);
void c_load_image_data(LIMFILEHANDLE f_handle, LIMPICTURE *p, LIMLOCALMETADATA *m, LIMUINT uiSeqIndex, LIMINT iStretchMode);
void c_to_rgb(LIMPICTURE *dstPicBuf, const LIMPICTURE *srcPicBuf);

/* -----------------------------------------------------------------------------

    Metadata access functions

----------------------------------------------------------------------------- */

PyObject* c_index_to_subscripts(LIMUINT seq_index, LIMEXPERIMENT *exp, LIMUINT *coords);
LIMUINT c_subscripts_to_index(LIMEXPERIMENT *exp, LIMUINT *coords);
PyObject* c_parse_stage_coords(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr, int iUseAlignment);
PyObject* c_get_recorded_data_int(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr);
PyObject* c_get_recorded_data_double(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr);
PyObject* c_get_recorded_data_string(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr);
PyObject* c_get_custom_data(LIMFILEHANDLE f_handle);
PyObject* c_get_multi_point_names(LIMFILEHANDLE f_handle, LIMUINT n_multi_points);
PyObject* c_get_binary_descr(LIMFILEHANDLE f_handle);
LIMUINT c_get_num_binary_descriptors(LIMFILEHANDLE f_handle);
PyObject* c_get_large_image_dimensions(LIMFILEHANDLE f_handle);
PyObject* c_get_alignment_points(LIMFILEHANDLE f_handle);
PyObject* c_get_user_events(LIMFILEHANDLE f_handle);

#endif
