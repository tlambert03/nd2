#include "nd2Reader_helper.h"
#include <stdio.h>

/* -----------------------------------------------------------------------------

    Debug functions: dump the LIM structures for control

    These functions are only compiled if a DEBUG build is asked.

----------------------------------------------------------------------------- */

#ifdef DEBUG


void c_dump_LIMATTRIBUTES_struct(const LIMATTRIBUTES *s)
{
    printf("uiWidth             = %d\n", (long)s->uiWidth);
    printf("uiWidthBytes        = %d\n", (long)s->uiWidthBytes);
    printf("uiHeight            = %d\n", (long)s->uiHeight);
    printf("uiComp              = %d\n", (long)s->uiComp);
    printf("uiBpcInMemory       = %d\n", (long)s->uiBpcInMemory);
    printf("uiBpcSignificant    = %d\n", (long)s->uiBpcSignificant);
    printf("uiSequenceCount     = %d\n", (long)s->uiSequenceCount);
    printf("uiTileWidth         = %d\n", (long)s->uiTileWidth);
    printf("uiTileHeight        = %d\n", (long)s->uiTileHeight);
    printf("uiCompression       = %d\n", (long)s->uiCompression);
    printf("uiQuality           = %d\n", (long)s->uiQuality);
}

void c_dump_LIMMETADATA_DESC_struct(const LIMMETADATA_DESC *s)
{
	printf("dTimeStart          = %f\n", (double)s->dTimeStart);
	printf("dAngle              = %f\n", (double)s->dAngle);
	printf("dCalibration        = %f\n", (double)s->dCalibration);
	printf("dObjectiveMag       = %f\n", (double)s->dObjectiveMag);
	printf("dObjectiveNA        = %f\n", (double)s->dObjectiveNA);
	printf("dRefractIndex1      = %f\n", (double)s->dRefractIndex1);
	printf("dRefractIndex2      = %f\n", (double)s->dRefractIndex2);
	printf("dPinholeRadius      = %f\n", (double)s->dPinholeRadius);
	printf("dZoom               = %f\n", (double)s->dZoom);
	printf("dProjectiveMag      = %f\n", (double)s->dProjectiveMag);
	printf("uiImageType         = %d\n", (LIMUINT)s->uiImageType);
	printf("uiPlaneCount        = %d\n", (LIMUINT)s->uiPlaneCount);
	printf("uiComponentCount    = %d\n", (LIMUINT)s->uiComponentCount);
	printf("wszObjectiveName    = %ls\n", (LIMWSTR)s->wszObjectiveName);
	// Print just the metadata for the first two planes
	// @TODO: make sure that the correct number is given by 'uiPlaneCount'!
	for (int i = 0; i < (int)s->uiPlaneCount; i++)
	{
		LIMPICTUREPLANE_DESC p = s->pPlanes[i];
		printf("Plane %d\n", i);
		c_dump_LIMPICTUREPLANE_DESC_struct(&p);
	}
}

void c_dump_LIMTEXTINFO_struct(const LIMTEXTINFO *s)
{
    printf("wszImageID          = %ls\n", (LIMWSTR)s->wszImageID);
    printf("wszType             = %ls\n", (LIMWSTR)s->wszType);
    printf("wszGroup            = %ls\n", (LIMWSTR)s->wszGroup);
    printf("wszSampleID         = %ls\n", (LIMWSTR)s->wszSampleID);
    printf("wszAuthor           = %ls\n", (LIMWSTR)s->wszAuthor);
    printf("wszDescription      = %ls\n", (LIMWSTR)s->wszDescription);
    printf("wszCapturing        = %ls\n", (LIMWSTR)s->wszCapturing);
    printf("wszSampling         = %ls\n", (LIMWSTR)s->wszSampling);
    printf("wszLocation         = %ls\n", (LIMWSTR)s->wszLocation);
    printf("wszDate             = %ls\n", (LIMWSTR)s->wszDate);
    printf("wszConclusion       = %ls\n", (LIMWSTR)s->wszConclusion);
    printf("wszInfo1            = %ls\n", (LIMWSTR)s->wszInfo1);
    printf("wszInfo2            = %ls\n", (LIMWSTR)s->wszInfo2);
    printf("wszOptics           = %ls\n", (LIMWSTR)s->wszOptics);
    printf("wszAppVersion       = %ls\n", (LIMWSTR)s->wszAppVersion);
}

void c_dump_LIMPICTUREPLANE_DESC_struct(const LIMPICTUREPLANE_DESC *s)
{
    printf("    uiCompCount     = %d\n", (long)s->uiCompCount);
    printf("    uiColorRGB      = %d\n", (long)s->uiColorRGB);
    printf("    wszName         = %ls\n", (LIMWSTR)s->wszName);
    printf("    wszOCName       = %ls\n", (LIMWSTR)s->wszOCName);
    printf("    dEmissionWL     = %f\n", (double)s->dEmissionWL);
}

void c_dump_LIMEXPERIMENTLEVEL_struct(const LIMEXPERIMENTLEVEL *s)
{
    printf("    uiExpType       = %d\n", (long)s->uiExpType);
    printf("    uiLoopSize      = %d\n", (long)s->uiLoopSize);
    printf("    dInterval       = %f\n", (double)s->dInterval);
}

void c_dump_LIMEXPERIMENT_struct(const LIMEXPERIMENT *s)
{
    printf("uiLevelCount        = %d\n", (long)s->uiLevelCount);
    for (int i = 0; i < LIMMAXEXPERIMENTLEVEL; i++)
    {
        LIMEXPERIMENTLEVEL l = s->pAllocatedLevels[i];
        c_dump_LIMEXPERIMENTLEVEL_struct(&l);
    }
}

void c_dump_LIMPICTURE_struct(const LIMPICTURE *s)
{
    printf("uiWidth             = %d\n", (long)s->uiWidth);
    printf("uiHeight            = %d\n", (long)s->uiHeight);
    printf("uiBitsPerComp       = %d\n", (long)s->uiBitsPerComp);
    printf("uiComponents        = %d\n", (long)s->uiComponents);
    printf("uiWidthBytes        = %d\n", (long)s->uiWidthBytes);
    printf("uiSize              = %d\n", (long)s->uiSize);
    printf("&pImageData         = %p\n", (void *)s->pImageData);
}

void c_dump_LIMLOCALMETADATA_struct(const LIMLOCALMETADATA *s)
{
    printf("dTimeMSec           = %f\n", (double)s->dTimeMSec);
    printf("dXPos               = %f\n", (double)s->dXPos);
    printf("dYPos               = %f\n", (double)s->dYPos);
    printf("dZPos               = %f\n", (double)s->dZPos);
}

void c_dump_LIMBINARIES_struct(const LIMBINARIES *s)
{
    printf("uiCount             = %d\n", (long)s->uiCount);

    for (unsigned int i = 0; i < s->uiCount; i++)
    {
        c_dump_LIMBINARYDESCRIPTOR_struct(&s->pDescriptors[i]);
    }
}

void c_dump_LIMBINARYDESCRIPTOR_struct(const LIMBINARYDESCRIPTOR *s)
{
    printf("    wszImageID      = %ls\n", (LIMWSTR)s->wszName);
    printf("    wszCompName     = %ls\n", (LIMWSTR)s->wszCompName);
    printf("    uiColorRGB      = %d\n", (long)s->uiColorRGB);
}

void c_dump_LIMFILEUSEREVENT_struct(const LIMFILEUSEREVENT *s)
{
    printf("uiID                = %d\n", (long)s->uiID);
    printf("dTime               = %f\n", (double)s->dTime);
    printf("wsType              = %ls\n", (LIMWSTR)s->wsType);
    printf("wsDescription       = %ls\n", (LIMWSTR)s->wsDescription);
}

#endif // DEBUG

/* -----------------------------------------------------------------------------

    Conversion functions: map LIM structures to python dictionaries;
    and other mappings.

----------------------------------------------------------------------------- */

double min(double a, double b) {
    return a<b ? a : b;
}
double max(double a, double b) {
    return a>b ? a : b;
}

PyObject* c_LIMATTRIBUTES_to_dict(const LIMATTRIBUTES *s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMATTRIBUTES_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "uiWidth",
            PyLong_FromLong((long)s->uiWidth));
    PyDict_SetItemString(d, "uiWidthBytes",
            PyLong_FromLong((long)s->uiWidthBytes));
    PyDict_SetItemString(d, "uiHeight",
            PyLong_FromLong((long)s->uiHeight));
    PyDict_SetItemString(d, "uiComp",
            PyLong_FromLong((long)s->uiComp));
    PyDict_SetItemString(d, "uiBpcInMemory",
            PyLong_FromLong((long)s->uiBpcInMemory));
    PyDict_SetItemString(d, "uiBpcSignificant",
            PyLong_FromLong((long)s->uiBpcSignificant));
    PyDict_SetItemString(d, "uiSequenceCount",
            PyLong_FromLong((long)s->uiSequenceCount));
    PyDict_SetItemString(d, "uiTileWidth",
            PyLong_FromLong((long)s->uiTileWidth));
    PyDict_SetItemString(d, "uiTileHeight",
            PyLong_FromLong((long)s->uiTileHeight));
    PyDict_SetItemString(d, "uiCompression",
            PyLong_FromLong((long)s->uiCompression));
    PyDict_SetItemString(d, "uiQuality",
            PyLong_FromLong((long)s->uiQuality));

    // Return
    return d;
}

PyObject* c_LIMMETADATA_DESC_to_dict(const LIMMETADATA_DESC *s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMMETADATA_DESC_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add (single) values
    PyDict_SetItemString(d, "dTimeStart",
            PyFloat_FromDouble((double)s->dTimeStart));
    PyDict_SetItemString(d, "dAngle",
            PyFloat_FromDouble((double)s->dAngle));
    PyDict_SetItemString(d, "dCalibration",
            PyFloat_FromDouble((double)s->dCalibration));
    PyDict_SetItemString(d, "dObjectiveMag",
            PyFloat_FromDouble((double)s->dObjectiveMag));
    PyDict_SetItemString(d, "dObjectiveNA",
            PyFloat_FromDouble((double)s->dObjectiveNA));
    PyDict_SetItemString(d, "dRefractIndex1",
            PyFloat_FromDouble((double)s->dRefractIndex1));
    PyDict_SetItemString(d, "dRefractIndex2",
            PyFloat_FromDouble((double)s->dRefractIndex2));
    PyDict_SetItemString(d, "dPinholeRadius",
            PyFloat_FromDouble((double)s->dPinholeRadius));
    PyDict_SetItemString(d, "dZoom",
            PyFloat_FromDouble((double)s->dZoom));
    PyDict_SetItemString(d, "dProjectiveMag",
            PyFloat_FromDouble((double)s->dProjectiveMag));
    PyDict_SetItemString(d, "uiPlaneCount",
            PyLong_FromLong((long)s->uiPlaneCount));
    PyDict_SetItemString(d, "uiComponentCount",
            PyLong_FromLong((long)s->uiComponentCount));
    PyDict_SetItemString(d, "wszObjectiveName",
            PyUnicode_FromWideChar((LIMWSTR)s->wszObjectiveName, -1));

    // Create a list to add the LIMPICTUREPLANE_DESC objects.
    // @TODO: make sure that the correct number is given by 'uiPlaneCount'!
    PyObject* l = PyList_New((Py_ssize_t) s->uiPlaneCount);

    for (int i = 0; i < (Py_ssize_t) s->uiPlaneCount; i++)
    {
        // Map the LIMPICTUREPLANE_DESC struct (plane) to a dictionary
        PyObject* p = c_LIMPICTUREPLANE_DESC_to_dict(&(s->pPlanes[i]));

        // Store it
        PyList_SetItem(l, i, p);
    }

    // Add the list
    PyDict_SetItemString(d, "pPlanes", l);

    // Return
    return d;
}

PyObject* c_LIMTEXTINFO_to_dict(const LIMTEXTINFO * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMTEXTINFO_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "wszImageID",
            PyUnicode_FromWideChar((LIMWSTR)s->wszImageID, -1));
    PyDict_SetItemString(d, "wszType",
            PyUnicode_FromWideChar((LIMWSTR)s->wszType, -1));
    PyDict_SetItemString(d, "wszGroup",
            PyUnicode_FromWideChar((LIMWSTR)s->wszGroup, -1));
    PyDict_SetItemString(d, "wszSampleID",
            PyUnicode_FromWideChar((LIMWSTR)s->wszSampleID, -1));
    PyDict_SetItemString(d, "wszAuthor",
            PyUnicode_FromWideChar((LIMWSTR)s->wszAuthor, -1));
    PyDict_SetItemString(d, "wszDescription",
            PyUnicode_FromWideChar((LIMWSTR)s->wszDescription, -1));
    PyDict_SetItemString(d, "wszCapturing",
            PyUnicode_FromWideChar((LIMWSTR)s->wszCapturing, -1));
    PyDict_SetItemString(d, "wszSampling",
            PyUnicode_FromWideChar((LIMWSTR)s->wszSampling, -1));
    PyDict_SetItemString(d, "wszLocation",
            PyUnicode_FromWideChar((LIMWSTR)s->wszLocation, -1));
    PyDict_SetItemString(d, "wszDate",
            PyUnicode_FromWideChar((LIMWSTR)s->wszDate, -1));
    PyDict_SetItemString(d, "wszConclusion",
            PyUnicode_FromWideChar((LIMWSTR)s->wszConclusion, -1));
    PyDict_SetItemString(d, "wszInfo1",
            PyUnicode_FromWideChar((LIMWSTR)s->wszInfo1, -1));
    PyDict_SetItemString(d, "wszInfo2",
            PyUnicode_FromWideChar((LIMWSTR)s->wszInfo2, -1));
    PyDict_SetItemString(d, "wszOptics",
            PyUnicode_FromWideChar((LIMWSTR)s->wszOptics, -1));
    PyDict_SetItemString(d, "wszAppVersion",
            PyUnicode_FromWideChar((LIMWSTR)s->wszAppVersion, -1));

    // Return
    return d;
}

PyObject* c_LIMPICTUREPLANE_DESC_to_dict(const LIMPICTUREPLANE_DESC * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMPICTUREPLANE_DESC_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "uiCompCount",
            PyLong_FromLong((long)s->uiCompCount));
    PyDict_SetItemString(d, "uiColorRGB",
            PyLong_FromLong((long)s->uiColorRGB));
    PyDict_SetItemString(d, "wszName",
            PyUnicode_FromWideChar((LIMWSTR)s->wszName, -1));
    PyDict_SetItemString(d, "wszOCName",
            PyUnicode_FromWideChar((LIMWSTR)s->wszOCName, -1));
    PyDict_SetItemString(d, "dEmissionWL",
            PyFloat_FromDouble((double)s->dEmissionWL));

    // Return
    return d;
}

PyObject* c_LIMEXPERIMENTLEVEL_to_dict(const LIMEXPERIMENTLEVEL * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMEXPERIMENTLEVEL_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "uiExpType",
            PyLong_FromLong((long)s->uiExpType));
    if (s->uiExpType == LIMLOOP_TIME)
    {
        PyDict_SetItemString(d, "uiExpTypeStr",
            PyUnicode_FromWideChar(L"Time", -1));
    }
    else if (s->uiExpType == LIMLOOP_MULTIPOINT)
    {
        PyDict_SetItemString(d, "uiExpTypeStr",
            PyUnicode_FromWideChar(L"Multipoint", -1));
    }
    else if (s->uiExpType == LIMLOOP_Z)
    {
        PyDict_SetItemString(d, "uiExpTypeStr",
            PyUnicode_FromWideChar(L"Z", -1));
    }
    else if (s->uiExpType == LIMLOOP_OTHER)
    {
        PyDict_SetItemString(d, "uiExpTypeStr",
            PyUnicode_FromWideChar(L"Other", -1));
    }
    else
    {
        printf("Error: unexpected uiExpType found!");
        PyDict_SetItemString(d, "uiExpTypeStr",
            PyUnicode_FromWideChar(L"LIMLOOP_ERROR", -1));
    }
    PyDict_SetItemString(d, "uiLoopSize",
            PyLong_FromLong((long)s->uiLoopSize));
    PyDict_SetItemString(d, "dInterval",
            PyFloat_FromDouble((double)s->dInterval));

    // Return
    return d;
}

PyObject* c_LIMEXPERIMENT_to_dict(const LIMEXPERIMENT * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMEXPERIMENT_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "uiLevelCount",
            PyLong_FromLong((long)s->uiLevelCount));

    // Create a list to add the LIMEXPERIMENTLEVEL objects.
    PyObject* l = PyList_New((Py_ssize_t) s->uiLevelCount);

    for (int i = 0; i < (Py_ssize_t) s->uiLevelCount; i++)
    {
        // Map the LIMEXPERIMENTLEVEL struct to a dictionary
        PyObject* p = c_LIMEXPERIMENTLEVEL_to_dict(&(s->pAllocatedLevels[i]));

        // Store it
        PyList_SetItem(l, i, p);
    }

    // Add the list
    PyDict_SetItemString(d, "pAllocatedLevels", l);


    // Return
    return d;
}

PyObject* c_LIMLOCALMETADATA_to_dict(const LIMLOCALMETADATA * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMLOCALMETADATA_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "dTimeMSec",
            PyFloat_FromDouble((double)s->dTimeMSec));
    PyDict_SetItemString(d, "dXPos",
            PyFloat_FromDouble((double)s->dXPos));
    PyDict_SetItemString(d, "dYPos",
            PyFloat_FromDouble((double)s->dYPos));
    PyDict_SetItemString(d, "dZPos",
            PyFloat_FromDouble((double)s->dZPos));

    // Return
    return d;
}

PyObject* c_LIMBINARIES_to_dict(const LIMBINARIES * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMBINARIES_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "uiCount",
            PyLong_FromLong((long)s->uiCount));

    // Create a list to add the LIMEXPERIMENTLEVEL objects.
    PyObject* l = PyList_New((Py_ssize_t) s->uiCount);

    for (int i = 0; i < (Py_ssize_t) s->uiCount; i++)
    {
        // Map the LIMBINARYDESCRIPTOR struct to a dictionary
        PyObject* p = c_LIMBINARYDESCRIPTOR_to_dict(&(s->pDescriptors[i]));

        // Store it
        PyList_SetItem(l, i, p);
    }

    // Add the list
    PyDict_SetItemString(d, "pDescriptors", l);

    // Return
    return d;

}

PyObject* c_LIMBINARYDESCRIPTOR_to_dict(const LIMBINARYDESCRIPTOR * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMBINARYDESCRIPTOR_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    PyDict_SetItemString(d, "wszName",
            PyUnicode_FromWideChar((LIMWSTR)s->wszName, -1));
    PyDict_SetItemString(d, "wszCompName",
            PyUnicode_FromWideChar((LIMWSTR)s->wszCompName, -1));
    PyDict_SetItemString(d, "uiColorRGB",
            PyLong_FromLong((long)s->uiColorRGB));

    // Return
    return d;
}

PyObject* c_LIMFILEUSEREVENT_to_dict(const LIMFILEUSEREVENT * s)
{
    #ifdef DEBUG
        #ifdef VERBOSE
            c_dump_LIMFILEUSEREVENT_struct(s);
        #endif
    #endif

    // Create a dictionary
    PyObject* d = PyDict_New();

    PyDict_SetItemString(d, "uiID",
            PyLong_FromLong((long)s->uiID));
    PyDict_SetItemString(d, "dTime",
            PyFloat_FromDouble((double)s->dTime));
    PyDict_SetItemString(d, "wsType",
            PyUnicode_FromWideChar((LIMWSTR)s->wsType, -1));
    PyDict_SetItemString(d, "wsDescription",
            PyUnicode_FromWideChar((LIMWSTR)s->wsDescription, -1));

    // Return
    return d;
}

/* -----------------------------------------------------------------------------

    Data access/conversion functions

----------------------------------------------------------------------------- */

float *c_get_float_pointer_to_picture_data(const LIMPICTURE * p)
{
    return (float *)p->pImageData;
}


uint16_t *c_get_uint16_pointer_to_picture_data(const LIMPICTURE * p)
{
    return (uint16_t *)p->pImageData;
}


uint8_t *c_get_uint8_pointer_to_picture_data(const LIMPICTURE * p)
{
    return (uint8_t *)p->pImageData;
}

/**
 The LIMPICTURE must be initialized already!
*/
void c_load_image_data(LIMFILEHANDLE f_handle, LIMPICTURE *picture,
    LIMLOCALMETADATA *meta, LIMUINT uiSeqIndex, LIMINT iStretchMode)
{
    LIMATTRIBUTES attr;

    // Read the attributes
    Lim_FileGetAttributes(f_handle, &attr);

    #ifdef DEBUG
        printf("Image size from the file attributes is (%dx%d).\n",
            attr.uiWidth, attr.uiHeight);
        printf("Image size from the LIMPICTURE is (%dx%d).\n",
            picture->uiWidth, picture->uiHeight);
    #endif

    // Check whether the required Picture size corresponds to the size
    // stored in the attributes. If yes, we can read the image faster.
    if (attr.uiWidth == picture->uiWidth && attr.uiHeight == picture->uiHeight)
    {
        #ifdef DEBUG
            printf("Using fast loading into buffer (no resampling).\n");
        #endif

        // Load the picture into the prepared buffer
        Lim_FileGetImageData(f_handle, uiSeqIndex, picture, meta);
    }
    else
    {
        #ifdef DEBUG
            printf("Loading and resampling into buffer.\n");
        #endif

        // Load and rescale the image into the prepared buffer
        Lim_FileGetImageRectData(f_handle, uiSeqIndex, picture->uiWidth,
            picture->uiHeight, 0, 0, picture->uiWidth, picture->uiHeight,
            picture->pImageData, picture->uiWidthBytes, iStretchMode, meta);
    }

}

PyObject* c_get_multi_point_names(LIMFILEHANDLE f_handle, LIMUINT n_multi_points)
{

    // Create a list
    PyObject* l = PyList_New((Py_ssize_t) n_multi_points);

    for (unsigned int i = 0; i < n_multi_points; i++)
    {

        // Read current name
        LIMWCHAR wstrPointName[128];
        Lim_GetMultipointName(f_handle, i, wstrPointName);

        // Pack it into the list
        PyList_SetItem(l, i,
            PyUnicode_FromWideChar(wstrPointName, -1));
    }

    // Return the list
    return l;
}

void c_to_rgb(LIMPICTURE *dstPicBuf, const LIMPICTURE *srcPicBuf)
{
// float point image
   if (32 == srcPicBuf->uiBitsPerComp)
   {
      LIMUINT c = srcPicBuf->uiComponents;
      float fMin = 9999999, fMax = -1;
      for (LIMUINT i = 0; i < srcPicBuf->uiHeight; i++)
      {
         float *pSrcWlk = (float *)srcPicBuf->pImageData + i * srcPicBuf->uiWidthBytes / sizeof(float);
         float *pSrcEnd = pSrcWlk + c * srcPicBuf->uiWidth;

         while (pSrcWlk != pSrcEnd)
         {
            float fVal = 0.0;
            for (LIMUINT j = 0; j < c; j++)
               fVal += pSrcWlk[j];
            fVal /= c;

            if (fVal < fMin) fMin = fVal;
            if (fMax < fVal) fMax = fVal;
            pSrcWlk += c;
         }
      }

      for (LIMUINT i = 0; i < min(dstPicBuf->uiHeight, srcPicBuf->uiHeight); i++)
      {
         unsigned char* pDstWlk = (unsigned char*) dstPicBuf->pImageData + i * dstPicBuf->uiWidthBytes;
         unsigned char* pDstEnd = pDstWlk + 3 * dstPicBuf->uiWidth;

         float *pSrcWlk = (float *)srcPicBuf->pImageData + i * srcPicBuf->uiWidthBytes / sizeof(float);
         float *pSrcEnd = pSrcWlk + c * srcPicBuf->uiWidth;

         while (pSrcWlk != pSrcEnd && pDstWlk != pDstEnd)
         {
            float fVal = 0.0;
            for (LIMUINT j = 0; j < c; j++)
               fVal += pSrcWlk[j];
            fVal /= c;

            unsigned char val = (unsigned char)((fVal - fMin) / (fMax - fMin) * 255.0f);
            pDstWlk[0] = pDstWlk[1] = pDstWlk[2] = val;

            pSrcWlk += c;
            pDstWlk += 3;
         }
      }
   }

   else
   {
      for (LIMUINT i = 0; i < min(dstPicBuf->uiHeight, srcPicBuf->uiHeight); i++)
      {
         unsigned char* pDstWlk = (unsigned char*) dstPicBuf->pImageData + i * dstPicBuf->uiWidthBytes;
         //unsigned char* pDstEnd = pDstWlk + 3 * dstPicBuf->uiWidth;
         if (8 == srcPicBuf->uiBitsPerComp)
         {
            unsigned char* pSrcWlk = (unsigned char*) srcPicBuf->pImageData + i * srcPicBuf->uiWidthBytes;
            unsigned char* pSrcEnd = pSrcWlk + srcPicBuf->uiComponents * srcPicBuf->uiWidth;

            if (srcPicBuf->uiComponents == 3)
               memcpy(pDstWlk, pSrcWlk, pSrcEnd - pSrcWlk);

            else
            {
               while (pSrcWlk < pSrcEnd)
               {
                  LIMUINT val = 0;
                  for (LIMUINT j = 0; j < srcPicBuf->uiComponents; j++)
                     val += *pSrcWlk++;
                  pDstWlk[0] = pDstWlk[1] = pDstWlk[2] = val / srcPicBuf->uiComponents;
                  pDstWlk += 3;
               }
            }
         }
         else if (8 < srcPicBuf->uiBitsPerComp && srcPicBuf->uiBitsPerComp <= 16)
         {
            LIMUINT iShift = srcPicBuf->uiBitsPerComp - dstPicBuf->uiBitsPerComp;
            uint16_t* pSrcWlk = (uint16_t*) srcPicBuf->pImageData + i * srcPicBuf->uiWidthBytes / 2;
            uint16_t* pSrcEnd = pSrcWlk + srcPicBuf->uiComponents * srcPicBuf->uiWidth;

            if (srcPicBuf->uiComponents == 3)
            {
               for (LIMUINT j = 0; j < srcPicBuf->uiWidth; j++)
               {
                  pDstWlk[3*j+0] = pSrcWlk[3*j+0] >> iShift;
                  pDstWlk[3*j+1] = pSrcWlk[3*j+1] >> iShift;
                  pDstWlk[3*j+2] = pSrcWlk[3*j+2] >> iShift;
               }
            }

            else
            {
               while (pSrcWlk < pSrcEnd)
               {
                  LIMUINT val = 0;
                  for (LIMUINT j = 0; j < srcPicBuf->uiComponents; j++)
                     val += *pSrcWlk++;
                  pDstWlk[0] = pDstWlk[1] = pDstWlk[2] = (val / srcPicBuf->uiComponents) >> iShift;
                  pDstWlk += 3;
               }
            }
         }
      }
    }
}

/* -----------------------------------------------------------------------------

    Metadata access functions

----------------------------------------------------------------------------- */

PyObject* c_index_to_subscripts(LIMUINT seq_index, LIMEXPERIMENT *exp, LIMUINT *coords)
{
    // Convert the linear index to coordinates the experiment coordinate system
    Lim_GetCoordsFromSeqIndex(exp, seq_index, coords);

    // Now return the coordinate as a dictionary
    PyObject* d = PyDict_New();

    // Add values
    PyDict_SetItemString(d, "time",
            PyLong_FromLong((long)coords[0]));
    PyDict_SetItemString(d, "point",
            PyLong_FromLong((long)coords[1]));
    PyDict_SetItemString(d, "plane",
            PyLong_FromLong((long)coords[2]));
    PyDict_SetItemString(d, "other",
            PyLong_FromLong((long)coords[3]));

    // Return
    return d;
}

LIMUINT c_subscripts_to_index(LIMEXPERIMENT *exp, LIMUINT *coords)
{
    // Convert the point in the experiment coordinate system to a linear index
    return Lim_GetSeqIndexFromCoords(exp, coords);
}

PyObject* c_parse_stage_coords(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr,
    int iUseAlignment)
{
    LIMUINT uiPosCount = attr.uiSequenceCount;

    // Packages them into 2D list
    PyObject* l = PyList_New((Py_ssize_t) uiPosCount);

    for (int i = 0; i < (Py_ssize_t) uiPosCount; i++)
    {
        // Get the x, y, and z coordinates of current position
        PyObject* c = PyList_New((Py_ssize_t) 3);

        // The stage coordinate is relative to the center of the image
        LIMUINT puiXPos = attr.uiWidth / 2;
        LIMUINT puiYPos = attr.uiHeight / 2;
        double pdXPos = 0.0;
        double pdYPos = 0.0;
        double pdZPos = 0.0;

        // Retrieve the coordinates (one at a time)
        Lim_GetStageCoordinates(f_handle, 1,
            &i, &puiXPos, &puiYPos, &pdXPos,
            &pdYPos, &pdZPos, iUseAlignment);

        PyList_SetItem(c, 0, PyFloat_FromDouble(pdXPos));
        PyList_SetItem(c, 1, PyFloat_FromDouble(pdYPos));
        PyList_SetItem(c, 2, PyFloat_FromDouble(pdZPos));

        // Store it
        PyList_SetItem(l, i, c);
    }

    return l;
}

PyObject* c_get_recorded_data_int(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr)
{
    // Initialize a dictionary
    PyObject* d = PyDict_New();

    // Fill the dictionary
    PyDict_SetItemString(d, "unknown_keys_for_int_data",
        PyUnicode_FromString("Not implemented yet."));

    // Return the dictionary
    return d;
}

PyObject* c_get_recorded_data_double(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr)
{
    // Sequence count
    LIMUINT uiPosCount = attr.uiSequenceCount;

    #ifdef DEBUG
        printf("Processing %d sequences.\n", uiPosCount);
    #endif

    // Load Recorded Data
    const LIMWCHAR *doubleValues[] = {L"X", L"Y", L"Z", L"Z1", L"Z2",
                                      L"HEATSTAGE_T", L"ADC_VOLTAGE_0"};
    const char *key_names[] = {"X", "Y", "Z", "Z1", "Z2", "HEATSTAGE_T",
                               "ADC_VOLTAGE_0"};

    // Initialize a dictionary
    PyObject* d = PyDict_New();

    int n_values = sizeof(doubleValues)/sizeof(doubleValues[0]);
    for (int i = 0; i < n_values; i++)
    {
        // First check if there are recordings for current value
        double test;
        if (Lim_GetRecordedDataDouble(f_handle, doubleValues[i], 0, &test) != LIM_OK)
        {
               // Continue to the next recording
               #ifdef DEBUG
                   printf("No recordings found for key %s.\n", key_names[i]);
               #endif

               continue;
        }

        // Create a list to store the values
        PyObject* l = PyList_New((Py_ssize_t) uiPosCount);

        double doubleVal = 0;
        for (unsigned int j = 0; j < uiPosCount; j++)
        {
            if (Lim_GetRecordedDataDouble(f_handle, doubleValues[i], j, &doubleVal) == LIM_OK)
            {
                PyList_SetItem(l, j, PyFloat_FromDouble(doubleVal));
            }
        }

        // Add the list to the dictionary with the correct key
        #ifdef DEBUG
            printf("Adding the list to the dictionary under key %s.\n", key_names[i]);
        #endif

        PyDict_SetItemString(d, key_names[i], l);

    }

    // Return the dictionary
    return d;
}

PyObject* c_get_recorded_data_string(LIMFILEHANDLE f_handle, LIMATTRIBUTES attr)
{
    // Initialize a dictionary
    PyObject* d = PyDict_New();

    // Fill the dictionary
    PyDict_SetItemString(d, "unknown_keys_for_string_data",
        PyUnicode_FromString("Not implemented yet."));

    // Return the dictionary
    return d;
}

PyObject* c_get_custom_data(LIMFILEHANDLE f_handle)
{
    LIMUINT count = Lim_GetCustomDataCount(f_handle);
    LIMWCHAR wszName[256], wszDescription[256], wszValue[256];
    LIMINT iType = 0, iFlags = 0, iLength = 256;

    const LIMWCHAR *types[] = {L"Label", L"Number", L"Text",
                               L"Selection", L"Long Text", L"Date"};

    // Initialize a list
    PyObject* l = PyList_New((Py_ssize_t) count);

    for(LIMUINT i = 0; i < count; i++)
    {
        // Initialize a dictionary
        PyObject* d = PyDict_New();

        if (Lim_GetCustomDataInfo(f_handle, i, wszName, wszDescription, &iType, &iFlags) == LIM_OK)
        {
            // Name
            PyDict_SetItemString(d, "Name",
                PyUnicode_FromWideChar(wszName, -1));

            // Type
            if (iType >= 1 && iType <=6)
            {
                PyDict_SetItemString(d, "Type",
                    PyUnicode_FromWideChar(types[iType - 1], -1));
            }
            else
            {
                PyDict_SetItemString(d, "Type",
                    PyUnicode_FromWideChar(L"Unknown", -1));
            }

            // Description
            PyDict_SetItemString(d, "Description",
                PyUnicode_FromWideChar(wszDescription, -1));

            // Mandatory
            if (iFlags & 2)
            {
                PyDict_SetItemString(d, "Mandatory",
                    PyUnicode_FromWideChar(L"Yes", -1));
            }
            else
            {
                PyDict_SetItemString(d, "Mandatory",
                    PyUnicode_FromWideChar(L"No", -1));
            }

            // Value
            iLength = 256;
            if (Lim_GetCustomDataString(f_handle, i, wszValue, &iLength) == LIM_OK)
            {
                PyDict_SetItemString(d, "Value",
                    PyUnicode_FromWideChar(wszValue, -1));
            }

        }

        // Add the dictionary to the list (it could potentially be empty)
        PyList_SetItem(l, i, d);
    }

    // Return the list
    return l;

}

PyObject* c_get_binary_descr(LIMFILEHANDLE f_handle)
{
    LIMBINARIES binaries;
    Lim_FileGetBinaryDescriptors(f_handle, &binaries);
    return c_LIMBINARIES_to_dict(&binaries);
}

LIMUINT c_get_num_binary_descriptors(LIMFILEHANDLE f_handle)
{
    LIMBINARIES binaries;
    Lim_FileGetBinaryDescriptors(f_handle, &binaries);
    return binaries.uiCount;
}

PyObject* c_get_large_image_dimensions(LIMFILEHANDLE f_handle)
{
    // Initialize fields to 0
    LIMUINT uiXFields = 0;
    LIMUINT uiYFields = 0;
    double dOverlap = 0.0;

    // Initialize a dictionary
    PyObject* d = PyDict_New();

    // Retrieve from file
    Lim_GetLargeImageDimensions(f_handle, &uiXFields, &uiYFields, &dOverlap);

    // Fill the dictionary
    PyDict_SetItemString(d, "uiXFiels", PyLong_FromLong(uiXFields));
    PyDict_SetItemString(d, "uiYFiels", PyLong_FromLong(uiYFields));
    PyDict_SetItemString(d, "dOverlap", PyFloat_FromDouble(dOverlap));

    // Return the dictionary
    return d;
}

PyObject* c_get_alignment_points(LIMFILEHANDLE f_handle)
{
    LIMUINT uiAlignmentPointsCount = 0;

    // Initialize a dictionary
    PyObject* d = PyDict_New();

    // Get the number of alignment points
    Lim_GetAlignmentPoints(f_handle, &uiAlignmentPointsCount, 0, 0, 0, 0, 0);

    // Pack the alignment points into the dictionary
    if (uiAlignmentPointsCount > 0)
    {
        // Allocate memory to read the positions
        LIMUINT *point_seq_idx = (LIMUINT *)malloc(uiAlignmentPointsCount * sizeof *point_seq_idx);
        LIMUINT *point_x = (LIMUINT *)malloc(uiAlignmentPointsCount * sizeof *point_x);
        LIMUINT *point_y = (LIMUINT *)malloc(uiAlignmentPointsCount * sizeof *point_y);
        double *d_point_x = (double *)malloc(uiAlignmentPointsCount * sizeof *d_point_x);
        double *d_point_y = (double *)malloc(uiAlignmentPointsCount * sizeof *d_point_y);

        // Create python lists to store the values
        PyObject* l_seq_idx = PyList_New((Py_ssize_t) uiAlignmentPointsCount);
        PyObject* l_x = PyList_New((Py_ssize_t) uiAlignmentPointsCount);
        PyObject* l_y = PyList_New((Py_ssize_t) uiAlignmentPointsCount);
        PyObject* l_dx = PyList_New((Py_ssize_t) uiAlignmentPointsCount);
        PyObject* l_dy = PyList_New((Py_ssize_t) uiAlignmentPointsCount);

        // Read the positions
        Lim_GetAlignmentPoints(f_handle, &uiAlignmentPointsCount, point_seq_idx,
            point_x, point_y, d_point_x, d_point_y);

        // Now pack them into the lists
        for (LIMUINT uiPoint = 0; uiPoint < uiAlignmentPointsCount; uiPoint++)
        {
            PyList_SetItem(l_seq_idx, uiPoint,
                PyLong_FromLong(point_seq_idx[uiPoint]));

            PyList_SetItem(l_x, uiPoint,
                PyLong_FromLong(point_x[uiPoint]));

            PyList_SetItem(l_y, uiPoint,
                PyLong_FromLong(point_y[uiPoint]));

            PyList_SetItem(l_dx, uiPoint,
                PyFloat_FromDouble(d_point_x[uiPoint]));

            PyList_SetItem(l_dy, uiPoint,
                PyFloat_FromDouble(d_point_y[uiPoint]));
        }

        // Add the lists to the dictionary
        PyDict_SetItemString(d, "uiAlignmentPointsSeqIdx", l_seq_idx);
        PyDict_SetItemString(d, "uiAlignmentPointsX", l_x);
        PyDict_SetItemString(d, "uiAlignmentPointsY", l_y);
        PyDict_SetItemString(d, "dAlignmentPointsX", l_dx);
        PyDict_SetItemString(d, "dAlignmentPointsY", l_dy);

        // Delete the allocated memory
        free(point_seq_idx);
        free(point_x);
        free(point_y);
        free(d_point_x);
        free(d_point_y);
    }

    // Return the dictionary
    return d;

}

PyObject* c_get_user_events(LIMFILEHANDLE f_handle)
{
    // Initialize an empty list, since we do not know how many
    // user event we will find.
    PyObject* l = PyList_New((Py_ssize_t) 0);

    // Instantiate needed arguments
    LIMUINT eventIndex = 1;
    LIMFILEUSEREVENT event;

    while(Lim_GetNextUserEvent(f_handle, &eventIndex, &event) == LIM_OK)
    {
        // Retrieve the event
        PyObject* d = c_LIMFILEUSEREVENT_to_dict(&event);

        // Append the object at the end of the list
        PyList_Append(l, d);
    }

    // Return the dictionary
    return l;
}
