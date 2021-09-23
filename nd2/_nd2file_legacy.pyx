# ND2Reader
#
# Uses the Nikon SDK for accessing data and metadata from ND2 files.
import numpy as np
cimport numpy as np
np.import_array()

# from libc.stdlib cimport free
from libc.stddef cimport wchar_t

DEF DEBUG = False

# Binary class
cdef class Binary:
    cdef LIMPICTURE picture
    cdef unsigned int seq_index
    cdef unsigned int bin_index
    cdef unsigned int width
    cdef unsigned int height

    def __cinit__(self, LIMFILEHANDLE hFile, unsigned int seq_index, unsigned int bin_index):
        """
        Constructor of the Binary picture class.

        :param hFile: handle of the open file.
        :type hFile: LIMFILEHANDLE
        :param seq_index: index of the sequence for which a binary map is
                          to be load.
        :type seq_index: unsigned int
        :param bin_index: index of the binary map for the selected sequence
                          to load.
        :type bin_index: unsigned int
        """
        cdef LIMATTRIBUTES attr
        cdef LIMPICTURE temp_pic

        self.seq_index = seq_index
        self.bin_index = bin_index

        # Initialize the LIMPicture
        if _Lim_FileGetAttributes(hFile, &attr) != LIM_OK:
            raise Exception("Could not retrieve file attributes!")

        # Get attributes into a python dictionary
        attrib = c_LIMATTRIBUTES_to_dict(&attr)

        self.width = attrib['uiWidth']
        self.height = attrib['uiHeight']

        if _Lim_InitPicture(&temp_pic, self.width,
                            self.height, 8, 1) == 0:
            raise Exception("Could not initialize picture!")

        # Load the binary data
        _Lim_FileGetBinary(hFile, seq_index, bin_index, &temp_pic);

        # Store the loaded binary data
        self.picture = temp_pic

    def __dealloc__(self):
        """
        Destructor.

        When the Picture object is destroyed, we make sure
        to destroy also the LIM picture it refers to.
        """
        if DEBUG:
            print("Deleting binary picture %d for sequence %d" %
                  (self.bin_index, self.seq_index))

        _Lim_DestroyPicture(&self.picture)

    def __getitem__(self, comp):
        """
        Return binary image as 2D numpy array (memoryview).

        Same as Binary.image()

        @see image()

        :param comp: can only be 0.
        :type comp: int
        :return: image
        :rtype: np.array (memoryview)
        """
        if comp != 0:
            raise ValueError("comp can ony be 0!")

        return self.image()

    def __repr__(self):
        """
        Summary of the Binary map.

        :return: summary of the Binary map.
        :rtype: string
        """
        return self.__str__()

    def __str__(self):
        """
        Summary of the Binary map.

        :return: summary of the Binary map.
        :rtype: string
        """
        str = "Binary:\n" \
              "   XY = (%dx%d), sequence index = %d, binary index = %d\n" % \
              (self.width, self.height, self.seq_index, self.bin_index)

        return str

    def image(self):
        """
        Return binary image as 2D numpy array (memoryview).

        :return: image
        :rtype: numpy.ndarray (memoryview)
        """
        cdef np.ndarray np_arr

        # Create a memory view to the data
        np_arr = to_uint8_numpy_array(&self.picture, self.height,
                                      self.width, 1, 0)

        return np_arr


# Picture class
cdef class Picture:
    cdef LIMPICTURE picture
    cdef LIMLOCALMETADATA metadata
    cdef unsigned int width
    cdef unsigned int height
    cdef unsigned int bpc
    cdef unsigned int n_components
    cdef unsigned int seq_index
    cdef int stretch_mode

    def __cinit__(self, unsigned int width, unsigned int height,
                  unsigned int bpc, unsigned int n_components,
                 LIMFILEHANDLE hFile, unsigned int seq_index):
        """
        Constructor of the Picture class.

        :param width: width of the picture in pixels
        :type width: unsigned int
        :param height: height of the picture in pixels
        :type height: unsigned int
        :param bpc: bit per component
        :type bpc: unsigned int
        :param n_components: number of components in the sequence. 
        :type n_components: unsigned int
        :param hFile: handle of the open file. 
        :type hFile: LIMFILEHANDLE
        :param seq_index: index of the sequence to load
        :type seq_index: LIMUINT
        """
        if width == 0 or height == 0:
            raise ValueError("The Picture cannot have 0 size!")

        # Initialize stretch mode to default
        self.stretch_mode = LIMSTRETCH_LINEAR

        # Store some arguments for easier access
        self.width = width
        self.height = height
        self.bpc = bpc
        self.n_components = n_components
        self.seq_index = seq_index

        # Load the data into the LIMPicture structure
        cdef LIMPICTURE temp_pic
        if _Lim_InitPicture(&temp_pic, width, height, bpc, n_components) == 0:
            raise Exception("Could not initialize picture!")

        # Load the image and store it
        cdef LIMLOCALMETADATA temp_metadata
        c_load_image_data(hFile, &temp_pic, &temp_metadata, seq_index,
                        self.stretch_mode)
        self.picture = temp_pic
        self.metadata = temp_metadata

    def __dealloc__(self):
        """
        Destructor.

        When the Picture object is destroyed, we make sure
        to destroy also the LIM picture it refers to.
        """
        if DEBUG:
            print("Deleting picture for sequence " + str(self.seq_index) + ".\n")

        _Lim_DestroyPicture(&self.picture)

    def __getitem__(self, comp):
        """
        Return image at given component number as 2D numpy array (memoryview).

        Same as Picture.image(comp)

        @see image()

        :param comp: component number
        :type comp: int
        :return: image
        :rtype: numpy.ndarray (memoryview)
        """
        return self.image(comp)

    def __repr__(self):
        """
        Return a summary string of the Picture.

        :return: summary of the Picture.
        :rtype: string
        """
        return self.__str__()

    def __str__(self):
        """
        Return a summary string of the Picture.

        :return: summary of the Picture.
        :rtype: string
        """
        # Get the geometry
        metadata = c_LIMLOCALMETADATA_to_dict(&self.metadata)

        str = "Picture:\n" \
              "   XY = (%dx%d), sequence index = %d, components = %d\n" \
              "   Metadata:\n" \
              "      X pos    : %f\n" \
              "      Y pos    : %f\n" \
              "      Z pos    : %f\n" \
              "      Time (ms): %f\n" % \
              (self.width, self.height, self.seq_index, self.n_components,
               metadata['dXPos'], metadata['dYPos'], metadata['dZPos'],
               metadata['dTimeMSec'])

        return str

    @property
    def n_components(self):
        """
        Number of components in the Picture.

        :return: number of components
        :rtype: int
        """
        return self.n_components

    @property
    def metadata(self):
        """
        Picture (local) metadata.

        :return: Picture (local) metadata.
        :rtype: dict
        """
        return c_LIMLOCALMETADATA_to_dict(&self.metadata)

    def image(self, comp):
        """
        Return image at given component number as 2D numpy array (memoryview).

        :param comp: component number
        :type comp: int
        :return: image
        :rtype: numpy.ndarray (memoryview)
        """
        cdef np.ndarray np_arr

        if comp >= self.n_components:
            raise Exception("The Picture only has " +
                            str(self.n_components) + " components!")

        # Create a memory view to the data
        if self.bpc == 8:
            np_arr = to_uint8_numpy_array(&self.picture, self.height,
                                          self.width, self.n_components, comp)
        elif 8 < self.bpc <= 16:
            np_arr = to_uint16_numpy_array(&self.picture, self.height,
                                           self.width, self.n_components, comp)
        elif self.bpc == 32:
            np_arr = to_float_numpy_array(&self.picture, self.height,
                                          self.width, self.n_components, comp)
        else:
            raise ValueError("Unexpected value for bpc!")

        return np_arr


cdef class nd2Reader:
    """
    ND2 Reader class.
    """

    cdef LIMFILEHANDLE file_handle
    cdef LIMEXPERIMENT exp
    cdef LIMATTRIBUTES attr
    cdef LIMMETADATA_DESC meta
    cdef LIMTEXTINFO info

    cdef public file_name
    cdef dict Pictures

    def __cinit__(self):
        """
        Constructor.
        """
        self.file_name = ""
        self.file_handle = 0
        self.Pictures = {}

    def __dealloc__(self):
        """
        Destructor.
        """
        # Close the file, if open
        if self.is_open():
            self.close()


    def __repr__(self):
        """
        Display summary of the reader state.
        :return: summary of the reader state
        :rtype: string
        """
        return self.__str__()

    def __str__(self):
        """
        Display summary of the reader state.
        :return: summary of the reader state
        :rtype: string
        """
        if not self.is_open():
            return "nd2Reader: no file opened"
        else:

            # Get the geometry
            geometry = self.get_geometry()

            str = "File opened: %s\n" \
                  "   XYZ = (%dx%dx%d), C = %d, T = %d\n" \
                  "   Number of positions = %d (other = %d)\n" \
                  "   %d bits (%d significant)\n" % \
                  (self.file_name,
                   geometry[0], geometry[1], geometry[2],
                   geometry[3], geometry[4], geometry[5],
                   geometry[6], geometry[8], geometry[9])

            return str

    def clear_cache(self):
        """
        Clears all loaded Pictures from the cache.
        """
        self.Pictures.clear()

    def close(self):
        """
        Closes the file.
        """

        # Clear the cache and close the file
        if self.is_open():

            # Clear cache
            self.clear_cache()

            # Close file
            if DEBUG:
                print("Closing file " + self.file_name + ".\n")

            self.file_handle = _Lim_FileClose(self.file_handle)

        return self.file_handle

    def get_alignment_points(self):
        """
        Get the alignment points.

        :return: Alignment points.
        :rtype: dict
        """

        if not self.is_open():
            return {}

        return c_get_alignment_points(self.file_handle)

    def get_attributes(self):
        """
        Retrieves the file attributes.

        The attributes are the LIMATTRIBUTES C structure mapped to a
        python dictionary.

        :return: File attributes.
        :rtype: dict
        """
        if not self.is_open():
            return {}

        # Convert the attribute structure to dict
        return c_LIMATTRIBUTES_to_dict(&self.attr)

    def get_binary_descriptors(self):
        """
        Retrieves the file binary descriptors.

        The attributes are the LIMBINARIES and LIMBINARYDESCRIPTOR C structures
        mapped to a python dictionary.

        :return: File binary descriptors.
        :rtype: dict
        """
        if not self.is_open():
            return {}

        # Read the binary descriptors and return them in a dictionary
        return c_get_binary_descr(self.file_handle)

    def get_custom_data(self):
        """
        Return the file custom data entities.

        All custom data entities are stored in a python dictionary.

        @TODO Investigate how to use Lim_GetCustomDataDouble()

        :return: File custom data.
        :rtype: dict
        """
        if not self.is_open():
            return {}

        return c_get_custom_data(self.file_handle)

    def get_custom_data_count(self):
        """
        Return the number of file custom data entities.
        :return: number of custom data entities.
        :rtype: int
        """
        if not self.is_open():
            return 0

        return _Lim_GetCustomDataCount(self.file_handle)

    def get_experiment(self):
        """
        Retrieves the experiment info.

        The experiment is the LIMEXPERIMENT C structure mapped to a
        python dictionary.

        :return: File experiment info.
        :rtype: dict
        """
        if not self.is_open():
            return {}

        # Convert the experiment structure to dict
        return c_LIMEXPERIMENT_to_dict(&self.exp)

    def get_geometry(self):
        """
        Returns the geometry of the dataset.

            geometry = [x, y, z, c, t, m, o, g, b, s]

            x: width
            y: height
            z: number of planes
            c: number of channels
            t: number of time points
            m: number of positions
            o: other
            g: total number of sequences
            b: bit depth
            s: significant bits

        :return: Geometry vector.
        :rtype: list
        """
        if not self.is_open():
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Get data in pythonic form
        exp = self.get_experiment()
        attr = self.get_attributes()

        x = 0 # Width
        y = 0 # Height
        z = 1 # Number of planes
        c = 1 # Number of channels
        t = 1 # Number of timepoints
        m = 1 # Number of positions
        o = 0 # Other (?)
        g = 0 # Total number of sequences
        b = 0 # Bit depth
        s = 0 # Significant bits

        cdef int n_levels = exp['uiLevelCount']
        for i in range(n_levels):
            if exp['pAllocatedLevels'][i]['uiExpType'] == LIMLOOP_TIME:
                t = exp['pAllocatedLevels'][i]['uiLoopSize']
            elif exp['pAllocatedLevels'][i]['uiExpType'] == LIMLOOP_MULTIPOINT:
                m = exp['pAllocatedLevels'][i]['uiLoopSize']
            elif exp['pAllocatedLevels'][i]['uiExpType'] == LIMLOOP_Z:
                z = exp['pAllocatedLevels'][i]['uiLoopSize']
            elif exp['pAllocatedLevels'][i]['uiExpType'] == LIMLOOP_OTHER:
                o = exp['pAllocatedLevels'][i]['uiLoopSize']
            else:
                raise Exception("Unexpected experiment level!")

        g = attr['uiSequenceCount']
        x = attr['uiWidth']
        y = attr['uiHeight']
        c = attr['uiComp']
        b = attr['uiBpcInMemory']
        s = attr['uiBpcSignificant']

        return [x, y, z, c, t, m, o, g, b, s]

    def get_metadata(self):
        """
        Retrieves the file metadata.

        The metadata is the LIMMETADATA_DESC C structure mapped to a
        python dictionary.

        :return: File metadata
        :rtype: dict
        """
        if not self.is_open():
            return {}

        # Convert metadata structure to dict
        return c_LIMMETADATA_DESC_to_dict(&self.meta)

    def get_user_events(self):
        """
        Return user events stored in file.

        The LIMFILEUSEREVENTs are converted into python dictionaries
        and stored in a list.

        :return: User events.
        :rtype: list
        """
        if not self.is_open():
            return []

        return c_get_user_events(self.file_handle)

    def get_num_binaries(self):
        """
        Retrieves the number of file binary masks.

        :return: Number of binary masks in the file.
        :rtype: int
        """
        if not self.is_open():
            return 0

        return c_get_num_binary_descriptors(self.file_handle)

    def get_large_image_dimensions(self):
        """
        Get large image dimensions.

        :return: dimensions (X and Y) and overlap.
        :rtype: dict
        """
        if not self.is_open():
            return 0

        # Retrieve the dimensions and overlap
        return c_get_large_image_dimensions(self.file_handle)

    def get_position_names(self):
        """
        Return the names of the positions.

        :return: List of position names.
        :rtype: list
        """
        if not self.is_open():
            return []

        return c_get_multi_point_names(self.file_handle, self.positions)

    def get_recorded_data(self):
        """
        Get the recorded data and return it in a python dictionary.

        @TODO Sort out how to use Lim_GetRecordedData{Int|String}

        :return: Recorded data
        :rtype: dict
        """
        if not self.is_open():
            return {}

        # Initialize the dictionary
        d = {}

        # Retrieve the data
        double_data = c_get_recorded_data_double(self.file_handle,
                                               self.attr)

        int_data = c_get_recorded_data_int(self.file_handle,
                                         self.attr)

        string_data = c_get_recorded_data_string(self.file_handle,
                                               self.attr)

        # Store it
        d['double_data'] = double_data
        d['integer_data'] = int_data
        d['string_data'] = string_data

        # Return it
        return d

    def get_stage_coordinates(self, use_alignment=False):
        """
        Get stage coordinates.

        :param use_alignment: use manual alignment
        :type use_alignment: bool
        :return: stage coordinates (per position)
        :rtype: list ([x y z]_n)
        """
        # Make sure the file is open
        if not self.is_open():
            return []

        # Make sure the attributes have ben read
        self.get_attributes()

        # The c code takes an integer as use_alignment flag
        if use_alignment:
            align = 1
        else:
            align = 0
        stage_coords = c_parse_stage_coords(self.file_handle,
                                          self.attr,
                                          align)

        return stage_coords

    def get_text_info(self):
        """
        Retrieves the File text info.

        The text info is the LIMTEXTINFO C structure mapped to a
        python dictionary.

        :return: File text info
        :rtype: dict
        """

        if not self.is_open():
            return {}

        if _Lim_FileGetTextinfo(self.file_handle, &self.info) != LIM_OK:
            raise Exception("Could not retrieve the text info!")

        # Convert to dict
        return c_LIMTEXTINFO_to_dict(&self.info)

    def get_z_stack_home(self):
        """
        Return the Z stack home.

        :return: Z stack home
        :rtype: int
        """
        cdef LIMINT home = 0

        if not self.is_open():
            return None

        # Get the experiment as a python dictionary
        exp = self.get_experiment()

        # Retrieve the home value if we have a z stack only
        for i in range(exp['uiLevelCount']):
            if exp['pAllocatedLevels'][i]['uiExpType'] == LIMLOOP_Z:
                home = _Lim_GetZStackHome(self.file_handle)
                if home < 0:
                    home = 0

        return home

    def is_open(self):
        """
        Checks if the file is open.

        :return: True if the file is open, False otherwise.
        :rtype: bool
        """
        return self.file_handle != 0

    def load(self, LIMUINT time, LIMUINT position, LIMUINT plane,
             LIMUINT other = 0, LIMUINT width=-1, LIMUINT height=-1):
        """
        Load, optionally store and return the picture.

        SYNOPSIS 1:

            r.load(time, position, plane, other)

        SYNOPSIS 2:

            r.load(time, position, plane, other, width, height)

        If width and height are specified, the image is resampled on loading
        from the original size in the file to the specified (width x height).

        When using SYNOPSIS 1, the loaded Picture is cached. When using
        SYNOPSIS 2, it is not.

        :param time: time point
        :type time: unsigned int
        :param position: position index
        :type position: unsigned int
        :param plane: plane index (z level)
        :type plane: unsigned int
        :param other: ? (optional, default = 0)
        :type other: unsigned int
        :param width: width of the image (optional, if not set the value is
                      read from the file)
        :type width: unsigned int
        :param height: height of the image (optional, if not set the value is
                       read from the file)
        :type height: unsigned int
        :return: Picture object
        :rtype: PyND2SDK.nd2Reader.Picture
        """
        if not self.is_open():
            return None

        # Map the subs to a linear index
        index = self.map_subscripts_to_index(time, position, plane, other)

        # Load the Picture
        p = self.load_by_index(index, width, height)

        # Return the Picture
        return p

    def load_binary_by_index(self, LIMUINT seq_index, LIMUINT bin_index):
        """
        Loads and returns the binary image at given sequence and binary indices.

        :param seq_index: sequence index
        :type seq_index: unsigned int
        :param bin_index: binary sequence
        :type bin_index: unsigned int
        :return: Binay object
        :rtype: PyND2SDK.nd2Reader.Binary
        """
        if not self.is_open():
            return None

        cdef unsigned int num_binaries = self.get_num_binaries()

        if num_binaries == 0:
            return None

        # Load the binary picture
        b = Binary(self.file_handle, seq_index, bin_index)

        # Return the binary
        return b

    def load_by_index(self, LIMUINT index, LIMUINT width=-1, LIMUINT height=-1):
        """
        Load, optionally store and return the picture.

        SYNOPSIS 1:

            r.load(index)

        SYNOPSIS 2:

            r.load(index, width, height)

        If width and height are specified, the image is resampled on loading
        from the original size in the file to the specified (width x height).

        When using SYNOPSIS 1, the loaded Picture is cached. When using
        SYNOPSIS 2, it is not.

        :param index: linear index of the sequence in the file.
        :type index: unsigned int
        :return: Picture object
        :rtype: PyND2SDK.nd2Reader.Picture
        """
        if not self.is_open():
            return None

        # If the image was loaded already, return it from the cache
        if index in self.Pictures and width == -1 and height == -1:
            if DEBUG:
                print("Returning picture from cache.")
            return self.Pictures[index]

        # Get the attributes
        attr = self.get_attributes()

        if index >= attr['uiSequenceCount']:
            raise Exception("The requested sequence does not exist in the file!")

        # Create a new Picture objects that loads the requested image
        store = False
        if width == -1:
            width = attr['uiWidth']
            store = True
        if height == -1:
            height = attr['uiHeight']
            store = True

        p = Picture(width,
                    height,
                    attr['uiBpcSignificant'],
                    attr['uiComp'],
                    self.file_handle,
                    index)

        # Store the picture if full size
        if store:
            if DEBUG:
                print("Adding Picture to the cache.")
            self.Pictures[index] = p
        else:
            if DEBUG:
                print("The Picture is resized and is NOT being added to the cache.")

        # Return the Picture
        return p

    def map_index_to_subscripts(self, seq_index):
        """
        Map linear index to subscripts.

        :param seq_index: linear sequence index
        :type seq_index: unsigmed int
        :return: subscripts [time, position, plane, other]
        :rtype: list
        """
        if not self.is_open():
            return None

        cdef LIMUINT[LIMMAXEXPERIMENTLEVEL] pExpCoords;
        return c_index_to_subscripts(seq_index, &self.exp, pExpCoords)

    def map_subscripts_to_index(self, time, position, plane, other = 0):
        """
        Map subscripts to linear sequence index.

        :param time: time index
        :type time: unsigned int
        :param position: position index
        :type position: unsigned int
        :param plane: plane (z lavel)
        :type plane: unsigned int
        :param other: ?
        :type other: unsigned int
        :return: linear sequence
        :rtype: int
        """
        if not self.is_open():
            return {}

        cdef LIMUINT[LIMMAXEXPERIMENTLEVEL] pExpCoords;
        pExpCoords[0] = time
        pExpCoords[1] = position
        pExpCoords[2] = plane
        pExpCoords[3] = other

        return c_subscripts_to_index(&self.exp, pExpCoords)

    def open(self, filename):
        """
        Opens the file with given filename and returns the file handle.
        :param filename: file name with full path
        :type filename: string
        :return: file handle
        :rtype: int
        """
        # Make sure the string is unicode
        self.file_name = unicode(filename)
        cdef Py_ssize_t length
        cdef wchar_t *w_filename = PyUnicode_AsWideCharString(self.file_name, &length)

        # Open the file and return the handle
        self.file_handle = _Lim_FileOpenForRead(w_filename)
        if self.file_handle == 0:
            raise Exception("Could not open file " + filename + "!")

        # Load the experiment
        if _Lim_FileGetExperiment(self.file_handle, &self.exp) != LIM_OK:
            raise Exception("Could not retrieve the experiment info!")

        # Load the attributes
        if _Lim_FileGetAttributes(self.file_handle, &self.attr) != LIM_OK:
            raise Exception("Could not retrieve the file attributes!")

        # Load the metadata
        if _Lim_FileGetMetadata(self.file_handle, &self.meta) != LIM_OK:
            raise Exception("Could not retrieve the file metadata!")

        return self.file_handle

    @property
    def file_handle(self):
        """
        :return: File handle (0 if no file is open).
        :rtype: int
        """
        return self.file_handle

    @property
    def width(self):
        """
        :return: Image width in pixels.
        :rtype: int
        """
        return self.get_geometry()[0]

    @property
    def height(self):
        """
        :return: Image height in pixels.
        :rtype: int
        """
        return self.get_geometry()[1]

    @property
    def planes(self):
        """
        :return: Number of planes (z levels).
        :rtype: int
        """
        return self.get_geometry()[2]

    @property
    def channels(self):
        """
        :return: Number of channels.
        :rtype: int
        """
        return self.get_geometry()[3]

    @property
    def timepoints(self):
        """
        :return: Number of timepoints.
        :rtype: int
        """
        return self.get_geometry()[4]

    @property
    def positions(self):
        """
        :return: Number of positions.
        :rtype: int
        """
        return self.get_geometry()[5]

    @property
    def other(self):
        """
        :return: ?
        :rtype: int
        """
        return self.get_geometry()[6]

    @property
    def sequences(self):
        """
        :return: Total number of sequences.
        :rtype: int
        """
        return self.get_geometry()[7]

    @property
    def bits(self):
        """
        :return: Image bit depth.
        :rtype: int
        """
        return self.get_geometry()[8]

    @property
    def significant_bits(self):
        """
        :return: Image significant bit depth.
        :rtype: int
        """
        return self.get_geometry()[9]


# Clean (own) memory when finalizing the array
cdef class _finalizer:
    """
    Finalizer.
    """
    cdef void *_data

    def __dealloc__(self):
        """
        Destructor.
        """
        # The data is deleted by Lim_DestroyPicture in the
        # pLIMPICTURE destructor.
        #if self._data is not NULL:
        #    free(self._data)
        pass

# Convenience function to create a _finalizer
cdef void set_base(np.ndarray arr, void *carr):
    cdef _finalizer f = _finalizer()
    f._data = <void*>carr
    np.set_array_base(arr, f)

cdef to_uint8_numpy_array(LIMPICTURE * pPicture, int n_rows, int n_cols,
                          int n_components, int component):
    """
    Create a memoryview for the requested component from the picture data
    stored in the LIMPICTURE structure (no copies are made of the data).
    The view is returned and can be used an numpy array with type np.uint8.
    
    Please notice that if the LIMPICTURE object that owns the data is
    destroyed, the memoryview will be invalid!!
        
    :param pPicture: Pointer to a LIMPICTURE structure. 
    :type pPicture: LIMPICTURE *
    :param n_rows: number of rows
    :type n_rows: int
    :param n_cols: number of columns
    :type n_cols: int
    :param n_components: number of components (channels) 
    :type n_components: int
    :param component: target component (channel)  
    :type component: int
    :return: memoryview on the underlying C array as 2D numpy array 
    :rtype: numpy.ndarray
    """
    # Get a uint8_t pointer to the picture data
    cdef uint8_t *mat = c_get_uint8_pointer_to_picture_data(pPicture)

    # Create a contiguous 1D memory view over the whole array
    n_elements = n_rows * n_cols * n_components
    cdef uint8_t[:] mv = <uint8_t[:n_elements]>mat

    # Now skip over the number of components
    mv = mv[component::n_components]

    # Now reshape the view as a 2D numpy array
    cdef np.ndarray arr = np.asarray(mv).reshape(n_rows, n_cols).view(np.uint8)

    # Set the base of the array to the picture data location
    set_base(arr, mat)

    return arr

cdef to_uint16_numpy_array(LIMPICTURE * pPicture, int n_rows, int n_cols,
                           int n_components, int component):
    """
    Create a memoryview for the requested component from the picture data
    stored in the LIMPICTURE structure (no copies are made of the data).
    The view is returned and can be used an numpy array with type np.uint16.
    
    Please notice that if the LIMPICTURE object that owns the data is
    destroyed, the memoryview will be invalid!!
        
    :param pPicture: Pointer to a LIMPICTURE structure. 
    :type pPicture: LIMPICTURE *
    :param n_rows: number of rows
    :type n_rows: int
    :param n_cols: number of columns
    :type n_cols: int
    :param n_components: number of components (channels) 
    :type n_components: int
    :param component: target component (channel)  
    :type component: int
    :return: memoryview on the underlying C array as 2D numpy array 
    :rtype: numpy.ndarray
    """
    # Get a uint16_t pointer to the picture data
    cdef uint16_t *mat = c_get_uint16_pointer_to_picture_data(pPicture)

    # Create a contiguous 1D memory view over the whole array
    n_elements = n_rows * n_cols * n_components
    cdef uint16_t[:] mv = <uint16_t[:n_elements]>mat

    # Now skip over the number of components
    mv = mv[component::n_components]

    # Now reshape the view as a 2D numpy array
    cdef np.ndarray arr = np.asarray(mv).reshape(n_rows, n_cols).view(np.uint16)

    # Set the base of the array to the picture data location
    set_base(arr, mat)

    return arr

cdef to_float_numpy_array(LIMPICTURE * pPicture, int n_rows, int n_cols,
                          int n_components, int component):
    """
    Create a memoryview for the requested component from the picture data
    stored in the LIMPICTURE structure (no copies are made of the data).
    The view is returned and can be used an numpy array with type np.float32.
    
    Please notice that if the LIMPICTURE object that owns the data is
    destroyed, the memoryview will be invalid!!
        
    :param pPicture: Pointer to a LIMPICTURE structure. 
    :type pPicture: LIMPICTURE *
    :param n_rows: number of rows
    :type n_rows: int
    :param n_cols: number of columns
    :type n_cols: int
    :param n_components: number of components (channels) 
    :type n_components: int
    :param component: target component (channel)  
    :type component: int
    :return: memoryview on the underlying C array as 2D numpy array 
    :rtype: numpy.ndarray
    """
    # Get a float pointer to the picture data
    cdef float *mat = c_get_float_pointer_to_picture_data(pPicture)

    # Create a contiguous 1D memory view over the whole array
    n_elements = n_rows * n_cols * n_components
    cdef float[:] mv = <float[:n_elements]>mat

    # Now skip over the number of components
    mv = mv[component::n_components]

    # Now reshape the view as a 2D numpy array
    cdef np.ndarray arr = np.asarray(mv).reshape(n_rows, n_cols).view(np.float32)

    # Set the base of the array to the picture data location
    set_base(arr, mat)

    return arr
