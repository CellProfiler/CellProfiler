"""
Readers are modular classes designed to read in image data. Like modules, readers can also be added as plugins
using this template class. Place readers into the plugins directory to load them on startup.

A typical reader plugin will add support for a specific file format or protocol.

N.b. readers should not assume that cellprofiler (and by extension wx/the gui) is always available. In a
headless environment cellprofiler_core may be installed alone.

To add custom file URI schemas to CellProfiler, e.g. "omero:iid=123",
append new entries to the list in cellprofiler_core/constants/image.py.
This will allow the main file list to accept those URIs.

If you need to add menu entries to the GUI, access cellprofiler.gui.plugins_menu and add
entries to the container within. Further instructions are within that file.
"""
import uuid

from abc import ABC, abstractmethod

import numpy
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float32


class Reader(ABC):
    """
    Derive from this abstract Reader class to create your own image reader in Python

    You need to implement the methods below in the derived class.

    Use this block of text to describe your reader to the user.
    """

    def __init__(self, image_file):
        """
        Your init should define a variable_revision_number representing the reader version.
        Defining reader_name also allows you to register the reader with a custom name,
        otherwise it'll match the class name.
        """
        self.file = image_file
        self.id = uuid.uuid4()

    @property
    @abstractmethod
    def supported_filetypes(self):
        # This should be a class property. Give the reader a set of supported filetypes (extensions).
        pass

    @property
    @abstractmethod
    def supported_schemes(self):
        # This should be a class property. Give the reader a set of supported schemes (e.g. file, https, s3, ...).
        pass

    @property
    @abstractmethod
    def variable_revision_number(self):
        # This should be a class property. Give the reader a version number (int).
        pass

    @property
    @abstractmethod
    def reader_name(self):
        # This should be a class property. Give the reader a human-readable name (string).
        pass

    @abstractmethod
    def read(self,
             series=None,
             index=None,
             c=None,
             z=None,
             t=None,
             autoscale=True,
             xywh=None,
             wants_max_intensity=False,
             channel_names=None,
             ):
        """Read a single plane from the image file. Mimics the Bioformats API
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        :param series: series for ``.flex`` and similar multi-stack formats
        :param index: if `None`, fall back to ``zct``, otherwise load the indexed frame
        :param autoscale: `True` to autoscale the intensity scale to 0 and 1; `False` to
                  return the raw values native to the file.
        :param xywh: a (x, y, w, h) tuple
        :param wants_max_intensity: if `False`, only return the image; if `True`,
                  return a tuple of image and max intensity
        :param channel_names: provide the channel names for the OME metadata

        Should return a data array with channel order X, Y, (C)
        """
        return None

    def read_volume(self,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    autoscale=True,
                    xywh=None,
                    wants_max_intensity=False,
                    channel_names=None,
                    ):
        """Read a series of planes from the image file. Mimics the Bioformats API
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        n.b. either z or t should be "None" to specify which channel to read across.
        :param series: series for ``.flex`` and similar multi-stack formats
        :param autoscale: `True` to autoscale the intensity scale to 0 and 1; `False` to
                  return the raw values native to the file.
        :param xywh: a (x, y, w, h) tuple
        :param wants_max_intensity: if `False`, only return the image; if `True`,
                  return a tuple of image and max intensity
        :param channel_names: provide the channel names for the OME metadata

        Should return a data array with channel order Z, X, Y, (C)
        """
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support 3D reading.")

    @classmethod
    @abstractmethod
    def supports_format(cls, image_file, allow_open=False, volume=False):
        """This function needs to evaluate whether a given ImageFile object
        can be read by this reader class.

        Return value should be an integer representing suitability:
        -1 - 'I can't read this at all'
        1 - 'I am the one true reader for this format, don't even bother checking any others'
        2 - 'I am well-suited to this format'
        3 - 'I can read this format, but I might not be the best',
        4 - 'I can give it a go, if you must'

        The allow_open parameter dictates whether the reader is permitted to read the file when
        making this decision. If False the decision should be made using file extension only.
        Any opened files should be closed before returning. For now readers are selected with
        allow_open disabled, but this may change in the future.

        The volume parameter specifies whether the reader will need to return a 3D array.
        ."""
        return -1

    @classmethod
    def clear_cached_readers(cls):
        # This should clear any cached reader objects if your class stores unused readers.
        pass

    @classmethod
    def supports_url(cls):
        # This function defines whether the reader class supports reading data directly from a URL.
        # If False, CellProfiler will download compatible web-based images to a temporary directory.
        # If True, the reader will be passed the source URL.
        return False

    @abstractmethod
    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        pass

    @abstractmethod
    def get_series_metadata(self):
        """Should return a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        MD_SERIES_NAME - [Optional] list of human-readable series names, one string per series.
                                    Must not contain '|' as this is used as a separator for storage.
        """
        pass

    @staticmethod
    def get_settings():
        """
        This function should return a list of settings objects configurable
        by the reader class. Each entry is a tuple. This list should not
        include the default '.enabled' key, used to disable readers.

        Tuple Format: (config key, name, description, type, default)

        config key - A name for the key in internal storage.
        Slashes should not be used in key names. The reader's name will
        be prefixed automatically.

        name - a human-readable name to be shown to the user

        description - extra text to describe the setting

        type - one of str, bool, float or int. These types are supported by
        wx config files.

        default - value to use if no existing config exists

        :return: list of setting tuples
        """
        return []

    @staticmethod
    def float_cast(dtype):
        '''
        For a given dtype, return the minimally sized floating dtype needed
        to store the values of the given dtype without loss in precision.

        For example, uint32 cannot simply be cast to float32 because
        float32 can only accurately represent consecutive integer values
        between -16_777_216 to 16_777_216. The further beyond that range, the
        more precision is lost.
        e.g. The uint32 value of 16_777_217, when cast to float32,
        becomes 16_777_216.0, but when cast to float64 remains 16_777_217.0.

        Since an integer to float conversion entails casting up to a larger
        size, the minimal floating type needed is returned (rather than
        unconditionally returning float64) in order to save memory.
        '''
        if dtype == numpy.uint8:
            return numpy.float16
        elif dtype == numpy.uint16:
            return numpy.float32
        elif dtype == numpy.uint32:
            return numpy.float64
        elif dtype == numpy.int8:
            return numpy.float16
        elif dtype == numpy.int16:
            return numpy.float32
        elif dtype == numpy.int32:
            return numpy.float64
        elif dtype == numpy.float16:
            return numpy.float16
        elif dtype == numpy.float32:
            return numpy.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")


    @staticmethod
    def __uint_to_float32(data):
        # out_range set to float64 seems wasteful for small types like
        # [u]int[8,16] (and it is), but it gets us a bit of extra precision
        # before a final cast to float32, and the internals of
        # rescale_intensity do it *anyway* (it does np.clip, w/ min/max values
        # set to float64, causing the img to become float64) so we don't
        # incur *extra* inefficiencies on top of what skimage is already doing
        data = rescale_intensity(data, in_range="dtype", out_range="float64")
        return data.astype("float32")

    @staticmethod
    def __int_to_float32(data):
        data = rescale_intensity(data, in_range="dtype", out_range=(0., 1.))
        return data.astype("float32")

    @staticmethod
    def __float_to_float32(data):
        # data is normalized float, just cast to 32 bit
        if data.min() >= 0 and data.max() <= 1:
            data = img_as_float32(data)
        # data is normalized float, but not in [0, 1]
        # adjust and cast to 32 bit
        elif data.min() >= -1 and data.max() <= 1:
            # a bit nasty to cast to 64,
            # but it buys us some extra precision until final cast to 32
            data = (data.astype("float64") + 1) / 2.
            data = img_as_float32(data)
        # data is unnormalized, with non-negative values
        # treat as uint of the same bitdepth and normalize
        elif data.min() >= 0:
            data = Reader.__uint_to_float32(data.astype(f"u{data.dtype.itemsize}"))
        # data is unnormalized, with negative values
        # treat as int of the same bitdepth and normalize
        else:
            data = Reader.__int_to_float32(data.astype(f"i{data.dtype.itemsize}"))

        return data

    @staticmethod
    def normalize_to_float32(data):
        '''
        Convert an array of given data of some type to float32,
        normalizing it to the range 0-1.

        The scaling IS NOT done such that the data's minimal pixel value
        is forced to 0. and the maximal pixel value is forced to 1.
        (i.e. stretching/shrinking the data).
        Instead, scaling IS done such that relative ranges are preserved
        (i.e. range of the data wrt range of the data type is kept).

        For example, for unsigned type, the scaling *will not* be
        data / (max_val - min_val)
        but rather
        data / (2^bit_depth - 1)

        The conversion also handles signed types by shifting the data first.

        Finally the conversion handles three conditions present in floating
        point images:
        1. Floating point images which already have values inside the range 0-1.
           In this case, it is simply cast to float32.
        2. Floating point images which have values normalized inside the range -1-1.
           In this case, the data is renormalized to the range 0-1, and then cast to float32.
        2. Floating point values which *represent* signed/unsigned integers but stored as float
           e.g. ..., -65535.0, -255.0, -1.0, 0.0, 1.0, 65535.0, ...
           i.e. data that really should be an signed/unsigned integer type, but for whatever
           reason, are stored as float type.
           In this case, it is converted to the appropriate integer type first,
           and then normalized and cast to float32 in the range 0-1.
        '''

        if numpy.issubdtype(data.dtype, numpy.floating):
            return Reader.__float_to_float32(data)
        elif numpy.issubdtype(data.dtype, numpy.signedinteger):
            return Reader.__int_to_float32(data)
        elif numpy.issubdtype(data.dtype, numpy.unsignedinteger):
            return Reader.__uint_to_float32(data)
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

    # TODO - 4955: Temporary. This needs to be fixed, it is currently very naive
    @staticmethod
    def naive_scale(data):
        if numpy.issubdtype(data.dtype, numpy.integer):
            scale = numpy.iinfo(data.dtype).max
        elif numpy.issubdtype(data.dtype, numpy.floating): # assume
            # assume float is already normalized
            scale = 1
        else:
            raise NotImplementedError(f"Unsupported dtype: {data.dtype}")
        return scale

