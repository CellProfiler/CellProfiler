import atexit
import hashlib
import logging
import os
import tempfile
import urllib.parse
import urllib.request
import weakref

import numpy
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float32

import cellprofiler_core.preferences
from .._abstract_image import AbstractImage
from ..._image import Image
from ....reader import get_image_reader, get_image_reader_class
from ....utilities.image import is_numpy_file, download_to_temp_file
from ....utilities.image import is_matlab_file
from ....utilities.image import loadmat
from ....utilities.image import load_data_file
from ....utilities.image import generate_presigned_url
from ....constants.image import FILE_SCHEME, PASSTHROUGH_SCHEMES, NO_RESCALE
from ....utilities.pathname import pathname2url, url2pathname

LOGGER = logging.getLogger(__name__)

# A set of readers with open file locks.
ACTIVE_READERS = weakref.WeakSet()


class FileImage(AbstractImage):
    """Base for image providers: handle pathname and filename & URLs"""

    def __init__(
        self,
        name,
        pathname,
        filename,
        rescale_range=None, # autoscale by default
        metadata_rescale=False,
        series=None,
        index=None,
        channel=None,
        volume=False,
        spacing=None,
        z=None,
        t=None
    ):
        """
        :param name: Name of image to be provided
        :type name:
        :param pathname: Path to file or base of URL
        :type pathname:
        :param filename: Filename of file or last chunk of URL
        :type filename:
        :param rescale_range: a 2-tuple of min/max float values dictating the values to manually rescale the image from
                              if `None`, autorescaling will occur by dtype (if `metadata_rescale` is `False`)
                              or by values dictated in the image fle header's metadata (if `metadata_rescale` is `True`)
                              if (None, None) is given, no rescaling will occur (see NO_RESCALE constant), regardless of
                              image values
        :type rescale_range: (float, float)
        :param metadata_rescale: If `True`, rescale the image by the bitdepth values located in image file header's metadata
                                 (useful for e.g. 12-bit images stored in float16 dtype)
        :type metadata_rescale: bool
        :param series:
        :type series:
        :param index:
        :type index:
        :param channel:
        :type channel:
        :param volume:
        :type volume:
        :param spacing:
        :type spacing:
        """
        if pathname.startswith(FILE_SCHEME):
            pathname = url2pathname(pathname)
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__cached_file = None
        self.__is_cached = False
        self.__cacheing_tried = False
        self.__image = None
        self.__preferred_reader = None
        self.__image_file = None
        self.__reader = None

        if pathname is None:
            self.__url = filename
        elif any([pathname.startswith(s + ":") for s in PASSTHROUGH_SCHEMES]):
            if filename is not None:
                self.__url = pathname + "/" + filename
            else:
                self.__url = pathname
        elif filename is None:
            self.__url = pathname2url(pathname)
        else:
            self.__url = pathname2url(os.path.join(pathname, filename))
        self.__scale = None
        if rescale_range != None and metadata_rescale != False:
            LOGGER.error("Cannot specify both rescale_range and metadata_rescale, prioritizing rescale_range")
            metadata_rescale = False
        self.__rescale_range = rescale_range
        self.__metadata_rescale = metadata_rescale
        self.__series = series
        self.__channel = channel
        self.__index = index
        self.__volume = volume
        self.__spacing = spacing
        if volume:
            if z is not None and t is not None:
                raise ValueError(f"T- and Z-plane indexes were specified while in 3D mode."
                                 f"If extracting planes you should disable separation of an axis to "
                                 f"work in 3D.")
            self.__z = z
            self.__t = t
        else:
            self.__z = z if z is not None else 0
            self.__t = t if t is not None else 0

    @staticmethod
    def __validate_rescale_range(rescale_range):
        if not isinstance(rescale_range, tuple):
            raise ValueError(f"rescale_range must be a tuple, got {type(rescale_range)}")
        if len(rescale_range) != 2:
            raise ValueError("rescale_range must be a tuple of length 2")
        if rescale_range == NO_RESCALE:
            return
        if not isinstance(rescale_range[0], (float, int)):
            raise ValueError("rescale_range must be a tuple of floats")
        if not isinstance(rescale_range[1], (float, int)):
            raise ValueError("rescale_range must be a tuple of floats")
        if rescale_range[0] > rescale_range[1]:
            raise ValueError("rescale_range must be min-max")

    @staticmethod
    def __uint_to_float32(data, wants_inscale=False):
        scale = float(2**(8*data.dtype.itemsize) - 1)
        # out_range set to float64 seems wasteful for small types like
        # [u]int[8,16] (and it is), but it gets us a bit of extra precision
        # before a final cast to float32, and the internals of
        # rescale_intensity do it *anyway* (it does np.clip, w/ min/max values
        # set to float64, causing the img to become float64) so we don't
        # incur *extra* inefficiencies on top of what skimage is already doing
        data = rescale_intensity(data, in_range="dtype", out_range="float64").astype("float32")
        if wants_inscale:
            return data, scale
        return data

    @staticmethod
    def __int_to_float32(data, wants_inscale=False):
        scale = float(2**(8*data.dtype.itemsize) - 1)
        data = rescale_intensity(data, in_range="dtype", out_range=(0., 1.)).astype("float32")
        if wants_inscale:
            return data, scale
        return data

    @staticmethod
    def __float_to_float32(data, wants_inscale=False):
        # data is normalized float, just cast to 32 bit
        if data.min() >= 0 and data.max() <= 1:
            scale = 1.0
            data = img_as_float32(data)
        # data is normalized float, but not in [0, 1]
        # adjust and cast to 32 bit
        elif data.min() >= -1 and data.max() <= 1:
            scale = 2.0
            # a bit nasty to cast to 64,
            # but it buys us some extra precision until final cast to 32
            data = (data.astype("float64") + 1) / 2.
            data = img_as_float32(data)
        # data is unnormalized, with non-negative values
        # treat as uint of the same bitdepth and normalize
        elif data.min() >= 0:
            scale = float(2**(8*data.dtype.itemsize)-1)
            data = FileImage.__uint_to_float32(data.astype(f"u{data.dtype.itemsize}"))
        # data is unnormalized, with negative values
        # treat as int of the same bitdepth and normalize
        else:
            data, scale = FileImage.__int_to_float32(
                data.astype(f"i{data.dtype.itemsize}"),
                wants_inscale=True
            )

        if wants_inscale:
            return data, scale
        return data

    @staticmethod
    def normalize_to_float32(data, in_range=None, wants_inscale=False):
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
        3. Floating point values which *represent* signed/unsigned integers but stored as float
           e.g. ..., -65535.0, -255.0, -1.0, 0.0, 1.0, 65535.0, ...
           i.e. data that really should be an signed/unsigned integer type, but for whatever
           reason, are stored as float type.
           In this case, it is converted to the appropriate integer type first,
           and then normalized and cast to float32 in the range 0-1.

        param: data - the numpy data to be converted to rescaled and cast as float32
        param: in_range - `None` for automatic range detection, or a tuple of (min, max)
                          dictating the range of values in the input data
                          if equal to (None, None) no rescaling or type converstion will be performed
        param: wants_inscale - if True, return the rescaled data, and scale factor used to rescale it
        '''

        if in_range is not None:
            if in_range == NO_RESCALE:
                scale = 1.0
                # do not scale or typecast data
            if in_range != NO_RESCALE:
                scale = float(in_range[1] - in_range[0])
                # see notes above about casting to float64
                data = rescale_intensity(data.astype("float64"), in_range=in_range, out_range=(0., 1.)).astype("float32")
        elif numpy.issubdtype(data.dtype, numpy.floating):
            data, scale = FileImage.__float_to_float32(data, wants_inscale=True)
        elif numpy.issubdtype(data.dtype, numpy.signedinteger):
            data, scale = FileImage.__int_to_float32(data, wants_inscale=True)
        elif numpy.issubdtype(data.dtype, numpy.unsignedinteger):
            data, scale = FileImage.__uint_to_float32(data, wants_inscale=True)
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")

        if wants_inscale:
            return data, scale
        return data

    @property
    def series(self):
        return self.__series

    @series.setter
    def series(self, series):
        # Invalidate the cached image
        self.__image = None
        self.__series = series

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, index):
        # Invalidate the cached image
        self.__image = None
        self.__channel = index

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, index):
        # Invalidate the cached image
        self.__image = None
        self.__index = index

    @property
    def z(self):
        return self.__z

    @property
    def t(self):
        return self.__t

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if self.__scale is not None:
            LOGGER.warning(f"Setting scale to {scale} but scale was {self.__scale}")
        if scale is None:
            LOGGER.warning(f"Setting scale to None, but scale was {self.__scale}")
            self.__scale = scale
        elif isinstance(scale, int):
            self.__scale = float(scale)
        elif isinstance(scale, float):
            self.__scale = scale
        else:
            raise ValueError("scale must be a float or int")

    @property
    def rescale_range(self):
        return self.__rescale_range

    @rescale_range.setter
    def rescale_range(self, rescale_range):
        if rescale_range != None:
            FileImage.__validate_rescale_range(rescale_range)

        if (self.__rescale_range != None):
            LOGGER.warning(f"Rescale range was {str(self.__rescale_range)}, now {str(rescale_range)}")
        self.__rescale_range = rescale_range


    @property
    def metadata_rescale(self):
        return self.__metadata_rescale

    @metadata_rescale.setter
    def metadata_rescale(self, metadata_rescale):
        if not isinstance(metadata_rescale, bool):
            raise ValueError("metadata_rescale must be a boolean")
        self.__metadata_rescale = metadata_rescale

    def get_reader(self, create=True, volume=False):
        if self.__reader is None and create:
            image_file = self.get_image_file()
            self.__reader = get_image_reader(image_file, volume=volume)
            ACTIVE_READERS.add(self)
        return self.__reader

    def get_name(self):
        return self.__name

    def get_pathname(self):
        return self.__pathname

    def get_filename(self):
        return self.__filename

    def cache_file(self):
        """Cache a file that needs to be HTTP downloaded

        Return True if the file has been cached
        """
        if self.__cacheing_tried:
            return self.__is_cached
        self.__cacheing_tried = True
        #
        # Check to see if the pathname can be accessed as a directory
        # If so, handle normally
        #
        path = self.get_pathname()
        if len(path) == 0:
            filename = self.get_filename()
            if os.path.exists(filename):
                return False
            parsed_path = urllib.parse.urlparse(filename)
            url = filename
            if len(parsed_path.scheme) < 2:
                raise IOError("Test for access to file failed. File: %s" % filename)
        elif os.path.exists(path):
            return False
        else:
            parsed_path = urllib.parse.urlparse(path)
            url = "/".join((path, self.get_filename()))
            #
            # Scheme length == 0 means no scheme
            # Scheme length == 1 - probably DOS drive letter
            #
            if len(parsed_path.scheme) < 2:
                raise IOError(
                    "Test for access to directory failed. Directory: %s" % path
                )
        if parsed_path.scheme == "file":
            self.__cached_file = url2pathname(path)
        elif is_numpy_file(self.__filename):
            #
            # urlretrieve uses the suffix of the path component of the URL
            # to name the temporary file, so we replicate that behavior
            #
            temp_dir = cellprofiler_core.preferences.get_temporary_directory()
            tempfd, temppath = tempfile.mkstemp(suffix=".npy", dir=temp_dir)
            self.__cached_file = temppath
            try:
                url = generate_presigned_url(url)
                self.__cached_file, headers = urllib.request.urlretrieve(
                    url, filename=temppath
                )
            finally:
                os.close(tempfd)
        else:
            from ....pipeline import ImageFile
            image_file = self.get_image_file()
            rdr_class = get_image_reader_class(image_file, volume=self.__volume)
            if not rdr_class.supports_url() and parsed_path.scheme.lower() != 'omero':
                cached_file = download_to_temp_file(image_file.url)
                if cached_file is None:
                    return False
                self.__cached_file = pathname2url(cached_file)
                self.__image_file = ImageFile(self.__cached_file)
            else:
                return False
        self.__is_cached = True
        return True

    def get_full_name(self):
        self.cache_file()
        if self.__is_cached:
            return self.__cached_file
        return os.path.join(self.get_pathname(), self.get_filename())

    def get_url(self):
        """Get the URL representation of the file location"""
        return self.__url

    def get_md5_hash(self, measurements):
        """Compute the MD5 hash of the underlying file or use cached value

        measurements - backup for case where MD5 is calculated on image data
                       directly retrieved from URL
        """
        #
        # Cache the MD5 hash on the image reader
        #
        if is_matlab_file(self.__filename) or is_numpy_file(self.__filename):
            rdr = None
        else:
            rdr = self.get_reader()

        if rdr is None or not hasattr(rdr, "md5_hash"):
            hasher = hashlib.md5()
            path = self.get_full_name()
            if not os.path.isfile(path):
                # No file here - hash the image
                image = self.provide_image(measurements)
                hasher.update(image.pixel_data.tostring())
            else:
                with open(self.get_full_name(), "rb") as fd:
                    while True:
                        buf = fd.read(65536)
                        if len(buf) == 0:
                            break
                        hasher.update(buf)
            if rdr is None:
                return hasher.hexdigest()
            # TODO - LIS: rdr.md5_hash is unused
            rdr.md5_hash = hasher.hexdigest()
        return rdr.md5_hash

    def release_memory(self):
        """Release any image memory

        Possibly delete the temporary file"""
        if self.__is_cached:
            if is_matlab_file(self.__filename) or is_numpy_file(self.__filename):
                try:
                    os.remove(self.__cached_file)
                except:
                    LOGGER.warning(
                        "Could not delete file %s", self.__cached_file, exc_info=True
                    )
            self.__is_cached = False
            self.__cacheing_tried = False
            self.__cached_file = None

        rdr = self.get_reader(create=False)
        if rdr is not None:
            rdr.close()
            self.__reader = None
            if self in ACTIVE_READERS:
                ACTIVE_READERS.remove(self)
        self.__image = None

    def __del__(self):
        # using __del__ is all kinds of bad, but we need to remove the
        # files to keep the system from filling up.
        self.release_memory()

    def get_image_file(self):
        if self.__image_file is None:
            from ....pipeline import ImageFile
            self.__image_file = ImageFile(self.get_url())
            self.__image_file.preferred_reader = self.__preferred_reader
        return self.__image_file

    def __set_image(self):
        if self.__volume:
            self.__set_image_volume()
            return

        self.cache_file()
        channel_names = []

        # .mat and .npy are special cases, intended for flatfield illumination correction images
        # we do not want to rescale them
        if is_matlab_file(self.__filename):
            img = load_data_file(self.get_full_name(), loadmat)
            self.rescale_range = NO_RESCALE
            self.__scale = 1.0
        elif is_numpy_file(self.__filename):
            img = load_data_file(self.get_full_name(), numpy.load)
            self.rescale_range = NO_RESCALE
            self.__scale = 1.0
        else:
            rdr = self.get_reader()
            _rescale_range = None
            if numpy.isscalar(self.index) or self.index is None:
                img, _rescale_range = rdr.read(
                    wants_metadata_rescale=True,
                    c=self.channel,
                    z=self.z,
                    t=self.t,
                    series=self.series,
                    index=self.index,
                    channel_names=channel_names,
                )
            else:
                # It's a stack
                stack = []
                if numpy.isscalar(self.series):
                    series_list = [self.series] * len(self.index)
                else:
                    series_list = self.series
                if not numpy.isscalar(self.channel):
                    channel_list = [self.channel] * len(self.index)
                else:
                    channel_list = self.channel
                for series, index, channel in zip(
                    series_list, self.index, channel_list
                ):
                    img, _rescale_range = rdr.read(
                        wants_metadata_rescale=True,
                        c=channel,
                        z=self.z,
                        t=self.t,
                        series=series,
                        index=index,
                        channel_names=channel_names,
                    )
                    stack.append(img)
                img = numpy.dstack(stack)

            # if rescale_range is not manually set in __init__
            # then set it from the metadata, if requested
            if self.rescale_range is None and self.metadata_rescale:
                # _rescale_range might stil be None (reader didn't find anything)
                # but that's okay
                self.rescale_range = _rescale_range

            img, self.__scale = FileImage.normalize_to_float32(img, in_range=self.rescale_range, wants_inscale=True)

        self.__image = Image(
            img,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            scale=self.__scale,
            channelstack=img.ndim == 3 and img.shape[-1]>3
        )

        if img.ndim == 3 and len(channel_names) == img.shape[2]:
            self.__image.channel_names = list(channel_names)

    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        if self.__image is None:
            self.__set_image()
        return self.__image

    def __set_image_volume(self):
        pathname = url2pathname(self.get_url())

        # Volume loading is currently limited to tiffs/numpy files only
        if is_numpy_file(self.__filename):
            img = numpy.load(pathname)
        else:
            reader = self.get_reader(volume=True)
            img, self.rescale_range = reader.read_volume(
                wants_metadata_rescale=True,
                c=self.channel,
                z=self.z,
                t=self.t,
                series=self.series
            )

        if self.metadata_rescale:
            img, self.__scale = FileImage.normalize_to_float32(img, in_range=self.rescale_range, wants_inscale=True)
        else:
            img, self.__scale = FileImage.normalize_to_float32(img, wants_inscale=True)

        self.__image = Image(
            image=img,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            dimensions=3,
            scale=self.__scale,
            spacing=self.__spacing,
        )


@atexit.register
def shut_down_readers():
    """
    Ensures that reader file locks are released before we shut down CP.
    This isn't a problem on UNIX, but on Windows any cached files can't be
    deleted if they're still linked to a reader instance.
    """
    for reader in list(ACTIVE_READERS):
        reader.release_memory()
