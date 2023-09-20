import atexit
import hashlib
import logging
import os
import tempfile
import urllib.parse
import urllib.request
import weakref

import numpy

import cellprofiler_core.preferences
from .._abstract_image import AbstractImage
from ..._image import Image
from ....reader import get_image_reader, get_image_reader_class
from ....utilities.image import is_numpy_file, download_to_temp_file
from ....utilities.image import is_matlab_file
from ....utilities.image import loadmat
from ....utilities.image import load_data_file
from ....utilities.image import generate_presigned_url
from ....constants.image import FILE_SCHEME, PASSTHROUGH_SCHEMES
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
        rescale=True,
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
        :param rescale:
        :type rescale:
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
        self.rescale = rescale
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
        self.scale = None

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
        if is_matlab_file(self.__filename):
            img = load_data_file(self.get_full_name(), loadmat)
            self.scale = 1.0
        elif is_numpy_file(self.__filename):
            img = load_data_file(self.get_full_name(), numpy.load)
            self.scale = 1.0
        else:
            rdr = self.get_reader()
            if numpy.isscalar(self.index) or self.index is None:
                img, self.scale = rdr.read(
                    c=self.channel,
                    z=self.z,
                    t=self.t,
                    series=self.series,
                    index=self.index,
                    rescale=self.rescale if isinstance(self.rescale, bool) else False,
                    wants_max_intensity=True,
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
                    img, self.scale = rdr.read(
                        c=channel,
                        z=self.z,
                        t=self.t,
                        series=series,
                        index=index,
                        rescale=self.rescale if isinstance(self.rescale, bool) else False,
                        wants_max_intensity=True,
                        channel_names=channel_names,
                    )
                    stack.append(img)
                img = numpy.dstack(stack)
        if isinstance(self.rescale, float):
            # Apply a manual rescale
            img = img.astype(numpy.float32) / self.rescale
        self.__image = Image(
            img,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            scale=self.scale,
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
            data = numpy.load(pathname)
        else:
            reader = self.get_reader(volume=True)
            data = reader.read_volume(c=self.channel,
                                      z=self.z,
                                      t=self.t,
                                      series=self.series,
                                      rescale=self.rescale,
                                      wants_max_intensity=False)

        # https://github.com/CellProfiler/python-bioformats/blob/855f2fb7807f00ef41e6d169178b7f3d22530b79/bioformats/formatreader.py#L768-L791
        if data.dtype in [numpy.int8, numpy.uint8]:
            self.scale = 255
        elif data.dtype in [numpy.int16, numpy.uint16]:
            self.scale = 65535
        elif data.dtype == numpy.int32:
            self.scale = 2 ** 32 - 1
        elif data.dtype == numpy.uint32:
            self.scale = 2 ** 32
        else:
            self.scale = 1

        self.__image = Image(
            image=data,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            dimensions=3,
            scale=self.scale,
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
