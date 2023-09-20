import os
import urllib.parse
import urllib.request
import logging
from functools import cached_property

import numpy

from cellprofiler_core.constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, \
    MD_SIZE_KEYS, MD_SERIES_NAME
from cellprofiler_core.constants.modules.metadata import COL_PATH, COL_SERIES, COL_INDEX, COL_URL
from cellprofiler_core.constants.measurement import RESERVED_METADATA_KEYS
from cellprofiler_core.reader import get_image_reader, Reader
from cellprofiler_core.utilities.image import url_to_modpath, is_file_url

LOGGER = logging.getLogger(__name__)


class ImageFile:
    """This class represents an image file

    A file is considered as a single URL.

    A single file can contain multiple planes.

    This class assists in extracting information about the contents of
    a single file.

    """

    def __init__(self, url):
        self._url = url
        self._extracted = False
        self._index_mode = False
        self._reader = None
        self._metadata_dict = {
            COL_PATH: self.path,
            COL_SERIES: 0,
            COL_INDEX: 0,
            COL_URL: url,
            MD_SIZE_S: 0,
            MD_SIZE_C: [],
            MD_SIZE_Z: [],
            MD_SIZE_T: [],
            MD_SIZE_Y: [],
            MD_SIZE_X: [],
            MD_SERIES_NAME: [],
        }
        self._plane_details = []
        self._modpath = None
        # Records the name of the reader class selected for this file.
        self.preferred_reader = None

    def __repr__(self):
        return f"ImageFile object for {self.url}. Metadata extracted:{self.extracted}, Indexed:{self.index_mode}"

    def __lt__(self, other):
        if isinstance(other, str):
            return self.url < other
        elif isinstance(other, ImageFile):
            return self.url < other.url
        else:
            raise NotImplementedError("Unsupported comparison")

    def __gt__(self, other):
        if isinstance(other, str):
            return self.url > other
        elif isinstance(other, ImageFile):
            return self.url > other.url
        else:
            raise NotImplementedError("Unsupported comparison")

    def __eq__(self, other):
        if isinstance(other, str):
            return self.url == other
        elif isinstance(other, ImageFile):
            return self.url == other.url
        else:
            raise NotImplementedError("Unsupported comparison")

    def extract_planes(self, workspace=None):
        if self._extracted:
            return
        # Figure out the number of planes, indexes or series in the file.
        try:
            reader = self.get_reader()
        except:
            LOGGER.error(f"May not be an image: {self.url}", exc_info=True)
            self.metadata[MD_SIZE_S] = 0
            return
        meta_dict = reader.get_series_metadata()
        if meta_dict[MD_SIZE_S] == 0:
            LOGGER.error(f"File {self.filename} appears to contain no images.")
            self.metadata[MD_SIZE_S] = 0
            self.release_reader()
            return
        elif MD_SERIES_NAME not in meta_dict:
            meta_dict[MD_SERIES_NAME] = [''] * meta_dict[MD_SIZE_S]
        assert MD_SIZE_KEYS.issubset(meta_dict.keys()), "Returned metadata keys are incomplete"
        self.metadata.update(meta_dict)
        for S, C, Z, T, Y, X, name in self.get_plane_iterator():
            self._plane_details.append(f"Series {S:>2}{f' ({name})' if name else ''}"
                                       f": {X:>5} x {Y:<5}, {C} Channels, {Z:>2} Planes, {T:>2} Timepoints")
        if workspace is not None:
            metadata_array = numpy.transpose([
                self.metadata[MD_SIZE_C],
                self.metadata[MD_SIZE_Z],
                self.metadata[MD_SIZE_T],
                self.metadata[MD_SIZE_Y],
                self.metadata[MD_SIZE_X]])
            names_array = self.metadata[MD_SERIES_NAME]
            workspace.file_list.add_metadata(self.url, metadata_array.flatten(), names_array)
        self._extracted = True
        self.release_reader()

    def load_plane_metadata(self, data, names=''):
        # Metadata is stored in the HDF5 file as an array of int values, 5 per series for axis sizes (CZTYX)
        if len(data) < 5 or len(data) % 5 != 0:
            # No metadata or bad format
            if len(data) > 0:
                LOGGER.warning(f"Unable to load saved metadata for {self.filename}")
            return
        if len(data) == 5 and numpy.all(data == -1):
            # Unfilled metadata
            return
        num_series = len(data) // 5
        self.metadata[MD_SIZE_S] = num_series
        self.metadata[MD_SERIES_NAME] = names
        for i in range(num_series):
            C, Z, T, Y, X = data[:5]
            self.metadata[MD_SIZE_C].append(int(C))
            self.metadata[MD_SIZE_Z].append(int(Z))
            self.metadata[MD_SIZE_T].append(int(T))
            self.metadata[MD_SIZE_Y].append(int(Y))
            self.metadata[MD_SIZE_X].append(int(X))
        for S, C, Z, T, Y, X, name in self.get_plane_iterator():
            self._plane_details.append(
                f"Series {S:>2}{f' ({name})' if name else ''}"
                f": {X:>5} x {Y:<5}, {C} Channels, {Z:>2} Planes, {T:>2} Timepoints")
        self._extracted = True

    @property
    def extracted(self):
        return self._extracted

    @property
    def index_mode(self):
        return self._index_mode

    @property
    def url(self):
        return self._url

    @property
    def scheme(self):
        protocol_idx = self._url.find(":")
        if protocol_idx >= 0:
            return self._url.lower()[0: protocol_idx]
        else:
            return None

    @cached_property
    def filename(self):
        return os.path.basename(self.path)

    @cached_property
    def dirname(self):
        return os.path.dirname(self.path)

    @cached_property
    def path(self):
        """The file path if a file: URL, otherwise the URL"""
        if is_file_url(self.url):
            parsed = urllib.parse.urlparse(self.url)
            return urllib.request.url2pathname(parsed.path)
        return self.url

    @cached_property
    def modpath(self):
        """The directory, filename and extension broken up into a tuple"""
        if self._modpath is None:
            self._modpath = url_to_modpath(self.url)
        return self._modpath

    @cached_property
    def file_extension(self):
        return os.path.splitext(self.path)[-1].lower()

    @cached_property
    def full_extension(self):
        file_name = os.path.basename(self.path).lower()
        return file_name[file_name.find('.'):]

    @property
    def metadata(self):
        return self._metadata_dict

    def get_reader(self):
        if self._reader is None:
            self._reader = get_image_reader(self)
        return self._reader

    def release_reader(self):
        if self._reader is not None:
            if isinstance(self._reader, Reader):
                self._reader.close()
            self._reader = None

    def get_plane_iterator(self):
        # Returns an iterator which provides an entry for each series
        # in the file consisting of a tuple of the Series number, C, Z, T, Y and X dimension sizes.
        return zip(range(self.metadata[MD_SIZE_S]), self.metadata[MD_SIZE_C], self.metadata[MD_SIZE_Z],
                   self.metadata[MD_SIZE_T], self.metadata[MD_SIZE_Y], self.metadata[MD_SIZE_X],
                   self.metadata[MD_SERIES_NAME])

    @property
    def plane_details_text(self):
        return self._plane_details

    def add_metadata(self, meta_dict):
        # Used to bulk-add metadata keys from the Metadata module. Other modules should use set_metadata method.
        for key, value in meta_dict.items():
            if key in RESERVED_METADATA_KEYS:
                LOGGER.error(f"Unable to set protected metadata key '{key}' to '{value}' for file {self.filename}. "
                             f"Please choose another key name.")
                continue
            self._metadata_dict[key] = value

    def set_metadata(self, key, value, force=False):
        if key in RESERVED_METADATA_KEYS and not force:
            raise PermissionError(f"Cannot override protected metadata key '{key}'")
        else:
            if force:
                LOGGER.warning(f"Overwriting protected key {key}. This may break functionality.")
            self._metadata_dict[key] = value

    def clear_metadata(self):
        keep = (COL_PATH, COL_URL, COL_SERIES, COL_INDEX, MD_SIZE_S,
                MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, MD_SERIES_NAME)
        self._metadata_dict = {k: self._metadata_dict[k] for k in keep}
