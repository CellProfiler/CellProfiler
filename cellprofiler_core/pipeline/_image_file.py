import os
import urllib.request
import logging

from cellprofiler_core.constants.modules.metadata import COL_PATH, COL_SERIES, COL_INDEX, COL_URL
from cellprofiler_core.constants.pipeline import RESERVED_KEYS
from cellprofiler_core.utilities.image import url_to_modpath

logger = logging.getLogger(__name__)

MD_RGB = "RGB"
MD_PLANAR = "Planar"
MD_SIZE_S = "SizeS"
MD_SIZE_C = "SizeC"
MD_SIZE_Z = "SizeZ"
MD_SIZE_T = "SizeT"
MD_SIZE_X = "SizeX"
MD_SIZE_Y = "SizeY"
MD_CHANNEL_NAME = "ChannelName"


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
        self._xml_metadata = None
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
        }
        self._plane_details = []
        self._modpath = None

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

    def extract_planes(self):
        if self._extracted:
            return
        # Figure out the number of planes, indexes or series in the file.
        try:
            reader = self.get_reader().rdr
        except Exception as e:
            logger.error(f"May not be an image: {self.url}")
            logger.error(e)
            self.metadata[MD_SIZE_S] = 0
            return
        # self._xml_metadata = reader.get_omexml_metadata()
        series_count = reader.getSeriesCount()
        self.metadata[MD_SIZE_S] = series_count
        for i in range(series_count):
            reader.setSeries(i)
            self.metadata[MD_SIZE_C].append(reader.getSizeC())
            self.metadata[MD_SIZE_Z].append(reader.getSizeZ())
            self.metadata[MD_SIZE_T].append(reader.getSizeT())
            self.metadata[MD_SIZE_Y].append(reader.getSizeY())
            self.metadata[MD_SIZE_X].append(reader.getSizeX())
        for S, C, Z, T, Y, X in self.get_plane_iterator():
            self._plane_details.append(f"Series {S:>2}: {X:>5} x {Y:<5}, {C} Channels, {Z:>2} Planes, {T:>2} Timepoints")
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
    def filename(self):
        return os.path.basename(self.path)

    @property
    def dirname(self):
        return os.path.dirname(self.path)

    @property
    def path(self):
        """The file path if a file: URL, otherwise the URL"""
        if self.url.startswith("file:"):
            return urllib.request.url2pathname(self.url[5:])
        return self.url

    @property
    def modpath(self):
        """The directory, filename and extension broken up into a tuple"""
        if self._modpath is None:
            self._modpath = url_to_modpath(self.url)
        return self._modpath

    @property
    def metadata(self):
        return self._metadata_dict

    def get_reader(self):
        if self._reader is None:
            from bioformats.formatreader import get_image_reader
            self._reader = get_image_reader(key=self.url, url=self.url)
        return self._reader

    def get_xml_metadata(self):
        if self._xml_metadata is None:
            self.extract_planes()
        return self._xml_metadata

    def get_plane_iterator(self):
        # Returns an iterator which provides an entry for each series
        # in the file consisting of a tuple of the Series number, C, Z, T, Y and X dimension sizes.
        return zip(range(self.metadata[MD_SIZE_S]), self.metadata[MD_SIZE_C], self.metadata[MD_SIZE_Z],
                   self.metadata[MD_SIZE_T], self.metadata[MD_SIZE_Y], self.metadata[MD_SIZE_X])

    @property
    def plane_details_text(self):
        return self._plane_details

    def add_metadata(self, meta_dict):
        # Used to bulk-add metadata keys from the Metadata module. Other modules should use set_metadata method.
        for key, value in meta_dict.items():
            if key in RESERVED_KEYS:
                logger.error(f"Unable to set protected metadata key '{key}' to '{value}' for file {self.filename}. "
                             f"Please choose another key name.")
                continue
            self._metadata_dict[key] = value

    def set_metadata(self, key, value, force=False):
        if key in RESERVED_KEYS and not force:
            raise PermissionError(f"Cannot override protected metadata key '{key}'")
        else:
            if force:
                logger.warning(f"Overwriting protected key {key}. This may break functionality.")
            self._metadata_dict[key] = value

    def clear_metadata(self):
        keep = (COL_PATH, COL_URL, COL_SERIES, COL_INDEX, MD_SIZE_S,
                MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X)
        self._metadata_dict = {k: self._metadata_dict[k] for k in keep}
