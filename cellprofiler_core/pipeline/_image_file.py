import os
import urllib.request
import logging

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
        self._extracted = False
        self._index_mode = False
        self._reader = None
        self._xml_metadata = None
        self._metadata_dict = {
            MD_SIZE_S: 0,
            MD_SIZE_C: [],
            MD_SIZE_Z: [],
            MD_SIZE_T: [],
            MD_SIZE_Y: [],
            MD_SIZE_X: [],
        }
        self._url = url
        self._plane_details = []

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
        except:
            logger.error(f"May not be an image: {self.url}")
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
    def path(self):
        """The file path if a file: URL, otherwise the URL"""
        if self.url.startswith("file:"):
            return urllib.request.url2pathname(self.url[5:])
        return self.url

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


