import collections
import logging

from cellprofiler_core.constants.measurement import RESERVED_METADATA_KEYS, \
    C_MONOCHROME, C_RGB, C_TILE, C_URL, \
    C_SERIES, C_C, C_Z, C_T, C_INDEX, C_SERIES_NAME
from cellprofiler_core.pipeline import ImageFile

LOGGER = logging.getLogger(__name__)


class ImagePlane:
    """This class represents the location and metadata for a 2-d image plane

    A single ImageFile can be referenced by multiple ImagePlanes.

    You need a few pieces of information to reference an image plane:

    * The ImageFile object representing the source file

    Either:

    * The series number, index and channel number (if source is read with BioFormats)

    * The series, index (field), channel, z and t plane numbers (for other loaders)

    The xywh parameter can be supplied as a tuple (xstart, ystart, width, height) to isolate a
    specific region from a larger image.

    In addition, image planes have associated metadata which is represented
    as a dictionary of keys and values.
    """

    def __init__(self, file: ImageFile, series=None, index=None, channel=None, z=None, t=None, xywh=None, color=None,
                 name=None):
        self._file = file
        self._metadata_dict = collections.defaultdict(get_missing)

        self._metadata_dict[C_URL] = file.url
        self._metadata_dict[C_SERIES] = series
        self._metadata_dict[C_SERIES_NAME] = name
        self._metadata_dict[C_INDEX] = index
        self._metadata_dict[C_C] = channel
        self._metadata_dict[C_T] = t
        self._metadata_dict[C_Z] = z
        self._metadata_dict[C_TILE] = xywh
        self._multichannel = color

    def __repr__(self):
        return f"ImagePlane object Series:{self.series}, C:{self.channel}, Z:{self.z}, T:{self.t} for {self.url}"

    def __eq__(self, other):
        return all((self.url == other.url, self.series == other.series,
                   self.channel == other.channel, self.z == other.z, self.t == other.t))

    def __str__(self):
        plane_string = self.file.filename
        if self.series is not None:
            plane_string += f", Series {self.series}"
        if self.series_name is not None:
            plane_string += f" ({self.series_name})"
        if self.index is not None:
            plane_string += f", Index {self.index}"
        if self.channel is not None:
            plane_string += f", Channel {self.channel}"
        if self.z is not None:
            plane_string += f", Z {self.z}"
        if self.t is not None:
            plane_string += f", T {self.t}"
        return plane_string

    def __getstate__(self):
        # This is the object supplied to pickle.
        # We don't want to compress the parent file.
        # Let's also compress a record of the selected reader.
        self._metadata_dict[':PREFERRED_READER:'] = self.file.preferred_reader
        return self._metadata_dict

    def __setstate__(self, state):
        # State should be a dictionary.
        self._metadata_dict = state
        # Rebuild the parent file from the URL.
        self._file = ImageFile(self._metadata_dict['URL'])
        if ':PREFERRED_READER:' in self._metadata_dict:
            self._file.preferred_reader = self._metadata_dict[':PREFERRED_READER:']
            del self._metadata_dict[':PREFERRED_READER:']

    @property
    def file(self):
        return self._file

    @property
    def path(self):
        return self._file.path

    @property
    def url(self):
        return self._file.url

    @property
    def reader_name(self):
        return self._file.preferred_reader

    @property
    def series(self):
        return self.get_metadata(C_SERIES)

    @property
    def series_name(self):
        return self.get_metadata(C_SERIES_NAME)

    @property
    def index(self):
        return self.get_metadata(C_INDEX)

    @property
    def channel(self):
        return self.get_metadata(C_C)

    @property
    def z(self):
        return self.get_metadata(C_Z)

    @property
    def t(self):
        return self.get_metadata(C_T)

    @property
    def tile(self):
        return self.get_metadata(C_TILE)

    @property
    def multichannel(self):
        # True = color, False = monochrome, None = unknown
        return self._multichannel

    @property
    def color_format(self):
        if self.multichannel is None or not self.multichannel:
            return C_MONOCHROME
        return C_RGB

    @property
    def modpath(self):
        return self._file.modpath

    @property
    def metadata(self):
        return self._metadata_dict.items()

    def get_metadata(self, key):
        if key in self._metadata_dict and self._metadata_dict[key] != None:
            # Use Plane metadata
            return self._metadata_dict[key]
        elif key in self.file.metadata and self.file.metadata[key] != None:
            # Use File metadata
            return self.file.metadata[key]
        else:
            return None

    def get_all_metadata(self):
        return {**self.file.metadata, **self._metadata_dict}

    def set_metadata(self, key, value, force=False):
        if key in RESERVED_METADATA_KEYS and not force:
            raise PermissionError(f"Cannot override protected metadata key '{key}'")
        else:
            if force:
                LOGGER.warning(f"Overwriting protected key {key}. This may break functionality.")
            self._metadata_dict[key] = value


def get_missing():
    # Return None for missing keys. Can't be a lambda function if we want to use Pickle.
    return None
