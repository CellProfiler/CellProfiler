import collections
import logging

from cellprofiler_core.constants.pipeline import RESERVED_KEYS
from cellprofiler_core.pipeline import ImageFile

logger = logging.getLogger(__name__)

MD_COLOR_FORMAT = "ColorFormat"
MD_MONOCHROME = "monochrome"
MD_RGB = "RGB"


class ImagePlaneV2:
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

    def __init__(self, file: ImageFile, series=None, index=None, channel=None, z=None, t=None, xywh=None, color=None):
        self._file = file
        self._metadata_dict = collections.defaultdict(get_missing)
        self._metadata_dict['URL'] = file.url
        self._metadata_dict['Series'] = series
        self._metadata_dict['Index'] = index
        self._metadata_dict['Channel'] = channel
        self._metadata_dict['Timepoint'] = t
        self._metadata_dict['ZPlane'] = z
        self._metadata_dict['TileXYWH'] = xywh
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
        return self._metadata_dict

    def __setstate__(self, state):
        # State should be a dictionary.
        self._metadata_dict = state
        # Rebuild the parent file from the URL.
        self._file = ImageFile(self._metadata_dict['URL'])


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
    def series(self):
        return self._metadata_dict['Series']

    @property
    def index(self):
        return self._metadata_dict['Index']

    @property
    def channel(self):
        return self._metadata_dict['Channel']

    @property
    def z(self):
        return self._metadata_dict['ZPlane']

    @property
    def t(self):
        return self._metadata_dict['Timepoint']

    @property
    def tile(self):
        return self._metadata_dict['TileXYWH']

    @property
    def multichannel(self):
        # True = color, False = monochrome, None = unknown
        return self._multichannel

    @property
    def color_format(self):
        if self.multichannel is None or not self.multichannel:
            return MD_MONOCHROME
        return MD_RGB

    @property
    def modpath(self):
        return self._file.modpath

    @property
    def metadata(self):
        return self._metadata_dict.items()

    def get_metadata(self, key):
        if key in self._metadata_dict:
            # Use Plane metadata
            return self._metadata_dict[key]
        elif key in self.file.metadata:
            # Use File metadata
            return self.file.metadata[key]
        else:
            return None

    def set_metadata(self, key, value, force=False):
        if key in RESERVED_KEYS and not force:
            raise PermissionError(f"Cannot override protected metadata key '{key}'")
        else:
            if force:
                logger.warning(f"Overwriting protected key {key}. This may break functionality.")
            self._metadata_dict[key] = value


def get_missing():
    # Return None for missing keys. Can't be a lambda function if we want to use Pickle.
    return None
