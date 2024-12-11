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
             wants_metadata_rescale=False,
             series=None,
             index=None,
             c=None,
             z=None,
             t=None,
             xywh=None,
             channel_names=None,
             ):
        """Read a single plane from the image file. Mimics the Bioformats API
        :param wants_metadata_rescale: if `True`, return a tuple of image and a
               tuple of (min, max) for range values of image dtype gathered from
               file metadata; if `False`, returns only the image
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        :param series: series for ``.flex`` and similar multi-stack formats
        :param index: if `None`, fall back to ``zct``, otherwise load the indexed frame
        :param xywh: a (x, y, w, h) tuple
        :param channel_names: provide the channel names for the OME metadata

        Should return a data array with channel order X, Y, (C)
        """
        return None

    def read_volume(self,
                    wants_metadata_rescale=False,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    xywh=None,
                    channel_names=None,
                    ):
        """Read a series of planes from the image file. Mimics the Bioformats API
        :param wants_metadata_rescale: if `True`, return a tuple of image and a
               tuple of (min, max) for range values of image dtype gathered from
               file metadata; if `False`, returns only the image
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        n.b. either z or t should be "None" to specify which channel to read across.
        :param series: series for ``.flex`` and similar multi-stack formats
        :param xywh: a (x, y, w, h) tuple
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
