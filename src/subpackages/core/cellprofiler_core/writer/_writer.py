import uuid

from abc import ABC, abstractmethod


class Writer(ABC):
    """
    Derive from this abstract Writer class to create your own image writer

    You need to implement the methods below in the derived class.

    Use this block of text to describe your writer to the user.
    """

    def __init__(self, file_path):
        """
        Your init should define a variable_revision_number representing the writer version.
        Defining writer_name also allows you to register the writer with a custom name,
        otherwise it'll match the class name.
        """
        self.file_path = file_path
        self.id = uuid.uuid4()

    @property
    @abstractmethod
    def supported_filetypes(self):
        # This should be a class property. Give the writer a set of supported filetypes (extensions).
        pass

    @property
    @abstractmethod
    def supported_schemes(self):
        # This should be a class property. Give the writer a set of supported schemes (e.g. file, https, s3, ...).
        pass

    @property
    @abstractmethod
    def variable_revision_number(self):
        # This should be a class property. Give the writer a version number (int).
        pass

    @property
    @abstractmethod
    def writer_name(self):
        # This should be a class property. Give the writer a human-writeable name (string).
        pass

    def write(self,
             series=None,
             c=None,
             z=None,
             t=None,
             xywh=None,
             channel_names=None,
             ):
        """Write a single plane from the image file. Mimics the Bioformats API
        :param series: series (pyramid level)
        :param c: write from this channel. `None` = write color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        :param xywh: a (x, y, w, h) tuple (ROI / bounding box)
        :param channel_names: provide the channel names for the OME metadata
        """
        raise NotImplementedError(f"This writer ({self.writer_name}) does not support plain writes (perhaps it only supports volume or tiled writing).")

    def write_volume(self,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    xywh=None,
                    channel_names=None,
                    ):
        """Write a series of planes from the image file. Mimics the Bioformats API
        :param series: series (pyramid level)
        :param c: write from this channel. `None` = write color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        n.b. either z or t should be "None" to specify which channel to write across.
        :param xywh: a (x, y, w, h) tuple
        :param channel_names: provide the channel names for the OME metadata
        """
        raise NotImplementedError(f"This writer ({self.writer_name}) does not support 3D writing.")

    @classmethod
    @abstractmethod
    def supports_format(cls, image_file, allow_open=False, volume=False, tiled=False):
        """This function needs to evaluate whether a given ImageFile object
        can be written by this writer class.

        Return value should be an integer representing suitability:
        -1 - 'I can't write this at all'
        1 - 'I am the one true writer for this format, don't even bother checking any others'
        2 - 'I am well-suited to this format'
        3 - 'I can write this format, but I might not be the best',
        4 - 'I can give it a go, if you must'

        The allow_open parameter dictates whether the writer is permitted to write the file when
        making this decision. If False the decision should be made using file extension only.
        Any opened files should be closed before returning. For now writers are selected with
        allow_open disabled, but this may change in the future.

        The volume parameter specifies whether the writer will need to return a 3D array.
        """
        return -1

    @classmethod
    def clear_cached_writers(cls):
        # This should clear any cached writer objects if your class stores unused writers.
        pass

    @classmethod
    def supports_url(cls):
        # This function defines whether the writer class supports writing data directly from a URL.
        # If False, CellProfiler will download compatible web-based images to a temporary directory.
        # If True, the writer will be passed the source URL.
        return False

    @abstractmethod
    def close(self):
        # If your writer opens a file, this needs to release any active lock,
        pass

    @abstractmethod
    def set_series_metadata(self):
        """Takes a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        MD_SERIES_NAME - [Optional] list of human-writeable series names, one string per series.
                                    Must not contain '|' as this is used as a separator for storage.
        """
        pass
