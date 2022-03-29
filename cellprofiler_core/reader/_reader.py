import uuid

from abc import ABC, abstractmethod


class Reader(ABC):
    """ Derive from this abstract Reader class to create your own image reader in Python

    You need to implement the methods below in the derived class.
    """

    def __init__(self, image_file):
        """
        Your init should define a variable_revision_number representing the reader version.
        Defining reader_name also allows you to register the reader with a custom name,
        otherwise it'll match the class name.
        """
        self.file = image_file
        self.id = uuid.uuid4()
        if not hasattr(self, "variable_revision_number"):
            self.variable_revision_number = 0
        if not hasattr(self, "reader_name"):
            self.reader_name = self.__class__.__name__

    @abstractmethod
    def read(self,
             series=None,
             index=None,
             c=None,
             z=None,
             t=None,
             rescale=True,
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
        :param rescale: `True` to rescale the intensity scale to 0 and 1; `False` to
                  return the raw values native to the file.
        :param xywh: a (x, y, w, h) tuple
        :param wants_max_intensity: if `False`, only return the image; if `True`,
                  return a tuple of image and max intensity
        :param channel_names: provide the channel names for the OME metadata
        """
        return None

    def read_volume(self,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    rescale=True,
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
        :param rescale: `True` to rescale the intensity scale to 0 and 1; `False` to
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
    def supports_format(cls, image_file, allow_open=True, volume=False):
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
        Any opened files should be closed before returning.

        The volume parameter specifies whether the reader will need to return a 3D array.
        ."""
        # Todo: Allow_open is a setting
        return -1

    @classmethod
    def clear_cached_readers(cls):
        # This should clear any cached reader objects if your class stores unused readers.
        pass

    @abstractmethod
    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        pass

    @abstractmethod
    def get_series_dimensions(self):
        """Should return a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        """
        pass
