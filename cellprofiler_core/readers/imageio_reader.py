import collections

import numpy

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X

from ..reader import Reader

import imageio

SUPPORTED_EXTENSIONS = {'.png', '.bmp', '.jpeg', '.jpg', '.gif'}


class ImageIOReader(Reader):
    """ Derive from this abstract Reader class to create your own image reader in Python

    You need to implement the methods below in the derived class.
    """

    reader_name = "ImageIO"

    def __init__(self, image_file):
        self.variable_revision_number = 1
        self._reader = None
        self._volume = False
        super().__init__(image_file)

    def get_reader(self, volume=False):
        if self._reader is None or volume != self._volume:
            url = self.file.url
            if url.startswith("file:/") and url[6] != '/':
                url = url.replace("file:/", 'file:///')
            if volume:
                self._reader = imageio.get_reader(url, mode='v')
            else:
                self._reader = imageio.get_reader(url, mode='i')
            self._volume = volume
        return self._reader

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
        """Read a single plane from the image file.
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
        reader = self.get_reader()
        if series is None:
            series = 0
        data = reader.get_data(series)
        if c is not None and len(data.shape) > 2:
            data = data[:, :, c, ...]
        elif c is None and len(data.shape) > 2 and data.shape[2] == 4:
            # Remove alpha channel
            data = data[:, :, :3, ...]
        if rescale:
            imax = self.find_scale_to_match_bioformats(data)
            data = data.astype(numpy.float32) / float(imax)
            if wants_max_intensity:
                return data, 1
            return data
        if wants_max_intensity:
            return data, numpy.iinfo(data.dtype).max
        return data

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
        reader = self.get_reader(volume=True)
        if series is None:
            series = 0
        data = reader.get_data(series)
        if c is not None and len(data.shape) > 3:
            data = data[:, :, :,  c, ...]
        if rescale:
            imax = self.find_scale_to_match_bioformats(data)
            data = data.astype(numpy.float32) / float(imax)

            if wants_max_intensity:
                return data, 1
            return data
        return data

    @staticmethod
    def find_scale_to_match_bioformats(data):
        # We'd love to use skimage.exposure.rescale_intensity.
        # But instead we follow the funky custom rescaling that bf uses.
        if data.dtype in (numpy.int8, numpy.uint8):
            return 255
        elif data.dtype in (numpy.int16, numpy.uint16):
            return 65535
        elif data.dtype == numpy.int32:
            return 2 ** 32 - 1
        elif data.dtype == numpy.uint32:
            return 2 ** 32
        else:
            return 1

    @classmethod
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
        if image_file.url.lower().startswith("omero:"):
            return -1
        if image_file.file_extension in SUPPORTED_EXTENSIONS:
            return 2
        return -1

    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        if self._reader:
            self._reader.close()
        self._reader = None

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
        meta_dict = collections.defaultdict(list)
        reader = self.get_reader()
        series_count = reader.get_length()
        meta_dict[MD_SIZE_S] = series_count
        for i in range(series_count):
            data = reader.get_data(index=i)
            dims = data.shape
            meta_dict[MD_SIZE_Z].append(dims[4] if len(dims) > 4 else 1)
            meta_dict[MD_SIZE_T].append(dims[3] if len(dims) > 3 else 1)
            meta_dict[MD_SIZE_C].append(dims[2] if len(dims) > 2 else 1)
            meta_dict[MD_SIZE_Y].append(dims[1])
            meta_dict[MD_SIZE_X].append(dims[0])
        return meta_dict
