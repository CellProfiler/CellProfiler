import collections
import numpy
import imageio

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X
from ..preferences import config_read_typed
from ..reader import Reader


SUPPORTED_EXTENSIONS = {'.png', '.bmp', '.jpeg', '.jpg', '.gif'}
# bioformats returns 2 for these, imageio reader returns 3
SEMI_SUPPORTED_EXTENSIONS = {'.tiff', '.tif', '.ome.tif', '.ome.tiff'}
SUPPORTED_SCHEMES = {'file', 'http', 'https', 'ftp', 'ftps'}

class ImageIOReader(Reader):
    """
    Reads basic image formats using ImageIO.
    """

    reader_name = "ImageIO"
    variable_revision_number = 1
    supported_filetypes = SUPPORTED_EXTENSIONS.union(SEMI_SUPPORTED_EXTENSIONS)
    supported_schemes = SUPPORTED_SCHEMES

    def __init__(self, image_file):
        self.variable_revision_number = 1
        self._reader = None
        self._volume = False
        super().__init__(image_file)

    def __del__(self):
        self.close()

    def get_reader(self, volume=False):
        if self._reader is None or volume != self._volume:
            path = self.file.path
            if volume:
                self._reader = imageio.get_reader(path, mode='v')
            else:
                self._reader = imageio.get_reader(path, mode='i')
            self._volume = volume
        return self._reader

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
        """
        reader = self.get_reader()
        if series is None:
            series = 0
        img = reader.get_data(series)
        # https://imageio.readthedocs.io/en/v2.8.0/devapi.html#imageio.core.Array
        data = numpy.asarray(img)
        if c is not None and len(data.shape) > 2:
            data = data[:, :, c, ...]
        elif c is None and len(data.shape) > 2 and data.shape[2] == 4:
            # Remove alpha channel
            data = data[:, :, :3, ...]
        if wants_metadata_rescale == True:
            # tiff-specific
            scale = getattr(img.meta, "MaxSampleValue", None)
            return data, (0.0, float(scale)) if scale else None
        return data

    def read_volume(self,
                    wants_metadata_rescale=False,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    xywh=None,
                    channel_names=None,
                    ):
        reader = self.get_reader(volume=True)
        if series is None:
            series = 0
        img = reader.get_data(series)
        data = numpy.asarray(img)
        if c is not None and len(data.shape) > 3:
            data = data[:, :, :,  c, ...]
        if wants_metadata_rescale == True:
            # tiff-specific
            scale = getattr(img.meta, "BitsPerSample", None)
            return data, (0.0, float(2**scale-1)) if scale else None
        return data

    @classmethod
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
        Any opened files should be closed before returning.

        The volume parameter specifies whether the reader will need to return a 3D array.
        ."""
        if image_file.scheme not in SUPPORTED_SCHEMES:
            return -1
        if image_file.file_extension in SUPPORTED_EXTENSIONS:
            return 2
        if image_file.full_extension in SEMI_SUPPORTED_EXTENSIONS:
            if config_read_typed(f"Reader.{ImageIOReader.reader_name}.read_tif", bool):
                return 2
            return 3

        return -1

    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        if self._reader:
            self._reader.close()
        self._reader = None

    def get_series_metadata(self):
        """Should return a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        MD_SERIES_NAME - list of series names, one element per series.
        """
        meta_dict = collections.defaultdict(list)
        reader = self.get_reader()
        series_count = reader.get_length()
        meta_dict[MD_SIZE_S] = series_count
        for i in range(series_count):
            data = reader.get_data(index=i)
            dims = data.shape
            # expects dim ordering of: [H, W, C?, T?, Z?]
            meta_dict[MD_SIZE_Z].append(dims[4] if len(dims) > 4 else 1)
            meta_dict[MD_SIZE_T].append(dims[3] if len(dims) > 3 else 1)
            meta_dict[MD_SIZE_C].append(dims[2] if len(dims) > 2 else 1)
            meta_dict[MD_SIZE_X].append(dims[1])
            meta_dict[MD_SIZE_Y].append(dims[0])
        return meta_dict

    @staticmethod
    def get_settings():
        return [
            ('read_tif',
             "Read TIFF files",
             """
             If enabled, this reader will attempt to read TIFF files.
             Note that this reader cannot properly handle complex, multi-series
             TIFF formats or special compression methods. 
             Only enable this option if you're loading simple TIFF images.
             """,
             bool,
             False)
        ]
