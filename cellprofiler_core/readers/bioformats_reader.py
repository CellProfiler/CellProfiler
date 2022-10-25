import collections
import itertools

import numpy

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, \
    BIOFORMATS_IMAGE_EXTENSIONS

from ..reader import Reader

from bioformats.formatreader import get_image_reader, release_image_reader, clear_image_reader_cache

from ..utilities.java import start_java

# bioformats returns 2 for these, imageio reader returns 3
SUPPORTED_EXTENSIONS = {'.tiff', '.tif', '.ome.tif', '.ome.tiff'}
SEMI_SUPPORTED_EXTENSIONS = BIOFORMATS_IMAGE_EXTENSIONS

class BioformatsReader(Reader):
    """ Derive from this abstract Reader class to create your own image reader in Python

    You need to implement the methods below in the derived class.
    """

    reader_name = "Bio-Formats"

    def __init__(self, image_file):
        self.variable_revision_number = 1
        self._reader = None
        start_java()
        super().__init__(image_file)

    def get_reader(self):
        if self._reader is None:

            self._reader = get_image_reader(id(self), url=self.file.url)

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
        return reader.read(
            c=c,
            z=z,
            t=t,
            series=series,
            index=index,
            rescale=rescale,
            wants_max_intensity=wants_max_intensity,
            channel_names=channel_names,
        )

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
        # Whether a volume has planes stored in the z or t axis is often ambiguous.
        # If z-size > 1 we'll use z, else we'll use t. Otherwise user should choose
        # an axis to split in Images.
        reader = self.get_reader()
        bf_reader = reader.rdr
        if series is None:
            series = 0
        bf_reader.setSeries(series)
        z_len = 0
        if z is None:
            z_len = bf_reader.getSizeZ()
            z_range = range(z_len)
        else:
            z_range = [z]
        if t is None:
            if z_len > 1:
                t_range = 1
            else:
                t_range = range(bf_reader.getSizeT())
        else:
            t_range = [t]
        image_stack = []
        for z_index, t_index in itertools.product(z_range, t_range):
            data = reader.read(
                series=series,
                c=c,
                z=z_index,
                t=t_index,
                rescale=rescale,
                XYWH=xywh,
                wants_max_intensity=False,
                channel_names=channel_names,
            )
            image_stack.append(data)
        return numpy.stack(image_stack)

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
        if image_file.url.lower().startswith("omero:"):
            return 1
        if image_file.full_extension in SUPPORTED_EXTENSIONS:
            return 2
        if not allow_open:
            if image_file.file_extension in SEMI_SUPPORTED_EXTENSIONS:
                return 3
            return -1
        else:
            try:
                get_image_reader(id(image_file), url=image_file.url)
                return 3
            except:
                return -1

    @classmethod
    def clear_cached_readers(cls):
        clear_image_reader_cache()

    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        release_image_reader(id(self))
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
        reader = self.get_reader().rdr
        series_count = reader.getSeriesCount()
        meta_dict[MD_SIZE_S] = series_count
        for i in range(series_count):
            reader.setSeries(i)
            meta_dict[MD_SIZE_C].append(reader.getSizeC())
            meta_dict[MD_SIZE_Z].append(reader.getSizeZ())
            meta_dict[MD_SIZE_T].append(reader.getSizeT())
            meta_dict[MD_SIZE_Y].append(reader.getSizeY())
            meta_dict[MD_SIZE_X].append(reader.getSizeX())
        return meta_dict
