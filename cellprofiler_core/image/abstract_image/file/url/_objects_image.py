import bioformats
import imageio
import numpy

from .... import Image
from .....utilities.image import convert_image_to_objects
from .....utilities.pathname import url2pathname
from ._url_image import URLImage


class ObjectsImage(URLImage):
    """Provide a multi-plane integer image, interpreting an image file as objects"""

    def __init__(self, name, url, series, index, volume=False, spacing=None):
        self.__data = None
        self.volume = volume
        if volume:
            index = self.get_indexes(url)
            series = None
        self.__image = None
        self.__spacing = spacing
        URLImage.__init__(
            self, name, url, rescale=False, series=series, index=index, volume=volume
        )

    def provide_image(self, image_set):
        """Load an image from a pathname
        """

        if self.__image is not None:
            return self.__image

        if self.volume:
            return self.get_image_volume()

        self.cache_file()
        filename = self.get_filename()
        channel_names = []
        url = self.get_url()
        properties = {}
        if self.index is None:
            metadata = bioformats.get_omexml_metadata(self.get_full_name())

            ometadata = bioformats.omexml.OMEXML(metadata)
            pixel_metadata = ometadata.image(
                0 if self.series is None else self.series
            ).Pixels
            nplanes = pixel_metadata.SizeC * pixel_metadata.SizeZ * pixel_metadata.SizeT
            indexes = list(range(nplanes))
        elif numpy.isscalar(self.index):
            indexes = [self.index]
        else:
            indexes = self.index
        planes = []
        offset = 0
        for i, index in enumerate(indexes):
            properties["index"] = str(index)
            if self.series is not None:
                if numpy.isscalar(self.series):
                    properties["series"] = self.series
                else:
                    properties["series"] = self.series[i]
            img = bioformats.load_image(
                self.get_full_name(), rescale=False, **properties
            ).astype(int)
            img = convert_image_to_objects(img).astype(numpy.int32)
            img[img != 0] += offset
            offset += numpy.max(img)
            planes.append(img)

        image = Image(
            numpy.dstack(planes),
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            convert=False,
        )
        self.__image = image
        return image

    def get_indexes(self, url):
        pathname = url2pathname(url)
        # Javabridge gave us dud indexes, let's find our own planes
        self.__data = imageio.volread(pathname).astype(int)
        indexes = list(range(self.__data.shape[0]))
        return indexes

    def get_image_volume(self):
        imdata = self.__data
        planes = []
        for planeid in range(imdata.shape[0]):
            planes.append(imdata[planeid].astype(numpy.int32))
        imdata = numpy.stack(planes)

        image = Image(
            imdata,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            convert=False,
            dimensions=3,
        )
        self.__image = image
        return image
