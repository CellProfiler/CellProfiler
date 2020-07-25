import bioformats
import numpy

import cellprofiler_core.image
import cellprofiler_core.utilities
import cellprofiler_core.utilities.image
from ._url import URL


class Objects(URL):
    """Provide a multi-plane integer image, interpreting an image file as objects"""

    def __init__(self, name, url, series, index):
        cellprofiler_core.image.abstract.file.url._url.URL.__init__(
            self, name, url, rescale=False, series=series, index=index, volume=False
        )

    def provide_image(self, image_set):
        """Load an image from a pathname
        """
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
            img = cellprofiler_core.utilities.image.convert_image_to_objects(
                img
            ).astype(numpy.int32)
            img[img != 0] += offset
            offset += numpy.max(img)
            planes.append(img)

        image = cellprofiler_core.image.Image(
            numpy.dstack(planes),
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            convert=False,
        )
        return image
