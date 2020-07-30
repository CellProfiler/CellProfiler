import numpy.random

from cellprofiler_core.image import GrayscaleImage
from cellprofiler_core.image import Image


class TestGrayscaleImage:
    def test_pixel_data(self):
        data = numpy.random.random((224, 224, 3))

        image = Image(data)

        grayscale_image = GrayscaleImage(image)

        assert grayscale_image.pixel_data.shape == (224, 224)
