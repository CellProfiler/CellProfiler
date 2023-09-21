import numpy
import numpy.random
import numpy.testing

from cellprofiler_core.image import Image


class TestImage:
    def test_multichannel(self):
        x = numpy.zeros((224, 224, 3), numpy.float32)

        image = Image(x)

        assert image.multichannel

    def test_volumetric(self):
        x = numpy.zeros((100, 224, 224, 3), numpy.float32)

        image = Image(x, dimensions=3)

        assert image.volumetric

    def test_spacing(self):
        x = numpy.zeros((100, 224, 224, 3), numpy.float32)

        image = Image(x, dimensions=3)

        assert image.spacing == (1.0, 1.0, 1.0)

    def test_get_image(self):
        x = numpy.zeros((224, 224, 3), numpy.float32)

        image = Image(x)

        numpy.testing.assert_array_equal(image.get_image(), x)

    def test_set_image(self):
        x = numpy.zeros((224, 224, 3), numpy.float32)

        image = Image(x)

        y = numpy.zeros((224, 224, 3), numpy.float32)

        image.set_image(y)

        numpy.testing.assert_array_equal(image.get_image(), y)

    def test_has_parent_image(self):
        x = numpy.zeros((224, 224, 3), numpy.float32)

        parent_image = Image(x)

        assert not parent_image.has_parent_image

        image = Image(x, parent_image=parent_image)

        assert image.has_parent_image

    def test_has_masking_objects(self):
        pass

    def test_labels(self):
        pass

    def test_mask(self):
        pass

    def test_has_mask(self):
        pass

    def test_crop_mask(self):
        pass

    def test_has_crop_mask(self):
        pass

    def test_crop_image_similarly(self):
        pass

    def test_file_name(self):
        pass

    def test_path_name(self):
        pass

    def test_has_channel_names(self):
        pass

    def test_scale(self):
        pass
