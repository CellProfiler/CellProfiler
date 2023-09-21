import math

import numpy

import cellprofiler_core.image


class TestImage:
    def test_init(self):
        cellprofiler_core.image.Image()

    def test_init_image(self):
        x = cellprofiler_core.image.Image(numpy.zeros((10, 10)))

    def test_init_image_mask(self):
        x = cellprofiler_core.image.Image(
            image=numpy.zeros((10, 10)), mask=numpy.ones((10, 10), dtype=bool)
        )

    def test_set_image(self):
        x = cellprofiler_core.image.Image()
        x.Image = numpy.ones((10, 10))

    def test_set_mask(self):
        x = cellprofiler_core.image.Image()
        x.Mask = numpy.ones((10, 10))

    def test_image_casts(self):
        one_target = numpy.ones((10, 10), dtype=numpy.float64)
        zero_target = numpy.zeros((10, 10), dtype=numpy.float64)
        tests = [
            (numpy.float64, 0, 1.0),
            (numpy.float32, 0, 1.0),
            (numpy.uint32, 0, math.pow(2.0, 32.0) - 1),
            (numpy.uint16, 0, math.pow(2.0, 16.0) - 1),
            (numpy.uint8, 0, math.pow(2.0, 8.0) - 1),
            (numpy.int32, -math.pow(2.0, 31.0), math.pow(2.0, 31.0) - 1),
            (numpy.int16, -math.pow(2.0, 15.0), math.pow(2.0, 15.0) - 1),
            (numpy.int8, -math.pow(2.0, 7.0), math.pow(2.0, 7.0) - 1),
        ]
        for dtype, zval, oval in tests:
            x = cellprofiler_core.image.Image()
            x.set_image((one_target * zval).astype(dtype))
            assert (x.image == zero_target).all(), "Failed setting %s to min" % (
                repr(dtype)
            )
            x.set_image((one_target * oval).astype(dtype))
            y = x.image == one_target
            assert (x.image == one_target).all(), "Failed setting %s to max" % (
                repr(dtype)
            )

    def test_mask_of3D(self):
        """The mask of a 3-d image should be 2-d"""
        x = cellprofiler_core.image.Image()
        x.image = numpy.ones((10, 10, 3))
        assert x.mask.ndim == 2

    def test_cropping(self):
        x = cellprofiler_core.image.Image()
        x.image = numpy.ones((7, 7))
        crop_mask = numpy.zeros((10, 10), bool)
        crop_mask[2:-1, 1:-2] = True
        x.crop_mask = crop_mask
        i, j = numpy.mgrid[0:10, 0:10]
        test = i + j * 10
        test_out = x.crop_image_similarly(test)
        assert numpy.all(test_out == test[2:-1, 1:-2])

    def test_init_volume(self):
        data = numpy.ones((5, 10, 10))

        x = cellprofiler_core.image.Image(image=data, dimensions=3)

        assert numpy.all(x.pixel_data == data)

        assert x.dimensions == 3

    def test_multichannel_grayscale_image(self):
        data = numpy.ones((10, 10))

        x = cellprofiler_core.image.Image(image=data)

        assert not x.multichannel

    def test_multichannel_rgb_image(self):
        data = numpy.ones((10, 10, 3))

        x = cellprofiler_core.image.Image(image=data)

        assert x.multichannel

    def test_multichannel_grayscale_volume(self):
        data = numpy.ones((5, 10, 10))

        x = cellprofiler_core.image.Image(image=data, dimensions=3)

        assert not x.multichannel

    def test_spacing_image_default(self):
        data = numpy.ones((5, 5))

        x = cellprofiler_core.image.Image(image=data)

        assert x.spacing == (1.0, 1.0)

    def test_spacing_image(self):
        data = numpy.ones((5, 5))

        x = cellprofiler_core.image.Image(image=data, spacing=(0.33, 0.33))

        assert x.spacing == (1.0, 1.0)

    def test_spacing_parent_image(self):
        data = numpy.ones((5, 5))

        px = cellprofiler_core.image.Image(image=data, spacing=(0.33, 0.33))

        x = cellprofiler_core.image.Image(image=data, parent_image=px)

        assert x.spacing == (1.0, 1.0)

    def test_spacing_volume_default(self):
        data = numpy.ones((5, 10, 10))

        x = cellprofiler_core.image.Image(image=data, dimensions=3)

        assert x.spacing == (1.0, 1.0, 1.0)

    def test_spacing_volume(self):
        data = numpy.ones((5, 10, 10))

        x = cellprofiler_core.image.Image(
            image=data, dimensions=3, spacing=(0.77, 0.33, 0.33)
        )

        assert x.spacing == (0.77 / 0.33, 1.0, 1.0)

    def test_spacing_volume_parent_image(self):
        data = numpy.ones((5, 10, 10))

        px = cellprofiler_core.image.Image(
            image=data, dimensions=3, spacing=(0.77, 0.33, 0.33)
        )

        x = cellprofiler_core.image.Image(
            image=data, parent_image=px, spacing=(0.77, 0.33, 0.33)
        )

        assert x.spacing == (0.77 / 0.33, 1.0, 1.0)

    def test_channelstack(self):
        data = numpy.ones((5, 10, 10))

        x = cellprofiler_core.image.Image(image=data)

        assert x.channelstack == False

        x.channelstack = True

        assert x.channelstack == True
