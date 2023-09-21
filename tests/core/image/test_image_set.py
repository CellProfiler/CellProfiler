import numpy
import pytest

import cellprofiler_core.image


class TestImageSet:
    def test_add(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20))))
        assert len(x.providers) == 1
        assert x.providers[0].name == "image"

    def test_get_image(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20))))
        image = x.get_image("image")
        assert tuple(image.pixel_data.shape) == (10, 20)

    def test_must_be_binary(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20), bool)))
        image = x.get_image("image", must_be_binary=True)
        assert tuple(image.pixel_data.shape) == (10, 20)

    def test_must_be_binary_throws(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20), float)))
        with pytest.raises(ValueError):
            x.get_image("image", must_be_binary=True)

    def test_must_be_gray(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20), float)))
        image = x.get_image("image", must_be_grayscale=True)
        assert tuple(image.pixel_data.shape) == (10, 20)

    def test_must_be_gray_throws(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(
            "image",
            cellprofiler_core.image.Image(numpy.random.uniform(size=(10, 20, 3))),
        )
        with pytest.raises(ValueError):
            x.get_image("image", must_be_grayscale=True)

    def test_must_be_gray_color(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image("image", must_be_grayscale=True)
        assert tuple(image.pixel_data.shape) == (10, 20)

    def test_must_be_color(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image("image", must_be_color=True)
        assert tuple(image.pixel_data.shape) == (10, 20, 3)

    def test_must_be_color_throws(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(
            "image", cellprofiler_core.image.Image(numpy.random.uniform(size=(10, 20)))
        )
        with pytest.raises(ValueError):
            x.get_image("image", must_be_color=True)

    def test_must_be_rgb(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image("image", must_be_rgb=True)
        assert tuple(image.pixel_data.shape) == (10, 20, 3)

    def test_must_be_rgb_throws_gray(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(
            "image", cellprofiler_core.image.Image(numpy.random.uniform(size=(10, 20)))
        )
        with pytest.raises(ValueError):
            x.get_image("image", must_be_rgb=True)

    def test_must_be_rgb_throws_5_channel(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(
            "image",
            cellprofiler_core.image.Image(numpy.random.uniform(size=(10, 20, 5))),
        )
        with pytest.raises(ValueError):
            x.get_image("image", must_be_rgb=True)

    def test_must_be_rgb_alpha(self):
        x = cellprofiler_core.image.ImageSet(0, {}, {})
        x.add("image", cellprofiler_core.image.Image(numpy.zeros((10, 20, 4), float)))
        image = x.get_image("image", must_be_rgb=True)
        assert tuple(image.pixel_data.shape) == (10, 20, 3)
