import unittest

import cellprofiler.image
import numpy


class TestImage(unittest.TestCase):
    def test_00_00_init(self):
        cellprofiler.image.Image()

    def test_01_01_init_image(self):
        x = cellprofiler.image.Image(numpy.zeros((10, 10)))

    def test_01_02_init_image_mask(self):
        x = cellprofiler.image.Image(data=numpy.zeros((10, 10)),
                                     mask=numpy.ones((10, 10), dtype=numpy.bool))

    def test_02_01_set_image(self):
        x = cellprofiler.image.Image()
        x.Image = numpy.ones((10, 10))

    def test_02_02_set_mask(self):
        x = cellprofiler.image.Image()
        x.Mask = numpy.ones((10, 10))

    def test_04_01_image_mask_missize(self):
        x = cellprofiler.image.Image()
        x.image = numpy.ones((10, 10))
        self.assertRaises(AssertionError, x.set_mask, numpy.ones((5, 5)))

    def test_05_01_mask_of3D(self):
        """The mask of a 3-d image should be 2-d"""
        x = cellprofiler.image.Image()
        x.image = numpy.ones((10, 10, 3))
        self.assertTrue(x.mask.ndim == 2)

    def test_06_01_cropping(self):
        x = cellprofiler.image.Image()
        x.pixel_data = numpy.ones((7, 7))
        crop_mask = numpy.zeros((10, 10), bool)
        crop_mask[2:-1, 1:-2] = True
        x.crop_mask = crop_mask
        i, j = numpy.mgrid[0:10, 0:10]
        test = i + j * 10
        test_out = x.crop_image_similarly(test)
        self.assertTrue(numpy.all(test_out == test[2:-1, 1:-2]))

IMAGE_NAME = "image"


class TestImageSet(unittest.TestCase):
    def test_01_01_add(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20))))
        self.assertEqual(len(x.providers), 1)
        self.assertEqual(x.providers[0].name, IMAGE_NAME)

    def test_01_02_get_image(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20))))
        image = x.get_image(IMAGE_NAME)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_02_01_must_be_binary(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20), bool)))
        image = x.get_image(IMAGE_NAME, must_be_binary=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_02_02_must_be_binary_throws(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20), float)))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME,
                          must_be_binary=True)

    def test_03_01_must_be_gray(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_03_02_must_be_gray_throws(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.random.uniform(size=(10, 20, 3))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME,
                          must_be_grayscale=True)

    def test_03_03_must_be_gray_color(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_04_01_must_be_color(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_color=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))

    def test_04_02_must_be_color_throws(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.random.uniform(size=(10, 20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME,
                          must_be_color=True)

    def test_05_01_must_be_rgb(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))

    def test_05_02_must_be_rgb_throws_gray(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.random.uniform(size=(10, 20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME,
                          must_be_rgb=True)

    def test_05_03_must_be_rgb_throws_5_channel(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.random.uniform(size=(10, 20, 5))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME,
                          must_be_rgb=True)

    def test_05_04_must_be_rgb_alpha(self):
        x = cellprofiler.image.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cellprofiler.image.Image(numpy.zeros((10, 20, 4), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))


class TestImageSetList(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.image.ImageSetList()
        self.assertEqual(x.count(), 0, "# of elements of an empty image set list is %d, not zero" % (x.count()))

    def test_01_01_add_image_set_by_number(self):
        x = cellprofiler.image.ImageSetList()
        y = x.get_image_set(0)
        self.assertEqual(x.count(), 1, "# of elements was %d, should be 1" % (x.count()))
        self.assertEqual(y.number, 0, "The image set should be #0, was %d" % (y.number))
        self.assertTrue(y.keys.has_key("number"), "The image set was missing a number key")
        self.assertEqual(y.keys["number"], 0,
                         "The number key should be zero, was %s" % (repr(y.keys["number"])))

    def test_01_02_add_image_set_by_key(self):
        x = cellprofiler.image.ImageSetList()
        key = {"key": "value"}
        y = x.get_image_set(key)
        self.assertEqual(x.count(), 1, "# of elements was %d, should be 1" % (x.count()))
        self.assertEqual(y.number, 0, "The image set should be #0, was %d" % (y.number))
        self.assertEquals(y, x.get_image_set(0), "The image set should be retrievable by index")
        self.assertEquals(y, x.get_image_set(key), "The image set should be retrievable by key")
        self.assertEquals(repr(key), repr(y.keys))

    def test_01_03_add_two_image_sets(self):
        x = cellprofiler.image.ImageSetList()
        y = x.get_image_set(0)
        z = x.get_image_set(1)
        self.assertEqual(x.count(), 2, "# of elements was %d, should be 2" % (x.count()))
        self.assertEqual(y.number, 0, "The image set should be #0, was %d" % (y.number))
        self.assertEqual(z.number, 1, "The image set should be #1, was %d" % (y.number))
        self.assertEquals(y, x.get_image_set(0), "The first image set was not retrieved by index")
        self.assertEquals(z, x.get_image_set(1), "The second image set was not retrieved by index")
