import imaging.eukaryote.image.image
import imaging.eukaryote.image.image_set
import numpy
import unittest

IMAGE_NAME = "image"


class TestImageSet(unittest.TestCase):
    def test_01_01_add(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(np.zeros((10, 20))))
        self.assertEqual(len(x.providers), 1)
        self.assertEqual(x.providers[0].name, IMAGE_NAME)

    def test_01_02_get_image(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(np.zeros((10, 20))))
        image = x.get_image(IMAGE_NAME)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_02_01_must_be_binary(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20), bool)))
        image = x.get_image(IMAGE_NAME, must_be_binary=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_02_02_must_be_binary_throws(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20), float)))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, must_be_binary=True)

    def test_03_01_must_be_gray(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_03_02_must_be_gray_throws(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.random.uniform(size=(10, 20, 3))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, must_be_grayscale=True)

    def test_03_03_must_be_gray_color(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20))

    def test_04_01_must_be_color(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_color=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))

    def test_04_02_must_be_color_throws(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.random.uniform(size=(10, 20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, must_be_color=True)

    def test_05_01_must_be_rgb(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))

    def test_05_02_must_be_rgb_throws_gray(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.random.uniform(size=(10, 20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, must_be_rgb=True)

    def test_05_03_must_be_rgb_throws_5_channel(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        numpy.random.seed(22)
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.random.uniform(size=(10, 20, 5))))

    def test_05_04_must_be_rgb_alpha(self):
        x = imaging.eukaryote.image.image_set.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, imaging.eukaryote.image.image.Image(numpy.zeros((10, 20, 4), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10, 20, 3))
