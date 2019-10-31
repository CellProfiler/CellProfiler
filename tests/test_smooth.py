from __future__ import absolute_import
import unittest
import numpy as np
import scipy.ndimage
import centrosome.smooth as cpms
from six.moves import range


class TestSmoothWithNoise(unittest.TestCase):
    def test_01_smooth_zero(self):
        """Make sure smooth doesn't crash on all zeros"""
        out = cpms.smooth_with_noise(np.zeros((3, 3)), 5)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1))

    def test_02_smooth_half(self):
        out = cpms.smooth_with_noise(np.ones((100, 100)) * 0.5, 5)
        # Roughly 1/2 (5000) should be above .5 and half below
        # Over 1/2 should be within 1/32 of .5
        self.assertTrue(out.ndim == 2)
        self.assertTrue(out.shape[0] == 100)
        self.assertTrue(out.shape[1] == 100)
        self.assertTrue(
            abs(np.sum(out > 0.5) - 5000) < 300
        )  # unless we are 3sd unlucky
        self.assertTrue(
            np.sum(np.abs(out - 0.5) < 1.0 / 32.0) > 4700
        )  # unless we are > 3sd unlucky

    def test_03_img_1201(self):
        """Regression test of img-1201"""
        #
        # Using internal knowledge: smooth_with_noise always seeds with 0
        #
        np.random.seed(0)
        r = np.random.normal(size=(10, 20))
        img = np.random.uniform(size=(10, 20))
        for bits in range(8, 17):
            result = cpms.smooth_with_noise(img, bits)
            im = img.copy()
            delta = 1.0 / float(2 ** bits)
            im[im < delta] = delta
            expected = np.exp2(
                np.log2(im) + (np.log2(im + 2.0 ** -bits) - np.log2(im)) * r
            )
            expected = np.clip(expected, 0, 1)
            np.testing.assert_almost_equal(result, expected)


class TestSmoothWithFunctionAndMask(unittest.TestCase):
    def function(self, image):
        return scipy.ndimage.gaussian_filter(image, 1.0)

    def test_01_smooth_zero(self):
        """Make sure smooth_with_function_and_mask doesn't crash on all-zeros"""

        result = cpms.smooth_with_function_and_mask(
            np.zeros((10, 10)), self.function, np.zeros((10, 10), bool)
        )
        self.assertTrue(np.all(np.abs(result) < 0.00001))

    def test_02_smooth_masked_square(self):
        """smooth_with_function_and_mask on a masked square does nothing"""

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        mask = np.zeros((10, 10), bool)
        mask[2:7, 3:8] = True
        image[mask] = 1
        result = cpms.smooth_with_function_and_mask(image, self.function, mask)
        self.assertTrue(np.all(np.abs(result[mask] - 1) < 0.00001))

    def test_03_smooth_unmasked_square(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        mask = np.ones((10, 10), bool)
        image[2:7, 3:8] = 1
        expected = self.function(image)
        result = cpms.smooth_with_function_and_mask(image, self.function, mask)
        self.assertTrue(np.all(np.abs(result - expected) < 0.00001))


class TestCircularGaussianKernel(unittest.TestCase):
    def test_01_convolve_1(self):
        """The center of a large uniform image, convolved with a Gaussian should not change value"""
        image = np.ones((100, 100))
        kernel = cpms.circular_gaussian_kernel(1, 3)
        result = scipy.ndimage.convolve(image, kernel)
        self.assertTrue(np.all(np.abs(result[40:60, 40:60] - 1) < 0.00001))

    def test_02_convolve_random(self):
        """Convolve a random image with a large circular Gaussian kernel"""

        np.random.seed(0)
        image = np.random.uniform(size=(100, 100))
        kernel = cpms.circular_gaussian_kernel(1, 10)
        expected = scipy.ndimage.gaussian_filter(image, 1)
        result = scipy.ndimage.convolve(image, kernel)
        self.assertTrue(np.all(np.abs(result - expected) < 0.001))
