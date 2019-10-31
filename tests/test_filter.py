from __future__ import absolute_import
from __future__ import division
import base64
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, convolve
import unittest
import pytest

import centrosome.filter as F
from six.moves import range

"""Perform line-integration per-column of the image"""
VERTICAL = "vertical"
"""Perform line-integration per-row of the image"""
HORIZONTAL = "horizontal"
"""Perform line-integration along diagonals from top left to bottom right"""
DIAGONAL = "diagonal"
"""Perform line-integration along diagonals from top right to bottom left"""
ANTI_DIAGONAL = "anti-diagonal"


class TestStretch(unittest.TestCase):
    def test_00_00_empty(self):
        result = F.stretch(np.zeros((0,)))
        self.assertEqual(len(result), 0)

    def test_00_01_empty_plus_mask(self):
        result = F.stretch(np.zeros((0,)), np.zeros((0,), bool))
        self.assertEqual(len(result), 0)

    def test_00_02_zeros(self):
        result = F.stretch(np.zeros((10, 10)))
        self.assertTrue(np.all(result == 0))

    def test_00_03_zeros_plus_mask(self):
        result = F.stretch(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_04_half(self):
        result = F.stretch(np.ones((10, 10)) * 0.5)
        self.assertTrue(np.all(result == 0.5))

    def test_00_05_half_plus_mask(self):
        result = F.stretch(np.ones((10, 10)) * 0.5, np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0.5))

    def test_01_01_rescale(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10, 10))
        image[0, 0] = -2
        image[9, 9] = 2
        expected = (image + 2.0) / 4.0
        result = F.stretch(image)
        self.assertTrue(np.all(result == expected))

    def test_01_02_rescale_plus_mask(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10, 10))
        mask = np.zeros((10, 10), bool)
        mask[1:9, 1:9] = True
        image[0, 0] = -4
        image[9, 9] = 4
        image[1, 1] = -2
        image[8, 8] = 2
        expected = (image[1:9, 1:9] + 2.0) / 4.0
        result = F.stretch(image, mask)
        self.assertTrue(np.all(result[1:9, 1:9] == expected))


class TestMedianFilter(unittest.TestCase):
    def test_00_00_zeros(self):
        """The median filter on an array of all zeros should be zero"""
        result = F.median_filter(np.zeros((10, 10)), np.ones((10, 10), bool), 3)
        self.assertTrue(np.all(result == 0))

    def test_00_01_all_masked(self):
        """Test a completely masked image

        Regression test of IMG-1029"""
        result = F.median_filter(np.zeros((10, 10)), np.zeros((10, 10), bool), 3)
        self.assertTrue(np.all(result == 0))

    def test_00_02_all_but_one_masked(self):
        mask = np.zeros((10, 10), bool)
        mask[5, 5] = True
        result = F.median_filter(np.zeros((10, 10)), mask, 3)

    def test_01_01_mask(self):
        """The median filter, masking a single value"""
        img = np.zeros((10, 10))
        img[5, 5] = 1
        mask = np.ones((10, 10), bool)
        mask[5, 5] = False
        result = F.median_filter(img, mask, 3)
        self.assertTrue(np.all(result[mask] == 0))

    def test_02_01_median(self):
        """A median filter larger than the image = median of image"""
        np.random.seed(0)
        img = np.random.uniform(size=(9, 9))
        result = F.median_filter(img, np.ones((9, 9), bool), 20)
        self.assertEqual(result[0, 0], np.median(img))
        self.assertTrue(np.all(result == np.median(img)))

    def test_02_02_median_bigger(self):
        """Use an image of more than 255 values to test approximation"""
        np.random.seed(0)
        img = np.random.uniform(size=(20, 20))
        result = F.median_filter(img, np.ones((20, 20), bool), 40)
        sorted = np.ravel(img)
        sorted.sort()
        min_acceptable = sorted[198]
        max_acceptable = sorted[202]
        self.assertTrue(np.all(result >= min_acceptable))
        self.assertTrue(np.all(result <= max_acceptable))

    def test_03_01_shape(self):
        """Make sure the median filter is the expected octagonal shape"""

        radius = 5
        a_2 = int(radius / 2.414213)
        i, j = np.mgrid[-10:11, -10:11]
        octagon = np.ones((21, 21), bool)
        #
        # constrain the octagon mask to be the points that are on
        # the correct side of the 8 edges
        #
        octagon[i < -radius] = False
        octagon[i > radius] = False
        octagon[j < -radius] = False
        octagon[j > radius] = False
        octagon[i + j < -radius - a_2] = False
        octagon[j - i > radius + a_2] = False
        octagon[i + j > radius + a_2] = False
        octagon[i - j > radius + a_2] = False
        np.random.seed(0)
        img = np.random.uniform(size=(21, 21))
        result = F.median_filter(img, np.ones((21, 21), bool), radius)
        sorted = img[octagon]
        sorted.sort()
        min_acceptable = sorted[len(sorted) // 2 - 1]
        max_acceptable = sorted[len(sorted) // 2 + 1]
        self.assertTrue(result[10, 10] >= min_acceptable)
        self.assertTrue(result[10, 10] <= max_acceptable)

    def test_04_01_half_masked(self):
        """Make sure that the median filter can handle large masked areas."""
        img = np.ones((20, 20))
        mask = np.ones((20, 20), bool)
        mask[10:, :] = False
        img[~mask] = 2
        img[1, 1] = 0  # to prevent short circuit for uniform data.
        result = F.median_filter(img, mask, 5)
        # in partial coverage areas, the result should be only from the masked pixels
        self.assertTrue(np.all(result[:14, :] == 1))
        # in zero coverage areas, the result should be the lowest valud in the valid area
        self.assertTrue(np.all(result[15:, :] == np.min(img[mask])))


@pytest.mark.skipif(sys.version_info > (3, 0), reason="requires Python 2.7")
class TestBilateralFilter(unittest.TestCase):
    def test_00_00_zeros(self):
        """Test the bilateral filter of an array of all zeros"""
        result = F.bilateral_filter(
            np.zeros((10, 10)), np.ones((10, 10), bool), 5.0, 0.1
        )
        self.assertTrue(np.all(result == 0))

    def test_00_01_all_masked(self):
        """Test the bilateral filter of a completely masked array"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        result = F.bilateral_filter(image, np.zeros((10, 10), bool), 5.0, 0.1)
        self.assertTrue(np.all(result == image))


class TestLaplacianOfGaussian(unittest.TestCase):
    def test_00_00_zeros(self):
        result = F.laplacian_of_gaussian(np.zeros((10, 10)), None, 9, 3)
        self.assertTrue(np.all(result == 0))

    def test_00_01_zeros_mask(self):
        result = F.laplacian_of_gaussian(
            np.zeros((10, 10)), np.zeros((10, 10), bool), 9, 3
        )
        self.assertTrue(np.all(result == 0))

    def test_01_01_ring(self):
        """The LoG should have its lowest value in the center of the ring"""
        i, j = np.mgrid[-20:21, -20:21].astype(float)
        # A ring of radius 3, more or less
        image = (np.abs(i ** 2 + j ** 2 - 3) < 2).astype(float)
        result = F.laplacian_of_gaussian(image, None, 9, 3)
        self.assertTrue(
            (np.argmin(result) % 41, int(np.argmin(result) / 41)) == (20, 20)
        )


class TestCanny(unittest.TestCase):
    def test_00_00_zeros(self):
        """Test that the Canny filter finds no points for a blank field"""
        result = F.canny(np.zeros((20, 20)), np.ones((20, 20), bool), 4, 0, 0)
        self.assertFalse(np.any(result))

    def test_00_01_zeros_mask(self):
        """Test that the Canny filter finds no points in a masked image"""
        result = F.canny(
            np.random.uniform(size=(20, 20)), np.zeros((20, 20), bool), 4, 0, 0
        )
        self.assertFalse(np.any(result))

    def test_01_01_circle(self):
        """Test that the Canny filter finds the outlines of a circle"""
        i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        result = F.canny(c.astype(float), np.ones(c.shape, bool), 4, 0, 0)
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=3)
        ce = binary_erosion(c, iterations=3)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        #
        # The circle has a radius of 100. There are two rings here, one
        # for the inside edge and one for the outside. So that's 100 * 2 * 2 * 3
        # for those places where pi is still 3. The edge contains both pixels
        # if there's a tie, so we bump the count a little.
        #
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)

    def test_01_02_circle_with_noise(self):
        """Test that the Canny filter finds the circle outlines in a noisy image"""
        np.random.seed(0)
        i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        cf = c.astype(float) * 0.5 + np.random.uniform(size=c.shape) * 0.5
        result = F.canny(cf, np.ones(c.shape, bool), 4, 0.1, 0.2)
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=4)
        ce = binary_erosion(c, iterations=4)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)


class TestRoberts(unittest.TestCase):
    def test_00_00_zeros(self):
        """Roberts on an array of all zeros"""
        result = F.roberts(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Roberts on a masked array should be zero"""
        np.random.seed(0)
        result = F.roberts(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01(self):
        """Roberts on a diagonal edge should recreate the diagonal line"""

        i, j = np.mgrid[0:10, 0:10]
        image = (i >= j).astype(float)
        result = F.roberts(image)
        #
        # Do something a little sketchy to keep from measuring the points
        # at 0,0 and -1,-1 which are eroded
        #
        i[0, 0] = 10000
        i[-1, -1] = 10000
        self.assertTrue(np.all(result[i == j] == 1))
        self.assertTrue(np.all(result[np.abs(i - j) > 1] == 0))

    def test_01_02(self):
        """Roberts on an anti-diagonal edge should recreate the line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i > -j).astype(float)
        result = F.roberts(image)
        i[0, -1] = 10000
        i[-1, 0] = 10000
        self.assertTrue(np.all(result[i == -j] == 1))
        self.assertTrue(np.all(result[np.abs(i + j) > 1] == 0))


class TestSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        """Sobel on an array of all zeros"""
        result = F.sobel(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Sobel on a masked array should be zero"""
        np.random.seed(0)
        result = F.sobel(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_horizontal(self):
        """Sobel on an edge should be a horizontal line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.sobel(image)
        # Fudge the eroded points
        i[np.abs(j) == 5] = 10000
        self.assertTrue(np.all(result[i == 0] == 1))
        self.assertTrue(np.all(result[np.abs(i) > 1] == 0))

    def test_01_02_vertical(self):
        """Sobel on a vertical edge should be a vertical line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.sobel(image)
        j[np.abs(i) == 5] = 10000
        self.assertTrue(np.all(result[j == 0] == 1))
        self.assertTrue(np.all(result[np.abs(j) > 1] == 0))


class TestHSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        """Horizontal sobel on an array of all zeros"""
        result = F.hsobel(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Horizontal Sobel on a masked array should be zero"""
        np.random.seed(0)
        result = F.hsobel(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_horizontal(self):
        """Horizontal Sobel on an edge should be a horizontal line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.hsobel(image)
        # Fudge the eroded points
        i[np.abs(j) == 5] = 10000
        self.assertTrue(np.all(result[i == 0] == 1))
        self.assertTrue(np.all(result[np.abs(i) > 1] == 0))

    def test_01_02_vertical(self):
        """Horizontal Sobel on a vertical edge should be zero"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.hsobel(image)
        self.assertTrue(np.all(result == 0))


class TestVSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        """Vertical sobel on an array of all zeros"""
        result = F.vsobel(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Vertical Sobel on a masked array should be zero"""
        np.random.seed(0)
        result = F.vsobel(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_vertical(self):
        """Vertical Sobel on an edge should be a vertical line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.vsobel(image)
        # Fudge the eroded points
        j[np.abs(i) == 5] = 10000
        self.assertTrue(np.all(result[j == 0] == 1))
        self.assertTrue(np.all(result[np.abs(j) > 1] == 0))

    def test_01_02_horizontal(self):
        """vertical Sobel on a horizontal edge should be zero"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.vsobel(image)
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result) < eps))


class TestPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        """Prewitt on an array of all zeros"""
        result = F.prewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Prewitt on a masked array should be zero"""
        np.random.seed(0)
        result = F.prewitt(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result) < eps))

    def test_01_01_horizontal(self):
        """Prewitt on an edge should be a horizontal line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.prewitt(image)
        # Fudge the eroded points
        i[np.abs(j) == 5] = 10000
        eps = 0.000001
        self.assertTrue(np.all(result[i == 0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(i) > 1]) < eps))

    def test_01_02_vertical(self):
        """Prewitt on a vertical edge should be a vertical line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.prewitt(image)
        eps = 0.000001
        j[np.abs(i) == 5] = 10000
        self.assertTrue(np.all(result[j == 0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(j) > 1]) < eps))


class TestHPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        """Horizontal sobel on an array of all zeros"""
        result = F.hprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Horizontal prewitt on a masked array should be zero"""
        np.random.seed(0)
        result = F.hprewitt(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result) < eps))

    def test_01_01_horizontal(self):
        """Horizontal prewitt on an edge should be a horizontal line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.hprewitt(image)
        # Fudge the eroded points
        i[np.abs(j) == 5] = 10000
        eps = 0.000001
        self.assertTrue(np.all(result[i == 0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(i) > 1]) < eps))

    def test_01_02_vertical(self):
        """Horizontal prewitt on a vertical edge should be zero"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.hprewitt(image)
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result) < eps))


class TestVPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        """Vertical prewitt on an array of all zeros"""
        result = F.vprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_00_01_mask(self):
        """Vertical prewitt on a masked array should be zero"""
        np.random.seed(0)
        result = F.vprewitt(np.random.uniform(size=(10, 10)), np.zeros((10, 10), bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_vertical(self):
        """Vertical prewitt on an edge should be a vertical line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.vprewitt(image)
        # Fudge the eroded points
        j[np.abs(i) == 5] = 10000
        self.assertTrue(np.all(result[j == 0] == 1))
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result[np.abs(j) > 1]) < eps))

    def test_01_02_horizontal(self):
        """vertical prewitt on a horizontal edge should be zero"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.vprewitt(image)
        eps = 0.000001
        self.assertTrue(np.all(np.abs(result) < eps))


class TestEnhanceDarkHoles(unittest.TestCase):
    def test_00_00_zeros(self):
        result = F.enhance_dark_holes(np.zeros((15, 19)), 1, 5)
        self.assertTrue(np.all(result == 0))

    def test_01_01_positive(self):
        """See if we pick up holes of given sizes"""

        i, j = np.mgrid[-25:26, -25:26].astype(float)
        for r in range(5, 11):
            image = (np.abs(np.sqrt(i ** 2 + j ** 2) - r) <= 0.5).astype(float)
            eimg = F.enhance_dark_holes(image, r - 1, r)
            self.assertTrue(np.all(eimg[np.sqrt(i ** 2 + j ** 2) < r - 1] == 1))
            self.assertTrue(np.all(eimg[np.sqrt(i ** 2 + j ** 2) >= r] == 0))

    def test_01_01_negative(self):
        """See if we miss holes of the wrong size"""
        i, j = np.mgrid[-25:26, -25:26].astype(float)
        for r in range(5, 11):
            image = (np.abs(np.sqrt(i ** 2 + j ** 2) - r) <= 0.5).astype(float)
            for lo, hi in ((r - 3, r - 2), (r + 1, r + 2)):
                eimg = F.enhance_dark_holes(image, lo, hi)
                self.assertTrue(np.all(eimg == 0))


class TestKalmanFilter(unittest.TestCase):
    def test_00_00_none(self):
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.zeros(0, int),
            np.zeros((0, 2)),
            np.zeros((0, 4, 4)),
            np.zeros((0, 2, 2)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 0)

    def test_01_01_add_one(self):
        np.random.seed(11)
        locs = np.random.randint(0, 1000, size=(1, 2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.ones(1, int) * -1,
            locs,
            np.zeros((1, 4, 4)),
            np.zeros((1, 2, 2)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))

    def test_01_02_same_loc_twice(self):
        np.random.seed(12)
        locs = np.random.randint(0, 1000, size=(1, 2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.ones(1, int) * -1,
            locs,
            np.zeros((1, 4, 4)),
            np.zeros((1, 2, 2)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))

        result = F.kalman_filter(
            result,
            np.zeros(1, int),
            locs,
            np.eye(4)[np.newaxis, :, :],
            np.eye(2)[np.newaxis, :, :],
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

    def test_01_03_same_loc_thrice(self):
        np.random.seed(13)
        locs = np.random.randint(0, 1000, size=(1, 2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.ones(1, int) * -1,
            locs,
            np.zeros((1, 4, 4)),
            np.zeros((1, 2, 2)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))

        result = F.kalman_filter(
            result,
            np.zeros(1, int),
            locs,
            np.eye(4)[np.newaxis, :, :],
            np.eye(2)[np.newaxis, :, :],
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))
        #
        # The third time through exercises some code to join the state_noise
        #
        result = F.kalman_filter(
            result,
            np.zeros(1, int),
            locs,
            np.eye(4)[np.newaxis, :, :],
            np.eye(2)[np.newaxis, :, :],
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

    def test_01_04_disappear(self):
        np.random.seed(13)
        locs = np.random.randint(0, 1000, size=(1, 2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.ones(1, int) * -1,
            locs,
            np.zeros((1, 4, 4)),
            np.zeros((1, 2, 2)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))

        result = F.kalman_filter(
            kalman_state,
            np.zeros(0, int),
            np.zeros((0, 2)),
            np.zeros((0, 4, 4)),
            np.zeros((0, 2, 2)),
        )
        self.assertEqual(len(result.state_vec), 0)

    def test_01_05_follow_2(self):
        np.random.seed(15)
        locs = np.random.randint(0, 1000, size=(2, 2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(
            kalman_state,
            np.ones(2, int) * -1,
            locs,
            np.zeros((0, 2, 2)),
            np.zeros((0, 4, 4)),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))

        result = F.kalman_filter(
            result,
            np.arange(2),
            locs,
            np.array([np.eye(4)] * 2),
            np.array([np.eye(2)] * 2),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

        result = F.kalman_filter(
            result,
            np.arange(2),
            locs,
            np.array([np.eye(4)] * 2),
            np.array([np.eye(2)] * 2),
        )
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:, :2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

    def test_01_06_follow_with_movement(self):
        np.random.seed(16)
        vi = 5
        vj = 7
        e = np.ones(2)
        locs = np.random.randint(0, 1000, size=(1, 2))
        kalman_state = F.velocity_kalman_model()
        for i in range(100):
            kalman_state = F.kalman_filter(
                kalman_state,
                [0] if i > 0 else [-1],
                locs,
                np.eye(4)[np.newaxis, :, :] * 2,
                np.eye(2)[np.newaxis, :, :] * 0.5,
            )
            locs[0, 0] += vi
            locs[0, 1] += vj
            if i > 0:
                new_e = np.abs(kalman_state.predicted_obs_vec[0] - locs[0])
                self.assertTrue(np.all(new_e <= e + np.finfo(np.float32).eps))
                e = new_e

    def test_01_07_scramble_and_follow(self):
        np.random.seed(17)
        nfeatures = 20
        v = np.random.uniform(size=(nfeatures, 2)) * 5
        locs = np.random.uniform(size=(nfeatures, 2)) * 100
        e = np.ones((nfeatures, 2))
        q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
        r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 0.5

        kalman_state = F.kalman_filter(
            F.velocity_kalman_model(), -np.ones(nfeatures, int), locs, q, r
        )
        locs += v
        for i in range(100):
            scramble = np.random.permutation(np.arange(nfeatures))
            # scramble = np.arange(nfeatures)
            locs = locs[scramble]
            v = v[scramble]
            e = e[scramble]
            kalman_state = F.kalman_filter(kalman_state, scramble, locs, q, r)
            locs += v
            new_e = np.abs(kalman_state.predicted_obs_vec - locs)
            self.assertTrue(np.all(new_e <= e + np.finfo(np.float32).eps))
            e = new_e

    def test_01_08_scramble_add_and_remove(self):
        np.random.seed(18)
        nfeatures = 20
        v = np.random.uniform(size=(nfeatures, 2)) * 5
        locs = np.random.uniform(size=(nfeatures, 2)) * 100
        e = np.ones((nfeatures, 2))
        q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
        r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 0.5

        kalman_state = F.kalman_filter(
            F.velocity_kalman_model(), -np.ones(nfeatures, int), locs, q, r
        )
        locs += v
        for i in range(100):
            add = np.random.randint(1, 10)
            remove = np.random.randint(1, nfeatures - 1)
            scramble = np.random.permutation(np.arange(nfeatures))[remove:]
            locs = locs[scramble]
            v = v[scramble]
            e = e[scramble]
            new_v = np.random.uniform(size=(add, 2)) * 5
            new_locs = np.random.uniform(size=(add, 2)) * 100
            new_e = np.ones((add, 2))
            scramble = np.hstack((scramble, -np.ones(add, int)))
            v = np.vstack((v, new_v))
            locs = np.vstack((locs, new_locs))
            e = np.vstack((e, new_e))
            nfeatures += add - remove
            q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
            r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 0.5
            kalman_state = F.kalman_filter(kalman_state, scramble, locs, q, r)
            locs += v
            new_e = np.abs(kalman_state.predicted_obs_vec - locs)
            self.assertTrue(np.all(new_e[:-add] <= e[:-add] + np.finfo(np.float32).eps))
            e = new_e

    def test_02_01_with_noise(self):
        np.random.seed(21)
        nfeatures = 20
        nsteps = 200
        vq = np.random.uniform(size=nfeatures) * 2
        vr = np.random.uniform(size=nfeatures) * 0.5
        sdq = np.sqrt(vq)
        sdr = np.sqrt(vr)
        v = np.random.uniform(size=(nfeatures, 2)) * 10
        locs = np.random.uniform(size=(nfeatures, 2)) * 200
        locs = (
            locs[np.newaxis, :, :]
            + np.arange(nsteps)[:, np.newaxis, np.newaxis] * v[np.newaxis, :, :]
        )
        process_error = np.random.normal(
            scale=sdq, size=(nsteps, 2, nfeatures)
        ).transpose((0, 2, 1))
        measurement_error = np.random.normal(
            scale=sdr, size=(nsteps, 2, nfeatures)
        ).transpose((0, 2, 1))
        locs = locs + np.cumsum(process_error, 0)
        meas = locs + measurement_error
        q = (
            np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)]
            * vq[:, np.newaxis, np.newaxis]
        )
        r = (
            np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)]
            * vr[:, np.newaxis, np.newaxis]
        )

        obs = np.zeros((nsteps, nfeatures, 2))
        kalman_state = F.kalman_filter(
            F.velocity_kalman_model(), -np.ones(nfeatures, int), meas[0], q, r
        )
        obs[0] = kalman_state.state_vec[:, :2]
        for i in range(1, nsteps):
            kalman_state = F.kalman_filter(
                kalman_state, np.arange(nfeatures), meas[i], q, r
            )
            obs[i] = kalman_state.predicted_obs_vec
        #
        # The true variance between the real location and the predicted
        #
        k_var = np.array(
            [np.var(obs[:, i, 0] - locs[:, i, 0]) for i in range(nfeatures)]
        )
        #
        # I am not sure if the difference between the estimated process
        # variance and the real process variance is reasonable.
        #
        self.assertTrue(np.all(k_var / kalman_state.noise_var[:, 0] < 4))
        self.assertTrue(np.all(kalman_state.noise_var[:, 0] / k_var < 4))


class TestPermutations(unittest.TestCase):
    def test_01_01_permute_one(self):
        np.random.seed(11)
        a = [np.random.uniform()]
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 1)
        self.assertEqual(len(b[0]), 1)
        self.assertEqual(b[0][0], a[0])

    def test_01_02_permute_two(self):
        np.random.seed(12)
        a = np.random.uniform(size=2)
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 2)
        self.assertEqual(len(b[0]), 2)
        self.assertTrue(np.all(np.array(b) == a[np.array([[0, 1], [1, 0]])]))

    def test_01_03_permute_three(self):
        np.random.seed(13)
        a = np.random.uniform(size=3)
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 6)
        self.assertEqual(len(b[0]), 3)
        expected = np.array(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        )
        self.assertTrue(np.all(np.array(b) == a[expected]))


class TestParity(unittest.TestCase):
    def test_01_01_one(self):
        self.assertEqual(F.parity([1]), 1)

    def test_01_02_lots(self):
        np.random.seed(12)
        for i in range(100):
            size = np.random.randint(3, 20)
            a = np.arange(size)
            n = np.random.randint(1, 20)
            for j in range(n):
                k, l = np.random.permutation(np.arange(size))[:2]
                a[k], a[l] = a[l], a[k]
            self.assertEqual(F.parity(a), 1 - (n % 2) * 2)


class TestDotN(unittest.TestCase):
    def test_00_00_dot_nothing(self):
        result = F.dot_n(np.zeros((0, 4, 4)), np.zeros((0, 4, 4)))
        self.assertEqual(len(result), 0)

    def test_01_01_dot_2x2(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1, 2, 2))
        b = np.random.uniform(size=(1, 2, 2))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[0], b[0])])
        np.testing.assert_array_almost_equal(result, expected)

    def test_01_02_dot_2x3(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1, 3, 2))
        b = np.random.uniform(size=(1, 2, 3))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[0], b[0])])
        np.testing.assert_array_almost_equal(result, expected)

    def test_01_02_dot_nx2x3(self):
        np.random.seed(13)
        a = np.random.uniform(size=(20, 3, 2))
        b = np.random.uniform(size=(20, 2, 3))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[i], b[i]) for i in range(20)])
        np.testing.assert_array_almost_equal(result, expected)


class TestDetN(unittest.TestCase):
    def test_00_00_det_nothing(self):
        result = F.det_n(np.zeros((0, 4, 4)))
        self.assertEqual(len(result), 0)

    def test_01_01_det_1x1x1(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1, 1, 1))
        result = F.det_n(a)
        self.assertEqual(len(result), 1)
        self.assertEqual(a[0, 0, 0], result[0])

    def test_01_02_det_1x2x2(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1, 2, 2))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_03_det_1x3x3(self):
        np.random.seed(13)
        a = np.random.uniform(size=(1, 3, 3))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_04_det_nx3x3(self):
        np.random.seed(14)
        a = np.random.uniform(size=(21, 3, 3))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)


class TestCofactorN(unittest.TestCase):
    def test_01_01_cofactor_1x2x2(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1, 2, 2))
        ii, jj = np.mgrid[: (a.shape[1] - 1), : (a.shape[1] - 1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r != i]
                    jjj = r[r != j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])

    def test_01_02_cofactor_1x3x3(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1, 3, 3))
        ii, jj = np.mgrid[: (a.shape[1] - 1), : (a.shape[1] - 1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r != i]
                    jjj = r[r != j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])

    def test_01_03_cofactor_nx4x4(self):
        np.random.seed(13)
        a = np.random.uniform(size=(21, 4, 4))
        ii, jj = np.mgrid[: (a.shape[1] - 1), : (a.shape[1] - 1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r != i]
                    jjj = r[r != j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])


class TestInvN(unittest.TestCase):
    def test_01_01_inv_1x1x1(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1, 1, 1))
        result = F.inv_n(a)
        self.assertEqual(len(result), 1)
        self.assertEqual(a[0, 0, 0], 1 / result[0])

    def test_01_02_inv_1x2x2(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1, 2, 2))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_03_inv_1x3x3(self):
        np.random.seed(13)
        a = np.random.uniform(size=(1, 3, 3))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_04_inv_nx3x3(self):
        np.random.seed(14)
        a = np.random.uniform(size=(21, 3, 3))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)


class TestConvexHullTransform(unittest.TestCase):
    def test_01_01_zeros(self):
        """The convex hull transform of an array of identical values is itself"""
        self.assertTrue(np.all(F.convex_hull_transform(np.zeros((10, 20))) == 0))

    def test_01_02_point(self):
        """The convex hull transform of 1 foreground pixel is itself"""
        image = np.zeros((10, 20))
        image[5, 10] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))

    def test_01_03_line(self):
        """The convex hull transform of a line of foreground pixels is itself"""
        image = np.zeros((10, 20))
        image[5, 7:14] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))

    def test_01_04_convex(self):
        """The convex hull transform of a convex figure is itself"""

        image = np.zeros((10, 20))
        image[2:7, 7:14] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))

    def test_01_05_concave(self):
        """The convex hull transform of a concave figure is the convex hull"""
        expected = np.zeros((10, 20))
        expected[2:8, 7:14] = 1
        image = expected.copy()
        image[4:6, 7:10] = 0.5
        self.assertTrue(np.all(F.convex_hull_transform(image) == expected))

    def test_02_01_two_levels(self):
        """Test operation on two grayscale levels"""

        expected = np.zeros((20, 30))
        expected[3:18, 3:27] = 0.5
        expected[8:15, 10:20] = 1
        image = expected.copy()
        image[:, 15] = 0
        image[10, :] = 0
        # need an odd # of bins in order to have .5 be a bin
        self.assertTrue(np.all(F.convex_hull_transform(image, 7) == expected))

    def test_03_01_masked(self):
        """Test operation on a masked image"""

        expected = np.zeros((20, 30))
        expected[3:18, 3:27] = 0.5
        expected[8:15, 10:20] = 1
        image = expected.copy()
        image[:, 15] = 0
        image[10, :] = 0
        mask = np.ones((20, 30), bool)
        mask[:, 0] = False
        image[:, 0] = 0.75

        result = F.convex_hull_transform(image, levels=7, mask=mask)
        self.assertTrue(np.all(result == expected))

    def test_03_02_all_masked(self):
        # Regression test of 1286 - exception if totally masked image
        #
        i = np.linspace(-1, 1, 11) ** 2
        i, j = i[:, np.newaxis], i[np.newaxis, :]
        image = np.sqrt(i * i + j * j)
        mask = np.zeros(image.shape, bool)
        F.convex_hull_transform(image, levels=8, mask=mask)

    def test_04_01_many_chunks(self):
        """Test the two-pass at a single level chunk looping"""
        np.random.seed(41)
        #
        # Make an image that monotonically decreases from the center
        #
        i, j = np.mgrid[-50:51, -50:51].astype(float) / 100.0
        image = 1 - np.sqrt(i ** 2 + j ** 2)
        expected = image.copy()
        #
        # Riddle it with holes
        #
        holes = np.random.uniform(size=image.shape) < 0.01
        image[holes] = 0
        result = F.convex_hull_transform(
            image, levels=256, chunksize=1000, pass_cutoff=256
        )
        diff = np.abs(result - expected)
        self.assertTrue(np.sum(diff > 1 / 256.0) <= np.sum(holes))
        expected = F.convex_hull_transform(image, pass_cutoff=256)
        np.testing.assert_equal(result, expected)

    def test_04_02_two_pass(self):
        """Test the two-pass at multiple levels chunk looping"""
        np.random.seed(42)
        #
        # Make an image that monotonically decreases from the center
        #
        i, j = np.mgrid[-50:51, -50:51].astype(float) / 100.0
        image = 1 - np.sqrt(i ** 2 + j ** 2)
        #
        # Riddle it with holes
        #
        holes = np.random.uniform(size=image.shape) < 0.01
        image[holes] = 0
        result = F.convex_hull_transform(
            image, levels=256, chunksize=1000, pass_cutoff=256
        )
        expected = F.convex_hull_transform(image, pass_cutoff=256)
        np.testing.assert_equal(result, expected)


class TestCircularHough(unittest.TestCase):
    def test_01_01_nothing(self):
        img = np.zeros((10, 20))
        result = F.circular_hough(img, 4)
        self.assertTrue(np.all(result == 0))

    def test_01_02_circle(self):
        i, j = np.mgrid[-15:16, -15:16]
        circle = np.abs(np.sqrt(i * i + j * j) - 6) <= 1.5
        expected = convolve(circle.astype(float), circle.astype(float)) / np.sum(circle)
        img = F.circular_hough(circle, 6)
        self.assertEqual(img[15, 15], 1)
        self.assertTrue(np.all(img[np.abs(np.sqrt(i * i + j * j) - 6) < 1.5] < 0.25))

    def test_01_03_masked(self):
        img = np.zeros((31, 62))
        mask = np.ones((31, 62), bool)
        i, j = np.mgrid[-15:16, -15:16]
        circle = np.abs(np.sqrt(i * i + j * j) - 6) <= 1.5
        # Do one circle
        img[:, :31] = circle
        # Do a second, but mask it
        img[:, 31:] = circle
        mask[:, 31:][circle] = False
        result = F.circular_hough(img, 6, mask=mask)
        self.assertEqual(result[15, 15], 1)
        self.assertEqual(result[15, 15 + 31], 0)


class TestLineIntegration(unittest.TestCase):
    def test_01_01_nothing(self):
        img = np.zeros((23, 17))
        result = F.line_integration(img, 0, 0.95, 2.0)
        np.testing.assert_almost_equal(result, 0)

    def test_01_02_two_lines(self):
        img = np.ones((20, 30)) * 0.5
        img[8, 10:20] = 1
        img[12, 10:20] = 0
        result = F.line_integration(img, 0, 1, 0)
        expected = np.zeros((20, 30))
        expected[9:12, 10:20] = 1
        expected[8, 10:20] = 0.5
        expected[12, 10:20] = 0.5
        np.testing.assert_almost_equal(result, expected)

    def test_01_03_diagonal_lines(self):
        img = np.ones((20, 30)) * 0.5
        i, j = np.mgrid[0:20, 0:30]
        img[(i == j - 3) & (i <= 15)] = 1
        img[(i == j + 3)] = 0
        expected = np.zeros((20, 30), bool)
        expected[(i >= j - 3) & (i <= j + 3)] = True
        result = F.line_integration(img, -45, 1, 0)
        self.assertTrue(np.mean(result[expected]) > 0.5)
        self.assertTrue(np.mean(result[~expected]) < 0.25)

    def test_01_04_decay(self):
        img = np.ones((25, 23)) * 0.5
        img[10, 10] = 1
        img[20, 10] = 0
        result = F.line_integration(img, 0, 0.9, 0)
        decay_part = result[11:20, 10]
        expected = 0.9 ** np.arange(1, 10) + 0.9 ** np.arange(9, 0, -1)
        expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
        decay_part = (decay_part - np.min(decay_part)) / (
            np.max(decay_part) - np.min(decay_part)
        )
        np.testing.assert_almost_equal(decay_part, expected)

    def test_01_05_smooth(self):
        img = np.ones((30, 20)) * 0.5
        img[10, 10] = 1
        img[20, 10] = 0
        result = F.line_integration(img, 0, 1, 0.5)
        part = result[15, :]
        part = (part - np.min(part)) / (np.max(part) - np.min(part))
        expected = np.exp(-((np.arange(20) - 10) ** 2) * 2)
        expected = (expected - np.min(expected)) / (np.max(expected) - np.min(expected))
        np.testing.assert_almost_equal(part, expected)


class TestVarianceTransform(unittest.TestCase):
    def test_01_00_zeros(self):
        result = F.variance_transform(np.zeros((30, 20)), 1)
        np.testing.assert_almost_equal(result, 0)

    def test_01_01_transform(self):
        r = np.random.RandomState()
        r.seed(11)
        img = r.uniform(size=(21, 18))
        sigma = 1.5
        result = F.variance_transform(img, sigma)
        #
        # Calculate the variance for one point
        #
        center_i, center_j = 10, 9
        i, j = np.mgrid[
            -center_i : (img.shape[0] - center_i), -center_j : (img.shape[1] - center_j)
        ]
        weight = np.exp(-(i * i + j * j) / (2 * sigma * sigma))
        weight = weight / np.sum(weight)
        mean = np.sum(img * weight)
        norm = img - mean
        var = np.sum(norm * norm * weight)
        self.assertAlmostEqual(var, result[center_i, center_j], 5)

    def test_01_02_transform_masked(self):
        r = np.random.RandomState()
        r.seed(12)
        center_i, center_j = 10, 9
        img = r.uniform(size=(21, 18))
        mask = r.uniform(size=(21, 18)) > 0.25
        mask[center_i, center_j] = True
        sigma = 1.7
        result = F.variance_transform(img, sigma, mask)
        #
        # Calculate the variance for one point
        #
        i, j = np.mgrid[
            -center_i : (img.shape[0] - center_i), -center_j : (img.shape[1] - center_j)
        ]
        weight = np.exp(-(i * i + j * j) / (2 * sigma * sigma))
        weight[~mask] = 0
        weight = weight / np.sum(weight)
        mean = np.sum(img * weight)
        norm = img - mean
        var = np.sum(norm * norm * weight)
        self.assertAlmostEqual(var, result[center_i, center_j], 5)


class TestPoissonEquation(unittest.TestCase):
    def test_00_00_nothing(self):
        image = np.zeros((11, 14), bool)
        p = F.poisson_equation(image)
        np.testing.assert_array_equal(p, 0)

    def test_00_01_single(self):
        image = np.zeros((11, 14), bool)
        image[7, 3] = True
        p = F.poisson_equation(image)
        np.testing.assert_array_equal(p[image], 1)
        np.testing.assert_array_equal(p[~image], 0)

    def test_01_01_simple(self):
        image = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            bool,
        )
        a = 5.0 / 3.0
        b = a + 1
        self.assertAlmostEqual(b / 4 + 1, a)
        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, a, 0, 0],
                [0, a, b, a, 0],
                [0, 0, a, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        p = F.poisson_equation(image, convergence=0.00001)
        np.testing.assert_almost_equal(p, expected, 4)

    def test_01_02_boundary(self):
        # Test an image with pixels at the boundaries.
        image = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        a = 5.0 / 3.0
        b = a + 1
        self.assertAlmostEqual(b / 4 + 1, a)
        expected = np.array([[0, a, 0], [a, b, a], [0, a, 0]])
        p = F.poisson_equation(image, convergence=0.00001)
        np.testing.assert_almost_equal(p, expected, 4)

    def test_01_03_subsampling(self):
        # Test an image that is large enough to undergo some subsampling
        #
        r = np.random.RandomState()
        r.seed(13)
        image = r.uniform(size=(300, 300)) < 0.001
        i, j = np.mgrid[-8:9, -8:9]
        kernel = i * i + j * j <= 64
        image = binary_dilation(image, kernel)
        p = F.poisson_equation(image, convergence=0.001)
        i, j = np.mgrid[0 : p.shape[0], 0 : p.shape[1]]
        mask = image & (i > 0) & (i < p.shape[0] - 1) & (j > 0) & (j < p.shape[1] - 1)
        i, j = i[mask], j[mask]
        expected = (p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1]) / 4 + 1
        np.testing.assert_almost_equal(p[mask], expected, 0)
