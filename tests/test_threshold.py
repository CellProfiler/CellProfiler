from __future__ import absolute_import
import hashlib
import numpy as np
from scipy.ndimage import convolve1d
import scipy.stats
import unittest

import centrosome.threshold as T


class TestThreshold(unittest.TestCase):
    def test_01_00_nothing(self):
        result = T.get_otsu_threshold(-np.ones((10, 10)))

    def test_01_01_negative_log_otsu(self):
        """regression test of img-1466"""

        r = np.random.RandomState()
        r.seed(11)
        img = r.uniform(size=(10, 10))
        img[0, 0] = -1
        unmasked = T.get_otsu_threshold(img)
        masked = T.get_otsu_threshold(img, img >= 0)
        self.assertEqual(unmasked, masked)

    def test_02_00_mct_zeros(self):
        result = T.get_maximum_correlation_threshold(np.zeros(0))
        r = np.random.RandomState()
        r.seed(11)
        result = T.get_maximum_correlation_threshold(
            r.uniform(size=(10, 10)), mask=np.zeros((10, 10), bool)
        )
        result = T.get_maximum_correlation_threshold(np.ones((10, 10)) * 0.5)
        self.assertEqual(result, 0.5)

    def test_02_01_mct_matches_reference_implementation(self):
        image = np.array(
            [
                0,
                255,
                231,
                161,
                58,
                218,
                95,
                17,
                136,
                56,
                179,
                196,
                1,
                70,
                173,
                113,
                192,
                101,
                223,
                65,
                127,
                27,
                234,
                224,
                205,
                61,
                74,
                168,
                63,
                209,
                120,
                41,
                218,
                22,
                66,
                135,
                244,
                178,
                193,
                238,
                140,
                215,
                96,
                194,
                158,
                20,
                169,
                61,
                55,
                1,
                130,
                17,
                240,
                237,
                15,
                228,
                136,
                207,
                65,
                90,
                191,
                253,
                63,
                101,
                206,
                91,
                154,
                76,
                43,
                89,
                213,
                26,
                17,
                107,
                251,
                164,
                206,
                191,
                73,
                32,
                51,
                191,
                80,
                48,
                61,
                57,
                4,
                152,
                74,
                174,
                103,
                91,
                106,
                217,
                194,
                161,
                248,
                59,
                198,
                24,
                22,
                36,
            ],
            float,
        )
        self.assertEqual(127, T.get_maximum_correlation_threshold(image))

    def test_03_01_adaptive_threshold_same(self):
        r = np.random.RandomState()
        r.seed(31)
        block = r.uniform(size=(10, 10))
        i, j = np.mgrid[0:10:2, 0:10:2]
        block[i, j] *= 0.5
        i, j = np.mgrid[0:50, 0:50]
        img = block[i % 10, j % 10]
        global_threshold = T.get_global_threshold(T.TM_OTSU, block)
        adaptive_threshold = T.get_adaptive_threshold(
            T.TM_OTSU, img, global_threshold, adaptive_window_size=10
        )
        np.testing.assert_almost_equal(adaptive_threshold, global_threshold)

    def test_03_02_adaptive_threshold_different(self):
        r = np.random.RandomState()
        r.seed(31)
        block = r.uniform(size=(10, 10))
        i, j = np.mgrid[0:10:2, 0:10:2]
        block[i, j] *= 0.5
        i, j = np.mgrid[0:50, 0:50]
        img = block[i % 10, j % 10] * 0.5
        #
        # Make the middle higher in intensity
        #
        img[20:30, 20:30] *= 2
        global_threshold = T.get_global_threshold(T.TM_OTSU, block)
        adaptive_threshold = T.get_adaptive_threshold(
            T.TM_OTSU, img, global_threshold, adaptive_window_size=10
        )
        #
        # Check that the gradients are positive for i,j<15 and negative
        # for i,j>=15
        #
        gradient = convolve1d(adaptive_threshold, [-1, 0, 1], 0)
        self.assertTrue(np.all(gradient[20:25, 20:30] < 0))
        self.assertTrue(np.all(gradient[25:30, 20:30] > 0))
        gradient = convolve1d(adaptive_threshold, [-1, 0, 1], 1)
        self.assertTrue(np.all(gradient[20:30, 20:25] < 0))
        self.assertTrue(np.all(gradient[20:30, 25:30] > 0))

    def get_random_state(self, *args):
        h = hashlib.sha1()
        h.update(np.array(args))
        seed = np.frombuffer(h.digest(), np.uint32)[0]
        r = np.random.RandomState()
        r.seed(seed)
        return r

    def make_mog_image(self, loc1, sigma1, loc2, sigma2, frac1, size):
        """Make an image that is a mixture of gaussians
        
        loc{1,2} - mean of distribution # 1 and 2
        sigma{1,2} - standard deviation of distribution # 1 and 2
        frac1 - the fraction of pixels that are in distribution 1
        size - the shape of the image.
        """
        r = self.get_random_state(loc1, sigma1, loc2, sigma2, frac1, *tuple(size))
        n_pixels = np.prod(size)
        p = r.permutation(n_pixels).reshape(size)
        s1 = int(n_pixels * frac1)
        s2 = n_pixels - s1
        d = np.hstack(
            [
                r.normal(loc=loc1, scale=sigma1, size=s1),
                r.normal(loc=loc2, scale=sigma2, size=s2),
            ]
        )
        d[d < 0] = 0
        d[d > 1] = 1
        return d[p]

    def test_04_01_robust_background(self):
        img = self.make_mog_image(0.1, 0.05, 0.5, 0.2, 0.975, (45, 35))
        t = T.get_global_threshold(T.TM_ROBUST_BACKGROUND, img)
        self.assertLess(abs(t - 0.2), 0.025)

    def test_04_02_robust_background_lower_outliers(self):
        img = self.make_mog_image(0.1, 0.05, 0.5, 0.2, 0.5, (45, 35))
        t0 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND, img, lower_outlier_fraction=0
        )
        t05 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND, img, lower_outlier_fraction=0.05
        )
        self.assertNotEqual(t0, t05)

    def test_04_03_robust_background_upper_outliers(self):
        img = self.make_mog_image(0.1, 0.05, 0.5, 0.2, 0.9, (45, 35))
        t0 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND, img, upper_outlier_fraction=0
        )
        t05 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND, img, upper_outlier_fraction=0.05
        )
        self.assertNotEqual(t0, t05)

    def test_04_04_robust_background_sd(self):
        img = self.make_mog_image(0.5, 0.1, 0.8, 0.01, 0.99, (45, 35))
        t2 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND,
            img,
            lower_outlier_fraction=0,
            upper_outlier_fraction=0,
        )
        self.assertLess(abs(t2 - 0.7), 0.02)
        t3 = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND,
            img,
            lower_outlier_fraction=0,
            upper_outlier_fraction=0,
            deviations_above_average=2.5,
        )
        self.assertLess(abs(t3 - 0.75), 0.02)

    def test_04_05_robust_background_median(self):
        img = self.make_mog_image(0.3, 0.05, 0.5, 0.2, 0.9, (45, 35))
        t = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND,
            img,
            average_fn=np.median,
            deviations_above_average=0,
            lower_outlier_fraction=0,
            upper_outlier_fraction=0,
        )
        self.assertLess(abs(t - 0.3), 0.01)

    def test_04_06_robust_background_mode(self):
        img = self.make_mog_image(0.3, 0.05, 0.5, 0.2, 0.9, (45, 35))
        img[(img > 0.25) & (img < 0.35)] = 0.304
        t = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND,
            img,
            average_fn=T.binned_mode,
            deviations_above_average=0,
            lower_outlier_fraction=0,
            upper_outlier_fraction=0,
        )
        self.assertAlmostEqual(t, 0.304)

    def test_04_08_mad(self):
        img = self.make_mog_image(0.3, 0.05, 0.5, 0.2, 0.95, (45, 35))
        t = T.get_global_threshold(
            T.TM_ROBUST_BACKGROUND,
            img,
            variance_fn=T.mad,
            deviations_above_average=2,
            lower_outlier_fraction=0,
            upper_outlier_fraction=0,
        )
        norm = scipy.stats.norm(0, 0.05)
        # the MAD should be the expected value at the 75th percentile
        expected = 0.3 + 2 * norm.ppf(0.75)
        self.assertLess(np.abs(t - expected), 0.02)


if __name__ == "__main__":
    unittest.main()
