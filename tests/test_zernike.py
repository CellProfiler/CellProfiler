from __future__ import absolute_import
from __future__ import division
import numpy as np
import scipy.ndimage as scind
import unittest
import centrosome.zernike as z
from centrosome.cpmorphology import fill_labeled_holes, draw_line
from six.moves import range


class TestZernike(unittest.TestCase):
    def make_zernike_indexes(self):
        """Make an Nx2 array of all the zernike indexes for n<10"""
        zernike_n_m = []
        for n in range(10):
            for m in range(n + 1):
                if (m + n) & 1 == 0:
                    zernike_n_m.append((n, m))
        return np.array(zernike_n_m)

    def score_rotations(self, labels, n):
        """Score the result of n rotations of the label matrix then test for equality"""
        self.assertEqual(
            labels.shape[0], labels.shape[1], "Must use square matrix for test"
        )
        self.assertEqual(labels.shape[0] & 1, 1, "Must be odd width/height")

        zi = self.make_zernike_indexes()
        test_labels = np.zeros((labels.shape[0] * n, labels.shape[0]))
        test_x = np.zeros((labels.shape[0] * n, labels.shape[0]))
        test_y = np.zeros((labels.shape[0] * n, labels.shape[0]))
        diameter = labels.shape[0]
        radius = labels.shape[0] // 2
        y, x = (
            np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(float) / radius
        )
        anti_mask = x ** 2 + y ** 2 > 1
        x[anti_mask] = 0
        y[anti_mask] = 0
        min_pixels = 100000
        max_pixels = 0
        for i in range(0, n):
            angle = 360 * i // n  # believe it or not, in degrees!
            off_x = labels.shape[0] * i
            off_y = 0
            rotated_labels = scind.rotate(labels, angle, order=0, reshape=False)
            pixels = np.sum(rotated_labels)
            min_pixels = min(min_pixels, pixels)
            max_pixels = max(max_pixels, pixels)
            x_mask = x.copy()
            y_mask = y.copy()
            x_mask[rotated_labels == 0] = 0
            y_mask[rotated_labels == 0] = 0
            test_labels[
                off_x : off_x + diameter, off_y : off_y + diameter
            ] = rotated_labels * (i + 1)
            test_x[off_x : off_x + diameter, off_y : off_y + diameter] = x_mask
            test_y[off_x : off_x + diameter, off_y : off_y + diameter] = y_mask
        zf = z.construct_zernike_polynomials(test_x, test_y, zi)
        scores = z.score_zernike(zf, np.ones((n,)) * radius, test_labels)
        score_0 = scores[0]
        epsilon = 2.0 * (max(1, max_pixels - min_pixels)) / max_pixels
        for score in scores[1:, :]:
            self.assertTrue(np.all(np.abs(score - score_0) < epsilon))

    def score_scales(self, labels, n):
        """Score the result of n 3x scalings of the label matrix then test for equality"""
        self.assertEqual(
            labels.shape[0], labels.shape[1], "Must use square matrix for test"
        )
        self.assertEqual(labels.shape[0] & 1, 1, "Must be odd width/height")

        width = labels.shape[0] * 3 ** n
        height = width * (n + 1)
        zi = self.make_zernike_indexes()
        test_labels = np.zeros((height, width))
        test_x = np.zeros((height, width))
        test_y = np.zeros((height, width))
        radii = []
        for i in range(n + 1):
            scaled_labels = scind.zoom(labels, 3 ** i, order=0)
            diameter = scaled_labels.shape[0]
            radius = scaled_labels.shape[0] // 2
            radii.append(radius)
            y, x = (
                np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(float)
                / radius
            )
            anti_mask = x ** 2 + y ** 2 > 1
            x[anti_mask] = 0
            y[anti_mask] = 0
            off_x = width * i
            off_y = 0
            x[scaled_labels == 0] = 0
            y[scaled_labels == 0] = 0
            test_labels[
                off_x : off_x + diameter, off_y : off_y + diameter
            ] = scaled_labels * (i + 1)
            test_x[off_x : off_x + diameter, off_y : off_y + diameter] = x
            test_y[off_x : off_x + diameter, off_y : off_y + diameter] = y
        zf = z.construct_zernike_polynomials(test_x, test_y, zi)
        scores = z.score_zernike(zf, np.array(radii), test_labels)
        score_0 = scores[0]
        epsilon = 0.02
        for score in scores[1:, :]:
            self.assertTrue(np.all(np.abs(score - score_0) < epsilon))

    def test_00_00_zeros(self):
        """Test construct_zernike_polynomials on an empty image"""
        zi = self.make_zernike_indexes()
        zf = z.construct_zernike_polynomials(
            np.zeros((100, 100)), np.zeros((100, 100)), zi
        )
        # All zernikes with m!=0 should be zero
        m_ne_0 = np.array([i for i in range(zi.shape[0]) if zi[i, 1]])
        m_eq_0 = np.array([i for i in range(zi.shape[0]) if zi[i, 1] == 0])
        self.assertTrue(np.all(zf[:, :, m_ne_0] == 0))
        self.assertTrue(np.all(zf[:, :, m_eq_0] != 0))
        scores = z.score_zernike(zf, np.array([]), np.zeros((100, 100), int))
        self.assertEqual(np.product(scores.shape), 0)

    def test_01_01_one_object(self):
        """Test Zernike on one single circle"""
        zi = self.make_zernike_indexes()
        y, x = np.mgrid[-50:51, -50:51].astype(float) / 50
        labels = x ** 2 + y ** 2 <= 1
        x[labels == 0] = 0
        y[labels == 0] = 0
        zf = z.construct_zernike_polynomials(x, y, zi)
        scores = z.score_zernike(zf, [50], labels)
        # Zernike 0,0 should be 1 and others should be zero within
        # an approximation of 1/radius
        epsilon = 1.0 / 50.0
        self.assertTrue(abs(scores[0, 0] - 1) < epsilon)
        self.assertTrue(np.all(scores[0, 1:] < epsilon))

    def test_02_01_half_circle_rotate(self):
        y, x = np.mgrid[-10:11, -10:11].astype(float) / 10
        labels = x ** 2 + y ** 2 <= 1
        labels[y > 0] = False
        labels = labels.astype(int)
        self.score_rotations(labels, 12)

    def test_02_02_triangle_rotate(self):
        labels = np.zeros((31, 31), int)
        draw_line(labels, (15, 0), (5, 25))
        draw_line(labels, (5, 25), (25, 25))
        draw_line(labels, (25, 25), (15, 0))
        labels = fill_labeled_holes(labels)
        labels = labels > 0
        self.score_rotations(labels, 12)

    def test_02_03_random_objects_rotate(self):
        np.random.seed(0)
        y, x = np.mgrid[-50:50, -50:50].astype(float) / 50
        min = int(50 / np.sqrt(2)) + 1
        max = 100 - min
        for points in range(4, 12):
            labels = np.zeros((101, 101), int)
            coords = np.random.uniform(low=min, high=max, size=(points, 2)).astype(int)
            angles = np.array([np.arctan2(y[yi, xi], x[yi, xi]) for xi, yi in coords])
            order = np.argsort(angles)
            for i in range(points - 1):
                draw_line(labels, coords[i], coords[i + 1])
            draw_line(labels, coords[i], coords[0])
            fill_labeled_holes(labels)
            self.score_rotations(labels, 12)

    def test_03_01_half_circle_scale(self):
        y, x = np.mgrid[-10:11, -10:11].astype(float) / 10
        labels = x ** 2 + y ** 2 <= 1
        labels[y >= 0] = False
        self.score_scales(labels, 2)

    def test_03_02_triangle_scale(self):
        labels = np.zeros((31, 31), int)
        draw_line(labels, (15, 0), (5, 25))
        draw_line(labels, (5, 25), (25, 25))
        draw_line(labels, (25, 25), (15, 0))
        labels = fill_labeled_holes(labels)
        labels = labels > 0
        self.score_scales(labels, 2)

    def test_03_03_random_objects_scale(self):
        np.random.seed(0)
        y, x = np.mgrid[-20:20, -20:20].astype(float) / 20
        min = int(20 / np.sqrt(2)) + 1
        max = 40 - min
        for points in range(4, 12):
            labels = np.zeros((41, 41), int)
            coords = np.random.uniform(low=min, high=max, size=(points, 2)).astype(int)
            angles = np.array([np.arctan2(y[yi, xi], x[yi, xi]) for xi, yi in coords])
            order = np.argsort(angles)
            for i in range(points - 1):
                draw_line(labels, coords[i], coords[i + 1])
            draw_line(labels, coords[i], coords[0])
            fill_labeled_holes(labels)
            self.score_scales(labels, 2)


class TestGetZerikeNumbers(unittest.TestCase):
    def test_01_01_test_3(self):
        expected = np.array(((0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3)), int)
        result = np.array(z.get_zernike_indexes(4))
        order = np.lexsort((result[:, 1], result[:, 0]))
        result = result[order]
        self.assertTrue(np.all(expected == result))
