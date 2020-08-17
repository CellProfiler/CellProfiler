#!/usr/bin/env python
""" test_haralick -- tests for centrosome.haralick
"""
import unittest

import centrosome.haralick as haralick
import numpy as np

gray4 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 3, 3]])
gray = gray4 / (1.0 * gray4.max())
labels = np.ones((4, 4), dtype="int32")


# labels = np.array[[1,1,1,1],[1,1,1,2],[1,1,2,2],[1,2,2,2]], dtype=int32)


class TestHaralick(unittest.TestCase):
    def test_quantize(self):
        q = haralick.quantize(gray, 7)
        correct = np.array(
            [[0, 0, 2, 2], [0, 0, 2, 2], [0, 4, 4, 4], [4, 4, 6, 6]], dtype="i4"
        )
        self.assertTrue((q == correct).all())

    def test_quantize_fixpoint(self):
        q = haralick.quantize(gray, 4)
        self.assertTrue((q == gray4).all())

    def test_normalize_per_object(self):
        norm = haralick.normalized_per_object(gray4, labels)
        self.assertTrue((norm == gray).all())

    def test_cooccurrence(self):
        P, levels = haralick.cooccurrence(gray4, labels, 0, 1)
        correct = np.array(
            [[[2, 2, 1, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, 0, 1]]], float
        )
        correct = correct / np.sum(correct)
        self.assertEqual(levels, 4)
        self.assertTrue((P == correct).all())

    def test_H1(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H1()[0], 0.1667, 4)

    def test_H2(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H2()[0], 0.5833, 4)

    def test_H3(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.mux[0], 2.0833, 4)
        self.assertAlmostEqual(h.muy[0], 2.5000, 4)
        self.assertAlmostEqual(h.sigmax[0], 1.0375, 4)
        self.assertAlmostEqual(h.sigmay[0], 0.9574, 4)
        self.assertAlmostEqual(h.H3()[0], 0.7970, 4)

    def test_H4(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H4()[0], 1.0764, 4)

    def test_H5(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H5()[0], 0.8083, 4)

    def test_H6(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H6()[0], 4.5833, 4)

    def test_H7(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H7()[0], 3.5764, 4)

    def test_H8(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H8()[0], 1.7046, 4)

    def test_H9(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H9()[0], 1.8637, 4)

    def test_H10(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H10()[0], 0.4097, 4)

    def test_H11(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H11()[0], 0.8240, 4)

    def test_H12(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H12()[0], -0.5285, 4)

    def test_H13(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        self.assertAlmostEqual(h.H13()[0], 0.8687, 4)

    def test_all_01(self):
        h = haralick.Haralick(gray, labels, 0, 1, nlevels=4)
        fv = h.all()
        self.assertTrue((np.array(list(map(len, fv))) == 1).all())

    def test_01_01_regression_edge(self):
        """Test coocurrence with an object smaller than the scal near the edge.

        There's a corner case here where the co-occurrence is being done
        on pixels "scale" to the right of objects and all objects are
        within "scale" of the right edge. That means that all objects get
        compared to nothing and the histogram code blows up.
        """
        labels = np.zeros((100, 100), int)
        labels[90:99, 90:99] = 1
        np.random.seed(0)
        image = (np.random.uniform(size=(100, 100)) * 8).astype(int)
        c, nlevels = haralick.cooccurrence(image, labels, 0, 30)
        self.assertTrue(np.all(c == 0))

    def test_01_02_mask(self):
        """Test with a masked image"""
        labels = np.ones((10, 20), int)
        np.random.seed(12)
        image = np.random.uniform(size=(10, 20)).astype(np.float32)
        image[:, :10] *= 0.5
        mask = np.ones((10, 20), bool)
        mask[:, 10:] = False
        # Masked haralick
        hm = haralick.Haralick(image, labels, 0, 1, mask=mask)
        # Expected haralick
        he = haralick.Haralick(image[:, :10], labels[:, :10], 0, 1)
        for measured, expected in zip(hm.all(), he.all()):
            self.assertEqual(measured, expected)

    def test_02_01_angles(self):
        """Test that measurements are stable on i,j swap"""

        labels = np.ones((10, 20), int)
        np.random.seed(12)
        image = np.random.uniform(size=(10, 20)).astype(np.float32)
        hi = haralick.Haralick(image, labels, 1, 0)
        hj = haralick.Haralick(image.transpose(), labels.transpose(), 0, 1)
        for him, hjm in zip(hi.all(), hj.all()):
            self.assertEqual(him, hjm)


if __name__ == "__main__":
    unittest.main()
