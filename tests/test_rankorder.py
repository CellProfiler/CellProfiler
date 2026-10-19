from __future__ import absolute_import
import unittest
import numpy as np
from centrosome.rankorder import rank_order


class TestRankOrder(unittest.TestCase):
    def test_00_zeros(self):
        """Test rank_order on a matrix of all zeros"""
        x = np.zeros((5, 5))
        output = rank_order(x)[0]
        self.assertTrue(np.all(output == 0))
        self.assertTrue(output.dtype.type == np.uint32)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(x.shape[1], 5)

    def test_01_3D(self):
        x = np.zeros((5, 5, 5))
        output = rank_order(x)[0]
        self.assertTrue(np.all(output == 0))
        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(x.shape[1], 5)
        self.assertEqual(x.shape[2], 5)

    def test_02_two_values(self):
        x = np.zeros((5, 10))
        x[3, 5] = 2
        x[4, 7] = 2
        output, orig = rank_order(x)
        self.assertEqual(output[3, 5], 1)
        self.assertEqual(output[4, 7], 1)
        self.assertEqual(len(orig), 2)
        self.assertEqual(orig[0], 0)
        self.assertEqual(orig[1], 2)
        self.assertEqual(np.sum(output == 0), 48)

    def test_03_three_values(self):
        x = np.zeros((5, 10))
        x[3, 5] = 4
        x[4, 7] = 4
        x[0, 9] = 3
        output, orig = rank_order(x)
        self.assertEqual(output[0, 9], 1)
        self.assertEqual(output[3, 5], 2)
        self.assertEqual(output[4, 7], 2)
        self.assertEqual(len(orig), 3)
        self.assertEqual(orig[0], 0)
        self.assertEqual(orig[1], 3)
        self.assertEqual(orig[2], 4)
        self.assertEqual(np.sum(output == 0), 47)

    def test_04_decimate(self):
        # The bins originally go from 0 to 9 and we should
        # merge bins 0 and 1 because they are the smallest.
        x = np.sqrt(np.arange(100)).astype(int)
        output, orig = rank_order(x, nbins=9)
        self.assertEqual(len(orig), 9)
        self.assertTrue(np.all(output[:4] == 0))
        self.assertIn(orig[0], (0, 1))
        np.testing.assert_array_equal(orig[1:], np.arange(2, 10))
