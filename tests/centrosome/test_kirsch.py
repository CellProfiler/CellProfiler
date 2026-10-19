from __future__ import absolute_import
import centrosome.kirsch
import scipy.misc
import numpy as np
import numpy.testing
import unittest


class TestKirsch(unittest.TestCase):
    def test_01_01_kirsch(self):
        #
        # Test a maximum at all possible orientations
        #
        r = np.random.RandomState([ord(_) for _ in "kirsch"])
        for coords in (
            ((0, -1), (-1, -1), (-1, 0)),
            ((-1, -1), (-1, 0), (-1, 1)),
            ((-1, 0), (-1, 1), (0, 1)),
            ((-1, 1), (0, 1), (1, 1)),
            ((0, 1), (1, 1), (1, 0)),
            ((1, 1), (1, 0), (1, -1)),
            ((1, 0), (1, -1), (0, -1)),
            ((1, -1), (0, -1), (-1, -1)),
        ):
            img = r.uniform(size=(3, 3)) * 0.1
            expected = -3 * img
            for ioff, joff in coords:
                img[ioff + 1, joff + 1] += 0.5
                expected[ioff + 1, joff + 1] = img[ioff + 1, joff + 1] * 5
            expected[1, 1] = 0
            result = centrosome.kirsch.kirsch(img)
            numpy.testing.assert_approx_equal(result[1, 1], np.sum(expected))
