"""test_smooth - test the smoothing module

"""
__version__="$Revision$"

import unittest
import numpy
import cellprofiler.cpmath.smooth as cpms

class TestSmoothWithNoise(unittest.TestCase):
    def test_01_smooth_zero(self):
        """Make sure smooth doesn't crash on all zeros"""
        out = cpms.smooth_with_noise(numpy.zeros((3,3)), 5)
        self.assertTrue(numpy.all(out >=0))
        self.assertTrue(numpy.all(out <=1))
    
    def test_02_smooth_half(self):
        out = cpms.smooth_with_noise(numpy.ones((100,100))*.5, 5)
        # Roughly 1/2 (5000) should be above .5 and half below
        # Over 1/2 should be within 1/32 of .5
        self.assertTrue(out.ndim == 2)
        self.assertTrue(out.shape[0] == 100)
        self.assertTrue(out.shape[1] == 100)
        self.assertTrue(abs(numpy.sum(out > .5)-5000) < 300) # unless we are 3sd unlucky
        self.assertTrue(numpy.sum(numpy.abs(out-.5)< 1.0 / 32.0) > 4700) # unless we are > 3sd unlucky 