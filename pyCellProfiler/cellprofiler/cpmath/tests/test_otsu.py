"""Test Otsu module

"""
import unittest
import numpy
import numpy.random
from cellprofiler.cpmath.otsu import otsu

class testOtsu(unittest.TestCase):
    def test_01_TwoValues(self):
        x = otsu([.2,.8])
        self.assertTrue(x >= .2)
        self.assertTrue(x <= .8)
    
    def test_02_TwoDistributions(self):
        numpy.random.seed(0)
        x0 = numpy.random.uniform(.1,.4,size=1000)
        x1 = numpy.random.uniform(.6,1.0,size=1000)
        x = numpy.append(x0,x1)
        numpy.random.shuffle(x)
        threshold = otsu(x)
        self.assertTrue(threshold >= .4)
        self.assertTrue(threshold <= .6)