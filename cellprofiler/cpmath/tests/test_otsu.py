"""Test Otsu module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import unittest
import numpy
import numpy.random
from cellprofiler.cpmath.otsu import otsu

class testOtsu(unittest.TestCase):
    def test_01_TwoValues(self):
        """Test Otsu of two values is between the two"""
        x = otsu([.2,.8])
        self.assertTrue(x >= .2)
        self.assertTrue(x <= .8)
    
    def test_02_TwoDistributions(self):
        """Test Otsu of two distributions with no points in between is between the two distributions"""
        numpy.random.seed(0)
        x0 = numpy.random.uniform(.1,.4,size=1000)
        x1 = numpy.random.uniform(.6,1.0,size=1000)
        x = numpy.append(x0,x1)
        numpy.random.shuffle(x)
        threshold = otsu(x)
        self.assertTrue(threshold >= .4)
        self.assertTrue(threshold <= .6)
    
    def test_03_min_threshold(self):
        """Test Otsu with a min_threshold"""
        numpy.random.seed(0)
        #
        # There should be three peaks with the otsu
        # between the first and second peaks.
        # With a fixed min threshold, the otsu
        # should be between the second two peaks.
        x0 = numpy.random.binomial(40,.1,10000).astype(float)/40.0
        x1 = numpy.random.binomial(40,.5,2000).astype(float)/40.0
        x2 = numpy.random.binomial(40,.9,2000).astype(float)/40.0
        x = numpy.concatenate((x0,x1,x2))
        self.assertTrue(otsu(x) >=.1)
        self.assertTrue(otsu(x) <=.5)
        self.assertTrue(otsu(x,min_threshold=.5) >= .5)
        self.assertTrue(otsu(x,min_threshold=.5) < .9)
        
    def test_04_max_threshold(self):
        """Test Otsu with a max_threshold"""
        numpy.random.seed(0)
        #
        # There should be three peaks with the otsu
        # between the second and third
        # With a fixed max threshold, the otsu
        # should be between the first two peaks.
        x0 = numpy.random.binomial(40,.1,2000).astype(float)/40.0
        x1 = numpy.random.binomial(40,.5,2000).astype(float)/40.0
        x2 = numpy.random.binomial(40,.9,10000).astype(float)/40.0
        x = numpy.concatenate((x0,x1,x2))
        self.assertTrue(otsu(x) > .5)
        self.assertTrue(otsu(x) < .9)
        self.assertTrue(otsu(x,max_threshold=.5) >=.1)
        self.assertTrue(otsu(x,max_threshold=.5) <=.5)
        
    def test_05_threshold_of_flat(self):
        """Test Otsu with a threshold and all input values the same
        
        This is a regression test of an apparent bug where the Otsu
        of an all-zero image has a threshold of zero even though
        the min_threshold was .1
        """
        numpy.random.seed(0)
        x = numpy.zeros((10,))
        self.assertTrue(otsu(x,min_threshold=.1)>=.1)

