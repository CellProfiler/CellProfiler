"""Test Otsu module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import unittest
import numpy as np
from cellprofiler.cpmath.otsu import otsu, entropy, otsu3, entropy3

class TestOtsu(unittest.TestCase):
    def test_01_TwoValues(self):
        """Test Otsu of two values is between the two"""
        x = otsu([.2,.8])
        self.assertTrue(x >= .2)
        self.assertTrue(x <= .8)
    
    def test_02_TwoDistributions(self):
        """Test Otsu of two distributions with no points in between is between the two distributions"""
        np.random.seed(0)
        x0 = np.random.uniform(.1,.4,size=1000)
        x1 = np.random.uniform(.6,1.0,size=1000)
        x = np.append(x0,x1)
        np.random.shuffle(x)
        threshold = otsu(x)
        self.assertTrue(threshold >= .4)
        self.assertTrue(threshold <= .6)
    
    def test_03_min_threshold(self):
        """Test Otsu with a min_threshold"""
        np.random.seed(0)
        #
        # There should be three peaks with the otsu
        # between the first and second peaks.
        # With a fixed min threshold, the otsu
        # should be between the second two peaks.
        x0 = np.random.binomial(40,.1,10000).astype(float)/40.0
        x1 = np.random.binomial(40,.5,2000).astype(float)/40.0
        x2 = np.random.binomial(40,.9,2000).astype(float)/40.0
        x = np.concatenate((x0,x1,x2))
        self.assertTrue(otsu(x) >=.1)
        self.assertTrue(otsu(x) <=.5)
        self.assertTrue(otsu(x,min_threshold=.5) >= .5)
        self.assertTrue(otsu(x,min_threshold=.5) < .9)
        
    def test_04_max_threshold(self):
        """Test Otsu with a max_threshold"""
        np.random.seed(0)
        #
        # There should be three peaks with the otsu
        # between the second and third
        # With a fixed max threshold, the otsu
        # should be between the first two peaks.
        x0 = np.random.binomial(40,.1,2000).astype(float)/40.0
        x1 = np.random.binomial(40,.5,2000).astype(float)/40.0
        x2 = np.random.binomial(40,.9,10000).astype(float)/40.0
        x = np.concatenate((x0,x1,x2))
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
        np.random.seed(0)
        x = np.zeros((10,))
        self.assertTrue(otsu(x,min_threshold=.1)>=.1)
        
    def test_06_NaN(self):
        """Regression test of Otsu with NaN in input (issue #624)"""
        r = np.random.RandomState()
        r.seed(6)
        data = r.uniform(size = 100)
        data[r.uniform(size = 100) > .8] = np.NaN
        self.assertEqual(otsu(data), otsu(data[~ np.isnan(data)]))
        self.assertEqual(entropy(data), entropy(data[~ np.isnan(data)]))
        self.assertEqual(otsu3(data), otsu3(data[~ np.isnan(data)]))
        self.assertEqual(entropy3(data), entropy3(data[~ np.isnan(data)]))
        
    def test_07_entropy(self):
        '''Test entropy with two normal distributions'''
        r = np.random.RandomState()
        r.seed(7)
        x1 = r.normal(.2, .1, 10000)
        x2 = r.normal(.5, .25, 5000)
        data = np.hstack((x1, x2))
        data = data[(data > 0) & (data < 1)]
        threshold = entropy(data)
        self.assertTrue(threshold > .2)
        self.assertTrue(threshold < .5)
        
    def test_08_otsu3(self):
        '''Test 3-class Otsu with three normal distributions'''

        r = np.random.RandomState()
        r.seed(8)
        x1 = r.normal(.2, .1, 10000)
        x2 = r.normal(.5, .25, 5000)
        x3 = r.normal(.8, .1, 5000)
        data = np.hstack((x1, x2, x3))
        data = data[(data > 0) & (data < 1)]
        threshold1, threshold2 = otsu3(data)
        self.assertTrue(threshold1 > .2)
        self.assertTrue(threshold1 < .5)
        self.assertTrue(threshold2 > .5)
        self.assertTrue(threshold2 < .8)

    def test_09_entropy3(self):
        '''Test 3-class entropy with three normal distributions'''

        r = np.random.RandomState()
        r.seed(8)
        x1 = r.normal(.2, .1, 10000)
        x2 = r.normal(.5, .25, 5000)
        x3 = r.normal(.8, .1, 5000)
        data = np.hstack((x1, x2, x3))
        data = data[(data > 0) & (data < 1)]
        threshold1, threshold2 = entropy3(data)
        self.assertTrue(threshold1 > .2)
        self.assertTrue(threshold1 < .5)
        self.assertTrue(threshold2 > .5)
        self.assertTrue(threshold2 < .8)
