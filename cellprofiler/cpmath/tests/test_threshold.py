'''test_threshold - test the threshold module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
from scipy.ndimage import convolve1d
import unittest

import cellprofiler.cpmath.threshold as T

class TestThreshold(unittest.TestCase):
    def test_01_00_nothing(self):
        result = T.get_otsu_threshold(-np.ones((10,10)))
        
    def test_01_01_negative_log_otsu(self):
        '''regression test of img-1466'''
        
        r = np.random.RandomState()
        r.seed(11)
        img = r.uniform(size=(10,10))
        img[0,0] = -1
        unmasked = T.get_otsu_threshold(img)
        masked = T.get_otsu_threshold(img, img >= 0)
        self.assertEqual(unmasked, masked)
        
    def test_02_00_mct_zeros(self):
        result = T.get_maximum_correlation_threshold(np.zeros(0))
        r = np.random.RandomState()
        r.seed(11)
        result = T.get_maximum_correlation_threshold(r.uniform(size=(10,10)),
                                                     mask=np.zeros((10,10), bool))
        result = T.get_maximum_correlation_threshold(np.ones((10,10)) * .5)
        self.assertEqual(result, .5)
        
    def test_02_01_mct_matches_reference_implementation(self):
        image = np.array([0,255,231,161,58,218,95,17,136,56,179,196,1,70,173,113,192,101,223,65,127,27,234,224,205,61,74,168,63,209,120,41,218,22,66,135,244,178,193,238,140,215,96,194,158,20,169,61,55,1,130,17,240,237,15,228,136,207,65,90,191,253,63,101,206,91,154,76,43,89,213,26,17,107,251,164,206,191,73,32,51,191,80,48,61,57,4,152,74,174,103,91,106,217,194,161,248,59,198,24,22,36], float)
        self.assertEqual(127, T.get_maximum_correlation_threshold(image))
        
    def test_03_01_adaptive_threshold_same(self):
        r = np.random.RandomState()
        r.seed(31)
        block = r.uniform(size=(10,10))
        i,j = np.mgrid[0:10:2,0:10:2]
        block[i,j] *= .5
        i,j = np.mgrid[0:50,0:50]
        img = block[i%10, j%10]
        global_threshold = T.get_global_threshold(T.TM_OTSU, block)
        adaptive_threshold = T.get_adaptive_threshold(
            T.TM_OTSU, img, global_threshold,
            adaptive_window_size = 10)
        np.testing.assert_almost_equal(adaptive_threshold, global_threshold)
        
    def test_03_02_adaptive_threshold_different(self):
        r = np.random.RandomState()
        r.seed(31)
        block = r.uniform(size=(10,10))
        i,j = np.mgrid[0:10:2,0:10:2]
        block[i,j] *= .5
        i,j = np.mgrid[0:50,0:50]
        img = block[i%10, j%10] * .5
        #
        # Make the middle higher in intensity
        #
        img[20:30, 20:30] *= 2
        global_threshold = T.get_global_threshold(T.TM_OTSU, block)
        adaptive_threshold = T.get_adaptive_threshold(
            T.TM_OTSU, img, global_threshold,
            adaptive_window_size = 10)
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
        
if __name__=="__main__":
    unittest.main()
    
