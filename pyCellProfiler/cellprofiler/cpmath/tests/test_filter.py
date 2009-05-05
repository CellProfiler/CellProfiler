'''test_filter - test the filter module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision: 1 "

import numpy as np
import unittest

import cellprofiler.cpmath.filter as F

'''Perform line-integration per-column of the image'''
VERTICAL = 'vertical'
'''Perform line-integration per-row of the image'''
HORIZONTAL = 'horizontal'
'''Perform line-integration along diagonals from top left to bottom right'''
DIAGONAL = 'diagonal'
'''Perform line-integration along diagonals from top right to bottom left'''
ANTI_DIAGONAL = 'anti-diagonal'

class TestStretch(unittest.TestCase):
    def test_00_00_empty(self):
        result = F.stretch(np.zeros((0,)))
        self.assertEqual(len(result), 0)
    
    def test_00_01_empty_plus_mask(self):
        result = F.stretch(np.zeros((0,)), np.zeros((0,),bool))
        self.assertEqual(len(result), 0)
    
    def test_00_02_zeros(self):
        result = F.stretch(np.zeros((10,10)))
        self.assertTrue(np.all(result == 0))
    
    def test_00_03_zeros_plus_mask(self):
        result = F.stretch(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result == 0))
    
    def test_00_04_half(self):
        result = F.stretch(np.ones((10,10))*.5)
        self.assertTrue(np.all(result == .5))
    
    def test_00_05_half_plus_mask(self):
        result = F.stretch(np.ones((10,10))*.5, np.ones((10,10),bool))
        self.assertTrue(np.all(result == .5))

    def test_01_01_rescale(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10,10))
        image[0,0] = -2
        image[9,9] = 2
        expected = (image + 2.0)/4.0
        result = F.stretch(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_rescale_plus_mask(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10,10))
        mask = np.zeros((10,10), bool)
        mask[1:9,1:9] = True
        image[0,0] = -4
        image[9,9] = 4
        image[1,1] = -2
        image[8,8] = 2
        expected = (image[1:9,1:9] + 2.0)/4.0
        result = F.stretch(image, mask)
        self.assertTrue(np.all(result[1:9,1:9] == expected))

class TestMedianFilter(unittest.TestCase):
    def test_00_00_zeros(self):
        '''The median filter on an array of all zeros should be zero'''
        result = F.median_filter(np.zeros((10,10)), np.ones((10,10),bool), 3)
        self.assertTrue(np.all(result == 0))
    
    def test_01_01_mask(self):
        '''The median filter, masking a single value'''
        img = np.zeros((10,10))
        img[5,5] = 1
        mask = np.ones((10,10),bool)
        mask[5,5] = False
        result = F.median_filter(img, mask, 3)
        self.assertTrue(np.all(result[mask] == 0))
        self.assertEqual(result[5,5], 1)
    
    def test_02_01_median(self):
        '''A median filter larger than the image = median of image'''
        np.random.seed(0)
        img = np.random.uniform(size=(9,9))
        result = F.median_filter(img, np.ones((9,9),bool), 20)
        self.assertEqual(result[0,0], np.median(img))
        self.assertTrue(np.all(result == np.median(img)))
    
    def test_02_02_median_bigger(self):
        '''Use an image of more than 255 values to test approximation'''
        np.random.seed(0)
        img = np.random.uniform(size=(20,20))
        result = F.median_filter(img, np.ones((20,20),bool),40)
        sorted = np.ravel(img)
        sorted.sort()
        min_acceptable = sorted[198]
        max_acceptable = sorted[202]
        self.assertTrue(np.all(result >= min_acceptable))
        self.assertTrue(np.all(result <= max_acceptable))
        
    def test_03_01_shape(self):
        '''Make sure the median filter is the expected octagonal shape'''
        
        radius = 5
        a_2 = int(radius / 2.414213)
        i,j = np.mgrid[-10:11,-10:11]
        octagon = np.ones((21,21), bool)
        #
        # constrain the octagon mask to be the points that are on
        # the correct side of the 8 edges
        #
        octagon[i < -radius] = False
        octagon[i > radius]  = False
        octagon[j < -radius] = False
        octagon[j > radius]  = False
        octagon[i+j < -radius-a_2] = False
        octagon[j-i >  radius+a_2] = False
        octagon[i+j >  radius+a_2] = False
        octagon[i-j >  radius+a_2] = False
        np.random.seed(0)
        img = np.random.uniform(size=(21,21))
        result = F.median_filter(img, np.ones((21,21),bool), radius)
        sorted = img[octagon]
        sorted.sort()
        min_acceptable = sorted[len(sorted)/2-1]
        max_acceptable = sorted[len(sorted)/2+1]
        self.assertTrue(result[10,10] >= min_acceptable)
        self.assertTrue(result[10,10] <= max_acceptable)
        
