"""Tests for CellProfiler.Objects

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import numpy as np
import scipy.ndimage
import unittest
import cellprofiler.objects as cpo

class TestObjects(unittest.TestCase):
    def setUp(self):
        self.__image10 = np.zeros((10,10),dtype=np.bool)
        self.__image10[2:4,2:4] = 1
        self.__image10[5:7,5:7] = 1
        self.__unedited_segmented10,count =scipy.ndimage.label(self.__image10)
        assert count==2
        self.__segmented10 = self.__unedited_segmented10.copy()
        self.__segmented10[self.__segmented10==2] = 0
        self.__small_removed_segmented10 = self.__unedited_segmented10.copy()
        self.__small_removed_segmented10[self.__segmented10==1] = 0
        
    
    def test_01_01_set_segmented(self):
        x = cpo.Objects()
        x.set_segmented(self.__segmented10)
        self.assertTrue((self.__segmented10==x.segmented).all())
    
    def test_01_02_segmented(self):
        x = cpo.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((self.__segmented10==x.segmented).all())
    
    def test_01_03_set_unedited_segmented(self):
        x = cpo.Objects()
        x.set_unedited_segmented(self.__unedited_segmented10)
        self.assertTrue((self.__unedited_segmented10 == x.unedited_segmented).all())
    
    def test_01_04_unedited_segmented(self):
        x = cpo.Objects()
        x.unedited_segmented = self.__unedited_segmented10
        self.assertTrue((self.__unedited_segmented10== x.unedited_segmented).all())
    
    def test_01_05_set_small_removed_segmented(self):
        x = cpo.Objects()
        x.set_small_removed_segmented(self.__small_removed_segmented10)
        self.assertTrue((self.__small_removed_segmented10==x.small_removed_segmented).all())
    
    def test_01_06_unedited_segmented(self):
        x = cpo.Objects()
        x.small_removed_segmented = self.__small_removed_segmented10
        self.assertTrue((self.__small_removed_segmented10== x.small_removed_segmented).all())
        
    def test_02_01_set_all(self):
        x = cpo.Objects()
        x.segmented = self.__segmented10
        x.unedited_segmented = self.__unedited_segmented10
        x.small_removed_segmented = self.__small_removed_segmented10

    def test_03_01_default_unedited_segmented(self):
        x = cpo.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((x.unedited_segmented==x.segmented).all())
    
    def test_03_02_default_small_removed_segmented(self):
        x = cpo.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((x.small_removed_segmented == self.__segmented10).all())
        x.unedited_segmented = self.__unedited_segmented10
        self.assertTrue((x.small_removed_segmented == self.__unedited_segmented10).all())
    
    def test_04_01_mis_size(self):
        x = cpo.Objects()
        x.segmented = self.__segmented10
        self.assertRaises(AssertionError,x.set_unedited_segmented,np.ones((5,5)))
        self.assertRaises(AssertionError,x.set_small_removed_segmented,np.ones((5,5)))
    
    def test_05_01_relate_zero_parents_and_children(self):
        """Test the relate method if both parent and child label matrices are zeros"""
        x = cpo.Objects()
        x.segmented = np.zeros((10,10),int)
        y = cpo.Objects()
        y.segmented = np.zeros((10,10),int)
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 0)
        self.assertEqual(np.product(parents_of_children.shape), 0)
    
    def test_05_02_relate_zero_parents_one_child(self): 
        x = cpo.Objects()
        x.segmented = np.zeros((10,10),int)
        y = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 0)
        self.assertEqual(np.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],0)
    
    def test_05_03_relate_one_parent_no_children(self):
        x = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        x.segmented = labels
        y = cpo.Objects()
        y.segmented = np.zeros((10,10),int)
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 0)
        self.assertEqual(np.product(parents_of_children.shape), 0)
        
    def test_05_04_relate_one_parent_one_child(self):
        x = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        x.segmented = labels
        y = cpo.Objects()
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 1)
        self.assertEqual(np.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],1)
    
    def test_05_05_relate_two_parents_one_child(self):
        x = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        labels[3:6,7:9] = 2
        x.segmented = labels
        y = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,5:9] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 2)
        self.assertEqual(children_per_parent[0], 0)
        self.assertEqual(children_per_parent[1], 1)
        self.assertEqual(np.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],2)
        
    def test_05_06_relate_one_parent_two_children(self):
        x = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:9] = 1
        x.segmented = labels
        y = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        labels[3:6,7:9] = 2
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(np.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 2)
        self.assertEqual(np.product(parents_of_children.shape), 2)
        self.assertEqual(parents_of_children[0],1)
        self.assertEqual(parents_of_children[1],1)

class TestDownsampleLabels(unittest.TestCase):
    def test_01_01_downsample_255(self):
        i,j = np.mgrid[0:16, 0:16]
        labels = (i*16 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.uint8))
        self.assertTrue(np.all(result == labels))
        
    def test_01_02_downsample_256(self):
        i,j = np.mgrid[0:16, 0:16]
        labels = (i*16 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.uint16))
        self.assertTrue(np.all(result == labels))
    
    def test_01_03_downsample_65535(self):
        i,j = np.mgrid[0:256, 0:256]
        labels = (i*256 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.uint16))
        self.assertTrue(np.all(result == labels))

    def test_01_04_downsample_65536(self):
        i,j = np.mgrid[0:256, 0:256]
        labels = (i*256 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.uint32))
        self.assertTrue(np.all(result == labels))

class TestCropLabelsAndImage(unittest.TestCase):
    def test_01_01_crop_same(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)), 
                                                  np.zeros((10,20)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))
        
    def test_01_02_crop_image(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)), 
                                                  np.zeros((10,30)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)), 
                                                  np.zeros((20, 20)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))

    def test_01_03_crop_labels(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 30)), 
                                                  np.zeros((10,20)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))
        labels, image = cpo.crop_labels_and_image(np.zeros((20, 20)), 
                                                  np.zeros((10, 20)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))
        
    def test_01_04_crop_both(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 30)), 
                                                  np.zeros((20,20)))
        self.assertEqual(tuple(labels.shape), (10,20))
        self.assertEqual(tuple(image.shape), (10,20))

class TestSizeSimilarly(unittest.TestCase):
    def test_01_01_size_same(self):
        secondary, mask = cpo.size_similarly(np.zeros((10,20)), 
                                             np.zeros((10,20)))
        self.assertEqual(tuple(secondary.shape), (10,20))
        self.assertTrue(np.all(mask))
        
    def test_01_02_larger_secondary(self):
        secondary, mask = cpo.size_similarly(np.zeros((10,20)), 
                                             np.zeros((10,30)))
        self.assertEqual(tuple(secondary.shape), (10,20))
        self.assertTrue(np.all(mask))
        secondary, mask = cpo.size_similarly(np.zeros((10,20)), 
                                             np.zeros((20,20)))
        self.assertEqual(tuple(secondary.shape), (10,20))
        self.assertTrue(np.all(mask))
    
    def test_01_03_smaller_secondary(self):
        secondary, mask = cpo.size_similarly(np.zeros((10,20), int), 
                                             np.zeros((10,15), np.float32))
        self.assertEqual(tuple(secondary.shape), (10,20))
        self.assertTrue(np.all(mask[:10,:15]))
        self.assertTrue(np.all(~mask[:10,15:]))
        self.assertEqual(secondary.dtype, np.dtype(np.float32))
        