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
import cellprofiler.cpimage as cpi
from cellprofiler.cpmath.outline import outline

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
        
    def test_05_07_relate_ijv_none(self):
        child_counts, parents_of = cpo.Objects.relate_ijv(
            np.zeros((0,3), int), np.zeros((0,3), int))
        self.assertEqual(len(child_counts), 0)
        self.assertEqual(len(parents_of), 0)
        
        child_counts, parents_of = cpo.Objects.relate_ijv(
            np.zeros((0,3), int), np.array([[1,2,3]]))
        self.assertEqual(len(child_counts), 0)
        self.assertEqual(len(parents_of), 1)
        self.assertEqual(parents_of[0], 0)
        
        child_counts, parents_of = cpo.Objects.relate_ijv(
            np.array([[1,2,3]]), np.zeros((0,3), int))
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 0)
        self.assertEqual(len(parents_of), 0)
        
    def test_05_08_relate_ijv_no_match(self):
        child_counts, parents_of = cpo.Objects.relate_ijv(
            np.array([[3,2,1]]), np.array([[5,6,1]]))
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 0)
        self.assertEqual(len(parents_of), 1)
        self.assertEqual(parents_of[0], 0)
        
    def test_05_09_relate_ijv_one_match(self):
        child_counts, parents_of = cpo.Objects.relate_ijv(
            np.array([[3,2,1]]), np.array([[3,2,1]]))
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        self.assertEqual(len(parents_of), 1)
        self.assertEqual(parents_of[0], 1)
        
    def test_05_10_relate_ijv_many_points_one_match(self):
        r = np.random.RandomState()
        r.seed(510)
        parent_ijv = np.column_stack((
            r.randint(0,10,size=(100,2)), np.ones(100, int)))
        child_ijv = np.column_stack((
            r.randint(0,10,size=(100,2)), np.ones(100, int)))
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        self.assertEqual(len(parents_of), 1)
        self.assertEqual(parents_of[0], 1)

    def test_05_11_relate_many_many(self):
        r = np.random.RandomState()
        r.seed(511)
        parent_ijv = np.column_stack((
            r.randint(0,10,size=(100,2)), np.ones(100, int)))
        child_ijv = np.column_stack((
            r.randint(0,10,size=(100,2)), np.ones(100, int)))
        parent_ijv[parent_ijv[:,0] >= 5, 2] = 2
        child_ijv[:,2] = (
            1 + (child_ijv[:,0] >= 5).astype(int) + 
            2 * (child_ijv[:,1] >= 5).astype(int))
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),2)
        self.assertEqual(tuple(child_counts), (2,2))
        self.assertEqual(len(parents_of), 4)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 2)
        self.assertEqual(parents_of[2], 1)
        self.assertEqual(parents_of[3], 2)
        
    def test_05_12_relate_many_parent_missing_child(self):
        parent_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        child_ijv = np.array([[1,0,1], [3,0,2]])
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),3)
        self.assertEqual(tuple(child_counts), (1, 0, 1))
        self.assertEqual(len(parents_of), 2)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 3)
        
    def test_05_13_relate_many_child_missing_parent(self):
        child_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        parent_ijv = np.array([[1,0,1], [3,0,2]])
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),2)
        self.assertEqual(tuple(child_counts), (1, 1))
        self.assertEqual(len(parents_of), 3)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 0)
        self.assertEqual(parents_of[2], 2)
        
    def test_05_14_relate_many_parent_missing_child_end(self):
        parent_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        child_ijv = np.array([[1,0,1], [2,0,2]])
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),3)
        self.assertEqual(tuple(child_counts), (1, 1, 0))
        self.assertEqual(len(parents_of), 2)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 2)
        
    def test_05_15_relate_many_child_missing_end(self):
        child_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        parent_ijv = np.array([[1,0,1], [2,0,2]])
        child_counts, parents_of = cpo.Objects.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),2)
        self.assertEqual(tuple(child_counts), (1, 1))
        self.assertEqual(len(parents_of), 3)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 2)
        self.assertEqual(parents_of[2], 0)
        
        
    def test_06_01_segmented_to_ijv(self):
        '''Convert the segmented representation to an IJV one'''
        x = cpo.Objects()
        np.random.seed(61)
        labels = np.random.randint(0,10,size=(20,20))
        x.segmented = labels
        ijv = x.get_ijv()
        new_labels = np.zeros(labels.shape, int)
        new_labels[ijv[:,0],ijv[:,1]] = ijv[:,2]
        self.assertTrue(np.all(labels == new_labels))
        
    def test_06_02_ijv_to_labels_empty(self):
        '''Convert a blank ijv representation to labels'''
        x = cpo.Objects()
        x.ijv = np.zeros((0,3), int)
        y = x.get_labels()
        self.assertEqual(len(y), 1)
        labels, indices = y[0]
        self.assertEqual(len(indices), 0)
        self.assertTrue(np.all(labels == 0))
        
    def test_06_03_ijv_to_labels_simple(self):
        '''Convert an ijv representation w/o overlap to labels'''
        x = cpo.Objects()
        np.random.seed(63)
        labels = np.random.randint(0,10,size=(20,20))
        x.segmented = labels
        ijv = x.get_ijv()
        x = cpo.Objects()
        x.ijv = ijv
        labels_out = x.get_labels()
        self.assertEqual(len(labels_out), 1)
        labels_out, indices = labels_out[0]
        self.assertTrue(np.all(labels_out == labels))
        self.assertEqual(len(indices), 9)
        self.assertTrue(np.all(np.unique(indices)==np.arange(1,10)))
        
    def test_06_04_ijv_to_labels_overlapping(self):
        '''Convert an ijv representation with overlap to labels'''
        ijv = np.array([[1,1,1],
                        [1,2,1],
                        [2,1,1],
                        [2,2,1],
                        [1,3,2],
                        [2,3,2],
                        [2,3,3],
                        [4,4,4],
                        [4,5,4],
                        [4,5,5],
                        [5,5,5]])
        x = cpo.Objects()
        x.ijv = ijv
        labels = x.get_labels()
        self.assertEqual(len(labels), 2)
        unique_a = np.unique(labels[0][0])[1:]
        unique_b = np.unique(labels[1][0])[1:]
        for a in unique_a:
            self.assertTrue(a not in unique_b)
        for b in unique_b:
            self.assertTrue(b not in unique_a)
        for i, j, v in ijv:
            mylabels = labels[0][0] if v in unique_a else labels[1][0]
            self.assertEqual(mylabels[i,j], v)
            
    def test_07_00_make_ivj_outlines_empty(self):
        np.random.seed(70)
        x = cpo.Objects()
        x.segmented = np.zeros((10,20), int)
        image = x.make_ijv_outlines(np.random.uniform(size=(5,3)))
        self.assertTrue(np.all(image == 0))
        
    def test_07_01_make_ijv_outlines(self):
        np.random.seed(70)
        x = cpo.Objects()
        ii,jj = np.mgrid[0:10,0:20]
        masks = [(ii-ic)**2 + (jj - jc) **2 < r **2 
                 for ic, jc, r in ((4,5,5), (4,12,5), (6, 8, 5))]
        i = np.hstack([ii[mask] for mask in masks])
        j = np.hstack([jj[mask] for mask in masks])
        v = np.hstack([[k+1] * np.sum(mask) for k, mask in enumerate(masks)])
        
        x.ijv = np.column_stack((i,j,v))
        x.parent_image = cpi.Image(np.zeros((10,20)))
        colors = np.random.uniform(size=(3, 3)).astype(np.float32)
        image = x.make_ijv_outlines(colors)
        i1 = [i for i, color in enumerate(colors) if np.all(color == image[0,5,:])]
        self.assertEqual(len(i1), 1)
        i2 = [i for i, color in enumerate(colors) if np.all(color == image[0,12,:])]
        self.assertEqual(len(i2), 1)
        i3 = [i for i, color in enumerate(colors) if np.all(color == image[-1,8,:])]
        self.assertEqual(len(i3), 1)
        self.assertNotEqual(i1[0], i2[0])
        self.assertNotEqual(i2[0], i3[0])
        colors = colors[np.array([i1[0], i2[0], i3[0]])]
        outlines = np.zeros((10,20,3), np.float32)
        alpha = np.zeros((10,20))
        for i, (color, mask) in enumerate(zip(colors, masks)):
            my_outline = outline(mask)
            outlines[my_outline] += color
            alpha[my_outline] += 1
        alpha[alpha == 0] = 1
        outlines /= alpha[:,:,np.newaxis]
        np.testing.assert_almost_equal(outlines, image)

class TestDownsampleLabels(unittest.TestCase):
    def test_01_01_downsample_127(self):
        i,j = np.mgrid[0:16, 0:8]
        labels = (i*8 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int8))
        self.assertTrue(np.all(result == labels))
        
    def test_01_02_downsample_128(self):
        i,j = np.mgrid[0:16, 0:8]
        labels = (i*8 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int16))
        self.assertTrue(np.all(result == labels))
    
    def test_01_03_downsample_32767(self):
        i,j = np.mgrid[0:256, 0:128]
        labels = (i*128 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int16))
        self.assertTrue(np.all(result == labels))

    def test_01_04_downsample_32768(self):
        i,j = np.mgrid[0:256, 0:128]
        labels = (i*128 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int32))
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
        
    def test_01_04_size_color(self):
        secondary, mask = cpo.size_similarly(np.zeros((10,20), int), 
                                             np.zeros((10,15,3), np.float32))
        self.assertEqual(tuple(secondary.shape), (10,20,3))
        self.assertTrue(np.all(mask[:10,:15]))
        self.assertTrue(np.all(~mask[:10,15:]))
        self.assertEqual(secondary.dtype, np.dtype(np.float32))
        