"""Tests for CellProfiler.Objects

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import numpy as np
import scipy.ndimage
import unittest
import cStringIO
import bz2
import base64

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

    def relate_ijv(self, parent_ijv, children_ijv):
        p = cpo.Objects()
        p.ijv = parent_ijv
        c = cpo.Objects()
        c.ijv = children_ijv
        return p.relate_children(c)
    
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
        child_counts, parents_of = self.relate_ijv(
            np.zeros((0,3), int), np.zeros((0,3), int))
        self.assertEqual(len(child_counts), 0)
        self.assertEqual(len(parents_of), 0)
        
        child_counts, parents_of = self.relate_ijv(
            np.zeros((0,3), int), np.array([[1,2,3]]))
        self.assertEqual(len(child_counts), 0)
        self.assertEqual(len(parents_of), 3)
        self.assertEqual(parents_of[2], 0)
        
        child_counts, parents_of = self.relate_ijv(
            np.array([[1,2,3]]), np.zeros((0,3), int))
        self.assertEqual(len(child_counts), 3)
        self.assertEqual(child_counts[2], 0)
        self.assertEqual(len(parents_of), 0)
        
    def test_05_08_relate_ijv_no_match(self):
        child_counts, parents_of = self.relate_ijv(
            np.array([[3,2,1]]), np.array([[5,6,1]]))
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 0)
        self.assertEqual(len(parents_of), 1)
        self.assertEqual(parents_of[0], 0)
        
    def test_05_09_relate_ijv_one_match(self):
        child_counts, parents_of = self.relate_ijv(
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
        child_counts, parents_of = self.relate_ijv(
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
        child_counts, parents_of = self.relate_ijv(
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
        child_counts, parents_of = self.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),3)
        self.assertEqual(tuple(child_counts), (1, 0, 1))
        self.assertEqual(len(parents_of), 2)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 3)
        
    def test_05_13_relate_many_child_missing_parent(self):
        child_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        parent_ijv = np.array([[1,0,1], [3,0,2]])
        child_counts, parents_of = self.relate_ijv(
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
        child_counts, parents_of = self.relate_ijv(
            parent_ijv, child_ijv)
        self.assertEqual(len(child_counts),3)
        self.assertEqual(tuple(child_counts), (1, 1, 0))
        self.assertEqual(len(parents_of), 2)
        self.assertEqual(parents_of[0], 1)
        self.assertEqual(parents_of[1], 2)
        
    def test_05_15_relate_many_child_missing_end(self):
        child_ijv = np.array([[1,0,1], [2,0,2],[3,0,3]])
        parent_ijv = np.array([[1,0,1], [2,0,2]])
        child_counts, parents_of = self.relate_ijv(
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
        labels = np.zeros((20,20), int)
        labels[1:-1,1:-1] = np.random.randint(0,10,size=(18,18))
        
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
            
    def test_06_05_ijv_three_overlapping(self):
        #
        # This is a regression test of a bug where a segmentation consists
        # of only one point, labeled three times yielding two planes instead
        # of three.
        #
        ijv = np.array([[4, 5, 1],
                        [4, 5, 2], 
                        [4, 5, 3]])
        x = cpo.Objects()
        x.set_ijv(ijv, (8, 9))
        labels = []
        indices = np.zeros(3, bool)
        for l, i in x.get_labels():
            labels.append(l)
            self.assertEqual(len(i), 1)
            self.assertTrue(i[0] in (1, 2, 3))
            indices[i[0]-1] = True
        self.assertTrue(np.all(indices))
        self.assertEqual(len(labels), 3)
        lstacked = np.dstack(labels)
        i, j, k = np.mgrid[0:lstacked.shape[0],
                           0:lstacked.shape[1],
                           0:lstacked.shape[2]]
        self.assertTrue(np.all(lstacked[(i != 4) | (j != 5)] == 0))
        self.assertEqual((1, 2, 3), tuple(sorted(lstacked[4, 5, :])))
            
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
        
        x.set_ijv(np.column_stack((i,j,v)), ii.shape)
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

    def test_07_02_labels_same_as_ijv(self):
        d = ('QlpoOTFBWSZTWeu0qJwGoDt///////////////9///////9//3///3//f3//f/9////4YCAfH0ki'
             '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')
        stream = cStringIO.StringIO(bz2.decompress(base64.b64decode(d)))
        x = cpo.Objects()
        x.segmented = np.load(stream)
        y = cpo.Objects()
        y.segmented = np.load(stream)
        labels_children_per_parent, labels_parents_of_children = x.relate_children(y)
        # force generation of ijv
        x.ijv, y.ijv
        ijv_children_per_parent, ijv_parents_of_children = x.relate_children(y)
        np.testing.assert_array_equal(labels_children_per_parent, ijv_children_per_parent)
        np.testing.assert_array_equal(labels_parents_of_children, ijv_parents_of_children)
        
    def test_08_01_cache(self):
        import h5py
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        import os
        import tempfile
        x = cpo.Objects()
        r = np.random.RandomState()
        r.seed(81)
        segmented_unedited = r.randint(0, 5, size=(10, 15))
        segmented_small_removed = segmented_unedited.copy()
        segmented_small_removed[segmented_small_removed == 4] = 0
        segmented = segmented_small_removed.copy()
        segmented[segmented == 3] = 0
        x.segmented = segmented
        x.small_removed_segmented = segmented_small_removed
        x.unedited_segmented = segmented_unedited
        y = cpo.Objects()
        y.segmented = segmented
        y.small_removed_segmented = segmented_small_removed
        y.unedited_segmented = segmented_unedited
        
        fd, path = tempfile.mkstemp(".h5")
        f = h5py.File(path)
        try:
            cache = HDF5ObjectSet(f)
            x.cache(cache, "whatever")
            np.testing.assert_array_equal(x.segmented, segmented)
            np.testing.assert_array_equal(x.small_removed_segmented,
                                          segmented_small_removed)
            np.testing.assert_array_equal(x.unedited_segmented,
                                          segmented_unedited)
            np.testing.assert_array_equal(y.ijv, x.ijv)
        finally:
            f.close()
            os.close(fd)
            os.remove(path)
            

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
