"""Tests for CellProfiler.Objects

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import numpy
import scipy.ndimage
import unittest
import cellprofiler.objects

class TestObjects(unittest.TestCase):
    def setUp(self):
        self.__image10 = numpy.zeros((10,10),dtype=numpy.bool)
        self.__image10[2:4,2:4] = 1
        self.__image10[5:7,5:7] = 1
        self.__unedited_segmented10,count =scipy.ndimage.label(self.__image10)
        assert count==2
        self.__segmented10 = self.__unedited_segmented10.copy()
        self.__segmented10[self.__segmented10==2] = 0
        self.__small_removed_segmented10 = self.__unedited_segmented10.copy()
        self.__small_removed_segmented10[self.__segmented10==1] = 0
        
    
    def test_01_01_set_segmented(self):
        x = cellprofiler.objects.Objects()
        x.set_segmented(self.__segmented10)
        self.assertTrue((self.__segmented10==x.segmented).all())
    
    def test_01_02_segmented(self):
        x = cellprofiler.objects.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((self.__segmented10==x.segmented).all())
    
    def test_01_03_set_unedited_segmented(self):
        x = cellprofiler.objects.Objects()
        x.set_unedited_segmented(self.__unedited_segmented10)
        self.assertTrue((self.__unedited_segmented10 == x.unedited_segmented).all())
    
    def test_01_04_unedited_segmented(self):
        x = cellprofiler.objects.Objects()
        x.unedited_segmented = self.__unedited_segmented10
        self.assertTrue((self.__unedited_segmented10== x.unedited_segmented).all())
    
    def test_01_05_set_small_removed_segmented(self):
        x = cellprofiler.objects.Objects()
        x.set_small_removed_segmented(self.__small_removed_segmented10)
        self.assertTrue((self.__small_removed_segmented10==x.small_removed_segmented).all())
    
    def test_01_06_unedited_segmented(self):
        x = cellprofiler.objects.Objects()
        x.small_removed_segmented = self.__small_removed_segmented10
        self.assertTrue((self.__small_removed_segmented10== x.small_removed_segmented).all())
        
    def test_02_01_set_all(self):
        x = cellprofiler.objects.Objects()
        x.segmented = self.__segmented10
        x.unedited_segmented = self.__unedited_segmented10
        x.small_removed_segmented = self.__small_removed_segmented10

    def test_03_01_default_unedited_segmented(self):
        x = cellprofiler.objects.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((x.unedited_segmented==x.segmented).all())
    
    def test_03_02_default_small_removed_segmented(self):
        x = cellprofiler.objects.Objects()
        x.segmented = self.__segmented10
        self.assertTrue((x.small_removed_segmented == 0).all())
        x.unedited_segmented = self.__unedited_segmented10
        self.assertTrue((x.small_removed_segmented == self.__small_removed_segmented10).all())
    
    def test_04_01_mis_size(self):
        x = cellprofiler.objects.Objects()
        x.segmented = self.__segmented10
        self.assertRaises(AssertionError,x.set_unedited_segmented,numpy.ones((5,5)))
        self.assertRaises(AssertionError,x.set_small_removed_segmented,numpy.ones((5,5)))
    
    def test_05_01_relate_zero_parents_and_children(self):
        """Test the relate method if both parent and child label matrices are zeros"""
        x = cellprofiler.objects.Objects()
        x.segmented = numpy.zeros((10,10),int)
        y = cellprofiler.objects.Objects()
        y.segmented = numpy.zeros((10,10),int)
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 0)
        self.assertEqual(numpy.product(parents_of_children.shape), 0)
    
    def test_05_02_relate_zero_parents_one_child(self): 
        x = cellprofiler.objects.Objects()
        x.segmented = numpy.zeros((10,10),int)
        y = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:6] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 0)
        self.assertEqual(numpy.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],0)
    
    def test_05_03_relate_one_parent_no_children(self):
        x = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:6] = 1
        x.segmented = labels
        y = cellprofiler.objects.Objects()
        y.segmented = numpy.zeros((10,10),int)
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 0)
        self.assertEqual(numpy.product(parents_of_children.shape), 0)
        
    def test_05_04_relate_one_parent_one_child(self):
        x = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:6] = 1
        x.segmented = labels
        y = cellprofiler.objects.Objects()
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 1)
        self.assertEqual(numpy.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],1)
    
    def test_05_05_relate_two_parents_one_child(self):
        x = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:6] = 1
        labels[3:6,7:9] = 2
        x.segmented = labels
        y = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,5:9] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 2)
        self.assertEqual(children_per_parent[0], 0)
        self.assertEqual(children_per_parent[1], 1)
        self.assertEqual(numpy.product(parents_of_children.shape), 1)
        self.assertEqual(parents_of_children[0],2)
        
    def test_05_06_relate_one_parent_two_children(self):
        x = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:9] = 1
        x.segmented = labels
        y = cellprofiler.objects.Objects()
        labels = numpy.zeros((10,10),int)
        labels[3:6,3:6] = 1
        labels[3:6,7:9] = 2
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        self.assertEqual(numpy.product(children_per_parent.shape), 1)
        self.assertEqual(children_per_parent[0], 2)
        self.assertEqual(numpy.product(parents_of_children.shape), 2)
        self.assertEqual(parents_of_children[0],1)
        self.assertEqual(parents_of_children[1],1)
        
if __name__ == "__main__":
    unittest.main()
