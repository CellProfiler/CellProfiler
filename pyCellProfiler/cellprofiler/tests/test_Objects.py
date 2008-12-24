"""Tests for CellProfiler.Objects

"""
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
        
        
if __name__ == "__main__":
    unittest.main()
