"""Tests for CellProfiler.Objects

"""
import numpy
import scipy.ndimage
import unittest
import CellProfiler.Objects

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
        
    
    def test_01_01_SetSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.SetSegmented(self.__segmented10)
        self.assertTrue((self.__segmented10==x.Segmented).all())
    
    def test_01_02_Segmented(self):
        x = CellProfiler.Objects.Objects()
        x.Segmented = self.__segmented10
        self.assertTrue((self.__segmented10==x.Segmented).all())
    
    def test_01_03_SetUneditedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.SetUneditedSegmented(self.__unedited_segmented10)
        self.assertTrue((self.__unedited_segmented10 == x.UneditedSegmented).all())
    
    def test_01_04_UneditedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.UneditedSegmented = self.__unedited_segmented10
        self.assertTrue((self.__unedited_segmented10== x.UneditedSegmented).all())
    
    def test_01_05_SetSmallRemovedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.SetSmallRemovedSegmented(self.__small_removed_segmented10)
        self.assertTrue((self.__small_removed_segmented10==x.SmallRemovedSegmented).all())
    
    def test_01_06_UneditedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.SmallRemovedSegmented = self.__small_removed_segmented10
        self.assertTrue((self.__small_removed_segmented10== x.SmallRemovedSegmented).all())
        
    def test_02_01_SetAll(self):
        x = CellProfiler.Objects.Objects()
        x.Segmented = self.__segmented10
        x.UneditedSegmented = self.__unedited_segmented10
        x.SmallRemovedSegmented = self.__small_removed_segmented10

    def test_03_01_DefaultUneditedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.Segmented = self.__segmented10
        self.assertTrue((x.UneditedSegmented==x.Segmented).all())
    
    def test_03_02_DefaultSmallRemovedSegmented(self):
        x = CellProfiler.Objects.Objects()
        x.Segmented = self.__segmented10
        self.assertTrue((x.SmallRemovedSegmented == 0).all())
        x.UneditedSegmented = self.__unedited_segmented10
        self.assertTrue((x.SmallRemovedSegmented == self.__small_removed_segmented10).all())
    
    def test_04_01_MisSize(self):
        x = CellProfiler.Objects.Objects()
        x.Segmented = self.__segmented10
        self.assertRaises(AssertionError,x.SetUneditedSegmented,numpy.ones((5,5)))
        self.assertRaises(AssertionError,x.SetSmallRemovedSegmented,numpy.ones((5,5)))
        
        
if __name__ == "__main__":
    unittest.main()
