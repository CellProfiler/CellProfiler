""" test_Measurements.py - tests for CellProfiler.Measurements
"""
import unittest
import numpy
import CellProfiler.Measurements

class TestMeasurements(unittest.TestCase):
    def test_00_00_Init(self):
        x = CellProfiler.Measurements.Measurements()
    
    def test_01_01_ImageNumberIsZero(self):
        x = CellProfiler.Measurements.Measurements()
        self.assertEqual(x.ImageSetNumber,0)
    
    def test_01_01_NextImage(self):
        x = CellProfiler.Measurements.Measurements()
        x.NextImageSet()
        self.assertEqual(x.ImageSetNumber,1)
    
    def test_02_01_AddImageMeasurement(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value" )
        self.assertEqual(x.GetCurrentMeasurement("Image", "Feature"),"Value")
        self.assertTrue("Image" in x.GetObjectNames())
        self.assertTrue("Feature" in x.GetFeatureNames("Image"))
    
    def test_02_02_AddObjectMeasurement(self):
        x = CellProfiler.Measurements.Measurements()
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.AddMeasurement("Nuclei", "Feature",m)
        self.assertTrue((x.GetCurrentMeasurement("Nuclei", "Feature")==m).all)
        self.assertTrue("Nuclei" in x.GetObjectNames())
        self.assertTrue("Feature" in x.GetFeatureNames("Nuclei"))
    
    def test_02_03_AddTwoMeasurements(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value" )
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.AddMeasurement("Nuclei", "Feature",m)
        self.assertEqual(x.GetCurrentMeasurement("Image", "Feature"),"Value")
        self.assertTrue((x.GetCurrentMeasurement("Nuclei", "Feature")==m).all())
        self.assertTrue("Image" in x.GetObjectNames())
        self.assertTrue("Nuclei" in x.GetObjectNames())
        self.assertTrue("Feature" in x.GetFeatureNames("Image"))
    
    def test_02_04_AddTwoMeasurementsToObject(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature1","Value1" )
        x.AddMeasurement("Image", "Feature2","Value2" )
        self.assertEqual(x.GetCurrentMeasurement("Image", "Feature1"),"Value1")
        self.assertEqual(x.GetCurrentMeasurement("Image", "Feature2"),"Value2")
        self.assertTrue("Image" in x.GetObjectNames())
        self.assertTrue("Feature1" in x.GetFeatureNames("Image"))
        self.assertTrue("Feature2" in x.GetFeatureNames("Image"))
    
    def test_03_03_MultipleImageSets(self):
        numpy.random.seed(0)
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value1" )
        m1 = numpy.random.rand(10)
        x.AddMeasurement("Nuclei", "Feature",m1)
        x.NextImageSet()
        x.AddMeasurement("Image", "Feature","Value2" )
        m2 = numpy.random.rand(10)
        x.AddMeasurement("Nuclei", "Feature",m2)
        self.assertEqual(x.GetCurrentMeasurement("Image", "Feature"),"Value2")
        self.assertTrue((x.GetCurrentMeasurement("Nuclei", "Feature")==m2).all())
        for a,b in zip(x.GetAllMeasurements("Image", "Feature"),["Value1","Value2"]):
            self.assertEqual(a,b)
        for a,b in zip(x.GetAllMeasurements("Nuclei","Feature"),[m1,m2]):
            self.assertTrue((a==b).all())
    
    def test_04_01_NegativeAddTwice(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.AddMeasurement,"Image", "Feature","Value")
    
    def test_04_02_NegativeAddOnSecondPass(self):
        x = CellProfiler.Measurements.Measurements()
        x.NextImageSet()
        self.assertRaises(AssertionError,x.AddMeasurement,"Image", "Feature","Value")
    
    def test_04_03_NegativeAddOtherOnSecondPass(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature1","Value1" )
        x.NextImageSet()
        x.AddMeasurement("Image", "Feature1","Value1" )
        self.assertRaises(AssertionError,x.AddMeasurement,"Image", "Feature2","Value2")
        
    def test_04_04_NegativeAddTwiceOnSecondPass(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value" )
        x.NextImageSet()
        x.AddMeasurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.AddMeasurement,"Image", "Feature","Value")
    
    def test_05_01_TestHasCurrentMeasurements(self):
        x = CellProfiler.Measurements.Measurements()
        self.assertFalse(x.HasCurrentMeasurements('Image', 'Feature'))
                         
    def test_05_02_TestHasCurrentMeasurements(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "OtherFeature","Value" )
        self.assertFalse(x.HasCurrentMeasurements('Image', 'Feature'))

    def test_05_03_TestHasCurrentMeasurements(self):
        x = CellProfiler.Measurements.Measurements()
        x.AddMeasurement("Image", "Feature","Value" )
        self.assertTrue(x.HasCurrentMeasurements('Image', 'Feature'))

if __name__ == "__main__":
    unittest.main()
