""" test_Measurements.py - tests for CellProfiler.Measurements
"""
import unittest
import numpy
import cellprofiler.measurements

class TestMeasurements(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.measurements.Measurements()
    
    def test_01_01_image_number_is_zero(self):
        x = cellprofiler.measurements.Measurements()
        self.assertEqual(x.image_set_number,0)
    
    def test_01_01_next_image(self):
        x = cellprofiler.measurements.Measurements()
        x.next_image_set()
        self.assertEqual(x.image_set_number,1)
    
    def test_02_01_add_image_measurement(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))
    
    def test_02_02_add_object_measurement(self):
        x = cellprofiler.measurements.Measurements()
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m)
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m).all)
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Nuclei"))
    
    def test_02_03_add_two_measurements(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m)
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m).all())
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))
    
    def test_02_04_add_two_measurements_to_object(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature1","Value1" )
        x.add_measurement("Image", "Feature2","Value2" )
        self.assertEqual(x.get_current_measurement("Image", "Feature1"),"Value1")
        self.assertEqual(x.get_current_measurement("Image", "Feature2"),"Value2")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature1" in x.get_feature_names("Image"))
        self.assertTrue("Feature2" in x.get_feature_names("Image"))
    
    def test_03_03_MultipleImageSets(self):
        numpy.random.seed(0)
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value1" )
        m1 = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m1)
        x.next_image_set()
        x.add_measurement("Image", "Feature","Value2" )
        m2 = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m2)
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value2")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m2).all())
        for a,b in zip(x.get_all_measurements("Image", "Feature"),["Value1","Value2"]):
            self.assertEqual(a,b)
        for a,b in zip(x.get_all_measurements("Nuclei","Feature"),[m1,m2]):
            self.assertTrue((a==b).all())
    
    def test_04_01_negative_add_twice(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_04_02_negative_add_on_second_pass(self):
        x = cellprofiler.measurements.Measurements()
        x.next_image_set()
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_04_03_negative_add_other_on_second_pass(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature1","Value1" )
        x.next_image_set()
        x.add_measurement("Image", "Feature1","Value1" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature2","Value2")
        
    def test_04_04_negative_add_twice_on_second_pass(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        x.next_image_set()
        x.add_measurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_05_01_test_has_current_measurements(self):
        x = cellprofiler.measurements.Measurements()
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))
                         
    def test_05_02_test_has_current_measurements(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "OtherFeature","Value" )
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))

    def test_05_03_test_has_current_measurements(self):
        x = cellprofiler.measurements.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertTrue(x.has_current_measurements('Image', 'Feature'))

if __name__ == "__main__":
    unittest.main()
