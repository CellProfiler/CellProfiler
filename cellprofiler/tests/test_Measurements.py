""" test_Measurements.py - tests for CellProfiler.Measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import unittest
import numpy
import cellprofiler.measurements

class TestMeasurements(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.measurements.Measurements()
    
    def test_01_01_image_number_is_zero(self):
        x = cellprofiler.measurements.Measurements()
        self.assertEqual(x.image_set_number,1)
    
    def test_01_01_next_image(self):
        x = cellprofiler.measurements.Measurements()
        x.next_image_set()
        self.assertEqual(x.image_set_number,2)
    
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
    
    def test_06_00_00_dont_apply_metadata(self):
        x = cellprofiler.measurements.Measurements()
        value = "P12345"
        expected = "pre_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_00_01_dont_apply_metadata_with_slash(self):
        x = cellprofiler.measurements.Measurements()
        value = "P12345"
        expected = "pre\\post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre\\\\post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_01_apply_metadata(self):
        x = cellprofiler.measurements.Measurements()
        value = "P12345"
        expected = "pre_"+value+"_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"pre_\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_02_apply_metadata_with_slash(self):
        x = cellprofiler.measurements.Measurements()
        value = "P12345"
        expected = "\\"+value+"_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_03_apply_metadata_with_two_slashes(self):
        '''Regression test of img-1144'''
        x = cellprofiler.measurements.Measurements()
        plate = "P12345"
        well = "A01"
        expected = "\\"+plate+"\\"+well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\\\g<Plate>\\\g<Well>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_04_apply_metadata_when_user_messes_with_your_head(self):
        x = cellprofiler.measurements.Measurements()
        value = "P12345"
        expected = r"\g<Plate>"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\g<Plate>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_05_apply_metadata_twice(self):
        '''Regression test of img-1144 (second part)'''
        x = cellprofiler.measurements.Measurements()
        plate = "P12345"
        well = "A01"
        expected = plate+"_"+well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\g<Plate>_\g<Well>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        

if __name__ == "__main__":
    unittest.main()
