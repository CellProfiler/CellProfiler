""" test_Measurements.py - tests for CellProfiler.Measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import base64
import unittest
import numpy as np
import os
import tempfile
import zlib

import cellprofiler.measurements as cpmeas
from cStringIO import StringIO

OBJECT_NAME = "myobjects"

class TestMeasurements(unittest.TestCase):
    def test_00_00_init(self):
        x = cpmeas.Measurements()
    
    def test_01_01_image_number_is_zero(self):
        x = cpmeas.Measurements()
        self.assertEqual(x.image_set_number,1)
    
    def test_01_01_next_image(self):
        x = cpmeas.Measurements()
        x.next_image_set()
        self.assertEqual(x.image_set_number,2)
    
    def test_02_01_add_image_measurement(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))
    
    def test_02_02_add_object_measurement(self):
        x = cpmeas.Measurements()
        np.random.seed(0)
        m = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m)
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m).all)
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Nuclei"))
    
    def test_02_03_add_two_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        np.random.seed(0)
        m = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m)
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m).all())
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))
    
    def test_02_04_add_two_measurements_to_object(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature1","Value1" )
        x.add_measurement("Image", "Feature2","Value2" )
        self.assertEqual(x.get_current_measurement("Image", "Feature1"),"Value1")
        self.assertEqual(x.get_current_measurement("Image", "Feature2"),"Value2")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature1" in x.get_feature_names("Image"))
        self.assertTrue("Feature2" in x.get_feature_names("Image"))
    
    def test_03_03_MultipleImageSets(self):
        np.random.seed(0)
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value1" )
        m1 = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m1)
        x.next_image_set()
        x.add_measurement("Image", "Feature","Value2" )
        m2 = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature",m2)
        self.assertEqual(x.get_current_measurement("Image", "Feature"),"Value2")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature")==m2).all())
        for a,b in zip(x.get_all_measurements("Image", "Feature"),["Value1","Value2"]):
            self.assertEqual(a,b)
        for a,b in zip(x.get_all_measurements("Nuclei","Feature"),[m1,m2]):
            self.assertTrue((a==b).all())
    
    def test_04_01_negative_add_twice(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_04_02_negative_add_on_second_pass(self):
        x = cpmeas.Measurements()
        x.next_image_set()
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_04_03_negative_add_other_on_second_pass(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature1","Value1" )
        x.next_image_set()
        x.add_measurement("Image", "Feature1","Value1" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature2","Value2")
        
    def test_04_04_negative_add_twice_on_second_pass(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        x.next_image_set()
        x.add_measurement("Image", "Feature","Value" )
        self.assertRaises(AssertionError,x.add_measurement,"Image", "Feature","Value")
    
    def test_05_01_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))
                         
    def test_05_02_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "OtherFeature","Value" )
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))

    def test_05_03_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature","Value" )
        self.assertTrue(x.has_current_measurements('Image', 'Feature'))
    
    def test_06_00_00_dont_apply_metadata(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "pre_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_00_01_dont_apply_metadata_with_slash(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "pre\\post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre\\\\post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_01_apply_metadata(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "pre_"+value+"_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"pre_\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_02_apply_metadata_with_slash(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "\\"+value+"_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_03_apply_metadata_with_two_slashes(self):
        '''Regression test of img-1144'''
        x = cpmeas.Measurements()
        plate = "P12345"
        well = "A01"
        expected = "\\"+plate+"\\"+well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\\\g<Plate>\\\g<Well>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_04_apply_metadata_when_user_messes_with_your_head(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = r"\g<Plate>"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\g<Plate>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_06_05_apply_metadata_twice(self):
        '''Regression test of img-1144 (second part)'''
        x = cpmeas.Measurements()
        plate = "P12345"
        well = "A01"
        expected = plate+"_"+well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\g<Plate>_\g<Well>"
        self.assertEqual(x.apply_metadata(pattern), expected)
        
    def test_07_01_copy(self):
        x = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(71)
        areas = [ r.randint(100, 200, size=r.randint(100, 200))
                  for _ in range(12)]
                            
        for i in range(12):
            x.add_measurement(cpmeas.IMAGE, "Metadata_Well", "A%02d" % (i+1),
                              image_set_number = (i+1))
            x.add_measurement(OBJECT_NAME, "AreaShape_Area", areas[i],
                              image_set_number = (i+1))
            
        y = cpmeas.Measurements(copy = x)
        for i in range(12):
            self.assertEqual(
                y.get_measurement(cpmeas.IMAGE, "Metadata_Well", (i+1)),
                "A%02d" % (i+1))
            values = y.get_measurement(OBJECT_NAME, "AreaShape_Area", (i+1))
            np.testing.assert_equal(values, areas[i])

    def test_08_01_load(self):
        data = ('eJzt3M1LFGEcwPFnZtZ2VcSXS29Gewm8BJu9KHgxUtvAl6UX8GDpmmMWauIL'
                'SPQH6M2jl6Bb0qGgCK92M+rWRZAg6BJB4LFL2D7zPM/uztS4smtYu9/P5fH5'
                'zTPP7M7+5pnxML/lZFdPXc2JGiHFYiIiGkS+XW31jb9vto/o1tLtkm7XbBO3'
                'vG3HdLxRzx8cd/N6d7ccvRtgjvO5SrUxgUqU7L6cku2g7pt82rT94/rc9NzC'
                'rDvlTs/Pyb7Jy84ijxuWv0lH9Sd0XhbK36/VqiV/K1Mwf80qu1TlH3dtKn3P'
                'Fbm8TZZ43Bv9A10yg03exZ0SJ6xQYetAKqr6M/r6LrQO7NSp1syDyhJcB1p0'
                'u1LtH+etA/0LU6PurNfvc+fTY+n59HBqMj3v5taHxD6PG1wHWqLFfoPKFrYO'
                'DNaq/qK+vgutA0l9A/A/baNShD3Prtb5x8lrXrb3p8fcRVH686xaB+zsOpCo'
                '3Xu8+Vyx20UesExZIqp+C/2DVGXOqPzTdlTEyVzZ8o+GzImOeCPiej91Ri1L'
                '7Wgex8z6UCvavf0s/X+N7c1ribmJhfHxSVfPJbyVXMbH3HHvhmDiscz8cn/H'
                'trd6dSwisptFk/7l5Zi+nub+4jPJL5hX7fV7jzfr3krjgRy+bFjiSJF5pZ4k'
                'LNufV8H8Mv1cnjhPm3QsooZ70/6eJzMH/E33R95HrT/dRws8+Gw1BwL6CnD8'
                'XQAAAJQJK6+181onrwUAAAAAAAAAAAAAAAAAAAAAAIcr7L3RUyHjY7cCgYiv'
                'AQAAZWjxzve3Vl3mOcCJv/zZsf3FedWWiXauRx/+ENHMxs11S8RjIvF+Y/dk'
                '/c6zjtNr71aXZCmXje3W10d3lp9/04X8hh68+CRr6ix+sB49dj4e4lcCAAAF'
                'hNV1TOi6d6lh1Raq67g1olrqOlamsLqOM8P+cQdd17FwPUJV7+24pXKzITvU'
                '1HkrrS6hyfd91CWceKJjkbzLJL/e3JBXb669yDOhBOsRbg7vOTz7QeKjJR22'
                '7BxCPcLEXR0Lq0c49B/WI5yYCgSoRwgAAFDWqEcIAAAAAAAAAAAAAAAAAAAA'
                'AMC/L+y90Y6Q8WdC6hFSVwAAgPJ19cpAr/fOoL7hm/cHo7pNnT3Xev7CRWH/'
                'FpfhS9n3CXNxGW7Lzr9W/5c+OAAAAAAA2LdfZABKkA==')
        data = zlib.decompress(base64.b64decode(data))
        fd, filename = tempfile.mkstemp('.h5')
        try:
            f = os.fdopen(fd, "wb")
            f.write(data)
            f.close()
            m = cpmeas.load_measurements(filename)
            for i in range(1, 4):
                self.assertEqual(m.get_measurement(
                    cpmeas.IMAGE, 'ImageNumber', i), i)
            for i, plate in enumerate(('P-12345', 'P-23456', 'P-34567')):
                self.assertEqual(m.get_measurement(
                    cpmeas.IMAGE, 'Metadata_Plate', i+1), plate)
        finally:
            try:
                os.unlink(filename)
            except:
                print "Failed to remove file %s" % filename
        
        

if __name__ == "__main__":
    unittest.main()
