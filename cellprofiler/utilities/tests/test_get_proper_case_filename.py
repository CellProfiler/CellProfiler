'''test_get_proper_case_filename - test the get_proper_case_filename module

'''
import os
import sys
import tempfile
import unittest

import cellprofiler.utilities.get_proper_case_filename as gpcf

if sys.platform == 'win32':
    class Test_GetProperCaseFilename(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            fd, cls.temp = tempfile.mkstemp()
            os.close(fd)
            
        @classmethod
        def tearDownClass(cls):
            os.remove(cls.temp)

        def test_01_01_constants(self):
            self.assertEqual(gpcf.FILE_ATTRIBUTE_ARCHIVE, 0x20)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_HIDDEN, 0x02)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_NOT_CONTENT_INDEXED, 0x2000)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_OFFLINE, 0x1000)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_READONLY, 0x1)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_SYSTEM, 0x4)
            self.assertEqual(gpcf.FILE_ATTRIBUTE_TEMPORARY, 0x100)
            
        def test_01_02_not_readonly(self):
            attrs = gpcf.get_file_attributes(self.temp)
            self.assertEqual(attrs & gpcf.FILE_ATTRIBUTE_READONLY, 0)
            
        def test_01_03_set_readonly(self):
            attrs = gpcf.get_file_attributes(self.temp)
            gpcf.set_file_attributes(
                self.temp, attrs | gpcf.FILE_ATTRIBUTE_READONLY)
            self.assertRaises(IOError, lambda :open(self.temp, "w"))
            gpcf.set_file_attributes(
                self.temp, attrs & ~ gpcf.FILE_ATTRIBUTE_READONLY)
            
        def test_01_04_read_and_write(self):
            attrs = gpcf.get_file_attributes(self.temp)
            for flag in (gpcf.FILE_ATTRIBUTE_HIDDEN,
                         gpcf.FILE_ATTRIBUTE_READONLY,
                         gpcf.FILE_ATTRIBUTE_TEMPORARY):
                gpcf.set_file_attributes(self.temp, attrs ^ flag)
                new_attrs = gpcf.get_file_attributes(self.temp)
                self.assertEqual(new_attrs, attrs ^ flag)
                gpcf.set_file_attributes(self.temp, attrs)