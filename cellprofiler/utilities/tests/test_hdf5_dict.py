'''test_hdf5_dict - test the hdf5_dict module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
Copyright (c) 2011 Institut Curie
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import h5py
import numpy as np
import os
import tempfile
import unittest

import cellprofiler.utilities.hdf5_dict as H5DICT

class TestHDFCSV(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename)
        
    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        
    def test_01_01_init(self):
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        self.assertEqual(len(csv), 0)
        
    def test_01_02_init_twice(self):
        H5DICT.HDFCSV(self.hdf_file, "csv")
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        self.assertEqual(len(csv), 0)
    
    def test_02_01_add_column_empty(self):
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        column = csv.add_column("kolumn")
        self.assertTrue("kolumn" in csv.top_level_group)
        
    def test_02_02_add_column_with_data(self):
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        strings = ["foo", "bar"]
        column = csv.add_column("kolumn", strings)
        self.assertEqual(len(column), len(strings))
        for s0, s1 in zip(strings, column):
            self.assertEqual(s0, s1)
            
    def test_03_01_getitem(self):
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        strings = ["foo", "bar"]
        csv.add_column("kolumn", strings)
        column = csv["kolumn"]
        self.assertEqual(len(column), len(strings))
        for s0, s1 in zip(strings, column):
            self.assertEqual(s0, s1)
            
    def test_04_01_set_all(self):
        d = { "random":["foo", "bar", "baz"],
              "fruits":["lemon", "cherry", "orange", "apple"],
              "rocks":["granite", "basalt", "limestone"] }
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        csv.set_all(d)
        for key, strings in d.iteritems():
            column = csv[key]
            self.assertEqual(len(column), len(strings))
            for s0, s1 in zip(strings, column):
                self.assertEqual(s0, s1)
                
    def test_05_01_get_column_names_etc(self):
        d = { "random":["foo", "bar", "baz"],
              "fruits":["lemon", "cherry", "orange", "apple"],
              "rocks":["granite", "basalt", "limestone"] }
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        csv.set_all(d)
        for key in d:
            self.assertIn(key, csv.get_column_names())
            self.assertIn(key, csv)
            self.assertIn(key, csv.keys())
            self.assertIn(key, csv.iterkeys())
            
class TestVStringArray(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename)
        
    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        
    def test_01_01_init(self):
        H5DICT.VStringArray(self.hdf_file)
        self.assertIn("index", self.hdf_file)
        self.assertIn("data", self.hdf_file)
        self.assertEqual(self.hdf_file["index"].shape[0], 0)
        
    def test_01_02_init_existing(self):
        H5DICT.VStringArray(self.hdf_file)
        H5DICT.VStringArray(self.hdf_file)
        self.assertIn("index", self.hdf_file)
        self.assertIn("data", self.hdf_file)
        self.assertEqual(self.hdf_file["index"].shape[0], 0)
        
    def test_02_01_set_one(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 0], 0)
        self.assertEqual(self.hdf_file["index"][0, 1], 3)
        self.assertEqual(self.hdf_file["data"][:].tostring(), "foo")
        
    def test_02_02_set_none(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = None
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 0], a.VS_NULL)
        self.assertEqual(self.hdf_file["index"][0, 1], 0)
        
    def test_02_03_set_and_set_shorter(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[0] = "hu"
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 1] - 
                         self.hdf_file["index"][0, 0], 2)
        self.assertEqual(self.hdf_file["data"][
            self.hdf_file["index"][0, 0]:
            self.hdf_file["index"][0, 1]].tostring(), "hu")
        
    def test_02_04_set_and_set_longer(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[0] = "whoops"
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 1] - 
                         self.hdf_file["index"][0, 0], 6)
        self.assertEqual(self.hdf_file["data"][
            self.hdf_file["index"][0, 0]:
            self.hdf_file["index"][0, 1]].tostring(), "whoops")
        
    def test_02_05_set_empty(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = ""
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 0], 0)
        self.assertEqual(self.hdf_file["index"][0, 1], 0)
        
    def test_03_01_len(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[1] = "bar"
        self.assertEqual(len(a), 2)
        
    def test_04_01_get(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        self.assertEqual(a[0], "foo")
        
    def test_04_02_get_null(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = None
        self.assertIsNone(a[0])
        
    def test_04_03_get_zero_len(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = ""
        self.assertEqual(a[0], "")
        
    def test_04_04_get_unicode(self):
        s = u"\u03b4x"
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = s
        self.assertEqual(a[0], s)
        
    def test_04_05_get_two(self):
        a = H5DICT.VStringArray(self.hdf_file)
        strings = ["foo", "bar"]
        for i, s in enumerate(strings):
            a[i] = s
        for i, s in enumerate(strings):
            self.assertEqual(s, a[i])
        
    def test_05_01_del_one(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        del a[0]
        self.assertEqual(len(a), 0)
        
    def test_05_02_del_last(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[1] = "bar"
        del a[1]
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0], "foo")
        
    def test_05_03_del_not_last(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[1] = "bar"
        del a[0]
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0], "bar")
        
    def test_06_01_iter(self):
        a = H5DICT.VStringArray(self.hdf_file)
        strings = ["foo", "bar"]
        for i, s in enumerate(strings):
            a[i] = s
        for s0, s1 in zip(a, strings):
            self.assertEqual(s0, s1)
            
    def test_07_00_set_empty(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all([])
        self.assertEqual(len(a), 0)
        
    def test_07_00_00_set_empty_dirty(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a.set_all([])
        self.assertEqual(len(a), 0)
        
    def test_07_01_set_all(self):
        a = H5DICT.VStringArray(self.hdf_file)
        strings = ["foo", "bar"]
        a.set_all(strings)
        for s0, s1 in zip(a, strings):
            self.assertEqual(s0, s1)
            
    def test_07_02_set_all_dirty(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(["blah", "blah", "blah", "blah"])
        strings = ["foo", "bar"]
        a.set_all(strings)
        self.assertEqual(len(a), len(strings))
        for s0, s1 in zip(a, strings):
            self.assertEqual(s0, s1)
            
    def test_07_03_set_all_with_nulls(self):
        a = H5DICT.VStringArray(self.hdf_file)
        strings = ["foo", None, "bar"]
        a.set_all(strings)
        self.assertEqual(len(a), len(strings))
        for s0, s1 in zip(a, strings):
            self.assertEqual(s0, s1)
            
    def test_08_01_reopen(self):
        strings = ["foo", "bar"]
        a = H5DICT.VStringArray(self.hdf_file)
        strings = ["foo", "bar"]
        a.set_all(strings)
        
        b = H5DICT.VStringArray(self.hdf_file)
        for s0, s1 in zip(b, strings):
            self.assertEqual(s0, s1)
        

class TestStringReference(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename)
        
    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)

    def test_01_01_init(self):
        sr = H5DICT.StringReferencer(self.hdf_file.create_group("test"))
        
    def test_01_02_insert_one(self):
        sr = H5DICT.StringReferencer(self.hdf_file.create_group("test"))
        s = "Hello"
        result = sr.get_string_refs([s])
        self.assertEqual(len(result), 1)
        rstrings = sr.get_strings(result)
        self.assertEqual(len(rstrings), 1)
        self.assertEqual(s, rstrings[0])
        
    def test_01_03_insert_two(self):
        for s1, s2 in (("foo", "bar"), 
                       ("bar", "foo"),
                       ("foo", "fooo"),
                       ("fo", "foo")):
            for how in ("together", "separate"):
                sr = H5DICT.StringReferencer(self.hdf_file.create_group(
                    "test"+s1+s2+how))
                if how == "together":
                    result = sr.get_string_refs([s1, s2])
                else:
                    result = [sr.get_string_refs([x])[0] for x in (s1, s2)]
                self.assertEqual(len(result), 2)
                rstrings = sr.get_strings(result)
                self.assertEqual(len(rstrings), 2)
                self.assertEqual(rstrings[0], s1)
                self.assertEqual(rstrings[1], s2)
    
            
    def test_01_04_duplicate(self):
        sr = H5DICT.StringReferencer(self.hdf_file.create_group("test"))
        strings = ["foo", "bar", "foo"]
        result = sr.get_string_refs(strings)
        self.assertEqual(len(result), len(strings))
        self.assertTrue(result[0] == result[2])
        rstrings = sr.get_strings(result)
        self.assertTrue(all([s == r for s, r in zip(strings, rstrings)]))
        
    def test_01_05_unicode(self):
        sr = H5DICT.StringReferencer(self.hdf_file.create_group("test"))
        s = u'\u03c0r\u00b2'
        result = sr.get_string_refs([s])
        self.assertEqual(len(result), 1)
        rstrings = sr.get_strings(result)
        self.assertEqual(len(rstrings), 1)
        self.assertEqual(s, rstrings[0])

    def test_01_06_two_tier(self):
        r = np.random.RandomState()
        r.seed(16)
        for i in range(100):
            sr = H5DICT.StringReferencer(
                self.hdf_file.create_group("test%d"% i), 4)
            strings = [chr(65+ r.randint(0, 26)) for j in range(10)]
            result = sr.get_string_refs([strings])
            self.assertEqual(len(result), 10)
            rstrings = sr.get_strings(result)
            self.assertSequenceEqual(strings, rstrings)
            
    def test_01_07_three_tier(self):
        r = np.random.RandomState()
        r.seed(16)
        for i in range(100):
            sr = H5DICT.StringReferencer(
                self.hdf_file.create_group("test%d"% i), 4)
            strings = [chr(65+ r.randint(0, 26))+chr(65 + r.randint(0,26))
                       for j in range(50)]
            result = sr.get_string_refs([strings])
            self.assertEqual(len(result), 50)
            rstrings = sr.get_strings(result)
            self.assertSequenceEqual(strings, rstrings)
        