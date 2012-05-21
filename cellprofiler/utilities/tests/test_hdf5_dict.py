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
E = H5DICT.HDF5Dict.encode
D = H5DICT.HDF5Dict.decode


class TestHDF5Dict(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_dict = H5DICT.HDF5Dict(self.temp_filename)
        self.hdf_file = self.hdf_dict.hdf5_file
        
    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        
    def test_01_01_encode_alphanumeric(self):
        r = np.random.RandomState()
        r.seed(101)
        s = r.permutation(np.frombuffer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-", "S1")).tostring()
        self.assertEqual(s, E(s))
        
    def test_01_02_decode_alphanumeric(self):
        r = np.random.RandomState()
        r.seed(102)
        s = r.permutation(np.frombuffer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-", "S1")).tostring()
        self.assertEqual(s, D(s))
        
    def test_01_03_decode_of_encode_is_same(self):
        r = np.random.RandomState()
        r.seed(103)
        s = r.permutation(np.array([chr(x) for x in range(32, 127)])).tostring()
        self.assertEqual(s, D(E(s)))
        
    def test_02_01_get_new_filelist_group(self):
        g = self.hdf_dict.get_filelist_group()
        self.assertIn(H5DICT.FILE_LIST_GROUP, self.hdf_file)
        self.assertIn(H5DICT.DEFAULT_GROUP,
                      self.hdf_file[H5DICT.FILE_LIST_GROUP])
        
    def test_02_02_get_existing_filelist_group(self):
        g1 = self.hdf_dict.get_filelist_group()
        g2 = self.hdf_dict.get_filelist_group()
        self.assertEqual(g1, g2)
        
    def test_03_00_add_no_file(self):
        self.hdf_dict.add_files_to_filelist([])
        
    def test_03_01_add_file(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(["file://foo/bar.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g["file"])
        filenames = H5DICT.VStringArray(g[E("file")][E("//foo")])
        self.assertEqual(len(filenames) ,1)
        self.assertIn("bar.jpg", filenames)
        
    def test_03_02_add_two_files(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
        self.assertEqual(len(filenames) ,2)
        self.assertEqual(filenames[0], "bar.jpg")
        self.assertEqual(filenames[1], "baz.jpg")
        
    def test_03_03_add_two_directories(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://bar/baz.jpg"])
        for subdir, filename in (("//foo", "bar.jpg"),
                                 ("//bar", "baz.jpg")):
            self.assertIn(E("file"), g)
            self.assertIn(E(subdir), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E(subdir)]))
            self.assertEqual(len(filenames) ,1)
            self.assertEqual(filenames[0], filename)
            
    def test_03_04_add_a_file_and_a_file(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg"])
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/baz.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
        self.assertEqual(len(filenames) ,2)
        self.assertEqual(filenames[0], "bar.jpg")
        self.assertEqual(filenames[1], "baz.jpg")
        
    def test_03_05_add_a_file_with_a_stupid_DOS_name(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file:///C:/foo/bar.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("///C:"), g[E("file")])
        self.assertIn(E("foo"), g[E("file")][E("///C:")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("///C:")][E("foo")]))
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], "bar.jpg")
        
    def test_03_06_add_a_file_to_the_base(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(["file://foo.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//")]))
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], "foo.jpg")
        
    def test_03_07_add_a_file_to_the_schema(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(["file:foo.jpg"])
        self.assertIn(E("file"), g)
        filenames = list(H5DICT.VStringArray(g[E("file")]))
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0], "foo.jpg")
        
    def test_04_00_remove_none(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg"])
        self.hdf_dict.remove_files_from_filelist([])
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
        self.assertEqual(len(filenames) ,2)
        self.assertEqual(filenames[0], "bar.jpg")
        self.assertEqual(filenames[1], "baz.jpg")
        
    def test_04_01_remove_file(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg", "file://foo/a.jpg"])
        self.hdf_dict.remove_files_from_filelist(
            ["file://foo/bar.jpg"])
                                                 
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
        self.assertEqual(len(filenames) ,2)
        self.assertEqual(filenames[0], "a.jpg")
        self.assertEqual(filenames[1], "baz.jpg")
        
    def test_04_02_remove_all_files_in_dir(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://bar/baz.jpg"])
        self.hdf_dict.remove_files_from_filelist(
            ["file://foo/bar.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//bar"), g[E("file")])
        self.assertNotIn(E("//foo"), g[E("file")])
        
    def test_04_03_remove_all_files_in_parent(self):
        g = self.hdf_dict.get_filelist_group()
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file:baz.jpg"])
        self.assertTrue(H5DICT.VStringArray.has_vstring_array(g[E("file")]))
        self.hdf_dict.remove_files_from_filelist(
            ["file:baz.jpg"])
        self.assertIn(E("file"), g)
        self.assertIn(E("//foo"), g[E("file")])
        filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
        self.assertEqual(len(filenames), 1)
        self.assertIn("bar.jpg", filenames)
        self.assertFalse(H5DICT.VStringArray.has_vstring_array(g[E("file")]))
        
    def test_05_01_get_filelist(self):
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg"])
        urls = self.hdf_dict.get_filelist()
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], "file://foo/bar.jpg")
        self.assertEqual(urls[1], "file://foo/baz.jpg")
        
    def test_05_02_get_multidir_filelist(self):
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://bar/baz.jpg", "file://foo.jpg"])
        urls = self.hdf_dict.get_filelist()
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls[2], "file://foo/bar.jpg")
        self.assertEqual(urls[1], "file://bar/baz.jpg")
        self.assertEqual(urls[0], "file://foo.jpg")
        
    def test_06_00_walk_empty(self):
        def fn(root, directories, urls):
            raise AssertionError("Whoops! Should never be called")
        self.hdf_dict.walk(fn)
        
    def test_06_01_walk_one(self):
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg"])
        roots = []
        directories = []
        urls = []
        def fn(r, d, u):
            roots.append(r)
            directories.append(d)
            urls.append(u)
        self.hdf_dict.walk(fn)
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[0], "file:")
        self.assertEqual(roots[1], "file://foo/")
        self.assertEqual(len(directories[0]), 1)
        self.assertEqual(directories[0][0], "//foo")
        self.assertEqual(len(directories[1]), 0)
        self.assertEqual(len(urls[0]), 0)
        self.assertEqual(len(urls[1]), 2)
        self.assertEqual(urls[1][0], "bar.jpg")
        self.assertEqual(urls[1][1], "baz.jpg")
        
    def test_06_02_walk_many(self):
        self.hdf_dict.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg",
             "file://foo/bar/baz.jpg", "file://bar/foo.jpg",
             "file://foo/baz/bar.jpg"])
        roots = []
        directories = []
        urls = []
        def fn(r, d, u):
            roots.append(r)
            directories.append(d)
            urls.append(u)
        self.hdf_dict.walk(fn)
        self.assertEqual(len(roots), 5)
        
        self.assertEqual(roots[0], "file:")
        self.assertEqual(len(directories[0]), 2)
        self.assertEqual(directories[0][0], "//bar")
        self.assertEqual(directories[0][1], "//foo")
        self.assertEqual(len(urls[0]), 0)
    
        self.assertEqual(roots[1], "file://bar/")
        self.assertEqual(len(directories[1]), 0)
        self.assertEqual(len(urls[1]), 1)
        self.assertEqual(urls[1][0], "foo.jpg")
        
        self.assertEqual(roots[2], "file://foo/")
        self.assertEqual(len(directories[2]), 2)
        self.assertEqual(directories[2][0], "bar")
        self.assertEqual(directories[2][1], "baz")
        self.assertEqual(len(urls[2]), 2)
        self.assertEqual(urls[2][0], "bar.jpg")
        self.assertEqual(urls[2][1], "baz.jpg")
        
        self.assertEqual(roots[3], "file://foo/bar/")
        self.assertEqual(len(directories[3]), 0)
        self.assertEqual(len(urls[3]), 1)
        self.assertEqual(urls[3][0], "baz.jpg")
        
        self.assertEqual(roots[4], "file://foo/baz/")
        self.assertEqual(len(directories[4]), 0)
        self.assertEqual(len(urls[4]), 1)
        self.assertEqual(urls[4][0], "bar.jpg")
        
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
            
    def test_09_00_sort_none(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a.sort()
        
    def test_09_01_sort_one(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "hello"
        a.sort()
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0], "hello")
        
    def test_09_02_sort(self):
        r = np.random.RandomState()
        r.seed(92)
        lengths = r.randint(3, 10, 100)
        idx = np.hstack([[0], np.cumsum(lengths)])
        chars = r.randint(ord('A'), ord('F')+1, idx[-1]).astype(np.uint8)
        strings = [chars[i:j].tostring() for i,j in zip(idx[:-1], idx[1:])]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(strings)
        a.sort()
        for s0, s1 in zip(sorted(strings), a):
            self.assertEqual(s0, s1)
            
    def test_09_03_sort_none(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(["hello", None, "world", None])
        a.sort()
        self.assertIsNone(a[0])
        self.assertIsNone(a[1])
        self.assertEqual(a[2], "hello")
        self.assertEqual(a[3], "world")
        
            
    def test_10_01_insert(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(["hello", "world"])
        a.insert(2, ".")
        a.insert(1, "brave")
        a.insert(2, "new")
        for s0, s1 in zip(["hello", "brave", "new", "world", "."], a):
            self.assertEqual(s0, s1)
            
    def test_11_00_bisect_left_none(self):
        a = H5DICT.VStringArray(self.hdf_file)
        self.assertEqual(a.bisect_left("foo"), 0)
        a[0] = "foo"
        self.assertEqual(a.bisect_left(None), 0)
        
    def test_11_01_bisect_left(self):
        a = H5DICT.VStringArray(self.hdf_file)
        r = np.random.RandomState()
        r.seed(1100)
        lengths = r.randint(3, 10, 100)
        idx = np.hstack([[0], np.cumsum(lengths)])
        chars = r.randint(ord('A'), ord('F')+1, idx[-1]).astype(np.uint8)
        strings = [chars[i:j].tostring() for i,j in zip(idx[:-1], idx[1:])]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(strings[:50])
        a.sort()
        for s in strings[50:]:
            idx = a.bisect_left(s)
            if idx > 0:
                self.assertLessEqual(a[idx-1], s)
            if idx < len(a):
                self.assertGreaterEqual(a[idx], s)

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
        