'''test_hdf5_dict - test the hdf5_dict module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
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
import sys

import cellprofiler.utilities.hdf5_dict as H5DICT
E = H5DICT.HDF5FileList.encode
D = H5DICT.HDF5FileList.decode

OBJECT_NAME = "objectname"
FEATURE_NAME = "featurename"
ALT_FEATURE_NAME = "featurename2"

class TestHDF5Dict(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename)
        
    def tearDown(self):
        self.hdf5_dict.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        self.assertFalse(os.path.exists(self.temp_filename),
                         "If the file can't be removed, it's a bug. "
                         "Clean up your trash: %s" % self.temp_filename)
        
    def test_00_00_init(self):
        # Test setup and teardown - create HDF5 file / close
        pass
    
    def test_01_01_read_write_ints(self):
        r = np.random.RandomState()
        r.seed(101)
        data = r.randint(0, 100, 100)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = data
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        np.testing.assert_array_equal(data, result)
        
    def test_01_02_read_write_floats(self):
        r = np.random.RandomState()
        r.seed(102)
        data = r.uniform(size=100)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = data
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        np.testing.assert_array_equal(data, result)
        
    def test_01_03_read_write_strings(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        
    def test_02_01_read_write_twice(self):
        r = np.random.RandomState()
        r.seed(201)
        data = r.randint(0, 100, (10,100))
        p = r.permutation(data.shape[0])
        for i, d in enumerate(data[p, :]):
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, p[i]+1] = d
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, i+1]
            np.testing.assert_array_equal(d, result)
        
    def test_02_02_read_none(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = "Hello"
        self.assertIsNone(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1])
        
    def test_02_03_write_numeric_imagesets(self):
        for dtype, code, ftr in ((int, 'i', FEATURE_NAME), 
                                 (float, 'f', ALT_FEATURE_NAME)):
            data = [
                np.zeros(0, dtype),
                np.array([1, 10, 100], dtype), 
                np.array([2, 20], dtype),
                np.array([3, 30, 300, 3000], dtype)]
            self.hdf5_dict[OBJECT_NAME, ftr, [1, 3]] = data[::2]
            self.hdf5_dict[OBJECT_NAME, ftr, [2, 4]] = data[1::2]
            d2 = self.hdf5_dict[OBJECT_NAME, ftr, [1, 2, 3, 4]]
            self.assertEqual(len(d2), 4)
            self.assertIsNone(d2[0])
            self.assertTrue(all([d.dtype.kind == code for d in d2[1:]]))
            self.assertTrue(all([np.all(d == e) for d, e in zip(d2[1:], data[1:])]))
        
    def test_02_04_write_string_imagesets(self):
        data = [ None, "foo", "bar", None, "baz"]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [1, 2] ] = data[:2]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [3, 4, 5] ] = data[2:]
        d2 = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [1,2,3,4,5]]
        self.assertIsNone(d2[0])
        self.assertIsNone(d2[3])
        for i in (1, 2, 4):
            self.assertEqual(data[i], d2[i][0])
            
    def test_02_05_00_upgrade_none(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.zeros(0)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = np.arange(5)
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2].tolist(), range(5))
        
    def test_02_05_01_upgrade_none_multiple(self):
        # Regression test of issue #1011
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.zeros(0)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, np.arange(2,4)] = [
            np.zeros(1), np.zeros(0)]
        
    def test_02_06_upgrade_int_to_float(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.arange(5)
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].dtype.kind, "i")
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = np.arange(5).astype(float) / 2.0
        
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].dtype.kind, "f")
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].tolist(), range(5))
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2].tolist(),
            map(lambda x: float(x)/2, range(5)))
        
    def test_02_07_upgrade_float_to_string(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = [2.5]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = ["alfalfa"]
        data = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        self.assertIsInstance(data[0], basestring)
        
    def test_02_08_write_twice(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2)] = [1.2, 3.4]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (2, 3)] = [5.6, 7.8]
        
        data = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2, 3)]
        self.assertEqual(tuple(data), (1.2, 5.6, 7.8))
        index = self.hdf5_dict.top_group[OBJECT_NAME][FEATURE_NAME][H5DICT.INDEX]
        self.assertEqual(index.shape[0], 3)
        
    def test_02_09_write_with_dtype(self):
        self.hdf5_dict[
            OBJECT_NAME, FEATURE_NAME, 1, np.uint8] = np.zeros(5, np.uint16)
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        self.assertEqual(len(result), 5)
        self.assertEqual(result.dtype, np.uint8)
        
    def test_02_10_write_with_dtype_and_nulls(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2), np.uint8] = \
            [ None, np.zeros(5, np.uint8)]
        self.assertIsNone(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1])
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2]
        self.assertEqual(len(result), 5)
        self.assertEqual(result.dtype, np.uint8)
        
    def test_02_11_write_empty_dtype(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1, np.uint8] = None
        self.assertEqual(
            self.hdf5_dict.get_feature_dtype(OBJECT_NAME, FEATURE_NAME),
            np.uint8)
        
    def test_03_01_add_all(self):
        r = np.random.RandomState()
        r.seed(301)
        data = r.randint(0, 100, (10,100))
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data)
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, i+1]
            np.testing.assert_array_equal(d, result)

    def test_03_02_add_all_out_of_order(self):        
        r = np.random.RandomState()
        r.seed(302)
        data = r.randint(0, 100, (10,100))
        p = r.permutation(data.shape[0]) + 1
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data, p)
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, p[i]]
            np.testing.assert_array_equal(d, result)
            
    def test_03_03_add_all_with_dtype(self):
        r = np.random.RandomState()
        r.seed(303)
        data = [ [], r.randint(0, 255, 10).astype(np.uint8)]
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data,
                               idxs = np.arange(1, len(data)+1),
                               data_type = np.uint8)
        self.assertIsNone(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1])
        m2 = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2]
        self.assertEqual(m2.dtype, np.uint8)
        np.testing.assert_array_equal(m2, data[1])

    def test_04_01_reopen_read_only(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
        self.hdf5_dict.close()
    
        self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename, mode="r")
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        with self.assertRaises(Exception):
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = "World"
            
    def test_04_02_reopen_with_write(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
        self.hdf5_dict.close()
        self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename, mode="r+")
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = "World"
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2], "World")
        self.hdf5_dict.close()
        self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename, mode="a")
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 3] = "Append"
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 3], "Append")
        
    def test_05_01_in_memory(self):
        if sys.platform == "darwin":
            self.assertRaises(NotImplementedError, H5DICT.HDF5Dict, None)
        else:
            hdf5_dict = H5DICT.HDF5Dict(None)
            hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
            self.assertEqual(hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        
    def test_06_01_reorder(self):
        r = np.random.RandomState()
        r.seed(601)
        values = ["v%d" % i for i in range(1, 11)]
        for idx in r.permutation(len(values)):
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, idx+1] = values[idx]
        new_image_numbers = np.hstack(([0], r.permutation(len(values)) + 1))
        self.hdf5_dict.reorder(OBJECT_NAME, FEATURE_NAME, new_image_numbers)
        for reopen in (False, True):
            if reopen:
                self.hdf5_dict.close()
                self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename, mode="r")
            for idx, image_number in enumerate(new_image_numbers[1:]):
                self.assertEqual(
                    values[idx], 
                    self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, image_number])
                
    def test_07_01_file_contents(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
        contents = self.hdf5_dict.file_contents()
        fd, filename = tempfile.mkstemp(".h5")
        if sys.platform.startswith("win"):
            import msvcrt
            msvcrt.setmode(fd, os.O_BINARY)
        os.write(fd, contents)
        os.close(fd)
        h5copy = None
        try:
            h5copy = H5DICT.HDF5Dict(filename, mode = "r")
            self.assertTrue(h5copy.has_feature(OBJECT_NAME, FEATURE_NAME))
            self.assertEqual(h5copy[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        finally:
            if h5copy is not None:
                h5copy.close()
            os.unlink(filename)
            
    def test_08_01_copy(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = "Hello"
        temp_fd, temp_filename = tempfile.mkstemp(".h5")
        try:
            h5copy = H5DICT.HDF5Dict(temp_filename,
                                     copy = self.hdf5_dict.top_group)
            
            self.assertTrue(h5copy.has_object(OBJECT_NAME))
            self.assertTrue(h5copy.has_feature(OBJECT_NAME, FEATURE_NAME))
            self.assertEqual(h5copy[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        finally:
            h5copy.close()
            os.close(temp_fd)
            os.remove(temp_filename)
            self.assertFalse(os.path.exists(temp_filename),
                             "If the file can't be removed, it's a bug. "
                             "Clean up your trash: %s" % temp_filename)
            
    def test_09_01_delete_imageset(self):
        # Delete an image set's measurements from one image set
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2, 3)] = ["A", "B", "C"]
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2], "B")
        del self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2]
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2], None)
        
    def test_09_02_delete_measurement(self):
        # delete an entire measurement
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2, 3)] = ["A", "B", "C"]
        self.assertTrue(self.hdf5_dict.has_feature(OBJECT_NAME, FEATURE_NAME))
        del self.hdf5_dict[OBJECT_NAME, FEATURE_NAME]
        self.assertFalse(self.hdf5_dict.has_feature(OBJECT_NAME, FEATURE_NAME))
        
class TestHDF5FileList(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename)
        self.filelist = H5DICT.HDF5FileList(self.hdf_file)
        
        self.temp_fd_nocache, self.temp_filename_nocache = tempfile.mkstemp(".h5")
        self.hdf_file_nocache = h5py.File(self.temp_filename_nocache)
        self.filelist_nocache = H5DICT.HDF5FileList(self.hdf_file_nocache)
        
        self.temp_fd_empty, self.temp_filename_empty = tempfile.mkstemp(".h5")
        self.hdf_file_empty = h5py.File(self.temp_filename_empty)
        
        
    def tearDown(self):
        if isinstance(self.hdf_file, h5py.File):
            self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        if isinstance(self.hdf_file_nocache, h5py.File):
            self.hdf_file_nocache.close()
        os.close(self.temp_fd_nocache)
        os.remove(self.temp_filename_nocache)
        if isinstance(self.hdf_file_empty, h5py.File):
            self.hdf_file_empty.close()
        os.close(self.temp_fd_empty)
        os.remove(self.temp_filename_empty)
        
    def test_01_01_encode_alphanumeric(self):
        r = np.random.RandomState()
        r.seed(101)
        s = r.permutation(np.frombuffer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-+.%=", "S1")).tostring()
        self.assertEqual(s, E(s))
        
    def test_01_02_decode_alphanumeric(self):
        r = np.random.RandomState()
        r.seed(102)
        s = r.permutation(np.frombuffer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-+.%=", "S1")).tostring()
        self.assertEqual(s, D(s))
        
    def test_01_03_decode_of_encode_is_same(self):
        r = np.random.RandomState()
        r.seed(103)
        s = r.permutation(np.array([chr(x) for x in range(32, 127)])).tostring()
        self.assertEqual(s, D(E(s)))
        
    def test_02_01_get_new_filelist_group(self):
        g = self.filelist.get_filelist_group()
        self.assertIn(H5DICT.FILE_LIST_GROUP, self.hdf_file)
        self.assertIn(H5DICT.DEFAULT_GROUP,
                      self.hdf_file[H5DICT.FILE_LIST_GROUP])
        
    def test_02_02_get_existing_filelist_group(self):
        g1 = self.filelist.get_filelist_group()
        g2 = self.filelist.get_filelist_group()
        self.assertEqual(g1, g2)
        
    def test_02_03_has_file_list(self):
        self.assertTrue(H5DICT.HDF5FileList.has_file_list(self.hdf_file))
        
    def test_02_04_no_file_list(self):
        self.assertFalse(H5DICT.HDF5FileList.has_file_list(self.hdf_file_empty))
        
    def test_02_05_copy(self):
        url = "file://foo/bar.jpg"
        metadata = "fakemetadata"
        self.filelist.add_files_to_filelist([url])
        self.filelist.add_metadata(url, metadata)
        H5DICT.HDF5FileList.copy(self.hdf_file, self.hdf_file_empty)
        
        dest_filelist = H5DICT.HDF5FileList(self.hdf_file_empty)
        self.assertSequenceEqual([url], dest_filelist.get_filelist())
        self.assertEqual(metadata, dest_filelist.get_metadata(url))
        
    def test_03_00_add_no_file(self):
        self.filelist.add_files_to_filelist([])
        
    def test_03_01_add_file(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(["file://foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g["file"])
            filenames = H5DICT.VStringArray(g[E("file")][E("//foo")])
            self.assertEqual(len(filenames) ,1)
            self.assertIn("bar.jpg", filenames)
        
    def test_03_02_add_two_files(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            if not cache:
                filelist.clear_cache()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg"])
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
            self.assertEqual(len(filenames) ,2)
            self.assertEqual(filenames[0], "bar.jpg")
            self.assertEqual(filenames[1], "baz.jpg")
        
    def test_03_03_add_two_directories(self):
        g = self.filelist.get_filelist_group()
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://bar/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            for subdir, filename in (("//foo", "bar.jpg"),
                                     ("//bar", "baz.jpg")):
                self.assertIn(E("file"), g)
                self.assertIn(E(subdir), g[E("file")])
                filenames = list(H5DICT.VStringArray(g[E("file")][E(subdir)]))
                self.assertEqual(len(filenames) ,1)
                self.assertEqual(filenames[0], filename)
            
    def test_03_04_add_a_file_and_a_file(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            filelist.add_files_to_filelist(
                ["file://foo/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
            self.assertEqual(len(filenames) ,2)
            self.assertEqual(filenames[0], "bar.jpg")
            self.assertEqual(filenames[1], "baz.jpg")
        
    def test_03_05_add_a_file_with_a_stupid_DOS_name(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file:///C:/foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("///C:"), g[E("file")])
            self.assertIn(E("foo"), g[E("file")][E("///C:")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("///C:")][E("foo")]))
            self.assertEqual(len(filenames), 1)
            self.assertEqual(filenames[0], "bar.jpg")
        
    def test_03_06_add_a_file_to_the_base(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(["file://foo.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("//"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//")]))
            self.assertEqual(len(filenames), 1)
            self.assertEqual(filenames[0], "foo.jpg")
        
    def test_03_07_add_a_file_to_the_schema(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(["file:foo.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            filenames = list(H5DICT.VStringArray(g[E("file")]))
            self.assertEqual(len(filenames), 1)
            self.assertEqual(filenames[0], "foo.jpg")
        
    def test_03_08_what_if_the_user_has_a_directory_named_index(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            #
            # Another in the endless progression of disgusting corner cases
            #
            filelist.add_files_to_filelist(["file://foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            filelist.add_files_to_filelist(
                [ "file://foo/index/baz.jpg",
                  "file://foo/data/xyz.jpg"])
            if not cache:
                filelist.clear_cache()
            result = filelist.get_filelist()
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], "file://foo/bar.jpg")
            self.assertEqual(result[1], "file://foo/data/xyz.jpg")
            self.assertEqual(result[2], "file://foo/index/baz.jpg")
            
            filelist.add_files_to_filelist(
                ["file://bar/index/baz.jpg",
                 "file://bar/data/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            filelist.add_files_to_filelist(["file://bar/baz.jpg"])
            if not cache:
                filelist.clear_cache()
        
    def test_04_00_remove_none(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            self.filelist.remove_files_from_filelist([])
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
            self.assertEqual(len(filenames) ,2)
            self.assertEqual(filenames[0], "bar.jpg")
            self.assertEqual(filenames[1], "baz.jpg")
        
    def test_04_01_remove_file(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg", "file://foo/a.jpg"])
            if not cache:
                filelist.clear_cache()
            filelist.remove_files_from_filelist(
                ["file://foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
                                                     
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
            self.assertEqual(len(filenames) ,2)
            self.assertEqual(filenames[0], "a.jpg")
            self.assertEqual(filenames[1], "baz.jpg")
        
    def test_04_02_remove_all_files_in_dir(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://bar/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            filelist.remove_files_from_filelist(
                ["file://foo/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("//bar"), g[E("file")])
            self.assertNotIn(E("//foo"), g[E("file")])
        
    def test_04_03_remove_all_files_in_parent(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            g = filelist.get_filelist_group()
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file:baz.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertTrue(H5DICT.VStringArray.has_vstring_array(g[E("file")]))
            filelist.remove_files_from_filelist(
                ["file:baz.jpg"])
            if not cache:
                filelist.clear_cache()
            self.assertIn(E("file"), g)
            self.assertIn(E("//foo"), g[E("file")])
            filenames = list(H5DICT.VStringArray(g[E("file")][E("//foo")]))
            self.assertEqual(len(filenames), 1)
            self.assertIn("bar.jpg", filenames)
            self.assertFalse(H5DICT.VStringArray.has_vstring_array(g[E("file")]))
        
    def test_05_01_get_filelist(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            self.filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            urls = self.filelist.get_filelist()
            self.assertEqual(len(urls), 2)
            self.assertEqual(urls[0], "file://foo/bar.jpg")
            self.assertEqual(urls[1], "file://foo/baz.jpg")
        
    def test_05_02_get_multidir_filelist(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://bar/baz.jpg", "file://foo.jpg"])
            if not cache:
                filelist.clear_cache()
            urls = filelist.get_filelist()
            self.assertEqual(len(urls), 3)
            self.assertEqual(urls[2], "file://foo/bar.jpg")
            self.assertEqual(urls[1], "file://bar/baz.jpg")
            self.assertEqual(urls[0], "file://foo.jpg")
            
    def test_05_03_get_sub_filelist(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://bar/baz.jpg", "file://foo/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            urls = filelist.get_filelist("file://foo")
            self.assertEqual(len(urls), 2)
            self.assertEqual(urls[0], "file://foo/bar.jpg")
            self.assertEqual(urls[1], "file://foo/baz.jpg")
        
    def test_06_00_walk_empty(self):
        def fn(root, directories, urls):
            raise AssertionError("Whoops! Should never be called")
        self.filelist.walk(fn)
        
    def test_06_01_walk_one(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            roots = []
            directories = []
            urls = []
            def fn(r, d, u):
                roots.append(r)
                directories.append(d)
                urls.append(u)
            self.filelist.walk(fn)
            self.assertEqual(len(roots), 2)
            self.assertEqual(roots[0], "file:")
            self.assertEqual(roots[1], "file://foo")
            self.assertEqual(len(directories[0]), 1)
            self.assertEqual(directories[0][0], "//foo")
            self.assertEqual(len(directories[1]), 0)
            self.assertEqual(len(urls[0]), 0)
            self.assertEqual(len(urls[1]), 2)
            self.assertEqual(urls[1][0], "bar.jpg")
            self.assertEqual(urls[1][1], "baz.jpg")
        
    def test_06_02_walk_many(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(
                ["file://foo/bar.jpg", "file://foo/baz.jpg",
                 "file://foo/bar/baz.jpg", "file://bar/foo.jpg",
                 "file://foo/baz/bar.jpg"])
            if not cache:
                filelist.clear_cache()
            roots = []
            directories = []
            urls = []
            def fn(r, d, u):
                roots.append(r)
                directories.append(d)
                urls.append(u)
            self.filelist.walk(fn)
            self.assertEqual(len(roots), 5)
            
            self.assertEqual(roots[0], "file:")
            self.assertEqual(len(directories[0]), 2)
            self.assertEqual(directories[0][0], "//bar")
            self.assertEqual(directories[0][1], "//foo")
            self.assertEqual(len(urls[0]), 0)
        
            self.assertEqual(roots[1], "file://bar")
            self.assertEqual(len(directories[1]), 0)
            self.assertEqual(len(urls[1]), 1)
            self.assertEqual(urls[1][0], "foo.jpg")
            
            self.assertEqual(roots[2], "file://foo")
            self.assertEqual(len(directories[2]), 2)
            self.assertEqual(directories[2][0], "bar")
            self.assertEqual(directories[2][1], "baz")
            self.assertEqual(len(urls[2]), 2)
            self.assertEqual(urls[2][0], "bar.jpg")
            self.assertEqual(urls[2][1], "baz.jpg")
            
            self.assertEqual(roots[3], "file://foo/bar")
            self.assertEqual(len(directories[3]), 0)
            self.assertEqual(len(urls[3]), 1)
            self.assertEqual(urls[3][0], "baz.jpg")
            
            self.assertEqual(roots[4], "file://foo/baz")
            self.assertEqual(len(directories[4]), 0)
            self.assertEqual(len(urls[4]), 1)
            self.assertEqual(urls[4][0], "bar.jpg")
        
    def test_07_00_list_files_in_empty_dir(self):
        self.assertEqual(len(self.filelist.list_files("file://foo")), 0)
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        self.assertEqual(len(self.filelist.list_files("file://foo")), 0)
        self.assertEqual(len(self.filelist.list_files("file://foo/baz")), 0)
        
    def test_07_01_list_files(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
            if not cache:
                filelist.clear_cache()
            result = filelist.list_files("file://foo/bar")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], "baz.jpg")
        
    def test_08_00_list_directories_in_empty_dir(self):
        self.assertEqual(len(self.filelist.list_directories("file://foo")), 0)
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        self.assertEqual(len(self.filelist.list_directories("file://bar")), 0)
        self.assertEqual(len(self.filelist.list_directories("file://foo/bar")), 0)
        
    def test_08_01_list_directories(self):
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        result = self.filelist.list_directories("file://foo")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "bar")
        
    def test_09_00_get_type_none(self):
        self.assertEqual(self.filelist.get_type("file://foo/bar.jpg"),
                         self.filelist.TYPE_NONE)
        
    def test_09_01_get_type_file(self):
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            urls = [
                "file://foo/bar/baz.jpg",
                "file://foo/bar/foo.jpg",
                "file://foo/bar/abc.jpg"]
            filelist.add_files_to_filelist(urls)
            if not cache:
                filelist.clear_cache()
            for url in urls:
                self.assertEqual(filelist.get_type(url), 
                                 filelist.TYPE_FILE)
            
    def test_09_02_get_type_dir(self):
        urls = [
            "file://foo.jpg",
            "file://foo/bar.jpg",
            "file://foo/bar/baz.jpg"]
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(urls)
            if not cache:
                filelist.clear_cache()
            for url in ["file://", "file://foo", "file://foo/bar"] :
                self.assertEqual(filelist.get_type(url),
                                 filelist.TYPE_DIRECTORY)
                
    def test_10_01_get_no_metadata(self):
        urls = [
            "file://foo.jpg",
            "file://foo/bar.jpg",
            "file://foo/bar/baz.jpg"]
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(urls)
            if not cache:
                filelist.clear_cache()
            for url in urls:
                self.assertIsNone(filelist.get_metadata(url))
                
    def test_10_02_get_metadata(self):
        urls = [
            "file://foo.jpg",
            "file://foo/bar.jpg",
            "file://foo/bar/baz.jpg"]
        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            return r.randint(ord(" "), ord("~"), 300).astype(np.uint8).tostring()
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(urls)
            for url in urls:
                filelist.add_metadata(url, fn_metadata(url))
            if not cache:
                filelist.clear_cache()
            for url in urls:
                expected = fn_metadata(url)
                self.assertEqual(expected, filelist.get_metadata(url))
                
    def test_10_03_get_metadata_after_insert(self):
        urls = [
            "file://foo/foo.jpg",
            "file://foo/bar.jpg",
            "file://foo/baz.jpg"]
        extend = ["file://foo/pleasework.jpg",
                  "file://foo/beesaregreat.jpg"]
        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            return r.randint(ord(" "), ord("~"), 300).astype(np.uint8).tostring()
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(urls)
            for url in urls:
                filelist.add_metadata(url, fn_metadata(url))
            filelist.add_files_to_filelist(extend)
            for url in extend:
                filelist.add_metadata(url, fn_metadata(url))
            if not cache:
                filelist.clear_cache()
            for url in urls + extend:
                expected = fn_metadata(url)
                self.assertEqual(expected, filelist.get_metadata(url))
        
    def test_10_04_get_metadata_after_remove(self):
        to_remove = "file://foo/bar.jpg"
        urls = [
            "file://foo/foo.jpg",
            to_remove,
            "file://foo/baz.jpg"]
        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            return r.randint(ord(" "), ord("~"), 300).astype(np.uint8).tostring()
        for filelist, cache in ((self.filelist, True), 
                                (self.filelist_nocache, False)):
            filelist.add_files_to_filelist(urls)
            for url in urls:
                filelist.add_metadata(url, fn_metadata(url))
            filelist.remove_files_from_filelist([to_remove])
            if not cache:
                filelist.clear_cache()
            for url in urls:
                if url == to_remove:
                    self.assertIsNone(filelist.get_metadata(url))
                else:
                    expected = fn_metadata(url)
                    self.assertEqual(expected, filelist.get_metadata(url))
                    
    def test_11_01_hasnt_files(self):
        self.assertFalse(self.filelist.has_files())
        
    def test_11_02_has_files(self):
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        self.assertTrue(self.filelist.has_files())
        self.filelist.clear_cache()
        self.assertTrue(self.filelist.has_files())
        #
        # Make sure cache wasn't screwed up
        #
        roots = []
        directories = []
        urls = []
        def fn(r, d, u):
            roots.append(r)
            directories.append(d)
            urls.extend(u)
        self.filelist.walk(fn)
        self.assertEqual(len(urls), 1)
        
    def test_11_03_hasnt_files_after_remove(self):
        url = "file://foo/bar/baz.jpg"
        self.filelist.add_files_to_filelist([url])
        self.filelist.remove_files_from_filelist([url])
        self.assertFalse(self.filelist.has_files())
        self.filelist.clear_cache()
        self.assertFalse(self.filelist.has_files())
        
class TestHDF5ImageSet(unittest.TestCase):
    CHANNEL_NAME = "channelname"
    ALT_CHANNEL_NAME = "alt_channelname"
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename)
        
    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        
    def test_01_01_init(self):
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        self.assertEqual(image_set.root.name, "/" + H5DICT.IMAGES_GROUP)
        
    def test_01_02_reinit(self):
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        self.assertEqual(image_set.root.name, "/" + H5DICT.IMAGES_GROUP)
        
    def test_02_01_set_and_get(self):
        r = np.random.RandomState()
        r.seed(21)
        data = r.uniform(size=(2, 3, 4, 5, 6))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data)
        np.testing.assert_array_equal(
            data, image_set.get_image(self.CHANNEL_NAME))
        
    def test_02_02_set_reattach_and_get(self):
        r = np.random.RandomState()
        r.seed(22)
        data = r.uniform(size=(2, 3, 4, 5, 6))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data)
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        np.testing.assert_array_equal(
            data, image_set.get_image(self.CHANNEL_NAME))

    def test_02_03_ensure_no_overwrite(self):
        r = np.random.RandomState()
        r.seed(23)
        data = r.uniform(size=(2, 3, 4, 5, 6))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data)
        copy = image_set.get_image(self.CHANNEL_NAME)
        copy[:] = 0
        np.testing.assert_array_equal(
            data, image_set.get_image(self.CHANNEL_NAME))
        
    def test_02_04_set_and_get_two(self):
        r = np.random.RandomState()
        r.seed(24)
        data1 = r.uniform(size=(2, 3, 4, 5, 6))
        data2 = r.uniform(size=(6, 5, 4, 3, 2))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data1)
        image_set.set_image(self.ALT_CHANNEL_NAME, data2)
        np.testing.assert_array_equal(
            data1, image_set.get_image(self.CHANNEL_NAME))
        np.testing.assert_array_equal(
            data2, image_set.get_image(self.ALT_CHANNEL_NAME))
        
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
                
    def test_12_00_extend_none(self):
        data = ["hello", "world"]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.extend([])
        self.assertSequenceEqual(data, a)
        
    def test_12_01_extend_one(self):
        data = ["hello", "world"]
        extend = ["."]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.extend(extend)
        self.assertSequenceEqual(data + extend, a)
        
    def test_12_02_extend_many(self):
        data = ["hello", "world"]
        extend = ["that", "is", "observing", "the", "transit", "of", "venus"]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.extend(extend)
        self.assertSequenceEqual(data + extend, a)
        
    def test_13_00_reorder_deleting_all(self):
        data = ["hello", "world"]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.reorder([])
        self.assertEqual(len(a), 0)
        
    def test_13_01_reorder(self):
        data = ["hello", "green", "world"]
        order = [ 2, 0, 1]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.reorder(order)
        self.assertSequenceEqual([data[o] for o in order], a)
        
    def test_13_02_reorder_with_delete(self):
        data = ["hello", "green", "world"]
        order = [ 2, 0]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.reorder(order)
        self.assertEqual(len(a), len(order))
        self.assertSequenceEqual([data[o] for o in order], a)
        
    def test_14_01_is_not_none_many(self):
        data = ["hello", None, "world"]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        self.assertSequenceEqual(list(a.is_not_none()), [True, False, True])
        
    def test_14_02_is_not_none_single(self):
        data = ["hello", None, "world"]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        for i, expected in enumerate( [True, False, True]):
            self.assertEqual(a.is_not_none(i), expected)
        
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
        