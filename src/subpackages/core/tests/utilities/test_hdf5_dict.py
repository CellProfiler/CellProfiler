"""test_hdf5_dict - test the hdf5_dict module
"""

import os
import random
import string
import sys
import tempfile
import unittest

import h5py
import numpy
import numpy as np
import six

import cellprofiler_core.utilities.hdf5_dict as H5DICT

OBJECT_NAME = "objectname"
FEATURE_NAME = "featurename"
ALT_FEATURE_NAME = "featurename2"


class HDF5DictTessstBase(unittest.TestCase):
    """Base class for HDF5Dict test cases

    This class creates a .h5 file per test case during setUp and
    deletes it during tearDown.

    Note: misspelling of Tessst is intentional.
    """

    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf_file = h5py.File(self.temp_filename, "w")

    def tearDown(self):
        self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        self.assertFalse(
            os.path.exists(self.temp_filename),
            "If the file can't be removed, it's a bug. "
            "Clean up your trash: %s" % self.temp_filename,
        )


class TestHDF5Dict(unittest.TestCase):
    def setUp(self):
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename)

    def tearDown(self):
        self.hdf5_dict.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        self.assertFalse(
            os.path.exists(self.temp_filename),
            "If the file can't be removed, it's a bug. "
            "Clean up your trash: %s" % self.temp_filename,
        )

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
        data = r.randint(0, 100, (10, 100))
        p = r.permutation(data.shape[0])
        for i, d in enumerate(data[p, :]):
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, p[i] + 1] = d
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, i + 1]
            np.testing.assert_array_equal(d, result)

    def test_02_02_read_none(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = "Hello"
        self.assertIsNone(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1])

    def test_02_03_write_numeric_imagesets(self):
        for dtype, code, ftr in (
            (int, "i", FEATURE_NAME),
            (float, "f", ALT_FEATURE_NAME),
        ):
            data = [
                np.zeros(0, dtype),
                np.array([1, 10, 100], dtype),
                np.array([2, 20], dtype),
                np.array([3, 30, 300, 3000], dtype),
            ]
            self.hdf5_dict[OBJECT_NAME, ftr, [1, 3]] = data[::2]
            self.hdf5_dict[OBJECT_NAME, ftr, [2, 4]] = data[1::2]
            d2 = self.hdf5_dict[OBJECT_NAME, ftr, [1, 2, 3, 4]]
            self.assertEqual(len(d2), 4)
            self.assertIsNone(d2[0])
            self.assertTrue(all([d.dtype.kind == code for d in d2[1:]]))
            self.assertTrue(all([np.all(d == e) for d, e in zip(d2[1:], data[1:])]))

    def test_02_04_write_string_imagesets(self):
        data = [None, "foo", "bar", None, "baz"]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [1, 2]] = data[:2]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [3, 4, 5]] = data[2:]
        d2 = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, [1, 2, 3, 4, 5]]
        self.assertIsNone(d2[0])
        self.assertIsNone(d2[3])
        for i in (1, 2, 4):
            self.assertEqual(data[i], d2[i][0])

    def test_02_05_00_upgrade_none(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.zeros(0)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = np.arange(5)
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2].tolist(), list(range(5))
        )

    def test_02_05_01_upgrade_none_multiple(self):
        # Regression test of issue #1011
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.zeros(0)
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, np.arange(2, 4)] = [
            np.zeros(1),
            np.zeros(0),
        ]

    def test_02_06_upgrade_int_to_float(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = np.arange(5)
        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].dtype.kind, "i")
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = np.arange(5).astype(float) / 2.0

        self.assertEqual(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].dtype.kind, "f")
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1].tolist(), list(range(5))
        )
        self.assertSequenceEqual(
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2].tolist(),
            [float(x) / 2 for x in range(5)],
        )

    def test_02_07_upgrade_float_to_string(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1] = [2.5]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2] = ["alfalfa"]
        data = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        self.assertIsInstance(data[0], str)

    def test_02_08_write_twice(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2)] = [1.2, 3.4]
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (2, 3)] = [5.6, 7.8]

        data = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2, 3)]
        self.assertEqual(tuple(data), (1.2, 5.6, 7.8))
        index = self.hdf5_dict.top_group[OBJECT_NAME][FEATURE_NAME][H5DICT.INDEX]
        self.assertEqual(index.shape[0], 3)

    def test_02_09_write_with_dtype(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1, np.uint8] = np.zeros(5, np.uint16)
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1]
        self.assertEqual(len(result), 5)
        self.assertEqual(result.dtype, np.uint8)

    def test_02_10_write_with_dtype_and_nulls(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, (1, 2), np.uint8] = [
            None,
            np.zeros(5, np.uint8),
        ]
        self.assertIsNone(self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1])
        result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 2]
        self.assertEqual(len(result), 5)
        self.assertEqual(result.dtype, np.uint8)

    def test_02_11_write_empty_dtype(self):
        self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, 1, np.uint8] = None
        self.assertEqual(
            self.hdf5_dict.get_feature_dtype(OBJECT_NAME, FEATURE_NAME), np.uint8
        )

    def test_03_01_add_all(self):
        r = np.random.RandomState()
        r.seed(301)
        data = r.randint(0, 100, (10, 100))
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data)
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, i + 1]
            np.testing.assert_array_equal(d, result)

    def test_03_02_add_all_out_of_order(self):
        r = np.random.RandomState()
        r.seed(302)
        data = r.randint(0, 100, (10, 100))
        p = r.permutation(data.shape[0]) + 1
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data, p)
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, p[i]]
            np.testing.assert_array_equal(d, result)

    def test_03_02_01_add_all_out_of_order_object(self):
        r = np.random.RandomState()
        r.seed(302)
        data = np.array(["World", "Hello"], object)
        p = np.array([2, 1], int)
        self.hdf5_dict.add_all(OBJECT_NAME, FEATURE_NAME, data, p)
        for i, d in enumerate(data):
            result = self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, p[i]]
            self.assertEqual(result, d)

    def test_03_03_add_all_with_dtype(self):
        r = np.random.RandomState()
        r.seed(303)
        data = [[], r.randint(0, 255, 10).astype(np.uint8)]
        self.hdf5_dict.add_all(
            OBJECT_NAME,
            FEATURE_NAME,
            data,
            idxs=np.arange(1, len(data) + 1),
            data_type=np.uint8,
        )
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
            self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, idx + 1] = values[idx]
        new_image_numbers = np.hstack(([0], r.permutation(len(values)) + 1))
        self.hdf5_dict.reorder(OBJECT_NAME, FEATURE_NAME, new_image_numbers)
        for reopen in (False, True):
            if reopen:
                self.hdf5_dict.close()
                self.hdf5_dict = H5DICT.HDF5Dict(self.temp_filename, mode="r")
            for idx, image_number in enumerate(new_image_numbers[1:]):
                self.assertEqual(
                    values[idx], self.hdf5_dict[OBJECT_NAME, FEATURE_NAME, image_number]
                )

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
            h5copy = H5DICT.HDF5Dict(filename, mode="r")
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
            h5copy = H5DICT.HDF5Dict(temp_filename, copy=self.hdf5_dict.top_group)

            self.assertTrue(h5copy.has_object(OBJECT_NAME))
            self.assertTrue(h5copy.has_feature(OBJECT_NAME, FEATURE_NAME))
            self.assertEqual(h5copy[OBJECT_NAME, FEATURE_NAME, 1], "Hello")
        finally:
            h5copy.close()
            os.close(temp_fd)
            os.remove(temp_filename)
            self.assertFalse(
                os.path.exists(temp_filename),
                "If the file can't be removed, it's a bug. "
                "Clean up your trash: %s" % temp_filename,
            )

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
        self.hdf_file = h5py.File(self.temp_filename, "w")
        self.filelist = H5DICT.HDF5FileList(self.hdf_file)

        self.temp_fd_empty, self.temp_filename_empty = tempfile.mkstemp(".h5")
        self.hdf_file_empty = h5py.File(self.temp_filename_empty, "w")

    def tearDown(self):
        if isinstance(self.hdf_file, h5py.File):
            self.hdf_file.close()
        os.close(self.temp_fd)
        os.remove(self.temp_filename)
        if isinstance(self.hdf_file_empty, h5py.File):
            self.hdf_file_empty.close()
        os.close(self.temp_fd_empty)
        os.remove(self.temp_filename_empty)

    def test_01_01_encode_alphanumeric(self):
        s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-+.%="
        self.assertEqual(s, s)

    def test_01_02_decode_alphanumeric(self):
        s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-+.%="
        self.assertEqual(s, s)

    def test_01_03_decode_of_encode_is_same(self):
        s = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        self.assertEqual(s, s)

    def test_02_01_get_new_filelist_group(self):
        g = self.filelist.get_filelist_group()
        self.assertIn(H5DICT.FILE_LIST_GROUP, self.hdf_file)
        self.assertIn(H5DICT.DEFAULT_GROUP, self.hdf_file[H5DICT.FILE_LIST_GROUP])

    def test_02_02_get_existing_filelist_group(self):
        g1 = self.filelist.get_filelist_group()
        g2 = self.filelist.get_filelist_group()
        self.assertEqual(g1, g2)

    def test_02_03_has_file_list(self):
        self.assertTrue(H5DICT.HDF5FileList.has_file_list(self.hdf_file))

    def test_02_04_no_file_list(self):
        self.assertFalse(H5DICT.HDF5FileList.has_file_list(self.hdf_file_empty))

    def test_02_05_copy(self):
        url = "file:///foo/bar.jpg"
        metadata = numpy.array([1, 2, 3, 4, 5], dtype=int)
        self.filelist.add_files_to_filelist([url])
        self.filelist.add_metadata(url, metadata)
        H5DICT.HDF5FileList.copy(self.hdf_file, self.hdf_file_empty)

        dest_filelist = H5DICT.HDF5FileList(self.hdf_file_empty)
        self.assertSequenceEqual([url], dest_filelist.get_filelist())
        numpy.testing.assert_array_equal(metadata, dest_filelist.get_metadata(url)[0])

    def test_02_06_copy_corruption(self):
        # Check that we don't corrupt file strings containing the schema
        url = "file:///foo/cellprofiler/bar.jpg"
        self.filelist.add_files_to_filelist([url])
        H5DICT.HDF5FileList.copy(self.hdf_file, self.hdf_file_empty)
        dest_filelist = H5DICT.HDF5FileList(self.hdf_file_empty)
        self.assertSequenceEqual([url], dest_filelist.get_filelist())

    def test_03_00_add_no_file(self):
        self.filelist.add_files_to_filelist([])

    def test_03_01_add_file(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:/foo/bar.jpg"])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 1)
        self.assertIn(b"bar.jpg", filenames)

    def test_03_02_add_two_files(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:/foo/bar.jpg", "file:/foo/baz.jpg"])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 2)
        self.assertEqual(filenames[0].decode(), "bar.jpg")
        self.assertEqual(filenames[1].decode(), "baz.jpg")

    def test_03_03_add_two_directories(self):
        filelist = self.filelist
        g = self.filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:/foo/bar.jpg", "file:/bar/baz.jpg"])
        for subdir, filename in (("foo", "bar.jpg"), ("bar", "baz.jpg")):
            self.assertIn("file:", g)
            self.assertIn(subdir, g["file:"])
            filenames = g["file:"][subdir][H5DICT.FILES][:]
            self.assertEqual(len(filenames), 1)
            self.assertEqual(filenames[0].decode(), filename)

    def test_03_04_add_a_file_and_a_file(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:/foo/bar.jpg"])
        filelist.add_files_to_filelist(["file:/foo/baz.jpg"])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 2)
        self.assertEqual(filenames[0].decode(), "bar.jpg")
        self.assertEqual(filenames[1].decode(), "baz.jpg")

    def test_03_05_add_a_file_with_a_stupid_DOS_name(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:///C:/foo/bar.jpg"])
        self.assertIn("file:", g)
        self.assertIn("C:", g["file:"])
        self.assertIn("foo", g["file:"]["C:"])
        filenames = g["file:/C:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0].decode(), "bar.jpg")

    def test_03_06_add_a_file_to_the_base(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:///foo.jpg"])
        self.assertIn("file:", g)
        self.assertIn(H5DICT.FILES, g["file:"])
        filenames = g["file:"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0].decode(), "foo.jpg")
        returned_list = filelist.get_filelist()
        assert len(returned_list) == 1
        assert returned_list[0] == "file:///foo.jpg"


    def test_03_07_add_a_file_to_the_schema(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file:foo.jpg"])
        self.assertIn(H5DICT.ROOT, g)
        filenames = g[H5DICT.ROOT][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 1)
        self.assertEqual(filenames[0].decode(), "file:foo.jpg")
        returned_list = filelist.get_filelist()
        assert len(returned_list) == 1
        assert returned_list[0] == "file:foo.jpg"


    def test_03_08_what_if_the_user_has_a_directory_named_index(self):
        filelist = self.filelist
        #
        # Another in the endless progression of disgusting corner cases
        #
        filelist.add_files_to_filelist(["file:///foo/bar.jpg"])
        filelist.add_files_to_filelist(
            ["file:///foo/index/baz.jpg", "file:///foo/data/xyz.jpg"]
        )
        result = filelist.get_filelist()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "file:///foo/bar.jpg")
        self.assertEqual(result[1], "file:///foo/data/xyz.jpg")
        self.assertEqual(result[2], "file:///foo/index/baz.jpg")


    def test_04_00_remove_none(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file://foo/bar.jpg", "file://foo/baz.jpg"])
        self.filelist.remove_files_from_filelist([])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 2)
        self.assertEqual(filenames[0].decode(), "bar.jpg")
        self.assertEqual(filenames[1].decode(), "baz.jpg")

    def test_04_01_remove_file(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(
            ["file://foo/bar.jpg", "file://foo/baz.jpg", "file://foo/a.jpg"]
        )
        filelist.remove_files_from_filelist(["file://foo/bar.jpg"])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 2)
        self.assertEqual(filenames[0].decode(), "a.jpg")
        self.assertEqual(filenames[1].decode(), "baz.jpg")

    def test_04_02_remove_all_files_in_dir(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file://foo/bar.jpg", "file://bar/baz.jpg"])
        filelist.remove_files_from_filelist(["file://foo/bar.jpg"])
        self.assertIn("file:", g)
        self.assertIn("bar", g["file:"])
        self.assertNotIn("foo", g["file:"])

    def test_04_03_remove_all_files_in_parent(self):
        filelist = self.filelist
        g = filelist.get_filelist_group()
        filelist.add_files_to_filelist(["file://foo/bar.jpg", "file:baz.jpg"])
        self.assertIn(H5DICT.ROOT, g)
        filelist.remove_files_from_filelist(["file:baz.jpg"])
        self.assertIn("file:", g)
        self.assertIn("foo", g["file:"])
        self.assertNotIn(H5DICT.ROOT, g)
        filenames = g["file:/foo"][H5DICT.FILES][:]
        self.assertEqual(len(filenames), 1)
        self.assertIn("bar.jpg", filenames[0].decode())

    def test_05_01_get_filelist(self):
        filelist = self.filelist
        self.filelist.add_files_to_filelist(
            ["file:///foo/bar.jpg", "file:///foo/baz.jpg"]
        )
        urls = self.filelist.get_filelist()
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], "file:///foo/bar.jpg")
        self.assertEqual(urls[1], "file:///foo/baz.jpg")

    def test_05_02_get_multidir_filelist(self):
        filelist = self.filelist
        filelist.add_files_to_filelist(
            ["file:///foo/bar.jpg", "file:///bar/baz.jpg", "file:///foo.jpg"]
        )
        urls = filelist.get_filelist()
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls[2], "file:///foo/bar.jpg")
        self.assertEqual(urls[1], "file:///bar/baz.jpg")
        self.assertEqual(urls[0], "file:///foo.jpg")

    def test_05_03_get_sub_filelist(self):
        filelist = self.filelist
        filelist.add_files_to_filelist(
            ["file:/foo/bar.jpg", "file:/bar/baz.jpg", "file:/foo/baz.jpg"]
        )
        urls = filelist.get_filelist("file:/foo")
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls[0], "file:/foo/bar.jpg")
        self.assertEqual(urls[1], "file:/foo/baz.jpg")

    def test_06_00_walk_empty(self):
        assert len(self.filelist.get_filelist()) == 0

    def test_06_01_walk_one(self):
        filelist = self.filelist
        filelist.add_files_to_filelist(["file:/foo/bar.jpg", "file:/foo/baz.jpg"])
        root = self.filelist.get_filelist_group()
        full_list = self.filelist.get_filelist()
        assert len(full_list) == 2
        self.assertEqual(1, len(root.keys()))
        assert 'file:' in root
        level = root['file:']
        self.assertEqual(1, len(level.keys()))
        assert 'foo' in level
        level2 = level['foo']
        self.assertEqual(3, len(level2.keys()))
        assert H5DICT.FILES in level2
        assert H5DICT.METADATA in level2
        assert H5DICT.SERIESNAMES in level2
        data = level2[H5DICT.FILES][:]
        self.assertEqual(2, len(data))
        # H5Py returns strings as bytes, manually decode.
        data = set([d.decode() for d in data])
        assert data == {"bar.jpg", "baz.jpg"}

    def test_06_02_walk_many(self):
        filelist = self.filelist
        filelist.add_files_to_filelist(
            [
                "file://foo/bar.jpg",
                "file://foo/baz.jpg",
                "file://foo/bar/baz.jpg",
                "file://bar/foo.jpg",
                "file://foo/baz/bar.jpg",
            ]
        )

        root = self.filelist.get_filelist_group()
        full_list = self.filelist.get_filelist()
        assert len(full_list) == 5
        self.assertEqual(1, len(root.keys()))
        assert 'file:' in root
        level = root['file:']
        self.assertEqual(2, len(level.keys()))
        assert 'foo' in level
        assert 'bar' in level

        level2_bar = level["bar"]
        self.assertEqual(3, len(level2_bar.keys()))
        assert H5DICT.FILES in level2_bar
        assert H5DICT.METADATA in level2_bar
        assert H5DICT.SERIESNAMES in level2_bar
        data = level2_bar[H5DICT.FILES][:]
        self.assertEqual(1, len(data))
        assert data[0].decode() == "foo.jpg"

        level2_foo = level["foo"]
        self.assertEqual(5, len(level2_foo.keys()))
        expected_keys = {H5DICT.FILES, H5DICT.METADATA, H5DICT.SERIESNAMES, "bar", "baz"}
        assert set(level2_foo.keys()) == expected_keys
        data = level2_foo[H5DICT.FILES][:]
        self.assertEqual(2, len(data))
        # H5Py returns strings as bytes, manually decode.
        data = set([d.decode() for d in data])
        assert data == {"bar.jpg", "baz.jpg"}

        level3_bar = level2_foo["bar"]
        self.assertEqual(3, len(level3_bar.keys()))
        assert H5DICT.FILES in level3_bar
        assert H5DICT.METADATA in level3_bar
        assert H5DICT.SERIESNAMES in level3_bar
        data = level3_bar[H5DICT.FILES][:]
        self.assertEqual(1, len(data))
        assert data[0].decode() == "baz.jpg"

        level3_baz = level2_foo["baz"]
        self.assertEqual(3, len(level3_baz.keys()))
        assert H5DICT.FILES in level3_baz
        assert H5DICT.METADATA in level3_baz
        assert H5DICT.SERIESNAMES in level3_baz
        data = level3_baz[H5DICT.FILES][:]
        self.assertEqual(1, len(data))
        assert data[0].decode() == "bar.jpg"

    def test_07_00_list_files_in_empty_dir(self):
        self.assertEqual(len(self.filelist.get_filelist("file://foo")), 0)
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        self.assertEqual(len(self.filelist.get_filelist("file://foo")), 1)
        self.assertEqual(len(self.filelist.get_filelist("file://foo/baz")), 0)
        self.assertEqual(len(self.filelist.get_filelist()), 1)

    def test_07_01_list_files(self):
        filelist = self.filelist
        filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        result = filelist.get_filelist("file://foo/bar")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "file://foo/bar/baz.jpg")

    def test_10_01_get_no_metadata(self):
        urls = ["file://foo.jpg", "file://foo/bar.jpg", "file://foo/bar/baz.jpg"]
        filelist = self.filelist
        filelist.add_files_to_filelist(urls)
        for url in urls:
            numpy.testing.assert_array_equal([-1, -1, -1, -1, -1], filelist.get_metadata(url)[0])

    def test_10_02_get_metadata(self):
        urls = ["file://foo.jpg", "file://foo/bar.jpg", "file://foo/bar/baz.jpg"]

        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            r.randint(0, 200, 5)
            return r.randint(0, 200, 5)

        def fn_name(url):
            random.seed(url)
            return [''.join(random.choice(string.ascii_letters) for i in range(10))]

        filelist = self.filelist
        filelist.add_files_to_filelist(urls)
        for url in urls:
            filelist.add_metadata(url, fn_metadata(url), fn_name(url))
        for url in urls:
            expected_meta = fn_metadata(url)
            expected_names = fn_name(url)
            actual_meta, actual_names = filelist.get_metadata(url)
            numpy.testing.assert_array_equal(expected_meta, actual_meta)
            numpy.testing.assert_array_equal(expected_names, actual_names)

    def test_10_03_get_metadata_after_insert(self):
        urls = ["file://foo/foo.jpg", "file://foo/bar.jpg", "file://foo/baz.jpg"]
        extend = ["file://foo/pleasework.jpg", "file://foo/beesaregreat.jpg"]

        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            r.randint(0, 200, 5)
            return r.randint(0, 200, 5)

        def fn_name(url):
            random.seed(url)
            return [''.join(random.choice(string.ascii_letters) for i in range(10))]

        filelist = self.filelist
        filelist.add_files_to_filelist(urls)
        for url in urls:
            filelist.add_metadata(url, fn_metadata(url), fn_name(url))
        filelist.add_files_to_filelist(extend)
        for url in extend:
            filelist.add_metadata(url, fn_metadata(url), fn_name(url))
        for url in urls + extend:
            expected_meta = fn_metadata(url)
            expected_name = fn_name(url)
            actual_meta, actual_name = filelist.get_metadata(url)
            numpy.testing.assert_array_equal(expected_meta, actual_meta)
            numpy.testing.assert_array_equal(expected_name, actual_name)

    def test_10_04_get_metadata_after_remove(self):
        to_remove = "file://foo/bar.jpg"
        urls = ["file://foo/foo.jpg", to_remove, "file://foo/baz.jpg"]

        def fn_metadata(url):
            r = np.random.RandomState()
            r.seed(np.fromstring(url, np.uint8))
            r.randint(0, 200, 5)
            return r.randint(0, 200, 5)

        def fn_name(url):
            random.seed(url)
            return [''.join(random.choice(string.ascii_letters) for i in range(10))]

        filelist = self.filelist
        filelist.add_files_to_filelist(urls)
        for url in urls:
            filelist.add_metadata(url, fn_metadata(url), fn_name(url))
        filelist.remove_files_from_filelist([to_remove])
        for url in urls:
            if url == to_remove:
                self.assertIsNone(filelist.get_metadata(url))
            else:
                expected_meta = fn_metadata(url)
                expected_name = fn_name(url)
                actual_meta, actual_name = filelist.get_metadata(url)
                numpy.testing.assert_array_equal(expected_meta, actual_meta)
                numpy.testing.assert_array_equal(expected_name, actual_name)

    def test_11_01_hasnt_files(self):
        self.assertFalse(self.filelist.has_files())

    def test_11_02_has_files(self):
        self.filelist.add_files_to_filelist(["file://foo/bar/baz.jpg"])
        self.assertTrue(self.filelist.has_files())
        urls = self.filelist.get_filelist()
        self.assertEqual(len(urls), 1)

    def test_11_03_hasnt_files_after_remove(self):
        url = "file://foo/bar/baz.jpg"
        self.filelist.add_files_to_filelist([url])
        self.filelist.remove_files_from_filelist([url])
        self.assertFalse(self.filelist.has_files())


class TestHDF5ImageSet(HDF5DictTessstBase):
    CHANNEL_NAME = "channelname"
    ALT_CHANNEL_NAME = "alt_channelname"

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
        np.testing.assert_array_equal(data, image_set.get_image(self.CHANNEL_NAME))

    def test_02_02_set_reattach_and_get(self):
        r = np.random.RandomState()
        r.seed(22)
        data = r.uniform(size=(2, 3, 4, 5, 6))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data)
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        np.testing.assert_array_equal(data, image_set.get_image(self.CHANNEL_NAME))

    def test_02_03_ensure_no_overwrite(self):
        r = np.random.RandomState()
        r.seed(23)
        data = r.uniform(size=(2, 3, 4, 5, 6))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data)
        copy = image_set.get_image(self.CHANNEL_NAME)
        copy[:] = 0
        np.testing.assert_array_equal(data, image_set.get_image(self.CHANNEL_NAME))

    def test_02_04_set_and_get_two(self):
        r = np.random.RandomState()
        r.seed(24)
        data1 = r.uniform(size=(2, 3, 4, 5, 6))
        data2 = r.uniform(size=(6, 5, 4, 3, 2))
        image_set = H5DICT.HDF5ImageSet(self.hdf_file)
        image_set.set_image(self.CHANNEL_NAME, data1)
        image_set.set_image(self.ALT_CHANNEL_NAME, data2)
        np.testing.assert_array_equal(data1, image_set.get_image(self.CHANNEL_NAME))
        np.testing.assert_array_equal(data2, image_set.get_image(self.ALT_CHANNEL_NAME))


class TestHDF5ObjectSet(HDF5DictTessstBase):
    OBJECTS_NAME = "objectsname"
    ALT_OBJECTS_NAME = "altobjectsname"
    SEGMENTATION_NAME = "segmentationname"
    ALT_SEGMENTATION_NAME = "altsegmentationname"

    def test_01_01_init(self):
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)

    def test_01_02_set_has_get_dense(self):
        # Test set_dense, has_dense, get_dense
        r = np.random.RandomState()
        r.seed(12)
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        self.assertFalse(
            object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        expected = r.randint(0, 10, size=(11, 13))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        self.assertTrue(object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME))
        self.assertFalse(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        np.testing.assert_array_equal(
            expected, object_set.get_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )

    def test_01_03_set_has_get_sparse(self):
        # test set_sparse, has_sparse, get_sparse
        r = np.random.RandomState()
        r.seed(13)
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        self.assertFalse(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        expected = np.core.records.fromarrays(
            r.randint(0, 10, (3, 9)),
            [
                (object_set.AXIS_Y, np.uint32, 1),
                (object_set.AXIS_X, np.uint32, 1),
                (object_set.AXIS_LABELS, np.uint32, 1),
            ],
        )
        object_set.set_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        self.assertTrue(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        self.assertFalse(
            object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        np.testing.assert_array_equal(
            expected, object_set.get_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )

    def test_01_04_clear(self):
        r = np.random.RandomState()
        r.seed(14)
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        expected = r.randint(0, 10, size=(11, 13))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        expected = np.core.records.fromarrays(
            r.randint(0, 10, (3, 9)),
            [
                (object_set.AXIS_Y, np.uint32, 1),
                (object_set.AXIS_X, np.uint32, 1),
                (object_set.AXIS_LABELS, np.uint32, 1),
            ],
        )
        object_set.set_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        object_set.clear(self.OBJECTS_NAME, self.ALT_SEGMENTATION_NAME)
        self.assertTrue(object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME))
        object_set.clear(self.OBJECTS_NAME)
        self.assertFalse(
            object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        self.assertFalse(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        expected = r.randint(0, 10, size=(11, 13))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        self.assertTrue(object_set.has_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME))
        self.assertFalse(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )

    def test_01_05_set_dense_twice_same_size(self):
        r = np.random.RandomState()
        r.seed(15)
        expected = r.randint(0, 10, size=(11, 13))
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        expected = r.randint(0, 10, size=(11, 13))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        np.testing.assert_array_equal(
            expected, object_set.get_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )

    def test_01_06_set_dense_different_size(self):
        r = np.random.RandomState()
        r.seed(16)
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        expected = r.randint(0, 10, size=(11, 13))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        expected = r.randint(0, 10, size=(13, 11))
        object_set.set_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        np.testing.assert_array_equal(
            expected, object_set.get_dense(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )

    def test_01_07_set_sparse_many(self):
        r = np.random.RandomState()
        r.seed(13)
        object_set = H5DICT.HDF5ObjectSet(self.hdf_file)
        self.assertFalse(
            object_set.has_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        expected = np.core.records.fromarrays(
            r.randint(0, 10, (3, 9)),
            [
                (object_set.AXIS_Y, np.uint32, 1),
                (object_set.AXIS_X, np.uint32, 1),
                (object_set.AXIS_LABELS, np.uint32, 1),
            ],
        )
        object_set.set_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        expected = np.core.records.fromarrays(
            r.randint(0, 10, (3, 20)),
            [
                (object_set.AXIS_Y, np.uint32, 1),
                (object_set.AXIS_X, np.uint32, 1),
                (object_set.AXIS_LABELS, np.uint32, 1),
            ],
        )
        object_set.set_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        np.testing.assert_array_equal(
            expected, object_set.get_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )
        expected = np.core.records.fromarrays(
            r.randint(0, 10, (3, 6)),
            [
                (object_set.AXIS_Y, np.uint32, 1),
                (object_set.AXIS_X, np.uint32, 1),
                (object_set.AXIS_LABELS, np.uint32, 1),
            ],
        )
        object_set.set_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME, expected)
        np.testing.assert_array_equal(
            expected, object_set.get_sparse(self.OBJECTS_NAME, self.SEGMENTATION_NAME)
        )


class TestHDFCSV(HDF5DictTessstBase):
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
        d = {
            "random": ["foo", "bar", "baz"],
            "fruits": ["lemon", "cherry", "orange", "apple"],
            "rocks": ["granite", "basalt", "limestone"],
        }
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        csv.set_all(d)
        for key, strings in list(d.items()):
            column = csv[key]
            self.assertEqual(len(column), len(strings))
            for s0, s1 in zip(strings, column):
                self.assertEqual(s0, s1)

    def test_05_01_get_column_names_etc(self):
        d = {
            "random": ["foo", "bar", "baz"],
            "fruits": ["lemon", "cherry", "orange", "apple"],
            "rocks": ["granite", "basalt", "limestone"],
        }
        csv = H5DICT.HDFCSV(self.hdf_file, "csv")
        csv.set_all(d)
        for key in d:
            self.assertIn(key, csv.get_column_names())
            self.assertIn(key, csv)
            self.assertIn(key, list(csv.keys()))
            self.assertIn(key, iter(list(csv.keys())))


class TestVStringArray(HDF5DictTessstBase):
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
        self.assertEqual(self.hdf_file["data"][:].tostring().decode("utf-8"), "foo")

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
        self.assertEqual(self.hdf_file["index"][0, 1] - self.hdf_file["index"][0, 0], 2)
        self.assertEqual(
            self.hdf_file["data"][
                self.hdf_file["index"][0, 0] : self.hdf_file["index"][0, 1]
            ]
            .tostring()
            .decode("utf-8"),
            "hu",
        )

    def test_02_04_set_and_set_longer(self):
        a = H5DICT.VStringArray(self.hdf_file)
        a[0] = "foo"
        a[0] = "whoops"
        self.assertEqual(self.hdf_file["index"].shape[0], 1)
        self.assertEqual(self.hdf_file["index"][0, 1] - self.hdf_file["index"][0, 0], 6)
        self.assertEqual(
            self.hdf_file["data"][
                self.hdf_file["index"][0, 0] : self.hdf_file["index"][0, 1]
            ]
            .tostring()
            .decode("utf-8"),
            "whoops",
        )

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
        s = "\\u03b4x"
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
        chars = r.randint(ord("A"), ord("F") + 1, idx[-1]).astype(np.uint8)
        strings = [
            chars[i:j].tostring().decode("utf-8") for i, j in zip(idx[:-1], idx[1:])
        ]
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
        chars = r.randint(ord("A"), ord("F") + 1, idx[-1]).astype(np.uint8)
        strings = [
            chars[i:j].tostring().decode("utf-8") for i, j in zip(idx[:-1], idx[1:])
        ]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(strings[:50])
        a.sort()
        for s in strings[50:]:
            idx = a.bisect_left(s)
            if idx > 0:
                self.assertLessEqual(a[idx - 1], s)
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
        order = [2, 0, 1]
        a = H5DICT.VStringArray(self.hdf_file)
        a.set_all(data)
        a.reorder(order)
        self.assertSequenceEqual([data[o] for o in order], a)

    def test_13_02_reorder_with_delete(self):
        data = ["hello", "green", "world"]
        order = [2, 0]
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
        for i, expected in enumerate([True, False, True]):
            self.assertEqual(a.is_not_none(i), expected)


class TestStringReference(HDF5DictTessstBase):
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
        for s1, s2 in (("foo", "bar"), ("bar", "foo"), ("foo", "fooo"), ("fo", "foo")):
            for how in ("together", "separate"):
                sr = H5DICT.StringReferencer(
                    self.hdf_file.create_group("test" + s1 + s2 + how)
                )
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
        s = "\\u03c0r\\u00b2"
        result = sr.get_string_refs([s])
        self.assertEqual(len(result), 1)
        rstrings = sr.get_strings(result)
        self.assertEqual(len(rstrings), 1)
        self.assertEqual(s, rstrings[0])

    def test_01_06_two_tier(self):
        r = np.random.RandomState()
        r.seed(16)
        for i in range(100):
            sr = H5DICT.StringReferencer(self.hdf_file.create_group("test%d" % i), 4)
            strings = [chr(65 + r.randint(0, 26)) for j in range(10)]
            result = sr.get_string_refs([strings])
            self.assertEqual(len(result), 10)
            rstrings = sr.get_strings(result)
            self.assertSequenceEqual(strings, rstrings)

    def test_01_07_three_tier(self):
        r = np.random.RandomState()
        r.seed(16)
        for i in range(100):
            sr = H5DICT.StringReferencer(self.hdf_file.create_group("test%d" % i), 4)
            strings = [
                chr(65 + r.randint(0, 26)) + chr(65 + r.randint(0, 26))
                for j in range(50)
            ]
            result = sr.get_string_refs([strings])
            self.assertEqual(len(result), 50)
            rstrings = sr.get_strings(result)
            self.assertSequenceEqual(strings, rstrings)
