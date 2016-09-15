""" test_Measurements.py - tests for CellProfiler.Measurements
"""

import base64
import os
import sys
import tempfile
import unittest
import uuid
import zlib
from cStringIO import StringIO

import h5py
import numpy as np

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas

OBJECT_NAME = "myobjects"
FEATURE_NAME = "feature"


class TestMeasurements(unittest.TestCase):
    def test_00_00_init(self):
        x = cpmeas.Measurements()

    def test_00_01_wrap_unwrap(self):
        test0 = [u"foo", u"foo\\", u"foo\\u0384", u"foo\u0384"]
        test = test0 + [x.encode("utf-8") for x in test0]
        # numpy.object_
        test += np.array(test0, object).tolist()
        for case in test:
            result = cpmeas.Measurements.unwrap_string(
                    cpmeas.Measurements.wrap_string(case))
            if not isinstance(case, unicode):
                case = case.decode("utf-8")
            self.assertEqual(result, case)

    def test_01_01_image_number_is_zero(self):
        x = cpmeas.Measurements()
        self.assertEqual(x.image_set_number, 1)

    def test_01_01_next_image(self):
        x = cpmeas.Measurements()
        x.next_image_set()
        self.assertEqual(x.image_set_number, 2)

    def test_02_01_add_image_measurement(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        self.assertEqual(x.get_current_measurement("Image", "Feature"), "Value")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))

    def test_02_01b_add_image_measurement_arrayinterface(self):
        x = cpmeas.Measurements()
        x["Image", "Feature"] = "Value"
        self.assertEqual(x["Image", "Feature"], "Value")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))

    def test_02_02_add_object_measurement(self):
        x = cpmeas.Measurements()
        np.random.seed(0)
        m = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m)
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature") == m).all)
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Nuclei"))

    def test_02_02b_add_object_measurement_arrayinterface(self):
        x = cpmeas.Measurements()
        np.random.seed(0)
        m = np.random.rand(10)
        x["Nuclei", "Feature"] = m
        self.assertTrue((x["Nuclei", "Feature"] == m).all)
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Nuclei"))

    def test_02_03_add_two_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        np.random.seed(0)
        m = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m)
        self.assertEqual(x.get_current_measurement("Image", "Feature"), "Value")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature") == m).all())
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))

    def test_02_03b_add_two_measurements_arrayinterface(self):
        x = cpmeas.Measurements()
        x["Image", "Feature"] = "Value"
        np.random.seed(0)
        m = np.random.rand(10)
        x["Nuclei", "Feature"] = m
        self.assertEqual(x["Image", "Feature"], "Value")
        self.assertTrue((x["Nuclei", "Feature"] == m).all())
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Nuclei" in x.get_object_names())
        self.assertTrue("Feature" in x.get_feature_names("Image"))

    def test_02_04_add_two_measurements_to_object(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature1", "Value1")
        x.add_measurement("Image", "Feature2", "Value2")
        self.assertEqual(x.get_current_measurement("Image", "Feature1"), "Value1")
        self.assertEqual(x.get_current_measurement("Image", "Feature2"), "Value2")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature1" in x.get_feature_names("Image"))
        self.assertTrue("Feature2" in x.get_feature_names("Image"))

    def test_02_04b_add_two_measurements_to_object_arrayinterface(self):
        x = cpmeas.Measurements()
        x["Image", "Feature1"] = "Value1"
        x["Image", "Feature2"] = "Value2"
        self.assertEqual(x["Image", "Feature1"], "Value1")
        self.assertEqual(x["Image", "Feature2"], "Value2")
        self.assertTrue("Image" in x.get_object_names())
        self.assertTrue("Feature1" in x.get_feature_names("Image"))
        self.assertTrue("Feature2" in x.get_feature_names("Image"))

    def test_03_03_MultipleImageSets(self):
        np.random.seed(0)
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature", "Value1")
        m1 = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m1)
        x.next_image_set()
        x.add_measurement("Image", "Feature", "Value2")
        m2 = np.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m2)
        self.assertEqual(x.get_current_measurement("Image", "Feature"), "Value2")
        self.assertTrue((x.get_current_measurement("Nuclei", "Feature") == m2).all())
        for a, b in zip(x.get_all_measurements("Image", "Feature"), ["Value1", "Value2"]):
            self.assertEqual(a, b)
        for a, b in zip(x.get_all_measurements("Nuclei", "Feature"), [m1, m2]):
            self.assertTrue((a == b).all())

    def test_03_03b_MultipleImageSets_arrayinterface(self):
        np.random.seed(0)
        x = cpmeas.Measurements()
        x["Image", "Feature"] = "Value1"
        m1 = np.random.rand(10)
        x["Nuclei", "Feature"] = m1
        x.next_image_set()
        x["Image", "Feature"] = "Value2"
        m2 = np.random.rand(10)
        x["Nuclei", "Feature"] = m2
        self.assertEqual(x["Image", "Feature"], "Value2")
        self.assertTrue((x["Nuclei", "Feature"] == m2).all())
        for a, b in zip(x["Image", "Feature", :], ["Value1", "Value2"]):
            self.assertEqual(a, b)
        for a, b in zip(x["Nuclei", "Feature", :], [m1, m2]):
            self.assertTrue((a == b).all())

    def test_04_01_get_all_image_measurements_float(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(41)
        vals = r.uniform(size=100)
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(cpmeas.IMAGE, "Feature", vals[image_number - 1],
                              image_set_number=image_number)
        result = m.get_all_measurements(cpmeas.IMAGE, "Feature")
        np.testing.assert_equal(result, vals)

    def test_04_01b_get_all_image_measurements_float_arrayinterface(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(41)
        vals = r.uniform(size=100)
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m[cpmeas.IMAGE, "Feature", image_number] = vals[image_number - 1]
        result = m[cpmeas.IMAGE, "Feature", :]
        np.testing.assert_equal(result, vals)
        for image_number in bad_order:
            result = m[cpmeas.IMAGE, "Feature", image_number]
            np.testing.assert_equal(result, vals[image_number - 1])

    def test_04_02_get_all_image_measurements_string(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = r.uniform(size=100)
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(cpmeas.IMAGE, "Feature",
                              unicode(vals[image_number - 1]),
                              image_set_number=image_number)
        result = m.get_all_measurements(cpmeas.IMAGE, "Feature")
        self.assertTrue(all([r == unicode(v) for r, v in zip(result, vals)]))

    def test_04_02b_get_all_image_measurements_string_arrayinterface(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = r.uniform(size=100)
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m[cpmeas.IMAGE, "Feature", image_number] = unicode(vals[image_number - 1])
        result = m[cpmeas.IMAGE, "Feature", :]
        self.assertTrue(all([r == unicode(v) for r, v in zip(result, vals)]))

    def test_04_03_get_all_image_measurements_unicode(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = [u"\u2211" + str(r.uniform()) for _ in range(100)]
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(cpmeas.IMAGE, "Feature",
                              vals[image_number - 1],
                              image_set_number=image_number)
        result = m.get_all_measurements(cpmeas.IMAGE, "Feature")
        self.assertTrue(all([r == unicode(v) for r, v in zip(result, vals)]))

    def test_04_03b_get_all_image_measurements_unicode_arrayinterface(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = [u"\u2211" + str(r.uniform()) for _ in range(100)]
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m[cpmeas.IMAGE, "Feature", image_number] = vals[image_number - 1]
        result = m[cpmeas.IMAGE, "Feature", :]
        self.assertTrue(all([r == unicode(v) for r, v in zip(result, vals)]))

    def test_04_04_get_all_object_measurements(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = [r.uniform(size=r.randint(10, 100)) for _ in range(100)]
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(OBJECT_NAME, "Feature",
                              vals[image_number - 1],
                              image_set_number=image_number)
        result = m.get_all_measurements(OBJECT_NAME, "Feature")
        self.assertTrue(all([np.all(r == v) and len(r) == len(v)
                             for r, v in zip(result, vals)]))

    def test_04_04b_get_all_object_measurements_arrayinterface(self):
        r = np.random.RandomState()
        m = cpmeas.Measurements()
        r.seed(42)
        vals = [r.uniform(size=r.randint(10, 100)) for _ in range(100)]
        bad_order = r.permutation(np.arange(1, 101))
        for image_number in bad_order:
            m[OBJECT_NAME, "Feature", image_number] = vals[image_number - 1]
        result = m[OBJECT_NAME, "Feature", :]
        self.assertTrue(all([np.all(r == v) and len(r) == len(v)
                             for r, v in zip(result, vals)]))

    # def test_04_05_get_many_string_measurements(self):
    #     #
    #     # Add string measurements that are likely to break things, then
    #     # read them back in one shot
    #     #
    #     test = [u"foo", u"foo\\", u"foo\\u0384", u"foo\u0384"]
    #     test = test + [x.encode('utf-8') for x in test]
    #     test.append(None)
    #     expected = [x if isinstance(x, unicode)
    #                 or x is None
    #                 else x.decode('utf-8')
    #                 for x in test]
    #     m = cpmeas.Measurements()
    #     for i, v in enumerate(test):
    #         m[cpmeas.IMAGE, "Feature", i+1] = v
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     self.assertSequenceEqual(expected, result)

    # def test_04_06_set_many_string_measurements(self):
    #     #
    #     # Add string measurements all at once
    #     #
    #     test = [u"foo", u"foo\\", u"foo\\u0384", u"foo\u0384"]
    #     test = test + [x.encode('utf-8') for x in test]
    #     test.append(None)
    #     expected = [x if isinstance(x, unicode)
    #                 or x is None
    #                 else x.decode('utf-8')
    #                 for x in test]
    #     m = cpmeas.Measurements()
    #     m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)] = test
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     self.assertSequenceEqual(expected, result)

    # def test_04_07_set_many_numeric_measurements(self):
    #     test = [1.5, np.NaN, 3.0]
    #     m = cpmeas.Measurements()
    #     m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)] = test
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     np.testing.assert_array_equal(test, result)

    def test_04_08_set_one_blob_measurement(self):
        r = np.random.RandomState(408)
        test = r.randint(0, 255, 10).astype(np.uint8)
        m = cpmeas.Measurements()
        m[cpmeas.IMAGE, "Feature", 1, np.uint8] = test
        np.testing.assert_array_equal(test, m[cpmeas.IMAGE, "Feature", 1])

    def test_04_09_set_many_blob_measurements(self):
        #
        # This is a regression test which ran into the exception
        # "ValueError: setting an array element with a sequence"
        # when CP attempted to execute something like:
        # np.array([np.nan, np.zeros(5)])
        #
        r = np.random.RandomState(408)
        test = [None, r.randint(0, 255, 10).astype(np.uint8)]
        m = cpmeas.Measurements()
        image_numbers = np.arange(1, len(test) + 1)
        m[cpmeas.IMAGE, "Feature", image_numbers, np.uint8] = test
        result = m[cpmeas.IMAGE, "Feature", image_numbers]
        self.assertIsNone(result[0])
        np.testing.assert_array_equal(test[1], result[1])

    def test_05_01_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))

    def test_05_02_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "OtherFeature", "Value")
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))

    def test_05_02b_test_has_current_measurements_arrayinterface(self):
        x = cpmeas.Measurements()
        x["Image", "OtherFeature"] = "Value"
        self.assertFalse(x.has_current_measurements('Image', 'Feature'))

    def test_05_03_test_has_current_measurements(self):
        x = cpmeas.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        self.assertTrue(x.has_current_measurements('Image', 'Feature'))

    def test_05_03b_test_has_current_measurements_arrayinterface(self):
        x = cpmeas.Measurements()
        x["Image", "Feature"] = "Value"
        self.assertTrue(x.has_current_measurements('Image', 'Feature'))

    def test_06_00_00_dont_apply_metadata(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "pre_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre_post"
        self.assertEqual(x.apply_metadata(pattern), expected)

    def test_06_00_00b_dont_apply_metadata_arrayinterface(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "pre_post"
        x["Image", "Metadata_Plate"] = value
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
        expected = "pre_" + value + "_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"pre_\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)

    def test_06_02_apply_metadata_with_slash(self):
        x = cpmeas.Measurements()
        value = "P12345"
        expected = "\\" + value + "_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\\g<Plate>_post"
        self.assertEqual(x.apply_metadata(pattern), expected)

    def test_06_03_apply_metadata_with_two_slashes(self):
        '''Regression test of img-1144'''
        x = cpmeas.Measurements()
        plate = "P12345"
        well = "A01"
        expected = "\\" + plate + "\\" + well
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
        expected = plate + "_" + well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\g<Plate>_\g<Well>"
        self.assertEqual(x.apply_metadata(pattern), expected)

    def test_06_06_apply_series_and_frame_metadata(self):
        x = cpmeas.Measurements()
        x[cpmeas.IMAGE, cpmeas.C_SERIES + "_DNA"] = 1
        x[cpmeas.IMAGE, cpmeas.C_SERIES + "_DNAIllum"] = 0
        x[cpmeas.IMAGE, cpmeas.C_FRAME + "_DNA"] = 2
        x[cpmeas.IMAGE, cpmeas.C_FRAME + "_DNAIllum"] = 0
        pattern = r"\g<%s>_\g<%s>" % (cpmeas.C_SERIES, cpmeas.C_FRAME)
        self.assertEqual(x.apply_metadata(pattern), "1_2")

    def test_07_01_copy(self):
        x = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(71)
        areas = [r.randint(100, 200, size=r.randint(100, 200))
                 for _ in range(12)]

        for i in range(12):
            x.add_measurement(cpmeas.IMAGE, "Metadata_Well", "A%02d" % (i + 1),
                              image_set_number=(i + 1))
            x.add_measurement(OBJECT_NAME, "AreaShape_Area", areas[i],
                              image_set_number=(i + 1))

        y = cpmeas.Measurements(copy=x)
        for i in range(12):
            self.assertEqual(
                    y.get_measurement(cpmeas.IMAGE, "Metadata_Well", (i + 1)),
                    "A%02d" % (i + 1))
            values = y.get_measurement(OBJECT_NAME, "AreaShape_Area", (i + 1))
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
                        cpmeas.IMAGE, 'Metadata_Plate', i + 1), plate)
        finally:
            try:
                os.unlink(filename)
            except:
                print "Failed to remove file %s" % filename

    def test_09_01_group_by_metadata(self):
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(91)
        aa = [None] * 100
        bb = [None] * 100
        for image_number in r.permutation(np.arange(1, 101)):
            a = r.randint(1, 3)
            aa[image_number - 1] = a
            b = "A%02d" % r.randint(1, 12)
            bb[image_number - 1] = b
            m.add_measurement(cpmeas.IMAGE, "Metadata_A", a,
                              image_set_number=image_number)
            m.add_measurement(cpmeas.IMAGE, "Metadata_B", b,
                              image_set_number=image_number)
        result = m.group_by_metadata(["A", "B"])
        for d in result:
            for image_number in d.image_numbers:
                self.assertEqual(d["A"], aa[image_number - 1])
                self.assertEqual(d["B"], bb[image_number - 1])

    def test_09_02_get_groupings(self):
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(91)
        aa = [None] * 100
        bb = [None] * 100
        for image_number in r.permutation(np.arange(1, 101)):
            a = r.randint(1, 3)
            aa[image_number - 1] = a
            b = "A%02d" % r.randint(1, 12)
            bb[image_number - 1] = b
            m.add_measurement(cpmeas.IMAGE, "Metadata_A", a,
                              image_set_number=image_number)
            m.add_measurement(cpmeas.IMAGE, "Metadata_B", b,
                              image_set_number=image_number)
        result = m.get_groupings(["Metadata_A", "Metadata_B"])
        for d, image_numbers in result:
            for image_number in image_numbers:
                self.assertEqual(d["Metadata_A"], unicode(aa[image_number - 1]))
                self.assertEqual(d["Metadata_B"], unicode(bb[image_number - 1]))

    def test_10_01_remove_image_measurement(self):
        m = cpmeas.Measurements()
        m.add_measurement(cpmeas.IMAGE, "M", "Hello", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "M", "World", image_set_number=2)
        m.remove_measurement(cpmeas.IMAGE, "M", 1)
        self.assertTrue(m.get_measurement(cpmeas.IMAGE, "M", 1) is None)
        self.assertEqual(m.get_measurement(cpmeas.IMAGE, "M", 2), "World")

    def test_10_02_remove_object_measurement(self):
        m = cpmeas.Measurements()
        m.add_measurement(OBJECT_NAME, "M", np.arange(5), image_set_number=1)
        m.add_measurement(OBJECT_NAME, "M", np.arange(7), image_set_number=2)
        m.remove_measurement(OBJECT_NAME, "M", 1)
        self.assertEqual(len(m.get_measurement(OBJECT_NAME, "M", 1)), 0)
        np.testing.assert_equal(m.get_measurement(OBJECT_NAME, "M", 2),
                                np.arange(7))

    def test_10_03_remove_image_number(self):
        m = cpmeas.Measurements()
        m.add_measurement(cpmeas.IMAGE, "M", "Hello", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "M", "World", image_set_number=2)
        np.testing.assert_equal(np.array(m.get_image_numbers()),
                                np.arange(1, 3))
        m.remove_measurement(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER, 1)
        np.testing.assert_equal(np.array(m.get_image_numbers()), np.array([2]))

    def test_11_00_match_metadata_by_order_nil(self):
        m = cpmeas.Measurements()
        result = m.match_metadata(("Metadata_foo", "Metadata_bar"),
                                  (np.zeros(3), np.zeros(3)))
        self.assertEqual(len(result), 3)
        self.assertTrue(all([len(x) == 1 and x[0] == i + 1
                             for i, x in enumerate(result)]))

    def test_11_01_match_metadata_by_order(self):
        m = cpmeas.Measurements()
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Hello", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Hello", image_set_number=2)
        result = m.match_metadata(("Metadata_bar",), (np.zeros(2),))
        self.assertEqual(len(result), 2)
        self.assertTrue(all([len(x) == 1 and x[0] == i + 1
                             for i, x in enumerate(result)]))

    def test_11_02_match_metadata_equal_length(self):
        m = cpmeas.Measurements()
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Hello", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "Metadata_bar", "World", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Goodbye", image_set_number=2)
        m.add_measurement(cpmeas.IMAGE, "Metadata_bar", "Phobos", image_set_number=2)
        result = m.match_metadata(("Metadata_foo", "Metadata_bar"),
                                  (("Goodbye", "Hello"), ("Phobos", "World")))
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0], 2)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1][0], 1)

    def test_11_03_match_metadata_different_length(self):
        m = cpmeas.Measurements()
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Hello", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "Metadata_bar", "World", image_set_number=1)
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Goodbye", image_set_number=2)
        m.add_measurement(cpmeas.IMAGE, "Metadata_bar", "Phobos", image_set_number=2)
        m.add_measurement(cpmeas.IMAGE, "Metadata_foo", "Hello", image_set_number=3)
        m.add_measurement(cpmeas.IMAGE, "Metadata_bar", "Phobos", image_set_number=3)
        result = m.match_metadata(("Metadata_foo",),
                                  (("Goodbye", "Hello"),))
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0], 2)
        self.assertEqual(len(result[1]), 2)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][1], 3)

    def test_12_00_load_version_0(self):
        data = ('eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KTiig8mHyCVCaEUp3QOkVTDBxRrCc'
                'BFRcEGo+urqQIFdXkOr/aABmzwNWCM3BMApGIvBwdQwA0RFQPiw9nWBCVeeb'
                'mlhcWpSam5pXUgziw9KlA5n24kq/HswQfgY0XRJKvy84IfRo+h2ZAD39wkrZ'
                'DlZUdZ65iempDIh060GhvcF+/i6gFAxLdwrMFBo4QgGuciCAHcIvgOZvQuXA'
                'B14IDTNnFIwsgF4OKEDpCZyo6sDlgF9pblJqEZiflp8PpmHlggWJ9qKXAxrs'
                'JBowCsAAVzkQwQ3hV0DzN6FywANqAGprexSMFICrPTuDF1VdSmJJIojOzEtJ'
                'rWCgvD0LKQeY4OWAATd+9TB3LZAi08JhChgZ2CBxAY0QVmCIgphMTExgEWZg'
                'zgYxBIABzQJWoQDVB8nxjIwwcyCAAxjSIDYzowe0SGBBsU8IGmMgNbzNhn6I'
                'HvTgAIwMCL+AaEsDVP4oGAWjYBSMglEwCoYfQG9XWvDjVw/r90yQpKWrhh4g'
                'v10JaakzwuddIAA23IdoX/pAg54HxV7k9qU8uH3ZQEVfkQ9w9bdPQPsjL2Qg'
                'NKH+doEchB7tb49MgKu/zSGLqo7a/W3a9RMjCPYTwwdhPxG9nkiQxascnl93'
                'yNPSVUMP0L6eiCFYT+QMonoCAIv/qlY=')
        data = zlib.decompress(base64.b64decode(data))
        fid, name = tempfile.mkstemp(".h5")
        m = None
        try:
            fd = os.fdopen(fid, "wb")
            fd.write(data)
            fd.close()
            m = cpmeas.load_measurements(name)
            self.assertEqual(tuple(m.get_image_numbers()), (1,))
            self.assertEqual(m.get_measurement(cpmeas.IMAGE, "foo",
                                               image_set_number=1), 12345)
        finally:
            if m is not None:
                del m
            os.unlink(name)

    def test_12_01_load_version_1(self):
        data = ('eJzt3E9sFFUcwPH3Zrft0lIpDUGCBWqiUg8124ZAQQslUqwxtkSQlAvplu7a'
                'Ymlr/yQ9mWKIKZgoRw5GS2JMI2rqQSFeEE94MZyM8WBKwMDBIkFDMGDrzr73'
                'dneGzi4sq7vd/X44/Hbe/HZmOrx582aT+R1v27W7snxtubAFAsIvqkSyBW3/'
                'UelYNuu7dJQ6Tuo4bZl2GVu3Rrev1Nt35+17rbXVzl5wMfsJ+lUMCBSjttad'
                'e+zYqZdrdbxoOfP2h4dH+gYHxKvh0MjYcPhIeGB0xG43/bPpIfcrRan6rt5A'
                'SbQH2h8ty4q1+KL92f5QFe2Y/qQjk7qfS2m2owSiV4L92Se7dFf2O/ZXrXu4'
                'nfPr8Yb2xJWSH/a2d+ySwoqPEuMlqfPN9Vrr+y+PaunxGhen9Pn8rkzFdOPi'
                'nWoVGReLk9e4OFvmzGsMNjTUB7fUNzbVNzTaHzY1xtozHxcX779demCYqDZ5'
                '0THi4NwFWSnEcmvq7vzpucsT9+zWK+U75oX/czH7ijjstbUvdbf+Uc1P0l4N'
                '+5er6JzDoFi4rwbTn26UO/N6QqMhO/YN9ITHReIqaMlwv1KULT5L8Ik0swR1'
                'hFJPE8xkxvTriuh1aa+ReoUV264UI71jkUh/OD53kLq9JxzpD40m2uOzDcvq'
                '/FC3+ZOmHMmzDd8Je7aR6RlwMrMEc1faXZE631yvH1VmZfcFw3P2mbZf1alo'
                'OfuVmYSZ/mWWE/3EV9ut2/wqPbbZ+/vJUJb/0gcTu5Msi36wpq4vXH/72C+l'
                '0c9NWzrmRUl01cUnpQgGxMxL32+v+G3yxEarf/j8+qF9V2uit6UzpU81yLDn'
                '0+gnann1ChXd40hQx80rnMfTHRqOxZePhN4It48d6Q6r5Y7uw+FDo4kGM77U'
                'pfn7vI5v+lO1/I3H8Zn8n1zHl+1xzj7JDStFbUC0bH3vqwMnjz3z1sZ37n3w'
                '8bWr1u5vXz+7Ifji7+v+ERPnm5uX+VXfSnfHFo+rwB27OHn15KlqZ17sCovG'
                '9rFD/eG+R+/J6g4l43eoJh6gMuI1YvWuUsuTq1VMNw4EntDxfzpu5Bev59jp'
                '1c68yOBgLLrvuJk+x7pnqi2rUueb+9SNxx5yRwXOaxwY0g0n16qYbhzoWqci'
                '84Hi5DUfmFnrzMv2zNb9u/a7aW5EZnXbmpRpRWcJ/BJy84Buy8UvIXtqUueb'
                '/je9Piu7Lxg5+CXk3Abdlo+/hNj3UbnYfTSY+ns3n3U16CvA51wEAABAgZBJ'
                '0UqKAAAAAAAAAAAAAAAAAAAAAAAgf3i9N/q0R/4P7joXfkcAAAAFKNM6E539'
                'rgbqTAAAABQ06kwAAAAAAAAAAAAAAAAAAAAAAJD/vN4brfHI7wm4GqgzAQBA'
                'wbPnC2KxOhPaxWkV3dMEFId0/aP2jIpVuTpA5NTe9o5dUvji//+zf6TOr9Vx'
                'qEmmSouPN++veISDW0KkKFPvbuvTUhI9A7F3t32qxT7D9oeqgHk2q9XfW6Oi'
                'VF80z23mOq0QTbHvSf0SuBXbrhQjvWORSH84ni91e0840h8aTbQHotu3v++z'
                'rJlrus2f9HhYrf+n7JwdJxrahWjJyvlQ/cqK94Oqz1Lnm/7X+3lWdl8wpCjN'
                'sF/VqWipL5oaAqaOlOlfZjnRT3zjX+g2v0qPbfb+fjKU5b/0wWRaV2vmrKuB'
                'uloAAAAFzUqKpqYWcz4AAAAAAAAAAAAAAAAAAAAAAPKL13uj9R75z7kLIlBX'
                'CwCAgidj/4SuziLESl2hJHBTLdfdUjFdfaWZP1WkvlJxamvduceOnXrZ9Ke2'
                'W868ntBoyI59Az3hcREv95JxNZ4lUI+oJaiPLRf1iKZupUyPX693/srK7gtG'
                'DuoRBW7rtkKqR9QSv8I16hEBAAAUNOoRAQAAAAAAAAAAAAAAAAAAAACQ/7ze'
                'G13nkV9zdPH3RXlnAACAwuVVj+hUk5oXnNumYrp6RC0vqDzqERUnr3pEP29z'
                'zi+LsB7RpfEc1iMKPi9T5pvrdbI5dV6xyUE9olPbE/2kYOoRXWqjHhEAAEAx'
                'oR4RAAAAAAAAAAAAAAAAAAAAAAD572HrEW2lHhEAAEVn/ODcBVkpxHJr6u78'
                '6bnLE/fs+cCV8h3zwp+07s2FS7fvTGzeGl03+7Uc/FuU3bdyU7NeeVuU/Qv/'
                'J+C3')
        data = zlib.decompress(base64.b64decode(data))
        fid, name = tempfile.mkstemp(".h5")
        m = None
        try:
            fd = os.fdopen(fid, "wb")
            fd.write(data)
            fd.close()
            m = cpmeas.load_measurements(name)
            self.assertEqual(tuple(m.get_image_numbers()), (1, 2))
            self.assertEqual(m.get_measurement(cpmeas.IMAGE, "foo",
                                               image_set_number=1), 12345)
            self.assertEqual(m.get_measurement(cpmeas.IMAGE, "foo",
                                               image_set_number=2), 23456)
            for image_number, expected in ((1, np.array([234567, 123456])),
                                           (2, np.array([123456, 234567]))):
                values = m.get_measurement("Nuclei", "bar",
                                           image_set_number=image_number)
                self.assertEqual(len(expected), len(values))
                self.assertTrue(np.all(expected == values))
        finally:
            if m is not None:
                del m
            os.unlink(name)

    def test_13_01_load_matlab(self):
        data = ('eJztHEtzHEd5pCghVkhi3gmVQ5NylS2XJe3KWlueCpFkSbYV9CpLsRMss+nd'
                '6d3teF7MQ7tLTBXHcM2JOweuUFw4pfIDuFHAkRTwA3wCDqSK7+vu2XnsrPYh'
                'aYuAxrX2dvf36u/VX/f0env1YGv1NinNFcj26sFsjZuM7Jk0qDmepRM7uEbW'
                'PEYDZhDH1slBIySrYZ0UFkmxoC8s6cUlslAoFrXRnonN7Vfw33c07QX450X4'
                'TKqh51V7IvH5Cnwa1DZM5sP4lPay6l+Ez1roecwOJPI2o37oMQs6fG3PYzUG'
                'Y1XA0vZZEHC77ndEQP73pgbjL/untO+o9hfw2QmtCvN2a5sWrTOgrijDt9sM'
                'GK3a1Gz/hBkKO4LedowQZ6GefXrEdsPADYN7TnO3FjBbdB9wi+0H1As6+BIa'
                'u4B2xDLZHbrr3GPVwPHaqnud1WhoBpJ8PKi6BY0EimgfOI7p3wFPsKklhex0'
                '32OmmzIh6m9Ji/X3Qh/9XVD96lk5DfxCAv+rA9hvUsk9LN6EwttL4E3l4H03'
                'gXdRtTFKZgtLs3Hk3Jy7daNUKBRiPY4qT1J/efK8qKXlwfaafhh4of1E+H8f'
                '/G9m8LFdBXxmuYcVahxyqz5bXFwqRPM4bXpugt5EDr2kXhB+pQ//lzL8sS08'
                'nATo+bn2yPpl9CT9MsLbe2O4fBLJ8z34bLRc5nHMXFIkCbUTVk3GE2wFH/Lq'
                'cHy+rtqXUEbuMpPbrBx9Ac48KEMSCUI/w+feK8Pp/7VXjtf/319O6x/ba8w0'
                '9zwH1x+PRDLppBEErj4/32w256oA4SqIOcerTz9gns9hVVqY3n+wc58dcdEq'
                'FhcLi9PTWw41hAJ9/ZElsm3ZDi29+NQ/sstHCvPwcrF4/Vbp8PLTI+pxWgEg'
                'T9FBaEjVQO6p33Ca5Sa3DaepH3ghe2o7AZA9bJVuw2f9aYUG1UbZB80xnXoe'
                'bV+Jhq4RI2i77Psht4OlmcfTBB7MqwR7wdlIhRETBGWGDuT5ETdCahIuxI6B'
                'fWZCegaZiMWChmPoB6wVzG60aDUgFvIWoHJpIU5N4RNuE0arDVL3nNBd1q8L'
                'qAPBuAEfoAFfaCBarFU1QxAjwm3AgoQEqo5lgZ7WHQJzJqHPBBG1qBFqmsQP'
                'KzXHNEChpMmDBuAgPSky0JNjy/qOY0vcTRuWIcmGiGLDdKoUJ6erFUlB3BGI'
                'h62bax3UtQarPlGoPqy0QNwjFvd9WAsJfDVC1+RATBL2l/X3lRbvogaiqVXa'
                'qEZq0ICiWAJgQ06fVJkXUJiCwlej+y6r8lqMBsPMNHy0n9AtkNQloJh1UiUA'
                'QqWyJITMMlUntAO9KA2StIPP8g1ArlSpz2Z9Zvs84EdsRvdhpeeGoLDnYCcA'
                'genBAL7SUMr8itlGK/CE13Rm4jkWaTagPEqY6D6rhyb1wClcj/kYDFLAGvio'
                'H+MqUwsjYqGg/+jK8t5bWDuyt+euzpSx9RBi9m2MhtXZPQwI/FqYvYVfP1r4'
                '6UzZR6B9DhiJkZm0p3pDidNRPnFp0NDnropYVM88kkeW60rI3MH7oY1jl5TT'
                'UdtmZspm0p+Qm+UccVCBh3VSx1/EMPqkCtdNO2CeycCiRiZSq5K4TyDnK0PJ'
                'MMXsJRhwEQzUj/wC3NypfAheBuw24zyxAwJI48t0IsH19Z3V/GFJQ5fLijK6'
                '8HwnDDDx+ihdCjaeniLGIljFa0tAC5l2FZGUAqN8qrj5VWri9ALp01G4Tm8a'
                'sPRBtO3BIki99q5knsrhC905/HrxuBx+65RTuIrz2D652mbElXOINKgSPpcz'
                '5JD0E/oHd4fcZRKDA3Yg3UPhXUPfdnkLlBjaHOhc2eb2tW3amtGLhWuLBYG/'
                'zkGjXsdaaBwfOAk5OjQ9atdZx5AHIBqIZDFPVj7EtzCjRxQwnRObUQ8yJsRf'
                'nXldrpBlGjhhtYGej1wrjmfIeQhFoYrilJxQYdCAsG5AzCZCZjfwQ3LXdCrU'
                'lKJGMBCHnqdWwxrFTYvyqC2nCcxgc0hCF6OpAvEK6UEkC4WrF+YK4rlWVF/k'
                'auZCVdEC+QIRydUol3KVq8HRIAoq7Xj2QEcy3RbSohIN7uOWLOR+g8BaYrlx'
                '4Ij4RydvZ3E82iRi1ceZy7CrsKDJmN2Hxj6H1RdE9C3HCYTCIQsHGF1ySvug'
                'A8yVYnE1oUjA6cmsST2Yk+n4oCJo2zIrWNzmVmjhgg5qNMRsKGyY9ZvRAshQ'
                'r6iEUKy2CObNAgvHDIW+uCqfRVZOsY1t3iNxRJGeTBpQ95gEbCbLmDhiYhtE'
                'RFfDwAHTYeyYbQJ/V0NcgYjfQ0X9MXspo2Ob/NltUxuLt5S3JV29wm3MBXLO'
                'ibW2K+2KaMmZsYq5XmKnQkmZCftEKMmwioXdwCWJbFHXpFVOhbvfpSEsrjQd'
                'LyUZfU1ntmpSXyw/OMxUMxm6yzqAEdEfaQQViWZAMZqM1xtYEYosjeoUtKD0'
                'tAPPcdvL+sMshJwuCFVX6c+PFnmLG0Zi9WgTJY4jKxIHSgYP4z/iUaHVJ7Jn'
                'Wb/TGe2rzsiHovRWU+Ty9Zb1L6Vj564a6aRhXWr1Hp5kidI15sBrWIpjlQPl'
                'n92pEzrD8dKx5sA3O2TK91rCZbsw9FKhkM23Vnw4JjUW+Qwm/Ngz4zwVKwUq'
                'LRfrTyKXUqEgWYaIr6nclIHF1BTtDxvacPvJkc43HMs1QduI/6/p4fbJl1X7'
                '00mkA3VfuXv/nXlEzVeWld0xYJ11rHwH0oEZN9MMttdL6xxqu6AMNUVvavdE'
                'xBwP05Ft0zZY6ziwWJgoEh+oQExJh1tSzOR92O5B9T0AWMx0P7R2axsiGUA1'
                'mFHIRotVxUKDR6LlQjHe3aeoPeRG0OirjwTTXY/Xexkhy3QhvzTV5Fnuhuc5'
                'Xm8gTXv3/lZ/yTRtH6INkkI/0BTTXurA56TxNtx57LNlxP9gSH6jnnuOY15/'
                '+vG/v3hV+7WY18+G5NfvnJho6byF7VK1xJboYmmpslC7sUhLxVKhyG5cv1Uq'
                'FW4slYzFceh3S+GdNZ9x2lFL4J/2efK0lrYjtuUBzZxr12M6rSH5nvY5+jj0'
                '/MmFvzyr/+H1z8Zm19//9aPSP341lrzz7hj1mMw7Y9Hje5+89ed//nYseozw'
                'Ph6Sz3va8fFwOQF/UbXxeFKfn59f0+cxLuYhLuajuJhPBOlY89Dnzso49fzZ'
                'xHD177dUe1mL3liXZR0TFbhb6sy+vMZwk1N+HzGzne91+J/1PMe5jvxwZueP'
                'P59aWxkXv998+2+//ME7q4Lfs+dGv6ewhxto3JtpqUe9dJHlZNftgEEvDUT7'
                '8DXHdLxtqm4IbNEKM9Nd+OwHHnc7bx3j7ifcFfVsuoZd5z5sttsgIHtAzTBC'
                'uAPb3+7ZRNAPxZYzReg06toV7fj8czGBf1F9ovfsh+rKx5fhfXuhD72pFL0p'
                're7R9iB4z6XwntM+ZMFA/LJ478M+ZxC8yRTepLbjnCy/DIs3rXX7XRZvsgsv'
                'TQ/xP780XNy/ptrP4PNAvRQRwZOIiKh/0645+KotGpJuuhNd/Ok8MvP7u7UI'
                'UQ33yCsd+tF7eYWv6Gd7O3zxtUyKzn7Dad7B92sZ+rfxjQ3eVsgMoL5+d7F3'
                'Xkb9vZnoHyR+vqal4wfbXe/rE/SG9Wfc64+aX7J3AQY5L3shQwfb0dnHIPJP'
                'pPAntOvaYOd0z2f4Ylu8wdJG32fFVxMG1/8geWGQvIdnpSfh1+984PXMfLGd'
                'd0viKQqiDT7/UfNpL7yVPnh5960Sb/gGlntUu2X9tXhCfqPki653j2r8JPfM'
                'hvVXfG95En4rffhdyMwb253acAi5e9nrtPiPkh/lVnE0vykWztZPs3g3B8Qb'
                'NZ/1ygONPng3M3rFdq9LQ49WZ/cePyrM3nqcuSgk+mYG8odXM/ywnXnbPNC5'
                'wqUMHWzPXX10eDj/OHmZqNMRXSDSBrTDqPl4VH85y3r6rNeNxGUqbVT9Qh48'
                'UZ0zalyNknei1zrjtE9Uj45q17OuP7vtWTqRflb64OX5YeKyg+gfpC74RoYO'
                'tnOuzQ1ML6/O6LpFocbPct3Nq8vjaxaD8z/tuvSs6qqsnko5eMPwG+1+g7z/'
                'cZbzzOq1VOjOm2dZr+bVI6dZr+b5bXybJV/u05xvrzrx48TvOiYyeHnnF+PU'
                'jzjsEFeVT85/lPwmLzbJe/fcNpiboHcadjrHO8c7xzvHO8f778VbSeANek4e'
                'r1ty2fgyzfccb7z+c9Z1z5fV7/9f67WVPvM+zzfneOd4/7t4FydivOx+PLuP'
                'R/gPEnzy8sVVLZ0vsJ363w3kj139OfwFrrxdMJe94o98Wn343M7wud2Lj/pd'
                'UVv9YFX9eGiu528ZcvR5IYd/Ui+T8OeNN0/nHtawfF+60C3vS33wpoa4p5q1'
                '/5Vj4KPnJPDDzn9iQuL9Ysh5HMcnK9fkCfH+A86LP/E=')
        data = zlib.decompress(base64.b64decode(data))
        fid, name = tempfile.mkstemp(".mat")
        m = None
        try:
            fd = os.fdopen(fid, "wb")
            fd.write(data)
            fd.close()
            m = cpmeas.load_measurements(name)
            self.assertEqual(tuple(m.get_image_numbers()), (1,))
            self.assertEqual(m.get_measurement(cpmeas.IMAGE, "Count_Nuclei",
                                               image_set_number=1), 1)
            values = m.get_measurement("Nuclei", "Location_Center_X", 1)
            self.assertEqual(len(values), 1)
            self.assertAlmostEqual(values[0], 34.580433355219959)
        finally:
            if m is not None:
                del m
            os.unlink(name)

    def test_15_01_get_object_names(self):
        '''Test the get_object_names() function'''
        m = cpmeas.Measurements()
        try:
            m.add_measurement(OBJECT_NAME, "Foo", np.zeros(3))
            object_names = m.get_object_names()
            self.assertEqual(len(object_names), 2)
            self.assertTrue(OBJECT_NAME in object_names)
            self.assertTrue(cpmeas.IMAGE in object_names)
        finally:
            del m

    def test_15_02_get_object_names_relationships(self):
        '''Regression test - don't return Relationships'''
        m = cpmeas.Measurements()
        try:
            m.add_image_measurement(cpmeas.GROUP_NUMBER, 1)
            m.add_image_measurement(cpmeas.GROUP_INDEX, 0)
            m.add_measurement(OBJECT_NAME, "Foo", np.zeros(3))
            m.add_relate_measurement(1, "Foo", OBJECT_NAME, OBJECT_NAME,
                                     np.zeros(3), np.zeros(3),
                                     np.zeros(3), np.zeros(3))
            object_names = m.get_object_names()
            self.assertEqual(len(object_names), 2)
            self.assertTrue(OBJECT_NAME in object_names)
            self.assertTrue(cpmeas.IMAGE in object_names)
        finally:
            del m

    def test_15_03_relate_no_objects(self):
        # regression test of issue #886 - add_relate_measurement called
        #                                 with no objects
        m = cpmeas.Measurements()
        m.add_image_measurement(cpmeas.GROUP_NUMBER, 1)
        m.add_image_measurement(cpmeas.GROUP_INDEX, 0)
        m.add_measurement(OBJECT_NAME, "Foo", np.zeros(3))
        m.add_relate_measurement(1, "Foo", OBJECT_NAME, OBJECT_NAME,
                                 np.zeros(3), np.zeros(3),
                                 np.zeros(3), np.zeros(3))
        m.add_relate_measurement(1, "Foo", OBJECT_NAME, OBJECT_NAME,
                                 np.zeros(0), np.zeros(0),
                                 np.zeros(0), np.zeros(0))

    def test_16_01_get_feature_names(self):
        m = cpmeas.Measurements()
        try:
            m.add_measurement(OBJECT_NAME, "Foo", np.zeros(3))
            feature_names = m.get_feature_names(OBJECT_NAME)
            self.assertEqual(len(feature_names), 1)
            self.assertTrue("Foo" in feature_names)
        finally:
            del m

    def test_16_01b_get_feature_names_arrayinterface(self):
        m = cpmeas.Measurements()
        try:
            m[OBJECT_NAME, "Foo"] = np.zeros(3)
            feature_names = m.get_feature_names(OBJECT_NAME)
            self.assertEqual(len(feature_names), 1)
            self.assertTrue("Foo" in feature_names)
        finally:
            del m

    def test_17_01_aggregate_measurements(self):
        m = cpmeas.Measurements()
        try:
            values = np.arange(5).astype(float)
            m.add_measurement(OBJECT_NAME, "Foo", values)
            d = m.compute_aggregate_measurements(1)
            for agg_name, expected in (
                    (cpmeas.AGG_MEAN, np.mean(values)),
                    (cpmeas.AGG_MEDIAN, np.median(values)),
                    (cpmeas.AGG_STD_DEV, np.std(values))):
                feature = "%s_%s_Foo" % (agg_name, OBJECT_NAME)
                self.assertTrue(d.has_key(feature))
                self.assertAlmostEqual(d[feature], expected)
        finally:
            del m

    def test_17_02_aggregate_measurements_with_relate(self):
        '''regression test of img-1554'''
        m = cpmeas.Measurements()
        try:
            values = np.arange(5).astype(float)
            m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1, image_set_number=1)
            m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1, image_set_number=2)
            m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX, 1, image_set_number=1)
            m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX, 2, image_set_number=1)
            m.add_measurement(OBJECT_NAME, "Foo", values, image_set_number=1)
            m.add_measurement(OBJECT_NAME, "Foo", values, image_set_number=2)
            m.add_relate_measurement(1, "R", "A1", "A2",
                                     np.array([1, 1, 1, 1, 1], int),
                                     np.array([1, 2, 3, 4, 5], int),
                                     np.array([2, 2, 2, 2, 2], int),
                                     np.array([5, 4, 3, 2, 1], int))
            d = m.compute_aggregate_measurements(1)
            for agg_name, expected in (
                    (cpmeas.AGG_MEAN, np.mean(values)),
                    (cpmeas.AGG_MEDIAN, np.median(values)),
                    (cpmeas.AGG_STD_DEV, np.std(values))):
                feature = "%s_%s_Foo" % (agg_name, OBJECT_NAME)
                self.assertTrue(d.has_key(feature))
                self.assertAlmostEqual(d[feature], expected)
        finally:
            del m

    def test_18_01_test_add_all_measurements_string(self):
        m = cpmeas.Measurements()
        try:
            values = ["Foo", "Bar", "Baz"]
            m.add_all_measurements(cpmeas.IMAGE, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(cpmeas.IMAGE, FEATURE_NAME,
                                          image_set_number=i + 1)
                self.assertEqual(expected, value)
        finally:
            del m

    def test_18_02_test_add_all_measurements_unicode(self):
        m = cpmeas.Measurements()
        try:
            values = [u"Foo", u"Bar", u"Baz", u"-\u221E < \u221E"]
            m.add_all_measurements(cpmeas.IMAGE, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(cpmeas.IMAGE, FEATURE_NAME,
                                          image_set_number=i + 1)
                self.assertEqual(expected, value)
        finally:
            del m

    def test_18_03_test_add_all_measurements_number(self):
        m = cpmeas.Measurements()
        try:
            r = np.random.RandomState()
            r.seed(1803)
            values = r.randint(0, 10, size=5)
            m.add_all_measurements(cpmeas.IMAGE, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(cpmeas.IMAGE, FEATURE_NAME,
                                          image_set_number=i + 1)
                self.assertEqual(expected, value)
        finally:
            del m

    def test_18_04_test_add_all_measurements_nulls(self):
        m = cpmeas.Measurements()
        try:
            values = [u"Foo", u"Bar", None, u"Baz", None, u"-\u221E < \u221E"]
            m.add_all_measurements(cpmeas.IMAGE, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(cpmeas.IMAGE, FEATURE_NAME,
                                          image_set_number=i + 1)
                if expected is None:
                    self.assertTrue(value is None)
                else:
                    self.assertEqual(expected, value)
        finally:
            del m

    def test_18_05_test_add_all_per_object_measurements(self):
        m = cpmeas.Measurements()
        try:
            r = np.random.RandomState()
            r.seed(1803)
            values = [r.uniform(size=5), np.zeros(0), r.uniform(size=7),
                      np.zeros(0), r.uniform(size=9), None, r.uniform(size=10)]
            m.add_all_measurements(OBJECT_NAME, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(OBJECT_NAME, FEATURE_NAME,
                                          image_set_number=i + 1)
                if expected is None:
                    self.assertEqual(len(value), 0)
                else:
                    np.testing.assert_almost_equal(expected, value)
        finally:
            del m

    def test_19_01_load_image_sets(self):
        expected_features = [cpmeas.GROUP_NUMBER, cpmeas.GROUP_INDEX,
                             "URL_DNA", "PathName_DNA", "FileName_DNA"]
        expected_values = [[1, 1, "file://foo/bar.tif", "/foo", "bar.tif"],
                           [1, 2, "file://bar/foo.tif", "/bar", "foo.tif"],
                           [2, 1, "file://baz/foobar.tif", "/baz", "foobar.tif"]]

        data = """"%s","%s","URL_DNA","PathName_DNA","FileName_DNA"
1,1,"file://foo/bar.tif","/foo","bar.tif"
1,2,"file://bar/foo.tif","/bar","foo.tif"
2,1,"file://baz/foobar.tif","/baz","foobar.tif"
""" % (cpmeas.GROUP_NUMBER, cpmeas.GROUP_INDEX)
        m = cpmeas.Measurements()
        try:
            m.load_image_sets(StringIO(data))
            features = m.get_feature_names(cpmeas.IMAGE)
            self.assertItemsEqual(features, expected_features)
            for i, row_values in enumerate(expected_values):
                image_number = i + 1
                for value, feature_name in zip(row_values, expected_features):
                    self.assertEqual(value, m.get_measurement(
                            cpmeas.IMAGE, feature_name, image_set_number=image_number))
        finally:
            m.close()

    def test_19_02_write_and_load_image_sets(self):
        m = cpmeas.Measurements()
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, [1, 1, 2])
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_INDEX, [1, 2, 1])
        m.add_all_measurements(
                cpmeas.IMAGE, "URL_DNA",
                ["file://foo/bar.tif", "file://bar/foo.tif", "file://baz/foobar.tif"])
        m.add_all_measurements(
                cpmeas.IMAGE, "PathName_DNA", ["/foo", "/bar", "/baz"])
        m.add_all_measurements(
                cpmeas.IMAGE, "FileName_DNA", ["bar.tif", "foo.tif", "foobar.tif"])
        m.add_all_measurements(
                cpmeas.IMAGE, "Metadata_test",
                ["quotetest\"", "backslashtest\\", "unicodeescapetest\\u0384"])
        m.add_all_measurements(
                cpmeas.IMAGE, "Metadata_testunicode",
                [u"quotetest\"", u"backslashtest\\", u"unicodeescapetest\u0384"])
        m.add_all_measurements(
                cpmeas.IMAGE, "Metadata_testnull",
                ["Something", None, "SomethingElse"])
        m.add_all_measurements(
                cpmeas.IMAGE, "Dont_copy", ["do", "not", "copy"])
        fd = StringIO()
        m.write_image_sets(fd)
        fd.seek(0)
        mdest = cpmeas.Measurements()
        mdest.load_image_sets(fd)
        expected_features = [
            feature_name for feature_name in m.get_feature_names(cpmeas.IMAGE)
            if feature_name != "Dont_copy"]
        self.assertItemsEqual(expected_features, mdest.get_feature_names(cpmeas.IMAGE))
        image_numbers = m.get_image_numbers()
        for feature_name in expected_features:
            src = m.get_measurement(cpmeas.IMAGE, feature_name, image_numbers)
            dest = mdest.get_measurement(cpmeas.IMAGE, feature_name, image_numbers)
            self.assertSequenceEqual(list(src), list(dest))

    def test_19_03_delete_tempfile(self):
        m = cpmeas.Measurements()
        filename = m.hdf5_dict.filename
        del m
        self.assertFalse(os.path.exists(filename))

    def test_19_04_dont_delete_file(self):
        fd, filename = tempfile.mkstemp(suffix=".h5")
        m = cpmeas.Measurements(filename=filename)
        os.close(fd)
        del m
        self.assertTrue(os.path.exists(filename))
        os.unlink(filename)

    def test_20_01_add_one_relationship_measurement(self):
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(2001)
        image_numbers1, object_numbers1 = [
            x.flatten() for x in np.mgrid[1:4, 1:10]]
        order = r.permutation(len(image_numbers1))
        image_numbers2, object_numbers2 = [
            x[order] for x in image_numbers1, object_numbers1]

        m.add_relate_measurement(1, "Foo", "O1", "O2",
                                 image_numbers1, object_numbers1,
                                 image_numbers2, object_numbers2)
        rg = m.get_relationship_groups()
        self.assertEqual(len(rg), 1)
        self.assertEqual(rg[0].module_number, 1)
        self.assertEqual(rg[0].relationship, "Foo")
        self.assertEqual(rg[0].object_name1, "O1")
        self.assertEqual(rg[0].object_name2, "O2")
        r = m.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key] for key in
            cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
            cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
        order = np.lexsort((ro1, ri1))
        np.testing.assert_array_equal(image_numbers1, ri1[order])
        np.testing.assert_array_equal(object_numbers1, ro1[order])
        np.testing.assert_array_equal(image_numbers2, ri2[order])
        np.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_02_add_two_sets_of_relationships(self):
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(2002)
        image_numbers1, object_numbers1 = [
            x.flatten() for x in np.mgrid[1:4, 1:10]]
        order = r.permutation(len(image_numbers1))
        image_numbers2, object_numbers2 = [
            x[order] for x in image_numbers1, object_numbers1]

        split = int(len(image_numbers1) / 2)
        m.add_relate_measurement(
                1, "Foo", "O1", "O2",
                image_numbers1[:split], object_numbers1[:split],
                image_numbers2[:split], object_numbers2[:split])
        m.add_relate_measurement(
                1, "Foo", "O1", "O2",
                image_numbers1[split:], object_numbers1[split:],
                image_numbers2[split:], object_numbers2[split:])
        r = m.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key] for key in
            cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
            cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
        order = np.lexsort((ro1, ri1))
        np.testing.assert_array_equal(image_numbers1, ri1[order])
        np.testing.assert_array_equal(object_numbers1, ro1[order])
        np.testing.assert_array_equal(image_numbers2, ri2[order])
        np.testing.assert_array_equal(object_numbers2, ro2[order])

    # def test_20_03_add_many_different_relationships(self):
    #     m = cpmeas.Measurements()
    #     r = np.random.RandomState()
    #     r.seed(2003)
    #     image_numbers1, object_numbers1 = [
    #         x.flatten() for x in np.mgrid[1:4, 1:10]]
    #     module_numbers = [1, 2]
    #     relationship_names = ["Foo", "Bar"]
    #     first_object_names = ["Nuclei", "Cells"]
    #     second_object_names = ["Alice", "Bob"]
    #     d = {}
    #     midxs, ridxs, on1idxs, on2idxs = [
    #         x.flatten() for x in np.mgrid[0:2, 0:2, 0:2, 0:2]]
    #     for midx, ridx, on1idx, on2idx in zip(midxs, ridxs, on1idxs, on2idxs):
    #         key = (module_numbers[midx], relationship_names[ridx],
    #                first_object_names[on1idx], second_object_names[on2idx])
    #         order = r.permutation(len(image_numbers1))
    #         image_numbers2, object_numbers2 = [
    #             x[order] for x in image_numbers1, object_numbers1]
    #         d[key] = (image_numbers2, object_numbers2)
    #         m.add_relate_measurement(key[0], key[1], key[2], key[3],
    #                                  image_numbers1, object_numbers1,
    #                                  image_numbers2, object_numbers2)
    #
    #     rg = [(x.module_number, x.relationship, x.object_name1, x.object_name2)
    #           for x in m.get_relationship_groups()]
    #     self.assertItemsEqual(d.keys(), rg)
    #
    #     for key in d:
    #         image_numbers2, object_numbers2 = d[key]
    #         r = m.get_relationships(key[0], key[1], key[2], key[3])
    #         ri1, ro1, ri2, ro2 = [
    #             r[key] for key in
    #             cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
    #             cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
    #         order = np.lexsort((ro1, ri1))
    #         np.testing.assert_array_equal(image_numbers1, ri1[order])
    #         np.testing.assert_array_equal(object_numbers1, ro1[order])
    #         np.testing.assert_array_equal(image_numbers2, ri2[order])
    #         np.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_04_saved_relationships(self):
        #
        # Test closing and reopening a measurements file with
        # relationships.
        #
        fd, filename = tempfile.mkstemp(suffix=".h5")
        m = cpmeas.Measurements(filename=filename)
        os.close(fd)
        try:
            r = np.random.RandomState()
            r.seed(2004)
            image_numbers1, object_numbers1 = [
                x.flatten() for x in np.mgrid[1:4, 1:10]]
            module_numbers = [1, 2]
            relationship_names = ["Foo", "Bar"]
            first_object_names = ["Nuclei", "Cells"]
            second_object_names = ["Alice", "Bob"]
            d = {}
            midxs, ridxs, on1idxs, on2idxs = [
                x.flatten() for x in np.mgrid[0:2, 0:2, 0:2, 0:2]]
            for midx, ridx, on1idx, on2idx in zip(midxs, ridxs, on1idxs, on2idxs):
                key = (module_numbers[midx], relationship_names[ridx],
                       first_object_names[on1idx], second_object_names[on2idx])
                order = r.permutation(len(image_numbers1))
                image_numbers2, object_numbers2 = [
                    x[order] for x in image_numbers1, object_numbers1]
                d[key] = (image_numbers2, object_numbers2)
                m.add_relate_measurement(key[0], key[1], key[2], key[3],
                                         image_numbers1, object_numbers1,
                                         image_numbers2, object_numbers2)

            m.close()
            m = cpmeas.Measurements(filename=filename, mode="r")

            rg = [(x.module_number, x.relationship, x.object_name1, x.object_name2)
                  for x in m.get_relationship_groups()]
            self.assertItemsEqual(d.keys(), rg)

            for key in d:
                image_numbers2, object_numbers2 = d[key]
                r = m.get_relationships(key[0], key[1], key[2], key[3])
                ri1, ro1, ri2, ro2 = [
                    r[key] for key in
                    cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
                    cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
                order = np.lexsort((ro1, ri1))
                np.testing.assert_array_equal(image_numbers1, ri1[order])
                np.testing.assert_array_equal(object_numbers1, ro1[order])
                np.testing.assert_array_equal(image_numbers2, ri2[order])
                np.testing.assert_array_equal(object_numbers2, ro2[order])
        finally:
            m.close()
            os.unlink(filename)

    def test_20_05_copy_relationships(self):
        m1 = cpmeas.Measurements()
        m2 = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(2005)
        image_numbers1, object_numbers1 = [
            x.flatten() for x in np.mgrid[1:4, 1:10]]
        image_numbers2 = r.permutation(image_numbers1)
        object_numbers2 = r.permutation(object_numbers1)
        m1.add_relate_measurement(
                1, "Foo", "O1", "O2",
                image_numbers1, object_numbers1,
                image_numbers2, object_numbers2)
        m2.copy_relationships(m1)
        rg = m2.get_relationship_groups()
        self.assertEqual(len(rg), 1)
        self.assertEqual(rg[0].module_number, 1)
        self.assertEqual(rg[0].relationship, "Foo")
        self.assertEqual(rg[0].object_name1, "O1")
        self.assertEqual(rg[0].object_name2, "O2")
        r = m2.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key] for key in
            cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
            cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
        order = np.lexsort((ro1, ri1))
        np.testing.assert_array_equal(image_numbers1, ri1[order])
        np.testing.assert_array_equal(object_numbers1, ro1[order])
        np.testing.assert_array_equal(image_numbers2, ri2[order])
        np.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_06_get_relationship_range(self):
        #
        # Test writing and reading relationships with a variety of ranges
        # over the whole extent of the storage
        #
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(2005)
        image_numbers1, image_numbers2 = r.randint(1, 1001, (2, 4000))
        object_numbers1, object_numbers2 = r.randint(1, 10, (2, 4000))
        for i in range(0, 4000, 500):
            m.add_relate_measurement(
                    1, "Foo", "O1", "O2",
                    *[x[i:(i + 500)] for x in image_numbers1, object_numbers1,
                                              image_numbers2, object_numbers2])

        for _ in range(50):
            image_numbers = r.randint(1, 1001, 3)
            result = m.get_relationships(1, "Foo", "O1", "O2", image_numbers)
            ri1, ro1, ri2, ro2 = [
                result[key] for key in
                cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
                cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
            rorder = np.lexsort((ro2, ri2, ro1, ri1))
            i, j = [x.flatten() for x in np.mgrid[0:2, 0:3]]
            mask = reduce(
                    np.logical_or,
                    [(image_numbers1 if ii == 0 else image_numbers2) == image_numbers[jj]
                     for ii, jj in zip(i, j)])
            ei1, eo1, ei2, eo2 = map(
                    lambda x: x[mask], (image_numbers1, object_numbers1,
                                        image_numbers2, object_numbers2))
            eorder = np.lexsort((eo2, ei2, eo1, ei1))
            np.testing.assert_array_equal(ri1[rorder], ei1[eorder])
            np.testing.assert_array_equal(ri2[rorder], ei2[eorder])
            np.testing.assert_array_equal(ro1[rorder], eo1[eorder])
            np.testing.assert_array_equal(ro2[rorder], eo2[eorder])

    def test_21_01_load_measurements_from_buffer(self):
        r = np.random.RandomState()
        r.seed(51)
        m_in = cpmeas.Measurements()
        m_in[cpmeas.IMAGE, FEATURE_NAME, 1] = r.uniform()
        m_in[OBJECT_NAME, FEATURE_NAME, 1] = r.uniform(size=100)
        m_out = cpmeas.load_measurements_from_buffer(
                m_in.file_contents())
        self.assertAlmostEqual(m_in[cpmeas.IMAGE, FEATURE_NAME, 1],
                               m_out[cpmeas.IMAGE, FEATURE_NAME, 1])
        np.testing.assert_array_almost_equal(
                m_in[OBJECT_NAME, FEATURE_NAME, 1],
                m_out[OBJECT_NAME, FEATURE_NAME, 1])


IMAGE_NAME = "ImageName"
ALT_IMAGE_NAME = "AltImageName"
OBJECT_NAME = "ObjectName"
ALT_OBJECT_NAME = "AltObjectName"
METADATA_NAMES = ["Metadata_%d" % i for i in range(1, 10)]


if __name__ == "__main__":
    unittest.main()
