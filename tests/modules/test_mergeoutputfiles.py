'''test_mergeoutputfiles.py - test the MergeOutputFiles module'''

import os
import tempfile
import unittest

import numpy as np

import cellprofiler.measurement as cpmeas
import cellprofiler.modules.mergeoutputfiles as M
import cellprofiler.pipeline as cpp
from cellprofiler.modules.loadimages import LoadImages


class TestMergeOutputFiles(unittest.TestCase):
    def execute_merge_files(self, mm):
        input_files = []
        output_fd, output_file = tempfile.mkstemp(".mat")
        pipeline = cpp.Pipeline()
        li = LoadImages()
        li.module_num = 1
        pipeline.add_module(li)

        for m in mm:
            input_fd, input_file = tempfile.mkstemp(".mat")
            pipeline.save_measurements(input_file, m)
            input_files.append((input_fd, input_file))

        M.MergeOutputFiles.merge_files(output_file, [x[1] for x in input_files])
        m = cpmeas.load_measurements(output_file)
        os.close(output_fd)
        os.remove(output_file)
        for fd, filename in input_files:
            os.close(fd)
            os.remove(filename)
        return m

    def write_image_measurements(self, m, feature, image_count):
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for i in range(image_count):
            if i > 0:
                m.next_image_set(i + 1)
            m.add_image_measurement(feature, np.random.uniform())

    def write_object_measurements(self, m, object_name, feature, object_counts):
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for i, count in enumerate(object_counts):
            object_measurements = np.random.uniform(size=i)
            m.add_measurement(object_name, feature, object_measurements,
                              image_set_number=i + 1)

    def write_experiment_measurement(self, m, feature):
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        m.add_experiment_measurement(feature, np.random.uniform())

    def test_00_00_nothing(self):
        '''Make sure merge_files doesn't crash if no inputs'''
        M.MergeOutputFiles.merge_files("nope", [])

    def test_01_01_one(self):
        '''Test "merging" one file'''
        np.random.seed(11)
        m = cpmeas.Measurements()
        self.write_image_measurements(m, "foo", 5)
        self.write_object_measurements(m, "myobjects", "bar", [3, 6, 2, 9, 16])
        self.write_experiment_measurement(m, "baz")
        result = self.execute_merge_files([m])
        self.assertAlmostEqual(result.get_experiment_measurement("baz"),
                               m.get_experiment_measurement("baz"))
        ro = result.get_all_measurements("myobjects", "bar")
        mo = m.get_all_measurements("myobjects", "bar")
        for i in range(5):
            self.assertAlmostEqual(
                    result.get_all_measurements(cpmeas.IMAGE, "foo")[i],
                    m.get_all_measurements(cpmeas.IMAGE, "foo")[i])
            self.assertEqual(len(ro[i]), len(mo[i]))
            np.testing.assert_almost_equal(ro[i], mo[i])

    def test_01_02_two(self):
        np.random.seed(12)
        mm = []
        for i in range(2):
            m = cpmeas.Measurements()
            self.write_image_measurements(m, "foo", 5)
            self.write_object_measurements(m, "myobjects", "bar", [3, 6, 2, 9, 16])
            self.write_experiment_measurement(m, "baz")
            mm.append(m)
        result = self.execute_merge_files(mm)
        self.assertAlmostEqual(result.get_experiment_measurement("baz"),
                               mm[0].get_experiment_measurement("baz"))
        ro = result.get_all_measurements("myobjects", "bar")
        moo = [m.get_all_measurements("myobjects", "bar") for m in mm]
        for i in range(5):
            for j in range(2):
                np.testing.assert_almost_equal(
                        ro[i + j * 5],
                        moo[j][i])
            self.assertEqual(len(ro[i + j * 5]), len(moo[j][i]))
            np.testing.assert_almost_equal(ro[i + j * 5], moo[j][i])

    def test_01_03_different_measurements(self):
        np.random.seed(13)
        mm = []
        for i in range(2):
            m = cpmeas.Measurements()
            self.write_image_measurements(m, "foo", 5)
            self.write_object_measurements(m, "myobjects", "bar%d" % i, [3, 6, 2, 9, 16])
            self.write_experiment_measurement(m, "baz")
            mm.append(m)
        result = self.execute_merge_files(mm)
        self.assertAlmostEqual(result.get_experiment_measurement("baz"),
                               mm[0].get_experiment_measurement("baz"))
        for imgidx in range(10):
            imgnum = imgidx + 1
            if imgidx < 5:
                ro = result.get_measurement("myobjects", "bar0", imgnum)
                mo = mm[0].get_measurement("myobjects", "bar0", imgnum)
                np.testing.assert_almost_equal(ro, mo)
                self.assertEqual(len(result.get_measurement("myobjects", "bar1", imgnum)), 0)
            else:
                ro = result.get_measurement("myobjects", "bar1", imgnum)
                mo = mm[1].get_measurement("myobjects", "bar1", imgnum - 5)
                np.testing.assert_almost_equal(ro, mo)
                self.assertEqual(len(result.get_measurement("myobjects", "bar0", imgnum)), 0)

    def test_01_04_different_objects(self):
        np.random.seed(13)
        mm = []
        for i in range(2):
            m = cpmeas.Measurements()
            self.write_image_measurements(m, "foo", 5)
            self.write_object_measurements(m, "myobjects%d" % i, "bar", [3, 6, 2, 9, 16])
            self.write_experiment_measurement(m, "baz")
            mm.append(m)
        result = self.execute_merge_files(mm)
        self.assertAlmostEqual(result.get_experiment_measurement("baz"),
                               mm[0].get_experiment_measurement("baz"))
        for imgidx in range(10):
            imgnum = imgidx + 1
            if imgidx < 5:
                ro = result.get_measurement("myobjects0", "bar", imgnum)
                mo = mm[0].get_measurement("myobjects0", "bar", imgnum)
                np.testing.assert_almost_equal(ro, mo)
                self.assertEqual(len(result.get_measurement("myobjects1", "bar", imgnum)), 0)
            else:
                ro = result.get_measurement("myobjects1", "bar", imgnum)
                mo = mm[1].get_measurement("myobjects1", "bar", imgnum - 5)
                np.testing.assert_almost_equal(ro, mo)
                self.assertEqual(len(result.get_measurement("myobjects0", "bar", imgnum)), 0)
