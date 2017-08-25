'''test_classifyobjects - test the ClassifyObjects module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.grid as cpg
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.classifyobjects as C

OBJECTS_NAME = "myobjects"
MEASUREMENT_NAME_1 = "Measurement1"
MEASUREMENT_NAME_2 = "Measurement2"
IMAGE_NAME = "image"


class TestClassifyObjects(unittest.TestCase):
    def make_workspace(self, labels, contrast_choice,
                       measurement1=None, measurement2=None):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        measurements = cpmeas.Measurements()
        module = C.ClassifyObjects()
        m_names = []
        if measurement1 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_1,
                                         measurement1)
            m_names.append(MEASUREMENT_NAME_1)
        if measurement2 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_2,
                                         measurement2)
            module.add_single_measurement()
            m_names.append(MEASUREMENT_NAME_2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        module.contrast_choice.value = contrast_choice
        if module.contrast_choice == C.BY_SINGLE_MEASUREMENT:
            for i, m in enumerate(m_names):
                group = module.single_measurements[i]
                group.object_name.value = OBJECTS_NAME
                group.measurement.value = m
                group.image_name.value = IMAGE_NAME
        else:
            module.object_name.value = OBJECTS_NAME
            module.image_name.value = IMAGE_NAME
            module.first_measurement.value = MEASUREMENT_NAME_1
            module.second_measurement.value = MEASUREMENT_NAME_2
        module.module_num = 1
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements,
                                  image_set_list)
        return workspace, module

    def test_02_01_classify_single_none(self):
        '''Make sure the single measurement mode can handle no objects'''
        workspace, module = self.make_workspace(
                np.zeros((10, 10), int),
                C.BY_SINGLE_MEASUREMENT,
                np.zeros((0,), float))
        module.run(workspace)
        for m_name in ("Classify_Measurement1_Bin_1",
                       "Classify_Measurement1_Bin_2",
                       "Classify_Measurement1_Bin_3"):
            m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                               m_name)
            self.assertEqual(len(m), 0)

    def test_02_02_classify_single_even(self):
        m = np.array((.5, 0, 1, .1))
        labels = np.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = C.BC_EVEN
        module.single_measurements[0].low_threshold.value = .2
        module.single_measurements[0].high_threshold.value = .7
        module.single_measurements[0].bin_count.value = 1
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].wants_images.value = True

        expected_obj = dict(Classify_Measurement1_Bin_1=(0, 1, 0, 1),
                            Classify_Measurement1_Bin_2=(1, 0, 0, 0),
                            Classify_Measurement1_Bin_3=(0, 0, 1, 0))
        expected_img = dict(Classify_Measurement1_Bin_1_NumObjectsPerBin=2,
                            Classify_Measurement1_Bin_2_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_3_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_1_PctObjectsPerBin=50.0,
                            Classify_Measurement1_Bin_2_PctObjectsPerBin=25.0,
                            Classify_Measurement1_Bin_3_PctObjectsPerBin=25.0)
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)

        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cpmeas.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                        C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, cpmeas.IMAGE, "foo")
        self.assertEqual(len(names), 0)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cpmeas.IMAGE, C.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_03_classify_single_custom(self):
        m = np.array((.5, 0, 1, .1))
        labels = np.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = C.BC_CUSTOM
        module.single_measurements[0].custom_thresholds.value = ".2,.7"
        module.single_measurements[0].bin_count.value = 14  # should ignore
        module.single_measurements[0].wants_custom_names.value = True
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
        module.single_measurements[0].wants_images.value = True

        expected_img = dict(Classify_Three_NumObjectsPerBin=2,
                            Classify_Three_PctObjectsPerBin=50.0,
                            Classify_Blind_NumObjectsPerBin=1,
                            Classify_Blind_PctObjectsPerBin=25.0,
                            Classify_Mice_NumObjectsPerBin=1,
                            Classify_Mice_PctObjectsPerBin=25.0)
        expected_obj = dict(Classify_Three=(0, 1, 0, 1),
                            Classify_Blind=(1, 0, 0, 0),
                            Classify_Mice=(0, 0, 1, 0))
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cpmeas.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                        C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cpmeas.IMAGE, C.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_04_last_is_nan(self):
        # regression test for issue #1553
        #
        # Test that classify objects classifies an object whose measurement
        # is NaN as none of the categories. Test for no exception thrown
        # if showing the figure and last object has a measurement of NaN
        #
        for leave_last_out in (False, True):
            m = np.array((.5, 0, 1, np.NaN))
            if leave_last_out:
                m = m[:-1]
            labels = np.zeros((20, 10), int)
            labels[2:5, 3:7] = 1
            labels[12:15, 1:4] = 2
            labels[6:11, 5:9] = 3
            labels[16:19, 5:9] = 4
            workspace, module = self.make_workspace(
                    labels, C.BY_SINGLE_MEASUREMENT, m)
            module.single_measurements[0].bin_choice.value = C.BC_CUSTOM
            module.single_measurements[0].custom_thresholds.value = ".2,.7"
            module.single_measurements[0].bin_count.value = 14  # should ignore
            module.single_measurements[0].wants_custom_names.value = True
            module.single_measurements[0].wants_low_bin.value = True
            module.single_measurements[0].wants_high_bin.value = True
            module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
            module.single_measurements[0].wants_images.value = True

            expected_img = dict(Classify_Three_NumObjectsPerBin=1,
                                Classify_Three_PctObjectsPerBin=25.0,
                                Classify_Blind_NumObjectsPerBin=1,
                                Classify_Blind_PctObjectsPerBin=25.0,
                                Classify_Mice_NumObjectsPerBin=1,
                                Classify_Mice_PctObjectsPerBin=25.0)
            expected_obj = dict(Classify_Three=(0, 1, 0, 0),
                                Classify_Blind=(1, 0, 0, 0),
                                Classify_Mice=(0, 0, 1, 0))
            module.run(workspace)
            for measurement, expected_values in expected_obj.iteritems():
                values = workspace.measurements.get_current_measurement(
                        OBJECTS_NAME, measurement)
                self.assertEqual(len(values), 4)
                self.assertTrue(np.all(values == np.array(expected_values)))
            for measurement, expected_values in expected_img.iteritems():
                values = workspace.measurements.get_current_measurement(
                        cpmeas.IMAGE, measurement)
                self.assertTrue(values == expected_values)
            image = workspace.image_set.get_image(IMAGE_NAME)
            pixel_data = image.pixel_data
            self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
            colors = [pixel_data[x, y, :] for x, y in
                      ((2, 3), (12, 1), (6, 5), (16, 5))]
            for i, color in enumerate(colors + [colors[1]]):
                self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

    def test_03_01_two_none(self):
        workspace, module = self.make_workspace(
                np.zeros((10, 10), int),
                C.BY_TWO_MEASUREMENTS,
                np.zeros((0,), float), np.zeros((0,), float))
        module.run(workspace)
        for lh1 in ("low", "high"):
            for lh2 in ("low", "high"):
                m_name = ("Classify_Measurement1_%s_Measurement2_%s" %
                          (lh1, lh2))
                m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                   m_name)
                self.assertEqual(len(m), 0)

    def test_03_02_two(self):
        np.random.seed(0)
        labels = np.zeros((10, 20), int)
        index = 1
        for i_min, i_max in ((1, 4), (6, 9)):
            for j_min, j_max in ((2, 6), (8, 11), (13, 18)):
                labels[i_min:i_max, j_min:j_max] = index
                index += 1
        num_labels = index - 1
        exps = np.exp(np.arange(np.max(labels)))
        m1 = np.random.permutation(exps)
        m2 = np.random.permutation(exps)
        for wants_custom_names in (False, True):
            for tm1 in (C.TM_MEAN, C.TM_MEDIAN, C.TM_CUSTOM):
                for tm2 in (C.TM_MEAN, C.TM_MEDIAN, C.TM_CUSTOM):
                    workspace, module = self.make_workspace(labels,
                                                            C.BY_TWO_MEASUREMENTS,
                                                            m1, m2)
                    self.assertTrue(isinstance(module, C.ClassifyObjects))
                    module.first_threshold_method.value = tm1
                    module.first_threshold.value = 8
                    module.second_threshold_method.value = tm2
                    module.second_threshold.value = 70
                    module.wants_image.value = True

                    def cutoff(method, custom_cutoff):
                        if method == C.TM_MEAN:
                            return np.mean(exps)
                        elif method == C.TM_MEDIAN:
                            return np.median(exps)
                        else:
                            return custom_cutoff

                    c1 = cutoff(tm1, module.first_threshold.value)
                    c2 = cutoff(tm2, module.second_threshold.value)
                    m1_over = m1 >= c1
                    m2_over = m2 >= c2
                    if wants_custom_names:
                        f_names = ("TL", "TR", "BL", "BR")
                        module.wants_custom_names.value = True
                        module.low_low_custom_name.value = f_names[0]
                        module.low_high_custom_name.value = f_names[1]
                        module.high_low_custom_name.value = f_names[2]
                        module.high_high_custom_name.value = f_names[3]
                    else:
                        f_names = ("Measurement1_low_Measurement2_low",
                                   "Measurement1_low_Measurement2_high",
                                   "Measurement1_high_Measurement2_low",
                                   "Measurement1_high_Measurement2_high")
                    m_names = ["_".join((C.M_CATEGORY, name))
                               for name in f_names]

                    module.run(workspace)
                    columns = module.get_measurement_columns(None)
                    for column in columns:
                        if column[0] != OBJECTS_NAME:  # Must be image
                            self.assertEqual(column[0], cpmeas.IMAGE)
                            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                                    C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
                        else:
                            self.assertEqual(column[0], OBJECTS_NAME)
                            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

                    self.assertEqual(len(columns), 12)
                    self.assertEqual(len(set([column[1] for column in columns])), 12)  # no duplicates

                    categories = module.get_categories(None, cpmeas.IMAGE)
                    self.assertEqual(len(categories), 1)
                    categories = module.get_categories(None, OBJECTS_NAME)
                    self.assertEqual(len(categories), 1)
                    self.assertEqual(categories[0], C.M_CATEGORY)
                    names = module.get_measurements(None, OBJECTS_NAME, "foo")
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, "foo", C.M_CATEGORY)
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
                    self.assertEqual(len(names), 4)

                    for m_name, expected in zip(m_names,
                                                ((~m1_over) & (~m2_over),
                                                 (~m1_over) & m2_over,
                                                 m1_over & ~m2_over,
                                                 m1_over & m2_over)):
                        m = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                           '_'.join((m_name, C.F_NUM_PER_BIN)))
                        self.assertTrue(m == expected.astype(int).sum())
                        m = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                           '_'.join((m_name, C.F_PCT_PER_BIN)))
                        self.assertTrue(m == 100.0 * float(expected.astype(int).sum()) / num_labels)
                        m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                           m_name)
                        self.assertTrue(np.all(m == expected.astype(int)))
                        self.assertTrue(m_name in [column[1] for column in columns])
                        self.assertTrue(m_name in ["_".join((C.M_CATEGORY, name))
                                                   for name in names])
                    image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
                    self.assertTrue(np.all(image[labels == 0, :] == 0))
                    colors = image[(labels > 0) & (m[labels - 1] == 1), :]
                    if colors.shape[0] > 0:
                        self.assertTrue(all([np.all(colors[:, i] == colors[0, i])
                                             for i in range(3)]))

    def test_03_04_nans(self):
        # Test for NaN values in two measurements.
        #
        labels = np.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 3
        labels[3:5, 6:8] = 4
        labels[6:8, 6:8] = 5
        labels[3:5, 10:12] = 2

        m1 = np.array((1, 2, np.NaN, 1, np.NaN))
        m2 = np.array((1, 2, 1, np.NaN, np.NaN))
        for leave_last_out in (False, True):
            end = np.max(labels) - 1 if leave_last_out else np.max(labels)
            workspace, module = self.make_workspace(
                    labels,
                    C.BY_TWO_MEASUREMENTS,
                    m1[:end], m2[:end])
            self.assertTrue(isinstance(module, C.ClassifyObjects))
            module.first_threshold_method.value = C.TM_MEAN
            module.first_threshold.value = 2
            module.second_threshold_method.value = C.TM_MEAN
            module.second_threshold.value = 2
            module.wants_image.value = True
            module.wants_custom_names.value = False
            module.run(workspace)
            f_names = ("Measurement1_low_Measurement2_low",
                       "Measurement1_low_Measurement2_high",
                       "Measurement1_high_Measurement2_low",
                       "Measurement1_high_Measurement2_high")
            m_names = ["_".join((C.M_CATEGORY, name))
                       for name in f_names]
            m = workspace.measurements
            for m_name, expected in zip(
                    m_names,
                    [np.array((1, 0, 0, 0, 0)),
                     np.array((0, 0, 0, 0, 0)),
                     np.array((0, 0, 0, 0, 0)),
                     np.array((0, 1, 0, 0, 0))]):
                values = m[OBJECTS_NAME, m_name]
                np.testing.assert_array_equal(values, expected)

    def test_03_05_nan_offset_by_1(self):
        # Regression test of 1636
        labels = np.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 2

        m1 = np.array((4, np.NaN))
        m2 = np.array((4, 4))
        workspace, module = self.make_workspace(
                labels,
                C.BY_TWO_MEASUREMENTS,
                m1, m2)
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        module.first_threshold_method.value = C.TM_MEAN
        module.first_threshold.value = 2
        module.second_threshold_method.value = C.TM_MEAN
        module.second_threshold.value = 2
        module.wants_image.value = True
        module.wants_custom_names.value = False
        module.run(workspace)
        image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        colors = module.get_colors(4)
        reverse = np.zeros(image.shape[:2], int)
        for idx, color in enumerate(colors):
            reverse[
                np.all(image == color[np.newaxis, np.newaxis, :3], 2)] = idx
        self.assertTrue(np.all(reverse[labels == 1] == 4))
