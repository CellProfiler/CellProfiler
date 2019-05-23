"""test_identifytertiaryobjects.py - test the IdentifyTertiaryObjects module
"""

import base64
import unittest
import zlib
from StringIO import StringIO

import cellprofiler.measurement
import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.modules.identify as cpmi
import cellprofiler.modules.identifytertiaryobjects as cpmit
import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.object as cpo
import cellprofiler.measurement as cpm

PRIMARY = "primary"
SECONDARY = "secondary"
TERTIARY = "tertiary"
OUTLINES = "Outlines"


class TestIdentifyTertiaryObjects(unittest.TestCase):
    def on_pipeline_event(self, caller, event):
        self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

    def make_workspace(self, primary_labels, secondary_labels):
        """Make a workspace that has objects for the input labels

        returns a workspace with the following
            object_set - has object with name "primary" containing
                         the primary labels
                         has object with name "secondary" containing
                         the secondary labels
        """
        isl = cpi.ImageSetList()
        module = cpmit.IdentifyTertiarySubregion()
        module.module_num = 1
        module.primary_objects_name.value = PRIMARY
        module.secondary_objects_name.value = SECONDARY
        module.subregion_objects_name.value = TERTIARY
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  isl.get_image_set(0),
                                  cpo.ObjectSet(),
                                  cpm.Measurements(),
                                  isl)
        workspace.pipeline.add_module(module)

        for labels, name in ((primary_labels, PRIMARY),
                             (secondary_labels, SECONDARY)):
            objects = cpo.Objects()
            objects.segmented = labels
            workspace.object_set.add_objects(objects, name)
        return workspace

    def test_00_00_zeros(self):
        """Test IdentifyTertiarySubregion on an empty image"""
        primary_labels = np.zeros((10, 10), int)
        secondary_labels = np.zeros((10, 10), int)
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s" % TERTIARY
        self.assertTrue(count_feature in
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(np.product(value.shape), 1)
        self.assertEqual(value, 0)
        self.assertTrue(TERTIARY in workspace.object_set.get_object_names())
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == primary_labels))
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name in (cpm.IMAGE, PRIMARY, SECONDARY, TERTIARY):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_01_01_one_object(self):
        """Test creation of a single tertiary object"""
        primary_labels = np.zeros((10, 10), int)
        secondary_labels = np.zeros((10, 10), int)
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels = np.zeros((10, 10), int)
        expected_labels[2:7, 3:8] = 1
        expected_labels[4, 5] = 0
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s" % TERTIARY
        self.assertTrue(count_feature in
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(np.product(value.shape), 1)
        self.assertEqual(value, 1)

        self.assertTrue(TERTIARY in measurements.get_object_names())
        child_count_feature = "Children_%s_Count" % TERTIARY
        for parent_name in (PRIMARY, SECONDARY):
            parents_of_feature = ("Parent_%s" % parent_name)
            self.assertTrue(parents_of_feature in
                            measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY,
                                                         parents_of_feature)
            self.assertTrue(np.product(value.shape), 1)
            self.assertTrue(value[0], 1)
            self.assertTrue(child_count_feature in
                            measurements.get_feature_names(parent_name))
            value = measurements.get_current_measurement(parent_name,
                                                         child_count_feature)
            self.assertTrue(np.product(value.shape), 1)
            self.assertTrue(value[0], 1)

        for axis, expected in (("X", 5), ("Y", 4)):
            feature = "Location_Center_%s" % axis
            self.assertTrue(feature in measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY, feature)
            self.assertTrue(np.product(value.shape), 1)
            self.assertEqual(value[0], expected)

        self.assertTrue(TERTIARY in workspace.object_set.get_object_names())
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == expected_labels))

    def test_01_02_two_objects(self):
        """Test creation of two tertiary objects"""
        primary_labels = np.zeros((10, 20), int)
        secondary_labels = np.zeros((10, 20), int)
        expected_primary_parents = np.zeros((10, 20), int)
        expected_secondary_parents = np.zeros((10, 20), int)
        centers = ((4, 5, 1, 2), (4, 15, 2, 1))
        for x, y, primary_label, secondary_label in centers:
            primary_labels[x - 1:x + 2, y - 1:y + 2] = primary_label
            secondary_labels[x - 2:x + 3, y - 2:y + 3] = secondary_label
            expected_primary_parents[x - 2:x + 3, y - 2:y + 3] = primary_label
            expected_primary_parents[x, y] = 0
            expected_secondary_parents[x - 2:x + 3, y - 2:y + 3] = secondary_label
            expected_secondary_parents[x, y] = 0

        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        count_feature = "Count_%s" % TERTIARY
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(value, 2)

        child_count_feature = "Children_%s_Count" % TERTIARY
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        for parent_name, idx, parent_labels in ((PRIMARY, 2, expected_primary_parents),
                                                (SECONDARY, 3, expected_secondary_parents)):
            parents_of_feature = ("Parent_%s" % parent_name)
            cvalue = measurements.get_current_measurement(parent_name,
                                                          child_count_feature)
            self.assertTrue(np.all(cvalue == 1))
            pvalue = measurements.get_current_measurement(TERTIARY,
                                                          parents_of_feature)
            for value in (pvalue, cvalue):
                self.assertTrue(np.product(value.shape), 2)
            #
            # Make an array that maps the parent label index to the
            # corresponding child label index
            #
            label_map = np.zeros((len(centers) + 1,), int)
            for center in centers:
                label = center[idx]
                label_map[label] = pvalue[center[idx] - 1]
            expected_labels = label_map[parent_labels]
            self.assertTrue(np.all(expected_labels == output_labels))

    def test_01_03_overlapping_secondary(self):
        """Make sure that an overlapping tertiary is assigned to the larger parent"""
        expected_primary_parents = np.zeros((10, 20), int)
        expected_secondary_parents = np.zeros((10, 20), int)
        primary_labels = np.zeros((10, 20), int)
        secondary_labels = np.zeros((10, 20), int)
        primary_labels[3:6, 3:10] = 2
        primary_labels[3:6, 10:17] = 1
        secondary_labels[2:7, 2:12] = 1
        expected_primary_parents[2:7, 2:12] = 2
        expected_primary_parents[4, 4:12] = 0  # the middle of the primary
        expected_primary_parents[4, 9] = 2  # the outline of primary # 2
        expected_primary_parents[4, 10] = 2  # the outline of primary # 1
        expected_secondary_parents[expected_primary_parents > 0] = 1
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        self.assertTrue(isinstance(module, cpmit.IdentifyTertiarySubregion))
        module.run(workspace)
        measurements = workspace.measurements
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        for parent_name, parent_labels in ((PRIMARY, expected_primary_parents),
                                           (SECONDARY, expected_secondary_parents)):
            parents_of_feature = ("Parent_%s" % parent_name)
            pvalue = measurements.get_current_measurement(TERTIARY,
                                                          parents_of_feature)
            label_map = np.zeros((np.product(pvalue.shape) + 1,), int)
            label_map[1:] = pvalue.flatten()
            mapped_labels = label_map[output_labels]
            self.assertTrue(np.all(parent_labels == mapped_labels))

    def test_01_04_wrong_size(self):
        '''Regression test of img-961, what if objects have different sizes?

        Slightly bizarre use case: maybe if user wants to measure background
        outside of cells in a plate of wells???
        '''
        expected_primary_parents = np.zeros((20, 20), int)
        expected_secondary_parents = np.zeros((20, 20), int)
        primary_labels = np.zeros((10, 30), int)
        secondary_labels = np.zeros((20, 20), int)
        primary_labels[3:6, 3:10] = 2
        primary_labels[3:6, 10:17] = 1
        secondary_labels[2:7, 2:12] = 1
        expected_primary_parents[2:7, 2:12] = 2
        expected_primary_parents[4, 4:12] = 0  # the middle of the primary
        expected_primary_parents[4, 9] = 2  # the outline of primary # 2
        expected_primary_parents[4, 10] = 2  # the outline of primary # 1
        expected_secondary_parents[expected_primary_parents > 0] = 1
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        self.assertTrue(isinstance(module, cpmit.IdentifyTertiarySubregion))
        module.run(workspace)

    def test_03_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        module = cpmit.IdentifyTertiarySubregion()
        module.primary_objects_name.value = PRIMARY
        module.secondary_objects_name.value = SECONDARY
        module.subregion_objects_name.value = TERTIARY
        columns = module.get_measurement_columns(None)
        expected = ((cpm.IMAGE, cellprofiler.measurement.FF_COUNT % TERTIARY, cpm.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.measurement.M_LOCATION_CENTER_X, cpm.COLTYPE_FLOAT),
                    (TERTIARY, cellprofiler.measurement.M_LOCATION_CENTER_Y, cpm.COLTYPE_FLOAT),
                    (TERTIARY, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER, cpm.COLTYPE_INTEGER),
                    (PRIMARY, cellprofiler.measurement.FF_CHILDREN_COUNT % TERTIARY, cpm.COLTYPE_INTEGER),
                    (SECONDARY, cellprofiler.measurement.FF_CHILDREN_COUNT % TERTIARY, cpm.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.measurement.FF_PARENT % PRIMARY, cpm.COLTYPE_INTEGER),
                    (TERTIARY, cellprofiler.measurement.FF_PARENT % SECONDARY, cpm.COLTYPE_INTEGER))
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cv == ev for cv, ev in zip(column, ec)])
                                 for ec in expected]))

    def test_04_01_do_not_shrink(self):
        '''Test the option to not shrink the smaller objects'''
        primary_labels = np.zeros((10, 10), int)
        secondary_labels = np.zeros((10, 10), int)
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels = np.zeros((10, 10), int)
        expected_labels[2:7, 3:8] = 1
        expected_labels[3:6, 4:7] = 0

        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.shrink_primary.value = False
        module.run(workspace)
        measurements = workspace.measurements

        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == expected_labels))

    def test_04_02_do_not_shrink_identical(self):
        '''Test a case where the primary and secondary objects are identical'''
        primary_labels = np.zeros((20, 20), int)
        secondary_labels = np.zeros((20, 20), int)
        expected_labels = np.zeros((20, 20), int)

        # first and third objects have different sizes
        primary_labels[3:6, 4:7] = 1
        secondary_labels[2:7, 3:8] = 1
        expected_labels[2:7, 3:8] = 1
        expected_labels[3:6, 4:7] = 0

        primary_labels[13:16, 4:7] = 3
        secondary_labels[12:17, 3:8] = 3
        expected_labels[12:17, 3:8] = 3
        expected_labels[13:16, 4:7] = 0

        # second object and fourth have same size

        primary_labels[3:6, 14:17] = 2
        secondary_labels[3:6, 14:17] = 2
        primary_labels[13:16, 14:17] = 4
        secondary_labels[13:16, 14:17] = 4
        workspace = self.make_workspace(primary_labels, secondary_labels)

        module = workspace.module
        module.shrink_primary.value = False
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == expected_labels))

        measurements = workspace.measurements
        count_feature = "Count_%s" % TERTIARY
        value = measurements.get_current_measurement("Image", count_feature)
        self.assertEqual(value, 3)

        child_count_feature = "Children_%s_Count" % TERTIARY
        for parent_name in PRIMARY, SECONDARY:
            parent_of_feature = "Parent_%s" % parent_name
            parent_of = measurements.get_current_measurement(
                    TERTIARY, parent_of_feature)
            child_count = measurements.get_current_measurement(
                    parent_name, child_count_feature)
            for parent, expected_child_count in ((1, 1), (2, 0), (3, 1), (4, 0)):
                self.assertEqual(child_count[parent - 1], expected_child_count)
            for child in (1, 3):
                self.assertEqual(parent_of[child - 1], child)

        for location_feature in (
                cellprofiler.measurement.M_LOCATION_CENTER_X, cellprofiler.measurement.M_LOCATION_CENTER_Y):
            values = measurements.get_current_measurement(
                    TERTIARY, location_feature)
            self.assertTrue(np.all(np.isnan(values) == [False, True, False]))

    def test_04_03_do_not_shrink_missing(self):
        # Regression test of 705

        for missing in range(1, 3):
            for missing_primary in False, True:
                primary_labels = np.zeros((20, 20), int)
                secondary_labels = np.zeros((20, 20), int)
                expected_labels = np.zeros((20, 20), int)
                centers = ((5, 5), (15, 5), (5, 15))
                pidx = 1
                sidx = 1
                for idx, (i, j) in enumerate(centers):
                    if (idx + 1 != missing) or not missing_primary:
                        primary_labels[(i - 1):(i + 2), (j - 1):(j + 2)] = pidx
                        pidx += 1
                    if (idx + 1 != missing) or missing_primary:
                        secondary_labels[(i - 2):(i + 3), (j - 2):(j + 3)] = sidx
                        sidx += 1
                expected_labels = secondary_labels * (primary_labels == 0)
                workspace = self.make_workspace(primary_labels, secondary_labels)

                module = workspace.module
                module.shrink_primary.value = False
                module.run(workspace)
                output_objects = workspace.object_set.get_objects(TERTIARY)
                self.assertTrue(np.all(output_objects.segmented == expected_labels))

                m = workspace.measurements

                child_name = module.subregion_objects_name.value
                primary_name = module.primary_objects_name.value
                ftr = cellprofiler.measurement.FF_PARENT % primary_name
                pparents = m[child_name, ftr]
                self.assertEqual(len(pparents), 3 if missing_primary else 2)
                if missing_primary:
                    self.assertEqual(pparents[missing - 1], 0)

                secondary_name = module.secondary_objects_name.value
                ftr = cellprofiler.measurement.FF_PARENT % secondary_name
                pparents = m[child_name, ftr]
                self.assertEqual(len(pparents), 3 if missing_primary else 2)
                if not missing_primary:
                    self.assertTrue(all([x in pparents for x in range(1, 3)]))

                ftr = cellprofiler.measurement.FF_CHILDREN_COUNT % child_name
                children = m[primary_name, ftr]
                self.assertEqual(len(children), 2 if missing_primary else 3)
                if not missing_primary:
                    self.assertEqual(children[missing - 1], 0)
                    self.assertTrue(np.all(np.delete(children, missing - 1) == 1))
                else:
                    self.assertTrue(np.all(children == 1))

                children = m[secondary_name, ftr]
                self.assertEqual(len(children), 3 if missing_primary else 2)
                self.assertTrue(np.all(children == 1))

    def test_05_00_no_relationships(self):
        workspace = self.make_workspace(np.zeros((10, 10), int),
                                        np.zeros((10, 10), int))
        workspace.module.run(workspace)
        m = workspace.measurements
        for parent, relationship in (
                (PRIMARY, cpmit.R_REMOVED),
                (SECONDARY, cpmit.R_PARENT)):
            result = m.get_relationships(
                    workspace.module.module_num, relationship,
                    parent, TERTIARY)
            self.assertEqual(len(result), 0)

    def test_05_01_relationships(self):
        primary = np.zeros((10, 30), int)
        secondary = np.zeros((10, 30), int)
        for i in range(3):
            center_j = 5 + i * 10
            primary[3:6, (center_j - 1):(center_j + 2)] = i + 1
            secondary[2:7, (center_j - 2):(center_j + 3)] = i + 1
        workspace = self.make_workspace(primary, secondary)
        workspace.module.run(workspace)
        m = workspace.measurements
        for parent, relationship in (
                (PRIMARY, cpmit.R_REMOVED),
                (SECONDARY, cpmit.R_PARENT)):
            result = m.get_relationships(
                    workspace.module.module_num, relationship,
                    parent, TERTIARY)
            self.assertEqual(len(result), 3)
            for i in range(3):
                self.assertEqual(result[cpm.R_FIRST_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cpm.R_SECOND_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cpm.R_FIRST_OBJECT_NUMBER][i], i + 1)
                self.assertEqual(result[cpm.R_SECOND_OBJECT_NUMBER][i], i + 1)

    def test_06_01_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150319195827
GitHash:d8289bf
ModuleCount:1
HasImagePlaneDetails:False

IdentifyTertiaryObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the larger identified objects:IdentifySecondaryObjects
    Select the smaller identified objects:IdentifyPrimaryObjects
    Name the tertiary objects to be identified:IdentifyTertiaryObjects
    Shrink smaller object prior to subtraction?:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.loadtxt(StringIO(data))
        module = pipeline.modules()[0]

        assert module.secondary_objects_name.value == "IdentifySecondaryObjects"
        assert module.primary_objects_name.value == "IdentifyPrimaryObjects"
        assert module.subregion_objects_name.value == "IdentifyTertiaryObjects"
        assert module.shrink_primary.value
