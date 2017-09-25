import base64
import os
import unittest
import zlib
from StringIO import StringIO

import cellprofiler.measurement
import numpy as np
import scipy.ndimage

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.splitormergeobjects
import cellprofiler.modules.identify as I

INPUT_OBJECTS_NAME = 'inputobjects'
OUTPUT_OBJECTS_NAME = 'outputobjects'
IMAGE_NAME = 'image'
OUTLINE_NAME = 'outlines'


class TestSplitOrMergeObjects(unittest.TestCase):
    def test_01_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150319195827
GitHash:d8289bf
ModuleCount:1
HasImagePlaneDetails:False

SplitOrMergeObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:IdentifyPrimaryObjects
    Name the new objects:SplitOrMergeObjects
    Operation:Unify
    Maximum distance within which to unify objects:0
    Unify using a grayscale image?:No
    Select the grayscale image to guide unification:None
    Minimum intensity fraction:0.9
    Method to find object intensity:Closest point
    Unification method:Distance
    Select the parent object:None
    Output object type:Disconnected
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.loadtxt(StringIO(data))
        module = pipeline.modules()[0]

        assert module.objects_name.value == "IdentifyPrimaryObjects"
        assert module.output_objects_name.value == "SplitOrMergeObjects"
        assert module.relabel_option.value == "Merge"
        assert module.distance_threshold.value == 0
        assert not module.wants_image.value
        assert module.image_name.value == "None"
        assert module.minimum_intensity_fraction.value == 0.9
        assert module.where_algorithm.value == "Closest point"
        assert module.merge_option.value == "Distance"
        assert module.parent_object.value == "None"
        assert module.merging_method.value == "Disconnected"

    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150319195827
GitHash:d8289bf
ModuleCount:2
HasImagePlaneDetails:False

SplitOrMergeObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:blobs
    Name the new objects:RelabeledBlobs
    Operation:Unify
    Maximum distance within which to unify objects:2
    Unify using a grayscale image?:No
    Select the grayscale image to guide unification:Guide
    Minimum intensity fraction:0.8
    Method to find object intensity:Closest point
    Retain outlines of the relabeled objects?:No
    Name the outlines:RelabeledNucleiOutlines
    Unification method:Per-parent
    Select the parent object:Nuclei
    Output object type:Convex hull

SplitOrMergeObjects:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:blobs
    Name the new objects:RelabeledNuclei
    Operation:Split
    Maximum distance within which to unify objects:1
    Unify using a grayscale image?:Yes
    Select the grayscale image to guide unification:Guide
    Minimum intensity fraction:0.9
    Method to find object intensity:Centroids
    Retain outlines of the relabeled objects?:Yes
    Name the outlines:RelabeledNucleiOutlines
    Unification method:Distance
    Select the parent object:Nuclei
    Output object type:Disconnected
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.loadtxt(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects))
        self.assertEqual(module.objects_name, "blobs")
        self.assertEqual(module.output_objects_name, "RelabeledBlobs")
        self.assertEqual(module.relabel_option, cellprofiler.modules.splitormergeobjects.OPTION_MERGE)
        self.assertEqual(module.distance_threshold, 2)
        self.assertFalse(module.wants_image)
        self.assertEqual(module.image_name, "Guide")
        self.assertEqual(module.minimum_intensity_fraction, .8)
        self.assertEqual(module.where_algorithm, cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT)
        self.assertEqual(module.merge_option, cellprofiler.modules.splitormergeobjects.UNIFY_PARENT)
        self.assertEqual(module.parent_object, "Nuclei")
        self.assertEqual(module.merging_method, cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL)

        module = pipeline.modules()[1]
        self.assertEqual(module.relabel_option, cellprofiler.modules.splitormergeobjects.OPTION_SPLIT)
        self.assertTrue(module.wants_image)
        self.assertEqual(module.where_algorithm, cellprofiler.modules.splitormergeobjects.CA_CENTROIDS)
        self.assertEqual(module.merge_option, cellprofiler.modules.splitormergeobjects.UNIFY_DISTANCE)
        self.assertEqual(module.merging_method, cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED)

    def rruunn(self, input_labels, relabel_option,
               merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_DISTANCE,
               unify_method=cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED,
               distance_threshold=5,
               minimum_intensity_fraction=.9,
               where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT,
               image=None,
               parent_object="Parent_object",
               parents_of=None):
        '''Run the SplitOrMergeObjects module

        returns the labels matrix and the workspace.
        '''
        module = cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects()
        module.module_num = 1
        module.objects_name.value = INPUT_OBJECTS_NAME
        module.output_objects_name.value = OUTPUT_OBJECTS_NAME
        module.relabel_option.value = relabel_option
        module.merge_option.value = merge_option
        module.merging_method.value = unify_method
        module.parent_object.value = parent_object
        module.distance_threshold.value = distance_threshold
        module.minimum_intensity_fraction.value = minimum_intensity_fraction
        module.wants_image.value = (image is not None)
        module.where_algorithm.value = where_algorithm

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        if image is not None:
            img = cpi.Image(image)
            image_set.add(IMAGE_NAME, img)
            module.image_name.value = IMAGE_NAME

        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        o.segmented = input_labels
        object_set.add_objects(o, INPUT_OBJECTS_NAME)

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        if parents_of is not None:
            m = workspace.measurements
            ftr = cellprofiler.measurement.FF_PARENT % parent_object
            m[INPUT_OBJECTS_NAME, ftr] = parents_of
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        return output_objects.segmented, workspace

    def test_02_01_split_zero(self):
        labels, workspace = self.rruunn(np.zeros((10, 20), int),
                                        cellprofiler.modules.splitormergeobjects.OPTION_SPLIT)
        self.assertTrue(np.all(labels == 0))
        self.assertEqual(labels.shape[0], 10)
        self.assertEqual(labels.shape[1], 20)

        self.assertTrue(isinstance(workspace, cpw.Workspace))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        count = m.get_current_image_measurement(cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, 0)
        for feature_name in (cellprofiler.measurement.M_LOCATION_CENTER_X, cellprofiler.measurement.M_LOCATION_CENTER_Y):
            values = m.get_current_measurement(OUTPUT_OBJECTS_NAME,
                                               feature_name)
            self.assertEqual(len(values), 0)

        module = workspace.module
        self.assertTrue(isinstance(module, cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 6)
        for object_name, feature_name, coltype in (
                (OUTPUT_OBJECTS_NAME, cellprofiler.measurement.M_LOCATION_CENTER_X, cpmeas.COLTYPE_FLOAT),
                (OUTPUT_OBJECTS_NAME, cellprofiler.measurement.M_LOCATION_CENTER_Y, cpmeas.COLTYPE_FLOAT),
                (OUTPUT_OBJECTS_NAME, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
                (INPUT_OBJECTS_NAME, cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME,
                 cpmeas.COLTYPE_INTEGER),
                (OUTPUT_OBJECTS_NAME, cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS_NAME,
                 cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME, cpmeas.COLTYPE_INTEGER)):
            self.assertTrue(any([object_name == c[0] and
                                 feature_name == c[1] and
                                 coltype == c[2] for c in columns]))
        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Count")
        categories = module.get_categories(workspace.pipeline, OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 3)
        self.assertTrue(any(["Location" in categories]))
        self.assertTrue(any(["Parent" in categories]))
        self.assertTrue(any(["Number" in categories]))
        categories = module.get_categories(workspace.pipeline, INPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Children")
        f = module.get_measurements(workspace.pipeline, cpmeas.IMAGE, "Count")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], OUTPUT_OBJECTS_NAME)
        f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME,
                                    "Location")
        self.assertEqual(len(f), 2)
        self.assertTrue(all([any([x == y for y in f])
                             for x in ("Center_X", "Center_Y")]))
        f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME,
                                    "Parent")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], INPUT_OBJECTS_NAME)

        f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME,
                                    "Number")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], 'Object_Number')

        f = module.get_measurements(workspace.pipeline, INPUT_OBJECTS_NAME,
                                    "Children")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], "%s_Count" % OUTPUT_OBJECTS_NAME)

    def test_02_02_split_one(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_SPLIT)
        self.assertTrue(np.all(labels == labels_out))

        self.assertTrue(isinstance(workspace, cpw.Workspace))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        count = m.get_current_image_measurement(cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, 1)
        for feature_name, value in ((cellprofiler.measurement.M_LOCATION_CENTER_X, 5),
                                    (cellprofiler.measurement.M_LOCATION_CENTER_Y, 3),
                                    (cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS_NAME, 1)):
            values = m.get_current_measurement(OUTPUT_OBJECTS_NAME,
                                               feature_name)
            self.assertEqual(len(values), 1)
            self.assertAlmostEqual(values[0], value)

        values = m.get_current_measurement(INPUT_OBJECTS_NAME,
                                           cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 1)

    def test_02_03_split_one_into_two(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 1
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_SPLIT)
        index = np.array([labels_out[3, 5], labels_out[3, 15]])
        self.assertNotEqual(index[0], index[1])
        self.assertTrue(all([x in index for x in (1, 2)]))
        expected = np.zeros((10, 20), int)
        expected[2:5, 3:8] = index[0]
        expected[2:5, 13:18] = index[1]
        self.assertTrue(np.all(labels_out == expected))
        m = workspace.measurements
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME,
                                           cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS_NAME)
        self.assertEqual(len(values), 2)
        self.assertTrue(np.all(values == 1))
        values = m.get_current_measurement(INPUT_OBJECTS_NAME,
                                           cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 2)

    def test_03_01_unify_zero(self):
        labels, workspace = self.rruunn(np.zeros((10, 20), int),
                                        cellprofiler.modules.splitormergeobjects.OPTION_MERGE)
        self.assertTrue(np.all(labels == 0))
        self.assertEqual(labels.shape[0], 10)
        self.assertEqual(labels.shape[1], 20)

    def test_03_02_unify_one(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE)
        self.assertTrue(np.all(labels == labels_out))

    def test_03_03_unify_two_to_one(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=6)
        self.assertTrue(np.all(labels_out[labels != 0] == 1))
        self.assertTrue(np.all(labels_out[labels == 0] == 0))

    def test_03_04_unify_two_stays_two(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=4)
        self.assertTrue(np.all(labels_out == labels))

    def test_03_05_unify_image_centroids(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        image = np.ones((10, 20)) * (labels > 0) * .5
        image[3, 8:13] = .41
        image[3, 5] = .6
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=6,
                                            image=image,
                                            minimum_intensity_fraction=.8,
                                            where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CENTROIDS)
        self.assertTrue(np.all(labels_out[labels != 0] == 1))
        self.assertTrue(np.all(labels_out[labels == 0] == 0))

    def test_03_06_dont_unify_image_centroids(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        image = np.ones((10, 20)) * labels * .5
        image[3, 8:12] = .41
        image[3, 5] = .6
        image[3, 15] = .6
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=6,
                                            image=image,
                                            minimum_intensity_fraction=.8,
                                            where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CENTROIDS)
        self.assertTrue(np.all(labels_out == labels))

    def test_03_07_unify_image_closest_point(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        image = np.ones((10, 20)) * (labels > 0) * .6
        image[2, 8:13] = .41
        image[2, 7] = .5
        image[2, 13] = .5
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=6,
                                            image=image,
                                            minimum_intensity_fraction=.8,
                                            where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT)
        self.assertTrue(np.all(labels_out[labels != 0] == 1))
        self.assertTrue(np.all(labels_out[labels == 0] == 0))

    def test_03_08_dont_unify_image_closest_point(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        image = np.ones((10, 20)) * labels * .6
        image[3, 8:12] = .41
        image[2, 7] = .5
        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            distance_threshold=6,
                                            image=image,
                                            minimum_intensity_fraction=.8,
                                            where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT)
        self.assertTrue(np.all(labels_out == labels))

    def test_05_00_unify_per_parent(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2

        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
                                            parent_object="Parent_object",
                                            parents_of=np.array([1, 1]))
        self.assertTrue(np.all(labels_out[labels != 0] == 1))

    def test_05_01_unify_convex_hull(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 2
        expected = np.zeros(labels.shape, int)
        expected[2:5, 3:18] = 1

        labels_out, workspace = self.rruunn(labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                                            merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
                                            unify_method=cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL,
                                            parent_object="Parent_object",
                                            parents_of=np.array([1, 1]))
        self.assertTrue(np.all(labels_out == expected))

    def test_05_02_unify_nothing(self):
        labels = np.zeros((10, 20), int)
        for um in cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED, cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL:
            labels_out, workspace = self.rruunn(
                    labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
                    merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
                    unify_method=cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL,
                    parent_object="Parent_object",
                    parents_of=np.zeros(0, int))
            self.assertTrue(np.all(labels_out == 0))
