import StringIO
import cPickle
import contextlib
import os
import tempfile
import unittest

import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.filterobjects
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

INPUT_IMAGE = "input_image"
INPUT_OBJECTS = "input_objects"
ENCLOSING_OBJECTS = "my_enclosing_objects"
OUTPUT_OBJECTS = "output_objects"
TEST_FTR = "my_measurement"


class TestFilterObjects(unittest.TestCase):
    def make_workspace(self, object_dict={}, image_dict={}):
        '''Make a workspace for testing FilterByObjectMeasurement'''
        module = cellprofiler.modules.filterobjects.FilterByObjectMeasurement()
        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.object.ObjectSet()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        for key in image_dict.keys():
            image_set.add(key, cellprofiler.image.Image(image_dict[key]))
        for key in object_dict.keys():
            o = cellprofiler.object.Objects()
            o.segmented = object_dict[key]
            object_set.add_objects(o, key)
        return workspace, module

    @contextlib.contextmanager
    def make_classifier(self, module,
                        answers,
                        classes=None,
                        class_names = None,
                        rules_class = None,
                        name = "Classifier",
                        feature_names = ["Foo_"+TEST_FTR]):
        '''Returns the filename of the classifier pickle'''
        assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
        if classes is None:
            classes = numpy.arange(1, numpy.max(answers) + 1)
        if class_names is None:
            class_names = ["Class%d" for _ in classes]
        if rules_class is None:
            rules_class = class_names[0]
        s = make_classifier_pickle(answers, classes, class_names, name,
                                  feature_names)
        fd, filename = tempfile.mkstemp(".model")
        os.write(fd, s)
        os.close(fd)

        module.mode.value = cellprofiler.modules.filterobjects.MODE_CLASSIFIERS
        module.rules_class.value = rules_class
        module.rules_directory.set_custom_path(os.path.dirname(filename))
        module.rules_file_name.value = os.path.split(filename)[1]
        yield
        try:
            os.remove(filename)
        except:
            pass

    def test_00_01_zeros_single(self):
        '''Test keep single object on an empty labels matrix'''
        workspace, module = self.make_workspace({INPUT_OBJECTS: numpy.zeros((10, 10), int)})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == 0))

    def test_00_02_zeros_per_object(self):
        '''Test keep per object filtering on an empty labels matrix'''
        workspace, module = self.make_workspace(
                {INPUT_OBJECTS: numpy.zeros((10, 10), int),
                 ENCLOSING_OBJECTS: numpy.zeros((10, 10), int)})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == 0))

    def test_00_03_zeros_filter(self):
        '''Test object filtering on an empty labels matrix'''
        workspace, module = self.make_workspace({INPUT_OBJECTS: numpy.zeros((10, 10), int)})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[0].min_limit.value = 0
        module.measurements[0].max_limit.value = 1000
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == 0))

    def test_01_01_keep_single_min(self):
        '''Keep a single object (min) from among two'''
        labels = numpy.zeros((10, 10), int)
        labels[2:4, 3:5] = 1
        labels[6:9, 5:8] = 2
        expected = labels.copy()
        expected[labels == 1] = 0
        expected[labels == 2] = 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([2, 1]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))
        parents = m.get_current_measurement(
                OUTPUT_OBJECTS, cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], 2)
        self.assertEqual(m.get_current_image_measurement(
            cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS), 1)
        feature = cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS
        child_count = m.get_current_measurement(INPUT_OBJECTS, feature)
        self.assertEqual(len(child_count), 2)
        self.assertEqual(child_count[0], 0)
        self.assertEqual(child_count[1], 1)

    def test_01_02_keep_single_max(self):
        '''Keep a single object (max) from among two'''
        labels = numpy.zeros((10, 10), int)
        labels[2:4, 3:5] = 1
        labels[6:9, 5:8] = 2
        expected = labels.copy()
        expected[labels == 1] = 0
        expected[labels == 2] = 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 2]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_02_01_keep_one_min(self):
        '''Keep two sub-objects (min) from among four enclosed by two'''
        sub_labels = numpy.zeros((20, 20), int)
        expected = numpy.zeros((20, 20), int)
        for i, j, k, e in ((0, 0, 1, 0), (10, 0, 2, 1), (0, 10, 3, 2), (10, 10, 4, 0)):
            sub_labels[i + 2:i + 5, j + 3:j + 7] = k
            expected[i + 2:i + 5, j + 3:j + 7] = e
        labels = numpy.zeros((20, 20), int)
        labels[:, :10] = 1
        labels[:, 10:] = 2
        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels,
                                                 ENCLOSING_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([2, 1, 3, 4]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_02_02_keep_one_max(self):
        '''Keep two sub-objects (max) from among four enclosed by two'''
        sub_labels = numpy.zeros((20, 20), int)
        expected = numpy.zeros((20, 20), int)
        for i, j, k, e in ((0, 0, 1, 0), (10, 0, 2, 1), (0, 10, 3, 2), (10, 10, 4, 0)):
            sub_labels[i + 2:i + 5, j + 3:j + 7] = k
            expected[i + 2:i + 5, j + 3:j + 7] = e
        labels = numpy.zeros((20, 20), int)
        labels[:, :10] = 1
        labels[:, 10:] = 2
        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels,
                                                 ENCLOSING_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 2, 4, 3]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_02_03_keep_maximal_most_overlap(self):
        labels = numpy.zeros((10, 20), int)
        labels[:, :10] = 1
        labels[:, 10:] = 2
        sub_labels = numpy.zeros((10, 20), int)
        sub_labels[2, 4] = 1
        sub_labels[4:6, 8:15] = 2
        sub_labels[8, 15] = 3
        expected = sub_labels * (sub_labels != 3)
        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels,
                                                 ENCLOSING_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
        module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_02_04_keep_minimal_most_overlap(self):
        labels = numpy.zeros((10, 20), int)
        labels[:, :10] = 1
        labels[:, 10:] = 2
        sub_labels = numpy.zeros((10, 20), int)
        sub_labels[2, 4] = 1
        sub_labels[4:6, 8:15] = 2
        sub_labels[8, 15] = 3
        expected = sub_labels * (sub_labels != 3)
        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels,
                                                 ENCLOSING_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
        module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([4, 2, 3]))
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_03_01_filter(self):
        '''Filter objects by limits'''
        n = 40
        labels = numpy.zeros((10, n * 10), int)
        for i in range(40):
            labels[2:5, i * 10 + 3:i * 10 + 7] = i + 1
        numpy.random.seed(0)
        values = numpy.random.uniform(size=n)
        idx = 1
        my_min = .3
        my_max = .7
        expected = numpy.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min and value <= my_max:
                expected[labels == i + 1] = idx
                idx += 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[0].wants_minimum.value = True
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].wants_maximum.value = True
        module.measurements[0].max_limit.value = my_max
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_03_02_filter(self):
        '''Filter objects by min limits'''
        n = 40
        labels = numpy.zeros((10, n * 10), int)
        for i in range(40):
            labels[2:5, i * 10 + 3:i * 10 + 7] = i + 1
        numpy.random.seed(0)
        values = numpy.random.uniform(size=n)
        idx = 1
        my_min = .3
        expected = numpy.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min:
                expected[labels == i + 1] = idx
                idx += 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].max_limit.value = .7
        module.measurements[0].wants_maximum.value = False
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_03_03_filter(self):
        '''Filter objects by maximum limits'''
        n = 40
        labels = numpy.zeros((10, n * 10), int)
        for i in range(40):
            labels[2:5, i * 10 + 3:i * 10 + 7] = i + 1
        numpy.random.seed(0)
        values = numpy.random.uniform(size=n)
        idx = 1
        my_max = .7
        expected = numpy.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value <= my_max:
                expected[labels == i + 1] = idx
                idx += 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[0].min_limit.value = .3
        module.measurements[0].wants_minimum.value = False
        module.measurements[0].max_limit.value = my_max
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_03_04_filter_two(self):
        '''Filter objects by two measurements'''
        n = 40
        labels = numpy.zeros((10, n * 10), int)
        for i in range(40):
            labels[2:5, i * 10 + 3:i * 10 + 7] = i + 1
        numpy.random.seed(0)
        values = numpy.zeros((n, 2))
        values = numpy.random.uniform(size=(n, 2))
        idx = 1
        my_max = numpy.array([.7, .5])
        expected = numpy.zeros(labels.shape, int)
        for i, v1, v2 in zip(range(n), values[:, 0], values[:, 1]):
            if v1 <= my_max[0] and v2 <= my_max[1]:
                expected[labels == i + 1] = idx
                idx += 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.add_measurement()
        m = workspace.measurements
        for i in range(2):
            measurement_name = "measurement%d" % (i + 1)
            module.measurements[i].measurement.value = measurement_name
            module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
            module.measurements[i].min_limit.value = .3
            module.measurements[i].wants_minimum.value = False
            module.measurements[i].max_limit.value = my_max[i]
            m.add_measurement(INPUT_OBJECTS, measurement_name, values[:, i])
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_04_01_renumber_other(self):
        '''Renumber an associated object'''
        n = 40
        labels = numpy.zeros((10, n * 10), int)
        alternates = numpy.zeros((10, n * 10), int)
        for i in range(40):
            labels[2:5, i * 10 + 3:i * 10 + 7] = i + 1
            alternates[3:7, i * 10 + 2:i * 10 + 5] = i + 1
        numpy.random.seed(0)
        values = numpy.random.uniform(size=n)
        idx = 1
        my_min = .3
        my_max = .7
        expected = numpy.zeros(labels.shape, int)
        expected_alternates = numpy.zeros(alternates.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min and value <= my_max:
                expected[labels == i + 1] = idx
                expected_alternates[alternates == i + 1] = idx
                idx += 1
        workspace, module = self.make_workspace({INPUT_OBJECTS: labels,
                                                 "my_alternates": alternates})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].max_limit.value = my_max
        module.add_additional_object()
        module.additional_objects[0].object_name.value = "my_alternates"
        module.additional_objects[0].target_name.value = "my_additional_result"
        m = workspace.measurements
        m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        alternates = workspace.object_set.get_objects("my_additional_result")
        self.assertTrue(numpy.all(labels.segmented == expected))
        self.assertTrue(numpy.all(alternates.segmented == expected_alternates))

    def test_05_05_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8973

FilterObjects:[module_num:1|svn_version:\'8955\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Name the output objects:FilteredThings
    Select the object to filter:Things
    Select the measurement to filter by:Intensity_MeanIntensity_DNA
    Select the filtering method:Minimal
    What did you call the objects that contain the filtered objects?:Nuclei
    Filter using a minimum measurement value?:No
    Minimum value:0
    Filter using a maximum measurement value?:No
    Maximum value:1
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:None
    Filter using classifier rules or measurements?:Measurements
    Rules file location:Default output folder
    Rules folder name:.
    Rules file name:myrules.txt
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterObjects))
        self.assertEqual(module.x_name, "Things")
        self.assertEqual(module.y_name, "FilteredThings")
        self.assertEqual(module.mode, cellprofiler.modules.filterobjects.MODE_MEASUREMENTS)
        self.assertEqual(module.rules_directory.dir_choice,
                         cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurements[0].measurement, "Intensity_MeanIntensity_DNA")
        self.assertEqual(module.filter_choice, cellprofiler.modules.filterobjects.FI_MINIMAL)

    def test_05_06_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9025

FilterObjects:[module_num:1|svn_version:\'9000\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Name the output objects:MyFilteredObjects
    Select the object to filter:MyObjects
    Filter using classifier rules or measurements?:Measurements
    Select the filtering method:Limits
    What did you call the objects that contain the filtered objects?:None
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:FilteredObjects
    Rules file location:Default input folder
    Rules folder name:./rules
    Rules file name:myrules.txt
    Hidden:2
    Hidden:2
    Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
    Filter using a minimum measurement value?:Yes
    Minimum value:0.2
    Filter using a maximum measurement value?:No
    Maximum value:1.5
    Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
    Filter using a minimum measurement value?:No
    Minimum value:0.9
    Filter using a maximum measurement value?:Yes
    Maximum value:1.8
    Select additional object to relabel:Cells
    Name the relabeled objects:FilteredCells
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCells
    Select additional object to relabel:Cytoplasm
    Name the relabeled objects:FilteredCytoplasm
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCytoplasm
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterObjects))
        self.assertEqual(module.y_name, "MyFilteredObjects")
        self.assertEqual(module.x_name, "MyObjects")
        self.assertEqual(module.mode, cellprofiler.modules.filterobjects.MODE_MEASUREMENTS)
        self.assertEqual(module.filter_choice, cellprofiler.modules.filterobjects.FI_LIMITS)
        self.assertEqual(module.rules_directory.dir_choice, cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement,
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects, ('Cells', 'Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)

    def test_05_07_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9025

FilterObjects:[module_num:6|svn_version:\'9000\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Name the output objects:MyFilteredObjects
    Select the object to filter:MyObjects
    Filter using classifier rules or measurements?:Measurements
    Select the filtering method:Limits
    What did you call the objects that contain the filtered objects?:None
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:FilteredObjects
    Rules file location:Default input folder\x7C./rules
    Rules file name:myrules.txt
    Hidden:2
    Hidden:2
    Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
    Filter using a minimum measurement value?:Yes
    Minimum value:0.2
    Filter using a maximum measurement value?:No
    Maximum value:1.5
    Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
    Filter using a minimum measurement value?:No
    Minimum value:0.9
    Filter using a maximum measurement value?:Yes
    Maximum value:1.8
    Select additional object to relabel:Cells
    Name the relabeled objects:FilteredCells
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCells
    Select additional object to relabel:Cytoplasm
    Name the relabeled objects:FilteredCytoplasm
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCytoplasm
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterObjects))
        self.assertEqual(module.y_name, "MyFilteredObjects")
        self.assertEqual(module.x_name, "MyObjects")
        self.assertEqual(module.mode, cellprofiler.modules.filterobjects.MODE_MEASUREMENTS)
        self.assertEqual(module.filter_choice, cellprofiler.modules.filterobjects.FI_LIMITS)
        self.assertEqual(module.rules_directory.dir_choice, cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.rules_class, "1")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement,
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects, ('Cells', 'Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)

    def test_05_08_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
    Version:1
    SVNRevision:9025

    FilterObjects:[module_num:1|svn_version:\'9000\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D]
        Name the output objects:MyFilteredObjects
        Select the object to filter:MyObjects
        Filter using classifier rules or measurements?:Measurements
        Select the filtering method:Limits
        What did you call the objects that contain the filtered objects?:None
        Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
        Name the outline image:FilteredObjects
        Rules file location:Default input folder\x7C./rules
        Rules file name:myrules.txt
        Rules class:1
        Hidden:2
        Hidden:2
        Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
        Filter using a minimum measurement value?:Yes
        Minimum value:0.2
        Filter using a maximum measurement value?:No
        Maximum value:1.5
        Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
        Filter using a minimum measurement value?:No
        Minimum value:0.9
        Filter using a maximum measurement value?:Yes
        Maximum value:1.8
        Select additional object to relabel:Cells
        Name the relabeled objects:FilteredCells
        Save outlines of relabeled objects?:No
        Name the outline image:OutlinesFilteredCells
        Select additional object to relabel:Cytoplasm
        Name the relabeled objects:FilteredCytoplasm
        Save outlines of relabeled objects?:No
        Name the outline image:OutlinesFilteredCytoplasm
    """
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterObjects))
        self.assertEqual(module.y_name, "MyFilteredObjects")
        self.assertEqual(module.x_name, "MyObjects")
        self.assertEqual(module.mode, cellprofiler.modules.filterobjects.MODE_MEASUREMENTS)
        self.assertEqual(module.filter_choice, cellprofiler.modules.filterobjects.FI_LIMITS)
        self.assertEqual(module.per_object_assignment, cellprofiler.modules.filterobjects.PO_BOTH)
        self.assertEqual(module.rules_directory.dir_choice, cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.rules_class, "1")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement,
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects, ('Cells', 'Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)

    def test_05_09_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
    Version:1
    SVNRevision:9025

    FilterObjects:[module_num:1|svn_version:\'9000\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
        Name the output objects:MyFilteredObjects
        Select the object to filter:MyObjects
        Filter using classifier rules or measurements?:Measurements
        Select the filtering method:Limits
        What did you call the objects that contain the filtered objects?:None
        Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
        Name the outline image:FilteredObjects
        Rules file location:Default input folder\x7C./rules
        Rules file name:myrules.txt
        Rules class:1
        Hidden:2
        Hidden:2
        Assign overlapping child to:Parent with most overlap
        Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
        Filter using a minimum measurement value?:Yes
        Minimum value:0.2
        Filter using a maximum measurement value?:No
        Maximum value:1.5
        Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
        Filter using a minimum measurement value?:No
        Minimum value:0.9
        Filter using a maximum measurement value?:Yes
        Maximum value:1.8
        Select additional object to relabel:Cells
        Name the relabeled objects:FilteredCells
        Save outlines of relabeled objects?:No
        Name the outline image:OutlinesFilteredCells
        Select additional object to relabel:Cytoplasm
        Name the relabeled objects:FilteredCytoplasm
        Save outlines of relabeled objects?:No
        Name the outline image:OutlinesFilteredCytoplasm
    """
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterObjects))
        self.assertEqual(module.y_name, "MyFilteredObjects")
        self.assertEqual(module.x_name, "MyObjects")
        self.assertEqual(module.mode, cellprofiler.modules.filterobjects.MODE_MEASUREMENTS)
        self.assertEqual(module.filter_choice, cellprofiler.modules.filterobjects.FI_LIMITS)
        self.assertEqual(module.per_object_assignment, cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP)
        self.assertEqual(module.rules_directory.dir_choice, cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.rules_class, "1")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement,
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects, ('Cells', 'Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)

    def test_08_01_filter_by_rule(self):
        labels = numpy.zeros((10, 20), int)
        labels[3:5, 4:9] = 1
        labels[7:9, 6:12] = 2
        labels[4:9, 14:18] = 3
        workspace, module = self.make_workspace({"MyObjects": labels})
        self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterByObjectMeasurement))
        m = workspace.measurements
        m.add_measurement("MyObjects", "MyMeasurement",
                          numpy.array([1.5, 2.3, 1.8]))
        rules_file_contents = "IF (MyObjects_MyMeasurement > 2.0, [1.0,-1.0], [-1.0,1.0])\n"
        rules_path = tempfile.mktemp()
        fd = open(rules_path, 'wt')
        try:
            fd.write(rules_file_contents)
            fd.close()
            rules_dir, rules_file = os.path.split(rules_path)
            module.x_name.value = "MyObjects"
            module.mode.value = cellprofiler.modules.filterobjects.MODE_RULES
            module.rules_file_name.value = rules_file
            module.rules_directory.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            module.rules_directory.custom_path = rules_dir
            module.y_name.value = "MyTargetObjects"
            module.run(workspace)
            target_objects = workspace.object_set.get_objects("MyTargetObjects")
            target_labels = target_objects.segmented
            self.assertTrue(numpy.all(target_labels[labels == 2] > 0))
            self.assertTrue(numpy.all(target_labels[labels != 2] == 0))
        finally:
            os.remove(rules_path)

    def test_08_02_filter_by_3_class_rule(self):
        rules_file_contents = (
            "IF (MyObjects_MyMeasurement > 2.0, [1.0,-1.0,-1.0], [-0.5,0.5,0.5])\n"
            "IF (MyObjects_MyMeasurement > 1.6, [0.5,0.5,-0.5], [-1.0,-1.0,1.0])\n")
        expected_class = [None, "3", "1", "2"]
        rules_path = tempfile.mktemp()
        fd = open(rules_path, 'wt')
        fd.write(rules_file_contents)
        fd.close()
        try:
            for rules_class in ("1", "2", "3"):
                labels = numpy.zeros((10, 20), int)
                labels[3:5, 4:9] = 1
                labels[7:9, 6:12] = 2
                labels[4:9, 14:18] = 3
                workspace, module = self.make_workspace({"MyObjects": labels})
                self.assertTrue(isinstance(module, cellprofiler.modules.filterobjects.FilterByObjectMeasurement))
                m = workspace.measurements
                m.add_measurement("MyObjects", "MyMeasurement",
                                  numpy.array([1.5, 2.3, 1.8]))
                rules_dir, rules_file = os.path.split(rules_path)
                module.x_name.value = "MyObjects"
                module.mode.value = cellprofiler.modules.filterobjects.MODE_RULES
                module.rules_file_name.value = rules_file
                module.rules_directory.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
                module.rules_directory.custom_path = rules_dir
                module.rules_class.value = rules_class
                module.y_name.value = "MyTargetObjects"
                module.run(workspace)
                target_objects = workspace.object_set.get_objects("MyTargetObjects")
                target_labels = target_objects.segmented
                kept = expected_class.index(rules_class)
                self.assertTrue(numpy.all(target_labels[labels == kept] > 0))
                self.assertTrue(numpy.all(target_labels[labels != kept] == 0))
        finally:
            os.remove(rules_path)

    def test_09_01_discard_border_objects(self):
        '''Test the mode to discard border objects'''
        labels = numpy.zeros((10, 10), int)
        labels[1:4, 0:3] = 1
        labels[4:8, 1:5] = 2
        labels[:, 9] = 3

        expected = numpy.zeros((10, 10), int)
        expected[4:8, 1:5] = 1

        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(expected == output_objects.segmented))

    def test_09_02_discard_mask_objects(self):
        '''Test discarding objects that touch the mask of objects parent img'''
        mask = numpy.ones((10, 10), bool)
        mask[5, 5] = False
        labels = numpy.zeros((10, 10), int)
        labels[1:4, 1:4] = 1
        labels[5:8, 5:8] = 2
        expected = labels.copy()
        expected[expected == 2] = 0

        workspace, module = self.make_workspace({})
        parent_image = cellprofiler.image.Image(numpy.zeros((10, 10)), mask=mask)
        workspace.image_set.add(INPUT_IMAGE, parent_image)

        input_objects = cellprofiler.object.Objects()
        input_objects.segmented = labels
        input_objects.parent_image = parent_image

        workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(numpy.all(expected == output_objects.segmented))

    def test_10_01_unedited_segmented(self):
        # Test transferral of unedited segmented segmentation
        # from source to target

        unedited = numpy.zeros((10, 10), dtype=int)
        unedited[0:4, 0:4] = 1
        unedited[6:8, 0:4] = 2
        unedited[6:8, 6:8] = 3
        segmented = numpy.zeros((10, 10), dtype=int)
        segmented[6:8, 6:8] = 1
        segmented[6:8, 0:4] = 2

        workspace, module = self.make_workspace({})
        input_objects = cellprofiler.object.Objects()
        workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)
        input_objects.segmented = segmented
        input_objects.unedited_segmented = unedited
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.mode.value = cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
        m = workspace.measurements
        m[INPUT_OBJECTS, TEST_FTR] = numpy.array([2, 1])
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        numpy.testing.assert_equal(output_objects.unedited_segmented, unedited)

    def test_10_02_small_removed_segmented(self):
        # Test output objects' small_removed_segmented
        #
        # It should be the small_removed_segmented of the
        # source minus the filtered

        unedited = numpy.zeros((10, 10), dtype=int)
        unedited[0:4, 0:4] = 1
        unedited[6:8, 0:4] = 2
        unedited[6:8, 6:8] = 3
        segmented = numpy.zeros((10, 10), dtype=int)
        segmented[6:8, 6:8] = 1
        segmented[6:8, 0:4] = 2

        workspace, module = self.make_workspace({})
        input_objects = cellprofiler.object.Objects()
        input_objects.segmented = segmented
        input_objects.unedited_segmented = unedited
        workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.mode.value = cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
        m = workspace.measurements
        m[INPUT_OBJECTS, TEST_FTR] = numpy.array([2, 1])
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        small_removed = output_objects.small_removed_segmented
        mask = (unedited != 3) & (unedited != 0)
        self.assertTrue(numpy.all(small_removed[mask] != 0))
        self.assertTrue(numpy.all(small_removed[~mask] == 0))

    def test_11_00_classify_none(self):
        workspace, module = self.make_workspace(
            {INPUT_OBJECTS : numpy.zeros((10, 10), int)})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        with self.make_classifier(module, numpy.zeros(0, int), classes = [1]):
            workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(0)
            module.run(workspace)
            output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
            self.assertEqual(output_objects.count, 0)

    def test_11_01_classify_true(self):
        labels = numpy.zeros((10, 10), int)
        labels[4:7, 4:7] = 1
        workspace, module = self.make_workspace(
            {INPUT_OBJECTS : labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        with self.make_classifier(module, numpy.ones(1, int), classes = [1, 2]):
            workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(1)
            module.run(workspace)
            output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
            self.assertEqual(output_objects.count, 1)

    def test_11_02_classify_false(self):
        labels = numpy.zeros((10, 10), int)
        labels[4:7, 4:7] = 1
        workspace, module = self.make_workspace(
            {INPUT_OBJECTS : labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        with self.make_classifier(module, numpy.ones(1, int)*2, classes = [1, 2]):
            workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(1)
            module.run(workspace)
            output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
            self.assertEqual(output_objects.count, 0)

    def test_11_03_classify_many(self):
        labels = numpy.zeros((10, 10), int)
        labels[1:4, 1:4] = 1
        labels[5:7, 5:7] = 2
        workspace, module = self.make_workspace(
            {INPUT_OBJECTS : labels})
        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        with self.make_classifier(
            module, numpy.array([1, 2]), classes = [1, 2]):
            workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(2)
            module.run(workspace)
            output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
            self.assertEqual(output_objects.count, 1)
            labels_out = output_objects.get_labels()[0][0]
            numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
            numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)

    def test_measurements(self):
        workspace, module = self.make_workspace(object_dict={
            INPUT_OBJECTS: numpy.zeros((10, 10), dtype=numpy.uint8),
            "additional_objects": numpy.zeros((10, 10), dtype=numpy.uint8)
        })

        module.x_name.value = INPUT_OBJECTS

        module.y_name.value = OUTPUT_OBJECTS

        module.add_additional_object()

        module.additional_objects[0].object_name.value = "additional_objects"

        module.additional_objects[0].target_name.value = "additional_result"

        module.measurements[0].measurement.value = TEST_FTR

        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL

        measurements = workspace.measurements

        measurements.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))

        module.run(workspace)

        object_names = [(INPUT_OBJECTS, OUTPUT_OBJECTS), ("additional_objects", "additional_result")]

        for input_object_name, output_object_name in object_names:
            assert measurements.has_current_measurements(
                cellprofiler.measurement.IMAGE,
                cellprofiler.measurement.FF_COUNT % output_object_name
            )

            assert measurements.has_current_measurements(
                input_object_name,
                cellprofiler.measurement.FF_CHILDREN_COUNT % output_object_name
            )

            output_object_features = [
                cellprofiler.measurement.FF_PARENT % input_object_name,
                cellprofiler.measurement.M_LOCATION_CENTER_X,
                cellprofiler.measurement.M_LOCATION_CENTER_Y,
                cellprofiler.measurement.M_LOCATION_CENTER_Z,
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER
            ]

            for feature in output_object_features:
                assert measurements.has_current_measurements(output_object_name, feature)

    def test_discard_border_objects_volume(self):
        labels = numpy.zeros((9, 16, 16), dtype=numpy.uint8)
        labels[:3, 1:5, 1:5] = 1       # touches bottom z-slice
        labels[-3:, -5:-1, -5:-1] = 2  # touches top z-slice
        labels[2:7, 6:10, 6:10] = 3
        labels[2:5, :5, -5:-1] = 4     # touches top edge
        labels[6:9, -5:-1, :5] = 5     # touches left edge

        expected = numpy.zeros_like(labels)
        expected[2:7, 6:10, 6:10] = 1

        src_objects = cellprofiler.object.Objects()
        src_objects.segmented = labels

        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER

        module.run(workspace)

        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)

        numpy.testing.assert_array_equal(output_objects.segmented, expected)

    def test_keep_maximal_per_object_most_overlap_volume(self):
        labels = numpy.zeros((1, 10, 20), int)
        labels[:, :, :10] = 1
        labels[:, :, 10:] = 2

        sub_labels = numpy.zeros((1, 10, 20), int)
        sub_labels[:, 2, 4] = 1
        sub_labels[:, 4:6, 8:15] = 2
        sub_labels[:, 8, 15] = 3

        expected = sub_labels * (sub_labels != 3)

        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels})

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
        module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP

        m = workspace.measurements

        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

        module.run(workspace)

        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_keep_minimal_per_object_both_parents_image(self):
        labels = numpy.zeros((10, 20), int)
        labels[:, :10] = 1
        labels[:, 10:] = 2

        sub_labels = numpy.zeros((10, 20), int)
        sub_labels[2, 4] = 1
        sub_labels[4:6, 8:15] = 2
        sub_labels[8, 15] = 3

        expected = numpy.zeros_like(sub_labels)
        expected[2, 4] = 1
        expected[8, 15] = 2

        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels})

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
        module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_BOTH

        m = workspace.measurements

        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

        module.run(workspace)

        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_keep_maximal_both_parents_volume(self):
        labels = numpy.zeros((2, 10, 20), int)
        labels[:, :, :10] = 1
        labels[:, :, 10:] = 2

        sub_labels = numpy.zeros((2, 10, 20), int)
        sub_labels[:, 2, 4] = 1
        sub_labels[:, 4:6, 8:15] = 2
        sub_labels[:, 8, 15] = 3

        expected = numpy.zeros_like(sub_labels)
        expected[sub_labels == 2] = 1

        workspace, module = self.make_workspace({INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels})

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.enclosing_object_name.value = ENCLOSING_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
        module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_BOTH

        m = workspace.measurements

        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

        module.run(workspace)

        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

        self.assertTrue(numpy.all(labels.segmented == expected))

    def test_keep_maximal_volume(self):
        labels = numpy.zeros((2, 10, 20), int)
        labels[:, 2, 4] = 1
        labels[:, 4:6, 8:15] = 2
        labels[:, 8, 15] = 3

        expected = numpy.zeros_like(labels)
        expected[labels == 2] = 1

        workspace, module = self.make_workspace({INPUT_OBJECTS: labels})

        module.x_name.value = INPUT_OBJECTS
        module.y_name.value = OUTPUT_OBJECTS
        module.measurements[0].measurement.value = TEST_FTR
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL

        m = workspace.measurements

        m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

        module.run(workspace)

        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

        self.assertTrue(numpy.all(labels.segmented == expected))


class FakeClassifier(object):
    def __init__(self, answers, classes):
        '''initializer

        answers - a vector of answers to be returned by "predict"

        classes - a vector of class numbers to be used to populate self.classes_
        '''
        self.answers_ = answers
        self.classes_ = classes

    def predict(self, *args, **kwargs):
        return self.answers_

def make_classifier_pickle(answers, classes, class_names, name, feature_names):
    '''Make a pickle of a fake classifier

    answers - the answers you want to get back after calling classifier.predict
    classes - the class #s for the answers.
    class_names - one name per class in the order they appear in classes
    name - the name of the classifier
    feature_names - the names of the features fed into the classifier
    '''
    classifier = FakeClassifier(answers, classes)
    return cPickle.dumps([classifier, class_names, name, feature_names])
