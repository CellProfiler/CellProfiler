'''test_trackobjects.py - testing of the TrackObjects module
'''

import unittest
from StringIO import StringIO

import numpy as np

import cellprofiler.measurement
from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
from centrosome.filter import permutations

import cellprofiler.modules.trackobjects as T
from cellprofiler.measurement import C_COUNT

OBJECT_NAME = "objects"


class TestTrackObjects(unittest.TestCase):
    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9227

TrackObjects:[module_num:1|svn_version:\'9227\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Choose a tracking method:LAP
    Select the objects to track:Nuclei
    Select measurement to use:AreaShape_Area
    Select pixel distance:80
    Select display option:Color and Number
    Save color-coded image?:No
    Name the output image:TrackedCells
    Cost of being born:100
    Cost of dying:100
    Do you want to run the second phase of the LAP algorithm?:Yes
    Gap cost\x3A:40
    Split alternative cost\x3A:41
    Merge alternative cost\x3A:42
    Maximum gap displacement\x3A:53
    Maximum split score\x3A:54
    Maximum merge score\x3A:55
    Maximum gap:6
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, T.TrackObjects))
        self.assertEqual(module.tracking_method, "LAP")
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.pixel_radius.value, 80)
        self.assertEqual(module.display_type.value, "Color and Number")
        self.assertFalse(module.wants_image)
        self.assertEqual(module.measurement, "AreaShape_Area")
        self.assertEqual(module.image_name, "TrackedCells")
        self.assertTrue(module.wants_second_phase)
        self.assertEqual(module.split_cost, 41)
        self.assertEqual(module.merge_cost, 42)
        self.assertEqual(module.max_gap_score, 53)
        self.assertEqual(module.max_split_score, 54)
        self.assertEqual(module.max_merge_score, 55)
        self.assertEqual(module.max_frame_distance, 6)

    def test_01_05_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10400

TrackObjects:[module_num:1|svn_version:\'10373\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Choose a tracking method:Measurements
    Select the objects to track:Objs
    Select object measurement to use for tracking:Slothfulness
    Maximum pixel distance to consider matches:50
    Select display option:Color and Number
    Save color-coded image?:Yes
    Name the output image:TrackByLAP
    Motion model(s)\x3A:Both
    # standard deviations for radius:3
    Radius limit:3.0,10.0
    Run the second phase of the LAP algorithm?:Yes
    Gap cost:40
    Split alternative cost:1
    Merge alternative cost:1
    Maximum gap displacement:51
    Maximum split score:52
    Maximum merge score:53
    Maximum gap:4

TrackObjects:[module_num:2|svn_version:\'10373\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Choose a tracking method:Overlap
    Select the objects to track:Objs
    Select object measurement to use for tracking:Prescience
    Maximum pixel distance to consider matches:50
    Select display option:Color Only
    Save color-coded image?:No
    Name the output image:TrackByLAP
    Motion model(s)\x3A:Random
    # standard deviations for radius:3
    Radius limit:3.0,10.0
    Run the second phase of the LAP algorithm?:No
    Gap cost:40
    Split alternative cost:1
    Merge alternative cost:1
    Maximum gap displacement:51
    Maximum split score:52
    Maximum merge score:53
    Maximum gap:4

TrackObjects:[module_num:1|svn_version:\'10373\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Choose a tracking method:Distance
    Select the objects to track:Objs
    Select object measurement to use for tracking:Trepidation
    Maximum pixel distance to consider matches:50
    Select display option:Color and Number
    Save color-coded image?:Yes
    Name the output image:TrackByLAP
    Motion model(s)\x3A:Velocity
    # standard deviations for radius:3
    Radius limit:3.0,10.0
    Run the second phase of the LAP algorithm?:Yes
    Gap cost:40
    Split alternative cost:1
    Merge alternative cost:1
    Maximum gap displacement:51
    Maximum split score:52
    Maximum merge score:53
    Maximum gap:4
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        for module, tracking_method, model, save_img, phase2, meas, dop in zip(
                pipeline.modules(),
                ("Measurements", "Overlap", "Distance"),
                (T.M_BOTH, T.M_RANDOM, T.M_VELOCITY),
                (True, False, True), (True, False, True),
                ("Slothfulness", "Prescience", "Trepidation"),
                (T.DT_COLOR_AND_NUMBER, T.DT_COLOR_ONLY, T.DT_COLOR_AND_NUMBER)):
            self.assertTrue(isinstance(module, T.TrackObjects))
            self.assertEqual(module.tracking_method, tracking_method)
            self.assertEqual(module.model, model)
            self.assertEqual(module.wants_image.value, save_img)
            self.assertEqual(module.wants_second_phase.value, phase2)
            self.assertEqual(module.measurement, meas)
            self.assertEqual(module.pixel_radius, 50)
            self.assertEqual(module.display_type, dop)
            self.assertEqual(module.image_name, "TrackByLAP")
            self.assertEqual(module.radius_std, 3)
            self.assertEqual(module.radius_limit.min, 3.0)
            self.assertEqual(module.radius_limit.max, 10.0)
            self.assertEqual(module.gap_cost, 40)
            self.assertEqual(module.split_cost, 1)
            self.assertEqual(module.merge_cost, 1)
            self.assertEqual(module.max_gap_score, 51)
            self.assertEqual(module.max_split_score, 52)
            self.assertEqual(module.max_merge_score, 53)
            self.assertEqual(module.max_frame_distance, 4)

    def test_01_06_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140723174500
GitHash:6c2d896
ModuleCount:1
HasImagePlaneDetails:False

TrackObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Choose a tracking method:LAP
    Select the objects to track:Turtles
    Select object measurement to use for tracking:Steadiness
    Maximum pixel distance to consider matches:44
    Select display option:Color and Number
    Save color-coded image?:No
    Name the output image:TrackedTurtles
    Select the motion model:Both
    Number of standard deviations for search radius:3.0
    Search radius limit, in pixel units (Min,Max):3.0,11.0
    Run the second phase of the LAP algorithm?:Yes
    Gap cost:39
    Split alternative cost:41
    Merge alternative cost:42
    Maximum gap displacement, in frames:6
    Maximum split score:51
    Maximum merge score:52
    Maximum gap:8
    Filter objects by lifetime?:No
    Filter using a minimum lifetime?:Yes
    Minimum lifetime:2
    Filter using a maximum lifetime?:No
    Maximum lifetime:1000
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        m = pipeline.modules()[0]
        assert isinstance(m, T.TrackObjects)
        self.assertEqual(m.tracking_method, "LAP")
        self.assertEqual(m.object_name, "Turtles")
        self.assertEqual(m.measurement, "Steadiness")
        self.assertEqual(m.pixel_radius, 44)
        self.assertEqual(m.display_type, T.DT_COLOR_AND_NUMBER)
        self.assertFalse(m.wants_image)
        self.assertEqual(m.image_name, "TrackedTurtles")
        self.assertEqual(m.model, T.M_BOTH)
        self.assertEqual(m.radius_std, 3)
        self.assertEqual(m.radius_limit.min, 3)
        self.assertEqual(m.radius_limit.max, 11)
        self.assertTrue(m.wants_second_phase)
        self.assertEqual(m.gap_cost, 39)
        self.assertEqual(m.split_cost, 41)
        self.assertEqual(m.merge_cost, 42)
        self.assertEqual(m.max_frame_distance, 8)
        self.assertTrue(m.wants_minimum_lifetime)
        self.assertEqual(m.min_lifetime, 2)
        self.assertFalse(m.wants_maximum_lifetime)
        self.assertEqual(m.max_lifetime, 1000)

    def test_01_07_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140723174500
GitHash:6c2d896
ModuleCount:1
HasImagePlaneDetails:False

TrackObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Choose a tracking method:LAP
    Select the objects to track:Turtles
    Select object measurement to use for tracking:Steadiness
    Maximum pixel distance to consider matches:44
    Select display option:Color and Number
    Save color-coded image?:No
    Name the output image:TrackedTurtles
    Select the motion model:Both
    Number of standard deviations for search radius:3.0
    Search radius limit, in pixel units (Min,Max):3.0,11.0
    Run the second phase of the LAP algorithm?:Yes
    Gap cost:39
    Split alternative cost:41
    Merge alternative cost:42
    Maximum gap displacement, in frames:6
    Maximum split score:51
    Maximum merge score:52
    Maximum gap:8
    Filter objects by lifetime?:No
    Filter using a minimum lifetime?:Yes
    Minimum lifetime:2
    Filter using a maximum lifetime?:No
    Maximum lifetime:1000
    Mitosis alternative cost:79
    Mitosis max distance:41
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        m = pipeline.modules()[0]
        assert isinstance(m, T.TrackObjects)
        self.assertEqual(m.tracking_method, "LAP")
        self.assertEqual(m.object_name, "Turtles")
        self.assertEqual(m.measurement, "Steadiness")
        self.assertEqual(m.pixel_radius, 44)
        self.assertEqual(m.display_type, T.DT_COLOR_AND_NUMBER)
        self.assertFalse(m.wants_image)
        self.assertEqual(m.image_name, "TrackedTurtles")
        self.assertEqual(m.model, T.M_BOTH)
        self.assertEqual(m.radius_std, 3)
        self.assertEqual(m.radius_limit.min, 3)
        self.assertEqual(m.radius_limit.max, 11)
        self.assertTrue(m.wants_second_phase)
        self.assertEqual(m.gap_cost, 39)
        self.assertEqual(m.split_cost, 41)
        self.assertEqual(m.merge_cost, 42)
        self.assertEqual(m.max_frame_distance, 8)
        self.assertTrue(m.wants_minimum_lifetime)
        self.assertEqual(m.min_lifetime, 2)
        self.assertFalse(m.wants_maximum_lifetime)
        self.assertEqual(m.max_lifetime, 1000)
        self.assertEqual(m.mitosis_cost, 79)
        self.assertEqual(m.mitosis_max_distance, 41)

    def runTrackObjects(self, labels_list, fn=None, measurement=None):
        '''Run two cycles of TrackObjects

        labels1 - the labels matrix for the first cycle
        labels2 - the labels matrix for the second cycle
        fn - a callback function called with the module and workspace. It has
             the signature, fn(module, workspace, n) where n is 0 when
             called prior to prepare_run, 1 prior to first iteration
             and 2 prior to second iteration.

        returns the measurements
        '''
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.pixel_radius.value = 50
        module.measurement.value = "measurement"
        measurements = cpmeas.Measurements()
        measurements.add_all_measurements(cpmeas.IMAGE, cpp.GROUP_NUMBER,
                                          [1] * len(labels_list))
        measurements.add_all_measurements(cpmeas.IMAGE, cpp.GROUP_INDEX,
                                          range(1, len(labels_list) + 1))
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()

        if fn:
            fn(module, None, 0)
        module.prepare_run(cpw.Workspace(pipeline, module, None, None,
                                         measurements, image_set_list))

        first = True
        for labels, index in zip(labels_list, range(len(labels_list))):
            object_set = cpo.ObjectSet()
            objects = cpo.Objects()
            objects.segmented = labels
            object_set.add_objects(objects, OBJECT_NAME)
            image_set = image_set_list.get_image_set(index)
            if first:
                first = False
            else:
                measurements.next_image_set()
            if measurement is not None:
                measurements.add_measurement(OBJECT_NAME, "measurement",
                                             np.array(measurement[index]))
            workspace = cpw.Workspace(pipeline, module, image_set,
                                      object_set, measurements, image_set_list)
            if fn:
                fn(module, workspace, index + 1)

            module.run(workspace)
        return measurements

    def test_02_01_track_nothing(self):
        '''Run TrackObjects on an empty labels matrix'''
        columns = []

        def fn(module, workspace, index, columns=columns):
            if workspace is not None and index == 0:
                columns += module.get_measurement_columns(workspace.pipeline)

        measurements = self.runTrackObjects((np.zeros((10, 10), int),
                                             np.zeros((10, 10), int)), fn)

        features = [feature
                    for feature in measurements.get_feature_names(OBJECT_NAME)
                    if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == OBJECT_NAME]))
        for feature in T.F_ALL:
            name = "_".join((T.F_PREFIX, feature, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(value), 0)

        features = [feature for feature in measurements.get_feature_names(cpmeas.IMAGE)
                    if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == cpmeas.IMAGE]))
        for feature in T.F_IMAGE_ALL:
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_image_measurement(name)
            self.assertEqual(value, 0)

    def test_02_01_00_track_one_then_nothing(self):
        '''Run track objects on an object that disappears

        Regression test of IMG-1090
        '''
        labels = np.zeros((10, 10), int)
        labels[3:6, 2:7] = 1
        measurements = self.runTrackObjects((labels,
                                             np.zeros((10, 10), int)))
        feature = "_".join((T.F_PREFIX, T.F_LOST_OBJECT_COUNT,
                            OBJECT_NAME, "50"))
        value = measurements.get_current_image_measurement(feature)
        self.assertEqual(value, 1)

    def test_02_02_track_one_distance(self):
        '''Track an object that doesn't move using distance'''
        labels = np.zeros((10, 10), int)
        labels[3:6, 2:7] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = "Distance"

        measurements = self.runTrackObjects((labels, labels), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertAlmostEqual(m(T.F_TRAJECTORY_X), 0)
        self.assertAlmostEqual(m(T.F_TRAJECTORY_Y), 0)
        self.assertAlmostEqual(m(T.F_DISTANCE_TRAVELED), 0)
        self.assertAlmostEqual(m(T.F_INTEGRATED_DISTANCE), 0)
        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 1)
        self.assertEqual(m(T.F_PARENT_IMAGE_NUMBER), 1)
        self.assertEqual(m(T.F_LIFETIME), 2)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
        self.check_relationships(measurements, [1], [1], [2], [1])

    def test_02_03_track_one_moving(self):
        '''Track an object that moves'''

        labels_list = []
        distance = 0
        last_i, last_j = (0, 0)
        for i_off, j_off in ((0, 0), (2, 0), (2, 1), (0, 1)):
            distance = i_off - last_i + j_off - last_j
            last_i, last_j = (i_off, j_off)
            labels = np.zeros((10, 10), int)
            labels[4 + i_off:7 + i_off, 4 + j_off:7 + j_off] = 1
            labels_list.append(labels)

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 3
                module.tracking_method.value = "Distance"

        measurements = self.runTrackObjects(labels_list, fn)

        def m(feature, expected):
            name = "_".join((T.F_PREFIX, feature, "3"))
            value_set = measurements.get_all_measurements(OBJECT_NAME, name)
            self.assertEqual(len(expected), len(value_set))
            for values, x in zip(value_set, expected):
                self.assertEqual(len(values), 1)
                self.assertAlmostEqual(values[0], x)

        m(T.F_TRAJECTORY_X, [0, 0, 1, 0])
        m(T.F_TRAJECTORY_Y, [0, 2, 0, -2])
        m(T.F_DISTANCE_TRAVELED, [0, 2, 1, 2])
        m(T.F_INTEGRATED_DISTANCE, [0, 2, 3, 5])
        m(T.F_LABEL, [1, 1, 1, 1])
        m(T.F_LIFETIME, [1, 2, 3, 4])
        m(T.F_LINEARITY, [1, 1, np.sqrt(5) / 3, 1.0 / 5.0])

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "3"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
        image_numbers = np.arange(1, len(labels_list) + 1)
        object_numbers = np.ones(len(image_numbers))
        self.check_relationships(measurements,
                                 image_numbers[:-1], object_numbers[:-1],
                                 image_numbers[1:], object_numbers[1:])

    def test_02_04_track_split(self):
        '''Track an object that splits'''
        labels1 = np.zeros((11, 9), int)
        labels1[1:10, 1:8] = 1
        labels2 = np.zeros((10, 10), int)
        labels2[1:6, 1:8] = 1
        labels2[6:10, 1:8] = 2

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 5
                module.tracking_method.value = "Distance"

        measurements = self.runTrackObjects((labels1, labels2, labels2), fn)

        def m(feature, idx):
            name = "_".join((T.F_PREFIX, feature, "5"))
            values = measurements.get_measurement(OBJECT_NAME, name, idx + 1)
            self.assertEqual(len(values), 2)
            return values

        labels = m(T.F_LABEL, 2)
        self.assertEqual(len(labels), 2)
        self.assertTrue(np.all(labels == 1))
        parents = m(T.F_PARENT_OBJECT_NUMBER, 1)
        self.assertTrue(np.all(parents == 1))
        self.assertTrue(np.all(m(T.F_PARENT_IMAGE_NUMBER, 1) == 1))
        parents = m(T.F_PARENT_OBJECT_NUMBER, 2)
        self.assertTrue(np.all(parents == np.array([1, 2])))
        self.assertTrue(np.all(m(T.F_PARENT_IMAGE_NUMBER, 2) == 2))

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "5"))
            return measurements.get_all_measurements(cpmeas.IMAGE, name)[1]

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 1)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
        self.check_relationships(measurements,
                                 [1, 1, 2, 2], [1, 1, 1, 2],
                                 [2, 2, 3, 3], [1, 2, 1, 2])

    def test_02_05_track_negative(self):
        '''Track unrelated objects'''
        labels1 = np.zeros((10, 10), int)
        labels1[1:5, 1:5] = 1
        labels2 = np.zeros((10, 10), int)
        labels2[6:9, 6:9] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = "Distance"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 0)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)

    def test_02_06_track_ambiguous(self):
        '''Track disambiguation from among two possible parents'''
        labels1 = np.zeros((20, 20), int)
        labels1[1:4, 1:4] = 1
        labels1[16:19, 16:19] = 2
        labels2 = np.zeros((20, 20), int)
        labels2[10:15, 10:15] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 20
                module.tracking_method.value = "Distance"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "20"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 2)

    def test_03_01_overlap_positive(self):
        '''Track overlapping objects'''
        labels1 = np.zeros((10, 10), int)
        labels1[3:6, 4:7] = 1
        labels2 = np.zeros((10, 10), int)
        labels2[4:7, 5:9] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = "Overlap"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 1)

    def test_03_02_overlap_negative(self):
        '''Track objects that don't overlap'''
        labels1 = np.zeros((20, 20), int)
        labels1[3:6, 4:7] = 1
        labels2 = np.zeros((20, 20), int)
        labels2[14:17, 15:19] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = "Overlap"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 0)

    def test_03_03_overlap_ambiguous(self):
        '''Track an object that overlaps two parents'''
        labels1 = np.zeros((20, 20), int)
        labels1[1:5, 1:5] = 1
        labels1[15:19, 15:19] = 2
        labels2 = np.zeros((20, 20), int)
        labels2[4:18, 4:18] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = "Overlap"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 2)

    def test_04_01_measurement_positive(self):
        '''Test tracking an object by measurement'''
        labels1 = np.zeros((10, 10), int)
        labels1[3:6, 4:7] = 1
        labels2 = np.zeros((10, 10), int)
        labels2[4:7, 5:9] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = "Measurements"

        measurements = self.runTrackObjects((labels1, labels2), fn, [[1], [1]])

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 1)

    def test_04_02_measurement_negative(self):
        '''Test tracking with too great a jump between successive images'''
        labels1 = np.zeros((20, 20), int)
        labels1[3:6, 4:7] = 1
        labels2 = np.zeros((20, 20), int)
        labels2[14:17, 15:19] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = "Measurements"

        measurements = self.runTrackObjects((labels1, labels2), fn, [[1], [1]])

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 0)

    def test_04_03_ambiguous(self):
        '''Test measurement with ambiguous parent choice'''
        labels1 = np.zeros((20, 20), int)
        labels1[1:5, 1:5] = 1
        labels1[15:19, 15:19] = 2
        labels2 = np.zeros((20, 20), int)
        labels2[6:14, 6:14] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 4
                module.tracking_method.value = "Measurements"

        measurements = self.runTrackObjects((labels1, labels2), fn, [[1, 10], [9]])

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "4"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 2)

    def test_04_04_cross_numbered_objects(self):
        '''Test labeling when object 1 in one image becomes object 2 in next'''

        i, j = np.mgrid[0:10, 0:20]
        labels = (i > 5) + (j > 10) * 2
        pp = np.array(list(permutations([1, 2, 3, 4])))

        def fn(module, workspace, idx):
            if idx == 0:
                module.tracking_method.value = "LAP"

        measurements = self.runTrackObjects([np.array(p)[labels] for p in pp], fn)

        def m(feature, i):
            name = "_".join((T.F_PREFIX, feature))
            values = measurements[OBJECT_NAME, name, i + 1]
            self.assertEqual(len(values), 4)
            return values

        for i, p in enumerate(pp):
            l = m(T.F_LABEL, i)
            np.testing.assert_array_equal(np.arange(1, 5), p[l - 1])
            if i > 0:
                p_prev = pp[i - 1]
                order = np.lexsort([p])
                expected_po = p_prev[order]
                po = m(T.F_PARENT_OBJECT_NUMBER, i)
                np.testing.assert_array_equal(po, expected_po)
                pi = m(T.F_PARENT_IMAGE_NUMBER, i)
                np.testing.assert_array_equal(pi, i)
        image_numbers, _ = np.mgrid[1:(len(pp) + 1), 0:4]
        self.check_relationships(
            measurements,
            image_numbers[:-1, :].flatten(), pp[:-1, :].flatten(),
            image_numbers[1:, :].flatten(), pp[1:, :].flatten())

    def test_05_01_measurement_columns(self):
        '''Test get_measurement_columns function'''
        module = T.TrackObjects()
        module.object_name.value = OBJECT_NAME
        module.tracking_method.value = "Distance"
        module.pixel_radius.value = 10
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), len(T.F_ALL) + len(T.F_IMAGE_ALL))
        for object_name, features in ((OBJECT_NAME, T.F_ALL),
                                      (cpmeas.IMAGE, T.F_IMAGE_ALL)):
            for feature in features:
                if object_name == OBJECT_NAME:
                    name = "_".join((T.F_PREFIX, feature, "10"))
                else:
                    name = "_".join((T.F_PREFIX, feature,
                                     OBJECT_NAME, "10"))
                index = [column[1] for column in columns].index(name)
                self.assertTrue(index != -1)
                column = columns[index]
                self.assertEqual(column[0], object_name)

    def test_05_02_measurement_columns_lap(self):
        '''Test get_measurement_columns function for LAP'''
        module = T.TrackObjects()
        module.object_name.value = OBJECT_NAME
        module.tracking_method.value = "LAP"
        module.model.value = T.M_BOTH
        second_phase = [T.F_LINKING_DISTANCE, T.F_MOVEMENT_MODEL]
        for wants in (True, False):
            module.wants_second_phase.value = wants
            columns = module.get_measurement_columns(None)
            # 2, 2, 4 for the static model
            # 4, 4, 16 for the velocity model
            other_features = [T.F_AREA, T.F_LINKING_DISTANCE, T.F_LINK_TYPE,
                              T.F_MOVEMENT_MODEL, T.F_STANDARD_DEVIATION]
            if wants:
                other_features += [
                    T.F_GAP_LENGTH, T.F_GAP_SCORE, T.F_MERGE_SCORE,
                    T.F_SPLIT_SCORE, T.F_MITOSIS_SCORE]
            self.assertEqual(len(columns), len(T.F_ALL) + len(T.F_IMAGE_ALL) +
                             len(other_features) + 2 + 2 + 4 + 4 + 4 + 16)
            kalman_features = [
                T.kalman_feature(T.F_STATIC_MODEL, T.F_STATE, T.F_X),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_STATE, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_STATE, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_STATE, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_STATE, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_STATE, T.F_VY),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_NOISE, T.F_X),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_NOISE, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_NOISE, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_NOISE, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_NOISE, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_NOISE, T.F_VY),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_COV, T.F_X, T.F_X),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_COV, T.F_X, T.F_Y),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_COV, T.F_Y, T.F_X),
                T.kalman_feature(T.F_STATIC_MODEL, T.F_COV, T.F_X, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_X, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_X, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_X, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_X, T.F_VY),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_Y, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_Y, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_Y, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_Y, T.F_VY),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VX, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VX, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VX, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VX, T.F_VY),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VY, T.F_X),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VY, T.F_Y),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VY, T.F_VX),
                T.kalman_feature(T.F_VELOCITY_MODEL, T.F_COV, T.F_VY, T.F_VY)]
            for object_name, features in (
                    (OBJECT_NAME, T.F_ALL + kalman_features + other_features),
                    (cpmeas.IMAGE, T.F_IMAGE_ALL)):
                for feature in features:
                    if object_name == OBJECT_NAME:
                        name = "_".join((T.F_PREFIX, feature))
                    else:
                        name = "_".join((T.F_PREFIX, feature,
                                         OBJECT_NAME))
                    index = [column[1] for column in columns].index(name)
                    self.assertTrue(index != -1)
                    column = columns[index]
                    self.assertEqual(column[0], object_name)
                    if wants or feature in second_phase:
                        self.assertEqual(len(column), 4)
                        self.assertTrue(cpmeas.MCA_AVAILABLE_POST_GROUP in column[3])
                        self.assertTrue(column[3][cpmeas.MCA_AVAILABLE_POST_GROUP])
                    else:
                        self.assertTrue(
                            (len(column) == 3) or
                            (cpmeas.MCA_AVAILABLE_POST_GROUP not in column[3]) or
                            (not column[3][cpmeas.MCA_AVAILABLE_POST_GROUP]))

    def test_06_01_measurements(self):
        '''Test the different measurement pieces'''
        module = T.TrackObjects()
        module.object_name.value = OBJECT_NAME
        module.image_name.value = "image"
        module.pixel_radius.value = 10
        categories = module.get_categories(None, "Foo")
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(None, OBJECT_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], T.F_PREFIX)
        features = module.get_measurements(None, OBJECT_NAME, "Foo")
        self.assertEqual(len(features), 0)
        features = module.get_measurements(None, OBJECT_NAME, T.F_PREFIX)
        self.assertEqual(len(features), len(T.F_ALL))
        self.assertTrue(all([feature in T.F_ALL for feature in features]))
        scales = module.get_measurement_scales(None, OBJECT_NAME,
                                               T.F_PREFIX, "Foo", "image")
        self.assertEqual(len(scales), 0)
        for feature in T.F_ALL:
            scales = module.get_measurement_scales(None, OBJECT_NAME,
                                                   T.F_PREFIX, feature, "image")
            self.assertEqual(len(scales), 1)
            self.assertEqual(int(scales[0]), 10)

    def make_lap2_workspace(self, objs, nimages, group_numbers=None, group_indexes=None):
        '''Make a workspace to test the second half of LAP

        objs - a N x 5 array of "objects" composed of the
               following pieces per object
               objs[0] - image set # for object
               objs[1] - label for object
               objs[2] - parent image #
               objs[3] - parent object #
               objs[4] - x coordinate for object
               objs[5] - y coordinate for object
               objs[6] - area for object
        nimages - # of image sets
        group_numbers - group numbers for each image set, defaults to all 1
        group_indexes - group indexes for each image set, defaults to range
        '''
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.tracking_method.value = "LAP"
        module.wants_second_phase.value = True
        module.wants_lifetime_filtering.value = False
        module.wants_minimum_lifetime.value = False
        module.min_lifetime.value = 1
        module.wants_maximum_lifetime.value = False
        module.max_lifetime.value = 100

        module.pixel_radius.value = 50

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        m = cpmeas.Measurements()
        if objs.shape[0] > 0:
            nobjects = np.bincount(objs[:, 0].astype(int))
        else:
            nobjects = np.zeros(nimages, int)
        for i in range(nimages):
            m.next_image_set(i + 1)
            for index, feature, dtype in (
                    (1, module.measurement_name(T.F_LABEL), int),
                    (2, module.measurement_name(T.F_PARENT_IMAGE_NUMBER), int),
                    (3, module.measurement_name(T.F_PARENT_OBJECT_NUMBER), int),
                    (4, cellprofiler.measurement.M_LOCATION_CENTER_X, float),
                    (5, cellprofiler.measurement.M_LOCATION_CENTER_Y, float),
                    (6, module.measurement_name(T.F_AREA), float)):
                values = objs[objs[:, 0] == i, index].astype(dtype)
                m.add_measurement(OBJECT_NAME, feature, values, i + 1)
            m.add_measurement(cpmeas.IMAGE, "ImageNumber", i + 1)
            m.add_measurement(cpmeas.IMAGE, cpp.GROUP_NUMBER,
                              1 if group_numbers is None else group_numbers[i], i + 1)
            m.add_measurement(cpmeas.IMAGE, cpp.GROUP_INDEX,
                              i if group_indexes is None else group_indexes[i], i + 1)
            #
            # Add blanks of the right sizes for measurements that are recalculated
            #
            m.add_measurement(cpmeas.IMAGE, '_'.join((C_COUNT, OBJECT_NAME)),
                              nobjects[i], i + 1)
            for feature in (T.F_DISTANCE_TRAVELED, T.F_DISPLACEMENT,
                            T.F_INTEGRATED_DISTANCE, T.F_TRAJECTORY_X,
                            T.F_TRAJECTORY_Y, T.F_LINEARITY, T.F_LIFETIME,
                            T.F_FINAL_AGE, T.F_LINKING_DISTANCE,
                            T.F_LINK_TYPE, T.F_MOVEMENT_MODEL,
                            T.F_STANDARD_DEVIATION):
                dtype = int if feature in (
                    T.F_PARENT_OBJECT_NUMBER, T.F_PARENT_IMAGE_NUMBER,
                    T.F_LIFETIME, T.F_LINK_TYPE, T.F_MOVEMENT_MODEL) else float
                m.add_measurement(
                    OBJECT_NAME,
                    module.measurement_name(feature),
                    np.NaN * np.ones(nobjects[i], dtype) if feature == T.F_FINAL_AGE
                    else np.zeros(nobjects[i], dtype),
                    i + 1)
            for feature in (T.F_SPLIT_COUNT, T.F_MERGE_COUNT):
                m.add_measurement(cpmeas.IMAGE,
                                  module.image_measurement_name(feature),
                                  0, i + 1)
        #
        # Figure out how many new and lost objects per image set
        #
        label_sets = [set() for i in range(nimages)]
        for row in objs:
            label_sets[row[0]].add(row[1])
        if group_numbers is None:
            group_numbers = np.ones(nimages, int)
        if group_indexes is None:
            group_indexes = np.arange(nimages) + 1
        #
        # New objects are ones without matching labels in the previous set
        #
        for i in range(0, nimages):
            if group_indexes[i] == 1:
                new_objects = len(label_sets[i])
                lost_objects = 0
            else:
                new_objects = sum([1 for label in label_sets[i]
                                   if label not in label_sets[i - 1]])
                lost_objects = sum([1 for label in label_sets[i - 1]
                                    if label not in label_sets[i]])
            m.add_measurement(
                cpmeas.IMAGE,
                module.image_measurement_name(T.F_NEW_OBJECT_COUNT),
                new_objects, True, i + 1)
            m.add_measurement(
                cpmeas.IMAGE,
                module.image_measurement_name(T.F_LOST_OBJECT_COUNT),
                lost_objects, True, i + 1)
        m.image_set_number = nimages

        image_set_list = cpi.ImageSetList()
        for i in range(nimages):
            image_set = image_set_list.get_image_set(i)
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  m, image_set_list)
        return workspace, module

    def check_measurements(self, workspace, d):
        '''Check measurements against expected values

        workspace - workspace that was run
        d - dictionary of feature name and list of expected measurement values
        '''
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        module = workspace.module
        self.assertTrue(isinstance(module, T.TrackObjects))
        for feature, expected in d.iteritems():
            if np.isscalar(expected[0]):
                mname = module.image_measurement_name(feature)
                values = m.get_all_measurements(cpmeas.IMAGE, mname)
                self.assertEqual(len(expected), len(values),
                                 "Expected # image sets (%d) != actual (%d) for %s" %
                                 (len(expected), len(values), feature))
                self.assertTrue(all([v == e for v, e in zip(values, expected)]),
                                "Values don't match for " + feature)
            else:
                mname = module.measurement_name(feature)
                values = m.get_all_measurements(OBJECT_NAME, mname)
                self.assertEqual(len(expected), len(values),
                                 "Expected # image sets (%d) != actual (%d) for %s" %
                                 (len(expected), len(values), feature))
                for i, (e, v) in enumerate(zip(expected, values)):
                    self.assertEqual(len(e), len(v),
                                     "Expected # of objects (%d) != actual (%d) for %s:%d" %
                                     (len(e), len(v), feature, i))
                    np.testing.assert_almost_equal(v, e)

    def check_relationships(self, m,
                            expected_parent_image_numbers,
                            expected_parent_object_numbers,
                            expected_child_image_numbers,
                            expected_child_object_numbers):
        '''Check the relationship measurements against expected'''
        expected_parent_image_numbers = np.atleast_1d(expected_parent_image_numbers)
        expected_child_image_numbers = np.atleast_1d(expected_child_image_numbers)
        expected_parent_object_numbers = np.atleast_1d(expected_parent_object_numbers)
        expected_child_object_numbers = np.atleast_1d(expected_child_object_numbers)
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        r = m.get_relationships(
            1, T.R_PARENT, OBJECT_NAME, OBJECT_NAME)
        actual_parent_image_numbers = r[cpmeas.R_FIRST_IMAGE_NUMBER]
        actual_parent_object_numbers = r[cpmeas.R_FIRST_OBJECT_NUMBER]
        actual_child_image_numbers = r[cpmeas.R_SECOND_IMAGE_NUMBER]
        actual_child_object_numbers = r[cpmeas.R_SECOND_OBJECT_NUMBER]
        self.assertEqual(len(actual_parent_image_numbers),
                         len(expected_parent_image_numbers))
        #
        # Sort similarly
        #
        for i1, o1, i2, o2 in (
                (expected_parent_image_numbers, expected_parent_object_numbers,
                 expected_child_image_numbers, expected_child_object_numbers),
                (actual_parent_image_numbers, actual_parent_object_numbers,
                 actual_child_image_numbers, actual_child_object_numbers)):
            order = np.lexsort((i1, o1, i2, o2))
            for x in (i1, o1, i2, o2):
                x[:] = x[order]
        for expected, actual in zip(
                (expected_parent_image_numbers, expected_parent_object_numbers,
                 expected_child_image_numbers, expected_child_object_numbers),
                (actual_parent_image_numbers, actual_parent_object_numbers,
                 actual_child_image_numbers, actual_child_object_numbers)):
            np.testing.assert_array_equal(expected, actual)

    def test_07_01_lap_none(self):
        '''Run the second part of LAP on one image of nothing'''
        with self.MonkeyPatchedDelete(self):
            workspace, module = self.make_lap2_workspace(np.zeros((0, 7)), 1)
            self.assertTrue(isinstance(module, T.TrackObjects))
            module.run_as_data_tool(workspace)
            self.check_measurements(workspace, {
                T.F_LABEL: [np.zeros(0, int)],
                T.F_DISTANCE_TRAVELED: [np.zeros(0)],
                T.F_DISPLACEMENT: [np.zeros(0)],
                T.F_INTEGRATED_DISTANCE: [np.zeros(0)],
                T.F_TRAJECTORY_X: [np.zeros(0)],
                T.F_TRAJECTORY_Y: [np.zeros(0)],
                T.F_NEW_OBJECT_COUNT: [0],
                T.F_LOST_OBJECT_COUNT: [0],
                T.F_MERGE_COUNT: [0],
                T.F_SPLIT_COUNT: [0]
            })

    def test_07_02_lap_one(self):
        '''Run the second part of LAP on one image of one object'''
        with self.MonkeyPatchedDelete(self):
            workspace, module = self.make_lap2_workspace(
                np.array([[0, 1, 0, 0, 100, 100, 25]]), 1)
            self.assertTrue(isinstance(module, T.TrackObjects))
            module.run_as_data_tool(workspace)
            self.check_measurements(workspace, {
                T.F_LABEL: [np.array([1])],
                T.F_PARENT_IMAGE_NUMBER: [np.array([0])],
                T.F_PARENT_OBJECT_NUMBER: [np.array([0])],
                T.F_DISPLACEMENT: [np.zeros(1)],
                T.F_INTEGRATED_DISTANCE: [np.zeros(1)],
                T.F_TRAJECTORY_X: [np.zeros(1)],
                T.F_TRAJECTORY_Y: [np.zeros(1)],
                T.F_NEW_OBJECT_COUNT: [1],
                T.F_LOST_OBJECT_COUNT: [0],
                T.F_MERGE_COUNT: [0],
                T.F_SPLIT_COUNT: [0]
            })

    def test_07_03_bridge_gap(self):
        '''Bridge a gap of zero frames between two objects'''
        with self.MonkeyPatchedDelete(self):
            workspace, module = self.make_lap2_workspace(
                np.array([[0, 1, 0, 0, 1, 2, 25],
                          [2, 2, 0, 0, 101, 102, 25]]), 3)
            self.assertTrue(isinstance(module, T.TrackObjects))
            #
            # The cost of bridging the gap should be 141. We set the alternative
            # score to 142 so that bridging wins.
            #
            module.gap_cost.value = 142
            module.max_gap_score.value = 142
            module.run_as_data_tool(workspace)
            distance = np.array([np.sqrt(2 * 100 * 100)])
            self.check_measurements(workspace, {
                T.F_LABEL: [np.array([1]), np.zeros(0), np.array([1])],
                T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.zeros(0, int), np.array([1])],
                T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.zeros(0, int), np.array([1])],
                T.F_DISTANCE_TRAVELED: [np.zeros(1), np.zeros(0), distance],
                T.F_INTEGRATED_DISTANCE: [np.zeros(1), np.zeros(0), distance],
                T.F_TRAJECTORY_X: [np.zeros(1), np.zeros(0), np.array([100])],
                T.F_TRAJECTORY_Y: [np.zeros(1), np.zeros(0), np.array([100])],
                T.F_LINEARITY: [np.array([np.nan]), np.zeros(0), np.array([1])],
                T.F_LIFETIME: [np.ones(1), np.zeros(0), np.array([2])],
                T.F_FINAL_AGE: [np.array([np.nan]), np.zeros(0), np.array([2])],
                T.F_NEW_OBJECT_COUNT: [1, 0, 0],
                T.F_LOST_OBJECT_COUNT: [0, 0, 0],
                T.F_MERGE_COUNT: [0, 0, 0],
                T.F_SPLIT_COUNT: [0, 0, 0]
            })
            self.check_relationships(workspace.measurements,
                                     [1], [1], [3], [1])

    def test_07_04_maintain_gap(self):
        '''Maintain object identity across a large gap'''
        with self.MonkeyPatchedDelete(self):
            workspace, module = self.make_lap2_workspace(
                np.array([[0, 1, 0, 0, 1, 2, 25],
                          [2, 2, 0, 0, 101, 102, 25]]), 3)
            self.assertTrue(isinstance(module, T.TrackObjects))
            #
            # The cost of creating the gap should be 140 and the cost of
            # bridging the gap should be 141.
            #
            module.gap_cost.value = 140
            module.max_gap_score.value = 142
            module.run_as_data_tool(workspace)
            self.check_measurements(workspace, {
                T.F_LABEL: [np.array([1]), np.zeros(0), np.array([2])],
                T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.zeros(0), np.array([0])],
                T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.zeros(0), np.array([0])],
                T.F_NEW_OBJECT_COUNT: [1, 0, 1],
                T.F_LOST_OBJECT_COUNT: [0, 1, 0],
                T.F_MERGE_COUNT: [0, 0, 0],
                T.F_SPLIT_COUNT: [0, 0, 0]
            })

    def test_07_05_filter_gap(self):
        '''Filter a gap due to an unreasonable score'''
        with self.MonkeyPatchedDelete(self):
            workspace, module = self.make_lap2_workspace(
                np.array([[0, 1, 0, 0, 1, 2, 25],
                          [2, 2, 0, 0, 101, 102, 25]]), 3)
            self.assertTrue(isinstance(module, T.TrackObjects))
            #
            # The cost of creating the gap should be 142 and the cost of
            # bridging the gap should be 141. However, the gap should be filtered
            # by the max score
            #
            module.gap_cost.value = 142
            module.max_gap_score.value = 140
            module.run_as_data_tool(workspace)
            self.check_measurements(workspace, {
                T.F_LABEL: [np.array([1]), np.zeros(0), np.array([2])],
                T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.zeros(0), np.array([0])],
                T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.zeros(0), np.array([0])]
            })

    def test_07_06_split(self):
        '''Track an object splitting'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 100, 100, 50],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 0, 0, 90, 90, 25],
                      [2, 1, 2, 1, 113, 114, 25],
                      [2, 2, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The split score should be 20*sqrt(2) more than the null so a split
        # alternative cost of 15 is too much and 14 too little. Values
        # doulbed to mat
        #
        module.split_cost.value = 30
        module.max_split_score.value = 30
        module.run_as_data_tool(workspace)
        d200 = np.sqrt(200)
        tot = np.sqrt(13 ** 2 + 14 ** 2)
        lin = tot / (d200 + 5)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, 1]), np.array([1, 1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 1]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 1]), np.array([1, 2])],
            T.F_DISTANCE_TRAVELED: [np.zeros(1), np.ones(2) * d200, np.array([5, 5])],
            T.F_DISPLACEMENT: [np.zeros(1), np.ones(2) * d200, np.array([tot, tot])],
            T.F_INTEGRATED_DISTANCE: [np.zeros(1), np.ones(2) * d200, np.ones(2) * d200 + 5],
            T.F_TRAJECTORY_X: [np.zeros(1), np.array([10, -10]), np.array([3, -4])],
            T.F_TRAJECTORY_Y: [np.zeros(1), np.array([10, -10]), np.array([4, -3])],
            T.F_LINEARITY: [np.array([np.nan]), np.array([1, 1]), np.array([lin, lin])],
            T.F_LIFETIME: [np.ones(1), np.array([2, 2]), np.array([3, 3])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.array([np.nan, np.nan]), np.array([3, 3])],
            T.F_NEW_OBJECT_COUNT: [1, 0, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 1, 0]
        })

    def test_07_07_dont_split(self):
        '''Track an object splitting'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 100, 100, 50],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 0, 0, 90, 90, 25],
                      [2, 1, 2, 1, 110, 110, 25],
                      [2, 2, 2, 2, 90, 90, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.split_cost.value = 28
        module.max_split_score.value = 30
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, 2]), np.array([1, 2])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 0]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 0]), np.array([1, 2])],
            T.F_LIFETIME: [np.ones(1), np.array([2, 1]), np.array([3, 2])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.array([np.nan, np.nan]), np.array([3, 2])],
            T.F_NEW_OBJECT_COUNT: [1, 1, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_08_split_filter(self):
        '''Prevent a split by setting the filter too low'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 100, 100, 50],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 0, 0, 90, 90, 25],
                      [2, 1, 2, 1, 110, 110, 25],
                      [2, 2, 2, 2, 90, 90, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.split_cost.value = 30
        module.max_split_score.value = 28
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, 2]), np.array([1, 2])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 0]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 0]), np.array([1, 2])],
            T.F_LIFETIME: [np.array([1]), np.array([2, 1]), np.array([3, 2])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.array([np.nan, np.nan]), np.array([3, 2])],
            T.F_NEW_OBJECT_COUNT: [1, 1, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_09_merge(self):
        '''Merge two objects into one'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 110, 110, 25],
                      [0, 2, 0, 0, 90, 90, 25],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 1, 2, 90, 90, 25],
                      [2, 1, 2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.merge_cost.value = 30
        module.max_merge_score.value = 30
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1, 1]), np.array([1, 1]), np.array([1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0, 0]), np.array([1, 1]), np.array([2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0, 0]), np.array([1, 2]), np.array([1])],
            T.F_LIFETIME: [np.array([1, 1]), np.array([2, 2]), np.array([3])],
            T.F_FINAL_AGE: [np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([3])],
            T.F_NEW_OBJECT_COUNT: [2, 0, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 1],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_10_dont_merge(self):
        '''Don't merge because of low alternative merge cost'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 110, 110, 25],
                      [0, 2, 0, 0, 90, 90, 25],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 1, 2, 90, 90, 25],
                      [2, 1, 2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
        #
        module.merge_cost.value = 28
        module.max_merge_score.value = 30
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 2)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(labels[0][1], 2)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)
        self.assertEqual(len(labels[2]), 1)
        self.assertEqual(labels[2][0], 1)

    def test_07_11_filter_merge(self):
        '''Don't merge because of low alternative merge cost'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 110, 110, 25],
                      [0, 2, 0, 0, 90, 90, 25],
                      [1, 1, 1, 1, 110, 110, 25],
                      [1, 2, 1, 2, 90, 90, 25],
                      [2, 1, 2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
        #
        module.merge_cost.value = 30
        module.max_merge_score.value = 28
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 2)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(labels[0][1], 2)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)
        self.assertEqual(len(labels[2]), 1)
        self.assertEqual(labels[2][0], 1)

    def test_07_12_img_1111(self):
        '''Regression test of img-1111'''
        data = np.array([[9, 1, 0, 0, 225, 20, 50],
                         [9, 2, 0, 0, 116, 223, 31],
                         [25, 3, 0, 0, 43, 291, 26],
                         [28, 4, 0, 0, 410, 436, 24],
                         [29, 5, 0, 0, 293, 166, 23],
                         [29, 4, 29, 1, 409, 436, 24],
                         [30, 5, 30, 1, 293, 167, 30],
                         [32, 6, 0, 0, 293, 164, 69],
                         [33, 6, 33, 1, 292, 166, 37],
                         [35, 7, 0, 0, 290, 165, 63],
                         [36, 7, 36, 1, 290, 166, 38],
                         [39, 8, 0, 0, 287, 163, 28],
                         [40, 8, 40, 1, 287, 163, 21],
                         [44, 9, 0, 0, 54, 288, 20],
                         [77, 10, 0, 0, 514, 211, 49],
                         [78, 10, 78, 1, 514, 210, 42],
                         [79, 10, 79, 1, 514, 209, 73],
                         [80, 10, 80, 1, 514, 208, 49],
                         [81, 10, 81, 1, 515, 209, 38],
                         [98, 11, 0, 0, 650, 54, 24],
                         [102, 12, 0, 0, 586, 213, 46],
                         [104, 13, 0, 0, 586, 213, 27],
                         [106, 14, 0, 0, 587, 212, 54],
                         [107, 14, 107, 1, 587, 212, 40],
                         [113, 15, 0, 0, 17, 145, 51],
                         [116, 16, 0, 0, 45, 153, 21],
                         [117, 17, 0, 0, 53, 148, 44],
                         [117, 18, 0, 0, 90, 278, 87],
                         [119, 19, 0, 0, 295, 184, 75],
                         [120, 19, 120, 1, 295, 184, 79],
                         [121, 19, 121, 1, 295, 182, 75],
                         [123, 20, 0, 0, 636, 7, 20],
                         [124, 20, 124, 1, 635, 7, 45],
                         [124, 21, 0, 0, 133, 171, 22],
                         [124, 22, 0, 0, 417, 365, 65],
                         [126, 23, 0, 0, 125, 182, 77],
                         [126, 24, 0, 0, 358, 306, 48],
                         [126, 25, 0, 0, 413, 366, 60],
                         [127, 26, 0, 0, 141, 173, 71],
                         [127, 25, 127, 3, 413, 366, 35],
                         [128, 27, 0, 0, 131, 192, 76],
                         [129, 28, 0, 0, 156, 182, 74],
                         [130, 29, 0, 0, 147, 194, 56],
                         [131, 30, 0, 0, 152, 185, 56],
                         [132, 30, 132, 1, 154, 188, 78],
                         [133, 31, 0, 0, 142, 186, 64],
                         [133, 32, 0, 0, 91, 283, 23],
                         [134, 33, 0, 0, 150, 195, 80]])
        data = data[:8, :]
        workspace, module = self.make_lap2_workspace(data, np.max(data[:, 0]) + 1)
        module.run_as_data_tool(workspace)

    def test_07_12_multi_group(self):
        '''Run several tests in different groups'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 1, 2, 25],
                      [2, 2, 0, 0, 101, 102, 25],
                      [3, 1, 0, 0, 100, 100, 50],
                      [4, 1, 4, 1, 110, 110, 25],
                      [4, 2, 0, 0, 90, 90, 25],
                      [5, 1, 5, 1, 113, 114, 25],
                      [5, 2, 5, 2, 86, 87, 25],
                      [6, 1, 0, 0, 110, 110, 25],
                      [6, 2, 0, 0, 90, 90, 25],
                      [7, 1, 7, 1, 110, 110, 25],
                      [7, 2, 7, 2, 90, 90, 25],
                      [8, 1, 8, 1, 104, 102, 50]
                      ]), 9,
            group_numbers=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            group_indexes=[1, 2, 3, 1, 2, 3, 1, 2, 3]
        )
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of bridging the gap should be 141. We set the alternative
        # score to 142 so that bridging wins.
        #
        module.gap_cost.value = 142
        module.max_gap_score.value = 142
        module.split_cost.value = 30
        module.max_split_score.value = 30
        module.merge_cost.value = 30
        module.max_merge_score.value = 30
        module.run_as_data_tool(workspace)
        distance = np.array([np.sqrt(2 * 100 * 100)])
        d200 = np.sqrt(200)
        tot = np.sqrt(13 ** 2 + 14 ** 2)
        lin = tot / (d200 + 5)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.zeros(0), np.array([1]),
                        np.array([1]), np.array([1, 1]), np.array([1, 1]),
                        np.array([1, 1]), np.array([1, 1]), np.array([1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.zeros(0), np.array([1]),
                                      np.array([0]), np.array([4, 4]), np.array([5, 5]),
                                      np.array([0, 0]), np.array([7, 7]), np.array([8])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.zeros(0), np.array([1]),
                                       np.array([0]), np.array([1, 1]), np.array([1, 2]),
                                       np.array([0, 0]), np.array([1, 2]), np.array([1])],
            T.F_DISPLACEMENT: [np.zeros(1), np.zeros(0), distance,
                               np.zeros(1), np.ones(2) * d200, np.array([tot, tot]),
                               np.zeros(2), np.zeros(2), np.array([10])],
            T.F_INTEGRATED_DISTANCE: [np.zeros(1), np.zeros(0), distance,
                                      np.zeros(1), np.ones(2) * d200, np.ones(2) * d200 + 5,
                                      np.zeros(2), np.zeros(2), np.array([10])],
            T.F_DISTANCE_TRAVELED: [np.zeros(1), np.zeros(0), distance,
                                    np.zeros(1), np.ones(2) * d200, np.array([5, 5]),
                                    np.zeros(2), np.zeros(2), np.array([10])],
            T.F_TRAJECTORY_X: [np.zeros(1), np.zeros(0), np.array([100]),
                               np.zeros(1), np.array([10, -10]), np.array([3, -4]),
                               np.zeros(2), np.zeros(2), np.array([-6])],
            T.F_TRAJECTORY_Y: [np.zeros(1), np.zeros(0), np.array([100]),
                               np.zeros(1), np.array([10, -10]), np.array([4, -3]),
                               np.zeros(2), np.zeros(2), np.array([-8])],
            T.F_LINEARITY: [np.array([np.nan]), np.zeros(0), np.array([1]),
                            np.array([np.nan]), np.array([1, 1]), np.array([lin, lin]),
                            np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.ones(1)],
            T.F_LIFETIME: [np.ones(1), np.zeros(0), np.array([2]),
                           np.ones(1), np.array([2, 2]), np.array([3, 3]),
                           np.ones(2), np.array([2, 2]), np.array([3])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.zeros(0), np.array([2]),
                            np.array([np.nan]), np.array([np.nan, np.nan]), np.array([3, 3]),
                            np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([3])],
            T.F_NEW_OBJECT_COUNT: [1, 0, 0,
                                   1, 0, 0,
                                   2, 0, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 0,
                              0, 0, 0,
                              0, 0, 1],
            T.F_SPLIT_COUNT: [0, 0, 0,
                              0, 1, 0,
                              0, 0, 0]
        })

    def test_07_13_filter_by_final_age(self):
        '''Filter an object by the final age'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 100, 100, 50],
                      [1, 1, 1, 1, 110, 110, 50],
                      [1, 2, 0, 0, 90, 90, 25],
                      [2, 1, 2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The split score should be between 14 and 15.  Set the split
        # alternative cost to 28 so that the split is inhibited.
        #
        module.split_cost.value = 28
        module.max_split_score.value = 30
        #
        # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
        #
        module.merge_cost.value = 28
        module.max_merge_score.value = 30
        module.wants_lifetime_filtering.value = True
        module.wants_minimum_lifetime.value = True
        module.min_lifetime.value = 1
        module.wants_maximum_lifetime.value = False
        module.max_lifetime.value = 100
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, np.NaN]), np.array([1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 0]), np.array([2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 0]), np.array([1])],
            T.F_LIFETIME: [np.array([1]), np.array([2, 1]), np.array([3])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.array([np.nan, 1]), np.array([3])],
            T.F_NEW_OBJECT_COUNT: [1, 1, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 1],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_14_mitosis(self):
        '''Track a mitosis'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 103, 104, 50],
                      [1, 2, 0, 0, 110, 110, 25],
                      [1, 3, 0, 0, 90, 90, 25],
                      [2, 2, 2, 1, 113, 114, 25],
                      [2, 3, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
        # 4 loses and 6 wins
        #
        module.merge_cost.value = 1
        module.gap_cost.value = 1
        module.mitosis_cost.value = 6
        module.mitosis_max_distance.value = 20
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, 1]), np.array([1, 1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 1]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 1]), np.array([1, 2])],
            T.F_LIFETIME: [np.ones(1), np.array([2, 2]), np.array([3, 3])],
            T.F_FINAL_AGE: [np.array([np.nan]), np.array([np.nan, np.nan]), np.array([3, 3])],
            T.F_LINK_TYPE: [np.array([T.LT_NONE]),
                            np.array([T.LT_MITOSIS, T.LT_MITOSIS]),
                            np.array([T.LT_NONE, T.LT_NONE])],
            T.F_MITOSIS_SCORE: [np.array([np.nan]),
                                np.array([5, 5]),
                                np.array([np.nan, np.nan])],
            T.F_NEW_OBJECT_COUNT: [1, 0, 0],
            T.F_LOST_OBJECT_COUNT: [0, 0, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 1, 0],
        })

    def test_07_14_no_mitosis(self):
        '''Don't track a mitosis'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 103, 104, 50],
                      [1, 2, 0, 0, 110, 110, 25],
                      [1, 3, 0, 0, 90, 90, 25],
                      [2, 2, 2, 1, 113, 114, 25],
                      [2, 3, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
        # 4 loses and 6 wins
        #
        module.merge_cost.value = 1
        module.mitosis_cost.value = 4
        module.mitosis_max_distance.value = 20
        module.gap_cost.value = 1
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([2, 3]), np.array([2, 3])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([0, 0]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([0, 0]), np.array([1, 2])],
            T.F_LIFETIME: [np.ones(1), np.array([1, 1]), np.array([2, 2])],
            T.F_FINAL_AGE: [np.array([1]), np.array([np.nan, np.nan]), np.array([2, 2])],
            T.F_NEW_OBJECT_COUNT: [1, 2, 0],
            T.F_LOST_OBJECT_COUNT: [0, 1, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_15_mitosis_distance_filter(self):
        '''Don't track a mitosis'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 103, 104, 50],
                      [1, 2, 0, 0, 110, 110, 25],
                      [1, 3, 0, 0, 90, 90, 25],
                      [2, 2, 2, 1, 113, 114, 25],
                      [2, 3, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
        # 4 loses and 6 wins
        #
        module.merge_cost.value = 1
        module.mitosis_cost.value = 6
        module.mitosis_max_distance.value = 15
        module.gap_cost.value = 1
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([2, 3]), np.array([2, 3])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([0, 0]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([0, 0]), np.array([1, 2])],
            T.F_LIFETIME: [np.ones(1), np.array([1, 1]), np.array([2, 2])],
            T.F_FINAL_AGE: [np.array([1]), np.array([np.nan, np.nan]), np.array([2, 2])],
            T.F_NEW_OBJECT_COUNT: [1, 2, 0],
            T.F_LOST_OBJECT_COUNT: [0, 1, 0],
            T.F_MERGE_COUNT: [0, 0, 0],
            T.F_SPLIT_COUNT: [0, 0, 0]
        })

    def test_07_16_alternate_child_mitoses(self):
        # Test that LAP can pick the best of two possible child alternates
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 103, 104, 50],
                      [1, 2, 0, 0, 110, 110, 25],
                      [1, 3, 0, 0, 91, 91, 25],
                      [1, 4, 0, 0, 90, 90, 25],
                      [2, 2, 2, 1, 113, 114, 25],
                      [2, 3, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.merge_cost.value = 1
        module.gap_cost.value = 1
        module.mitosis_cost.value = 6
        module.mitosis_max_distance.value = 20
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1]), np.array([1, 1, 2]), np.array([1, 1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0]), np.array([1, 1, 0]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0]), np.array([1, 1, 0]), np.array([1, 2])]
        })

    def test_07_17_alternate_parent_mitoses(self):
        # Test that LAP can pick the best of two possible parent alternates
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 100, 100, 50],
                      [0, 2, 0, 0, 103, 104, 50],
                      [1, 3, 0, 0, 110, 110, 25],
                      [1, 4, 0, 0, 90, 90, 25],
                      [2, 3, 2, 1, 113, 114, 25],
                      [2, 4, 2, 2, 86, 87, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.merge_cost.value = 1
        module.gap_cost.value = 1
        module.mitosis_cost.value = 6
        module.mitosis_max_distance.value = 20
        module.run_as_data_tool(workspace)
        self.check_measurements(workspace, {
            T.F_LABEL: [np.array([1, 2]), np.array([1, 1]), np.array([1, 1])],
            T.F_PARENT_IMAGE_NUMBER: [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])],
            T.F_PARENT_OBJECT_NUMBER: [np.array([0, 0]), np.array([1, 1]), np.array([1, 2])]
        })

    class MonkeyPatchedDelete(object):
        '''Monkey patch np.delete inside of a scope

        For regression test of issue #1571 - negative
        indices in calls to numpy.delete

        Usage:
            with MonkeyPatchedDelete(self):
                ... do test ...
        '''

        def __init__(self, test):
            self.__test = test

        def __enter__(self):
            self.old_delete = np.delete
            np.delete = self.monkey_patched_delete

        def __exit__(self, type, value, traceback):
            np.delete = self.old_delete

        def monkey_patched_delete(self, array, indices, axis):
            self.__test.assertTrue(np.all(indices >= 0))
            return self.old_delete(array, indices, axis)

    def test_08_01_save_image(self):
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.pixel_radius.value = 50
        module.wants_image.value = True
        module.image_name.value = "outimage"
        measurements = cpmeas.Measurements()
        measurements.add_image_measurement(cpp.GROUP_NUMBER, 1)
        measurements.add_image_measurement(cpp.GROUP_INDEX, 1)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()

        module.prepare_run(cpw.Workspace(
            pipeline, module, None, None, measurements, image_set_list))

        first = True
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = np.zeros((640, 480), int)
        object_set.add_objects(objects, OBJECT_NAME)
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements, image_set_list)
        module.run(workspace)
        image = workspace.image_set.get_image(module.image_name.value)
        shape = image.pixel_data.shape
        self.assertEqual(shape[0], 640)
        self.assertEqual(shape[1], 480)

    def test_09_00_get_no_gap_pair_scores(self):
        for F, L, max_gap in (
                (np.zeros((0, 3)), np.zeros((0, 3)), 1),
                (np.ones((1, 3)), np.ones((1, 3)), 1),
                (np.ones((2, 3)), np.ones((2, 3)), 1)):
            t = T.TrackObjects()
            a, d = t.get_gap_pair_scores(F, L, max_gap)
            self.assertEqual(tuple(a.shape), (0, 2))
            self.assertEqual(len(d), 0)

    def test_09_01_get_gap_pair_scores(self):
        L = np.array([[0.0, 0.0, 1, 0, 0, 0, 1],
                      [1.0, 1.0, 5, 0, 0, 0, 1],
                      [3.0, 3.0, 8, 0, 0, 0, 1],
                      [2.0, 2.0, 9, 0, 0, 0, 1],
                      [0.0, 0.0, 9, 0, 0, 0, 1],
                      [0.0, 0.0, 9, 0, 0, 0, 1]])
        F = np.array([[0.0, 0.0, 0, 0, 0, 0, 1],
                      [1.0, 0.0, 4, 0, 0, 0, 1],
                      [3.0, 0.0, 6, 0, 0, 0, 1],
                      [4.0, 0.0, 7, 0, 0, 0, 1],
                      [1.0, 0.0, 2, 0, 0, 0, 2],
                      [1.0, 0.0, 2, 0, 0, 0, .5]])
        expected = np.array([[0, 1],
                             [0, 4],
                             [0, 5],
                             [1, 2],
                             [1, 3]])
        expected_d = np.sqrt(
            np.sum((L[expected[:, 0], :2] - F[expected[:, 1], :2]) ** 2, 1))
        expected_rho = np.array([1, 2, 2, 1, 1])
        t = T.TrackObjects()
        a, d = t.get_gap_pair_scores(F, L, 4)
        order = np.lexsort((a[:, 1], a[:, 0]))
        a, d = a[order], d[order]
        np.testing.assert_array_equal(a, expected)

        np.testing.assert_array_almost_equal(d, expected_d * expected_rho)

    def test_10_01_neighbour_track_nothing(self):
        '''Run TrackObjects on an empty labels matrix'''
        columns = []

        def fn(module, workspace, index, columns=columns):
            if workspace is not None and index == 0:
                columns += module.get_measurement_columns(workspace.pipeline)
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((np.zeros((10, 10), int),
                                             np.zeros((10, 10), int)), fn)

        features = [feature
                    for feature in measurements.get_feature_names(OBJECT_NAME)
                    if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == OBJECT_NAME]))
        for feature in T.F_ALL:
            name = "_".join((T.F_PREFIX, feature, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(value), 0)

        features = [feature for feature in measurements.get_feature_names(cpmeas.IMAGE)
                    if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == cpmeas.IMAGE]))
        for feature in T.F_IMAGE_ALL:
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_image_measurement(name)
            self.assertEqual(value, 0)

    def test_10_01_00_neighbour_track_one_then_nothing(self):
        '''Run track objects on an object that disappears

        Regression test of IMG-1090
        '''
        labels = np.zeros((10, 10), int)
        labels[3:6, 2:7] = 1

        def fn(module, workspace, index):
            if workspace is not None and index == 0:
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((labels,
                                             np.zeros((10, 10), int)), fn)
        feature = "_".join((T.F_PREFIX, T.F_LOST_OBJECT_COUNT,
                            OBJECT_NAME, "50"))
        value = measurements.get_current_image_measurement(feature)
        self.assertEqual(value, 1)

    def test_10_02_neighbour_track_one_by_distance(self):
        '''Track an object that doesn't move.'''
        labels = np.zeros((10, 10), int)
        labels[3:6, 2:7] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((labels, labels), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertAlmostEqual(m(T.F_TRAJECTORY_X), 0)
        self.assertAlmostEqual(m(T.F_TRAJECTORY_Y), 0)
        self.assertAlmostEqual(m(T.F_DISTANCE_TRAVELED), 0)
        self.assertAlmostEqual(m(T.F_INTEGRATED_DISTANCE), 0)
        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 1)
        self.assertEqual(m(T.F_PARENT_IMAGE_NUMBER), 1)
        self.assertEqual(m(T.F_LIFETIME), 2)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
        self.check_relationships(measurements, [1], [1], [2], [1])

    def test_10_03_neighbour_track_one_moving(self):
        '''Track an object that moves'''

        labels_list = []
        distance = 0
        last_i, last_j = (0, 0)
        for i_off, j_off in ((0, 0), (2, 0), (2, 1), (0, 1)):
            distance = i_off - last_i + j_off - last_j
            last_i, last_j = (i_off, j_off)
            labels = np.zeros((10, 10), int)
            labels[4 + i_off:7 + i_off, 4 + j_off:7 + j_off] = 1
            labels_list.append(labels)

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 3
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects(labels_list, fn)

        def m(feature, expected):
            name = "_".join((T.F_PREFIX, feature, "3"))
            value_set = measurements.get_all_measurements(OBJECT_NAME, name)
            self.assertEqual(len(expected), len(value_set))
            for values, x in zip(value_set, expected):
                self.assertEqual(len(values), 1)
                self.assertAlmostEqual(values[0], x)

        m(T.F_TRAJECTORY_X, [0, 0, 1, 0])
        m(T.F_TRAJECTORY_Y, [0, 2, 0, -2])
        m(T.F_DISTANCE_TRAVELED, [0, 2, 1, 2])
        m(T.F_INTEGRATED_DISTANCE, [0, 2, 3, 5])
        m(T.F_LABEL, [1, 1, 1, 1])
        m(T.F_LIFETIME, [1, 2, 3, 4])
        m(T.F_LINEARITY, [1, 1, np.sqrt(5) / 3, 1.0 / 5.0])

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "3"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
        image_numbers = np.arange(1, len(labels_list) + 1)
        object_numbers = np.ones(len(image_numbers))
        self.check_relationships(measurements,
                                 image_numbers[:-1], object_numbers[:-1],
                                 image_numbers[1:], object_numbers[1:])

    def test_10_04_neighbour_track_negative(self):
        '''Track unrelated objects'''
        labels1 = np.zeros((10, 10), int)
        labels1[1:5, 1:5] = 1
        labels2 = np.zeros((10, 10), int)
        labels2[6:9, 6:9] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 0)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)

        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)

    def test_10_05_neighbour_track_ambiguous(self):
        '''Track disambiguation from among two possible parents'''
        labels1 = np.zeros((20, 20), int)
        labels1[1:4, 1:4] = 1
        labels1[16:19, 16:19] = 2
        labels2 = np.zeros((20, 20), int)
        labels2[10:15, 10:15] = 1

        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 20
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "20"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT_OBJECT_NUMBER), 2)

    def test_10_06_neighbour_track_group_with_drop(self):
        '''Track groups with one lost'''
        labels1 = np.zeros((20, 20), int)
        labels1[2, 2] = 1
        labels1[4, 2] = 2
        labels1[2, 4] = 3
        labels1[4, 4] = 4

        labels2 = np.zeros((20, 20), int)
        labels2[16, 16] = 1
        labels2[18, 16] = 2
        # labels2[16,18] = 3 is no longer present
        labels2[18, 18] = 4

        def fn(module, workspace, idx):
            if idx == 0:
                module.drop_cost.value = 100  # make it always try to match
                module.pixel_radius.value = 200
                module.average_cell_diameter.value = 5
                module.tracking_method.value = "Follow Neighbors"

        measurements = self.runTrackObjects((labels1, labels2), fn)

        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "20"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]

        self.check_relationships(measurements, [1, 1, 1], [1, 2, 4], [2, 2, 2], [1, 2, 4])
