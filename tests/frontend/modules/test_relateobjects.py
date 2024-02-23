import numpy
import unittest

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
from cellprofiler_core.constants.measurement import (
    C_PARENT,
    GROUP_NUMBER,
    GROUP_INDEX,
    COLTYPE_FLOAT,
    COLTYPE_INTEGER,
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    FF_PARENT,
)

import cellprofiler.modules.relateobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

PARENT_OBJECTS = "parentobjects"
CHILD_OBJECTS = "childobjects"
MEASUREMENT = "Measurement"
IGNORED_MEASUREMENT = "%s_Foo" % C_PARENT


class TestRelateObjects(unittest.TestCase):
    def make_workspace(self, parents, children, fake_measurement=False):
        """Make a workspace for testing Relate"""
        pipeline = cellprofiler_core.pipeline.Pipeline()
        if fake_measurement:

            class FakeModule(cellprofiler_core.module.Module):
                def get_measurement_columns(self, pipeline):
                    return [
                        (CHILD_OBJECTS, MEASUREMENT, COLTYPE_FLOAT,),
                        (CHILD_OBJECTS, IGNORED_MEASUREMENT, COLTYPE_INTEGER,),
                    ]

            module = FakeModule()
            module.set_module_num(1)
            pipeline.add_module(module)
        module = cellprofiler.modules.relateobjects.Relate()
        module.x_name.value = PARENT_OBJECTS
        module.y_name.value = CHILD_OBJECTS
        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_NONE
        )
        module.wants_child_objects_saved.value = False
        new_module_num = 2 if fake_measurement else 1
        module.set_module_num(new_module_num)
        pipeline.add_module(module)
        object_set = cellprofiler_core.object.ObjectSet()
        image_set_list = cellprofiler_core.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        m = cellprofiler_core.measurement.Measurements()
        m.add_image_measurement(GROUP_NUMBER, 1)
        m.add_image_measurement(GROUP_INDEX, 1)
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, module, image_set, object_set, m, image_set_list
        )
        o = cellprofiler_core.object.Objects()
        if parents.shape[1] == 3:
            # IJV format
            o.ijv = parents
        else:
            o.segmented = parents
        object_set.add_objects(o, PARENT_OBJECTS)
        o = cellprofiler_core.object.Objects()
        if children.shape[1] == 3:
            o.ijv = children
        else:
            o.segmented = children
        object_set.add_objects(o, CHILD_OBJECTS)
        return workspace, module

    def features_and_columns_match(self, workspace):
        module = workspace.module
        pipeline = workspace.pipeline
        measurements = workspace.measurements
        object_names = [x for x in measurements.get_object_names() if x != "Image"]
        features = [
            [
                feature
                for feature in measurements.get_feature_names(object_name)
                if feature not in (MEASUREMENT, IGNORED_MEASUREMENT)
            ]
            for object_name in object_names
        ]
        columns = [
            x for x in module.get_measurement_columns(pipeline) if x[0] != "Image"
        ]
        self.assertEqual(sum([len(f) for f in features]), len(columns))
        for column in columns:
            index = object_names.index(column[0])
            self.assertTrue(column[1] in features[index])

    def test_02_01_relate_zeros(self):
        """Relate a field of empty parents to empty children"""
        labels = numpy.zeros((10, 10), int)
        workspace, module = self.make_workspace(labels, labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(
            CHILD_OBJECTS, "Parent_%s" % PARENT_OBJECTS
        )
        self.assertEqual(numpy.product(parents_of.shape), 0)
        child_count = m.get_current_measurement(
            PARENT_OBJECTS, "Children_%s_Count" % CHILD_OBJECTS
        )
        self.assertEqual(numpy.product(child_count.shape), 0)
        self.features_and_columns_match(workspace)

    def test_02_01_relate_one(self):
        """Relate one parent to one child"""
        parent_labels = numpy.ones((10, 10), int)
        child_labels = numpy.zeros((10, 10), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(
            CHILD_OBJECTS, "Parent_%s" % PARENT_OBJECTS
        )
        self.assertEqual(numpy.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(
            PARENT_OBJECTS, "Children_%s_Count" % CHILD_OBJECTS
        )
        self.assertEqual(numpy.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_02_relate_wrong_size(self):
        """Regression test of IMG-961

        Perhaps someone is trying to relate cells to wells and the grid
        doesn't completely cover the labels matrix.
        """
        parent_labels = numpy.ones((20, 10), int)
        parent_labels[10:, :] = 0
        child_labels = numpy.zeros((10, 20), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(
            CHILD_OBJECTS, "Parent_%s" % PARENT_OBJECTS
        )
        self.assertEqual(numpy.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(
            PARENT_OBJECTS, "Children_%s_Count" % CHILD_OBJECTS
        )
        self.assertEqual(numpy.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_03_relate_ijv(self):
        """Regression test of IMG-1317: relating objects in ijv form"""

        child_ijv = numpy.array([[5, 5, 1], [5, 6, 2], [20, 15, 3]])
        parent_ijv = numpy.array([[5, 5, 1], [5, 6, 1], [20, 15, 2]])
        workspace, module = self.make_workspace(parent_ijv, child_ijv)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(
            CHILD_OBJECTS, "Parent_%s" % PARENT_OBJECTS
        )
        self.assertEqual(numpy.product(parents_of.shape), 3)
        self.assertTrue(parents_of[0], 1)
        self.assertEqual(parents_of[1], 1)
        self.assertEqual(parents_of[2], 2)
        child_count = m.get_current_measurement(
            PARENT_OBJECTS, "Children_%s_Count" % CHILD_OBJECTS
        )
        self.assertEqual(numpy.product(child_count.shape), 2)
        self.assertEqual(child_count[0], 2)
        self.assertEqual(child_count[1], 1)

    def test_03_01_mean(self):
        """Compute the mean for two parents and four children"""
        i, j = numpy.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = (i / 10).astype(int) + (j / 10).astype(int) * 2 + 1
        workspace, module = self.make_workspace(
            parent_labels, child_labels, fake_measurement=True
        )
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m,cellprofiler_core.measurement.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT, numpy.array([1.0, 2.0, 3.0, 4.0]))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT, numpy.array([1, 2, 3, 4]))
        expected = numpy.array([2.0, 3.0])
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(numpy.all(data == expected))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_03_02_empty_mean(self):
        # Regression test - if there are no children, the per-parent means
        #                   should still be populated
        i, j = numpy.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = numpy.zeros(parent_labels.shape, int)
        workspace, module = self.make_workspace(
            parent_labels, child_labels, fake_measurement=True
        )
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m,cellprofiler_core.measurement.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT, numpy.zeros(0))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT, numpy.zeros(0, int))
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(numpy.all(numpy.isnan(data)))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_04_00_distance_empty(self):
        """Make sure we can handle labels matrices that are all zero"""
        empty_labels = numpy.zeros((10, 20), int)
        some_labels = numpy.zeros((10, 20), int)
        some_labels[2:7, 3:8] = 1
        some_labels[3:8, 12:17] = 2
        for parent_labels, child_labels, n in (
            (empty_labels, empty_labels, 0),
            (some_labels, empty_labels, 0),
            (empty_labels, some_labels, 2),
        ):
            workspace, module = self.make_workspace(parent_labels, child_labels)
            self.assertTrue(
                isinstance(module, cellprofiler.modules.relateobjects.Relate)
            )
            module.find_parent_child_distances.value = (
                cellprofiler.modules.relateobjects.D_BOTH
            )
            module.run(workspace)
            self.features_and_columns_match(workspace)
            meas = workspace.measurements
            for feature in (
                cellprofiler.modules.relateobjects.FF_CENTROID,
                cellprofiler.modules.relateobjects.FF_MINIMUM,
            ):
                m = feature % PARENT_OBJECTS
                v = meas.get_current_measurement(CHILD_OBJECTS, m)
                self.assertEqual(len(v), n)
                if n > 0:
                    self.assertTrue(numpy.all(numpy.isnan(v)))

    def test_04_01_distance_centroids(self):
        """Check centroid-centroid distance calculation"""
        i, j = numpy.mgrid[0:14, 0:30]
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        # Centers should be at i=3 and j=7
        parent_centers = numpy.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        child_labels = numpy.zeros(i.shape)
        numpy.random.seed(0)
        # Take 12 random points and label them
        child_centers = numpy.random.permutation(numpy.prod(i.shape))[:12]
        child_centers = numpy.vstack(
            (i.flatten()[child_centers], j.flatten()[child_centers])
        )
        child_labels[child_centers[0], child_centers[1]] = numpy.arange(1, 13)
        parent_indexes = parent_labels[child_centers[0], child_centers[1]] - 1
        expected = numpy.sqrt(
            numpy.sum(
                (parent_centers[parent_indexes, :] - child_centers.transpose()) ** 2, 1
            )
        )

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.relateobjects.Relate))
        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_BOTH
        )
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(
            CHILD_OBJECTS,
            cellprofiler.modules.relateobjects.FF_CENTROID % PARENT_OBJECTS,
        )
        assert v.shape[0] == 12
        assert numpy.all(numpy.abs(v - expected) < 0.0001)

    def test_distance_minima(self):
        parents = numpy.zeros((11, 11), dtype=numpy.uint8)

        children = numpy.zeros_like(parents)

        parents[1:10, 1:10] = 1

        children[2:3, 2:3] = 1

        children[3:8, 3:8] = 2

        workspace, module = self.make_workspace(parents, children)

        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_MINIMUM
        )

        module.run(workspace)

        expected = [1, 4]

        actual = workspace.measurements.get_current_measurement(
            CHILD_OBJECTS,
            cellprofiler.modules.relateobjects.FF_MINIMUM % PARENT_OBJECTS,
        )

        numpy.testing.assert_array_equal(actual, expected)

    def test_means_of_distances(self):
        #
        # Regression test of issue #1409
        #
        # Make sure means of minimum and mean distances of children
        # are recorded properly
        #
        i, j = numpy.mgrid[0:14, 0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        child_labels = numpy.zeros(i.shape)
        numpy.random.seed(0)
        # Take 12 random points and label them
        child_centers = numpy.random.permutation(numpy.prod(i.shape))[:12]
        child_centers = numpy.vstack(
            (i.flatten()[child_centers], j.flatten()[child_centers])
        )
        child_labels[child_centers[0], child_centers[1]] = numpy.arange(1, 13)
        parent_centers = numpy.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        parent_indexes = parent_labels[child_centers[0], child_centers[1]] - 1
        expected = numpy.sqrt(
            numpy.sum(
                (parent_centers[parent_indexes, :] - child_centers.transpose()) ** 2, 1
            )
        )

        workspace, module = self.make_workspace(parent_labels, child_labels)
        assert isinstance(module, cellprofiler.modules.relateobjects.Relate)
        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_CENTROID
        )
        module.wants_per_parent_means.value = True
        mnames = module.get_measurements(
            workspace.pipeline,
            PARENT_OBJECTS,
            "_".join((cellprofiler.modules.relateobjects.C_MEAN, CHILD_OBJECTS)),
        )
        assert cellprofiler.modules.relateobjects.FF_CENTROID % PARENT_OBJECTS in mnames
        feat_mean = cellprofiler.modules.relateobjects.FF_MEAN % (
            CHILD_OBJECTS,
            cellprofiler.modules.relateobjects.FF_CENTROID % PARENT_OBJECTS,
        )
        mcolumns = module.get_measurement_columns(workspace.pipeline)
        assert any([c[0] == PARENT_OBJECTS and c[1] == feat_mean for c in mcolumns])
        m = workspace.measurements
        m[CHILD_OBJECTS, M_LOCATION_CENTER_X, 1] = child_centers[1]
        m[CHILD_OBJECTS, M_LOCATION_CENTER_Y, 1] = child_centers[0]
        module.run(workspace)

        v = m[PARENT_OBJECTS, feat_mean, 1]

        plabel = m[
            CHILD_OBJECTS, "_".join((C_PARENT, PARENT_OBJECTS)), 1,
        ]

        assert len(v) == 4
        for idx in range(4):
            if numpy.any(plabel == idx + 1):
                assert (
                    round(abs(v[idx] - numpy.mean(expected[plabel == idx + 1])), 4) == 0
                )

    def test_calculate_centroid_distances_volume(self):
        parents = numpy.zeros((9, 11, 11), dtype=numpy.uint8)

        children = numpy.zeros_like(parents)

        k, i, j = numpy.mgrid[0:9, 0:11, 0:11]

        parents[(k - 4) ** 2 + (i - 5) ** 2 + (j - 5) ** 2 <= 16] = 1

        children[(k - 3) ** 2 + (i - 3) ** 2 + (j - 3) ** 2 <= 4] = 1

        children[(k - 4) ** 2 + (i - 7) ** 2 + (j - 7) ** 2 <= 4] = 2

        workspace, module = self.make_workspace(parents, children)

        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_CENTROID
        )

        module.run(workspace)

        expected = [3, numpy.sqrt(8)]

        actual = workspace.measurements.get_current_measurement(
            CHILD_OBJECTS,
            cellprofiler.modules.relateobjects.FF_CENTROID % PARENT_OBJECTS,
        )

        numpy.testing.assert_array_equal(actual, expected)

    def test_calculate_minimum_distances_volume(self):
        parents = numpy.zeros((9, 11, 11), dtype=numpy.uint8)

        children = numpy.zeros_like(parents)

        parents[1:8, 1:10, 1:10] = 1

        children[3:6, 2:3, 2:3] = 1

        children[4:7, 3:8, 3:8] = 2

        workspace, module = self.make_workspace(parents, children)

        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_MINIMUM
        )

        module.run(workspace)

        expected = [1, 2]

        actual = workspace.measurements.get_current_measurement(
            CHILD_OBJECTS,
            cellprofiler.modules.relateobjects.FF_MINIMUM % PARENT_OBJECTS,
        )

        numpy.testing.assert_array_equal(actual, expected)

    def test_relate_zeros_with_step_parent(self):
        # https://github.com/CellProfiler/CellProfiler/issues/2441
        parents = numpy.zeros((10, 10), dtype=numpy.uint8)

        children = numpy.zeros_like(parents)

        step_parents = numpy.zeros_like(parents)

        step_parents_object = cellprofiler_core.object.Objects()

        step_parents_object.segmented = step_parents

        workspace, module = self.make_workspace(parents, children)

        workspace.measurements.add_measurement("Step", FF_PARENT % PARENT_OBJECTS, [])

        module.step_parent_names[0].step_parent_name.value = "Step"

        workspace.object_set.add_objects(step_parents_object, "Step")

        module.wants_step_parent_distances.value = True

        module.find_parent_child_distances.value = (
            cellprofiler.modules.relateobjects.D_MINIMUM
        )

        module.run(workspace)

        expected = []

        actual = workspace.measurements.get_current_measurement(
            CHILD_OBJECTS, cellprofiler.modules.relateobjects.FF_MINIMUM % "Step"
        )
        numpy.testing.assert_array_equal(actual, expected)

    def test_relate_and_make_new_objects(self):
        """Relate one parent to one child, but save children as a new set"""
        parent_labels = numpy.ones((10, 10), int)
        child_labels = numpy.zeros((10, 10), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_child_objects_saved.value = True
        module.output_child_objects_name.value = "outputobjects"
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(
            CHILD_OBJECTS, "Parent_%s" % PARENT_OBJECTS
        )
        self.assertEqual(numpy.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(
            PARENT_OBJECTS, "Children_%s_Count" % CHILD_OBJECTS
        )
        self.assertEqual(numpy.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)
