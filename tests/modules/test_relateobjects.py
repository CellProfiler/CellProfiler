'''test_relateobjects.py - test the RelateObjects module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np
from scipy.ndimage import distance_transform_edt

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.relateobjects as R

PARENT_OBJECTS = 'parentobjects'
CHILD_OBJECTS = 'childobjects'
MEASUREMENT = 'Measurement'
IGNORED_MEASUREMENT = '%s_Foo' % R.C_PARENT


class TestRelateObjects(unittest.TestCase):
    def make_workspace(self, parents, children, fake_measurement=False):
        '''Make a workspace for testing Relate'''
        pipeline = cpp.Pipeline()
        if fake_measurement:
            class FakeModule(cpm.Module):
                def get_measurement_columns(self, pipeline):
                    return [(CHILD_OBJECTS, MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                            (CHILD_OBJECTS, IGNORED_MEASUREMENT, cpmeas.COLTYPE_INTEGER)]

            module = FakeModule()
            module.module_num = 1
            pipeline.add_module(module)
        module = R.Relate()
        module.parent_name.value = PARENT_OBJECTS
        module.sub_object_name.value = CHILD_OBJECTS
        module.find_parent_child_distances.value = R.D_NONE
        module.module_num = 2 if fake_measurement else 1
        pipeline.add_module(module)
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        m = cpmeas.Measurements()
        m.add_image_measurement(cpmeas.GROUP_NUMBER, 1)
        m.add_image_measurement(cpmeas.GROUP_INDEX, 1)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        o = cpo.Objects()
        if parents.shape[1] == 3:
            # IJV format
            o.ijv = parents
        else:
            o.segmented = parents
        object_set.add_objects(o, PARENT_OBJECTS)
        o = cpo.Objects()
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
        object_names = [x for x in measurements.get_object_names()
                        if x != cpmeas.IMAGE]
        features = [[feature
                     for feature in measurements.get_feature_names(object_name)
                     if feature not in (MEASUREMENT, IGNORED_MEASUREMENT)]
                    for object_name in object_names]
        columns = module.get_measurement_columns(pipeline)
        self.assertEqual(sum([len(f) for f in features]), len(columns))
        for column in columns:
            index = object_names.index(column[0])
            self.assertTrue(column[1] in features[index])

    def test_02_01_relate_zeros(self):
        '''Relate a field of empty parents to empty children'''
        labels = np.zeros((10, 10), int)
        workspace, module = self.make_workspace(labels, labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 0)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 0)
        self.features_and_columns_match(workspace)

    def test_02_01_relate_one(self):
        '''Relate one parent to one child'''
        parent_labels = np.ones((10, 10), int)
        child_labels = np.zeros((10, 10), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_02_relate_wrong_size(self):
        '''Regression test of IMG-961

        Perhaps someone is trying to relate cells to wells and the grid
        doesn't completely cover the labels matrix.
        '''
        parent_labels = np.ones((20, 10), int)
        parent_labels[10:, :] = 0
        child_labels = np.zeros((10, 20), int)
        child_labels[3:5, 4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0], 1)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 1)
        self.assertEqual(child_count[0], 1)
        self.features_and_columns_match(workspace)

    def test_02_03_relate_ijv(self):
        '''Regression test of IMG-1317: relating objects in ijv form'''

        child_ijv = np.array([[5, 5, 1], [5, 5, 2], [20, 15, 3]])
        parent_ijv = np.array([[5, 5, 1], [20, 15, 2]])
        workspace, module = self.make_workspace(parent_ijv, child_ijv)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS,
                                               "Parent_%s" % PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 3)
        self.assertTrue(parents_of[0], 1)
        self.assertEqual(parents_of[1], 1)
        self.assertEqual(parents_of[2], 2)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count" %
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 2)
        self.assertEqual(child_count[0], 2)
        self.assertEqual(child_count[1], 1)

    def test_03_01_mean(self):
        '''Compute the mean for two parents and four children'''
        i, j = np.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = (i / 10).astype(int) + (j / 10).astype(int) * 2 + 1
        workspace, module = self.make_workspace(parent_labels, child_labels,
                                                fake_measurement=True)
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT,
                          np.array([1.0, 2.0, 3.0, 4.0]))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT,
                          np.array([1, 2, 3, 4]))
        expected = np.array([2.0, 3.0])
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(np.all(data == expected))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_03_02_empty_mean(self):
        # Regression test - if there are no children, the per-parent means
        #                   should still be populated
        i, j = np.mgrid[0:20, 0:20]
        parent_labels = (i / 10 + 1).astype(int)
        child_labels = np.zeros(parent_labels.shape, int)
        workspace, module = self.make_workspace(parent_labels, child_labels,
                                                fake_measurement=True)
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        m.add_measurement(CHILD_OBJECTS, MEASUREMENT, np.zeros(0))
        m.add_measurement(CHILD_OBJECTS, IGNORED_MEASUREMENT, np.zeros(0, int))
        module.run(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, MEASUREMENT)
        self.assertTrue(name in m.get_feature_names(PARENT_OBJECTS))
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(np.all(np.isnan(data)))
        self.features_and_columns_match(workspace)
        name = "Mean_%s_%s" % (CHILD_OBJECTS, IGNORED_MEASUREMENT)
        self.assertFalse(name in m.get_feature_names(PARENT_OBJECTS))

    def test_04_00_distance_empty(self):
        '''Make sure we can handle labels matrices that are all zero'''
        empty_labels = np.zeros((10, 20), int)
        some_labels = np.zeros((10, 20), int)
        some_labels[2:7, 3:8] = 1
        some_labels[3:8, 12:17] = 2
        for parent_labels, child_labels, n in ((empty_labels, empty_labels, 0),
                                               (some_labels, empty_labels, 0),
                                               (empty_labels, some_labels, 2)):
            workspace, module = self.make_workspace(parent_labels, child_labels)
            self.assertTrue(isinstance(module, R.Relate))
            module.find_parent_child_distances.value = R.D_BOTH
            module.run(workspace)
            self.features_and_columns_match(workspace)
            meas = workspace.measurements
            for feature in (R.FF_CENTROID, R.FF_MINIMUM):
                m = feature % PARENT_OBJECTS
                v = meas.get_current_measurement(CHILD_OBJECTS, m)
                self.assertEqual(len(v), n)
                if n > 0:
                    self.assertTrue(np.all(np.isnan(v)))

    def test_04_01_distance_centroids(self):
        '''Check centroid-centroid distance calculation'''
        i, j = np.mgrid[0:14, 0:30]
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        # Centers should be at i=3 and j=7
        parent_centers = np.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        parent_indexes = parent_labels[child_centers[0],
                                       child_centers[1]] - 1
        expected = np.sqrt(np.sum((parent_centers[parent_indexes, :] -
                                   child_centers.transpose()) ** 2, 1))

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_CENTROID
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_CENTROID % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))

    def test_04_02_distance_minima(self):
        '''Check centroid-perimeter distance calculation'''
        i, j = np.mgrid[0:14, 0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i >= 6) * 1 + (j >= 14) * 2 + 1
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        #
        # Measure the distance from the child to the edge of its parent.
        # We do this using the distance transform with a background that's
        # the edges of the labels
        #
        background = ((i != 0) & (i != 5) & (i != 6) & (i != 13) &
                      (j != 0) & (j != 13) & (j != 14) & (j != 29))
        d = distance_transform_edt(background)
        expected = d[child_centers[0], child_centers[1]]

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_MINIMUM
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_MINIMUM % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))

    def test_04_03_means_of_distances(self):
        #
        # Regression test of issue #1409
        #
        # Make sure means of minimum and mean distances of children
        # are recorded properly
        #
        i, j = np.mgrid[0:14, 0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i >= 7) * 1 + (j >= 15) * 2 + 1
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers],
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0], child_centers[1]] = np.arange(1, 13)
        parent_centers = np.array([[3, 7], [10, 7], [3, 22], [10, 22]], float)
        parent_indexes = parent_labels[child_centers[0],
                                       child_centers[1]] - 1
        expected = np.sqrt(np.sum((parent_centers[parent_indexes, :] -
                                   child_centers.transpose()) ** 2, 1))

        workspace, module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_CENTROID
        module.wants_per_parent_means.value = True
        mnames = module.get_measurements(workspace.pipeline,
                                         PARENT_OBJECTS,
                                         "_".join((R.C_MEAN, CHILD_OBJECTS)))
        self.assertTrue(R.FF_CENTROID % PARENT_OBJECTS in mnames)
        feat_mean = R.FF_MEAN % (CHILD_OBJECTS, R.FF_CENTROID % PARENT_OBJECTS)
        mcolumns = module.get_measurement_columns(workspace.pipeline)
        self.assertTrue(any([c[0] == PARENT_OBJECTS and c[1] == feat_mean
                             for c in mcolumns]))
        m = workspace.measurements
        m[CHILD_OBJECTS, R.M_LOCATION_CENTER_X, 1] = child_centers[1]
        m[CHILD_OBJECTS, R.M_LOCATION_CENTER_Y, 1] = child_centers[0]
        module.run(workspace)

        v = m[PARENT_OBJECTS, feat_mean, 1]

        plabel = m[CHILD_OBJECTS, "_".join((R.C_PARENT, PARENT_OBJECTS)), 1]

        self.assertEqual(len(v), 4)
        for idx in range(4):
            if np.any(plabel == idx + 1):
                self.assertAlmostEqual(
                        v[idx], np.mean(expected[plabel == idx + 1]), 4)
