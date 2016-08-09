import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureobjectneighbors
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

OBJECTS_NAME = 'objectsname'
NEIGHBORS_NAME = 'neighborsname'


class TestMeasureObjectNeighbors(unittest.TestCase):
    def make_workspace(self, labels, mode, distance=0, neighbors_labels=None):
        '''Make a workspace for testing MeasureObjectNeighbors'''
        module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
        module.module_num = 1
        module.object_name.value = OBJECTS_NAME
        module.distance_method.value = mode
        module.distance.value = distance
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        object_set = cellprofiler.region.Set()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        measurements = cellprofiler.measurement.Measurements()
        measurements.group_index = 1
        measurements.group_number = 1
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        objects = cellprofiler.region.Region()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        if neighbors_labels is None:
            module.neighbors_name.value = OBJECTS_NAME
        else:
            module.neighbors_name.value = NEIGHBORS_NAME
            objects = cellprofiler.region.Region()
            objects.segmented = neighbors_labels
            object_set.add_objects(objects, NEIGHBORS_NAME)
        return workspace, module

    def test_01_02_load_v1(self):
        '''Load a pipeline with a v1 MeasureObjectNeighbors module'''
        data = ('eJztWt1u2zYUph0naFpg7VoMG9AbXjZdLMhusqbBkNqJu85b7RiN0aAo2o6'
                'W6JgFTRoSldodCuxyj7PLXe6R9ggTZTqWWSdS5Cy2AQkQpHPE7/zxkBQPWC'
                's3X5T34bZhwlq5mW8TimGDItHmTncXMrEJDxyMBLYhZ7vw2H/+4jFYeAwL5'
                'u7Wzm5xGxZN8wlIdmWqta/8R+chAGv+84Z/Z9WnVUVnQrekj7AQhJ24qyAH'
                'vlP8f/z7FXIIalH8ClEPu2MVI36VtXlz0Dv7VOO2R3EddcON/avudVvYcQ/'
                'bI6D63CB9TI/IJ6y5MGr2Ep8Sl3Cm8Eq+zj3Ty4WmV8bhr3vjOGS0OKz49/'
                '0QX7b/GYzb56bE7etQ+zuKJswmp8T2EIWki07OrJDyzAh5KxPyVkClXg5wO'
                'xG4Nc2OtSDOFsVkqLcUgb+j4eXdxH2Rf9ZHloBdJKzOVdgRFc9vNLykn/V7'
                'iNnQY4JQiOwPyMJMxIxnZkJeBjyKictN4HLgh80tM4m+bRAv/jc1vyVd4ZB'
                'xAT1XDYgk+fPaz77/E5edwGVBncez8zxcVH7c0+Ik6cPWB2yJOiYnnRZ3Dr'
                'gXZEc8ed9q8iRdwW3kUQGrcvDCCnF86dwZzBTHJOPmuYMxc7+M55qGH10j/'
                'Lp6xsm7W5peSR8K14PPKW8heiZn0fJOxxUMM5a/tzV/Jd3AjpxRmtyzOv66'
                'N5PdSdYL0zCDa7OgXtT3OHmzqsmT9L5anuPgb2h4SR90EGOYFpPMk74vhVn'
                'yNen6WIrArWt+SrrKhD++iBico/8q7dbXhQKINz7nbbfev3XOcJL5vWDOZu'
                'cfEfp+1eIk6XcPnjZ+lD/aeM/4fuO9pI4xpS/5x7035Xzj7caIc8Cp12V7b'
                '8z8k7e/FzaLn4eNj4iPDJgbseOl9/PjKbjL+N2J0Lej+S1paftrjBzl0Nbn'
                'jbxk1TgTHcUrKl4FDcacq5qvk+bfVa5rSf6nrsPuZYlvaud87TTV/8yi2Rl'
                'nPVgEO+PsSxbBzuXNz+2FnPen7WeaHzm0KHJdVYmZh91J9gXHckcry4Knsg'
                'DGLBySt2hxn7be/sQdfOL423F7dv29u5er212nn0GRTzraiy9nWp7yoI4xF'
                'nTd+RbSDwmzcS8kb57zUYqbH64ELs6jaXXj8XgYptEy+Zvi0jxYJlwJpP2S'
                '4lLcouCi/rvugsnxKGnuCUoY/uLHa5n8TnEpbhHWu7j7s2XxN8WluBQ3P9y'
                'fmTFOrzvpdVHZ/reQnmnz00MwOT9J2sKU9hwuzwU6Rjc4vOYalCN7eHrMeO'
                'G/VkMHyYJ6WISekqandJ4eYmMmSHvQc3xtnuBdJIhlVBW34XPLI67U24/Qu'
                '6/p3T9Pbxcj13PwsObE1Nkd16gN2ZNHekJ+h/txfYr+cH9kfer2/fUL+x+A'
                'yX4f58O/T5Poy65kAn3hcxe3InC5kE0jP/8Gl8u7Bxe0H/l4Xe3/A6vxr7o=')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
        self.assertEqual(module.object_name, 'Nuclei')
        self.assertEqual(module.neighbors_name, 'Nuclei')
        self.assertEqual(module.distance_method, cellprofiler.modules.measureobjectneighbors.D_EXPAND)
        self.assertTrue(module.wants_count_image.value)
        self.assertEqual(module.count_image_name.value, "ObjectNeighborCount")
        self.assertEqual(module.count_colormap.value, "Greens")
        self.assertTrue(module.wants_percent_touching_image.value)
        self.assertEqual(module.touching_image_name.value, "PercentTouching")
        self.assertEqual(module.touching_colormap.value, "Blues")

    def test_01_03_load_v2(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:11016

MeasureObjectNeighbors:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select objects to measure:glia
    Select neighboring objects to measure:neurites
    Method to determine neighbors:Expand until adjacent
    Neighbor distance:2
    Retain the image of objects colored by numbers of neighbors for use later in the pipeline (for example, in SaveImages)?:No
    Name the output image:countimage
    Select colormap:pink
    Retain the image of objects colored by percent of touching pixels for use later in the pipeline (for example, in SaveImages)?:No
    Name the output image:touchingimage
    Select a colormap:purple
'''
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
        self.assertEqual(module.object_name, "glia")
        self.assertEqual(module.neighbors_name, "neurites")
        self.assertEqual(module.distance_method, cellprofiler.modules.measureobjectneighbors.D_EXPAND)
        self.assertEqual(module.distance, 2)
        self.assertFalse(module.wants_count_image)
        self.assertEqual(module.count_image_name, "countimage")
        self.assertEqual(module.count_colormap, "pink")
        self.assertFalse(module.wants_percent_touching_image)
        self.assertEqual(module.touching_image_name, "touchingimage")
        self.assertEqual(module.touching_colormap, "purple")

    def test_02_02_empty(self):
        '''Test a labels matrix with no objects'''
        workspace, module = self.make_workspace(numpy.zeros((10, 10), int),
                                                cellprofiler.modules.measureobjectneighbors.D_EXPAND, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Expanded")
        self.assertEqual(len(neighbors), 0)
        features = m.get_feature_names(OBJECTS_NAME)
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(features), len(columns))
        for column in columns:
            self.assertEqual(column[0], OBJECTS_NAME)
            self.assertTrue(column[1] in features)
            self.assertTrue(column[2] == (cellprofiler.measurement.COLTYPE_INTEGER
                                          if column[1].find('Number') != -1
                                          else cellprofiler.measurement.COLTYPE_FLOAT))

    def test_02_03_one(self):
        '''Test a labels matrix with a single object'''
        labels = numpy.zeros((10, 10), int)
        labels[3:5, 4:6] = 1
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_EXPAND, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Expanded")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0], 0)
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Expanded")
        self.assertEqual(len(pct), 1)
        self.assertEqual(pct[0], 0)

    def test_02_04_two_expand(self):
        '''Test a labels matrix with two objects'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[8, 7] = 2
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_EXPAND, 5)
        module.run(workspace)
        self.assertEqual(tuple(module.get_categories(None, OBJECTS_NAME)),
                         ("Neighbors",))
        self.assertEqual(tuple(module.get_measurements(None, OBJECTS_NAME,
                                                       "Neighbors")),
                         tuple(cellprofiler.modules.measureobjectneighbors.M_ALL))
        self.assertEqual(tuple(module.get_measurement_scales(None,
                                                             OBJECTS_NAME,
                                                             "Neighbors",
                                                             "NumberOfNeighbors",
                                                             None)),
                         ("Expanded",))
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Expanded")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(numpy.all(neighbors == 1))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Expanded")
        #
        # This is what the patch looks like:
        #  P P P P P P P P P P
        #  P I I I I I I I O O
        #  P I I I I I O O O N
        #  P I I I I O O N N N
        #  P I I I O O N N N N
        #  P I I O O N N N N N
        #  P I O O N N N N N N
        #  O O O N N N N N N N
        #  O N N N N N N N N N
        #  N N N N N N N N N N
        #
        # where P = perimeter, but not overlapping the second object
        #       I = interior, not perimeter
        #       O = dilated 2nd object overlaps perimeter
        #       N = neigbor object, not overlapping
        #
        # There are 33 perimeter pixels (P + O) and 17 perimeter pixels
        # that overlap the dilated neighbor (O).
        #
        self.assertEqual(len(pct), 2)
        self.assertAlmostEqual(pct[0], 100.0 * 17.0 / 33.0)
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Expanded")
        self.assertEqual(len(fo), 2)
        self.assertEqual(fo[0], 2)
        self.assertEqual(fo[1], 1)
        x = m.get_current_measurement(OBJECTS_NAME,
                                      "Neighbors_FirstClosestDistance_Expanded")
        self.assertAlmostEqual(len(x), 2)
        self.assertAlmostEqual(x[0], numpy.sqrt(61))
        self.assertAlmostEqual(x[1], numpy.sqrt(61))

    def test_02_04_two_not_adjacent(self):
        '''Test a labels matrix with two objects, not adjacent'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[8, 7] = 2
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5)
        module.run(workspace)
        self.assertEqual(tuple(module.get_measurement_scales(None,
                                                             OBJECTS_NAME,
                                                             "Neighbors",
                                                             "NumberOfNeighbors",
                                                             None)),
                         ("Adjacent",))
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(numpy.all(neighbors == 0))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        self.assertEqual(len(pct), 2)
        self.assertTrue(numpy.all(pct == 0))

    def test_02_05_adjacent(self):
        '''Test a labels matrix with two objects, adjacent'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 3] = 2
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(numpy.all(neighbors == 1))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        self.assertEqual(len(pct), 2)
        self.assertAlmostEqual(pct[0], 100)
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Adjacent")
        self.assertEqual(len(fo), 2)
        self.assertEqual(fo[0], 2)
        self.assertEqual(fo[1], 1)

    def test_02_06_manual_not_touching(self):
        '''Test a labels matrix with two objects not touching'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1  # Pythagoras triangle 3-4-5
        labels[5, 6] = 2
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4)
        module.run(workspace)
        self.assertEqual(tuple(module.get_measurement_scales(None,
                                                             OBJECTS_NAME,
                                                             "Neighbors",
                                                             "NumberOfNeighbors",
                                                             None)),
                         ("4",))
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_4")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(numpy.all(neighbors == 0))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_4")
        self.assertEqual(len(pct), 2)
        self.assertAlmostEqual(pct[0], 0)

    def test_02_07_manual_touching(self):
        '''Test a labels matrix with two objects touching'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1  # Pythagoras triangle 3-4-5
        labels[5, 6] = 2
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_WITHIN, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_5")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(numpy.all(neighbors == 1))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_5")
        self.assertEqual(len(pct), 2)
        self.assertAlmostEqual(pct[0], 100)

        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_5")
        self.assertEqual(len(fo), 2)
        self.assertEqual(fo[0], 2)
        self.assertEqual(fo[1], 1)

    def test_02_08_three(self):
        '''Test the angles between three objects'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1  # x=3,y=4,5 triangle
        labels[2, 5] = 2
        labels[6, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_WITHIN, 5)
        module.run(workspace)
        m = workspace.measurements
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_5")
        self.assertEqual(len(fo), 3)
        self.assertEqual(fo[0], 2)
        self.assertEqual(fo[1], 1)
        self.assertEqual(fo[2], 1)
        so = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_SecondClosestObjectNumber_5")
        self.assertEqual(len(so), 3)
        self.assertEqual(so[0], 3)
        self.assertEqual(so[1], 3)
        self.assertEqual(so[2], 2)
        d = m.get_current_measurement(OBJECTS_NAME,
                                      "Neighbors_SecondClosestDistance_5")
        self.assertEqual(len(d), 3)
        self.assertAlmostEqual(d[0], 4)
        self.assertAlmostEqual(d[1], 5)
        self.assertAlmostEqual(d[2], 5)

        angle = m.get_current_measurement(OBJECTS_NAME,
                                          "Neighbors_AngleBetweenNeighbors_5")
        self.assertEqual(len(angle), 3)
        self.assertAlmostEqual(angle[0], 90)
        self.assertAlmostEqual(angle[1], numpy.arccos(3.0 / 5.0) * 180.0 / numpy.pi)
        self.assertAlmostEqual(angle[2], numpy.arccos(4.0 / 5.0) * 180.0 / numpy.pi)

    def test_02_09_touching_discarded(self):
        '''Make sure that we count edge-touching discarded objects

        Regression test of IMG-1012.
        '''
        labels = numpy.zeros((10, 10), int)
        labels[2, 3] = 1
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5)
        object_set = workspace.object_set
        self.assertTrue(isinstance(object_set, cellprofiler.region.Set))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(isinstance(objects, cellprofiler.region.Region))

        sm_labels = labels.copy() * 3
        sm_labels[-1, -1] = 1
        sm_labels[0:2, 3] = 2
        objects.small_removed_segmented = sm_labels
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 1)
        self.assertTrue(numpy.all(neighbors == 1))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        self.assertEqual(len(pct), 1)
        self.assertAlmostEqual(pct[0], 100)
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Adjacent")
        self.assertEqual(len(fo), 1)
        self.assertEqual(fo[0], 0)

        angle = m.get_current_measurement(OBJECTS_NAME,
                                          "Neighbors_AngleBetweenNeighbors_Adjacent")
        self.assertEqual(len(angle), 1)
        self.assertFalse(numpy.isnan(angle)[0])

    def test_02_10_all_discarded(self):
        '''Test the case where all objects touch the edge

        Regression test of a follow-on bug to IMG-1012
        '''
        labels = numpy.zeros((10, 10), int)
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5)
        object_set = workspace.object_set
        self.assertTrue(isinstance(object_set, cellprofiler.region.Set))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(isinstance(objects, cellprofiler.region.Region))

        # Needs 2 objects to trigger the bug
        sm_labels = numpy.zeros((10, 10), int)
        sm_labels[0:2, 3] = 1
        sm_labels[-3:-1, 5] = 2
        objects.small_removed_segmented = sm_labels
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 0)
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        self.assertEqual(len(pct), 0)
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Adjacent")
        self.assertEqual(len(fo), 0)

    def test_03_01_NeighborCountImage(self):
        '''Test production of a neighbor-count image'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1  # x=3,y=4,5 triangle
        labels[2, 5] = 2
        labels[6, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4)
        module.wants_count_image.value = True
        module.count_image_name.value = 'my_image'
        module.count_colormap.value = 'jet'
        module.run(workspace)
        image = workspace.image_set.get_image('my_image').pixel_data
        self.assertEqual(tuple(image.shape), (10, 10, 3))
        # Everything off of the images should be black
        self.assertTrue(numpy.all(image[labels[labels == 0], :] == 0))
        # The corners should match 1 neighbor and should get the same color
        self.assertTrue(numpy.all(image[2, 5, :] == image[6, 2, :]))
        # The pixel at the right angle should have a different color
        self.assertFalse(numpy.all(image[2, 2, :] == image[2, 5, :]))

    def test_04_01_PercentTouchingImage(self):
        '''Test production of a percent touching image'''
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 5] = 2
        labels[6, 2] = 3
        labels[7, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4)
        module.wants_percent_touching_image.value = True
        module.touching_image_name.value = 'my_image'
        module.touching_colormap.value = 'jet'
        module.run(workspace)
        image = workspace.image_set.get_image('my_image').pixel_data
        self.assertEqual(tuple(image.shape), (10, 10, 3))
        # Everything off of the images should be black
        self.assertTrue(numpy.all(image[labels[labels == 0], :] == 0))
        # 1 and 2 are at 100 %
        self.assertTrue(numpy.all(image[2, 2, :] == image[2, 5, :]))
        # 3 is at 50% and should have a different color
        self.assertFalse(numpy.all(image[2, 2, :] == image[6, 2, :]))

    def test_05_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
        module.object_name.value = OBJECTS_NAME
        module.neighbors_name.value = OBJECTS_NAME
        module.distance.value = 5
        for distance_method, scale in ((cellprofiler.modules.measureobjectneighbors.D_EXPAND, cellprofiler.modules.measureobjectneighbors.S_EXPANDED),
                                       (cellprofiler.modules.measureobjectneighbors.D_ADJACENT, cellprofiler.modules.measureobjectneighbors.S_ADJACENT),
                                       (cellprofiler.modules.measureobjectneighbors.D_WITHIN, "5")):
            module.distance_method.value = distance_method
            columns = module.get_measurement_columns(None)
            features = ["%s_%s_%s" % (cellprofiler.modules.measureobjectneighbors.C_NEIGHBORS, feature, scale)
                        for feature in cellprofiler.modules.measureobjectneighbors.M_ALL]
            self.assertEqual(len(columns), len(features))
            for column in columns:
                self.assertTrue(column[1] in features, "Unexpected column name: %s" % column[1])

    def test_05_02_get_measurement_columns_neighbors(self):
        module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
        module.object_name.value = OBJECTS_NAME
        module.neighbors_name.value = NEIGHBORS_NAME
        module.distance.value = 5
        for distance_method, scale in ((cellprofiler.modules.measureobjectneighbors.D_EXPAND, cellprofiler.modules.measureobjectneighbors.S_EXPANDED),
                                       (cellprofiler.modules.measureobjectneighbors.D_ADJACENT, cellprofiler.modules.measureobjectneighbors.S_ADJACENT),
                                       (cellprofiler.modules.measureobjectneighbors.D_WITHIN, "5")):
            module.distance_method.value = distance_method
            columns = module.get_measurement_columns(None)
            features = ["%s_%s_%s_%s" % (cellprofiler.modules.measureobjectneighbors.C_NEIGHBORS, feature, NEIGHBORS_NAME, scale)
                        for feature in cellprofiler.modules.measureobjectneighbors.M_ALL
                        if feature != cellprofiler.modules.measureobjectneighbors.M_PERCENT_TOUCHING]
            self.assertEqual(len(columns), len(features))
            for column in columns:
                self.assertTrue(column[1] in features, "Unexpected column name: %s" % column[1])

    def test_06_01_neighbors_zeros(self):
        blank_labels = numpy.zeros((20, 10), int)
        one_object = numpy.zeros((20, 10), int)
        one_object[2:-2, 2:-2] = 1

        cases = ((blank_labels, blank_labels, 0, 0),
                 (blank_labels, one_object, 0, 1),
                 (one_object, blank_labels, 1, 0))
        for olabels, nlabels, ocount, ncount in cases:
            for mode in cellprofiler.modules.measureobjectneighbors.D_ALL:
                workspace, module = self.make_workspace(
                        olabels, mode, neighbors_labels=nlabels)
                self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
                module.run(workspace)
                m = workspace.measurements
                self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                for feature in module.all_features:
                    v = m.get_current_measurement(
                            OBJECTS_NAME, module.get_measurement_name(feature))
                    self.assertEqual(len(v), ocount)

    def test_06_02_one_neighbor(self):
        olabels = numpy.zeros((20, 10), int)
        olabels[2, 2] = 1
        nlabels = numpy.zeros((20, 10), int)
        nlabels[-2, -2] = 1
        for mode in cellprofiler.modules.measureobjectneighbors.D_ALL:
            workspace, module = self.make_workspace(
                    olabels, mode, distance=20, neighbors_labels=nlabels)
            self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 1)
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_OBJECT_NUMBER))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 0)
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_DISTANCE))
            self.assertEqual(len(v), 1)
            self.assertAlmostEqual(v[0], numpy.sqrt(16 ** 2 + 6 ** 2))
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 0 if mode == cellprofiler.modules.measureobjectneighbors.D_ADJACENT else 1)

    def test_06_03_two_neighbors(self):
        olabels = numpy.zeros((20, 10), int)
        olabels[2, 2] = 1
        nlabels = numpy.zeros((20, 10), int)
        nlabels[5, 2] = 2
        nlabels[2, 6] = 1
        workspace, module = self.make_workspace(
                olabels, cellprofiler.modules.measureobjectneighbors.D_EXPAND, distance=20, neighbors_labels=nlabels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER))
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 2)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_OBJECT_NUMBER))
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 1)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_DISTANCE))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 3)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_DISTANCE))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 4)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_ANGLE_BETWEEN_NEIGHBORS))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 90)

    def test_07_01_relationships(self):
        labels = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        workspace, module = self.make_workspace(
                labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 2)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        k = m.get_relationship_groups()
        self.assertEqual(len(k), 1)
        k = k[0]
        self.assertTrue(isinstance(k, cellprofiler.measurement.RelationshipKey))
        self.assertEqual(k.module_number, 1)
        self.assertEqual(k.object_name1, OBJECTS_NAME)
        self.assertEqual(k.object_name2, OBJECTS_NAME)
        self.assertEqual(k.relationship, cellprofiler.measurement.NEIGHBORS)
        r = m.get_relationships(
                k.module_number,
                k.relationship,
                k.object_name1,
                k.object_name2)
        self.assertEqual(len(r), 8)
        ro1 = r[cellprofiler.measurement.R_FIRST_OBJECT_NUMBER]
        ro2 = r[cellprofiler.measurement.R_SECOND_OBJECT_NUMBER]
        numpy.testing.assert_array_equal(numpy.unique(ro1[ro2 == 3]),
                                         numpy.array([1, 2, 4, 5]))
        numpy.testing.assert_array_equal(numpy.unique(ro2[ro1 == 3]),
                                         numpy.array([1, 2, 4, 5]))

    def test_07_02_neighbors(self):
        labels = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        nlabels = numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        workspace, module = self.make_workspace(
                labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 2, nlabels)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        k = m.get_relationship_groups()
        self.assertEqual(len(k), 1)
        k = k[0]
        self.assertTrue(isinstance(k, cellprofiler.measurement.RelationshipKey))
        self.assertEqual(k.module_number, 1)
        self.assertEqual(k.object_name1, OBJECTS_NAME)
        self.assertEqual(k.object_name2, NEIGHBORS_NAME)
        self.assertEqual(k.relationship, cellprofiler.measurement.NEIGHBORS)
        r = m.get_relationships(
                k.module_number,
                k.relationship,
                k.object_name1,
                k.object_name2)
        self.assertEqual(len(r), 3)
        ro1 = r[cellprofiler.measurement.R_FIRST_OBJECT_NUMBER]
        ro2 = r[cellprofiler.measurement.R_SECOND_OBJECT_NUMBER]
        self.assertTrue(numpy.all(ro2 == 1))
        numpy.testing.assert_array_equal(numpy.unique(ro1), numpy.array([1, 3, 4]))

    def test_08_01_missing_object(self):
        # Regression test of issue 434
        #
        # Catch case of no pixels for an object
        #
        labels = numpy.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 3] = 3
        workspace, module = self.make_workspace(labels,
                                                cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        numpy.testing.assert_array_equal(neighbors, [1, 0, 1])
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        numpy.testing.assert_array_almost_equal(pct, [100.0, 0, 100.0])
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Adjacent")
        numpy.testing.assert_array_equal(fo, [3, 0, 1])

    def test_08_02_small_removed(self):
        # Regression test of issue #1179
        #
        # neighbor_objects.small_removed_segmented + objects touching border
        # with higher object numbers
        #
        neighbors = numpy.zeros((11, 13), int)
        neighbors[5:7, 4:8] = 1
        neighbors_unedited = numpy.zeros((11, 13), int)
        neighbors_unedited[5:7, 4:8] = 1
        neighbors_unedited[0:4, 4:8] = 2

        objects = numpy.zeros((11, 13), int)
        objects[1:6, 5:7] = 1

        workspace, module = self.make_workspace(
                objects, cellprofiler.modules.measureobjectneighbors.D_WITHIN, neighbors_labels=neighbors)
        no = workspace.object_set.get_objects(NEIGHBORS_NAME)
        no.unedited_segmented = neighbors_unedited
        no.small_removed_segmented = neighbors
        module.run(workspace)
        m = workspace.measurements
        v = m[OBJECTS_NAME,
              module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS), 1]
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 2)

    def test_08_03_object_is_missing(self):
        # regression test of #1639
        #
        # Object # 2 should match neighbor # 1, but because of
        # an error in masking distances, neighbor #1 is masked out
        #
        olabels = numpy.zeros((20, 10), int)
        olabels[2, 2] = 2
        nlabels = numpy.zeros((20, 10), int)
        nlabels[2, 3] = 1
        nlabels[5, 2] = 2
        workspace, module = self.make_workspace(
                olabels, cellprofiler.modules.measureobjectneighbors.D_EXPAND, distance=20, neighbors_labels=nlabels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        ftr = module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER)
        values = m[OBJECTS_NAME, ftr]
        self.assertEqual(values[1], 1)

    def test_08_04_small_removed_same(self):
        # Regression test of issue #1672
        #
        # Objects with small removed failed.
        #
        objects = numpy.zeros((11, 13), int)
        objects[5:7, 1:3] = 1
        objects[6:8, 5:7] = 2
        objects_unedited = objects.copy()
        objects_unedited[0:2, 0:2] = 3

        workspace, module = self.make_workspace(
                objects, cellprofiler.modules.measureobjectneighbors.D_EXPAND, distance=1)
        no = workspace.object_set.get_objects(OBJECTS_NAME)
        no.unedited_segmented = objects_unedited
        no.small_removed_segmented = objects
        module.run(workspace)
        m = workspace.measurements
        v = m[OBJECTS_NAME,
              module.get_measurement_name(cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS), 1]
        self.assertEqual(len(v), 2)
        self.assertEqual(v[0], 1)
