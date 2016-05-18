'''test_measureobjectneighbors.py Test the MeasureObjectNeighbors module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.measureobjectneighbors as M

OBJECTS_NAME = 'objectsname'
NEIGHBORS_NAME = 'neighborsname'


class TestMeasureObjectNeighbors(unittest.TestCase):
    def make_workspace(self, labels, mode, distance=0, neighbors_labels=None):
        '''Make a workspace for testing MeasureObjectNeighbors'''
        module = M.MeasureObjectNeighbors()
        module.module_num = 1
        module.object_name.value = OBJECTS_NAME
        module.distance_method.value = mode
        module.distance.value = distance
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        measurements = cpmeas.Measurements()
        measurements.group_index = 1
        measurements.group_number = 1
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        if neighbors_labels is None:
            module.neighbors_name.value = OBJECTS_NAME
        else:
            module.neighbors_name.value = NEIGHBORS_NAME
            objects = cpo.Objects()
            objects.segmented = neighbors_labels
            object_set.add_objects(objects, NEIGHBORS_NAME)
        return workspace, module

    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline with a MeasureObjectNeighbors module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUggH0l6leQqG5gqGZlbG5lZGxgpGBgaWCiQDBkZPX34GBoY5TA'
                'wMFXPuRsTnXzYQKL+9halt2SauI87yn3a056o0CPT7SDd5rVl7w6TxSPfsH'
                'WvVj68TSs/6mWLzyfSH2ZHcSVZx1/+8jmpRPW31/F15+f6Yc57fGBn+ijH0'
                'JtSKnNGK6KnSCF+i+cT7iOUy5fV8Jh+M7vyXPZP9IslC4YWUkHylgtzFuna'
                '+m1dzhOJ3JWzzDxC32lRksGufscO669JubQ+f2s1TKuqsV5ogM/FWeEm37Z'
                'mMDPlHq3dvjD8muv/3QZs70Z8CAitWsG5Wfcv9lz9h5fq1D9ZqvjQ/XNm67'
                'M+PxyJRMeUK248xHfc4uFK1IOvBnmqr7//tjq5b+CP/zM+tHIduProf/Ka1'
                'xqxtitKnXzcMkuVOHZs4rV41MX7m6+kPdc/6ed/IfHD3YrQyd7z/FJvWmnM'
                'lp3a8q058OO3J3SmV52f//77g2faZx5NZ3U+kG59TY6nii3i05Gmg74MQx+'
                'mip+fva5I+u1/jf+TP6hNVrwrPX7cXNGtQOXazJKZyfnaxe+GK7yctjIQLn'
                '5x97OtmvfW8hHrbzu5fEV/nlGm4Xzlco3ZUeO384A/zNCucjl05tv3GN8WL'
                'G5ndP778LfBR9K266yOLr5bcfEU5oh3Oa2dPFqpZ072Y496JVuPzTSyeFqm'
                '/7647v3pNocjaft/jhfd/hn6zm/JUcdP/ZWvr46f9dri5zyxfSnn6z8lvLF'
                'afszXb4SgWzzT1bbbQPLkvFiaaE96vet+Suo/7Ru6RnS8rzOeW3/q1//fv8'
                'OtbqmwYPOVaztT4GbiWcwUdyw17914/a772rfB9BhnlS8Qn6fy0CS9oubNR'
                '8GmFxevdNyaef6RToS9ioxiv8OKn3quz8rf1Di3+2NApL5F1P2/JhyDeJc1'
                'he6/8roir/7r8q/V/1ktvtgQ8d999/d/luB9/WT/d/X/Y9NfFG9P+xcv3CH'
                '8AAMz0fHw=')
        #
        # MeasureObjectNeighbors is module # 3
        # object_name = Nuclei
        # distance = 0 which should translate into D_EXPAND
        #
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
        self.assertEqual(module.object_name, 'Nuclei')
        self.assertEqual(module.distance_method, M.D_EXPAND)
        self.assertFalse(module.wants_count_image.value)
        self.assertFalse(module.wants_percent_touching_image.value)

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
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
        self.assertEqual(module.object_name, 'Nuclei')
        self.assertEqual(module.neighbors_name, 'Nuclei')
        self.assertEqual(module.distance_method, M.D_EXPAND)
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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
        self.assertEqual(module.object_name, "glia")
        self.assertEqual(module.neighbors_name, "neurites")
        self.assertEqual(module.distance_method, M.D_EXPAND)
        self.assertEqual(module.distance, 2)
        self.assertFalse(module.wants_count_image)
        self.assertEqual(module.count_image_name, "countimage")
        self.assertEqual(module.count_colormap, "pink")
        self.assertFalse(module.wants_percent_touching_image)
        self.assertEqual(module.touching_image_name, "touchingimage")
        self.assertEqual(module.touching_colormap, "purple")

    def test_02_02_empty(self):
        '''Test a labels matrix with no objects'''
        workspace, module = self.make_workspace(np.zeros((10, 10), int),
                                                M.D_EXPAND, 5)
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
            self.assertTrue(column[2] == (cpmeas.COLTYPE_INTEGER
                                          if column[1].find('Number') != -1
                                          else cpmeas.COLTYPE_FLOAT))

    def test_02_03_one(self):
        '''Test a labels matrix with a single object'''
        labels = np.zeros((10, 10), int)
        labels[3:5, 4:6] = 1
        workspace, module = self.make_workspace(labels,
                                                M.D_EXPAND, 5)
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
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[8, 7] = 2
        workspace, module = self.make_workspace(labels,
                                                M.D_EXPAND, 5)
        module.run(workspace)
        self.assertEqual(tuple(module.get_categories(None, OBJECTS_NAME)),
                         ("Neighbors",))
        self.assertEqual(tuple(module.get_measurements(None, OBJECTS_NAME,
                                                       "Neighbors")),
                         tuple(M.M_ALL))
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
        self.assertTrue(np.all(neighbors == 1))
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
        self.assertAlmostEqual(x[0], np.sqrt(61))
        self.assertAlmostEqual(x[1], np.sqrt(61))

    def test_02_04_two_not_adjacent(self):
        '''Test a labels matrix with two objects, not adjacent'''
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[8, 7] = 2
        workspace, module = self.make_workspace(labels,
                                                M.D_ADJACENT, 5)
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
        self.assertTrue(np.all(neighbors == 0))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        self.assertEqual(len(pct), 2)
        self.assertTrue(np.all(pct == 0))

    def test_02_05_adjacent(self):
        '''Test a labels matrix with two objects, adjacent'''
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 3] = 2
        workspace, module = self.make_workspace(labels,
                                                M.D_ADJACENT, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(np.all(neighbors == 1))
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
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1  # Pythagoras triangle 3-4-5
        labels[5, 6] = 2
        workspace, module = self.make_workspace(labels,
                                                M.D_WITHIN, 4)
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
        self.assertTrue(np.all(neighbors == 0))
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_4")
        self.assertEqual(len(pct), 2)
        self.assertAlmostEqual(pct[0], 0)

    def test_02_07_manual_touching(self):
        '''Test a labels matrix with two objects touching'''
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1  # Pythagoras triangle 3-4-5
        labels[5, 6] = 2
        workspace, module = self.make_workspace(labels,
                                                M.D_WITHIN, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_5")
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(np.all(neighbors == 1))
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
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1  # x=3,y=4,5 triangle
        labels[2, 5] = 2
        labels[6, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                M.D_WITHIN, 5)
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
        self.assertAlmostEqual(angle[1], np.arccos(3.0 / 5.0) * 180.0 / np.pi)
        self.assertAlmostEqual(angle[2], np.arccos(4.0 / 5.0) * 180.0 / np.pi)

    def test_02_09_touching_discarded(self):
        '''Make sure that we count edge-touching discarded objects

        Regression test of IMG-1012.
        '''
        labels = np.zeros((10, 10), int)
        labels[2, 3] = 1
        workspace, module = self.make_workspace(labels,
                                                M.D_ADJACENT, 5)
        object_set = workspace.object_set
        self.assertTrue(isinstance(object_set, cpo.ObjectSet))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(isinstance(objects, cpo.Objects))

        sm_labels = labels.copy() * 3
        sm_labels[-1, -1] = 1
        sm_labels[0:2, 3] = 2
        objects.small_removed_segmented = sm_labels
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        self.assertEqual(len(neighbors), 1)
        self.assertTrue(np.all(neighbors == 1))
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
        self.assertFalse(np.isnan(angle)[0])

    def test_02_10_all_discarded(self):
        '''Test the case where all objects touch the edge

        Regression test of a follow-on bug to IMG-1012
        '''
        labels = np.zeros((10, 10), int)
        workspace, module = self.make_workspace(labels,
                                                M.D_ADJACENT, 5)
        object_set = workspace.object_set
        self.assertTrue(isinstance(object_set, cpo.ObjectSet))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(isinstance(objects, cpo.Objects))

        # Needs 2 objects to trigger the bug
        sm_labels = np.zeros((10, 10), int)
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
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1  # x=3,y=4,5 triangle
        labels[2, 5] = 2
        labels[6, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                M.D_WITHIN, 4)
        module.wants_count_image.value = True
        module.count_image_name.value = 'my_image'
        module.count_colormap.value = 'jet'
        module.run(workspace)
        image = workspace.image_set.get_image('my_image').pixel_data
        self.assertEqual(tuple(image.shape), (10, 10, 3))
        # Everything off of the images should be black
        self.assertTrue(np.all(image[labels[labels == 0], :] == 0))
        # The corners should match 1 neighbor and should get the same color
        self.assertTrue(np.all(image[2, 5, :] == image[6, 2, :]))
        # The pixel at the right angle should have a different color
        self.assertFalse(np.all(image[2, 2, :] == image[2, 5, :]))

    def test_04_01_PercentTouchingImage(self):
        '''Test production of a percent touching image'''
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 5] = 2
        labels[6, 2] = 3
        labels[7, 2] = 3
        workspace, module = self.make_workspace(labels,
                                                M.D_WITHIN, 4)
        module.wants_percent_touching_image.value = True
        module.touching_image_name.value = 'my_image'
        module.touching_colormap.value = 'jet'
        module.run(workspace)
        image = workspace.image_set.get_image('my_image').pixel_data
        self.assertEqual(tuple(image.shape), (10, 10, 3))
        # Everything off of the images should be black
        self.assertTrue(np.all(image[labels[labels == 0], :] == 0))
        # 1 and 2 are at 100 %
        self.assertTrue(np.all(image[2, 2, :] == image[2, 5, :]))
        # 3 is at 50% and should have a different color
        self.assertFalse(np.all(image[2, 2, :] == image[6, 2, :]))

    def test_05_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        module = M.MeasureObjectNeighbors()
        module.object_name.value = OBJECTS_NAME
        module.neighbors_name.value = OBJECTS_NAME
        module.distance.value = 5
        for distance_method, scale in ((M.D_EXPAND, M.S_EXPANDED),
                                       (M.D_ADJACENT, M.S_ADJACENT),
                                       (M.D_WITHIN, "5")):
            module.distance_method.value = distance_method
            columns = module.get_measurement_columns(None)
            features = ["%s_%s_%s" % (M.C_NEIGHBORS, feature, scale)
                        for feature in M.M_ALL]
            self.assertEqual(len(columns), len(features))
            for column in columns:
                self.assertTrue(column[1] in features, "Unexpected column name: %s" % column[1])

    def test_05_02_get_measurement_columns_neighbors(self):
        module = M.MeasureObjectNeighbors()
        module.object_name.value = OBJECTS_NAME
        module.neighbors_name.value = NEIGHBORS_NAME
        module.distance.value = 5
        for distance_method, scale in ((M.D_EXPAND, M.S_EXPANDED),
                                       (M.D_ADJACENT, M.S_ADJACENT),
                                       (M.D_WITHIN, "5")):
            module.distance_method.value = distance_method
            columns = module.get_measurement_columns(None)
            features = ["%s_%s_%s_%s" % (M.C_NEIGHBORS, feature, NEIGHBORS_NAME, scale)
                        for feature in M.M_ALL
                        if feature != M.M_PERCENT_TOUCHING]
            self.assertEqual(len(columns), len(features))
            for column in columns:
                self.assertTrue(column[1] in features, "Unexpected column name: %s" % column[1])

    def test_06_01_neighbors_zeros(self):
        blank_labels = np.zeros((20, 10), int)
        one_object = np.zeros((20, 10), int)
        one_object[2:-2, 2:-2] = 1

        cases = ((blank_labels, blank_labels, 0, 0),
                 (blank_labels, one_object, 0, 1),
                 (one_object, blank_labels, 1, 0))
        for olabels, nlabels, ocount, ncount in cases:
            for mode in M.D_ALL:
                workspace, module = self.make_workspace(
                        olabels, mode, neighbors_labels=nlabels)
                self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
                module.run(workspace)
                m = workspace.measurements
                self.assertTrue(isinstance(m, cpmeas.Measurements))
                for feature in module.all_features:
                    v = m.get_current_measurement(
                            OBJECTS_NAME, module.get_measurement_name(feature))
                    self.assertEqual(len(v), ocount)

    def test_06_02_one_neighbor(self):
        olabels = np.zeros((20, 10), int)
        olabels[2, 2] = 1
        nlabels = np.zeros((20, 10), int)
        nlabels[-2, -2] = 1
        for mode in M.D_ALL:
            workspace, module = self.make_workspace(
                    olabels, mode, distance=20, neighbors_labels=nlabels)
            self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(M.M_FIRST_CLOSEST_OBJECT_NUMBER))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 1)
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(M.M_SECOND_CLOSEST_OBJECT_NUMBER))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 0)
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(M.M_FIRST_CLOSEST_DISTANCE))
            self.assertEqual(len(v), 1)
            self.assertAlmostEqual(v[0], np.sqrt(16 ** 2 + 6 ** 2))
            v = m.get_current_measurement(
                    OBJECTS_NAME,
                    module.get_measurement_name(M.M_NUMBER_OF_NEIGHBORS))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0], 0 if mode == M.D_ADJACENT else 1)

    def test_06_03_two_neighbors(self):
        olabels = np.zeros((20, 10), int)
        olabels[2, 2] = 1
        nlabels = np.zeros((20, 10), int)
        nlabels[5, 2] = 2
        nlabels[2, 6] = 1
        workspace, module = self.make_workspace(
                olabels, M.D_EXPAND, distance=20, neighbors_labels=nlabels)
        self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(M.M_FIRST_CLOSEST_OBJECT_NUMBER))
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 2)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(M.M_SECOND_CLOSEST_OBJECT_NUMBER))
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 1)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(M.M_FIRST_CLOSEST_DISTANCE))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 3)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(M.M_SECOND_CLOSEST_DISTANCE))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 4)
        v = m.get_current_measurement(
                OBJECTS_NAME,
                module.get_measurement_name(M.M_ANGLE_BETWEEN_NEIGHBORS))
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 90)

    def test_07_01_relationships(self):
        labels = np.array([
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
                labels, M.D_WITHIN, 2)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        k = m.get_relationship_groups()
        self.assertEqual(len(k), 1)
        k = k[0]
        self.assertTrue(isinstance(k, cpmeas.RelationshipKey))
        self.assertEqual(k.module_number, 1)
        self.assertEqual(k.object_name1, OBJECTS_NAME)
        self.assertEqual(k.object_name2, OBJECTS_NAME)
        self.assertEqual(k.relationship, cpmeas.NEIGHBORS)
        r = m.get_relationships(
                k.module_number,
                k.relationship,
                k.object_name1,
                k.object_name2)
        self.assertEqual(len(r), 8)
        ro1 = r[cpmeas.R_FIRST_OBJECT_NUMBER]
        ro2 = r[cpmeas.R_SECOND_OBJECT_NUMBER]
        np.testing.assert_array_equal(np.unique(ro1[ro2 == 3]),
                                      np.array([1, 2, 4, 5]))
        np.testing.assert_array_equal(np.unique(ro2[ro1 == 3]),
                                      np.array([1, 2, 4, 5]))

    def test_07_02_neighbors(self):
        labels = np.array([
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

        nlabels = np.array([
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
                labels, M.D_WITHIN, 2, nlabels)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        k = m.get_relationship_groups()
        self.assertEqual(len(k), 1)
        k = k[0]
        self.assertTrue(isinstance(k, cpmeas.RelationshipKey))
        self.assertEqual(k.module_number, 1)
        self.assertEqual(k.object_name1, OBJECTS_NAME)
        self.assertEqual(k.object_name2, NEIGHBORS_NAME)
        self.assertEqual(k.relationship, cpmeas.NEIGHBORS)
        r = m.get_relationships(
                k.module_number,
                k.relationship,
                k.object_name1,
                k.object_name2)
        self.assertEqual(len(r), 3)
        ro1 = r[cpmeas.R_FIRST_OBJECT_NUMBER]
        ro2 = r[cpmeas.R_SECOND_OBJECT_NUMBER]
        self.assertTrue(np.all(ro2 == 1))
        np.testing.assert_array_equal(np.unique(ro1), np.array([1, 3, 4]))

    def test_08_01_missing_object(self):
        # Regression test of issue 434
        #
        # Catch case of no pixels for an object
        #
        labels = np.zeros((10, 10), int)
        labels[2, 2] = 1
        labels[2, 3] = 3
        workspace, module = self.make_workspace(labels,
                                                M.D_ADJACENT, 5)
        module.run(workspace)
        m = workspace.measurements
        neighbors = m.get_current_measurement(OBJECTS_NAME,
                                              "Neighbors_NumberOfNeighbors_Adjacent")
        np.testing.assert_array_equal(neighbors, [1, 0, 1])
        pct = m.get_current_measurement(OBJECTS_NAME,
                                        "Neighbors_PercentTouching_Adjacent")
        np.testing.assert_array_almost_equal(pct, [100.0, 0, 100.0])
        fo = m.get_current_measurement(OBJECTS_NAME,
                                       "Neighbors_FirstClosestObjectNumber_Adjacent")
        np.testing.assert_array_equal(fo, [3, 0, 1])

    def test_08_02_small_removed(self):
        # Regression test of issue #1179
        #
        # neighbor_objects.small_removed_segmented + objects touching border
        # with higher object numbers
        #
        neighbors = np.zeros((11, 13), int)
        neighbors[5:7, 4:8] = 1
        neighbors_unedited = np.zeros((11, 13), int)
        neighbors_unedited[5:7, 4:8] = 1
        neighbors_unedited[0:4, 4:8] = 2

        objects = np.zeros((11, 13), int)
        objects[1:6, 5:7] = 1

        workspace, module = self.make_workspace(
                objects, M.D_WITHIN, neighbors_labels=neighbors)
        no = workspace.object_set.get_objects(NEIGHBORS_NAME)
        no.unedited_segmented = neighbors_unedited
        no.small_removed_segmented = neighbors
        module.run(workspace)
        m = workspace.measurements
        v = m[OBJECTS_NAME,
              module.get_measurement_name(M.M_NUMBER_OF_NEIGHBORS), 1]
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0], 2)

    def test_08_03_object_is_missing(self):
        # regression test of #1639
        #
        # Object # 2 should match neighbor # 1, but because of
        # an error in masking distances, neighbor #1 is masked out
        #
        olabels = np.zeros((20, 10), int)
        olabels[2, 2] = 2
        nlabels = np.zeros((20, 10), int)
        nlabels[2, 3] = 1
        nlabels[5, 2] = 2
        workspace, module = self.make_workspace(
                olabels, M.D_EXPAND, distance=20, neighbors_labels=nlabels)
        self.assertTrue(isinstance(module, M.MeasureObjectNeighbors))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        ftr = module.get_measurement_name(M.M_FIRST_CLOSEST_OBJECT_NUMBER)
        values = m[OBJECTS_NAME, ftr]
        self.assertEqual(values[1], 1)

    def test_08_04_small_removed_same(self):
        # Regression test of issue #1672
        #
        # Objects with small removed failed.
        #
        objects = np.zeros((11, 13), int)
        objects[5:7, 1:3] = 1
        objects[6:8, 5:7] = 2
        objects_unedited = objects.copy()
        objects_unedited[0:2, 0:2] = 3

        workspace, module = self.make_workspace(
                objects, M.D_EXPAND, distance=1)
        no = workspace.object_set.get_objects(OBJECTS_NAME)
        no.unedited_segmented = objects_unedited
        no.small_removed_segmented = objects
        module.run(workspace)
        m = workspace.measurements
        v = m[OBJECTS_NAME,
              module.get_measurement_name(M.M_NUMBER_OF_NEIGHBORS), 1]
        self.assertEqual(len(v), 2)
        self.assertEqual(v[0], 1)
