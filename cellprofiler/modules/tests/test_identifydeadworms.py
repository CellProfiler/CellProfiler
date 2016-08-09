import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.identifydeadworms
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import centrosome.cpmorphology
import numpy
import scipy.ndimage

cellprofiler.preferences.set_headless()

IMAGE_NAME = "myimage"
OBJECTS_NAME = "myobjects"


class TestIdentifyDeadWorms(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = '''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10479

IdentifyDeadWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Input image:BinaryWorms
    Objects name:DeadWorms
    Worm width:6
    Worm length:114
    Number of angles:180
'''
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))
        self.assertEqual(module.image_name, "BinaryWorms")
        self.assertEqual(module.object_name, "DeadWorms")
        self.assertEqual(module.worm_width, 6)
        self.assertEqual(module.worm_length, 114)
        self.assertEqual(module.angle_count, 180)
        self.assertTrue(module.wants_automatic_distance)

    def test_01_01_load_v2(self):
        data = '''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10479

IdentifyDeadWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Input image:BinaryWorms
    Objects name:DeadWorms
    Worm width:6
    Worm length:114
    Number of angles:180
    Automatically calculate distance parameters?:No
    Spatial distance:6
    Angular distance:45
'''
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))
        self.assertEqual(module.image_name, "BinaryWorms")
        self.assertEqual(module.object_name, "DeadWorms")
        self.assertEqual(module.worm_width, 6)
        self.assertEqual(module.worm_length, 114)
        self.assertEqual(module.angle_count, 180)
        self.assertFalse(module.wants_automatic_distance)
        self.assertEqual(module.space_distance, 6)
        self.assertEqual(module.angular_distance, 45)

    def make_workspace(self, pixel_data, mask=None):
        image = cellprofiler.image.Image(pixel_data, mask)
        image_set_list = cellprofiler.image.ImageSetList()

        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, image)

        module = cellprofiler.modules.identifydeadworms.IdentifyDeadWorms()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.object_name.value = OBJECTS_NAME

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_02_01_zeros(self):
        """Run the module with an image of all zeros"""
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        module.run(workspace)
        count = workspace.measurements.get_current_image_measurement(
                '_'.join((cellprofiler.modules.identifydeadworms.I.C_COUNT, OBJECTS_NAME)))
        self.assertEqual(count, 0)

    def test_02_02_one_worm(self):
        """Find a single worm"""
        image = numpy.zeros((20, 20), bool)
        index, count, i, j = centrosome.cpmorphology.get_line_pts(
                numpy.array([1, 6, 19, 14]),
                numpy.array([5, 0, 13, 18]),
                numpy.array([6, 19, 14, 1]),
                numpy.array([0, 13, 18, 5]))
        image[i, j] = True
        image = scipy.ndimage.binary_fill_holes(image)
        workspace, module = self.make_workspace(image)
        module.worm_length.value = 12
        module.worm_width.value = 5
        module.angle_count.value = 16
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        count = m.get_current_image_measurement(
                '_'.join((cellprofiler.modules.identifydeadworms.I.C_COUNT, OBJECTS_NAME)))
        self.assertEqual(count, 1)
        x = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_X)
        self.assertEqual(len(x), 1)
        self.assertAlmostEqual(x[0], 9., 1)
        y = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_Y)
        self.assertEqual(len(y), 1)
        self.assertAlmostEqual(y[0], 10., 1)
        a = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.M_ANGLE)
        self.assertEqual(len(a), 1)
        self.assertAlmostEqual(a[0], 135, 0)

    def test_02_03_crossing_worms(self):
        """Find two worms that cross"""
        image = numpy.zeros((20, 20), bool)
        index, count, i, j = centrosome.cpmorphology.get_line_pts(
                numpy.array([1, 4, 19, 16]),
                numpy.array([3, 0, 15, 18]),
                numpy.array([4, 19, 16, 1]),
                numpy.array([0, 15, 18, 3]))
        image[i, j] = True
        index, count, i, j = centrosome.cpmorphology.get_line_pts(
                numpy.array([0, 3, 18, 15]),
                numpy.array([16, 19, 4, 1]),
                numpy.array([3, 18, 15, 0]),
                numpy.array([19, 4, 1, 16])
        )
        image[i, j] = True
        image = scipy.ndimage.binary_fill_holes(image)
        workspace, module = self.make_workspace(image)
        module.worm_length.value = 17
        module.worm_width.value = 5
        module.angle_count.value = 16
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        count = m.get_current_image_measurement(
                '_'.join((cellprofiler.modules.identifydeadworms.I.C_COUNT, OBJECTS_NAME)))
        self.assertEqual(count, 2)
        a = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.M_ANGLE)
        self.assertEqual(len(a), 2)
        if a[0] > 90:
            order = numpy.array([0, 1])
        else:
            order = numpy.array([1, 0])
        self.assertAlmostEqual(a[order[0]], 135, 0)
        self.assertAlmostEqual(a[order[1]], 45, 0)
        x = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_X)
        self.assertEqual(len(x), 2)
        self.assertAlmostEqual(x[order[0]], 9., 0)
        self.assertAlmostEqual(x[order[1]], 10., 0)
        y = m.get_current_measurement(OBJECTS_NAME,
                                      cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_Y)
        self.assertEqual(len(y), 2)
        self.assertAlmostEqual(y[order[0]], 10., 0)
        self.assertAlmostEqual(y[order[1]], 9., 0)

    def test_03_01_measurement_columns(self):
        """Test get_measurement_columns"""
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))
        columns = module.get_measurement_columns(workspace.pipeline)
        expected = (
            (OBJECTS_NAME, cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_X, cellprofiler.measurement.COLTYPE_INTEGER),
            (OBJECTS_NAME, cellprofiler.modules.identifydeadworms.I.M_LOCATION_CENTER_Y, cellprofiler.measurement.COLTYPE_INTEGER),
            (OBJECTS_NAME, cellprofiler.modules.identifydeadworms.M_ANGLE, cellprofiler.measurement.COLTYPE_FLOAT),
            (OBJECTS_NAME, cellprofiler.modules.identifydeadworms.I.M_NUMBER_OBJECT_NUMBER, cellprofiler.measurement.COLTYPE_INTEGER),
            (cellprofiler.measurement.IMAGE, cellprofiler.modules.identifydeadworms.I.FF_COUNT % OBJECTS_NAME, cellprofiler.measurement.COLTYPE_INTEGER))
        self.assertEqual(len(columns), len(expected))
        for e in expected:
            self.assertTrue(any(all([x == y for x, y in zip(c, e)])
                                for c in columns), "could not find " + repr(e))

    def test_04_01_find_adjacent_by_distance_empty(self):
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))

        first, second = module.find_adjacent_by_distance(numpy.zeros(0),
                                                         numpy.zeros(0),
                                                         numpy.zeros(0))
        self.assertEqual(len(first), 0)
        self.assertEqual(len(second), 0)

    def test_04_02_find_adjacent_by_distance_one(self):
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))

        first, second = module.find_adjacent_by_distance(numpy.zeros(1),
                                                         numpy.zeros(1),
                                                         numpy.zeros(1))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0], 0)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0], 0)

    def test_04_03_find_adjacent_by_distance_easy(self):
        #
        # Feed "find_adjacent_by_distance" points whose "i" are all
        # within the space_distance
        #
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))
        module.space_distance.value = 10
        # Take find_adjacent_by_distance internals into account: consecutive i
        # will create a single cross-product
        #
        i = numpy.arange(10)
        j = numpy.arange(10)
        # Break into two groups: 0-4 (5x5) and 5-9 (5x5)
        j[5:] += 10
        a = numpy.zeros(10)
        first, second = module.find_adjacent_by_distance(i, j, a)
        order = numpy.lexsort((second, first))
        first = first[order]
        second = second[order]
        self.assertEqual(len(first), 50)
        self.assertEqual(len(second), 50)
        for i in range(50):
            self.assertEqual(first[i], int(i / 5))
        for i in range(25):
            self.assertEqual(second[i], i % 5)
            self.assertEqual(second[i + 25], (i % 5) + 5)

    def test_04_04_find_adjacent_by_distance_hard(self):
        #
        # Feed "find_adjacent_by_distance" points whose "i" are not all
        # within the space_distance
        #
        workspace, module = self.make_workspace(numpy.zeros((20, 10), bool))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms))
        module.space_distance.value = 10
        r = numpy.random.RandomState(44)
        for idx, scramble in enumerate(
                        [numpy.arange(13)] + [r.permutation(numpy.arange(13)) for ii in range(10)]):
            # Take find_adjacent_by_distance internals into account: non consecutive i
            # will create two cross-products
            #
            i = numpy.arange(13)
            j = numpy.arange(13)
            # Break into three groups: 0-2 (3x3), 3-6 (4x4) and 7-11 (5x5)
            # with one loner at end
            i[3:] += 10
            i[7:] += 10
            #
            # Make last in last group not match by i+j
            #
            i[-1] += 7
            j[-1] += 8
            a = numpy.zeros(13)
            #
            # Scramble i, j and a
            #
            i = i[scramble]
            j = j[scramble]
            a = a[scramble]
            #
            # a reported value of "n" corresponds to whatever index in scramble
            # that contains "n"
            #
            unscramble = numpy.zeros(13, int)
            unscramble[scramble] = numpy.arange(13)
            first, second = module.find_adjacent_by_distance(i, j, a)
            self.assertEqual(len(first), 9 + 16 + 25 + 1)
            self.assertEqual(len(second), 9 + 16 + 25 + 1)
            for f, s in zip(first, second):
                self.assertTrue((i[f] - i[s]) ** 2 + (j[f] - j[s]) ** 2 <= 100)
