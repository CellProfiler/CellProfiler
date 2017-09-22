import StringIO
import unittest

import numpy
import numpy.random
import scipy.ndimage

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureimageoverlap
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

GROUND_TRUTH_IMAGE_NAME = 'groundtruth'
TEST_IMAGE_NAME = 'test'
O_IMG = 'Foreground/background segmentation'
O_OBJ = 'Segmented objects'
GROUND_TRUTH_OBJ_IMAGE_NAME = 'DNA'
ID_OBJ_IMAGE_NAME = 'Protein'
GROUND_TRUTH_OBJ = 'Nuclei'
ID_OBJ = 'Protein'


class TestMeasureImageOverlap(unittest.TestCase):
    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9169

MeasureImageOverlap:[module_num:1|svn_version:\'9000\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Which image do you want to use as the basis for calculating the amount of overlap? :GroundTruth
    Which image do you want to compare for overlap?:Segmentation
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        self.assertEqual(module.ground_truth, "GroundTruth")
        self.assertEqual(module.test_img, "Segmentation")

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20131210175632
GitHash:63ec479
ModuleCount:2
HasImagePlaneDetails:False

MeasureImageOverlap:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Compare segmented objects, or foreground/background?:Segmented objects
    Select the image to be used as the ground truth basis for calculating the amount of overlap:Bar
    Select the image to be used to test for overlap:Foo
    Select the objects to be used as the ground truth basis for calculating the amount of overlap:Nuclei2_0
    Select the objects to be tested for overlap against the ground truth:Nuclei2_1

MeasureImageOverlap:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Compare segmented objects, or foreground/background?:Foreground/background segmentation
    Select the image to be used as the ground truth basis for calculating the amount of overlap:Foo
    Select the image to be used to test for overlap:Bar
    Select the objects to be used as the ground truth basis for calculating the amount of overlap:Cell2_0
    Select the objects to be tested for overlap against the ground truth:Cell2_1
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureimageoverlap.O_OBJ)
        self.assertEqual(module.ground_truth, "Bar")
        self.assertEqual(module.test_img, "Foo")
        self.assertEqual(module.object_name_GT, "Nuclei2_0")
        self.assertEqual(module.object_name_ID, "Nuclei2_1")
        self.assertFalse(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureimageoverlap.DM_KMEANS)
        self.assertEqual(module.max_distance, 250)
        self.assertEqual(module.max_points, 250)
        self.assertFalse(module.penalize_missing)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureimageoverlap.O_IMG)
        self.assertEqual(module.ground_truth, "Foo")
        self.assertEqual(module.test_img, "Bar")
        self.assertEqual(module.object_name_GT, "Cell2_0")
        self.assertEqual(module.object_name_ID, "Cell2_1")
        self.assertFalse(module.wants_emd)

    def test_01_04_load_v4(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20141015195823
GitHash:051040e
ModuleCount:2
HasImagePlaneDetails:False

MeasureImageOverlap:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Compare segmented objects, or foreground/background?:Segmented objects
    Select the image to be used as the ground truth basis for calculating the amount of overlap:Bar
    Select the image to be used to test for overlap:Foo
    Select the objects to be used as the ground truth basis for calculating the amount of overlap:Nuclei2_0
    Select the objects to be tested for overlap against the ground truth:Nuclei2_1
    Calculate earth mover\'s distance?:No
    Maximum # of points:201
    Point selection method:K Means
    Maximum distance:202
    Penalize missing pixels:No

MeasureImageOverlap:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Compare segmented objects, or foreground/background?:Foreground/background segmentation
    Select the image to be used as the ground truth basis for calculating the amount of overlap:Foo
    Select the image to be used to test for overlap:Bar
    Select the objects to be used as the ground truth basis for calculating the amount of overlap:Cell2_0
    Select the objects to be tested for overlap against the ground truth:Cell2_1
    Calculate earth mover\'s distance?:Yes
    Maximum # of points:101
    Point selection method:Skeleton
    Maximum distance:102
    Penalize missing pixels:Yes
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureimageoverlap.O_OBJ)
        self.assertEqual(module.ground_truth, "Bar")
        self.assertEqual(module.test_img, "Foo")
        self.assertEqual(module.object_name_GT, "Nuclei2_0")
        self.assertEqual(module.object_name_ID, "Nuclei2_1")
        self.assertFalse(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureimageoverlap.DM_KMEANS)
        self.assertEqual(module.max_distance, 202)
        self.assertEqual(module.max_points, 201)
        self.assertFalse(module.penalize_missing)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureimageoverlap.O_IMG)
        self.assertEqual(module.ground_truth, "Foo")
        self.assertEqual(module.test_img, "Bar")
        self.assertEqual(module.object_name_GT, "Cell2_0")
        self.assertEqual(module.object_name_ID, "Cell2_1")
        self.assertTrue(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureimageoverlap.DM_SKEL)
        self.assertEqual(module.max_distance, 102)
        self.assertEqual(module.max_points, 101)
        self.assertTrue(module.penalize_missing)

    def make_workspace(self, ground_truth, test, dimensions=2):
        '''Make a workspace with a ground-truth image and a test image

        ground_truth and test are dictionaries with the following keys:
        image     - the pixel data
        mask      - (optional) the mask data
        crop_mask - (optional) a cropping mask

        returns a workspace and module
        '''
        module = cellprofiler.modules.measureimageoverlap.MeasureImageOverlap()
        module.module_num = 1
        module.obj_or_img.value = O_IMG
        module.ground_truth.value = GROUND_TRUTH_IMAGE_NAME
        module.test_img.value = TEST_IMAGE_NAME
        module.wants_emd.value = True

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        for name, d in ((GROUND_TRUTH_IMAGE_NAME, ground_truth),
                        (TEST_IMAGE_NAME, test)):
            image = cellprofiler.image.Image(
                d["image"],
                mask=d.get("mask"),
                crop_mask=d.get("crop_mask"),
                dimensions=dimensions
            )
            image_set.add(name, image)

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     cellprofiler.object.ObjectSet(), cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def make_obj_workspace(self, ground_truth_obj, id_obj, ground_truth, id):
        '''make a workspace to test comparing objects'''
        ''' ground truth object and ID object  are dictionaires w/ the following keys'''
        '''i - i component of pixel coordinates
        j - j component of pixel coordinates
        l - label '''

        module = cellprofiler.modules.measureimageoverlap.MeasureImageOverlap()
        module.module_num = 1
        module.obj_or_img.value = O_OBJ
        module.object_name_GT.value = GROUND_TRUTH_OBJ
        module.object_name_ID.value = ID_OBJ
        module.wants_emd.value = True
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        for name, d in ((GROUND_TRUTH_OBJ_IMAGE_NAME, ground_truth),
                        (ID_OBJ_IMAGE_NAME, id)):
            image = cellprofiler.image.Image(d["image"],
                                             mask=d.get("mask"),
                                             crop_mask=d.get("crop_mask"))
            image_set.add(name, image)
        object_set = cellprofiler.object.ObjectSet()
        for name, d in ((GROUND_TRUTH_OBJ, ground_truth_obj),
                        (ID_OBJ, id_obj)):
            object = cellprofiler.object.Objects()
            if d.shape[1] == 3:
                object.ijv = d
            else:
                object.segmented = d
            object_set.add_objects(object, name)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     object_set, cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_03_01_zeros(self):
        '''Test ground-truth of zeros and image of zeros'''

        workspace, module = self.make_workspace(
                dict(image=numpy.ones((20, 10), bool)),
                dict(image=numpy.ones((20, 10), bool)))

        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        self.assertEqual(
                measurements.get_current_image_measurement("Overlap_FalseNegRate_test"),
                0)
        features = measurements.get_feature_names(cellprofiler.measurement.IMAGE)
        for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL + [cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE]:
            field = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            self.assertTrue(field in features,
                            "Missing feature: %s" % feature)
        ftr_emd = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)
        self.assertEqual(measurements[cellprofiler.measurement.IMAGE, ftr_emd], 0)

    def test_03_02_ones(self):
        '''Test ground-truth of ones and image of ones'''

        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
                          TEST_IMAGE_NAME))
        self.assertTrue(numpy.isnan(measurements.get_current_image_measurement(
                mname)))

    def test_03_03_masked(self):
        '''Test ground-truth of a masked image'''

        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool),
                     mask=numpy.zeros((20, 10), bool)))

        self.assertTrue(isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)
        for feature in (cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertTrue(numpy.isnan(value))

    def test_03_04_all_right(self):
        numpy.random.seed(34)
        image = numpy.random.uniform(size=(10, 20)) > .5
        workspace, module = self.make_workspace(
                dict(image=image), dict(image=image))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)

    def test_03_05_one_false_positive(self):
        i, j = numpy.mgrid[0:10, 0:20]
        ground_truth = ((i + j) % 2) == 0
        test = ground_truth.copy()
        test[0, 1] = True
        workspace, module = self.make_workspace(
                dict(image=ground_truth), dict(image=test))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 100.0 / 101.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.01),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.99),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
                                  (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong" % feature)

    def test_03_05_one_false_negative(self):
        i, j = numpy.mgrid[0:10, 0:20]
        ground_truth = ((i + j) % 2) == 0
        test = ground_truth.copy()
        test[0, 0] = False
        workspace, module = self.make_workspace(
                dict(image=ground_truth), dict(image=test))
        module.run(workspace)
        measurements = workspace.measurements
        recall = 0.99
        f_factor = 2 * recall / (1 + recall)
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0.01),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 0.99),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, recall),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
                                  (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong" % feature)

    def test_03_06_one_false_positive_and_mask(self):
        i, j = numpy.mgrid[0:10, 0:20]
        ground_truth = ((i + j) % 2) == 0
        test = ground_truth.copy()
        test[0, 1] = True
        mask = j < 10
        workspace, module = self.make_workspace(
                dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 50.0 / 51.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.02),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.98),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong" % feature)

    def test_03_07_one_false_negative_and_mask(self):
        i, j = numpy.mgrid[0:10, 0:20]
        ground_truth = ((i + j) % 2) == 0
        test = ground_truth.copy()
        test[0, 0] = False
        mask = j < 10
        workspace, module = self.make_workspace(
                dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        recall = 0.98
        f_factor = 2 * recall / (1 + recall)
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0.02),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 0.98),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, recall),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong" % feature)

    def test_03_08_masked_errors(self):
        numpy.random.seed(38)
        ground_truth = numpy.random.uniform(size=(20, 10)) > .5
        test = ground_truth.copy()
        mask = numpy.random.uniform(size=(20, 10)) > .5
        test[~ mask] = numpy.random.uniform(size=numpy.sum(~ mask)) > .5
        workspace, module = self.make_workspace(
                dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong" % feature)

    def test_03_09_cropped(self):
        numpy.random.seed(39)
        i, j = numpy.mgrid[0:10, 0:20]
        ground_truth = ((i + j) % 2) == 0
        test = ground_truth.copy()
        test[0, 1] = True
        cropping = numpy.zeros((20, 40), bool)
        cropping[10:20, 10:30] = True
        big_ground_truth = numpy.random.uniform(size=(20, 40)) > .5
        big_ground_truth[10:20, 10:30] = ground_truth
        workspace, module = self.make_workspace(
                dict(image=big_ground_truth),
                dict(image=test, crop_mask=cropping))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 100.0 / 101.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.01),
                                  (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.99),
                                  (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
                                  (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
                                  (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value,
                                   msg="%s is wrong. Expected %f, got %f" %
                                       (feature, expected, value))

    def test_03_10_rand_index(self):
        numpy.random.seed(310)
        i, j = numpy.mgrid[0:10, 0:20]
        #
        # Create a labeling with two objects 0:10, 0:5 and 0:10, 15:20.
        # The background class is 0:10, 5:15
        #
        ground_truth = (j < 5) | (j >= 15)
        #
        # Add a 3x4 square in the middle
        #
        test = ground_truth.copy()
        test[4:7, 8:12] = True
        #
        # I used R to generate the rand index and adjusted rand index
        # of the two segmentations: a 10 x 5 rectangle, a 10x10 background
        # and a 10x5 rectangle with 12 pixels that disagree in the middle
        #
        # The rand index is from rand.index in the fossil package and
        # the adjusted rand index is from cluster.stats in the fpc package.
        # There's an adjusted rand index in the fossil package but it gives
        # the wrong numbers (!!!!)
        #
        expected_rand_index = 0.9469347
        expected_adj_rand_index = 0.8830027
        workspace, module = self.make_workspace(
                dict(image=ground_truth),
                dict(image=test))
        module.run(workspace)
        measurements = workspace.measurements
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_rand_index, 6)
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
                          TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_adj_rand_index, 6)

    def test_03_11_masked_rand_index(self):
        numpy.random.seed(310)
        i, j = numpy.mgrid[0:10, 0:20]
        #
        # Create a labeling with two objects 0:10, 0:5 and 0:10, 15:20.
        # The background class is 0:10, 5:15
        #
        ground_truth = (j < 5) | (j >= 15)
        #
        # Add a 3x4 square in the middle
        #
        test = ground_truth.copy()
        test[4:7, 8:12] = True
        #
        # Remove both one correct and one incorect pixel
        #
        mask = numpy.ones(ground_truth.shape, bool)
        mask[4, 4] = False
        mask[5, 9] = False
        #
        # See notes from 03_10
        #
        expected_rand_index = 0.9503666
        expected_adj_rand_index = 0.8907784
        workspace, module = self.make_workspace(
                dict(image=ground_truth, mask=mask),
                dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements

        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_rand_index, 6)
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
                          TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_adj_rand_index, 6)

    def test_04_01_get_measurement_columns(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        assert isinstance(module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap)
        module.object_name_GT.value = GROUND_TRUTH_OBJ
        module.object_name_ID.value = ID_OBJ
        for obj_or_img, name in ((O_IMG, TEST_IMAGE_NAME),
                                 (O_OBJ, "_".join((GROUND_TRUTH_OBJ, ID_OBJ)))):
            module.obj_or_img.value = obj_or_img
            columns = module.get_measurement_columns(workspace.pipeline)
            # All columns should be unique
            self.assertEqual(len(columns), len(set([x[1] for x in columns])))
            # All columns should be floats and done on images
            self.assertTrue(all([x[0] == cellprofiler.measurement.IMAGE]))
            self.assertTrue(all([x[2] == cellprofiler.measurement.COLTYPE_FLOAT]))
            for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL:
                field = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, name))
                self.assertTrue(field in [x[1] for x in columns])

    def test_04_02_get_categories(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        categories = module.get_categories(workspace.pipeline, "Foo")
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(workspace.pipeline, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP)

    def test_04_03_get_measurements(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        for wants_emd, features in (
                (True, list(cellprofiler.modules.measureimageoverlap.FTR_ALL) + [cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE]),
                (False, cellprofiler.modules.measureimageoverlap.FTR_ALL)):
            module.wants_emd.value = wants_emd
            mnames = module.get_measurements(workspace.pipeline,
                                             cellprofiler.measurement.IMAGE, cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP)
            self.assertEqual(len(mnames), len(features))
            self.assertTrue(all(n in features for n in mnames))
            self.assertTrue(all(f in mnames for f in features))
            mnames = module.get_measurements(workspace.pipeline, "Foo",
                                             cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP)
            self.assertEqual(len(mnames), 0)
            mnames = module.get_measurements(workspace.pipeline, cellprofiler.measurement.IMAGE,
                                             "Foo")
            self.assertEqual(len(mnames), 0)

    def test_04_04_get_measurement_images(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL:
            imnames = module.get_measurement_images(workspace.pipeline,
                                                    cellprofiler.measurement.IMAGE,
                                                    cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                                                    feature)
            self.assertEqual(len(imnames), 1)
            self.assertEqual(imnames[0], TEST_IMAGE_NAME)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                cellprofiler.measurement.IMAGE,
                                                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                                                "Foo")
        self.assertEqual(len(imnames), 0)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                cellprofiler.measurement.IMAGE,
                                                "Foo",
                                                cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE)
        self.assertEqual(len(imnames), 0)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                "Foo",
                                                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                                                cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE)
        self.assertEqual(len(imnames), 0)

    def test_04_05_get_measurement_scales(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.obj_or_img.value = cellprofiler.modules.measureimageoverlap.O_OBJ
        module.object_name_GT.value = GROUND_TRUTH_OBJ
        module.object_name_ID.value = ID_OBJ

        scales = module.get_measurement_scales(
                workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, None)
        self.assertEqual(len(scales), 1)
        self.assertEqual(scales[0], "_".join((GROUND_TRUTH_OBJ, ID_OBJ)))

        module.obj_or_img.value = cellprofiler.modules.measureimageoverlap.O_IMG
        scales = module.get_measurement_scales(
                workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, None)
        self.assertEqual(len(scales), 0)

    def test_05_00_test_measure_overlap_no_objects(self):
        # Regression test of issue #934 - no objects
        workspace, module = self.make_obj_workspace(
                numpy.zeros((0, 3), int),
                numpy.zeros((0, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.run(workspace)
        m = workspace.measurements
        for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL:
            mname = module.measurement_name(feature)
            value = m[cellprofiler.measurement.IMAGE, mname, 1]
            if feature == cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE:
                self.assertEqual(value, 1)
            elif feature == cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE:
                self.assertEqual(value, 0)
            else:
                self.assertTrue(
                        numpy.isnan(value), msg="%s was %f. not nan" % (mname, value))
        #
        # Make sure they don't crash
        #
        workspace, module = self.make_obj_workspace(
                numpy.zeros((0, 3), int),
                numpy.ones((1, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.run(workspace)
        workspace, module = self.make_obj_workspace(
                numpy.ones((1, 3), int),
                numpy.zeros((0, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.run(workspace)

    def test_05_01_test_measure_overlap_objects(self):
        r = numpy.random.RandomState()
        r.seed(51)

        workspace, module = self.make_obj_workspace(
                numpy.column_stack([r.randint(0, 20, 150),
                                    r.randint(0, 10, 150),
                                    r.randint(1, 5, 150)]),
                numpy.column_stack([r.randint(0, 20, 175),
                                    r.randint(0, 10, 175),
                                    r.randint(1, 5, 175)]),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.wants_emd.value = False
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))

    def test_05_02_test_objects_rand_index(self):
        r = numpy.random.RandomState()
        r.seed(52)
        base = numpy.zeros((100, 100), bool)
        base[r.randint(0, 100, size=10),
             r.randint(0, 100, size=10)] = True
        gt = base.copy()
        gt[r.randint(0, 100, size=5),
           r.randint(0, 100, size=5)] = True
        test = base.copy()
        test[r.randint(0, 100, size=5),
             r.randint(0, 100, size=5)] = True
        gt = scipy.ndimage.binary_dilation(gt, numpy.ones((5, 5), bool))
        test = scipy.ndimage.binary_dilation(test, numpy.ones((5, 5), bool))
        workspace, module = self.make_workspace(
                dict(image=gt),
                dict(image=test))
        module.wants_emd.value = False
        module.run(workspace)

        measurements = workspace.measurements
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
                          TEST_IMAGE_NAME))
        expected_rand_index = measurements.get_current_image_measurement(mname)
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
                          TEST_IMAGE_NAME))
        expected_adjusted_rand_index = \
            measurements.get_current_image_measurement(mname)

        gt_labels, _ = scipy.ndimage.label(gt, numpy.ones((3, 3), bool))
        test_labels, _ = scipy.ndimage.label(test, numpy.ones((3, 3), bool))

        workspace, module = self.make_obj_workspace(
                gt_labels, test_labels,
                dict(image=numpy.ones(gt_labels.shape)),
                dict(image=numpy.ones(test_labels.shape)))
        module.run(workspace)
        measurements = workspace.measurements
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
                          GROUND_TRUTH_OBJ, ID_OBJ))
        rand_index = measurements.get_current_image_measurement(mname)
        self.assertAlmostEqual(rand_index, expected_rand_index)
        mname = '_'.join((cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
                          GROUND_TRUTH_OBJ, ID_OBJ))
        adjusted_rand_index = \
            measurements.get_current_image_measurement(mname)
        self.assertAlmostEqual(adjusted_rand_index, expected_adjusted_rand_index)

    def test_06_00_no_emd(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.ones((20, 10), bool)),
                dict(image=numpy.ones((20, 10), bool)))
        module.wants_emd.value = False
        module.run(workspace)
        self.assertFalse(workspace.measurements.has_feature(
                cellprofiler.measurement.IMAGE,
                module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)))

    def test_06_01_one_pixel(self):
        #
        # The earth movers distance should be sqrt((8-5)**2 + (7 - 3) ** 2) = 5
        #
        src = numpy.zeros((20, 10), bool)
        dest = numpy.zeros((20, 10), bool)
        src[5, 3] = True
        dest[8, 7] = True
        workspace, module = self.make_workspace(
                dict(image=src), dict(image=dest))
        module.run(workspace)
        self.assertEqual(workspace.measurements[
                             cellprofiler.measurement.IMAGE,
                             module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)], 5)

    def test_06_02_missing_penalty(self):
        #
        # Test that the missing penalty works
        #
        src = numpy.zeros((20, 10), bool)
        dest = numpy.zeros((20, 10), bool)
        src[2, 2] = True
        dest[2, 2] = True
        dest[8, 7] = True
        dest[2, 6] = True
        workspace, module = self.make_workspace(
                dict(image=src), dict(image=dest))
        module.penalize_missing.value = True
        module.max_distance.value = 8
        module.run(workspace)
        self.assertEqual(workspace.measurements[
                             cellprofiler.measurement.IMAGE,
                             module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)], 16)

    def test_06_03_max_distance(self):
        src = numpy.zeros((20, 10), bool)
        dest = numpy.zeros((20, 10), bool)
        src[5, 3] = True
        dest[8, 7] = True
        src[19, 9] = True
        dest[11, 9] = True
        workspace, module = self.make_workspace(
                dict(image=src), dict(image=dest))
        module.max_distance.value = 6
        module.run(workspace)
        self.assertEqual(workspace.measurements[
                             cellprofiler.measurement.IMAGE,
                             module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)], 11)

    def test_06_04_decimate_k_means(self):
        r = numpy.random.RandomState()
        r.seed(64)
        img = r.uniform(size=(10, 10)) > .5
        workspace, module = self.make_workspace(
                dict(image=img), dict(image=img.transpose()))
        #
        # Pick a single point for decimation - the emd should be zero
        #
        module.max_points._Number__minval = 1
        module.max_points.value = 1
        module.run(workspace)
        self.assertEqual(workspace.measurements[
                             cellprofiler.measurement.IMAGE,
                             module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)], 0)
        #
        # Pick a large number of points to get the real EMD
        #
        workspace, module = self.make_workspace(
                dict(image=img), dict(image=img.transpose()))
        module.max_points._Number__minval = 1
        module.max_points.value = 100
        module.run(workspace)
        emd = workspace.measurements[
            cellprofiler.measurement.IMAGE,
            module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)]
        #
        # The EMD after decimation is going to be randomly different,
        # but not by much.
        #
        workspace, module = self.make_workspace(
                dict(image=img), dict(image=img.transpose()))
        module.max_points._Number__minval = 1
        module.max_points.value = numpy.sum(img | img.transpose()) / 2
        module.run(workspace)
        decimated_emd = workspace.measurements[
            cellprofiler.measurement.IMAGE,
            module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)]
        self.assertLess(decimated_emd, emd * 2)
        self.assertGreater(decimated_emd, emd / 2)

    def test_06_05_decimate_skel(self):
        #
        # Mostly, this is to check that the skeleton method doesn't crash
        #
        i, j = numpy.mgrid[0:10, 0:20]
        image1 = ((i - 4) ** 2) * 4 + (j - 8) ** 2 < 32
        image2 = ((i - 6) ** 2) * 4 + (j - 12) ** 2 < 32
        workspace, module = self.make_workspace(
                dict(image=image1), dict(image=image2))
        module.max_points._Number__minval = 1
        module.max_points.value = 5
        module.decimation_method.value = cellprofiler.modules.measureimageoverlap.DM_SKEL
        module.run(workspace)
        emd = workspace.measurements[
            cellprofiler.measurement.IMAGE,
            module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE)]
        self.assertGreater(emd, numpy.sum(image1) * 3)
        self.assertLess(emd, numpy.sum(image1) * 6)

    def test_3D_perfect_overlap(self):  # All tests start with test_, what follows is usually a short description of the test.
        # Define an image
        image_data = numpy.random.rand(10, 100, 100) >= 0.5

        workspace, module = self.make_workspace(
            ground_truth={
                "image": image_data
            },
            test={
                "image": image_data
            },
            dimensions=3
        )

        module.run(workspace)

        # Assuming that succeeds we make assertions.
        measurements = workspace.measurements

        # Test of all FTRs (perfect overlap)
        ftr_precision = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_PRECISION)
        ftr_precision_measurement = measurements.get_current_image_measurement(ftr_precision)
        assert ftr_precision_measurement == 1.0

        ftr_true_pos_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE)
        ftr_true_pos_rate_measurement = measurements.get_current_image_measurement(ftr_true_pos_rate)
        assert ftr_true_pos_rate_measurement == 1.0

        ftr_false_pos_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE)
        ftr_false_pos_rate_measurement = measurements.get_current_image_measurement(ftr_false_pos_rate)
        assert ftr_false_pos_rate_measurement == 0

        ftr_true_neg_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE)
        ftr_true_neg_rate_measurement = measurements.get_current_image_measurement(ftr_true_neg_rate)
        assert ftr_true_neg_rate_measurement == 1.0

        ftr_false_neg_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE)
        ftr_false_neg_rate_measurement = measurements.get_current_image_measurement(ftr_false_neg_rate)
        assert ftr_false_neg_rate_measurement == 0

        ftr_rand_index = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX)
        ftr_rand_index_measurement = measurements.get_current_image_measurement(ftr_rand_index)
        assert ftr_rand_index_measurement == 1.0

        # TODO: Test non/partially overlapping images
        # TODO: Test 100% overlapping objects
        # TODO: Test non/partially overlapping objects/different numbers of objects

    def test_3D_half_overlap_objects(self):
        ground_truth_image_data = numpy.zeros((10, 100, 100), dtype=numpy.uint8)
        ground_truth_image_data[2:6, 30:40, 30:40] = 1
        ground_truth_image_data[8, 10:20, 10:20] = 1
        ground_truth_image_data[7:10, 50:60, 50:60] = 1

        test_image_data = numpy.zeros((10, 100, 100), dtype=numpy.uint8)
        test_image_data[2:6, 30:40, 35:45] = 1
        test_image_data[8, 15:25, 10:20] = 1
        test_image_data[5:8, 50:60, 50:60] = 1

        workspace, module = self.make_workspace(
            ground_truth={
                "image": ground_truth_image_data
            },
            test={
                "image": test_image_data
            },
            dimensions=3
        )

        module.run(workspace)

        # Assuming that succeeds we make assertions.
        measurements = workspace.measurements

        # Test of all FTRs (perfect overlap)
        ftr_precision = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_PRECISION)
        ftr_precision_measurement = measurements.get_current_image_measurement(ftr_precision)
        assert ftr_precision_measurement == 0.5

        ftr_true_pos_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE)
        ftr_true_pos_rate_measurement = measurements.get_current_image_measurement(ftr_true_pos_rate)
        assert ftr_true_pos_rate_measurement == 0.5

        ftr_false_pos_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE)
        ftr_false_pos_rate_measurement = measurements.get_current_image_measurement(ftr_false_pos_rate)
        assert ftr_false_pos_rate_measurement == 0.5

        ftr_true_neg_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE)
        ftr_true_neg_rate_measurement = measurements.get_current_image_measurement(ftr_true_neg_rate)
        assert ftr_true_neg_rate_measurement == 0.5

        ftr_false_neg_rate = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE)
        ftr_false_neg_rate_measurement = measurements.get_current_image_measurement(ftr_false_neg_rate)
        assert ftr_false_neg_rate_measurement == 0.5

        ftr_rand_index = module.measurement_name(cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX)
        ftr_rand_index_measurement = measurements.get_current_image_measurement(ftr_rand_index)
        assert ftr_rand_index_measurement == 0.5
