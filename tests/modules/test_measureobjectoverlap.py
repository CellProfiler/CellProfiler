import StringIO
import unittest

import numpy
import numpy.random
import scipy.ndimage

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureobjectoverlap
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
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap))
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
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureobjectoverlap.O_OBJ)
        self.assertEqual(module.ground_truth, "Bar")
        self.assertEqual(module.test_img, "Foo")
        self.assertEqual(module.object_name_GT, "Nuclei2_0")
        self.assertEqual(module.object_name_ID, "Nuclei2_1")
        self.assertFalse(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureobjectoverlap.DM_KMEANS)
        self.assertEqual(module.max_distance, 250)
        self.assertEqual(module.max_points, 250)
        self.assertFalse(module.penalize_missing)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureobjectoverlap.O_IMG)
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
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureobjectoverlap.O_OBJ)
        self.assertEqual(module.ground_truth, "Bar")
        self.assertEqual(module.test_img, "Foo")
        self.assertEqual(module.object_name_GT, "Nuclei2_0")
        self.assertEqual(module.object_name_ID, "Nuclei2_1")
        self.assertFalse(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureobjectoverlap.DM_KMEANS)
        self.assertEqual(module.max_distance, 202)
        self.assertEqual(module.max_points, 201)
        self.assertFalse(module.penalize_missing)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap))
        self.assertEqual(module.obj_or_img, cellprofiler.modules.measureobjectoverlap.O_IMG)
        self.assertEqual(module.ground_truth, "Foo")
        self.assertEqual(module.test_img, "Bar")
        self.assertEqual(module.object_name_GT, "Cell2_0")
        self.assertEqual(module.object_name_ID, "Cell2_1")
        self.assertTrue(module.wants_emd)
        self.assertEqual(module.decimation_method, cellprofiler.modules.measureobjectoverlap.DM_SKEL)
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
        module = cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap()
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

        module = cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap()
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
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX, TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_rand_index, 6)
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_ADJUSTED_RAND_INDEX,
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

        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX, TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_rand_index, 6)
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_ADJUSTED_RAND_INDEX,
                          TEST_IMAGE_NAME))
        self.assertAlmostEqual(
                measurements.get_current_image_measurement(mname),
                expected_adj_rand_index, 6)

    def test_04_01_get_measurement_columns(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        assert isinstance(module, cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap)
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
            for feature in cellprofiler.modules.measureobjectoverlap.FTR_ALL:
                field = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, feature, name))
                self.assertTrue(field in [x[1] for x in columns])

    def test_04_05_get_measurement_scales(self):
        workspace, module = self.make_workspace(
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.obj_or_img.value = cellprofiler.modules.measureobjectoverlap.O_OBJ
        module.object_name_GT.value = GROUND_TRUTH_OBJ
        module.object_name_ID.value = ID_OBJ

        scales = module.get_measurement_scales(
                workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP,
                cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX, None)
        self.assertEqual(len(scales), 1)
        self.assertEqual(scales[0], "_".join((GROUND_TRUTH_OBJ, ID_OBJ)))

        module.obj_or_img.value = cellprofiler.modules.measureobjectoverlap.O_IMG
        scales = module.get_measurement_scales(
                workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP,
                cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX, None)
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
        for feature in cellprofiler.modules.measurobjectoverlap.FTR_ALL:
            mname = module.measurement_name(feature)
            value = m[cellprofiler.measurement.IMAGE, mname, 1]
            if feature == cellprofiler.modules.measureobjectoverlap.FTR_TRUE_NEG_RATE:
                self.assertEqual(value, 1)
            elif feature == cellprofiler.modules.measureobjectoverlap.FTR_FALSE_POS_RATE:
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

        gt_labels, _ = scipy.ndimage.label(gt, numpy.ones((3, 3), bool))
        test_labels, _ = scipy.ndimage.label(test, numpy.ones((3, 3), bool))

        workspace, module = self.make_obj_workspace(
                gt_labels, test_labels,
                dict(image=numpy.ones(gt_labels.shape)),
                dict(image=numpy.ones(test_labels.shape)))
        module.run(workspace)

        measurements = workspace.measurements
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX,
                          GROUND_TRUTH_OBJ, ID_OBJ))

        rand_index = measurements.get_current_image_measurement(mname)
        self.assertAlmostEqual(rand_index, expected_rand_index)
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_ADJUSTED_RAND_INDEX,
                          GROUND_TRUTH_OBJ, ID_OBJ))
        adjusted_rand_index = \
            measurements.get_current_image_measurement(mname)
        self.assertAlmostEqual(adjusted_rand_index, expected_adjusted_rand_index)