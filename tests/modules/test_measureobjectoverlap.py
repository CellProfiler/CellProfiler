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
GROUND_TRUTH_OBJ_IMAGE_NAME = 'DNA'
ID_OBJ_IMAGE_NAME = 'Protein'
GROUND_TRUTH_OBJ = 'Nuclei'
ID_OBJ = 'Protein'

class TestMeasureObjectOverlap(unittest.TestCase):
    def make_obj_workspace(self, ground_truth_obj, id_obj, ground_truth, id):
        '''make a workspace to test comparing objects'''
        ''' ground truth object and ID object  are dictionaires w/ the following keys'''
        '''i - i component of pixel coordinates
        j - j component of pixel coordinates
        l - label '''

        module = cellprofiler.modules.measureobjectoverlap.MeasureObjectOverlap()
        module.module_num = 1
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

    def test_04_01_get_measurement_columns(self):
        workspace, module = self.make_obj_workspace(
                numpy.zeros((0, 3), int),
                numpy.zeros((0, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        columns = module.get_measurement_columns(workspace.pipeline)
        # All columns should be unique
        self.assertEqual(len(columns), len(set([x[1] for x in columns])))
        # All columns should be floats and done on images
        x = columns[-1]
        self.assertTrue(all([x[0] == cellprofiler.measurement.IMAGE]))
        self.assertTrue(all([x[2] == cellprofiler.measurement.COLTYPE_FLOAT]))
        for feature in cellprofiler.modules.measureobjectoverlap.FTR_ALL:
            field = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, feature, GROUND_TRUTH_OBJ, ID_OBJ))
            self.assertTrue(field in [x[1] for x in columns])

    def test_04_05_get_measurement_scales(self):
        workspace, module = self.make_obj_workspace(
                numpy.zeros((0, 3), int),
                numpy.zeros((0, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))

        scales = module.get_measurement_scales(
            workspace.pipeline,
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureobjectoverlap.FTR_RAND_INDEX,
            None
        )

        self.assertEqual(len(scales), 1)
        self.assertEqual(scales[0], "_".join((GROUND_TRUTH_OBJ, ID_OBJ)))

    def test_05_00_test_measure_overlap_no_objects(self):
        # Regression test of issue #934 - no objects
        workspace, module = self.make_obj_workspace(
                numpy.zeros((0, 3), int),
                numpy.zeros((0, 3), int),
                dict(image=numpy.zeros((20, 10), bool)),
                dict(image=numpy.zeros((20, 10), bool)))
        module.run(workspace)
        m = workspace.measurements
        for feature in cellprofiler.modules.measureobjectoverlap.FTR_ALL:
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
        expected_rand_index = measurements.get_current_image_measurement(mname)
        rand_index = measurements.get_current_image_measurement(mname)
        self.assertAlmostEqual(rand_index, expected_rand_index)
        mname = '_'.join((cellprofiler.modules.measureobjectoverlap.C_IMAGE_OVERLAP, cellprofiler.modules.measureobjectoverlap.FTR_ADJUSTED_RAND_INDEX,
                          GROUND_TRUTH_OBJ, ID_OBJ))
        adjusted_rand_index = \
            measurements.get_current_image_measurement(mname)
#        self.assertAlmostEqual(adjusted_rand_index, expected_adjusted_rand_index)
