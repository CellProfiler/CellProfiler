import StringIO
import base64
import os
import unittest
import zlib

import centrosome.threshold
import numpy
import scipy.ndimage

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.applythreshold
import cellprofiler.modules.identify
import cellprofiler.modules.identifyprimaryobjects
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.workspace
import tests.modules

IMAGE_NAME = "my_image"
OBJECTS_NAME = "my_objects"
MASKING_OBJECTS_NAME = "masking_objects"
MEASUREMENT_NAME = "my_measurement"


class TestIdentifyPrimaryObjects(unittest.TestCase):
    def load_error_handler(self, caller, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.fail(event.error.message)

    def make_workspace(self, image,
                       mask=None,
                       labels=None):
        '''Make a workspace and IdentifyPrimaryObjects module

        image - the intensity image for thresholding

        mask - if present, the "don't analyze" mask of the intensity image

        labels - if thresholding per-object, the labels matrix needed
        '''
        module = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        module.module_num = 1
        module.x_name.value = IMAGE_NAME
        module.y_name.value = OBJECTS_NAME

        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        m = cellprofiler.measurement.Measurements()
        cpimage = cellprofiler.image.Image(image, mask=mask)
        m.add(IMAGE_NAME, cpimage)
        object_set = cellprofiler.object.ObjectSet()
        if labels is not None:
            o = cellprofiler.object.Objects()
            o.segmented = labels
            object_set.add_objects(o, MASKING_OBJECTS_NAME)
        workspace = cellprofiler.workspace.Workspace(
                pipeline, module, m, object_set, m, None)
        return workspace, module

    def test_00_00_init(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()

    def test_02_000_test_zero_objects(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .1
        x.apply_threshold.threshold_range.max = 1
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        img = numpy.zeros((25, 25))
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image", "Count_my_object")
        self.assertEqual(count, 0)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 0)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
        self.assertTrue(isinstance(location_center_y, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape), 0)

    def test_02_001_test_zero_objects_wa_in_lo_in(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .1
        x.apply_threshold.threshold_range.max = 1
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        img = numpy.zeros((25, 25))
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))

    def test_02_002_test_zero_objects_wa_di_lo_in(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .1
        x.apply_threshold.threshold_range.max = 1
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        img = numpy.zeros((25, 25))
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))

    def test_02_003_test_zero_objects_wa_in_lo_sh(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .1
        x.apply_threshold.threshold_range.max = 1
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
        img = numpy.zeros((25, 25))
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))

    def test_02_004_test_zero_objects_wa_di_lo_sh(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .1
        x.apply_threshold.threshold_range.max = 1
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
        img = numpy.zeros((25, 25))
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))

    def test_02_01_test_one_object(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = one_cell_image()
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented[img > 0] == 1))
        self.assertTrue(numpy.all(img[segmented == 1] > 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image", "Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .5)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image", "Count_my_object")
        self.assertEqual(count, 1)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
        self.assertTrue(isinstance(location_center_y, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape), 1)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 1)
        self.assertTrue(location_center_x[0] > 13)
        self.assertTrue(location_center_x[0] < 16)
        columns = x.get_measurement_columns(pipeline)
        for object_name in (cellprofiler.measurement.IMAGE, "my_object"):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_02_02_test_two_objects(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = two_cell_image()
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image", "Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .6)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image", "Count_my_object")
        self.assertEqual(count, 2)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
        self.assertTrue(isinstance(location_center_y, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape), 2)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue(location_center_y[1] > 28)
        self.assertTrue(location_center_y[1] < 32)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 2)
        self.assertTrue(location_center_x[0] > 33)
        self.assertTrue(location_center_x[0] < 37)
        self.assertTrue(location_center_x[1] > 13)
        self.assertTrue(location_center_x[1] < 16)

    def test_02_03_test_threshold_range(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .7
        x.apply_threshold.threshold_range.max = 1
        x.apply_threshold.threshold_correction_factor.value = .95
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_LI
        x.exclude_size.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        img = two_cell_image()
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image", "Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .8)
        self.assertTrue(threshold > .6)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image", "Count_my_object")
        self.assertEqual(count, 1)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
        self.assertTrue(isinstance(location_center_y, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape), 1)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 1)
        self.assertTrue(location_center_x[0] > 33)
        self.assertTrue(location_center_x[0] < 36)

    def test_02_04_fill_holes(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.fill_holes.value = True
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.zeros((40, 40))
        draw_circle(img, (10, 10), 7, .5)
        draw_circle(img, (30, 30), 7, .5)
        img[10, 10] = 0
        img[30, 30] = 0
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10, 10] > 0)
        self.assertTrue(objects.segmented[30, 30] > 0)

    def test_02_05_dont_fill_holes(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.apply_threshold.threshold_range.min = .7
        x.apply_threshold.threshold_range.max = 1
        x.exclude_size.value = False
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        img = numpy.zeros((40, 40))
        draw_circle(img, (10, 10), 7, .5)
        draw_circle(img, (30, 30), 7, .5)
        img[10, 10] = 0
        img[30, 30] = 0
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10, 10] == 0)
        self.assertTrue(objects.segmented[30, 30] == 0)

    def test_02_05_01_fill_holes_within_holes(self):
        'Regression test of img-1431'
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.size_range.min = 1
        x.size_range.max = 2
        x.exclude_size.value = False
        x.fill_holes.value = cellprofiler.modules.identifyprimaryobjects.FH_DECLUMP
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.zeros((40, 40))
        draw_circle(img, (20, 20), 10, .5)
        draw_circle(img, (20, 20), 4, 0)
        img[20, 20] = 1
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[20, 20] == 1)
        self.assertTrue(objects.segmented[22, 20] == 1)
        self.assertTrue(objects.segmented[26, 20] == 1)

    def test_02_06_test_watershed_shape_shape(self):
        """Identify by local_maxima:shape & intensity:shape

        Create an object whose intensity is high near the middle
        but has an hourglass shape, then segment it using shape/shape
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .6, .6, .6, .6, .6, .6, .6, .6, .6, .6, 0, 0, 0],
                           [0, 0, 0, 0, .7, .7, .7, .7, .7, .7, .7, .7, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .8, .9, 1, 1, .9, .8, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .7, .7, .7, .7, .7, .7, .7, .7, 0, 0, 0, 0],
                           [0, 0, 0, .6, .6, .6, .6, .6, .6, .6, .6, .6, .6, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 2)

    def test_02_07_test_watershed_shape_intensity(self):
        """Identify by local_maxima:shape & watershed:intensity

        Create an object with an hourglass shape to get two maxima, but
        set the intensities so that one maximum gets the middle portion
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0],
                           [0, 0, 0, .4, .4, .4, .5, .5, .5, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, 0, 0, 0, .4, .4, .4, .4, .4, .4, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 2)
        self.assertEqual(objects.segmented[7, 11], objects.segmented[7, 4])

    def test_02_08_test_watershed_intensity_distance_single(self):
        """Identify by local_maxima:intensity & watershed:shape - one object

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (4, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .7, .8, .9, .9, .8, .7, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .25, mode='constant')
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 1)

    def test_02_08_test_watershed_intensity_distance_triple(self):
        """Identify by local_maxima:intensity & watershed:shape - 3 objects w/o filter

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.maxima_suppression_size.value = 7.1
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .8, .8, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .7, .8, .9, .9, .8, .7, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .8, .8, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 3)

    def test_02_09_test_watershed_intensity_distance_filter(self):
        """Identify by local_maxima:intensity & watershed:shape - filtered

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 1
        x.automatic_smoothing.value = 1
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .7, .8, .9, .9, .8, .7, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .6, .7, .8, .8, .7, .6, .5, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .6, .7, .7, .6, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .6, .6, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 1)

    def test_02_10_test_watershed_intensity_distance_double(self):
        """Identify by local_maxima:intensity & watershed:shape - two objects

        Create an object with an hourglass shape and peaks in the top and
        bottom, but with a distribution of values that's skewed so that,
        by intensity, one of the images occupies the middle. The middle
        should be shared because the watershed is done using the distance
        transform.
        """
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .9, .9, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0],
                           [0, 0, 0, .4, .4, .4, .5, .5, .5, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .9, .9, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, 0, 0, 0, .4, .4, .4, .4, .4, .4, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .5, mode='constant')
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 2)
        self.assertNotEqual(objects.segmented[12, 7], objects.segmented[4, 7])

    def test_02_11_propagate(self):
        """Test the propagate unclump method"""
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 7
        x.automatic_suppression.value = False
        x.apply_threshold.manual_threshold.value = .3
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.threshold_smoothing_scale.value = 0
        img = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                           [0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, .5, .9, .9, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0, .5, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5, 0, 0],
                           [0, 0, 0, 0, 0, 0, .5, .5, .5, .5, .5, .5, .5, .5, 0, 0],
                           [0, 0, 0, 0, 0, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, .4, .4, .4, .5, .5, .5, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .9, .9, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0],
                           [0, 0, 0, .4, .4, .4, .4, .4, .4, .4, .4, .4, .4, 0, 0, 0],
                           [0, 0, 0, 0, 0, .4, .4, .4, .4, .4, .4, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .5, mode='constant')
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(objects.segmented), 2)
        # This point has a closer "crow-fly" distance to the upper object
        # but should be in the lower one because of the serpentine path
        self.assertEqual(objects.segmented[14, 9], objects.segmented[9, 9])

    def test_02_12_fly(self):
        '''Run identify on the fly image'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:1
HasImagePlaneDetails:False

IdentifyPrimaryObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:13|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:CropBlue
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):15,40
    Discard objects outside the diameter range?:Yes
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:5
    Speed up by using lower-resolution image to find local maxima?:Yes
    Fill holes in identified objects?:After both thresholding and declumping
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Use advanced settings?:Yes
    Threshold setting version:3
    Threshold strategy:Global
    Thresholding method:Otsu
    Threshold smoothing scale:1.3488
    Threshold correction factor:1.6
    Lower and upper bounds on threshold:0,1
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:10
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(pipeline, event):
            self.assertFalse(isinstance(event, (cellprofiler.pipeline.RunExceptionEvent,
                                                cellprofiler.pipeline.LoadExceptionEvent)))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        x = pipeline.modules()[0]
        self.assertTrue(isinstance(x, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))

        img = fly_image()
        image = cellprofiler.image.Image(img)
        #
        # Make sure it runs both regular and with reduced image
        #
        for min_size in (9, 15):
            #
            # Exercise clumping / declumping options
            #
            x.size_range.min = min_size
            for unclump_method in (
                    cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY,
                    cellprofiler.modules.identifyprimaryobjects.UN_SHAPE,
            ):
                x.unclump_method.value = unclump_method
                for watershed_method in (
                        cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY,
                        cellprofiler.modules.identifyprimaryobjects.WA_SHAPE,
                        cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE
                ):
                    x.watershed_method.value = watershed_method
                    image_set_list = cellprofiler.image.ImageSetList()
                    image_set = image_set_list.get_image_set(0)
                    image_set.add(x.x_name.value, image)
                    object_set = cellprofiler.object.ObjectSet()
                    measurements = cellprofiler.measurement.Measurements()
                    x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))

    def test_02_13_maxima_suppression_zero(self):
        # Regression test for issue #877
        # if maxima_suppression_size = 1 or 0, use a 4-connected structuring
        # element.
        #
        img = numpy.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, .1, 0, 0, .1, 0, 0, .1, 0, 0],
                 [0, .1, 0, 0, 0, .2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, .1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        expected = numpy.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 2, 0, 0, 3, 0, 0],
                 [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        for distance in (0, 1):
            x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
            x.use_advanced.value = True
            x.x_name.value = "my_image"
            x.y_name.value = "my_object"
            x.exclude_size.value = False
            x.size_range.value = (2, 10)
            x.fill_holes.value = False
            x.smoothing_filter_size.value = 0
            x.automatic_smoothing.value = 0
            x.maxima_suppression_size.value = distance
            x.automatic_suppression.value = False
            x.apply_threshold.manual_threshold.value = .05
            x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
            x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
            x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
            x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
            x.apply_threshold.threshold_smoothing_scale.value = 0
            pipeline = cellprofiler.pipeline.Pipeline()
            x.module_num = 1
            pipeline.add_module(x)
            object_set = cellprofiler.object.ObjectSet()
            measurements = cellprofiler.measurement.Measurements()
            measurements.add(x.x_name.value, cellprofiler.image.Image(img))
            x.run(cellprofiler.workspace.Workspace(pipeline, x, measurements, object_set, measurements,
                                                   None))
            output = object_set.get_objects(x.y_name.value)
            self.assertEqual(output.count, 4)
            self.assertTrue(numpy.all(output.segmented[expected == 0] == 0))
            self.assertEqual(len(numpy.unique(output.segmented[expected == 1])), 1)

    def test_04_10_load_v10(self):
        # Sorry about this overly-long pipeline, it seemed like we need to
        # revisit many of the choices.
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130226215424
ModuleCount:11
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:Yes
    Extraction method count:2
    Extraction method:Automatic
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No
    Extraction method:Manual
    Source:From file name
    Regular expression:^(?P<StackName>\x5B^.\x5D+).flex
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assignment method:Assign all images
    Load as:Grayscale image
    Image name:Channel0
    :\x5B\x5D
    Assign channels by:Order
    Assignments count:1
    Match this rule:or (metadata does StackName "")
    Image name:DNA
    Objects name:Cell
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:Yes
    grouping metadata count:1
    Metadata category:StackName

IdentifyPrimaryObjects:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel0
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:CellOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Adaptive
    Threshold method:Otsu
    Smoothing for threshold:Automatic
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.90
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel1
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:Laplacian of Gaussian
    Method to draw dividing lines between clumped objects:None
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Erase
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Automatic
    Threshold method:MCT
    Smoothing for threshold:Manual
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel2
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:None
    Method to draw dividing lines between clumped objects:Propagate
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Truncate
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Binary image
    Threshold method:MoG
    Smoothing for threshold:No smoothing
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel3
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:Shape
    Method to draw dividing lines between clumped objects:Shape
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Truncate
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Global
    Threshold method:Background
    Smoothing for threshold:No smoothing
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:9|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel4
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:Shape
    Method to draw dividing lines between clumped objects:Shape
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Truncate
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Manual
    Threshold method:Kapur
    Smoothing for threshold:No smoothing
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:10|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel4
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:Shape
    Method to draw dividing lines between clumped objects:Shape
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Truncate
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Measurement
    Threshold method:RidlerCalvard
    Smoothing for threshold:No smoothing
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12

IdentifyPrimaryObjects:[module_num:11|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Channel4
    Name the primary objects to be identified:Cells
    Typical diameter of objects, in pixel units (Min,Max):15,45
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Method to distinguish clumped objects:Shape
    Method to draw dividing lines between clumped objects:Shape
    Size of smoothing filter:11
    Suppress local maxima that are closer than this minimum allowed distance:9
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:CellOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum allowed distance between local maxima?:No
    Retain outlines of the identified objects?:Yes
    Automatically calculate the threshold using the Otsu method?:No
    Enter Laplacian of Gaussian threshold:0.2
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter:3
    Handling of objects if excessive number of objects identified:Truncate
    Maximum number of objects:499
    Threshold setting version:1
    Threshold strategy:Per object
    Threshold method:RobustBackground
    Smoothing for threshold:No smoothing
    Threshold smoothing scale:2.0
    Threshold correction factor:.80
    Lower and upper bounds on threshold:0.01,0.09
    Approximate fraction of image covered by objects?:0.05
    Manual threshold:0.03
    Select the measurement to threshold with:Metadata_Threshold
    Select binary image:Segmentation
    Masking objects:Wells
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Method to calculate adaptive window size:Custom
    Size of adaptive window:12
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(
                    isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[4]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.x_name, "Channel0")
        self.assertEqual(module.y_name, "Cells")
        self.assertEqual(module.size_range.min, 15)
        self.assertEqual(module.size_range.max, 45)
        self.assertTrue(module.exclude_size)
        self.assertTrue(module.exclude_border_objects)
        self.assertEqual(module.unclump_method, cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY)
        self.assertEqual(module.watershed_method, cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY)
        self.assertTrue(module.automatic_smoothing)
        self.assertEqual(module.smoothing_filter_size, 11)
        self.assertTrue(module.automatic_suppression)
        self.assertEqual(module.maxima_suppression_size, 9)
        self.assertTrue(module.low_res_maxima)
        self.assertEqual(module.fill_holes, cellprofiler.modules.identifyprimaryobjects.FH_THRESHOLDING)
        self.assertEqual(module.limit_choice, cellprofiler.modules.identifyprimaryobjects.LIMIT_NONE)
        self.assertEqual(module.maximum_object_count, 499)
        #
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.identify.TS_ADAPTIVE)
        self.assertEqual(module.apply_threshold.local_operation.value, centrosome.threshold.TM_OTSU)
        self.assertEqual(module.apply_threshold.threshold_smoothing_scale, 1.3488)
        self.assertAlmostEqual(module.apply_threshold.threshold_correction_factor, .80)
        self.assertAlmostEqual(module.apply_threshold.threshold_range.min, 0.01)
        self.assertAlmostEqual(module.apply_threshold.threshold_range.max, 0.90)
        self.assertAlmostEqual(module.apply_threshold.manual_threshold, 0.03)
        self.assertEqual(module.apply_threshold.thresholding_measurement, "Metadata_Threshold")
        self.assertEqual(module.apply_threshold.two_class_otsu, cellprofiler.modules.identify.O_TWO_CLASS)
        self.assertEqual(module.apply_threshold.assign_middle_to_foreground, cellprofiler.modules.identify.O_FOREGROUND)
        self.assertEqual(module.apply_threshold.adaptive_window_size, 12)
        #
        # Test alternate settings using subsequent instances of IDPrimary
        #
        module = pipeline.modules()[5]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertFalse(module.exclude_size)
        self.assertFalse(module.exclude_border_objects)
        self.assertEqual(module.unclump_method, cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY)
        self.assertEqual(module.watershed_method, cellprofiler.modules.identifyprimaryobjects.WA_NONE)
        self.assertFalse(module.automatic_smoothing)
        self.assertFalse(module.automatic_suppression)
        self.assertFalse(module.low_res_maxima)
        self.assertEqual(module.fill_holes, cellprofiler.modules.identifyprimaryobjects.FH_NEVER)
        self.assertEqual(module.limit_choice, cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE)
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.identify.TS_GLOBAL)
        self.assertEqual(module.apply_threshold.global_operation.value, cellprofiler.modules.applythreshold.TM_LI)
        self.assertEqual(module.apply_threshold.two_class_otsu, cellprofiler.modules.identify.O_THREE_CLASS)
        self.assertEqual(module.apply_threshold.assign_middle_to_foreground, cellprofiler.modules.identify.O_BACKGROUND)
        self.assertTrue(module.use_advanced.value)

        module = pipeline.modules()[6]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.unclump_method, cellprofiler.modules.identifyprimaryobjects.UN_NONE)
        self.assertEqual(module.watershed_method, cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE)
        self.assertEqual(module.limit_choice, "None")
        self.assertEqual(module.apply_threshold.global_operation.value, "None")
        self.assertEqual(module.apply_threshold.threshold_smoothing_scale, 0)
        self.assertTrue(module.use_advanced.value)

        module = pipeline.modules()[7]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.unclump_method, cellprofiler.modules.identifyprimaryobjects.UN_SHAPE)
        self.assertEqual(module.watershed_method, cellprofiler.modules.identifyprimaryobjects.WA_SHAPE)
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.identify.TS_GLOBAL)
        self.assertEqual(module.apply_threshold.global_operation.value, centrosome.threshold.TM_ROBUST_BACKGROUND)
        self.assertEqual(module.apply_threshold.lower_outlier_fraction.value, 0.02)
        self.assertEqual(module.apply_threshold.upper_outlier_fraction.value, 0.02)
        self.assertEqual(module.apply_threshold.averaging_method.value, cellprofiler.modules.identify.RB_MODE)
        self.assertEqual(module.apply_threshold.variance_method.value, cellprofiler.modules.identify.RB_SD)
        self.assertEqual(module.apply_threshold.number_of_deviations.value, 0)
        self.assertEqual(module.apply_threshold.threshold_correction_factor.value, 1.6)
        self.assertTrue(module.use_advanced.value)

        module = pipeline.modules()[8]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.applythreshold.TS_GLOBAL)
        self.assertEqual(module.apply_threshold.global_operation.value, cellprofiler.modules.applythreshold.TM_MANUAL)
        self.assertTrue(module.use_advanced.value)

        module = pipeline.modules()[9]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.applythreshold.TS_GLOBAL)
        self.assertEqual(module.apply_threshold.global_operation.value, cellprofiler.modules.applythreshold.TM_MEASUREMENT)
        self.assertTrue(module.use_advanced.value)

        module = pipeline.modules()[10]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects))
        self.assertEqual(module.apply_threshold.threshold_scope, "None")
        self.assertEqual(module.apply_threshold.global_operation.value, centrosome.threshold.TM_ROBUST_BACKGROUND)
        self.assertEqual(module.apply_threshold.lower_outlier_fraction, .05)
        self.assertEqual(module.apply_threshold.upper_outlier_fraction, .05)
        self.assertEqual(module.apply_threshold.averaging_method, cellprofiler.modules.identify.RB_MEAN)
        self.assertEqual(module.apply_threshold.variance_method, cellprofiler.modules.identify.RB_SD)
        self.assertEqual(module.apply_threshold.number_of_deviations, 2)
        self.assertTrue(module.use_advanced.value)

    def test_04_10_01_load_new_robust_background(self):
        #
        # Test custom robust background parameters.
        #
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20141114191709
GitHash:d186f20
ModuleCount:3
HasImagePlaneDetails:False

IdentifyPrimaryObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7.0
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:After both thresholding and declumping
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.5
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5.0
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Threshold setting version:2
    Threshold strategy:Global
    Thresholding method:RobustBackground
    Select the smoothing method for thresholding:Automatic
    Threshold smoothing scale:1.0
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Approximate fraction of image covered by objects?:0.01
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Select binary image:None
    Masking objects:None
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:10
    Use default parameters?:Custom
    Lower outlier fraction:0.1
    Upper outlier fraction:0.2
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2.5

IdentifyPrimaryObjects:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7.0
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:After both thresholding and declumping
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.5
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5.0
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Threshold setting version:2
    Threshold strategy:Global
    Thresholding method:RobustBackground
    Select the smoothing method for thresholding:Automatic
    Threshold smoothing scale:1.0
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Approximate fraction of image covered by objects?:0.01
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Select binary image:None
    Masking objects:None
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:10
    Use default parameters?:Custom
    Lower outlier fraction:0.1
    Upper outlier fraction:0.2
    Averaging method:Median
    Variance method:Median absolute deviation
    # of deviations:2.5

IdentifyPrimaryObjects:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7.0
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:After both thresholding and declumping
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.5
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5.0
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Threshold setting version:2
    Threshold strategy:Global
    Thresholding method:RobustBackground
    Select the smoothing method for thresholding:Automatic
    Threshold smoothing scale:1.0
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Approximate fraction of image covered by objects?:0.01
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Select binary image:None
    Masking objects:None
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:10
    Use default parameters?:Custom
    Lower outlier fraction:0.1
    Upper outlier fraction:0.2
    Averaging method:Mode
    Variance method:Median absolute deviation
    # of deviations:2.5
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(
                    isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        for module, averaging_method, variance_method in zip(
                pipeline.modules(),
                (cellprofiler.modules.identify.RB_MEAN,
                 cellprofiler.modules.identify.RB_MEDIAN,
                 cellprofiler.modules.identify.RB_MODE),
                (cellprofiler.modules.identify.RB_SD,
                 cellprofiler.modules.identify.RB_MAD,
                 cellprofiler.modules.identify.RB_MAD)
        ):
            assert isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects)
            self.assertEqual(module.apply_threshold.lower_outlier_fraction, .1)
            self.assertEqual(module.apply_threshold.upper_outlier_fraction, .2)
            self.assertEqual(module.apply_threshold.number_of_deviations, 2.5)
            self.assertEqual(module.apply_threshold.averaging_method, averaging_method)
            self.assertEqual(module.apply_threshold.variance_method, variance_method)
            self.assertTrue(module.use_advanced.value)

    def test_05_01_discard_large(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .3
        img = numpy.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(objects.segmented[25, 25], 1, "The small object was not there")
        self.assertEqual(objects.segmented[100, 100], 0, "The large object was not filtered out")
        self.assertTrue(objects.small_removed_segmented[25, 25] > 0,
                        "The small object was not in the small_removed label set")
        self.assertTrue(objects.small_removed_segmented[100, 100] > 0,
                        "The large object was not in the small-removed label set")
        self.assertTrue(objects.unedited_segmented[25, 25], "The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100, 100], "The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 1)

    def test_05_02_keep_large(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .3
        img = numpy.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[25, 25], "The small object was not there")
        self.assertTrue(objects.segmented[100, 100], "The large object was filtered out")
        self.assertTrue(objects.unedited_segmented[25, 25], "The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100, 100], "The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 2)

    def test_05_03_discard_small(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 40
        x.size_range.max = 60
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .3
        img = numpy.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(objects.segmented[25, 25], 0, "The small object was not filtered out")
        self.assertEqual(objects.segmented[100, 100], 1, "The large object was not present")
        self.assertTrue(objects.small_removed_segmented[25, 25] == 0,
                        "The small object was in the small_removed label set")
        self.assertTrue(objects.small_removed_segmented[100, 100] > 0,
                        "The large object was not in the small-removed label set")
        self.assertTrue(objects.unedited_segmented[25, 25], "The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100, 100], "The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape), 1)

    def test_05_02_discard_edge(self):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_MANUAL
        x.apply_threshold.manual_threshold.value = .3
        img = numpy.zeros((100, 100))
        centers = [(50, 50), (10, 50), (50, 10), (90, 50), (50, 90)]
        present = [True, False, False, False, False]
        for center in centers:
            draw_circle(img, center, 15, .5)
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        for center, p in zip(centers, present):
            if p:
                self.assertTrue(objects.segmented[center[0], center[1]] > 0)
                self.assertTrue(objects.small_removed_segmented[center[0], center[1]] > 0)
            else:
                self.assertTrue(objects.segmented[center[0], center[1]] == 0)
                self.assertTrue(objects.small_removed_segmented[center[0], center[1]] == 0)
            self.assertTrue(objects.unedited_segmented[center[0], center[1]] > 0)

    def test_05_03_discard_with_mask(self):
        """Check discard of objects that are on the border of a mask"""
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_MANUAL
        x.apply_threshold.manual_threshold.value = .3
        img = numpy.zeros((200, 200))
        centers = [(100, 100), (30, 100), (100, 30), (170, 100), (100, 170)]
        present = [True, False, False, False, False]
        for center in centers:
            draw_circle(img, center, 15, .5)
        mask = numpy.zeros((200, 200))
        mask[25:175, 25:175] = 1
        image = cellprofiler.image.Image(img, mask)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        for center, p in zip(centers, present):
            if p:
                self.assertTrue(objects.segmented[center[0], center[1]] > 0)
                self.assertTrue(objects.small_removed_segmented[center[0], center[1]] > 0)
            else:
                self.assertTrue(objects.segmented[center[0], center[1]] == 0)
                self.assertTrue(objects.small_removed_segmented[center[0], center[1]] == 0)
            self.assertTrue(objects.unedited_segmented[center[0], center[1]] > 0)

    def test_06_01_regression_diagonal(self):
        """Regression test - was using one-connected instead of 3-connected structuring element"""
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.threshold_smoothing_scale.value = 0
        x.apply_threshold.manual_threshold.value = .5
        img = numpy.zeros((10, 10))
        img[4, 4] = 1
        img[5, 5] = 1
        image = cellprofiler.image.Image(img)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented[img > 0] == 1))
        self.assertTrue(numpy.all(img[segmented == 1] > 0))

    def test_06_02_regression_adaptive_mask(self):
        """Regression test - mask all but one pixel / adaptive"""
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.apply_threshold.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
        x.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        numpy.random.seed(62)
        img = numpy.random.uniform(size=(100, 100))
        mask = numpy.zeros(img.shape, bool)
        mask[-1, -1] = True
        image = cellprofiler.image.Image(img, mask)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.image.VanillaImageProvider("my_image", image))
        object_set = cellprofiler.object.ObjectSet()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented == 0))

    # def test_11_01_test_robust_background_fly(self):
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.apply_threshold.threshold_scope.value = I.TS_GLOBAL
    #     x.threshold_method.value = T.TM_ROBUST_BACKGROUND
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.09)
    #     self.assertTrue(threshold < 0.095)

    def test_16_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        oname = "my_object"
        x.y_name.value = oname
        x.x_name.value = "my_image"
        columns = x.get_measurement_columns(None)
        expected_columns = [
            (cellprofiler.measurement.IMAGE, format % oname, coltype)
            for format, coltype in (
                (cellprofiler.measurement.FF_COUNT, cellprofiler.measurement.COLTYPE_INTEGER),
                (cellprofiler.measurement.FF_FINAL_THRESHOLD, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.measurement.FF_ORIG_THRESHOLD, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.measurement.FF_WEIGHTED_VARIANCE, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.measurement.FF_SUM_OF_ENTROPIES, cellprofiler.measurement.COLTYPE_FLOAT)
            )]
        expected_columns += [(oname, feature, cellprofiler.measurement.COLTYPE_FLOAT)
                             for feature in (cellprofiler.measurement.M_LOCATION_CENTER_X,
                                             cellprofiler.measurement.M_LOCATION_CENTER_Y,
                                             cellprofiler.measurement.M_LOCATION_CENTER_Z)]
        expected_columns += [(oname,
                              cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                              cellprofiler.measurement.COLTYPE_INTEGER)]
        self.assertEqual(len(columns), len(expected_columns))
        for column in columns:
            self.assertTrue(any(all([colval == exval for colval, exval in zip(column, expected)])
                                for expected in expected_columns))

    def test_17_01_regression_holes(self):
        '''Regression test - fill holes caused by filtered object

        This was created as a regression test for the bug, IMG-191, but
        didn't exercise the bug. It's a good test of watershed and filling
        labeled holes in an odd case, so I'm leaving it in.
        '''
        #
        # This array has two intensity peaks separated by a border.
        # You should get two objects, one within the other.
        #
        pixels = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                              [0, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0],
                              [0, 0, 2, 1, 2, 9, 2, 2, 1, 2, 0, 0],
                              [0, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0],
                              [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 9, 9, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 9, 9, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], float) / 10.0
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                                [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                                [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                                [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                                [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        mask = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], bool)
        workspace, x = self.make_workspace(pixels)
        x.use_advanced.value = True
        x.exclude_size.value = True
        x.size_range.min = 6
        x.size_range.max = 50
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.threshold_smoothing_scale.value = 0
        x.apply_threshold.manual_threshold.value = .05
        x.apply_threshold.threshold_correction_factor.value = 1
        measurements = workspace.measurements
        x.run(workspace)
        my_objects = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(my_objects.segmented[3, 3] != 0)
        if my_objects.unedited_segmented[3, 3] == 2:
            unedited_segmented = my_objects.unedited_segmented
        else:
            unedited_segmented = numpy.array([0, 2, 1])[my_objects.unedited_segmented]
        self.assertTrue(numpy.all(unedited_segmented[mask] == expected[mask]))

    def test_17_02_regression_holes(self):
        '''Regression test - fill holes caused by filtered object

        This is the real regression test for IMG-191. The smaller object
        is surrounded by pixels below threshold. This prevents filling in
        the unedited case.
        '''
        # An update to fill_labeled_holes will remove both the filtered object
        # and the hole
        #
        if True:
            return
        pixels = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                              [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                              [0, 0, 3, 0, 0, 9, 2, 0, 0, 3, 0, 0],
                              [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                              [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                              [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 9, 2, 0, 0],
                              [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], float) / 10.0
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        mask = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], bool)
        image = cellprofiler.image.Image(pixels)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.object.ObjectSet()

        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 4
        x.size_range.max = 50
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.local_operation.value = centrosome.threshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .1
        x.apply_threshold.threshold_correction_factor.value = 1
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements,
                                                     image_set_list)
        x.run(workspace)
        my_objects = object_set.get_objects("my_object")
        self.assertTrue(my_objects.segmented[3, 3] != 0)
        self.assertTrue(numpy.all(my_objects.segmented[mask] == expected[mask]))

    def test_18_02_erase_objects(self):
        '''Set up a limit on the # of objects and exceed it - erasing objects'''
        maximum_object_count = 3
        pixels = numpy.zeros((20, 21))
        pixels[2:8, 2:8] = .5
        pixels[12:18, 2:8] = .5
        pixels[2:8, 12:18] = .5
        pixels[12:18, 12:18] = .5
        image = cellprofiler.image.Image(pixels)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.object.ObjectSet()

        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .25
        x.apply_threshold.threshold_correction_factor.value = 1
        x.limit_choice.value = cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE
        x.maximum_object_count.value = maximum_object_count
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements,
                                                     image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement(
                "Count_my_object"), 0)
        my_objects = object_set.get_objects("my_object")
        self.assertTrue(numpy.all(my_objects.segmented == 0))
        self.assertEqual(numpy.max(my_objects.unedited_segmented), 4)

    def test_18_03_dont_erase_objects(self):
        '''Ask to erase objects, but don't'''
        maximum_object_count = 5
        pixels = numpy.zeros((20, 21))
        pixels[2:8, 2:8] = .5
        pixels[12:18, 2:8] = .5
        pixels[2:8, 12:18] = .5
        pixels[12:18, 12:18] = .5
        image = cellprofiler.image.Image(pixels)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.object.ObjectSet()

        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "my_object"
        x.x_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        x.apply_threshold.manual_threshold.value = .25
        x.apply_threshold.threshold_correction_factor.value = 1
        x.limit_choice.value = cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE
        x.maximum_object_count.value = maximum_object_count
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements,
                                                     image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement(
                "Count_my_object"), 4)
        my_objects = object_set.get_objects("my_object")
        self.assertEqual(numpy.max(my_objects.segmented), 4)

    def test_19_01_threshold_by_measurement(self):
        '''Set threshold based on mean image intensity'''
        pixels = numpy.zeros((10, 10))
        pixels[2:6, 2:6] = .5

        image = cellprofiler.image.Image(pixels)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("MyImage", image)
        object_set = cellprofiler.object.ObjectSet()

        pipeline = cellprofiler.pipeline.Pipeline()
        measurements = cellprofiler.measurement.Measurements()
        measurements.add_image_measurement("MeanIntensity_MyImage", numpy.mean(pixels))

        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.y_name.value = "MyObject"
        x.x_name.value = "MyImage"
        x.exclude_size.value = False
        x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
        x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        x.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        x.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MEASUREMENT
        x.apply_threshold.threshold_smoothing_scale.value = 0
        x.apply_threshold.thresholding_measurement.value = "MeanIntensity_MyImage"
        x.apply_threshold.threshold_correction_factor.value = 1
        x.module_num = 1
        pipeline.add_module(x)

        workspace = cellprofiler.workspace.Workspace(pipeline, x, image_set, object_set, measurements,
                                                     image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement("Count_MyObject"), 1)
        self.assertEqual(measurements.get_current_image_measurement("Threshold_FinalThreshold_MyObject"),
                         numpy.mean(pixels))

    def test_20_01_threshold_smoothing_automatic(self):
        image = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, .4, .5, .4, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]])
        workspace, module = self.make_workspace(image)
        assert isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects)
        module.use_advanced.value = True
        module.exclude_size.value = False
        module.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
        module.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        # MCT on this image is zero, so set the threshold at .225
        # with the threshold minimum (manual = no smoothing)
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_LI
        module.apply_threshold.threshold_range.min = .225
        module.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        numpy.testing.assert_array_equal(expected, labels)

    def test_20_02_threshold_smoothing_manual(self):
        image = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, .4, .5, .4, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]])
        workspace, module = self.make_workspace(image)
        assert isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects)
        module.use_advanced.value = True
        module.exclude_size.value = False
        module.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
        module.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_LI
        module.apply_threshold.threshold_range.min = .125
        module.apply_threshold.threshold_smoothing_scale.value = 3
        module.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        numpy.testing.assert_array_equal(expected, labels)

    def test_20_03_threshold_no_smoothing(self):
        image = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, .4, .5, .4, 0, 0],
                             [0, 0, .4, .4, .4, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]])
        for tm in cellprofiler.modules.identify.TS_MANUAL, cellprofiler.modules.identify.TS_MEASUREMENT:
            workspace, module = self.make_workspace(image)
            assert isinstance(module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects)
            module.use_advanced.value = True
            module.exclude_size.value = False
            module.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
            module.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
            module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
            module.apply_threshold.global_operation.value = tm
            module.apply_threshold.manual_threshold.value = .125
            module.apply_threshold.thresholding_measurement.value = MEASUREMENT_NAME
            workspace.measurements[cellprofiler.measurement.IMAGE, MEASUREMENT_NAME] = .125
            module.apply_threshold.threshold_smoothing_scale.value = 3
            module.run(workspace)
            labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
            numpy.testing.assert_array_equal(expected, labels)


def add_noise(img, fraction):
    '''Add a fractional amount of noise to an image to make it look real'''
    numpy.random.seed(0)
    noise = numpy.random.uniform(low=1 - fraction / 2, high=1 + fraction / 2,
                                 size=img.shape)
    return img * noise


def one_cell_image():
    img = numpy.zeros((25, 25))
    draw_circle(img, (10, 15), 5, .5)
    return add_noise(img, .01)


def two_cell_image():
    img = numpy.zeros((50, 50))
    draw_circle(img, (10, 35), 5, .8)
    draw_circle(img, (30, 15), 5, .6)
    return add_noise(img, .01)


def fly_image():
    from bioformats import load_image
    path = os.path.join(os.path.dirname(__file__), '../resources/01_POS002_D.TIF')
    return load_image(path)


def draw_circle(img, center, radius, value):
    x, y = numpy.mgrid[0:img.shape[0], 0:img.shape[1]]
    distance = numpy.sqrt((x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1]))
    img[distance <= radius] = value


class TestWeightedVariance(unittest.TestCase):
    def test_01_masked_wv(self):
        output = centrosome.threshold.weighted_variance(numpy.zeros((3, 3)),
                                                        numpy.zeros((3, 3), bool), 1)
        self.assertEqual(output, 0)

    def test_02_zero_wv(self):
        output = centrosome.threshold.weighted_variance(numpy.zeros((3, 3)),
                                                        numpy.ones((3, 3), bool),
                                                        numpy.ones((3, 3), bool))
        self.assertEqual(output, 0)

    def test_03_fg_0_bg_0(self):
        """Test all foreground pixels same, all background same, wv = 0"""
        img = numpy.zeros((4, 4))
        img[:, 2:4] = 1
        binary_image = img > .5
        output = centrosome.threshold.weighted_variance(img, numpy.ones(img.shape, bool), binary_image)
        self.assertEqual(output, 0)

    def test_04_values(self):
        """Test with two foreground and two background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25 *2 + .25 *2)/4 = .25
        img = numpy.array([[1.0 / 16., 1.0 / 8.0], [1.0 / 4.0, 1.0 / 2.0]])
        binary_image = numpy.array([[False, False], [True, True]])
        output = centrosome.threshold.weighted_variance(img, numpy.ones((2, 2), bool), binary_image)
        self.assertAlmostEqual(output, .25)

    def test_05_mask(self):
        """Test, masking out one of the background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25*2 + .25 *2)/4 = .25
        img = numpy.array([[1.0 / 16., 1.0 / 16.0, 1.0 / 8.0], [1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
        mask = numpy.array([[False, True, True], [False, True, True]])
        binary_image = numpy.array([[False, False, False], [True, True, True]])
        output = centrosome.threshold.weighted_variance(img, mask, binary_image)
        self.assertAlmostEquals(output, .25)


class TestSumOfEntropies(unittest.TestCase):
    def test_01_all_masked(self):
        output = centrosome.threshold.sum_of_entropies(numpy.zeros((3, 3)),
                                                       numpy.zeros((3, 3), bool), 1)
        self.assertEqual(output, 0)

    def test_020_all_zero(self):
        """Can't take the log of zero, so all zero matrix = 0"""
        output = centrosome.threshold.sum_of_entropies(numpy.zeros((4, 2)),
                                                       numpy.ones((4, 2), bool),
                                                       numpy.ones((4, 2), bool))
        self.assertAlmostEqual(output, 0)

    def test_03_fg_bg_equal(self):
        img = numpy.ones((128, 128))
        img[0:64, :] *= .15
        img[64:128, :] *= .85
        img[0, 0] = img[-1, 0] = 0
        img[0, -1] = img[-1, -1] = 1
        binary_mask = numpy.zeros(img.shape, bool)
        binary_mask[64:, :] = True
        #
        # You need one foreground and one background pixel to defeat a
        # divide-by-zero (that's appropriately handled)
        #
        one_of_each = numpy.zeros(img.shape, bool)
        one_of_each[0, 0] = one_of_each[-1, -1] = True
        output = centrosome.threshold.sum_of_entropies(img, numpy.ones((128, 128), bool), binary_mask)
        ob = centrosome.threshold.sum_of_entropies(img, one_of_each | ~binary_mask, binary_mask)
        of = centrosome.threshold.sum_of_entropies(img, one_of_each | binary_mask, binary_mask)
        self.assertAlmostEqual(output, ob + of)

    def test_04_fg_bg_different(self):
        img = numpy.ones((128, 128))
        img[0:64, 0:64] *= .15
        img[0:64, 64:128] *= .3
        img[64:128, 0:64] *= .7
        img[64:128, 64:128] *= .85
        binary_mask = numpy.zeros(img.shape, bool)
        binary_mask[64:, :] = True
        one_of_each = numpy.zeros(img.shape, bool)
        one_of_each[0, 0] = one_of_each[-1, -1] = True
        output = centrosome.threshold.sum_of_entropies(img, numpy.ones((128, 128), bool), binary_mask)
        ob = centrosome.threshold.sum_of_entropies(img, one_of_each | ~binary_mask, binary_mask)
        of = centrosome.threshold.sum_of_entropies(img, one_of_each | binary_mask, binary_mask)
        self.assertAlmostEqual(output, ob + of)
