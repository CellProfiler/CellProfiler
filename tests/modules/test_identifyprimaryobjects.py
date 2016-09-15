"""test_identifyprimautomatic.py: test the IdentifyPrimAutomatic module
"""
import StringIO
import base64
import os
import unittest
import zlib

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.identify as I
import cellprofiler.modules.identifyprimaryobjects as ID
import cellprofiler.object as cpo
import cellprofiler.pipeline
import cellprofiler.setting
import centrosome.threshold as T
import numpy as np
import scipy.ndimage
import tests.modules
from cellprofiler.workspace import Workspace

IMAGE_NAME = "my_image"
OBJECTS_NAME = "my_objects"
BINARY_IMAGE_NAME = "binary_image"
MASKING_OBJECTS_NAME = "masking_objects"
MEASUREMENT_NAME = "my_measurement"


class test_IdentifyPrimaryObjects(unittest.TestCase):
    def load_error_handler(self, caller, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.fail(event.error.message)

    def make_workspace(self, image,
                       mask=None,
                       labels=None,
                       binary_image=None):
        '''Make a workspace and IdentifyPrimaryObjects module

        image - the intensity image for thresholding

        mask - if present, the "don't analyze" mask of the intensity image

        labels - if thresholding per-object, the labels matrix needed

        binary_image - if thresholding using a binary image, the image
        '''
        module = ID.IdentifyPrimaryObjects()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.object_name.value = OBJECTS_NAME
        module.binary_image.value = BINARY_IMAGE_NAME
        module.masking_objects.value = MASKING_OBJECTS_NAME

        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        cpimage = cpi.Image(image, mask=mask)
        m.add(IMAGE_NAME, cpimage)
        if binary_image is not None:
            m.add(BINARY_IMAGE_NAME, cpi.Image(binary_image))
        object_set = cpo.ObjectSet()
        if labels is not None:
            o = cpo.Objects()
            o.segmented = labels
            object_set.add_objects(o, MASKING_OBJECTS_NAME)
        workspace = cellprofiler.workspace.Workspace(
                pipeline, module, m, object_set, m, None)
        return workspace, module

    def test_00_00_init(self):
        x = ID.IdentifyPrimaryObjects()

    def test_02_000_test_zero_objects(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((25, 25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image", "Count_my_object")
        self.assertEqual(count, 0)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 0)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object", "Location_Center_Y")
        self.assertTrue(isinstance(location_center_y, np.ndarray))
        self.assertEqual(np.product(location_center_y.shape), 0)

    def test_02_001_test_zero_objects_wa_in_lo_in(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_INTENSITY
        x.unclump_method.value = ID.UN_INTENSITY
        img = np.zeros((25, 25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_002_test_zero_objects_wa_di_lo_in(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_SHAPE
        x.unclump_method.value = ID.UN_INTENSITY
        img = np.zeros((25, 25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_003_test_zero_objects_wa_in_lo_sh(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_INTENSITY
        x.unclump_method.value = ID.UN_SHAPE
        img = np.zeros((25, 25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_004_test_zero_objects_wa_di_lo_sh(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_SHAPE
        x.unclump_method.value = ID.UN_SHAPE
        img = np.zeros((25, 25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_01_test_one_object(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = one_cell_image()
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented[img > 0] == 1))
        self.assertTrue(np.all(img[segmented == 1] > 0))
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
        self.assertTrue(isinstance(location_center_y, np.ndarray))
        self.assertEqual(np.product(location_center_y.shape), 1)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 1)
        self.assertTrue(location_center_x[0] > 13)
        self.assertTrue(location_center_x[0] < 16)
        columns = x.get_measurement_columns(pipeline)
        for object_name in (cpmeas.IMAGE, "my_object"):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_02_02_test_two_objects(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = two_cell_image()
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        self.assertTrue(isinstance(location_center_y, np.ndarray))
        self.assertEqual(np.product(location_center_y.shape), 2)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue(location_center_y[1] > 28)
        self.assertTrue(location_center_y[1] < 32)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 2)
        self.assertTrue(location_center_x[0] > 33)
        self.assertTrue(location_center_x[0] < 37)
        self.assertTrue(location_center_x[1] > 13)
        self.assertTrue(location_center_x[1] < 16)

    def test_02_03_test_threshold_range(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .7
        x.threshold_range.max = 1
        x.threshold_correction_factor.value = .95
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = I.TM_MCT
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        img = two_cell_image()
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        self.assertTrue(isinstance(location_center_y, np.ndarray))
        self.assertEqual(np.product(location_center_y.shape), 1)
        self.assertTrue(location_center_y[0] > 8)
        self.assertTrue(location_center_y[0] < 12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 1)
        self.assertTrue(location_center_x[0] > 33)
        self.assertTrue(location_center_x[0] < 36)

    def test_02_04_fill_holes(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.fill_holes.value = True
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.zeros((40, 40))
        draw_circle(img, (10, 10), 7, .5)
        draw_circle(img, (30, 30), 7, .5)
        img[10, 10] = 0
        img[30, 30] = 0
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10, 10] > 0)
        self.assertTrue(objects.segmented[30, 30] > 0)

    def test_02_05_dont_fill_holes(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .7
        x.threshold_range.max = 1
        x.exclude_size.value = False
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((40, 40))
        draw_circle(img, (10, 10), 7, .5)
        draw_circle(img, (30, 30), 7, .5)
        img[10, 10] = 0
        img[30, 30] = 0
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10, 10] == 0)
        self.assertTrue(objects.segmented[30, 30] == 0)

    def test_02_05_01_fill_holes_within_holes(self):
        'Regression test of img-1431'
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.size_range.min = 1
        x.size_range.max = 2
        x.exclude_size.value = False
        x.fill_holes.value = ID.FH_DECLUMP
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.zeros((40, 40))
        draw_circle(img, (20, 20), 10, .5)
        draw_circle(img, (20, 20), 4, 0)
        img[20, 20] = 1
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[20, 20] == 1)
        self.assertTrue(objects.segmented[22, 20] == 1)
        self.assertTrue(objects.segmented[26, 20] == 1)

    def test_02_06_test_watershed_shape_shape(self):
        """Identify by local_maxima:shape & intensity:shape

        Create an object whose intensity is high near the middle
        but has an hourglass shape, then segment it using shape/shape
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_SHAPE
        x.watershed_method.value = ID.WA_SHAPE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 2)

    def test_02_07_test_watershed_shape_intensity(self):
        """Identify by local_maxima:shape & watershed:intensity

        Create an object with an hourglass shape to get two maxima, but
        set the intensities so that one maximum gets the middle portion
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_SHAPE
        x.watershed_method.value = ID.WA_INTENSITY
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 2)
        self.assertEqual(objects.segmented[7, 11], objects.segmented[7, 4])

    def test_02_08_test_watershed_intensity_distance_single(self):
        """Identify by local_maxima:intensity & watershed:shape - one object

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (4, 10)
        x.fill_holes.value = False
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 1)

    def test_02_08_test_watershed_intensity_distance_triple(self):
        """Identify by local_maxima:intensity & watershed:shape - 3 objects w/o filter

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.maxima_suppression_size.value = 7.1
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 3)

    def test_02_09_test_watershed_intensity_distance_filter(self):
        """Identify by local_maxima:intensity & watershed:shape - filtered

        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 1
        x.automatic_smoothing.value = 1
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        x.threshold_scope.value = I.TS_GLOBAL
        x.threshold_method.value = T.TM_OTSU
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 1)

    def test_02_10_test_watershed_intensity_distance_double(self):
        """Identify by local_maxima:intensity & watershed:shape - two objects

        Create an object with an hourglass shape and peaks in the top and
        bottom, but with a distribution of values that's skewed so that,
        by intensity, one of the images occupies the middle. The middle
        should be shared because the watershed is done using the distance
        transform.
        """
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 2)
        self.assertNotEqual(objects.segmented[12, 7], objects.segmented[4, 7])

    def test_02_11_propagate(self):
        """Test the propagate unclump method"""
        x = ID.IdentifyPrimaryObjects()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 7
        x.automatic_suppression.value = False
        x.manual_threshold.value = .3
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_PROPAGATE
        x.threshold_scope.value = I.TS_MANUAL
        x.threshold_smoothing_choice.value = I.TSM_NONE
        img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented), 2)
        # This point has a closer "crow-fly" distance to the upper object
        # but should be in the lower one because of the serpentine path
        self.assertEqual(objects.segmented[14, 9], objects.segmented[9, 9])

    def test_02_12_fly(self):
        '''Run identify on the fly image'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9722

IdentifyPrimaryObjects:[module_num:1|svn_version:\'9633\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D]
    Select the input image:CropBlue
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):15,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:MoG Global
    Threshold correction factor:1.6
    Lower and upper bounds on threshold:0,1
    Approximate fraction of image covered by objects?:0.2
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:5
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:None
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Manual threshold:0.0
    Select binary image:MoG Global
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(pipeline, event):
            self.assertFalse(isinstance(event, (cellprofiler.pipeline.RunExceptionEvent,
                                                cellprofiler.pipeline.LoadExceptionEvent)))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        x = pipeline.modules()[0]
        self.assertTrue(isinstance(x, ID.IdentifyPrimaryObjects))

        img = fly_image()
        image = cpi.Image(img)
        #
        # Make sure it runs both regular and with reduced image
        #
        for min_size in (9, 15):
            #
            # Exercise clumping / declumping options
            #
            x.size_range.min = min_size
            for unclump_method in (ID.UN_INTENSITY, ID.UN_SHAPE, ID.UN_LOG):
                x.unclump_method.value = unclump_method
                for watershed_method in (ID.WA_INTENSITY, ID.WA_SHAPE, ID.WA_PROPAGATE):
                    x.watershed_method.value = watershed_method
                    image_set_list = cpi.ImageSetList()
                    image_set = image_set_list.get_image_set(0)
                    image_set.add(x.image_name.value, image)
                    object_set = cpo.ObjectSet()
                    measurements = cpmeas.Measurements()
                    x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))

    def test_02_13_maxima_suppression_zero(self):
        # Regression test for issue #877
        # if maxima_suppression_size = 1 or 0, use a 4-connected structuring
        # element.
        #
        img = np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, .1, 0, 0, .1, 0, 0, .1, 0, 0],
                 [0, .1, 0, 0, 0, .2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, .1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        expected = np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 2, 0, 0, 3, 0, 0],
                 [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        for distance in (0, 1):
            x = ID.IdentifyPrimaryObjects()
            x.image_name.value = "my_image"
            x.object_name.value = "my_object"
            x.exclude_size.value = False
            x.size_range.value = (2, 10)
            x.fill_holes.value = False
            x.smoothing_filter_size.value = 0
            x.automatic_smoothing.value = 0
            x.maxima_suppression_size.value = distance
            x.automatic_suppression.value = False
            x.manual_threshold.value = .05
            x.unclump_method.value = ID.UN_INTENSITY
            x.watershed_method.value = ID.WA_INTENSITY
            x.threshold_scope.value = I.TS_MANUAL
            x.threshold_smoothing_choice.value = I.TSM_NONE
            pipeline = cellprofiler.pipeline.Pipeline()
            x.module_num = 1
            pipeline.add_module(x)
            object_set = cpo.ObjectSet()
            measurements = cpmeas.Measurements()
            measurements.add(x.image_name.value, cpi.Image(img))
            x.run(Workspace(pipeline, x, measurements, object_set, measurements,
                            None))
            output = object_set.get_objects(x.object_name.value)
            self.assertEqual(output.count, 4)
            self.assertTrue(np.all(output.segmented[expected == 0] == 0))
            self.assertEqual(len(np.unique(output.segmented[expected == 1])), 1)

    def test_02_14_automatic(self):
        # Regression test of issue 1071 - automatic should yield same
        # threshold regardless of manual parameters
        #
        r = np.random.RandomState()
        r.seed(214)
        image = r.uniform(size=(20, 20))
        workspace, module = self.make_workspace(image)
        assert isinstance(module, ID.IdentifyPrimaryObjects)
        module.threshold_scope.value = I.TS_AUTOMATIC
        module.run(workspace)
        m = workspace.measurements
        orig_threshold = m[cpmeas.IMAGE, I.FF_FINAL_THRESHOLD % OBJECTS_NAME]
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = I.TS_AUTOMATIC
        module.threshold_method.value = I.TM_OTSU
        module.threshold_smoothing_choice.value = I.TSM_MANUAL
        module.threshold_smoothing_scale.value = 100
        module.threshold_correction_factor.value = .1
        module.threshold_range.min = .8
        module.threshold_range.max = .81
        module.run(workspace)
        m = workspace.measurements
        threshold = m[cpmeas.IMAGE, I.FF_FINAL_THRESHOLD % OBJECTS_NAME]
        self.assertEqual(threshold, orig_threshold)

    def test_04_01_load_matlab_12(self):
        """Test loading a Matlab version 12 IdentifyPrimAutomatic pipeline


        """
        old_r12_file = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgRGVjIDMxIDExOjQxOjUxIDIwMDggICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAuAEAAHicxVRdT8IwFO3GWEQNEYmJvu3RB2K2xAcfNTFRHgQihujjBoXUbC3ZWgM++TP8Of4Uf4ot7LMSNqfgTZpx7u45p/eytg4AMGsA6Py5w5cKllENsZJaAvchpQhPgirQwHGY/+BrYPvIdlw4sF0GAxBHlG/jMXmYT+NXd2TEXNixvXQxjw7zHOgH3XFEDF/30Ay6ffQKQTaisnv4ggJEcMgP9eVs7Euo5Fvn61NL5qCsmEMzlRf1lyCp11bU76bqD0J8TQxMqMECmOhc5Ojoko6+mNPQhagYvyrxBbbM1rkZ+ps5/EqGXwFPfHZFeGqGp4IO+Z1f3rz3pD4F7tKAGTcucWw3nneev5LRUYBVck5myyrE0zI8DZhnplWk35rUr8BtTCEOEJ2H+f/WuWKUeDZFww3obOo7Knpu/0pn2+fvTVl/zzVS+bJ9Is+ewIlP2DTRuc3RaUg6AhPnGQ7pQshAeASnqX1t+5m3/0Np/wITRl2E4bcGhN4MrP8f0vdQEf8jyV/g9ghiisbzno+89BmSvx89x1/lv5oreGXvzyJ++yV4Gme+nyx5jz+c7+ma+iii/BfqTY0Q'
        pipeline = tests.modules.load_pipeline(self, old_r12_file)
        pipeline.add_listener(self.load_error_handler)
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.module(1)
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm, T.TM_OTSU)
        self.assertTrue(module.threshold_modifier, T.TM_GLOBAL)
        self.assertAlmostEqual(float(module.object_fraction.value), .01)
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.image_name.value, "Do not use")
        self.assertTrue(module.exclude_size.value)
        self.assertEqual(module.fill_holes.value, ID.FH_THRESHOLDING)
        self.assertTrue(module.exclude_border_objects.value)
        self.assertTrue(module.automatic_smoothing.value)
        self.assertTrue(module.automatic_suppression.value)
        self.assertFalse(module.merge_objects.value)
        self.assertTrue(module.image_name == cellprofiler.setting.DO_NOT_USE)
        self.assertFalse(module.should_save_outlines.value)
        self.assertEqual(module.save_outlines.value, "None")
        self.assertAlmostEqual(module.threshold_range.min, 0)
        self.assertAlmostEqual(module.threshold_range.max, 1)
        self.assertAlmostEqual(module.threshold_correction_factor.value, 1)
        self.assertEqual(module.watershed_method.value, "Intensity")
        self.assertEqual(module.unclump_method.value, "Intensity")
        self.assertAlmostEqual(module.maxima_suppression_size.value, 5)

    def test_04_001_load_matlab_regression(self):
        '''A regression test on a pipeline that misloaded the outlines variable'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUX'
                'AuSk0sSU1RyM+zUvDNz1PwKs1TMLBQMDS1MjayMjJTMDIwsFQgGTAw'
                'evryMzAwbGNiYKiY8zbCMf+ygUjZpWVaOVrJzJ3O/JZFEsqiMhabMj'
                'mUNi5Luqyiopf3SqxZOrwzeOsfqTo29zqpwtlL+m5KXed9zRexac3z'
                'Pd9/7j1/Xt8viqHhpjCD1MkbPrs4p531SnV+EbPPpedhgkjkAr55Sz'
                '/vn1zH68zzmyXWWWgxxxPd2eXNintn+X9yFy8REL7SmhxomXm34o57'
                '4hNe48NfCvnPC+w8Yi+gsc3nrfCsRxyXFbb6f3x6syb21JLSaM/63d'
                'sfHZxQsUL1r8eM+BfNU+v+st3jY/nbvCV+oWT1xzy22rR+xc/7i+aY'
                'q1r4crafjutwT+e8qvVtWsr5p8ZMze8zZfw6a/cmxLM/X24bnnq3bY'
                've9N0b/QXCHq9Xvbm9qFo/jYW9hrv8aPxxy7q3DFstvqlW68UfmOnb'
                'biZ3+KLS0tACOS+LGLvlZQ4zZd1fHgy4eT6KcTmbnbrLq2MPfQM9Ht'
                'y56yqTxnicJXbV9PORcm9m/V/1U/vwzckFO95s1Nh2X/hWu8rxlbfW'
                'G9X1MPUxWll/cr6n/nxH8IfkyxZxmrdO/nw5x2Ju7JPjzEBn5x0IEE'
                'g0E/9z8hi/akW/qo3e44SG5RUCzpvWtE5sCN9av+ury/H+yzMuPmHt'
                'r+W1f7LH8mZTf2ndiwe9Thb9NR4TGjbn7v0d/l77avGCV+15dSvuJZ'
                'f85Ig75PUtMVrO6Hfn1n9yutcac1/fWpTR4yTlv+r4Sbe5u9x+359w'
                'XqyhLOjxhZRmi/xd6RdTlz2Re1VXv+ZRzK7S2/vMVfasSa1YlqDeH/'
                'qzNP7x5aM/5c/fPVJ8//imqiKOrj2FkTb/kxwFC2cfe1savu7/rtJP'
                'yq3M4TtWrDzyOeTQw03WDoyHD1fqH0n+2Lfo0XVlzv7TL8sz/jnpnl'
                'afyW88ka9/zdp9/max52+Z//9VH5gW7l+6b8veb+e/Fd2NT9hcW7/P'
                'zT67fOl/9tZZsgEA6Ux4DA==')
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertTrue(module.should_save_outlines.value)
        self.assertEqual(module.save_outlines.value, "NucleiOutlines")

    def test_04_02_load_v1(self):
        file = 'TUFUTEFCIDUuMCBNQVQtZmlsZSBQbGF0Zm9ybTogbnQsIENyZWF0ZWQgb246IE1vbiBBcHIgMDYgMTI6MzQ6MjQgMjAwOQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABSU0OAAAAoA0AAAYAAAAIAAAAAgAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAgAAABTZXR0aW5ncwUABAAYAAAAAQAAAMAAAABWYXJpYWJsZVZhbHVlcwAAAAAAAAAAAABWYXJpYWJsZUluZm9UeXBlcwAAAAAAAABNb2R1bGVOYW1lcwAAAAAAAAAAAAAAAABOdW1iZXJzT2ZWYXJpYWJsZXMAAAAAAABQaXhlbFNpemUAAAAAAAAAAAAAAAAAAABWYXJpYWJsZVJldmlzaW9uTnVtYmVycwBNb2R1bGVSZXZpc2lvbk51bWJlcnMAAABNb2R1bGVOb3RlcwAAAAAAAAAAAAAAAAAOAAAAYAUAAAYAAAAIAAAAAQAAAAAAAAAFAAAACAAAAAEAAAAWAAAAAQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm9uZQ4AAAA4AAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAYAAAABAAAAAAAAABAAAAAGAAAATnVjbGVpAAAOAAAAOAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAFAAAAAQAAAAAAAAAQAAAABQAAADEwLDQwAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAACAAAAAQAAAAAAAAAQAAIATm8AAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACwAAAAEAAAAAAAAAEAAAAAsAAABPdHN1IEdsb2JhbAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEAAAAAAAAAEAABADEAAAAOAAAASAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAARAAAAAQAAAAAAAAAQAAAAEQAAADAuMDAwMDAwLDEuMDAwMDAwAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAQAAAABAAAAAAAAABAABAAwLjAxDgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACQAAAAEAAAAAAAAAEAAAAAkAAABJbnRlbnNpdHkAAAAAAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACQAAAAEAAAAAAAAAEAAAAAkAAABJbnRlbnNpdHkAAAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAgAAAAEAAAAAAAAAEAACADEwAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAAQAAEANwAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACgAAAAEAAAAAAAAAEAAAAAoAAABEbyBub3QgdXNlAAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMAWWVzAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADADAuMAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm9uZQ4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAIAAAABAAAAAAAAABAAAgBObwAADgAAAEgFAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAABAAAAFgAAAAEAAAAAAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACgAAAAEAAAAAAAAAEAAAAAoAAABpbWFnZWdyb3VwAAAAAAAADgAAAEgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAEQAAAAEAAAAAAAAAEAAAABEAAABvYmplY3Rncm91cCBpbmRlcAAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAABIAAAABAAAAAAAAABAAAAASAAAAb3V0bGluZWdyb3VwIGluZGVwAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAKAAAAAQAAAAAAAAAQAAAACgAAAGltYWdlZ3JvdXAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAACgAAAABgAAAAgAAAABAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAA4AAABwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAEAAAAABAAAAAAAAABAAAABAAAAAY2VsbHByb2ZpbGVyLm1vZHVsZXMuaWRlbnRpZnlwcmltYXV0b21hdGljLklkZW50aWZ5UHJpbUF1dG9tYXRpYw4AAAAwAAAABgAAAAgAAAAJAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAAIAAQAWAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAAAAAABAAAAAAAAAAkAAAAIAAAAAAAAAAAA8D8OAAAAMAAAAAYAAAAIAAAACQAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAACAAEAAQAAAA4AAAAwAAAABgAAAAgAAAALAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAAQAAgAAAAAADgAAAFgAAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEAAAAAAAAADgAAACgAAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAAAAAAAAQAAAAEAAAAAAAAA'
        pipeline = tests.modules.load_pipeline(self, file)
        pipeline.add_listener(self.load_error_handler)
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.module(1)
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.threshold_algorithm, T.TM_OTSU)
        self.assertEqual(module.threshold_modifier, T.TM_GLOBAL)
        self.assertTrue(module.image_name == 'None')

    def test_04_03_load_v3(self):
        data = ('eJztWVtP2zAUTktBMMbG9rJJaJIfYWurpIAGaAI6uku3tlTQcRFim9'
                'u6rSfXrhKH0U1Ie9zP2k/aT1gc0jY1l4T0IpgaFKXn5HznO+f4Ehtn'
                'k4VM8jVYjqsgmyzEKpggkCeQV5heXwOUR8GWjiBHZcDoGsgyCj6YFK'
                'grQEusLS6taQmQUNVVJdgVSmcfWI+DZ4oyYT0nrTvsvBp35JDrFvIu'
                '4hzTqjGuRJSnjv6Pde9BHcMiQXuQmMjoULT0aVphhWaj/SrLyiZBOV'
                'h3G1tXzqwXkW5sV1pA53UenyKyi38gKYWW2Q46wQZm1ME7/mVtm5dx'
                'iVfUoTHTqUNIqoOoy5xLL+zfKx37yCV1e+Syn3VkTMv4BJdNSACuw2'
                'o7CuFP9fA31uVvTEnlkjZu0wM3K8Uh7gI65bE3p7DEQR3yUk34WfHw'
                'MyH5EXLOLBGE/cUf6sKHlEUnby/ecYlXyJoaXVJ7wKczmU/ZgHU/tF'
                'rNDy7chQsrOeaP7yqcV397IuUp5BSqQJNwkBadDaSwjkqc6c2+5T0h'
                '4VpXCzflPP3002kpfiFvc8ME7wgrQtL2M6j2knFaXO2JL8j8oMZV+4'
                'pqzg9X/bz6+aTkT8hbNUgpIgk/eUS68BERizbIeWlKilfIacoRNTBv'
                'uvK+6byiKf76W7/45brlGEVBxrmmnvP98sB9lOIW8uf5jfwrsXBA6/'
                'EXC1+EtI8I2WHf14+SsfzxQkuzxYhZp+tHamz1+KcWTZydG+9iC2kr'
                'FwLX/aWDq3ngVqT4hSxiOERQdwJbOluICZW14OE1R5dwdCnY7Ghu4z'
                'x2T8pPyCkGKOPANFDHT1D+Yec74uvmUy/5LvSTz8980k8+P+uU/6v9'
                'lgc6/meU7vEv5EJNRwiUCDQMe83fC3+QdcU+wtWa2EaeiA0TLSGXv2'
                'HOg2+Zjqo6M2m54+fg/s32XcOM196kiYAbvfMHaTdW/Gat2O0AgLV3'
                'RI0+1GGEG+FGuN5xmy6c3/+7dOaT8+F8l/Id4W4Hzus78ljp7ndCZi'
                'YnmKILH5K7lPcIN9z5a9DroRFuhBsGbjJ09f5C3v8K+6/K9ePiudI9'
                'LoRcQoQ0dCbO7/R43T5kMuKEwfL5KU88Y/1Muw587PMmD55NiWfzKh'
                '5cRpTjSrOhW2wmZ3XIcSmedrR5S5tsaeU6Tl3C665H2Pp7OHd9/eW6'
                'd9rj70YQvvDYRb5pD1zEqaDA/VZu1t7z19i3cgtq/w8+vUjz')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.load_error_handler)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm, T.TM_OTSU)
        self.assertTrue(module.threshold_modifier, T.TM_GLOBAL)
        self.assertEqual(module.two_class_otsu.value, I.O_THREE_CLASS)
        self.assertEqual(module.use_weighted_variance.value,
                         I.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground.value,
                         I.O_FOREGROUND)

    def test_04_04_load_v4(self):
        data = ('eJztWd1u0zAUdrtu2piGBjdwM+RLhLYo2VYxekPLyqCo7SZWhrjDTd3WyLWr'
                'xJlWnoBLHolLHodHIO6SNvG6JUvaiaGkitLj+Dvf+bFP7KRRadUrb2BR02Gj'
                '0trpEorhCUWiy61BCTKxDQ8tjATuQM5K8Mgi8IND4e4+NIolQy/t7cFdXX8F'
                'kh25WuOhe/n5DIAV97rqnnnv1rIn5wKnlE+xEIT17GVQAE+99l/ueYYsgtoU'
                'nyHqYHtK4bfXWJe3RsPJrQbvOBQ30SDY2T2azqCNLfu46wO92yfkAtNT8h0r'
                'LvjdPuJzYhPOPLynX22d8HKh8Mo4/N6YxiGnxEHGZSvQLvu/B9P+hRlxexTo'
                'v+nJhHXIOek4iEIyQL2JFVKfHqFvKaRvCVSblTGuHIHbVOyQZwtfiJ23F8gU'
                'cICE2Zd6DiL0rCh6pNx0TIpJPPtzIXwO7Hl+R/EuK7xSNvTtfT0Fvlavf2ok'
                'jPsXN2txcPkQLg+aPB7fdbio8fZE8VPKVdxFDhWwJgcbrBILm4Jbo7n5vaLg'
                '/MPHrXnXOON0XbFfysfCduA7ytuITvQsKl8qztD0VHxJ6oOu6eNj2/D+BOK3'
                'KL8LIVxB2mDEmVerIGy/lA/7iDFMd+Pke03BS7nGBGY2EaMUfseti/PiV+ua'
                'EROnznNDj4dT89XkDCex82VMO5PyzWs8+nzlCNwDEM6nlKscMi6gY3sLhzT1'
                '667rZcYX5tNn1ON58sUZ5/Pki7M++L/yV1zo+mEDhOe/lFt9C2NoUmTb47V2'
                'Gv4kz/PPmPT6cvt2LjcqzMQBfYuKw6w6eMQt3LO4wzrp+f+1caY+14oe7uCW'
                '+7m7zMd48ycTMkzPn2Rc8vY3dycwNgC6e1I8nEMcMlyGy3D3D1cO4OK+P5rW'
                'r8vycZ/8zXDJniOPQXgcSJk7ghKGrzxI7pPfGe5u68mi10MZLsNluPS41dz1'
                '+yf1/YXs/zXAM2vevwDheS9lE1M6tLj87mlpg/HHOVujHHUuv45pdfdvLfCh'
                'TPIMI3jKCk/5Oh7SwUyQ7mhouWyO4AMkiKnVvNYTt7Xit6pxXJvBG4xH3v1t'
                'bt0cfzXu03z8eZ2Eb6lwlW89AlfwIihxP8Dt8v38hv6+b0n7/wXQ1Cms')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(
                    isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(
                StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm, T.TM_OTSU)
        self.assertTrue(module.threshold_modifier, T.TM_GLOBAL)
        self.assertEqual(module.two_class_otsu.value, I.O_THREE_CLASS)
        self.assertEqual(module.use_weighted_variance.value,
                         I.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground.value,
                         I.O_FOREGROUND)

    def test_04_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9008

LoadImages:[module_num:1|svn_version:\'8947\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:
    Do you want to check image sets for missing or duplicate files?:Yes
    Do you want to group image sets by metadata?:No
    Do you want to exclude certain files?:No
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):
    What do you want to call this image in CellProfiler?:DNA
    What is the position of this image in each group?:1
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path\x3A:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

IdentifyPrimaryObjects:[module_num:2|svn_version:\'8981\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the identified primary objects:MyObjects
    Typical diameter of objects, in pixel units (Min,Max)\x3A:12,42
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:RobustBackground Global
    Threshold correction factor:1.2
    Lower and upper bounds on threshold\x3A:0.1,0.6
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Shape
    Method to draw dividing lines between clumped objects:Distance
    Size of smoothing filter\x3A:10
    Suppress local maxima within this distance\x3A:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:MyOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum size of local maxima?:Yes
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:MyBinaryImage
    Save outlines of the identified objects?:No
    Calculate the Laplacian of Gaussian threshold automatically?:Yes
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter\x3A :5
    How do you want to handle images with large numbers of objects?:Truncate
    Maximum # of objects\x3A:305

IdentifyPrimaryObjects:[module_num:3|svn_version:\'8981\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the identified primary objects:MyObjects
    Typical diameter of objects, in pixel units (Min,Max)\x3A:12,42
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:Yes
    Discard objects touching the border of the image?:No
    Select the thresholding method:Otsu Adaptive
    Threshold correction factor:1.2
    Lower and upper bounds on threshold\x3A:0.1,0.6
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Propagate
    Size of smoothing filter\x3A:10
    Suppress local maxima within this distance\x3A:7
    Speed up by using lower-resolution image to find local maxima?:No
    Name the outline image:MyOutlines
    Fill holes in identified objects?:No
    Automatically calculate size of smoothing filter?:No
    Automatically calculate minimum size of local maxima?:No
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:MyBinaryImage
    Save outlines of the identified objects?:Yes
    Calculate the Laplacian of Gaussian threshold automatically?:No
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:No
    Enter LoG filter diameter\x3A :5
    How do you want to handle images with large numbers of objects?:Erase
    Maximum # of objects\x3A:305
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(
                    isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.size_range.min, 12)
        self.assertEqual(module.size_range.max, 42)
        self.assertTrue(module.exclude_size)
        self.assertFalse(module.merge_objects)
        self.assertTrue(module.exclude_border_objects)
        self.assertEqual(module.threshold_algorithm, T.TM_ROBUST_BACKGROUND)
        self.assertEqual(module.threshold_modifier, T.TM_GLOBAL)
        self.assertAlmostEqual(module.threshold_correction_factor.value, 1.2)
        self.assertAlmostEqual(module.threshold_range.min, 0.1)
        self.assertAlmostEqual(module.threshold_range.max, 0.6)
        self.assertEqual(module.object_fraction.value, "0.01")
        self.assertEqual(module.unclump_method, ID.UN_SHAPE)
        self.assertEqual(module.watershed_method, ID.WA_SHAPE)
        self.assertEqual(module.smoothing_filter_size, 10)
        self.assertEqual(module.maxima_suppression_size, 7)
        self.assertFalse(module.should_save_outlines)
        self.assertEqual(module.save_outlines, "MyOutlines")
        self.assertEqual(module.fill_holes, ID.FH_THRESHOLDING)
        self.assertTrue(module.automatic_smoothing)
        self.assertTrue(module.automatic_suppression)
        self.assertEqual(module.manual_threshold, 0)
        self.assertEqual(module.binary_image, "MyBinaryImage")
        self.assertTrue(module.wants_automatic_log_threshold)
        self.assertAlmostEqual(module.manual_log_threshold.value, 0.5)
        self.assertEqual(module.two_class_otsu, I.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance, I.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground, I.O_FOREGROUND)
        self.assertTrue(module.wants_automatic_log_diameter)
        self.assertEqual(module.log_diameter, 5)
        self.assertEqual(module.limit_choice, ID.LIMIT_TRUNCATE)
        self.assertEqual(module.maximum_object_count, 305)

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.size_range.min, 12)
        self.assertEqual(module.size_range.max, 42)
        self.assertFalse(module.exclude_size)
        self.assertTrue(module.merge_objects)
        self.assertFalse(module.exclude_border_objects)
        self.assertEqual(module.threshold_algorithm, T.TM_OTSU)
        self.assertEqual(module.threshold_modifier, T.TM_ADAPTIVE)
        self.assertAlmostEqual(module.threshold_correction_factor.value, 1.2)
        self.assertAlmostEqual(module.threshold_range.min, 0.1)
        self.assertAlmostEqual(module.threshold_range.max, 0.6)
        self.assertEqual(module.object_fraction.value, "0.01")
        self.assertEqual(module.unclump_method, ID.UN_INTENSITY)
        self.assertEqual(module.watershed_method, ID.WA_PROPAGATE)
        self.assertEqual(module.smoothing_filter_size, 10)
        self.assertEqual(module.maxima_suppression_size, 7)
        self.assertTrue(module.should_save_outlines)
        self.assertEqual(module.save_outlines, "MyOutlines")
        self.assertEqual(module.fill_holes, ID.FH_NEVER)
        self.assertFalse(module.automatic_smoothing)
        self.assertFalse(module.automatic_suppression)
        self.assertEqual(module.manual_threshold, 0)
        self.assertEqual(module.binary_image, "MyBinaryImage")
        self.assertFalse(module.wants_automatic_log_threshold)
        self.assertAlmostEqual(module.manual_log_threshold.value, 0.5)
        self.assertEqual(module.two_class_otsu, I.O_THREE_CLASS)
        self.assertEqual(module.use_weighted_variance, I.O_ENTROPY)
        self.assertEqual(module.assign_middle_to_foreground, I.O_BACKGROUND)
        self.assertFalse(module.wants_automatic_log_diameter)
        self.assertEqual(module.log_diameter, 5)
        self.assertEqual(module.limit_choice, ID.LIMIT_ERASE)
        self.assertEqual(module.maximum_object_count, 305)

    # Missing tests for versions 6-9 (!)

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
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.image_name, "Channel0")
        self.assertEqual(module.object_name, "Cells")
        self.assertEqual(module.size_range.min, 15)
        self.assertEqual(module.size_range.max, 45)
        self.assertTrue(module.exclude_size)
        self.assertFalse(module.merge_objects)
        self.assertTrue(module.exclude_border_objects)
        self.assertEqual(module.unclump_method, ID.UN_INTENSITY)
        self.assertEqual(module.watershed_method, ID.WA_INTENSITY)
        self.assertTrue(module.automatic_smoothing)
        self.assertEqual(module.smoothing_filter_size, 11)
        self.assertTrue(module.automatic_suppression)
        self.assertEqual(module.maxima_suppression_size, 9)
        self.assertTrue(module.low_res_maxima)
        self.assertFalse(module.should_save_outlines)
        self.assertEqual(module.save_outlines, "CellOutlines")
        self.assertEqual(module.fill_holes, ID.FH_THRESHOLDING)
        self.assertTrue(module.wants_automatic_log_threshold)
        self.assertEqual(module.manual_log_threshold, .2)
        self.assertTrue(module.wants_automatic_log_diameter)
        self.assertEqual(module.log_diameter, 3)
        self.assertEqual(module.limit_choice, ID.LIMIT_NONE)
        self.assertEqual(module.maximum_object_count, 499)
        #
        self.assertEqual(module.threshold_scope, I.TS_ADAPTIVE)
        self.assertEqual(module.threshold_method, I.TM_OTSU)
        self.assertEqual(module.threshold_smoothing_choice, I.TSM_AUTOMATIC)
        self.assertEqual(module.threshold_smoothing_scale, 2.0)
        self.assertAlmostEqual(module.threshold_correction_factor, .80)
        self.assertAlmostEqual(module.threshold_range.min, 0.01)
        self.assertAlmostEqual(module.threshold_range.max, 0.90)
        self.assertAlmostEqual(module.object_fraction, 0.05)
        self.assertAlmostEqual(module.manual_threshold, 0.03)
        self.assertEqual(module.thresholding_measurement, "Metadata_Threshold")
        self.assertEqual(module.binary_image, "Segmentation")
        self.assertEqual(module.masking_objects, "Wells")
        self.assertEqual(module.two_class_otsu, I.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance, I.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground, I.O_FOREGROUND)
        self.assertEqual(module.adaptive_window_method, I.FI_IMAGE_SIZE)
        self.assertEqual(module.adaptive_window_size, 12)
        #
        # Test alternate settings using subsequent instances of IDPrimary
        #
        module = pipeline.modules()[5]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertFalse(module.exclude_size)
        self.assertTrue(module.merge_objects)
        self.assertFalse(module.exclude_border_objects)
        self.assertEqual(module.unclump_method, ID.UN_LOG)
        self.assertEqual(module.watershed_method, ID.WA_NONE)
        self.assertFalse(module.automatic_smoothing)
        self.assertFalse(module.automatic_suppression)
        self.assertFalse(module.low_res_maxima)
        self.assertTrue(module.should_save_outlines)
        self.assertEqual(module.fill_holes, ID.FH_NEVER)
        self.assertFalse(module.wants_automatic_log_threshold)
        self.assertFalse(module.wants_automatic_log_diameter)
        self.assertEqual(module.limit_choice, ID.LIMIT_ERASE)
        self.assertEqual(module.threshold_scope, I.TS_AUTOMATIC)
        self.assertEqual(module.threshold_method, I.TM_MCT)
        self.assertEqual(module.threshold_smoothing_choice, I.TSM_MANUAL)
        self.assertEqual(module.two_class_otsu, I.O_THREE_CLASS)
        self.assertEqual(module.use_weighted_variance, I.O_ENTROPY)
        self.assertEqual(module.assign_middle_to_foreground, I.O_BACKGROUND)
        self.assertEqual(module.adaptive_window_method, I.FI_CUSTOM)
        module = pipeline.modules()[6]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.unclump_method, ID.UN_NONE)
        self.assertEqual(module.watershed_method, ID.WA_PROPAGATE)
        self.assertEqual(module.limit_choice, ID.LIMIT_TRUNCATE)
        self.assertEqual(module.threshold_scope, I.TS_BINARY_IMAGE)
        self.assertEqual(module.threshold_method, I.TM_MOG)
        self.assertEqual(module.threshold_smoothing_choice, I.TSM_NONE)
        module = pipeline.modules()[7]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.unclump_method, ID.UN_SHAPE)
        self.assertEqual(module.watershed_method, ID.WA_SHAPE)
        self.assertEqual(module.threshold_scope, I.TS_GLOBAL)
        self.assertEqual(module.threshold_method, T.TM_BACKGROUND)
        module = pipeline.modules()[8]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.threshold_scope, I.TS_MANUAL)
        self.assertEqual(module.threshold_method, T.TM_KAPUR)
        module = pipeline.modules()[9]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.threshold_scope, I.TS_MEASUREMENT)
        self.assertEqual(module.threshold_method, T.TM_RIDLER_CALVARD)
        module = pipeline.modules()[10]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.threshold_scope, I.TS_PER_OBJECT)
        self.assertEqual(module.threshold_method, T.TM_ROBUST_BACKGROUND)
        self.assertEqual(module.rb_custom_choice, I.RB_DEFAULT)
        self.assertEqual(module.lower_outlier_fraction, .05)
        self.assertEqual(module.upper_outlier_fraction, .05)
        self.assertEqual(module.averaging_method, I.RB_MEAN)
        self.assertEqual(module.variance_method, I.RB_SD)
        self.assertEqual(module.number_of_deviations, 2)

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
                (I.RB_MEAN, I.RB_MEDIAN, I.RB_MODE),
                (I.RB_SD, I.RB_MAD, I.RB_MAD)):
            assert isinstance(module, ID.IdentifyPrimaryObjects)
            self.assertEqual(module.lower_outlier_fraction, .1)
            self.assertEqual(module.upper_outlier_fraction, .2)
            self.assertEqual(module.number_of_deviations, 2.5)
            self.assertEqual(module.averaging_method, averaging_method)
            self.assertEqual(module.variance_method, variance_method)

    def test_05_01_discard_large(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.manual_threshold.value = .3
        img = np.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 1)

    def test_05_02_keep_large(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.manual_threshold.value = .3
        img = np.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[25, 25], "The small object was not there")
        self.assertTrue(objects.segmented[100, 100], "The large object was filtered out")
        self.assertTrue(objects.unedited_segmented[25, 25], "The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100, 100], "The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object", "Location_Center_X")
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 2)

    def test_05_03_discard_small(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 40
        x.size_range.max = 60
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.manual_threshold.value = .3
        img = np.zeros((200, 200))
        draw_circle(img, (100, 100), 25, .5)
        draw_circle(img, (25, 25), 10, .5)
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        self.assertTrue(isinstance(location_center_x, np.ndarray))
        self.assertEqual(np.product(location_center_x.shape), 1)

    def test_05_02_discard_edge(self):
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.manual_threshold.value = .3
        img = np.zeros((100, 100))
        centers = [(50, 50), (10, 50), (50, 10), (90, 50), (50, 90)]
        present = [True, False, False, False, False]
        for center in centers:
            draw_circle(img, center, 15, .5)
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.manual_threshold.value = .3
        img = np.zeros((200, 200))
        centers = [(100, 100), (30, 100), (100, 30), (170, 100), (100, 170)]
        present = [True, False, False, False, False]
        for center in centers:
            draw_circle(img, center, 15, .5)
        mask = np.zeros((200, 200))
        mask[25:175, 25:175] = 1
        image = cpi.Image(img, mask)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
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
        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_MANUAL
        x.threshold_smoothing_choice.value = I.TSM_NONE
        x.manual_threshold.value = .5
        img = np.zeros((10, 10))
        img[4, 4] = 1
        img[5, 5] = 1
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        self.assertEqual(len(object_set.object_names), 1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented[img > 0] == 1))
        self.assertTrue(np.all(img[segmented == 1] > 0))

    def test_06_02_regression_adaptive_mask(self):
        """Regression test - mask all but one pixel / adaptive"""
        for o_alg in (I.O_WEIGHTED_VARIANCE, I.O_ENTROPY):
            x = ID.IdentifyPrimaryObjects()
            x.use_weighted_variance.value = o_alg
            x.object_name.value = "my_object"
            x.image_name.value = "my_image"
            x.exclude_size.value = False
            x.threshold_scope.value = T.TM_ADAPTIVE
            x.threshold_method.value = T.TM_OTSU
            np.random.seed(62)
            img = np.random.uniform(size=(100, 100))
            mask = np.zeros(img.shape, bool)
            mask[-1, -1] = True
            image = cpi.Image(img, mask)
            image_set_list = cpi.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            image_set.providers.append(cpi.VanillaImageProvider("my_image", image))
            object_set = cpo.ObjectSet()
            measurements = cpmeas.Measurements()
            pipeline = cellprofiler.pipeline.Pipeline()
            x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
            self.assertEqual(len(object_set.object_names), 1)
            self.assertTrue("my_object" in object_set.object_names)
            objects = object_set.get_objects("my_object")
            segmented = objects.segmented
            self.assertTrue(np.all(segmented == 0))

    def test_07_01_adaptive_otsu_small(self):
        """Test the function, get_threshold, using Otsu adaptive / small

        Use a small image (125 x 125) to break the image into four
        pieces, check that the threshold is different in each block
        and that there are four blocks broken at the 75 boundary
        """
        np.random.seed(0)
        image = np.zeros((120, 110))
        for i0, i1 in ((0, 60), (60, 120)):
            for j0, j1 in ((0, 55), (55, 110)):
                dmin = float(i0 * 2 + j0) / 500.0
                dmult = 1.0 - dmin
                # use the sine here to get a bimodal distribution of values
                r = np.random.uniform(0, np.pi * 2, (60, 55))
                rsin = (np.sin(r) + 1) / 2
                image[i0:i1, j0:j1] = dmin + rsin * dmult
        workspace, x = self.make_workspace(image)
        assert isinstance(x, ID.IdentifyPrimaryObjects)
        x.threshold_scope.value = T.TM_ADAPTIVE
        x.threshold_method.value = T.TM_OTSU
        threshold, global_threshold = x.get_threshold(
                cpi.Image(image), np.ones((120, 110), bool), workspace)
        self.assertTrue(threshold[0, 0] != threshold[0, 109])
        self.assertTrue(threshold[0, 0] != threshold[119, 0])
        self.assertTrue(threshold[0, 0] != threshold[119, 109])

    def test_07_02_adaptive_otsu_big(self):
        """Test the function, get_threshold, using Otsu adaptive / big

        Use a large image (525 x 525) to break the image into 100
        pieces, check that the threshold is different in each block
        and that boundaries occur where expected
        """
        np.random.seed(0)
        image = np.zeros((525, 525))
        blocks = []
        for i in range(10):
            for j in range(10):
                # the following makes a pattern of thresholds where
                # each square has a different threshold from its 8-connected
                # neighbors
                dmin = float((i % 2) * 2 + (j % 2)) / 8.0
                dmult = 1.0 - dmin

                def b(x):
                    return int(float(x) * 52.5)

                dim = ((b(i), b(i + 1)), (b(j), b(j + 1)))
                blocks.append(dim)
                ((i0, i1), (j0, j1)) = dim
                # use the sine here to get a bimodal distribution of values
                r = np.random.uniform(0, np.pi * 2, (i1 - i0, j1 - j0))
                rsin = (np.sin(r) + 1) / 2
                image[i0:i1, j0:j1] = dmin + rsin * dmult
        workspace, x = self.make_workspace(image)
        assert isinstance(x, ID.IdentifyPrimaryObjects)
        x.threshold_scope.value = T.TM_ADAPTIVE
        x.threshold_method.value = T.TM_OTSU
        threshold, global_threshold = x.get_threshold(
                cpi.Image(image), np.ones((525, 525), bool), workspace)

    def test_08_01_per_object_otsu(self):
        """Test get_threshold using Otsu per-object"""

        image = np.ones((20, 20)) * .08
        draw_circle(image, (5, 5), 2, .1)
        draw_circle(image, (15, 15), 3, .1)
        draw_circle(image, (15, 15), 2, .2)
        labels = np.zeros((20, 20), int)
        draw_circle(labels, (5, 5), 3, 1)
        draw_circle(labels, (15, 15), 3, 2)
        workspace, x = self.make_workspace(image, labels=labels)
        x.threshold_scope.value = I.TS_PER_OBJECT
        x.threshold_method.value = T.TM_OTSU
        threshold, global_threshold = x.get_threshold(cpi.Image(image),
                                                      np.ones((20, 20), bool),
                                                      workspace)
        t1 = threshold[5, 5]
        t2 = threshold[15, 15]
        self.assertTrue(t1 < .1)
        self.assertTrue(t2 > .1)
        self.assertTrue(t2 < .2)
        self.assertTrue(np.all(threshold[labels == 1] == threshold[5, 5]))
        self.assertTrue(np.all(threshold[labels == 2] == threshold[15, 15]))

    def test_08_02_per_object_otsu_run(self):
        """Test IdentifyPrimAutomatic per object through the Run function"""

        image = np.ones((20, 20)) * 0.06
        draw_circle(image, (5, 5), 5, .05)
        draw_circle(image, (5, 5), 2, .15)
        draw_circle(image, (15, 15), 5, .05)
        draw_circle(image, (15, 15), 2, .15)
        image = add_noise(image, .01)
        labels = np.zeros((20, 20), int)
        draw_circle(labels, (5, 5), 5, 1)
        draw_circle(labels, (15, 15), 5, 2)

        expected_labels = np.zeros((20, 20), int)
        draw_circle(expected_labels, (5, 5), 2, 1)
        draw_circle(expected_labels, (15, 15), 2, 2)

        workspace, x = self.make_workspace(image, labels=labels)
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_PER_OBJECT
        x.threshold_method.value = T.TM_OTSU
        x.threshold_correction_factor.value = 1.05
        x.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        # Do a little indexing trick so we can ignore which object got
        # which label
        self.assertNotEqual(labels[5, 5], labels[15, 15])
        indexes = np.array([0, labels[5, 5], labels[15, 15]])

        self.assertTrue(np.all(indexes[labels] == expected_labels))

    def test_08_03_per_objects_image_mask(self):
        image = np.ones((20, 20)) * 0.06
        draw_circle(image, (5, 5), 5, .05)
        draw_circle(image, (5, 5), 2, .15)
        image = add_noise(image, .01)
        mask = np.zeros((20, 20), bool)
        draw_circle(mask, (5, 5), 5, 1)

        expected_labels = np.zeros((20, 20), int)
        draw_circle(expected_labels, (5, 5), 2, 1)

        workspace, x = self.make_workspace(image, mask=mask)
        x.masking_objects.value = I.O_FROM_IMAGE
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_PER_OBJECT
        x.threshold_method.value = T.TM_OTSU
        x.threshold_correction_factor.value = 1.05
        x.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        self.assertTrue(np.all(labels == expected_labels))

    def test_09_01_small_images(self):
        """Test mixture of gaussians thresholding with few pixels

        Run MOG to see if it blows up, given 0-10 pixels"""
        r = np.random.RandomState()
        r.seed(91)
        image = r.uniform(size=(9, 11))
        ii, jj = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        ii, jj = ii.flatten(), jj.flatten()

        for threshold_method in (T.TM_BACKGROUND, T.TM_KAPUR, T.TM_MCT,
                                 T.TM_MOG, T.TM_OTSU, T.TM_RIDLER_CALVARD,
                                 T.TM_ROBUST_BACKGROUND):
            for i in range(11):
                mask = np.zeros(image.shape, bool)
                if i:
                    p = r.permutation(np.prod(image.shape))[:i]
                    mask[ii[p], jj[p]] = True
                workspace, x = self.make_workspace(image, mask)
                x.threshold_method.value = threshold_method
                x.threshold_scope.value = I.TS_GLOBAL
                l, g = x.get_threshold(cpi.Image(image), mask, workspace)
                v = image[mask]
                image = r.uniform(size=(9, 11))
                image[mask] = v
                l1, g1 = x.get_threshold(cpi.Image(image), mask, workspace)
                self.assertAlmostEqual(l1, l)

    # def test_09_02_mog_fly(self):
    #     """Test mixture of gaussians thresholding on the fly image"""
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.threshold_method.value = T.TM_MOG
    #     x.threshold_scope.value = I.TS_GLOBAL
    #     x.object_fraction.value = '0.10'
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.040)
    #     self.assertTrue(threshold < 0.050)
    #     x.object_fraction.value = '0.50'
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.0085)
    #     self.assertTrue(threshold < 0.0090)

    # def test_10_02_test_background_fly(self):
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.threshold_method.value = T.TM_BACKGROUND
    #     x.threshold_scope.value = I.TS_GLOBAL
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.020)
    #     self.assertTrue(threshold < 0.025)

    def test_10_03_test_background_mog(self):
        '''Test the background method with a mixture of gaussian distributions'''
        np.random.seed(103)
        image = np.random.normal(.2, .01, size=10000)
        ind = np.random.permutation(int(image.shape[0]))[:image.shape[0] / 5]
        image[ind] = np.random.normal(.5, .2, size=len(ind))
        image[image < 0] = 0
        image[image > 1] = 1
        image[0] = 0
        image[1] = 1
        image.shape = (100, 100)
        workspace, x = self.make_workspace(image)
        x.threshold_method.value = T.TM_BACKGROUND
        x.threshold_scope.value = I.TS_GLOBAL
        local_threshold, threshold = x.get_threshold(
                cpi.Image(image), np.ones(image.shape, bool), workspace)
        self.assertTrue(threshold > .18 * 2)
        self.assertTrue(threshold < .22 * 2)

    # def test_11_01_test_robust_background_fly(self):
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.threshold_scope.value = I.TS_GLOBAL
    #     x.threshold_method.value = T.TM_ROBUST_BACKGROUND
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.09)
    #     self.assertTrue(threshold < 0.095)

    # def test_12_01_test_ridler_calvard_background_fly(self):
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.threshold_scope.value = I.TS_GLOBAL
    #     x.threshold_method.value = T.TM_RIDLER_CALVARD
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.015)
    #     self.assertTrue(threshold < 0.020)


    # def test_13_01_test_kapur_background_fly(self):
    #     image = fly_image()
    #     workspace, x = self.make_workspace(image)
    #     x.threshold_scope.value = I.TS_GLOBAL
    #     x.threshold_method.value = T.TM_KAPUR
    #     local_threshold,threshold = x.get_threshold(
    #         cpi.Image(image), np.ones(image.shape,bool), workspace)
    #     self.assertTrue(threshold > 0.015)
    #     self.assertTrue(threshold < 0.020)

    def test_14_01_test_manual_background(self):
        """Test manual background"""
        workspace, x = self.make_workspace(np.zeros((10, 10)))
        x = ID.IdentifyPrimaryObjects()
        x.threshold_scope.value = T.TM_MANUAL
        x.manual_threshold.value = .5
        local_threshold, threshold = x.get_threshold(cpi.Image(np.zeros((10, 10))),
                                                     np.ones((10, 10), bool),
                                                     workspace)
        self.assertTrue(threshold == .5)
        self.assertTrue(threshold == .5)

    def test_15_01_test_binary_background(self):
        img = np.zeros((200, 200), np.float32)
        thresh = np.zeros((200, 200), bool)
        draw_circle(thresh, (100, 100), 50, True)
        draw_circle(thresh, (25, 25), 20, True)
        workspace, x = self.make_workspace(img, binary_image=thresh)
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = I.TS_BINARY_IMAGE
        x.run(workspace)
        count_ftr = I.C_COUNT + "_" + OBJECTS_NAME
        m = workspace.measurements
        self.assertTrue(m.has_feature(cpmeas.IMAGE, count_ftr))
        count = m.get_current_measurement(cpmeas.IMAGE, count_ftr)
        self.assertEqual(count, 2)

    def test_16_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = ID.IdentifyPrimaryObjects()
        oname = "my_object"
        x.object_name.value = oname
        x.image_name.value = "my_image"
        columns = x.get_measurement_columns(None)
        expected_columns = [
            (cpmeas.IMAGE, format % oname, coltype)
            for format, coltype in ((I.FF_COUNT, cpmeas.COLTYPE_INTEGER),
                                    (I.FF_FINAL_THRESHOLD, cpmeas.COLTYPE_FLOAT),
                                    (I.FF_ORIG_THRESHOLD, cpmeas.COLTYPE_FLOAT),
                                    (I.FF_WEIGHTED_VARIANCE, cpmeas.COLTYPE_FLOAT),
                                    (I.FF_SUM_OF_ENTROPIES, cpmeas.COLTYPE_FLOAT))]
        expected_columns += [(oname, feature, cpmeas.COLTYPE_FLOAT)
                             for feature in (I.M_LOCATION_CENTER_X,
                                             I.M_LOCATION_CENTER_Y)]
        expected_columns += [(oname, I.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER)]
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
        pixels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        x.exclude_size.value = True
        x.size_range.min = 6
        x.size_range.max = 50
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.watershed_method.value = ID.WA_INTENSITY
        x.threshold_scope.value = T.TM_MANUAL
        x.threshold_smoothing_choice.value = I.TSM_NONE
        x.manual_threshold.value = .05
        x.threshold_correction_factor.value = 1
        x.should_save_outlines.value = True
        x.save_outlines.value = "outlines"
        measurements = workspace.measurements
        x.run(workspace)
        my_objects = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(my_objects.segmented[3, 3] != 0)
        if my_objects.unedited_segmented[3, 3] == 2:
            unedited_segmented = my_objects.unedited_segmented
        else:
            unedited_segmented = np.array([0, 2, 1])[my_objects.unedited_segmented]
        self.assertTrue(np.all(unedited_segmented[mask] == expected[mask]))
        outlines = workspace.image_set.get_image("outlines",
                                                 must_be_binary=True)
        self.assertTrue(np.all(my_objects.segmented[outlines.pixel_data] > 0))

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
        pixels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        image = cpi.Image(pixels)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cpo.ObjectSet()

        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 4
        x.size_range.max = 50
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .1
        x.threshold_correction_factor.value = 1
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cpmeas.Measurements()
        workspace = Workspace(pipeline, x, image_set, object_set, measurements,
                              image_set_list)
        x.run(workspace)
        my_objects = object_set.get_objects("my_object")
        self.assertTrue(my_objects.segmented[3, 3] != 0)
        self.assertTrue(np.all(my_objects.segmented[mask] == expected[mask]))

    def test_18_01_truncate_objects(self):
        '''Set up a limit on the # of objects and exceed it'''
        for maximum_object_count in range(2, 5):
            pixels = np.zeros((20, 21))
            pixels[2:8, 2:8] = .5
            pixels[12:18, 2:8] = .5
            pixels[2:8, 12:18] = .5
            pixels[12:18, 12:18] = .5
            image = cpi.Image(pixels)
            image_set_list = cpi.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            image_set.add("my_image", image)
            object_set = cpo.ObjectSet()

            x = ID.IdentifyPrimaryObjects()
            x.object_name.value = "my_object"
            x.image_name.value = "my_image"
            x.exclude_size.value = False
            x.unclump_method.value = ID.UN_NONE
            x.watershed_method.value = ID.WA_NONE
            x.threshold_scope.value = T.TM_MANUAL
            x.manual_threshold.value = .25
            x.threshold_smoothing_choice.value = I.TSM_NONE
            x.threshold_correction_factor.value = 1
            x.limit_choice.value = ID.LIMIT_TRUNCATE
            x.maximum_object_count.value = maximum_object_count
            x.module_num = 1
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.add_module(x)
            measurements = cpmeas.Measurements()
            workspace = Workspace(pipeline, x, image_set, object_set, measurements,
                                  image_set_list)
            x.run(workspace)
            self.assertEqual(measurements.get_current_image_measurement(
                    "Count_my_object"), maximum_object_count)
            my_objects = object_set.get_objects("my_object")
            self.assertEqual(np.max(my_objects.segmented), maximum_object_count)
            self.assertEqual(np.max(my_objects.unedited_segmented), 4)

    def test_18_02_erase_objects(self):
        '''Set up a limit on the # of objects and exceed it - erasing objects'''
        maximum_object_count = 3
        pixels = np.zeros((20, 21))
        pixels[2:8, 2:8] = .5
        pixels[12:18, 2:8] = .5
        pixels[2:8, 12:18] = .5
        pixels[12:18, 12:18] = .5
        image = cpi.Image(pixels)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cpo.ObjectSet()

        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = T.TM_MANUAL
        x.threshold_smoothing_choice.value = I.TSM_NONE
        x.manual_threshold.value = .25
        x.threshold_correction_factor.value = 1
        x.limit_choice.value = ID.LIMIT_ERASE
        x.maximum_object_count.value = maximum_object_count
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cpmeas.Measurements()
        workspace = Workspace(pipeline, x, image_set, object_set, measurements,
                              image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement(
                "Count_my_object"), 0)
        my_objects = object_set.get_objects("my_object")
        self.assertTrue(np.all(my_objects.segmented == 0))
        self.assertEqual(np.max(my_objects.unedited_segmented), 4)

    def test_18_03_dont_erase_objects(self):
        '''Ask to erase objects, but don't'''
        maximum_object_count = 5
        pixels = np.zeros((20, 21))
        pixels[2:8, 2:8] = .5
        pixels[12:18, 2:8] = .5
        pixels[2:8, 12:18] = .5
        pixels[12:18, 12:18] = .5
        image = cpi.Image(pixels)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cpo.ObjectSet()

        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = T.TM_MANUAL
        x.threshold_smoothing_choice.value = I.TSM_NONE
        x.manual_threshold.value = .25
        x.threshold_correction_factor.value = 1
        x.limit_choice.value = ID.LIMIT_ERASE
        x.maximum_object_count.value = maximum_object_count
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cpmeas.Measurements()
        workspace = Workspace(pipeline, x, image_set, object_set, measurements,
                              image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement(
                "Count_my_object"), 4)
        my_objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(my_objects.segmented), 4)

    def test_19_01_threshold_by_measurement(self):
        '''Set threshold based on mean image intensity'''
        pixels = np.zeros((10, 10))
        pixels[2:6, 2:6] = .5

        image = cpi.Image(pixels)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("MyImage", image)
        object_set = cpo.ObjectSet()

        pipeline = cellprofiler.pipeline.Pipeline()
        measurements = cpmeas.Measurements()
        measurements.add_image_measurement("MeanIntensity_MyImage", np.mean(pixels))

        x = ID.IdentifyPrimaryObjects()
        x.object_name.value = "MyObject"
        x.image_name.value = "MyImage"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_scope.value = T.TM_MEASUREMENT
        x.threshold_smoothing_choice.value = I.TSM_NONE
        x.thresholding_measurement.value = "MeanIntensity_MyImage"
        x.threshold_correction_factor.value = 1
        x.module_num = 1
        pipeline.add_module(x)

        workspace = Workspace(pipeline, x, image_set, object_set, measurements,
                              image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement("Count_MyObject"), 1)
        self.assertEqual(measurements.get_current_image_measurement("Threshold_FinalThreshold_MyObject"),
                         np.mean(pixels))

    def test_20_01_threshold_smoothing_automatic(self):
        image = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, .4, .5, .4, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
        expected = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        workspace, module = self.make_workspace(image)
        assert isinstance(module, ID.IdentifyPrimaryObjects)
        module.exclude_size.value = False
        module.unclump_method.value = ID.UN_NONE
        module.watershed_method.value = ID.WA_NONE
        # MCT on this image is zero, so set the threshold at .225
        # with the threshold minimum (manual = no smoothing)
        module.threshold_scope.value = I.TS_GLOBAL
        module.threshold_method.value = T.TM_MCT
        module.threshold_range.min = .225
        module.threshold_smoothing_choice.value = I.TSM_AUTOMATIC
        module.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        np.testing.assert_array_equal(expected, labels)

    def test_20_02_threshold_smoothing_manual(self):
        image = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, .4, .5, .4, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
        expected = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        workspace, module = self.make_workspace(image)
        assert isinstance(module, ID.IdentifyPrimaryObjects)
        module.exclude_size.value = False
        module.unclump_method.value = ID.UN_NONE
        module.watershed_method.value = ID.WA_NONE
        module.threshold_scope.value = I.TS_GLOBAL
        module.threshold_method.value = T.TM_MCT
        module.threshold_range.min = .125
        module.threshold_smoothing_choice.value = I.TSM_MANUAL
        module.threshold_smoothing_scale.value = 3
        module.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        np.testing.assert_array_equal(expected, labels)

    def test_20_03_threshold_no_smoothing(self):
        image = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, .4, .5, .4, 0, 0],
                          [0, 0, .4, .4, .4, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
        expected = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        for ts in I.TS_MANUAL, I.TS_MEASUREMENT:
            workspace, module = self.make_workspace(image)
            assert isinstance(module, ID.IdentifyPrimaryObjects)
            module.exclude_size.value = False
            module.unclump_method.value = ID.UN_NONE
            module.watershed_method.value = ID.WA_NONE
            module.threshold_scope.value = ts
            module.manual_threshold.value = .125
            module.thresholding_measurement.value = MEASUREMENT_NAME
            workspace.measurements[cpmeas.IMAGE, MEASUREMENT_NAME] = .125
            module.threshold_smoothing_choice.value = I.TSM_MANUAL
            module.threshold_smoothing_scale.value = 3
            module.run(workspace)
            labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
            np.testing.assert_array_equal(expected, labels)


def add_noise(img, fraction):
    '''Add a fractional amount of noise to an image to make it look real'''
    np.random.seed(0)
    noise = np.random.uniform(low=1 - fraction / 2, high=1 + fraction / 2,
                              size=img.shape)
    return img * noise


def one_cell_image():
    img = np.zeros((25, 25))
    draw_circle(img, (10, 15), 5, .5)
    return add_noise(img, .01)


def two_cell_image():
    img = np.zeros((50, 50))
    draw_circle(img, (10, 35), 5, .8)
    draw_circle(img, (30, 15), 5, .6)
    return add_noise(img, .01)


def fly_image():
    from bioformats import load_image
    path = os.path.join(os.path.dirname(__file__), '../resources/01_POS002_D.TIF')
    return load_image(path)


def draw_circle(img, center, radius, value):
    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    distance = np.sqrt((x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1]))
    img[distance <= radius] = value


class TestWeightedVariance(unittest.TestCase):
    def test_01_masked_wv(self):
        output = T.weighted_variance(np.zeros((3, 3)),
                                     np.zeros((3, 3), bool), 1)
        self.assertEqual(output, 0)

    def test_02_zero_wv(self):
        output = T.weighted_variance(np.zeros((3, 3)),
                                     np.ones((3, 3), bool),
                                     np.ones((3, 3), bool))
        self.assertEqual(output, 0)

    def test_03_fg_0_bg_0(self):
        """Test all foreground pixels same, all background same, wv = 0"""
        img = np.zeros((4, 4))
        img[:, 2:4] = 1
        binary_image = img > .5
        output = T.weighted_variance(img, np.ones(img.shape, bool), binary_image)
        self.assertEqual(output, 0)

    def test_04_values(self):
        """Test with two foreground and two background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25 *2 + .25 *2)/4 = .25
        img = np.array([[1.0 / 16., 1.0 / 8.0], [1.0 / 4.0, 1.0 / 2.0]])
        binary_image = np.array([[False, False], [True, True]])
        output = T.weighted_variance(img, np.ones((2, 2), bool), binary_image)
        self.assertAlmostEqual(output, .25)

    def test_05_mask(self):
        """Test, masking out one of the background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25*2 + .25 *2)/4 = .25
        img = np.array([[1.0 / 16., 1.0 / 16.0, 1.0 / 8.0], [1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
        mask = np.array([[False, True, True], [False, True, True]])
        binary_image = np.array([[False, False, False], [True, True, True]])
        output = T.weighted_variance(img, mask, binary_image)
        self.assertAlmostEquals(output, .25)


class TestSumOfEntropies(unittest.TestCase):
    def test_01_all_masked(self):
        output = T.sum_of_entropies(np.zeros((3, 3)),
                                    np.zeros((3, 3), bool), 1)
        self.assertEqual(output, 0)

    def test_020_all_zero(self):
        """Can't take the log of zero, so all zero matrix = 0"""
        output = T.sum_of_entropies(np.zeros((4, 2)),
                                    np.ones((4, 2), bool),
                                    np.ones((4, 2), bool))
        self.assertAlmostEqual(output, 0)

    def test_03_fg_bg_equal(self):
        img = np.ones((128, 128))
        img[0:64, :] *= .15
        img[64:128, :] *= .85
        img[0, 0] = img[-1, 0] = 0
        img[0, -1] = img[-1, -1] = 1
        binary_mask = np.zeros(img.shape, bool)
        binary_mask[64:, :] = True
        #
        # You need one foreground and one background pixel to defeat a
        # divide-by-zero (that's appropriately handled)
        #
        one_of_each = np.zeros(img.shape, bool)
        one_of_each[0, 0] = one_of_each[-1, -1] = True
        output = T.sum_of_entropies(img, np.ones((128, 128), bool), binary_mask)
        ob = T.sum_of_entropies(img, one_of_each | ~binary_mask, binary_mask)
        of = T.sum_of_entropies(img, one_of_each | binary_mask, binary_mask)
        self.assertAlmostEqual(output, ob + of)

    def test_04_fg_bg_different(self):
        img = np.ones((128, 128))
        img[0:64, 0:64] *= .15
        img[0:64, 64:128] *= .3
        img[64:128, 0:64] *= .7
        img[64:128, 64:128] *= .85
        binary_mask = np.zeros(img.shape, bool)
        binary_mask[64:, :] = True
        one_of_each = np.zeros(img.shape, bool)
        one_of_each[0, 0] = one_of_each[-1, -1] = True
        output = T.sum_of_entropies(img, np.ones((128, 128), bool), binary_mask)
        ob = T.sum_of_entropies(img, one_of_each | ~binary_mask, binary_mask)
        of = T.sum_of_entropies(img, one_of_each | binary_mask, binary_mask)
        self.assertAlmostEqual(output, ob + of)
