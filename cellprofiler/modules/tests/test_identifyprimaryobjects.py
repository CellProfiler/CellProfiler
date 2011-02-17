"""test_identifyprimautomatic.py: test the IdentifyPrimAutomatic module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"
import os

import base64
import unittest
import numpy as np
import Image as PILImage
import scipy.ndimage
import tempfile
import StringIO
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.modules.identifyprimaryobjects as ID
import cellprofiler.modules.identify as I
import cellprofiler.cpmath.threshold as T
from cellprofiler.modules.injectimage import InjectImage
import cellprofiler.settings
import cellprofiler.cpimage
import cellprofiler.objects
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline
from cellprofiler.workspace import Workspace
import cellprofiler.modules.tests

class test_IdentifyPrimaryObjects(unittest.TestCase):
    def load_error_handler(self, caller, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.fail(event.error.message)

    def test_00_00_init(self):
        x = ID.IdentifyPrimAutomatic()
    
    def test_01_01_image_name(self):
        x = ID.IdentifyPrimAutomatic()
        x.setting(ID.IMAGE_NAME_VAR+1).set_value("MyImage")
        self.assertEqual(x.image_name, "MyImage")
    
    def test_01_02_object_name(self):
        x = ID.IdentifyPrimAutomatic()
        x.setting(ID.OBJECT_NAME_VAR+1).set_value("MyObject")
        self.assertEqual(x.object_name, "MyObject")
    
    def test_01_03_size_range(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.size_range.min,10)
        self.assertEqual(x.size_range.max,40)
        x.setting(ID.SIZE_RANGE_VAR+1).set_value("5,100")
        self.assertEqual(x.size_range.min,5)
        self.assertEqual(x.size_range.max,100)
    
    def test_01_04_exclude_size(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.exclude_size.value,"Default should be yes")
        x.setting(ID.EXCLUDE_SIZE_VAR+1).set_value("No")
        self.assertFalse(x.exclude_size.value)
        x.setting(ID.EXCLUDE_SIZE_VAR+1).set_value("Yes")
        self.assertTrue(x.exclude_size.value)
        
    def test_01_05_merge_objects(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertFalse(x.merge_objects.value, "Default should be no")
        x.setting(ID.MERGE_CHOICE_VAR+1).set_value("Yes")
        self.assertTrue(x.merge_objects.value)
        x.setting(ID.MERGE_CHOICE_VAR+1).set_value("No")
        self.assertFalse(x.merge_objects.value)
    
    def test_01_06_exclude_border_objects(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.exclude_border_objects.value,"Default should be yes")
        x.setting(ID.EXCLUDE_BORDER_OBJECTS_VAR+1).set_value("Yes")
        self.assertTrue(x.exclude_border_objects.value)
        x.setting(ID.EXCLUDE_BORDER_OBJECTS_VAR+1).set_value("No")
        self.assertFalse(x.exclude_border_objects.value)
    
    def test_01_07_threshold_method(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.threshold_method, T.TM_OTSU_GLOBAL, "Default should be Otsu global")
        x.setting(ID.THRESHOLD_METHOD_VAR+1).set_value(T.TM_BACKGROUND_GLOBAL)
        self.assertEqual(x.threshold_method, T.TM_BACKGROUND_GLOBAL)
    
    def test_01_07_01_threshold_modifier(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.threshold_modifier, T.TM_GLOBAL)
        x.setting(ID.THRESHOLD_METHOD_VAR+1).set_value(T.TM_BACKGROUND_ADAPTIVE)
        self.assertEqual(x.threshold_modifier, T.TM_ADAPTIVE)

    def test_01_07_02_threshold_algorithm(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.threshold_algorithm == T.TM_OTSU, "Default should be Otsu")
        x.setting(ID.THRESHOLD_METHOD_VAR+1).set_value(T.TM_BACKGROUND_GLOBAL)
        self.assertTrue(x.threshold_algorithm == T.TM_BACKGROUND)

    def test_01_08_threshold_range(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.threshold_range.min,0)
        self.assertEqual(x.threshold_range.max,1)
        x.setting(ID.THRESHOLD_RANGE_VAR+1).set_value(".2,.8")
        self.assertEqual(x.threshold_range.min,.2)
        self.assertEqual(x.threshold_range.max,.8)
    
    def test_01_09_threshold_correction_factor(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.threshold_correction_factor.value,1)
        x.setting(ID.THRESHOLD_CORRECTION_VAR+1).set_value("1.5")
        self.assertEqual(x.threshold_correction_factor.value,1.5)
    
    def test_01_10_object_fraction(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.object_fraction.value,'0.01')
        x.setting(ID.OBJECT_FRACTION_VAR+1).set_value("0.2")
        self.assertEqual(x.object_fraction.value,'0.2')
        
    def test_01_11_unclump_method(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.unclump_method.value, ID.UN_INTENSITY, "Default should be intensity, was %s"%(x.unclump_method))
        x.setting(ID.UNCLUMP_METHOD_VAR+1).set_value(ID.UN_LOG)
        self.assertEqual(x.unclump_method.value, ID.UN_LOG)

    def test_01_12_watershed_method(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertEqual(x.watershed_method.value, ID.WA_INTENSITY, "Default should be intensity")
        x.setting(ID.WATERSHED_VAR+1).set_value(ID.WA_SHAPE)
        self.assertEqual(x.watershed_method.value, ID.WA_SHAPE)
        
    def test_01_13_smoothing_filter_size(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.automatic_smoothing.value, "Default should be automatic")
        x.automatic_smoothing.value = False
        x.setting(ID.SMOOTHING_SIZE_VAR+1).set_value("10")
        self.assertFalse(x.automatic_smoothing.value)
        self.assertEqual(x.smoothing_filter_size,10)
    
    def test_01_14_maxima_suppression_size(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.automatic_suppression.value, "Default should be automatic")
        x.automatic_suppression.value= False
        x.setting(ID.MAXIMA_SUPPRESSION_SIZE_VAR+1).set_value("10")
        self.assertFalse(x.automatic_suppression.value)
        self.assertEqual(x.maxima_suppression_size.value,10)
        
    def test_01_15_use_low_res(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.low_res_maxima.value)
        x.setting(ID.LOW_RES_MAXIMA_VAR+1).set_value("No")
        self.assertFalse(x.low_res_maxima.value)
        x.setting(ID.LOW_RES_MAXIMA_VAR+1).set_value("Yes")
        self.assertTrue(x.low_res_maxima.value)
        
    def test_01_17_fill_holes(self):
        x = ID.IdentifyPrimAutomatic()
        self.assertTrue(x.fill_holes.value)
        x.setting(ID.FILL_HOLES_OPTION_VAR+1).value = cellprofiler.settings.NO
        self.assertFalse(x.fill_holes.value)
        x.setting(ID.FILL_HOLES_OPTION_VAR+1).value = cellprofiler.settings.YES
        self.assertTrue(x.fill_holes.value)
        
    def test_02_000_test_zero_objects(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min =.1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((25,25))
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,0)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),0)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),0)

    def test_02_001_test_zero_objects_wa_in_lo_in(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_INTENSITY
        x.unclump_method.value = ID.UN_INTENSITY
        img = np.zeros((25,25))
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_002_test_zero_objects_wa_di_lo_in(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_SHAPE
        x.unclump_method.value = ID.UN_INTENSITY
        img = np.zeros((25,25))
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))
        
    def test_02_003_test_zero_objects_wa_in_lo_sh(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_INTENSITY
        x.unclump_method.value = ID.UN_SHAPE
        img = np.zeros((25,25))
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_004_test_zero_objects_wa_di_lo_sh(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .1
        x.threshold_range.max = 1
        x.watershed_method.value = ID.WA_SHAPE
        x.unclump_method.value = ID.UN_SHAPE
        img = np.zeros((25,25))
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))

    def test_02_01_test_one_object(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = ID.WA_NONE
        img = one_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented[img>0] == 1))
        self.assertTrue(np.all(img[segmented==1] > 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .5)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,1)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>8)
        self.assertTrue(location_center_y[0]<12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>13)
        self.assertTrue(location_center_x[0]<16)
        columns = x.get_measurement_columns(pipeline)
        for object_name in (cpmeas.IMAGE, "my_object"):
            ocolumns =[x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_02_02_test_two_objects(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        img = two_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .6)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,2)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>8)
        self.assertTrue(location_center_y[0]<12)
        self.assertTrue(location_center_y[1]>28)
        self.assertTrue(location_center_y[1]<32)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>33)
        self.assertTrue(location_center_x[0]<37)
        self.assertTrue(location_center_x[1]>13)
        self.assertTrue(location_center_x[1]<16)

    def test_02_03_test_threshold_range(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .7
        x.threshold_range.max = 1
        x.threshold_correction_factor.value = .95
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        img = two_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .8)
        self.assertTrue(threshold > .6)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,1)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>8)
        self.assertTrue(location_center_y[0]<12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>33)
        self.assertTrue(location_center_x[0]<36)
    
    def test_02_04_fill_holes(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.fill_holes.value = True
        x.automatic_smoothing.value = False
        x.smoothing_filter_size.value = 0
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((40,40))
        draw_circle(img, (10,10), 7, .5)
        draw_circle(img, (30,30), 7, .5)
        img[10,10] = 0
        img[30,30] = 0
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10,10] > 0)
        self.assertTrue(objects.segmented[30,30] > 0)
        
    def test_02_05_dont_fill_holes(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.threshold_range.min = .7
        x.threshold_range.max = 1
        x.exclude_size.value = False
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((40,40))
        draw_circle(img, (10,10), 7, .5)
        draw_circle(img, (30,30), 7, .5)
        img[10,10] = 0
        img[30,30] = 0
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[10,10] == 0)
        self.assertTrue(objects.segmented[30,30] == 0)
    
    def test_02_06_test_watershed_shape_shape(self):
        """Identify by local_maxima:shape & intensity:shape
        
        Create an object whose intensity is high near the middle
        but has an hourglass shape, then segment it using shape/shape
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_SHAPE
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6, 0, 0, 0],
                           [ 0, 0, 0, 0,.7,.7,.7,.7,.7,.7,.7,.7, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.8,.9, 1, 1,.9,.8, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.7,.7,.7,.7,.7,.7,.7,.7, 0, 0, 0, 0],
                           [ 0, 0, 0,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),2)
    
    def test_02_07_test_watershed_shape_intensity(self):
        """Identify by local_maxima:shape & watershed:intensity
        
        Create an object with an hourglass shape to get two maxima, but
        set the intensities so that one maximum gets the middle portion
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_SHAPE
        x.watershed_method.value = ID.WA_INTENSITY
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.5,.5,.5,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.4,.4,.4,.4,.4,.4, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),2)
        self.assertEqual(objects.segmented[7,11],objects.segmented[7,4])
    
    def test_02_08_test_watershed_intensity_distance_single(self):
        """Identify by local_maxima:intensity & watershed:shape - one object
        
        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (4,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.7,.8,.9,.9,.8,.7, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .25, mode='constant')
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),1)
    
    def test_02_08_test_watershed_intensity_distance_triple(self):
        """Identify by local_maxima:intensity & watershed:shape - 3 objects w/o filter
        
        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.8,.8,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.7,.8,.9,.9,.8,.7, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.8,.8,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),3)
    
    def test_02_09_test_watershed_intensity_distance_filter(self):
        """Identify by local_maxima:intensity & watershed:shape - filtered
        
        Create an object with an hourglass shape and a peak in the middle.
        It should be segmented into a single object.
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 1
        x.automatic_smoothing.value = 1
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.7,.8,.9,.9,.8,.7, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.6,.7,.8,.8,.7,.6,.5, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.6,.7,.7,.6,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.6,.6,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),1)
    
    def test_02_10_test_watershed_intensity_distance_double(self):
        """Identify by local_maxima:intensity & watershed:shape - two objects
        
        Create an object with an hourglass shape and peaks in the top and
        bottom, but with a distribution of values that's skewed so that,
        by intensity, one of the images occupies the middle. The middle
        should be shared because the watershed is done using the distance
        transform.
        """
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_SHAPE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.9,.9,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.5,.5,.5,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.9,.9,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.4,.4,.4,.4,.4,.4, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .5, mode='constant')
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),2)
        self.assertNotEqual(objects.segmented[12,7],objects.segmented[4,7])
        
    def test_02_11_propagate(self):
        """Test the propagate unclump method"""
        x = ID.IdentifyPrimAutomatic()
        x.image_name.value = "my_image"
        x.object_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2,10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = 3.6
        x.automatic_suppression.value = False
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .3
        x.unclump_method.value = ID.UN_INTENSITY
        x.watershed_method.value = ID.WA_PROPAGATE
        img = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5,.5,.9,.9,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0,.5,.5,.5,.5, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0],
                           [ 0, 0, 0, 0, 0, 0,.5,.5,.5,.5,.5,.5,.5,.5, 0, 0],
                           [ 0, 0, 0, 0, 0,.5,.5,.5,.5, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0,.5,.5,.5,.5,.5, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.5,.5,.5,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.9,.9,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0],
                           [ 0, 0, 0,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4, 0, 0, 0],
                           [ 0, 0, 0, 0, 0,.4,.4,.4,.4,.4,.4, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                           ])
        # We do a little blur here so that there's some monotonic decrease
        # from the central peak
        img = scipy.ndimage.gaussian_filter(img, .5, mode='constant')
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(np.max(objects.segmented),2)
        # This point has a closer "crow-fly" distance to the upper object
        # but should be in the lower one because of the serpentine path
        self.assertEqual(objects.segmented[14,9],objects.segmented[9,9])
        
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
        
        img = fly_image()[300:600,300:600]
        image = cellprofiler.cpimage.Image(img)
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
                    image_set_list = cellprofiler.cpimage.ImageSetList()
                    image_set = image_set_list.get_image_set(0)
                    image_set.add(x.image_name.value, image)
                    object_set = cellprofiler.objects.ObjectSet()
                    measurements = cpmeas.Measurements()
                    x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
    
    def test_03_01_run_inside_pipeline(self):
        pass # No longer supported

    def test_04_01_load_matlab_12(self):
        """Test loading a Matlab version 12 IdentifyPrimAutomatic pipeline
        
        
        """
        old_r12_file = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgRGVjIDMxIDExOjQxOjUxIDIwMDggICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAuAEAAHicxVRdT8IwFO3GWEQNEYmJvu3RB2K2xAcfNTFRHgQihujjBoXUbC3ZWgM++TP8Of4Uf4ot7LMSNqfgTZpx7u45p/eytg4AMGsA6Py5w5cKllENsZJaAvchpQhPgirQwHGY/+BrYPvIdlw4sF0GAxBHlG/jMXmYT+NXd2TEXNixvXQxjw7zHOgH3XFEDF/30Ay6ffQKQTaisnv4ggJEcMgP9eVs7Euo5Fvn61NL5qCsmEMzlRf1lyCp11bU76bqD0J8TQxMqMECmOhc5Ojoko6+mNPQhagYvyrxBbbM1rkZ+ps5/EqGXwFPfHZFeGqGp4IO+Z1f3rz3pD4F7tKAGTcucWw3nneev5LRUYBVck5myyrE0zI8DZhnplWk35rUr8BtTCEOEJ2H+f/WuWKUeDZFww3obOo7Knpu/0pn2+fvTVl/zzVS+bJ9Is+ewIlP2DTRuc3RaUg6AhPnGQ7pQshAeASnqX1t+5m3/0Np/wITRl2E4bcGhN4MrP8f0vdQEf8jyV/g9ghiisbzno+89BmSvx89x1/lv5oreGXvzyJ++yV4Gme+nyx5jz+c7+ma+iii/BfqTY0Q'
        pipeline = cellprofiler.modules.tests.load_pipeline(self, old_r12_file)
        pipeline.add_listener(self.load_error_handler)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertTrue(isinstance(module,ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm,T.TM_OTSU)
        self.assertTrue(module.threshold_modifier,T.TM_GLOBAL)
        self.assertAlmostEqual(float(module.object_fraction.value),.01)
        self.assertEqual(module.object_name.value,"Nuclei")
        self.assertEqual(module.image_name.value,"Do not use")
        self.assertTrue(module.exclude_size.value)
        self.assertTrue(module.fill_holes.value)
        self.assertTrue(module.exclude_border_objects.value)
        self.assertTrue(module.automatic_smoothing.value)
        self.assertTrue(module.automatic_suppression.value)
        self.assertFalse(module.merge_objects.value)
        self.assertTrue(module.image_name == cellprofiler.settings.DO_NOT_USE)
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
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimAutomatic))
        self.assertTrue(module.should_save_outlines.value)
        self.assertEqual(module.save_outlines.value, "NucleiOutlines")
    
    def test_04_02_load_v1(self):
        file = 'TUFUTEFCIDUuMCBNQVQtZmlsZSBQbGF0Zm9ybTogbnQsIENyZWF0ZWQgb246IE1vbiBBcHIgMDYgMTI6MzQ6MjQgMjAwOQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABSU0OAAAAoA0AAAYAAAAIAAAAAgAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAgAAABTZXR0aW5ncwUABAAYAAAAAQAAAMAAAABWYXJpYWJsZVZhbHVlcwAAAAAAAAAAAABWYXJpYWJsZUluZm9UeXBlcwAAAAAAAABNb2R1bGVOYW1lcwAAAAAAAAAAAAAAAABOdW1iZXJzT2ZWYXJpYWJsZXMAAAAAAABQaXhlbFNpemUAAAAAAAAAAAAAAAAAAABWYXJpYWJsZVJldmlzaW9uTnVtYmVycwBNb2R1bGVSZXZpc2lvbk51bWJlcnMAAABNb2R1bGVOb3RlcwAAAAAAAAAAAAAAAAAOAAAAYAUAAAYAAAAIAAAAAQAAAAAAAAAFAAAACAAAAAEAAAAWAAAAAQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm9uZQ4AAAA4AAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAYAAAABAAAAAAAAABAAAAAGAAAATnVjbGVpAAAOAAAAOAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAFAAAAAQAAAAAAAAAQAAAABQAAADEwLDQwAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAACAAAAAQAAAAAAAAAQAAIATm8AAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACwAAAAEAAAAAAAAAEAAAAAsAAABPdHN1IEdsb2JhbAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEAAAAAAAAAEAABADEAAAAOAAAASAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAARAAAAAQAAAAAAAAAQAAAAEQAAADAuMDAwMDAwLDEuMDAwMDAwAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAQAAAABAAAAAAAAABAABAAwLjAxDgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACQAAAAEAAAAAAAAAEAAAAAkAAABJbnRlbnNpdHkAAAAAAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACQAAAAEAAAAAAAAAEAAAAAkAAABJbnRlbnNpdHkAAAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAgAAAAEAAAAAAAAAEAACADEwAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAAQAAEANwAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACgAAAAEAAAAAAAAAEAAAAAoAAABEbyBub3QgdXNlAAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMAWWVzAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADADAuMAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm9uZQ4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAIAAAABAAAAAAAAABAAAgBObwAADgAAAEgFAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAABAAAAFgAAAAEAAAAAAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACgAAAAEAAAAAAAAAEAAAAAoAAABpbWFnZWdyb3VwAAAAAAAADgAAAEgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAEQAAAAEAAAAAAAAAEAAAABEAAABvYmplY3Rncm91cCBpbmRlcAAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAABIAAAABAAAAAAAAABAAAAASAAAAb3V0bGluZWdyb3VwIGluZGVwAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAKAAAAAQAAAAAAAAAQAAAACgAAAGltYWdlZ3JvdXAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAACgAAAABgAAAAgAAAABAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAA4AAABwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAEAAAAABAAAAAAAAABAAAABAAAAAY2VsbHByb2ZpbGVyLm1vZHVsZXMuaWRlbnRpZnlwcmltYXV0b21hdGljLklkZW50aWZ5UHJpbUF1dG9tYXRpYw4AAAAwAAAABgAAAAgAAAAJAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAAIAAQAWAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAAAAAABAAAAAAAAAAkAAAAIAAAAAAAAAAAA8D8OAAAAMAAAAAYAAAAIAAAACQAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAACAAEAAQAAAA4AAAAwAAAABgAAAAgAAAALAAAAAAAAAAUAAAAIAAAAAQAAAAEAAAABAAAAAAAAAAQAAgAAAAAADgAAAFgAAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEAAAAAAAAADgAAACgAAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAAAAAAAAQAAAAEAAAAAAAAA'
        pipeline = cellprofiler.modules.tests.load_pipeline(self,file)
        pipeline.add_listener(self.load_error_handler)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertTrue(isinstance(module,ID.IdentifyPrimaryObjects))
        self.assertEqual(module.threshold_algorithm,T.TM_OTSU)
        self.assertEqual(module.threshold_modifier,T.TM_GLOBAL)
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
        self.assertEqual(len(pipeline.modules()),2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module,ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm,T.TM_OTSU)
        self.assertTrue(module.threshold_modifier,T.TM_GLOBAL)
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
        def callback(caller,event):
            self.assertFalse(
                isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(
            StringIO.StringIO(zlib.decompress(base64.b64decode(data))))        
        self.assertEqual(len(pipeline.modules()),2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module,ID.IdentifyPrimaryObjects))
        self.assertTrue(module.threshold_algorithm,T.TM_OTSU)
        self.assertTrue(module.threshold_modifier,T.TM_GLOBAL)
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
        def callback(caller,event):
            self.assertFalse(
                isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, ID.IdentifyPrimaryObjects))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.size_range.min, 12)
        self.assertEqual(module.size_range.max, 42)
        self.assertTrue(module.exclude_size)
        self.assertFalse(module.merge_objects)
        self.assertTrue(module.exclude_border_objects)
        self.assertEqual(module.threshold_method, T.TM_ROBUST_BACKGROUND_GLOBAL)
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
        self.assertTrue(module.fill_holes)
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
        self.assertEqual(module.threshold_method, T.TM_OTSU_ADAPTIVE)
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
        self.assertFalse(module.fill_holes)
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

    def test_05_01_discard_large(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((200,200))
        draw_circle(img,(100,100),25,.5)
        draw_circle(img,(25,25),10,.5)
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(objects.segmented[25,25],1,"The small object was not there")
        self.assertEqual(objects.segmented[100,100],0,"The large object was not filtered out")
        self.assertTrue(objects.small_removed_segmented[25,25]>0,"The small object was not in the small_removed label set")
        self.assertTrue(objects.small_removed_segmented[100,100]>0,"The large object was not in the small-removed label set")
        self.assertTrue(objects.unedited_segmented[25,25],"The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100,100],"The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),1)

    def test_05_02_keep_large(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((200,200))
        draw_circle(img,(100,100),25,.5)
        draw_circle(img,(25,25),10,.5)
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertTrue(objects.segmented[25,25],"The small object was not there")
        self.assertTrue(objects.segmented[100,100],"The large object was filtered out")
        self.assertTrue(objects.unedited_segmented[25,25],"The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100,100],"The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)

    def test_05_03_discard_small(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 40
        x.size_range.max = 60
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((200,200))
        draw_circle(img,(100,100),25,.5)
        draw_circle(img,(25,25),10,.5)
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        self.assertEqual(objects.segmented[25,25],0,"The small object was not filtered out")
        self.assertEqual(objects.segmented[100,100],1,"The large object was not present")
        self.assertTrue(objects.small_removed_segmented[25,25]==0,"The small object was in the small_removed label set")
        self.assertTrue(objects.small_removed_segmented[100,100]>0,"The large object was not in the small-removed label set")
        self.assertTrue(objects.unedited_segmented[25,25],"The small object was not in the unedited set")
        self.assertTrue(objects.unedited_segmented[100,100],"The large object was not in the unedited set")
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),1)

    def test_05_02_discard_edge(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((100,100))
        centers = [(50,50),(10,50),(50,10),(90,50),(50,90)]
        present = [ True,  False,  False,  False,  False]
        for center in centers:
            draw_circle(img,center,15,.5)
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        for center, p in zip(centers,present):
            if p:
                self.assertTrue(objects.segmented[center[0],center[1]] > 0)
                self.assertTrue(objects.small_removed_segmented[center[0],center[1]] > 0)
            else:
                self.assertTrue(objects.segmented[center[0],center[1]] == 0)
                self.assertTrue(objects.small_removed_segmented[center[0],center[1]] == 0)
            self.assertTrue(objects.unedited_segmented[center[0],center[1]] > 0)

    def test_05_03_discard_with_mask(self):
        """Check discard of objects that are on the border of a mask"""
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.size_range.min = 10
        x.size_range.max = 40
        x.watershed_method.value = ID.WA_NONE
        img = np.zeros((200,200))
        centers = [(100,100),(30,100),(100,30),(170,100),(100,170)]
        present = [ True,  False,  False,  False,  False]
        for center in centers:
            draw_circle(img,center,15,.5)
        mask = np.zeros((200,200))
        mask[25:175,25:175]=1
        image = cellprofiler.cpimage.Image(img,mask)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects("my_object")
        for center, p in zip(centers,present):
            if p:
                self.assertTrue(objects.segmented[center[0],center[1]] > 0)
                self.assertTrue(objects.small_removed_segmented[center[0],center[1]] > 0)
            else:
                self.assertTrue(objects.segmented[center[0],center[1]] == 0)
                self.assertTrue(objects.small_removed_segmented[center[0],center[1]] == 0)
            self.assertTrue(objects.unedited_segmented[center[0],center[1]] > 0)

    def test_06_01_regression_diagonal(self):
        """Regression test - was using one-connected instead of 3-connected structuring element"""
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .5
        img = np.zeros((10,10))
        img[4,4]=1
        img[5,5]=1
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(np.all(segmented[img>0] == 1))
        self.assertTrue(np.all(img[segmented==1] > 0))
        
    def test_06_02_regression_adaptive_mask(self):
        """Regression test - mask all but one pixel / adaptive"""
        for o_alg in (I.O_WEIGHTED_VARIANCE, I.O_ENTROPY):
            x = ID.IdentifyPrimAutomatic()
            x.use_weighted_variance.value = o_alg
            x.object_name.value = "my_object"
            x.image_name.value = "my_image"
            x.exclude_size.value = False
            x.threshold_method.value = T.TM_OTSU_ADAPTIVE
            np.random.seed(62)
            img = np.random.uniform(size=(100,100))
            mask = np.zeros(img.shape, bool)
            mask[-1,-1] = True
            image = cellprofiler.cpimage.Image(img, mask)
            image_set_list = cellprofiler.cpimage.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
            object_set = cellprofiler.objects.ObjectSet()
            measurements = cpmeas.Measurements()
            pipeline = cellprofiler.pipeline.Pipeline()
            x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
            self.assertEqual(len(object_set.object_names),1)
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
        image = np.zeros((120,110))
        for i0,i1 in ((0,60),(60,120)):
            for j0,j1 in ((0,55),(55,110)):
                dmin = float(i0 * 2 + j0) / 500.0
                dmult = 1.0-dmin
                # use the sine here to get a bimodal distribution of values
                r = np.random.uniform(0,np.pi*2,(60,55))
                rsin = (np.sin(r) + 1) / 2
                image[i0:i1,j0:j1] = dmin + rsin * dmult
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_OTSU_ADAPTIVE
        threshold,global_threshold = x.get_threshold(image, 
                                                     np.ones((120,110),bool),
                                                     None,None)
        for i0,i1 in ((0,60),(60,120)):
            for j0,j1 in ((0,55),(55,110)):
                self.assertTrue(np.all(threshold[i0:i1,j0:j1] == threshold[i0,j0]))
        self.assertTrue(threshold[0,0] != threshold[0,109])
        self.assertTrue(threshold[0,0] != threshold[119,0])
        self.assertTrue(threshold[0,0] != threshold[119,109])

    def test_07_02_adaptive_otsu_big(self): 
        """Test the function, get_threshold, using Otsu adaptive / big
        
        Use a large image (525 x 525) to break the image into 100
        pieces, check that the threshold is different in each block
        and that boundaries occur where expected
        """
        np.random.seed(0)
        image = np.zeros((525,525))
        blocks = []
        for i in range(10):
            for j in range(10):
                # the following makes a pattern of thresholds where
                # each square has a different threshold from its 8-connected
                # neighbors
                dmin = float((i % 2) * 2 + (j%2)) / 8.0
                dmult = 1.0-dmin
                def b(x):
                    return int(float(x)*52.5)
                dim = ((b(i),b(i+1)),(b(j),b(j+1)))
                blocks.append(dim)
                ((i0,i1),(j0,j1)) = dim
                # use the sine here to get a bimodal distribution of values
                r = np.random.uniform(0,np.pi*2,(i1-i0,j1-j0))
                rsin = (np.sin(r) + 1) / 2
                image[i0:i1,j0:j1] = dmin + rsin * dmult
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_OTSU_ADAPTIVE
        threshold,global_threshold = x.get_threshold(image, 
                                                     np.ones((525,525),bool),
                                                     None,None)
        for ((i0,i1),(j0,j1)) in blocks:
                self.assertTrue(np.all(threshold[i0:i1,j0:j1] == threshold[i0,j0]))
    
    def test_08_01_per_object_otsu(self):
        """Test get_threshold using Otsu per-object"""
        
        image = np.ones((20,20)) * .08
        draw_circle(image,(5,5),2,.1)
        draw_circle(image,(15,15),3,.1)
        draw_circle(image,(15,15),2,.2)
        labels = np.zeros((20,20),int)
        draw_circle(labels,(5,5),3,1)
        draw_circle(labels,(15,15),3,2)
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_OTSU_PER_OBJECT
        threshold, global_threshold = x.get_threshold(image, 
                                                      np.ones((20,20), bool),
                                                      labels,None)
        t1 = threshold[5,5]
        t2 = threshold[15,15]
        self.assertTrue(t1 < .1)
        self.assertTrue(t2 > .1)
        self.assertTrue(t2 < .2)
        self.assertTrue(np.all(threshold[labels==1] == threshold[5,5]))
        self.assertTrue(np.all(threshold[labels==2] == threshold[15,15]))
    
    def test_08_02_per_object_otsu_run(self):
        """Test IdentifyPrimAutomatic per object through the Run function"""
        
        image = np.ones((20,20))*0.06
        draw_circle(image,(5,5),5,.05)
        draw_circle(image,(5,5),2,.15)
        draw_circle(image,(15,15),5,.05)
        draw_circle(image,(15,15),2,.15)
        image = add_noise(image, .01)
        labels = np.zeros((20,20),int)
        draw_circle(labels,(5,5),5,1)
        draw_circle(labels,(15,15),5,2)
        objects = cellprofiler.objects.Objects()
        objects.segmented = labels
        image = cellprofiler.cpimage.Image(image,
                                           masking_objects = objects)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.objects.ObjectSet()
        
        expected_labels = np.zeros((20,20),int)
        draw_circle(expected_labels,(5,5),2,1)
        draw_circle(expected_labels,(15,15),2,2)
        
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_OTSU_PER_OBJECT
        x.threshold_correction_factor.value = 1.05
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cpmeas.Measurements()
        workspace = Workspace(pipeline, x, image_set, object_set, measurements, 
                              image_set_list)
        x.run(workspace)
        labels = object_set.get_objects("my_object").segmented
        # Do a little indexing trick so we can ignore which object got
        # which label
        self.assertNotEqual(labels[5,5], labels[15,15])
        indexes = np.array([0, labels[5,5], labels[15,15]])


        self.assertTrue(np.all(indexes[labels] == expected_labels))
    
    def test_09_01_mog(self):
        """Test mixture of gaussians thresholding with few pixels
        
        Run MOG to see if it blows up, given 0-10 pixels"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_MOG_GLOBAL
        for i in range(11):
            if i:
                image = np.array(range(i),float) / float(i)
            else:
                image = np.array((0,))
            x.get_threshold(image, np.ones((i,),bool),None,None)

    def test_09_02_mog_fly(self):
        """Test mixture of gaussians thresholding on the fly image"""
        image = fly_image()
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_MOG_GLOBAL
        x.object_fraction.value = '0.10'
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.036)
        self.assertTrue(threshold < 0.040)
        x.object_fraction.value = '0.20'
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.0084)
        self.assertTrue(threshold < 0.0088)
        x.object_fraction.value = '0.50'
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.0082)
        self.assertTrue(threshold < 0.0086)
    
    def test_10_01_test_background(self):
        """Test simple mode background for problems with small images"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_BACKGROUND_GLOBAL
        for i in range(11):
            if i:
                image = np.array(range(i),float) / float(i)
            else:
                image = np.array((0,))
            x.get_threshold(image, np.ones((i,),bool),None,None)
    
    def test_10_02_test_background_fly(self):
        image = fly_image()
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_BACKGROUND_GLOBAL
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.030)
        self.assertTrue(threshold < 0.032)
        
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
        image.shape = (100,100)
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_BACKGROUND_GLOBAL
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > .18 * 2)
        self.assertTrue(threshold < .22 * 2)
        
    def test_11_01_test_robust_background(self):
        """Test robust background for problems with small images"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_ROBUST_BACKGROUND_GLOBAL
        for i in range(11):
            if i:
                image = np.array(range(i),float) / float(i)
            else:
                image = np.array((0,))
            x.get_threshold(image, np.ones((i,),bool),None,None)
    
    def test_11_02_test_robust_background_fly(self):
        image = fly_image()
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_ROBUST_BACKGROUND_GLOBAL
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.054)
        self.assertTrue(threshold < 0.056)
        
    def test_12_01_test_ridler_calvard_background(self):
        """Test ridler-calvard background for problems with small images"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_RIDLER_CALVARD_GLOBAL
        for i in range(11):
            if i:
                image = np.array(range(i),float) / float(i)
            else:
                image = np.array((0,))
            x.get_threshold(image, np.ones((i,),bool),None,None)

    def test_12_02_test_ridler_calvard_background_fly(self):
        image = fly_image()
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_RIDLER_CALVARD_GLOBAL
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.017)
        self.assertTrue(threshold < 0.019)
        
        
    def test_13_01_test_kapur_background(self):
        """Test kapur background for problems with small images"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_KAPUR_GLOBAL
        for i in range(11):
            if i:
                image = np.array(range(i),float) / float(i)
            else:
                image = np.array((0,))
            x.get_threshold(image, np.ones((i,),bool),None,None)
    
    def test_13_02_test_kapur_background_fly(self):
        image = fly_image()
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_KAPUR_GLOBAL
        local_threshold,threshold = x.get_threshold(image, np.ones(image.shape,bool),None,None)
        self.assertTrue(threshold > 0.015)
        self.assertTrue(threshold < 0.020)
    
    def test_14_01_test_manual_background(self):
        """Test manual background"""
        x = ID.IdentifyPrimAutomatic()
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .5
        local_threshold,threshold = x.get_threshold(np.zeros((10,10)), 
                                                    np.ones((10,10),bool),
                                                    None,None)
        self.assertTrue(threshold == .5)
        self.assertTrue(threshold == .5)
    
    def test_15_01_test_binary_background(self):
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_BINARY_IMAGE
        x.binary_image.value = "my_threshold"
        img = np.zeros((200,200),np.float32)
        thresh = np.zeros((200,200),bool)
        draw_circle(thresh,(100,100),50,True)
        draw_circle(thresh,(25,25),20,True)
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        threshold = cellprofiler.cpimage.Image(thresh)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_threshold",threshold))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,2)
    
    def test_16_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = ID.IdentifyPrimAutomatic()
        oname = "my_object"
        x.object_name.value = oname
        x.image_name.value = "my_image"
        columns = x.get_measurement_columns(None)
        expected_columns = [(cpmeas.IMAGE, format%oname, coltype )
                            for format,coltype in ((I.FF_COUNT, cpmeas.COLTYPE_INTEGER),
                                                   (ID.FF_FINAL_THRESHOLD, cpmeas.COLTYPE_FLOAT),
                                                   (ID.FF_ORIG_THRESHOLD, cpmeas.COLTYPE_FLOAT),
                                                   (ID.FF_WEIGHTED_VARIANCE, cpmeas.COLTYPE_FLOAT),
                                                   (ID.FF_SUM_OF_ENTROPIES, cpmeas.COLTYPE_FLOAT))]
        expected_columns += [(oname, feature, cpmeas.COLTYPE_FLOAT)
                             for feature in (I.M_LOCATION_CENTER_X,
                                             I.M_LOCATION_CENTER_Y)]
        expected_columns += [(oname, I.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER)]
        self.assertEqual(len(columns), len(expected_columns))
        for column in columns:
            self.assertTrue(any(all([colval==exval for colval, exval in zip(column, expected)])
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
        pixels = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,2,1,1,1,1,1,1,2,0,0],
                           [0,0,2,1,2,2,2,2,1,2,0,0],
                           [0,0,2,1,2,9,2,2,1,2,0,0],
                           [0,0,2,1,2,2,2,2,1,2,0,0],
                           [0,0,2,1,1,1,1,1,1,2,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],                           
                           [0,0,2,2,1,2,2,2,2,2,0,0],                           
                           [0,0,2,2,1,2,2,2,2,2,0,0],                           
                           [0,0,2,2,1,2,2,2,2,2,0,0],                           
                           [0,0,2,2,2,2,2,2,2,2,0,0],                           
                           [0,0,2,2,2,2,2,2,9,9,0,0],                           
                           [0,0,2,2,2,2,2,2,9,9,0,0],                           
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0]], float) / 10.0
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,1,1,1,1,1,1,2,0,0],
                             [0,0,2,1,1,1,1,1,1,2,0,0],
                             [0,0,2,1,1,1,1,1,1,2,0,0],
                             [0,0,2,1,1,1,1,1,1,2,0,0],
                             [0,0,2,1,1,1,1,1,1,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        mask = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,1,1,1,1,1,1,0,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,0,1,1,1,1,0,1,0,0],
                         [0,0,1,0,1,1,1,1,0,1,0,0],
                         [0,0,1,0,1,1,1,1,0,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,0,1,1,1,1,1,1,0,0,0],                           
                         [0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0]], bool)
        image = cellprofiler.cpimage.Image(pixels)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.objects.ObjectSet()
        
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = True
        x.size_range.min = 6
        x.size_range.max = 50
        x.maxima_suppression_size.value = 3
        x.automatic_suppression.value = False
        x.watershed_method.value = ID.WA_INTENSITY
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .05
        x.threshold_correction_factor.value = 1
        x.should_save_outlines.value = True
        x.save_outlines.value = "outlines"
        x.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(x)
        measurements = cpmeas.Measurements()
        workspace = Workspace(pipeline, x, image_set, object_set, measurements, 
                              image_set_list)
        x.run(workspace)
        my_objects = object_set.get_objects("my_object")
        self.assertTrue(my_objects.segmented[3,3] != 0)
        if my_objects.unedited_segmented[3,3] == 2:
            unedited_segmented = my_objects.unedited_segmented
        else:
            unedited_segmented = np.array([0,2,1])[my_objects.unedited_segmented]
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
        pixels = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,2,2,2,2,2,2,2,2,0,0],
                           [0,0,3,0,0,0,0,0,0,3,0,0],
                           [0,0,3,0,0,0,0,0,0,3,0,0],
                           [0,0,3,0,0,9,2,0,0,3,0,0],
                           [0,0,3,0,0,0,0,0,0,3,0,0],
                           [0,0,3,0,0,0,0,0,0,3,0,0],
                           [0,0,3,2,2,2,2,2,2,2,0,0],                           
                           [0,0,3,2,2,2,2,2,2,2,0,0],                           
                           [0,0,3,2,2,2,2,2,2,2,0,0],                           
                           [0,0,2,2,2,2,2,2,9,2,0,0],                           
                           [0,0,2,2,2,2,2,2,2,2,0,0],                           
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0]], float) / 10.0
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],
                             [0,0,2,0,0,0,0,0,0,2,0,0],
                             [0,0,2,0,0,0,0,0,0,2,0,0],
                             [0,0,2,0,0,1,1,0,0,2,0,0],
                             [0,0,2,0,0,0,0,0,0,2,0,0],
                             [0,0,2,0,0,0,0,0,0,2,0,0],
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,2,2,2,2,2,2,2,2,0,0],                           
                             [0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        mask = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,1,1,1,1,1,1,0,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,0,0,1,1,0,0,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,0,0,0,0,0,0,1,0,0],
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,1,1,1,1,1,1,1,1,0,0],                           
                         [0,0,0,1,1,1,1,1,1,0,0,0],                           
                         [0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0]], bool)
        image = cellprofiler.cpimage.Image(pixels)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.objects.ObjectSet()
        
        x = ID.IdentifyPrimAutomatic()
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
        self.assertTrue(my_objects.segmented[3,3] != 0)
        if my_objects.unedited_segmented[3,3] == 2:
            unedited_segmented = my_objects.unedited_segmented
        else:
            unedited_segmented = np.array([0,2,1])[my_objects.unedited_segmented]
        self.assertTrue(np.all(unedited_segmented[mask] == expected[mask]))
    
    def test_18_01_truncate_objects(self):
        '''Set up a limit on the # of objects and exceed it'''
        for maximum_object_count in range(2,5):
            pixels = np.zeros((20,21))
            pixels[2:8,2:8] = .5
            pixels[12:18,2:8] = .5
            pixels[2:8,12:18] = .5
            pixels[12:18,12:18] = .5
            image = cellprofiler.cpimage.Image(pixels)
            image_set_list = cellprofiler.cpimage.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            image_set.add("my_image", image)
            object_set = cellprofiler.objects.ObjectSet()
            
            x = ID.IdentifyPrimAutomatic()
            x.object_name.value = "my_object"
            x.image_name.value = "my_image"
            x.exclude_size.value = False
            x.unclump_method.value = ID.UN_NONE
            x.watershed_method.value = ID.WA_NONE
            x.threshold_method.value = T.TM_MANUAL
            x.manual_threshold.value = .25
            x.threshold_correction_factor.value = 1
            x.limit_choice = ID.LIMIT_TRUNCATE
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
        pixels = np.zeros((20,21))
        pixels[2:8,2:8] = .5
        pixels[12:18,2:8] = .5
        pixels[2:8,12:18] = .5
        pixels[12:18,12:18] = .5
        image = cellprofiler.cpimage.Image(pixels)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.objects.ObjectSet()
        
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .25
        x.threshold_correction_factor.value = 1
        x.limit_choice = ID.LIMIT_ERASE
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
        pixels = np.zeros((20,21))
        pixels[2:8,2:8] = .5
        pixels[12:18,2:8] = .5
        pixels[2:8,12:18] = .5
        pixels[12:18,12:18] = .5
        image = cellprofiler.cpimage.Image(pixels)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("my_image", image)
        object_set = cellprofiler.objects.ObjectSet()
        
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "my_object"
        x.image_name.value = "my_image"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_MANUAL
        x.manual_threshold.value = .25
        x.threshold_correction_factor.value = 1
        x.limit_choice = ID.LIMIT_ERASE
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
        pixels = np.zeros((10,10))
        pixels[2:6,2:6] = .5
        
        image = cellprofiler.cpimage.Image(pixels)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("MyImage", image)
        object_set = cellprofiler.objects.ObjectSet()

        pipeline = cellprofiler.pipeline.Pipeline()
        measurements = cpmeas.Measurements()
        measurements.add_image_measurement("MeanIntensity_MyImage", np.mean(pixels))
        
        x = ID.IdentifyPrimAutomatic()
        x.object_name.value = "MyObject"
        x.image_name.value = "MyImage"
        x.exclude_size.value = False
        x.unclump_method.value = ID.UN_NONE
        x.watershed_method.value = ID.WA_NONE
        x.threshold_method.value = T.TM_MEASUREMENT
        x.thresholding_measurement.value = "MeanIntensity_MyImage"
        x.threshold_correction_factor.value = 1
        x.module_num = 1
        pipeline.add_module(x)
        
        workspace = Workspace(pipeline, x, image_set, object_set, measurements, 
                              image_set_list)
        x.run(workspace)
        self.assertEqual(measurements.get_current_image_measurement("Count_MyObject"),1)
        self.assertEqual(measurements.get_current_image_measurement("Threshold_FinalThreshold_MyObject"),np.mean(pixels))
        
def add_noise(img, fraction):
    '''Add a fractional amount of noise to an image to make it look real'''
    np.random.seed(0)
    noise = np.random.uniform(low=1-fraction/2, high=1+fraction/2,
                                 size=img.shape)
    return img * noise

def one_cell_image():
    img = np.zeros((25,25))
    draw_circle(img,(10,15),5, .5)
    return add_noise(img,.01)

def two_cell_image():
    img = np.zeros((50,50))
    draw_circle(img,(10,35),5, .8)
    draw_circle(img,(30,15),5, .6)
    return add_noise(img,.01)

def fly_image():
    file = os.path.join(cellprofiler.modules.tests.example_images_directory(),
                        'ExampleFlyImages','01_POS002_D.TIF')
    img = np.asarray(PILImage.open(file))
    img = img.astype(float) / 255.0
    return img
    
def draw_circle(img,center,radius,value):
    x,y=np.mgrid[0:img.shape[0],0:img.shape[1]]
    distance = np.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
    img[distance<=radius]=value

class TestWeightedVariance(unittest.TestCase):
    def test_01_masked_wv(self):
        output = T.weighted_variance(np.zeros((3,3)), 
                                      np.zeros((3,3),bool), 1)
        self.assertEqual(output, 0)
    
    def test_02_zero_wv(self):
        output = T.weighted_variance(np.zeros((3,3)),
                                      np.ones((3,3),bool),1)
        self.assertEqual(output, 0)
    
    def test_03_fg_0_bg_0(self):
        """Test all foreground pixels same, all background same, wv = 0"""
        img = np.zeros((4,4))
        img[:,2:4]=1
        output = T.weighted_variance(img, np.ones(img.shape,bool),.5)
        self.assertEqual(output,0)
    
    def test_04_values(self):
        """Test with two foreground and two background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25 *2 + .25 *2)/4 = .25
        img = np.array([[1.0/16.,1.0/8.0],[1.0/4.0,1.0/2.0]])
        threshold = 3.0/16.0
        output = T.weighted_variance(img, np.ones((2,2),bool), threshold)
        self.assertAlmostEqual(output,.25)
    
    def test_05_mask(self):
        """Test, masking out one of the background values"""
        #
        # The log of this array is [-4,-3],[-2,-1] and
        # the variance should be (.25*2 + .25 *2)/4 = .25
        img = np.array([[1.0/16.,1.0/16.0,1.0/8.0],[1.0/4.0,1.0/4.0,1.0/2.0]])
        mask = np.array([[False,True,True],[False,True,True]])
        threshold = 3.0/16.0
        output = T.weighted_variance(img, mask, threshold)
        self.assertAlmostEquals(output,.25)

class TestSumOfEntropies(unittest.TestCase):
    def test_01_all_masked(self):
        output = T.sum_of_entropies(np.zeros((3,3)), 
                                     np.zeros((3,3),bool), 1)
        self.assertEqual(output,0)
    
    def test_020_all_zero(self):
        """Can't take the log of zero, so all zero matrix = 0"""
        output = T.sum_of_entropies(np.zeros((4,2)),np.ones((4,2),bool),1)
        self.assertAlmostEqual(output,0)
    
    def test_03_fg_bg_equal(self):
        img = np.ones((128,128))
        img[0:64,:] *= .1
        img[64:128,:] *= .9
        output = T.sum_of_entropies(img, np.ones((128,128),bool), .5)
    
    def test_04_fg_bg_different(self):
        img = np.ones((128,128))
        img[0:64,0:64] *= .1
        img[0:64,64:128] *= .3
        img[64:128,0:64] *= .7
        img[64:128,64:128] *= .9
        output = T.sum_of_entropies(img, np.ones((128,128),bool), .5)
        
