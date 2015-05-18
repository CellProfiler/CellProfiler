"""test_yeast_cell_segmentations.py: test the YeastSegmentation module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Credits (coding)
Filip Mroz, Adam Kaczmarek, Szymon Stoma.

Website: http://www.cellprofiler.org
"""
import os

import base64
import unittest
import numpy as np
import scipy.ndimage
import tempfile
import StringIO
import zlib

import cellprofiler.modules.yeast_cell_segmentation as YS
import cellprofiler.modules.identify as I
import cellprofiler.cpmath.threshold as T
from cellprofiler.modules.injectimage import InjectImage
import cellprofiler.settings
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline
from cellprofiler.workspace import Workspace
from cellprofiler.modules.tests import read_example_image

IMAGE_NAME = "my_image"
BACKGROUND_IMAGE_NAME = "background_image"
OBJECTS_NAME = "my_objects"
BINARY_IMAGE_NAME = "binary_image"
MASKING_OBJECTS_NAME = "masking_objects"
MEASUREMENT_NAME = "my_measurement"

class test_YeastSegmentation(unittest.TestCase):
    def load_error_handler(self, caller, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.fail(event.error.message)
            
    def make_workspace(self, image, 
                       mask = None,
                       labels = None, 
                       binary_image = None):
        '''Make a workspace and IdentifyPrimaryObjects module
        
        image - the intensity image for thresholding
        
        mask - if present, the "don't analyze" mask of the intensity image
        
        labels - if thresholding per-object, the labels matrix needed
        
        binary_image - if thresholding using a binary image, the image
        '''
        module = YS.YeastCellSegmentation()
        module.module_num = 1
        module.input_image_name.value = IMAGE_NAME
        module.object_name.value = OBJECTS_NAME
        module.binary_image.value = BINARY_IMAGE_NAME
        module.masking_objects.value = MASKING_OBJECTS_NAME
        
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        cpimage = cpi.Image(image, mask = mask)
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
        x = YS.YeastCellSegmentation()
        
    def test_01_00_test_zero_objects(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 5
        img = np.zeros((25,25))
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Count_" + OBJECTS_NAME in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_" + OBJECTS_NAME)
        self.assertEqual(count,0)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),0)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),0)

    def test_01_01_test_one_object(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 10
        img = convert_to_brightfield(get_one_cell_mask(), False)
        image = cpi.Image(img, file_name="test_01_01_test_one_object")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        segmented = objects.segmented
        self.assertTrue(is_segmentation_correct(get_one_cell_mask(), segmented))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Features_Quality" in measurements.get_feature_names(OBJECTS_NAME))
        quality = measurements.get_current_measurement(OBJECTS_NAME,"Features_Quality")
        self.assertTrue(quality[0] > 0)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>8)
        self.assertTrue(location_center_y[0]<12)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>13)
        self.assertTrue(location_center_x[0]<16)
        columns = x.get_measurement_columns(pipeline)
        for object_name in (cpmeas.IMAGE, OBJECTS_NAME):
            ocolumns =[x for x in columns if x[0] == object_name]
            features = measurements.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))

    def test_01_02_test_two_bright_objects(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 10
        img = convert_to_brightfield(get_two_cell_mask(), False)
        image = cpi.Image(img, file_name="test_01_02_test_two_bright_objects")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(is_segmentation_correct(get_two_cell_mask(), objects.segmented))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Features_Quality" in measurements.get_feature_names(OBJECTS_NAME))
        quality = measurements.get_current_measurement(OBJECTS_NAME,"Features_Quality")
        self.assertTrue(len(quality) == 2)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[1]>25)
        self.assertTrue(location_center_y[1]<45)
        self.assertTrue(location_center_y[0]>5)
        self.assertTrue(location_center_y[0]<25)
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[1]>3)
        self.assertTrue(location_center_x[1]<18)
        self.assertTrue(location_center_x[0]>20)
        self.assertTrue(location_center_x[0]<40)

    def test_01_03_test_two_dark_objects(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = True
        x.average_cell_diameter.value = 10
        img = convert_to_brightfield(get_two_cell_mask(), True)
        image = cpi.Image(img, file_name="test_01_03_test_two_dark_objects")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(is_segmentation_correct(get_two_cell_mask(), objects.segmented))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Features_Quality" in measurements.get_feature_names(OBJECTS_NAME))
        quality = measurements.get_current_measurement(OBJECTS_NAME,"Features_Quality")
        self.assertTrue(len(quality) == 2)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>25)
        self.assertTrue(location_center_y[0]<45)
        self.assertTrue(location_center_y[1]>5)
        self.assertTrue(location_center_y[1]<25)
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>3)
        self.assertTrue(location_center_x[0]<18)
        self.assertTrue(location_center_x[1]>20)
        self.assertTrue(location_center_x[1]<40)

    def test_01_04_test_two_flu_bright_objects(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.bright_field_image.value = False
        x.average_cell_diameter.value = 10
        img = convert_to_fluorescent(get_two_cell_mask(), False)
        image = cpi.Image(img, file_name="test_01_04_test_two_flu_bright_objects")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(is_segmentation_correct(get_two_cell_mask(), objects.segmented))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Features_Quality" in measurements.get_feature_names(OBJECTS_NAME))
        quality = measurements.get_current_measurement(OBJECTS_NAME,"Features_Quality")
        self.assertTrue(len(quality) == 2)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>25)
        self.assertTrue(location_center_y[0]<45)
        self.assertTrue(location_center_y[1]>5)
        self.assertTrue(location_center_y[1]<25)
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>3)
        self.assertTrue(location_center_x[0]<18)
        self.assertTrue(location_center_x[1]>20)
        self.assertTrue(location_center_x[1]<40)

    def test_01_05_test_two_flu_dark_objects(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = True
        x.bright_field_image.value = False
        x.average_cell_diameter.value = 10
        img = convert_to_fluorescent(get_two_cell_mask(), True)
        image = cpi.Image(img, file_name="test_01_05_test_two_flu_dark_objects")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue(OBJECTS_NAME in object_set.object_names)
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(is_segmentation_correct(get_two_cell_mask(), objects.segmented))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue(OBJECTS_NAME in measurements.get_object_names())
        self.assertTrue("Features_Quality" in measurements.get_feature_names(OBJECTS_NAME))
        quality = measurements.get_current_measurement(OBJECTS_NAME,"Features_Quality")
        self.assertTrue(len(quality) == 2)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names(OBJECTS_NAME))
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,np.ndarray))
        self.assertEqual(np.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>25)
        self.assertTrue(location_center_y[0]<45)
        self.assertTrue(location_center_y[1]>5)
        self.assertTrue(location_center_y[1]<25)
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>3)
        self.assertTrue(location_center_x[0]<18)
        self.assertTrue(location_center_x[1]>20)
        self.assertTrue(location_center_x[1]<40)

    def test_01_06_fill_holes(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 5

        img = np.zeros((40,40))
        draw_circle(img, (10,10), 7, .5)
        draw_circle(img, (30,30), 7, .5)
        img[10,10] = 0
        img[30,30] = 0
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(objects.segmented[10,10] > 0)
        self.assertTrue(objects.segmented[30,30] > 0)

    def test_02_01_discard_large(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 30
        x.min_cell_area.value = 100
        x.max_cell_area.value = 1000
        x.advanced_cell_filtering.value = True

        img = np.ones((200,200)) * 0.5
        draw_brightfield_cell(img,100,100,20,False)
        draw_brightfield_cell(img,25,25,10,False)
        draw_brightfield_cell(img,150,150,15,False)
        image = cpi.Image(img)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(objects.segmented[25,25]>0,1,"The small object was not there")
        self.assertEqual(objects.segmented[150,150]>0,1,"The medium object was not there")
        self.assertEqual(objects.segmented[100,100]>0,0,"The large object was not filtered out")
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)

    def test_02_02_discard_small(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 30
        x.min_cell_area.value = 500
        x.max_cell_area.value = 5000
        x.advanced_cell_filtering.value = True

        img = np.ones((200,200)) * 0.5
        draw_brightfield_cell(img,100,100,20,False)
        draw_brightfield_cell(img,25,25,10,False)
        draw_brightfield_cell(img,150,150,15,False)
        image = cpi.Image(img, file_name="test_02_02_discard_small")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(objects.segmented[25,25]>0,0,"The small object was not filtered out")
        self.assertEqual(objects.segmented[150,150]>0,1,"The medium object was not there")
        self.assertEqual(objects.segmented[100,100]>0,1,"The large object was not there")
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        self.assertTrue(isinstance(location_center_x,np.ndarray))
        self.assertEqual(np.product(location_center_x.shape),2)

    def test_02_03_use_background_image(self):
        x = YS.YeastCellSegmentation()
        x.object_name.value = OBJECTS_NAME
        x.input_image_name.value = IMAGE_NAME
        x.segmentation_precision.value = 11
        x.background_image_name.value = BACKGROUND_IMAGE_NAME
        x.background_elimination_strategy.value = YS.BKG_FILE
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 30

        img = np.ones((200,200)) * 0.5
        draw_brightfield_cell(img,100,100,20,False)
        draw_brightfield_cell(img,25,25,10,False)
        draw_brightfield_cell(img,150,150,15,False) # background blob

        bkg = np.ones((200,200)) * 0.5
        draw_brightfield_cell(bkg,150,150,15,False) # background blob

        image = cpi.Image(img, file_name="test_02_03_use_background_image")
        background = cpi.Image(bkg)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        image_set.providers.append(cpi.VanillaImageProvider(BACKGROUND_IMAGE_NAME,background))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(objects.segmented[25,25]>0,1,"The small object was not there")
        self.assertEqual(objects.segmented[100,100]>0,1,"The large object was not there")
        self.assertEqual(objects.segmented[150,150]>0,0,"The background blob was not filtered out")

def add_noise(img, fraction):
    '''Add a fractional amount of noise to an image to make it look real'''
    np.random.seed(0)
    noise = np.random.uniform(low=1-fraction/2, high=1+fraction/2,
                                 size=img.shape)
    return img * noise

def get_one_cell_mask():
    img = np.zeros((30,30))
    draw_circle(img,(10,15),5, 1)
    return img

def get_two_cell_mask():
    img = np.zeros((50,50))
    draw_circle(img,(10,35),5,1)
    draw_circle(img,(30,15),5,1)
    return img

def convert_to_brightfield(img, content_dark):
    if(content_dark):
        img *= 0.3
    else:
        img *= 0.6
    # get ring with dilation (5x5 radius)
    ring = (scipy.ndimage.morphology.binary_dilation(img, np.ones((3,3))) - (img > 0))
    img[ring] = .8
    # fill rest with background
    img[img == 0] = 0.5
    return add_noise(img, 0.000)

def draw_brightfield_cell(img,x,y,radius,content_dark=True):
    draw_circle(img,(x,y),radius+2, .8)
    if(content_dark):
        draw_circle(img,(x,y),radius, .3)
    else:
        draw_circle(img,(x,y),radius, .6)
    return img

def convert_to_fluorescent(img, content_dark):
    if content_dark:
        img = 1 - img + (img * 0.1)
    else:
        img *= 0.9
    img = scipy.ndimage.gaussian_filter(img, sigma=2)
    return add_noise(img, .000)

def is_segmentation_correct(ground_truth, segmentation):
    return are_masks_similar(segmentation > 0, ground_truth > 0)

def are_masks_similar(a, b):
    return 1.0 - (a&b).sum() / float((a|b).sum()) < 0.5

def draw_circle(img,center,radius,value):
    x,y=np.mgrid[0:img.shape[0],0:img.shape[1]]
    distance = np.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
    img[distance<=radius]=value
