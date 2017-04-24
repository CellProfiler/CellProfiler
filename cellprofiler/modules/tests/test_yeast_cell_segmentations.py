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

import ast
import unittest
import StringIO

import numpy as np
import time
import scipy.ndimage

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.modules.yeast_cell_segmentation as YS
import cellprofiler.objects as cpo
import cellprofiler.pipeline
import cellprofiler.settings
from cellprofiler.workspace import Workspace

IMAGE_NAME = "my_image"
BACKGROUND_IMAGE_NAME = "background_image"
MASK_IMAGE_NAME = "mask_image"
OBJECTS_NAME = "my_objects"
BINARY_IMAGE_NAME = "binary_image"
MASKING_OBJECTS_NAME = "masking_objects"
MEASUREMENT_NAME = "my_measurement"

class test_YeastSegmentation(unittest.TestCase):
    def load_error_handler(self, caller, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.fail(event.error.message)

    def assertRange(self, range_start, range_end, value):
        self.assertGreaterEqual(value, range_start)
        self.assertLessEqual(value, range_end)

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
        module = YS.IdentifyYeastCells()
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
        x = YS.IdentifyYeastCells()


    def test_00_01_image_loading_equivalence(self):
        # TODO test if image read with CP pipeline and by yeast_segmentation internals are the same
        # save image and load it using both methods
        # check png, tiff 8/16
        self.fail("TODO")

    def test_00_02_load_v6_8_precision(self):
        self.longMessage = True
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

IdentifyYeastCells:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Input
    Name the primary objects to be identified:YeastCells
    Average cell diameter in pixels:27.0
    Segmentation precision:13
    Maximal overlap allowed while final filtering of cells:0.2
    Select the empty field image:None
    Retain outlines of the identified objects?:No
    Name the outline image:PrimaryOutlines
    Use advanced configuration parameters:Yes
    Is the area without cells (background) brighter then cell interiors?:Yes
    Do you want to segment brightfield images?:Yes
    Minimal area of accepted cell in pixels:900
    Maximum area of accepted cell in pixels:4500
    Do you want to filter cells by area?:No
    Select the background calculation mode:Actual image
    Do you want to see autoadapted parameters?:Yes
    Autoadapted parameters\x3A :\x5B\x5B0.1, 0.0442, 304.45, 15.482, 189.40820000000002, 7.0\x5D, \x5B300, 10, 0, 18, 10\x5D\x5D
    Number of steps in the autoadaptation procedure:1
    Select ignore mask image:Leave blank

IdentifyYeastCells:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Input
    Name the primary objects to be identified:YeastCells
    Average cell diameter in pixels:27.0
    Segmentation precision:4
    Maximal overlap allowed while final filtering of cells:0.2
    Select the empty field image:None
    Retain outlines of the identified objects?:No
    Name the outline image:PrimaryOutlines
    Use advanced configuration parameters:Yes
    Is the area without cells (background) brighter then cell interiors?:Yes
    Do you want to segment brightfield images?:Yes
    Minimal area of accepted cell in pixels:900
    Maximum area of accepted cell in pixels:4500
    Do you want to filter cells by area?:No
    Select the background calculation mode:Actual image
    Do you want to see autoadapted parameters?:Yes
    Autoadapted parameters\x3A :\x5B\x5B0.1, 0.0442, 304.45, 15.482, 189.40820000000002, 7.0\x5D, \x5B300, 10, 0, 18, 10\x5D\x5D
    Number of steps in the autoadaptation procedure:1
    Select ignore mask image:Leave blank
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(
                isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()),3)
        module1 = pipeline.modules()[1]
        module2 = pipeline.modules()[2]
        self.assertTrue(isinstance(module1, YS.IdentifyYeastCells))
        self.assertTrue(isinstance(module2, YS.IdentifyYeastCells))

        # both module should be now exactly the same
        self.assertEqual(len(module1.settings()), len(module2.settings()))
        for i in range(len(module1.settings())):
            self.assertEqual(module1.settings()[i].text, module2.settings()[i].text)
            self.assertEqual(module1.settings()[i].value, module2.settings()[i].value, module1.settings()[i].text)

        # validate default values for precision dependencies
        self.assertEqual(8, module2.iterations.value)
        self.assertEqual(1, module2.seeds_border.value)
        self.assertEqual(2, module2.seeds_content.value)
        self.assertEqual(1, module2.seeds_centroid.value)
        self.assertEqual(52, module2.contour_points.value)
        self.assertEqual(75, int(.5 + module2.contour_precision.value))
        self.assertEqual(3, module2.weights_number.value)

        # check if precision maps to values after change
        module2.segmentation_precision.value = 1
        module2.on_setting_changed(module2.segmentation_precision, None)
        self.assertEqual(4, module2.iterations.value)
        self.assertEqual(0.7, module2.seeds_border.value)
        self.assertEqual(0.5, module2.seeds_content.value)
        self.assertEqual(1, module2.seeds_centroid.value)
        self.assertEqual(36, module2.contour_points.value)
        self.assertEqual(37, int(.5 + module2.contour_precision.value))
        self.assertEqual(1, module2.weights_number.value)

        # if we turn off precision details it is restored to default
        module2.seeds_border.value = 3.1
        module2.seeds_content.value = 1.5
        self.assertEqual(3.1, module2.seeds_border.value)
        self.assertEqual(1.5, module2.seeds_content.value)
        module2.specify_precision_details.value = 0
        module2.on_setting_changed(module2.specify_precision_details, None)
        self.assertEqual(0.7, module2.seeds_border.value)
        self.assertEqual(0.5, module2.seeds_content.value)



    def test_01_00_test_zero_objects(self):
        x = YS.IdentifyYeastCells()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 5
        img = np.zeros((25,25))
        image = cpi.Image(img, file_name="test_01_00_test_zero_objects")
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
        x = YS.IdentifyYeastCells()
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
        x = YS.IdentifyYeastCells()
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
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_Y")
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_X")
        positions = sorted(zip(location_center_x, location_center_y))
        self.assertEqual(2, len(positions))

        self.assertRange(3,18, positions[0][0])
        self.assertRange(25,45, positions[0][1])

        self.assertRange(20,40, positions[1][0])
        self.assertRange(5,25, positions[1][1])


    def test_01_03_test_two_dark_objects(self):
        x = YS.IdentifyYeastCells()
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
        location_center_y = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_Y")
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME, "Location_Center_X")
        positions = sorted(zip(location_center_x, location_center_y))
        self.assertEqual(2, len(positions))

        self.assertRange(3,18, positions[0][0])
        self.assertRange(25,45, positions[0][1])

        self.assertRange(20,40, positions[1][0])
        self.assertRange(5,25, positions[1][1])


    def test_01_04_test_two_flu_bright_objects(self):
        x = YS.IdentifyYeastCells()
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
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        positions = sorted(zip(location_center_x, location_center_y))
        self.assertEqual(2, len(positions))

        self.assertRange(3,18, positions[0][0])
        self.assertRange(25,45, positions[0][1])

        self.assertRange(20,40, positions[1][0])
        self.assertRange(5,25, positions[1][1])

    def test_01_05_test_two_flu_dark_objects(self):
        x = YS.IdentifyYeastCells()
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
        location_center_x = measurements.get_current_measurement(OBJECTS_NAME,"Location_Center_X")
        positions = sorted(zip(location_center_x, location_center_y))
        self.assertEqual(2, len(positions))

        self.assertRange(3,18, positions[0][0])
        self.assertRange(25,45, positions[0][1])

        self.assertRange(20,40, positions[1][0])
        self.assertRange(5,25, positions[1][1])

    def test_01_06_fill_holes(self):
        x = YS.IdentifyYeastCells()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 11
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 5

        img = np.zeros((40,40))
        draw_disc(img, (10, 10), 7, .5)
        draw_disc(img, (30, 30), 7, .5)
        img[10,10] = 0
        img[30,30] = 0
        image = cpi.Image(img, file_name="test_01_06_fill_holes")
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

    def test_01_07_extreme_params(self):
        x = YS.IdentifyYeastCells()
        x.object_name.value = OBJECTS_NAME
        x.segmentation_precision.value = 14
        x.input_image_name.value = IMAGE_NAME
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 77
        img = get_two_cell_mask()
        draw_disc(img, (5, 5), 2, 0.7)
        draw_disc(img, (35, 11), 3, 0.2)
        img = convert_to_brightfield(img, False)
        image = cpi.Image(img, file_name="test_01_07_extreme_params")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME,image))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline,x,image_set,object_set,measurements,None))
        objects = object_set.get_objects(OBJECTS_NAME)
        segmented = objects.segmented
        self.assertTrue(np.all(segmented == 0))  # no found because of parameters (no foreground)
        self.assertEqual(0, objects.count)

    def test_02_01_discard_large(self):
        x = YS.IdentifyYeastCells()
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
        image = cpi.Image(img, file_name="test_02_01_discard_large")
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
        x = YS.IdentifyYeastCells()
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
        x = YS.IdentifyYeastCells()
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
        image_set.providers.append(cpi.VanillaImageProvider(BACKGROUND_IMAGE_NAME, background))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(objects.segmented[25, 25] > 0, 1, "The small object was not there")
        self.assertEqual(objects.segmented[100, 100] > 0, 1, "The large object was not there")
        self.assertEqual(objects.segmented[150, 150] > 0, 0, "The background blob was not filtered out")

    def test_02_04_mask_input_image(self):
        x = YS.IdentifyYeastCells()
        x.object_name.value = OBJECTS_NAME
        x.input_image_name.value = IMAGE_NAME
        x.segmentation_precision.value = 11
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 30

        img = np.ones((200, 200)) * 0.5
        draw_brightfield_cell(img, 100, 100, 20, False)
        draw_brightfield_cell(img, 25, 25, 10, False)
        img[0:10, 0:10] = 1
        img[180:200, 180:200] = 0

        msk = np.zeros((200, 200))
        msk[0:10, 0:10] = 1
        msk[180:200, 180:200] = 1

        image = cpi.Image(img, file_name="test_02_04_mask_input_image")
        mask = cpi.Image(msk)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME, image))
        image_set.providers.append(cpi.VanillaImageProvider(MASK_IMAGE_NAME, mask))

        # first try without masking
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(0, objects.segmented.max(), "Cells should not be found due to the distractors")

        # now if we use masking option we should find these cells
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.ignore_mask_image_name.value = MASK_IMAGE_NAME
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(objects.segmented[25, 25] > 0, 1, "The small object was not there")
        self.assertEqual(objects.segmented[100, 100] > 0, 1, "The large object was not there")

    def test_03_01_simple_fitting(self):
        np.random.seed(1)

        x = YS.IdentifyYeastCells()
        x.object_name.value = OBJECTS_NAME

        x.input_image_name.value = IMAGE_NAME
        x.segmentation_precision.value = 9  # so that it is faster for fitting
        x.maximal_cell_overlap.value = 0.4
        x.background_brighter_then_cell_inside.value = False
        x.average_cell_diameter.value = 30
        x.autoadaptation_steps.value = 1

        img = np.ones((200, 200)) * 0.5
        draw_brightfield_cell(img, 100, 100, 15, False)
        draw_brightfield_cell(img, 120, 120, 15, False)
        draw_brightfield_cell(img, 110, 70, 15, False)
        draw_brightfield_cell(img, 160, 160, 10, False)
        draw_disc(img, (100, 100), 15, .65)
        draw_disc(img, (120, 120), 15, .65)
        draw_disc(img, (110, 70), 15, .65)
        draw_disc(img, (160, 160), 10, .65)
        img = img + np.random.normal(3., 0.01, img.shape)
        img = scipy.ndimage.gaussian_filter(img, 3)

        label = np.zeros((200, 200), dtype=int)
        draw_disc(label, (100, 100), 15, 1)
        draw_disc(label, (110, 70), 15, 2)

        image = cpi.Image(img, file_name="test_03_01_simple_fitting")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cpi.VanillaImageProvider(IMAGE_NAME, image))

        old_params = ast.literal_eval(x.autoadapted_params.value)
        input_processed, background_processed, ignore_mask_processed = x.preprocess_images(img, None, None)
        x.fit_parameters(input_processed, background_processed, ignore_mask_processed, label,
                         x.autoadaptation_steps.value * 2, lambda x: True, lambda secs: time.sleep(secs))
        new_params = ast.literal_eval(x.autoadapted_params.value)
        self.assertNotEqual(old_params[0], new_params[0])
        self.assertNotEqual(old_params[1], new_params[1])

        print "found parameters: ", x.autoadapted_params.value
        #x.autoadapted_params.value = "[[16.282366833092343, -12.278185398879907, 608.72017238611102, 17.441635091145478, 203.32510436137059, 7.1180878616033336], [1214.4324725382576, 2367.3881652432678, 216.10299086636189, 2620.6127639758142, 523.98667591841763]]"

        # now if we use new parameters option we should find these cells
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.segmentation_precision.value = 11
        x.run(Workspace(pipeline, x, image_set, object_set, measurements, None))
        objects = object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(4, objects.segmented.max())
        colours = sorted([objects.segmented[100, 100], objects.segmented[120, 120], objects.segmented[110, 70], objects.segmented[160, 160]])
        self.assertEqual(colours[0], 1)
        self.assertEqual(colours[1], 2)
        self.assertEqual(colours[2], 3)
        self.assertEqual(colours[3], 4)

    def test_03_02_fitting_background_masked(self):
        # Test if ignore and background can be used in fitting process (without it it should fade)
        self.fail("TODO")


def add_noise(img, fraction):
    '''Add a fractional amount of noise to an image to make it look real'''
    np.random.seed(0)
    noise = np.random.uniform(low=1 - fraction / 2, high=1 + fraction / 2,
                              size=img.shape)
    return img * noise

def get_one_cell_mask():
    img = np.zeros((30,30))
    draw_disc(img, (10, 15), 5, 1)
    return img

def get_two_cell_mask():
    img = np.zeros((50,50))
    draw_disc(img, (10, 35), 5, 1)
    draw_disc(img, (30, 15), 5, 1)
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
    draw_disc(img, (x, y), radius + 2, .8)
    if(content_dark):
        draw_disc(img, (x, y), radius, .3)
    else:
        draw_disc(img, (x, y), radius, .6)
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

def draw_disc(img, center, radius, value):
    x,y=np.mgrid[0:img.shape[0],0:img.shape[1]]
    distance = np.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
    img[distance<=radius]=value
