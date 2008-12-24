__version__ = "$Revision: 1 $"
import os

import unittest
import numpy
import tempfile

import cellprofiler.modules.identifyprimautomatic as ID
from cellprofiler.modules.injectimage import InjectImage
import cellprofiler.variable
import cellprofiler.cpimage
import cellprofiler.objects
import cellprofiler.measurements
import cellprofiler.pipeline
from cellprofiler.matlab.cputils import get_matlab_instance

class test_IdentifyPrimAutomatic(unittest.TestCase):
    def test_00_00_init(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
    
    def test_01_01_image_name(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        x.variable(ID.IMAGE_NAME_VAR).set_value("MyImage")
        self.assertEqual(x.image_name, "MyImage")
    
    def test_01_02_object_name(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        x.variable(ID.OBJECT_NAME_VAR).set_value("MyObject")
        self.assertEqual(x.object_name, "MyObject")
    
    def test_01_03_size_range(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.min_size,10)
        self.assertEqual(x.max_size,40)
        x.variable(ID.SIZE_RANGE_VAR).set_value("5,100")
        self.assertEqual(x.min_size,5)
        self.assertEqual(x.max_size,100)
    
    def test_01_04_exclude_size(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.exclude_size,"Default should be yes")
        x.variable(ID.EXCLUDE_SIZE_VAR).set_value("Yes")
        self.assertTrue(x.exclude_size)
        x.variable(ID.EXCLUDE_SIZE_VAR).set_value("No")
        self.assertFalse(x.exclude_size)
        
    def test_01_05_merge_objects(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertFalse(x.merge_objects, "Default should be no")
        x.variable(ID.MERGE_CHOICE_VAR).set_value("Yes")
        self.assertTrue(x.merge_objects)
        x.variable(ID.MERGE_CHOICE_VAR).set_value("No")
        self.assertFalse(x.merge_objects)
    
    def test_01_06_exclude_border_objects(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.exclude_border_objects,"Default should be yes")
        x.variable(ID.EXCLUDE_BORDER_OBJECTS_VAR).set_value("Yes")
        self.assertTrue(x.exclude_border_objects)
        x.variable(ID.EXCLUDE_BORDER_OBJECTS_VAR).set_value("No")
        self.assertFalse(x.exclude_border_objects)
    
    def test_01_07_threshold_method(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.threshold_method, ID.TM_OTSU_GLOBAL, "Default should be Otsu global")
        x.variable(ID.THRESHOLD_METHOD_VAR).set_value(ID.TM_BACKGROUND_GLOBAL)
        self.assertEqual(x.threshold_method, ID.TM_BACKGROUND_GLOBAL)
    
    def test_01_07_01_threshold_modifier(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.threshold_modifier, ID.TM_GLOBAL)
        x.variable(ID.THRESHOLD_METHOD_VAR).set_value(ID.TM_BACKGROUND_ADAPTIVE)
        self.assertEqual(x.threshold_modifier, ID.TM_ADAPTIVE)

    def test_01_07_02_threshold_algorithm(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.threshold_algorithm, ID.TM_OTSU, "Default should be Otsu")
        x.variable(ID.THRESHOLD_METHOD_VAR).set_value(ID.TM_BACKGROUND_GLOBAL)
        self.assertEqual(x.threshold_algorithm, ID.TM_BACKGROUND)

    def test_01_08_threshold_range(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.min_threshold,0)
        self.assertEqual(x.max_threshold,1)
        x.variable(ID.THRESHOLD_RANGE_VAR).set_value(".2,.8")
        self.assertEqual(x.min_threshold,.2)
        self.assertEqual(x.max_threshold,.8)
    
    def test_01_09_threshold_correction_factor(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.threshold_correction_factor,1)
        x.variable(ID.THRESHOLD_CORRECTION_VAR).set_value("1.5")
        self.assertEqual(x.threshold_correction_factor,1.5)
    
    def test_01_10_object_fraction(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.object_fraction,0.01)
        x.variable(ID.OBJECT_FRACTION_VAR).set_value("0.2")
        self.assertEqual(x.object_fraction,0.2)
        
    def test_01_11_unclump_method(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.unclump_method, ID.UN_INTENSITY, "Default should be intensity, was %s"%(x.unclump_method))
        x.variable(ID.UNCLUMP_METHOD_VAR).set_value(ID.UN_MANUAL)
        self.assertEqual(x.unclump_method, ID.UN_MANUAL)

    def test_01_12_watershed_method(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertEqual(x.watershed_method, ID.WA_INTENSITY, "Default should be intensity")
        x.variable(ID.WATERSHED_VAR).set_value(ID.WA_DISTANCE)
        self.assertEqual(x.watershed_method, ID.WA_DISTANCE)
        
    def test_01_13_smoothing_filter_size(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.automatic_smoothing_filter_size, "Default should be automatic")
        self.assertEqual(x.smoothing_filter_size, None)
        x.variable(ID.SMOOTHING_SIZE_VAR).set_value("10")
        self.assertFalse(x.automatic_smoothing_filter_size)
        self.assertEqual(x.smoothing_filter_size,10)
        x.variable(ID.SMOOTHING_SIZE_VAR).set_value(ID.AUTOMATIC)
        self.assertTrue(x.automatic_smoothing_filter_size)
        self.assertEqual(x.smoothing_filter_size, None)
    
    def test_01_14_maxima_suppression_size(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.automatic_maxima_suppression_size, "Default should be automatic")
        self.assertEqual(x.smoothing_filter_size, None)
        x.variable(ID.MAXIMA_SUPRESSION_SIZE_VAR).set_value("10")
        self.assertFalse(x.automatic_maxima_suppression_size)
        self.assertEqual(x.maxima_suppression_size,10)
        x.variable(ID.MAXIMA_SUPRESSION_SIZE_VAR).set_value(ID.AUTOMATIC)
        self.assertTrue(x.automatic_maxima_suppression_size)
        self.assertEqual(x.maxima_suppression_size, None)
        
    def test_01_15_use_low_res(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.use_low_res)
        x.variable(ID.LOW_RES_MAXIMA_VAR).set_value("No")
        self.assertFalse(x.use_low_res)
        x.variable(ID.LOW_RES_MAXIMA_VAR).set_value("Yes")
        self.assertTrue(x.use_low_res)
        
    def test_01_16_outline_name(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertFalse(x.save_outlines)
        x.variable(ID.SAVE_OUTLINES_VAR).value = "ImageOutline"
        self.assertTrue(x.save_outlines)
        self.assertEqual(x.outlines_name,"ImageOutline")
        x.variable(ID.SAVE_OUTLINES_VAR).value = cellprofiler.variable.DO_NOT_USE
        self.assertFalse(x.save_outlines)
    
    def test_01_17_fill_holes(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.fill_holes)
        x.variable(ID.FILL_HOLES_OPTION_VAR).value = cellprofiler.variable.NO
        self.assertFalse(x.fill_holes)
        x.variable(ID.FILL_HOLES_OPTION_VAR).value = cellprofiler.variable.YES
        self.assertTrue(x.fill_holes)
        
    def test_01_18_test_mode(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        self.assertTrue(x.test_mode)
        x.variable(ID.TEST_MODE_VAR).value = cellprofiler.variable.NO
        self.assertFalse(x.test_mode)
        x.variable(ID.TEST_MODE_VAR).value = cellprofiler.variable.YES
        self.assertTrue(x.test_mode)

    def test_02_01_test_one_object(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        x.variable(ID.OBJECT_NAME_VAR).value = "my_object"
        x.variable(ID.IMAGE_NAME_VAR).value = "my_image"
        img = one_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cellprofiler.measurements.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(pipeline,image_set,object_set,measurements,None)
        self.assertEqual(len(object_set.object_names),1)
        self.assertTrue("my_object" in object_set.object_names)
        objects = object_set.get_objects("my_object")
        segmented = objects.segmented
        self.assertTrue(numpy.all(segmented[img>0] == 1))
        self.assertTrue(numpy.all(img[segmented==1] > 0))
        self.assertTrue("Image" in measurements.get_object_names())
        self.assertTrue("my_object" in measurements.get_object_names())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.get_feature_names("Image"))
        threshold = measurements.get_current_measurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .5)
        self.assertTrue("Count_my_object" in measurements.get_feature_names("Image"))
        count = measurements.get_current_measurement("Image","Count_my_object")
        self.assertEqual(count,1)
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>13)
        self.assertTrue(location_center_y[0]<16)

    def test_02_02_test_two_objects(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        x.variable(ID.OBJECT_NAME_VAR).value = "my_object"
        x.variable(ID.IMAGE_NAME_VAR).value = "my_image"
        img = two_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cellprofiler.measurements.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(pipeline,image_set,object_set,measurements,None)
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
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue(location_center_x[1]>28)
        self.assertTrue(location_center_x[1]<32)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>33)
        self.assertTrue(location_center_y[0]<37)
        self.assertTrue(location_center_y[1]>13)
        self.assertTrue(location_center_y[1]<16)

    def test_02_03_test_threshold_range(self):
        x = ID.IdentifyPrimAutomatic()
        x.create_from_annotations()
        x.variable(ID.OBJECT_NAME_VAR).value = "my_object"
        x.variable(ID.IMAGE_NAME_VAR).value = "my_image"
        x.variable(ID.THRESHOLD_RANGE_VAR).value = ".7,1"
        img = two_cell_image()
        image = cellprofiler.cpimage.Image(img)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider("my_image",image))
        object_set = cellprofiler.objects.ObjectSet()
        measurements = cellprofiler.measurements.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        x.run(pipeline,image_set,object_set,measurements,None)
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
        self.assertTrue("Location_Center_X" in measurements.get_feature_names("my_object"))
        location_center_x = measurements.get_current_measurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue("Location_Center_Y" in measurements.get_feature_names("my_object"))
        location_center_y = measurements.get_current_measurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>33)
        self.assertTrue(location_center_y[0]<36)
    
    def test_03_01_run_inside_pipeline(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        inject_image = InjectImage("my_image", two_cell_image())
        inject_image.set_module_num(1)
        pipeline.add_module(inject_image)
        ipm = ID.IdentifyPrimAutomatic()
        ipm.set_module_num(2)
        ipm.create_from_annotations() 
        ipm.variable(ID.OBJECT_NAME_VAR).value = "my_object"
        ipm.variable(ID.IMAGE_NAME_VAR).value = "my_image"
        pipeline.add_module(ipm)
        measurements = pipeline.run()
        (matfd,matpath) = tempfile.mkstemp('.mat')
        matfh = os.fdopen(matfd,'wb')
        matfh.close()
        pipeline.save_measurements(matpath, measurements)
        matlab = get_matlab_instance()
        handles = matlab.load(matpath)
        handles = handles.handles
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,"Image")),'1')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,"Threshold_FinalThreshold_my_object")),'1')
        thresholds = handles.Measurements.Image.Threshold_FinalThreshold_my_object
        threshold = thresholds._[0][0,0]
        #self.assertTrue(threshold < .6)
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,"Count_my_object")),'1')
        counts = handles.Measurements.Image.Count_my_object
        count = counts._[0][0,0]
        self.assertEqual(count,2)
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,"my_object")),'1')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.my_object,"Location_Center_X")),'1')
        location_center_x = matlab.cell2mat(handles.Measurements.my_object.Location_Center_X[0])
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0,0]>8)
        self.assertTrue(location_center_x[0,0]<12)
        self.assertTrue(location_center_x[1,0]>28)
        self.assertTrue(location_center_x[1,0]<32)
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.my_object,"Location_Center_Y")),'1')
        location_center_y = matlab.cell2mat(handles.Measurements.my_object.Location_Center_Y[0])
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0,0]>33)
        self.assertTrue(location_center_y[0,0]<37)
        self.assertTrue(location_center_y[1,0]>13)
        self.assertTrue(location_center_y[1,0]<16)

def one_cell_image():
    img = numpy.zeros((25,25))
    draw_circle(img,(10,15),5, .5)
    return img

def two_cell_image():
    img = numpy.zeros((50,50))
    draw_circle(img,(10,35),5, .8)
    draw_circle(img,(30,15),5, .6)
    return img
    
def draw_circle(img,center,radius,value):
    x,y=numpy.mgrid[0:img.shape[0],0:img.shape[1]]
    distance = numpy.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
    img[distance<=radius]=value
