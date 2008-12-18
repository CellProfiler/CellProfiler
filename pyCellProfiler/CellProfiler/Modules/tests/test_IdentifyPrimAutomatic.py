import CellProfiler.Modules.IdentifyPrimAutomatic as ID
import CellProfiler.Variable
import CellProfiler.Image
import CellProfiler.Objects
import CellProfiler.Measurements
import CellProfiler.Pipeline
import unittest
import numpy

class test_IdentifyPrimAutomatic(unittest.TestCase):
    def test_00_00_Init(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
    
    def test_01_01_ImageName(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        x.Variable(ID.IMAGE_NAME_VAR).SetValue("MyImage")
        self.assertEqual(x.ImageName, "MyImage")
    
    def test_01_02_ObjectName(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        x.Variable(ID.OBJECT_NAME_VAR).SetValue("MyObject")
        self.assertEqual(x.ObjectName, "MyObject")
    
    def test_01_03_SizeRange(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.MinSize,10)
        self.assertEqual(x.MaxSize,40)
        x.Variable(ID.SIZE_RANGE_VAR).SetValue("5,100")
        self.assertEqual(x.MinSize,5)
        self.assertEqual(x.MaxSize,100)
    
    def test_01_04_ExcludeSize(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.ExcludeSize,"Default should be yes")
        x.Variable(ID.EXCLUDE_SIZE_VAR).SetValue("Yes")
        self.assertTrue(x.ExcludeSize)
        x.Variable(ID.EXCLUDE_SIZE_VAR).SetValue("No")
        self.assertFalse(x.ExcludeSize)
        
    def test_01_05_MergeObjects(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertFalse(x.MergeObjects, "Default should be no")
        x.Variable(ID.MERGE_CHOICE_VAR).SetValue("Yes")
        self.assertTrue(x.MergeObjects)
        x.Variable(ID.MERGE_CHOICE_VAR).SetValue("No")
        self.assertFalse(x.MergeObjects)
    
    def test_01_06_ExcludeBorderObjects(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.ExcludeBorderObjects,"Default should be yes")
        x.Variable(ID.EXCLUDE_BORDER_OBJECTS_VAR).SetValue("Yes")
        self.assertTrue(x.ExcludeBorderObjects)
        x.Variable(ID.EXCLUDE_BORDER_OBJECTS_VAR).SetValue("No")
        self.assertFalse(x.ExcludeBorderObjects)
    
    def test_01_07_ThresholdMethod(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.ThresholdMethod, ID.TM_OTSU_GLOBAL, "Default should be Otsu global")
        x.Variable(ID.THRESHOLD_METHOD_VAR).SetValue(ID.TM_BACKGROUND_GLOBAL)
        self.assertEqual(x.ThresholdMethod, ID.TM_BACKGROUND_GLOBAL)
    
    def test_01_07_01_ThresholdModifier(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.ThresholdModifier, ID.TM_GLOBAL)
        x.Variable(ID.THRESHOLD_METHOD_VAR).SetValue(ID.TM_BACKGROUND_ADAPTIVE)
        self.assertEqual(x.ThresholdModifier, ID.TM_ADAPTIVE)

    def test_01_07_02_ThresholdAlgorithm(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.ThresholdAlgorithm, ID.TM_OTSU, "Default should be Otsu")
        x.Variable(ID.THRESHOLD_METHOD_VAR).SetValue(ID.TM_BACKGROUND_GLOBAL)
        self.assertEqual(x.ThresholdAlgorithm, ID.TM_BACKGROUND)

    def test_01_08_ThresholdRange(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.MinThreshold,0)
        self.assertEqual(x.MaxThreshold,1)
        x.Variable(ID.THRESHOLD_RANGE_VAR).SetValue(".2,.8")
        self.assertEqual(x.MinThreshold,.2)
        self.assertEqual(x.MaxThreshold,.8)
    
    def test_01_09_ThresholdCorrectionFactor(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.ThresholdCorrectionFactor,1)
        x.Variable(ID.THRESHOLD_CORRECTION_VAR).SetValue("1.5")
        self.assertEqual(x.ThresholdCorrectionFactor,1.5)
    
    def test_01_10_ObjectFraction(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.ObjectFraction,0.01)
        x.Variable(ID.OBJECT_FRACTION_VAR).SetValue("0.2")
        self.assertEqual(x.ObjectFraction,0.2)
        
    def test_01_11_UnclumpMethod(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.UnclumpMethod, ID.UN_INTENSITY, "Default should be intensity, was %s"%(x.UnclumpMethod))
        x.Variable(ID.UNCLUMP_METHOD_VAR).SetValue(ID.UN_MANUAL)
        self.assertEqual(x.UnclumpMethod, ID.UN_MANUAL)

    def test_01_12_WatershedMethod(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertEqual(x.WatershedMethod, ID.WA_INTENSITY, "Default should be intensity")
        x.Variable(ID.WATERSHED_VAR).SetValue(ID.WA_DISTANCE)
        self.assertEqual(x.WatershedMethod, ID.WA_DISTANCE)
        
    def test_01_13_SmoothingFilterSize(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.AutomaticSmoothingFilterSize, "Default should be automatic")
        self.assertEqual(x.SmoothingFilterSize, None)
        x.Variable(ID.SMOOTHING_SIZE_VAR).SetValue("10")
        self.assertFalse(x.AutomaticSmoothingFilterSize)
        self.assertEqual(x.SmoothingFilterSize,10)
        x.Variable(ID.SMOOTHING_SIZE_VAR).SetValue(ID.AUTOMATIC)
        self.assertTrue(x.AutomaticSmoothingFilterSize)
        self.assertEqual(x.SmoothingFilterSize, None)
    
    def test_01_14_MaximaSuppressionSize(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.AutomaticMaximaSuppressionSize, "Default should be automatic")
        self.assertEqual(x.SmoothingFilterSize, None)
        x.Variable(ID.MAXIMA_SUPRESSION_SIZE_VAR).SetValue("10")
        self.assertFalse(x.AutomaticMaximaSuppressionSize)
        self.assertEqual(x.MaximaSuppressionSize,10)
        x.Variable(ID.MAXIMA_SUPRESSION_SIZE_VAR).SetValue(ID.AUTOMATIC)
        self.assertTrue(x.AutomaticMaximaSuppressionSize)
        self.assertEqual(x.MaximaSuppressionSize, None)
        
    def test_01_15_UseLowRes(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.UseLowRes)
        x.Variable(ID.LOW_RES_MAXIMA_VAR).SetValue("No")
        self.assertFalse(x.UseLowRes)
        x.Variable(ID.LOW_RES_MAXIMA_VAR).SetValue("Yes")
        self.assertTrue(x.UseLowRes)
        
    def test_01_16_OutlineName(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertFalse(x.SaveOutlines)
        x.Variable(ID.SAVE_OUTLINES_VAR).Value = "ImageOutline"
        self.assertTrue(x.SaveOutlines)
        self.assertEqual(x.OutlinesName,"ImageOutline")
        x.Variable(ID.SAVE_OUTLINES_VAR).Value = CellProfiler.Variable.DO_NOT_USE
        self.assertFalse(x.SaveOutlines)
    
    def test_01_17_FillHoles(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.FillHoles)
        x.Variable(ID.FILL_HOLES_OPTION_VAR).Value = CellProfiler.Variable.NO
        self.assertFalse(x.FillHoles)
        x.Variable(ID.FILL_HOLES_OPTION_VAR).Value = CellProfiler.Variable.YES
        self.assertTrue(x.FillHoles)
        
    def test_01_18_TestMode(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        self.assertTrue(x.TestMode)
        x.Variable(ID.TEST_MODE_VAR).Value = CellProfiler.Variable.NO
        self.assertFalse(x.TestMode)
        x.Variable(ID.TEST_MODE_VAR).Value = CellProfiler.Variable.YES
        self.assertTrue(x.TestMode)

    def test_02_01_TestOneObject(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        x.Variable(ID.OBJECT_NAME_VAR).Value = "my_object"
        x.Variable(ID.IMAGE_NAME_VAR).Value = "my_image"
        img = OneCellImage()
        image = CellProfiler.Image.Image(img)
        image_set_list = CellProfiler.Image.ImageSetList()
        image_set = image_set_list.GetImageSet(0)
        image_set.Providers.append(CellProfiler.Image.VanillaImageProvider("my_image",image))
        object_set = CellProfiler.Objects.ObjectSet()
        measurements = CellProfiler.Measurements.Measurements()
        pipeline = CellProfiler.Pipeline.Pipeline()
        x.Run(pipeline,image_set,object_set,measurements,None)
        self.assertEqual(len(object_set.ObjectNames),1)
        self.assertTrue("my_object" in object_set.ObjectNames)
        objects = object_set.GetObjects("my_object")
        segmented = objects.Segmented
        self.assertTrue(numpy.all(segmented[img>0] == 1))
        self.assertTrue(numpy.all(img[segmented==1] > 0))
        self.assertTrue("Image" in measurements.GetObjectNames())
        self.assertTrue("my_object" in measurements.GetObjectNames())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.GetFeatureNames("Image"))
        threshold = measurements.GetCurrentMeasurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .5)
        self.assertTrue("Count_my_object" in measurements.GetFeatureNames("Image"))
        count = measurements.GetCurrentMeasurement("Image","Count_my_object")
        self.assertEqual(count,1)
        self.assertTrue("Location_Center_X" in measurements.GetFeatureNames("my_object"))
        location_center_x = measurements.GetCurrentMeasurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue("Location_Center_Y" in measurements.GetFeatureNames("my_object"))
        location_center_y = measurements.GetCurrentMeasurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>13)
        self.assertTrue(location_center_y[0]<16)

    def test_02_02_TestTwoObjects(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        x.Variable(ID.OBJECT_NAME_VAR).Value = "my_object"
        x.Variable(ID.IMAGE_NAME_VAR).Value = "my_image"
        img = TwoCellImage()
        image = CellProfiler.Image.Image(img)
        image_set_list = CellProfiler.Image.ImageSetList()
        image_set = image_set_list.GetImageSet(0)
        image_set.Providers.append(CellProfiler.Image.VanillaImageProvider("my_image",image))
        object_set = CellProfiler.Objects.ObjectSet()
        measurements = CellProfiler.Measurements.Measurements()
        pipeline = CellProfiler.Pipeline.Pipeline()
        x.Run(pipeline,image_set,object_set,measurements,None)
        self.assertEqual(len(object_set.ObjectNames),1)
        self.assertTrue("my_object" in object_set.ObjectNames)
        objects = object_set.GetObjects("my_object")
        self.assertTrue("Image" in measurements.GetObjectNames())
        self.assertTrue("my_object" in measurements.GetObjectNames())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.GetFeatureNames("Image"))
        threshold = measurements.GetCurrentMeasurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .6)
        self.assertTrue("Count_my_object" in measurements.GetFeatureNames("Image"))
        count = measurements.GetCurrentMeasurement("Image","Count_my_object")
        self.assertEqual(count,2)
        self.assertTrue("Location_Center_X" in measurements.GetFeatureNames("my_object"))
        location_center_x = measurements.GetCurrentMeasurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),2)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue(location_center_x[1]>28)
        self.assertTrue(location_center_x[1]<32)
        self.assertTrue("Location_Center_Y" in measurements.GetFeatureNames("my_object"))
        location_center_y = measurements.GetCurrentMeasurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),2)
        self.assertTrue(location_center_y[0]>33)
        self.assertTrue(location_center_y[0]<37)
        self.assertTrue(location_center_y[1]>13)
        self.assertTrue(location_center_y[1]<16)

    def test_02_03_TestThresholdRange(self):
        x = ID.IdentifyPrimAutomatic()
        x.CreateFromAnnotations()
        x.Variable(ID.OBJECT_NAME_VAR).Value = "my_object"
        x.Variable(ID.IMAGE_NAME_VAR).Value = "my_image"
        x.Variable(ID.THRESHOLD_RANGE_VAR).Value = ".7,1"
        img = TwoCellImage()
        image = CellProfiler.Image.Image(img)
        image_set_list = CellProfiler.Image.ImageSetList()
        image_set = image_set_list.GetImageSet(0)
        image_set.Providers.append(CellProfiler.Image.VanillaImageProvider("my_image",image))
        object_set = CellProfiler.Objects.ObjectSet()
        measurements = CellProfiler.Measurements.Measurements()
        pipeline = CellProfiler.Pipeline.Pipeline()
        x.Run(pipeline,image_set,object_set,measurements,None)
        self.assertEqual(len(object_set.ObjectNames),1)
        self.assertTrue("my_object" in object_set.ObjectNames)
        objects = object_set.GetObjects("my_object")
        self.assertTrue("Image" in measurements.GetObjectNames())
        self.assertTrue("my_object" in measurements.GetObjectNames())
        self.assertTrue("Threshold_FinalThreshold_my_object" in measurements.GetFeatureNames("Image"))
        threshold = measurements.GetCurrentMeasurement("Image","Threshold_FinalThreshold_my_object")
        self.assertTrue(threshold < .8)
        self.assertTrue(threshold > .6)
        self.assertTrue("Count_my_object" in measurements.GetFeatureNames("Image"))
        count = measurements.GetCurrentMeasurement("Image","Count_my_object")
        self.assertEqual(count,1)
        self.assertTrue("Location_Center_X" in measurements.GetFeatureNames("my_object"))
        location_center_x = measurements.GetCurrentMeasurement("my_object","Location_Center_X")
        self.assertTrue(isinstance(location_center_x,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_x.shape),1)
        self.assertTrue(location_center_x[0]>8)
        self.assertTrue(location_center_x[0]<12)
        self.assertTrue("Location_Center_Y" in measurements.GetFeatureNames("my_object"))
        location_center_y = measurements.GetCurrentMeasurement("my_object","Location_Center_Y")
        self.assertTrue(isinstance(location_center_y,numpy.ndarray))
        self.assertEqual(numpy.product(location_center_y.shape),1)
        self.assertTrue(location_center_y[0]>33)
        self.assertTrue(location_center_y[0]<36)

def OneCellImage():
    img = numpy.zeros((25,25))
    DrawCircle(img,(10,15),5, .5)
    return img

def TwoCellImage():
    img = numpy.zeros((50,50))
    DrawCircle(img,(10,35),5, .8)
    DrawCircle(img,(30,15),5, .6)
    return img
    
def DrawCircle(img,center,radius,value):
    x,y=numpy.mgrid[0:img.shape[0],0:img.shape[1]]
    distance = numpy.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
    img[distance<=radius]=value
