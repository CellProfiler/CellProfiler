import CellProfiler.Modules.IdentifyPrimAutomatic as ID
import CellProfiler.Variable
import unittest

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
    
        