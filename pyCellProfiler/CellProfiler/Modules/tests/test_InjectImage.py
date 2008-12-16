import CellProfiler.Modules.InjectImage
import CellProfiler.Image
import CellProfiler.Pipeline
import unittest
import numpy

class testInjectImage(unittest.TestCase):
    def test_00_00_Init(self):
        image = numpy.zeros((10,10),dtype=float)
        x = CellProfiler.Modules.InjectImage.InjectImage("my_image", image)
    
    def test_01_01_GetFromImageSet(self):
        image = numpy.zeros((10,10),dtype=float)
        image_set_list = CellProfiler.Image.ImageSetList()
        ii = CellProfiler.Modules.InjectImage.InjectImage("my_image", image)
        pipeline = CellProfiler.Pipeline.Pipeline()
        ii.PrepareRun(pipeline, image_set_list)
        image_set = image_set_list.GetImageSet(0)
        self.assertTrue(image_set,"No image set returned from ImageSetList.GetImageSet")
        my_image = image_set.GetImage("my_image")
        self.assertTrue(my_image, "No image returned from ImageSet.GetImage")
        self.assertEqual(my_image.Image.shape[0],10,"Wrong image shape")

if __name__=="main":
    unittest.main()
        
