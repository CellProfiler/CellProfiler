"""Test the LoadImages module
"""
import unittest
import os
import CellProfiler.Modules.LoadImages as LI
import CellProfiler.Modules.tests as T
import CellProfiler.Image as I
import CellProfiler.Pipeline as P

class testLoadImages(unittest.TestCase):
    def test_00_00Init(self):
        x=LI.LoadImages()
        x.CreateFromAnnotations()
    
    def test_00_01Version(self):
        self.assertEqual(LI.LoadImages().VariableRevisionNumber(),4,"LoadImages' version number has changed")
    
    def test_01_01LoadImageTextMatch(self):
        l=LI.LoadImages()
        l.CreateFromAnnotations()
        l.Variable(LI.MATCH_STYLE_VAR).SetValue(LI.MS_EXACT_MATCH)
        l.Variable(LI.PATHNAME_VAR).SetValue(os.path.join(T.ExampleImagesDirectory(),"ExampleSBSImages"))
        l.Variable(LI.FIRST_IMAGE_VAR).SetValue("1-01-A-01.tif")
        l.Variable(LI.FIRST_IMAGE_VAR+1).SetValue("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.PrepareRun(pipeline, image_set_list)
        self.assertEqual(image_set_list.Count(),1,"Expected one image set in the list")
        image_set = image_set_list.GetImageSet(0)
        self.assertEqual(len(image_set.GetNames()),1)
        self.assertEqual(image_set.GetNames()[0],"my_image")
        self.assertTrue(image_set.GetImage("my_image"))
        
    def test_01_02LoadImageTextMatchMany(self):
        l=LI.LoadImages()
        l.CreateFromAnnotations()
        l.Variable(LI.MATCH_STYLE_VAR).SetValue(LI.MS_EXACT_MATCH)
        l.Variable(LI.PATHNAME_VAR).SetValue(os.path.join(T.ExampleImagesDirectory(),"ExampleSBSImages"))
        for i in range(0,LI.MAX_IMAGE_COUNT):
            ii = i+1
            l.Variable(LI.FIRST_IMAGE_VAR+i*2).SetValue("1-0%(ii)d-A-0%(ii)d.tif"%(locals()))
            l.Variable(LI.FIRST_IMAGE_VAR+1+i*2).SetValue("my_image%(i)d"%(locals()))
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.PrepareRun(pipeline, image_set_list)
        self.assertEqual(image_set_list.Count(),1,"Expected one image set, there were %d"%(image_set_list.Count()))
        image_set = image_set_list.GetImageSet(0)
        self.assertEqual(len(image_set.GetNames()),LI.MAX_IMAGE_COUNT)
        for i in range(0,LI.MAX_IMAGE_COUNT):
            self.assertTrue("my_image%d"%(i) in image_set.GetNames())
            self.assertTrue(image_set.GetImage("my_image%d"%(i)))
        
    def test_02_01LoadImageRegexMatch(self):
        l=LI.LoadImages()
        l.CreateFromAnnotations()
        l.Variable(LI.MATCH_STYLE_VAR).SetValue(LI.MS_REGULAR_EXPRESSIONS)
        l.Variable(LI.PATHNAME_VAR).SetValue(os.path.join(T.ExampleImagesDirectory(),"ExampleSBSImages"))
        l.Variable(LI.FIRST_IMAGE_VAR).SetValue("Channel1-[0-1][0-9]-A-01")
        l.Variable(LI.FIRST_IMAGE_VAR+1).SetValue("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.PrepareRun(pipeline, image_set_list)
        self.assertEqual(image_set_list.Count(),1,"Expected one image set in the list")
        image_set = image_set_list.GetImageSet(0)
        self.assertEqual(len(image_set.GetNames()),1)
        self.assertEqual(image_set.GetNames()[0],"my_image")
        self.assertTrue(image_set.GetImage("my_image"))
        
        
if __name__=="main":
    unittest.main()