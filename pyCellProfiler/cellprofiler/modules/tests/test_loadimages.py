"""Test the LoadImages module
"""
import unittest
import os
import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.tests as T
import cellprofiler.cpimage as I
import cellprofiler.pipeline as P

class testLoadImages(unittest.TestCase):
    def test_00_00init(self):
        x=LI.LoadImages()
        x.create_from_annotations()
    
    def test_00_01version(self):
        self.assertEqual(LI.LoadImages().variable_revision_number(),4,"LoadImages' version number has changed")
    
    def test_01_01load_image_text_match(self):
        l=LI.LoadImages()
        l.create_from_annotations()
        l.variable(LI.MATCH_STYLE_VAR).set_value(LI.MS_EXACT_MATCH)
        l.variable(LI.PATHNAME_VAR).set_value(os.path.join(T.example_images_directory(),"ExampleSBSImages"))
        l.variable(LI.FIRST_IMAGE_VAR).set_value("1-01-A-01.tif")
        l.variable(LI.FIRST_IMAGE_VAR+1).set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
        
    def test_01_02load_image_text_match_many(self):
        l=LI.LoadImages()
        l.create_from_annotations()
        l.variable(LI.MATCH_STYLE_VAR).set_value(LI.MS_EXACT_MATCH)
        l.variable(LI.PATHNAME_VAR).set_value(os.path.join(T.example_images_directory(),"ExampleSBSImages"))
        for i in range(0,LI.MAX_IMAGE_COUNT):
            ii = i+1
            l.variable(LI.FIRST_IMAGE_VAR+i*2).set_value("1-0%(ii)d-A-0%(ii)d.tif"%(locals()))
            l.variable(LI.FIRST_IMAGE_VAR+1+i*2).set_value("my_image%(i)d"%(locals()))
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set, there were %d"%(image_set_list.count()))
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),LI.MAX_IMAGE_COUNT)
        for i in range(0,LI.MAX_IMAGE_COUNT):
            self.assertTrue("my_image%d"%(i) in image_set.get_names())
            self.assertTrue(image_set.get_image("my_image%d"%(i)))
        
    def test_02_01load_image_regex_match(self):
        l=LI.LoadImages()
        l.create_from_annotations()
        l.variable(LI.MATCH_STYLE_VAR).set_value(LI.MS_REGULAR_EXPRESSIONS)
        l.variable(LI.PATHNAME_VAR).set_value(os.path.join(T.example_images_directory(),"ExampleSBSImages"))
        l.variable(LI.FIRST_IMAGE_VAR).set_value("Channel1-[0-1][0-9]-A-01")
        l.variable(LI.FIRST_IMAGE_VAR+1).set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
        
        
if __name__=="main":
    unittest.main()
