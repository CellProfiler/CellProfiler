import unittest
import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.settings as cps
from cellprofiler.modules import instantiate_module

INPUT_IMAGE_NAME = "inputimage"

class TestExample3b(unittest.TestCase):
    def setUp(self):
        self.module = instantiate_module("Example3b")
        self.module.input_image_name.value = INPUT_IMAGE_NAME
        self.E = __import__(self.module.__class__.__module__)
    
    def test_00_00_can_load(self):
        pass
        
    def test_01_01_get_categories_pos(self):
        c = self.module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(c), 1,
                         "You returned the wrong number of categories")
        
    def test_01_02_get_categories_neg(self):
        c = self.module.get_categories(None, "Grumblefish")
        self.assertEqual(
            len(c), 0,
            "You shouldn't return a category unless the object_name is Image")
        
    def test_02_01_get_measurements_pos(self):
        m = self.module.get_measurements(None, cpmeas.IMAGE, self.E.C_EXAMPLE3)
        self.assertEqual(len(m), 1, "You should return one measurement")
        self.assertEqual(m[0], self.E.FTR_VARIANCE)
        
    def test_02_02_get_measurements_neg(self):
        for object_name, category, err in (
            (cpmeas.IMAGE, "bogus", "the category isn't %s" % self.E.C_EXAMPLE3),
            ("bogus", self.E.C_EXAMPLE3, "the image name isn't %s" % cpmeas.IMAGE)):
            m = self.module.get_measurements(None, object_name, category)
            self.assertEqual(
                len(m), 0, 
                "You shouldn't return a measurement because " + err)
            
    def test_03_01_get_measurement_images_pos(self):
        i = self.module.get_measurement_images(
            None, cpmeas.IMAGE, self.E.C_EXAMPLE3, self.E.FTR_VARIANCE)
        self.assertEqual(len(i), 1, "You should return the input image name")
        self.assertEqual(i[0], INPUT_IMAGE_NAME)
        
    def test_03_02_get_measurement_images_neg(self):
        args = (cpmeas.IMAGE, self.E.C_EXAMPLE3, self.E.FTR_VARIANCE)
        items = ("object name", "category", "feature")
        for i in range(len(args)):
            args_copy = list(args)
            args_copy[i] = "bogus"
            imgs = self.module.get_measurement_images(
                None, *args_copy)
            self.assertEqual(len(imgs), 0,
                             "The %s isn't %s" % (items[i], args[i]))
        
        