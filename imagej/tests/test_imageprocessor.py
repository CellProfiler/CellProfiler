"""test_imageprocessor - test imageprocessor.py"""

__version__ = "$Revision$"

import unittest

import javabridge as J
import numpy as np

import imagej.imageplus as I
import imagej.imageprocessor as IP


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        J.attach()

    def tearDown(self):
        J.detach()

    def test_01_01_get_image(self):
        from cellprofiler.modules.tests import maybe_download_example_image

        folder = "ExampleCometAssay"
        fn = "CometTails.tif"
        file_name = maybe_download_example_image([folder], fn)
        imageplus_obj = I.load_imageplus(file_name)
        pixels = IP.get_image(imageplus_obj.getProcessor())
        pass

    def test_01_02_make_image_processor(self):
        np.random.seed(102)
        image = np.random.uniform(size=(30, 50)).astype(np.float32)
        image_processor = IP.make_image_processor(image)
        result = IP.get_image(image_processor)
        self.assertEqual(image.shape[0], result.shape[0])
        self.assertEqual(image.shape[1], result.shape[1])
        self.assertTrue(np.all(result == image))
