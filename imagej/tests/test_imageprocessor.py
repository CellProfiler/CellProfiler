'''test_imageprocessor - test imageprocessor.py

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import numpy as np
import os
import unittest

import cellprofiler.utilities.jutil as J
import imagej.imageplus as I
import imagej.imageprocessor as IP

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        from cellprofiler.modules.tests import example_images_directory
        self.root_dir = example_images_directory()
        J.attach()
        
    def tearDown(self):
        J.detach()
        
    def test_01_01_get_image(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleCometAssay", "CometTails.tif")
        imageplus_obj = I.load_imageplus(file_name)
        pixels = IP.get_image(imageplus_obj.getProcessor())
        pass
    
    def test_01_02_make_image_processor(self):
        np.random.seed(102)
        image = np.random.uniform(size=(30,50)).astype(np.float32)
        image_processor = IP.make_image_processor(image)
        result = IP.get_image(image_processor)
        self.assertEqual(image.shape[0], result.shape[0])
        self.assertEqual(image.shape[1], result.shape[1])
        self.assertTrue(np.all(result == image))


