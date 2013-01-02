'''test_imageprocessor - test imageprocessor.py

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
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
from cellprofiler.modules.tests import example_images_directory

class TestImageProcessor(unittest.TestCase):
        
    def test_01_01_get_image(self):
        file_name = os.path.join(example_images_directory(), 
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


