"""test_injectimage.py - test the InjectImage module (which is used for testing)

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import unittest
import numpy

from cellprofiler.preferences import set_headless
set_headless()

from cellprofiler.modules.injectimage import InjectImage
import cellprofiler.cpimage
import cellprofiler.pipeline
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw

class testInjectImage(unittest.TestCase):
    def test_00_00_init(self):
        image = numpy.zeros((10,10),dtype=float)
        x = InjectImage("my_image", image)
    
    def test_01_01_get_from_image_set(self):
        image = numpy.zeros((10,10),dtype=float)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        ii = InjectImage("my_image", image)
        pipeline = cellprofiler.pipeline.Pipeline()
        ii.prepare_run(cpw.Workspace(pipeline, ii, None, None,
                                     cpmeas.Measurements(), image_set_list))
        ii.prepare_group(pipeline, image_set_list, {}, [1])
        image_set = image_set_list.get_image_set(0)
        self.assertTrue(image_set,"No image set returned from ImageSetList.GetImageSet")
        my_image = image_set.get_image("my_image")
        self.assertTrue(my_image, "No image returned from ImageSet.GetImage")
        self.assertEqual(my_image.image.shape[0],10,"Wrong image shape")

if __name__=="main":
    unittest.main()
        
