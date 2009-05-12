'''test_enhanceorsuppressspeckles - test the EnhanceOrSuppressSpeckles module
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
import numpy as np
from StringIO import StringIO
import unittest
import zlib

import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.enhanceorsuppressspeckles as E

INPUT_IMAGE_NAME = 'myimage'
OUTPUT_IMAGE_NAME = 'myfilteredimage'

class TestEnhanceOrSuppressSpeckles(unittest.TestCase):
    def make_workspace(self, image,mask):
        '''Make a workspace for testing FilterByObjectMeasurement'''
        module = E.EnhanceOrSuppressSpeckles()
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image, mask))
        module.image_name.value = INPUT_IMAGE_NAME
        module.filtered_image_name.value = OUTPUT_IMAGE_NAME
        return workspace, module
    
    def test_00_00_enhance_zero(self):
        '''Test enhance of an image of all zeros'''
        workspace, module = self.make_workspace(np.zeros((10,10)), None)
        self.assertTrue(module.method.value == E.ENHANCE)
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == 0))
        
    def test_00_01_suppress_zero(self):
        '''Test suppress of an image of all zeros'''
        workspace, module = self.make_workspace(np.zeros((10,10)), None)
        module.method.value = E.SUPPRESS
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == 0))
        
    def test_01_01_load_v1(self):
        data = ( 'eJztWNFO2zAUdUqBsUkr28v26Ee60aotQ4NqKu0oEtUIVLRiQohtpnXba'
                 'EkcOQlrNyHtcZ+0x33OHvcJs4NDUhMIbccmpqay2nt9zz3Xx0lsV600dy'
                 'qv4Wo2B9VKM9PRdAzrOnI6hBpFaDrLcJNi5OA2JGYRqsSEKhrAfB7mV4o'
                 'rheLqOizkcutgvEupqQ/Z19pjAObY9z3WEqJrVthKqHG7gR1HM7v2LEiC'
                 'p8L/g7UDRDV0ouMDpLvYDih8f83skObAuuhSSdvV8S4ywsHs2nWNE0ztv'
                 'Y4PFN11rY/1hvYZS0Pww/bxqWZrxBR4kV/2XvASR+LlOnyfD3RQJB24Lq'
                 'mQn8dvgyA+GaHbo1D8orA1s62dam0X6VAzUPeiCm8eYvLNS/m4rQ5qPI2'
                 'HL8fgFyU8b03cdzJbfdRyoIGcVu8meVJSnpRXx5bZQ2YLt4N6cjF5lKE8'
                 'CliZQAfBfiMd7kt4blcJNIkDXRsH8xFXf2IoTwLkX46H2yWXcXMSzr983'
                 'AII6oy7D59I4+V2FXeQqzvQmy1Y1ShuOYQOJqrjX+Kixj0zNO4ZcMietr'
                 'vK9zdwk75/bktX+T2R/4P6RPElh/iS7Pk08SR8X2P43oBhXbn9bmmj/op'
                 'vBHAp+zz9nltvsa7vk0+lo0qmfpz2PZtEdw2zdJTLrB9/yS8Xzs6DGxpD'
                 'es505LhHqb8XU/+aVD+3eQ2HGFFR2IuzdIa72AbG6QlfQfiqaBB4Jqnz5'
                 '9xo6/e4POUYPaLWF2+x71LiWrfPH7XOB/yQbUGwdZfeS1PcFPc/4soh3P'
                 'Q5nuJGxS0qV6938jmDx38A199vz8Dw/cbtFttiWJTw/yVo1vAOz3ZWJ6h'
                 '9fnrN7rCftdBBlvP0Y3i2JZ7tq3jw+aGOUNu1LIpt27Zw6yPvEce9PdoQ'
                 'PQ3RI+u5EMEf1iXBPqnk9fMg6x/My6+NcfgSymW+BzG4pFCS476B0eZ96'
                 'Zp4f2zjxv8G/FcCeg==')
        pipeline = cpp.Pipeline()
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()),2)
        module = pipeline.modules()[1]
        self.assertEqual(module.module_name,'EnhanceOrSuppressSpeckles')
        self.assertEqual(module.image_name.value, 'MyImage')
        self.assertEqual(module.filtered_image_name.value, 'MyEnhancedImage')
        self.assertEqual(module.method.value, E.ENHANCE)
        self.assertEqual(module.object_size, 17)
    
    def test_02_01_enhance(self):
        '''Enhance an image composed of two circles of different diameters'''
        #
        # Make an image which has circles of diameters 9 and 7. We should
        # keep the smaller circle and erase the larger
        # 
        image = np.zeros((10,20))
        expected = np.zeros((10,20))
        i,j = np.mgrid[-5:5,-5:15]
        image[i**2+j**2 <= 16] = 1
        i,j = np.mgrid[-5:5,-15:5]
        image[i**2+j**2 <= 9] = 1
        expected[i**2+j**2 <= 9] = 1
        workspace, module = self.make_workspace(image,
                                                np.ones(image.shape, bool))
        module.method.value = E.ENHANCE
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == expected))
        
    def test_02_02_suppress(self):
        '''Suppress a speckle in an image composed of two circles'''
        image = np.zeros((10,20))
        expected = np.zeros((10,20))
        i,j = np.mgrid[-5:5,-5:15]
        image[i**2+j**2 <= 16] = 1
        expected[i**2+j**2 <= 16] = 1
        i,j = np.mgrid[-5:5,-15:5]
        image[i**2+j**2 <= 9] = 1
        workspace, module = self.make_workspace(image, None)
        module.method.value = E.SUPPRESS
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == expected))
    
    def test_03_01_enhancemask(self):
        '''Enhance a speckles image, masking a portion'''
        image = np.zeros((10,10))
        mask  = np.ones((10,10),bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation and it
        # should be zero after the subtraction
        #
        i,j = np.mgrid[-5:5,-5:5]
        image[5,5] = 1
        mask[np.logical_and(i**2+j**2<=16,image==0)] = False
        #
        # Prove that, without the mask, the image is zero
        #
        workspace, module = self.make_workspace(image, None)
        module.method.value = E.ENHANCE
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(np.all(result.pixel_data == image))
        #
        # rescue the point with the mask
        #
        workspace, module = self.make_workspace(image, mask)
        module.method.value = E.ENHANCE
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == 0))

    def test_03_02_suppressmask(self):
        '''Suppress a speckles image, masking a portion'''
        image = np.zeros((10,10))
        mask  = np.ones((10,10),bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation
        #
        i,j = np.mgrid[-5:5,-5:5]
        image[5,5] = 1
        mask[np.logical_and(i**2+j**2<=16,image==0)] = False
        #
        # Prove that, without the mask, the speckle is removed
        #
        workspace, module = self.make_workspace(image, None)
        module.method.value = E.SUPPRESS
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(np.all(result.pixel_data == 0))
        #
        # rescue the point with the mask
        #
        workspace, module = self.make_workspace(image, mask)
        module.method.value = E.SUPPRESS
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == image))        