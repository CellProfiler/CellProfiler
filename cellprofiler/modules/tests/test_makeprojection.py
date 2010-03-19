'''test_makeprojection - Test the MakeProjection module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
from matplotlib.image import pil_to_array
import numpy as np
import os
import Image as PILImage
import scipy.ndimage
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.makeprojection as M

IMAGE_NAME = 'image'
PROJECTED_IMAGE_NAME = 'projectedimage'

class TestMakeProjection(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwKs1RMDRTMLC0MjSxMjZXMDIwsFQgGTAwevryMzAw/Gdk'
                'YKiY8zbc2/+Qg4BcBuNThwmT2fltqz6K6ssdTFDqVBB0SnqU2sgaflvvZbNI'
                '/Uz3+lm7L1zaxNT9dKEh38ycNOvPb88cP1c2m5lBn31C2mO5yyo+Kk9vd3YU'
                '7W6UbZ9Ra82qJfPrs7xP9AmlFWsfi12KzTkc0vSz6+bisxcTfq3w2r2uL/tE'
                'h5Xxyp1u0tHfavU5vshf72z/ylZ52EC78TaznNDsgMv93z8evfly1xXBa6ki'
                'B6rVnigqflhgoOvybGe9oFN9KV/+z476e9fVvs2ZLM1fKnPWwe/5zMdzvAum'
                'SMqwntoqlGPsN7czeGHMqvCKO1NV9JSvnH57SSB6Rb9iXo1o5ZGC3q2vdL0e'
                'bTq066ZBPp/hNNNP+9NkBa37ja76vMpY13vYJk/VgpWx/Xa5SOnWroNem0yT'
                '7zDfPnw7ZO6jH/27Y2Mi61mtDvoeeNr3efLby8yM028feTNJ8eUuj+snKraf'
                'Oxi79d8TnjqhrBJjm3nHnhTGr5h+u5a79w0f1y3DsLpHlr9ORPz23Hek5oyx'
                'iXi7tV51vfvPqPL9febB9xe9S/hs0e0m+W/Pb7eO9RvDDjTf79j8tip1z7+d'
                'X4W6fzu8Wb7j97T9/7UnMpeKzpnTcPitVtXR0u59D/oOv3s5+2jnPO1MTn7P'
                'NNEQ02s/axk/XvPWPDW9eqmO39faeX1Rb57Xbz/w/d/7x6r/Gt+c+i/ct++O'
                'NwB/3SPw')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))        
        self.assertEqual(len(pipeline.modules()), 3)
        #
        # Module 2 - image name = DNA, projection_image_name = AverageDNA
        #            projection_type = Average
        #
        # Module 3 - image name = DNA, projection_image_name = MaxBlue
        #            projection_type = Maximum
        #
        for i,projection_image_name, projection_type\
         in ((1,"AverageDNA",M.P_AVERAGE),
             (2,"MaxBlue", M.P_MAXIMUM)):
            module = pipeline.modules()[i]
            self.assertTrue(isinstance(module, M.MakeProjection))
            self.assertEqual(module.image_name.value, "DNA")
            self.assertEqual(module.projection_image_name.value, 
                             projection_image_name)
            self.assertEqual(module.projection_type, projection_type)
    
    def test_01_02_load_v1(self):
        data = ('eJztWd1OGkEUHn60WtuGXrVJb+ZSWtgsaBskjYLStLSiRImNMbYdYYBpd3fI'
                'MGuljUkfq5d9lD5CH6EzuAvsCiwgupqwyQTP2fPNN+c7O7szYyFb2s5uwpeK'
                'CgvZUrxKNAyLGuJVyvQ0NHgMbjGMOK5AaqRhqW7C96YGE6+gmkqvpNKrKkyq'
                '6hqY7ArkC4/Ej/oMgHnxuyBa0Lo1Z9mBnibtfcw5MWrNORAGTy3/H9EOECPo'
                'RMMHSDNxs0th+/NGlZZajc6tAq2YGt5Bem+wuHZM/QSz5m7VBlq3i+QMa/vk'
                'B3alYIft4VPSJNSw8Fb/bm+Hl3IXr9Th71JXh4BLh5BokR6/jH8HuvHhPro9'
                '7omPWDYxKuSUVEykQaKjWmcUsr+UR39zrv6kvUU1ykbE33Phpb3LSG0PV9r4'
                'jAc+4sLLVsJnPP7mDJU51BEv1yfNY7+hEX6FPLKnmAk523jVAx9w4ANgZUTe'
                'BeDkXbD0e8tQaxT9Hrrw0i4y+hWXuXhGZRXASHW47+pH2jkKDcqh2bQmyCg6'
                'hBz9hEBCUS/h5l04+7Jxi2B0vqCDLwh2qD/j9Jq3T4BTX2nncBWZGod5OWlh'
                'jjBRNMpavus8zfq4x3ko3k5+8F31PeanPpM+9zcxzzIeuEXg1FXaF+83jI0B'
                '/NepLydVX/S97u/woO/Iplg6Xae+7u9eog/uNryHwo5xhkU9janpMg7ul8c4'
                'PwBnHaX9aXmj+Fou4PG68iL6WVofsabt0e/rR9l48Thqe8QDY+rG+pEaXzv+'
                'mYglzy+C94lAtp3RK49/UlzdI++UK29py7EfYsSshFbPo3HpKlCD1y1f0vLl'
                'UKvr8SO/5THX+ZPyZDx07LeOa28Kaoyajen34+c8vwlcBgzXqd/+pasTFFsz'
                '3JhmP3dFtxluhvNjnt2VfGf63k5cBkxHp2n1c1d0m+FmuLH2A4HB62X3vl3G'
                'fwHD59Nz4JxP0i6LLVGDUfn/D6bo7UP6pqJRVLk4JVe2xZ/5ngPzUXhiLp7Y'
                'IJ6y3LxzWmOopbQ38iUqz3Q7+XvwJF08yUE8OvqGG50DX6UgzO75b/86Lfbh'
                '69U7KKzIg9DQ+gLgrGu33v82JuELBQKXzjmWPHDhnjHZef4G4z1Xy0Pi7Rxv'
                'Kv4/DzlPxw==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, M.MakeProjection))
        self.assertEqual(module.image_name.value, "OrigRed")
        self.assertEqual(module.projection_image_name.value, "ProjectionRed") 
        self.assertEqual(module.projection_type, M.P_AVERAGE)
    
    def run_image_set(self, projection_type, images_and_masks):
        image_set_list = cpi.ImageSetList()
        for i in range(len(images_and_masks)):
            pixel_data, mask = images_and_masks[i]
            if mask is None:
                image = cpi.Image(pixel_data)
            else:
                image = cpi.Image(pixel_data, mask)
            image_set_list.get_image_set(i).add(IMAGE_NAME, image)
        
        pipeline = cpp.Pipeline()
        module = M.MakeProjection()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.projection_image_name.value = PROJECTED_IMAGE_NAME
        module.projection_type.value = projection_type
        pipeline.add_module(module)
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, {},
                             range(1,len(images_and_masks)+1))
        for i in range(len(images_and_masks)):
            workspace = cpw.Workspace(pipeline, module, 
                                      image_set_list.get_image_set(i),
                                      cpo.ObjectSet(),
                                      cpmeas.Measurements(),
                                      image_set_list)
            module.run(workspace)
            image = workspace.image_set.get_image(PROJECTED_IMAGE_NAME)
        return image
    
    def test_02_01_average(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(10,10)).astype(np.float32), None)
                             for i in range(3)]
        expected = np.zeros((10,10), np.float32)
        for image, mask in images_and_masks:
            expected += image
        expected = expected / len(images_and_masks)
        image = self.run_image_set(M.P_AVERAGE, images_and_masks)
        self.assertFalse(image.has_mask)
        self.assertTrue(np.all(np.abs(image.pixel_data - expected) < 
                               np.finfo(float).eps))
    
    def test_02_02_average_mask(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(100,100)).astype(np.float32), 
                             np.random.uniform(size=(100,100)) > .3)
                             for i in range(3)]
        expected = np.zeros((100,100), np.float32)
        expected_count = np.zeros((100,100), np.float32)
        expected_mask = np.zeros((100,100), bool)
        for image, mask in images_and_masks:
            expected[mask] += image[mask]
            expected_count[mask] += 1
            expected_mask = mask | expected_mask
        expected = expected / expected_count
        image = self.run_image_set(M.P_AVERAGE, images_and_masks)
        self.assertTrue(image.has_mask)
        self.assertTrue(np.all(expected_mask == image.mask))
        np.testing.assert_almost_equal(image.pixel_data[image.mask],
                                       expected[expected_mask])
    
    def test_02_03_average_color(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(10,10,3)).astype(np.float32), None)
                             for i in range(3)]
        expected = np.zeros((10,10,3), np.float32)
        for image, mask in images_and_masks:
            expected += image
        expected = expected / len(images_and_masks)
        image = self.run_image_set(M.P_AVERAGE, images_and_masks)
        self.assertFalse(image.has_mask)
        self.assertTrue(np.all(np.abs(image.pixel_data - expected) < 
                               np.finfo(float).eps))
    
    def test_03_01_maximum(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(10,10)).astype(np.float32), None)
                             for i in range(3)]
        expected = np.zeros((10,10), np.float32)
        for image, mask in images_and_masks:
            expected = np.maximum(expected,image)
        image = self.run_image_set(M.P_MAXIMUM, images_and_masks)
        self.assertFalse(image.has_mask)
        self.assertTrue(np.all(np.abs(image.pixel_data - expected) < 
                               np.finfo(float).eps))
    
    def test_03_02_maximum_mask(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(100,100)).astype(np.float32), 
                             np.random.uniform(size=(100,100)) > .3)
                             for i in range(3)]
        expected = np.zeros((100,100), np.float32)
        expected_mask = np.zeros((100,100), bool)
        for image, mask in images_and_masks:
            expected[mask] = np.maximum(expected[mask],image[mask])
            expected_mask = mask | expected_mask
        image = self.run_image_set(M.P_MAXIMUM, images_and_masks)
        self.assertTrue(image.has_mask)
        self.assertTrue(np.all(expected_mask == image.mask))
        self.assertTrue(np.all(np.abs(image.pixel_data[image.mask] -
                                      expected[expected_mask]) < 
                               np.finfo(float).eps))
    
    def test_03_03_maximum_color(self):
        np.random.seed(0)
        images_and_masks = [(np.random.uniform(size=(10,10,3)).astype(np.float32), None)
                             for i in range(3)]
        expected = np.zeros((10,10,3), np.float32)
        for image, mask in images_and_masks:
            expected = np.maximum(expected, image)
        image = self.run_image_set(M.P_MAXIMUM, images_and_masks)
        self.assertFalse(image.has_mask)
        self.assertTrue(np.all(np.abs(image.pixel_data - expected) < 
                               np.finfo(float).eps))
    
