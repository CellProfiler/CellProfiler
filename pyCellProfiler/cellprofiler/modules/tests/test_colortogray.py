"""test_colortogray.py - test the ColorToGray module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import unittest

import numpy

import cellprofiler.pipeline as cpp
import cellprofiler.measurements as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo

import cellprofiler.modules.injectimage as cpm_inject
import cellprofiler.modules.colortogray as cpm_ctg

import cellprofiler.modules.tests as cpmt
from cellprofiler.workspace import Workspace

class TestColorToGray(unittest.TestCase):
    def get_my_image(self):
        """A color image with red in the upper left, green in the lower left and blue in the upper right"""
        img = numpy.zeros((50,50,3))
        img[0:25,0:25,0] = 1
        img[0:25,25:50,1] = 1
        img[25:50,0:25,2] = 1
        return img
        
    def test_00_00_init(self):
        x = cpm_ctg.ColorToGray()
    
    def test_01_01_combine(self):
        img = self.get_my_image()
        inj = cpm_inject.InjectImage("my_image",img)
        inj.module_num = 1
        ctg = cpm_ctg.ColorToGray()
        ctg.module_num = 2
        ctg.image_name.value = "my_image"
        ctg.combine_or_split.value = cpm_ctg.COMBINE
        ctg.red_contribution.value = 1
        ctg.green_contribution.value = 2
        ctg.blue_contribution.value = 3
        ctg.grayscale_name.value = "my_grayscale"
        pipeline = cpp.Pipeline()
        pipeline.add_module(inj)
        pipeline.add_module(ctg)
        pipeline.test_valid()
        
        measurements = cpm.Measurements()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        inj.prepare_run(pipeline, image_set_list, None)
        inj.prepare_group(pipeline, image_set_list, {}, [1])
        image_set = image_set_list.get_image_set(0)
        inj.run(Workspace(pipeline,inj,image_set,object_set,measurements,None))
        ctg.run(Workspace(pipeline,ctg,image_set,object_set,measurements,None))
        grayscale = image_set.get_image("my_grayscale")
        self.assertTrue(grayscale)
        img = grayscale.image
        self.assertAlmostEqual(img[0,0],1.0/6.0)
        self.assertAlmostEqual(img[0,25],1.0/3.0)
        self.assertAlmostEqual(img[25,0],1.0/2.0)
        self.assertAlmostEqual(img[25,25],0)
     
    def test_01_02_split_all(self):
        img = self.get_my_image()
        inj = cpm_inject.InjectImage("my_image",img)
        inj.module_num = 1
        ctg = cpm_ctg.ColorToGray()
        ctg.module_num = 2
        ctg.image_name.value = "my_image"
        ctg.combine_or_split.value = cpm_ctg.SPLIT
        ctg.use_red.value = True
        ctg.use_blue.value = True
        ctg.use_green.value = True
        ctg.red_name.value   = "my_red"
        ctg.green_name.value = "my_green"
        ctg.blue_name.value  = "my_blue"
        pipeline = cpp.Pipeline()
        pipeline.add_module(inj)
        pipeline.add_module(ctg)
        pipeline.test_valid()
        
        measurements = cpm.Measurements()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        inj.prepare_run(pipeline, image_set_list, None)
        inj.prepare_group(pipeline, image_set_list, {}, [1])
        image_set = image_set_list.get_image_set(0)
        inj.run(Workspace(pipeline,inj,image_set,object_set,measurements,None))
        ctg.run(Workspace(pipeline,ctg,image_set,object_set,measurements,None))
        red = image_set.get_image("my_red")
        self.assertTrue(red)
        img = red.image
        self.assertAlmostEqual(img[0,0],1)
        self.assertAlmostEqual(img[0,25],0)
        self.assertAlmostEqual(img[25,0],0)
        self.assertAlmostEqual(img[25,25],0)
        green = image_set.get_image("my_green")
        self.assertTrue(green)
        img = green.image
        self.assertAlmostEqual(img[0,0],0)
        self.assertAlmostEqual(img[0,25],1)
        self.assertAlmostEqual(img[25,0],0)
        self.assertAlmostEqual(img[25,25],0)        
        blue = image_set.get_image("my_blue")
        self.assertTrue(blue)
        img = blue.image
        self.assertAlmostEqual(img[0,0],0)
        self.assertAlmostEqual(img[0,25],0)
        self.assertAlmostEqual(img[25,0],1)
        self.assertAlmostEqual(img[25,25],0)
    
    def test_2_1_load_matlab_combine(self):
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDEyIDA5OjE3OjA0IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAdAEAAHiczZS9TsMwEIAvTdpSKlURA2JkZONvYUR0QAwU1EYVq6teI0uJHeUHUZ6Ax+rMU2G3TupEoUlDB05ynDvfd2efdR4AwFcXoCPmIzFasJG20g1tSH2CcUyZG7XBgjNlX4kxJSElMw+nxEswgkxS+xNbcGcZZEvPfJ54OCK+7ixklPgzDKOXRQqq5Vf6gd6EfiLkJXUb4zuNKGeKV/GL1iwvjwt5B/JjbutglNThRLNL/zvY+lsl/nocW+kORvFjSJZ1+G6Bl/qQ+zPKcJ3/qoI3c7wJ7JIcZN9VeY0cb8C1qte+3E1D7rYm17Q+Zfci6zPG+Zq/r+B7Bb6X1ReRKftf7ulBtOBAi7PvvDJ290Ef8n1Qdd7jwj6lTn3iohvyJPh9H1Vx7UJcOxf3nLI5BnXOe6g8/y1OALvvUe+bOnn1e7eVPuQeDx0un4Ysjt53nYq8LfHXL+Gavit18hkNOEuQ36cb7m3Pul7s8E8ltf8AGWV9Kw=='
        pipeline = cpmt.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertEqual(module.image_name.value, "TestGray")
        self.assertTrue(module.should_combine)
        self.assertEqual(module.grayscale_name.value, "TestGray")
        self.assertEqual(module.red_contribution.value, 1)
        self.assertEqual(module.green_contribution.value, 2)
        self.assertEqual(module.blue_contribution.value, 3)
    
    def test_2_2_load_matlab_split(self):
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDEyIDA5OjI3OjMwIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAaQEAAHiczVTNToQwEJ6ysBg3IcSD8ejRm38Xj0YPxoOrWcjGazd0SROghB+z69Pt2afxESzZwpYGAXFNnGRSZjrffMN0WgsAVibAmK8HXDXYiiFsJGlhOyTLaOSnBuhwIvwbrnOcULwIyBwHOUmhktL/GC2Zu46rrSfm5QGZ4lAO5jLNwwVJ0udlCRTbL3RFAoe+E6hLGTYjbzSlLBJ4kV/1VrwsU3gtrp/arg+ooQ9Hkr+Iv4FdvN4QL+exhe2SNHtI8LoP3lDwhe3EAc0E/0UHflTDjyA6x3upu4sX1fAILnvWq+KuBuKu/7g/ptIfU/RnRrxB9U7hd/N0x6+cJfxD1g1qn/sJ1Of+tqPOQ6XOwqYh9omfsDz+vo6uvLaS167lPaWRR+I+/7svnv+WJ4b2c5Tnrg+vfO62sO9ZwBKXFU9BlUee93EHr8a/Jg24oe9IHz40AKdz5MfxFvf6w76etcSXUvq/AErXefI='
        pipeline = cpmt.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertEqual(module.image_name.value, "TestGray")
        self.assertTrue(module.should_split)
        self.assertTrue(module.use_red.value)
        self.assertEqual(module.red_name.value, "TestRed")
        self.assertFalse(module.use_green.value)
        self.assertTrue(module.use_blue.value)
        self.assertEqual(module.blue_name.value, "TestBlue")
    
    def test_2_3_load_combine(self):
        data = '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'
        pipeline = cpmt.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertEqual(module.image_name.value, "TestInput")
        self.assertTrue(module.should_combine)
        self.assertEqual(module.grayscale_name.value, "TestGray")
        self.assertEqual(module.red_contribution.value, 1)
        self.assertEqual(module.green_contribution.value, 2)
        self.assertEqual(module.blue_contribution.value, 3)
    
    def test_2_4_load_split(self):
        data = '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'
        pipeline = cpmt.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertEqual(module.image_name.value, "TestInput")
        self.assertTrue(module.should_split)
        self.assertTrue(module.use_red.value)
        self.assertEqual(module.red_name.value, "TestRed")
        self.assertTrue(module.use_green.value)
        self.assertEqual(module.green_name.value, "TestGreen")
        self.assertFalse(module.use_blue.value)
        
        
