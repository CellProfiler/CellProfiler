'''test_measureneurons.py - test the MeasureNeurons module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import base64
import numpy as np
import os
import scipy.ndimage
from StringIO import StringIO
import unittest
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.measureneurons as M

IMAGE_NAME = "MyImage"
OBJECT_NAME = "MyObject"

class TestMeasureNeurons(unittest.TestCase):
    def test_01_01_load_matlab(self):
        '''Load a Matlab version of MeasureNeurons'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggH0l6JeQoGZgqGplbGFlZGlgpGBoYGCiQDBkZPX34GBoYyJgaG'
                'ijlPw2PzLxuIlP4y7G1ctYmrc/vTie2GUY0X5MJXcihtXLPossq9AuHtEqpl'
                'U28Enza273L/x/S/Y4Za7+My30dlb1mmrXlh9X1edU1OvDovw4FK7obmF5US'
                'X7lmbK/KkQ4S/8Zg2HbSYk2zVIVc//+uX7w8BjJVL9iqZFoTLk4q7O558TVn'
                'U0WpZ+w7Relfm4qv5LzjDxSI65104EepvP9E8wP3BU9sO/JWeNdRxWUqff0f'
                'r150jTs1ZU/vinpLr1IDEdlVgfmejw/9Z1ryb6voLlPV/2l31l+csrfGvJXP'
                'uPbEnPVMy+c2B599EOJ4/l2TUuFWnxhenhX8pY+q7u4rnf+z5Hl5+J1Kv5jY'
                'SPHE8lvL8132udnEpdT2i8gkfl/hqWFc2RE0Z9PnHzJvz9Z+WJFSoHVg2f9t'
                'cpUW/aIyifEbJ8x+b2vVco8p/fovCUkZ5pJvR5xjl3/LVD+8+I1NnHhP8Guf'
                '32vu3jz+r2LJA47bSjwTBTbPPKJeWXfiYPWqB5zSwuL3k6Y7nz9iJ6Oft/b3'
                'g+XR8iUVJZbdeextsrxyr/ZZ3nboL7Up+XCMcU4r35cJLZsz917uPf84/ouM'
                'yNOFu+23nXW+2rynKTqy3bfU49Hsr1/vzyySvXD1ff5GCfYp/mX9m1/dW3I4'
                '/0Si5gWzwpJXq1qS3UzFfL8t+L7qf7NIul/21X+3p+WlFv4X38evet+59tnc'
                '7ckTrjremGLML7yl5EHKHefH73atuJZ+t/jXRKfrk+yWdJ15PrNJOee0a8vD'
                'i2vkE+L/tWkcWr3wAedvoYMT2E+42SxQidmXzlf0ZYXLtcLVmy7+d75Z81X9'
                '2KX/ZXExut8qqz89frLd8vxv4SnR38Wz/0zepv3f3ukznzkA9yxhQQ==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, M.MeasureNeurons))
        self.assertEqual(module.seed_objects_name, "Soma")
        self.assertEqual(module.image_name, "DNA")
        
    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8977

MeasureNeurons:[module_num:1|svn_version:\'8401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Seed objects name\x3A:Nucs
    Skeletonized image name\x3A:DNA
    Do you want to save the branchpoint image?:Yes
    Branchpoint image name\x3A:BPImg
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, M.MeasureNeurons))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.seed_objects_name, "Nucs")
        self.assertTrue(module.wants_branchpoint_image)
        self.assertEqual(module.branchpoint_image_name, "BPImg")

    def make_workspace(self, labels, image, mask = None):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        img = cpi.Image(image, mask)
        image_set.add(IMAGE_NAME, img)
        
        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        o.segmented = labels
        object_set.add_objects(o, OBJECT_NAME)
        
        module = M.MeasureNeurons()
        module.image_name.value = IMAGE_NAME
        module.seed_objects_name.value = OBJECT_NAME
        module.module_num = 1
        
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        return workspace, module
    
    def test_02_01_empty(self):
        workspace, module = self.make_workspace(np.zeros((20,10), int),
                                                np.zeros((20,10), bool))
        #
        # Make sure module tells us about the measurements
        #
        columns = module.get_measurement_columns(None)
        features = [c[1] for c in columns]
        features.sort()
        expected = [M.F_NUMBER_NON_TRUNK_BRANCHES, M.F_NUMBER_TRUNKS]
        expected.sort()
        for feature, expected in zip(features, expected):
            expected_feature = "_".join((M.C_NEURON, expected, IMAGE_NAME))
            self.assertEqual(feature, expected_feature)
        self.assertTrue(all([c[0] == OBJECT_NAME for c in columns]))
        self.assertTrue(all([c[2] == cpmeas.COLTYPE_INTEGER for c in columns]))
        
        categories = module.get_categories(None, OBJECT_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], M.C_NEURON)
        self.assertEqual(len(module.get_categories(None, "Foo")), 0)
        
        measurements = module.get_measurements(None, OBJECT_NAME, M.C_NEURON)
        self.assertEqual(len(measurements), 2)
        self.assertNotEqual(measurements[0], measurements[1])
        self.assertTrue(all([m in (M.F_NUMBER_NON_TRUNK_BRANCHES, 
                                   M.F_NUMBER_TRUNKS)
                             for m in measurements]))
        
        self.assertEqual(len(module.get_measurements(None,"Foo", M.C_NEURON)), 0)
        self.assertEqual(len(module.get_measurements(None,OBJECT_NAME, "Foo")), 0)
        
        for feature in (M.F_NUMBER_NON_TRUNK_BRANCHES, M.F_NUMBER_TRUNKS):
            images = module.get_measurement_images(None, OBJECT_NAME, 
                                                   M.C_NEURON, feature)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0], IMAGE_NAME)
        
                        
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature in (M.F_NUMBER_NON_TRUNK_BRANCHES, M.F_NUMBER_TRUNKS):
            mname = "_".join((M.C_NEURON, expected, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 0)
            
    def test_02_02_trunk(self):
        '''Create an image with one soma with one neurite'''
        image = np.zeros((20,15), bool)
        image[9,5:] = True
        labels = np.zeros((20,15), int)
        labels[6:12,2:8] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 0),
                                  (M.F_NUMBER_TRUNKS, 1)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected)
            
    def test_02_03_trunks(self):
        '''Create an image with two soma and a neurite that goes through both'''
        image = np.zeros((30,15),bool)
        image[1:25,7] = True
        labels = np.zeros((30,15),int)
        labels[6:13,3:10] = 1
        labels[18:26,3:10] = 2
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, [0,0]),
                                  (M.F_NUMBER_TRUNKS, [2,1])):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 2)
            for i in range(2):
                self.assertEqual(data[i], expected[i])

    def test_02_04_branch(self):
        '''Create an image with one soma and a neurite with a branch'''
        image = np.zeros((30,15),bool)
        image[6:15,7] = True
        image[15+np.arange(3),7+np.arange(3)] = True
        image[15+np.arange(3),7-np.arange(3)] = True
        labels = np.zeros((30,15), int)
        labels[1:8,3:10] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 1),
                                  (M.F_NUMBER_TRUNKS, 1)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected)
