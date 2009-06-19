'''test_measuretexture - test the MeasureTexture module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import base64
from matplotlib.image import pil_to_array
import numpy as np
import os
import Image as PILImage
from scipy.io.matlab import loadmat
import scipy.ndimage as scind
from StringIO import StringIO
import unittest
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.measuretexture as M
from cellprofiler.modules.tests import example_images_directory

INPUT_IMAGE_NAME = 'Cytoplasm'
INPUT_OBJECTS_NAME = 'inputobjects'
class TestMeasureTexture(unittest.TestCase):
    def make_workspace(self, image, labels):
        '''Make a workspace for testing MeasureTexture'''
        module = M.MeasureTexture()
        module.image_groups[0].image_name.value = INPUT_IMAGE_NAME
        module.object_groups[0].object_name.value = INPUT_OBJECTS_NAME
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
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image))
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, INPUT_OBJECTS_NAME)
        return workspace, module
    
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUX'
                'AuSk0sSU1RyM+zUggpTVXwKs1TMLBUMDS0MjSxMjFTMDIA8kgGDIye'
                'vvwMDAxzmBgYKua8jfDNvm0gEWZ3cpvQrmlO/JY/djhfs9zSLHf94k'
                'bWayprlhrccC0U8Z+dyxqtdbIusv9v5uQs64lTdYpy1C8ENirF/a6d'
                '/935k7KmL8OBj4IMpXZmFfrNG64q7VQpVYxX9tqYzm4awHr84mf2e/'
                'kL+x9ecNx+IVGD4bOY/fuq5CLj2WfiYycKFgfw79pklG/7jG+i/Jfj'
                'GxO/VDUsP7HzWL2Ax7aIt9K7DjOqxaXIP1zt/7yOM/SP2dHz9xbxqS'
                '7lE73Hv/S503/miBfh4Vdy4y/d7//FO+vS9vnLLixq4175NfjBnOwC'
                'Ka6+Cb/tttl/ChJPzM96s5rzt9aOPRIll3+s1a5rvZM8rbkgYn/u7e'
                'dro303ihcdz5nfWbV05v/pXXsn6Hc+FMzaoBB1wvXRi2cbJx1Y2fD+'
                'j3RFV+UUYYvUC8rnP288a1NisV5ERvF75oGe83ySTunPP4qY2i9l8e'
                'MscbC6F3lUevnbzd9+yx5fv05/813O7zt7fs5ftiX4O/fnI29O1HWf'
                'im//7HRQsOj64hPcBnM9Px55LM57L5vV/8QN6YfWNkkXDDdwv/25Mk'
                'n6Y263ePjRwoMaD9lFclI7nGPN1S1fbdTr06nYxp/eyCqr8jDl6cs9'
                '33b9uFzHnvn4xC67WV4yTnznE2vdZN90us5fLnfPOuKEzZO1zjn2K9'
                'buZz9hM2H6dTnPw984Z4v0C/dHVvTy1Ct22zFJzXl8IaXZJnwuu8qe'
                'Kxar3/l32xzeYLay/taOKx8t7/+3vp979HeIu2y8TEOKbIvPobkJnQ'
                'Ypd9xe3t+z6lN1mNwfrwcrtGWPXHpSvOQBx05rJ7mjj28eOL6uhu3k'
                'rvg4RQkD+c799T827PG/avpVo/jlgw0TX9/btHL/rbjKl/sWb17tZm'
                'Uf9ffa+7AX3+VyzN7nX3txPDf8Vz3jKX7jVwD5C2/v')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[2]
        #
        # image_name = OrigBlue
        # object_name = Nuclei
        # scale = 3,4,5
        #
        self.assertTrue(isinstance(module,M.MeasureTexture))
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(module.image_groups[0].image_name.value, "OrigBlue")
        
        self.assertEqual(module.object_count.value, 1)
        self.assertEqual(module.object_groups[0].object_name.value, "Nuclei")
        
        self.assertEqual(module.scale_count.value, 3)
        for scale, expected in zip([x.scale.value 
                                    for x in module.scale_groups],[3,4,5]):
            self.assertEqual(scale, expected)
        self.assertEqual(module.gabor_angles, 4)
    
    def test_01_02_load_v1(self):
        data = ('eJztWk1P2zAYdj9AdEiMcdkkLj5uE0RpB9LgspZ1bJ3oh0bFtNvS1C2eErt'
                'KHNbutJ+1437OjvsJi9uEJl4gaQptihJhhdfx4+fx69dOeHG90j6rnMBDSY'
                'b1Snu/hzUEW5rCetTQjyFhe/CtgRSGupCSY3hqYPjRIrBYgvLRsf1zcARLs'
                'nwE4l2ZWn3Lvv18AcC6fd+wS9Z5tObYGU/h9jliDJO+uQby4JlT/9suF4qB'
                'lY6GLhTNQuaUwq2vkR5tjwbXj+q0a2mooejexvbVsPQOMsxmzwU6j1t4iLR'
                'z/AMJQ3CbfUJX2MSUOHinf7H2mpcygZf74c/O1A8ZwQ85u+x66nn7D2DaPh'
                '/gtyee9tuOjUkXX+GupWgQ60r/WgXv73VIfxtCf9xuGrh/Yruc4+UQfMaHz'
                '4Ciw1sOwW0LvLy00ZDtvxsqKoO6wtTLKPrXhX643bBUDeGJjrj6Z8W9iujv'
                'NUEvt4vy3oE8J2+Yvx8JvNyuUkgog5bpLIAo/DlfPznwxY62RcRZ1ofPgga'
                'Npvcm3LxxFbZOnwp4bldRT7E0Bmt8kcIqNpDKqDGay+9x42VWnASixdmmMG'
                '5uN5lpwfca7SjauD4O/0FEXNw4uat94PCedYpxIO8V7zUO8j5cHlQrrVocn'
                'CzJRRG3LuDcy8UVnPtd7C/lEHxBwHO7RhgiJmYjj464+uP6edV0NyiJtZ8X'
                '5WTqjLK+k6Azyvshrs5ZcOUQnVvAH6/cnrxXmxbTMOEfr8vQvSr+TXUuV6f'
                '9Hkvkuorz3ZVE/65KHCRVp/helQ6XM+/lEJ1B8dr+TqGqKabpZDCWoTvs77'
                'qg/MtnhPuXPJ12xRNHREWe/pLm96A8wCk1UN+gFunOz/91xnzXIsc5To7xg'
                'Q7m519kfNHON6SysXCISRcN7kBHintYcRWmO2i/9cTV0nSnuBT3EHFlcPt6'
                'DPr/x3QfmWzzqzTeFJfikowL++7aAf71yG06yUj99+G1SuNOcSkuCe+7pH4'
                '3p7gUl+JWDzfMTHFinknM147zUh6eoP3pJfDvT9xWkaYNDMrPzxmSPj7kZU'
                'oaVbqTU1bSmf1rzXPgivMMQnjKAk/5Jh7cRYTh3mhg2GwWo7rCsCrVnNqWX'
                'VtxaznvZQhvSeAt3cSrI8W0DMTQkNk3qT4x2xMzeN4KAXxe/2dt6/Fu4db5'
                'BsA/z9P5//smDl82lxnzec8NbIbg8h5N7jh/gdni7Pkt7d0xLqr9P5vhZ1k=')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[2]
        #
        # image_name = DNA, Cytoplasm
        # object_name = Nuclei
        # scale = 3,4,5
        #
        self.assertTrue(isinstance(module,M.MeasureTexture))
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(module.image_groups[0].image_name.value, "OrigBlue")
        
        self.assertEqual(module.object_count.value, 1)
        self.assertEqual(module.object_groups[0].object_name.value, "Nuclei")
        
        self.assertEqual(module.scale_count.value, 3)
        for scale, expected in zip([x.scale.value 
                                    for x in module.scale_groups],[3,4,5]):
            self.assertEqual(scale, expected)
        self.assertEqual(module.gabor_angles.value, 3)
    
    def test_02_01_compare_to_matlab(self):
        path = os.path.split(__file__)[0]
        mask_file = os.path.join(path, 'Channel2-01-A-01Mask.png')
        mask = pil_to_array(PILImage.open(mask_file))
        mask = np.flipud(mask)
        texture_measurements = loadmat(os.path.join(path,'texturemeasurements.mat'), struct_as_record=True)
        texture_measurements = texture_measurements['m'][0,0]
        image_file = os.path.join(example_images_directory(), 
                                  'ExampleSBSImages', 'Channel1-01-A-01.tif')
        image = pil_to_array(PILImage.open(image_file))
        image = np.flipud(image[:,:,0])
        labels,count = scind.label(mask.astype(bool),np.ones((3,3),bool))
        centers = scind.center_of_mass(np.ones(labels.shape), labels, 
                                       np.arange(count)+1)
        centers = np.array(centers)
        X = 1 # the index of the X coordinate
        Y = 0 # the index of the Y coordinate
        order_python = np.lexsort((centers[:,X],centers[:,Y]))
        workspace, module = self.make_workspace(image, labels)
        module.scale_groups[0].scale.value = 3
        module.run(workspace)
        m = workspace.measurements
        tm_center_x = texture_measurements['Location_Center_X'][0,0][:,0]
        tm_center_y = texture_measurements['Location_Center_Y'][0,0][:,0]
        order_matlab = np.lexsort((tm_center_x,tm_center_y))
            
        for measurement in M.F_HARALICK:
            mname = '%s_%s_%s_%d'%(M.TEXTURE, measurement, INPUT_IMAGE_NAME, 3)
            pytm = m.get_current_measurement(INPUT_OBJECTS_NAME, mname)
            tm = texture_measurements[mname][0,0][:,]
            for i in range(count):
                self.assertAlmostEqual(tm[order_matlab[i]],
                                       pytm[order_python[i]],7,
                                       "Measurement = %s, Loc=(%.2f,%.2f), Matlab=%f, Python=%f"%
                                       (mname, tm_center_x[order_matlab[i]],
                                        tm_center_y[order_matlab[i]],
                                        tm[order_matlab[i]],
                                        pytm[order_python[i]]))
        image_measurements =\
        (('Texture_AngularSecondMoment_Cytoplasm_3', 0.5412),
         ('Texture_Contrast_Cytoplasm_3',0.1505),
         ('Texture_Correlation_Cytoplasm_3', 0.7740),
         ('Texture_Variance_Cytoplasm_3', 0.3330),
         ('Texture_InverseDifferenceMoment_Cytoplasm_3',0.9321),
         ('Texture_SumAverage_Cytoplasm_3',2.5684),
         ('Texture_SumVariance_Cytoplasm_3',1.1814),
         ('Texture_SumEntropy_Cytoplasm_3',0.9540),
         ('Texture_Entropy_Cytoplasm_3',1.0677),
         ('Texture_DifferenceVariance_Cytoplasm_3',0.1314),
         ('Texture_DifferenceEntropy_Cytoplasm_3',0.4147),
         ('Texture_InfoMeas1_Cytoplasm_3',-0.4236),
         ('Texture_InfoMeas2_Cytoplasm_3',0.6609))
        for feature_name, value in image_measurements:
            pytm = m.get_current_image_measurement(feature_name)
            self.assertAlmostEqual(pytm, value,3,
                                   "%s failed. Python=%f, Matlab=%f" %
                                   (feature_name, pytm, value))
            
    def test_03_01_gabor_null(self):
        '''Test for no score on a uniform image'''
        image = np.ones((10,10))*.5
        labels = np.ones((10,10),int)
        workspace, module = self.make_workspace(image, labels)
        module.scale_groups[0].scale.value = 2
        module.run(workspace)
        mname = '%s_%s_%s_%d'%(M.TEXTURE, M.F_GABOR, INPUT_IMAGE_NAME, 2)
        m = workspace.measurements.get_current_measurement(INPUT_OBJECTS_NAME, 
                                                           mname)
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0], 0)
    
    def test_03_02_gabor_horizontal(self):
        '''Compare the Gabor score on the horizontal with the one on the diagonal'''
        i,j = np.mgrid[0:10,0:10]
        labels = np.ones((10,10),int)
        himage = np.cos(np.pi*i)*.5 + .5
        dimage = np.cos(np.pi*(i-j)/np.sqrt(2)) * .5 + .5
        def run_me(image, angles):
            workspace, module = self.make_workspace(image, labels)
            module.scale_groups[0].scale.value = 2
            module.gabor_angles.value = angles
            module.run(workspace)
            mname = '%s_%s_%s_%d'%(M.TEXTURE, M.F_GABOR, INPUT_IMAGE_NAME, 2)
            m = workspace.measurements.get_current_measurement(INPUT_OBJECTS_NAME, 
                                                               mname)
            self.assertEqual(len(m), 1)
            return m[0]
        himage_2, himage_4, dimage_2, dimage_4 = [run_me(image, angles)
                                                  for image, angles in
                                                  ((himage, 2),(himage,4),
                                                   (dimage, 2),(dimage,4))]
        self.assertAlmostEqual(himage_2, himage_4)
        self.assertAlmostEqual(dimage_2, 0)
        self.assertNotAlmostEqual(dimage_4,0)
        
