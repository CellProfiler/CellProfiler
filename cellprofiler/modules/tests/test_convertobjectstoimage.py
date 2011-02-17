'''test_converttoimage.py - test the ConvertToImage module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import base64
import numpy as np
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
import cellprofiler.modules.convertobjectstoimage as C

OBJECTS_NAME = "inputobjects"
IMAGE_NAME = "outputimage"
 
class TestConvertObjectsToImage(unittest.TestCase):
    def make_workspace(self):
        module = C.ConvertToImage()
        labels = np.reshape(np.arange(256),(16,16))
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
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        module.image_name.value = IMAGE_NAME
        module.object_name.value = OBJECTS_NAME
        return (workspace,module)
    
    def test_01_01_load_matlab(self):
        '''load a matlab pipeline with ConvertToImage in it'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUnArylTwKs1TMDRSMDS2Mja3MjBUMDIwsFQgGTAwevryMzAwrG'
                'BiYKiY8zbcMf+WgUiZgsI1v45A31uNzn3WJbI5l7c81rLK9ZAO0V5lkikuz'
                'PP1yQ2NQoE307/umL/p9b+jSR5Nj5kMPa+8M/Ce9nXO933f9z6fLr6cj2FH'
                'LlMA77UoA+8EW62FOR7BJ8I7in2Wx7HOeBC547/2jOsXjha4XDBVsfvkYLP'
                'or9zcDQ+LxOx/HTpm51j74sSpgy9+n+PYeS9BSLKtX/8j00TtP8KNTzqu7F'
                'tk+c1g8cLajvXvjtVJhv+51vzzd8ZkdueTzpYz1C/tu1DPb/br2bQ9pip/3'
                'SzaMw7G10Q4zI7me/Kt90Dm8gdThC1SCy7ax/+sn54UHvonPPznVs5DDqf3'
                'LbvBvbOaWdP1Sajgv8ufytfckD/+zZbdL7CGL/Lz9LR0tTc+H7598Y1Wf9d'
                '1KPpErNsM7qLneyer3st3XJveWe70IOT5HDnNmtY7sUVmyheKD9XK/l/K4s'
                'f55eYtvU3+jwuXfBD4xLpZ4/G094q1crs/hZv0m8zJ/Wr/YF5+rUoOi2dP8'
                'f2fhkIWbC9sBPi/zrm1fC5Lv/PjirWZy/epv91c9lzxVfLt6uuLZQL5Py89'
                'oh5lv/SBbeCHlp6XEuc/3mg/cmjRyyqhuIt/Nn1r17E5V3uuid9gm/GHgNf'
                'PdNztvn080WRe+vDSZkVHIb97ddOv2vXuf1e94LRydq3IWeez7LWc3Jbtey'
                'o73kg/+1e97g77p29bQy/P2HVm/svHOb/fbbzIn/KZN+DePrWHkY79zpuZH'
                '90pW3//yM42bSH33vLrv+7Hb/z376rp2nuzFWayKxrPF/a5OLdpemH2sXf7'
                'FnxbvubvA+Pmx/NKJl7ofL6ySfnJPYnAPvXT3uLWn8xD2sNjDwoXTb9oWfR'
                'zQuv+aVnz1y57sljE46yw8fW/PwPqpx4/Nvf/SRNr4ys/70lt+bflWl3+wc'
                'Tn/2r660zeef/XfzaV/REATdyFtA==')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module,C.ConvertToImage))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.image_name.value, "NucleiImage")
        self.assertEqual(module.image_mode.value, "Color")
        self.assertEqual(module.colormap.value, "flag")
            
    def test_01_02_load_v1(self):
        '''load a pipeline with a variable_revision_number=1 ConvertToImage'''
        data = ('eJztWltv2jAUNpeidpW6rnvYpL74cZvaKKFF2ngZFNaNqVy0ok57WxoM9WR'
                'slDgU9kv2U/a4n7SfsBhCSbzQcGm5SIlkwXH8ne+c4xM7OXI5X7/In8GMos'
                'Jyvn7cxATBGtF5k5ntLKT8CBZMpHPUgIxm4bmJ4WebQi0NtZOspmZPMzCtq'
                'u/AfFesVN5zfn69BiDl/G47Le7e2nLlmKcJ+RJxjmnL2gJJ8NLt/+O0K93E'
                '+jVBVzqxkTWmGPWXaJPV+527W2XWsAmq6G3vYOeq2O1rZFrV5gjo3q7hHiK'
                'X+CeSXBgN+4K62MKMunhXv9x7x8u4xCviAJ6P4xCT4pBw2qGnX4z/BMbjkw'
                'Fxe+YZv+/KmDZwFzdsnUDc1lt3Vgh9b0P0bUv6hFw1cevMCfk0+JSETw3ib'
                'RCEh/y5EPy+hBetjnr8+ENPNzhs69y4WYYdOxJeyAVESEkE1BNPNURPzKcn'
                'Bk6mnIctiV/Imnp0qi6ALzDCTDCd/08kvJCLDFLGoW2h6f1P+PQkwDcnG+e'
                'dv1tMOTKn44378HFQYQ+HS0m40TXC7YBxfMKe3xeSn0IuoqZuEw4HuQaL2E'
                'QGZ2Z/oXjPa/+s+a2A6fJrV/JbyFVu2fAjYdc6CeR/SLvnzQ/ZX21N7ZTzQ'
                'D3SHtXOpI8vCYr5WmkenKqo2iJ2Lrq/5ULwQftCyVmWqIV5f0K8lhHnTbO7'
                'wiiaJ/81dT3tlNeFTADfOtj5mPvDLLhciJ17wJ+vQh6+v1VtTjAVL7WrsHt'
                'T4hvZuVo7nX1sLZ+rZb53RXmwvnbK+6qSWc2850LsDMrX+i2DBtEty61srM'
                'LusO+6oLrMV4RbN6LM1hUFJWpMqiOsQ9yDvv/PmYlaJrNpY3H+7wez1cGW6'
                'eegaCYc7UyvJyhP2fUP58t9rGjZ+ebhh5g2UGeGuATV4cZxGapb9boW4SJc'
                'hItwm4zLeXDROhzhItxqcWHvWQfA/zwKmQ0rUv+9aG2S3xEuwq3Dfjft99i'
                'm+BvhIlyEWx2uFxvj5DqTXK8d1KU8PEHr0xvgX5+EbCBCOiYT5+pMpT04/G'
                'UphOmN4ekr5cL5W/IcxBI8nRCenMSTm8SDG4hy3Ox3TIfN5qytc2woJbe35'
                'vTmR72C9yaENy3xpifxGox2kck5GzipFIZinXkOSMnzthPA541/3JGeHibv'
                'nW8A/PM8nv+/7+fhiydiAz7vuYHdEFzSY9PIz99gtjx7dc/4kY/LGv8PmsR'
                'r2g==')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module,C.ConvertToImage))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.image_name.value, "CellImage")
        self.assertEqual(module.image_mode.value, "Color")
        self.assertEqual(module.colormap.value, "winter")
    
    def test_02_01_bw(self):
        '''Test conversion of labels to black and white'''
        workspace, module = self.make_workspace()
        module.image_mode.value = C.IM_BINARY
        module.run(workspace)
        pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        self.assertFalse(pixel_data[0,0])
        i,j = np.mgrid[0:16,0:16]
        self.assertTrue(np.all(pixel_data[i*j > 0]))
    
    def test_02_02_gray(self):
        '''Test conversion of labels to grayscale'''
        workspace, module = self.make_workspace()
        module.image_mode.value = C.IM_GRAYSCALE
        module.run(workspace)
        pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        expected = np.reshape(np.arange(256).astype(np.float32)/255,(16,16))
        self.assertTrue(np.all(pixel_data == expected))
    
    def test_02_03_color(self):
        '''Test conversion of labels to color'''
        for color in C.COLORMAPS:
            workspace, module = self.make_workspace()
            module.image_mode.value = C.IM_COLOR
            module.colormap.value = color
            module.run(workspace)
            pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
            self.assertEqual(np.product(pixel_data.shape),256*3)
    
    def test_02_04_uint16(self):
        workspace, module = self.make_workspace()
        module.image_mode.value = C.IM_UINT16
        module.run(workspace)
        pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        expected = np.reshape(np.arange(256),(16,16))
        self.assertTrue(np.all(pixel_data== expected))
        