'''test_conservememory - Test the ConserveMemory module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import base64
from matplotlib.image import pil_to_array
import numpy as np
import os
import PIL.Image as PILImage
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

import cellprofiler.modules.conservememory as S

class TestConserveMemory(unittest.TestCase):
    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline with a SpeedUpCellProfiler module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggH0sGpBQrGBgqGplbGBlbGhgpGBgaWCiQDBkZPX34GBoaXjAwM'
                'FXPehlv7HzYQ2LeMV4snsLGVcUmPzlNFMbcjzouEpnQxCS3OvJNwsnOxzapp'
                'rVPjj5zxP9ymUuqUKRdyYEK3zmu977XTrXfbhTM3XNM6IDR/do3GB82sFdeX'
                'KH6x+1abNHseF+8Njrf5NXcUpy9g73ovwLuBfaPLtz7bFHNh9brSkz0Nj/Q2'
                'paTr3T1QdWBduWlPw8PXcpLJt9T9JSYJ7Yx9LPiRY7NzzYn5244VzpzftbrT'
                'ZtZlqytBpQKlkZPfHPrPsexfr2Rptue/tQV3Hmy8bye9uHF99Al5G655e3nm'
                'RTUtZ/tt+Vh/+SPuv08Eb5ot+52r2b/mx1cWf9l3lW9Z3hkwX29JqVkYv/OI'
                'zbPoFMEfa/6EB1UsnG90N22fXEXFrPqjwR9DbOIru+Z5s/836Xf9OCf9w5pl'
                'FS1+Oc09HYIKq4S5ZxZe/3m2NEjIhqvk5v13k+eyPp9/1Hg1/yy9WRXMLot/'
                't6idTf2dGPen/HWg2b693UI/lln9l1vsWaZ7vGbKR9vN7yzlZ98tiXzw4o96'
                'w9xNr22fav6Lrfy56s+qP/+K60Tvn/9x78Hvzwa1ctN7qw9OuDhRqmbe70NV'
                'Xe4vatmTH1+zPDjlW4yZishuw/l3xKcZfQr4sXnZ1NfnL/4/vehjyFaLdQfn'
                'X1/73LexujHp9Rbb4/aTNv17P8tDYD8AIz4bHQ==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, S.SpeedUpCellProfiler))
        self.assertEqual(module.how_to_remove, S.C_REMOVE)
        self.assertEqual(len(module.image_names), 1)
        self.assertEqual(module.image_names[0].image_name, "OrigBlue")
        
    def test_01_02_load_v1(self):
        data = ('eJztWdtO20AQXYeAuEgVfWrVp30kLYlMKBJECJKSVo1K0oikIIRou8QbspLt'
                'tXyB0Aqpj/2cfk4/o5/QXbCxvTVxLlzkyo4sM7Nz5syMZzfLpl5p71bewLWC'
                'DOuVdr5LVAybKrK71NRKULeX4Y6JkY0VSPUSPGDPFjbgqgxX1kqrayV5HRZl'
                'eQOMd0m1+hP2mH0BwAx/sjvjDk27shS4udzCtk30U2saZMFzV/+b3fvIJOhE'
                'xftIdbDlU3j6mt6l7QvjZqhOFUfFDaQFjdnVcLQTbFofux7QHW6SPlZb5BsW'
                'UvDM9vAZsQjVXbzrX9Te8FJb4G316Pk7k4Uj+Of1+bXg10cS6sPr9Syg5/bv'
                'gW+fjajn04D9oisTXSFnRHGQComGTm+i4/7WY/zNCP64bGKNnuFrfDkGvyjg'
                '+d3GfTv/to86NtSQ3elxP3KMn6mQnylQbVTAMDgphJPA6pB5Twtxc7nSYe0J'
                'hst7XsBzuUqhTm3oWNivfzD+GcGPd3l+5kA0Loo/E+LPgAadjC+u74J9uujK'
                'VdxFjmrDGm86WCUm7tjUvLizvB8aN0xfHrLZlaT3+ph8/ztu3HXsLvnE9W/l'
                'nvPLhviyrF90PAnfjxi+DyC87nD589J2c5NvdPBW4VXuC5cOsKru0fOto0q+'
                'eZzzNDtUdTR960jObxx/X1kuXl4btwhDXilzkXmPEn8vJv51IX4u8xgOMTLd'
                'wF5f5vJcVae63XN1RVdXRRe+5rH7fRTcpN+/DzVfivfMl86XZM6X/vxo+/ak'
                'zMsyGFz/qH3t1T8VpyZ1jOTzp7gUl+KShysHcMOeQ/jrBiS6go0k5Zviko0r'
                'g7RfU9z4uFnp9v2neG7D7b+Cwf32EoT7jcsdtuU3TMp/NzAL2tXhtlVQKVKu'
                'T5ELu+zPWuBAmfMYMTybAs/mbTyWgbHiGKGxFtd9MnaYrunqouo4F8EbrEeG'
                'fZ5NDa6/WHf/ffzZHocvK/3LtxCDy7oV5LifYLT3vTTA3sttEvtR85eY8Beb'
                '5Tby')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, S.SpeedUpCellProfiler))
        self.assertEqual(module.how_to_remove, S.C_REMOVE)
        self.assertEqual(len(module.image_names), 2)
        self.assertEqual(module.image_names[0].image_name, "DNA")
        self.assertEqual(module.image_names[1].image_name, "Actin")
        
    def test_02_01_erase_remove(self):
        module = S.SpeedUpCellProfiler()
        module.how_to_remove.value = S.C_REMOVE
        module.image_names[0].image_name.value = "Image1"
        module.module_num = 1
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("Image1", cpi.Image(np.zeros((10,10))))
        image_set.add("Image2", cpi.Image(np.zeros((10,10))))
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), cpmeas.Measurements(), 
                                  image_set_list)
        module.run(workspace)
        image = image_set.get_image("Image1")
        self.assertFalse(isinstance(image, cpi.Image))
        image = image_set.get_image("Image2")
        self.assertTrue(isinstance(image, cpi.Image))
        
    def test_02_02_erase_keep(self):
        module = S.SpeedUpCellProfiler()
        module.how_to_remove.value = S.C_KEEP
        module.image_names[0].image_name.value = "Image1"
        module.module_num = 1
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("Image1", cpi.Image(np.zeros((10,10))))
        image_set.add("Image2", cpi.Image(np.zeros((10,10))))
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        image = image_set.get_image("Image2")
        self.assertFalse(isinstance(image, cpi.Image))
        image = image_set.get_image("Image1")
        self.assertTrue(isinstance(image, cpi.Image))
