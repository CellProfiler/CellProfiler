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
import PIL.Image
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
    
    def test_01_02_load_v1(self):
        data = ('eJztWu9v2kYYPgiJmlWquk7TJvXLfWy6YBlGpiaaUljoNrZAUMNaVV'
                'XbXcwRbjrfIfucwqpK/bP2cX/S/oT5wA72xcSOCYRUtmSZ9/U99zz3'
                '+r0fnN2sdQ5rP8EdTYfNWqfYIxTDNkWixy1zDzKxDQ8sjATuQs72YM'
                'fB8DeHQX0Xlkp7FX2vUoZlXd8F6Y5co3nPveiPAdhwr3fcM+/dWvfs'
                'XOCU9jEWgrBTex0UwLee/1/3fIEsgk4ofoGog+0phe9vsB7vjAbnt5'
                'q861DcQmawsHu0HPMEW/ZRzwd6t9tkiOkx+RsrTfCLPcdnxCaceXiv'
                'ftV7zsuFwivjMPxqGoecEoc193wY8Mvyv4Jp+UJE3L4MlL/v2YR1yR'
                'npOohCYqLTcxXj5xBT31qovjVQb9US4XIhXA6UPb5qDO6+ol+eHTwU'
                'xWdDZAhoImH0ZT1PYurZUOqRdssxKCbJ2q3qLyWMl4r7PiGuEMIVwA'
                '/bFX0evrj4rCvxkXbj8PCPZkK9al68crNqkfmUD+HyoMWT6ZyFq8bg'
                'NpX4SPtgJPiAItv0/En64zdKPdKu4x5yqIAN2RlhnVjYENwazRX3Zf'
                'cHDSSL412FV9pHwnbgL5SfIHoex0X1q7R5o8a5pOmpdFaWrFPfLqXS'
                'uQOSjRt3QPh5SvugjxjDtJxmnNM1vaTiNhScf/i4Te86z3hTjcFF9f'
                '8GE5jZRIxm8F+nbjVO9Vq7cRt1tzjDafK/pK+mzln9ZtV0Jpmfr6Jz'
                'jnGhlCRv7yl4aU/mpyNHUMLkonUe/XH8Sef7Zcc97Tp71XSuSh5/iu'
                'H7HYTzQNpvHz1t/yj/ION97butd9J6iSl9zt/vv64V22+2fM8Bp47J'
                '9l/rxd03H0rb5Y+TwsfERY6dW4njdWFej1h/XKXd/Ri+J0q7pS21v8'
                'LI8hpU+bhVlK4mZ6Lv+cqer45GU8+i1oVp8/E68zjJOm0VdC6yfy8y'
                'ntrOzcSzGqMzKl877zk03PnB9nZUbkJ33P/PqP2gl5ic9uX23pncyG'
                'IGDtS3anH/QtEv7Z+5hU8t7rDu/PyfHlxt/22Z7Rxv1smGDubnX2Z+'
                '8ZO/sCHGwiFhXTy4Bh0Z7vPKq89Vd9Q8EegPN6Y7w2W468zzqPc00/'
                '45GfZvU3szXIZbZVzcOuwBCPdHafPJztmFhVg2LmS4DHc5rgouz/NV'
                'XZ9muAyX4W4fbpib4tR9KHU/V5b/M8ATNT49BuHxSdoGpnRgcfm9n6'
                'WZ44/SbI1y1J18FaYduj8bgQ/EJM8ghqeq8FRn8ZAuZoL0RgPLZXME'
                'N5EghtbwvG3XW/O9Sd5XlBXe8ixeEyPbsbDAQ+FetObE7EzM6Oe2Gc'
                'EXjH/etb5+uHnp8wYg/Jynz/+/p2n48mu5C+9h78bgCgFNfjv/AVfL'
                's0eXlPfbuKzy/wP0LZ97')
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
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.image_groups[0].image_name.value, "DNA")
        self.assertEqual(module.image_groups[1].image_name.value, "Cytoplasm")
        
        self.assertEqual(module.object_count.value, 1)
        self.assertEqual(module.object_groups[0].object_name.value, "Nuclei")
        
        self.assertEqual(module.scale_count.value, 3)
        for scale, expected in zip([x.scale.value 
                                    for x in module.scale_groups],[3,4,5]):
            self.assertEqual(scale, expected)
    
    def test_02_01_compare_to_matlab(self):
        path = os.path.split(__file__)[0]
        mask_file = os.path.join(path, 'Channel2-01-A-01Mask.png')
        mask = pil_to_array(PIL.Image.open(mask_file))
        mask = np.flipud(mask)
        texture_measurements = loadmat(os.path.join(path,'texturemeasurements.mat'), struct_as_record=True)
        texture_measurements = texture_measurements['m'][0,0]
        image_file = os.path.join(example_images_directory(), 
                                  'ExampleSBSImages', 'Channel1-01-A-01.tif')
        image = pil_to_array(PIL.Image.open(image_file))
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
