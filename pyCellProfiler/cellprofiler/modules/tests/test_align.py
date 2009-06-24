'''test_align.py - test the Align module

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

import cellprofiler.modules.align as A

class TestAlign(unittest.TestCase):
    def make_workspace(self, images, cropping_masks):
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = A.Align()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        for index, pixels, cropping_mask in zip(range(len(images)),images,cropping_masks):
            if cropping_mask is None:
                image = cpi.Image(pixels)
            else:
                image = cpi.Image(pixels, crop_mask = cropping_mask)
            input_name = 'Channel%d'%index
            output_name = 'Aligned%d'%index
            image_set.add(input_name, image)
            if index == 0:
                module.first_input_image.value = input_name
                module.first_output_image.value = output_name
            elif index == 1:
                module.second_input_image.value = input_name
                module.second_output_image.value = output_name
            else:
                module.add_image()
                ai = module.additional_images[-1]
                ai.input_image_name.value = input_name
                ai.output_image_name.value = output_name
        return workspace,module
        
    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUggH0l6leQpGJgqGxlamplbGJgpGBgaWCiQDBkZPX34GBoalTA'
                'wMFXPeRtzNum0gMTV495LYF7fieic25rPtjry2cXZsWUqz2DYdnaUGN45ei'
                'Shk3ZT6J/hS7BvLbxbV02LfGqpZhiVxW2+KNd+hzBRo5/q8xsp91ddiBgaL'
                'Sq4Der9vp25lFjvqkaQbWmnjmNgRa23QwDdP5e3+yL0tdm3hLLFOQgsZdLo'
                'z7MvTvzy6Fv3vS2eUoKPyiyXWm4265mg/YZ+fzjTjj82xYhe1H+xMN5V21T'
                'qaWxy44vuBc/+dXWe8nTXfM66+X60tb3KSfe7VxyHW7+u5wv5+lbnPF/yXf'
                '3JV47OHp/mLLXh92dsuZ58rXvTg5skPbxi+i/Z83h/6YJfC7P3Sz8ttmJxL'
                'pu/RmjPv4MsEgUeRknOmHcxMCLj0/PgkP7n9yjfKJqo+OWaj8DD4Z5pHjY1'
                'd+YJeoT8zmepVHvYH7Ttuw3x+q7RM/4PAn1v0P374YWsaKzUrsOFUv/vWdS'
                'GPX3hWyKl/Vk25271u28ctpTH/d+/b2LQcqK7CpZZdaNv/iGcXljsm8K18r'
                '7zt0c13l8r2+oJU+b5evct/b+bOeVa/vSvO9vnoXg9Nlz+2rjV/g6hjUtK+'
                '6yyadlf3lBvVyH7z1WiffD4wPvKa1uu1GuvfuEfuW1QcJnh/ZVM+f0rts41'
                'vD99O4OI89XC1VdzU+tCfs75tfjEn5sj37Py/to8DV/1dd/9uxcTbdr17nl'
                'Ve2vfGy2YRh1zh+jWL1N//2yLVfvLf1InXVfxjbWdf2jj3/ts/ec+37/2s/'
                'ftzwJ+9779u0pTlT92be2v62+8LZs0zXvujZlnjm99psf/tg3ZUvOE69fnJ'
                '3uKieZxluZsqItfU3u++f7+ljtXufZxp25suM5eWnlqrwyw/NeuvnW1+k97'
                'uevy37dpzts9Kz9a05WVsygfGw5b61/b135b/33z+Y73j1tLPSb71gbdu/Z'
                'c/7tH7BAAjAIGJ')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        #
        # The pipeline has four modules and Align is the third. It has
        # the following settings:
        # input image name = Template
        # output image name = AlignedTemplate
        # second input name = Image
        # second output name = AlignedImage
        # use normalized correlation
        #
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, A.Align))
        self.assertEqual(module.first_input_image.value, "Template")
        self.assertEqual(module.first_output_image.value, "AlignedTemplate")
        self.assertEqual(module.second_input_image.value, "Image")
        self.assertEqual(module.second_output_image.value, "AlignedImage")
        self.assertEqual(module.alignment_method.value, A.M_CROSS_CORRELATION)
        self.assertTrue(module.wants_cropping.value)
    
    def test_01_02_load_v1(self):
        '''Load a version 1 pipeline'''
        data = ('eJztWl9P2lAUvyAuui2b7mV7vCZ7kE0IFJYoWRQmS8YmjgjRGeO2K1zgJre'
                '9pLQqW0z2uI+xj7KPs8d9hN2LrS3XaoGKxtmSpj2n53fPOb+ec/uHlgu1jc'
                'Ib+CqZguVCLdEkFMMKRUaT6WoOasYSXNcxMnADMi0Hd/j2valBJQvT2Vw6l'
                'eM7Siq1AsZbIqXyI75pPwfgHt/O8DVqHZq25IhrFXIVGwbRWt1pEAPPLP1v'
                'vm4jnaADircRNXHXcWHrS1qT1Xqds0Nl1jAp3kSq25gvm6Z6gPXux6YNtA5'
                'XyDGmVfINSynYZlv4kHQJ0yy8Nb6sPfPLDMmv4OHXvMNDROJB8LLg0gv7d8'
                'Cxj3nwNu+yn7NkojXIIWmYiEKiotZZFGK8ZZ/xZqTxhLzeRpqGaVrg8z74O'
                'Qkv1ho+NhJvj1HdgCoy6u1x4yhQ0tJwox9HygcfGcBHQCZ4/sow+d+X8EIu'
                'MqgxA5pdHOw8WPkrw+QfHcBHwSYLnH9mmHp8KuGFXMRNZFIDlkQxwiLRcd1'
                'gei8oDxmZh3sS3l5s/Cxw+M/7+J2V/Aq5SlRCkU57rnH8zsPUwDhTYJd3Y8'
                'DzkL1uvxbfQ/mV6+4oPRzfXn1TxR2k8+uSTfhVzF9+9ftEwgu5bBpiLhWXF'
                '51PYHyud+IZdR5KD4m7qH/9cLEBXIzjNBykT374+Psg8SXkz4trldfiBgOv'
                'Jl/GvwhpB1O6xY5W9wqJyn7c1qwzaqra6l4qsbL/Pb2knJwaVwlH9pVxT55'
                'Hib/tE/+yFL+QRQy7GOlWYNmTeEKoykwz2pZOsXRF1HM0QeIcua+UYLwEvQ'
                '5OKk+5XxQP3FX6C/vljvRL5sb6JdB9yrj3uWG/hP0SqF+yN9Yv5+4zJ9kvW'
                'Q9c2C9hv3x6PNp7mkk9h3o9F/Vf6rR0ZnYm79/rfY7jHxKtgTth/mH+VzF/'
                'XgfOL9+Q7/8r7hB3N3F5ENZriLs9uDwI6zXE3R5cHoT1GuLGx/1x4eTna/l'
                '9lLD/Ci6vtxdgsN6EXMeUdnQmvnfRk2r/o4xukjLUOP0qIrnBd0uuDySEn4'
                'qPHyj5gRf5QeJP02T/r1OZp1mPcd35Rvlv4eHl/Mq8Onz/XRvHXzRy3t8DH'
                '1zMYkjgfoLRzufiJfZ2buPa/wMTtu04')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        #
        # There are two modules and Align is the second. It has the
        # following settings:
        #
        # First image:   input = Channel1, output = Aligned1
        # Second image:  input = Channel2, output = Aligned2
        # Third image:   input = Channel3, output = Aligned3, similarly aligned
        # Fourth image:  input = Channel4, output = Aligned4, separately
        # Mutual information
        # Don't crop
        self.assertEqual(len(pipeline.modules()),2)
        module = pipeline.modules()[1]
        self.assertEqual(len(module.additional_images),2)
        for i, input, output in ((1, module.first_input_image, module.first_output_image),
                                 (2, module.second_input_image, module.second_output_image),
                                 (3, module.additional_images[0].input_image_name, module.additional_images[0].output_image_name),
                                 (4, module.additional_images[1].input_image_name, module.additional_images[1].output_image_name)):
            
            self.assertEqual(input.value, 'Channel%d'%i)
            self.assertEqual(output.value, 'Aligned%d'%i)
        self.assertEqual(module.additional_images[0].align_choice.value, A.A_SIMILARLY)
        self.assertEqual(module.additional_images[1].align_choice.value, A.A_SEPARATELY)
        self.assertEqual(module.alignment_method.value, A.M_MUTUAL_INFORMATION)
        self.assertFalse(module.wants_cropping.value)
        
    def test_02_01_cross(self):
        '''Align two images using cross correlation'''
        np.random.seed(0)
        for offset in ((3,5),(-3,5),(3,-5),(-3,-5)):
            image1 = np.random.uniform(size=(50,50))
            i,j = np.mgrid[0:50,0:50]
            image2 = image1[(i+50-offset[0])%50,(j+50-offset[1])%50]
            mask = (((i+50+offset[0])%50 == i+offset[0]) &
                    (((j+50+offset[1])%50 == j+offset[1])))
            workspace, module = self.make_workspace((image1, image2),
                                                    (None,None))
            module.alignment_method.value = A.M_CROSS_CORRELATION
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned1')
            self.assertTrue(np.all(image1[mask] == output.pixel_data[mask]))
            self.assertTrue(np.all(mask == output.mask))
            m = workspace.measurements
            self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned1'),offset[1])
            self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned1'),offset[0])

    def test_02_02_mutual(self):
        '''Align two images using mutual information'''
        np.random.seed(0)
        i,j = np.mgrid[0:50,0:50]
        for offset in ((3,5),(-3,5),(3,-5),(-3,-5)):
            #
            # Do something to give the image some information over
            # the distance, 5,5
            #
            image1 = np.random.uniform(size=(5,5))[i/10,j/10]
            image1 = scipy.ndimage.gaussian_filter(image1,5)
            image2 = 1-image1[(i+50-offset[0])%50,(j+50-offset[1])%50]
            mask = (((i+50+offset[0])%50 == i+offset[0]) &
                    (((j+50+offset[1])%50 == j+offset[1])))
            workspace, module = self.make_workspace((image1, image2),
                                                    (None,None))
            module.alignment_method.value = A.M_MUTUAL_INFORMATION
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned1')
            self.assertTrue(np.all(np.abs(image1[mask] + output.pixel_data[mask] - 1) <= np.finfo(float).eps))
            self.assertTrue(np.all(mask == output.mask))
            m = workspace.measurements
            self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned1'),offset[1])
            self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned1'),offset[0])

    def test_03_01_align_similarly(self):
        '''Align a third image similarly to the other two'''
        np.random.seed(0)
        i,j = np.mgrid[0:50,0:50]
        for offset in ((3,5),(-3,5),(3,-5),(-3,-5)):
            image1 = np.random.uniform(size=(50,50))
            mask = (((i+50+offset[0])%50 == i+offset[0]) &
                    (((j+50+offset[1])%50 == j+offset[1])))
            image2 = image1[(i+50-offset[0])%50,(j+50-offset[1])%50]
            image3 = (i * 100 + j).astype(float)/10000
            expected = image3+float(offset[0]*100+offset[1])/10000
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            module.alignment_method.value = A.M_CROSS_CORRELATION
            module.additional_images[0].align_choice.value = A.A_SIMILARLY
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned2')
            self.assertTrue(np.all(np.abs(output.pixel_data[mask] - expected[mask]<=np.finfo(float).eps)))
            m = workspace.measurements
            self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned2'),offset[1])
            self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned2'),offset[0])

    def test_03_02_align_separately(self):
        '''Align a third image to the first image'''
        np.random.seed(0)
        i,j = np.mgrid[0:50,0:50]
        for offset in ((3,5),(-3,5),(3,-5),(-3,-5)):
            image1 = np.random.uniform(size=(50,50))
            mask = (((i+50+offset[0])%50 == i+offset[0]) &
                    (((j+50+offset[1])%50 == j+offset[1])))
            image2 = image1[(i+50-offset[0]-5)%50,(j+50-offset[1]-5)%50]
            image3 = image1[(i+50-offset[0])%50,(j+50-offset[1])%50]
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            module.alignment_method.value = A.M_CROSS_CORRELATION
            module.additional_images[0].align_choice.value = A.A_SEPARATELY
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned2')
            self.assertTrue(np.all(image1[mask] == output.pixel_data[mask]))
            m = workspace.measurements
            self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned1'),offset[1]+5)
            self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned1'),offset[0]+5)
            self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned2'),offset[1])
            self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned2'),offset[0])
    
    def test_04_01_crop(self):
        '''Align with cropping'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(50,50))
        image2 = image1[5:-6,7:-8]
        #
        # set up the cropping so that cropped image1 is offset by 1,-1
        #
        mask = np.zeros((50,50),bool)
        mask[4:-7,8:-7] = True
        workspace, module = self.make_workspace((image1, image2),
                                                (None, mask))
        module.alignment_method.value = A.M_CROSS_CORRELATION
        module.wants_cropping.value = True
        module.run(workspace)
        output1 = workspace.image_set.get_image('Aligned0')
        output2 = workspace.image_set.get_image('Aligned1')
        self.assertTrue(np.all(output1.pixel_data[output2.mask] ==
                               output2.pixel_data[output2.mask]))
        m = workspace.measurements
        self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned0_vs_Aligned1'),1)
        self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned0_vs_Aligned1'),-1)

        