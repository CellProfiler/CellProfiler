'''test_resize.py - test the Resize module
'''

import base64
import os
import unittest
import zlib
from StringIO import StringIO

import PIL.Image as PILImage
import numpy as np
import scipy.ndimage
from matplotlib.image import pil_to_array

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.resize as R

INPUT_IMAGE_NAME = 'input'
OUTPUT_IMAGE_NAME = 'output'


class TestResize(unittest.TestCase):
    def test_01_02_load_v1(self):
        data = ('eJztWF9v0zAQd7vuH0ioiAd4tPayFdYo7TZpq9C2siJRWEu1VWPTNMBdndZS'
                'EleJM1bQJB75WHykfQTsLmkSE5a03RBITRWld7nf/e7OzsVOrdzcL7+CG4oK'
                'a+VmXiM6hg0dMY1aRgmabBXuWRgx3IbULMFm14FvHR0W12Bho1TcLBXWYVFV'
                't8B4R6pae8Qv8AkAc/y6wM+0e2vWlVOBU8iHmDFiduxZkAHPXP1Pfh4hi6CW'
                'jo+Q7mDbp/D0VVOjzX5veKtG246O68gIGvOj7hgtbNnvNQ/o3m6QS6wfkq9Y'
                'SsEzO8AXxCbUdPGuf1k75KVM4hV1aCz4dUhJdRB1yQb0wv4N8O0zEXV7HLDP'
                'ujIx2+SCtB2kQ2KgzjAK4U+N8TcT8jcDKvXyALcbg8uCcBzibOJLln99ic4Z'
                'NBA77ybx80DyI+QDbPNBaYtQEueRCvlJgTUXdxyDW5b4l4f8sNWHCGo8G2pB'
                'qkHWxZBapENMXmdhkKhOUflVKDQpg46Nk+eXCfnJAFUpbiTBpUO4NKjT8eZF'
                'QVUTzc+nUr5CrmANOTqDVTE5YYVYWNS0P1EcQdychPMOD7cI/DrvxvBFzes6'
                'Rha2Gaxj0um2qDVO3Cf8qZwk7n+NL8m8GoVvM4ZvDoTHRch7OjVFs/ubdfX6'
                '433xyX2sEIG7Sz65r9R5SSfh+x7D9w6Ex1HIH1d2Gi/FAgVvKy9yn4T0Aev6'
                'Af2yfVrON85ynmaP6o5hbp+q+a2zb4XV4tWN8SHhyIEyF5n3KPF3Y+LflOIX'
                'sojhhPcIN7D1q1xeqGrUZF1XV3R1FdT3NZPEeTw/2rrivvpl1PttsAjpWNTp'
                '3T9/VL/2+SFfGuHeXT0/U9wUN8X9H7jdAG7aN6a4UXHXAZz8fpXXv8L+M7h9'
                'vj0H4fkm5HO+pOlZVHyfsRRj8BHBVnSK2je7eGWf/60GNvRJ9rNLEs/Sn3is'
                'wSZXudnrRtdrMcJ/MO80/2Xnb6+zXF+/7tc74/DNpH7nexiDy7iVErgfYLRx'
                'XbnF3sttXPtf+3UFIg==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertEqual(module.image_name, 'DNA')
        self.assertEqual(module.resized_image_name, 'ResizedDNA')
        self.assertEqual(module.size_method, R.R_BY_FACTOR)
        self.assertAlmostEqual(module.resizing_factor.value, .25)
        self.assertEqual(module.interpolation, R.I_NEAREST_NEIGHBOR)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10104

Resize:[module_num:1|svn_version:\'10104\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the output image:ResizedDNA
    Select resizing method:Resize by specifying desired final dimensions
    Resizing factor:0.25
    Width of the final image, in pixels:141
    Height of the final image, in pixels:169
    Interpolation method:Bilinear
    Additional image count:1
    Select the additional image?:Actin
    Name the output image:ResizedActin

Resize:[module_num:2|svn_version:\'10104\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the output image:ResizedDNA
    Select resizing method:Resize by specifying desired final dimensions
    Resizing factor:0.25
    Width of the final image, in pixels:100
    Height of the final image, in pixels:100
    Interpolation method:Bicubic
    Additional image count:0
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.Resize))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.resized_image_name, "ResizedDNA")
        self.assertEqual(module.size_method, R.R_TO_SIZE)
        self.assertAlmostEqual(module.resizing_factor.value, 0.25)
        self.assertEqual(module.specific_width, 141)
        self.assertEqual(module.specific_height, 169)
        self.assertEqual(module.interpolation, R.I_BILINEAR)
        self.assertEqual(module.additional_image_count.value, 1)
        additional_image = module.additional_images[0]
        self.assertEqual(additional_image.input_image_name, "Actin")
        self.assertEqual(additional_image.output_image_name, "ResizedActin")

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.Resize))
        self.assertEqual(module.interpolation, R.I_BICUBIC)

    def make_workspace(self, image, size_method, interpolation,
                       mask=None, cropping=None):
        module = R.Resize()
        module.image_name.value = INPUT_IMAGE_NAME
        module.resized_image_name.value = OUTPUT_IMAGE_NAME
        module.size_method.value = size_method
        module.interpolation.value = interpolation
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image = cpi.Image(image, mask, cropping)
        image_set.add(INPUT_IMAGE_NAME, image)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(),
                                  cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_02_01_rescale_triple_color(self):
        i, j = np.mgrid[0:10, 0:10]
        image = np.zeros((10, 10, 3))
        image[:, :, 0] = i
        image[:, :, 1] = j
        i, j = (np.mgrid[0:30, 0:30].astype(float) * 9.0 / 29.0 + .5).astype(int)
        expected = np.zeros((30, 30, 3))
        expected[:, :, 0] = i
        expected[:, :, 1] = j
        workspace, module = self.make_workspace(image, R.R_BY_FACTOR,
                                                R.I_NEAREST_NEIGHBOR)
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        result = result_image.pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.finfo(float).eps))
        self.assertTrue(result_image.parent_image is
                        workspace.image_set.get_image(INPUT_IMAGE_NAME))

    def test_02_02_rescale_triple_bw(self):
        i, j = np.mgrid[0:10, 0:10]
        image = i.astype(float)
        i, j = (np.mgrid[0:30, 0:30].astype(float) * 9.0 / 29.0 + .5).astype(int)
        expected = i
        workspace, module = self.make_workspace(image, R.R_BY_FACTOR,
                                                R.I_NEAREST_NEIGHBOR)
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.finfo(float).eps))

    def test_02_03_third(self):
        i, j = np.mgrid[0:30, 0:30]
        image = i.astype(float)
        expected = (np.mgrid[0:10, 0:10][0].astype(float) * 29.0 / 9.0 + .5).astype(int)
        workspace, module = self.make_workspace(image, R.R_BY_FACTOR,
                                                R.I_NEAREST_NEIGHBOR)
        module.resizing_factor.value = 1.0 / 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.finfo(float).eps))

    def test_03_01_bilinear(self):
        i, j = np.mgrid[0:10, 0:10]
        image = i.astype(np.float32)
        expected = np.mgrid[0:30, 0:30][0].astype(np.float32) * 9.0 / 29.0
        workspace, module = self.make_workspace(image, R.R_BY_FACTOR,
                                                R.I_BILINEAR)
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.sqrt(np.finfo(float).eps)))

    def test_03_02_bicubic(self):
        '''Test bicubic interpolation'''
        i, j = np.mgrid[0:10, 0:10]
        image = i.astype(float) ** 2
        # Bicubic here should be betweeen nearest neighbor and bilinear
        i, j = np.mgrid[0:19, 0:19]
        low_bound = (i / 2) ** 2
        upper_bound = (i.astype(float) / 2) ** 2
        odd_mask = i % 2 == 1
        upper_bound[odd_mask] = (upper_bound[np.maximum(i - 1, 0), j] +
                                 upper_bound[np.minimum(i + 1, 18), j])[odd_mask] / 2
        workspace, module = self.make_workspace(image, R.R_BY_FACTOR,
                                                R.I_BICUBIC)
        module.resizing_factor.value = 19.0 / 10.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        #
        # Need to account for edge effects at the top - a row of zeros
        # is used there
        #
        self.assertTrue(np.all(result[i < 17] >= low_bound[i < 17] -
                               np.sqrt(np.finfo(float).eps)))
        self.assertTrue(np.all(result[i < 17] <= upper_bound[i < 17] +
                               np.sqrt(np.finfo(float).eps)))

    def test_04_01_reshape_double(self):
        '''Make an image twice as large by changing the shape'''
        i, j = np.mgrid[0:10, 0:10].astype(float)
        image = i + j * 10
        i, j = np.mgrid[0:19, 0:19].astype(float) / 2.0
        expected = i + j * 10
        workspace, module = self.make_workspace(image, R.R_TO_SIZE,
                                                R.I_BILINEAR)
        module.specific_width.value = 19
        module.specific_height.value = 19
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.sqrt(np.finfo(float).eps)))

    def test_04_02_reshape_half(self):
        '''Make an image half as large by changing the shape'''
        i, j = np.mgrid[0:19, 0:19].astype(float) / 2.0
        image = i + j * 10
        i, j = np.mgrid[0:10, 0:10].astype(float)
        expected = i + j * 10
        workspace, module = self.make_workspace(image, R.R_TO_SIZE,
                                                R.I_BILINEAR)
        module.specific_width.value = 10
        module.specific_height.value = 10
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.sqrt(np.finfo(float).eps)))

    def test_04_03_reshape_half_and_double(self):
        '''Make an image twice as large in one dimension and half in other'''
        i, j = np.mgrid[0:10, 0:19].astype(float)
        image = i + j * 5.0
        i, j = np.mgrid[0:19, 0:10].astype(float)
        expected = i * .5 + j * 10
        workspace, module = self.make_workspace(image, R.R_TO_SIZE,
                                                R.I_BILINEAR)
        module.specific_width.value = 10
        module.specific_height.value = 19
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(np.abs(result - expected) <=
                               np.sqrt(np.finfo(float).eps)))

    def test_04_04_reshape_using_another_images_dimensions(self):
        ''''Resize to another image's dimensions'''
        i, j = np.mgrid[0:10, 0:19].astype(float)
        image = i + j
        i, j = np.mgrid[0:19, 0:10].astype(float)
        expected = i + j
        workspace, module = self.make_workspace(image, R.R_TO_SIZE,
                                                R.I_BILINEAR)
        module.use_manual_or_image.value = R.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(module.specific_image.value, cpi.Image(expected))
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(expected.shape == result.shape)

    def test_05_01_resize_with_cropping(self):
        # This is a regression test for issue # 967
        r = np.random.RandomState()
        r.seed(501)
        i, j = np.mgrid[0:10, 0:20]
        image = i + j
        mask = r.uniform(size=image.shape) > .5
        imask = mask.astype(int)
        cropping = np.zeros((30, 40), bool)
        cropping[10:20, 10:30] = True
        workspace, module = self.make_workspace(
                image, R.R_BY_FACTOR, R.I_BILINEAR, mask, cropping)
        assert isinstance(module, R.Resize)
        module.resizing_factor.value = .5
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.mask.shape), (5, 10))
        self.assertEqual(tuple(result.crop_mask.shape), (15, 20))
        x = result.crop_image_similarly(np.zeros(result.crop_mask.shape))
        self.assertEqual(tuple(x.shape), (5, 10))

    def test_05_02_resize_with_cropping_bigger(self):
        # This is a regression test for issue # 967
        r = np.random.RandomState()
        r.seed(501)
        i, j = np.mgrid[0:10, 0:20]
        image = i + j
        mask = r.uniform(size=image.shape) > .5
        imask = mask.astype(int)
        cropping = np.zeros((30, 40), bool)
        cropping[10:20, 10:30] = True
        workspace, module = self.make_workspace(
                image, R.R_BY_FACTOR, R.I_BILINEAR, mask, cropping)
        assert isinstance(module, R.Resize)
        module.resizing_factor.value = 2
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.mask.shape), (20, 40))
        self.assertEqual(tuple(result.crop_mask.shape), (60, 80))
        x = result.crop_image_similarly(np.zeros(result.crop_mask.shape))
        self.assertEqual(tuple(x.shape), (20, 40))

    def test_05_03_resize_color(self):
        # Regression test of issue #1416
        image = np.zeros((20, 22, 3))
        workspace, module = self.make_workspace(
                image, R.R_BY_FACTOR, R.I_BILINEAR)
        assert isinstance(module, R.Resize)
        module.resizing_factor.value = .5
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (10, 11, 3))

    def test_05_04_resize_color_bw(self):
        # Regression test of issue #1416
        image = np.zeros((20, 22, 3))
        tgt_image = np.zeros((5, 11))
        workspace, module = self.make_workspace(
                image, R.R_TO_SIZE, R.I_BILINEAR)
        assert isinstance(module, R.Resize)
        module.use_manual_or_image.value = R.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(module.specific_image.value,
                                cpi.Image(tgt_image))
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (5, 11, 3))

    def test_05_05_resize_color_color(self):
        # Regression test of issue #1416
        image = np.zeros((20, 22, 3))
        tgt_image = np.zeros((10, 11, 3))
        workspace, module = self.make_workspace(
                image, R.R_TO_SIZE, R.I_BILINEAR)
        assert isinstance(module, R.Resize)
        module.use_manual_or_image.value = R.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(module.specific_image.value,
                                cpi.Image(tgt_image))
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (10, 11, 3))
