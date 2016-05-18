'''test_enhanceorsuppressspeckles - test the EnhanceOrSuppressSpeckles module'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.enhanceorsuppressfeatures as E
from centrosome.filter import enhance_dark_holes

INPUT_IMAGE_NAME = 'myimage'
OUTPUT_IMAGE_NAME = 'myfilteredimage'


class TestEnhanceOrSuppressSpeckles(unittest.TestCase):
    def make_workspace(self, image, mask):
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
        workspace, module = self.make_workspace(np.zeros((10, 10)), None)
        self.assertTrue(module.method.value == E.ENHANCE)
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == 0))

    def test_00_01_suppress_zero(self):
        '''Test suppress of an image of all zeros'''
        workspace, module = self.make_workspace(np.zeros((10, 10)), None)
        module.method.value = E.SUPPRESS
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == 0))

    def test_01_00_check_version(self):
        '''Make sure the test covers the latest revision number'''
        # Create a new test and update this one after changing settings
        self.assertEqual(E.EnhanceOrSuppressFeatures.variable_revision_number, 5)

    def test_01_01_load_v1(self):
        data = ('eJztWNFO2zAUdUqBsUkr28v26Ee60aotQ4NqKu0oEtUIVLRiQohtpnXba'
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

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertEqual(module.module_name, 'EnhanceOrSuppressFeatures')
        self.assertEqual(module.image_name.value, 'MyImage')
        self.assertEqual(module.filtered_image_name.value, 'MyEnhancedImage')
        self.assertEqual(module.method.value, E.ENHANCE)
        self.assertEqual(module.object_size, 17)

    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10583

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Initial
    Name the output image:EnhancedSpeckles
    Select the operation:Enhance
    Feature size:11
    Feature type:Speckles
    Range of hole sizes:1,10

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:EnhancedSpeckles
    Name the output image:EnhancedNeurites
    Select the operation:Enhance
    Feature size:9
    Feature type:Neurites
    Range of hole sizes:1,10

EnhanceOrSuppressFeatures:[module_num:3|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:EnhancedNeurites
    Name the output image:EnhancedDarkHoles
    Select the operation:Enhance
    Feature size:9
    Feature type:Dark holes
    Range of hole sizes:4,11

EnhanceOrSuppressFeatures:[module_num:4|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:EnhancedDarkHoles
    Name the output image:EnhancedCircles
    Select the operation:Enhance
    Feature size:9
    Feature type:Circles
    Range of hole sizes:4,11

EnhanceOrSuppressFeatures:[module_num:5|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:EnhancedCircles
    Name the output image:Suppressed
    Select the operation:Suppress
    Feature size:13
    Feature type:Circles
    Range of hole sizes:4,11
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        for module, (input_name, output_name, operation, feature_size,
                     feature_type, min_range, max_range) in zip(
                pipeline.modules(), (
                        ("Initial", "EnhancedSpeckles", E.ENHANCE, 11, E.E_SPECKLES, 1, 10),
                        ("EnhancedSpeckles", "EnhancedNeurites", E.ENHANCE, 9, E.E_NEURITES, 1, 10),
                        ("EnhancedNeurites", "EnhancedDarkHoles", E.ENHANCE, 9, E.E_DARK_HOLES, 4, 11),
                        ("EnhancedDarkHoles", "EnhancedCircles", E.ENHANCE, 9, E.E_CIRCLES, 4, 11),
                        ("EnhancedCircles", "Suppressed", E.SUPPRESS, 13, E.E_CIRCLES, 4, 11))):
            self.assertEqual(module.module_name, 'EnhanceOrSuppressFeatures')
            self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
            self.assertEqual(module.image_name, input_name)
            self.assertEqual(module.filtered_image_name, output_name)
            self.assertEqual(module.method, operation)
            self.assertEqual(module.enhance_method, feature_type)
            self.assertEqual(module.object_size, feature_size)
            self.assertEqual(module.hole_size.min, min_range)
            self.assertEqual(module.hole_size.max, max_range)

    def test_01_03_test_load_v3(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10999

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'10591\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the output image:EnhancedTexture
    Select the operation:Enhance
    Feature size:10
    Feature type:Texture
    Range of hole sizes:1,10
    Smoothing scale:3.5
    Shear angle:45
    Decay:0.90

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'10591\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:EnhancedTexture
    Name the output image:EnhancedDIC
    Select the operation:Enhance
    Feature size:10
    Feature type:DIC
    Range of hole sizes:1,10
    Smoothing scale:1.5
    Shear angle:135
    Decay:0.99
'''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.filtered_image_name, "EnhancedTexture")
        self.assertEqual(module.method, E.ENHANCE)
        self.assertEqual(module.enhance_method, E.E_TEXTURE)
        self.assertEqual(module.smoothing, 3.5)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 45)
        self.assertEqual(module.decay, .9)
        self.assertEqual(module.speckle_accuracy, E.S_SLOW)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.enhance_method, E.E_DIC)

    def test_01_05_load_v5(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150414135713
GitHash:3bad577
ModuleCount:2
HasImagePlaneDetails:False

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Dendrite
    Name the output image:EnhancedDendrite
    Select the operation:Enhance
    Feature size:10
    Feature type:Neurites
    Range of hole sizes:1,10
    Smoothing scale:2.0
    Shear angle:0
    Decay:0.95
    Enhancement method:Tubeness
    Speed and accuracy:Slow / circular

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Axon
    Name the output image:EnhancedAxon
    Select the operation:Enhance
    Feature size:10
    Feature type:Neurites
    Range of hole sizes:1,10
    Smoothing scale:2.0
    Shear angle:0
    Decay:0.95
    Enhancement method:Line structures
    Speed and accuracy:Fast / hexagonal
'''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.image_name, "Dendrite")
        self.assertEqual(module.filtered_image_name, "EnhancedDendrite")
        self.assertEqual(module.method, E.ENHANCE)
        self.assertEqual(module.enhance_method, E.E_NEURITES)
        self.assertEqual(module.smoothing, 2.0)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.decay, .95)
        self.assertEqual(module.neurite_choice, E.N_TUBENESS)
        self.assertEqual(module.speckle_accuracy, E.S_SLOW)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.speckle_accuracy, E.S_FAST)

    def test_01_05_load_v5(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120516145742

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:Dendrite
    Name the output image:EnhancedDendrite
    Select the operation:Enhance
    Feature size:10
    Feature type:Neurites
    Range of hole sizes:1,10
    Smoothing scale:2.0
    Shear angle:0
    Decay:0.95
    Enhancement method:Tubeness

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:Axon
    Name the output image:EnhancedAxon
    Select the operation:Enhance
    Feature size:10
    Feature type:Neurites
    Range of hole sizes:1,10
    Smoothing scale:2.0
    Shear angle:0
    Decay:0.95
    Enhancement method:Line structures
'''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.image_name, "Dendrite")
        self.assertEqual(module.filtered_image_name, "EnhancedDendrite")
        self.assertEqual(module.method, E.ENHANCE)
        self.assertEqual(module.enhance_method, E.E_NEURITES)
        self.assertEqual(module.smoothing, 2.0)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.decay, .95)
        self.assertEqual(module.neurite_choice, E.N_TUBENESS)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        self.assertEqual(module.neurite_choice, E.N_GRADIENT)

    def test_02_01_enhance(self):
        '''Enhance an image composed of two circles of different diameters'''
        #
        # Make an image which has circles of diameters 10 and 7. We should
        # keep the smaller circle and erase the larger
        #
        image = np.zeros((11, 20))
        expected = np.zeros((11, 20))
        i, j = np.mgrid[-5:6, -5:16]
        image[i ** 2 + j ** 2 < 23] = 1
        i, j = np.mgrid[-5:6, -15:5]
        image[i ** 2 + j ** 2 <= 9] = 1
        expected[i ** 2 + j ** 2 <= 9] = 1
        workspace, module = self.make_workspace(image,
                                                np.ones(image.shape, bool))
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_SPECKLES
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == expected))

    def test_02_02_suppress(self):
        '''Suppress a speckle in an image composed of two circles'''
        image = np.zeros((11, 20))
        expected = np.zeros((11, 20))
        i, j = np.mgrid[-5:6, -5:15]
        image[i ** 2 + j ** 2 <= 22] = 1
        expected[i ** 2 + j ** 2 <= 22] = 1
        i, j = np.mgrid[-5:6, -15:5]
        image[i ** 2 + j ** 2 <= 9] = 1
        workspace, module = self.make_workspace(image, None)
        module.method.value = E.SUPPRESS
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(np.all(result.pixel_data == expected))

    def test_03_01_enhancemask(self):
        '''Enhance a speckles image, masking a portion'''
        image = np.zeros((10, 10))
        mask = np.ones((10, 10), bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation and it
        # should be zero after the subtraction
        #
        i, j = np.mgrid[-5:5, -5:5]
        image[5, 5] = 1
        mask[np.logical_and(i ** 2 + j ** 2 <= 16, image == 0)] = False
        for speckle_accuracy in E.S_SLOW, E.S_FAST:
            #
            # Prove that, without the mask, the image is zero
            #
            workspace, module = self.make_workspace(image, None)
            assert isinstance(module, E.EnhanceOrSuppressFeatures)
            module.method.value = E.ENHANCE
            module.enhance_method.value = E.E_SPECKLES
            module.speckle_accuracy.value = speckle_accuracy
            module.object_size.value = 7
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(np.all(result.pixel_data == image))
            #
            # rescue the point with the mask
            #
            workspace, module = self.make_workspace(image, mask)
            module.method.value = E.ENHANCE
            module.enhance_method.value = E.E_SPECKLES
            module.speckle_accuracy.value = speckle_accuracy
            module.object_size.value = 7
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertFalse(result is None)
            self.assertTrue(np.all(result.pixel_data == 0))

    def test_03_02_suppressmask(self):
        '''Suppress a speckles image, masking a portion'''
        image = np.zeros((10, 10))
        mask = np.ones((10, 10), bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation
        #
        i, j = np.mgrid[-5:5, -5:5]
        image[5, 5] = 1
        mask[np.logical_and(i ** 2 + j ** 2 <= 16, image == 0)] = False
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

    def test_04_01_enhance_neurites(self):
        '''Check enhance neurites against Matlab'''
        data = np.array([[7, 9, 5, 6, 3, 3, 6, 4, 1, 6, 4, 4, 3, 3, 9, 6, 2, 8, 7, 2, 7, 7, 7, 3, 7, 5, 4, 7, 4, 4, 4,
                          4, 6, 3, 4, 6, 4, 5, 4, 1, 4, 10, 4, 8, 8, 3, 9, 5, 3, 5, 4, 7, 4, 7, 8, 12, 8, 7, 7, 5, 6, 8,
                          6, 4, 6, 8, 3, 4, 4, 4, 4, 3, 6, 10, 10, 8, 11, 4, 4, 4, 10, 3, 10, 4, 4, 4, 3, 10, 8, 7, 3,
                          8, 7, 10, 10, 2, 7, 8, 7, 7, 5, 3, 3, 8, 4, 7, 7, 6, 4, 4, 9, 6, 2, 2, 5, 4, 3, 8, 8, 5, 7, 5,
                          5, 6, 4, 3, 0, 5, 3, 1, 6, 2, 6, 6, 7, 2, 6, 8, 3, 5, 5, 6, 6, 6, 3, 5, 9, 10, 4, 2, 5, 6, 8,
                          8, 4, 8, 6, 6, 6, 5, 5, 6, 7, 5, 5, 2, 5, 1, 5, 7, 5, 2, 5, 5, 3, 5, 5, 7, 3, 7, 2, 8, 9, 6,
                          8, 0, 9, 7, 5, 4, 6, 6, 1, 2, 2, 6, 9, 10, 8, 5, 3],
                         [4, 1, 7, 1, 4, 6, 6, 6, 5, 5, 8, 8, 4, 4, 3, 3, 5, 2, 7, 8, 5, 3, 5, 1, 3, 2, 4, 5, 2, 9, 10,
                          4, 4, 8, 3, 8, 3, 2, 1, 9, 7, 7, 7, 3, 8, 3, 8, 4, 4, 7, 3, 4, 4, 6, 11, 6, 3, 5, 8, 3, 1, 3,
                          4, 3, 4, 3, 6, 6, 6, 9, 8, 10, 8, 10, 8, 1, 8, 10, 3, 8, 4, 6, 3, 11, 4, 4, 7, 3, 5, 8, 5, 3,
                          7, 5, 4, 5, 2, 5, 5, 5, 7, 5, 6, 6, 2, 7, 2, 7, 6, 1, 3, 3, 5, 5, 8, 7, 5, 7, 3, 3, 5, 5, 5,
                          3, 9, 4, 6, 10, 6, 6, 6, 2, 6, 6, 7, 7, 9, 8, 2, 5, 6, 6, 6, 3, 3, 8, 6, 10, 4, 2, 5, 6, 6, 0,
                          8, 4, 4, 6, 6, 1, 3, 8, 4, 5, 7, 5, 9, 4, 5, 7, 5, 12, 9, 3, 2, 3, 0, 7, 3, 7, 7, 3, 5, 12,
                          12, 9, 4, 3, 5, 8, 6, 6, 6, 6, 1, 2, 8, 6, 3, 7, 4],
                         [6, 5, 4, 3, 5, 5, 6, 6, 6, 5, 5, 3, 3, 4, 2, 7, 2, 3, 3, 2, 3, 7, 7, 5, 8, 7, 9, 4, 5, 5, 5,
                          3, 8, 4, 8, 6, 3, 8, 3, 7, 3, 5, 5, 3, 1, 4, 4, 3, 3, 6, 3, 6, 7, 6, 8, 7, 4, 1, 3, 1, 5, 7,
                          3, 6, 8, 4, 4, 3, 2, 0, 3, 8, 1, 8, 2, 5, 3, 10, 6, 4, 8, 1, 10, 6, 3, 7, 5, 7, 8, 7, 5, 8, 5,
                          7, 5, 3, 2, 5, 3, 5, 7, 2, 6, 3, 4, 7, 9, 7, 4, 2, 9, 8, 8, 8, 5, 7, 3, 7, 3, 3, 3, 3, 8, 4,
                          8, 6, 7, 2, 1, 8, 6, 9, 8, 8, 8, 6, 9, 9, 9, 2, 5, 3, 6, 5, 3, 5, 5, 8, 8, 6, 5, 3, 10, 6, 6,
                          6, 3, 6, 6, 6, 3, 7, 1, 2, 2, 6, 9, 6, 6, 10, 6, 5, 8, 1, 6, 10, 6, 6, 3, 2, 7, 5, 7, 7, 10,
                          11, 11, 11, 5, 6, 5, 6, 5, 3, 3, 4, 3, 9, 4, 2, 8],
                         [8, 7, 7, 2, 6, 8, 5, 5, 3, 5, 6, 3, 5, 3, 2, 2, 5, 7, 3, 7, 1, 2, 5, 5, 7, 3, 7, 7, 4, 2, 7,
                          6, 6, 8, 8, 0, 7, 3, 8, 5, 5, 8, 7, 7, 4, 3, 4, 4, 4, 4, 8, 1, 4, 6, 8, 11, 7, 3, 10, 7, 1, 5,
                          3, 3, 5, 3, 3, 4, 3, 8, 5, 4, 7, 6, 2, 2, 5, 5, 8, 8, 11, 0, 3, 10, 8, 3, 8, 5, 3, 5, 8, 2, 5,
                          3, 3, 7, 7, 5, 5, 2, 3, 6, 8, 7, 7, 4, 9, 10, 7, 7, 2, 11, 8, 5, 7, 8, 8, 1, 5, 5, 3, 5, 3, 6,
                          8, 6, 9, 6, 4, 7, 10, 5, 8, 3, 3, 3, 6, 5, 1, 5, 9, 5, 3, 5, 3, 8, 3, 5, 9, 1, 1, 6, 6, 8, 11,
                          5, 1, 5, 8, 6, 2, 8, 12, 2, 8, 4, 2, 6, 7, 4, 5, 5, 5, 5, 2, 6, 9, 4, 2, 7, 3, 5, 7, 7, 12, 8,
                          5, 8, 6, 1, 5, 6, 1, 6, 2, 3, 5, 0, 5, 1, 6],
                         [5, 1, 2, 3, 2, 5, 5, 8, 7, 7, 3, 8, 4, 5, 7, 2, 3, 2, 7, 1, 5, 3, 7, 5, 8, 2, 4, 2, 10, 6, 7,
                          4, 4, 4, 7, 2, 5, 1, 5, 2, 8, 3, 5, 8, 4, 4, 4, 6, 4, 3, 3, 8, 8, 8, 6, 11, 8, 5, 2, 3, 2, 10,
                          7, 5, 3, 2, 10, 6, 3, 5, 9, 9, 8, 5, 8, 3, 2, 5, 3, 5, 6, 5, 6, 7, 5, 8, 3, 5, 3, 5, 8, 5, 5,
                          3, 1, 7, 3, 5, 8, 2, 7, 5, 10, 6, 6, 7, 7, 7, 5, 7, 3, 5, 7, 3, 8, 7, 5, 7, 5, 2, 3, 2, 3, 6,
                          8, 5, 4, 2, 6, 6, 8, 2, 5, 8, 3, 8, 5, 5, 5, 12, 3, 7, 6, 6, 5, 6, 5, 3, 4, 2, 2, 6, 1, 3, 6,
                          1, 5, 6, 6, 0, 3, 12, 7, 7, 5, 1, 1, 6, 8, 2, 3, 5, 0, 4, 6, 6, 6, 1, 7, 1, 9, 5, 3, 9, 7, 6,
                          5, 3, 7, 2, 5, 8, 2, 2, 3, 3, 3, 3, 1, 6, 8],
                         [2, 8, 6, 7, 3, 5, 9, 3, 3, 5, 5, 9, 3, 0, 5, 2, 5, 2, 5, 5, 5, 3, 3, 5, 5, 9, 8, 6, 9, 4, 2,
                          2, 9, 7, 5, 1, 8, 3, 7, 3, 8, 7, 8, 3, 6, 6, 8, 4, 3, 6, 3, 6, 2, 6, 8, 8, 6, 10, 5, 5, 2, 7,
                          11, 6, 6, 5, 8, 6, 3, 7, 9, 6, 6, 6, 5, 6, 3, 3, 7, 10, 6, 6, 3, 8, 5, 3, 8, 5, 7, 7, 7, 5, 5,
                          7, 7, 7, 10, 7, 5, 7, 3, 6, 7, 10, 7, 8, 7, 5, 3, 3, 4, 7, 7, 2, 5, 7, 5, 5, 5, 5, 7, 3, 8, 5,
                          5, 5, 7, 7, 6, 5, 6, 6, 3, 6, 3, 3, 5, 5, 7, 3, 10, 10, 1, 5, 5, 5, 3, 6, 2, 8, 6, 6, 3, 4, 4,
                          6, 6, 3, 5, 4, 7, 10, 9, 3, 9, 7, 4, 6, 6, 6, 3, 5, 2, 2, 2, 6, 9, 2, 2, 1, 9, 3, 5, 5, 9, 6,
                          10, 8, 10, 9, 4, 4, 6, 2, 3, 2, 9, 2, 6, 3, 3],
                         [0, 4, 11, 3, 4, 3, 4, 8, 6, 1, 4, 8, 1, 3, 3, 8, 8, 3, 7, 7, 7, 5, 5, 3, 8, 3, 2, 5, 3, 6, 7,
                          1, 8, 4, 6, 9, 4, 8, 1, 7, 5, 5, 1, 4, 9, 6, 4, 4, 4, 6, 3, 8, 7, 8, 5, 7, 8, 12, 8, 5, 2, 7,
                          10, 0, 2, 0, 5, 8, 8, 7, 7, 10, 3, 5, 5, 8, 8, 5, 5, 1, 6, 1, 3, 2, 2, 8, 2, 7, 0, 7, 7, 8, 8,
                          5, 5, 5, 8, 7, 1, 8, 5, 3, 8, 5, 5, 8, 3, 5, 6, 6, 5, 3, 5, 2, 3, 3, 7, 2, 8, 2, 2, 6, 8, 3,
                          5, 4, 2, 2, 4, 3, 8, 5, 5, 7, 7, 7, 9, 1, 7, 4, 1, 3, 9, 5, 8, 5, 2, 4, 4, 2, 6, 4, 2, 6, 3,
                          6, 11, 4, 2, 9, 2, 4, 7, 5, 3, 3, 1, 6, 1, 3, 9, 5, 8, 1, 6, 4, 6, 6, 1, 5, 3, 3, 5, 2, 5, 5,
                          5, 5, 6, 9, 3, 9, 5, 9, 5, 5, 8, 8, 4, 5, 5],
                         [4, 3, 8, 4, 4, 1, 9, 4, 6, 6, 8, 4, 4, 9, 8, 9, 7, 3, 5, 1, 11, 6, 3, 1, 3, 7, 8, 3, 5, 5, 5,
                          9, 3, 9, 6, 4, 1, 3, 1, 9, 7, 7, 3, 4, 2, 3, 4, 4, 8, 4, 4, 10, 7, 5, 5, 8, 8, 2, 9, 5, 2, 7,
                          5, 7, 7, 6, 3, 5, 1, 7, 7, 4, 8, 5, 5, 6, 8, 3, 6, 5, 3, 6, 6, 7, 3, 5, 9, 9, 7, 5, 5, 7, 2,
                          5, 3, 8, 5, 2, 5, 5, 7, 5, 5, 3, 7, 8, 5, 5, 5, 2, 5, 7, 7, 8, 7, 8, 2, 7, 5, 7, 7, 5, 5, 8,
                          2, 5, 6, 6, 9, 8, 6, 6, 5, 5, 5, 3, 5, 4, 9, 4, 9, 5, 5, 5, 4, 4, 2, 4, 8, 4, 11, 11, 9, 5,
                          10, 7, 4, 5, 7, 2, 5, 5, 5, 2, 5, 5, 6, 2, 7, 3, 3, 2, 4, 2, 4, 6, 6, 6, 3, 5, 7, 5, 5, 3, 1,
                          5, 1, 8, 6, 7, 5, 5, 5, 9, 2, 8, 8, 4, 6, 5, 4],
                         [4, 9, 4, 3, 6, 8, 3, 3, 8, 4, 8, 8, 6, 4, 4, 7, 5, 3, 6, 3, 4, 4, 6, 4, 5, 3, 3, 7, 8, 8, 7,
                          5, 9, 6, 4, 3, 6, 4, 4, 6, 3, 4, 8, 7, 7, 5, 5, 7, 8, 2, 6, 9, 7, 3, 7, 7, 9, 9, 7, 2, 7, 5,
                          5, 6, 5, 3, 1, 4, 4, 9, 6, 9, 7, 6, 6, 6, 5, 6, 2, 6, 8, 8, 3, 10, 5, 5, 3, 9, 2, 3, 5, 5, 7,
                          7, 5, 3, 3, 3, 3, 7, 9, 3, 3, 1, 6, 2, 7, 11, 5, 3, 8, 2, 2, 3, 7, 5, 2, 8, 5, 10, 9, 5, 3,
                          10, 8, 8, 3, 3, 6, 3, 3, 5, 2, 3, 5, 3, 3, 6, 6, 5, 5, 5, 5, 2, 5, 9, 9, 8, 8, 4, 8, 2, 2, 1,
                          2, 9, 4, 2, 1, 5, 2, 4, 2, 10, 4, 2, 8, 10, 7, 3, 3, 0, 10, 4, 4, 4, 9, 6, 3, 7, 5, 3, 5, 10,
                          5, 3, 6, 6, 10, 4, 7, 1, 7, 2, 2, 2, 4, 2, 6, 5, 5],
                         [1, 6, 3, 3, 1, 3, 4, 3, 1, 4, 1, 0, 8, 10, 5, 5, 2, 3, 5, 9, 6, 11, 1, 6, 7, 5, 5, 7, 8, 2, 7,
                          8, 8, 8, 0, 8, 6, 3, 3, 4, 4, 10, 1, 3, 5, 7, 3, 3, 3, 5, 9, 6, 7, 8, 5, 7, 10, 7, 10, 7, 2,
                          7, 0, 2, 2, 8, 9, 4, 6, 2, 7, 2, 7, 6, 8, 3, 6, 6, 2, 1, 3, 6, 5, 5, 3, 3, 6, 4, 4, 3, 7, 9,
                          7, 9, 2, 9, 7, 7, 5, 3, 5, 9, 1, 5, 8, 8, 3, 5, 3, 5, 3, 3, 7, 5, 7, 10, 5, 6, 6, 5, 5, 11, 6,
                          5, 5, 5, 3, 3, 8, 5, 2, 7, 9, 5, 5, 7, 1, 2, 6, 5, 9, 5, 5, 4, 7, 3, 6, 6, 8, 2, 6, 4, 1, 2,
                          7, 5, 5, 4, 6, 5, 5, 4, 4, 5, 3, 1, 5, 4, 0, 9, 0, 3, 8, 2, 2, 4, 1, 6, 2, 8, 9, 5, 7, 2, 3,
                          5, 5, 4, 5, 4, 7, 5, 5, 2, 4, 4, 4, 7, 0, 4, 5],
                         [3, 8, 3, 3, 4, 8, 4, 4, 6, 3, 4, 4, 4, 5, 5, 7, 8, 5, 3, 8, 7, 3, 7, 6, 8, 5, 3, 3, 7, 3, 5,
                          5, 5, 2, 7, 0, 3, 4, 6, 6, 8, 6, 8, 3, 7, 8, 7, 3, 3, 3, 7, 2, 4, 7, 10, 11, 8, 7, 7, 7, 2, 5,
                          6, 1, 6, 11, 2, 4, 4, 4, 2, 7, 4, 4, 4, 9, 9, 7, 6, 8, 6, 10, 5, 7, 0, 7, 6, 6, 4, 4, 3, 7, 5,
                          3, 9, 5, 7, 9, 9, 3, 3, 11, 5, 1, 10, 10, 7, 3, 9, 5, 5, 5, 5, 9, 5, 3, 11, 5, 3, 3, 3, 6, 2,
                          5, 6, 3, 5, 3, 5, 5, 3, 7, 3, 3, 5, 9, 5, 5, 8, 5, 5, 4, 4, 4, 5, 9, 2, 3, 8, 5, 4, 5, 9, 5,
                          2, 5, 6, 8, 4, 6, 7, 5, 2, 4, 8, 4, 5, 2, 5, 2, 5, 4, 9, 4, 8, 4, 5, 8, 1, 4, 4, 2, 2, 3, 1,
                          5, 4, 7, 9, 9, 5, 7, 4, 9, 4, 6, 1, 3, 8, 3, 5],
                         [7, 7, 7, 7, 9, 2, 5, 7, 9, 9, 6, 0, 2, 5, 2, 7, 5, 2, 7, 3, 7, 3, 5, 2, 3, 7, 8, 8, 5, 5, 5,
                          5, 3, 5, 10, 5, 3, 3, 5, 8, 5, 8, 1, 5, 5, 3, 5, 5, 7, 5, 6, 4, 6, 9, 9, 9, 10, 7, 5, 10, 2,
                          0, 6, 5, 3, 7, 3, 6, 4, 4, 10, 9, 7, 7, 2, 7, 10, 4, 9, 1, 3, 9, 3, 7, 1, 4, 6, 6, 4, 6, 6, 9,
                          7, 7, 9, 10, 6, 7, 3, 3, 6, 6, 3, 1, 6, 2, 0, 5, 7, 7, 9, 3, 3, 9, 3, 5, 5, 7, 5, 6, 6, 1, 7,
                          3, 7, 2, 3, 2, 3, 5, 5, 9, 3, 7, 2, 7, 5, 7, 7, 7, 5, 4, 5, 2, 5, 2, 4, 4, 5, 2, 5, 8, 5, 5,
                          0, 9, 9, 9, 4, 8, 6, 4, 4, 7, 0, 8, 4, 6, 4, 5, 8, 4, 9, 6, 6, 5, 6, 8, 6, 8, 6, 8, 7, 5, 3,
                          5, 2, 9, 9, 5, 5, 4, 7, 5, 2, 8, 8, 7, 5, 7, 3],
                         [8, 3, 7, 5, 7, 6, 6, 5, 3, 5, 5, 6, 3, 3, 2, 7, 3, 5, 5, 7, 5, 7, 5, 0, 5, 11, 8, 5, 3, 5, 5,
                          7, 5, 3, 6, 5, 6, 5, 5, 6, 6, 10, 5, 7, 7, 7, 7, 3, 3, 9, 6, 2, 2, 7, 2, 9, 5, 9, 11, 8, 6, 3,
                          6, 3, 6, 2, 2, 7, 9, 7, 8, 6, 2, 9, 6, 3, 5, 5, 6, 6, 2, 5, 7, 7, 7, 6, 6, 2, 4, 4, 2, 2, 5,
                          3, 5, 3, 3, 3, 5, 3, 5, 10, 3, 10, 5, 6, 4, 10, 7, 5, 5, 3, 2, 6, 5, 7, 5, 3, 7, 2, 3, 10, 9,
                          5, 5, 3, 2, 2, 9, 2, 3, 2, 3, 7, 2, 9, 5, 7, 5, 4, 7, 2, 5, 7, 5, 4, 7, 7, 7, 4, 7, 10, 5, 4,
                          6, 5, 6, 4, 4, 9, 7, 4, 4, 10, 4, 2, 8, 6, 6, 11, 4, 9, 6, 9, 8, 3, 6, 3, 1, 4, 2, 4, 8, 10,
                          2, 4, 5, 5, 5, 4, 5, 5, 7, 2, 4, 2, 6, 5, 7, 4, 5],
                         [5, 2, 8, 6, 6, 6, 6, 4, 5, 1, 5, 4, 5, 2, 7, 5, 2, 8, 2, 3, 5, 4, 3, 3, 5, 3, 3, 5, 5, 5, 3,
                          7, 2, 7, 1, 8, 3, 6, 8, 6, 6, 4, 6, 4, 4, 3, 9, 9, 3, 0, 2, 6, 6, 9, 1, 9, 7, 9, 8, 4, 6, 5,
                          3, 5, 6, 5, 2, 4, 3, 8, 5, 2, 5, 6, 6, 3, 3, 6, 8, 8, 9, 3, 3, 5, 2, 4, 7, 2, 6, 4, 6, 7, 3,
                          9, 8, 1, 8, 5, 5, 3, 0, 5, 7, 5, 7, 4, 7, 9, 0, 3, 9, 10, 1, 3, 7, 7, 7, 3, 2, 9, 2, 8, 3, 5,
                          7, 5, 2, 5, 5, 7, 5, 7, 9, 9, 5, 3, 9, 4, 9, 10, 1, 4, 7, 9, 9, 5, 5, 5, 4, 2, 7, 5, 4, 9, 8,
                          2, 5, 5, 0, 4, 8, 2, 4, 9, 4, 2, 6, 9, 4, 4, 8, 8, 8, 6, 2, 3, 6, 1, 5, 6, 6, 2, 8, 2, 5, 9,
                          2, 7, 4, 4, 7, 9, 7, 4, 4, 4, 9, 1, 5, 4, 5],
                         [7, 5, 7, 7, 6, 6, 2, 7, 6, 8, 3, 3, 7, 8, 3, 3, 7, 7, 3, 7, 5, 5, 6, 8, 6, 5, 3, 3, 3, 2, 5,
                          1, 5, 8, 3, 3, 6, 5, 6, 1, 4, 6, 11, 6, 6, 9, 6, 8, 7, 6, 5, 5, 7, 9, 7, 7, 7, 9, 8, 6, 12, 3,
                          8, 5, 10, 6, 1, 8, 5, 5, 5, 3, 5, 8, 8, 6, 6, 6, 8, 3, 7, 9, 2, 5, 9, 2, 8, 8, 4, 6, 7, 7, 9,
                          10, 10, 5, 10, 6, 3, 5, 3, 6, 7, 4, 2, 7, 9, 5, 9, 5, 7, 5, 5, 1, 1, 3, 3, 5, 7, 7, 7, 10, 5,
                          3, 5, 5, 5, 5, 9, 1, 7, 2, 3, 4, 9, 5, 9, 7, 4, 4, 7, 3, 3, 10, 2, 5, 2, 2, 7, 5, 7, 5, 1, 2,
                          6, 10, 4, 9, 7, 4, 4, 6, 8, 6, 7, 5, 5, 1, 6, 6, 6, 6, 1, 6, 6, 0, 3, 5, 3, 6, 5, 4, 2, 2, 1,
                          5, 7, 4, 4, 5, 9, 7, 4, 5, 7, 7, 2, 5, 7, 10, 4],
                         [8, 8, 2, 2, 6, 6, 4, 6, 2, 5, 5, 5, 11, 5, 3, 5, 7, 7, 3, 6, 5, 5, 3, 8, 5, 6, 6, 3, 5, 7, 4,
                          2, 6, 10, 3, 5, 5, 5, 6, 4, 7, 2, 2, 9, 4, 8, 9, 9, 9, 9, 9, 3, 5, 2, 6, 3, 3, 9, 7, 6, 8, 7,
                          1, 6, 8, 4, 6, 5, 5, 9, 7, 5, 6, 5, 3, 3, 5, 5, 5, 6, 10, 9, 7, 5, 5, 5, 6, 4, 1, 4, 4, 5, 5,
                          3, 3, 5, 1, 0, 7, 10, 6, 2, 2, 2, 1, 9, 5, 5, 3, 5, 9, 10, 6, 5, 3, 7, 5, 7, 3, 7, 9, 5, 5, 7,
                          5, 5, 7, 7, 8, 10, 7, 2, 4, 2, 7, 9, 5, 5, 5, 2, 4, 1, 4, 7, 5, 4, 5, 5, 1, 5, 2, 5, 4, 2, 6,
                          8, 0, 8, 5, 6, 2, 2, 6, 2, 8, 5, 2, 7, 9, 6, 8, 1, 6, 2, 6, 6, 10, 6, 5, 8, 8, 8, 1, 9, 4, 8,
                          5, 9, 5, 6, 6, 6, 4, 2, 5, 4, 3, 3, 8, 6, 11],
                         [3, 5, 6, 6, 2, 6, 7, 7, 4, 7, 2, 5, 3, 2, 3, 8, 3, 6, 1, 7, 8, 3, 8, 8, 5, 9, 6, 5, 3, 2, 1,
                          11, 6, 1, 5, 8, 8, 2, 9, 7, 6, 4, 2, 10, 6, 8, 2, 0, 7, 6, 8, 7, 7, 5, 6, 6, 7, 12, 8, 10, 12,
                          7, 4, 8, 9, 6, 9, 12, 9, 2, 7, 8, 5, 6, 6, 6, 3, 5, 5, 1, 5, 9, 5, 3, 9, 3, 5, 8, 4, 6, 3, 5,
                          5, 6, 6, 5, 5, 6, 12, 9, 4, 6, 4, 9, 5, 5, 7, 5, 9, 5, 10, 5, 8, 6, 6, 3, 3, 9, 5, 0, 8, 3, 5,
                          1, 2, 6, 1, 4, 9, 5, 5, 5, 7, 4, 5, 9, 7, 2, 4, 5, 5, 4, 9, 5, 5, 4, 5, 10, 2, 9, 4, 4, 2, 5,
                          11, 5, 3, 8, 3, 4, 6, 4, 6, 4, 6, 5, 5, 5, 8, 9, 9, 4, 4, 4, 11, 6, 4, 6, 7, 9, 0, 4, 4, 11,
                          9, 4, 4, 1, 4, 6, 4, 6, 4, 6, 11, 10, 10, 6, 8, 11, 4],
                         [5, 7, 5, 2, 4, 2, 4, 6, 8, 3, 5, 2, 3, 3, 7, 7, 6, 5, 5, 7, 5, 6, 3, 3, 3, 2, 7, 1, 4, 9, 2,
                          3, 3, 5, 11, 10, 6, 2, 6, 9, 4, 7, 7, 9, 6, 8, 3, 5, 6, 6, 6, 8, 5, 5, 5, 8, 8, 12, 10, 12, 8,
                          11, 5, 9, 4, 8, 4, 4, 4, 5, 5, 1, 8, 5, 6, 6, 3, 8, 5, 2, 8, 5, 9, 5, 5, 11, 3, 10, 6, 3, 10,
                          1, 5, 5, 5, 5, 5, 2, 6, 1, 1, 6, 4, 2, 3, 5, 9, 3, 5, 7, 7, 6, 6, 5, 3, 10, 5, 5, 3, 1, 6, 5,
                          6, 0, 7, 7, 3, 6, 3, 3, 6, 3, 7, 4, 4, 2, 4, 2, 5, 9, 6, 2, 1, 5, 9, 9, 8, 9, 7, 5, 7, 4, 5,
                          5, 5, 8, 3, 7, 5, 8, 8, 6, 8, 4, 6, 5, 5, 9, 6, 4, 6, 8, 4, 6, 6, 6, 2, 2, 5, 6, 5, 4, 4, 4,
                          6, 2, 10, 1, 1, 1, 3, 3, 0, 2, 2, 8, 8, 4, 13, 5, 4],
                         [5, 2, 2, 2, 6, 2, 7, 3, 8, 8, 3, 1, 3, 5, 9, 6, 5, 3, 3, 6, 3, 5, 5, 5, 4, 4, 5, 9, 5, 9, 2,
                          9, 10, 5, 8, 1, 2, 6, 6, 6, 6, 4, 9, 4, 5, 6, 7, 2, 1, 3, 5, 7, 1, 2, 1, 3, 7, 5, 9, 12, 8,
                          11, 9, 7, 3, 5, 6, 4, 2, 1, 5, 5, 4, 6, 8, 5, 6, 2, 4, 7, 5, 8, 2, 4, 5, 9, 9, 7, 6, 5, 5, 8,
                          3, 5, 3, 3, 5, 9, 6, 8, 9, 9, 6, 4, 5, 5, 5, 7, 3, 7, 2, 7, 8, 6, 5, 7, 9, 2, 2, 2, 8, 6, 10,
                          1, 8, 10, 6, 5, 5, 4, 6, 6, 6, 10, 4, 5, 4, 7, 0, 11, 10, 4, 4, 4, 4, 8, 2, 2, 7, 5, 4, 5, 4,
                          5, 10, 7, 3, 8, 7, 5, 1, 8, 4, 6, 2, 5, 4, 4, 8, 4, 1, 9, 6, 3, 3, 5, 8, 6, 7, 8, 3, 6, 8, 4,
                          6, 8, 5, 7, 5, 1, 8, 8, 3, 8, 4, 8, 4, 6, 7, 12, 10],
                         [5, 7, 9, 1, 8, 0, 3, 7, 8, 7, 7, 10, 6, 6, 10, 1, 8, 8, 6, 8, 4, 7, 7, 9, 4, 5, 7, 0, 5, 2, 5,
                          7, 5, 3, 6, 2, 2, 4, 7, 6, 7, 13, 3, 9, 6, 6, 7, 2, 9, 5, 5, 5, 5, 1, 8, 8, 5, 5, 6, 11, 10,
                          9, 11, 11, 11, 8, 6, 6, 4, 6, 10, 5, 7, 2, 7, 2, 5, 7, 4, 4, 5, 2, 1, 4, 7, 7, 7, 4, 7, 6, 6,
                          3, 6, 1, 3, 5, 3, 7, 6, 4, 4, 4, 6, 1, 8, 7, 9, 5, 7, 5, 5, 3, 5, 10, 6, 2, 6, 4, 4, 6, 4, 2,
                          5, 5, 6, 6, 3, 6, 8, 3, 0, 5, 7, 5, 4, 4, 2, 6, 2, 6, 8, 4, 6, 8, 10, 1, 4, 5, 5, 4, 7, 2, 10,
                          7, 5, 6, 7, 5, 8, 8, 5, 3, 4, 8, 5, 4, 6, 6, 6, 6, 2, 6, 5, 3, 7, 3, 5, 9, 0, 10, 1, 2, 8, 4,
                          4, 5, 5, 8, 8, 8, 7, 5, 3, 8, 6, 6, 10, 6, 6, 12, 10],
                         [1, 3, 4, 7, 4, 7, 7, 2, 5, 7, 5, 2, 5, 1, 6, 6, 6, 8, 6, 6, 6, 6, 6, 2, 9, 7, 9, 3, 2, 4, 4,
                          8, 0, 0, 6, 4, 6, 2, 9, 4, 9, 8, 10, 4, 4, 5, 2, 3, 5, 7, 7, 5, 4, 7, 3, 5, 5, 5, 9, 7, 11,
                          12, 9, 8, 2, 6, 4, 4, 2, 7, 9, 4, 7, 9, 9, 9, 7, 4, 4, 5, 7, 6, 6, 1, 4, 2, 7, 7, 7, 7, 1, 0,
                          5, 3, 10, 9, 5, 9, 2, 8, 5, 9, 3, 9, 9, 4, 3, 5, 9, 2, 2, 5, 7, 1, 6, 4, 6, 6, 6, 4, 4, 4, 9,
                          1, 2, 8, 4, 1, 5, 3, 4, 6, 4, 7, 4, 2, 6, 6, 10, 4, 8, 2, 2, 6, 2, 2, 7, 7, 10, 2, 4, 5, 7, 7,
                          4, 4, 8, 5, 5, 3, 7, 8, 8, 10, 6, 1, 8, 10, 4, 6, 6, 1, 1, 8, 8, 3, 3, 5, 3, 3, 3, 5, 6, 6, 5,
                          1, 5, 5, 5, 5, 3, 8, 5, 1, 4, 4, 2, 8, 10, 10, 10],
                         [2, 6, 6, 4, 6, 4, 4, 6, 10, 5, 5, 7, 7, 2, 5, 4, 6, 8, 8, 6, 6, 6, 0, 6, 4, 4, 9, 10, 6, 6, 1,
                          6, 4, 3, 6, 4, 4, 2, 8, 6, 8, 4, 2, 2, 1, 10, 4, 5, 9, 4, 7, 7, 2, 5, 5, 5, 7, 3, 10, 9, 9, 8,
                          10, 3, 5, 8, 4, 4, 5, 8, 7, 7, 5, 4, 2, 5, 4, 7, 5, 2, 6, 6, 4, 1, 5, 5, 9, 9, 4, 4, 5, 3, 11,
                          8, 1, 5, 1, 2, 1, 5, 4, 4, 5, 10, 5, 4, 1, 5, 3, 7, 7, 9, 5, 0, 6, 6, 8, 6, 4, 2, 5, 3, 7, 8,
                          9, 9, 6, 6, 3, 5, 2, 1, 2, 2, 1, 10, 6, 2, 10, 4, 2, 6, 1, 4, 5, 4, 9, 4, 2, 4, 4, 4, 2, 1, 6,
                          6, 1, 6, 4, 10, 4, 10, 10, 4, 8, 4, 2, 2, 8, 1, 2, 8, 7, 1, 3, 5, 1, 3, 10, 3, 8, 7, 0, 2, 5,
                          5, 7, 5, 7, 1, 3, 1, 3, 5, 5, 4, 9, 10, 6, 6, 10],
                         [5, 6, 4, 7, 2, 4, 2, 4, 9, 2, 7, 5, 9, 5, 7, 10, 4, 5, 5, 1, 6, 3, 5, 5, 2, 1, 2, 6, 4, 4, 8,
                          9, 7, 3, 5, 9, 6, 2, 2, 8, 6, 4, 6, 2, 2, 5, 8, 3, 10, 5, 9, 9, 7, 7, 9, 5, 7, 5, 7, 11, 11,
                          12, 8, 8, 7, 5, 2, 1, 0, 4, 8, 5, 7, 7, 5, 7, 5, 7, 4, 1, 4, 9, 6, 5, 5, 7, 2, 7, 2, 2, 10, 3,
                          3, 4, 4, 4, 2, 2, 4, 2, 3, 8, 4, 4, 2, 7, 7, 2, 7, 5, 5, 3, 9, 2, 8, 8, 2, 8, 2, 9, 7, 7, 11,
                          5, 8, 9, 6, 8, 9, 2, 7, 4, 6, 10, 4, 2, 0, 6, 8, 4, 4, 8, 6, 7, 4, 2, 2, 9, 5, 2, 5, 5, 2, 5,
                          0, 8, 4, 6, 4, 4, 10, 1, 1, 6, 2, 1, 10, 5, 5, 3, 4, 3, 7, 1, 5, 5, 5, 5, 5, 7, 0, 0, 10, 8,
                          5, 1, 5, 5, 5, 1, 7, 9, 7, 3, 3, 9, 1, 10, 3, 10, 4],
                         [7, 4, 7, 6, 7, 6, 7, 7, 3, 3, 9, 3, 7, 3, 5, 9, 9, 8, 6, 3, 8, 8, 5, 5, 9, 4, 2, 1, 8, 4, 9,
                          4, 7, 2, 10, 1, 8, 6, 4, 9, 8, 4, 6, 8, 4, 7, 1, 9, 8, 5, 2, 7, 4, 5, 1, 4, 7, 5, 4, 10, 10,
                          8, 10, 10, 7, 7, 10, 1, 7, 6, 8, 10, 9, 2, 5, 5, 4, 10, 4, 9, 6, 9, 4, 5, 4, 2, 4, 4, 5, 2, 5,
                          8, 6, 5, 9, 7, 8, 9, 4, 6, 9, 9, 4, 8, 3, 5, 3, 5, 2, 2, 5, 2, 9, 9, 8, 9, 9, 9, 9, 12, 4, 11,
                          9, 7, 7, 11, 9, 4, 6, 5, 5, 4, 7, 6, 6, 6, 2, 2, 8, 6, 8, 6, 1, 2, 7, 9, 5, 4, 7, 9, 7, 5, 5,
                          7, 4, 4, 8, 6, 10, 6, 2, 2, 8, 2, 6, 6, 8, 7, 7, 3, 7, 2, 3, 7, 5, 7, 0, 7, 3, 10, 5, 10, 7,
                          7, 8, 5, 7, 3, 0, 3, 5, 3, 3, 7, 7, 1, 5, 9, 12, 5, 11],
                         [9, 10, 4, 2, 4, 6, 4, 5, 5, 7, 3, 5, 5, 7, 7, 9, 6, 1, 6, 6, 6, 5, 5, 6, 8, 6, 6, 4, 1, 2, 4,
                          4, 5, 2, 7, 5, 5, 4, 8, 8, 6, 8, 4, 9, 6, 9, 5, 5, 7, 3, 1, 7, 2, 7, 5, 7, 1, 4, 4, 5, 11, 10,
                          11, 10, 10, 4, 10, 7, 5, 0, 6, 4, 1, 5, 5, 2, 5, 6, 7, 6, 4, 1, 6, 5, 5, 7, 5, 2, 4, 5, 5, 9,
                          6, 5, 0, 5, 8, 11, 9, 4, 6, 13, 4, 8, 8, 3, 6, 8, 3, 5, 8, 2, 8, 8, 8, 10, 9, 11, 11, 8, 10,
                          9, 9, 7, 7, 6, 7, 4, 4, 5, 3, 9, 5, 2, 8, 10, 6, 6, 2, 10, 1, 2, 4, 5, 4, 4, 4, 4, 7, 2, 6, 4,
                          6, 4, 6, 6, 4, 2, 4, 2, 2, 2, 8, 4, 8, 0, 5, 5, 0, 1, 5, 7, 10, 8, 5, 7, 8, 7, 5, 1, 5, 7, 3,
                          3, 5, 10, 5, 0, 6, 0, 1, 3, 5, 1, 1, 7, 7, 5, 5, 5, 6],
                         [2, 2, 8, 4, 4, 4, 3, 5, 3, 5, 7, 7, 7, 5, 9, 2, 4, 8, 5, 6, 5, 8, 6, 5, 6, 4, 4, 6, 4, 2, 8,
                          4, 7, 4, 4, 7, 2, 5, 4, 9, 8, 9, 9, 6, 4, 7, 9, 4, 2, 5, 5, 3, 10, 2, 2, 2, 10, 6, 2, 8, 10,
                          6, 7, 13, 7, 4, 4, 1, 3, 10, 2, 6, 10, 4, 10, 1, 2, 8, 2, 4, 4, 2, 8, 6, 4, 2, 4, 2, 4, 4, 5,
                          5, 8, 6, 11, 8, 4, 6, 7, 9, 9, 9, 9, 2, 8, 8, 6, 6, 6, 8, 6, 6, 8, 8, 9, 5, 7, 4, 9, 9, 4, 5,
                          7, 9, 9, 6, 9, 11, 9, 12, 3, 2, 9, 4, 6, 4, 2, 11, 4, 2, 10, 4, 1, 9, 7, 4, 5, 9, 7, 4, 4, 6,
                          4, 2, 11, 5, 7, 9, 8, 2, 8, 4, 4, 2, 6, 5, 4, 4, 7, 2, 5, 8, 5, 8, 6, 0, 10, 5, 7, 8, 7, 5, 0,
                          7, 5, 5, 9, 7, 6, 3, 7, 7, 9, 5, 1, 3, 11, 7, 7, 9, 7],
                         [3, 2, 5, 3, 3, 2, 7, 5, 3, 3, 9, 3, 7, 5, 9, 2, 4, 8, 6, 5, 6, 3, 5, 6, 9, 8, 2, 0, 4, 1, 6,
                          2, 4, 4, 5, 5, 5, 9, 6, 6, 11, 6, 6, 5, 5, 2, 9, 1, 0, 5, 3, 3, 5, 4, 5, 7, 8, 11, 2, 5, 7,
                          10, 5, 8, 10, 2, 7, 5, 5, 7, 6, 4, 4, 8, 6, 8, 4, 10, 8, 6, 11, 0, 10, 7, 9, 5, 2, 7, 5, 5, 5,
                          2, 3, 10, 9, 9, 8, 11, 11, 4, 5, 6, 8, 8, 5, 5, 6, 9, 6, 6, 8, 8, 12, 12, 13, 10, 2, 8, 6, 9,
                          9, 5, 9, 9, 9, 4, 2, 6, 4, 6, 10, 5, 1, 5, 5, 4, 4, 4, 4, 8, 4, 5, 2, 4, 4, 2, 2, 9, 7, 4, 6,
                          6, 11, 4, 4, 4, 7, 5, 7, 6, 6, 8, 8, 10, 9, 6, 6, 7, 9, 5, 9, 3, 7, 5, 7, 1, 10, 7, 1, 10, 10,
                          0, 5, 5, 7, 7, 5, 5, 9, 4, 11, 1, 3, 7, 5, 2, 7, 12, 8, 9, 2],
                         [0, 10, 5, 0, 10, 7, 9, 2, 2, 3, 3, 10, 5, 6, 10, 8, 4, 6, 8, 6, 8, 8, 0, 6, 8, 6, 2, 4, 11, 9,
                          4, 4, 1, 1, 5, 5, 2, 5, 4, 10, 10, 1, 10, 7, 5, 2, 0, 6, 1, 6, 4, 4, 4, 4, 5, 7, 8, 5, 5, 5,
                          7, 11, 9, 10, 7, 1, 7, 5, 5, 5, 8, 8, 6, 6, 6, 8, 4, 4, 6, 8, 3, 4, 4, 10, 2, 4, 1, 9, 7, 7,
                          7, 5, 10, 12, 8, 8, 1, 4, 5, 7, 11, 4, 5, 8, 8, 8, 8, 9, 2, 2, 11, 9, 7, 5, 14, 9, 2, 6, 8, 8,
                          4, 9, 5, 2, 8, 4, 8, 4, 10, 10, 6, 10, 5, 7, 5, 7, 5, 6, 4, 0, 6, 2, 6, 7, 4, 4, 2, 7, 1, 6,
                          2, 1, 9, 6, 6, 8, 2, 3, 1, 7, 6, 0, 7, 10, 6, 4, 2, 6, 4, 9, 7, 2, 5, 5, 3, 4, 2, 9, 7, 4, 4,
                          2, 9, 7, 9, 7, 10, 4, 6, 9, 1, 4, 6, 1, 2, 9, 4, 8, 6, 8, 1],
                         [5, 5, 6, 4, 6, 5, 3, 3, 5, 9, 3, 3, 2, 10, 6, 2, 6, 1, 5, 5, 3, 1, 3, 1, 8, 2, 4, 2, 6, 4, 2,
                          11, 9, 4, 7, 5, 7, 2, 7, 4, 1, 10, 5, 5, 5, 4, 7, 6, 6, 11, 2, 5, 4, 5, 7, 8, 5, 8, 6, 7, 13,
                          5, 7, 7, 7, 5, 5, 5, 3, 2, 2, 8, 4, 4, 8, 10, 6, 10, 2, 5, 7, 2, 4, 2, 10, 3, 7, 2, 7, 7, 10,
                          8, 8, 8, 2, 7, 10, 5, 11, 2, 8, 5, 7, 8, 5, 5, 3, 6, 11, 9, 7, 4, 8, 4, 8, 4, 6, 8, 0, 6, 6,
                          2, 2, 10, 8, 2, 4, 6, 6, 6, 10, 6, 2, 6, 5, 9, 4, 4, 4, 4, 6, 7, 7, 6, 2, 6, 7, 7, 6, 2, 6, 6,
                          2, 8, 4, 2, 6, 4, 0, 2, 11, 3, 5, 11, 0, 4, 7, 9, 6, 7, 7, 0, 5, 5, 5, 4, 4, 0, 7, 5, 5, 5, 2,
                          7, 2, 6, 6, 7, 0, 5, 9, 8, 4, 2, 6, 9, 8, 8, 4, 4, 4],
                         [8, 5, 9, 6, 4, 6, 5, 6, 3, 5, 7, 9, 3, 9, 10, 6, 2, 4, 6, 6, 6, 3, 8, 1, 2, 4, 5, 7, 2, 6, 4,
                          4, 2, 8, 9, 2, 4, 4, 9, 2, 4, 8, 8, 4, 5, 2, 4, 4, 5, 5, 1, 1, 5, 5, 5, 5, 5, 7, 8, 4, 3, 7,
                          15, 9, 9, 5, 5, 3, 7, 4, 4, 1, 10, 6, 4, 4, 4, 10, 7, 8, 3, 7, 7, 9, 7, 6, 8, 8, 10, 8, 10, 7,
                          7, 11, 4, 2, 5, 7, 10, 1, 9, 3, 5, 2, 5, 7, 7, 8, 6, 2, 11, 4, 8, 6, 8, 6, 2, 9, 6, 4, 3, 5,
                          4, 4, 0, 4, 0, 6, 4, 8, 6, 4, 2, 11, 7, 3, 3, 4, 1, 7, 2, 7, 7, 6, 4, 4, 4, 2, 6, 6, 4, 2, 6,
                          2, 2, 4, 6, 2, 9, 7, 0, 6, 6, 6, 4, 7, 6, 7, 4, 6, 4, 4, 9, 4, 6, 4, 7, 9, 0, 7, 0, 9, 7, 6,
                          4, 9, 5, 7, 5, 3, 7, 4, 4, 2, 6, 9, 6, 6, 6, 2, 8],
                         [6, 3, 6, 4, 6, 4, 8, 0, 8, 2, 1, 7, 5, 4, 7, 5, 6, 4, 3, 6, 6, 5, 6, 2, 5, 7, 4, 4, 5, 7, 1,
                          1, 4, 8, 6, 4, 5, 4, 7, 5, 6, 9, 7, 5, 2, 9, 7, 7, 7, 5, 10, 5, 5, 7, 5, 3, 3, 7, 6, 4, 10,
                          13, 10, 9, 7, 1, 5, 4, 1, 4, 11, 4, 4, 6, 8, 8, 6, 8, 8, 8, 13, 7, 10, 7, 10, 9, 10, 8, 6, 7,
                          7, 10, 8, 7, 7, 7, 5, 2, 10, 2, 5, 4, 9, 1, 4, 7, 3, 7, 9, 8, 5, 7, 5, 3, 9, 5, 8, 5, 2, 6, 8,
                          8, 1, 12, 9, 0, 2, 6, 6, 2, 8, 8, 6, 4, 3, 5, 9, 2, 2, 6, 6, 6, 4, 6, 9, 4, 6, 4, 8, 8, 6, 4,
                          4, 2, 4, 4, 4, 4, 7, 2, 4, 8, 6, 7, 2, 6, 4, 6, 7, 4, 7, 2, 6, 7, 4, 6, 7, 0, 2, 7, 4, 6, 0,
                          0, 7, 9, 5, 5, 1, 1, 3, 8, 4, 2, 8, 9, 6, 6, 8, 4, 4],
                         [1, 2, 4, 8, 6, 4, 6, 10, 2, 2, 3, 9, 5, 4, 9, 2, 2, 6, 3, 5, 8, 9, 2, 2, 5, 2, 7, 4, 4, 9, 1,
                          1, 2, 4, 4, 4, 5, 9, 5, 9, 11, 8, 10, 5, 5, 5, 5, 4, 7, 11, 7, 4, 7, 5, 5, 2, 9, 3, 3, 8, 9,
                          10, 8, 10, 6, 1, 7, 4, 2, 2, 6, 2, 2, 10, 4, 2, 8, 1, 8, 7, 13, 11, 8, 11, 10, 5, 9, 6, 10, 7,
                          5, 10, 5, 9, 5, 7, 4, 7, 4, 2, 7, 7, 5, 2, 8, 4, 2, 3, 8, 5, 7, 3, 5, 3, 7, 3, 4, 1, 6, 4, 6,
                          4, 6, 4, 8, 8, 6, 2, 2, 2, 2, 4, 6, 7, 5, 3, 5, 7, 1, 4, 7, 2, 6, 6, 2, 4, 1, 8, 2, 10, 6, 4,
                          9, 6, 9, 6, 3, 3, 5, 2, 2, 6, 6, 6, 7, 4, 0, 6, 7, 4, 2, 6, 10, 7, 9, 0, 9, 3, 1, 2, 9, 6, 2,
                          6, 5, 5, 5, 7, 7, 4, 3, 4, 9, 4, 6, 6, 2, 8, 6, 8, 8],
                         [4, 2, 4, 4, 6, 2, 8, 8, 4, 4, 5, 4, 5, 2, 8, 8, 8, 2, 4, 11, 8, 6, 2, 4, 4, 4, 2, 4, 7, 5, 8,
                          4, 1, 2, 2, 8, 6, 1, 3, 7, 6, 6, 11, 4, 3, 1, 4, 5, 3, 4, 2, 2, 7, 4, 4, 2, 5, 3, 8, 8, 9, 8,
                          10, 8, 12, 5, 2, 4, 2, 4, 4, 0, 2, 4, 8, 2, 4, 11, 11, 14, 14, 11, 13, 2, 6, 11, 9, 8, 3, 7,
                          2, 9, 4, 4, 2, 4, 4, 7, 4, 4, 7, 4, 6, 4, 4, 1, 5, 2, 7, 2, 3, 3, 5, 3, 5, 2, 5, 7, 6, 10, 4,
                          2, 2, 6, 4, 7, 2, 2, 6, 10, 10, 10, 2, 11, 7, 3, 3, 7, 6, 6, 2, 9, 7, 6, 9, 6, 4, 9, 5, 3, 1,
                          4, 2, 4, 2, 6, 6, 3, 7, 0, 6, 4, 9, 8, 6, 7, 4, 6, 4, 2, 7, 6, 5, 9, 4, 3, 5, 5, 9, 7, 9, 4,
                          0, 6, 5, 5, 7, 5, 4, 4, 6, 4, 8, 6, 4, 8, 2, 4, 4, 3, 5],
                         [2, 4, 2, 1, 2, 2, 8, 9, 2, 4, 4, 7, 9, 4, 4, 2, 4, 0, 4, 6, 8, 10, 8, 1, 9, 4, 7, 5, 2, 0, 2,
                          6, 6, 6, 0, 4, 7, 2, 4, 7, 4, 8, 6, 5, 7, 4, 5, 3, 4, 2, 6, 4, 6, 9, 2, 4, 7, 5, 6, 7, 13, 10,
                          12, 9, 7, 3, 6, 6, 2, 2, 4, 6, 6, 6, 6, 4, 11, 12, 16, 11, 15, 7, 11, 8, 7, 7, 8, 3, 7, 2, 5,
                          5, 7, 9, 5, 5, 7, 5, 5, 4, 7, 2, 4, 4, 6, 7, 2, 4, 5, 7, 5, 8, 3, 5, 4, 4, 5, 9, 1, 4, 4, 9,
                          2, 4, 4, 5, 5, 2, 4, 2, 4, 8, 4, 5, 5, 3, 7, 9, 4, 4, 4, 4, 4, 4, 1, 4, 5, 5, 7, 5, 7, 4, 11,
                          4, 9, 3, 1, 1, 5, 9, 9, 8, 8, 2, 6, 2, 6, 9, 9, 9, 7, 0, 7, 0, 3, 3, 3, 3, 3, 5, 5, 0, 9, 7,
                          5, 3, 3, 3, 6, 4, 2, 2, 6, 6, 6, 8, 10, 6, 8, 7, 1],
                         [2, 9, 6, 4, 9, 4, 4, 9, 1, 4, 2, 8, 4, 8, 6, 8, 6, 6, 8, 8, 6, 4, 2, 2, 4, 4, 2, 7, 4, 8, 7,
                          3, 8, 5, 2, 4, 7, 2, 7, 11, 10, 4, 6, 4, 4, 7, 6, 8, 6, 6, 4, 6, 2, 6, 6, 6, 7, 3, 6, 10, 10,
                          8, 5, 7, 7, 9, 2, 7, 6, 4, 4, 6, 4, 1, 4, 10, 6, 12, 14, 7, 9, 12, 8, 3, 10, 6, 3, 2, 2, 5, 5,
                          5, 5, 5, 5, 5, 7, 2, 1, 7, 7, 8, 8, 5, 1, 2, 9, 4, 2, 7, 9, 9, 7, 7, 4, 5, 2, 4, 4, 5, 2, 5,
                          4, 1, 9, 7, 4, 10, 4, 6, 2, 1, 3, 5, 7, 5, 3, 1, 4, 8, 7, 9, 7, 1, 6, 1, 5, 3, 0, 3, 3, 5, 1,
                          4, 11, 5, 5, 5, 1, 5, 3, 0, 8, 9, 0, 8, 4, 4, 6, 4, 10, 6, 4, 3, 7, 3, 5, 1, 5, 3, 1, 3, 1, 3,
                          4, 3, 1, 1, 11, 6, 10, 8, 10, 4, 6, 6, 10, 1, 3, 3, 3],
                         [6, 5, 7, 3, 8, 5, 3, 9, 7, 1, 8, 6, 4, 4, 2, 6, 2, 2, 7, 10, 4, 4, 6, 4, 10, 10, 1, 6, 6, 4,
                          10, 5, 5, 11, 4, 4, 9, 8, 5, 7, 14, 10, 7, 9, 5, 5, 2, 4, 6, 2, 6, 4, 8, 4, 4, 6, 10, 1, 11,
                          4, 6, 9, 7, 4, 10, 3, 2, 6, 6, 7, 7, 6, 4, 4, 7, 14, 18, 14, 20, 13, 7, 7, 5, 9, 4, 3, 4, 4,
                          5, 2, 1, 4, 2, 5, 5, 7, 4, 9, 7, 5, 3, 7, 7, 3, 5, 5, 9, 4, 7, 9, 4, 5, 5, 5, 4, 9, 5, 5, 4,
                          9, 4, 5, 2, 5, 2, 7, 4, 10, 1, 5, 5, 4, 4, 4, 6, 9, 8, 2, 10, 1, 3, 7, 7, 5, 10, 5, 5, 9, 4,
                          8, 2, 8, 0, 8, 6, 5, 1, 7, 1, 9, 1, 5, 8, 6, 4, 9, 8, 4, 6, 3, 3, 5, 4, 6, 9, 9, 7, 5, 5, 5,
                          7, 5, 5, 3, 3, 0, 1, 7, 5, 6, 3, 4, 8, 5, 1, 9, 1, 5, 3, 1, 5],
                         [5, 7, 3, 1, 7, 7, 5, 3, 8, 3, 6, 2, 6, 4, 8, 4, 8, 6, 7, 3, 2, 6, 10, 1, 8, 2, 2, 6, 6, 4, 10,
                          4, 8, 7, 3, 5, 3, 3, 1, 5, 8, 7, 14, 12, 6, 4, 9, 2, 6, 6, 4, 6, 4, 8, 6, 2, 3, 6, 12, 9, 8,
                          11, 5, 2, 6, 7, 5, 1, 4, 4, 7, 9, 3, 9, 9, 17, 18, 16, 16, 13, 4, 4, 2, 0, 5, 7, 2, 9, 4, 7,
                          5, 4, 5, 5, 4, 4, 4, 8, 5, 5, 8, 5, 5, 3, 3, 2, 7, 7, 4, 4, 7, 5, 7, 2, 7, 5, 4, 4, 10, 4, 5,
                          2, 4, 9, 7, 7, 4, 4, 4, 7, 10, 4, 8, 2, 8, 3, 5, 9, 5, 3, 3, 7, 5, 7, 3, 1, 5, 7, 10, 8, 4,
                          10, 6, 10, 8, 1, 3, 3, 5, 5, 1, 5, 1, 4, 8, 12, 6, 8, 1, 7, 7, 5, 5, 4, 12, 1, 5, 7, 9, 1, 7,
                          7, 5, 7, 1, 7, 3, 3, 5, 5, 4, 3, 5, 9, 3, 7, 7, 1, 7, 5, 3],
                         [5, 5, 8, 7, 7, 5, 3, 7, 5, 5, 5, 2, 8, 4, 6, 4, 4, 3, 8, 7, 3, 2, 10, 2, 8, 10, 6, 6, 6, 8,
                          10, 6, 5, 4, 5, 5, 7, 3, 6, 8, 8, 14, 15, 13, 8, 6, 4, 4, 6, 2, 2, 6, 2, 9, 2, 5, 9, 4, 6, 11,
                          12, 9, 11, 8, 4, 3, 7, 5, 7, 3, 7, 2, 3, 7, 13, 10, 20, 13, 7, 12, 7, 3, 5, 7, 7, 4, 7, 5, 2,
                          9, 5, 8, 9, 2, 2, 4, 8, 4, 5, 7, 8, 3, 5, 5, 7, 7, 5, 5, 5, 7, 4, 7, 9, 4, 4, 2, 7, 5, 4, 5,
                          4, 4, 5, 2, 6, 4, 7, 7, 7, 6, 6, 6, 2, 4, 0, 5, 3, 1, 5, 7, 5, 5, 7, 3, 5, 5, 5, 2, 6, 6, 6,
                          1, 8, 8, 3, 5, 5, 1, 5, 5, 9, 3, 3, 10, 7, 6, 0, 6, 1, 9, 5, 7, 7, 5, 6, 3, 3, 5, 5, 3, 7, 1,
                          7, 5, 9, 3, 5, 5, 3, 1, 7, 3, 7, 9, 5, 5, 3, 5, 3, 1, 5],
                         [5, 5, 5, 8, 8, 5, 1, 3, 7, 3, 3, 7, 2, 4, 4, 6, 1, 7, 8, 5, 5, 2, 1, 4, 6, 10, 9, 5, 5, 6, 6,
                          4, 0, 6, 2, 2, 5, 4, 2, 6, 8, 6, 14, 11, 12, 8, 7, 6, 7, 6, 4, 4, 2, 7, 2, 7, 7, 5, 6, 10, 12,
                          12, 9, 4, 4, 5, 7, 3, 5, 7, 9, 6, 7, 13, 19, 15, 18, 15, 10, 7, 6, 9, 5, 5, 2, 9, 7, 4, 7, 7,
                          7, 0, 1, 4, 2, 6, 4, 8, 7, 4, 5, 8, 7, 10, 8, 5, 2, 9, 5, 7, 4, 9, 7, 2, 2, 7, 2, 4, 5, 7, 7,
                          5, 7, 7, 4, 7, 2, 6, 0, 8, 8, 4, 6, 1, 0, 3, 5, 5, 7, 7, 3, 9, 7, 5, 5, 7, 0, 0, 8, 8, 11, 5,
                          3, 3, 10, 10, 9, 1, 3, 7, 9, 7, 5, 3, 8, 8, 8, 0, 7, 3, 5, 7, 5, 5, 11, 4, 4, 3, 4, 10, 3, 9,
                          1, 1, 7, 5, 3, 7, 7, 9, 7, 7, 3, 7, 7, 5, 9, 9, 3, 5, 1],
                         [5, 7, 7, 3, 7, 3, 7, 7, 5, 3, 8, 7, 1, 8, 5, 8, 1, 7, 9, 0, 5, 8, 6, 6, 5, 7, 5, 5, 1, 7, 10,
                          9, 4, 2, 2, 4, 10, 8, 2, 8, 6, 12, 7, 11, 13, 6, 5, 5, 1, 7, 6, 4, 11, 2, 2, 7, 8, 3, 3, 10,
                          10, 9, 9, 8, 10, 10, 5, 5, 9, 3, 4, 6, 12, 14, 15, 17, 18, 14, 9, 6, 3, 3, 9, 6, 9, 4, 9, 2,
                          2, 9, 10, 9, 9, 2, 6, 6, 8, 4, 4, 5, 10, 3, 5, 5, 5, 2, 4, 5, 5, 5, 8, 2, 8, 4, 5, 5, 0, 4, 9,
                          9, 6, 12, 8, 2, 2, 2, 6, 6, 9, 7, 3, 5, 1, 9, 7, 4, 4, 6, 1, 3, 9, 7, 5, 3, 5, 1, 7, 3, 9, 7,
                          1, 1, 8, 5, 6, 5, 1, 7, 7, 1, 5, 3, 6, 6, 8, 3, 9, 3, 3, 7, 3, 1, 5, 1, 9, 4, 6, 1, 4, 4, 10,
                          5, 3, 5, 7, 1, 5, 3, 3, 5, 7, 1, 5, 7, 9, 0, 1, 9, 3, 5, 4],
                         [5, 7, 0, 3, 8, 2, 7, 5, 1, 4, 1, 7, 5, 3, 3, 1, 7, 5, 7, 4, 6, 11, 6, 7, 9, 7, 7, 7, 3, 7, 9,
                          5, 4, 8, 8, 4, 4, 4, 6, 5, 6, 6, 11, 15, 11, 6, 5, 5, 2, 7, 3, 1, 1, 4, 8, 2, 6, 2, 7, 7, 8,
                          9, 9, 14, 10, 8, 4, 3, 3, 4, 7, 5, 13, 18, 17, 20, 14, 13, 6, 3, 3, 5, 3, 6, 6, 6, 6, 5, 4, 4,
                          0, 3, 7, 4, 4, 8, 2, 6, 4, 2, 5, 5, 5, 7, 1, 4, 5, 4, 5, 2, 8, 8, 8, 8, 4, 2, 4, 10, 6, 6, 9,
                          6, 6, 6, 6, 2, 2, 6, 4, 7, 5, 7, 5, 3, 3, 8, 6, 6, 6, 5, 7, 5, 7, 3, 5, 3, 3, 9, 7, 1, 10, 12,
                          0, 1, 8, 5, 5, 3, 3, 1, 5, 1, 5, 0, 7, 7, 9, 9, 1, 7, 3, 9, 9, 3, 3, 7, 10, 7, 7, 5, 3, 3, 5,
                          5, 7, 7, 5, 12, 5, 3, 3, 0, 6, 6, 10, 1, 7, 5, 2, 0, 2],
                         [5, 2, 4, 5, 5, 2, 5, 7, 7, 4, 5, 3, 3, 7, 9, 3, 1, 7, 3, 0, 1, 4, 4, 4, 8, 0, 6, 4, 0, 5, 7,
                          5, 6, 2, 8, 6, 4, 6, 7, 4, 3, 8, 13, 14, 14, 14, 10, 5, 4, 1, 5, 8, 8, 2, 9, 8, 4, 6, 7, 5, 3,
                          13, 7, 12, 7, 8, 4, 8, 3, 5, 4, 11, 17, 19, 19, 15, 13, 8, 6, 7, 9, 3, 6, 9, 2, 9, 6, 1, 9, 1,
                          6, 9, 7, 5, 8, 4, 8, 6, 8, 1, 2, 6, 6, 4, 2, 2, 9, 4, 8, 2, 6, 6, 6, 2, 3, 6, 2, 4, 6, 2, 1,
                          8, 6, 8, 4, 4, 6, 9, 8, 5, 1, 7, 1, 6, 3, 8, 6, 6, 3, 3, 5, 7, 5, 1, 5, 1, 1, 3, 6, 5, 6, 5,
                          5, 5, 10, 3, 3, 8, 10, 1, 6, 2, 6, 10, 3, 5, 9, 3, 5, 1, 5, 3, 3, 1, 5, 5, 5, 5, 3, 1, 3, 1,
                          5, 9, 3, 3, 1, 7, 5, 3, 5, 9, 4, 10, 9, 5, 9, 9, 7, 0, 7],
                         [9, 9, 10, 1, 7, 4, 4, 2, 5, 5, 4, 9, 1, 9, 5, 1, 1, 5, 5, 9, 5, 7, 6, 4, 4, 6, 2, 6, 4, 9, 3,
                          1, 7, 8, 8, 4, 6, 6, 1, 6, 6, 7, 7, 9, 14, 12, 8, 7, 2, 8, 0, 4, 6, 8, 2, 9, 6, 6, 8, 5, 6,
                          11, 14, 10, 9, 1, 6, 6, 6, 11, 6, 9, 18, 17, 21, 19, 9, 6, 10, 9, 5, 2, 6, 4, 4, 6, 6, 9, 1,
                          9, 8, 2, 5, 1, 9, 8, 8, 2, 4, 2, 2, 2, 6, 6, 6, 4, 6, 2, 2, 8, 4, 8, 2, 1, 7, 4, 1, 6, 6, 7,
                          5, 1, 1, 6, 4, 4, 4, 8, 3, 7, 5, 3, 3, 6, 3, 4, 3, 3, 6, 3, 9, 9, 9, 9, 7, 12, 8, 5, 6, 5, 3,
                          1, 6, 3, 5, 1, 3, 5, 6, 5, 5, 3, 3, 3, 3, 6, 1, 3, 9, 3, 7, 5, 3, 0, 7, 7, 9, 7, 9, 3, 7, 0,
                          3, 3, 5, 5, 5, 5, 9, 5, 7, 10, 3, 5, 5, 7, 9, 9, 3, 5, 4],
                         [9, 5, 3, 2, 5, 4, 7, 7, 7, 7, 4, 6, 9, 7, 7, 3, 5, 3, 5, 9, 4, 4, 4, 8, 6, 4, 4, 6, 4, 6, 9,
                          5, 5, 1, 6, 6, 6, 3, 5, 7, 7, 7, 7, 16, 16, 16, 11, 4, 4, 3, 1, 4, 4, 8, 1, 4, 6, 4, 6, 2, 11,
                          11, 10, 10, 9, 5, 9, 3, 1, 4, 5, 13, 16, 19, 19, 21, 11, 6, 6, 9, 4, 4, 4, 2, 6, 4, 9, 5, 5,
                          3, 1, 10, 2, 5, 5, 3, 11, 1, 8, 2, 6, 4, 8, 6, 8, 10, 4, 9, 4, 8, 2, 8, 4, 5, 5, 4, 6, 10, 9,
                          5, 9, 5, 5, 9, 8, 4, 4, 5, 5, 5, 7, 5, 1, 8, 6, 11, 4, 8, 6, 6, 9, 5, 7, 5, 7, 5, 7, 6, 3, 1,
                          5, 7, 3, 5, 5, 3, 0, 6, 7, 9, 12, 8, 12, 9, 7, 5, 5, 7, 1, 9, 1, 1, 3, 6, 6, 5, 3, 1, 5, 7, 4,
                          2, 9, 3, 5, 3, 3, 5, 5, 5, 7, 5, 9, 9, 7, 3, 7, 9, 1, 9, 4],
                         [7, 7, 9, 4, 4, 7, 4, 4, 4, 7, 4, 2, 2, 3, 3, 5, 3, 7, 7, 1, 9, 2, 6, 6, 6, 8, 2, 4, 6, 2, 6,
                          5, 5, 9, 3, 9, 5, 7, 5, 2, 4, 8, 9, 12, 14, 7, 15, 9, 5, 4, 7, 6, 2, 8, 6, 8, 6, 1, 6, 4, 11,
                          10, 10, 14, 8, 11, 7, 9, 3, 7, 9, 8, 17, 23, 24, 14, 2, 6, 1, 4, 9, 6, 6, 2, 6, 5, 9, 7, 5, 9,
                          5, 7, 1, 6, 2, 2, 6, 2, 4, 10, 4, 8, 6, 10, 1, 10, 6, 2, 6, 2, 6, 8, 7, 3, 10, 7, 5, 5, 0, 3,
                          5, 3, 3, 9, 5, 7, 1, 7, 7, 7, 3, 9, 3, 9, 8, 8, 4, 2, 11, 5, 7, 3, 1, 3, 5, 3, 9, 1, 3, 3, 7,
                          7, 3, 0, 1, 7, 4, 5, 10, 12, 12, 13, 14, 14, 6, 12, 12, 7, 5, 1, 7, 9, 0, 2, 8, 2, 8, 0, 1, 5,
                          9, 7, 4, 5, 1, 9, 5, 7, 9, 5, 9, 7, 2, 5, 7, 7, 9, 9, 3, 3, 6],
                         [3, 7, 6, 6, 6, 4, 6, 2, 6, 4, 6, 6, 6, 9, 9, 7, 5, 7, 5, 6, 2, 2, 7, 9, 6, 4, 6, 4, 6, 4, 4,
                          3, 5, 10, 3, 3, 9, 1, 5, 4, 2, 8, 4, 8, 11, 14, 17, 12, 7, 1, 3, 12, 6, 6, 6, 1, 6, 2, 11, 4,
                          8, 6, 8, 12, 14, 14, 11, 8, 7, 6, 11, 13, 19, 19, 14, 9, 6, 11, 12, 3, 6, 2, 6, 1, 5, 3, 3, 5,
                          7, 7, 3, 7, 5, 1, 8, 4, 1, 6, 10, 4, 6, 6, 6, 4, 4, 10, 7, 1, 6, 4, 2, 0, 3, 5, 5, 1, 1, 6, 4,
                          7, 1, 7, 3, 3, 3, 5, 3, 7, 1, 7, 5, 3, 5, 7, 0, 6, 3, 3, 3, 5, 5, 5, 5, 3, 9, 5, 9, 5, 9, 3,
                          3, 0, 3, 5, 7, 7, 11, 5, 5, 12, 12, 17, 18, 16, 9, 13, 15, 12, 11, 8, 8, 5, 5, 1, 4, 6, 4, 3,
                          5, 5, 9, 3, 5, 0, 5, 5, 3, 3, 9, 7, 3, 9, 4, 9, 5, 5, 7, 7, 3, 11, 2],
                         [5, 1, 0, 7, 4, 7, 6, 7, 7, 6, 7, 2, 6, 7, 5, 5, 5, 7, 6, 8, 3, 2, 4, 6, 2, 6, 6, 6, 8, 3, 4,
                          3, 6, 4, 1, 9, 9, 9, 7, 3, 6, 9, 8, 11, 12, 17, 18, 13, 9, 7, 6, 4, 4, 4, 2, 4, 10, 8, 6, 8,
                          15, 8, 15, 17, 14, 12, 10, 4, 8, 7, 15, 15, 21, 21, 19, 11, 6, 5, 1, 1, 3, 4, 6, 9, 5, 9, 9,
                          9, 7, 9, 5, 5, 3, 5, 11, 6, 6, 6, 6, 1, 2, 6, 10, 8, 2, 6, 5, 3, 5, 5, 3, 6, 4, 3, 7, 3, 6, 6,
                          6, 7, 3, 3, 7, 5, 7, 5, 7, 3, 4, 3, 5, 5, 5, 6, 11, 6, 7, 5, 1, 3, 1, 9, 5, 5, 5, 7, 9, 1, 3,
                          10, 5, 11, 5, 3, 9, 0, 2, 3, 5, 9, 7, 12, 12, 15, 10, 16, 15, 11, 12, 10, 9, 5, 10, 3, 7, 6,
                          3, 5, 1, 3, 5, 5, 7, 5, 2, 3, 3, 7, 5, 7, 3, 5, 1, 5, 5, 5, 9, 1, 1, 5, 1],
                         [5, 7, 4, 2, 7, 6, 7, 6, 2, 4, 6, 2, 1, 5, 3, 10, 4, 4, 6, 4, 4, 4, 6, 4, 2, 4, 6, 3, 6, 4, 0,
                          4, 6, 8, 8, 10, 5, 7, 5, 3, 0, 9, 3, 9, 16, 16, 18, 18, 15, 3, 5, 3, 8, 6, 1, 3, 5, 3, 6, 10,
                          10, 12, 17, 15, 7, 3, 10, 12, 12, 11, 6, 18, 21, 21, 24, 10, 11, 3, 1, 3, 11, 12, 4, 7, 7, 7,
                          5, 5, 7, 7, 7, 3, 3, 7, 2, 2, 6, 2, 8, 8, 6, 4, 2, 8, 6, 2, 7, 5, 3, 1, 6, 7, 7, 4, 6, 2, 6,
                          4, 1, 7, 7, 7, 5, 5, 5, 5, 5, 10, 4, 10, 5, 3, 2, 11, 4, 6, 8, 5, 5, 7, 0, 7, 5, 7, 9, 5, 7,
                          5, 3, 0, 3, 9, 3, 5, 1, 1, 3, 3, 4, 4, 9, 5, 7, 11, 9, 16, 10, 7, 18, 11, 6, 10, 7, 1, 5, 5,
                          1, 7, 3, 1, 7, 5, 5, 5, 3, 6, 4, 0, 3, 1, 11, 10, 10, 7, 7, 5, 7, 1, 1, 1, 8],
                         [10, 11, 7, 4, 6, 4, 2, 6, 2, 6, 7, 5, 3, 5, 4, 1, 8, 4, 8, 10, 4, 4, 1, 6, 4, 4, 6, 2, 9, 4,
                          6, 1, 6, 4, 10, 3, 3, 8, 4, 1, 6, 2, 7, 6, 13, 12, 18, 21, 21, 15, 3, 3, 10, 3, 3, 3, 1, 5, 9,
                          11, 13, 11, 10, 1, 7, 1, 8, 11, 4, 8, 10, 25, 24, 22, 20, 16, 4, 8, 8, 2, 2, 4, 3, 1, 7, 7, 3,
                          1, 3, 7, 3, 9, 7, 7, 10, 7, 5, 3, 8, 4, 4, 2, 6, 4, 6, 10, 1, 3, 3, 4, 4, 9, 4, 7, 6, 4, 7, 7,
                          3, 3, 3, 9, 0, 5, 10, 6, 6, 4, 5, 0, 6, 2, 0, 8, 6, 4, 4, 6, 4, 6, 8, 10, 7, 1, 5, 9, 3, 5, 0,
                          1, 5, 3, 0, 7, 3, 9, 5, 5, 3, 6, 6, 6, 6, 8, 7, 9, 12, 12, 9, 16, 12, 5, 5, 4, 5, 5, 9, 3, 9,
                          3, 3, 1, 5, 9, 5, 7, 6, 6, 3, 5, 12, 6, 6, 3, 5, 5, 5, 3, 5, 3, 10],
                         [4, 2, 0, 5, 3, 3, 3, 3, 6, 7, 10, 4, 2, 3, 3, 4, 3, 1, 3, 4, 3, 10, 8, 6, 8, 4, 2, 4, 2, 8, 4,
                          6, 0, 3, 8, 1, 6, 1, 3, 4, 2, 8, 7, 6, 9, 13, 14, 21, 21, 17, 8, 4, 7, 4, 10, 1, 0, 11, 7, 13,
                          10, 13, 9, 5, 5, 6, 8, 9, 10, 8, 17, 22, 22, 20, 18, 13, 7, 3, 2, 6, 6, 4, 4, 7, 3, 5, 9, 7,
                          5, 3, 7, 1, 7, 4, 3, 7, 5, 5, 5, 11, 6, 4, 8, 1, 1, 4, 3, 3, 0, 4, 6, 7, 6, 2, 4, 0, 3, 9, 7,
                          1, 5, 5, 3, 9, 7, 4, 10, 3, 0, 6, 4, 8, 4, 0, 10, 4, 6, 6, 4, 10, 6, 2, 3, 5, 5, 7, 7, 4, 10,
                          11, 7, 7, 7, 9, 7, 5, 3, 8, 6, 2, 9, 3, 8, 4, 4, 5, 9, 8, 9, 10, 9, 8, 9, 5, 3, 3, 5, 7, 3, 3,
                          1, 1, 5, 7, 1, 5, 0, 5, 7, 5, 9, 7, 0, 8, 1, 5, 5, 5, 3, 7, 6],
                         [6, 11, 3, 8, 4, 8, 5, 5, 7, 7, 1, 1, 6, 3, 4, 6, 4, 4, 6, 1, 10, 1, 6, 8, 1, 8, 2, 2, 6, 4, 4,
                          6, 0, 5, 3, 3, 8, 6, 1, 0, 6, 2, 9, 5, 9, 8, 13, 17, 19, 17, 17, 6, 12, 5, 3, 8, 3, 5, 16, 14,
                          19, 14, 12, 8, 8, 5, 3, 4, 6, 11, 20, 22, 27, 18, 14, 11, 6, 3, 6, 6, 6, 8, 4, 1, 3, 5, 10, 1,
                          7, 5, 7, 5, 6, 9, 9, 5, 5, 5, 3, 9, 8, 8, 2, 1, 2, 6, 6, 7, 4, 0, 6, 9, 4, 7, 2, 5, 5, 5, 3,
                          9, 7, 9, 5, 3, 4, 9, 6, 2, 4, 10, 2, 4, 4, 8, 2, 4, 6, 2, 8, 6, 4, 8, 6, 7, 7, 7, 8, 6, 10, 0,
                          9, 5, 7, 7, 11, 9, 6, 2, 2, 4, 6, 8, 2, 8, 2, 5, 9, 3, 3, 9, 16, 13, 9, 9, 3, 0, 7, 5, 10, 3,
                          9, 5, 7, 1, 3, 9, 3, 1, 3, 5, 9, 9, 11, 3, 5, 9, 7, 7, 7, 3, 4],
                         [6, 10, 9, 6, 3, 3, 4, 6, 6, 6, 3, 6, 4, 6, 4, 0, 2, 8, 4, 2, 6, 2, 8, 2, 2, 8, 4, 1, 10, 8, 4,
                          4, 3, 0, 1, 1, 3, 8, 9, 6, 8, 8, 3, 10, 3, 8, 11, 16, 17, 23, 15, 11, 10, 4, 8, 5, 5, 12, 10,
                          10, 13, 8, 9, 2, 8, 5, 9, 3, 10, 19, 22, 25, 23, 18, 11, 8, 8, 2, 9, 8, 6, 4, 1, 3, 1, 7, 4,
                          11, 4, 10, 4, 6, 5, 7, 3, 1, 5, 5, 1, 7, 10, 6, 4, 4, 4, 1, 7, 7, 6, 6, 6, 12, 4, 6, 4, 5, 9,
                          1, 1, 5, 7, 7, 9, 3, 1, 5, 10, 8, 8, 0, 6, 4, 2, 4, 10, 8, 4, 8, 4, 6, 10, 6, 4, 8, 0, 6, 2,
                          2, 6, 4, 6, 1, 3, 7, 9, 6, 6, 4, 8, 4, 8, 6, 2, 4, 8, 5, 7, 2, 7, 10, 13, 9, 3, 9, 7, 3, 2, 4,
                          5, 7, 3, 5, 3, 3, 5, 2, 11, 3, 3, 3, 7, 7, 5, 7, 5, 10, 8, 6, 9, 2, 8],
                         [6, 4, 4, 4, 3, 4, 6, 4, 4, 6, 1, 11, 1, 4, 9, 8, 6, 8, 10, 6, 8, 2, 6, 6, 8, 10, 4, 1, 4, 4,
                          1, 0, 3, 6, 3, 3, 4, 4, 8, 3, 4, 3, 4, 7, 5, 3, 6, 12, 12, 19, 17, 21, 11, 9, 7, 5, 5, 11, 11,
                          11, 13, 10, 3, 8, 2, 2, 7, 8, 11, 19, 25, 24, 18, 14, 6, 4, 8, 2, 6, 6, 6, 6, 7, 7, 5, 9, 6,
                          10, 1, 2, 6, 2, 5, 5, 9, 5, 5, 5, 3, 5, 5, 4, 4, 0, 3, 3, 9, 4, 4, 9, 9, 7, 7, 6, 6, 1, 9, 1,
                          5, 3, 5, 5, 3, 7, 11, 5, 8, 4, 6, 4, 8, 8, 8, 8, 8, 8, 0, 2, 0, 5, 7, 5, 6, 6, 10, 8, 6, 6, 8,
                          6, 2, 7, 5, 7, 7, 4, 2, 6, 6, 2, 8, 6, 6, 6, 4, 3, 7, 3, 3, 7, 9, 7, 11, 14, 9, 7, 7, 7, 3, 7,
                          7, 3, 7, 9, 6, 9, 3, 2, 6, 6, 1, 8, 4, 6, 1, 5, 3, 0, 2, 6, 10],
                         [9, 4, 4, 4, 3, 4, 3, 4, 4, 4, 6, 6, 11, 12, 12, 12, 13, 8, 10, 10, 10, 10, 8, 6, 8, 2, 4, 2,
                          6, 3, 11, 6, 4, 4, 4, 10, 6, 8, 4, 4, 4, 4, 3, 1, 5, 3, 3, 8, 8, 15, 21, 25, 18, 20, 11, 8, 6,
                          8, 11, 9, 13, 7, 5, 4, 0, 2, 3, 6, 11, 21, 23, 29, 26, 15, 12, 8, 4, 8, 4, 6, 4, 6, 9, 5, 5,
                          4, 4, 10, 7, 5, 9, 8, 7, 3, 5, 7, 5, 0, 7, 7, 3, 1, 9, 9, 5, 11, 4, 6, 9, 7, 7, 4, 7, 7, 9, 0,
                          5, 5, 7, 9, 3, 7, 3, 3, 7, 8, 0, 7, 7, 5, 0, 6, 4, 6, 8, 6, 4, 8, 3, 5, 5, 3, 3, 0, 5, 7, 9,
                          5, 0, 2, 0, 3, 5, 9, 5, 6, 2, 2, 2, 8, 6, 0, 4, 4, 2, 3, 5, 5, 7, 9, 5, 9, 9, 11, 11, 10, 10,
                          10, 3, 5, 1, 5, 0, 10, 5, 0, 6, 6, 4, 5, 7, 3, 6, 8, 10, 2, 6, 11, 12, 15, 14],
                         [13, 8, 1, 3, 6, 4, 8, 8, 3, 9, 12, 13, 8, 12, 13, 13, 13, 10, 6, 12, 6, 15, 15, 9, 11, 6, 6,
                          2, 2, 5, 1, 1, 6, 3, 4, 6, 3, 3, 6, 6, 8, 6, 4, 7, 8, 3, 5, 10, 4, 10, 12, 14, 15, 20, 11, 19,
                          11, 11, 11, 6, 6, 7, 1, 1, 5, 4, 10, 8, 15, 23, 26, 29, 22, 13, 8, 10, 8, 3, 2, 6, 2, 1, 9, 1,
                          8, 8, 5, 0, 3, 1, 1, 5, 6, 3, 7, 5, 7, 5, 8, 2, 2, 7, 3, 8, 0, 6, 9, 8, 8, 6, 7, 4, 7, 6, 8,
                          6, 7, 7, 5, 5, 1, 5, 5, 9, 0, 4, 5, 3, 7, 5, 7, 9, 3, 4, 0, 6, 10, 5, 3, 7, 9, 7, 5, 3, 5, 1,
                          3, 1, 3, 9, 9, 13, 4, 5, 7, 9, 1, 4, 4, 6, 4, 4, 9, 2, 6, 5, 7, 7, 5, 5, 7, 5, 13, 11, 11, 11,
                          17, 12, 9, 6, 6, 8, 5, 6, 6, 4, 2, 6, 3, 11, 7, 10, 3, 4, 10, 4, 12, 3, 10, 13, 14],
                         [10, 8, 4, 4, 3, 3, 6, 3, 4, 7, 10, 13, 16, 12, 12, 13, 6, 6, 3, 8, 6, 13, 8, 13, 11, 9, 9, 3,
                          11, 2, 8, 3, 3, 8, 1, 6, 8, 4, 6, 6, 4, 4, 4, 7, 10, 10, 10, 2, 6, 10, 12, 15, 16, 14, 15, 23,
                          22, 18, 17, 7, 9, 3, 2, 7, 1, 4, 7, 6, 11, 19, 28, 30, 21, 10, 6, 3, 1, 3, 1, 11, 6, 6, 11, 4,
                          4, 0, 9, 4, 7, 5, 5, 7, 5, 10, 9, 9, 7, 5, 5, 2, 2, 12, 6, 4, 4, 8, 6, 4, 0, 6, 8, 8, 6, 6, 8,
                          6, 3, 5, 3, 11, 5, 7, 9, 0, 4, 1, 7, 7, 10, 7, 5, 9, 3, 1, 4, 10, 3, 3, 3, 3, 5, 3, 9, 9, 3,
                          9, 1, 5, 9, 5, 5, 10, 2, 4, 7, 3, 5, 2, 8, 4, 6, 6, 6, 5, 2, 5, 5, 5, 7, 5, 3, 5, 7, 11, 7,
                          11, 11, 12, 11, 7, 6, 8, 8, 7, 2, 6, 2, 5, 11, 3, 9, 11, 8, 6, 2, 2, 12, 15, 13, 16, 9],
                         [3, 11, 2, 6, 8, 3, 8, 7, 11, 8, 13, 16, 12, 11, 7, 6, 3, 4, 8, 12, 6, 5, 8, 11, 7, 9, 11, 1,
                          10, 4, 2, 10, 5, 7, 9, 3, 8, 6, 4, 6, 6, 3, 5, 1, 5, 6, 6, 4, 3, 1, 9, 11, 7, 12, 18, 24, 26,
                          20, 13, 14, 7, 5, 6, 3, 0, 6, 5, 6, 15, 18, 26, 27, 15, 15, 8, 0, 0, 0, 8, 2, 4, 6, 4, 3, 5,
                          5, 4, 8, 9, 3, 5, 3, 3, 4, 7, 9, 3, 1, 3, 7, 4, 2, 7, 5, 7, 4, 4, 9, 6, 9, 9, 0, 2, 0, 9, 11,
                          3, 4, 3, 4, 3, 5, 3, 4, 8, 3, 3, 3, 7, 1, 5, 5, 3, 5, 3, 4, 11, 5, 1, 5, 1, 5, 1, 7, 5, 9, 7,
                          3, 11, 7, 3, 5, 0, 9, 5, 3, 5, 7, 3, 7, 9, 4, 4, 4, 1, 3, 5, 5, 3, 3, 7, 3, 7, 3, 9, 7, 10, 9,
                          12, 7, 13, 9, 7, 9, 3, 3, 2, 5, 9, 7, 10, 4, 7, 7, 6, 3, 12, 13, 11, 13, 11],
                         [13, 10, 6, 6, 2, 10, 4, 9, 12, 12, 15, 13, 9, 7, 11, 10, 5, 3, 3, 8, 6, 8, 5, 9, 5, 11, 7, 7,
                          10, 8, 6, 4, 6, 5, 7, 5, 7, 6, 5, 6, 6, 7, 3, 10, 5, 0, 0, 6, 11, 3, 4, 7, 10, 8, 14, 19, 23,
                          25, 22, 15, 9, 5, 4, 5, 1, 1, 2, 8, 13, 16, 29, 27, 25, 8, 5, 10, 5, 6, 6, 4, 4, 6, 1, 6, 3,
                          3, 7, 0, 5, 3, 5, 5, 7, 3, 6, 1, 5, 7, 7, 7, 0, 3, 5, 5, 5, 1, 6, 6, 0, 9, 6, 2, 8, 6, 2, 6,
                          0, 4, 4, 6, 1, 3, 10, 2, 8, 7, 7, 7, 13, 5, 3, 3, 5, 7, 7, 4, 9, 5, 3, 3, 3, 5, 9, 5, 5, 1, 5,
                          9, 5, 9, 5, 7, 5, 7, 1, 5, 7, 3, 11, 5, 3, 9, 9, 9, 6, 5, 5, 2, 10, 5, 7, 5, 1, 3, 5, 9, 9,
                          13, 13, 13, 7, 16, 11, 11, 6, 6, 14, 5, 8, 12, 12, 10, 6, 2, 8, 10, 7, 6, 11, 5, 10],
                         [12, 14, 4, 2, 1, 5, 9, 10, 12, 15, 4, 11, 11, 7, 1, 8, 5, 1, 8, 1, 5, 3, 9, 11, 7, 5, 5, 7, 4,
                          4, 6, 4, 10, 5, 1, 3, 0, 0, 1, 3, 5, 3, 1, 5, 5, 3, 1, 4, 2, 6, 2, 6, 5, 10, 8, 10, 19, 21,
                          20, 19, 6, 3, 1, 1, 9, 5, 4, 10, 11, 22, 30, 32, 20, 8, 6, 4, 8, 8, 6, 6, 4, 2, 6, 1, 5, 5, 1,
                          6, 3, 4, 7, 9, 8, 4, 6, 10, 7, 7, 7, 5, 7, 5, 3, 7, 3, 9, 5, 0, 4, 0, 8, 6, 2, 10, 8, 6, 10,
                          7, 1, 10, 1, 6, 6, 8, 8, 2, 1, 3, 3, 1, 3, 1, 5, 7, 3, 0, 0, 7, 9, 5, 12, 9, 7, 3, 9, 9, 11,
                          5, 3, 9, 3, 1, 7, 7, 3, 5, 9, 3, 1, 3, 9, 7, 9, 3, 3, 9, 11, 8, 6, 5, 3, 3, 9, 9, 3, 3, 9, 11,
                          13, 16, 19, 19, 15, 11, 16, 13, 10, 7, 4, 17, 11, 7, 10, 6, 4, 13, 11, 11, 3, 8, 6],
                         [12, 12, 4, 3, 5, 3, 10, 8, 15, 12, 12, 7, 5, 10, 4, 5, 3, 5, 8, 1, 10, 7, 1, 3, 0, 0, 3, 9, 5,
                          4, 2, 5, 9, 3, 7, 5, 6, 1, 7, 3, 3, 9, 12, 1, 7, 3, 4, 8, 2, 4, 6, 10, 10, 5, 6, 14, 14, 19,
                          27, 17, 14, 11, 3, 6, 1, 5, 1, 5, 16, 20, 28, 31, 24, 12, 5, 5, 0, 5, 3, 5, 8, 0, 1, 2, 6, 3,
                          8, 3, 4, 4, 3, 6, 3, 6, 8, 1, 2, 4, 3, 1, 5, 1, 5, 3, 3, 5, 7, 9, 6, 4, 4, 8, 8, 4, 10, 6, 6,
                          5, 9, 10, 8, 0, 6, 6, 4, 2, 2, 3, 3, 3, 9, 7, 7, 5, 0, 4, 6, 5, 3, 13, 7, 5, 3, 3, 5, 7, 3, 5,
                          7, 3, 3, 3, 9, 9, 5, 1, 9, 1, 11, 7, 1, 7, 5, 5, 4, 3, 9, 0, 5, 5, 3, 3, 3, 6, 7, 7, 9, 7, 10,
                          11, 13, 11, 21, 25, 20, 12, 17, 8, 14, 14, 10, 9, 11, 9, 10, 8, 11, 9, 7, 8, 4],
                         [16, 14, 10, 2, 4, 2, 14, 12, 9, 10, 2, 2, 6, 4, 6, 2, 5, 5, 0, 4, 6, 2, 1, 11, 5, 3, 5, 3, 11,
                          3, 2, 1, 6, 3, 5, 5, 0, 7, 5, 3, 9, 1, 5, 5, 7, 3, 8, 2, 6, 4, 4, 6, 2, 4, 9, 14, 5, 15, 22,
                          28, 22, 12, 9, 5, 2, 0, 6, 6, 13, 22, 24, 32, 21, 9, 6, 5, 10, 7, 5, 5, 4, 3, 4, 4, 6, 8, 4,
                          5, 5, 8, 4, 10, 1, 10, 6, 2, 2, 10, 4, 6, 5, 5, 3, 3, 7, 7, 7, 5, 5, 4, 6, 6, 8, 0, 0, 8, 4,
                          3, 6, 6, 4, 8, 8, 0, 9, 3, 7, 0, 7, 7, 5, 9, 1, 4, 0, 4, 6, 7, 9, 5, 7, 7, 3, 9, 5, 3, 5, 3,
                          5, 5, 5, 3, 7, 7, 3, 3, 3, 7, 3, 3, 11, 10, 12, 8, 8, 7, 5, 7, 7, 3, 3, 3, 5, 7, 6, 3, 3, 9,
                          7, 9, 11, 20, 25, 34, 29, 19, 14, 9, 10, 6, 14, 6, 8, 12, 14, 10, 13, 14, 7, 8, 4],
                         [12, 16, 11, 2, 2, 5, 10, 10, 6, 10, 8, 1, 2, 6, 4, 6, 1, 5, 5, 3, 11, 11, 1, 6, 5, 9, 4, 6, 1,
                          7, 5, 1, 11, 3, 4, 7, 7, 3, 5, 3, 3, 7, 2, 7, 7, 0, 4, 8, 6, 10, 8, 4, 4, 4, 6, 4, 6, 15, 18,
                          24, 26, 21, 14, 7, 4, 4, 1, 8, 14, 20, 28, 32, 12, 11, 11, 9, 12, 8, 12, 12, 6, 5, 9, 9, 9, 8,
                          6, 10, 12, 16, 13, 13, 6, 10, 10, 11, 4, 4, 4, 5, 8, 7, 7, 1, 5, 3, 5, 7, 9, 6, 2, 0, 5, 5, 5,
                          3, 11, 0, 10, 8, 6, 3, 9, 3, 5, 3, 1, 1, 5, 9, 4, 0, 2, 2, 0, 8, 8, 0, 11, 15, 7, 11, 1, 5, 3,
                          9, 1, 8, 6, 5, 9, 1, 3, 5, 7, 5, 5, 1, 1, 5, 11, 11, 12, 12, 11, 10, 10, 9, 7, 6, 5, 7, 3, 3,
                          1, 1, 7, 4, 5, 9, 14, 26, 38, 45, 41, 32, 23, 17, 15, 11, 15, 10, 14, 18, 9, 10, 11, 15, 12,
                          14, 13],
                         [11, 13, 16, 12, 5, 11, 7, 8, 8, 5, 2, 1, 1, 3, 5, 7, 9, 9, 13, 16, 19, 20, 22, 19, 15, 13, 8,
                          11, 6, 10, 7, 4, 5, 5, 6, 5, 3, 6, 6, 4, 7, 5, 3, 2, 7, 7, 7, 4, 6, 6, 8, 4, 6, 2, 1, 0, 8, 9,
                          9, 22, 26, 24, 14, 7, 1, 8, 1, 6, 7, 19, 22, 29, 23, 22, 19, 17, 17, 22, 21, 17, 21, 19, 16,
                          14, 16, 21, 19, 15, 15, 21, 17, 22, 19, 14, 12, 15, 11, 10, 8, 4, 5, 5, 7, 10, 6, 12, 5, 9, 7,
                          4, 0, 5, 3, 5, 9, 1, 1, 0, 2, 6, 8, 8, 9, 7, 5, 7, 5, 7, 3, 5, 3, 4, 6, 4, 1, 9, 5, 13, 8, 11,
                          11, 7, 7, 5, 7, 3, 2, 8, 10, 10, 7, 7, 7, 5, 3, 5, 5, 7, 10, 11, 14, 20, 25, 26, 29, 27, 25,
                          14, 5, 13, 9, 6, 7, 3, 7, 3, 5, 8, 4, 9, 16, 34, 57, 62, 65, 47, 35, 27, 20, 22, 20, 16, 7, 9,
                          11, 4, 11, 14, 7, 11, 11],
                         [11, 11, 15, 9, 11, 9, 10, 7, 7, 4, 3, 8, 6, 11, 11, 13, 13, 20, 28, 28, 35, 29, 27, 35, 37,
                          31, 30, 25, 19, 18, 10, 14, 15, 17, 11, 13, 12, 11, 5, 5, 7, 3, 7, 3, 9, 1, 3, 7, 6, 8, 4, 2,
                          2, 0, 12, 6, 7, 3, 12, 18, 16, 22, 18, 20, 7, 0, 3, 3, 11, 18, 23, 43, 40, 34, 25, 26, 24, 26,
                          22, 26, 24, 24, 20, 18, 22, 18, 12, 18, 14, 20, 15, 17, 16, 16, 16, 13, 12, 6, 4, 4, 6, 1, 3,
                          6, 5, 3, 3, 3, 6, 6, 7, 5, 5, 5, 3, 1, 7, 10, 5, 2, 4, 2, 8, 5, 3, 1, 3, 9, 9, 5, 5, 3, 6, 7,
                          3, 1, 1, 9, 12, 15, 9, 9, 5, 1, 3, 6, 6, 6, 10, 12, 12, 9, 9, 5, 5, 8, 3, 3, 8, 10, 19, 21,
                          31, 40, 49, 55, 45, 29, 18, 10, 7, 7, 7, 8, 7, 7, 1, 4, 5, 13, 23, 55, 73, 93, 81, 63, 57, 40,
                          40, 26, 21, 15, 10, 6, 4, 2, 4, 1, 4, 13, 11],
                         [7, 9, 15, 15, 11, 7, 13, 7, 9, 7, 11, 9, 17, 21, 21, 25, 27, 32, 33, 32, 27, 27, 27, 32, 35,
                          25, 35, 35, 37, 29, 29, 31, 27, 28, 25, 27, 24, 22, 24, 19, 16, 11, 10, 3, 7, 5, 5, 1, 7, 3,
                          1, 3, 3, 1, 6, 4, 3, 5, 5, 11, 16, 23, 20, 14, 11, 7, 3, 2, 13, 26, 37, 51, 57, 47, 36, 19,
                          24, 13, 12, 17, 12, 15, 12, 16, 14, 12, 6, 16, 10, 8, 11, 8, 12, 10, 3, 6, 9, 2, 6, 4, 8, 7,
                          5, 7, 3, 6, 1, 0, 6, 8, 7, 3, 5, 5, 5, 4, 6, 4, 1, 8, 2, 2, 8, 2, 7, 5, 3, 5, 7, 3, 3, 3, 9,
                          9, 7, 1, 9, 7, 10, 13, 12, 9, 11, 11, 11, 4, 8, 6, 10, 13, 13, 8, 1, 12, 2, 0, 3, 4, 10, 15,
                          26, 27, 32, 34, 45, 56, 67, 58, 48, 29, 14, 9, 8, 5, 3, 7, 7, 5, 9, 12, 30, 71, 137, 153, 140,
                          121, 77, 72, 56, 40, 25, 16, 8, 6, 4, 2, 1, 0, 10, 2, 7],
                         [6, 12, 13, 19, 15, 9, 14, 14, 12, 15, 19, 27, 34, 22, 35, 35, 35, 35, 31, 19, 20, 19, 15, 17,
                          19, 22, 25, 27, 32, 30, 33, 33, 33, 23, 31, 33, 23, 35, 28, 32, 27, 25, 19, 13, 12, 16, 9, 3,
                          7, 4, 6, 10, 3, 4, 8, 0, 1, 7, 1, 5, 15, 14, 23, 22, 18, 6, 7, 10, 16, 43, 63, 76, 79, 64, 47,
                          27, 15, 10, 19, 11, 8, 9, 6, 5, 6, 8, 4, 3, 7, 5, 8, 8, 14, 9, 5, 6, 3, 10, 8, 2, 1, 3, 0, 10,
                          1, 7, 4, 8, 2, 6, 3, 3, 1, 9, 8, 6, 4, 4, 2, 2, 2, 4, 8, 4, 3, 5, 3, 7, 5, 1, 1, 9, 5, 7, 3,
                          8, 11, 5, 10, 13, 16, 4, 3, 0, 3, 4, 10, 15, 14, 10, 11, 12, 8, 6, 0, 1, 4, 7, 11, 22, 24, 18,
                          12, 17, 22, 41, 57, 69, 62, 47, 28, 16, 12, 7, 10, 10, 12, 16, 22, 22, 53, 117, 176, 176, 177,
                          178, 150, 112, 79, 49, 27, 15, 6, 4, 6, 6, 1, 1, 5, 3, 4],
                         [11, 13, 18, 27, 23, 17, 23, 14, 21, 26, 31, 35, 37, 37, 30, 32, 23, 25, 27, 18, 14, 12, 10, 9,
                          17, 20, 18, 14, 10, 22, 19, 21, 21, 24, 19, 21, 23, 27, 21, 29, 31, 30, 28, 26, 20, 19, 19,
                          10, 12, 9, 8, 8, 5, 4, 4, 5, 5, 7, 8, 9, 12, 12, 20, 25, 20, 19, 20, 24, 51, 67, 89, 94, 86,
                          66, 53, 45, 37, 31, 18, 20, 11, 9, 5, 8, 8, 2, 1, 5, 3, 1, 8, 2, 2, 7, 7, 3, 8, 2, 2, 4, 3, 1,
                          5, 7, 9, 5, 12, 8, 0, 8, 4, 3, 5, 6, 4, 6, 4, 7, 4, 4, 2, 4, 4, 8, 3, 1, 5, 3, 3, 3, 3, 8, 3,
                          3, 1, 3, 3, 5, 12, 18, 14, 15, 8, 1, 0, 0, 1, 13, 15, 16, 12, 5, 6, 6, 2, 4, 4, 11, 21, 24,
                          24, 14, 6, 5, 15, 24, 43, 57, 71, 65, 53, 33, 23, 15, 18, 19, 23, 36, 52, 77, 124, 175, 177,
                          177, 178, 179, 179, 125, 103, 75, 44, 22, 17, 5, 8, 4, 4, 10, 4, 8, 8],
                         [33, 33, 41, 38, 33, 39, 35, 40, 39, 33, 29, 23, 31, 27, 21, 21, 21, 18, 20, 16, 8, 12, 5, 4,
                          14, 7, 6, 14, 14, 17, 15, 18, 14, 14, 16, 12, 12, 16, 16, 22, 23, 21, 19, 24, 16, 15, 19, 14,
                          8, 10, 9, 1, 8, 5, 0, 13, 5, 4, 3, 3, 5, 14, 12, 22, 29, 36, 38, 54, 75, 97, 93, 80, 71, 62,
                          52, 49, 44, 37, 36, 30, 26, 19, 9, 5, 5, 2, 4, 1, 5, 0, 0, 4, 2, 5, 10, 8, 2, 6, 2, 6, 1, 5,
                          7, 5, 7, 5, 7, 1, 4, 6, 6, 8, 2, 4, 4, 6, 7, 5, 6, 8, 2, 4, 4, 11, 3, 11, 4, 4, 8, 1, 0, 6, 0,
                          1, 2, 3, 7, 9, 15, 22, 20, 13, 3, 5, 3, 3, 3, 11, 19, 18, 10, 8, 0, 6, 6, 4, 8, 9, 22, 32, 22,
                          9, 3, 5, 0, 8, 15, 26, 41, 76, 80, 72, 66, 45, 48, 57, 69, 95, 131, 162, 177, 173, 177, 178,
                          179, 180, 125, 84, 79, 79, 64, 55, 33, 22, 19, 14, 10, 18, 11, 18, 7],
                         [53, 61, 72, 68, 62, 54, 49, 42, 41, 29, 21, 13, 18, 18, 12, 13, 11, 13, 12, 10, 12, 9, 8, 6,
                          5, 7, 3, 10, 9, 9, 7, 9, 7, 9, 12, 15, 13, 7, 13, 15, 18, 11, 18, 10, 12, 15, 13, 15, 5, 16,
                          15, 5, 3, 5, 7, 3, 2, 8, 3, 12, 5, 12, 15, 25, 42, 48, 72, 98, 101, 92, 65, 48, 42, 34, 35,
                          31, 36, 35, 30, 22, 28, 33, 26, 23, 22, 17, 16, 8, 16, 9, 6, 1, 5, 0, 2, 1, 7, 3, 3, 6, 1, 3,
                          3, 0, 5, 11, 1, 3, 5, 9, 11, 1, 7, 0, 9, 3, 9, 7, 7, 6, 8, 6, 2, 2, 5, 6, 2, 8, 2, 7, 6, 12,
                          7, 6, 2, 5, 3, 7, 16, 16, 27, 15, 7, 1, 1, 9, 9, 15, 15, 19, 10, 2, 2, 6, 0, 8, 13, 15, 22,
                          30, 19, 8, 0, 0, 0, 3, 6, 12, 33, 61, 104, 124, 139, 146, 148, 161, 176, 176, 177, 177, 176,
                          177, 178, 178, 158, 98, 60, 65, 62, 62, 70, 66, 55, 52, 38, 28, 29, 29, 32, 30, 22],
                         [57, 69, 94, 104, 106, 81, 67, 52, 26, 26, 11, 13, 7, 14, 11, 7, 9, 9, 11, 8, 7, 8, 6, 5, 5, 3,
                          8, 3, 1, 9, 2, 3, 8, 5, 9, 11, 5, 7, 7, 9, 5, 13, 9, 12, 8, 12, 12, 12, 1, 11, 9, 10, 6, 10,
                          6, 0, 2, 2, 6, 6, 3, 8, 17, 24, 43, 62, 103, 126, 119, 88, 49, 32, 21, 20, 17, 25, 27, 26, 25,
                          19, 24, 27, 29, 29, 27, 33, 28, 26, 22, 18, 15, 10, 5, 7, 0, 1, 4, 3, 5, 3, 2, 6, 0, 1, 5, 7,
                          5, 5, 3, 3, 1, 7, 5, 7, 5, 5, 7, 9, 1, 7, 6, 6, 4, 6, 1, 4, 6, 4, 7, 5, 1, 8, 1, 8, 4, 3, 7,
                          11, 18, 22, 21, 15, 3, 1, 3, 5, 1, 11, 17, 22, 14, 3, 4, 1, 5, 0, 3, 14, 21, 31, 22, 13, 4, 0,
                          3, 2, 10, 26, 57, 85, 125, 156, 176, 176, 176, 177, 177, 177, 177, 178, 178, 178, 178, 144,
                          93, 56, 37, 34, 40, 44, 50, 56, 61, 64, 53, 40, 28, 24, 28, 17, 14],
                         [45, 66, 86, 111, 123, 107, 83, 60, 36, 26, 14, 10, 5, 5, 3, 9, 5, 7, 5, 9, 14, 10, 3, 2, 0,
                          10, 1, 1, 1, 6, 4, 8, 4, 6, 1, 3, 4, 6, 4, 6, 4, 7, 3, 2, 4, 10, 6, 10, 10, 3, 7, 4, 4, 8, 4,
                          4, 5, 5, 6, 8, 8, 14, 19, 32, 52, 77, 117, 138, 132, 82, 48, 18, 13, 12, 13, 15, 21, 23, 23,
                          19, 19, 17, 19, 21, 27, 25, 25, 23, 24, 24, 24, 26, 19, 19, 19, 17, 7, 13, 10, 7, 9, 8, 5, 5,
                          5, 3, 3, 3, 3, 1, 7, 9, 3, 7, 5, 9, 7, 7, 7, 0, 5, 1, 5, 9, 5, 1, 9, 9, 0, 5, 3, 12, 5, 2, 4,
                          2, 3, 5, 12, 24, 22, 18, 9, 3, 5, 5, 9, 13, 21, 17, 15, 8, 1, 8, 1, 3, 1, 11, 20, 27, 33, 21,
                          21, 16, 27, 46, 59, 68, 86, 105, 124, 153, 173, 177, 177, 177, 177, 178, 178, 178, 178, 179,
                          166, 102, 72, 34, 26, 16, 14, 27, 38, 48, 57, 58, 54, 36, 18, 19, 10, 12, 5],
                         [34, 44, 66, 85, 99, 95, 80, 56, 33, 20, 8, 10, 12, 7, 5, 7, 5, 9, 5, 2, 12, 6, 12, 10, 7, 6,
                          0, 0, 2, 2, 2, 4, 1, 7, 1, 1, 3, 7, 1, 1, 3, 3, 2, 1, 2, 4, 4, 2, 2, 2, 1, 5, 4, 4, 8, 5, 9,
                          8, 8, 8, 8, 14, 19, 31, 75, 109, 147, 149, 125, 81, 37, 18, 12, 1, 3, 7, 11, 18, 18, 12, 19,
                          10, 9, 20, 18, 18, 20, 15, 22, 23, 26, 27, 23, 29, 19, 28, 26, 26, 19, 13, 17, 14, 9, 7, 3, 3,
                          0, 0, 9, 0, 7, 5, 5, 5, 3, 3, 1, 9, 5, 7, 5, 7, 5, 2, 9, 7, 7, 7, 5, 9, 7, 3, 0, 3, 4, 8, 5,
                          12, 16, 24, 28, 18, 2, 1, 7, 5, 5, 18, 21, 19, 8, 10, 3, 1, 1, 5, 5, 10, 13, 25, 34, 38, 51,
                          64, 71, 79, 68, 75, 79, 86, 90, 106, 113, 118, 127, 160, 178, 178, 178, 179, 179, 166, 115,
                          84, 68, 39, 19, 18, 25, 20, 20, 30, 34, 34, 49, 38, 20, 17, 13, 1, 0],
                         [20, 29, 43, 43, 59, 73, 62, 41, 27, 17, 16, 9, 10, 7, 10, 14, 5, 7, 12, 10, 5, 10, 8, 6, 11,
                          5, 6, 4, 1, 7, 9, 7, 3, 7, 5, 4, 2, 6, 0, 5, 1, 8, 7, 0, 7, 6, 8, 2, 4, 4, 1, 5, 1, 2, 7, 5,
                          9, 10, 12, 11, 16, 20, 44, 59, 110, 147, 174, 164, 119, 57, 32, 16, 7, 0, 2, 2, 6, 13, 15, 19,
                          8, 7, 2, 10, 10, 12, 14, 10, 15, 17, 19, 19, 21, 19, 19, 21, 22, 15, 21, 13, 15, 22, 13, 11,
                          9, 9, 9, 4, 4, 4, 1, 3, 11, 7, 7, 3, 5, 3, 5, 7, 9, 1, 10, 8, 4, 5, 9, 3, 3, 1, 9, 7, 5, 3, 3,
                          4, 2, 7, 14, 18, 28, 23, 12, 8, 0, 0, 6, 8, 21, 19, 3, 8, 1, 6, 8, 3, 5, 7, 11, 18, 32, 47,
                          62, 71, 69, 67, 57, 55, 47, 55, 59, 59, 63, 74, 83, 101, 134, 179, 179, 166, 131, 97, 65, 61,
                          51, 41, 33, 25, 25, 14, 0, 0, 1, 14, 33, 34, 22, 13, 4, 0, 0],
                         [14, 16, 19, 24, 25, 36, 40, 33, 28, 15, 7, 7, 7, 11, 8, 5, 5, 5, 8, 8, 6, 10, 10, 6, 11, 10,
                          10, 5, 6, 4, 1, 6, 6, 6, 11, 12, 0, 2, 8, 4, 0, 4, 5, 1, 5, 0, 4, 6, 2, 2, 2, 2, 3, 7, 10, 13,
                          17, 27, 42, 48, 48, 55, 66, 106, 148, 177, 177, 176, 119, 51, 23, 16, 6, 1, 0, 6, 8, 10, 20,
                          19, 15, 2, 3, 1, 7, 7, 7, 9, 11, 9, 13, 14, 10, 10, 18, 17, 15, 17, 15, 13, 13, 17, 13, 13,
                          11, 11, 11, 6, 8, 2, 4, 7, 3, 7, 7, 5, 5, 9, 1, 5, 1, 2, 4, 4, 6, 10, 0, 5, 3, 0, 0, 1, 5, 3,
                          9, 9, 1, 5, 13, 19, 27, 25, 15, 6, 0, 1, 4, 4, 21, 22, 17, 8, 11, 6, 2, 0, 3, 6, 11, 18, 32,
                          44, 57, 54, 46, 48, 42, 34, 30, 24, 25, 29, 31, 34, 41, 54, 77, 110, 109, 82, 74, 56, 33, 21,
                          28, 24, 29, 27, 14, 5, 0, 0, 0, 0, 11, 32, 29, 13, 0, 0, 0],
                         [14, 17, 14, 17, 18, 27, 29, 28, 19, 14, 14, 5, 5, 8, 4, 7, 5, 8, 8, 6, 6, 4, 8, 6, 13, 10, 8,
                          8, 6, 8, 6, 4, 6, 8, 3, 2, 2, 6, 0, 6, 6, 1, 1, 2, 2, 6, 1, 3, 7, 3, 3, 9, 5, 16, 26, 39, 53,
                          67, 73, 80, 75, 72, 95, 120, 159, 178, 177, 177, 127, 45, 25, 10, 2, 0, 0, 7, 8, 12, 18, 21,
                          16, 11, 3, 3, 7, 8, 9, 5, 5, 10, 5, 7, 9, 11, 10, 10, 10, 13, 14, 14, 10, 10, 14, 12, 12, 12,
                          11, 6, 8, 4, 4, 10, 6, 9, 5, 7, 3, 0, 5, 10, 10, 6, 10, 4, 4, 6, 4, 8, 4, 4, 10, 5, 5, 5, 5,
                          1, 5, 5, 12, 17, 19, 26, 19, 15, 0, 2, 10, 8, 27, 23, 6, 10, 8, 0, 6, 8, 3, 4, 10, 18, 24, 39,
                          39, 40, 29, 33, 24, 18, 18, 20, 13, 15, 20, 13, 24, 31, 50, 64, 72, 53, 44, 20, 8, 0, 0, 6,
                          14, 20, 10, 2, 0, 0, 0, 0, 0, 13, 28, 8, 4, 0, 7],
                         [10, 13, 11, 14, 12, 25, 23, 25, 20, 13, 10, 10, 8, 4, 5, 3, 3, 6, 4, 8, 6, 8, 4, 4, 12, 8, 6,
                          8, 6, 12, 6, 8, 6, 6, 3, 6, 0, 3, 3, 1, 13, 0, 6, 3, 6, 8, 10, 5, 12, 8, 15, 23, 24, 40, 53,
                          64, 72, 77, 71, 68, 61, 56, 65, 98, 139, 178, 178, 177, 132, 62, 35, 13, 6, 2, 0, 0, 9, 21,
                          22, 20, 18, 9, 8, 8, 8, 7, 5, 5, 5, 1, 9, 9, 9, 7, 6, 4, 4, 8, 7, 9, 7, 10, 9, 11, 12, 11, 10,
                          8, 10, 6, 6, 4, 2, 1, 5, 3, 7, 5, 7, 4, 0, 10, 6, 4, 6, 8, 6, 6, 8, 3, 11, 10, 5, 5, 1, 11, 1,
                          7, 10, 19, 27, 31, 25, 13, 10, 4, 6, 8, 18, 25, 19, 7, 4, 8, 8, 0, 1, 7, 10, 23, 31, 31, 31,
                          26, 23, 23, 11, 16, 8, 8, 6, 9, 6, 6, 12, 23, 38, 44, 46, 29, 20, 0, 0, 0, 0, 0, 11, 27, 14,
                          7, 0, 0, 0, 0, 0, 4, 16, 20, 18, 9, 30],
                         [13, 10, 9, 7, 8, 12, 18, 16, 14, 10, 8, 4, 6, 3, 8, 8, 5, 5, 1, 10, 0, 9, 3, 2, 1, 10, 8, 12,
                          6, 6, 8, 3, 6, 6, 8, 1, 5, 7, 3, 1, 7, 7, 5, 9, 12, 10, 14, 12, 24, 30, 38, 51, 61, 64, 70,
                          71, 63, 56, 45, 33, 40, 36, 47, 56, 94, 116, 150, 151, 128, 102, 52, 19, 12, 8, 3, 10, 16, 14,
                          20, 12, 14, 9, 8, 4, 7, 7, 2, 2, 4, 6, 11, 5, 5, 3, 5, 4, 6, 5, 5, 7, 5, 1, 6, 8, 6, 9, 7, 7,
                          3, 5, 9, 10, 3, 8, 8, 4, 6, 1, 4, 10, 1, 5, 5, 7, 6, 8, 0, 6, 6, 1, 10, 5, 3, 8, 3, 8, 5, 2,
                          7, 15, 19, 22, 32, 17, 6, 0, 0, 8, 18, 28, 21, 11, 7, 6, 1, 6, 4, 10, 15, 26, 27, 22, 21, 14,
                          12, 12, 9, 6, 4, 8, 2, 2, 0, 3, 13, 29, 27, 30, 21, 11, 0, 0, 0, 0, 0, 0, 43, 53, 47, 34, 18,
                          21, 23, 16, 19, 42, 42, 57, 53, 52, 50],
                         [8, 11, 8, 8, 11, 10, 23, 19, 14, 10, 4, 4, 6, 11, 8, 2, 5, 3, 1, 6, 9, 7, 5, 4, 4, 1, 3, 1, 5,
                          3, 7, 5, 1, 4, 8, 6, 4, 6, 10, 10, 13, 13, 20, 23, 27, 35, 39, 46, 54, 61, 68, 71, 68, 57, 57,
                          44, 37, 30, 30, 33, 16, 32, 32, 46, 51, 65, 83, 91, 118, 130, 99, 58, 17, 14, 6, 8, 12, 19,
                          18, 12, 10, 4, 4, 10, 7, 5, 4, 8, 8, 2, 3, 5, 1, 2, 4, 11, 1, 2, 6, 8, 5, 1, 5, 7, 1, 7, 7, 5,
                          9, 3, 7, 3, 7, 5, 8, 6, 4, 7, 2, 3, 5, 3, 5, 3, 5, 10, 0, 4, 4, 3, 3, 3, 5, 3, 6, 1, 3, 2, 10,
                          9, 17, 15, 29, 25, 8, 8, 0, 6, 20, 26, 20, 14, 11, 1, 6, 4, 10, 10, 18, 26, 21, 17, 16, 11, 7,
                          12, 3, 9, 4, 2, 6, 2, 10, 6, 26, 29, 29, 20, 4, 6, 0, 0, 0, 0, 4, 47, 91, 106, 105, 102, 86,
                          87, 88, 83, 84, 91, 99, 98, 90, 64, 52],
                         [13, 15, 7, 8, 12, 12, 18, 14, 9, 6, 8, 6, 1, 1, 4, 2, 11, 0, 10, 3, 7, 5, 2, 2, 5, 4, 6, 3, 3,
                          4, 6, 4, 6, 8, 6, 14, 15, 15, 17, 26, 31, 35, 49, 54, 57, 60, 66, 64, 68, 64, 54, 58, 47, 40,
                          32, 30, 26, 25, 18, 20, 12, 20, 23, 30, 33, 36, 51, 61, 86, 107, 115, 98, 65, 37, 19, 12, 7,
                          13, 8, 15, 10, 5, 14, 13, 5, 7, 2, 10, 6, 2, 0, 3, 7, 0, 3, 1, 7, 3, 5, 11, 1, 3, 5, 6, 7, 5,
                          1, 5, 5, 5, 5, 5, 7, 9, 9, 7, 3, 3, 4, 5, 9, 9, 1, 7, 9, 5, 6, 0, 4, 8, 5, 3, 5, 3, 8, 3, 6,
                          10, 6, 2, 13, 20, 27, 30, 15, 11, 3, 7, 15, 27, 23, 18, 4, 4, 6, 9, 8, 14, 19, 20, 16, 16, 9,
                          6, 5, 6, 9, 5, 9, 7, 1, 6, 10, 18, 28, 22, 17, 9, 1, 0, 0, 5, 5, 25, 76, 120, 136, 137, 133,
                          121, 104, 103, 100, 104, 89, 96, 112, 111, 99, 80, 61],
                         [9, 6, 7, 7, 20, 20, 19, 17, 11, 6, 4, 1, 6, 8, 4, 1, 5, 7, 0, 5, 7, 2, 10, 8, 8, 5, 4, 6, 4,
                          7, 11, 11, 17, 14, 24, 29, 37, 42, 42, 48, 55, 61, 59, 57, 47, 51, 53, 46, 41, 36, 29, 33, 26,
                          20, 20, 21, 13, 12, 11, 11, 12, 13, 18, 21, 24, 25, 25, 36, 54, 78, 97, 113, 105, 75, 47, 25,
                          18, 10, 12, 10, 14, 10, 5, 1, 7, 3, 3, 4, 8, 1, 5, 7, 5, 8, 3, 7, 1, 5, 1, 9, 3, 9, 4, 9, 3,
                          7, 7, 4, 6, 9, 7, 1, 7, 7, 7, 5, 7, 7, 2, 1, 3, 9, 5, 3, 5, 5, 8, 8, 2, 4, 0, 10, 8, 3, 1, 6,
                          6, 3, 11, 6, 8, 15, 25, 27, 23, 24, 18, 11, 23, 28, 30, 21, 5, 6, 6, 6, 4, 12, 10, 14, 9, 14,
                          14, 6, 7, 6, 3, 3, 3, 3, 6, 4, 16, 22, 22, 14, 3, 0, 1, 5, 6, 30, 59, 89, 120, 130, 121, 105,
                          89, 77, 68, 69, 67, 60, 69, 69, 70, 82, 87, 75, 71],
                         [11, 11, 13, 7, 14, 15, 15, 13, 10, 4, 3, 3, 0, 6, 4, 4, 2, 4, 2, 2, 2, 4, 8, 7, 13, 16, 8, 18,
                          18, 17, 22, 18, 29, 36, 46, 50, 51, 52, 50, 52, 51, 43, 41, 38, 30, 32, 29, 31, 27, 22, 26,
                          23, 16, 14, 16, 11, 9, 6, 5, 6, 4, 12, 13, 20, 18, 14, 16, 19, 34, 48, 71, 96, 108, 107, 70,
                          47, 26, 11, 16, 10, 12, 5, 7, 8, 8, 6, 5, 3, 8, 1, 3, 3, 6, 7, 6, 8, 6, 5, 1, 3, 1, 5, 9, 7,
                          3, 5, 7, 3, 6, 5, 7, 3, 1, 7, 7, 9, 11, 9, 4, 6, 9, 1, 5, 7, 3, 9, 2, 7, 2, 4, 5, 0, 1, 3, 6,
                          6, 6, 8, 8, 1, 8, 13, 16, 21, 25, 23, 30, 19, 25, 34, 34, 26, 9, 4, 0, 2, 8, 12, 18, 15, 11,
                          11, 12, 8, 7, 6, 3, 3, 3, 3, 6, 14, 18, 19, 15, 8, 5, 0, 0, 9, 20, 50, 81, 106, 119, 95, 74,
                          62, 50, 44, 42, 44, 45, 43, 31, 42, 46, 46, 54, 55, 56],
                         [9, 11, 8, 6, 16, 15, 20, 17, 7, 1, 6, 10, 8, 8, 9, 3, 4, 6, 6, 2, 5, 9, 16, 19, 20, 27, 33,
                          33, 35, 35, 41, 36, 43, 41, 39, 50, 38, 34, 33, 33, 29, 29, 23, 17, 19, 22, 21, 20, 17, 17,
                          12, 12, 11, 9, 7, 6, 10, 6, 4, 5, 13, 14, 15, 18, 11, 5, 11, 13, 18, 27, 41, 64, 86, 105, 100,
                          83, 47, 25, 17, 10, 7, 8, 7, 9, 5, 4, 10, 3, 2, 2, 3, 11, 4, 9, 9, 2, 6, 2, 9, 3, 5, 0, 5, 5,
                          5, 9, 9, 5, 5, 5, 5, 7, 5, 3, 3, 3, 9, 5, 7, 6, 2, 7, 3, 3, 1, 12, 1, 4, 4, 4, 6, 1, 6, 6, 6,
                          3, 1, 10, 1, 6, 11, 3, 13, 14, 18, 30, 25, 33, 41, 37, 43, 34, 14, 11, 5, 5, 2, 7, 13, 11, 9,
                          15, 14, 10, 9, 4, 4, 6, 3, 10, 10, 17, 15, 12, 11, 6, 1, 1, 2, 16, 39, 65, 82, 94, 89, 71, 43,
                          34, 31, 30, 22, 25, 28, 23, 18, 21, 17, 24, 28, 32, 32],
                         [5, 8, 11, 13, 19, 18, 20, 11, 6, 6, 6, 8, 8, 5, 9, 9, 11, 11, 11, 13, 16, 25, 28, 32, 40, 40,
                          42, 34, 38, 37, 37, 35, 32, 34, 28, 28, 24, 23, 23, 20, 22, 13, 23, 15, 10, 8, 11, 10, 10, 7,
                          11, 5, 11, 5, 4, 2, 8, 7, 3, 6, 9, 13, 13, 6, 14, 5, 9, 9, 11, 13, 25, 42, 46, 78, 96, 99, 77,
                          50, 28, 12, 10, 4, 6, 9, 5, 5, 4, 5, 7, 4, 1, 8, 6, 7, 7, 1, 6, 4, 6, 1, 6, 2, 8, 5, 5, 5, 9,
                          1, 4, 6, 4, 0, 5, 1, 3, 5, 11, 9, 1, 5, 6, 1, 3, 5, 8, 1, 2, 6, 2, 4, 6, 4, 10, 6, 0, 6, 6, 6,
                          4, 6, 6, 1, 9, 11, 13, 13, 17, 21, 34, 40, 48, 38, 16, 8, 11, 11, 11, 9, 13, 4, 15, 19, 15, 8,
                          6, 6, 4, 9, 6, 8, 7, 11, 11, 13, 12, 2, 4, 9, 12, 27, 44, 63, 69, 70, 57, 43, 29, 18, 16, 13,
                          12, 10, 12, 10, 8, 13, 7, 10, 12, 14, 22],
                         [11, 5, 8, 15, 18, 23, 13, 11, 8, 2, 5, 1, 9, 11, 15, 21, 23, 27, 22, 27, 32, 35, 32, 37, 35,
                          30, 30, 27, 24, 24, 21, 17, 21, 22, 21, 20, 18, 11, 10, 15, 6, 12, 10, 8, 5, 8, 6, 10, 8, 4,
                          3, 6, 6, 4, 4, 1, 6, 11, 1, 10, 14, 13, 19, 16, 5, 11, 4, 7, 7, 9, 15, 21, 28, 54, 71, 84, 99,
                          77, 50, 25, 14, 8, 7, 12, 7, 12, 10, 4, 3, 2, 9, 4, 4, 1, 7, 7, 7, 5, 7, 0, 2, 4, 10, 2, 5, 5,
                          1, 10, 2, 2, 8, 10, 6, 5, 5, 5, 9, 7, 11, 5, 3, 10, 6, 6, 5, 7, 8, 8, 8, 4, 2, 2, 6, 2, 6, 8,
                          6, 4, 4, 6, 1, 8, 9, 9, 6, 11, 9, 9, 33, 45, 50, 43, 24, 16, 18, 15, 11, 9, 15, 13, 19, 17,
                          11, 9, 6, 6, 6, 10, 9, 12, 9, 13, 9, 9, 7, 11, 1, 4, 26, 36, 50, 59, 56, 52, 36, 24, 16, 13,
                          10, 8, 8, 6, 6, 6, 11, 6, 5, 8, 9, 9, 14],
                         [4, 9, 15, 16, 18, 16, 11, 11, 2, 6, 6, 11, 13, 11, 13, 18, 25, 35, 28, 32, 35, 32, 29, 22, 16,
                          20, 15, 18, 15, 13, 17, 17, 18, 13, 12, 14, 13, 4, 10, 6, 10, 5, 9, 9, 3, 3, 5, 5, 3, 5, 5, 3,
                          9, 3, 6, 7, 5, 1, 4, 9, 8, 14, 11, 11, 5, 4, 3, 1, 4, 7, 7, 9, 16, 28, 50, 66, 98, 108, 93,
                          53, 30, 17, 12, 20, 18, 17, 5, 9, 3, 3, 5, 3, 2, 9, 9, 7, 5, 5, 5, 6, 4, 2, 10, 8, 6, 0, 4, 4,
                          6, 4, 8, 6, 4, 0, 7, 3, 5, 9, 5, 3, 6, 6, 2, 4, 3, 7, 6, 2, 4, 4, 6, 2, 4, 2, 10, 3, 3, 3, 5,
                          5, 6, 2, 6, 7, 7, 6, 6, 11, 26, 36, 58, 48, 41, 24, 21, 21, 16, 15, 17, 14, 22, 11, 9, 9, 9,
                          9, 5, 7, 9, 12, 13, 9, 9, 4, 4, 7, 9, 15, 27, 36, 36, 45, 32, 28, 18, 14, 11, 10, 6, 6, 4, 2,
                          2, 8, 2, 0, 5, 5, 5, 3, 11],
                         [8, 10, 21, 17, 23, 17, 11, 5, 4, 6, 1, 4, 10, 8, 13, 13, 21, 19, 25, 27, 23, 20, 18, 18, 13,
                          15, 13, 18, 10, 12, 11, 7, 11, 5, 11, 3, 3, 10, 3, 8, 6, 3, 3, 5, 7, 7, 1, 5, 5, 3, 5, 9, 9,
                          5, 6, 2, 4, 8, 6, 9, 16, 21, 13, 7, 9, 5, 5, 4, 11, 8, 9, 7, 13, 18, 32, 46, 73, 86, 114, 89,
                          66, 41, 30, 27, 21, 18, 21, 18, 9, 11, 8, 5, 4, 7, 1, 7, 7, 7, 6, 4, 6, 4, 8, 10, 4, 8, 4, 4,
                          6, 2, 8, 2, 6, 2, 2, 9, 5, 3, 5, 0, 6, 8, 8, 6, 1, 5, 5, 2, 4, 4, 8, 2, 9, 8, 5, 9, 9, 3, 1,
                          8, 6, 8, 4, 6, 5, 4, 6, 7, 14, 34, 49, 57, 53, 43, 31, 27, 23, 21, 21, 19, 12, 9, 9, 7, 9, 8,
                          7, 11, 7, 12, 1, 10, 8, 4, 2, 17, 20, 29, 34, 25, 27, 13, 21, 18, 12, 10, 6, 6, 1, 8, 4, 2, 6,
                          6, 6, 2, 0, 4, 6, 4, 10],
                         [8, 19, 20, 23, 13, 14, 11, 7, 5, 1, 8, 1, 6, 10, 13, 11, 13, 8, 15, 15, 16, 16, 10, 10, 11, 8,
                          6, 12, 6, 7, 7, 5, 7, 7, 6, 7, 2, 8, 8, 2, 1, 7, 0, 8, 6, 6, 4, 6, 3, 5, 3, 5, 3, 1, 5, 2, 6,
                          10, 6, 11, 13, 19, 10, 12, 3, 8, 4, 1, 7, 6, 6, 9, 9, 7, 13, 16, 46, 63, 92, 99, 88, 62, 23,
                          16, 17, 16, 18, 9, 12, 7, 6, 4, 0, 8, 2, 6, 4, 0, 2, 2, 4, 2, 4, 6, 6, 6, 2, 10, 8, 4, 8, 4,
                          10, 4, 6, 2, 2, 12, 5, 8, 12, 4, 10, 10, 5, 7, 9, 5, 6, 8, 2, 7, 12, 8, 7, 9, 9, 4, 4, 8, 6,
                          6, 8, 1, 6, 10, 5, 7, 15, 31, 53, 70, 56, 48, 38, 34, 28, 26, 26, 25, 17, 7, 14, 12, 9, 9, 9,
                          9, 9, 9, 9, 9, 4, 5, 14, 16, 29, 26, 26, 19, 15, 20, 11, 11, 9, 4, 4, 6, 10, 8, 4, 4, 6, 4,
                          10, 4, 0, 5, 6, 9, 4],
                         [4, 17, 23, 20, 20, 11, 12, 3, 7, 6, 8, 6, 10, 10, 4, 8, 11, 7, 14, 14, 7, 13, 13, 4, 3, 3, 8,
                          11, 6, 1, 4, 2, 4, 2, 2, 8, 4, 6, 6, 8, 1, 12, 8, 6, 3, 4, 6, 8, 4, 4, 4, 1, 5, 3, 7, 6, 4,
                          10, 4, 14, 16, 16, 9, 11, 4, 1, 9, 1, 8, 7, 8, 8, 9, 12, 7, 15, 21, 48, 67, 89, 98, 77, 41,
                          17, 14, 14, 18, 18, 17, 6, 5, 4, 1, 5, 1, 10, 8, 5, 4, 8, 8, 4, 0, 10, 2, 4, 4, 0, 0, 6, 2, 4,
                          4, 4, 6, 0, 2, 5, 7, 5, 12, 3, 13, 7, 9, 1, 5, 3, 1, 1, 3, 0, 6, 4, 9, 7, 6, 8, 2, 8, 4, 4, 8,
                          9, 6, 8, 5, 11, 21, 31, 59, 68, 62, 58, 45, 41, 36, 35, 33, 28, 20, 25, 31, 27, 23, 18, 14,
                          11, 9, 7, 12, 14, 14, 16, 16, 26, 18, 21, 20, 18, 7, 13, 9, 6, 11, 0, 7, 7, 4, 3, 5, 3, 2, 0,
                          3, 8, 4, 3, 1, 8, 6],
                         [5, 19, 13, 20, 13, 4, 3, 3, 7, 8, 6, 4, 8, 4, 10, 6, 8, 11, 2, 8, 4, 8, 8, 4, 5, 1, 4, 8, 12,
                          8, 4, 4, 4, 6, 4, 6, 4, 4, 2, 4, 6, 3, 2, 2, 6, 8, 8, 6, 6, 8, 1, 5, 7, 7, 4, 4, 4, 17, 11,
                          16, 16, 14, 9, 2, 4, 11, 10, 10, 6, 1, 3, 5, 3, 7, 8, 13, 15, 31, 45, 60, 90, 91, 77, 41, 21,
                          12, 8, 12, 18, 9, 12, 8, 3, 8, 8, 5, 5, 10, 1, 4, 6, 6, 6, 6, 8, 0, 4, 4, 8, 6, 2, 6, 4, 6, 2,
                          10, 3, 5, 8, 5, 7, 0, 7, 7, 1, 3, 7, 5, 7, 7, 2, 1, 2, 6, 10, 8, 4, 4, 8, 6, 6, 6, 6, 6, 4, 7,
                          10, 14, 21, 36, 59, 67, 73, 65, 53, 49, 44, 48, 41, 34, 36, 35, 31, 33, 33, 24, 24, 21, 18, 9,
                          17, 19, 23, 19, 15, 25, 17, 21, 16, 11, 13, 6, 10, 6, 7, 7, 3, 10, 4, 2, 8, 8, 6, 11, 0, 6, 6,
                          0, 8, 5, 8],
                         [14, 18, 23, 18, 10, 8, 7, 7, 6, 2, 4, 10, 6, 8, 6, 1, 7, 7, 11, 6, 8, 10, 6, 6, 2, 6, 6, 4, 8,
                          2, 2, 6, 1, 8, 4, 2, 8, 2, 6, 8, 8, 5, 4, 6, 4, 8, 6, 10, 0, 8, 7, 3, 5, 11, 6, 14, 8, 10, 14,
                          14, 16, 16, 13, 10, 6, 5, 1, 8, 4, 0, 1, 5, 5, 7, 6, 5, 11, 13, 17, 40, 54, 86, 95, 74, 45,
                          28, 15, 12, 21, 16, 9, 4, 4, 9, 3, 5, 3, 5, 5, 5, 0, 4, 4, 2, 8, 6, 2, 8, 8, 8, 0, 8, 4, 2, 0,
                          12, 5, 10, 1, 8, 8, 2, 9, 1, 9, 7, 5, 7, 5, 10, 0, 12, 10, 6, 0, 6, 4, 6, 2, 6, 6, 9, 9, 4,
                          11, 9, 12, 17, 27, 45, 72, 75, 87, 77, 65, 69, 61, 57, 54, 44, 34, 30, 25, 20, 36, 31, 32, 30,
                          34, 27, 26, 26, 24, 20, 18, 24, 15, 17, 18, 6, 8, 4, 9, 11, 3, 5, 7, 8, 8, 4, 2, 2, 2, 5, 6,
                          6, 9, 2, 10, 10, 1],
                         [11, 16, 19, 19, 12, 8, 5, 2, 6, 8, 4, 4, 1, 4, 6, 8, 8, 0, 7, 7, 9, 6, 2, 6, 4, 6, 8, 8, 6, 6,
                          4, 6, 11, 6, 2, 2, 6, 6, 4, 6, 2, 5, 11, 0, 1, 1, 4, 1, 0, 5, 5, 7, 9, 9, 15, 9, 6, 9, 11, 5,
                          16, 12, 14, 10, 10, 4, 9, 8, 4, 6, 6, 4, 5, 7, 3, 6, 6, 10, 15, 26, 36, 65, 83, 85, 70, 50,
                          25, 19, 10, 14, 10, 9, 10, 3, 5, 3, 3, 5, 5, 3, 6, 2, 10, 2, 6, 2, 4, 2, 4, 10, 6, 8, 1, 3, 2,
                          3, 3, 5, 5, 3, 6, 4, 1, 3, 3, 9, 7, 1, 9, 13, 8, 0, 1, 8, 1, 7, 4, 8, 6, 6, 8, 4, 1, 4, 6, 11,
                          14, 19, 34, 57, 84, 89, 89, 88, 89, 93, 85, 81, 68, 48, 37, 33, 27, 34, 39, 33, 32, 35, 34,
                          30, 31, 26, 26, 25, 20, 19, 19, 11, 8, 10, 10, 0, 10, 7, 7, 5, 4, 6, 2, 0, 5, 5, 7, 2, 2, 4,
                          6, 4, 6, 3, 10],
                         [12, 14, 14, 14, 9, 9, 4, 2, 2, 4, 4, 4, 8, 0, 0, 2, 0, 9, 7, 5, 0, 7, 11, 2, 2, 6, 6, 9, 9, 4,
                          2, 5, 3, 7, 4, 6, 4, 6, 6, 4, 4, 9, 6, 1, 6, 6, 1, 6, 5, 9, 15, 8, 6, 12, 9, 5, 5, 4, 6, 8, 8,
                          7, 11, 13, 8, 6, 9, 4, 6, 9, 4, 4, 1, 3, 3, 3, 8, 8, 10, 10, 18, 41, 57, 72, 86, 82, 51, 28,
                          15, 7, 10, 12, 7, 3, 1, 3, 10, 1, 6, 10, 0, 4, 8, 4, 6, 12, 5, 9, 3, 3, 10, 1, 3, 5, 2, 4, 8,
                          5, 8, 8, 8, 4, 3, 1, 9, 3, 0, 0, 2, 8, 4, 8, 3, 7, 7, 7, 5, 6, 4, 7, 7, 9, 8, 9, 9, 15, 19,
                          24, 43, 72, 111, 117, 106, 96, 99, 108, 100, 79, 59, 38, 33, 30, 32, 23, 26, 21, 21, 14, 21,
                          23, 23, 21, 28, 22, 19, 15, 13, 8, 1, 11, 0, 7, 10, 7, 10, 7, 6, 4, 0, 3, 7, 2, 5, 5, 4, 12,
                          6, 9, 11, 2, 6],
                         [16, 14, 12, 12, 6, 3, 4, 9, 6, 6, 1, 2, 6, 9, 4, 9, 4, 11, 7, 2, 6, 3, 3, 4, 6, 11, 12, 4, 9,
                          6, 9, 7, 7, 3, 11, 4, 6, 4, 4, 2, 1, 8, 5, 5, 3, 7, 7, 7, 11, 1, 12, 10, 12, 9, 5, 5, 3, 7, 5,
                          3, 8, 8, 7, 6, 13, 10, 6, 4, 0, 11, 6, 6, 6, 4, 5, 7, 9, 3, 10, 8, 16, 22, 32, 52, 71, 76, 87,
                          61, 43, 26, 17, 10, 10, 9, 8, 9, 5, 7, 8, 3, 1, 2, 10, 6, 0, 7, 3, 5, 5, 7, 1, 3, 9, 1, 2, 4,
                          6, 10, 6, 3, 6, 4, 1, 5, 3, 2, 6, 4, 6, 7, 11, 5, 5, 8, 3, 9, 3, 8, 6, 6, 4, 8, 9, 10, 11, 9,
                          15, 29, 57, 91, 155, 170, 145, 140, 126, 106, 88, 71, 46, 38, 26, 25, 23, 12, 15, 11, 14, 14,
                          13, 16, 16, 12, 16, 12, 7, 10, 6, 13, 5, 10, 10, 1, 3, 7, 5, 5, 6, 5, 0, 2, 5, 7, 2, 0, 2, 6,
                          2, 2, 2, 10, 2],
                         [11, 6, 11, 4, 8, 7, 1, 2, 2, 4, 4, 9, 6, 6, 10, 4, 0, 4, 5, 5, 5, 11, 3, 5, 11, 11, 4, 4, 1,
                          7, 7, 5, 1, 3, 4, 4, 6, 4, 4, 2, 1, 3, 7, 5, 3, 9, 9, 11, 9, 9, 11, 6, 5, 3, 7, 7, 5, 3, 5, 5,
                          4, 6, 10, 7, 13, 4, 4, 1, 6, 4, 6, 4, 4, 4, 8, 6, 4, 0, 5, 6, 8, 8, 17, 36, 50, 71, 86, 86,
                          81, 60, 38, 26, 19, 15, 7, 3, 2, 10, 5, 1, 1, 0, 6, 6, 5, 5, 3, 5, 5, 1, 9, 7, 3, 9, 3, 6, 4,
                          4, 6, 8, 0, 12, 2, 9, 8, 8, 6, 10, 5, 4, 13, 9, 15, 9, 6, 4, 6, 6, 7, 7, 8, 10, 15, 9, 9, 14,
                          22, 50, 75, 133, 175, 184, 175, 153, 130, 103, 81, 56, 42, 33, 16, 12, 18, 11, 13, 6, 13, 10,
                          7, 10, 6, 12, 7, 12, 9, 6, 6, 8, 7, 7, 7, 3, 1, 0, 6, 11, 9, 7, 7, 2, 7, 5, 5, 11, 9, 6, 9, 4,
                          6, 2, 10],
                         [9, 9, 8, 4, 2, 4, 4, 2, 4, 6, 6, 6, 9, 6, 6, 8, 6, 7, 7, 5, 5, 7, 0, 7, 2, 6, 4, 1, 5, 5, 11,
                          1, 4, 5, 5, 5, 1, 9, 7, 5, 3, 7, 5, 7, 3, 14, 7, 14, 4, 11, 6, 7, 5, 5, 9, 9, 8, 6, 2, 4, 7,
                          6, 8, 12, 13, 10, 6, 0, 6, 0, 6, 4, 6, 0, 4, 6, 2, 10, 4, 11, 9, 4, 13, 17, 28, 43, 61, 78,
                          77, 70, 67, 52, 35, 23, 19, 14, 6, 6, 6, 3, 5, 8, 0, 6, 4, 2, 8, 0, 9, 5, 7, 7, 5, 3, 7, 7, 6,
                          2, 10, 8, 10, 1, 0, 10, 3, 8, 6, 1, 4, 3, 7, 16, 17, 13, 7, 11, 10, 8, 8, 8, 13, 10, 13, 15,
                          15, 21, 30, 63, 122, 175, 184, 185, 185, 156, 128, 95, 74, 56, 39, 25, 15, 3, 12, 4, 6, 1, 4,
                          4, 10, 10, 6, 3, 10, 5, 1, 9, 7, 2, 3, 5, 0, 4, 6, 8, 10, 8, 4, 2, 9, 5, 2, 2, 0, 7, 0, 0, 9,
                          4, 2, 2, 5],
                         [12, 6, 10, 7, 3, 4, 6, 4, 2, 6, 9, 4, 4, 4, 4, 4, 5, 1, 7, 5, 7, 7, 9, 1, 5, 7, 11, 5, 3, 5,
                          5, 6, 9, 3, 9, 7, 5, 3, 5, 9, 7, 3, 5, 5, 4, 12, 12, 11, 6, 6, 2, 5, 5, 3, 1, 9, 9, 2, 2, 9,
                          4, 10, 10, 8, 6, 10, 8, 6, 4, 6, 6, 6, 4, 6, 11, 0, 10, 4, 4, 6, 1, 5, 7, 11, 15, 24, 40, 47,
                          55, 60, 65, 64, 65, 55, 38, 26, 12, 13, 8, 4, 4, 4, 1, 3, 2, 4, 6, 4, 3, 7, 7, 5, 7, 1, 3, 5,
                          1, 8, 6, 9, 9, 5, 7, 4, 10, 6, 3, 3, 1, 9, 15, 21, 23, 16, 12, 9, 6, 11, 11, 12, 13, 16, 19,
                          19, 27, 39, 60, 111, 158, 184, 184, 185, 185, 171, 135, 93, 69, 50, 34, 27, 14, 9, 4, 6, 6, 1,
                          4, 10, 5, 1, 7, 5, 5, 4, 1, 10, 1, 1, 5, 5, 7, 4, 1, 6, 6, 0, 6, 2, 7, 7, 7, 11, 6, 0, 4, 2,
                          2, 0, 6, 6, 2],
                         [5, 5, 4, 6, 8, 4, 8, 4, 11, 4, 6, 4, 9, 6, 4, 6, 0, 9, 3, 7, 5, 7, 9, 11, 7, 13, 9, 1, 1, 9,
                          4, 2, 2, 3, 9, 9, 7, 5, 3, 3, 5, 3, 9, 6, 6, 12, 6, 6, 6, 11, 4, 3, 3, 0, 3, 1, 9, 2, 4, 9, 7,
                          8, 7, 4, 10, 6, 4, 0, 2, 6, 4, 6, 2, 4, 6, 4, 4, 6, 8, 8, 6, 5, 10, 7, 6, 11, 24, 37, 38, 38,
                          49, 59, 66, 69, 67, 56, 37, 21, 7, 8, 4, 6, 8, 1, 8, 2, 6, 6, 2, 7, 7, 7, 9, 0, 9, 1, 6, 4, 9,
                          12, 16, 16, 10, 4, 3, 15, 0, 6, 7, 13, 18, 27, 29, 22, 15, 11, 11, 13, 14, 16, 18, 21, 24, 29,
                          45, 89, 111, 150, 184, 184, 184, 185, 185, 161, 104, 79, 57, 42, 33, 27, 14, 9, 5, 7, 5, 4,
                          10, 8, 9, 2, 4, 10, 4, 7, 3, 1, 5, 5, 10, 1, 5, 1, 10, 8, 8, 8, 2, 7, 5, 2, 6, 2, 11, 0, 4, 0,
                          6, 0, 13, 4, 9],
                         [7, 2, 1, 5, 6, 4, 4, 4, 4, 6, 9, 4, 2, 9, 9, 4, 2, 1, 3, 5, 5, 7, 7, 6, 12, 12, 7, 11, 1, 4,
                          2, 11, 6, 2, 5, 3, 9, 5, 7, 5, 9, 9, 11, 4, 6, 9, 6, 5, 1, 5, 11, 5, 0, 2, 7, 5, 5, 2, 7, 2,
                          1, 7, 1, 6, 6, 6, 6, 11, 4, 6, 4, 2, 9, 11, 5, 5, 4, 1, 4, 4, 8, 3, 5, 6, 7, 13, 10, 20, 24,
                          26, 27, 31, 47, 55, 60, 67, 64, 46, 33, 16, 10, 6, 8, 4, 6, 4, 4, 2, 6, 9, 13, 10, 10, 7, 6,
                          6, 2, 3, 9, 13, 20, 19, 17, 9, 3, 0, 1, 3, 7, 0, 14, 28, 31, 30, 21, 16, 16, 21, 22, 26, 26,
                          37, 40, 73, 103, 143, 155, 172, 174, 178, 168, 150, 131, 98, 84, 55, 44, 34, 31, 31, 25, 18,
                          5, 4, 7, 7, 1, 2, 4, 4, 2, 4, 3, 11, 1, 5, 5, 1, 1, 5, 5, 6, 1, 0, 1, 11, 7, 9, 9, 5, 7, 4, 2,
                          9, 2, 2, 2, 7, 8, 8, 6],
                         [4, 7, 2, 5, 4, 4, 4, 6, 1, 5, 8, 6, 4, 6, 6, 6, 1, 9, 3, 5, 5, 7, 0, 4, 6, 7, 9, 12, 9, 9, 9,
                          6, 2, 4, 6, 7, 5, 5, 7, 9, 9, 9, 11, 10, 6, 4, 5, 8, 6, 6, 6, 6, 6, 4, 3, 7, 7, 5, 7, 1, 7, 9,
                          10, 4, 10, 4, 9, 0, 4, 11, 11, 11, 5, 3, 1, 5, 1, 4, 10, 6, 9, 4, 5, 3, 9, 11, 9, 20, 16, 15,
                          15, 15, 30, 37, 44, 57, 67, 67, 60, 49, 25, 14, 6, 6, 10, 8, 6, 11, 7, 9, 3, 10, 9, 4, 4, 4,
                          6, 6, 9, 14, 22, 32, 29, 17, 10, 8, 4, 4, 3, 12, 22, 34, 34, 32, 30, 27, 26, 27, 29, 48, 66,
                          86, 114, 145, 173, 176, 173, 148, 128, 114, 82, 86, 75, 71, 63, 50, 39, 23, 23, 29, 27, 24,
                          17, 9, 2, 3, 4, 1, 1, 1, 3, 5, 7, 1, 6, 6, 7, 5, 7, 1, 0, 1, 4, 0, 4, 5, 5, 9, 5, 9, 7, 0, 11,
                          2, 6, 4, 2, 6, 6, 1, 3],
                         [2, 7, 4, 5, 5, 2, 4, 4, 3, 9, 9, 4, 4, 8, 5, 3, 8, 11, 5, 7, 3, 9, 6, 4, 0, 10, 7, 13, 5, 9,
                          9, 9, 4, 2, 4, 11, 6, 12, 10, 10, 12, 8, 12, 2, 11, 1, 6, 1, 6, 1, 6, 6, 4, 7, 1, 7, 9, 3, 3,
                          5, 5, 9, 5, 9, 9, 4, 8, 6, 8, 4, 0, 5, 1, 3, 7, 9, 5, 5, 9, 4, 6, 4, 4, 5, 4, 10, 15, 16, 9,
                          13, 9, 8, 14, 16, 32, 40, 47, 59, 69, 63, 54, 41, 20, 12, 7, 6, 5, 1, 0, 7, 8, 8, 11, 6, 2, 4,
                          1, 8, 5, 8, 21, 36, 47, 35, 28, 15, 13, 8, 16, 15, 34, 46, 59, 48, 42, 41, 45, 47, 68, 113,
                          147, 176, 182, 181, 167, 129, 117, 96, 63, 60, 48, 44, 55, 49, 44, 33, 26, 16, 13, 26, 26, 27,
                          25, 21, 12, 7, 2, 7, 5, 3, 5, 5, 1, 6, 6, 1, 1, 3, 5, 0, 6, 8, 8, 1, 5, 0, 2, 5, 7, 9, 2, 0,
                          7, 6, 6, 9, 6, 6, 3, 1, 1],
                         [7, 4, 2, 9, 5, 0, 8, 5, 5, 5, 9, 11, 2, 6, 11, 6, 2, 4, 4, 2, 4, 4, 4, 9, 4, 4, 8, 13, 0, 4,
                          9, 6, 2, 6, 2, 12, 9, 9, 12, 14, 9, 6, 11, 6, 12, 4, 4, 6, 10, 2, 9, 3, 4, 7, 7, 3, 7, 7, 0,
                          7, 5, 9, 5, 11, 4, 6, 6, 4, 4, 8, 4, 5, 3, 5, 3, 3, 5, 1, 5, 6, 6, 2, 1, 6, 3, 6, 11, 16, 15,
                          5, 8, 7, 8, 10, 15, 21, 26, 41, 45, 53, 64, 62, 51, 30, 14, 10, 6, 1, 10, 3, 0, 5, 5, 3, 5, 5,
                          5, 8, 2, 12, 19, 32, 49, 56, 45, 35, 18, 18, 27, 36, 52, 69, 90, 80, 85, 109, 144, 164, 182,
                          182, 182, 182, 183, 151, 118, 91, 68, 56, 37, 34, 24, 28, 37, 43, 31, 22, 14, 12, 12, 12, 13,
                          18, 33, 28, 18, 8, 8, 4, 4, 3, 0, 0, 8, 6, 1, 6, 0, 0, 4, 6, 10, 8, 4, 9, 9, 7, 7, 7, 0, 5, 5,
                          5, 9, 4, 6, 1, 8, 6, 3, 8, 0],
                         [4, 4, 5, 0, 4, 10, 4, 3, 7, 12, 9, 10, 8, 4, 9, 6, 9, 6, 6, 6, 4, 4, 11, 0, 4, 4, 4, 4, 7, 2,
                          6, 6, 11, 9, 3, 7, 14, 2, 4, 1, 6, 3, 5, 10, 6, 10, 1, 6, 9, 9, 5, 1, 7, 2, 4, 11, 3, 5, 7, 9,
                          0, 5, 9, 1, 6, 1, 8, 4, 1, 6, 4, 6, 5, 7, 5, 1, 3, 3, 1, 5, 3, 1, 6, 6, 9, 7, 11, 9, 18, 8, 9,
                          1, 6, 4, 6, 15, 16, 22, 33, 41, 47, 60, 55, 50, 35, 18, 10, 8, 8, 5, 8, 8, 11, 3, 3, 5, 1, 4,
                          10, 8, 14, 21, 41, 53, 60, 48, 37, 37, 45, 59, 81, 129, 169, 180, 179, 181, 181, 182, 182,
                          182, 182, 175, 138, 96, 72, 56, 41, 30, 24, 18, 15, 21, 22, 29, 28, 18, 14, 11, 10, 10, 12,
                          21, 28, 28, 22, 15, 8, 8, 4, 7, 4, 7, 2, 4, 2, 0, 9, 4, 10, 12, 8, 4, 10, 2, 2, 9, 5, 7, 0, 2,
                          7, 2, 7, 2, 11, 1, 6, 6, 1, 1, 3],
                         [7, 3, 6, 8, 0, 4, 10, 7, 9, 17, 13, 10, 4, 10, 6, 9, 5, 9, 0, 2, 4, 2, 7, 2, 4, 0, 4, 4, 3, 5,
                          4, 14, 7, 14, 6, 14, 11, 6, 6, 6, 6, 8, 10, 2, 6, 2, 5, 5, 1, 5, 0, 3, 5, 9, 11, 3, 1, 3, 3,
                          5, 7, 10, 7, 7, 9, 6, 6, 10, 4, 4, 6, 4, 6, 3, 5, 5, 5, 7, 1, 5, 5, 9, 4, 6, 8, 7, 15, 12, 10,
                          4, 3, 12, 5, 8, 6, 9, 4, 15, 18, 24, 37, 43, 51, 61, 56, 38, 24, 11, 11, 6, 3, 8, 1, 3, 1, 3,
                          1, 2, 6, 7, 12, 17, 30, 47, 62, 80, 79, 72, 86, 104, 159, 180, 180, 181, 181, 181, 182, 182,
                          182, 157, 137, 105, 79, 59, 42, 30, 21, 19, 17, 9, 14, 17, 20, 22, 15, 12, 8, 4, 10, 4, 10, 8,
                          21, 27, 27, 16, 10, 11, 4, 0, 10, 0, 9, 7, 2, 7, 9, 4, 7, 7, 7, 4, 2, 9, 5, 0, 9, 7, 5, 2, 2,
                          7, 4, 2, 2, 3, 3, 3, 6, 1, 8],
                         [1, 6, 6, 10, 4, 7, 10, 9, 11, 7, 10, 6, 10, 4, 6, 6, 0, 6, 4, 4, 7, 2, 4, 0, 9, 2, 9, 1, 5, 1,
                          0, 11, 2, 7, 14, 8, 13, 6, 6, 4, 7, 6, 2, 2, 2, 1, 0, 6, 5, 7, 9, 7, 3, 7, 0, 5, 3, 1, 5, 5,
                          7, 5, 5, 5, 4, 6, 10, 4, 6, 4, 2, 4, 9, 7, 1, 7, 3, 3, 3, 9, 9, 2, 6, 6, 8, 7, 9, 12, 11, 9,
                          5, 3, 3, 1, 8, 6, 9, 10, 8, 15, 20, 26, 40, 46, 60, 57, 46, 31, 22, 12, 8, 10, 5, 0, 2, 2, 8,
                          6, 2, 8, 7, 13, 15, 33, 59, 92, 108, 112, 130, 179, 180, 180, 180, 181, 181, 177, 160, 150,
                          124, 96, 84, 60, 43, 33, 22, 18, 15, 7, 7, 6, 15, 20, 18, 13, 12, 12, 11, 1, 7, 2, 7, 4, 19,
                          18, 27, 19, 10, 2, 4, 2, 4, 4, 4, 2, 0, 4, 9, 7, 7, 0, 2, 9, 0, 5, 2, 2, 7, 0, 2, 7, 7, 0, 2,
                          6, 6, 6, 6, 8, 8, 6, 3],
                         [6, 4, 0, 4, 4, 4, 9, 6, 11, 7, 5, 10, 11, 12, 12, 3, 12, 3, 3, 1, 8, 2, 0, 7, 4, 6, 8, 0, 1,
                          7, 11, 7, 4, 9, 7, 14, 11, 6, 10, 6, 3, 4, 4, 4, 9, 9, 4, 1, 3, 7, 9, 3, 5, 5, 7, 3, 7, 10, 3,
                          3, 5, 3, 5, 3, 10, 1, 8, 1, 6, 1, 2, 7, 2, 9, 7, 5, 5, 5, 7, 5, 9, 6, 2, 2, 4, 6, 1, 5, 9, 3,
                          9, 5, 4, 1, 6, 1, 4, 5, 10, 8, 10, 13, 25, 29, 40, 54, 56, 49, 49, 29, 17, 11, 4, 5, 2, 2, 6,
                          6, 1, 3, 3, 6, 17, 33, 57, 97, 134, 167, 179, 179, 180, 180, 181, 181, 174, 121, 91, 74, 66,
                          59, 42, 31, 22, 19, 15, 10, 9, 6, 4, 13, 15, 12, 14, 9, 10, 3, 10, 4, 9, 4, 5, 5, 8, 17, 21,
                          25, 13, 7, 2, 7, 5, 7, 2, 7, 7, 0, 7, 4, 2, 0, 2, 7, 9, 7, 5, 0, 7, 3, 2, 11, 7, 7, 2, 11, 2,
                          4, 2, 6, 3, 6, 6],
                         [6, 10, 2, 7, 2, 7, 2, 8, 1, 12, 17, 7, 14, 12, 10, 14, 9, 5, 5, 3, 1, 1, 0, 4, 4, 6, 4, 11, 7,
                          4, 7, 7, 2, 7, 13, 7, 14, 13, 10, 4, 8, 7, 7, 7, 7, 5, 7, 5, 1, 7, 9, 9, 3, 3, 5, 1, 10, 1, 3,
                          7, 3, 7, 5, 1, 1, 0, 4, 2, 7, 2, 12, 9, 1, 9, 3, 5, 5, 7, 7, 9, 5, 2, 6, 4, 10, 6, 8, 6, 7, 3,
                          3, 6, 6, 4, 6, 0, 4, 6, 7, 5, 7, 9, 13, 20, 28, 33, 43, 52, 56, 54, 43, 24, 11, 8, 5, 7, 3, 3,
                          0, 4, 5, 6, 27, 47, 78, 115, 174, 179, 179, 180, 180, 181, 181, 181, 131, 76, 49, 46, 34, 25,
                          21, 17, 12, 9, 8, 5, 5, 7, 4, 15, 15, 17, 12, 7, 6, 9, 7, 5, 5, 5, 7, 5, 5, 16, 19, 22, 17,
                          16, 5, 2, 3, 5, 0, 7, 7, 4, 4, 2, 7, 9, 7, 7, 7, 5, 0, 2, 0, 1, 5, 2, 2, 5, 5, 5, 5, 0, 0, 13,
                          6, 3, 3],
                         [4, 8, 9, 4, 4, 4, 6, 7, 4, 7, 11, 7, 16, 13, 13, 11, 11, 9, 12, 3, 3, 1, 6, 4, 6, 8, 11, 11,
                          11, 9, 11, 14, 4, 6, 6, 8, 12, 5, 9, 9, 4, 2, 7, 7, 5, 0, 3, 3, 3, 9, 3, 3, 7, 5, 5, 3, 5, 7,
                          5, 3, 10, 10, 5, 3, 11, 2, 1, 4, 4, 4, 9, 3, 5, 5, 5, 3, 5, 7, 3, 3, 3, 3, 6, 6, 4, 2, 6, 6,
                          6, 8, 4, 4, 6, 4, 6, 4, 4, 1, 3, 2, 7, 9, 8, 10, 14, 22, 32, 40, 49, 60, 60, 53, 39, 26, 18,
                          12, 8, 10, 6, 10, 15, 29, 38, 71, 106, 141, 179, 179, 180, 180, 180, 181, 181, 153, 90, 50,
                          31, 18, 14, 14, 12, 9, 6, 5, 1, 3, 0, 7, 9, 11, 15, 15, 9, 10, 6, 9, 5, 0, 3, 0, 0, 1, 5, 9,
                          12, 18, 24, 16, 10, 3, 4, 5, 1, 2, 9, 2, 2, 2, 2, 9, 0, 2, 9, 0, 9, 2, 5, 8, 5, 5, 2, 5, 9, 2,
                          0, 0, 0, 10, 2, 2, 10],
                         [6, 0, 2, 7, 2, 4, 5, 4, 8, 4, 5, 9, 12, 5, 3, 9, 11, 13, 10, 6, 3, 5, 9, 3, 1, 9, 11, 4, 9,
                          11, 11, 4, 9, 2, 8, 6, 4, 12, 7, 12, 6, 2, 1, 9, 1, 7, 7, 5, 1, 5, 3, 1, 10, 1, 7, 3, 1, 5, 1,
                          11, 5, 5, 5, 4, 2, 5, 5, 10, 1, 2, 4, 4, 5, 5, 5, 9, 1, 0, 9, 5, 3, 3, 0, 1, 2, 9, 8, 2, 1,
                          10, 6, 6, 1, 5, 5, 4, 4, 6, 5, 11, 4, 2, 4, 7, 8, 11, 17, 25, 31, 42, 51, 58, 64, 53, 51, 43,
                          36, 32, 26, 26, 21, 26, 53, 84, 104, 126, 154, 179, 180, 180, 181, 181, 181, 106, 68, 44, 16,
                          8, 6, 6, 5, 0, 4, 0, 3, 1, 3, 4, 7, 15, 17, 5, 8, 8, 1, 1, 1, 4, 2, 6, 6, 8, 3, 10, 13, 14,
                          15, 22, 16, 9, 7, 10, 7, 0, 2, 4, 2, 7, 2, 5, 5, 7, 7, 2, 7, 8, 5, 8, 1, 5, 0, 0, 5, 9, 8, 6,
                          0, 7, 7, 0, 0],
                         [6, 6, 7, 4, 6, 10, 4, 8, 14, 12, 12, 8, 4, 4, 8, 8, 13, 12, 12, 6, 4, 5, 7, 5, 7, 15, 12, 14,
                          8, 9, 4, 7, 11, 2, 4, 6, 6, 8, 12, 10, 7, 4, 7, 0, 6, 5, 3, 1, 3, 3, 9, 5, 7, 5, 3, 1, 5, 10,
                          7, 5, 7, 1, 5, 7, 0, 5, 3, 7, 4, 2, 9, 7, 4, 1, 3, 3, 3, 6, 10, 1, 4, 4, 11, 7, 2, 4, 2, 4, 6,
                          8, 7, 1, 5, 0, 5, 3, 11, 4, 0, 7, 8, 8, 4, 4, 6, 7, 8, 11, 19, 26, 36, 44, 44, 53, 57, 63, 69,
                          72, 56, 51, 46, 42, 60, 82, 101, 113, 133, 159, 180, 180, 181, 181, 141, 89, 63, 29, 13, 7, 8,
                          4, 4, 8, 4, 1, 3, 5, 1, 9, 11, 14, 16, 10, 6, 7, 4, 6, 6, 6, 1, 1, 6, 8, 5, 8, 7, 11, 16, 22,
                          18, 9, 4, 9, 5, 4, 9, 2, 4, 7, 0, 7, 0, 7, 2, 5, 0, 3, 3, 8, 1, 0, 5, 3, 3, 3, 0, 3, 10, 1,
                          10, 2, 0],
                         [8, 0, 5, 1, 10, 12, 12, 12, 1, 4, 9, 6, 10, 7, 7, 9, 4, 8, 15, 7, 17, 17, 9, 11, 11, 12, 6,
                          12, 11, 3, 9, 3, 5, 3, 5, 1, 7, 11, 8, 12, 8, 4, 0, 6, 6, 1, 7, 7, 7, 5, 3, 7, 10, 3, 10, 5,
                          7, 3, 3, 3, 1, 3, 3, 7, 4, 7, 1, 2, 7, 2, 4, 7, 4, 7, 3, 9, 3, 1, 4, 0, 10, 5, 11, 1, 5, 9, 6,
                          8, 6, 6, 0, 6, 4, 8, 6, 7, 3, 5, 4, 8, 5, 8, 6, 3, 8, 4, 6, 7, 8, 15, 22, 28, 32, 39, 40, 48,
                          51, 54, 62, 59, 58, 57, 57, 66, 77, 80, 92, 103, 116, 126, 135, 124, 109, 79, 59, 33, 19, 10,
                          6, 2, 2, 0, 1, 1, 1, 2, 4, 10, 15, 18, 8, 12, 8, 4, 7, 4, 8, 1, 6, 6, 4, 6, 4, 4, 7, 13, 15,
                          21, 18, 7, 4, 2, 7, 6, 7, 2, 2, 0, 7, 5, 5, 3, 1, 8, 1, 3, 8, 10, 0, 0, 10, 3, 8, 5, 6, 0, 8,
                          6, 3, 6, 0],
                         [8, 10, 3, 3, 10, 10, 14, 4, 1, 2, 2, 0, 5, 3, 3, 9, 7, 11, 8, 7, 7, 11, 11, 15, 12, 6, 1, 7,
                          9, 3, 3, 0, 0, 6, 8, 8, 5, 5, 10, 10, 14, 8, 4, 4, 6, 6, 1, 3, 3, 1, 5, 3, 5, 3, 10, 5, 10, 7,
                          5, 5, 3, 7, 7, 10, 6, 11, 7, 4, 4, 2, 2, 4, 4, 4, 3, 3, 4, 10, 6, 8, 4, 3, 9, 7, 5, 4, 8, 8,
                          8, 4, 6, 8, 6, 0, 6, 6, 3, 7, 5, 10, 8, 7, 1, 7, 3, 4, 6, 8, 6, 5, 13, 12, 18, 19, 22, 26, 28,
                          37, 41, 42, 44, 56, 53, 47, 53, 39, 53, 61, 71, 72, 88, 87, 89, 80, 59, 44, 29, 19, 12, 5, 2,
                          0, 7, 5, 2, 6, 15, 15, 18, 15, 4, 12, 8, 4, 4, 4, 8, 4, 4, 8, 1, 8, 8, 8, 7, 6, 15, 21, 15, 9,
                          9, 4, 4, 5, 9, 4, 2, 7, 0, 2, 5, 1, 5, 3, 1, 3, 5, 3, 5, 8, 3, 5, 0, 10, 10, 6, 3, 6, 8, 11,
                          1],
                         [4, 0, 5, 7, 12, 10, 8, 12, 9, 4, 2, 1, 2, 0, 7, 0, 1, 5, 4, 12, 9, 9, 7, 10, 10, 1, 7, 5, 3,
                          7, 6, 4, 4, 0, 4, 4, 8, 12, 8, 6, 12, 11, 8, 4, 0, 8, 8, 7, 7, 3, 5, 7, 7, 1, 7, 10, 10, 3, 7,
                          1, 1, 7, 10, 0, 4, 0, 2, 2, 2, 4, 2, 9, 2, 2, 9, 6, 6, 6, 4, 8, 6, 5, 1, 7, 6, 6, 4, 1, 6, 4,
                          4, 8, 1, 6, 0, 1, 4, 0, 7, 5, 2, 7, 1, 3, 6, 0, 6, 8, 1, 4, 11, 5, 3, 3, 5, 16, 11, 13, 13,
                          27, 31, 33, 41, 24, 28, 30, 35, 46, 49, 45, 57, 76, 78, 68, 55, 48, 39, 24, 11, 12, 4, 0, 6,
                          11, 8, 12, 10, 15, 6, 5, 10, 7, 10, 7, 8, 1, 8, 6, 1, 4, 8, 8, 6, 4, 0, 10, 11, 16, 15, 10, 9,
                          5, 1, 5, 10, 2, 7, 4, 5, 0, 7, 0, 3, 5, 1, 5, 3, 10, 3, 8, 8, 5, 8, 1, 3, 8, 3, 0, 3, 3, 0],
                         [4, 2, 10, 11, 4, 10, 6, 6, 4, 4, 2, 6, 7, 2, 4, 3, 6, 5, 2, 6, 2, 7, 9, 6, 6, 4, 9, 5, 1, 1,
                          1, 4, 11, 2, 4, 1, 1, 8, 12, 12, 12, 14, 7, 8, 1, 8, 12, 4, 0, 3, 7, 10, 3, 3, 5, 3, 7, 7, 7,
                          5, 7, 7, 5, 4, 8, 1, 7, 4, 4, 4, 2, 7, 7, 2, 0, 6, 0, 8, 6, 6, 0, 9, 3, 3, 1, 6, 1, 4, 8, 4,
                          0, 4, 10, 0, 4, 4, 6, 8, 3, 9, 6, 13, 4, 0, 8, 6, 10, 0, 3, 3, 6, 1, 0, 0, 5, 0, 0, 1, 0, 3,
                          4, 8, 8, 9, 4, 6, 11, 16, 22, 11, 33, 37, 48, 48, 56, 56, 52, 43, 32, 24, 14, 8, 19, 17, 5, 9,
                          7, 7, 6, 12, 5, 5, 2, 1, 8, 4, 8, 0, 1, 1, 8, 4, 6, 11, 0, 6, 13, 10, 13, 12, 9, 5, 3, 0, 6,
                          1, 9, 0, 7, 7, 7, 11, 3, 3, 0, 3, 3, 8, 5, 3, 5, 1, 8, 3, 6, 6, 0, 8, 6, 1, 6]])
        expected = np.array([[43, 67, 24, 35, 0, 0, 31, 8, 0, 31, 8, 8, 0, 0, 63, 31, 0, 55, 43, 0, 43, 43, 47, 0, 43,
                              20, 8, 39, 4, 0, 0, 4, 27, 0, 4, 27, 4, 24, 12, 0, 8, 75, 8, 55, 47, 0, 59, 16, 0, 16, 4,
                              39, 4, 39, 39, 82, 43, 39, 47, 24, 35, 51, 27, 4, 27, 51, 0, 4, 4, 4, 8, 0, 27, 75, 75,
                              51, 82, 0, 0, 0, 71, 0, 63, 0, 0, 0, 0, 67, 51, 39, 0, 51, 39, 71, 71, 0, 43, 55, 43, 43,
                              20, 0, 0, 55, 8, 43, 43, 27, 4, 8, 63, 31, 0, 0, 20, 8, 0, 51, 51, 16, 39, 16, 16, 27, 0,
                              0, 0, 16, 0, 0, 27, 0, 27, 27, 39, 0, 27, 51, 0, 16, 16, 27, 27, 27, 0, 16, 59, 67, 0, 0,
                              12, 27, 51, 63, 12, 59, 35, 35, 35, 24, 24, 35, 47, 24, 20, 0, 20, 0, 20, 43, 16, 0, 16,
                              16, 0, 24, 27, 43, 0, 43, 0, 55, 63, 16, 39, 0, 59, 47, 24, 12, 35, 35, 0, 0, 0, 35, 67,
                              75, 55, 20, 0],
                             [8, 0, 47, 0, 8, 31, 31, 31, 20, 20, 55, 55, 8, 8, 0, 0, 20, 0, 43, 55, 20, 0, 24, 0, 0, 0,
                              8, 16, 0, 59, 67, 4, 4, 51, 0, 51, 0, 0, 0, 67, 47, 47, 47, 0, 47, 0, 51, 4, 4, 39, 0, 4,
                              4, 27, 75, 16, 0, 16, 59, 0, 0, 0, 4, 0, 4, 0, 27, 27, 27, 63, 55, 75, 51, 75, 51, 0, 51,
                              75, 0, 47, 0, 16, 0, 75, 0, 0, 35, 0, 16, 51, 16, 0, 39, 20, 0, 20, 0, 20, 20, 20, 43, 20,
                              31, 31, 0, 43, 0, 39, 27, 0, 0, 0, 20, 20, 55, 43, 16, 39, 0, 0, 16, 16, 16, 0, 59, 0, 27,
                              75, 27, 27, 27, 0, 27, 27, 39, 39, 63, 51, 0, 16, 27, 27, 27, 0, 0, 51, 27, 67, 0, 0, 12,
                              27, 27, 0, 59, 12, 12, 35, 35, 0, 0, 59, 12, 24, 43, 20, 67, 8, 20, 43, 16, 90, 63, 0, 0,
                              0, 0, 43, 0, 43, 43, 0, 16, 86, 86, 55, 0, 0, 24, 59, 35, 35, 35, 35, 0, 0, 59, 31, 0, 47,
                              12],
                             [31, 24, 12, 0, 20, 20, 31, 31, 31, 20, 20, 0, 0, 8, 0, 43, 0, 0, 0, 0, 0, 47, 47, 20, 55,
                              43, 63, 4, 16, 12, 12, 0, 51, 4, 51, 27, 0, 59, 0, 47, 0, 24, 24, 0, 0, 4, 4, 0, 0, 27, 0,
                              27, 39, 27, 51, 27, 0, 0, 0, 0, 24, 43, 0, 27, 51, 4, 4, 0, 0, 0, 0, 51, 0, 47, 0, 20, 0,
                              75, 27, 0, 47, 0, 67, 20, 0, 35, 16, 39, 51, 39, 16, 51, 20, 43, 20, 0, 0, 20, 0, 20, 43,
                              0, 31, 0, 8, 43, 63, 39, 4, 0, 63, 55, 55, 55, 20, 43, 0, 39, 0, 0, 0, 0, 51, 4, 51, 27,
                              39, 0, 0, 47, 27, 63, 51, 51, 51, 27, 63, 63, 63, 0, 16, 0, 27, 16, 0, 16, 16, 51, 47, 24,
                              16, 0, 75, 27, 35, 35, 0, 35, 35, 35, 0, 47, 0, 0, 0, 31, 67, 31, 31, 75, 27, 16, 51, 0,
                              27, 75, 35, 35, 0, 0, 43, 16, 39, 39, 63, 78, 78, 82, 24, 35, 24, 35, 24, 0, 0, 12, 0, 67,
                              12, 0, 59],
                             [55, 47, 47, 0, 31, 55, 20, 20, 0, 20, 31, 0, 20, 0, 0, 0, 20, 43, 0, 43, 0, 0, 20, 20, 43,
                              0, 39, 39, 4, 0, 35, 27, 27, 51, 51, 0, 47, 0, 59, 24, 24, 59, 47, 47, 4, 0, 4, 4, 4, 4,
                              51, 0, 4, 27, 51, 75, 35, 0, 71, 43, 0, 20, 0, 0, 16, 0, 0, 4, 0, 51, 16, 0, 35, 31, 0, 0,
                              20, 20, 51, 51, 82, 0, 0, 67, 47, 0, 51, 16, 0, 16, 51, 0, 20, 0, 0, 43, 43, 20, 20, 0, 0,
                              31, 55, 43, 43, 4, 63, 71, 39, 39, 0, 78, 55, 20, 43, 55, 55, 0, 16, 16, 0, 16, 0, 27, 55,
                              31, 63, 27, 4, 39, 71, 16, 51, 0, 0, 0, 27, 16, 0, 16, 63, 16, 0, 16, 0, 51, 0, 16, 59, 0,
                              0, 27, 27, 51, 82, 24, 0, 24, 59, 35, 0, 55, 86, 0, 55, 8, 0, 31, 43, 8, 16, 16, 16, 16,
                              0, 31, 67, 12, 0, 43, 0, 16, 39, 39, 86, 47, 12, 51, 35, 0, 24, 35, 0, 31, 0, 0, 20, 0,
                              24, 0, 35],
                             [20, 0, 0, 0, 0, 20, 20, 55, 43, 43, 0, 55, 4, 20, 43, 0, 0, 0, 43, 0, 24, 0, 43, 20, 55,
                              0, 4, 0, 71, 27, 39, 0, 4, 4, 47, 0, 24, 0, 24, 0, 59, 0, 24, 55, 4, 4, 4, 27, 4, 0, 0,
                              55, 55, 55, 31, 78, 47, 12, 0, 0, 0, 75, 43, 20, 0, 0, 71, 24, 0, 16, 59, 59, 55, 16, 51,
                              0, 0, 20, 0, 16, 27, 16, 24, 35, 12, 47, 0, 16, 0, 16, 55, 20, 20, 0, 0, 43, 0, 20, 55, 0,
                              43, 20, 71, 27, 27, 39, 35, 35, 12, 35, 0, 20, 43, 0, 55, 43, 20, 43, 20, 0, 0, 0, 0, 31,
                              55, 20, 4, 0, 27, 27, 51, 0, 16, 51, 0, 51, 16, 16, 20, 90, 0, 43, 27, 27, 16, 27, 16, 0,
                              4, 0, 0, 27, 0, 0, 24, 0, 24, 35, 35, 0, 0, 86, 39, 39, 16, 0, 0, 31, 55, 0, 0, 16, 0, 8,
                              31, 31, 31, 0, 43, 0, 63, 16, 0, 63, 39, 24, 12, 0, 43, 0, 24, 59, 0, 0, 0, 0, 0, 0, 0,
                              35, 59],
                             [0, 55, 31, 47, 0, 20, 63, 0, 0, 24, 20, 63, 0, 0, 20, 0, 20, 0, 24, 24, 24, 0, 0, 20, 20,
                              63, 51, 27, 63, 4, 0, 0, 63, 47, 24, 0, 59, 0, 47, 0, 59, 47, 59, 0, 27, 27, 51, 4, 0, 27,
                              0, 31, 0, 31, 47, 47, 20, 71, 12, 12, 0, 35, 82, 24, 24, 12, 47, 31, 0, 39, 59, 24, 27,
                              27, 16, 27, 0, 0, 43, 75, 27, 27, 0, 51, 12, 0, 51, 16, 39, 43, 43, 20, 20, 43, 43, 43,
                              71, 43, 20, 43, 0, 27, 35, 71, 35, 47, 35, 12, 0, 0, 8, 43, 43, 0, 20, 43, 20, 20, 20, 20,
                              43, 0, 55, 20, 20, 20, 39, 39, 27, 16, 27, 27, 0, 27, 0, 0, 16, 16, 43, 0, 75, 75, 0, 16,
                              16, 16, 0, 27, 0, 51, 27, 24, 0, 0, 0, 24, 24, 0, 24, 8, 39, 71, 63, 0, 63, 39, 8, 31, 31,
                              31, 0, 16, 0, 0, 0, 31, 67, 0, 0, 0, 67, 0, 16, 16, 63, 24, 75, 51, 75, 67, 12, 8, 31, 0,
                              0, 0, 63, 0, 31, 0, 0],
                             [0, 8, 82, 0, 8, 0, 4, 51, 31, 0, 8, 55, 0, 0, 0, 55, 55, 0, 47, 47, 47, 24, 24, 0, 51, 0,
                              0, 20, 0, 31, 39, 0, 55, 8, 31, 67, 8, 55, 0, 47, 24, 24, 0, 8, 59, 27, 4, 4, 4, 27, 0,
                              55, 43, 47, 12, 35, 47, 86, 47, 12, 0, 35, 71, 0, 0, 0, 20, 55, 55, 39, 39, 67, 0, 16, 16,
                              51, 51, 16, 20, 0, 27, 0, 0, 0, 0, 51, 0, 43, 0, 43, 43, 55, 55, 20, 20, 20, 51, 39, 0,
                              51, 16, 0, 51, 12, 12, 47, 0, 12, 31, 31, 20, 0, 20, 0, 0, 0, 43, 0, 55, 0, 0, 31, 55, 0,
                              20, 4, 0, 0, 4, 0, 51, 16, 16, 39, 39, 39, 63, 0, 43, 8, 0, 0, 63, 20, 55, 20, 0, 4, 4, 0,
                              27, 0, 0, 27, 0, 27, 82, 8, 0, 63, 0, 4, 39, 16, 0, 0, 0, 31, 0, 0, 63, 16, 55, 0, 31, 8,
                              31, 31, 0, 20, 0, 0, 16, 0, 16, 16, 16, 16, 27, 67, 0, 63, 16, 63, 16, 16, 55, 55, 8, 24,
                              24],
                             [8, 0, 55, 8, 8, 0, 67, 8, 31, 31, 55, 8, 4, 63, 51, 63, 43, 0, 20, 0, 82, 31, 0, 0, 0, 43,
                              55, 0, 20, 20, 20, 63, 0, 67, 31, 8, 0, 0, 0, 67, 47, 47, 0, 4, 0, 0, 4, 4, 51, 4, 8, 71,
                              35, 12, 12, 47, 47, 0, 59, 12, 0, 35, 12, 43, 43, 31, 0, 20, 0, 39, 39, 0, 51, 16, 16, 27,
                              51, 0, 31, 20, 0, 27, 27, 43, 0, 16, 63, 63, 43, 20, 20, 43, 0, 20, 0, 51, 16, 0, 16, 16,
                              39, 16, 16, 0, 35, 47, 12, 12, 20, 0, 20, 43, 43, 55, 43, 55, 0, 43, 20, 43, 43, 20, 20,
                              55, 0, 16, 27, 27, 63, 51, 27, 27, 16, 16, 16, 0, 16, 8, 67, 8, 63, 16, 16, 16, 4, 4, 0,
                              4, 51, 4, 78, 78, 59, 16, 75, 43, 8, 20, 43, 0, 16, 16, 16, 0, 16, 16, 31, 0, 39, 0, 0, 0,
                              8, 0, 8, 31, 31, 31, 0, 20, 43, 20, 20, 0, 0, 16, 0, 47, 24, 39, 20, 16, 16, 63, 0, 55,
                              55, 8, 31, 24, 12],
                             [8, 67, 8, 0, 31, 55, 0, 0, 55, 8, 55, 55, 27, 4, 4, 43, 20, 0, 27, 0, 4, 8, 31, 4, 20, 0,
                              0, 43, 55, 55, 43, 20, 63, 31, 8, 0, 31, 8, 8, 31, 0, 8, 55, 43, 39, 16, 16, 39, 51, 0,
                              27, 63, 35, 0, 35, 35, 59, 59, 35, 0, 35, 12, 12, 31, 16, 0, 0, 4, 4, 63, 27, 59, 35, 27,
                              27, 27, 16, 27, 0, 31, 51, 51, 0, 75, 20, 20, 0, 63, 0, 0, 20, 20, 43, 43, 16, 0, 0, 0, 0,
                              39, 63, 0, 0, 0, 24, 0, 35, 78, 20, 0, 55, 0, 0, 0, 43, 20, 0, 55, 20, 71, 63, 16, 0, 71,
                              51, 51, 0, 0, 27, 0, 0, 16, 0, 0, 16, 0, 0, 31, 31, 16, 16, 16, 16, 0, 16, 63, 63, 51, 51,
                              4, 51, 0, 0, 0, 0, 67, 8, 0, 0, 16, 0, 4, 0, 71, 4, 0, 55, 75, 39, 0, 0, 0, 75, 8, 8, 8,
                              67, 31, 0, 43, 20, 0, 20, 75, 16, 0, 24, 24, 71, 4, 39, 0, 39, 0, 0, 0, 8, 0, 35, 24, 24],
                             [0, 31, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 51, 71, 20, 20, 0, 0, 20, 63, 27, 78, 0, 27, 43, 20,
                              20, 43, 55, 0, 43, 55, 55, 51, 0, 55, 31, 0, 0, 8, 8, 75, 0, 0, 16, 39, 0, 0, 0, 16, 63,
                              27, 39, 47, 12, 35, 67, 35, 71, 35, 0, 35, 0, 0, 0, 55, 63, 4, 27, 0, 39, 0, 35, 24, 51,
                              0, 27, 27, 0, 0, 0, 27, 16, 20, 0, 0, 31, 8, 8, 0, 39, 63, 39, 63, 0, 63, 39, 39, 16, 0,
                              16, 63, 0, 16, 51, 47, 0, 12, 0, 20, 0, 0, 43, 20, 43, 71, 12, 24, 24, 12, 12, 78, 24, 16,
                              16, 16, 0, 0, 51, 16, 0, 39, 63, 16, 16, 39, 0, 0, 31, 16, 63, 16, 16, 4, 39, 0, 27, 27,
                              51, 0, 27, 4, 0, 0, 43, 20, 20, 8, 31, 20, 16, 4, 4, 16, 0, 0, 16, 4, 0, 67, 0, 0, 55, 0,
                              0, 8, 0, 35, 0, 59, 67, 20, 43, 0, 0, 12, 16, 4, 16, 4, 39, 16, 16, 0, 8, 8, 8, 47, 0, 12,
                              24],
                             [0, 55, 0, 0, 4, 51, 4, 8, 31, 0, 8, 12, 4, 20, 20, 43, 55, 20, 0, 55, 43, 0, 43, 27, 55,
                              20, 0, 0, 43, 0, 20, 20, 20, 0, 43, 0, 0, 8, 31, 31, 55, 27, 55, 0, 39, 51, 39, 0, 0, 0,
                              39, 0, 4, 39, 71, 78, 43, 35, 35, 35, 0, 12, 27, 0, 27, 78, 0, 4, 4, 4, 0, 39, 4, 0, 0,
                              59, 63, 39, 24, 51, 27, 75, 20, 43, 0, 43, 31, 31, 8, 8, 0, 39, 16, 0, 63, 16, 39, 63, 63,
                              0, 0, 78, 16, 0, 75, 75, 35, 0, 63, 16, 20, 20, 20, 63, 16, 0, 78, 12, 0, 0, 0, 24, 0, 16,
                              27, 0, 16, 0, 16, 16, 0, 39, 0, 0, 16, 63, 16, 16, 55, 16, 16, 4, 4, 4, 16, 63, 0, 0, 51,
                              16, 4, 16, 59, 20, 0, 20, 31, 55, 8, 31, 43, 16, 0, 4, 51, 4, 16, 0, 16, 0, 16, 8, 67, 8,
                              55, 8, 24, 59, 0, 12, 12, 0, 0, 0, 0, 16, 4, 39, 63, 63, 16, 39, 4, 63, 8, 31, 0, 0, 59,
                              0, 24],
                             [43, 43, 43, 43, 63, 0, 20, 43, 67, 67, 31, 0, 0, 20, 0, 43, 20, 0, 43, 0, 43, 0, 20, 0, 0,
                              43, 55, 55, 20, 20, 20, 20, 0, 20, 75, 20, 0, 0, 20, 55, 16, 51, 0, 12, 16, 0, 16, 16, 39,
                              12, 27, 4, 27, 63, 63, 63, 67, 27, 8, 67, 0, 0, 27, 16, 0, 35, 0, 27, 4, 4, 71, 63, 39,
                              39, 0, 35, 67, 0, 59, 0, 0, 63, 0, 39, 0, 8, 31, 31, 8, 31, 31, 63, 39, 39, 63, 71, 24,
                              39, 0, 0, 24, 24, 0, 0, 27, 0, 0, 12, 35, 39, 63, 0, 0, 63, 0, 12, 16, 39, 16, 24, 24, 0,
                              39, 0, 39, 0, 0, 0, 0, 16, 16, 63, 0, 39, 0, 39, 16, 39, 39, 39, 16, 4, 16, 0, 16, 0, 4,
                              4, 16, 0, 16, 51, 12, 20, 0, 67, 67, 67, 8, 55, 31, 4, 4, 39, 0, 55, 8, 31, 4, 16, 55, 8,
                              67, 31, 31, 24, 35, 59, 35, 59, 35, 59, 43, 20, 0, 16, 0, 63, 63, 16, 16, 4, 39, 16, 0,
                              55, 55, 47, 24, 47, 0],
                             [55, 0, 43, 20, 47, 35, 35, 20, 0, 24, 20, 27, 0, 0, 0, 43, 0, 20, 20, 43, 20, 43, 20, 0,
                              20, 78, 55, 20, 0, 20, 20, 43, 20, 0, 31, 20, 31, 20, 20, 31, 31, 75, 16, 35, 35, 39, 39,
                              0, 0, 59, 27, 0, 0, 39, 0, 63, 12, 51, 75, 43, 20, 0, 27, 0, 27, 0, 0, 35, 59, 35, 51, 27,
                              0, 59, 27, 0, 16, 16, 20, 24, 0, 16, 39, 39, 43, 31, 31, 0, 8, 8, 0, 0, 16, 0, 16, 0, 0,
                              0, 12, 0, 12, 75, 0, 75, 20, 31, 8, 78, 35, 12, 16, 0, 0, 24, 12, 39, 16, 0, 39, 0, 0, 75,
                              63, 16, 16, 0, 0, 0, 63, 0, 0, 0, 0, 39, 0, 63, 16, 39, 16, 4, 39, 0, 16, 39, 16, 4, 39,
                              39, 39, 4, 39, 71, 12, 4, 31, 24, 31, 8, 8, 67, 39, 4, 4, 71, 4, 0, 55, 31, 31, 82, 8, 67,
                              31, 67, 55, 0, 35, 0, 0, 12, 0, 12, 55, 75, 0, 4, 16, 16, 16, 4, 16, 16, 39, 0, 4, 0, 31,
                              24, 47, 4, 16],
                             [20, 0, 55, 35, 35, 35, 35, 12, 20, 0, 20, 4, 20, 0, 43, 20, 0, 55, 0, 0, 20, 8, 0, 0, 20,
                              0, 0, 20, 20, 20, 0, 43, 0, 47, 0, 55, 0, 31, 55, 31, 31, 4, 27, 4, 0, 0, 59, 59, 0, 0, 0,
                              27, 27, 63, 0, 59, 35, 59, 39, 0, 20, 8, 0, 16, 27, 16, 0, 0, 0, 51, 16, 0, 16, 27, 27, 0,
                              0, 27, 51, 43, 63, 0, 0, 16, 0, 8, 43, 0, 31, 8, 31, 43, 0, 63, 47, 0, 47, 12, 12, 0, 0,
                              16, 43, 20, 43, 8, 43, 67, 0, 0, 59, 71, 0, 0, 39, 39, 39, 0, 0, 63, 0, 51, 0, 16, 39, 16,
                              0, 16, 16, 39, 16, 39, 63, 63, 16, 0, 63, 4, 63, 71, 0, 4, 39, 63, 63, 16, 16, 16, 4, 0,
                              39, 12, 4, 63, 51, 0, 24, 24, 0, 8, 55, 0, 8, 63, 4, 0, 31, 67, 8, 8, 55, 55, 55, 31, 0,
                              0, 35, 0, 24, 35, 35, 0, 55, 0, 16, 63, 0, 39, 4, 4, 39, 63, 39, 4, 4, 4, 63, 0, 16, 4,
                              16],
                             [43, 20, 47, 47, 35, 35, 0, 47, 35, 55, 0, 0, 43, 55, 0, 0, 43, 43, 0, 43, 20, 20, 31, 55,
                              31, 20, 0, 0, 0, 0, 20, 0, 24, 51, 0, 0, 31, 20, 31, 0, 4, 27, 78, 27, 27, 59, 24, 47, 39,
                              31, 20, 20, 43, 67, 35, 35, 35, 59, 35, 12, 78, 0, 51, 16, 75, 27, 0, 47, 16, 16, 16, 0,
                              16, 51, 51, 27, 27, 27, 51, 0, 39, 63, 0, 16, 63, 0, 55, 55, 8, 31, 43, 43, 63, 71, 71,
                              16, 75, 27, 0, 16, 0, 31, 43, 8, 0, 43, 63, 16, 63, 12, 35, 12, 12, 0, 0, 0, 0, 16, 39,
                              39, 43, 75, 20, 0, 16, 16, 16, 16, 63, 0, 39, 0, 0, 4, 63, 16, 63, 39, 4, 4, 39, 0, 0, 71,
                              0, 16, 0, 0, 39, 16, 35, 16, 0, 0, 27, 75, 12, 71, 47, 8, 8, 31, 55, 31, 39, 16, 16, 0,
                              31, 31, 31, 31, 0, 31, 31, 0, 0, 24, 0, 35, 24, 8, 0, 0, 0, 16, 39, 4, 4, 16, 63, 39, 4,
                              16, 39, 39, 0, 12, 35, 63, 0],
                             [55, 55, 0, 0, 35, 35, 12, 35, 0, 20, 20, 20, 78, 20, 0, 20, 43, 43, 0, 31, 20, 20, 0, 55,
                              20, 31, 27, 0, 20, 43, 12, 0, 27, 75, 0, 20, 20, 20, 27, 4, 39, 0, 0, 63, 4, 51, 63, 63,
                              67, 67, 67, 0, 24, 0, 27, 0, 0, 55, 20, 4, 27, 27, 0, 24, 47, 0, 24, 16, 16, 63, 43, 16,
                              27, 16, 0, 0, 16, 16, 16, 27, 71, 63, 39, 16, 16, 16, 31, 8, 0, 8, 8, 16, 12, 0, 0, 16, 0,
                              0, 39, 75, 31, 0, 0, 0, 0, 63, 16, 16, 0, 16, 59, 71, 24, 12, 0, 43, 16, 39, 0, 43, 67,
                              16, 20, 43, 16, 16, 39, 39, 55, 71, 39, 0, 4, 0, 39, 63, 16, 16, 16, 0, 4, 0, 0, 35, 12,
                              4, 16, 16, 0, 12, 0, 12, 0, 0, 27, 55, 0, 59, 24, 31, 0, 0, 31, 0, 55, 16, 0, 39, 67, 31,
                              55, 0, 31, 0, 31, 31, 78, 35, 24, 59, 55, 55, 0, 67, 4, 55, 16, 63, 16, 31, 31, 31, 8, 0,
                              16, 4, 0, 0, 39, 16, 71],
                             [0, 20, 35, 35, 0, 35, 47, 47, 12, 43, 0, 20, 0, 0, 0, 55, 0, 31, 0, 43, 55, 0, 55, 55, 20,
                              63, 27, 16, 0, 0, 0, 82, 27, 0, 16, 55, 55, 0, 63, 39, 27, 4, 0, 71, 27, 51, 0, 0, 43, 31,
                              59, 47, 47, 24, 27, 27, 31, 75, 27, 51, 75, 16, 0, 47, 59, 24, 59, 86, 67, 0, 43, 51, 16,
                              27, 27, 27, 0, 16, 16, 0, 20, 63, 16, 0, 63, 0, 16, 55, 8, 35, 0, 12, 12, 27, 27, 16, 16,
                              27, 90, 67, 8, 31, 8, 67, 16, 16, 39, 16, 63, 16, 71, 12, 47, 24, 24, 0, 0, 67, 20, 0, 51,
                              0, 16, 0, 0, 31, 0, 8, 67, 16, 16, 16, 39, 4, 16, 63, 39, 0, 4, 16, 16, 4, 63, 12, 12, 0,
                              12, 75, 0, 59, 0, 0, 0, 12, 82, 20, 0, 59, 0, 8, 31, 8, 31, 8, 31, 16, 16, 16, 55, 67, 67,
                              8, 8, 8, 82, 27, 4, 31, 47, 71, 0, 8, 8, 82, 63, 4, 8, 0, 8, 31, 8, 31, 8, 31, 78, 67, 67,
                              16, 39, 71, 0],
                             [20, 43, 20, 0, 12, 0, 12, 35, 55, 0, 20, 0, 0, 0, 43, 43, 31, 24, 24, 43, 20, 31, 0, 0, 0,
                              0, 39, 0, 8, 67, 0, 0, 0, 16, 82, 75, 27, 0, 27, 63, 4, 39, 39, 59, 27, 51, 0, 20, 31, 35,
                              35, 59, 24, 24, 16, 51, 43, 75, 51, 75, 27, 67, 0, 51, 0, 47, 0, 0, 0, 20, 24, 0, 51, 16,
                              27, 27, 0, 51, 20, 0, 59, 20, 63, 16, 16, 78, 0, 71, 24, 0, 75, 0, 16, 16, 16, 16, 16, 0,
                              31, 0, 0, 31, 8, 0, 0, 16, 63, 0, 16, 39, 39, 24, 24, 12, 0, 75, 16, 16, 0, 0, 31, 16, 27,
                              0, 43, 43, 0, 31, 0, 0, 27, 0, 39, 4, 4, 0, 4, 0, 16, 67, 27, 0, 0, 16, 63, 63, 51, 59,
                              35, 12, 35, 0, 12, 12, 12, 55, 0, 47, 24, 59, 55, 31, 55, 8, 31, 16, 16, 63, 31, 8, 31,
                              55, 8, 31, 31, 31, 0, 0, 20, 35, 24, 8, 4, 4, 27, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 43, 39,
                              0, 86, 0, 0],
                             [20, 0, 0, 0, 35, 0, 47, 0, 55, 55, 0, 0, 0, 12, 63, 31, 24, 0, 0, 31, 0, 20, 20, 20, 4, 4,
                              16, 67, 20, 67, 0, 63, 75, 16, 51, 0, 0, 27, 27, 27, 27, 4, 59, 0, 16, 27, 39, 0, 0, 0,
                              24, 47, 0, 0, 0, 0, 31, 4, 43, 75, 27, 67, 43, 20, 0, 8, 24, 0, 0, 0, 20, 20, 8, 27, 51,
                              16, 27, 0, 8, 43, 20, 59, 0, 4, 16, 63, 63, 39, 24, 12, 16, 51, 0, 16, 0, 0, 20, 67, 31,
                              55, 67, 67, 31, 8, 16, 16, 16, 39, 0, 39, 0, 39, 47, 24, 12, 39, 63, 0, 0, 0, 51, 31, 75,
                              0, 51, 75, 31, 20, 20, 8, 27, 27, 27, 71, 4, 16, 4, 39, 0, 82, 75, 4, 4, 4, 4, 51, 0, 0,
                              35, 12, 0, 12, 0, 12, 71, 43, 0, 59, 47, 24, 0, 55, 8, 31, 0, 16, 4, 8, 55, 8, 0, 67, 31,
                              0, 0, 24, 55, 27, 43, 55, 0, 27, 51, 4, 27, 51, 24, 47, 24, 0, 59, 59, 0, 55, 8, 47, 0,
                              16, 24, 78, 55],
                             [20, 43, 63, 0, 59, 0, 0, 43, 55, 43, 43, 71, 27, 27, 75, 0, 59, 59, 35, 59, 8, 43, 43, 63,
                              4, 16, 43, 0, 20, 0, 16, 39, 16, 0, 31, 0, 0, 4, 39, 27, 39, 94, 0, 59, 27, 27, 43, 0, 67,
                              24, 24, 24, 24, 0, 51, 51, 8, 4, 12, 67, 51, 43, 67, 67, 78, 47, 24, 24, 0, 27, 71, 16,
                              43, 0, 43, 0, 20, 43, 8, 8, 20, 0, 0, 4, 39, 39, 39, 4, 39, 24, 27, 0, 27, 0, 0, 20, 0,
                              43, 31, 8, 8, 8, 31, 0, 55, 39, 63, 16, 39, 16, 16, 0, 16, 71, 27, 0, 27, 4, 4, 27, 4, 0,
                              16, 16, 27, 31, 0, 31, 55, 0, 0, 16, 39, 16, 0, 4, 0, 31, 0, 27, 51, 8, 31, 55, 75, 0, 0,
                              12, 12, 0, 35, 0, 71, 35, 12, 27, 43, 24, 59, 59, 24, 0, 8, 55, 16, 4, 31, 31, 31, 31, 0,
                              31, 24, 0, 47, 0, 24, 67, 0, 78, 0, 0, 51, 4, 4, 24, 24, 59, 59, 59, 47, 24, 0, 55, 31,
                              27, 63, 16, 16, 78, 55],
                             [0, 0, 8, 43, 8, 43, 43, 0, 20, 43, 16, 0, 16, 0, 27, 35, 35, 59, 35, 35, 35, 31, 31, 0,
                              63, 43, 67, 0, 0, 8, 8, 55, 0, 0, 31, 4, 27, 0, 63, 4, 63, 51, 71, 0, 0, 20, 0, 0, 20, 43,
                              47, 24, 12, 47, 0, 16, 12, 4, 47, 20, 67, 75, 43, 31, 0, 24, 0, 0, 0, 39, 59, 4, 39, 63,
                              63, 63, 43, 8, 8, 20, 43, 35, 35, 0, 4, 0, 39, 39, 39, 39, 0, 0, 16, 0, 75, 67, 20, 67, 0,
                              55, 16, 63, 0, 63, 63, 8, 0, 16, 63, 0, 0, 16, 39, 0, 27, 4, 27, 27, 27, 4, 4, 4, 59, 0,
                              0, 51, 4, 0, 20, 0, 8, 31, 4, 39, 0, 0, 27, 27, 75, 4, 55, 0, 0, 31, 0, 0, 39, 39, 71, 0,
                              4, 16, 35, 35, 4, 4, 51, 20, 24, 0, 47, 59, 59, 75, 31, 0, 55, 75, 8, 31, 31, 0, 0, 59,
                              59, 0, 0, 24, 0, 0, 0, 20, 27, 27, 24, 0, 24, 24, 24, 24, 0, 59, 24, 0, 4, 4, 0, 39, 63,
                              55, 55],
                             [0, 31, 31, 8, 31, 8, 8, 31, 71, 16, 16, 39, 39, 0, 12, 12, 35, 59, 59, 35, 35, 35, 0, 35,
                              8, 8, 67, 75, 31, 31, 0, 31, 8, 0, 31, 4, 4, 0, 51, 27, 51, 4, 0, 0, 0, 75, 8, 20, 67, 8,
                              43, 47, 0, 24, 16, 12, 35, 0, 59, 43, 43, 27, 55, 0, 8, 47, 0, 0, 16, 51, 35, 39, 16, 4,
                              0, 16, 4, 43, 20, 0, 35, 35, 8, 0, 16, 16, 63, 63, 4, 4, 12, 0, 82, 51, 0, 20, 0, 0, 0,
                              16, 4, 4, 16, 71, 16, 4, 0, 16, 0, 39, 39, 63, 16, 0, 27, 27, 51, 27, 4, 0, 12, 0, 31, 47,
                              59, 63, 27, 27, 0, 16, 0, 0, 0, 0, 0, 75, 27, 0, 75, 8, 0, 31, 0, 8, 16, 4, 63, 4, 0, 4,
                              4, 4, 0, 0, 27, 27, 0, 27, 4, 75, 4, 75, 75, 4, 51, 4, 0, 0, 55, 0, 0, 59, 47, 0, 0, 24,
                              0, 0, 78, 0, 55, 43, 0, 0, 24, 24, 47, 24, 47, 0, 0, 0, 0, 20, 20, 4, 63, 67, 16, 8, 55],
                             [16, 31, 8, 43, 0, 8, 0, 8, 63, 0, 39, 16, 63, 16, 39, 71, 12, 24, 24, 0, 35, 0, 24, 24, 0,
                              0, 0, 31, 8, 8, 55, 67, 43, 0, 20, 63, 27, 0, 0, 51, 27, 4, 27, 0, 0, 20, 55, 0, 75, 20,
                              67, 67, 43, 43, 67, 12, 35, 12, 31, 67, 67, 75, 39, 43, 31, 8, 0, 0, 0, 4, 51, 16, 39, 39,
                              16, 39, 16, 39, 8, 0, 12, 67, 31, 16, 16, 39, 0, 39, 0, 0, 71, 0, 0, 0, 4, 4, 0, 0, 4, 0,
                              0, 51, 4, 4, 0, 39, 39, 0, 39, 16, 16, 0, 63, 0, 51, 51, 0, 51, 0, 59, 31, 31, 75, 8, 47,
                              63, 27, 51, 63, 0, 39, 4, 31, 75, 4, 0, 0, 27, 51, 8, 8, 55, 31, 39, 4, 0, 0, 63, 16, 0,
                              16, 16, 0, 16, 0, 51, 4, 27, 4, 4, 75, 0, 0, 27, 0, 0, 75, 20, 20, 0, 8, 0, 47, 0, 24, 24,
                              24, 27, 20, 43, 0, 0, 78, 55, 24, 0, 24, 24, 24, 0, 47, 67, 43, 0, 0, 67, 0, 67, 0, 55,
                              0],
                             [39, 8, 43, 31, 43, 31, 43, 43, 0, 0, 63, 0, 39, 0, 16, 63, 63, 59, 35, 0, 59, 59, 24, 24,
                              67, 8, 0, 0, 55, 8, 67, 8, 43, 0, 75, 0, 51, 27, 4, 63, 51, 4, 27, 51, 4, 43, 0, 67, 55,
                              16, 0, 43, 8, 20, 0, 8, 35, 12, 0, 63, 63, 39, 63, 67, 31, 31, 71, 0, 39, 27, 51, 75, 63,
                              0, 16, 16, 4, 71, 8, 67, 31, 67, 8, 16, 4, 0, 4, 4, 16, 0, 16, 47, 24, 12, 59, 39, 51, 63,
                              4, 27, 63, 63, 4, 51, 0, 16, 0, 16, 0, 0, 16, 0, 63, 63, 51, 59, 63, 63, 59, 82, 0, 75,
                              55, 31, 31, 78, 63, 4, 27, 16, 16, 4, 35, 27, 27, 27, 0, 0, 51, 31, 55, 31, 0, 0, 39, 63,
                              16, 4, 39, 63, 39, 16, 16, 39, 4, 4, 51, 27, 75, 27, 0, 0, 51, 0, 27, 27, 55, 43, 43, 0,
                              43, 0, 0, 47, 24, 47, 0, 43, 0, 78, 20, 78, 43, 43, 55, 24, 47, 0, 0, 0, 20, 0, 0, 43, 43,
                              0, 20, 55, 82, 0, 71],
                             [59, 71, 8, 0, 8, 31, 8, 16, 16, 39, 0, 16, 16, 39, 39, 63, 31, 0, 35, 35, 35, 24, 24, 35,
                              55, 31, 31, 8, 0, 0, 8, 8, 20, 0, 43, 16, 16, 4, 51, 51, 27, 51, 4, 63, 27, 67, 20, 20,
                              43, 0, 0, 43, 0, 39, 16, 39, 0, 0, 0, 4, 75, 63, 71, 67, 67, 0, 71, 39, 16, 0, 27, 4, 0,
                              16, 16, 0, 12, 27, 39, 31, 8, 0, 31, 16, 16, 39, 16, 0, 4, 16, 16, 63, 24, 12, 0, 12, 43,
                              78, 63, 4, 27, 94, 4, 51, 51, 0, 27, 51, 0, 16, 51, 0, 51, 51, 47, 71, 59, 78, 75, 43, 67,
                              55, 55, 31, 31, 24, 31, 4, 4, 16, 0, 59, 12, 0, 51, 75, 27, 27, 0, 75, 0, 0, 8, 16, 4, 4,
                              4, 4, 39, 0, 27, 4, 27, 4, 27, 27, 4, 0, 4, 0, 0, 0, 51, 4, 51, 0, 24, 24, 0, 0, 20, 43,
                              75, 59, 24, 47, 55, 43, 20, 0, 20, 43, 0, 0, 20, 78, 27, 0, 39, 0, 0, 0, 20, 0, 0, 43, 43,
                              8, 0, 0, 12],
                             [0, 0, 47, 8, 8, 8, 0, 16, 0, 16, 39, 39, 39, 16, 63, 0, 8, 55, 24, 35, 24, 59, 35, 24, 31,
                              8, 8, 31, 8, 0, 55, 8, 43, 8, 4, 39, 0, 16, 4, 63, 51, 59, 63, 27, 8, 43, 67, 8, 0, 20,
                              16, 0, 71, 0, 0, 0, 71, 24, 0, 39, 63, 16, 24, 90, 31, 0, 4, 0, 0, 75, 0, 27, 75, 4, 75,
                              0, 0, 51, 0, 4, 8, 0, 55, 31, 4, 0, 4, 0, 4, 4, 16, 16, 47, 24, 78, 47, 0, 20, 31, 55, 55,
                              63, 63, 0, 51, 51, 27, 27, 27, 51, 27, 27, 51, 47, 59, 12, 31, 4, 59, 59, 0, 8, 31, 55,
                              55, 24, 59, 78, 55, 86, 0, 0, 59, 4, 27, 4, 0, 82, 4, 0, 75, 8, 0, 63, 39, 4, 16, 63, 39,
                              4, 4, 27, 4, 0, 82, 20, 43, 67, 51, 0, 51, 4, 4, 0, 35, 24, 12, 12, 47, 0, 20, 55, 20, 51,
                              27, 0, 78, 20, 43, 55, 43, 20, 0, 43, 20, 20, 67, 47, 35, 0, 43, 43, 67, 20, 0, 0, 82, 35,
                              27, 59, 35],
                             [0, 0, 12, 0, 0, 0, 39, 16, 0, 0, 63, 0, 39, 12, 63, 0, 8, 55, 35, 24, 35, 0, 24, 35, 67,
                              55, 0, 0, 8, 0, 31, 0, 8, 8, 16, 16, 16, 63, 27, 27, 78, 24, 27, 20, 20, 0, 67, 0, 0, 20,
                              0, 0, 20, 8, 20, 43, 55, 78, 0, 4, 27, 63, 0, 43, 67, 0, 43, 20, 20, 43, 27, 4, 4, 51, 27,
                              51, 4, 75, 51, 27, 82, 0, 71, 39, 63, 16, 0, 39, 16, 16, 16, 0, 0, 71, 55, 55, 43, 78, 78,
                              0, 8, 27, 51, 51, 16, 16, 27, 63, 27, 27, 51, 51, 86, 86, 94, 67, 0, 51, 27, 59, 59, 12,
                              55, 55, 59, 0, 0, 24, 0, 24, 71, 12, 0, 12, 12, 0, 8, 8, 8, 55, 8, 16, 0, 4, 4, 0, 0, 63,
                              39, 4, 27, 27, 82, 4, 4, 4, 43, 20, 43, 27, 27, 51, 51, 78, 71, 35, 35, 47, 71, 24, 71, 0,
                              43, 20, 43, 0, 78, 43, 0, 78, 78, 0, 24, 24, 47, 43, 20, 20, 67, 4, 82, 0, 0, 43, 20, 0,
                              39, 86, 51, 63, 0],
                             [0, 78, 20, 0, 71, 39, 63, 0, 0, 0, 0, 71, 12, 24, 71, 55, 8, 31, 59, 35, 59, 59, 0, 35,
                              55, 31, 0, 8, 82, 67, 8, 8, 0, 0, 16, 16, 0, 16, 4, 71, 71, 0, 75, 43, 20, 0, 0, 31, 0,
                              31, 4, 8, 8, 8, 20, 43, 55, 20, 8, 4, 27, 67, 43, 67, 35, 0, 43, 20, 20, 20, 55, 51, 27,
                              27, 27, 51, 4, 4, 27, 51, 0, 0, 0, 71, 0, 4, 0, 63, 39, 39, 39, 12, 71, 86, 43, 43, 0, 0,
                              8, 31, 78, 4, 16, 51, 51, 51, 51, 63, 0, 0, 78, 55, 31, 8, 102, 63, 0, 27, 51, 51, 0, 59,
                              12, 0, 47, 0, 47, 0, 71, 71, 24, 71, 12, 35, 12, 39, 16, 31, 8, 0, 27, 0, 27, 39, 4, 4, 0,
                              39, 0, 31, 0, 0, 63, 27, 27, 51, 0, 0, 0, 43, 27, 0, 43, 78, 35, 12, 0, 35, 12, 71, 47, 0,
                              24, 24, 0, 12, 0, 71, 47, 12, 12, 0, 71, 47, 71, 47, 78, 12, 35, 67, 0, 8, 31, 0, 0, 63,
                              4, 51, 31, 55, 0],
                             [20, 20, 31, 8, 31, 12, 0, 0, 16, 63, 0, 0, 0, 71, 24, 0, 31, 0, 24, 24, 0, 0, 0, 0, 55, 0,
                              4, 0, 31, 8, 0, 82, 67, 8, 43, 16, 39, 0, 39, 4, 0, 75, 20, 20, 20, 8, 43, 31, 31, 82, 0,
                              20, 8, 20, 43, 55, 20, 51, 24, 31, 90, 0, 20, 35, 35, 20, 20, 20, 0, 0, 0, 51, 4, 4, 51,
                              75, 27, 75, 0, 12, 35, 0, 0, 0, 71, 0, 39, 0, 39, 39, 71, 47, 47, 43, 0, 35, 71, 12, 78,
                              0, 51, 16, 39, 51, 16, 16, 0, 27, 78, 55, 31, 0, 51, 4, 51, 4, 27, 51, 0, 27, 27, 0, 0,
                              71, 47, 0, 0, 24, 24, 24, 71, 24, 0, 24, 16, 63, 4, 4, 8, 4, 27, 39, 39, 27, 0, 27, 39,
                              39, 31, 0, 31, 27, 0, 51, 4, 0, 27, 4, 0, 0, 86, 0, 16, 86, 0, 12, 47, 71, 35, 47, 47, 0,
                              24, 24, 24, 12, 12, 0, 47, 24, 24, 24, 0, 47, 0, 35, 35, 47, 0, 20, 67, 55, 4, 0, 27, 63,
                              51, 55, 8, 8, 8],
                             [55, 20, 67, 31, 8, 31, 12, 24, 0, 16, 39, 63, 0, 63, 71, 31, 0, 8, 35, 35, 35, 0, 59, 0,
                              0, 4, 16, 39, 0, 31, 8, 8, 0, 55, 67, 0, 4, 4, 63, 0, 0, 43, 47, 8, 20, 0, 8, 8, 20, 12,
                              0, 0, 20, 20, 20, 20, 16, 39, 47, 0, 0, 20, 106, 59, 59, 20, 20, 0, 43, 8, 8, 0, 75, 27,
                              4, 4, 4, 75, 35, 47, 0, 31, 35, 59, 35, 24, 47, 47, 71, 47, 71, 35, 35, 78, 4, 0, 16, 35,
                              71, 0, 63, 0, 16, 0, 16, 39, 39, 51, 27, 0, 78, 4, 51, 27, 51, 27, 0, 63, 31, 8, 0, 16, 4,
                              0, 0, 8, 0, 24, 0, 47, 24, 0, 0, 78, 39, 0, 0, 4, 0, 39, 0, 39, 39, 27, 4, 4, 4, 0, 31,
                              31, 4, 0, 27, 0, 0, 4, 27, 0, 67, 47, 0, 35, 35, 35, 12, 47, 35, 47, 12, 35, 12, 12, 71,
                              12, 35, 12, 47, 71, 0, 47, 0, 71, 47, 35, 12, 71, 20, 43, 20, 0, 43, 4, 4, 0, 27, 63, 31,
                              31, 31, 0, 55],
                             [31, 0, 31, 8, 31, 8, 55, 0, 55, 0, 0, 39, 16, 4, 39, 16, 31, 8, 0, 35, 35, 24, 31, 0, 16,
                              39, 4, 4, 16, 39, 0, 0, 8, 55, 31, 8, 20, 8, 43, 12, 20, 55, 35, 12, 0, 67, 43, 43, 39, 8,
                              71, 16, 20, 43, 20, 0, 0, 35, 24, 0, 63, 86, 55, 59, 35, 0, 20, 8, 0, 8, 82, 4, 4, 27, 51,
                              51, 27, 51, 47, 47, 94, 31, 67, 35, 71, 59, 71, 47, 24, 35, 35, 71, 47, 39, 39, 39, 16, 0,
                              71, 0, 24, 12, 67, 0, 4, 39, 0, 39, 63, 51, 16, 39, 16, 0, 63, 16, 51, 16, 0, 31, 55, 55,
                              0, 90, 63, 0, 0, 24, 24, 0, 47, 47, 24, 0, 0, 16, 63, 0, 0, 27, 27, 27, 4, 27, 63, 4, 27,
                              8, 55, 55, 24, 4, 4, 0, 4, 4, 4, 4, 43, 0, 12, 59, 35, 47, 0, 35, 12, 35, 47, 12, 47, 0,
                              35, 47, 12, 35, 47, 0, 0, 47, 12, 35, 0, 0, 47, 67, 20, 20, 0, 0, 0, 51, 4, 0, 51, 63, 31,
                              31, 55, 8, 8],
                             [0, 0, 8, 55, 31, 8, 31, 75, 0, 0, 0, 63, 16, 4, 63, 0, 0, 31, 0, 24, 59, 67, 0, 0, 16, 0,
                              39, 4, 4, 63, 0, 0, 0, 8, 8, 8, 20, 67, 12, 59, 71, 35, 63, 12, 12, 20, 20, 4, 39, 78, 39,
                              4, 39, 20, 20, 0, 59, 0, 0, 43, 51, 55, 31, 55, 24, 0, 43, 8, 0, 0, 31, 0, 0, 75, 4, 0,
                              51, 0, 39, 20, 86, 71, 43, 78, 71, 12, 59, 24, 71, 35, 12, 71, 16, 63, 16, 39, 4, 39, 8,
                              0, 47, 47, 24, 0, 55, 4, 0, 0, 51, 16, 39, 0, 16, 0, 39, 0, 4, 0, 31, 8, 31, 8, 31, 8, 55,
                              51, 24, 0, 0, 0, 0, 0, 24, 31, 16, 0, 16, 39, 0, 4, 39, 0, 27, 27, 0, 4, 0, 55, 0, 75, 24,
                              4, 63, 27, 63, 27, 0, 0, 20, 0, 0, 35, 35, 35, 47, 12, 0, 35, 47, 12, 0, 35, 78, 47, 71,
                              0, 67, 0, 0, 0, 71, 35, 0, 35, 20, 20, 20, 43, 43, 4, 0, 4, 63, 4, 27, 27, 0, 55, 31, 55,
                              55],
                             [8, 0, 8, 8, 31, 0, 55, 55, 8, 4, 16, 4, 16, 0, 55, 55, 55, 0, 8, 82, 55, 31, 0, 4, 4, 4,
                              0, 4, 39, 16, 55, 8, 0, 0, 0, 55, 31, 0, 0, 31, 12, 12, 71, 0, 0, 0, 4, 16, 0, 8, 0, 0,
                              39, 4, 4, 0, 12, 0, 43, 43, 51, 31, 55, 31, 78, 16, 0, 8, 0, 8, 8, 0, 0, 8, 51, 0, 0, 71,
                              67, 94, 94, 63, 94, 0, 20, 78, 59, 51, 0, 39, 0, 63, 4, 4, 0, 4, 4, 43, 8, 8, 47, 12, 35,
                              12, 12, 0, 16, 0, 39, 0, 0, 0, 16, 0, 16, 0, 16, 39, 31, 75, 8, 0, 0, 31, 8, 43, 0, 0, 24,
                              71, 71, 71, 0, 78, 39, 0, 0, 39, 31, 31, 0, 63, 39, 27, 63, 31, 8, 67, 20, 0, 0, 0, 0, 4,
                              0, 27, 27, 0, 43, 0, 35, 12, 71, 59, 35, 47, 12, 35, 12, 0, 47, 35, 20, 71, 12, 0, 20, 20,
                              67, 43, 71, 12, 0, 35, 20, 20, 43, 20, 4, 4, 27, 4, 51, 27, 4, 51, 0, 8, 8, 0, 20],
                             [0, 8, 0, 0, 0, 0, 55, 67, 0, 4, 4, 39, 63, 8, 8, 0, 8, 0, 8, 27, 51, 75, 55, 0, 63, 4, 39,
                              16, 0, 0, 0, 31, 31, 31, 0, 8, 43, 0, 0, 31, 0, 35, 12, 8, 31, 0, 16, 0, 8, 0, 31, 8, 31,
                              63, 0, 4, 35, 12, 20, 27, 86, 55, 78, 47, 27, 0, 27, 31, 0, 0, 8, 31, 31, 31, 31, 0, 71,
                              78, 110, 55, 102, 16, 71, 47, 35, 35, 51, 0, 39, 0, 16, 16, 39, 63, 16, 16, 39, 20, 20,
                              12, 43, 0, 8, 12, 35, 43, 0, 4, 16, 39, 16, 51, 0, 16, 4, 4, 16, 63, 0, 4, 4, 63, 0, 8, 8,
                              20, 20, 0, 0, 0, 0, 47, 0, 16, 16, 0, 39, 63, 8, 8, 8, 4, 4, 4, 0, 8, 20, 20, 39, 16, 39,
                              4, 78, 4, 63, 0, 0, 0, 20, 67, 71, 59, 59, 0, 35, 0, 35, 71, 71, 71, 47, 0, 43, 0, 0, 0,
                              0, 0, 0, 20, 20, 0, 71, 47, 20, 0, 0, 0, 27, 4, 0, 0, 27, 27, 27, 51, 71, 31, 55, 47, 0],
                             [0, 67, 31, 8, 67, 8, 8, 67, 0, 8, 0, 55, 8, 55, 31, 55, 31, 31, 59, 51, 27, 4, 0, 0, 8, 4,
                              0, 39, 8, 55, 47, 0, 59, 24, 0, 8, 43, 0, 31, 71, 59, 0, 12, 0, 0, 31, 27, 55, 31, 31, 8,
                              31, 0, 31, 27, 27, 39, 0, 16, 59, 55, 31, 0, 27, 27, 63, 0, 43, 31, 8, 4, 27, 4, 0, 0, 55,
                              0, 55, 86, 8, 39, 78, 39, 0, 71, 27, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 39, 0, 0, 43,
                              39, 51, 51, 20, 0, 0, 63, 4, 0, 39, 63, 63, 39, 39, 4, 16, 0, 4, 4, 16, 0, 16, 4, 0, 67,
                              43, 8, 75, 0, 24, 0, 0, 0, 16, 39, 16, 0, 0, 8, 55, 43, 67, 43, 0, 31, 0, 20, 0, 0, 0, 0,
                              16, 0, 0, 82, 20, 20, 20, 0, 20, 0, 0, 59, 71, 0, 59, 12, 12, 35, 12, 75, 27, 4, 0, 43, 0,
                              20, 0, 20, 0, 0, 0, 0, 0, 12, 0, 0, 0, 78, 24, 71, 47, 71, 0, 24, 24, 71, 0, 4, 4, 4],
                             [35, 24, 47, 0, 59, 24, 0, 67, 47, 0, 55, 31, 8, 8, 0, 31, 0, 0, 47, 75, 4, 4, 27, 4, 75,
                              75, 0, 31, 31, 8, 75, 20, 20, 82, 8, 8, 67, 55, 4, 27, 94, 51, 16, 47, 8, 8, 0, 8, 31, 0,
                              31, 8, 55, 8, 4, 27, 71, 0, 75, 0, 8, 47, 24, 0, 71, 0, 0, 31, 31, 39, 39, 27, 4, 0, 24,
                              86, 118, 71, 133, 86, 20, 27, 8, 63, 4, 0, 4, 4, 16, 0, 0, 4, 0, 16, 16, 39, 4, 63, 43,
                              16, 0, 39, 39, 0, 20, 20, 63, 4, 39, 63, 4, 16, 16, 16, 4, 63, 16, 16, 4, 63, 4, 16, 0,
                              16, 0, 39, 8, 75, 0, 16, 16, 8, 8, 4, 27, 63, 55, 0, 75, 0, 0, 43, 43, 20, 75, 16, 16, 63,
                              8, 55, 0, 55, 0, 51, 27, 20, 0, 43, 0, 67, 0, 20, 59, 35, 12, 71, 59, 12, 35, 0, 0, 20, 4,
                              27, 67, 67, 43, 20, 20, 20, 43, 20, 20, 0, 0, 0, 0, 43, 20, 27, 0, 4, 51, 12, 0, 63, 0,
                              27, 4, 0, 27],
                             [24, 47, 0, 0, 47, 47, 24, 0, 59, 0, 31, 0, 31, 8, 55, 8, 55, 31, 47, 0, 0, 27, 75, 0, 51,
                              0, 0, 27, 27, 4, 75, 4, 55, 39, 0, 12, 0, 0, 0, 4, 31, 4, 86, 71, 12, 0, 55, 0, 31, 31, 8,
                              31, 8, 55, 27, 0, 0, 24, 82, 47, 31, 71, 0, 0, 24, 39, 16, 0, 4, 4, 39, 63, 0, 47, 47,
                              110, 118, 86, 102, 78, 0, 0, 0, 0, 16, 39, 0, 63, 4, 39, 16, 4, 16, 16, 4, 8, 8, 55, 16,
                              16, 51, 16, 16, 0, 0, 0, 39, 39, 4, 4, 39, 16, 39, 0, 39, 16, 4, 4, 71, 4, 16, 0, 4, 63,
                              39, 39, 4, 8, 8, 43, 75, 8, 55, 0, 55, 0, 20, 67, 20, 0, 0, 43, 20, 43, 0, 0, 16, 39, 78,
                              55, 8, 78, 27, 75, 51, 0, 0, 0, 20, 20, 0, 20, 0, 12, 59, 94, 35, 59, 0, 43, 43, 20, 20,
                              4, 90, 0, 20, 43, 67, 0, 43, 43, 20, 43, 0, 43, 0, 0, 20, 20, 4, 0, 16, 67, 0, 43, 43, 0,
                              51, 27, 4],
                             [24, 24, 59, 47, 47, 24, 0, 47, 24, 24, 24, 0, 55, 8, 31, 8, 8, 0, 59, 43, 0, 0, 75, 0, 51,
                              75, 27, 27, 27, 51, 75, 27, 20, 4, 12, 12, 35, 0, 24, 31, 31, 86, 94, 78, 35, 12, 0, 0,
                              31, 0, 0, 31, 0, 63, 0, 12, 59, 0, 20, 71, 82, 51, 75, 43, 0, 0, 39, 16, 39, 0, 39, 0, 0,
                              24, 78, 24, 133, 67, 8, 71, 31, 0, 12, 39, 39, 4, 39, 16, 0, 63, 16, 55, 63, 0, 0, 8, 55,
                              8, 16, 39, 51, 0, 16, 16, 35, 39, 16, 16, 16, 39, 4, 39, 63, 4, 4, 0, 39, 16, 0, 12, 4, 4,
                              16, 0, 27, 4, 39, 39, 43, 31, 31, 31, 0, 8, 0, 20, 0, 0, 20, 43, 20, 20, 43, 0, 20, 20,
                              16, 0, 31, 31, 31, 0, 51, 51, 0, 20, 20, 0, 20, 20, 67, 0, 0, 75, 43, 35, 0, 35, 0, 67,
                              20, 43, 43, 20, 24, 0, 0, 20, 20, 0, 43, 0, 43, 20, 67, 0, 20, 20, 0, 0, 43, 0, 43, 67,
                              20, 20, 0, 27, 4, 0, 27],
                             [24, 24, 24, 59, 59, 24, 0, 0, 47, 0, 0, 47, 0, 8, 8, 31, 0, 47, 59, 20, 20, 0, 0, 4, 27,
                              75, 67, 20, 20, 27, 27, 4, 0, 27, 0, 0, 12, 0, 0, 8, 31, 0, 86, 55, 71, 35, 27, 20, 39,
                              31, 8, 8, 0, 39, 0, 39, 39, 12, 20, 67, 82, 82, 51, 0, 0, 16, 39, 0, 16, 39, 59, 20, 24,
                              78, 125, 78, 118, 94, 47, 24, 20, 59, 12, 16, 0, 63, 39, 4, 39, 39, 39, 0, 0, 8, 0, 31, 8,
                              55, 39, 4, 16, 51, 39, 67, 47, 16, 0, 63, 16, 39, 4, 63, 39, 0, 0, 39, 0, 4, 12, 35, 35,
                              16, 39, 39, 4, 39, 0, 27, 0, 55, 55, 8, 31, 0, 0, 0, 20, 20, 43, 43, 0, 67, 43, 20, 20,
                              43, 0, 0, 55, 55, 86, 20, 0, 0, 75, 75, 67, 0, 0, 43, 67, 43, 20, 0, 59, 59, 59, 0, 43, 0,
                              20, 43, 20, 20, 82, 4, 4, 0, 4, 75, 0, 67, 0, 0, 43, 20, 0, 43, 43, 67, 43, 43, 0, 43, 43,
                              20, 67, 67, 4, 27, 0],
                             [24, 47, 47, 0, 47, 0, 51, 51, 27, 0, 59, 47, 0, 59, 24, 59, 0, 47, 67, 0, 20, 51, 27, 31,
                              20, 43, 20, 20, 0, 43, 75, 67, 4, 0, 0, 4, 71, 55, 0, 39, 8, 71, 8, 51, 78, 12, 4, 4, 0,
                              39, 27, 4, 78, 0, 0, 39, 51, 0, 0, 67, 67, 47, 47, 43, 67, 67, 16, 16, 63, 0, 0, 16, 71,
                              86, 78, 102, 118, 86, 47, 12, 0, 0, 59, 27, 63, 4, 63, 0, 0, 63, 75, 67, 67, 0, 31, 31,
                              55, 8, 8, 16, 67, 0, 8, 8, 16, 0, 4, 16, 16, 16, 55, 0, 55, 4, 16, 16, 0, 4, 59, 59, 24,
                              86, 51, 0, 0, 0, 27, 27, 63, 43, 0, 20, 0, 67, 43, 4, 4, 27, 0, 0, 67, 43, 20, 0, 20, 0,
                              43, 0, 63, 39, 0, 0, 51, 16, 27, 16, 0, 43, 43, 0, 20, 0, 27, 27, 59, 0, 67, 0, 0, 43, 0,
                              0, 20, 0, 67, 4, 27, 0, 4, 4, 75, 20, 0, 20, 43, 0, 20, 0, 0, 20, 43, 0, 20, 43, 67, 0, 0,
                              67, 0, 27, 16],
                             [24, 47, 0, 0, 59, 0, 51, 27, 0, 16, 0, 47, 24, 0, 0, 0, 47, 20, 43, 12, 35, 82, 31, 43,
                              67, 43, 43, 43, 0, 43, 67, 20, 4, 51, 51, 4, 8, 8, 31, 16, 16, 8, 55, 94, 51, 12, 4, 4, 0,
                              43, 0, 0, 0, 4, 51, 0, 27, 0, 35, 35, 43, 47, 47, 98, 67, 43, 0, 0, 0, 0, 31, 0, 78, 118,
                              102, 133, 86, 78, 12, 0, 0, 12, 0, 27, 27, 27, 27, 16, 4, 8, 0, 0, 43, 8, 8, 55, 0, 31, 8,
                              0, 8, 8, 8, 35, 0, 4, 16, 4, 16, 0, 55, 55, 55, 55, 4, 0, 8, 75, 24, 24, 63, 27, 27, 27,
                              27, 0, 0, 27, 4, 43, 20, 43, 20, 0, 0, 51, 27, 27, 27, 20, 43, 20, 43, 0, 20, 0, 0, 63,
                              39, 0, 75, 90, 0, 0, 51, 16, 16, 0, 0, 0, 20, 0, 16, 0, 43, 43, 67, 67, 0, 43, 0, 67, 67,
                              0, 0, 43, 75, 43, 43, 20, 0, 0, 20, 20, 43, 43, 20, 90, 16, 0, 0, 0, 31, 31, 75, 0, 43,
                              20, 0, 0, 0],
                             [24, 0, 12, 24, 27, 0, 24, 47, 47, 12, 24, 0, 0, 47, 67, 0, 0, 43, 0, 0, 0, 8, 4, 4, 51, 0,
                              35, 12, 0, 20, 43, 20, 27, 0, 51, 31, 8, 31, 43, 4, 0, 39, 78, 82, 90, 90, 59, 4, 4, 0,
                              16, 55, 55, 0, 63, 51, 4, 27, 35, 12, 0, 90, 20, 75, 27, 43, 4, 47, 0, 4, 0, 67, 110, 125,
                              125, 75, 78, 31, 12, 35, 59, 0, 27, 63, 0, 63, 27, 0, 67, 0, 31, 67, 43, 20, 55, 8, 55,
                              31, 55, 0, 0, 24, 24, 4, 0, 0, 63, 4, 55, 0, 31, 31, 31, 0, 0, 31, 0, 4, 27, 0, 0, 51, 27,
                              51, 4, 4, 27, 63, 51, 20, 0, 43, 0, 27, 0, 51, 27, 27, 0, 0, 20, 43, 20, 0, 20, 0, 0, 0,
                              27, 16, 27, 16, 16, 20, 75, 0, 0, 51, 75, 0, 24, 0, 20, 67, 0, 20, 67, 0, 20, 0, 20, 0, 0,
                              0, 20, 20, 20, 20, 0, 0, 0, 0, 20, 67, 0, 0, 0, 39, 16, 0, 16, 63, 8, 75, 67, 20, 67, 67,
                              43, 0, 47],
                             [67, 67, 75, 0, 47, 12, 12, 0, 24, 24, 12, 63, 0, 67, 20, 0, 0, 20, 20, 67, 20, 39, 27, 4,
                              4, 27, 0, 27, 4, 67, 0, 0, 43, 51, 51, 8, 31, 31, 0, 27, 24, 31, 27, 27, 86, 67, 35, 24,
                              0, 47, 0, 8, 31, 55, 0, 63, 27, 27, 51, 12, 20, 75, 98, 51, 51, 0, 27, 27, 20, 71, 12, 43,
                              122, 102, 141, 122, 35, 8, 67, 59, 12, 0, 27, 4, 4, 27, 27, 67, 0, 67, 55, 0, 20, 0, 67,
                              55, 55, 0, 4, 0, 0, 0, 27, 27, 27, 4, 27, 0, 0, 55, 8, 55, 0, 0, 47, 8, 0, 27, 31, 43, 20,
                              0, 0, 27, 4, 4, 4, 51, 0, 43, 20, 0, 0, 27, 0, 4, 0, 0, 27, 0, 67, 67, 67, 67, 43, 90, 51,
                              16, 27, 16, 0, 0, 27, 0, 20, 0, 0, 16, 24, 8, 8, 0, 0, 0, 0, 24, 0, 0, 67, 0, 43, 20, 0,
                              0, 43, 43, 67, 43, 67, 0, 43, 0, 0, 0, 20, 20, 16, 16, 63, 16, 39, 71, 0, 20, 20, 43, 67,
                              67, 0, 20, 8],
                             [63, 16, 0, 0, 24, 12, 47, 47, 47, 47, 12, 35, 63, 43, 43, 0, 20, 0, 20, 67, 4, 4, 4, 51,
                              27, 4, 4, 27, 4, 27, 67, 20, 20, 0, 27, 31, 31, 0, 24, 39, 35, 35, 27, 106, 106, 106, 67,
                              0, 0, 0, 0, 8, 8, 55, 0, 4, 27, 4, 27, 0, 75, 75, 51, 51, 55, 8, 59, 0, 0, 0, 0, 78, 94,
                              125, 118, 141, 59, 20, 20, 59, 4, 4, 4, 0, 27, 4, 67, 20, 20, 0, 0, 75, 0, 20, 20, 0, 82,
                              0, 51, 0, 27, 4, 51, 27, 51, 75, 4, 63, 8, 55, 0, 55, 8, 24, 24, 4, 27, 75, 67, 20, 67,
                              20, 20, 67, 51, 4, 4, 20, 20, 20, 43, 20, 0, 51, 27, 78, 4, 51, 27, 27, 67, 20, 43, 20,
                              43, 20, 43, 27, 0, 0, 16, 39, 0, 20, 20, 0, 0, 24, 31, 55, 82, 35, 78, 51, 27, 0, 4, 27,
                              0, 67, 0, 0, 0, 35, 35, 20, 0, 0, 20, 43, 12, 0, 67, 0, 20, 0, 0, 16, 16, 16, 39, 16, 63,
                              67, 43, 0, 43, 67, 0, 67, 8],
                             [39, 43, 67, 12, 12, 47, 12, 12, 12, 47, 12, 0, 0, 0, 0, 20, 0, 43, 43, 0, 63, 0, 27, 27,
                              27, 51, 0, 4, 27, 0, 27, 20, 20, 67, 0, 67, 24, 47, 24, 0, 0, 43, 55, 75, 82, 0, 102, 47,
                              4, 0, 39, 31, 0, 55, 31, 55, 31, 0, 27, 0, 78, 67, 55, 102, 31, 78, 35, 59, 0, 20, 31, 20,
                              106, 157, 165, 86, 0, 16, 0, 4, 63, 27, 27, 0, 27, 20, 67, 43, 20, 67, 20, 43, 0, 27, 0,
                              0, 27, 0, 4, 75, 4, 51, 27, 75, 0, 75, 31, 0, 27, 0, 31, 55, 47, 0, 75, 43, 20, 20, 0, 0,
                              20, 0, 0, 67, 20, 43, 0, 43, 43, 43, 0, 67, 0, 67, 51, 43, 0, 0, 78, 20, 43, 0, 0, 0, 20,
                              0, 67, 0, 0, 0, 39, 39, 0, 0, 0, 43, 0, 8, 67, 82, 82, 86, 94, 94, 4, 75, 75, 27, 4, 0,
                              43, 67, 0, 0, 59, 0, 59, 0, 0, 20, 67, 43, 12, 20, 0, 67, 16, 39, 63, 16, 63, 39, 0, 20,
                              43, 43, 67, 67, 0, 0, 31],
                             [0, 43, 39, 35, 35, 12, 35, 0, 35, 12, 35, 35, 35, 67, 67, 43, 20, 43, 20, 31, 0, 0, 39,
                              63, 27, 4, 27, 4, 27, 4, 4, 0, 20, 75, 0, 0, 67, 0, 24, 8, 0, 43, 0, 35, 51, 86, 118, 78,
                              24, 0, 0, 90, 31, 31, 31, 0, 31, 0, 78, 0, 43, 8, 31, 78, 102, 102, 82, 39, 20, 0, 55, 71,
                              125, 110, 47, 35, 0, 75, 90, 0, 27, 0, 27, 0, 20, 0, 0, 20, 43, 43, 0, 43, 20, 0, 51, 0,
                              0, 27, 75, 4, 27, 27, 27, 4, 4, 75, 43, 0, 27, 4, 0, 0, 0, 20, 20, 0, 0, 35, 12, 43, 0,
                              43, 0, 0, 0, 20, 0, 43, 0, 43, 20, 0, 20, 43, 0, 20, 0, 0, 0, 20, 20, 20, 20, 0, 67, 20,
                              67, 20, 67, 0, 0, 0, 0, 20, 43, 43, 78, 4, 4, 82, 78, 118, 125, 106, 27, 75, 98, 75, 67,
                              39, 47, 20, 20, 0, 12, 35, 12, 0, 20, 20, 67, 0, 20, 0, 20, 20, 0, 0, 63, 39, 0, 63, 8,
                              67, 20, 20, 43, 43, 0, 82, 0],
                             [16, 0, 0, 47, 12, 47, 35, 47, 47, 35, 47, 0, 35, 43, 20, 20, 20, 43, 31, 55, 0, 0, 4, 27,
                              0, 27, 27, 27, 51, 0, 4, 0, 31, 8, 0, 67, 67, 67, 43, 0, 27, 59, 43, 71, 63, 118, 125, 78,
                              43, 24, 12, 0, 0, 8, 0, 8, 75, 47, 20, 43, 110, 27, 110, 125, 102, 90, 59, 0, 24, 8, 94,
                              94, 141, 129, 106, 59, 0, 0, 0, 0, 0, 8, 31, 67, 20, 67, 67, 67, 43, 67, 20, 20, 0, 20,
                              82, 24, 27, 27, 27, 0, 0, 27, 75, 51, 0, 27, 20, 0, 20, 20, 0, 35, 12, 0, 43, 0, 35, 35,
                              35, 47, 0, 0, 43, 20, 43, 20, 43, 0, 8, 0, 20, 20, 20, 35, 86, 31, 39, 16, 0, 0, 0, 67,
                              20, 20, 20, 43, 67, 0, 0, 75, 20, 86, 20, 0, 67, 0, 0, 0, 12, 59, 31, 78, 78, 98, 35, 106,
                              98, 51, 75, 59, 55, 8, 75, 0, 43, 35, 0, 20, 0, 0, 20, 20, 43, 20, 0, 0, 0, 43, 16, 39, 0,
                              16, 0, 20, 20, 20, 67, 0, 0, 20, 0],
                             [20, 47, 20, 0, 47, 35, 47, 35, 0, 12, 35, 0, 0, 20, 0, 75, 8, 8, 31, 8, 8, 8, 27, 4, 0, 4,
                              27, 0, 27, 4, 0, 8, 31, 55, 51, 75, 20, 43, 20, 0, 0, 59, 0, 55, 110, 110, 125, 125, 106,
                              0, 0, 0, 43, 20, 0, 0, 16, 0, 20, 67, 51, 75, 125, 110, 31, 0, 55, 78, 75, 47, 0, 118,
                              129, 129, 165, 47, 59, 0, 0, 0, 82, 90, 8, 43, 43, 43, 20, 20, 43, 43, 43, 0, 0, 43, 0, 0,
                              27, 0, 51, 51, 27, 4, 0, 51, 27, 0, 43, 20, 0, 0, 35, 47, 47, 12, 35, 0, 35, 12, 0, 43,
                              43, 43, 20, 20, 20, 20, 20, 75, 4, 75, 20, 0, 0, 86, 8, 31, 55, 16, 16, 39, 0, 43, 20, 43,
                              67, 20, 43, 20, 0, 0, 0, 63, 0, 16, 0, 0, 0, 0, 0, 0, 55, 8, 31, 67, 43, 106, 35, 0, 122,
                              63, 20, 67, 35, 0, 20, 20, 0, 43, 0, 0, 43, 20, 20, 20, 0, 35, 12, 0, 0, 0, 82, 75, 75,
                              43, 43, 20, 43, 0, 0, 0, 51],
                             [75, 86, 51, 12, 35, 12, 0, 35, 0, 35, 47, 16, 0, 20, 4, 0, 51, 4, 51, 75, 8, 8, 0, 27, 4,
                              4, 27, 0, 63, 4, 31, 0, 31, 8, 75, 0, 0, 55, 8, 0, 27, 0, 35, 24, 86, 63, 125, 153, 153,
                              106, 0, 0, 67, 0, 0, 0, 0, 16, 51, 75, 82, 59, 51, 0, 31, 0, 31, 67, 0, 24, 35, 173, 165,
                              141, 133, 102, 0, 39, 51, 0, 0, 8, 0, 0, 43, 43, 0, 0, 0, 43, 0, 67, 43, 43, 75, 43, 20,
                              0, 51, 4, 4, 0, 27, 4, 27, 75, 0, 0, 0, 12, 12, 71, 12, 47, 35, 12, 47, 43, 0, 0, 0, 67,
                              0, 20, 75, 27, 27, 4, 16, 0, 31, 0, 0, 55, 31, 8, 8, 31, 8, 31, 55, 75, 43, 0, 20, 67, 0,
                              20, 0, 0, 16, 0, 0, 35, 0, 63, 16, 16, 0, 24, 24, 20, 20, 43, 27, 51, 75, 75, 24, 114, 82,
                              8, 12, 0, 20, 20, 67, 0, 67, 0, 0, 0, 20, 67, 20, 43, 35, 35, 0, 20, 90, 27, 27, 0, 20,
                              20, 20, 0, 20, 0, 75],
                             [0, 0, 0, 20, 0, 0, 0, 0, 35, 47, 71, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 75, 55, 31, 51, 4, 0,
                              4, 0, 55, 8, 31, 0, 0, 55, 0, 31, 0, 0, 8, 0, 51, 35, 24, 55, 90, 90, 153, 153, 118, 31,
                              0, 27, 0, 67, 0, 0, 75, 27, 82, 43, 82, 51, 4, 12, 16, 35, 43, 47, 24, 110, 137, 141, 133,
                              118, 78, 24, 0, 0, 27, 27, 8, 8, 43, 0, 20, 67, 43, 20, 0, 43, 0, 43, 4, 0, 43, 20, 20,
                              20, 82, 27, 4, 51, 0, 0, 4, 0, 0, 0, 12, 35, 47, 35, 0, 12, 0, 0, 67, 43, 0, 20, 20, 0,
                              71, 43, 4, 75, 0, 0, 31, 8, 55, 8, 0, 78, 8, 31, 31, 8, 78, 31, 0, 0, 20, 20, 43, 43, 12,
                              78, 86, 39, 39, 35, 59, 31, 16, 0, 55, 31, 0, 59, 0, 43, 0, 0, 4, 51, 35, 47, 43, 43, 43,
                              59, 12, 0, 0, 20, 43, 0, 0, 0, 0, 20, 43, 0, 20, 0, 20, 43, 12, 59, 35, 0, 51, 0, 20, 20,
                              20, 0, 39, 27],
                             [24, 82, 0, 51, 4, 51, 20, 20, 43, 43, 0, 0, 27, 0, 4, 27, 4, 4, 27, 0, 75, 0, 27, 55, 0,
                              51, 0, 0, 31, 8, 8, 31, 0, 20, 0, 0, 55, 31, 0, 0, 27, 0, 59, 12, 55, 43, 90, 122, 125,
                              102, 118, 8, 78, 4, 0, 43, 0, 0, 106, 82, 137, 98, 82, 47, 47, 8, 0, 0, 0, 59, 133, 137,
                              188, 118, 86, 63, 20, 0, 27, 27, 27, 51, 8, 0, 0, 20, 75, 0, 43, 20, 43, 20, 27, 67, 67,
                              20, 20, 20, 0, 67, 51, 51, 0, 0, 0, 27, 27, 47, 12, 0, 35, 71, 12, 47, 0, 20, 20, 20, 0,
                              67, 43, 67, 20, 0, 12, 67, 31, 0, 8, 78, 0, 12, 12, 55, 0, 8, 31, 0, 55, 31, 8, 55, 31,
                              39, 43, 43, 59, 31, 78, 0, 63, 12, 35, 35, 78, 63, 31, 0, 0, 8, 31, 51, 0, 51, 0, 12, 59,
                              0, 0, 47, 114, 90, 55, 59, 0, 0, 43, 20, 75, 0, 67, 20, 43, 0, 0, 67, 0, 0, 0, 12, 59, 59,
                              82, 0, 16, 67, 43, 43, 39, 0, 4],
                             [24, 75, 59, 27, 0, 0, 4, 27, 27, 27, 0, 27, 4, 27, 4, 0, 0, 47, 0, 0, 24, 0, 47, 0, 0, 51,
                              8, 0, 75, 55, 8, 8, 0, 0, 0, 0, 0, 47, 59, 27, 51, 55, 0, 67, 0, 43, 75, 110, 118, 165,
                              78, 47, 55, 0, 31, 0, 0, 59, 35, 35, 78, 35, 55, 0, 47, 8, 55, 0, 51, 125, 149, 173, 145,
                              118, 59, 27, 43, 0, 63, 51, 27, 4, 0, 0, 0, 43, 4, 82, 4, 75, 4, 27, 20, 43, 0, 0, 20, 20,
                              0, 43, 75, 27, 4, 4, 4, 0, 43, 43, 35, 35, 35, 94, 12, 35, 12, 20, 67, 0, 0, 20, 43, 43,
                              67, 0, 0, 20, 78, 55, 55, 0, 35, 12, 0, 12, 78, 55, 8, 55, 8, 31, 78, 31, 8, 55, 0, 31, 0,
                              0, 31, 8, 31, 0, 0, 35, 63, 31, 31, 8, 55, 8, 55, 31, 0, 4, 51, 16, 39, 0, 35, 67, 90, 43,
                              0, 55, 35, 0, 0, 0, 16, 39, 0, 20, 0, 0, 20, 0, 82, 0, 0, 0, 35, 35, 12, 35, 16, 75, 51,
                              27, 63, 0, 47],
                             [24, 8, 4, 4, 0, 4, 27, 4, 4, 27, 0, 82, 0, 4, 63, 47, 24, 47, 71, 24, 47, 0, 24, 24, 47,
                              71, 0, 0, 4, 4, 0, 0, 0, 27, 0, 0, 0, 4, 51, 0, 4, 0, 8, 35, 8, 0, 20, 78, 67, 133, 94,
                              149, 43, 39, 16, 0, 0, 47, 47, 55, 82, 59, 0, 47, 0, 0, 31, 31, 63, 125, 173, 161, 86, 82,
                              0, 0, 43, 0, 27, 27, 27, 27, 43, 43, 20, 67, 27, 75, 0, 0, 27, 0, 20, 20, 67, 20, 20, 20,
                              0, 20, 20, 4, 4, 0, 0, 0, 67, 12, 12, 71, 71, 47, 47, 35, 35, 0, 67, 0, 20, 0, 20, 20, 0,
                              43, 82, 20, 55, 8, 35, 12, 59, 59, 59, 59, 59, 55, 0, 0, 0, 16, 39, 20, 35, 35, 78, 55,
                              31, 31, 55, 31, 0, 35, 12, 43, 43, 8, 0, 31, 31, 0, 55, 31, 27, 27, 4, 0, 39, 0, 0, 35,
                              55, 20, 67, 98, 55, 31, 31, 31, 0, 39, 39, 0, 43, 67, 35, 67, 0, 0, 20, 20, 0, 43, 0, 20,
                              0, 16, 0, 0, 0, 20, 67],
                             [59, 8, 4, 4, 0, 4, 0, 4, 4, 4, 20, 12, 71, 90, 86, 86, 94, 47, 71, 71, 71, 71, 47, 24, 47,
                              0, 0, 0, 27, 0, 82, 27, 4, 4, 4, 67, 27, 51, 4, 4, 4, 4, 0, 0, 8, 0, 0, 39, 31, 102, 149,
                              176, 110, 137, 43, 8, 0, 8, 39, 35, 82, 24, 12, 8, 0, 0, 0, 8, 63, 141, 149, 204, 180, 94,
                              71, 35, 0, 51, 4, 27, 4, 31, 67, 20, 20, 4, 4, 75, 43, 20, 67, 51, 43, 0, 20, 43, 20, 0,
                              43, 43, 0, 0, 67, 67, 20, 86, 12, 35, 71, 47, 47, 12, 47, 47, 71, 0, 20, 20, 43, 67, 0,
                              43, 0, 0, 39, 55, 0, 39, 43, 20, 0, 35, 12, 35, 59, 31, 8, 55, 0, 16, 20, 0, 0, 0, 20, 39,
                              63, 16, 0, 0, 0, 0, 20, 67, 20, 35, 0, 0, 0, 55, 31, 0, 4, 4, 0, 0, 12, 12, 35, 59, 8, 55,
                              43, 67, 67, 67, 67, 67, 0, 8, 0, 16, 0, 78, 20, 0, 24, 20, 0, 8, 31, 0, 20, 43, 67, 0, 20,
                              78, 82, 106, 94],
                             [94, 51, 0, 0, 27, 4, 51, 51, 0, 55, 78, 86, 27, 75, 94, 94, 94, 63, 16, 86, 16, 110, 110,
                              55, 78, 24, 24, 0, 0, 12, 0, 0, 27, 0, 4, 27, 0, 0, 27, 27, 51, 27, 4, 47, 47, 0, 8, 67,
                              0, 55, 63, 82, 86, 129, 27, 125, 31, 35, 39, 0, 12, 27, 0, 0, 24, 4, 59, 31, 94, 157, 180,
                              204, 149, 71, 27, 59, 39, 0, 0, 27, 0, 0, 67, 0, 55, 55, 20, 0, 0, 0, 0, 20, 27, 0, 43,
                              20, 43, 20, 59, 0, 0, 43, 0, 59, 0, 35, 71, 59, 59, 35, 47, 12, 47, 35, 59, 35, 43, 43,
                              20, 20, 0, 20, 16, 63, 0, 8, 16, 0, 43, 20, 43, 67, 0, 12, 0, 31, 78, 16, 0, 43, 67, 43,
                              20, 0, 20, 0, 0, 0, 0, 63, 63, 98, 8, 20, 43, 67, 0, 8, 8, 31, 12, 8, 67, 0, 27, 16, 35,
                              35, 12, 12, 31, 8, 90, 67, 67, 67, 122, 82, 43, 8, 8, 31, 8, 24, 24, 0, 0, 20, 0, 78, 31,
                              67, 0, 0, 67, 0, 86, 0, 47, 82, 94],
                             [59, 47, 4, 4, 0, 0, 27, 0, 0, 24, 55, 86, 110, 75, 75, 94, 16, 16, 0, 39, 16, 94, 35, 94,
                              78, 55, 55, 0, 82, 0, 51, 0, 0, 51, 0, 27, 51, 4, 27, 27, 4, 4, 4, 47, 75, 71, 71, 0, 20,
                              59, 75, 98, 102, 71, 63, 157, 149, 118, 110, 8, 47, 0, 0, 43, 0, 4, 27, 8, 47, 125, 196,
                              212, 137, 35, 4, 0, 0, 0, 0, 82, 24, 24, 82, 8, 8, 0, 71, 12, 43, 20, 20, 43, 20, 75, 67,
                              67, 43, 20, 20, 0, 0, 94, 35, 12, 12, 59, 35, 12, 0, 35, 59, 59, 35, 35, 59, 35, 0, 20, 0,
                              82, 16, 39, 63, 0, 8, 0, 39, 39, 75, 43, 20, 67, 0, 0, 12, 78, 0, 0, 0, 0, 20, 0, 67, 67,
                              0, 67, 0, 16, 63, 12, 16, 78, 0, 8, 43, 0, 20, 0, 59, 12, 31, 31, 31, 20, 0, 16, 12, 12,
                              35, 12, 0, 8, 31, 75, 27, 67, 67, 55, 59, 20, 8, 31, 31, 31, 0, 24, 0, 8, 75, 0, 51, 78,
                              43, 20, 0, 0, 82, 106, 82, 114, 31],
                             [0, 78, 0, 27, 51, 0, 51, 35, 71, 31, 86, 110, 75, 71, 24, 20, 0, 0, 39, 86, 16, 4, 35, 78,
                              31, 55, 78, 0, 75, 4, 0, 75, 16, 43, 67, 0, 51, 27, 4, 27, 27, 0, 24, 0, 16, 27, 24, 0, 0,
                              0, 43, 67, 16, 63, 118, 165, 180, 125, 63, 86, 24, 12, 27, 0, 0, 27, 4, 8, 94, 114, 173,
                              176, 67, 94, 27, 0, 0, 0, 47, 0, 0, 24, 8, 0, 20, 24, 12, 59, 67, 0, 20, 0, 0, 4, 43, 67,
                              0, 0, 0, 43, 16, 0, 47, 20, 43, 12, 12, 71, 35, 71, 71, 0, 0, 0, 71, 86, 0, 0, 0, 4, 0,
                              16, 0, 8, 59, 0, 0, 0, 39, 0, 20, 20, 0, 20, 0, 12, 82, 16, 0, 20, 0, 20, 0, 43, 20, 67,
                              43, 0, 82, 39, 0, 20, 0, 67, 20, 0, 20, 43, 0, 43, 67, 8, 8, 8, 0, 0, 16, 12, 0, 0, 35, 0,
                              35, 0, 59, 27, 43, 31, 55, 8, 75, 39, 16, 43, 0, 0, 0, 0, 47, 24, 59, 0, 31, 31, 20, 0,
                              82, 86, 59, 86, 63],
                             [94, 67, 20, 24, 0, 67, 0, 51, 78, 78, 102, 86, 47, 24, 71, 71, 8, 0, 0, 39, 16, 39, 8, 55,
                              8, 78, 35, 35, 75, 51, 27, 4, 27, 20, 47, 24, 47, 27, 16, 27, 27, 47, 0, 75, 16, 0, 0, 27,
                              78, 0, 0, 24, 55, 27, 86, 125, 153, 173, 149, 94, 47, 12, 4, 20, 0, 0, 0, 31, 71, 90, 204,
                              176, 173, 24, 0, 59, 16, 27, 27, 0, 0, 31, 0, 31, 0, 0, 43, 0, 20, 0, 20, 20, 43, 0, 27,
                              0, 20, 43, 43, 47, 0, 0, 24, 24, 20, 0, 35, 35, 0, 71, 35, 0, 59, 35, 0, 31, 0, 4, 4, 27,
                              0, 0, 78, 0, 59, 43, 43, 43, 98, 20, 0, 0, 20, 43, 43, 12, 67, 12, 0, 0, 0, 20, 67, 20,
                              20, 0, 20, 59, 16, 67, 20, 43, 20, 43, 0, 20, 43, 0, 82, 20, 0, 67, 67, 67, 31, 20, 16, 0,
                              67, 12, 35, 12, 0, 0, 12, 59, 39, 78, 78, 63, 0, 90, 51, 51, 4, 4, 86, 0, 31, 78, 71, 51,
                              4, 0, 35, 59, 20, 4, 63, 0, 51],
                             [78, 102, 0, 0, 0, 4, 51, 59, 82, 102, 0, 71, 71, 24, 0, 47, 12, 0, 47, 0, 4, 0, 55, 78,
                              31, 12, 12, 35, 4, 4, 27, 4, 75, 24, 0, 0, 0, 0, 0, 0, 24, 0, 0, 16, 20, 0, 0, 0, 0, 24,
                              0, 16, 0, 55, 27, 39, 125, 129, 125, 125, 8, 0, 0, 0, 67, 20, 0, 55, 47, 149, 212, 227,
                              118, 24, 8, 0, 51, 51, 27, 27, 0, 0, 31, 0, 20, 20, 0, 24, 0, 0, 43, 67, 55, 4, 27, 75,
                              43, 43, 47, 24, 47, 24, 0, 47, 0, 67, 20, 0, 12, 0, 59, 35, 0, 78, 55, 31, 78, 43, 0, 75,
                              0, 35, 35, 59, 59, 0, 0, 0, 0, 0, 0, 0, 20, 43, 0, 0, 0, 43, 59, 12, 90, 67, 43, 0, 67,
                              67, 82, 16, 0, 67, 0, 0, 43, 43, 0, 20, 67, 0, 0, 0, 67, 43, 59, 0, 0, 59, 82, 47, 27, 12,
                              0, 0, 67, 59, 0, 0, 47, 63, 75, 98, 114, 114, 71, 24, 82, 63, 31, 4, 0, 110, 59, 12, 47,
                              0, 0, 90, 63, 63, 0, 27, 4],
                             [78, 78, 0, 0, 8, 0, 59, 35, 106, 78, 78, 24, 0, 71, 0, 12, 0, 12, 47, 0, 67, 31, 0, 0, 0,
                              0, 0, 59, 12, 4, 0, 16, 63, 0, 47, 24, 39, 0, 47, 0, 0, 67, 90, 0, 43, 0, 4, 51, 0, 0, 24,
                              71, 63, 0, 8, 86, 86, 125, 188, 94, 90, 75, 0, 31, 0, 16, 0, 0, 102, 125, 192, 212, 161,
                              71, 0, 0, 0, 4, 0, 4, 39, 0, 0, 0, 24, 0, 47, 0, 0, 0, 0, 31, 0, 27, 51, 0, 0, 0, 0, 0,
                              24, 0, 24, 0, 0, 20, 43, 67, 35, 12, 12, 59, 59, 12, 78, 31, 31, 20, 67, 78, 59, 0, 35,
                              35, 12, 0, 0, 0, 0, 0, 67, 43, 43, 20, 0, 12, 35, 20, 0, 98, 35, 12, 0, 0, 20, 35, 0, 16,
                              43, 0, 0, 0, 67, 67, 20, 0, 67, 0, 82, 35, 0, 27, 4, 4, 0, 0, 51, 0, 12, 12, 0, 0, 0, 31,
                              35, 35, 51, 24, 47, 43, 55, 12, 110, 141, 102, 27, 98, 0, 75, 75, 27, 35, 59, 35, 43, 27,
                              63, 39, 16, 27, 0],
                             [102, 86, 55, 0, 0, 0, 102, 75, 39, 55, 0, 0, 24, 0, 24, 0, 12, 12, 0, 0, 20, 0, 0, 82, 4,
                              0, 4, 0, 82, 0, 0, 0, 27, 0, 16, 16, 0, 39, 16, 0, 59, 0, 20, 20, 43, 0, 51, 0, 27, 0, 0,
                              24, 0, 0, 47, 90, 0, 98, 153, 200, 157, 75, 51, 8, 0, 0, 24, 0, 67, 145, 145, 216, 118, 0,
                              0, 0, 55, 20, 0, 0, 0, 0, 0, 0, 20, 47, 0, 12, 12, 47, 0, 67, 0, 71, 24, 0, 0, 71, 8, 31,
                              24, 24, 0, 0, 47, 43, 43, 20, 20, 12, 35, 35, 59, 0, 0, 55, 8, 0, 31, 35, 8, 55, 55, 0,
                              67, 0, 43, 0, 43, 43, 20, 67, 0, 12, 0, 12, 35, 43, 59, 4, 35, 35, 0, 59, 12, 0, 12, 0,
                              20, 20, 20, 0, 43, 43, 0, 0, 0, 43, 0, 0, 67, 55, 75, 31, 31, 20, 0, 20, 27, 0, 0, 0, 12,
                              35, 31, 0, 0, 51, 20, 27, 24, 102, 133, 192, 153, 71, 43, 4, 16, 0, 75, 0, 16, 63, 86, 47,
                              86, 98, 16, 27, 0],
                             [55, 102, 51, 0, 0, 0, 47, 47, 4, 59, 39, 0, 0, 24, 0, 24, 0, 4, 4, 0, 71, 67, 0, 4, 0, 47,
                              0, 12, 0, 35, 12, 0, 75, 0, 4, 39, 39, 0, 16, 0, 0, 27, 0, 43, 43, 0, 4, 51, 27, 71, 47,
                              0, 0, 0, 16, 0, 0, 98, 122, 173, 192, 153, 98, 27, 0, 4, 0, 8, 75, 129, 176, 208, 0, 0, 0,
                              24, 59, 16, 63, 63, 4, 0, 43, 51, 55, 43, 24, 67, 82, 114, 90, 90, 20, 71, 71, 78, 0, 0,
                              0, 20, 55, 47, 47, 0, 20, 0, 20, 43, 67, 35, 0, 0, 20, 20, 20, 0, 86, 0, 78, 55, 31, 0,
                              67, 0, 20, 0, 0, 0, 20, 67, 12, 0, 0, 0, 0, 59, 59, 0, 75, 114, 35, 82, 0, 12, 0, 59, 0,
                              47, 27, 20, 67, 0, 0, 20, 43, 20, 20, 0, 0, 0, 67, 63, 75, 75, 67, 55, 55, 43, 20, 16, 4,
                              27, 0, 0, 0, 0, 35, 0, 4, 27, 47, 141, 208, 227, 196, 125, 90, 67, 51, 20, 86, 27, 75,
                              118, 27, 47, 63, 106, 75, 98, 86],
                             [43, 67, 102, 63, 0, 47, 0, 12, 24, 0, 0, 0, 0, 0, 0, 12, 27, 27, 67, 94, 118, 137, 157,
                              133, 102, 86, 35, 63, 4, 51, 16, 0, 0, 0, 12, 8, 0, 24, 20, 0, 27, 0, 0, 0, 43, 43, 43, 4,
                              27, 24, 47, 0, 24, 0, 0, 0, 31, 39, 31, 161, 192, 169, 67, 8, 0, 39, 0, 0, 0, 102, 98,
                              141, 71, 82, 71, 71, 75, 133, 133, 86, 133, 125, 106, 82, 106, 149, 133, 90, 90, 153, 106,
                              161, 137, 98, 75, 110, 78, 71, 47, 0, 20, 24, 47, 75, 27, 90, 20, 67, 43, 12, 0, 20, 0,
                              20, 67, 0, 0, 0, 0, 35, 59, 59, 67, 43, 20, 43, 20, 43, 0, 20, 0, 12, 35, 12, 0, 67, 20,
                              98, 31, 59, 75, 35, 35, 12, 35, 0, 0, 47, 71, 75, 43, 43, 43, 20, 0, 20, 20, 39, 59, 67,
                              90, 137, 176, 184, 208, 192, 176, 90, 0, 82, 51, 16, 27, 0, 35, 0, 12, 43, 0, 27, 63, 176,
                              322, 302, 325, 184, 137, 106, 71, 114, 114, 94, 0, 24, 47, 0, 59, 98, 16, 63, 63],
                             [43, 43, 90, 24, 47, 24, 35, 0, 0, 0, 0, 20, 0, 55, 55, 67, 67, 122, 184, 184, 243, 173,
                              161, 259, 275, 227, 212, 173, 125, 118, 35, 82, 98, 114, 59, 82, 75, 67, 0, 0, 16, 0, 24,
                              0, 59, 0, 0, 43, 24, 47, 0, 0, 0, 0, 94, 24, 31, 0, 78, 129, 82, 145, 114, 137, 0, 0, 0,
                              0, 8, 82, 86, 259, 235, 188, 118, 145, 137, 165, 118, 173, 157, 157, 125, 106, 153, 114,
                              47, 122, 75, 145, 98, 122, 114, 118, 118, 94, 86, 24, 0, 0, 24, 0, 0, 27, 16, 0, 0, 0, 35,
                              35, 43, 20, 20, 20, 0, 0, 43, 75, 24, 0, 8, 0, 55, 20, 0, 0, 0, 67, 67, 20, 20, 0, 35, 43,
                              0, 0, 0, 51, 78, 106, 51, 51, 12, 0, 0, 24, 24, 24, 71, 90, 90, 67, 67, 20, 20, 55, 0, 0,
                              31, 55, 129, 129, 224, 294, 365, 412, 333, 204, 114, 39, 8, 20, 27, 39, 35, 35, 0, 0, 0,
                              43, 90, 306, 365, 510, 416, 267, 263, 176, 208, 125, 106, 63, 24, 0, 0, 0, 0, 0, 0, 86,
                              63],
                             [0, 20, 78, 78, 39, 0, 71, 0, 24, 0, 43, 20, 106, 137, 129, 161, 176, 216, 224, 216, 176,
                              180, 180, 231, 259, 133, 251, 251, 267, 196, 196, 220, 184, 200, 169, 192, 169, 145, 169,
                              129, 106, 63, 59, 0, 35, 12, 12, 0, 39, 0, 0, 0, 0, 0, 24, 0, 0, 4, 0, 59, 98, 161, 129,
                              63, 27, 0, 0, 0, 24, 129, 188, 322, 369, 290, 204, 47, 122, 12, 0, 59, 16, 55, 31, 82, 67,
                              43, 0, 98, 27, 16, 51, 20, 71, 47, 0, 16, 59, 0, 24, 4, 51, 43, 16, 39, 0, 27, 0, 0, 35,
                              59, 43, 0, 20, 20, 20, 8, 31, 12, 0, 55, 0, 0, 55, 0, 43, 20, 0, 20, 43, 0, 0, 0, 67, 67,
                              43, 0, 59, 27, 59, 82, 82, 51, 75, 82, 82, 0, 47, 24, 67, 98, 98, 43, 0, 90, 0, 0, 0, 0,
                              55, 98, 184, 192, 231, 247, 333, 420, 502, 427, 349, 192, 75, 24, 31, 4, 0, 27, 27, 0, 8,
                              4, 31, 341, 855, 980, 878, 729, 384, 380, 302, 208, 118, 59, 0, 0, 0, 0, 0, 0, 59, 0, 20],
                             [0, 35, 39, 98, 63, 16, 67, 67, 39, 75, 122, 184, 239, 98, 239, 239, 239, 239, 208, 82,
                              122, 118, 71, 106, 129, 149, 173, 188, 227, 204, 235, 235, 235, 122, 216, 239, 122, 255,
                              184, 231, 192, 176, 129, 78, 67, 114, 55, 0, 39, 4, 27, 75, 0, 0, 47, 0, 0, 27, 0, 0, 98,
                              75, 157, 149, 110, 0, 0, 0, 24, 259, 416, 518, 541, 424, 290, 98, 16, 0, 82, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 4, 0, 20, 24, 94, 47, 4, 24, 0, 71, 51, 0, 0, 0, 0, 75, 0, 39, 12, 59, 0,
                              35, 0, 0, 0, 67, 55, 31, 12, 12, 0, 0, 0, 8, 55, 8, 0, 20, 0, 43, 20, 0, 0, 67, 20, 43, 0,
                              47, 75, 4, 59, 82, 114, 0, 0, 0, 0, 0, 71, 114, 106, 63, 78, 90, 43, 20, 0, 0, 0, 27, 71,
                              153, 161, 67, 0, 55, 106, 282, 420, 514, 451, 318, 169, 78, 47, 16, 39, 39, 47, 63, 82, 0,
                              129, 620, 1000, 1000, 1000, 1000, 957, 659, 475, 227, 82, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [24, 39, 67, 141, 110, 43, 114, 20, 114, 161, 200, 247, 263, 263, 192, 216, 129, 153, 176,
                              98, 67, 55, 31, 31, 118, 133, 118, 71, 24, 149, 118, 141, 141, 169, 114, 137, 161, 192,
                              122, 208, 224, 216, 200, 188, 145, 137, 137, 59, 82, 63, 51, 51, 8, 0, 0, 8, 4, 27, 31,
                              35, 71, 59, 133, 165, 90, 63, 71, 102, 322, 447, 620, 659, 596, 439, 337, 275, 231, 196,
                              67, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 24, 27, 0, 47, 0, 0, 8, 0, 0, 20, 47, 67,
                              20, 94, 59, 0, 59, 12, 0, 20, 31, 8, 31, 12, 47, 8, 8, 0, 8, 8, 55, 0, 0, 20, 0, 0, 0, 0,
                              59, 0, 0, 0, 0, 0, 4, 82, 129, 90, 106, 35, 0, 0, 0, 0, 90, 114, 122, 90, 8, 20, 24, 0, 0,
                              0, 71, 153, 176, 137, 20, 0, 0, 27, 125, 302, 412, 510, 463, 325, 169, 90, 43, 67, 75, 90,
                              141, 204, 302, 620, 1000, 1000, 1000, 1000, 1000, 1000, 761, 663, 455, 196, 71, 59, 0, 8,
                              0, 0, 43, 0, 31, 31],
                             [204, 188, 247, 200, 137, 208, 161, 235, 251, 200, 157, 106, 216, 184, 133, 129, 129, 98,
                              122, 90, 8, 55, 0, 0, 94, 12, 0, 86, 86, 110, 86, 118, 82, 82, 106, 59, 59, 106, 106, 153,
                              161, 137, 114, 173, 102, 90, 137, 90, 35, 67, 55, 0, 51, 16, 0, 98, 4, 0, 0, 0, 0, 86, 43,
                              102, 153, 188, 153, 278, 443, 682, 647, 525, 478, 408, 329, 306, 286, 243, 239, 192, 161,
                              82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 47, 0, 27, 0, 31, 0, 20, 43, 24, 43, 20, 39,
                              0, 12, 35, 35, 59, 0, 8, 8, 31, 47, 24, 31, 55, 0, 8, 8, 78, 0, 82, 8, 8, 55, 0, 0, 35, 0,
                              0, 0, 0, 27, 51, 106, 161, 145, 82, 0, 0, 0, 0, 0, 67, 145, 137, 67, 43, 0, 24, 24, 0, 35,
                              43, 161, 239, 122, 0, 0, 0, 0, 0, 0, 35, 157, 502, 529, 451, 396, 173, 180, 235, 271, 424,
                              706, 949, 1000, 1000, 1000, 1000, 1000, 1000, 761, 439, 475, 486, 388, 373, 204, 118, 110,
                              55, 8, 125, 43, 133, 4],
                             [314, 404, 471, 435, 388, 325, 286, 231, 243, 145, 82, 8, 114, 114, 59, 67, 39, 63, 51, 31,
                              55, 31, 20, 4, 0, 24, 0, 55, 47, 47, 24, 51, 27, 43, 75, 98, 82, 12, 82, 98, 122, 39, 125,
                              43, 71, 106, 82, 106, 0, 122, 114, 16, 0, 16, 39, 0, 0, 35, 0, 78, 0, 55, 51, 122, 235,
                              231, 420, 624, 647, 569, 349, 208, 227, 184, 196, 169, 235, 227, 192, 98, 169, 216, 173,
                              153, 153, 94, 90, 0, 90, 20, 0, 0, 0, 0, 0, 0, 39, 0, 0, 27, 0, 0, 0, 0, 20, 86, 0, 0, 16,
                              63, 86, 0, 47, 0, 67, 0, 67, 47, 47, 31, 55, 31, 0, 0, 16, 31, 0, 55, 0, 43, 35, 94, 35,
                              20, 0, 4, 0, 24, 114, 90, 200, 106, 12, 0, 0, 43, 43, 114, 98, 145, 63, 0, 0, 27, 0, 39,
                              90, 106, 161, 216, 82, 0, 0, 0, 0, 0, 0, 0, 16, 322, 702, 851, 957, 984, 949, 992, 1000,
                              1000, 1000, 1000, 1000, 1000, 1000, 1000, 953, 533, 204, 365, 341, 369, 490, 463, 376,
                              353, 259, 176, 188, 188, 235, 220, 133],
                             [314, 435, 639, 718, 733, 537, 427, 310, 67, 106, 0, 39, 0, 82, 47, 0, 20, 20, 43, 8, 8,
                              20, 4, 0, 0, 0, 43, 0, 0, 63, 0, 0, 39, 4, 43, 63, 0, 16, 16, 39, 0, 86, 43, 82, 35, 82,
                              82, 82, 0, 78, 63, 75, 27, 75, 27, 0, 0, 0, 8, 8, 0, 8, 67, 90, 192, 341, 663, 843, 788,
                              545, 153, 110, 31, 59, 43, 149, 165, 161, 153, 82, 141, 169, 196, 200, 184, 239, 200, 184,
                              153, 114, 90, 31, 0, 0, 0, 0, 0, 0, 12, 0, 0, 27, 0, 0, 16, 39, 16, 16, 0, 0, 0, 47, 20,
                              43, 20, 20, 43, 67, 0, 47, 31, 27, 4, 31, 0, 8, 31, 8, 43, 24, 0, 47, 0, 43, 0, 0, 24, 71,
                              129, 161, 129, 106, 0, 0, 0, 0, 0, 67, 122, 169, 106, 0, 0, 0, 16, 0, 0, 98, 149, 220,
                              110, 4, 0, 0, 0, 0, 0, 0, 275, 549, 859, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                              1000, 1000, 1000, 1000, 741, 439, 157, 35, 94, 192, 286, 337, 380, 424, 447, 376, 275,
                              165, 118, 173, 67, 39],
                             [196, 424, 576, 773, 867, 741, 553, 373, 184, 125, 51, 24, 0, 0, 0, 31, 0, 8, 0, 31, 90,
                              51, 0, 0, 0, 75, 0, 0, 0, 31, 4, 51, 4, 27, 0, 0, 0, 8, 0, 8, 0, 20, 0, 0, 0, 59, 12, 67,
                              67, 0, 39, 4, 4, 51, 4, 0, 0, 0, 0, 12, 12, 55, 75, 129, 263, 427, 741, 906, 890, 498,
                              231, 4, 4, 31, 43, 67, 122, 137, 137, 94, 98, 75, 110, 137, 192, 176, 176, 153, 165, 161,
                              161, 184, 129, 129, 129, 118, 16, 86, 59, 27, 51, 51, 8, 4, 4, 0, 0, 0, 0, 0, 39, 67, 0,
                              43, 20, 67, 43, 43, 43, 0, 20, 0, 20, 67, 20, 0, 67, 67, 0, 24, 0, 94, 12, 0, 0, 0, 0, 0,
                              67, 176, 141, 129, 35, 0, 0, 0, 43, 78, 161, 122, 114, 47, 0, 47, 0, 0, 0, 71, 133, 188,
                              220, 63, 63, 0, 94, 275, 365, 408, 576, 710, 851, 1000, 1000, 1000, 1000, 1000, 1000,
                              1000, 1000, 1000, 1000, 1000, 1000, 498, 310, 12, 31, 0, 0, 90, 220, 322, 392, 416, 384,
                              235, 47, 67, 0, 8, 0],
                             [153, 251, 420, 569, 678, 647, 529, 341, 161, 78, 0, 31, 67, 8, 0, 16, 0, 39, 0, 0, 86, 16,
                              82, 67, 31, 27, 0, 0, 0, 0, 0, 8, 0, 43, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 16, 4, 4, 47, 0, 47, 12, 12, 12, 8, 35, 24, 39, 388, 651, 953, 992, 804, 490, 145, 43,
                              31, 0, 0, 0, 24, 86, 82, 16, 98, 4, 0, 137, 114, 114, 137, 78, 153, 161, 184, 192, 145,
                              208, 94, 200, 188, 184, 129, 67, 114, 90, 51, 27, 0, 0, 0, 0, 63, 0, 35, 12, 20, 20, 0, 0,
                              0, 67, 20, 43, 20, 43, 20, 0, 67, 43, 43, 43, 24, 71, 47, 0, 0, 0, 0, 43, 8, 82, 114, 169,
                              200, 122, 0, 0, 20, 0, 0, 137, 153, 137, 24, 71, 0, 0, 0, 12, 4, 51, 78, 157, 212, 243,
                              345, 447, 502, 573, 439, 478, 522, 561, 584, 698, 725, 714, 725, 933, 1000, 1000, 1000,
                              1000, 1000, 980, 600, 404, 400, 184, 0, 0, 67, 8, 8, 125, 173, 149, 345, 259, 78, 43, 0,
                              0, 0],
                             [59, 153, 243, 239, 365, 475, 388, 224, 129, 67, 94, 24, 43, 8, 43, 98, 0, 24, 82, 67, 12,
                              71, 39, 20, 78, 12, 31, 8, 0, 43, 67, 43, 0, 43, 20, 0, 0, 20, 0, 8, 0, 43, 31, 0, 31, 20,
                              43, 0, 0, 0, 0, 16, 0, 0, 24, 0, 20, 27, 47, 31, 59, 35, 192, 200, 663, 953, 1000, 1000,
                              733, 271, 125, 63, 0, 0, 0, 0, 0, 47, 51, 98, 0, 0, 0, 47, 47, 71, 90, 43, 98, 114, 129,
                              129, 145, 122, 122, 145, 153, 75, 145, 51, 75, 153, 82, 75, 51, 51, 51, 0, 0, 0, 0, 0, 82,
                              43, 43, 0, 20, 0, 20, 43, 67, 0, 75, 51, 4, 20, 67, 0, 0, 0, 67, 43, 12, 0, 0, 0, 0, 31,
                              90, 118, 200, 161, 51, 16, 0, 0, 0, 0, 153, 137, 0, 39, 0, 24, 47, 0, 4, 16, 47, 86, 196,
                              314, 431, 502, 486, 471, 396, 380, 243, 333, 349, 341, 361, 420, 439, 522, 729, 1000,
                              1000, 980, 706, 459, 212, 318, 290, 235, 153, 67, 67, 0, 0, 0, 0, 0, 157, 212, 102, 0, 0,
                              0, 0],
                             [43, 59, 75, 90, 90, 184, 216, 180, 153, 55, 0, 0, 20, 67, 31, 4, 4, 4, 43, 43, 24, 71, 71,
                              16, 78, 67, 75, 20, 31, 8, 0, 31, 31, 31, 82, 90, 0, 0, 47, 0, 0, 0, 12, 0, 12, 0, 0, 20,
                              0, 0, 0, 0, 0, 24, 39, 55, 86, 165, 286, 322, 302, 310, 282, 631, 961, 1000, 1000, 1000,
                              710, 200, 55, 63, 0, 0, 0, 0, 24, 31, 125, 114, 71, 0, 0, 0, 12, 24, 24, 47, 67, 43, 82,
                              90, 43, 43, 122, 114, 90, 114, 98, 75, 75, 114, 82, 90, 75, 75, 75, 20, 47, 0, 0, 39, 0,
                              43, 43, 20, 20, 67, 0, 20, 0, 0, 4, 4, 27, 75, 0, 24, 0, 0, 0, 0, 12, 0, 59, 59, 0, 0, 82,
                              129, 192, 176, 75, 0, 0, 0, 0, 0, 153, 161, 122, 39, 75, 24, 0, 0, 0, 4, 47, 86, 196, 290,
                              392, 369, 298, 322, 282, 216, 184, 75, 82, 114, 122, 133, 161, 212, 333, 541, 533, 322,
                              290, 149, 0, 0, 47, 47, 114, 90, 0, 0, 0, 0, 0, 0, 0, 188, 184, 31, 0, 0, 0],
                             [43, 78, 43, 59, 43, 141, 157, 153, 86, 63, 78, 0, 0, 31, 0, 27, 4, 43, 43, 20, 24, 0, 47,
                              24, 94, 67, 47, 51, 31, 55, 31, 8, 31, 47, 0, 0, 0, 24, 0, 24, 24, 0, 0, 0, 0, 16, 0, 0,
                              24, 0, 0, 27, 0, 82, 157, 259, 369, 478, 525, 565, 478, 412, 545, 741, 1000, 1000, 1000,
                              1000, 796, 129, 71, 16, 0, 0, 0, 12, 24, 63, 114, 149, 102, 63, 0, 0, 35, 35, 47, 0, 0,
                              55, 0, 20, 43, 67, 55, 59, 55, 82, 90, 90, 43, 43, 90, 82, 82, 82, 75, 24, 47, 0, 0, 71,
                              31, 67, 20, 43, 0, 0, 24, 78, 75, 27, 75, 4, 4, 31, 8, 55, 8, 8, 75, 12, 12, 12, 12, 0,
                              12, 0, 75, 106, 98, 184, 106, 75, 0, 0, 24, 0, 200, 169, 0, 63, 39, 0, 27, 51, 0, 0, 35,
                              86, 106, 251, 251, 259, 157, 204, 141, 90, 82, 106, 24, 47, 75, 0, 75, 94, 196, 294, 357,
                              184, 114, 0, 0, 0, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 0, 0, 165, 0, 0, 0, 0],
                             [20, 51, 20, 51, 20, 137, 114, 137, 110, 67, 43, 55, 31, 0, 4, 0, 0, 20, 0, 47, 24, 47, 0,
                              0, 86, 43, 24, 47, 24, 90, 24, 47, 24, 20, 0, 20, 0, 0, 0, 0, 98, 0, 12, 0, 0, 24, 47, 0,
                              63, 16, 75, 137, 145, 267, 369, 455, 518, 557, 510, 471, 373, 282, 310, 569, 890, 1000,
                              1000, 1000, 835, 310, 176, 20, 0, 0, 0, 0, 20, 141, 149, 141, 125, 39, 27, 47, 47, 35, 0,
                              0, 0, 0, 43, 43, 43, 20, 12, 0, 0, 31, 20, 43, 20, 55, 51, 75, 82, 75, 71, 47, 71, 24, 24,
                              0, 0, 0, 20, 0, 43, 20, 43, 4, 0, 75, 27, 4, 31, 55, 31, 31, 55, 0, 82, 75, 16, 16, 0, 82,
                              0, 20, 55, 129, 192, 224, 173, 35, 27, 0, 0, 0, 94, 184, 137, 27, 0, 47, 51, 0, 0, 12, 35,
                              137, 188, 188, 188, 149, 125, 133, 24, 90, 0, 0, 0, 4, 0, 0, 0, 67, 176, 224, 239, 47, 0,
                              0, 0, 0, 0, 0, 0, 118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 31, 0, 200],
                             [55, 20, 12, 0, 0, 16, 75, 63, 78, 43, 20, 0, 12, 0, 39, 39, 8, 8, 0, 75, 0, 59, 0, 0, 0,
                              67, 43, 90, 20, 20, 43, 0, 20, 20, 43, 0, 0, 24, 0, 0, 12, 12, 0, 27, 63, 39, 71, 47, 149,
                              192, 255, 357, 435, 455, 502, 510, 447, 392, 306, 169, 224, 149, 239, 239, 537, 710, 976,
                              984, 820, 659, 271, 12, 12, 8, 0, 31, 78, 51, 129, 47, 94, 39, 27, 0, 35, 35, 0, 0, 0, 24,
                              82, 4, 4, 0, 0, 0, 12, 0, 0, 20, 0, 0, 24, 47, 24, 63, 39, 39, 0, 16, 63, 71, 0, 55, 55,
                              8, 31, 0, 8, 75, 0, 20, 20, 43, 27, 55, 0, 31, 31, 0, 75, 16, 0, 51, 0, 51, 12, 0, 20, 98,
                              129, 133, 231, 86, 0, 0, 0, 0, 94, 204, 149, 71, 35, 24, 0, 24, 0, 47, 82, 165, 169, 118,
                              110, 47, 39, 51, 24, 0, 0, 24, 0, 0, 0, 0, 20, 180, 122, 153, 47, 0, 0, 0, 0, 0, 0, 0,
                              235, 353, 306, 165, 0, 20, 8, 0, 0, 220, 220, 384, 353, 345, 329],
                             [4, 39, 8, 8, 35, 8, 149, 118, 78, 43, 0, 0, 12, 75, 39, 0, 12, 0, 0, 31, 59, 35, 12, 0, 0,
                              0, 0, 0, 0, 0, 24, 0, 0, 0, 31, 8, 0, 0, 43, 43, 78, 71, 125, 149, 180, 235, 267, 322,
                              380, 435, 490, 514, 490, 400, 400, 298, 243, 176, 176, 212, 0, 161, 133, 231, 200, 310,
                              451, 537, 784, 922, 667, 365, 0, 27, 0, 0, 27, 118, 114, 55, 39, 0, 0, 63, 35, 12, 0, 47,
                              47, 0, 0, 12, 0, 0, 0, 78, 0, 0, 20, 43, 12, 0, 20, 43, 0, 43, 39, 16, 63, 0, 39, 0, 39,
                              20, 55, 31, 8, 43, 0, 0, 20, 0, 20, 0, 20, 75, 0, 8, 8, 0, 0, 0, 16, 0, 27, 0, 0, 0, 67,
                              43, 114, 78, 216, 173, 0, 0, 0, 0, 106, 176, 133, 94, 71, 0, 24, 0, 59, 35, 114, 169, 125,
                              82, 82, 31, 8, 67, 0, 55, 0, 0, 8, 0, 51, 0, 184, 204, 204, 98, 0, 0, 0, 0, 0, 0, 0, 192,
                              651, 769, 761, 737, 616, 631, 651, 592, 580, 651, 714, 706, 643, 439, 298],
                             [67, 90, 0, 8, 47, 35, 106, 67, 31, 4, 35, 12, 0, 0, 0, 0, 82, 0, 78, 0, 35, 12, 0, 0, 4,
                              0, 8, 0, 0, 0, 8, 0, 0, 24, 0, 86, 94, 94, 110, 180, 212, 243, 353, 392, 408, 431, 478,
                              455, 490, 459, 365, 412, 325, 271, 204, 188, 157, 149, 86, 110, 16, 94, 90, 118, 129, 141,
                              259, 337, 576, 741, 827, 678, 443, 227, 75, 12, 0, 55, 4, 90, 35, 0, 94, 86, 12, 35, 0,
                              71, 24, 0, 0, 0, 39, 0, 0, 0, 43, 0, 20, 82, 0, 0, 20, 31, 43, 20, 0, 16, 16, 16, 16, 16,
                              43, 67, 67, 43, 0, 0, 8, 20, 67, 67, 0, 43, 67, 20, 27, 0, 8, 55, 16, 0, 16, 0, 51, 0, 27,
                              75, 20, 0, 90, 145, 200, 224, 55, 8, 0, 0, 31, 173, 145, 122, 0, 0, 24, 47, 24, 71, 122,
                              129, 78, 90, 16, 0, 0, 12, 55, 8, 55, 31, 0, 8, 47, 125, 216, 141, 82, 0, 0, 0, 0, 0, 0,
                              8, 533, 878, 1000, 1000, 980, 886, 753, 753, 722, 776, 576, 659, 816, 808, 714, 537, 314],
                             [24, 0, 0, 0, 129, 129, 122, 106, 59, 4, 0, 0, 24, 47, 0, 0, 12, 39, 0, 12, 35, 0, 63, 35,
                              27, 0, 0, 8, 0, 8, 55, 55, 106, 75, 165, 204, 267, 306, 306, 345, 400, 447, 431, 416, 306,
                              353, 376, 322, 278, 239, 169, 216, 161, 110, 110, 118, 51, 39, 31, 31, 20, 27, 71, 82, 94,
                              98, 98, 184, 325, 537, 686, 820, 757, 525, 306, 133, 82, 8, 51, 35, 82, 43, 0, 0, 35, 0,
                              0, 0, 47, 0, 16, 39, 12, 55, 0, 43, 0, 20, 0, 67, 0, 67, 8, 67, 0, 43, 43, 8, 27, 63, 39,
                              0, 43, 39, 43, 20, 43, 39, 0, 0, 0, 67, 20, 0, 20, 20, 51, 51, 0, 4, 0, 75, 51, 0, 0, 27,
                              27, 0, 82, 20, 39, 106, 184, 200, 137, 157, 75, 0, 125, 184, 200, 129, 0, 0, 12, 0, 0, 47,
                              24, 63, 4, 78, 86, 0, 24, 12, 0, 0, 0, 0, 20, 0, 118, 165, 165, 75, 0, 0, 0, 0, 0, 122,
                              408, 643, 878, 957, 886, 761, 635, 541, 467, 490, 471, 396, 478, 478, 486, 580, 620, 525,
                              478],
                             [47, 47, 71, 0, 63, 71, 75, 59, 51, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 27, 8, 75, 98,
                              8, 118, 114, 102, 145, 98, 200, 255, 337, 369, 376, 384, 361, 376, 369, 306, 290, 267,
                              188, 212, 180, 204, 173, 114, 161, 137, 82, 59, 82, 43, 27, 4, 0, 0, 0, 20, 27, 102, 82,
                              35, 51, 75, 192, 302, 490, 686, 784, 776, 467, 306, 141, 16, 86, 35, 59, 0, 16, 31, 39,
                              16, 8, 0, 47, 0, 0, 0, 31, 43, 31, 55, 31, 20, 0, 0, 0, 20, 67, 43, 0, 20, 43, 0, 31, 20,
                              43, 0, 0, 39, 39, 63, 82, 63, 8, 31, 67, 0, 20, 43, 0, 67, 0, 47, 0, 12, 16, 0, 0, 0, 27,
                              27, 27, 51, 51, 0, 39, 90, 114, 149, 173, 125, 200, 71, 141, 231, 231, 169, 0, 0, 0, 0, 8,
                              55, 118, 78, 31, 47, 63, 35, 24, 12, 0, 0, 0, 0, 20, 98, 133, 141, 94, 8, 0, 0, 0, 0, 67,
                              337, 580, 776, 871, 682, 518, 424, 329, 282, 259, 294, 306, 298, 157, 263, 298, 298, 361,
                              369, 384],
                             [24, 47, 12, 0, 86, 71, 129, 110, 16, 0, 24, 71, 47, 43, 51, 0, 0, 12, 12, 0, 0, 39, 98,
                              122, 129, 184, 231, 231, 247, 247, 294, 235, 310, 286, 263, 369, 275, 243, 235, 227, 196,
                              196, 149, 78, 106, 133, 125, 118, 94, 94, 51, 51, 43, 27, 12, 4, 51, 8, 0, 0, 82, 90, 59,
                              94, 12, 0, 24, 39, 75, 145, 251, 439, 612, 761, 722, 588, 306, 129, 86, 24, 0, 20, 16, 43,
                              4, 0, 63, 0, 0, 0, 0, 82, 8, 67, 67, 0, 31, 0, 67, 0, 20, 0, 20, 20, 20, 67, 67, 20, 20,
                              20, 20, 43, 20, 0, 0, 0, 59, 16, 39, 31, 0, 43, 0, 0, 0, 90, 0, 12, 12, 12, 35, 0, 31, 27,
                              27, 0, 0, 75, 0, 27, 82, 0, 86, 94, 118, 208, 141, 224, 286, 239, 302, 231, 59, 27, 0, 0,
                              0, 0, 67, 31, 8, 94, 86, 55, 47, 0, 0, 16, 0, 67, 67, 122, 102, 67, 47, 0, 0, 0, 0, 31,
                              255, 455, 588, 682, 643, 494, 275, 204, 180, 173, 98, 137, 180, 141, 90, 133, 55, 125,
                              157, 196, 196],
                             [0, 12, 47, 71, 125, 110, 133, 55, 0, 20, 24, 47, 43, 4, 47, 47, 71, 71, 71, 82, 98, 169,
                              192, 224, 286, 286, 302, 224, 271, 263, 263, 247, 216, 239, 192, 196, 165, 157, 157, 125,
                              141, 35, 149, 86, 43, 20, 47, 39, 39, 16, 59, 0, 67, 0, 0, 0, 39, 27, 0, 12, 35, 82, 59,
                              0, 82, 0, 43, 27, 43, 43, 133, 267, 267, 549, 682, 710, 518, 329, 169, 27, 24, 0, 0, 35,
                              0, 0, 0, 4, 31, 0, 0, 55, 31, 43, 43, 0, 31, 8, 31, 0, 35, 0, 59, 20, 20, 20, 67, 0, 8,
                              31, 8, 0, 16, 0, 0, 16, 82, 59, 0, 16, 31, 0, 0, 20, 51, 0, 0, 35, 0, 12, 35, 12, 75, 31,
                              0, 27, 27, 31, 8, 31, 31, 0, 55, 71, 78, 75, 63, 94, 231, 275, 322, 243, 51, 0, 24, 24,
                              24, 0, 43, 0, 75, 122, 90, 31, 8, 12, 0, 51, 16, 43, 24, 67, 55, 78, 67, 0, 0, 20, 0, 161,
                              294, 439, 486, 494, 392, 282, 165, 78, 63, 51, 47, 31, 55, 31, 16, 71, 8, 31, 47, 55,
                              118],
                             [51, 0, 12, 94, 110, 157, 78, 55, 24, 0, 12, 0, 51, 71, 102, 149, 165, 196, 133, 192, 224,
                              247, 212, 263, 247, 208, 208, 184, 161, 161, 137, 90, 137, 145, 137, 133, 118, 47, 35, 94,
                              0, 63, 39, 20, 4, 35, 12, 55, 35, 0, 0, 20, 20, 0, 0, 0, 27, 82, 0, 59, 94, 82, 133, 106,
                              0, 67, 0, 20, 12, 27, 67, 102, 141, 361, 486, 541, 710, 518, 337, 129, 43, 8, 0, 59, 4,
                              75, 59, 0, 0, 0, 63, 8, 8, 0, 43, 43, 43, 20, 43, 0, 0, 12, 78, 0, 20, 20, 0, 75, 0, 0,
                              59, 78, 31, 16, 16, 16, 63, 35, 82, 16, 0, 75, 27, 27, 20, 47, 59, 59, 59, 12, 0, 0, 31,
                              0, 27, 51, 31, 8, 8, 31, 0, 55, 55, 55, 4, 63, 20, 0, 224, 298, 337, 275, 98, 43, 78, 47,
                              16, 0, 67, 43, 122, 98, 43, 31, 8, 8, 12, 59, 51, 78, 47, 90, 47, 31, 8, 55, 0, 0, 161,
                              231, 341, 412, 384, 353, 227, 133, 71, 51, 39, 24, 24, 8, 4, 4, 63, 8, 0, 27, 31, 27, 75],
                             [0, 27, 98, 86, 106, 98, 55, 59, 0, 20, 24, 82, 86, 63, 86, 125, 180, 259, 173, 220, 247,
                              224, 200, 145, 82, 129, 78, 114, 78, 59, 106, 106, 114, 71, 63, 86, 78, 0, 47, 0, 47, 0,
                              43, 51, 0, 0, 12, 12, 0, 12, 12, 0, 59, 0, 20, 39, 16, 0, 0, 47, 24, 86, 51, 67, 0, 0, 0,
                              0, 0, 12, 12, 20, 55, 141, 329, 427, 706, 792, 675, 353, 173, 59, 16, 110, 110, 102, 0,
                              43, 0, 0, 16, 0, 0, 67, 67, 43, 20, 20, 20, 35, 12, 0, 78, 55, 31, 0, 4, 4, 27, 8, 55, 31,
                              8, 0, 39, 0, 16, 59, 12, 0, 27, 27, 0, 4, 0, 47, 35, 0, 12, 12, 35, 0, 8, 0, 75, 0, 0, 0,
                              20, 20, 31, 0, 20, 31, 31, 4, 4, 43, 149, 227, 392, 275, 220, 67, 82, 82, 51, 43, 67, 51,
                              137, 27, 20, 31, 31, 39, 0, 24, 39, 78, 90, 47, 47, 0, 0, 8, 8, 75, 161, 239, 231, 302,
                              200, 165, 86, 55, 43, 39, 8, 8, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 75],
                             [20, 39, 145, 98, 157, 110, 59, 0, 0, 24, 0, 4, 63, 27, 86, 86, 149, 125, 176, 192, 161,
                              129, 114, 114, 67, 90, 67, 118, 47, 67, 55, 12, 59, 0, 67, 0, 0, 67, 0, 35, 12, 0, 0, 12,
                              35, 35, 0, 12, 12, 0, 12, 59, 59, 12, 31, 0, 0, 39, 12, 47, 110, 149, 86, 20, 47, 4, 4, 0,
                              71, 27, 35, 12, 67, 78, 188, 290, 518, 596, 831, 635, 455, 259, 173, 165, 118, 98, 133,
                              118, 43, 71, 47, 16, 8, 43, 0, 43, 43, 43, 35, 12, 35, 12, 55, 78, 8, 55, 8, 8, 31, 0, 55,
                              0, 31, 0, 0, 63, 16, 0, 12, 0, 27, 51, 51, 27, 0, 20, 24, 0, 12, 12, 59, 0, 67, 51, 20,
                              67, 67, 0, 0, 55, 31, 55, 8, 20, 8, 0, 4, 12, 27, 204, 255, 322, 294, 216, 122, 106, 90,
                              78, 82, 71, 20, 0, 4, 0, 31, 20, 8, 55, 8, 63, 0, 51, 27, 0, 0, 110, 133, 204, 239, 145,
                              169, 20, 114, 90, 47, 35, 8, 8, 0, 47, 0, 0, 24, 24, 27, 0, 0, 0, 20, 0, 67],
                             [0, 125, 133, 161, 55, 90, 63, 24, 8, 0, 51, 0, 16, 63, 86, 63, 86, 24, 102, 98, 106, 106,
                              39, 39, 55, 20, 0, 71, 0, 12, 16, 0, 20, 20, 20, 39, 0, 55, 55, 0, 0, 31, 0, 55, 31, 31,
                              8, 31, 0, 12, 0, 12, 0, 0, 16, 0, 16, 63, 12, 51, 71, 133, 55, 78, 0, 39, 0, 0, 27, 12, 4,
                              35, 35, 4, 43, 16, 298, 416, 659, 714, 627, 424, 90, 20, 71, 59, 98, 4, 71, 20, 16, 0, 0,
                              55, 0, 31, 8, 0, 0, 0, 12, 0, 8, 31, 31, 31, 0, 78, 55, 8, 55, 8, 78, 8, 31, 0, 0, 94, 12,
                              47, 90, 4, 75, 75, 20, 43, 67, 24, 35, 59, 0, 43, 94, 51, 43, 67, 67, 4, 4, 51, 31, 31,
                              55, 0, 20, 59, 0, 12, 35, 137, 275, 408, 282, 224, 129, 133, 110, 102, 102, 118, 47, 0,
                              43, 31, 12, 24, 24, 24, 27, 27, 27, 27, 0, 0, 75, 86, 204, 180, 180, 114, 55, 114, 35, 43,
                              27, 0, 0, 27, 75, 51, 0, 0, 24, 4, 78, 8, 0, 16, 20, 55, 0],
                             [0, 102, 165, 137, 137, 63, 78, 0, 31, 24, 51, 27, 75, 63, 0, 39, 71, 24, 94, 94, 12, 82,
                              82, 0, 0, 0, 24, 59, 0, 0, 0, 0, 0, 0, 0, 55, 8, 31, 31, 55, 0, 90, 55, 31, 0, 8, 31, 55,
                              8, 8, 8, 0, 12, 0, 35, 16, 0, 59, 0, 86, 110, 110, 43, 71, 0, 0, 59, 0, 47, 27, 35, 27,
                              35, 67, 4, 67, 71, 314, 463, 631, 706, 541, 243, 27, 27, 35, 98, 110, 102, 0, 0, 0, 0, 20,
                              0, 75, 55, 20, 12, 59, 55, 8, 0, 78, 0, 8, 8, 0, 0, 31, 0, 8, 8, 8, 31, 0, 0, 8, 31, 8,
                              94, 0, 98, 43, 67, 0, 20, 0, 0, 0, 0, 0, 31, 4, 67, 43, 27, 51, 0, 51, 4, 8, 55, 55, 20,
                              35, 0, 43, 82, 110, 322, 349, 282, 278, 161, 153, 141, 137, 129, 118, 47, 118, 196, 169,
                              137, 106, 75, 43, 20, 4, 63, 71, 71, 94, 82, 180, 90, 129, 129, 118, 0, 75, 35, 4, 71, 0,
                              39, 39, 4, 0, 24, 0, 0, 0, 0, 55, 8, 0, 0, 43, 20],
                             [0, 114, 47, 141, 86, 0, 0, 0, 35, 51, 24, 4, 51, 4, 63, 16, 39, 71, 0, 35, 0, 35, 35, 0,
                              0, 0, 0, 43, 90, 47, 0, 0, 0, 31, 8, 31, 8, 8, 0, 8, 31, 0, 0, 0, 31, 55, 55, 31, 31, 55,
                              0, 12, 35, 35, 0, 0, 0, 118, 51, 110, 110, 86, 43, 0, 0, 78, 71, 71, 24, 0, 0, 4, 0, 20,
                              31, 63, 67, 180, 290, 388, 643, 635, 541, 243, 82, 12, 0, 27, 110, 27, 71, 39, 0, 51, 51,
                              16, 16, 75, 0, 8, 31, 31, 31, 31, 55, 0, 8, 8, 55, 31, 0, 31, 8, 31, 0, 78, 0, 16, 51, 16,
                              31, 0, 43, 43, 0, 0, 43, 20, 43, 43, 0, 0, 0, 31, 75, 51, 4, 4, 51, 27, 27, 27, 31, 20, 0,
                              20, 35, 55, 82, 141, 286, 290, 349, 271, 188, 192, 157, 227, 184, 129, 180, 196, 169, 212,
                              216, 145, 153, 129, 106, 0, 102, 122, 153, 106, 47, 169, 82, 141, 94, 35, 78, 4, 51, 12,
                              43, 43, 0, 75, 12, 0, 59, 59, 39, 86, 0, 31, 31, 0, 51, 16, 51],
                             [71, 118, 161, 122, 55, 39, 35, 35, 27, 0, 4, 75, 27, 51, 27, 0, 35, 35, 78, 20, 39, 63,
                              16, 16, 0, 20, 20, 0, 47, 0, 0, 24, 0, 55, 8, 0, 55, 0, 31, 55, 55, 24, 4, 31, 12, 55, 31,
                              75, 0, 55, 35, 0, 12, 75, 16, 94, 24, 47, 94, 86, 110, 110, 86, 59, 12, 8, 0, 47, 0, 0, 0,
                              12, 12, 35, 16, 4, 47, 43, 35, 247, 318, 612, 682, 514, 294, 161, 35, 27, 133, 94, 31, 0,
                              0, 55, 0, 16, 0, 16, 16, 16, 0, 8, 8, 0, 55, 31, 0, 55, 55, 55, 0, 51, 4, 0, 0, 90, 16,
                              75, 0, 51, 51, 0, 71, 0, 67, 43, 20, 43, 20, 75, 0, 94, 78, 31, 0, 27, 4, 27, 0, 27, 27,
                              63, 67, 0, 67, 27, 47, 67, 106, 176, 341, 306, 400, 322, 239, 322, 286, 275, 275, 208,
                              129, 110, 86, 47, 235, 192, 204, 188, 231, 176, 176, 176, 153, 106, 86, 161, 75, 106, 118,
                              0, 27, 0, 47, 86, 0, 20, 43, 63, 59, 12, 0, 0, 0, 27, 27, 31, 67, 0, 75, 75, 0],
                             [47, 106, 129, 129, 78, 43, 12, 0, 27, 51, 4, 8, 0, 4, 27, 51, 51, 0, 31, 31, 55, 20, 0,
                              24, 0, 24, 43, 47, 24, 24, 0, 24, 82, 31, 0, 0, 31, 31, 8, 31, 0, 24, 82, 0, 0, 0, 4, 0,
                              0, 20, 12, 35, 51, 51, 106, 59, 20, 55, 71, 0, 110, 71, 94, 59, 59, 0, 55, 47, 0, 27, 31,
                              8, 12, 35, 0, 16, 12, 35, 67, 141, 200, 447, 596, 604, 463, 333, 125, 82, 4, 71, 43, 39,
                              63, 0, 16, 0, 0, 16, 16, 0, 31, 0, 78, 0, 31, 0, 8, 0, 8, 78, 27, 51, 0, 0, 0, 0, 0, 16,
                              16, 0, 27, 8, 0, 0, 0, 67, 43, 0, 67, 102, 47, 0, 0, 51, 0, 35, 0, 47, 24, 20, 43, 0, 0,
                              0, 4, 51, 51, 75, 133, 224, 376, 353, 353, 341, 408, 475, 447, 443, 361, 227, 153, 145,
                              110, 212, 259, 208, 196, 239, 231, 196, 216, 176, 176, 169, 122, 122, 122, 55, 20, 51, 59,
                              0, 75, 39, 43, 20, 12, 39, 0, 0, 27, 27, 51, 0, 0, 4, 27, 4, 27, 0, 71],
                             [59, 82, 90, 90, 55, 55, 0, 0, 0, 4, 8, 8, 55, 0, 0, 0, 0, 67, 39, 16, 0, 35, 82, 0, 0, 24,
                              20, 59, 59, 0, 0, 12, 0, 31, 8, 31, 8, 31, 31, 8, 8, 67, 31, 0, 27, 27, 0, 27, 16, 63,
                              106, 35, 12, 82, 59, 12, 12, 0, 20, 39, 39, 20, 63, 86, 35, 12, 47, 0, 24, 63, 8, 8, 0, 0,
                              0, 0, 39, 27, 35, 20, 71, 267, 388, 506, 616, 584, 341, 161, 35, 0, 27, 67, 16, 0, 0, 0,
                              75, 0, 27, 75, 0, 8, 55, 8, 31, 94, 24, 71, 0, 0, 75, 0, 0, 20, 0, 8, 51, 16, 51, 51, 51,
                              8, 0, 0, 71, 0, 0, 0, 0, 55, 0, 35, 0, 31, 35, 35, 12, 24, 0, 31, 31, 55, 27, 31, 27, 82,
                              90, 94, 169, 282, 525, 553, 467, 349, 404, 557, 529, 400, 290, 145, 110, 122, 188, 122,
                              157, 122, 122, 47, 129, 153, 153, 129, 192, 145, 122, 90, 75, 20, 0, 82, 0, 43, 78, 39,
                              78, 43, 35, 16, 0, 4, 51, 0, 27, 27, 4, 94, 24, 59, 78, 0, 24],
                             [106, 90, 75, 75, 20, 0, 4, 63, 27, 31, 0, 0, 27, 63, 8, 67, 8, 86, 39, 0, 27, 0, 0, 0, 24,
                              82, 90, 0, 59, 24, 59, 35, 35, 0, 78, 8, 31, 8, 8, 0, 0, 55, 12, 12, 0, 31, 31, 39, 82, 0,
                              82, 59, 82, 59, 12, 12, 0, 35, 12, 0, 39, 39, 20, 4, 86, 59, 12, 0, 0, 86, 35, 31, 31, 8,
                              12, 35, 59, 0, 51, 20, 86, 125, 176, 349, 498, 498, 624, 420, 278, 145, 59, 16, 31, 39,
                              43, 59, 12, 35, 51, 0, 0, 0, 78, 31, 0, 47, 0, 24, 20, 43, 0, 0, 67, 0, 0, 8, 31, 75, 27,
                              0, 27, 8, 0, 24, 0, 0, 35, 12, 31, 39, 67, 0, 4, 43, 0, 59, 0, 35, 12, 12, 0, 27, 31, 39,
                              35, 12, 43, 114, 224, 369, 851, 969, 773, 733, 624, 486, 400, 337, 188, 149, 75, 125, 133,
                              24, 75, 31, 75, 75, 71, 98, 98, 51, 98, 67, 8, 43, 0, 98, 16, 75, 78, 0, 0, 43, 16, 20,
                              35, 27, 0, 0, 27, 51, 0, 0, 0, 27, 0, 0, 0, 67, 0],
                             [63, 4, 63, 0, 43, 39, 0, 0, 0, 8, 4, 63, 27, 27, 71, 8, 0, 8, 24, 24, 24, 86, 0, 12, 82,
                              82, 0, 0, 0, 35, 35, 12, 0, 0, 8, 4, 31, 8, 8, 0, 0, 0, 35, 12, 0, 51, 51, 75, 47, 59, 82,
                              16, 12, 0, 35, 35, 12, 0, 12, 12, 0, 16, 63, 20, 86, 0, 0, 0, 27, 4, 27, 12, 8, 8, 55, 31,
                              0, 0, 4, 4, 20, 12, 75, 224, 333, 498, 616, 616, 576, 412, 239, 145, 110, 78, 16, 0, 0,
                              71, 12, 0, 0, 0, 31, 31, 24, 24, 0, 24, 20, 0, 67, 43, 0, 67, 0, 31, 8, 8, 27, 51, 0, 94,
                              0, 71, 59, 59, 35, 78, 16, 0, 90, 43, 94, 39, 8, 0, 12, 12, 24, 20, 27, 43, 82, 12, 12,
                              35, 67, 196, 294, 678, 1000, 1000, 1000, 835, 655, 443, 345, 220, 165, 129, 20, 4, 94, 27,
                              59, 0, 75, 51, 16, 51, 8, 67, 8, 67, 35, 4, 27, 51, 39, 39, 43, 0, 0, 0, 31, 86, 71, 47,
                              51, 0, 51, 27, 27, 86, 63, 27, 59, 0, 20, 0, 67],
                             [39, 39, 39, 0, 0, 4, 8, 0, 8, 27, 27, 27, 63, 27, 27, 47, 31, 47, 47, 24, 24, 47, 0, 35,
                              0, 24, 0, 0, 12, 12, 82, 0, 0, 16, 16, 16, 0, 59, 35, 12, 0, 35, 12, 35, 0, 98, 16, 98, 0,
                              82, 24, 43, 12, 12, 59, 59, 47, 24, 0, 0, 31, 16, 39, 78, 86, 59, 12, 0, 27, 0, 27, 4, 35,
                              0, 12, 27, 0, 75, 4, 75, 47, 0, 71, 75, 161, 278, 420, 553, 545, 490, 467, 349, 216, 141,
                              110, 71, 8, 8, 16, 0, 8, 51, 0, 27, 8, 0, 55, 0, 67, 20, 43, 43, 20, 0, 43, 43, 31, 0, 71,
                              47, 71, 0, 0, 78, 0, 55, 31, 0, 4, 0, 20, 102, 106, 75, 12, 63, 55, 35, 27, 27, 63, 20,
                              43, 59, 43, 59, 59, 200, 663, 1000, 1000, 1000, 1000, 859, 631, 380, 290, 220, 153, 90,
                              27, 0, 67, 0, 4, 0, 0, 0, 55, 55, 8, 0, 63, 4, 0, 63, 39, 0, 0, 16, 0, 8, 31, 55, 78, 55,
                              4, 0, 71, 27, 0, 0, 0, 39, 0, 0, 63, 0, 0, 0, 8],
                             [78, 16, 63, 35, 0, 4, 31, 8, 0, 27, 63, 4, 4, 4, 4, 4, 24, 0, 47, 24, 47, 47, 71, 0, 12,
                              35, 82, 12, 0, 12, 12, 20, 55, 0, 63, 39, 12, 0, 12, 59, 35, 0, 12, 12, 0, 82, 82, 75, 16,
                              24, 0, 20, 20, 0, 0, 63, 63, 0, 0, 63, 0, 63, 63, 39, 12, 59, 47, 27, 4, 27, 27, 27, 4,
                              27, 86, 0, 75, 4, 4, 27, 0, 4, 20, 47, 59, 129, 255, 310, 373, 412, 451, 435, 451, 392,
                              259, 165, 47, 75, 27, 0, 0, 0, 0, 0, 0, 8, 31, 8, 0, 43, 43, 20, 43, 0, 0, 20, 0, 51, 24,
                              59, 47, 0, 24, 0, 78, 31, 0, 0, 0, 43, 94, 137, 137, 78, 47, 27, 4, 43, 43, 43, 43, 63,
                              75, 75, 106, 165, 224, 580, 945, 1000, 1000, 1000, 1000, 976, 694, 365, 271, 196, 133,
                              114, 20, 0, 0, 8, 12, 0, 4, 75, 16, 0, 31, 8, 8, 0, 0, 75, 0, 0, 16, 16, 39, 8, 0, 31, 31,
                              0, 27, 0, 47, 47, 47, 86, 27, 0, 4, 0, 0, 0, 20, 20, 0],
                             [8, 8, 4, 27, 51, 4, 51, 4, 78, 4, 27, 4, 63, 27, 4, 27, 0, 71, 0, 47, 24, 47, 71, 82, 35,
                              98, 59, 0, 0, 59, 0, 0, 0, 0, 63, 59, 35, 12, 0, 0, 12, 0, 59, 16, 16, 82, 16, 16, 16, 82,
                              0, 0, 0, 0, 0, 0, 63, 0, 4, 63, 39, 47, 27, 0, 63, 24, 0, 0, 0, 27, 4, 27, 0, 0, 27, 4, 4,
                              27, 51, 51, 27, 8, 55, 20, 0, 27, 129, 231, 239, 239, 325, 404, 459, 502, 486, 400, 251,
                              125, 4, 27, 0, 16, 39, 0, 51, 0, 31, 31, 0, 47, 43, 39, 67, 0, 67, 0, 27, 0, 59, 82, 114,
                              114, 59, 0, 0, 118, 0, 20, 31, 82, 114, 165, 169, 110, 51, 35, 35, 51, 55, 63, 71, 82, 94,
                              114, 235, 565, 639, 898, 1000, 1000, 1000, 1000, 1000, 898, 451, 341, 231, 165, 129, 122,
                              20, 0, 0, 24, 8, 0, 75, 51, 63, 0, 4, 75, 4, 35, 0, 0, 16, 16, 75, 0, 16, 0, 78, 55, 55,
                              55, 0, 47, 24, 0, 27, 0, 86, 0, 4, 0, 27, 0, 102, 4, 63],
                             [39, 0, 0, 20, 31, 4, 4, 4, 0, 27, 63, 4, 0, 63, 63, 4, 0, 0, 0, 24, 24, 47, 47, 24, 90,
                              90, 31, 82, 0, 0, 0, 78, 20, 0, 16, 0, 59, 12, 35, 12, 59, 59, 75, 0, 12, 51, 16, 8, 0,
                              16, 82, 20, 0, 0, 43, 20, 20, 0, 39, 0, 0, 43, 0, 27, 27, 24, 24, 86, 4, 27, 4, 0, 59, 82,
                              20, 20, 8, 0, 4, 4, 55, 0, 12, 8, 20, 67, 20, 98, 129, 145, 153, 184, 329, 392, 431, 486,
                              463, 322, 227, 94, 47, 8, 31, 0, 16, 4, 8, 0, 31, 71, 102, 78, 78, 43, 31, 31, 0, 0, 55,
                              90, 145, 118, 102, 16, 0, 0, 0, 0, 20, 0, 47, 161, 161, 153, 75, 43, 43, 82, 86, 102, 102,
                              161, 184, 443, 690, 988, 1000, 1000, 1000, 1000, 1000, 812, 663, 404, 380, 208, 173, 137,
                              145, 180, 145, 90, 0, 0, 31, 35, 0, 0, 4, 4, 0, 4, 0, 82, 0, 16, 16, 0, 0, 16, 16, 31, 0,
                              0, 0, 86, 47, 71, 71, 24, 47, 4, 0, 63, 0, 0, 0, 43, 63, 63, 39],
                             [8, 43, 0, 20, 8, 8, 4, 27, 0, 12, 51, 27, 4, 27, 27, 24, 0, 67, 0, 24, 24, 47, 0, 4, 24,
                              31, 55, 90, 59, 55, 55, 20, 0, 0, 20, 35, 12, 12, 35, 59, 59, 51, 75, 59, 12, 0, 8, 51,
                              27, 27, 27, 31, 35, 12, 0, 43, 43, 20, 43, 0, 43, 67, 75, 4, 75, 0, 63, 0, 8, 86, 86, 82,
                              20, 0, 0, 20, 0, 8, 75, 31, 67, 0, 12, 0, 43, 63, 12, 118, 59, 47, 47, 67, 196, 251, 306,
                              408, 486, 486, 439, 353, 165, 78, 8, 8, 55, 39, 27, 82, 35, 63, 0, 75, 63, 4, 4, 4, 24,
                              24, 55, 98, 141, 220, 188, 94, 20, 4, 0, 0, 0, 24, 94, 141, 141, 125, 118, 106, 98, 106,
                              114, 255, 388, 545, 765, 1000, 1000, 1000, 1000, 973, 816, 647, 333, 380, 294, 278, 278,
                              220, 176, 67, 94, 176, 149, 145, 98, 43, 0, 0, 0, 0, 0, 0, 0, 16, 39, 0, 31, 31, 39, 16,
                              39, 0, 0, 0, 8, 0, 8, 24, 24, 71, 24, 71, 47, 0, 86, 0, 24, 0, 0, 35, 35, 0, 4],
                             [0, 43, 8, 20, 20, 0, 4, 4, 0, 59, 55, 4, 4, 51, 12, 0, 47, 82, 24, 47, 0, 63, 27, 4, 0,
                              71, 31, 98, 12, 55, 55, 55, 0, 0, 0, 78, 16, 82, 59, 59, 82, 35, 82, 0, 75, 0, 27, 0, 27,
                              0, 27, 27, 12, 47, 0, 47, 67, 0, 0, 20, 20, 67, 16, 63, 59, 4, 51, 27, 51, 4, 0, 20, 0, 0,
                              43, 67, 20, 20, 67, 8, 31, 8, 0, 12, 0, 55, 94, 90, 0, 47, 0, 0, 63, 86, 212, 275, 329,
                              431, 510, 463, 392, 286, 122, 59, 12, 4, 0, 0, 0, 39, 51, 51, 82, 27, 0, 4, 0, 47, 4, 16,
                              133, 243, 329, 224, 169, 67, 43, 4, 59, 12, 133, 180, 282, 196, 153, 145, 192, 208, 400,
                              757, 1000, 1000, 1000, 1000, 1000, 882, 784, 620, 306, 282, 161, 129, 259, 212, 173, 129,
                              106, 24, 0, 157, 157, 157, 176, 149, 78, 35, 0, 35, 16, 0, 16, 16, 0, 31, 31, 0, 0, 0, 16,
                              0, 31, 55, 55, 0, 24, 0, 0, 24, 47, 71, 0, 0, 43, 24, 24, 67, 35, 35, 0, 0, 0],
                             [43, 8, 0, 67, 20, 0, 51, 12, 12, 12, 55, 78, 0, 20, 78, 20, 0, 0, 4, 0, 4, 4, 4, 63, 4, 4,
                              51, 102, 0, 0, 55, 20, 0, 20, 0, 86, 51, 47, 82, 98, 51, 16, 78, 20, 90, 4, 4, 27, 75, 0,
                              67, 0, 8, 47, 47, 0, 47, 47, 0, 43, 20, 67, 16, 82, 4, 27, 27, 4, 4, 51, 8, 20, 0, 20, 0,
                              0, 20, 0, 20, 31, 31, 0, 0, 24, 0, 20, 47, 102, 82, 0, 20, 8, 16, 31, 78, 125, 173, 290,
                              322, 384, 467, 451, 365, 196, 71, 35, 4, 0, 63, 0, 0, 16, 12, 0, 16, 16, 16, 47, 0, 63,
                              110, 212, 333, 388, 294, 204, 35, 35, 106, 141, 204, 271, 408, 290, 349, 663, 969, 1000,
                              1000, 1000, 1000, 1000, 1000, 1000, 796, 584, 416, 306, 157, 133, 55, 86, 169, 216, 122,
                              63, 39, 39, 39, 47, 31, 82, 243, 204, 125, 35, 47, 0, 0, 0, 0, 0, 55, 31, 0, 31, 0, 0, 8,
                              31, 78, 55, 8, 71, 71, 47, 47, 47, 0, 24, 24, 20, 67, 0, 24, 0, 59, 35, 0, 59, 0],
                             [8, 8, 20, 0, 12, 75, 4, 0, 35, 86, 55, 63, 39, 0, 55, 20, 55, 24, 27, 27, 4, 4, 86, 0, 4,
                              4, 4, 4, 39, 0, 20, 20, 78, 55, 0, 27, 102, 0, 0, 0, 24, 0, 12, 75, 27, 75, 0, 31, 67, 67,
                              20, 0, 47, 0, 4, 86, 0, 20, 43, 67, 0, 16, 63, 0, 27, 0, 51, 4, 0, 31, 8, 31, 20, 43, 20,
                              0, 0, 0, 0, 20, 0, 0, 31, 24, 59, 35, 55, 27, 122, 27, 39, 0, 8, 0, 0, 90, 94, 141, 227,
                              286, 333, 435, 369, 349, 235, 102, 27, 12, 20, 4, 39, 51, 82, 0, 0, 16, 0, 0, 71, 24, 71,
                              114, 271, 357, 400, 247, 114, 114, 176, 231, 318, 694, 1000, 1000, 1000, 1000, 1000, 1000,
                              1000, 1000, 1000, 1000, 953, 624, 435, 310, 204, 118, 94, 59, 35, 82, 86, 141, 133, 86,
                              63, 39, 31, 31, 67, 149, 204, 204, 157, 102, 35, 47, 0, 39, 8, 47, 0, 12, 0, 0, 71, 12,
                              78, 94, 55, 8, 78, 0, 0, 71, 24, 47, 0, 0, 47, 0, 43, 0, 82, 0, 35, 35, 0, 0, 4],
                             [43, 0, 35, 59, 0, 4, 75, 35, 59, 125, 90, 63, 0, 63, 16, 55, 8, 63, 0, 0, 4, 0, 47, 0, 4,
                              0, 4, 4, 0, 16, 0, 102, 20, 102, 8, 94, 59, 0, 16, 24, 24, 51, 75, 0, 31, 0, 20, 20, 0,
                              20, 0, 0, 24, 71, 86, 0, 0, 0, 0, 20, 39, 75, 39, 39, 63, 27, 27, 75, 4, 8, 31, 8, 31, 0,
                              20, 20, 20, 43, 0, 20, 20, 67, 8, 31, 47, 35, 106, 78, 55, 0, 0, 90, 20, 43, 20, 55, 0,
                              86, 110, 153, 255, 298, 361, 439, 400, 259, 149, 39, 47, 4, 0, 39, 0, 0, 0, 0, 0, 0, 24,
                              20, 55, 82, 176, 298, 357, 482, 392, 251, 337, 408, 839, 1000, 1000, 1000, 1000, 1000,
                              1000, 1000, 1000, 1000, 941, 694, 490, 333, 200, 122, 82, 75, 59, 0, 31, 63, 98, 114, 63,
                              47, 16, 0, 63, 0, 63, 24, 149, 196, 196, 110, 59, 78, 0, 0, 78, 0, 71, 47, 0, 47, 71, 12,
                              47, 47, 47, 12, 0, 71, 24, 0, 71, 47, 24, 0, 0, 43, 0, 0, 0, 0, 0, 0, 35, 0, 63],
                             [0, 35, 35, 78, 4, 39, 75, 59, 78, 24, 51, 8, 63, 0, 16, 16, 0, 24, 0, 0, 47, 0, 12, 0, 71,
                              0, 63, 0, 16, 0, 0, 78, 0, 20, 94, 24, 82, 4, 16, 0, 35, 24, 0, 0, 0, 0, 0, 35, 20, 43,
                              67, 43, 0, 43, 0, 16, 0, 0, 16, 16, 39, 16, 16, 16, 4, 27, 75, 4, 27, 8, 0, 8, 67, 43, 0,
                              43, 0, 0, 0, 67, 67, 0, 31, 31, 55, 35, 55, 82, 75, 63, 16, 0, 0, 0, 47, 20, 55, 55, 20,
                              82, 122, 165, 275, 322, 431, 408, 322, 204, 141, 59, 20, 59, 0, 0, 0, 0, 51, 27, 0, 43,
                              12, 35, 4, 75, 306, 498, 541, 475, 565, 949, 957, 996, 1000, 1000, 1000, 1000, 957, 984,
                              812, 592, 525, 341, 208, 129, 86, 71, 51, 0, 0, 0, 71, 106, 90, 43, 47, 63, 71, 0, 39, 0,
                              31, 0, 133, 114, 196, 133, 59, 0, 0, 0, 12, 12, 12, 0, 0, 12, 71, 47, 47, 0, 0, 71, 0, 24,
                              0, 0, 47, 0, 0, 47, 47, 0, 0, 24, 24, 24, 35, 59, 59, 39, 4],
                             [35, 12, 0, 4, 4, 4, 63, 27, 78, 24, 0, 51, 67, 82, 82, 0, 82, 0, 0, 0, 55, 0, 0, 47, 8,
                              31, 47, 0, 0, 35, 78, 31, 0, 35, 12, 94, 59, 4, 63, 16, 0, 4, 4, 8, 67, 67, 12, 0, 0, 43,
                              67, 0, 20, 20, 39, 0, 39, 75, 0, 0, 16, 0, 16, 0, 75, 0, 51, 0, 27, 0, 0, 43, 0, 67, 43,
                              20, 20, 20, 43, 20, 67, 31, 0, 0, 8, 27, 0, 16, 63, 0, 67, 24, 12, 0, 35, 0, 0, 8, 55, 20,
                              39, 63, 157, 188, 275, 384, 400, 345, 353, 196, 102, 55, 0, 0, 0, 0, 24, 27, 0, 0, 0, 0,
                              20, 75, 204, 455, 647, 855, 949, 949, 957, 957, 1000, 1000, 984, 525, 416, 388, 357, 329,
                              208, 122, 86, 75, 59, 20, 20, 0, 0, 63, 82, 39, 71, 24, 47, 0, 67, 8, 67, 4, 8, 8, 39,
                              110, 149, 180, 75, 27, 0, 35, 16, 47, 0, 47, 47, 0, 47, 12, 0, 0, 0, 47, 71, 47, 24, 0,
                              47, 0, 0, 86, 43, 43, 0, 82, 0, 0, 0, 35, 4, 39, 39],
                             [31, 78, 0, 39, 0, 39, 0, 51, 0, 82, 122, 8, 98, 75, 51, 98, 47, 0, 4, 0, 0, 0, 0, 8, 8,
                              31, 0, 82, 35, 0, 31, 31, 0, 16, 86, 16, 94, 86, 63, 0, 43, 39, 39, 43, 43, 20, 43, 20, 0,
                              43, 67, 67, 0, 0, 16, 0, 75, 0, 0, 39, 0, 39, 16, 0, 0, 0, 4, 0, 39, 0, 90, 67, 0, 67, 0,
                              20, 20, 43, 43, 67, 20, 0, 31, 8, 75, 27, 51, 27, 39, 0, 0, 35, 35, 12, 35, 0, 12, 20, 24,
                              0, 12, 27, 63, 118, 180, 220, 298, 376, 408, 392, 306, 157, 55, 35, 0, 31, 0, 0, 0, 0, 0,
                              0, 98, 184, 306, 498, 910, 949, 949, 957, 957, 965, 965, 1000, 588, 239, 118, 200, 133,
                              98, 82, 67, 47, 24, 16, 0, 0, 8, 0, 94, 86, 110, 71, 20, 16, 59, 43, 24, 24, 24, 43, 8, 8,
                              110, 133, 157, 118, 110, 8, 0, 0, 16, 0, 47, 47, 12, 12, 0, 47, 71, 47, 47, 47, 24, 0, 0,
                              0, 0, 24, 0, 0, 20, 20, 20, 20, 0, 0, 102, 39, 4, 4],
                             [8, 55, 63, 4, 4, 4, 27, 39, 0, 24, 63, 16, 114, 90, 90, 67, 67, 47, 82, 0, 0, 0, 31, 8,
                              31, 55, 82, 82, 82, 55, 78, 102, 0, 12, 12, 35, 78, 0, 43, 51, 0, 0, 39, 43, 20, 0, 0, 0,
                              0, 67, 0, 0, 43, 16, 16, 0, 16, 39, 16, 0, 75, 75, 16, 0, 82, 0, 0, 4, 4, 8, 67, 0, 20,
                              20, 20, 0, 20, 43, 0, 0, 0, 0, 31, 31, 4, 0, 31, 31, 27, 51, 12, 12, 35, 12, 35, 16, 16,
                              0, 0, 0, 20, 35, 20, 27, 71, 133, 220, 282, 353, 439, 439, 384, 275, 184, 125, 78, 35, 59,
                              8, 31, 39, 125, 114, 373, 525, 702, 949, 949, 957, 957, 957, 965, 965, 784, 349, 125, 63,
                              8, 12, 27, 27, 24, 0, 0, 0, 0, 0, 8, 31, 55, 94, 102, 51, 63, 20, 67, 24, 0, 4, 0, 0, 0,
                              8, 51, 63, 122, 173, 110, 63, 0, 0, 16, 0, 0, 71, 0, 0, 0, 0, 71, 0, 0, 71, 0, 71, 0, 24,
                              59, 24, 20, 0, 20, 67, 0, 0, 0, 0, 78, 0, 0, 78],
                             [31, 0, 0, 39, 0, 4, 16, 0, 43, 0, 0, 47, 82, 0, 0, 43, 67, 90, 59, 12, 0, 12, 63, 0, 0,
                              59, 82, 0, 59, 78, 78, 0, 55, 0, 35, 12, 0, 78, 20, 78, 20, 0, 0, 67, 0, 43, 43, 20, 0,
                              20, 0, 0, 75, 0, 39, 0, 0, 16, 0, 82, 16, 16, 16, 0, 0, 16, 16, 75, 0, 0, 4, 4, 20, 20,
                              20, 67, 0, 0, 67, 20, 0, 0, 0, 0, 0, 67, 55, 0, 0, 75, 35, 35, 0, 24, 27, 16, 16, 27, 8,
                              78, 0, 0, 0, 12, 20, 55, 102, 165, 212, 298, 369, 424, 471, 396, 384, 322, 271, 235, 173,
                              169, 51, 20, 196, 475, 510, 584, 753, 949, 957, 957, 965, 965, 965, 416, 267, 165, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 102, 118, 0, 39, 43, 0, 0, 0, 12, 0, 35, 35, 59, 0, 63,
                              86, 86, 86, 157, 110, 51, 31, 71, 39, 0, 0, 12, 0, 47, 0, 24, 24, 47, 47, 0, 47, 59, 24,
                              59, 0, 20, 0, 0, 20, 67, 55, 31, 0, 43, 43, 0, 0],
                             [31, 31, 39, 4, 27, 75, 0, 43, 106, 90, 82, 43, 0, 0, 43, 39, 90, 78, 78, 12, 0, 12, 39,
                              16, 35, 114, 82, 106, 43, 59, 0, 31, 78, 0, 0, 12, 12, 35, 78, 55, 39, 4, 43, 0, 35, 20,
                              0, 0, 0, 0, 67, 20, 39, 16, 0, 0, 16, 75, 39, 16, 39, 0, 16, 39, 0, 16, 0, 43, 4, 0, 63,
                              39, 4, 0, 0, 0, 0, 35, 78, 0, 12, 12, 82, 43, 0, 8, 0, 12, 35, 59, 39, 0, 24, 0, 27, 4,
                              86, 4, 0, 31, 43, 43, 0, 0, 8, 16, 24, 55, 118, 173, 251, 314, 325, 396, 431, 478, 529,
                              549, 420, 349, 239, 149, 298, 490, 584, 580, 639, 792, 957, 957, 965, 965, 690, 416, 247,
                              55, 0, 0, 0, 0, 0, 47, 0, 0, 0, 8, 0, 31, 55, 90, 110, 55, 20, 31, 8, 35, 35, 35, 0, 0,
                              35, 59, 16, 47, 27, 63, 110, 157, 125, 51, 0, 59, 16, 8, 71, 0, 12, 47, 0, 47, 0, 47, 0,
                              24, 0, 0, 0, 59, 0, 0, 20, 0, 0, 0, 0, 0, 78, 0, 78, 0, 0],
                             [55, 0, 16, 0, 67, 82, 82, 82, 0, 0, 63, 20, 67, 35, 35, 59, 0, 35, 106, 12, 122, 122, 39,
                              63, 63, 90, 20, 90, 82, 0, 59, 0, 8, 0, 8, 0, 24, 71, 31, 78, 43, 0, 0, 35, 35, 0, 43, 43,
                              43, 20, 0, 43, 75, 0, 75, 16, 39, 0, 0, 0, 0, 0, 0, 39, 8, 39, 0, 0, 39, 0, 4, 39, 4, 39,
                              0, 67, 0, 0, 8, 0, 78, 16, 82, 0, 20, 67, 35, 59, 35, 35, 0, 31, 16, 63, 39, 47, 0, 24, 8,
                              47, 8, 43, 20, 0, 39, 0, 8, 16, 20, 86, 141, 200, 231, 290, 298, 365, 388, 408, 467, 431,
                              373, 349, 333, 369, 451, 420, 416, 404, 455, 533, 604, 557, 573, 369, 239, 86, 24, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 43, 102, 125, 31, 78, 43, 0, 43, 12, 59, 0, 35, 35, 12, 35, 12, 0,
                              27, 86, 102, 145, 122, 24, 0, 0, 35, 31, 47, 0, 0, 0, 47, 24, 24, 0, 0, 59, 0, 0, 59, 78,
                              0, 0, 78, 0, 55, 20, 31, 0, 55, 31, 0, 27, 0],
                             [55, 78, 0, 0, 67, 59, 98, 0, 0, 0, 0, 0, 12, 0, 0, 59, 35, 75, 35, 12, 12, 63, 63, 110,
                              86, 20, 0, 35, 59, 0, 0, 0, 0, 24, 47, 47, 12, 8, 55, 55, 94, 43, 0, 12, 35, 35, 0, 0, 0,
                              0, 20, 0, 16, 0, 75, 16, 75, 39, 16, 16, 0, 39, 39, 75, 31, 82, 39, 4, 4, 0, 0, 4, 4, 4,
                              0, 0, 12, 78, 31, 55, 8, 0, 67, 43, 20, 12, 59, 59, 59, 12, 39, 63, 31, 0, 35, 35, 0, 47,
                              24, 78, 47, 35, 0, 35, 0, 0, 16, 35, 12, 0, 75, 35, 86, 82, 94, 137, 129, 224, 255, 235,
                              259, 349, 302, 216, 216, 4, 102, 118, 153, 110, 267, 314, 451, 404, 333, 216, 102, 55, 27,
                              0, 0, 0, 31, 8, 0, 8, 102, 102, 125, 102, 0, 78, 43, 0, 8, 12, 59, 12, 12, 59, 0, 59, 59,
                              59, 27, 12, 98, 145, 98, 47, 47, 0, 0, 16, 71, 12, 0, 47, 0, 0, 24, 0, 24, 0, 0, 0, 24, 0,
                              20, 55, 0, 20, 0, 78, 78, 31, 0, 31, 55, 86, 0],
                             [8, 0, 8, 31, 82, 59, 35, 82, 55, 4, 0, 0, 0, 0, 47, 0, 0, 12, 0, 82, 39, 39, 16, 63, 63,
                              0, 35, 12, 0, 35, 24, 0, 4, 0, 0, 0, 47, 78, 31, 8, 78, 67, 43, 0, 0, 59, 59, 43, 43, 0,
                              20, 39, 39, 0, 39, 75, 75, 0, 39, 0, 0, 39, 75, 0, 8, 0, 0, 0, 0, 4, 0, 63, 0, 0, 67, 35,
                              35, 35, 8, 55, 31, 20, 0, 43, 35, 35, 12, 0, 35, 12, 16, 63, 0, 35, 0, 0, 12, 0, 47, 24,
                              0, 35, 0, 0, 27, 0, 27, 51, 0, 0, 71, 0, 0, 0, 0, 24, 0, 0, 0, 90, 106, 129, 173, 0, 0, 0,
                              0, 51, 39, 0, 20, 322, 380, 404, 298, 231, 161, 63, 0, 27, 0, 0, 20, 75, 31, 67, 43, 90,
                              0, 0, 55, 27, 63, 43, 59, 0, 59, 35, 0, 12, 59, 59, 35, 12, 0, 55, 59, 106, 98, 59, 47, 0,
                              0, 16, 78, 0, 47, 12, 24, 0, 47, 0, 0, 24, 0, 24, 0, 78, 0, 55, 55, 20, 55, 0, 0, 55, 0,
                              0, 0, 0, 0],
                             [8, 0, 67, 75, 0, 59, 12, 12, 0, 4, 0, 31, 47, 0, 12, 0, 31, 16, 0, 16, 0, 16, 39, 16, 16,
                              0, 59, 12, 0, 0, 0, 0, 86, 0, 0, 0, 0, 31, 78, 78, 78, 94, 20, 43, 0, 59, 94, 12, 0, 0,
                              39, 75, 0, 0, 16, 0, 39, 39, 39, 16, 39, 39, 16, 8, 55, 0, 39, 4, 4, 4, 0, 39, 39, 0, 0,
                              35, 0, 59, 31, 31, 0, 67, 0, 0, 0, 35, 0, 12, 59, 12, 0, 16, 78, 0, 12, 12, 35, 59, 0, 71,
                              31, 102, 8, 0, 55, 31, 78, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 122, 169, 310, 310, 278, 243, 208, 149, 75, 20, 145, 125, 0, 24, 0, 0, 0, 78,
                              0, 4, 0, 0, 59, 12, 59, 0, 0, 0, 59, 12, 35, 86, 0, 8, 82, 35, 75, 75, 47, 0, 0, 0, 31, 0,
                              71, 0, 47, 47, 47, 86, 0, 0, 0, 0, 0, 55, 20, 0, 20, 0, 55, 0, 31, 31, 0, 55, 31, 0, 27]])
        data = data.astype(float) / 255
        expected = expected.astype(float) / 1000
        workspace, module = self.make_workspace(data, None)
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressSpeckles))
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_NEURITES
        module.neurite_choice.value = E.N_GRADIENT
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(np.all(np.abs(result.pixel_data - expected) < .002))

    def test_04_02_enhance_neurites_tubeness_positive(self):
        image = np.zeros((20, 30))
        image[5:15, 10:20] = np.identity(10)
        workspace, module = self.make_workspace(image, None)
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressSpeckles))
        module.method.value = E.ENHANCE
        module.neurite_choice.value = E.N_TUBENESS
        module.smoothing.value = 1.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = result.pixel_data
        self.assertTrue(np.all(pixel_data[image > 0] > 0))

    def test_04_03_enhance_neurites_tubeness_negative(self):
        image = np.ones((20, 30))
        image[5:15, 10:20] -= np.identity(10)
        workspace, module = self.make_workspace(image, None)
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressSpeckles))
        module.method.value = E.ENHANCE
        module.neurite_choice.value = E.N_TUBENESS
        module.smoothing.value = 1.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = result.pixel_data
        np.testing.assert_array_almost_equal(pixel_data, 0)

    def test_05_01_enhance_dark_holes(self):
        '''Check enhancement of dark holes'''
        #
        # enhance_dark_holes's function is tested more extensively
        # in test_filter
        #
        np.random.seed(0)
        for i, j in ((2, 5), (3, 7), (4, 4)):
            data = np.random.uniform(size=(40, 40)).astype(np.float32)
            expected = enhance_dark_holes(data, i, j)
            workspace, module = self.make_workspace(data, None)
            self.assertTrue(isinstance(module, E.EnhanceOrSuppressSpeckles))
            module.method.value = E.ENHANCE
            module.enhance_method.value = E.E_DARK_HOLES
            module.hole_size.min = i * 2
            module.hole_size.max = j * 2
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(np.all(result.pixel_data == expected))

    def test_06_01_enhance_circles(self):
        i, j = np.mgrid[-15:16, -15:16]
        circle = np.abs(np.sqrt(i * i + j * j) - 6) <= 1.5
        workspace, module = self.make_workspace(circle, None)
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_CIRCLES
        module.object_size.value = 12
        module.run(workspace)
        img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertEqual(img[15, 15], 1)
        self.assertTrue(np.all(img[np.abs(np.sqrt(i * i + j * j) - 6) < 1.5] < .25))

    def test_06_02_enhance_masked_circles(self):
        img = np.zeros((31, 62))
        mask = np.ones((31, 62), bool)
        i, j = np.mgrid[-15:16, -15:16]
        circle = np.abs(np.sqrt(i * i + j * j) - 6) <= 1.5
        # Do one circle
        img[:, :31] = circle
        # Do a second, but mask it
        img[:, 31:] = circle
        mask[:, 31:][circle] = False
        workspace, module = self.make_workspace(img, mask)
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_CIRCLES
        module.object_size.value = 12
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertEqual(result[15, 15], 1)
        self.assertEqual(result[15, 15 + 31], 0)

    def test_07_01_enhance_dic(self):
        img = np.ones((21, 43)) * .5
        img[5:15, 10] = 1
        img[5:15, 15] = 0
        workspace, module = self.make_workspace(img, np.ones(img.shape))
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_DIC
        module.angle.value = 90
        module.decay.value = 1
        module.smoothing.value = 0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        expected = np.zeros(img.shape)
        expected[5:15, 10] = .5
        expected[5:15, 11:15] = 1
        expected[5:15, 15] = .5
        np.testing.assert_almost_equal(result, expected)

        module.decay.value = .9
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(result[5:15, 12:14] < 1))

        module.decay.value = 1
        module.smoothing.value = 1
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(np.all(result[4, 11:15] > .1))

    def test_08_01_enhance_variance(self):
        r = np.random.RandomState()
        r.seed(81)
        img = r.uniform(size=(19, 24))
        sigma = 2.1
        workspace, module = self.make_workspace(img, np.ones(img.shape))
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_TEXTURE
        module.smoothing.value = sigma
        module.run(workspace)
        expected = E.variance_transform(img, sigma)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        np.testing.assert_almost_equal(result, expected)

    def test_08_02_enhance_variance_masked(self):
        r = np.random.RandomState()
        r.seed(81)
        img = r.uniform(size=(19, 24))
        mask = r.uniform(size=img.shape) > .25
        sigma = 2.1
        workspace, module = self.make_workspace(img, mask)
        self.assertTrue(isinstance(module, E.EnhanceOrSuppressFeatures))
        module.method.value = E.ENHANCE
        module.enhance_method.value = E.E_TEXTURE
        module.smoothing.value = sigma
        module.run(workspace)
        expected = E.variance_transform(img, sigma, mask)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        np.testing.assert_almost_equal(result[mask], expected[mask])
