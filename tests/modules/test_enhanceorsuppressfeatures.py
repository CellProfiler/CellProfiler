import StringIO
import base64
import os.path
import unittest
import zlib

import centrosome.filter
import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.enhanceorsuppressfeatures
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import pytest


cellprofiler.preferences.set_headless()


@pytest.fixture(scope="function")
def image(request):
    image = cellprofiler.image.Image()

    image.dimensions = request.param

    return image


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures()

    module.x_name.value = "input"

    module.y_name.value = "output"

    return module


@pytest.fixture(scope="function")
def workspace(image, module):
    image_set_list = cellprofiler.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add("input", image)

    return cellprofiler.workspace.Workspace(
        pipeline=cellprofiler.pipeline.Pipeline(),
        module=module,
        image_set=image_set,
        object_set=cellprofiler.object.ObjectSet(),
        measurements=cellprofiler.measurement.Measurements(),
        image_set_list=image_set_list
    )


def test_enhance_zero(image, module, workspace):
    image.pixel_data = numpy.zeros((10, 10))

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    assert numpy.all(actual == 0)


def test_suppress_zero(image, module, workspace):
    image.pixel_data = numpy.zeros((10, 10))

    module.method.value = "Suppress"

    module.object_size.value = 10

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    assert numpy.all(actual == 0)


INPUT_IMAGE_NAME = 'myimage'
OUTPUT_IMAGE_NAME = 'myfilteredimage'


class TestEnhanceOrSuppressSpeckles(unittest.TestCase):
    def make_workspace(self, image, mask):
        '''Make a workspace for testing FilterByObjectMeasurement'''
        module = cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressSpeckles()
        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.object.ObjectSet()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        image_set.add(INPUT_IMAGE_NAME, cellprofiler.image.Image(image, mask))
        module.x_name.value = INPUT_IMAGE_NAME
        module.y_name.value = OUTPUT_IMAGE_NAME
        return workspace, module

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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertEqual(module.module_name, 'EnhanceOrSuppressFeatures')
        self.assertEqual(module.x_name.value, 'MyImage')
        self.assertEqual(module.y_name.value, 'MyEnhancedImage')
        self.assertEqual(module.method.value, cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE)
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        for module, (input_name, output_name, operation, feature_size,
                     feature_type, min_range, max_range) in zip(
                pipeline.modules(), (
                        ("Initial", "EnhancedSpeckles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 11, cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES, 1, 10),
                        ("EnhancedSpeckles", "EnhancedNeurites", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES, 1, 10),
                        ("EnhancedNeurites", "EnhancedDarkHoles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_DARK_HOLES, 4, 11),
                        ("EnhancedDarkHoles", "EnhancedCircles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES, 4, 11),
                        ("EnhancedCircles", "Suppressed", cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS, 13, cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES, 4, 11))):
            self.assertEqual(module.module_name, 'EnhanceOrSuppressFeatures')
            self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
            self.assertEqual(module.x_name, input_name)
            self.assertEqual(module.y_name, output_name)
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.x_name, "DNA")
        self.assertEqual(module.y_name, "EnhancedTexture")
        self.assertEqual(module.method, cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE)
        self.assertEqual(module.enhance_method, cellprofiler.modules.enhanceorsuppressfeatures.E_TEXTURE)
        self.assertEqual(module.smoothing, 3.5)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 45)
        self.assertEqual(module.decay, .9)
        self.assertEqual(module.speckle_accuracy, cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.enhance_method, cellprofiler.modules.enhanceorsuppressfeatures.E_DIC)

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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.x_name, "Dendrite")
        self.assertEqual(module.y_name, "EnhancedDendrite")
        self.assertEqual(module.method, cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE)
        self.assertEqual(module.enhance_method, cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES)
        self.assertEqual(module.smoothing, 2.0)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.decay, .95)
        self.assertEqual(module.neurite_choice, cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS)
        self.assertEqual(module.speckle_accuracy, cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.speckle_accuracy, cellprofiler.modules.enhanceorsuppressfeatures.S_FAST)

    def test_01_05_load_v4(self):
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.x_name, "Dendrite")
        self.assertEqual(module.y_name, "EnhancedDendrite")
        self.assertEqual(module.method, cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE)
        self.assertEqual(module.enhance_method, cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES)
        self.assertEqual(module.smoothing, 2.0)
        self.assertEqual(module.object_size, 10)
        self.assertEqual(module.hole_size.min, 1)
        self.assertEqual(module.hole_size.max, 10)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.decay, .95)
        self.assertEqual(module.neurite_choice, cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        self.assertEqual(module.neurite_choice, cellprofiler.modules.enhanceorsuppressfeatures.N_GRADIENT)

    def test_02_01_enhance(self):
        '''Enhance an image composed of two circles of different diameters'''
        #
        # Make an image which has circles of diameters 10 and 7. We should
        # keep the smaller circle and erase the larger
        #
        image = numpy.zeros((11, 20))
        expected = numpy.zeros((11, 20))
        i, j = numpy.mgrid[-5:6, -5:16]
        image[i ** 2 + j ** 2 < 23] = 1
        i, j = numpy.mgrid[-5:6, -15:5]
        image[i ** 2 + j ** 2 <= 9] = 1
        expected[i ** 2 + j ** 2 <= 9] = 1
        workspace, module = self.make_workspace(image,
                                                numpy.ones(image.shape, bool))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(numpy.all(result.pixel_data == expected))

    def test_02_02_suppress(self):
        '''Suppress a speckle in an image composed of two circles'''
        image = numpy.zeros((11, 20))
        expected = numpy.zeros((11, 20))
        i, j = numpy.mgrid[-5:6, -5:15]
        image[i ** 2 + j ** 2 <= 22] = 1
        expected[i ** 2 + j ** 2 <= 22] = 1
        i, j = numpy.mgrid[-5:6, -15:5]
        image[i ** 2 + j ** 2 <= 9] = 1
        workspace, module = self.make_workspace(image, None)
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(numpy.all(result.pixel_data == expected))

    def test_03_01_enhancemask(self):
        '''Enhance a speckles image, masking a portion'''
        image = numpy.zeros((10, 10))
        mask = numpy.ones((10, 10), bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation and it
        # should be zero after the subtraction
        #
        i, j = numpy.mgrid[-5:5, -5:5]
        image[5, 5] = 1
        mask[numpy.logical_and(i ** 2 + j ** 2 <= 16, image == 0)] = False
        for speckle_accuracy in cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW, cellprofiler.modules.enhanceorsuppressfeatures.S_FAST:
            #
            # Prove that, without the mask, the image is zero
            #
            workspace, module = self.make_workspace(image, None)
            assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
            module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
            module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES
            module.speckle_accuracy.value = speckle_accuracy
            module.object_size.value = 7
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(result.pixel_data == image))
            #
            # rescue the point with the mask
            #
            workspace, module = self.make_workspace(image, mask)
            module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
            module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES
            module.speckle_accuracy.value = speckle_accuracy
            module.object_size.value = 7
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertFalse(result is None)
            self.assertTrue(numpy.all(result.pixel_data == 0))

    def test_03_02_suppressmask(self):
        '''Suppress a speckles image, masking a portion'''
        image = numpy.zeros((10, 10))
        mask = numpy.ones((10, 10), bool)
        #
        # Put a single point in the middle of the image. The mask
        # should protect the point against the opening operation
        #
        i, j = numpy.mgrid[-5:5, -5:5]
        image[5, 5] = 1
        mask[numpy.logical_and(i ** 2 + j ** 2 <= 16, image == 0)] = False
        #
        # Prove that, without the mask, the speckle is removed
        #
        workspace, module = self.make_workspace(image, None)
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(result.pixel_data == 0))
        #
        # rescue the point with the mask
        #
        workspace, module = self.make_workspace(image, mask)
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS
        module.object_size.value = 7
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertFalse(result is None)
        self.assertTrue(numpy.all(result.pixel_data == image))

    def test_04_01_enhance_neurites(self):
        '''Check enhance neurites against Matlab'''
        resources = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "resources"))
        data = numpy.load(os.path.join(resources, "neurite.npy"))
        expected = numpy.load(os.path.join(resources, "enhanced_neurite.npy"))
        data = data.astype(float) / 255
        expected = expected.astype(float) / 1000
        workspace, module = self.make_workspace(data, None)
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressSpeckles))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES
        module.neurite_choice.value = cellprofiler.modules.enhanceorsuppressfeatures.N_GRADIENT
        module.object_size.value = 8
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(numpy.abs(result.pixel_data - expected) < .002))

    def test_04_02_enhance_neurites_tubeness_positive(self):
        image = numpy.zeros((20, 30))
        image[5:15, 10:20] = numpy.identity(10)
        workspace, module = self.make_workspace(image, None)
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressSpeckles))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.neurite_choice.value = cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS
        module.smoothing.value = 1.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = result.pixel_data
        self.assertTrue(numpy.all(pixel_data[image > 0] > 0))

    def test_04_03_enhance_neurites_tubeness_negative(self):
        image = numpy.ones((20, 30))
        image[5:15, 10:20] -= numpy.identity(10)
        workspace, module = self.make_workspace(image, None)
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressSpeckles))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.neurite_choice.value = cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS
        module.smoothing.value = 1.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = result.pixel_data
        numpy.testing.assert_array_almost_equal(pixel_data, 0)

    def test_05_01_enhance_dark_holes(self):
        '''Check enhancement of dark holes'''
        #
        # enhance_dark_holes's function is tested more extensively
        # in test_filter
        #
        numpy.random.seed(0)
        for i, j in ((2, 5), (3, 7), (4, 4)):
            data = numpy.random.uniform(size=(40, 40)).astype(numpy.float32)
            expected = centrosome.filter.enhance_dark_holes(data, i, j)
            workspace, module = self.make_workspace(data, None)
            self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressSpeckles))
            module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
            module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_DARK_HOLES
            module.hole_size.min = i * 2
            module.hole_size.max = j * 2
            module.run(workspace)
            result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            self.assertTrue(numpy.all(result.pixel_data == expected))

    def test_06_01_enhance_circles(self):
        i, j = numpy.mgrid[-15:16, -15:16]
        circle = numpy.abs(numpy.sqrt(i * i + j * j) - 6) <= 1.5
        workspace, module = self.make_workspace(circle, None)
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES
        module.object_size.value = 12
        module.run(workspace)
        img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertEqual(img[15, 15], 1)
        self.assertTrue(numpy.all(img[numpy.abs(numpy.sqrt(i * i + j * j) - 6) < 1.5] < .25))

    def test_06_02_enhance_masked_circles(self):
        img = numpy.zeros((31, 62))
        mask = numpy.ones((31, 62), bool)
        i, j = numpy.mgrid[-15:16, -15:16]
        circle = numpy.abs(numpy.sqrt(i * i + j * j) - 6) <= 1.5
        # Do one circle
        img[:, :31] = circle
        # Do a second, but mask it
        img[:, 31:] = circle
        mask[:, 31:][circle] = False
        workspace, module = self.make_workspace(img, mask)
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES
        module.object_size.value = 12
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertEqual(result[15, 15], 1)
        self.assertEqual(result[15, 15 + 31], 0)

    def test_07_01_enhance_dic(self):
        img = numpy.ones((21, 43)) * .5
        img[5:15, 10] = 1
        img[5:15, 15] = 0
        workspace, module = self.make_workspace(img, numpy.ones(img.shape))
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_DIC
        module.angle.value = 90
        module.decay.value = 1
        module.smoothing.value = 0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        expected = numpy.zeros(img.shape)
        expected[5:15, 10] = .5
        expected[5:15, 11:15] = 1
        expected[5:15, 15] = .5
        numpy.testing.assert_almost_equal(result, expected)

        module.decay.value = .9
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(numpy.all(result[5:15, 12:14] < 1))

        module.decay.value = 1
        module.smoothing.value = 1
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(numpy.all(result[4, 11:15] > .1))

    def test_08_01_enhance_variance(self):
        r = numpy.random.RandomState()
        r.seed(81)
        img = r.uniform(size=(19, 24))
        sigma = 2.1
        workspace, module = self.make_workspace(img, numpy.ones(img.shape))
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_TEXTURE
        module.smoothing.value = sigma
        module.run(workspace)
        expected = centrosome.filter.variance_transform(img, sigma)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_almost_equal(result, expected)

    def test_08_02_enhance_variance_masked(self):
        r = numpy.random.RandomState()
        r.seed(81)
        img = r.uniform(size=(19, 24))
        mask = r.uniform(size=img.shape) > .25
        sigma = 2.1
        workspace, module = self.make_workspace(img, mask)
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures))
        module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
        module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_TEXTURE
        module.smoothing.value = sigma
        module.run(workspace)
        expected = centrosome.filter.variance_transform(img, sigma, mask)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_almost_equal(result[mask], expected[mask])
