import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.colortogray
import cellprofiler.modules.injectimage
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy
import tests.modules

cellprofiler.preferences.set_headless()

IMAGE_NAME = "image"
OUTPUT_IMAGE_F = "outputimage%d"


class TestColorToGray(unittest.TestCase):
    def get_my_image(self):
        """A color image with red in the upper left, green in the lower left and blue in the upper right"""
        img = numpy.zeros((50, 50, 3))
        img[0:25, 0:25, 0] = 1
        img[0:25, 25:50, 1] = 1
        img[25:50, 0:25, 2] = 1
        return img

    def test_00_00_init(self):
        x = cellprofiler.modules.colortogray.ColorToGray()

    def test_01_01_combine(self):
        img = self.get_my_image()
        inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
        inj.module_num = 1
        ctg = cellprofiler.modules.colortogray.ColorToGray()
        ctg.module_num = 2
        ctg.image_name.value = "my_image"
        ctg.combine_or_split.value = cellprofiler.modules.colortogray.COMBINE
        ctg.red_contribution.value = 1
        ctg.green_contribution.value = 2
        ctg.blue_contribution.value = 3
        ctg.grayscale_name.value = "my_grayscale"
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(inj)
        pipeline.add_module(ctg)
        pipeline.test_valid()

        measurements = cellprofiler.measurement.Measurements()
        object_set = cellprofiler.region.Set()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, inj, None, None, measurements,
                                                     image_set_list, None)
        inj.prepare_run(workspace)
        inj.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        inj.run(cellprofiler.workspace.Workspace(pipeline, inj, image_set, object_set, measurements, None))
        ctg.run(cellprofiler.workspace.Workspace(pipeline, ctg, image_set, object_set, measurements, None))
        grayscale = image_set.get_image("my_grayscale")
        self.assertTrue(grayscale)
        img = grayscale.image
        self.assertAlmostEqual(img[0, 0], 1.0 / 6.0)
        self.assertAlmostEqual(img[0, 25], 1.0 / 3.0)
        self.assertAlmostEqual(img[25, 0], 1.0 / 2.0)
        self.assertAlmostEqual(img[25, 25], 0)

    def test_01_02_split_all(self):
        img = self.get_my_image()
        inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
        inj.module_num = 1
        ctg = cellprofiler.modules.colortogray.ColorToGray()
        ctg.module_num = 2
        ctg.image_name.value = "my_image"
        ctg.combine_or_split.value = cellprofiler.modules.colortogray.SPLIT
        ctg.use_red.value = True
        ctg.use_blue.value = True
        ctg.use_green.value = True
        ctg.red_name.value = "my_red"
        ctg.green_name.value = "my_green"
        ctg.blue_name.value = "my_blue"
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(inj)
        pipeline.add_module(ctg)
        pipeline.test_valid()

        measurements = cellprofiler.measurement.Measurements()
        object_set = cellprofiler.region.Set()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, inj, None, None, measurements,
                                                     image_set_list, None)
        inj.prepare_run(workspace)
        inj.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        inj.run(cellprofiler.workspace.Workspace(pipeline, inj, image_set, object_set, measurements, None))
        ctg.run(cellprofiler.workspace.Workspace(pipeline, ctg, image_set, object_set, measurements, None))
        red = image_set.get_image("my_red")
        self.assertTrue(red)
        img = red.image
        self.assertAlmostEqual(img[0, 0], 1)
        self.assertAlmostEqual(img[0, 25], 0)
        self.assertAlmostEqual(img[25, 0], 0)
        self.assertAlmostEqual(img[25, 25], 0)
        green = image_set.get_image("my_green")
        self.assertTrue(green)
        img = green.image
        self.assertAlmostEqual(img[0, 0], 0)
        self.assertAlmostEqual(img[0, 25], 1)
        self.assertAlmostEqual(img[25, 0], 0)
        self.assertAlmostEqual(img[25, 25], 0)
        blue = image_set.get_image("my_blue")
        self.assertTrue(blue)
        img = blue.image
        self.assertAlmostEqual(img[0, 0], 0)
        self.assertAlmostEqual(img[0, 25], 0)
        self.assertAlmostEqual(img[25, 0], 1)
        self.assertAlmostEqual(img[25, 25], 0)

    def test_01_03_combine_channels(self):
        numpy.random.seed(13)
        image = numpy.random.uniform(size=(20, 10, 5))
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

        module = cellprofiler.modules.colortogray.ColorToGray()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.combine_or_split.value = cellprofiler.modules.colortogray.COMBINE
        module.grayscale_name.value = OUTPUT_IMAGE_F % 1
        module.rgb_or_channels.value = cellprofiler.modules.colortogray.CH_CHANNELS
        module.add_channel()
        module.add_channel()

        channel_indexes = numpy.array([2, 0, 3])
        factors = numpy.random.uniform(size=3)
        divisor = numpy.sum(factors)
        expected = numpy.zeros((20, 10))
        for i, channel_index in enumerate(channel_indexes):
            module.channels[i].channel_choice.value = module.channel_names[channel_index]
            module.channels[i].contribution.value_text = "%.10f" % factors[i]
            expected += image[:, :, channel_index] * factors[i] / divisor

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        pixels = image_set.get_image(module.grayscale_name.value).pixel_data
        self.assertEqual(pixels.ndim, 2)
        self.assertEqual(tuple(pixels.shape), (20, 10))
        numpy.testing.assert_almost_equal(expected, pixels)

    def test_01_04_split_channels(self):
        numpy.random.seed(13)
        image = numpy.random.uniform(size=(20, 10, 5))
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

        module = cellprofiler.modules.colortogray.ColorToGray()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.combine_or_split.value = cellprofiler.modules.colortogray.SPLIT
        module.rgb_or_channels.value = cellprofiler.modules.colortogray.CH_CHANNELS
        module.add_channel()
        module.add_channel()

        channel_indexes = numpy.array([1, 4, 2])
        for i, channel_index in enumerate(channel_indexes):
            module.channels[i].channel_choice.value = module.channel_names[channel_index]
            module.channels[i].image_name.value = OUTPUT_IMAGE_F % i

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        for i, channel_index in enumerate(channel_indexes):
            pixels = image_set.get_image(module.channels[i].image_name.value).pixel_data
            self.assertEqual(pixels.ndim, 2)
            self.assertEqual(tuple(pixels.shape), (20, 10))
            numpy.testing.assert_almost_equal(image[:, :, channel_index], pixels)

    def test_2_3_load_combine(self):
        data = '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'
        pipeline = tests.modules.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertEqual(module.image_name.value, "TestInput")
        self.assertTrue(module.should_combine)
        self.assertEqual(module.grayscale_name.value, "TestGray")
        self.assertEqual(module.red_contribution.value, 1)
        self.assertEqual(module.green_contribution.value, 2)
        self.assertEqual(module.blue_contribution.value, 3)

    def test_2_4_load_split(self):
        data = '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'
        pipeline = tests.modules.load_pipeline(self, data)
        module = pipeline.module(1)
        self.assertTrue(isinstance(module, cellprofiler.modules.colortogray.ColorToGray))
        self.assertEqual(module.image_name.value, "TestInput")
        self.assertTrue(module.should_split)
        self.assertTrue(module.use_red.value)
        self.assertEqual(module.red_name.value, "TestRed")
        self.assertTrue(module.use_green.value)
        self.assertEqual(module.green_name.value, "TestGreen")
        self.assertFalse(module.use_blue.value)
        self.assertEqual(module.rgb_or_channels, cellprofiler.modules.colortogray.CH_RGB)

    def test_2_5_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10308

ColorToGray:[module_num:1|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Conversion method:Combine
    Image type\x3A:Channels
    Name the output image:OrigGrayw
    Relative weight of the red channel:1
    Relative weight of the green channel:3
    Relative weight of the blue channel:5
    Convert red to gray?:Yes
    Name the output image:OrigRedx
    Convert green to gray?:Yes
    Name the output image:OrigGreeny
    Convert blue to gray?:Yes
    Name the output image:OrigBluez
    Channel count:3
    Channel number\x3A:Red\x3A 1
    Relative weight of the channel:1
    Image name\x3A:RedChannel1
    Channel number\x3A:Blue\x3A 3
    Relative weight of the channel:2
    Image name\x3A:GreenChannel2
    Channel number\x3A:Green\x3A 2
    Relative weight of the channel:3
    Image name\x3A:BlueChannel3
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.colortogray.ColorToGray))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.combine_or_split, cellprofiler.modules.colortogray.COMBINE)
        self.assertEqual(module.rgb_or_channels, cellprofiler.modules.colortogray.CH_CHANNELS)
        self.assertEqual(module.grayscale_name, "OrigGrayw")
        self.assertEqual(module.red_contribution, 1)
        self.assertEqual(module.green_contribution, 3)
        self.assertEqual(module.blue_contribution, 5)
        self.assertTrue(module.use_red)
        self.assertTrue(module.use_green)
        self.assertTrue(module.use_blue)
        self.assertEqual(module.red_name, "OrigRedx")
        self.assertEqual(module.green_name, "OrigGreeny")
        self.assertEqual(module.blue_name, "OrigBluez")
        self.assertEqual(module.channel_count.value, 3)
        self.assertEqual(module.channels[0].channel_choice, module.channel_names[0])
        self.assertEqual(module.channels[1].channel_choice, module.channel_names[2])
        self.assertEqual(module.channels[2].channel_choice, module.channel_names[1])
        self.assertEqual(module.channels[0].contribution, 1)
        self.assertEqual(module.channels[1].contribution, 2)
        self.assertEqual(module.channels[2].contribution, 3)
        self.assertEqual(module.channels[0].image_name, "RedChannel1")
        self.assertEqual(module.channels[1].image_name, "GreenChannel2")
        self.assertEqual(module.channels[2].image_name, "BlueChannel3")
