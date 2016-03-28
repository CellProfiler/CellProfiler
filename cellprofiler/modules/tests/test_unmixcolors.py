'''test_unmixcolors - test the unmixcolors module
'''

import unittest
from StringIO import StringIO

import numpy as np

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.unmixcolors as U
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw

INPUT_IMAGE = "inputimage"


def output_image_name(idx):
    return "outputimage%d" % idx


class TestUnmixColors(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10268

UnmixColors:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Stain count:13
    Color image\x3A:Color
    Image name\x3A:Hematoxylin
    Stain:Hematoxylin
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:Eosin
    Stain:Eosin
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:DAB
    Stain:DAB
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:FastRed
    Stain:Fast red
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:FastBlue
    Stain:Fast blue
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:MethylGreen
    Stain:Methyl green
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:AEC
    Stain:AEC
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name:AnilineBlue
    Stain:Aniline blue
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:Azocarmine
    Stain:Azocarmine
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:AlicanBlue
    Stain:Alican blue
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:PAS
    Stain:PAS
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:HematoxylinAndPAS
    Stain:Hematoxylin and PAS
    Red absorbance\x3A:0.5
    Green absorbance\x3A:0.5
    Blue absorbance\x3A:0.5
    Image name\x3A:RedWine
    Stain:Custom
    Red absorbance\x3A:0.1
    Green absorbance\x3A:0.2
    Blue absorbance\x3A:0.3
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, U.UnmixColors))
        self.assertEqual(module.input_image_name, "Color")
        self.assertEqual(module.stain_count.value, 13)
        self.assertEqual(module.outputs[0].image_name, "Hematoxylin")
        self.assertEqual(module.outputs[-1].image_name, "RedWine")
        for i, stain in enumerate((
                U.CHOICE_HEMATOXYLIN, U.CHOICE_EOSIN, U.CHOICE_DAB,
                U.CHOICE_FAST_RED, U.CHOICE_FAST_BLUE, U.CHOICE_METHYL_GREEN,
                U.CHOICE_AEC, U.CHOICE_ANILINE_BLUE, U.CHOICE_AZOCARMINE,
                U.CHOICE_ALICAN_BLUE, U.CHOICE_PAS)):
            self.assertEqual(module.outputs[i].stain_choice, stain)
        self.assertAlmostEqual(module.outputs[-1].red_absorbance.value, .1)
        self.assertAlmostEqual(module.outputs[-1].green_absorbance.value, .2)
        self.assertAlmostEqual(module.outputs[-1].blue_absorbance.value, .3)

    def make_workspace(self, pixels, choices):
        '''Make a workspace for running UnmixColors

        pixels - input image
        choices - a list of choice strings for the images desired
        '''
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)

        module = U.UnmixColors()
        module.input_image_name.value = INPUT_IMAGE
        module.outputs[0].image_name.value = output_image_name(0)
        module.outputs[0].stain_choice.value = choices[0]
        for i, choice in enumerate(choices[1:]):
            module.add_image()
            module.outputs[i + 1].image_name.value = output_image_name(i + 1)
            module.outputs[i + 1].stain_choice.value = choice

        module.module_num = 1
        pipeline.add_module(module)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image = cpi.Image(pixels)
        image_set.add(INPUT_IMAGE, image)

        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        return workspace, module

    @staticmethod
    def make_image(expected, absorbances):
        eps = 1.0 / 256.0 / 2.0
        absorbance = 1 - expected
        log_absorbance = np.log(absorbance + eps)
        absorbances = np.array(absorbances)
        absorbances = absorbances / np.sqrt(np.sum(absorbances ** 2))
        log_absorbance = log_absorbance[:, :, np.newaxis] * absorbances[np.newaxis, np.newaxis, :]
        image = np.exp(log_absorbance) - eps
        return image

    def test_02_01_zeros(self):
        '''Test on an image of all zeros'''
        workspace, module = self.make_workspace(np.zeros((10, 20, 3)),
                                                [U.CHOICE_HEMATOXYLIN])
        module.run(workspace)
        image = workspace.image_set.get_image(output_image_name(0))
        #
        # All zeros in brightfield should be all 1 in stain
        #
        np.testing.assert_almost_equal(image.pixel_data, 1, 2)

    def test_02_02_ones(self):
        '''Test on an image of all ones'''
        workspace, module = self.make_workspace(np.ones((10, 20, 3)),
                                                [U.CHOICE_HEMATOXYLIN])
        module.run(workspace)
        image = workspace.image_set.get_image(output_image_name(0))
        #
        # All ones in brightfield should be no stain
        #
        np.testing.assert_almost_equal(image.pixel_data, 0, 2)

    def test_02_03_one_stain(self):
        '''Test on a single stain'''

        np.random.seed(23)
        expected = np.random.uniform(size=(10, 20))
        image = self.make_image(expected, U.ST_HEMATOXYLIN)
        workspace, module = self.make_workspace(image, [U.CHOICE_HEMATOXYLIN])
        module.run(workspace)
        image = workspace.image_set.get_image(output_image_name(0))
        np.testing.assert_almost_equal(image.pixel_data, expected, 2)

    def test_02_04_two_stains(self):
        '''Test on two stains mixed together'''
        np.random.seed(24)
        expected_1 = np.random.uniform(size=(10, 20)) * .5
        expected_2 = np.random.uniform(size=(10, 20)) * .5
        #
        # The absorbances should add in log space and multiply in
        # the image space
        #
        image = self.make_image(expected_1, U.ST_HEMATOXYLIN)
        image *= self.make_image(expected_2, U.ST_EOSIN)
        workspace, module = self.make_workspace(image, [
            U.CHOICE_HEMATOXYLIN, U.CHOICE_EOSIN])
        module.run(workspace)
        image_1 = workspace.image_set.get_image(output_image_name(0))
        np.testing.assert_almost_equal(image_1.pixel_data, expected_1, 2)
        image_2 = workspace.image_set.get_image(output_image_name(1))
        np.testing.assert_almost_equal(image_2.pixel_data, expected_2, 2)

    def test_02_05_custom_stain(self):
        '''Test on a custom value for the stains'''
        np.random.seed(25)
        absorbance = np.random.uniform(size=3)
        expected = np.random.uniform(size=(10, 20))
        image = self.make_image(expected, absorbance)
        workspace, module = self.make_workspace(image, [U.CHOICE_CUSTOM])
        (module.outputs[0].red_absorbance.value,
         module.outputs[0].green_absorbance.value,
         module.outputs[0].blue_absorbance.value) = absorbance
        module.run(workspace)
        image = workspace.image_set.get_image(output_image_name(0))
        np.testing.assert_almost_equal(image.pixel_data, expected, 2)
