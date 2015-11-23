import unittest
from cStringIO import StringIO

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
from cellprofiler.modules import instantiate_module

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"

class TestExample2b(unittest.TestCase):
    def make_instance(self):
        return instantiate_module("Example2b")
    
    def test_00_00_can_load(self):
        self.assertFalse(self.make_instance() is None)
        
    def make_workspace(self, pixel_data, mask=None):
        input_image = cpi.Image(pixel_data, mask)
        #
        # In the upcoming version of CellProfiler, Measurements has been
        # duck-typed as an image set and the image set list has gone away,
        # so we test for that and code it so it works with old and new
        #
        measurements = cpmeas.Measurements()
        if hasattr(measurements, "get_image"):
            image_set = measurements
            image_set_list = None
        else:
            image_set_list = cpi.ImageSetList()
            image_set = image_set_list.get_image_set(1)
        image_set.add(INPUT_IMAGE_NAME, input_image)
        #
        # Make the module
        #
        module = self.make_instance()
        module.module_num = 1
        module.input_image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        #
        # Make the pipeline
        #
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        #
        # Make the workspace
        #
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), measurements,
                                  image_set_list)
        return workspace, module
    
    def test_01_01_run_without_mask(self):
        r = np.random.RandomState(11)
        pixel_data = r.uniform(size=(114, 33))
        workspace, module = self.make_workspace(pixel_data)
        
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        
        self.assertEqual(tuple(output_image.pixel_data.shape),
                         tuple(pixel_data.shape))
        
    def test_01_02_run_with_all_unmasked(self):
        r = np.random.RandomState(12)
        pixel_data = r.uniform(size=(54, 73))
        workspace, module = self.make_workspace(pixel_data)
        
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        output_unmasked = output_image.pixel_data
        
        mask = np.ones(pixel_data.shape, bool)
        workspace, module = self.make_workspace(pixel_data, mask)

        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(output_image.has_mask, 
                        "You should either explicitly give the output image a mask "
                        "or you should implicitly inherit the mask by specifying "
                        "the input image as the output_image's parent image")
        np.testing.assert_array_equal(output_image.mask, mask)
        np.testing.assert_almost_equal(output_image.pixel_data, output_unmasked)
        
    def test_01_03_run_with_mask(self):
        r = np.random.RandomState(12)
        pixel_data = r.uniform(size=(54, 73))
        #
        # Mask 1/2 of the pixels
        #
        mask = r.uniform(size=pixel_data.shape) > .5
        workspace, module = self.make_workspace(pixel_data, mask)
        
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        np.testing.assert_array_equal(output_image.mask, mask)
        np.testing.assert_almost_equal(
            pixel_data[~mask], output_image.pixel_data[~mask],
            err_msg = "The mask=False portion of your image should remain unchanged")
        