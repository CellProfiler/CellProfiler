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

class TestExample2a(unittest.TestCase):
    def make_instance(self):
        return instantiate_module("Example2a")
    
    def test_00_00_can_load(self):
        self.assertFalse(self.make_instance() is None)
        
    def test_01_01_can_get_settings(self):
        module = self.make_instance()
        self.assertTrue(hasattr(module, "input_image_name"),
                        "Your module needs an input_image_name setting")
        self.assertTrue(
            isinstance(module.input_image_name, cps.ImageNameSubscriber),
            "Your module's input_image_name should be an ImageNameSubscriber")
        self.assertTrue(hasattr(module, "output_image_name"),
                        "Your module needs an output_image_name setting")
        self.assertTrue(
            isinstance(module.output_image_name, cps.ImageNameProvider),
            "Your module's output_image_name should be an ImageNameProvider")
        
    def test_01_02_can_load_and_save(self):
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        module = self.make_instance()
        module.module_num = 1
        module.input_image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        pipeline.add_module(module)
        #
        # Save the pipeline to a string-stream
        #
        fd = StringIO()
        pipeline.save(fd)
        fd.seek(0)
        #
        # Clear the pipeline
        #
        pipeline.clear()
        #
        # Load the pipeline
        #
        pipeline.load(fd)
        module = pipeline.modules()[-1]
        self.assertEqual(module.input_image_name.value, INPUT_IMAGE_NAME)
        self.assertEqual(module.output_image_name.value, OUTPUT_IMAGE_NAME)
        
    def test_02_01_run_it(self):
        # Get the same pseudo-random number generator each time
        r = np.random.RandomState()
        r.seed(21)
        #
        # make an image
        #
        pixel_data = r.uniform(size=(100, 200))
        input_image = cpi.Image(pixel_data)
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
        #
        # Run your module
        #
        module.run(workspace)
        #
        # Get the output image. 
        #
        self.assertTrue(OUTPUT_IMAGE_NAME in image_set.names,
                        "Could not find the output image in the image set")
        output_image = image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = output_image.pixel_data
