import numpy as np
import scipy.ndimage
import unittest

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw

from cellprofiler.modules import instantiate_module

MODULE_NAME = "Example6a"
INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"

class TestExample6a(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.module = instantiate_module(MODULE_NAME)
        self.module.input_image_name.value = INPUT_IMAGE_NAME
        self.module.output_image_name.value = OUTPUT_IMAGE_NAME
        self.module.module_num = 1
        self.pipeline = cpp.Pipeline()
        self.pipeline.add_module(self.module)
        self.package = __import__(self.module.__class__.__module__)
        
    def test_01_01_e6_state_functions(self):
        d = {}
        r = np.random.RandomState()
        r.seed(11)
        stack = r.uniform(size=(40, 50, 10))
        image_numbers = range(1, stack.shape[2]+1)
        self.package.e6_state_init(d, image_numbers)
        for image_number in image_numbers:
            self.package.e6_state_append(d, stack[:, :, (image_number-1)], 1)
            
        result = self.package.e6_state_median(d)
        expected = np.median(stack, 2)
        np.testing.assert_almost_equal(result, expected)
        
    def test_01_01_e6_state_functions(self):
        d = {}
        r = np.random.RandomState()
        r.seed(11)
        stack = r.uniform(size=(40, 50, 10))
        image_numbers = range(1, stack.shape[2]+1)
        self.package.e6_state_init(d, image_numbers)
        for image_number in image_numbers:
            self.package.e6_state_append(d, stack[:, :, (image_number-1)], 
                                         image_number, 1)
            
        result = self.package.e6_state_median(d)
        expected = np.median(stack, 2)
        np.testing.assert_almost_equal(result, expected)
        
    def test_01_02_e6_state_shrink(self):
        d = {}
        r = np.random.RandomState()
        r.seed(11)
        stack = r.uniform(size=(10, 15, 10))
        x = 11
        half_x = int(x/2)
        i2big, j2big = (np.mgrid[0:(stack.shape[0] * x),
                                 0:(stack.shape[1] * x)] / x).astype(int)
        #
        # Sample the middle pixel
        #
        i2small, j2small = np.mgrid[half_x:(stack.shape[0]*x):x,
                                    half_x:(stack.shape[1]*x):x]
        image_numbers = range(1, stack.shape[2]+1)
        self.package.e6_state_init(d, image_numbers)
        for image_number in image_numbers:
            image = stack[i2big, j2big, (image_number-1)]
            self.package.e6_state_append(d, image, image_number, .5)
            
        result = self.package.e6_state_median(d)[i2small, j2small]
        expected = np.median(stack, 2)
        np.testing.assert_almost_equal(result, expected, decimal = 2)
        
    def test_02_01_prepare_group(self):
        #
        # Just make sure prepare_group can run
        #
        measurements = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(self.pipeline, 
                                  self.module,
                                  None,
                                  None,
                                  measurements,
                                  image_set_list)
        self.module.prepare_group(workspace, (), range(10))

    def test_02_02_run(self):
        #
        # Run through an entire group
        #
        measurements = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        r = np.random.RandomState()
        r.seed(22)
        stack = r.uniform(size=(45, 49, 10))
        image_numbers = range(1, stack.shape[2]+1)
        measurements = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(self.pipeline, 
                                  self.module,
                                  None,
                                  None,
                                  measurements,
                                  image_set_list)
        self.module.scale.value = .5
        d = {}
        self.module.prepare_group(workspace, (), image_numbers)
        self.package.e6_state_init(d, image_numbers)
        for i, image_number in enumerate(image_numbers):
            measurements.next_image_set(image_number)
            if hasattr(measurements, "get_image"):
                image_set = measurements
            else:
                image_set = image_set_list.get_image_set(image_number)
            workspace = cpw.Workspace(self.pipeline, 
                                      self.module,
                                      image_set,
                                      None,
                                      measurements,
                                      image_set_list)
            image_set.add(INPUT_IMAGE_NAME, cpi.Image(stack[:, :, i]))
            self.module.run(workspace)
            self.package.e6_state_append(d, stack[:, :, i], image_number, .5)
        image = image_set.get_image(OUTPUT_IMAGE_NAME)
        expected = self.package.e6_state_median(d)
        np.testing.assert_almost_equal(image.pixel_data, expected)
            