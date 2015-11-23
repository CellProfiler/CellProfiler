import unittest

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

class TestExample3a(unittest.TestCase):
    def make_instance(self):
        return instantiate_module("Example3a")
    
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

    def test_01_01_easy_variance(self):
        r = np.random.RandomState()
        r.seed(11)
        pixel_data = r.uniform(size=(88, 66)).astype(np.float32)
        workspace, module = self.make_workspace(pixel_data)
        module.run(workspace)
        expected = np.var(pixel_data)
        
        m = workspace.measurements
        ftr = module.get_feature_name()
        self.assertTrue(m.has_feature(cpmeas.IMAGE, ftr),
                        "Your module doesn't seem to have added the feature, " +
                        ftr)
        value = m.get_measurement(cpmeas.IMAGE, ftr)
        self.assertAlmostEqual(expected, value)
        
    def test_01_02_get_measurement_columns(self):
        module = self.make_instance()
        image_name = "Hepzibah"
        module.input_image_name.value = image_name
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1,
                         "You should only return a single feature definition")
        self.assertEqual(len(columns[0]), 3,
                         "You should have three elements in your tuple")
        self.assertEqual(columns[0][0], cpmeas.IMAGE)
        self.assertEqual(columns[0][1], "Example3_Variance_Hepzibah")
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_FLOAT)
        
    def test_02_01_mask(self):
        r = np.random.RandomState()
        r.seed(21)
        pixel_data = r.uniform(size=(88, 66)).astype(np.float32)
        mask = r.uniform(size=pixel_data.shape) > .5
        workspace, module = self.make_workspace(pixel_data, mask)
        module.run(workspace)
        expected = np.var(pixel_data[mask])
        
        m = workspace.measurements
        ftr = module.get_feature_name()
        self.assertTrue(m.has_feature(cpmeas.IMAGE, ftr),
                        "Your module doesn't seem to have added the feature, " +
                        ftr)
        value = m.get_measurement(cpmeas.IMAGE, ftr)
        self.assertAlmostEqual(expected, value)
        
    def test_02_02_all_masked(self):
        r = np.random.RandomState()
        r.seed(21)
        pixel_data = r.uniform(size=(88, 66)).astype(np.float32)
        mask = np.zeros(pixel_data.shape, bool)
        workspace, module = self.make_workspace(pixel_data, mask)
        module.run(workspace)
        
        m = workspace.measurements
        ftr = module.get_feature_name()
        self.assertTrue(m.has_feature(cpmeas.IMAGE, ftr),
                        "Your module doesn't seem to have added the feature, " +
                        ftr)
        value = m.get_measurement(cpmeas.IMAGE, ftr)
        self.assertTrue(np.isnan(value),
                        "The variance of a completely masked image should be NaN")
        