"""test_correctillumination_apply.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__ = "$Revision$"
import numpy as np
import unittest

import cellprofiler.modules.correctillumination_apply as cpmcia
import cellprofiler.modules.injectimage as inj
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm

class TestCorrectIlluminationApply(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, cpp.RunExceptionEvent):
            self.fail(event.error.message)

    def test_01_01_divide(self):
        """Test correction by division"""
        np.random.seed(0)
        image = np.random.uniform(size=(10,10))
        illum = np.random.uniform(size=(10,10))
        expected = image / illum
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage",image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIllumination_Apply()
        module.module_num = 3
        pipeline.add_module(module)
        module.image_name.value = "InputImage"
        module.illum_correct_function_image_name.value = "IllumImage"
        module.corrected_image_name.value = "OutputImage"
        module.divide_or_subtract.value = cpmcia.DOS_DIVIDE
        module.rescale_option = cpmcia.RE_NONE
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))

    def test_01_01_subtract(self):
        """Test correction by subtraction"""
        np.random.seed(0)
        image = np.random.uniform(size=(10,10))
        illum = np.random.uniform(size=(10,10))
        expected = image - illum
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage",image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIllumination_Apply()
        module.module_num = 3
        pipeline.add_module(module)
        module.image_name.value = "InputImage"
        module.illum_correct_function_image_name.value = "IllumImage"
        module.corrected_image_name.value = "OutputImage"
        module.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        module.rescale_option = cpmcia.RE_NONE
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))
    
    def test_02_01_stretch(self):
        """Test rescaling by stretching"""
        np.random.seed(0)
        image = np.random.uniform(low = 0.1, high = 0.9, size=(10,10))
        image[0,0] = .1
        image[9,9] = .9
        illum = np.ones((10,10))
        expected = (image - .1) / .8
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage",image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIllumination_Apply()
        module.module_num = 3
        pipeline.add_module(module)
        module.image_name.value = "InputImage"
        module.illum_correct_function_image_name.value = "IllumImage"
        module.corrected_image_name.value = "OutputImage"
        module.divide_or_subtract.value = cpmcia.DOS_DIVIDE
        module.rescale_option = cpmcia.RE_STRETCH
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))
        self.assertAlmostEqual(output_image.pixel_data.max(), 1)
        self.assertAlmostEqual(output_image.pixel_data.min(),0)
        
    def test_02_01_match(self):
        """Test rescaling by matching maxima"""
        np.random.seed(0)
        image = np.random.uniform(low = 0.1, high = 0.9, size=(10,10))
        image[9,9] = .9
        illum = np.random.uniform(size=(10,10))
        expected = image / illum
        expected = .9 * expected / expected.max()
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage",image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIllumination_Apply()
        module.module_num = 3
        pipeline.add_module(module)
        module.image_name.value = "InputImage"
        module.illum_correct_function_image_name.value = "IllumImage"
        module.corrected_image_name.value = "OutputImage"
        module.divide_or_subtract.value = cpmcia.DOS_DIVIDE
        module.rescale_option = cpmcia.RE_MATCH
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))
        self.assertAlmostEqual(output_image.pixel_data.max(), .9)
        

    