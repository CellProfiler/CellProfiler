"""test_correctilluminationapply.py
"""

import base64
import os
import sys
import tempfile
import unittest
import zlib
from six.moves import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.modules.correctilluminationapply as cpmcia
import cellprofiler.modules.injectimage as inj
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.object as cpo
import cellprofiler.measurement as cpm


class TestCorrectIlluminationApply:
    def error_callback(self, calller, event):
        if isinstance(event, cpp.RunExceptionEvent):
            self.fail(event.error.message)

    def test_01_01_divide(self):
        """Test correction by division"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image / illum
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.set_module_num(1)
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.set_module_num(2)
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.set_module_num(3)
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_DIVIDE
        image.rescale_option = cpmcia.RE_NONE
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            pipeline, input_module, image_set, object_set, measurements, image_set_list
        )
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        assert np.all(output_image.pixel_data == expected)

    def test_01_02_subtract(self):
        """Test correction by subtraction"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image - illum
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.set_module_num(1)
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.set_module_num(2)
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.set_module_num(3)
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            pipeline, input_module, image_set, object_set, measurements, image_set_list
        )
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        assert np.all(output_image.pixel_data == expected)

    def test_02_01_color_by_bw(self):
        """Correct a color image with a black & white illumination fn"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image - illum[:, :, np.newaxis]
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.set_module_num(1)
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.set_module_num(2)
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.set_module_num(3)
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            pipeline, input_module, image_set, object_set, measurements, image_set_list
        )
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        assert np.all(output_image.pixel_data == expected)

    def test_02_02_color_by_color(self):
        """Correct a color image with a black & white illumination fn"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        expected = image - illum
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.set_module_num(1)
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.set_module_num(2)
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.set_module_num(3)
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            pipeline, input_module, image_set, object_set, measurements, image_set_list
        )
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        assert np.all(output_image.pixel_data == expected)
