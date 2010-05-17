'''test_run_imagej.py - test the run_imagej module'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
__version__="$Revision$"

import numpy as np
from StringIO import StringIO
import sys
import unittest

import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.preferences as cpprefs
import cellprofiler.workspace as cpw

import cellprofiler.modules.run_imagej as R

cpprefs.set_headless()

run_tests = (sys.platform.startswith("win"))

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"

class TestRunImageJ(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9954

RunImageJ:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Command or macro?:Command
    Command\x3A:Sharpen
    Macro\x3A:run("Invert");
    Options\x3A:nothing
    Set the current image?:Yes
    Current image\x3A:DNA
    Get the current image?:Yes
    Final image\x3A:OutputImage

RunImageJ:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Command or macro?:Macro
    Command\x3A:Sharpen
    Macro\x3A:run("Invert");
    Options\x3A:something
    Set the current image?:No
    Current image\x3A:DNA
    Get the current image?:No
    Final image\x3A:OutputImage
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_COMMAND)
        self.assertEqual(module.command, "Sharpen")
        self.assertEqual(module.options, "nothing")
        self.assertEqual(module.macro, 'run("Invert");')
        self.assertTrue(module.wants_to_set_current_image)
        self.assertTrue(module.wants_to_get_current_image)
        self.assertEqual(module.current_input_image_name, "DNA")
        self.assertEqual(module.current_output_image_name, "OutputImage")
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_MACRO)
        self.assertFalse(module.wants_to_set_current_image)
        self.assertFalse(module.wants_to_get_current_image)

    def make_workspace(self, input_image = None, wants_output_image = False):
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        module = R.RunImageJ()
        module.module_num = 1
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        if input_image is None:
            module.wants_to_set_current_image.value = False
        else:
            module.wants_to_set_current_image.value = True
            module.current_input_image_name.value = INPUT_IMAGE_NAME
            image_set.add(INPUT_IMAGE_NAME, cpi.Image(input_image))
        module.wants_to_get_current_image.value = wants_output_image
        if wants_output_image:
            module.current_output_image_name.value = OUTPUT_IMAGE_NAME
        workspace = cpw.Workspace(pipeline, module, image_set, 
                                  cpo.ObjectSet(), cpm.Measurements(),
                                  image_set_list)
        return workspace, module
    
    if run_tests:
        def test_02_01_run_null_command(self):
            if not run_tests:
                return
            workspace, module = self.make_workspace()
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Select None"
            module.run(workspace)
            
        def test_02_02_run_input_command(self):
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Invert"
            module.run(workspace)
        
        def test_02_03_run_io_command(self):
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image, True)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Invert"
            module.run(workspace)
            img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            np.testing.assert_almost_equal(image, 1-img.pixel_data, 4)
            
        def test_02_04_parameterized_command(self):
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image, True)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Divide..."
            module.options.value = "value=2"
            module.run(workspace)
            img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            np.testing.assert_almost_equal(image, img.pixel_data*2, 4)
            
            
        def test_02_05_run_i_then_o(self):
            '''Run two modules, checking to see if the ImageJ image stayed in place'''
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image, False)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Invert"
            module.run(workspace)
            workspace, module = self.make_workspace(wants_output_image=True)
            module.command_or_macro.value = R.CM_COMMAND
            module.command.value = "Divide..."
            module.options.value = "value=2"
            module.run(workspace)
            img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            np.testing.assert_almost_equal((1-image) / 2, img.pixel_data, 4)
            
        def test_03_01_null_macro(self):
            '''Run a macro that does something innocuous'''
            workspace, module = self.make_workspace()
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_MACRO
            module.macro.value = 'print("Hello, world");\n'
            module.run(workspace)
            
        def test_03_02_input_macro(self):
            '''Run a macro that supplies an image input'''
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_MACRO
            module.macro.value = 'run("Invert");\n'
            module.run(workspace)
            
        def test_03_03_run_io_macro(self):
            image = np.zeros((20,10))
            image[10:15,5:8] = 1
            workspace, module = self.make_workspace(image, True)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_MACRO
            module.command.value = 'run("Invert");\n'
            module.run(workspace)
            img = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            np.testing.assert_almost_equal(image, 1-img.pixel_data, 4)
