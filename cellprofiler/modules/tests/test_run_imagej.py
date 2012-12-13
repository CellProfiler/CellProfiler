'''test_run_imagej.py - test the run_imagej module'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#

import numpy as np
from StringIO import StringIO
import os
from scipy.ndimage import grey_erosion
import sys
import unittest

import cellprofiler.preferences as cpprefs
cpprefs.set_headless()
cpprefs.set_ij_plugin_directory(os.path.split(__file__)[0])
cpprefs.set_ij_version(cpprefs.IJ_1)

import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw

import cellprofiler.modules.run_imagej as R

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
        
    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10033

RunImageJ:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Command or macro?:Command
    Command\x3A:Sharpen
    Macro\x3A:run("Invert");
    Options\x3A:No Options
    Set the current image?:Yes
    Current image\x3A:MyInputImage
    Get the current image?:Yes
    Final image\x3A:MyOutputImage
    Wait for ImageJ?:No
    Run before each group?:Nothing
    Command\x3A:None
    Macro\x3A:run("Invert");
    Options\x3A:
    Run after each group?:Nothing
    Command\x3A:None
    Macro\x3A:run("Invert");
    Options\x3A:
    Save the selected image?:No
    Image name\x3A:ImageJGroupImage

RunImageJ:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Command or macro?:Macro
    Command\x3A:None
    Macro\x3A:run("Invert");
    Options\x3A:
    Set the current image?:No
    Current image\x3A:None
    Get the current image?:No
    Final image\x3A:ImageJImage
    Wait for ImageJ?:No
    Run before each group?:Command
    Command\x3A:Straighten
    Macro\x3A:run("Invert");
    Options\x3A:15
    Run after each group?:Command
    Command\x3A:Twist
    Macro\x3A:run("Invert");
    Options\x3A:236
    Save the selected image?:No
    Image name\x3A:ImageJGroupImage

RunImageJ:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Command or macro?:Macro
    Command\x3A:None
    Macro\x3A:run("Invert");
    Options\x3A:
    Set the current image?:No
    Current image\x3A:None
    Get the current image?:No
    Final image\x3A:ImageJImage
    Wait for ImageJ?:No
    Run before each group?:Macro
    Command\x3A:None
    Macro\x3A:run("Invert");
    Options\x3A:
    Run after each group?:Macro
    Command\x3A:None
    Macro\x3A:run("Revert");
    Options\x3A:
    Save the selected image?:Yes
    Image name\x3A:FinalImage
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_COMMAND)
        self.assertEqual(module.command, "Sharpen")
        self.assertEqual(module.options, "No Options")
        self.assertTrue(module.wants_to_set_current_image)
        self.assertEqual(module.current_input_image_name, "MyInputImage")
        self.assertEqual(module.current_output_image_name, "MyOutputImage")
        self.assertTrue(module.wants_to_get_current_image)
        self.assertFalse(module.pause_before_proceeding)
        self.assertEqual(module.prepare_group_choice, R.CM_NOTHING)
        self.assertEqual(module.post_group_choice, R.CM_NOTHING)
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_MACRO)
        self.assertEqual(module.macro.value, 'run("Invert");')
        self.assertFalse(module.wants_to_get_current_image)
        self.assertFalse(module.wants_to_set_current_image)
        self.assertEqual(module.prepare_group_choice, R.CM_COMMAND)
        self.assertEqual(module.prepare_group_command, "Straighten")
        self.assertEqual(module.prepare_group_options, "15")
        self.assertEqual(module.post_group_choice, R.CM_COMMAND)
        self.assertEqual(module.post_group_command, "Twist")
        self.assertEqual(module.post_group_options, "236")
        self.assertFalse(module.wants_post_group_image)

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.prepare_group_choice, R.CM_MACRO)
        self.assertEqual(module.prepare_group_macro, 'run("Invert");')
        self.assertEqual(module.post_group_choice, R.CM_MACRO)
        self.assertEqual(module.post_group_macro, 'run("Revert");')
        self.assertTrue(module.wants_post_group_image)
        self.assertEqual(module.post_group_output_image, "FinalImage")
    
    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10521

LoadImages:[module_num:1|svn_version:\'10503\'|variable_revision_number:7|show_window:True|notes:\x5B\'Loads all the images except the image correction images. OrigN = original nuclei image; OrigD = original dendrite image; OrigA = original axon image.\'\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7C.
    Check image sets for missing or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:Plate
    Image count:3
    Text that these images have in common (case-sensitive):d0.TIF
    Position of this image in each group:.*_\x5BB-G\x5D.*d0.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<System>.+)_(?P<Plate>.+)_(?P<WellRow>\x5BA-P\x5D)(?P<WellColumn>\x5B0-9\x5D\x5B0-9\x5D)f(?P<Site>\x5B0-9\x5D\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:(?P<PlateFolderNumber>PANDORA_\x5B0-9\x5D*)
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Name this loaded image:Nuclei
    Channel number:1
    Text that these images have in common (case-sensitive):d1.TIF
    Position of this image in each group:.*_\x5BB-G\x5D.*d1.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Name this loaded image:Dendrite
    Channel number:1
    Text that these images have in common (case-sensitive):d2.TIF
    Position of this image in each group:.*_\x5BB-G\x5D.*d2.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Name this loaded image:Axon
    Channel number:1

RunImageJ:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Command or macro?:Command
    Command\x3A:Invert
    Macro\x3A:run("Tubeness ", "sigma=1.0000 use");
    Options\x3A:whatever
    Set the current image?:No
    Current image\x3A:Axon
    Get the current image?:No
    Final image\x3A:AxonsTubeness
    Wait for ImageJ?:No
    Run before each group?:Command
    Command\x3A:Tubeness2 0
    Macro\x3A:print("Enter macro here")\n
    Options\x3A:
    Run after each group?:Command
    Command\x3A:Tubeness2 0
    Macro\x3A:print("Enter macro here")\n
    Options\x3A:
    Save the selected image?:No
    Image name\x3A:AggregateImage
    Command settings count:0
    Prepare group command settings count:0
    Post-group command settings count:0
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_COMMAND)
        self.assertEqual(module.command, "Invert")
        self.assertEqual(module.macro, 'run("Tubeness ", "sigma=1.0000 use");')
        self.assertEqual(module.options, "whatever")
        self.assertFalse(module.wants_to_set_current_image)
        self.assertFalse(module.wants_to_get_current_image)
        self.assertEqual(module.current_input_image_name, "Axon")
        self.assertEqual(module.current_output_image_name, "AxonsTubeness")
        self.assertFalse(module.pause_before_proceeding)
        self.assertEqual(module.prepare_group_choice, R.CM_COMMAND)
        self.assertEqual(module.post_group_choice, R.CM_COMMAND)
        self.assertEqual(module.prepare_group_command, "Tubeness2 0")
        self.assertEqual(module.post_group_command, "Tubeness2 0")
        self.assertEqual(module.prepare_group_macro, 'print("Enter macro here")\n')
        self.assertEqual(module.post_group_macro, 'print("Enter macro here")\n')
        self.assertFalse(module.wants_post_group_image)
        self.assertEqual(module.post_group_output_image, 'AggregateImage')
        
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
        module.prepare_group(workspace, {}, [1]);
        return workspace, module
    
    def make_workspaces(self, input_images):
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        module = R.RunImageJ()
        module.module_num = 1
        module.wants_to_set_current_image.value = False
        module.current_input_image_name.value = INPUT_IMAGE_NAME
        module.current_output_image_name.value = OUTPUT_IMAGE_NAME
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        workspaces = []
        for i, input_image in enumerate(input_images):
            image_set = image_set_list.get_image_set(i)
            image_set.add(INPUT_IMAGE_NAME, cpi.Image(input_image))
            workspace = cpw.Workspace(pipeline, module, image_set, 
                                      cpo.ObjectSet(), cpm.Measurements(),
                                      image_set_list)
            workspaces.append(workspace)
        module.prepare_group(workspaces[0], {}, 
                             list(range(1,len(input_images)+1)));
        return workspaces, module
    
    def test_02_01_run_null_command(self):
        workspace, module = self.make_workspace()
        self.assertTrue(isinstance(module, R.RunImageJ))
        module.command_or_macro.value = R.CM_COMMAND
        module.command.value = "File|Close All"
        module.wants_to_set_current_image.value = False
        module.wants_to_get_current_image.value = False
        module.run(workspace)
        
    def test_02_02_run_input_command(self):
        image = np.zeros((20,10))
        image[10:15,5:8] = 1
        workspace, module = self.make_workspace(image)
        self.assertTrue(isinstance(module, R.RunImageJ))
        module.wants_to_set_current_image.value = False
        module.wants_to_get_current_image.value = False
        module.command_or_macro.value = R.CM_COMMAND
        module.command.value = "Edit|Invert [IJ2]"
        module.on_setting_changed(module.command, workspace.pipeline)
        self.assertEqual(len(module.command_settings), 2)
        subscriber, provider = module.command_settings
        self.assertIsInstance(subscriber, cps.ImageNameSubscriber)
        self.assertIsInstance(provider, cps.ImageNameProvider)
        subscriber.value = INPUT_IMAGE_NAME
        provider.value = OUTPUT_IMAGE_NAME
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        output_pixel_data = output_image.pixel_data
        np.testing.assert_array_almost_equal(1 - output_pixel_data, image)
        
    def test_02_03_set_and_get_image(self):
        image = np.zeros((15, 17))
        image[3:6, 10:13] = 1
        workspace, module = self.make_workspace(image)
        self.assertTrue(isinstance(module, R.RunImageJ))
        module.command_or_macro.value = R.CM_COMMAND
        module.command.value = "Process|Binary|Make Binary"
        module.wants_to_set_current_image.value = True
        module.current_input_image_name.value = INPUT_IMAGE_NAME
        module.wants_to_get_current_image.value = False
        module.run(workspace)
        
        module.command.value = "Process|Binary|Erode"
        module.wants_to_set_current_image.value = False
        module.wants_to_get_current_image.value = True
        module.current_output_image_name.value = OUTPUT_IMAGE_NAME
        module.run(workspace)
        output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        output_pixel_data = output_image.pixel_data
        expected = grey_erosion(image, footprint = np.ones((3,3), bool))
        np.testing.assert_array_almost_equal(output_pixel_data, expected)
        
