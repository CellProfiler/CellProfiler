'''test_run_imagej.py - test the run_imagej module'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
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

import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw

import cellprofiler.modules.run_imagej as R
import imagej.imagej2 as ij2

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"

from cellprofiler.modules import instantiate_module

try:
   instantiate_module("RunImageJ")
   skip_tests = False
except:
   skip_tests = True


@unittest.skipIf(skip_tests, "RunImageJ did not load (headless?)")
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
        self.assertEqual(module.command_or_macro, R.CM_MACRO)
        self.assertEqual(module.macro, 'run("Sharpen");')
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
        self.assertEqual(module.command_or_macro, R.CM_MACRO)
        self.assertEqual(module.macro, 'run("Sharpen");')
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
        self.assertEqual(module.prepare_group_choice, R.CM_MACRO)
        self.assertEqual(module.prepare_group_macro, 'run("Straighten");')
        self.assertEqual(module.post_group_choice, R.CM_MACRO)
        self.assertEqual(module.post_group_macro, 'run("Twist");')
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
    Command or macro?:Macro
    Command\x3A:Invert
    Macro\x3A:run("Tubeness ", "sigma=1.0000 use");
    Options\x3A:whatever
    Set the current image?:No
    Current image\x3A:Axon
    Get the current image?:No
    Final image\x3A:AxonsTubeness
    Wait for ImageJ?:No
    Run before each group?:Command
    Command\x3A:Straighten
    Macro\x3A:print("Enter macro here")\n
    Options\x3A:
    Run after each group?:Command
    Command\x3A:Twist
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
        self.assertEqual(module.command_or_macro, R.CM_MACRO)
        self.assertEqual(module.command, "Invert")
        self.assertEqual(module.macro, 'run("Tubeness ", "sigma=1.0000 use");')
        self.assertFalse(module.wants_to_set_current_image)
        self.assertFalse(module.wants_to_get_current_image)
        self.assertEqual(module.current_input_image_name, "Axon")
        self.assertEqual(module.current_output_image_name, "AxonsTubeness")
        self.assertFalse(module.pause_before_proceeding)
        self.assertEqual(module.prepare_group_choice, R.CM_MACRO)
        self.assertEqual(module.post_group_choice, R.CM_MACRO)
        self.assertEqual(module.prepare_group_macro, 'run("Straighten");')
        self.assertEqual(module.post_group_macro, 'run("Twist");')
        self.assertFalse(module.wants_post_group_image)
        self.assertEqual(module.post_group_output_image, 'AggregateImage')
        
    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20121213213625
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :{"ShowFiltered"\x3A true}
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:No
    Extraction method count:1
    Extraction method:Automatic
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assignment method:Assign all images
    Load as:Grayscale image
    Image name:DNA
    :\x5B\x5D
    Assign channels by:Order
    Assignments count:1
    Match this rule:or (file does contain "")
    Image name:DNA
    Objects name:Cell
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:No
    grouping metadata count:1
    Image name:DNA
    Metadata category:None

RunImageJ:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Run an ImageJ command or macro?:Command
    Command:Edit\x7CInvert...
    Macro:var svcClass=java.lang.ClassLoader.getSystemClassLoader().loadClass(\'imagej.ui.UIService\');\\u000avar uiService=ImageJ.getService(svcClass);\\u000auiService.createUI();
    Macro language:Beanshell
    Input the currently active image in ImageJ?:No
    Select the input image:GFP
    Retrieve the currently active image from ImageJ?:No
    Name the current output image:GFPOut
    Wait for ImageJ before continuing?:No
    Run a command or macro before each group of images?:Script
    Command:Image\x7cCrop
    Macro:var svcClass=java.lang.ClassLoader.getSystemClassLoader().loadClass(\'imagej.ui.UIService\');\\u000avar uiService=ImageJ.getService(svcClass);\\u000auiService.createUI();
    Run a command or macro after each group of images?:Macro
    Command:None
    Macro:run("Smooth");
    Retrieve the image output by the group operation?:No
    Name the group output image:Projection
    Command settings count:3
    Prepare group command settings count:0
    Post-group command settings count:0
     (Input):DNA
    Apply to all planes:No
     (Output):InvertedDNA
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, R.RunImageJ))
        self.assertEqual(module.command_or_macro, R.CM_COMMAND)
        self.assertEqual(module.command, "Edit|Invert...")
        self.assertEqual(module.macro, u"var svcClass=java.lang.ClassLoader.getSystemClassLoader().loadClass('imagej.ui.UIService');\u000avar uiService=ImageJ.getService(svcClass);\u000auiService.createUI();")
        self.assertEqual(module.macro_language, "Beanshell")
        self.assertFalse(module.wants_to_set_current_image)
        self.assertEqual(module.current_input_image_name, "GFP")
        self.assertFalse(module.wants_to_get_current_image)
        self.assertEqual(module.current_output_image_name, "GFPOut")
        self.assertEqual(module.prepare_group_choice, R.CM_SCRIPT)
        self.assertEqual(module.prepare_group_command, "Image|Crop")
        self.assertEqual(module.prepare_group_macro.value, "var svcClass=java.lang.ClassLoader.getSystemClassLoader().loadClass('imagej.ui.UIService');\nvar uiService=ImageJ.getService(svcClass);\nuiService.createUI();")
        self.assertEqual(module.post_group_choice, R.CM_MACRO)
        self.assertEqual(module.post_group_command, "None")
        self.assertEqual(module.post_group_macro, 'run("Smooth");')
        self.assertFalse(module.wants_post_group_image)
        self.assertEqual(module.post_group_output_image, "Projection")
        self.assertEqual(module.command_settings_count.value, 3)
        self.assertEqual(module.pre_command_settings_count.value, 0)
        self.assertEqual(module.post_command_settings_count.value, 0)
        self.assertEqual(len(module.command_settings), 3)
        self.assertEqual(module.command_settings[0], "DNA")
        self.assertFalse(module.command_settings[1])
        self.assertEqual(module.command_settings[2], "InvertedDNA")
        
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
    
    if sys.platform != 'darwin':
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
            module.command.value = "Edit|Invert..."
            module.on_setting_changed(module.command, workspace.pipeline)
            self.assertEqual(len(module.command_settings), 3)
            subscriber, apply_to_all_planes, provider = module.command_settings
            self.assertIsInstance(subscriber, cps.ImageNameSubscriber)
            self.assertIsInstance(apply_to_all_planes, cps.Binary)
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
            module.command_or_macro.value = R.CM_SCRIPT
            module.macro_language.value = "ECMAScript"
            script_svc = ij2.get_script_service(R.get_context())
            factory = script_svc.getByName(module.macro_language.value)
            output_statement = factory.getOutputStatement("Hello, world!")
            module.macro.value = output_statement
            module.wants_to_set_current_image.value = True
            module.current_input_image_name.value = INPUT_IMAGE_NAME
            module.wants_to_get_current_image.value = True
            module.current_output_image_name.value = OUTPUT_IMAGE_NAME
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            output_pixel_data = output_image.pixel_data
            np.testing.assert_array_equal(output_pixel_data, image)
            
        
        def test_02_04_macro(self):
            image = np.zeros((15, 17))
            image[3:6, 10:13] = 1
            workspace, module = self.make_workspace(image)
            self.assertTrue(isinstance(module, R.RunImageJ))
            module.command_or_macro.value = R.CM_MACRO
            module.macro.value = 'run("Invert");'
            module.wants_to_set_current_image.value = True
            module.current_input_image_name.value = INPUT_IMAGE_NAME
            module.wants_to_get_current_image.value = True
            module.current_output_image_name.value = OUTPUT_IMAGE_NAME
            module.run(workspace)
            output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            output_pixel_data = output_image.pixel_data
            np.testing.assert_array_almost_equal(1 - output_pixel_data, image)
            
