'''test_morph - test the morphology module
'''

import StringIO
import base64
import unittest
import zlib

import numpy as np
import pytest
import scipy.ndimage as scind

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
import cellprofiler.modules.morph as morph
import centrosome.cpmorphology as cpmorph
import centrosome.filter as cpfilter


class TestMorph(unittest.TestCase):
    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10016

Morph:[module_num:1|svn_version:\'9935\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:InputImage
    Name the output image:MorphImage
    Select the operation to perform:branchpoints
    Repeat operation:Forever
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:bridge
    Repeat operation:Custom
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:clean
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:convex hull
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:diag
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:distance
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:endpoints
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:fill
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:hbreak
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:majority
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:remove
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:shrink
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:skelpe
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:spur
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:thicken
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:thin
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:vbreak
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        ops = [morph.F_BRANCHPOINTS, morph.F_BRIDGE,
               morph.F_CLEAN, morph.F_CONVEX_HULL,
               morph.F_DIAG, morph.F_DISTANCE,
               morph.F_ENDPOINTS, morph.F_FILL,
               morph.F_HBREAK,
               morph.F_MAJORITY, morph.F_REMOVE,
               morph.F_SHRINK, morph.F_SKELPE, morph.F_SPUR,
               morph.F_THICKEN, morph.F_THIN, morph.F_VBREAK]
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, morph.Morph))
        self.assertEqual(module.image_name, "InputImage")
        self.assertEqual(module.output_image_name, "MorphImage")
        self.assertEqual(len(module.functions), len(ops))

    # https://github.com/CellProfiler/CellProfiler/issues/3349
    def test_load_with_extracted_operations(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160418141927
GitHash:9969f42
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:6|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:\x5B\x5D
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Maximum intensity:255.0
    Select the rule criteria:and (file does contain "")
    Name to assign these images:DNA
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedOutlines
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

Morph:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the output image:MorphBlue
    Select the operation to perform:bothat
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:close
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:dilate
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:erode
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:fill small holes
    Number of times to repeat operation:Once
    Maximum hole area:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:invert
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:open
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:skel
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
    Select the operation to perform:tophat
    Number of times to repeat operation:Once
    Repetition number:2
    Diameter:3.0
    Structuring element:Disk
    X offset:1.0
    Y offset:1.0
    Angle:0.0
    Width:3.0
    Height:3.0
    Custom:5,5,1111111111111111111111111
    Rescale values from 0 to 1?:Yes
"""
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline = cpp.Pipeline()
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))

        module = pipeline.modules()[-1]
        with pytest.raises(cps.ValidationError):
            module.test_valid(pipeline)

    def execute(self, image, function, mask=None, custom_repeats=None, scale=None, module=None):
        '''Run the morph module on an input and return the resulting image'''
        INPUT_IMAGE_NAME = 'input'
        OUTPUT_IMAGE_NAME = 'output'
        if module is None:
            module = morph.Morph()
        module.functions[0].function.value = function
        module.image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        if custom_repeats is None:
            module.functions[0].repeats_choice.value = morph.R_ONCE
        elif custom_repeats == -1:
            module.functions[0].repeats_choice.value = morph.R_FOREVER
        else:
            module.functions[0].repeats_choice.value = morph.R_CUSTOM
            module.functions[0].custom_repeats.value = custom_repeats
        if scale is not None:
            module.functions[0].scale.value = scale
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image, mask=mask))
        module.run(workspace)
        output = image_set.get_image(OUTPUT_IMAGE_NAME)
        return output.pixel_data

    def binary_tteesstt(self, function_name, function, gray_out=False, scale=None, custom_repeats=None):
        np.random.seed(map(ord, function_name))
        input = np.random.uniform(size=(20, 20)) > .7
        output = self.execute(input, function_name, scale=scale, custom_repeats=custom_repeats)
        if scale is None:
            expected = function(input)
        else:
            footprint = cpmorph.strel_disk(float(scale) / 2.0)
            expected = function(input, footprint=footprint)
        if not gray_out:
            expected = expected > 0
            self.assertTrue(np.all(output == expected))
        else:
            self.assertTrue(np.all(np.abs(output - expected) < np.finfo(np.float32).eps))

    def test_02_015_binary_branchpoints(self):
        self.binary_tteesstt('branchpoints', cpmorph.branchpoints)

    def test_02_02_binary_bridge(self):
        self.binary_tteesstt('bridge', cpmorph.bridge)

    def test_02_03_binary_clean(self):
        self.binary_tteesstt('clean', cpmorph.clean)

    def test_02_05_binary_diag(self):
        self.binary_tteesstt('diag', cpmorph.diag)

    def test_02_065_binary_endpoints(self):
        self.binary_tteesstt('endpoints', cpmorph.endpoints)

    def test_02_08_binary_fill(self):
        self.binary_tteesstt('fill', cpmorph.fill)

    def test_02_09_binary_hbreak(self):
        self.binary_tteesstt('hbreak', cpmorph.hbreak)

    def test_02_10_binary_majority(self):
        self.binary_tteesstt('majority', cpmorph.majority)

    def test_02_12_binary_remove(self):
        self.binary_tteesstt('remove', cpmorph.remove)

    def test_02_13_binary_shrink(self):
        self.binary_tteesstt('shrink', lambda x: cpmorph.binary_shrink(x, 1))

    def test_02_15_binary_spur(self):
        self.binary_tteesstt('spur', cpmorph.spur)

    def test_02_16_binary_thicken(self):
        self.binary_tteesstt('thicken', cpmorph.thicken)

    def test_02_17_binary_thin(self):
        self.binary_tteesstt('thin', cpmorph.thin)

    def test_02_19_binary_vbreak(self):
        self.binary_tteesstt('vbreak', cpmorph.vbreak)

    def test_02_20_binary_distance(self):
        def distance(x):
            y = scind.distance_transform_edt(x)
            if np.max(y) == 0:
                return y
            else:
                return y / np.max(y)

        self.binary_tteesstt('distance', distance, True)

    def test_02_21_binary_convex_hull(self):
        #
        # Set the four points of a square to True
        #
        image = np.zeros((20, 15), bool)
        image[2, 3] = True
        image[17, 3] = True
        image[2, 12] = True
        image[17, 12] = True
        expected = np.zeros((20, 15), bool)
        expected[2:18, 3:13] = True
        result = self.execute(image, 'convex hull')
        self.assertTrue(np.all(result == expected))

    def test_02_26_binary_skelpe(self):
        def fn(x):
            d = scind.distance_transform_edt(x)
            pe = cpfilter.poisson_equation(x)
            return cpmorph.skeletonize(x, ordering=pe * d)

        self.binary_tteesstt('skelpe', fn)
