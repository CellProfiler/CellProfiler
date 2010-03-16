'''test_loadsingleimage - Test the LoadSingleImage module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import base64
import numpy as np
import os
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.loadsingleimage as L

class TestLoadSingleImage(unittest.TestCase):
    def test_01_00_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggpTVXwTSxSMDRTMDSxMjW3MrJQMDIwNFAgGTAwevryMzAwrGZk'
                'YKiY8zbM1v+wgcDeFpapDEuFuIOvchpu3WDY9MJBVaNj0boTqn6pmp2LaxRO'
                '6Sc86S9JT3mSnrjd77VExqRLnF8XBX+ZU/1+nl78OqYDt80OqFmHR7wy4A1O'
                '8PXqq7bo67Tv8TF44LCmfsObxRMWNHb/PuFbwLLZ47r1Puf37gffXDLdKixe'
                'PlFdfPMLtXsM7Rd7JwsdfRr9qeeuXOXBCb1b3vDZT+wIiP/Qum+X1Wvv5KX5'
                'U5+utpzfvOxM4/mjk65V/jU887pX/tk2xavXJT5Fv/Dfc1lm3syHvoWbnwZo'
                '/dE7bJ/DG6DxI93yT2zr+Y1vF7M/WqiYd+yI5orNi18U3Hk3rzzG/GPLmaDi'
                'FKnWZwGNOf+7rsz/rF/84zfX/MfHA32YxV3j0qSOPkvUrJLZnnl4eaFy5xHu'
                'QJd074sPfyh9ZT1aGvY0fe3ma5FLPq+c/ltuuu3zn2s07Sr97t1L9Ji8wLFG'
                'mn31lcjfv+//u/fw+/OZybKbzhfH3bddqn88XOSm7TbHZGu9dwlLrT79LzM+'
                'vv8c32mtb/OvObxbf5Cz5rnoy3SJp5Vs1se+FtZu+t9c9P15hmOt1Olr9YzH'
                'iy8IAQDsQ/za')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(module.directory.dir_choice, cps.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.directory.custom_path, "./foo")
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "dna_image.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "cytoplasm_image.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(module.directory.dir_choice, cps.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(module.directory.custom_path, "./bar")
        self.assertEqual(len(module.file_settings), 1)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "DNAIllum.tif")
        self.assertEqual(fs.image_name, "DNAIllum")
        
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9524

LoadSingleImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Input Folder
    Name of the folder containing the image file:path1
    Filename of the image to load (Include the extension, e.g., .tif):foo.tif
    Name the image that will be loaded:DNA
    Filename of the image to load (Include the extension, e.g., .tif):bar.tif
    Name the image that will be loaded:Cytoplasm

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Output Folder
    Name of the folder containing the image file:path2
    Filename of the image to load (Include the extension, e.g., .tif):baz.tif
    Name the image that will be loaded:GFP

LoadSingleImage:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Custom folder
    Name of the folder containing the image file:path3
    Filename of the image to load (Include the extension, e.g., .tif):baz.tif
    Name the image that will be loaded:GFP

LoadSingleImage:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Custom with metadata
    Name of the folder containing the image file:path4
    Filename of the image to load (Include the extension, e.g., .tif):baz.tif
    Name the image that will be loaded:GFP
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 4)
        
        dir_choice = [ 
            cps.DEFAULT_INPUT_FOLDER_NAME, cps.DEFAULT_OUTPUT_FOLDER_NAME,
            cps.ABSOLUTE_FOLDER_NAME, cps.ABSOLUTE_FOLDER_NAME]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, L.LoadSingleImage))
            self.assertEqual(module.directory.dir_choice, dir_choice[i])
            self.assertEqual(module.directory.custom_path,
                             "path%d" % (i+1))
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        
    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9524

LoadSingleImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):foo.tif
    Name the image that will be loaded:DNA
    Filename of the image to load (Include the extension, e.g., .tif):bar.tif
    Name the image that will be loaded:Cytoplasm

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Output Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):baz.tif
    Name the image that will be loaded:GFP

LoadSingleImage:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Elsewhere...\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):baz.tif
    Name the image that will be loaded:GFP

LoadSingleImage:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:URL\x7Chttps\x3A//svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/ExampleSBSImages
    Filename of the image to load (Include the extension, e.g., .tif):Channel1-01-A-01.tif
    Name the image that will be loaded:DNA1
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 4)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        
        dir_choice = [ 
            cps.DEFAULT_INPUT_FOLDER_NAME, cps.DEFAULT_OUTPUT_FOLDER_NAME,
            cps.ABSOLUTE_FOLDER_NAME, cps.URL_FOLDER_NAME]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, L.LoadSingleImage))
            self.assertEqual(module.directory.dir_choice, dir_choice[i])

