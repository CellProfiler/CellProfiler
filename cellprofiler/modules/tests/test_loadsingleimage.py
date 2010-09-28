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
import hashlib
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.loadsingleimage as L
from cellprofiler.modules.tests import example_images_directory

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

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9524

LoadSingleImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):foo.tif
    Name the image that will be loaded:DNA
    Filename of the image to load (Include the extension, e.g., .tif):bar.tif
    Name the image that will be loaded:Cytoplasm
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        self.assertTrue(fs.rescale)

    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9524

LoadSingleImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):foo.tif
    Name the image that will be loaded:DNA
    Rescale image?:No
    Filename of the image to load (Include the extension, e.g., .tif):bar.tif
    Name the image that will be loaded:Cytoplasm
    Rescale image?:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_name, "DNA")
        self.assertFalse(fs.rescale)
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        self.assertTrue(fs.rescale)

    def get_image_name(self, idx):
        return "MyImage%d" % idx
    
    def make_workspace(self, file_names):
        module = L.LoadSingleImage()
        module.module_num = 1
        for i, file_name in enumerate(file_names):
            if i > 0:
                module.add_file()
            module.file_settings[i].image_name.value = self.get_image_name(i)
            module.file_settings[i].file_name.value = file_name
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(pipeline, module, 
                                  image_set_list.get_image_set(0),
                                  cpo.ObjectSet(), cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module
    
    def test_02_01_load_one(self):
        path = os.path.join(example_images_directory(), "ExampleSpecklesImages")
        cpprefs.set_default_image_directory(path)
        file_name = "1-162hrh2ax2.tif"
        workspace, module = self.make_workspace([file_name])
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        f = m.get_all_measurements(cpmeas.IMAGE, 
                                   "_".join((L.C_FILE_NAME, self.get_image_name(0))))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], file_name)
        p = m.get_all_measurements(cpmeas.IMAGE, 
                                   "_".join((L.C_PATH_NAME, self.get_image_name(0))))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0], path)
        s = m.get_all_measurements(cpmeas.IMAGE,
                                   "_".join((L.C_SCALING, self.get_image_name(0))))
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0], 4095)
        md = m.get_all_measurements(cpmeas.IMAGE,
                                   "_".join((L.C_MD5_DIGEST, self.get_image_name(0))))
        self.assertEqual(len(md), 1)
        md5 = hashlib.md5()
        image = workspace.image_set.get_image(self.get_image_name(0))
        md5.update(np.ascontiguousarray(image.pixel_data).data)
        self.assertEqual(md5.hexdigest(), md[0])
        
    def test_02_02_scale(self):
        '''Load an image twice, as scaled and unscaled'''
        path = os.path.join(example_images_directory(), "ExampleSpecklesImages")
        cpprefs.set_default_image_directory(path)
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        module.file_settings[0].rescale.value = False
        module.file_settings[1].rescale.value = True
        module.run(workspace)
        unscaled, scaled = [workspace.image_set.get_image(self.get_image_name(i)).pixel_data
                            for i in range(2)]
        np.testing.assert_almost_equal(unscaled * 65535. / 4095., scaled)
        
    def test_03_01_measurement_columns(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 8)
        self.assertTrue([c[0] == cpmeas.IMAGE for c in columns])
        for image_name in [self.get_image_name(i) for i in range(2)]:
            for feature in (L.C_FILE_NAME, L.C_MD5_DIGEST, L.C_PATH_NAME,
                            L.C_SCALING):
                measurement = "_".join((feature, image_name))
                self.assertTrue(measurement in [c[1] for c in columns])
                
    def test_03_02_categories(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        categories = module.get_categories(workspace.pipeline, "Foo")
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 4)
        for category in (L.C_FILE_NAME, L.C_MD5_DIGEST, L.C_PATH_NAME,
                            L.C_SCALING):
            self.assertTrue(category in categories)
            
    def test_03_03_measurements(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, L.LoadSingleImage))
        measurements = module.get_measurements(workspace.pipeline, "foo", "bar")
        self.assertEqual(len(measurements), 0)
        measurements = module.get_measurements(workspace.pipeline, cpmeas.IMAGE, "bar")
        self.assertEqual(len(measurements), 0)
        measurements = module.get_measurements(workspace.pipeline, "foo", L.C_PATH_NAME)
        self.assertEqual(len(measurements), 0)
        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 4)
        for category in (categories):
            measurements = module.get_measurements(workspace.pipeline,
                                                   cpmeas.IMAGE, category)
            for i in range(2):
                self.assertTrue(self.get_image_name(i) in measurements)
        