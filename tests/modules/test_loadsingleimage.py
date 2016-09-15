'''test_loadsingleimage - Test the LoadSingleImage module
'''

import base64
import hashlib
import os
import tempfile
import unittest
import zlib
from StringIO import StringIO

import PIL.Image
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.loadimages
import cellprofiler.modules.loadsingleimage
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import cellprofiler.workspace
import numpy
import cellprofiler.modules.identify
import tests.modules
import tests.modules.test_loadimages

OBJECTS_NAME = "myobjects"
OUTLINES_NAME = "myoutlines"


class TestLoadSingleImage(unittest.TestCase, tests.modules.test_loadimages.ConvtesterMixin):
    @classmethod
    def setUpClass(cls):
        tests.modules.maybe_download_sbs()
        cls.test_filename = "1-162hrh2ax2.tif"
        cls.test_folder = "loadsingleimage"
        cls.test_shape = (27, 18)
        path = tests.modules.make_12_bit_image(cls.test_folder, cls.test_filename, (27, 18))
        cls.test_path = os.path.dirname(path)
        with open(path, "rb") as fd:
            cls.test_md5 = hashlib.md5(fd.read()).hexdigest()

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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        self.assertEqual(module.directory.dir_choice, cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.directory.custom_path, "./foo")
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "dna_image.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "cytoplasm_image.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        self.assertEqual(module.directory.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME)
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 4)

        dir_choice = [
            cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME,
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, cellprofiler.setting.ABSOLUTE_FOLDER_NAME]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
            self.assertEqual(module.directory.dir_choice, dir_choice[i])
            self.assertEqual(module.directory.custom_path,
                             "path%d" % (i + 1))
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 4)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_name, "DNA")
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        module = pipeline.modules()[3]
        fs = module.file_settings[0]
        self.assertEqual(
                fs.file_name,
                "https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/"
                "ExampleSBSImages/Channel1-01-A-01.tif")

        dir_choice = [
            cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME,
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, cellprofiler.setting.URL_FOLDER_NAME]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_objects_choice, cellprofiler.modules.loadsingleimage.IO_IMAGES)
        self.assertEqual(fs.image_name, "DNA")
        self.assertFalse(fs.rescale)
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_name, "Cytoplasm")
        self.assertTrue(fs.rescale)

    def test_01_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9524

LoadSingleImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Folder containing the image file:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):foo.tif
    Load as images or objects?:Images
    Name the image that will be loaded:DNA
    Name the objects that will be loaded:MyObjects
    Do you want to save outlines?:Yes
    Name the outlines:MyOutlines
    Rescale image?:No
    Filename of the image to load (Include the extension, e.g., .tif):bar.tif
    Load as images or objects?:Objects
    Name the image that will be loaded:Cytoplasm
    Name the objects that will be loaded:Cells
    Do you want to save outlines?:No
    Name the outlines:MyOutlines
    Rescale image?:Yes
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        self.assertEqual(len(module.file_settings), 2)
        fs = module.file_settings[0]
        self.assertEqual(fs.file_name, "foo.tif")
        self.assertEqual(fs.image_objects_choice, cellprofiler.modules.loadsingleimage.IO_IMAGES)
        self.assertEqual(fs.image_name, "DNA")
        self.assertEqual(fs.objects_name, "MyObjects")
        self.assertTrue(fs.wants_outlines)
        self.assertEqual(fs.outlines_name, "MyOutlines")
        self.assertFalse(fs.rescale)
        fs = module.file_settings[1]
        self.assertEqual(fs.file_name, "bar.tif")
        self.assertEqual(fs.image_objects_choice, cellprofiler.modules.loadsingleimage.IO_OBJECTS)
        self.assertEqual(fs.image_name, "Cytoplasm")
        self.assertEqual(fs.objects_name, "Cells")
        self.assertFalse(fs.wants_outlines)
        self.assertEqual(fs.outlines_name, "MyOutlines")
        self.assertTrue(fs.rescale)

    def get_image_name(self, idx):
        return "MyImage%d" % idx

    def make_workspace(self, file_names):
        module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
        module.module_num = 1
        module.directory.set_dir_choice(cellprofiler.modules.loadsingleimage.DEFAULT_INPUT_FOLDER_NAME)
        for i, file_name in enumerate(file_names):
            if i > 0:
                module.add_file()
            module.file_settings[i].image_name.value = self.get_image_name(i)
            module.file_settings[i].file_name.value = file_name
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     cellprofiler.object.ObjectSet(), cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_02_01_load_one(self):
        folder = self.test_folder
        file_name = self.test_filename
        cellprofiler.preferences.set_default_image_directory(self.test_path)
        workspace, module = self.make_workspace([file_name])
        assert isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage)
        module.prepare_run(workspace)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        self.assertEqual(m.image_set_count, 1)
        f = m.get_all_measurements(
                cellprofiler.measurement.IMAGE,
                "_".join((cellprofiler.modules.loadsingleimage.C_FILE_NAME, self.get_image_name(0))))
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], file_name)
        p = m.get_all_measurements(
                cellprofiler.measurement.IMAGE,
                "_".join((cellprofiler.modules.loadsingleimage.C_PATH_NAME, self.get_image_name(0))))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0], self.test_path)
        s = m.get_all_measurements(
                cellprofiler.measurement.IMAGE,
                "_".join((cellprofiler.modules.loadsingleimage.C_SCALING, self.get_image_name(0))))
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0], 4095)
        md = m.get_all_measurements(
                cellprofiler.measurement.IMAGE,
                "_".join((cellprofiler.modules.loadsingleimage.C_MD5_DIGEST, self.get_image_name(0))))
        self.assertEqual(len(md), 1)
        self.assertEqual(self.test_md5, md[0])

    def test_02_02_scale(self):
        '''Load an image twice, as scaled and unscaled'''
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        path = self.test_path
        cellprofiler.preferences.set_default_image_directory(path)
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        module.file_settings[0].rescale.value = False
        module.file_settings[1].rescale.value = True
        module.prepare_run(workspace)
        module.run(workspace)
        unscaled, scaled = [workspace.image_set.get_image(self.get_image_name(i)).pixel_data
                            for i in range(2)]
        numpy.testing.assert_almost_equal(unscaled * 65535. / 4095., scaled)

    def test_02_03_prepare_run(self):
        # regression test for issue #673 and #1161
        #
        # If LoadSingleImage appears first, pathname data does not show
        # up in the measurements.
        #
        tests.modules.maybe_download_sbs()
        folder = "ExampleSBSImages"
        path = os.path.join(tests.modules.example_images_directory(), folder)
        filename = "Channel1-01-A-01.tif"
        pipeline = cellprofiler.pipeline.Pipeline()
        lsi = cellprofiler.modules.loadsingleimage.LoadSingleImage()
        lsi.module_num = 1
        lsi.directory.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        lsi.directory.custom_path = path
        lsi.file_settings[0].image_name.value = self.get_image_name(0)
        lsi.file_settings[0].file_name.value = filename
        pipeline.add_module(lsi)
        li = cellprofiler.modules.loadimages.LoadImages()
        li.module_num = 2
        pipeline.add_module(li)
        li.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        li.location.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        li.location.custom_path = path
        li.images[0].common_text.value = "Channel2-"
        m = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, lsi, m, cellprofiler.object.ObjectSet(), m,
                                                     cellprofiler.image.ImageSetList())
        self.assertTrue(pipeline.prepare_run(workspace))
        self.assertGreater(m.image_set_count, 1)
        pipeline.prepare_group(workspace, {}, m.get_image_numbers())
        #
        # Skip to the second image set
        #
        m.next_image_set(2)
        lsi.run(workspace)
        #
        # Are the measurements populated?
        #
        m_file = "_".join((cellprofiler.measurement.C_FILE_NAME, self.get_image_name(0)))
        self.assertEqual(m[cellprofiler.measurement.IMAGE, m_file, 2], filename)
        #
        # Can we retrieve the image?
        #
        pixel_data = m.get_image(self.get_image_name(0)).pixel_data
        self.assertFalse(numpy.isscalar(pixel_data))

    def test_03_01_measurement_columns(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 12)
        self.assertTrue([c[0] == cellprofiler.measurement.IMAGE for c in columns])
        for image_name in [self.get_image_name(i) for i in range(2)]:
            for feature in (cellprofiler.modules.loadsingleimage.C_FILE_NAME, cellprofiler.modules.loadsingleimage.C_MD5_DIGEST, cellprofiler.modules.loadsingleimage.C_PATH_NAME,
                            cellprofiler.modules.loadsingleimage.C_SCALING, cellprofiler.modules.loadsingleimage.C_HEIGHT, cellprofiler.modules.loadsingleimage.C_WIDTH):
                measurement = "_".join((feature, image_name))
                self.assertTrue(measurement in [c[1] for c in columns])

    def test_03_02_categories(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        categories = module.get_categories(workspace.pipeline, "Foo")
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(workspace.pipeline, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 6)
        for category in (cellprofiler.modules.loadsingleimage.C_FILE_NAME, cellprofiler.modules.loadsingleimage.C_MD5_DIGEST, cellprofiler.modules.loadsingleimage.C_PATH_NAME,
                         cellprofiler.modules.loadsingleimage.C_SCALING, cellprofiler.modules.loadsingleimage.C_HEIGHT, cellprofiler.modules.loadsingleimage.C_WIDTH):
            self.assertTrue(category in categories)

    def test_03_03_measurements(self):
        file_names = ["1-162hrh2ax2.tif", "1-162hrh2ax2.tif"]
        workspace, module = self.make_workspace(file_names)
        self.assertTrue(isinstance(module, cellprofiler.modules.loadsingleimage.LoadSingleImage))
        measurements = module.get_measurements(workspace.pipeline, "foo", "bar")
        self.assertEqual(len(measurements), 0)
        measurements = module.get_measurements(workspace.pipeline, cellprofiler.measurement.IMAGE, "bar")
        self.assertEqual(len(measurements), 0)
        measurements = module.get_measurements(workspace.pipeline, "foo", cellprofiler.modules.loadsingleimage.C_PATH_NAME)
        self.assertEqual(len(measurements), 0)
        categories = module.get_categories(workspace.pipeline, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 6)
        for category in categories:
            measurements = module.get_measurements(workspace.pipeline,
                                                   cellprofiler.measurement.IMAGE, category)
            for i in range(2):
                self.assertTrue(self.get_image_name(i) in measurements)

    def test_03_04_object_measurement_columns(self):
        module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
        module.file_settings[0].image_objects_choice.value = cellprofiler.modules.loadsingleimage.IO_OBJECTS
        module.file_settings[0].objects_name.value = OBJECTS_NAME
        columns = module.get_measurement_columns(None)
        expected_columns = (
            (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME),
            (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME),
            (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_COUNT + "_" + OBJECTS_NAME),
            (OBJECTS_NAME, cellprofiler.modules.loadsingleimage.C_LOCATION + "_" + cellprofiler.modules.loadsingleimage.FTR_CENTER_X),
            (OBJECTS_NAME, cellprofiler.modules.loadsingleimage.C_LOCATION + "_" + cellprofiler.modules.loadsingleimage.FTR_CENTER_Y),
            (OBJECTS_NAME, cellprofiler.modules.loadsingleimage.C_NUMBER + "_" + cellprofiler.modules.loadsingleimage.FTR_OBJECT_NUMBER))
        for expected_column in expected_columns:
            self.assertTrue(any([column[0] == expected_column[0] and
                                 column[1] == expected_column[1]
                                 for column in columns]))

        for column in columns:
            self.assertTrue(any([column[0] == expected_column[0] and
                                 column[1] == expected_column[1]
                                 for expected_column in expected_columns]))

    def test_03_05_object_categories(self):
        module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
        module.file_settings[0].image_objects_choice.value = cellprofiler.modules.loadsingleimage.IO_OBJECTS
        module.file_settings[0].objects_name.value = OBJECTS_NAME
        for object_name, expected_categories in (
                (cellprofiler.measurement.IMAGE, (cellprofiler.modules.loadsingleimage.C_COUNT, cellprofiler.modules.loadsingleimage.C_OBJECTS_FILE_NAME, cellprofiler.modules.loadsingleimage.C_OBJECTS_PATH_NAME)),
                (OBJECTS_NAME, (cellprofiler.modules.loadsingleimage.C_NUMBER, cellprofiler.modules.loadsingleimage.C_LOCATION))):
            categories = module.get_categories(None, object_name)
            self.assertTrue(all([category in expected_categories
                                 for category in categories]))
            self.assertTrue(all([expected_category in categories
                                 for expected_category in expected_categories]))

    def test_03_06_object_measurements(self):
        module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
        module.file_settings[0].image_objects_choice.value = cellprofiler.modules.loadsingleimage.IO_OBJECTS
        module.file_settings[0].objects_name.value = OBJECTS_NAME
        for object_name, category, expected_features in (
                (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_COUNT, (OBJECTS_NAME,)),
                (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_OBJECTS_FILE_NAME, (OBJECTS_NAME,)),
                (cellprofiler.measurement.IMAGE, cellprofiler.modules.loadsingleimage.C_OBJECTS_PATH_NAME, (OBJECTS_NAME,)),
                (OBJECTS_NAME, cellprofiler.modules.loadsingleimage.C_NUMBER, (cellprofiler.modules.loadsingleimage.FTR_OBJECT_NUMBER,)),
                (OBJECTS_NAME, cellprofiler.modules.loadsingleimage.C_LOCATION, (cellprofiler.modules.loadsingleimage.FTR_CENTER_X, cellprofiler.modules.loadsingleimage.FTR_CENTER_Y))):
            features = module.get_measurements(None, object_name, category)
            self.assertTrue(all([feature in expected_features
                                 for feature in features]))
            self.assertTrue(all([expected_feature in features
                                 for expected_feature in expected_features]))

    def test_04_01_load_objects(self):
        r = numpy.random.RandomState()
        r.seed(41)
        labels = numpy.random.randint(0, 10, size=(30, 40))
        filename = "myobjects.png"
        directory = tempfile.mkdtemp()
        cellprofiler.preferences.set_default_image_directory(directory)
        pilimage = PIL.Image.fromarray(labels.astype(numpy.uint8), "L")
        pilimage.save(os.path.join(directory, filename))
        del pilimage
        try:
            module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
            module.module_num = 1
            module.directory.set_dir_choice(cellprofiler.modules.loadsingleimage.DEFAULT_INPUT_FOLDER_NAME)
            fs = module.file_settings[0]
            fs.file_name.value = filename
            fs.image_objects_choice.value = cellprofiler.modules.loadsingleimage.IO_OBJECTS
            fs.objects_name.value = OBJECTS_NAME
            pipeline = cellprofiler.pipeline.Pipeline()

            def callback(caller, event):
                self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

            pipeline.add_listener(callback)
            pipeline.add_module(module)
            m = cellprofiler.measurement.Measurements()
            object_set = cellprofiler.object.ObjectSet()
            image_set_list = cellprofiler.image.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            workspace = cellprofiler.workspace.Workspace(
                    pipeline, module, image_set, object_set, m, image_set_list)
            module.prepare_run(workspace)
            module.run(workspace)

            o = object_set.get_objects(OBJECTS_NAME)
            numpy.testing.assert_equal(labels, o.segmented)
            self.assertEqual(m.get_current_image_measurement(
                    "_".join((cellprofiler.modules.loadsingleimage.C_COUNT, OBJECTS_NAME))), 9)
            self.assertEqual(m.get_current_image_measurement(
                    "_".join((cellprofiler.modules.loadsingleimage.C_OBJECTS_FILE_NAME, OBJECTS_NAME))), filename)
            self.assertEqual(m.get_current_image_measurement(
                    "_".join((cellprofiler.modules.loadsingleimage.C_OBJECTS_PATH_NAME, OBJECTS_NAME))), directory)
            for feature in (cellprofiler.modules.identify.M_LOCATION_CENTER_X, cellprofiler.modules.identify.M_LOCATION_CENTER_Y, cellprofiler.modules.identify.M_NUMBER_OBJECT_NUMBER):
                values = m.get_current_measurement(OBJECTS_NAME, feature)
                self.assertEqual(len(values), 9)
        finally:
            try:
                os.remove(os.path.join(directory, filename))
                os.rmdir(directory)
            except:
                print "Failed to delete directory " + directory

    def test_04_02_object_outlines(self):
        labels = numpy.zeros((30, 40), int)
        labels[10:15, 20:30] = 1
        expected_outlines = labels != 0
        expected_outlines[11:14, 21:29] = False
        filename = "myobjects.png"
        directory = tempfile.mkdtemp()
        cellprofiler.preferences.set_default_image_directory(directory)
        pilimage = PIL.Image.fromarray(labels.astype(numpy.uint8), "L")
        pilimage.save(os.path.join(directory, filename))
        del pilimage
        try:
            module = cellprofiler.modules.loadsingleimage.LoadSingleImage()
            module.module_num = 1
            module.directory.set_dir_choice(cellprofiler.modules.loadsingleimage.DEFAULT_INPUT_FOLDER_NAME)
            fs = module.file_settings[0]
            fs.file_name.value = filename
            fs.image_objects_choice.value = cellprofiler.modules.loadsingleimage.IO_OBJECTS
            fs.objects_name.value = OBJECTS_NAME
            fs.wants_outlines.value = True
            fs.outlines_name.value = OUTLINES_NAME
            pipeline = cellprofiler.pipeline.Pipeline()

            def callback(caller, event):
                self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

            pipeline.add_listener(callback)
            pipeline.add_module(module)
            m = cellprofiler.measurement.Measurements()
            object_set = cellprofiler.object.ObjectSet()
            image_set_list = cellprofiler.image.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            workspace = cellprofiler.workspace.Workspace(
                    pipeline, module, image_set, object_set, m, image_set_list)
            module.prepare_run(workspace)
            module.run(workspace)

            outlines = image_set.get_image(OUTLINES_NAME)
            numpy.testing.assert_equal(outlines.pixel_data, expected_outlines)
        finally:
            try:
                os.remove(os.path.join(directory, filename))
                os.rmdir(directory)
            except:
                print "Failed to delete directory " + directory

    def test_05_01_convert_single_image(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120831182904
ModuleCount:2
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C
    Check image sets for unmatched or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:2
    Text that these images have in common (case-sensitive):Channel1-
    Position of this image in each group:1
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):Channel2-
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:DNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Input image file location:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):Channel1ILLUM.mat
    Load as images or objects?:Images
    Name the image that will be loaded:IllumGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:No
"""
        tests.modules.maybe_download_sbs()
        directory = os.path.join(tests.modules.example_images_directory(),
                                 "ExampleSBSImages")
        self.convtester(pipeline_text, directory)

    def test_05_02_convert_two_images(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120831182904
ModuleCount:2
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C
    Check image sets for unmatched or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:2
    Text that these images have in common (case-sensitive):Channel1-
    Position of this image in each group:1
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):Channel2-
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:DNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Input image file location:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):Channel1ILLUM.mat
    Load as images or objects?:Images
    Name the image that will be loaded:IllumGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:No
    Filename of the image to load (Include the extension, e.g., .tif):Channel2ILLUM.mat
    Load as images or objects?:Images
    Name the image that will be loaded:IllumDNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:No
"""
        tests.modules.maybe_download_sbs()
        directory = os.path.join(tests.modules.example_images_directory(),
                                 "ExampleSBSImages")
        self.convtester(pipeline_text, directory)

    def test_05_03_convert_with_metadata(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120831182904
ModuleCount:2
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C
    Check image sets for unmatched or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:1
    Text that these images have in common (case-sensitive):Channel1-
    Position of this image in each group:1
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Input image file location:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):Channel2-\\\\g<ImageNumber>-\\\\g<Row>-\\\\g<Column>.tif
    Load as images or objects?:Images
    Name the image that will be loaded:rawDNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:Yes
    Filename of the image to load (Include the extension, e.g., .tif):Channel1ILLUM.mat
    Load as images or objects?:Images
    Name the image that will be loaded:IllumGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:No
    Filename of the image to load (Include the extension, e.g., .tif):Channel2ILLUM.mat
    Load as images or objects?:Images
    Name the image that will be loaded:IllumDNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:Yes
"""
        tests.modules.maybe_download_sbs()
        directory = os.path.join(tests.modules.example_images_directory(),
                                 "ExampleSBSImages")
        self.convtester(pipeline_text, directory)

    def test_05_04_convert_objects(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120831182904
ModuleCount:2
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C
    Check image sets for unmatched or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:2
    Text that these images have in common (case-sensitive):Channel1-
    Position of this image in each group:1
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):Channel2-
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:DNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadSingleImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Input image file location:Default Input Folder\x7CNone
    Filename of the image to load (Include the extension, e.g., .tif):Channel1ILLUM.mat
    Load as images or objects?:Objects
    Name the image that will be loaded:IllumGFP
    Name this loaded object:MyObjects
    Retain outlines of loaded objects?:No
    Name the outlines:NucleiOutlines
    Rescale intensities?:No
"""
        tests.modules.maybe_download_sbs()
        directory = os.path.join(tests.modules.example_images_directory(),
                                 "ExampleSBSImages")
        self.convtester(pipeline_text, directory)
