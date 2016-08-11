import StringIO
import os
import tempfile
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.renameorrenumberfiles
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


IMAGE_NAME = 'myimage'


class TestRenameOrRenumberFiles(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()

    def tearDown(self):
        for file_name in os.listdir(self.path):
            try:
                os.remove(os.path.join(self.path, file_name))
            except:
                pass
        try:
            os.removedirs(self.path)
        except:
            pass

    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9161

RenameOrRenumberFiles:[module_num:1|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Image name\x3A:orig
    Number of characters to retain at start of file name\x3A:22
    Number of characters to retain at the end of file name\x3A:4
    What do you want to do with the remaining characters?:Delete
    How many numerical digits would you like to use?:4
    Do you want to add text to the file name?:No
    Replacement text:Do not use

RenameOrRenumberFiles:[module_num:2|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Image name\x3A:other
    Number of characters to retain at start of file name\x3A:14
    Number of characters to retain at the end of file name\x3A:3
    What do you want to do with the remaining characters?:Renumber
    How many numerical digits would you like to use?:5
    Do you want to add text to the file name?:Yes
    Replacement text:Text
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 22)
        self.assertEqual(module.number_characters_suffix, 4)
        self.assertEqual(module.action, cellprofiler.modules.renameorrenumberfiles.A_DELETE)
        self.assertFalse(module.wants_text)
        self.assertFalse(module.wants_to_replace_spaces)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "other")
        self.assertEqual(module.number_characters_prefix, 14)
        self.assertEqual(module.number_characters_suffix, 3)
        self.assertEqual(module.action, cellprofiler.modules.renameorrenumberfiles.A_RENUMBER)
        self.assertTrue(module.wants_text)
        self.assertEqual(module.number_digits, 5)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9161

RenameOrRenumberFiles:[module_num:1|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Image name\x3A:orig
    Number of characters to retain at start of file name\x3A:22
    Number of characters to retain at the end of file name\x3A:4
    What do you want to do with the remaining characters?:Delete
    How many numerical digits would you like to use?:4
    Do you want to add text to the file name?:No
    Replacement text:Do not use
    Replace spaces?:Yes
    Space replacement\x3A:+
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 22)
        self.assertEqual(module.number_characters_suffix, 4)
        self.assertEqual(module.action, cellprofiler.modules.renameorrenumberfiles.A_DELETE)
        self.assertFalse(module.wants_text)
        self.assertTrue(module.wants_to_replace_spaces)
        self.assertEqual(module.space_replacement, "+")

    def make_workspace(self, file_name):
        fd = open(os.path.join(self.path, file_name), 'w')
        fd.write("As the poet said, 'Only God can make a tree' -- probably "
                 "because it's so hard to figure out how to get the "
                 "bark on.\n-- Woody Allen\n")
        fd.close()
        module = cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles()
        module.image_name.value = IMAGE_NAME
        module.module_num = 1

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        m = cellprofiler.measurement.Measurements()
        m.add_image_measurement("FileName_%s" % IMAGE_NAME,
                                file_name)
        m.add_image_measurement("PathName_%s" % IMAGE_NAME,
                                self.path)

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     cellprofiler.region.Set(), m, image_set_list)
        return workspace, module

    def test_02_01_rename_delete(self):
        file_name = "myfile.txt"
        expected_name = "my.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        module.action.value = cellprofiler.modules.renameorrenumberfiles.A_DELETE
        module.number_characters_prefix.value = 2
        module.number_characters_suffix.value = 4
        module.wants_text.value = False

        self.assertTrue(os.path.exists(os.path.join(self.path, file_name)))
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))

    def test_02_02_rename_replace(self):
        file_name = "myfile.txt"
        expected_name = "myfiche.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        module.action.value = cellprofiler.modules.renameorrenumberfiles.A_DELETE
        module.number_characters_prefix.value = 2
        module.number_characters_suffix.value = 4
        module.wants_text.value = True
        module.text_to_add.value = "fiche"

        self.assertTrue(os.path.exists(os.path.join(self.path, file_name)))
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))

    def test_02_03_renumber(self):
        file_name = "myfile1.txt"
        expected_name = "myfile01.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        module.action.value = cellprofiler.modules.renameorrenumberfiles.A_RENUMBER
        module.number_characters_prefix.value = 6
        module.number_characters_suffix.value = 4
        module.wants_text.value = False
        module.number_digits.value = 2

        self.assertTrue(os.path.exists(os.path.join(self.path, file_name)))
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))

    def test_02_04_renumber_append(self):
        file_name = "myfile1.txt"
        expected_name = "myfile001_eek.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        module.action.value = cellprofiler.modules.renameorrenumberfiles.A_RENUMBER
        module.number_characters_prefix.value = 6
        module.number_characters_suffix.value = 4
        module.wants_text.value = True
        module.number_digits.value = 3
        module.text_to_add.value = '_eek'

        self.assertTrue(os.path.exists(os.path.join(self.path, file_name)))
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))

    def test_02_05_replace_spaces(self):
        file_name = "my file.txt"
        expected_name = "my+file.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, cellprofiler.modules.renameorrenumberfiles.RenameOrRenumberFiles))
        #
        # Delete nothing here.
        #
        module.action.value = cellprofiler.modules.renameorrenumberfiles.A_DELETE
        module.number_characters_prefix.value = 7
        module.number_characters_suffix.value = 4
        module.wants_to_replace_spaces.value = True
        module.space_replacement.value = "+"
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))
