'''test_renameorrenumberfiles.py - test the RenameOrRenumberFiles module
'''

import base64
import os
import tempfile
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.renameorrenumberfiles as R

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

    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggpTVXwSsxTMDJTMDSyMjSzMrZQMDIwNFAgGTAwevryMzAwvGNk'
                'YKiY8zbstN8hB5G9urp2Yiy+Pj4RN5ZKC34QnK53xWNRQsOEHTIJKyZP4XvZ'
                '3KBtf3rp/QPiPzT9FjpYawZ4HZ1RMPnkE7/3luXPvpxTZmXYKeIw6evuNZ0G'
                'Zje/svoX/9HaprNo9TIeIZndj+PXXJ3oGOB8uHTJzRstK32PdVfNvsZ29+Xy'
                '9FubTGOcZPMNNk+6m5lwWGe7usP3z9wvk9d63TP8kSsUdtA2SWjV7orp8+Xn'
                'PlsqNn26J8N9a2NZFr92luz2eYXq/8TSPgY15GX619zKNb2z/dy73ysyNWVT'
                'TO/LzLsjJHfyuDKjaccuz5htq9sPddrN4Wg9FLE8tPDn6YIdK14L11gwRuwK'
                '+Flxh7VldXPth+uNj+8Hui6fljQ7z/H4yc2r8vx8/x0v/fD/sdazPzX7n8bm'
                'z8/Jquv3ruC3iDuuXtX9xeO02ywnK4+59/gCr52fn64q1d93s8PqXeHr0pqr'
                'DsybO01kDoiX1XHK5zPFzcyqXyluL/9XhvVTTUf3+8r1ca+vvV4xI6T51t0f'
                'lc++nv7RfE/VuK41l/VO7JH9Z8X9920x2M/3s6ovytxfZGvXPGbdMz8rjny0'
                'qGN9/WZtnFmEfZW3/YSPl/+Y/9jz22ani/jEyvZNMi9tzrqfUec+xFG+mcmv'
                'KHmdf9vMxLQ534XDfv9fzv1IfdvKH/Ir75/sBQB5QhVW')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 22)
        self.assertEqual(module.number_characters_suffix, 4)
        self.assertEqual(module.action, R.A_DELETE)
        self.assertFalse(module.wants_text)

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 14)
        self.assertEqual(module.number_characters_suffix, 3)
        self.assertEqual(module.action, R.A_RENUMBER)
        self.assertTrue(module.wants_text)
        self.assertEqual(module.number_digits, 5)

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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 22)
        self.assertEqual(module.number_characters_suffix, 4)
        self.assertEqual(module.action, R.A_DELETE)
        self.assertFalse(module.wants_text)
        self.assertFalse(module.wants_to_replace_spaces)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "other")
        self.assertEqual(module.number_characters_prefix, 14)
        self.assertEqual(module.number_characters_suffix, 3)
        self.assertEqual(module.action, R.A_RENUMBER)
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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        self.assertEqual(module.image_name, "orig")
        self.assertEqual(module.number_characters_prefix, 22)
        self.assertEqual(module.number_characters_suffix, 4)
        self.assertEqual(module.action, R.A_DELETE)
        self.assertFalse(module.wants_text)
        self.assertTrue(module.wants_to_replace_spaces)
        self.assertEqual(module.space_replacement, "+")

    def make_workspace(self, file_name):
        fd = open(os.path.join(self.path, file_name), 'w')
        fd.write("As the poet said, 'Only God can make a tree' -- probably "
                 "because it's so hard to figure out how to get the "
                 "bark on.\n-- Woody Allen\n")
        fd.close()
        module = R.RenameOrRenumberFiles()
        module.image_name.value = IMAGE_NAME
        module.module_num = 1

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        m = cpmeas.Measurements()
        m.add_image_measurement("FileName_%s" % IMAGE_NAME,
                                file_name)
        m.add_image_measurement("PathName_%s" % IMAGE_NAME,
                                self.path)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), m, image_set_list)
        return workspace, module

    def test_02_01_rename_delete(self):
        file_name = "myfile.txt"
        expected_name = "my.txt"

        workspace, module = self.make_workspace(file_name)
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        module.action.value = R.A_DELETE
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
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        module.action.value = R.A_DELETE
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
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        module.action.value = R.A_RENUMBER
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
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        module.action.value = R.A_RENUMBER
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
        self.assertTrue(isinstance(module, R.RenameOrRenumberFiles))
        #
        # Delete nothing here.
        #
        module.action.value = R.A_DELETE
        module.number_characters_prefix.value = 7
        module.number_characters_suffix.value = 4
        module.wants_to_replace_spaces.value = True
        module.space_replacement.value = "+"
        module.run(workspace)
        self.assertFalse(os.path.exists(os.path.join(self.path, file_name)))
        self.assertTrue(os.path.exists(os.path.join(self.path, expected_name)))
