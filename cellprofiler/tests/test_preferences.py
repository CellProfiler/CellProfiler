# coding: latin-1
"""test_preferences.py - test the preferences module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import unittest
import tempfile

import cellprofiler.preferences as cpprefs

class TestPreferences(unittest.TestCase):
    def test_01_01_folder_translations(self):
        for expected, value in (
            (cpprefs.ABSOLUTE_FOLDER_NAME, cpprefs.ABSOLUTE_FOLDER_NAME),
            (cpprefs.DEFAULT_INPUT_FOLDER_NAME, cpprefs.DEFAULT_INPUT_FOLDER_NAME),
            (cpprefs.DEFAULT_OUTPUT_FOLDER_NAME, cpprefs.DEFAULT_OUTPUT_FOLDER_NAME),
            (cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME, cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME),
            (cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME, cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME),
            (cpprefs.ABSOLUTE_FOLDER_NAME, 'Absolute path elsewhere'),
            (cpprefs.DEFAULT_INPUT_FOLDER_NAME, 'Default Input Folder'),
            (cpprefs.DEFAULT_OUTPUT_FOLDER_NAME, 'Default Output Folder'),
            (cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default input directory sub-folder'),
            (cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default output directory sub-folder')):
            self.assertTrue(value in cpprefs.FOLDER_CHOICE_TRANSLATIONS.keys(), "%s not in dictionary" % value)
            self.assertEqual(expected, cpprefs.FOLDER_CHOICE_TRANSLATIONS[value])
            
    def test_01_02_slot_translations(self):
        for expected, value in (
            (cpprefs.ABSOLUTE_FOLDER_NAME, cpprefs.ABSOLUTE_FOLDER_NAME),
            (cpprefs.DEFAULT_INPUT_FOLDER_NAME, cpprefs.DEFAULT_INPUT_FOLDER_NAME),
            (cpprefs.DEFAULT_OUTPUT_FOLDER_NAME, cpprefs.DEFAULT_OUTPUT_FOLDER_NAME),
            (cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME, cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME),
            (cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME, cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME),
            (cpprefs.ABSOLUTE_FOLDER_NAME, 'Absolute path elsewhere'),
            (cpprefs.DEFAULT_INPUT_FOLDER_NAME, 'Default Input Folder'),
            (cpprefs.DEFAULT_OUTPUT_FOLDER_NAME, 'Default Output Folder'),
            (cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default input directory sub-folder'),
            (cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default Input Folder sub-folder'),
            (cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default output directory sub-folder'),
            (cpprefs.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default Output Folder sub-folder'),
            (cpprefs.DEFAULT_INPUT_FOLDER_NAME, 'Default Image Directory')):
            for i in range(3):
                setting_values = ["foo", "bar", "baz"]
                setting_values[i] = value
                self.assertEqual(cpprefs.standardize_default_folder_names(
                    setting_values, i)[i], expected)

    def test_01_03_unicode_directory(self):
        print "Skipping test that does not yet work"
        return
        old = cpprefs.get_default_image_directory()
        unicode_dir = u'P125 à 144 Crible Chimiothèque HBEC'
        unicode_dir = tempfile.mkdtemp(prefix=unicode_dir)
        cpprefs.set_default_image_directory(unicode_dir)
        self.assertEqual(cpprefs.get_config().Read(cpprefs.DEFAULT_IMAGE_DIRECTORY),
                         unicode_dir)
        self.assertEqual(cpprefs.get_default_image_directory(), unicode_dir)
        cpprefs.set_default_image_directory(old)
