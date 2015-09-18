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
        old = cpprefs.get_default_image_directory()
        unicode_dir = u'P125 à 144 Crible Chimiothèque HBEC'
        unicode_dir = tempfile.mkdtemp(prefix=unicode_dir)
        cpprefs.set_default_image_directory(unicode_dir)
        self.assertEqual(cpprefs.config_read(cpprefs.DEFAULT_IMAGE_DIRECTORY),
                         unicode_dir)
        self.assertEqual(cpprefs.get_default_image_directory(), unicode_dir)
        cpprefs.set_default_image_directory(old)

    def test_01_04_old_users_directory(self):
        gotcha = "c:\\users\\default"
        cpprefs.set_preferences_from_dict({})  # clear cache
        cpprefs.get_config().Write("test_preferences", gotcha)
        result = cpprefs.config_read("test_preferences")
        self.assertEqual(result, gotcha)
