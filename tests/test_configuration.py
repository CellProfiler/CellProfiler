import cellprofiler.configuration
import tempfile
import unittest


class TestPreferences(unittest.TestCase):
    def test_01_01_folder_translations(self):
        for expected, value in (
                (cellprofiler.configuration.ABSOLUTE_FOLDER_NAME, cellprofiler.configuration.ABSOLUTE_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME, cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME, cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME, cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME, cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME),
                (cellprofiler.configuration.ABSOLUTE_FOLDER_NAME, 'Absolute path elsewhere'),
                (cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME, 'Default Input Folder'),
                (cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME, 'Default Output Folder'),
                (cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default input directory sub-folder'),
                (cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default output directory sub-folder')):
            self.assertTrue(value in cellprofiler.configuration.FOLDER_CHOICE_TRANSLATIONS.keys(), "%s not in dictionary" % value)
            self.assertEqual(expected, cellprofiler.configuration.FOLDER_CHOICE_TRANSLATIONS[value])

    def test_01_02_slot_translations(self):
        for expected, value in (
                (cellprofiler.configuration.ABSOLUTE_FOLDER_NAME, cellprofiler.configuration.ABSOLUTE_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME, cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME, cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME, cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME),
                (cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME, cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME),
                (cellprofiler.configuration.ABSOLUTE_FOLDER_NAME, 'Absolute path elsewhere'),
                (cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME, 'Default Input Folder'),
                (cellprofiler.configuration.DEFAULT_OUTPUT_FOLDER_NAME, 'Default Output Folder'),
                (cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default input directory sub-folder'),
                (cellprofiler.configuration.DEFAULT_INPUT_SUBFOLDER_NAME, 'Default Input Folder sub-folder'),
                (cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default output directory sub-folder'),
                (cellprofiler.configuration.DEFAULT_OUTPUT_SUBFOLDER_NAME, 'Default Output Folder sub-folder'),
                (cellprofiler.configuration.DEFAULT_INPUT_FOLDER_NAME, 'Default Image Directory')):
            for i in range(3):
                setting_values = ["foo", "bar", "baz"]
                setting_values[i] = value
                self.assertEqual(cellprofiler.configuration.standardize_default_folder_names(
                        setting_values, i)[i], expected)

    def test_01_03_unicode_directory(self):
        old = cellprofiler.configuration.get_default_image_directory()
        unicode_dir = u'P125 � 144 Crible Chimioth�que HBEC'
        unicode_dir = tempfile.mkdtemp(prefix=unicode_dir)
        cellprofiler.configuration.set_default_image_directory(unicode_dir)
        self.assertEqual(cellprofiler.configuration.config_read(cellprofiler.configuration.DEFAULT_IMAGE_DIRECTORY),
                         unicode_dir)
        self.assertEqual(cellprofiler.configuration.get_default_image_directory(), unicode_dir)
        cellprofiler.configuration.set_default_image_directory(old)

    def test_01_04_old_users_directory(self):
        gotcha = "c:\\users\\default"
        cellprofiler.configuration.set_preferences_from_dict({})  # clear cache
        cellprofiler.configuration.get_config().Write("test_preferences", gotcha)
        result = cellprofiler.configuration.config_read("test_preferences")
        self.assertEqual(result, gotcha)


class TestPreferences_02(unittest.TestCase):
    def setUp(self):
        # force the test to use a special config object
        class FakeConfig:
            def Exists(self, arg):
                return True

            def Read(self, arg):
                return None

            def Write(self, arg, val):
                pass

            def GetEntryType(self, kwd):
                return 1

        self.old_headless = cellprofiler.configuration.__dict__['__is_headless']
        self.old_headless_config = cellprofiler.configuration.__dict__['__headless_config']
        cellprofiler.configuration.__dict__['__is_headless'] = True
        cellprofiler.configuration.__dict__['__headless_config'] = FakeConfig()

    def tearDown(self):
        cellprofiler.configuration.__dict__['__is_headless'] = self.old_headless
        cellprofiler.configuration.__dict__['__headless_config'] = self.old_headless_config

    def test_01_01_default_directory_none(self):
        print cellprofiler.configuration.get_default_image_directory()
        self.assertTrue(True)
