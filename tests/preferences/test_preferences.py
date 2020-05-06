import tempfile
import unittest

import cellprofiler_core.preferences


class TestPreferences(unittest.TestCase):
    def test_01_01_folder_translations(self):
        for expected, value in (
            (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
                "Absolute path elsewhere",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
                "Default Input Folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                "Default Output Folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                "Default input directory sub-folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                "Default output directory sub-folder",
            ),
        ):
            assert value in list(
                cellprofiler_core.preferences.FOLDER_CHOICE_TRANSLATIONS.keys()
            ), ("%s not in dictionary" % value)
            assert (
                expected
                == cellprofiler_core.preferences.FOLDER_CHOICE_TRANSLATIONS[value]
            )

    def test_01_02_slot_translations(self):
        for expected, value in (
            (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
            ),
            (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
                "Absolute path elsewhere",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
                "Default Input Folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                "Default Output Folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                "Default input directory sub-folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                "Default Input Folder sub-folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                "Default output directory sub-folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                "Default Output Folder sub-folder",
            ),
            (
                cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
                "Default Image Directory",
            ),
        ):
            for i in range(3):
                setting_values = ["foo", "bar", "baz"]
                setting_values[i] = value
                assert (
                    cellprofiler_core.preferences.standardize_default_folder_names(
                        setting_values, i
                    )[i]
                    == expected
                )

    def test_01_03_unicode_directory(self):
        old = cellprofiler_core.preferences.get_default_image_directory()
        unicode_dir = "P125 � 144 Crible Chimioth�que HBEC"
        unicode_dir = tempfile.mkdtemp(prefix=unicode_dir)
        cellprofiler_core.preferences.set_default_image_directory(unicode_dir)
        assert (
            cellprofiler_core.preferences.config_read(
                cellprofiler_core.preferences.DEFAULT_IMAGE_DIRECTORY
            )
            == unicode_dir
        )
        assert (
            cellprofiler_core.preferences.get_default_image_directory() == unicode_dir
        )
        cellprofiler_core.preferences.set_default_image_directory(old)

    def test_01_04_old_users_directory(self):
        gotcha = "c:\\users\\default"
        cellprofiler_core.preferences.set_preferences_from_dict({})  # clear cache
        cellprofiler_core.preferences.get_config().Write("test_preferences", gotcha)
        result = cellprofiler_core.preferences.config_read("test_preferences")
        assert result == gotcha

    def test_01_05_unicode_value(self):
        # If the item is already in unicode, don't re-decode
        gotcha = "c:\\users\\default"
        cellprofiler_core.preferences.set_preferences_from_dict({})  # clear cache
        cellprofiler_core.preferences.get_config().Write("test_preferences", gotcha)
        result = cellprofiler_core.preferences.config_read("test_preferences")
        assert result == gotcha


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

        self.old_headless = cellprofiler_core.preferences.__dict__["__is_headless"]
        self.old_headless_config = cellprofiler_core.preferences.__dict__[
            "__headless_config"
        ]
        cellprofiler_core.preferences.__dict__["__is_headless"] = True
        cellprofiler_core.preferences.__dict__["__headless_config"] = FakeConfig()

    def tearDown(self):
        cellprofiler_core.preferences.__dict__["__is_headless"] = self.old_headless
        cellprofiler_core.preferences.__dict__[
            "__headless_config"
        ] = self.old_headless_config

    def test_01_01_default_directory_none(self):
        print((cellprofiler_core.preferences.get_default_image_directory()))
        assert True
