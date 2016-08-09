"""test_variable.py - test cellprofiler.settings
"""

import os
import tempfile
import unittest

import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps


class TestVariable(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.Setting("text", "value")
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "value")
        self.assertTrue(x.key())

    def test_01_01_equality(self):
        x = cps.Setting("text", "value")
        self.assertTrue(x == "value")
        self.assertTrue(x != "text")
        self.assertFalse(x != "value")
        self.assertFalse(x == "text")
        self.assertEqual(x.value, "value")


class TestText(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.Text("text", "value")
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "value")
        self.assertTrue(x.key())


class TestInteger(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.Integer("text", 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, 5)

    def test_01_01_numeric_value(self):
        x = cps.Integer("text", 5)
        self.assertTrue(isinstance(x.value, int))

    def test_01_02_equality(self):
        x = cps.Integer("text", 5)
        self.assertTrue(x == 5)
        self.assertTrue(x.value == 5)
        self.assertFalse(x == 6)
        self.assertTrue(x != 6)

    def test_01_03_assign_str(self):
        x = cps.Integer("text", 5)
        x.value = 6
        self.assertTrue(x == 6)

    def test_02_01_neg_assign_number(self):
        x = cps.Integer("text", 5)
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)


class TestBinary(unittest.TestCase):
    def test_00_01_init_true(self):
        x = cps.Binary("text", True)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertTrue(x.value == True)
        self.assertTrue(x == True)
        self.assertTrue(x != False)
        self.assertFalse(x != True)
        self.assertFalse(x == False)

    def test_00_02_init_false(self):
        x = cps.Binary("text", False)
        self.assertTrue(x.value == False)
        self.assertFalse(x == True)
        self.assertFalse(x != False)
        self.assertTrue(x != True)

    def test_01_01_set_true(self):
        x = cps.Binary("text", False)
        x.value = True
        self.assertTrue(x.value == True)

    def test_01_02_set_false(self):
        x = cps.Binary("text", True)
        x.value = False
        self.assertTrue(x.value == False)


class TestChoice(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.Choice("text", ["choice"])
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "choice")
        self.assertEqual(len(x.choices), 1)
        self.assertEqual(x.choices[0], "choice")

    def test_01_01_assign(self):
        x = cps.Choice("text", ["foo", "bar"], "bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")

    def test_02_01_neg_assign(self):
        x = cps.Choice("text", ["choice"])
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)


class TestCustomChoice(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.CustomChoice("text", ["choice"])
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "choice")
        self.assertEqual(len(x.choices), 1)
        self.assertEqual(x.choices[0], "choice")

    def test_01_01_assign(self):
        x = cps.CustomChoice("text", ["foo", "bar"], "bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")

    def test_01_02_assign_other(self):
        x = cps.CustomChoice("text", ["foo", "bar"], "bar")
        x.value = "other"
        self.assertTrue(x == "other")
        self.assertEqual(len(x.choices), 3)
        self.assertEqual(x.choices[0], "other")


class TestIntegerRange(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(str(x), "1,2")
        self.assertEqual(x.min, 1)
        self.assertEqual(x.max, 2)
        x.test_valid(None)

    def test_01_01_assign_tuple(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.value = (2, 5)
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_01_02_assign_string(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.value = "2,5"
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.value = (0, 2)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_02_neg_max(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.value = (1, 6)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_03_neg_order(self):
        x = cps.IntegerRange("text", (1, 2), 1, 5)
        x.value = (2, 1)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x = cps.IntegerRange("text", (1, 2))
        x.test_valid(None)


class TestFloatRange(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, (1, 2))
        self.assertEqual(x.min, 1)
        self.assertEqual(x.max, 2)
        x.test_valid(None)

    def test_01_01_assign_tuple(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.value = (2, 5)
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_01_02_assign_string(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.value = "2,5"
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.value = (0, 2)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_02_neg_max(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.value = (1, 6)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_03_neg_order(self):
        x = cps.FloatRange("text", (1, 2), 1, 5)
        x.value = (2, 1)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x = cps.FloatRange("text", (1, 2))
        x.test_valid(None)


class TestDirectoryPath(unittest.TestCase):
    def setUp(self):
        #
        # Make three temporary directory structures
        #
        cpprefs.set_headless()
        self.directories = [tempfile.mkdtemp() for i in range(3)]
        for directory in self.directories:
            for i in range(3):
                os.mkdir(os.path.join(directory, str(i)))
                for j in range(3):
                    os.mkdir(os.path.join(directory, str(i), str(j)))
                    for k in range(3):
                        os.mkdir(os.path.join(directory, str(i), str(j), str(k)))
        cpprefs.set_default_image_directory(os.path.join(self.directories[0], "1"))
        cpprefs.set_default_output_directory(os.path.join(self.directories[1], "1"))
        self.root_directory = os.path.join(self.directories[2], "1")

    def tearDown(self):
        for directory in self.directories:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        os.rmdir(os.path.join(directory, str(i), str(j), str(k)))
                    os.rmdir(os.path.join(directory, str(i), str(j)))
                os.rmdir(os.path.join(directory, str(i)))
            os.rmdir(directory)

    def test_01_01_static_split_and_join(self):
        gibberish = "aqwura[oijs|fd"
        for dir_choice in cps.DirectoryPath.DIR_ALL + [cps.NO_FOLDER_NAME]:
            value = cps.DirectoryPath.static_join_string(dir_choice, gibberish)
            out_dir_choice, custom_path = cps.DirectoryPath.split_string(value)
            self.assertEqual(dir_choice, out_dir_choice)
            self.assertEqual(custom_path, gibberish)

    def test_01_02_split_and_join(self):
        gibberish = "aqwura[oijs|fd"
        for dir_choice in cps.DirectoryPath.DIR_ALL + [cps.NO_FOLDER_NAME]:
            s = cps.DirectoryPath("whatever")
            value = s.join_parts(dir_choice, gibberish)
            out_dir_choice = s.dir_choice
            custom_path = s.custom_path
            self.assertEqual(dir_choice, out_dir_choice)
            self.assertEqual(custom_path, gibberish)

    def test_01_03_is_custom_choice(self):
        for dir_choice, expected in (
                (cps.DEFAULT_INPUT_FOLDER_NAME, False),
                (cps.DEFAULT_INPUT_SUBFOLDER_NAME, True),
                (cps.DEFAULT_OUTPUT_FOLDER_NAME, False),
                (cps.DEFAULT_OUTPUT_SUBFOLDER_NAME, True),
                (cps.ABSOLUTE_FOLDER_NAME, True),
                (cps.URL_FOLDER_NAME, True)):
            s = cps.DirectoryPath("whatever")
            s.dir_choice = dir_choice
            self.assertEqual(s.is_custom_choice, expected)

    def test_01_04_get_parts_from_image_folder_path(self):
        s = cps.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                cpprefs.get_default_image_directory())
        self.assertEqual(dir_choice, cps.DEFAULT_INPUT_FOLDER_NAME)

    def test_01_05_get_parts_from_output_folder_path(self):
        s = cps.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(cpprefs.get_default_output_directory())
        self.assertEqual(dir_choice, cps.DEFAULT_OUTPUT_FOLDER_NAME)

    def test_01_06_get_parts_from_image_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                os.path.join(cpprefs.get_default_image_directory(), "1"))
        self.assertEqual(dir_choice, cps.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(custom_path, "1")

    def test_01_07_get_parts_from_output_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                os.path.join(cpprefs.get_default_output_directory(), "2"))
        self.assertEqual(dir_choice, cps.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(custom_path, "2")

    def test_01_08_get_parts_from_abspath(self):
        s = cps.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(self.root_directory)
        self.assertEqual(dir_choice, cps.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(custom_path, self.root_directory)

    def test_02_01_get_default_input_folder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_INPUT_FOLDER_NAME
        self.assertEqual(s.get_absolute_path(), cpprefs.get_default_image_directory())

    def test_02_02_get_default_output_folder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_OUTPUT_FOLDER_NAME
        self.assertEqual(s.get_absolute_path(), cpprefs.get_default_output_directory())

    def test_02_03_get_input_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
        s.custom_path = "2"
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(cpprefs.get_default_image_directory(), "2"))

    def test_02_04_get_output_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
        s.custom_path = "0"
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(cpprefs.get_default_output_directory(), "0"))

    def test_02_05_get_absolute_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        s.custom_path = os.path.join(self.root_directory, "..", "1", "2")
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(self.root_directory, "2"))

    def test_02_06_get_url(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.URL_FOLDER_NAME
        s.custom_path = "http://www.cellprofiler.org"
        self.assertEqual(s.get_absolute_path(),
                         "http://www.cellprofiler.org")

    def test_02_07_no_folder(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.NO_FOLDER_NAME
        s.custom_path = "gibberish"
        self.assertEqual(s.get_absolute_path(), '')

    def test_03_07_metadata(self):
        m = cpmeas.Measurements()
        m.add_image_measurement("Metadata_Path", "2")
        s = cps.DirectoryPath("whatever", allow_metadata=True)
        for dir_choice, expected in (
                (cps.DEFAULT_INPUT_SUBFOLDER_NAME,
                 os.path.join(cpprefs.get_default_image_directory(), "0", "2")),
                (cps.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                 os.path.join(cpprefs.get_default_output_directory(), "0", "2")),
                (cps.ABSOLUTE_FOLDER_NAME,
                 os.path.join(self.root_directory, "2")),
                (cps.URL_FOLDER_NAME, "http://www.cellprofiler.org/2")):
            s.dir_choice = dir_choice
            if dir_choice in (cps.DEFAULT_INPUT_SUBFOLDER_NAME,
                              cps.DEFAULT_OUTPUT_SUBFOLDER_NAME):
                s.custom_path = "0" + os.path.sep.replace('\\', '\\\\') + "\\g<Path>"
            elif dir_choice == cps.ABSOLUTE_FOLDER_NAME:
                s.custom_path = self.root_directory + os.path.sep.replace('\\', '\\\\') + "\\g<Path>"
            else:
                s.custom_path = "http://www.cellprofiler.org/\\g<Path>"
            self.assertEqual(s.get_absolute_path(m), expected)

    @staticmethod
    def fn_alter_path(path, **kwargs):
        '''Add "altered" to the path'''
        return path + "altered"

    def test_04_01_alter_input_folder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_INPUT_FOLDER_NAME
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                cpprefs.get_default_image_directory())

    def test_04_02_alter_output_folder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_OUTPUT_FOLDER_NAME
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                cpprefs.get_default_output_directory())

    def test_04_03_alter_input_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
        s.custom_path = "2"

        def fn_alter_path(path, **kwargs):
            self.assertEqual(path, "2")
            return "3"

        s.alter_for_create_batch_files(fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                os.path.join(cpprefs.get_default_image_directory(), "3"))

    def test_04_04_alter_output_subfolder_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
        s.custom_path = "0"

        def fn_alter_path(path, **kwargs):
            self.assertEqual(path, "0")
            return "5"

        s.alter_for_create_batch_files(fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                os.path.join(cpprefs.get_default_output_directory(), "5"))

    def test_04_05_alter_absolute_path(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        s.custom_path = os.path.join(self.root_directory, "..", "1", "2")
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(self.root_directory, "2altered"))

    def test_04_06_alter_url(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.URL_FOLDER_NAME
        s.custom_path = "http://www.cellprofiler.org"
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(s.get_absolute_path(),
                         "http://www.cellprofiler.org")

    def test_04_07_no_folder(self):
        s = cps.DirectoryPath("whatever")
        s.dir_choice = cps.NO_FOLDER_NAME
        s.custom_path = "gibberish"
        self.assertEqual(s.get_absolute_path(), '')
