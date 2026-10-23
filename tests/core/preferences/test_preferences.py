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


class TestUserDataDirectory:
    def test_darwin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "darwin")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / "Library" / "Application Support" / "CellProfiler"
        )

    def test_windows(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / "CellProfiler"
        )

    def test_linux(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / "CellProfiler"
        )

    # the XDG Base Directory Specification requires an unset, empty, or
    # relative XDG_DATA_HOME to be ignored in favour of ~/.local/share
    def test_linux_xdg_data_home_unset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / ".local" / "share" / "CellProfiler"
        )

    def test_linux_xdg_data_home_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / ".local" / "share" / "CellProfiler"
        )

    def test_linux_xdg_data_home_relative(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", "relative/share")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_data_directory() == str(
            tmp_path / ".local" / "share" / "CellProfiler"
        )


class TestUserConfigDirectory:
    # everywhere but linux the config sits in the data directory
    def test_darwin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "darwin")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / "Library" / "Application Support" / "CellProfiler"
        )

    def test_windows(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / "CellProfiler"
        )

    # on linux XDG separates config from data, and we want a directory of our
    # own rather than wx's bare $XDG_CONFIG_HOME/CellProfilerLocal.cfg
    def test_linux(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / "CellProfiler"
        )

    # the XDG Base Directory Specification requires an unset, empty, or
    # relative XDG_CONFIG_HOME to be ignored in favour of ~/.config
    def test_linux_xdg_config_home_unset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / ".config" / "CellProfiler"
        )

    def test_linux_xdg_config_home_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / ".config" / "CellProfiler"
        )

    def test_linux_xdg_config_home_relative(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cellprofiler_core.preferences.sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cellprofiler_core.preferences.get_user_config_directory() == str(
            tmp_path / ".config" / "CellProfiler"
        )


class TestCopyLegacyConfig:
    def _set_legacy_path(self, monkeypatch, path):
        monkeypatch.setattr(
            cellprofiler_core.preferences, "get_legacy_config_path", lambda: str(path)
        )

    def test_copies_legacy_config(self, monkeypatch, tmp_path):
        legacy_path = tmp_path / "legacy" / "CellProfilerLocal.cfg"
        legacy_path.parent.mkdir()
        legacy_path.write_text("[Preferences]\n")
        self._set_legacy_path(monkeypatch, legacy_path)
        config_path = tmp_path / "data" / "CellProfilerLocal.cfg"
        config_path.parent.mkdir()

        assert cellprofiler_core.preferences._copy_legacy_config(str(config_path))

        assert config_path.read_text() == "[Preferences]\n"
        # left in place for CellProfiler 4, which shares that location
        assert legacy_path.read_text() == "[Preferences]\n"

    def test_keeps_existing_config(self, monkeypatch, tmp_path):
        legacy_path = tmp_path / "legacy" / "CellProfilerLocal.cfg"
        legacy_path.parent.mkdir()
        legacy_path.write_text("legacy\n")
        self._set_legacy_path(monkeypatch, legacy_path)
        config_path = tmp_path / "data" / "CellProfilerLocal.cfg"
        config_path.parent.mkdir()
        config_path.write_text("current\n")

        assert not cellprofiler_core.preferences._copy_legacy_config(str(config_path))

        assert config_path.read_text() == "current\n"
        assert legacy_path.read_text() == "legacy\n"

    def test_no_legacy_config(self, monkeypatch, tmp_path):
        self._set_legacy_path(monkeypatch, tmp_path / "legacy" / "CellProfilerLocal.cfg")
        config_path = tmp_path / "data" / "CellProfilerLocal.cfg"
        config_path.parent.mkdir()

        assert not cellprofiler_core.preferences._copy_legacy_config(str(config_path))

        assert not config_path.exists()


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


def test_report_progress_from_another_thread_only_reaches_its_callbacks():
    import threading

    reports = []

    def callback(operation_id, progress, message):
        reports.append((threading.current_thread().name, message))

    cellprofiler_core.preferences.add_progress_callback(callback)
    try:
        errors = []

        def report_from_thread():
            try:
                # no callbacks are registered on this thread
                cellprofiler_core.preferences.report_progress("op", None, "thread")
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=report_from_thread, name="reporter")
        thread.start()
        thread.join()
        cellprofiler_core.preferences.report_progress("op", None, "main")
    finally:
        cellprofiler_core.preferences.remove_progress_callback(callback)

    assert errors == []
    assert reports == [(threading.current_thread().name, "main")]
