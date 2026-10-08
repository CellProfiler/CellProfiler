import io
import sys
import zipfile

import cellprofiler_core.constants.modules
import cellprofiler_core.constants.reader
from cellprofiler_core.utilities.core import plugins


class TestPlugins:
    def test_plugin_list(self, tmp_path):
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "_bar.py").write_text("")
        (tmp_path / "baz.txt").write_text("")
        assert plugins.plugin_list(str(tmp_path)) == ["foo"]

    def test_plugin_list_missing_directory(self, tmp_path):
        assert plugins.plugin_list(str(tmp_path / "does_not_exist")) == []
        assert plugins.plugin_list(None) == []

    def test_get_official_plugins_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plugins, "get_default_output_directory", lambda: str(tmp_path))
        expected = str(tmp_path / "CellProfiler-plugins" / "active_plugins")
        assert plugins.get_official_plugins_directory() == expected

    def test_official_plugins_repo_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plugins, "get_default_output_directory", lambda: str(tmp_path))
        assert not plugins.official_plugins_repo_exists()
        official_dir = tmp_path / "CellProfiler-plugins" / "active_plugins"
        official_dir.mkdir(parents=True)
        assert not plugins.official_plugins_repo_exists()
        (official_dir / "foo.py").write_text("")
        assert plugins.official_plugins_repo_exists()

    def _make_zip_bytes(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("CellProfiler-plugins-cp5/README.md", "not a plugin")
            zf.writestr("CellProfiler-plugins-cp5/CP5/active_plugins/", "")
            zf.writestr("CellProfiler-plugins-cp5/CP5/active_plugins/foo.py", "X = 1\n")
        return buffer.getvalue()

    def test_download_official_plugins_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plugins, "get_default_output_directory", lambda: str(tmp_path))

        class FakeResponse:
            def raise_for_status(self_inner):
                pass

            def iter_content(self_inner, chunk_size):
                yield self._make_zip_bytes()

        class FakeRequests:
            get = staticmethod(lambda *a, **k: FakeResponse())

        monkeypatch.setattr(plugins, "requests", FakeRequests)

        result = plugins.download_official_plugins_repo()

        official_dir = tmp_path / "CellProfiler-plugins" / "active_plugins"
        assert result == str(official_dir)
        assert (official_dir / "foo.py").read_text() == "X = 1\n"
        assert not (official_dir / "README.md").exists()

    def test_download_official_plugins_repo_replaces_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plugins, "get_default_output_directory", lambda: str(tmp_path))
        official_dir = tmp_path / "CellProfiler-plugins" / "active_plugins"
        official_dir.mkdir(parents=True)
        (official_dir / "stale.py").write_text("")

        class FakeResponse:
            def raise_for_status(self_inner):
                pass

            def iter_content(self_inner, chunk_size):
                yield self._make_zip_bytes()

        class FakeRequests:
            get = staticmethod(lambda *a, **k: FakeResponse())

        monkeypatch.setattr(plugins, "requests", FakeRequests)

        plugins.download_official_plugins_repo()

        assert not (official_dir / "stale.py").exists()
        assert (official_dir / "foo.py").exists()

    def test_load_plugin_tracks_status(self, tmp_path, monkeypatch):
        good_source = (
            "from cellprofiler_core.module import Module\n\n"
            "class SomeTestModule(Module):\n"
            "    module_name = 'SomeTestModule'\n"
            "    category = 'Other'\n"
            "    variable_revision_number = 1\n\n"
            "    def create_settings(self):\n"
            "        pass\n\n"
            "    def settings(self):\n"
            "        return []\n\n"
            "    def run(self, workspace):\n"
            "        pass\n"
        )
        bad_source = "import this_module_does_not_exist_anywhere\n"
        (tmp_path / "goodplugin.py").write_text(good_source)
        (tmp_path / "badplugin.py").write_text(bad_source)

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(plugins, "_plugin_directories", lambda: [str(tmp_path)])
        try:
            plugins.load_plugin("goodplugin", directory=str(tmp_path))
            plugins.load_plugin("badplugin", directory=str(tmp_path))

            good_status = plugins.PLUGIN_STATUS["goodplugin"]
            assert good_status == {
                "loaded": True,
                "kind": "module",
                "error": None,
                "directory": str(tmp_path),
                "appose_env_spec": None,
            }

            bad_status = plugins.PLUGIN_STATUS["badplugin"]
            assert bad_status["loaded"] is False
            assert bad_status["kind"] is None
            assert "this_module_does_not_exist_anywhere" in bad_status["error"]
            assert bad_status["directory"] == str(tmp_path)

            statuses = {s["name"]: s for s in plugins.get_plugin_statuses()}
            assert statuses["goodplugin"]["loaded"] is True
            assert statuses["badplugin"]["loaded"] is False

            # A second call should skip the already-loaded plugin rather than
            # re-registering it (which would log a "multiple definitions" warning).
            plugins.load_plugin("goodplugin", directory=str(tmp_path))
            assert cellprofiler_core.constants.modules.all_modules["SomeTestModule"].__name__ == "SomeTestModule"
        finally:
            for name in ("goodplugin", "badplugin"):
                plugins.PLUGIN_STATUS.pop(name, None)
                sys.modules.pop(name, None)
            cellprofiler_core.constants.modules.all_modules.pop("SomeTestModule", None)

    def test_get_plugin_statuses_hides_shadowed_official_plugin(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "user"
        official_dir = tmp_path / "official"
        user_dir.mkdir()
        official_dir.mkdir()

        module_source = (
            "from cellprofiler_core.module import Module\n\n"
            "class Shared(Module):\n"
            "    module_name = 'Shared'\n"
            "    category = 'Other'\n"
            "    variable_revision_number = 1\n\n"
            "    def create_settings(self):\n"
            "        pass\n\n"
            "    def settings(self):\n"
            "        return []\n\n"
            "    def run(self, workspace):\n"
            "        pass\n"
        )
        (user_dir / "shared.py").write_text(module_source)
        (official_dir / "shared.py").write_text(module_source)

        monkeypatch.setattr(
            plugins, "_plugin_directories", lambda: [str(user_dir), str(official_dir)]
        )
        try:
            plugins.load_plugins()

            statuses = [s for s in plugins.get_plugin_statuses() if s["name"] == "shared"]
            assert len(statuses) == 1
            assert statuses[0]["loaded"] is True
            assert statuses[0]["directory"] == str(user_dir)
        finally:
            plugins.PLUGIN_STATUS.pop("shared", None)
            sys.modules.pop("shared", None)
            cellprofiler_core.constants.modules.all_modules.pop("Shared", None)

    def test_load_plugin_captures_appose_env_spec(self, tmp_path, monkeypatch):
        source = (
            "from cellprofiler_core.module import Module\n\n"
            "class ApposeTestModule(Module):\n"
            "    module_name = 'ApposeTestModule'\n"
            "    category = 'Other'\n"
            "    variable_revision_number = 1\n"
            "    appose_env_spec = '/some/plugin/pixi.toml'\n\n"
            "    def create_settings(self):\n"
            "        pass\n\n"
            "    def settings(self):\n"
            "        return []\n\n"
            "    def run(self, workspace):\n"
            "        pass\n"
        )
        (tmp_path / "apposeplugin.py").write_text(source)

        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            plugins.load_plugin("apposeplugin", directory=str(tmp_path))

            status = plugins.PLUGIN_STATUS["apposeplugin"]
            assert status["loaded"] is True
            assert status["appose_env_spec"] == "/some/plugin/pixi.toml"
        finally:
            plugins.PLUGIN_STATUS.pop("apposeplugin", None)
            sys.modules.pop("apposeplugin", None)
            cellprofiler_core.constants.modules.all_modules.pop("ApposeTestModule", None)

    def test_load_plugin_captures_appose_env_spec_for_reader(self, tmp_path, monkeypatch):
        source = (
            "from cellprofiler_core.reader import Reader\n\n"
            "class ApposeTestReader(Reader):\n"
            "    reader_name = 'ApposeTestReader'\n"
            "    appose_env_spec = '/some/reader/pixi.toml'\n"
        )
        (tmp_path / "apposereader.py").write_text(source)

        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            plugins.load_plugin("apposereader", directory=str(tmp_path))

            status = plugins.PLUGIN_STATUS["apposereader"]
            assert status["loaded"] is True
            assert status["kind"] == "reader"
            assert status["appose_env_spec"] == "/some/reader/pixi.toml"
        finally:
            plugins.PLUGIN_STATUS.pop("apposereader", None)
            sys.modules.pop("apposereader", None)
            cellprofiler_core.constants.reader.ALL_READERS.pop("ApposeTestReader", None)
            cellprofiler_core.constants.reader.AVAILABLE_READERS.pop("ApposeTestReader", None)
