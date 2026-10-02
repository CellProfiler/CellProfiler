import io
import sys
import zipfile

import cellprofiler_core.constants.modules
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
        try:
            plugins.load_plugin("goodplugin")
            plugins.load_plugin("badplugin")

            good_status = plugins.PLUGIN_STATUS["goodplugin"]
            assert good_status == {"loaded": True, "kind": "module", "error": None}

            bad_status = plugins.PLUGIN_STATUS["badplugin"]
            assert bad_status["loaded"] is False
            assert bad_status["kind"] is None
            assert "this_module_does_not_exist_anywhere" in bad_status["error"]

            statuses = {s["name"]: s for s in plugins.get_plugin_statuses(str(tmp_path))}
            assert statuses["goodplugin"]["loaded"] is True
            assert statuses["badplugin"]["loaded"] is False

            # A second call should skip the already-loaded plugin rather than
            # re-registering it (which would log a "multiple definitions" warning).
            plugins.load_plugin("goodplugin")
            assert cellprofiler_core.constants.modules.all_modules["SomeTestModule"].__name__ == "SomeTestModule"
        finally:
            for name in ("goodplugin", "badplugin"):
                plugins.PLUGIN_STATUS.pop(name, None)
                sys.modules.pop(name, None)
            cellprofiler_core.constants.modules.all_modules.pop("SomeTestModule", None)
