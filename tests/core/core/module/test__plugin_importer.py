import importlib
import sys

import pytest

from cellprofiler_core.module import _plugin_importer


@pytest.fixture
def plugin_directory(tmp_path, monkeypatch):
    import cellprofiler.modules.plugins

    monkeypatch.setattr(_plugin_importer, "get_plugin_directory", lambda: str(tmp_path))
    before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - before:
        if name.startswith("cellprofiler.modules.plugins."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("fullname", ["other.plugin", "cellprofiler.modules.plugins", "cellprofiler.modules.plugins.nested.plugin"])
def test_ignores_other_namespaces(plugin_directory, fullname):
    (plugin_directory / "plugin.py").write_text("VALUE = 19\n")
    assert _plugin_importer.PluginImporter().find_spec(fullname) is None


def test_missing_directory(monkeypatch):
    monkeypatch.setattr(_plugin_importer, "get_plugin_directory", lambda: None)
    assert _plugin_importer.PluginImporter().find_spec("cellprofiler.modules.plugins.missing") is None


def test_import_relative_and_reload(plugin_directory):
    (plugin_directory / "migration_dependency.py").write_text("VALUE = 19\n")
    filename = plugin_directory / "migration_plugin.py"
    filename.write_text("from .migration_dependency import VALUE\n")
    name = "cellprofiler.modules.plugins.migration_plugin"
    module = importlib.import_module(name)
    assert module.VALUE == 19
    assert module.__file__ == str(filename)
    assert module.__package__ == "cellprofiler.modules.plugins"
    assert module.__spec__.name == name
    assert importlib.import_module(name) is module
    module.VALUE = 23
    assert importlib.reload(module) is module
    assert module.VALUE == 19


@pytest.mark.parametrize("source, error", [("raise RuntimeError('plugin error')\n", RuntimeError), ("def invalid(:\n", SyntaxError), ("raise SystemExit(2)\n", SystemExit)])
@pytest.mark.parametrize("legacy", [False, True])
def test_failed_import_cleanup(plugin_directory, source, error, legacy):
    (plugin_directory / "migration_failed.py").write_text(source)
    name = "cellprofiler.modules.plugins.migration_failed"
    with pytest.raises(error):
        if legacy:
            _plugin_importer.PluginImporter().load_module(name)
        else:
            importlib.import_module(name)
    assert name not in sys.modules


def test_legacy_entry_points(plugin_directory):
    (plugin_directory / "migration_legacy.py").write_text("VALUE = 19\n")
    name = "cellprofiler.modules.plugins.migration_legacy"
    finder = _plugin_importer.PluginImporter()
    assert finder.find_module(name) is finder
    module = finder.load_module(name)
    assert module.VALUE == 19
    assert finder.load_module(name) is module
    assert finder.find_module("other.plugin") is None
    with pytest.raises(ModuleNotFoundError):
        finder.load_module("cellprofiler.modules.plugins.missing")
