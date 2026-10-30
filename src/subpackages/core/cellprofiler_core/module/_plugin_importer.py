import importlib.abc
import importlib.util
import os
import sys

from ..preferences import get_plugin_directory


class PluginImporter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix, _, modname = fullname.rpartition(".")
        if prefix != "cellprofiler.modules.plugins":
            return None
        plugin_directory = get_plugin_directory()
        if not plugin_directory:
            return None
        filename = os.path.join(plugin_directory, modname + ".py")
        if not os.path.isfile(filename):
            return None
        return importlib.util.spec_from_file_location(fullname, filename)

    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return self if spec is not None else None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        spec = self.find_spec(fullname)
        if spec is None:
            raise ModuleNotFoundError(fullname)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[fullname]
            raise
        return module
