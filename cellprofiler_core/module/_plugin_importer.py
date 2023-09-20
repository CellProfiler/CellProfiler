import os
import sys
import types

from ..preferences import get_plugin_directory


class PluginImporter(object):
    def find_module(self, fullname, path=None):
        if not fullname.startswith("cellprofiler.modules.plugins"):
            return None
        prefix, modname = fullname.rsplit(".", 1)
        if prefix != "cellprofiler.modules.plugins":
            return None
        if os.path.exists(os.path.join(get_plugin_directory(), modname + ".py")):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        prefix, modname = fullname.rsplit(".", 1)
        assert prefix == "cellprofiler_core.modules.plugins"

        try:
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod
            mod.__loader__ = self
            mod.__file__ = os.path.join(get_plugin_directory(), modname + ".py")

            contents = open(mod.__file__, "r").read()
            exec(compile(contents, mod.__file__, "exec"), mod.__dict__)
            return mod
        except:
            if fullname in sys.modules:
                del sys.modules[fullname]
