# coding=utf-8

import glob
import logging
import os
import os.path
import re
import sys
import types

import cellprofiler_core.module
import cellprofiler_core.preferences

logger = logging.getLogger(__name__)


def plugin_list():
    plugin_dir = cellprofiler_core.preferences.get_plugin_directory()
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "*.py"))
        return [
            os.path.basename(f)[:-3] for f in file_list if not f.endswith("__init__.py")
        ]
    return []


class PluginImporter(object):
    def find_module(self, fullname, path=None):
        if not fullname.startswith("cellprofiler_core.modules.plugins"):
            return None
        prefix, modname = fullname.rsplit(".", 1)
        if prefix != "cellprofiler_core.modules.plugins":
            return None
        if os.path.exists(
            os.path.join(cellprofiler_core.preferences.get_plugin_directory(), modname + ".py")
        ):
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
            mod.__file__ = os.path.join(
                cellprofiler_core.preferences.get_plugin_directory(), modname + ".py"
            )

            contents = open(mod.__file__, "r").read()
            exec(compile(contents, mod.__file__, "exec"), mod.__dict__)
            return mod
        except:
            if fullname in sys.module:
                del sys.modules[fullname]


sys.meta_path.append(PluginImporter())

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {
    "align": "Align",
    "images": "Images",
    "loadimages": "LoadImages",
    "measurementfixture": "MeasurementFixture"
}

# the builtin CP modules that will be loaded from the cellprofiler.modules directory
builtin_modules = ["align", "images", "loadimages", "measurementfixture"]

all_modules = {}
svn_revisions = {}
pymodules = []
badmodules = []
datatools = []
pure_datatools = {}

do_not_override = ["set_settings", "create_from_handles", "test_valid", "module_class"]
should_override = ["create_settings", "settings", "run"]


def check_module(module, name):
    if hasattr(module, "do_not_check"):
        return
    assert (
        name == module.module_name
    ), "Module %s should have module_name %s (is %s)" % (name, name, module.module_name)
    for method_name in do_not_override:
        assert getattr(module, method_name) == getattr(
            cellprofiler_core.module.Module, method_name
        ), "Module %s should not override method %s" % (name, method_name)
    for method_name in should_override:
        assert getattr(module, method_name) != getattr(
            cellprofiler_core.module.Module, method_name
        ), "Module %s should override method %s" % (name, method_name)


def find_cpmodule(m):
    """Returns the CPModule from within the loaded Python module

    m - an imported module

    returns the CPModule class
    """
    for v, val in list(m.__dict__.items()):
        if isinstance(val, type) and issubclass(val, cellprofiler_core.module.Module):
            return val
    raise ValueError("Could not find cellprofiler_core.module.Module class in %s" % m.__file__)


def fill_modules():
    del pymodules[:]
    del badmodules[:]
    del datatools[:]
    all_modules.clear()
    svn_revisions.clear()

    def add_module(mod, check_svn):
        try:
            m = __import__(mod, globals(), locals(), ["__all__"], 0)
            cp_module = find_cpmodule(m)
            name = cp_module.module_name
        except Exception as e:
            logger.warning("Could not load %s", mod, exc_info=True)
            badmodules.append((mod, e))
            return

        try:
            pymodules.append(m)
            if name in all_modules:
                logger.warning(
                    "Multiple definitions of module %s\n\told in %s\n\tnew in %s",
                    name,
                    sys.modules[all_modules[name].__module__].__file__,
                    m.__file__,
                )
            all_modules[name] = cp_module
            check_module(cp_module, name)
            # attempt to instantiate
            if not hasattr(cp_module, "do_not_check"):
                cp_module()
            if hasattr(cp_module, "run_as_data_tool"):
                datatools.append(name)
            if check_svn and hasattr(m, "__version__"):
                match = re.match("^\$Revision: ([0-9]+) \$$", m.__version__)
                if match is not None:
                    svn_revisions[name] = match.groups()[0]
            if not hasattr(all_modules[name], "settings"):
                # No settings = pure data tool
                pure_datatools[name] = all_modules[name]
                del all_modules[name]
        except Exception as e:
            logger.warning("Failed to load %s", name, exc_info=True)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

    for mod in builtin_modules:
        add_module("cellprofiler_core.modules." + mod, True)

    plugin_directory = cellprofiler_core.preferences.get_plugin_directory()
    if plugin_directory is not None:
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for mod in plugin_list():
                add_module(mod, False)
        finally:
            sys.path = old_path

    datatools.sort()
    if len(badmodules) > 0:
        logger.warning(
            "could not load these modules: %s", ",".join([x[0] for x in badmodules])
        )


def add_module_for_tst(module_class):
    all_modules[module_class.module_name] = module_class


fill_modules()

__all__ = [
    "instantiate_module",
    "get_module_names",
    "reload_modules",
    "add_module_for_tst",
    "builtin_modules",
]

replaced_modules = {
    "LoadImageDirectory": ["LoadImages", "LoadData"],
    "GroupMovieFrames": ["LoadImages"],
    "IdentifyPrimLoG": ["IdentifyPrimaryObjects"],
    "FileNameMetadata": ["LoadImages"],
}
depricated_modules = ["CorrectIllumination_Calculate_kate", "SubtractBackground"]
unimplemented_modules = ["LabelImages", "Restart", "SplitOrSpliceMovie"]


def get_module_class(module_name):
    module_class = module_name.split(".")[-1]
    if module_class not in all_modules:
        if module_class in pure_datatools:
            return pure_datatools[module_class]
        if module_class in unimplemented_modules:
            raise ValueError(
                (
                    "The %s module has not yet been implemented. "
                    "It will be available in a later version "
                    "of CellProfiler."
                )
                % module_class
            )
        if module_class in depricated_modules:
            raise ValueError(
                (
                    "The %s module has been deprecated and will "
                    "not be implemented in CellProfiler 2.0."
                )
                % module_class
            )
        if module_class in replaced_modules:
            raise ValueError(
                (
                    "The %s module no longer exists. You can find "
                    "similar functionality in: %s"
                )
                % (module_class, ", ".join(replaced_modules[module_class]))
            )
        raise ValueError("Could not find the %s module" % module_class)
    return all_modules[module_class]


def instantiate_module(module_name):
    module = get_module_class(module_name)()
    if module_name in svn_revisions:
        module.svn_version = svn_revisions[module_name]
    return module


def get_module_names():
    return list(all_modules.keys())


def get_data_tool_names():
    return datatools


def reload_modules():
    for m in pymodules:
        try:
            del sys.modules[m.__name__]
        except:
            pass
    fill_modules()

