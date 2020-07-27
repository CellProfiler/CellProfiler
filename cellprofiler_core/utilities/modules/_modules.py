import glob
import logging
import os
import re
import sys

from ...module import Module
from ...constants.modules import (
    builtin_modules,
    all_modules,
    svn_revisions,
    pymodules,
    badmodules,
    do_not_override,
    should_override,
    replaced_modules,
    depricated_modules,
    unimplemented_modules,
)
from ...preferences import get_plugin_directory


def plugin_list(plugin_dir):
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "*.py"))
        return [
            os.path.basename(f)[:-3]
            for f in file_list
            if not f.endswith(("__init__.py", "_help.py"))
        ]
    return []


def check_module(module, name):
    if hasattr(module, "do_not_check"):
        return
    assert (
        name == module.module_name
    ), "Module %s should have module_name %s (is %s)" % (name, name, module.module_name)
    for method_name in do_not_override:
        assert getattr(module, method_name) == getattr(
            Module, method_name
        ), "Module %s should not override method %s" % (name, method_name)
    for method_name in should_override:
        assert getattr(module, method_name) != getattr(
            Module, method_name
        ), "Module %s should override method %s" % (name, method_name)


def find_cpmodule(m):
    """Returns the CPModule from within the loaded Python module

    m - an imported module

    returns the CPModule class
    """
    for v, val in list(m.__dict__.items()):
        if isinstance(val, type) and issubclass(val, Module):
            return val
    raise ValueError(
        "Could not find cellprofiler_core.module.Module class in %s" % m.__file__
    )


def fill_modules():
    del pymodules[:]
    del badmodules[:]
    all_modules.clear()
    svn_revisions.clear()

    def add_module(mod, check_svn):
        try:
            m = __import__(mod, globals(), locals(), ["__all__"], 0)
            cp_module = find_cpmodule(m)
            name = cp_module.module_name
        except Exception as e:
            logging.warning("Could not load %s", mod, exc_info=True)
            badmodules.append((mod, e))
            return

        try:
            pymodules.append(m)
            if name in all_modules:
                logging.warning(
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
            if check_svn and hasattr(m, "__version__"):
                match = re.match("^\$Revision: ([0-9]+) \$$", m.__version__)
                if match is not None:
                    svn_revisions[name] = match.groups()[0]
        except Exception as e:
            logging.warning("Failed to load %s", name, exc_info=True)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

    # Import core modules
    for mod in builtin_modules:
        add_module("cellprofiler_core.modules." + mod, True)

    # Import CellProfiler modules if CellProfiler is installed
    cpinstalled = False
    try:
        import cellprofiler.modules

        cpinstalled = True
    except ImportError:
        print("No CellProfiler installation detected, only base modules will be loaded")
    if cpinstalled:
        for mod in cellprofiler.modules.builtin_modules:
            add_module("cellprofiler.modules." + mod, True)

    # Find and import plugins
    plugin_directory = get_plugin_directory()
    if plugin_directory is not None:
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for mod in plugin_list(plugin_directory):
                add_module(mod, False)
        finally:
            sys.path = old_path

    if len(badmodules) > 0:
        logging.warning(
            "could not load these modules: %s", ",".join([x[0] for x in badmodules])
        )


def add_module_for_tst(module_class):
    all_modules[module_class.module_name] = module_class


def get_module_class(module_name):
    module_class = module_name.split(".")[-1]
    if module_class not in all_modules:
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
                    "not be implemented in CellProfiler 4.0."
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
    names = list(all_modules.keys())
    names.sort()
    return names


def reload_modules():
    for m in pymodules:
        try:
            del sys.modules[m.__name__]
        except:
            pass
    fill_modules()
