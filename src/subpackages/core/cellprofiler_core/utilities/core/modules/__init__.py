import logging
import os
import re
import sys

from ....module import Module
from ....constants.modules import (
    do_not_override,
    should_override,
    all_modules,
    svn_revisions,
    pymodules,
    badmodules,
    builtin_modules,
    renamed_modules,
    unimplemented_modules,
    depricated_modules,
    replaced_modules,
)
from ..plugins import load_plugins


LOGGER = logging.getLogger(__name__)

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
    modulename = os.path.splitext(os.path.basename(m.__file__))[0]
    for key, item in list(m.__dict__.items()):
        if key.lower() == modulename and issubclass(item, Module):
            return item
    raise ValueError(
        "Could not find cellprofiler_core.module.Module class in %s" % m.__file__
    )


def fill_modules():
    del pymodules[:]
    del badmodules[:]
    all_modules.clear()
    svn_revisions.clear()

    def add_module(mod, check_svn, class_name=None):
        try:
            m = __import__(mod, globals(), locals(), ["__all__"], 0)
            if class_name:
                cp_module = m.__dict__[class_name]
                assert issubclass(cp_module, Module)
            else:
                cp_module = find_cpmodule(m)
            name = cp_module.module_name
        except Exception as e:
            LOGGER.warning("Could not load %s", mod, exc_info=True)
            badmodules.append((mod, e))
            return

        try:
            pymodules.append(m)
            if name in all_modules:
                LOGGER.warning(
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
            LOGGER.warning("Failed to load %s", name, exc_info=True)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

    # Import core modules
    for modname, classname in builtin_modules.items():
        add_module("cellprofiler_core.modules." + modname, True, class_name=classname)

    # Import CellProfiler modules if CellProfiler is installed
    cpinstalled = False
    try:
        import cellprofiler.modules

        cpinstalled = True
    except ImportError:
        print("No CellProfiler installation detected, only base modules will be loaded")
    if cpinstalled:
        for modname, classname in cellprofiler.modules.builtin_modules.items():
            add_module("cellprofiler.modules." + modname, True, class_name=classname)

    if len(badmodules) > 0:
        LOGGER.warning(
            "could not load these modules: %s", ",".join([x[0] for x in badmodules])
        )


def add_module_for_tst(module_class):
    all_modules[module_class.module_name] = module_class


def get_module_class(module_name):
    module_class = module_name.split(".")[-1]
    if module_class not in all_modules:
        if module_class in renamed_modules:
            return all_modules[renamed_modules[module_class]]
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
    load_plugins(modules_only=True)
