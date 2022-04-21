import glob
import inspect
import logging
import os
import sys


from cellprofiler_core.constants.modules import all_modules
from cellprofiler_core.constants.modules import pymodules
from cellprofiler_core.constants.reader import all_readers
from cellprofiler_core.preferences import get_plugin_directory
from cellprofiler_core.module import Module
from cellprofiler_core.reader import Reader

LOGGER = logging.getLogger(__name__)


def plugin_list(plugin_dir):
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "[!_]*.py"))
        return [os.path.basename(f)[:-3] for f in file_list]
    return []


def load_plugins(modules_only=False):
    # Find and import plugins
    plugin_directory = get_plugin_directory()
    if plugin_directory is not None:
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for plugin in plugin_list(plugin_directory):
                load_plugin(plugin, modules_only=modules_only)
        finally:
            sys.path = old_path


def load_plugin(source, modules_only=False):
    try:
        m = __import__(source, globals(), locals(), ["__all__"], 0)
        pymodules.append(m)
        available_classes = inspect.getmembers(
            m, lambda member: inspect.isclass(member) and member.__module__ == m.__name__)
        for name, plugin_class in available_classes:
            if issubclass(plugin_class, Module):
                add_module(plugin_class)
                break
            elif modules_only:
                continue
            elif issubclass(plugin_class, Reader):
                add_reader(plugin_class)
                break
        else:
            LOGGER.warning(f"Could not find Module{' or Reader' if not modules_only else ''} class in {m.__file__}")
    except Exception as e:
        LOGGER.warning("Could not load %s", source, exc_info=True)
        return


def add_module(cp_module):
    LOGGER.debug("Registering ", cp_module.__name__)
    name = None
    try:
        name = cp_module.module_name
        if name in all_modules:
            LOGGER.warning(
                "Multiple definitions of module %s\n\told in %s\n\tnew in %s",
                name,
                sys.modules[all_modules[name].__module__].__file__,
                inspect.getfile(cp_module),
            )
        all_modules[name] = cp_module
        from cellprofiler_core.utilities.core.modules import check_module
        check_module(cp_module, name)
        # attempt to instantiate
        if not hasattr(cp_module, "do_not_check"):
            cp_module()
    except Exception as e:
        LOGGER.warning("Failed to load %s", cp_module, exc_info=True)
        if name in all_modules:
            del all_modules[name]
            del pymodules[-1]


def add_reader(cp_reader):
    LOGGER.debug("Registering ", cp_reader.__name__)
    name = None
    try:
        name = cp_reader.reader_name

        if name in all_readers:
            LOGGER.warning(
                "Multiple definitions of reader %s\n\told in %s\n\tnew in %s",
                name,
                sys.modules[all_readers[name].__module__].__file__,
                inspect.getfile(cp_reader),
            )
        all_readers[name] = cp_reader
    except Exception as e:
        LOGGER.warning("Failed to load %s", name, exc_info=True)
        if name in all_readers:
            del all_readers[name]
            del pymodules[-1]
