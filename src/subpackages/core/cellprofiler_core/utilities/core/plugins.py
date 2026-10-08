import glob
import importlib
import importlib.util
import inspect
import logging
import os
import re
import shutil
import sys
import tempfile
import traceback
import zipfile

import requests

from cellprofiler_core.constants.modules import all_modules
from cellprofiler_core.constants.modules import pymodules
from cellprofiler_core.constants.reader import ALL_READERS, BAD_READERS, AVAILABLE_READERS
from cellprofiler_core.preferences import get_plugin_directory, get_default_output_directory, config_read_typed
from cellprofiler_core.module import Module
from cellprofiler_core.reader import Reader

LOGGER = logging.getLogger(__name__)

OFFICIAL_PLUGINS_REPO_ZIP_URL = (
    "https://github.com/CellProfiler/CellProfiler-plugins/archive/refs/heads/cp5.zip"
)
OFFICIAL_PLUGINS_ZIP_SUBPATH = "CP5/active_plugins/"

# Per-plugin load status, keyed by plugin source name (as returned by plugin_list).
# Each value is {"loaded": bool, "kind": "module"|"reader"|None, "error": str|None,
# "directory": str|None, "appose_env_spec": str|pathlib.Path|None} -
# "directory" is whichever plugin directory this name was actually last
# attempted from. If the same name exists in more than one plugin directory,
# only the highest-priority one that successfully loads is ever attempted (see
# load_plugin's skip-guard, and _plugin_directories for priority order) - so
# this dict only ever holds one entry per name, reflecting whichever attempt
# is actually responsible for that name's current state, never a
# shadowed/superseded one. "appose_env_spec" is the plugin class's (Module
# or Reader) own declared `appose_env_spec` class attribute (see
# load_plugin), or None for plugins that don't run via Appose.
PLUGIN_STATUS = {}

# Prepended to a plugin module's display name (GUI only, never to module_name
# itself, which is also the pipeline-file serialization key).
PLUGIN_NAME_PREFIX = "\U0001F9E9 "


def get_module_display_name(module):
    """Return a module's display name, decorated if it came from a plugin.

    `module` may be a Module subclass or an instance of one. Does not use
    or affect `module_name`, which is the pipeline-file serialization key.
    """
    name = module.module_name
    if getattr(module, "is_plugin", False):
        return PLUGIN_NAME_PREFIX + name
    return name


def plugin_list(plugin_dir):
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "[!_]*.py"))
        return [os.path.basename(f)[:-3] for f in file_list]
    return []


def get_official_plugins_directory():
    return os.path.join(
        get_default_output_directory(), "CellProfiler-plugins", "active_plugins"
    )


def official_plugins_repo_exists():
    directory = get_official_plugins_directory()
    return os.path.isdir(directory) and len(plugin_list(directory)) > 0


def download_official_plugins_repo():
    """Download the officially-supported CellProfiler-plugins subset and
    extract it into get_official_plugins_directory(), replacing any
    existing copy there.
    """
    response = requests.get(OFFICIAL_PLUGINS_REPO_ZIP_URL, stream=True, timeout=30)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmp_zip.write(chunk)
        tmp_zip_path = tmp_zip.name
    try:
        with zipfile.ZipFile(tmp_zip_path) as zf:
            members = [
                name
                for name in zf.namelist()
                if OFFICIAL_PLUGINS_ZIP_SUBPATH in name and not name.endswith("/")
            ]
            if not members:
                raise RuntimeError(
                    f"Could not find {OFFICIAL_PLUGINS_ZIP_SUBPATH!r} in the downloaded archive"
                )
            target_directory = get_official_plugins_directory()
            if os.path.isdir(target_directory):
                shutil.rmtree(target_directory)
            os.makedirs(target_directory, exist_ok=True)
            for name in members:
                prefix, _, relative_path = name.partition(OFFICIAL_PLUGINS_ZIP_SUBPATH)
                destination = os.path.join(target_directory, relative_path)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with zf.open(name) as src, open(destination, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        os.remove(tmp_zip_path)
    return get_official_plugins_directory()


def get_plugin_statuses():
    """
    Return one status entry per plugin *name* found across all plugin
    directories (see _plugin_directories), merging in PLUGIN_STATUS where
    available. A name present in more than one directory (e.g. a user
    plugin shadowing a same-named official one) still produces exactly one
    entry, reflecting whichever directory's attempt is actually responsible
    for that name's current state - never a separate entry for the
    shadowed/superseded one.
    """
    default_directory_by_name = {}
    for directory in _plugin_directories():
        for name in plugin_list(directory):
            default_directory_by_name.setdefault(name, directory)
    statuses = []
    for name in sorted(default_directory_by_name):
        status = PLUGIN_STATUS.get(name)
        if status is None:
            status = {
                "loaded": False,
                "kind": None,
                "error": None,
                "directory": default_directory_by_name[name],
                "appose_env_spec": None,
            }
        statuses.append({"name": name, **status})
    return statuses


def _plugin_directories():
    directories = []
    seen_realpaths = set()
    for directory in (get_plugin_directory(), get_official_plugins_directory()):
        if directory is None or not os.path.isdir(directory):
            continue
        real = os.path.realpath(directory)
        if real in seen_realpaths:
            continue
        seen_realpaths.add(real)
        directories.append(directory)
    return directories


def load_plugins(modules_only=False):
    # Find and import plugins
    for plugin_directory in _plugin_directories():
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for plugin in plugin_list(plugin_directory):
                load_plugin(plugin, directory=plugin_directory, modules_only=modules_only)
        finally:
            sys.path = old_path


def load_plugin(source, directory=None, modules_only=False):
    if PLUGIN_STATUS.get(source, {}).get("loaded"):
        return
    try:
        m = importlib.import_module(source)
        pymodules.append(m)
        available_classes = inspect.getmembers(
            m, lambda member: inspect.isclass(member) and member.__module__ == m.__name__)
        for name, plugin_class in available_classes:
            if issubclass(plugin_class, Module):
                loaded, error = add_module(plugin_class)
                PLUGIN_STATUS[source] = {
                    "loaded": loaded,
                    "kind": "module",
                    "error": error,
                    "directory": directory,
                    "appose_env_spec": getattr(plugin_class, "appose_env_spec", None),
                }
                break
            elif modules_only:
                continue
            elif issubclass(plugin_class, Reader):
                loaded, error = add_reader(plugin_class)
                PLUGIN_STATUS[source] = {
                    "loaded": loaded,
                    "kind": "reader",
                    "error": error,
                    "directory": directory,
                    "appose_env_spec": getattr(plugin_class, "appose_env_spec", None),
                }
                break
        else:
            message = f"Could not find Module{' or Reader' if not modules_only else ''} class in {m.__file__}"
            LOGGER.warning(message)
            PLUGIN_STATUS[source] = {
                "loaded": False,
                "kind": None,
                "error": message,
                "directory": directory,
                "appose_env_spec": None,
            }
    except Exception as e:
        tb = traceback.format_exc()
        if not modules_only:
            # Figure out and store Reader class name, if present
            try:
                spec = importlib.util.find_spec(source)
                with open(spec.origin) as fd:
                    for line in fd:
                        if line.endswith("Reader):\n"):
                            result = re.search(r'.*class (?P<classname>.*)\(.*Reader\)', line)
                            if result:
                                reader_name = result.group('classname')
                                BAD_READERS[reader_name] = tb
                            break
                        elif line.endswith("Module):\n"):
                            break
            except Exception:
                pass
        LOGGER.warning("Could not load %s", source, exc_info=False)
        PLUGIN_STATUS[source] = {
            "loaded": False,
            "kind": None,
            "error": tb,
            "directory": directory,
            "appose_env_spec": None,
        }
        return


def add_module(cp_module):
    LOGGER.debug("Registering %s", cp_module.__name__)
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
        cp_module.is_plugin = True
        from cellprofiler_core.utilities.core.modules import check_module
        check_module(cp_module, name)
        # attempt to instantiate
        if not hasattr(cp_module, "do_not_check"):
            cp_module()
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.warning("Failed to load %s", cp_module, exc_info=True)
        if name in all_modules:
            del all_modules[name]
            del pymodules[-1]
        return False, tb
    return True, None


def add_reader(cp_reader):
    LOGGER.debug("Registering %s", cp_reader.__name__)
    name = None
    try:
        name = cp_reader.reader_name

        if name in ALL_READERS:
            LOGGER.warning(
                "Multiple definitions of reader %s\n\told in %s\n\tnew in %s",
                name,
                sys.modules[ALL_READERS[name].__module__].__file__,
                inspect.getfile(cp_reader),
            )
        ALL_READERS[name] = cp_reader
        enabled = config_read_typed(f'Reader.{name}.enabled', bool)
        if enabled or enabled is None:
            AVAILABLE_READERS[name] = cp_reader
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.warning("Failed to load %s", name, exc_info=True)
        if name in ALL_READERS:
            del ALL_READERS[name]
            del pymodules[-1]
        return False, tb
    return True, None
