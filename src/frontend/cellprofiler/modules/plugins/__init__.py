"""Plugin modules - redirect module loads to the user's plugin
directory.
"""

# These examples:
# http://blog.dowski.com/2008/07/31/customizing-the-python-import-system/
# http://www.secomputing.co.uk/2009/05/importing-python-packages-in-memory.html
# were helpful when writing this code.

import glob
import os.path
import sys

import cellprofiler_core.preferences as cpprefs
from cellprofiler_core.module._plugin_importer import PluginImporter


def plugin_list():
    plugin_dir = cpprefs.get_plugin_directory()
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "*.py"))
        return [
            os.path.basename(f)[:-3] for f in file_list if not f.endswith("__init__.py")
        ]
    return []


sys.meta_path.append(PluginImporter())
