import sys

from ..module._plugin_importer import PluginImporter

from ..utilities.core.modules import fill_modules
from ..utilities.core.plugins import load_plugins

sys.meta_path.append(PluginImporter())

fill_modules()
load_plugins()
