import sys

from ..module._plugin_importer import PluginImporter

from ..utilities.core.modules import fill_modules

sys.meta_path.append(PluginImporter())

fill_modules()
