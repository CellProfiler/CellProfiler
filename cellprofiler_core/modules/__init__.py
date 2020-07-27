import sys

from ..module import PluginImporter

from ..utilities.modules import fill_modules

sys.meta_path.append(PluginImporter())

fill_modules()
