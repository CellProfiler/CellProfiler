import os.path


__test__ = False

__version_file__ = open(os.path.join(os.path.dirname(__file__), 'VERSION'))
__version__ = __version_file__.read().strip()