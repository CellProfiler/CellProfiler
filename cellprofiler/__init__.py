import pkg_resources

__test__ = False


__version_file__ = open(pkg_resources.resource_filename("cellprofiler", "data/VERSION"))
__version__ = __version_file__.read().strip()
