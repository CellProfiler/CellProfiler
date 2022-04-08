import os
import sys

from ._reader import Reader
from ..constants.reader import all_readers, builtin_readers, bad_readers

import logging


LOGGER = logging.getLogger(__name__)


def add_reader(reader_name, check_svn, class_name=None):
    LOGGER.debug("Registering ", reader_name)
    try:
        rdr = __import__(reader_name, globals(), locals(), ["__all__"], 0)
        if class_name:
            cp_reader = rdr.__dict__[class_name]
            assert issubclass(cp_reader, Reader)
        else:
            cp_reader = find_cp_reader(rdr)
        name = cp_reader.reader_name
    except Exception as e:
        LOGGER.warning("Could not load %s", reader_name, exc_info=True)
        bad_readers.append((reader_name, e))
        return

    try:
        if name in all_readers:
            LOGGER.warning(
                "Multiple definitions of reader %s\n\told in %s\n\tnew in %s",
                name,
                sys.modules[all_readers[name].__module__].__file__,
                rdr.__file__,
            )
        all_readers[name] = cp_reader
    except Exception as e:
        LOGGER.warning("Failed to load %s", name, exc_info=True)
        bad_readers.append((reader_name, e))
        if name in all_readers:
            del all_readers[name]


def fill_readers():
    all_readers.clear()
    bad_readers.clear()

    # Import core modules
    for reader_name, classname in builtin_readers.items():
        add_reader("cellprofiler_core.readers." + reader_name, True, class_name=classname)

    # Find and import plugins
    # Todo: Plugin system
    # plugin_directory = get_plugin_directory()
    # if plugin_directory is not None:
    #     old_path = sys.path
    #     sys.path.insert(0, plugin_directory)
    #     try:
    #         for rdr in plugin_list(plugin_directory):
    #             add_module(mod, False)
    #     finally:
    #         sys.path = old_path

    if len(bad_readers) > 0:
        LOGGER.warning(
            "could not load these modules: %s", ",".join([x[0] for x in bad_readers])
        )
    if len(all_readers) == 0:
        LOGGER.critical("No image readers are available, CellProfiler won't be able to load data!")


def find_cp_reader(rdr):
    """Returns the CPModule from within the loaded Python module

    m - an imported module

    returns the CPModule class
    """
    reader_name = os.path.splitext(os.path.basename(rdr.__file__))[0]
    for key, item in list(rdr.__dict__.items()):
        if key.lower() == reader_name and issubclass(item, Reader):
            return item
    raise ValueError(
        "Could not find cellprofiler_core.reader.Reader class in %s" % rdr.__file__
    )


def get_image_reader_class(image_file, use_cached_name=True, volume=False):
    if use_cached_name and image_file.preferred_reader in all_readers:
        reader_class = get_image_reader_by_name(image_file.preferred_reader)
        return reader_class
    LOGGER.debug(f"Choosing reader for {image_file.filename}")
    best_reader = None
    best_value = 5
    for reader_name, reader_class in all_readers.items():
        result = reader_class.supports_format(image_file, volume=volume, allow_open=False)
        if result == 1:
            LOGGER.debug(f"Selected {reader_name}")
            image_file.preferred_reader = reader_name
            return reader_class(image_file)
        elif 1 < result < best_value:
            best_value = result
            best_reader = reader_class
    if best_reader is None:
        raise NotImplementedError(f"No reader available for {image_file.filename}")
    LOGGER.debug(f"Selected {best_reader}")
    return best_reader


def get_image_reader(image_file, use_cached_name=True, volume=False):
    reader_class = get_image_reader_class(image_file, use_cached_name=use_cached_name, volume=volume)
    return reader_class(image_file)


def get_image_reader_by_name(reader_name):
    assert reader_name in all_readers, f"Reader {reader_name} not found"
    return all_readers[reader_name]


fill_readers()
