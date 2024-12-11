import os
import sys
import traceback

from ._reader import Reader
from ..constants.reader import ALL_READERS, builtin_readers, BAD_READERS, AVAILABLE_READERS

import logging

from ..preferences import config_read_typed

LOGGER = logging.getLogger(__name__)


def add_reader(reader_name, check_svn, class_name=None):
    LOGGER.debug(f"Registering {reader_name}")
    try:
        rdr = __import__(reader_name, globals(), locals(), ["__all__"], 0)
        if class_name:
            cp_reader = rdr.__dict__[class_name]
            assert issubclass(cp_reader, Reader)
        else:
            cp_reader = find_cp_reader(rdr)
        name = cp_reader.reader_name
    except Exception as e:
        reader_name = reader_name.replace("cellprofiler_core.readers.", "")
        LOGGER.warning(f"Could not load {reader_name}", exc_info=True)
        BAD_READERS[reader_name] = traceback.format_exc()
        return
    try:
        if name in ALL_READERS:
            LOGGER.warning(
                "Multiple definitions of reader %s\n\told in %s\n\tnew in %s",
                name,
                sys.modules[ALL_READERS[name].__module__].__file__,
                rdr.__file__,
            )
        ALL_READERS[name] = cp_reader
    except Exception as e:
        LOGGER.warning("Failed to load %s", name, exc_info=True)
        BAD_READERS[reader_name] = traceback.format_exc()
        if name in ALL_READERS:
            del ALL_READERS[name]


def fill_readers(check_config=True):
    ALL_READERS.clear()
    BAD_READERS.clear()

    # Import core modules
    for reader_name, classname in builtin_readers.items():
        add_reader("cellprofiler_core.readers." + reader_name, True, class_name=classname)

    if len(BAD_READERS) > 0:
        LOGGER.warning(
            f"could not load these readers: {', '.join(BAD_READERS)}"
        )

    activate_readers(check_config)


def activate_readers(check_config=True):
    AVAILABLE_READERS.clear()
    for reader_name, classname in ALL_READERS.items():
        if check_config:
            enabled = config_read_typed(f'Reader.{reader_name}.enabled', bool)
            if enabled or enabled is None:
                AVAILABLE_READERS[reader_name] = classname
        else:
            AVAILABLE_READERS[reader_name] = classname

    if len(AVAILABLE_READERS) == 0:
        LOGGER.critical("No image readers are available, CellProfiler won't be able to load data!")
    else:
        LOGGER.debug("Image readers available and active: %s", ", ".join(AVAILABLE_READERS))

def filter_active_readers(readers_to_keep, from_all_readers=True, by_module_name=True):
    """
    Filter out readers that are not in the readers_to_keep list.
    from_all_readers - if True, use ALL_READERS instead of AVAILABLE_READERS
    by_module_name - if True, use the module name, if False use the reader name
    """
    if from_all_readers:
        readers_to_filter = ALL_READERS.copy()
    else:
        readers_to_filter = AVAILABLE_READERS.copy()

    if by_module_name:
        filtered_readers = {reader_name: classname for reader_name, classname in readers_to_filter.items() if classname.__module__.split('.')[-1] in readers_to_keep}
    else:
        filtered_readers = {reader_name: classname for reader_name, classname in readers_to_filter.items() if classname.reader_name in readers_to_keep}

    if len(filtered_readers) == 0:
        LOGGER.critical("No image readers are available after filtering, CellProfiler won't be able to load data!")
    else:
        LOGGER.debug("Image readers available and active after filtering: %s", ", ".join(filtered_readers))

    AVAILABLE_READERS.clear()
    AVAILABLE_READERS.update(filtered_readers)

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
    if not AVAILABLE_READERS:
        raise Exception("No image readers are enabled.\n"
                        "Please check reader configuration in the File menu.")
    if use_cached_name and image_file.preferred_reader in AVAILABLE_READERS:
        reader_class = get_image_reader_by_name(image_file.preferred_reader)
        return reader_class
    LOGGER.debug(f"Choosing reader for {image_file.filename}")
    best_reader = None
    best_value = 5
    for reader_name, reader_class in AVAILABLE_READERS.items():
        result = reader_class.supports_format(image_file, volume=volume, allow_open=False)
        if result == 1:
            LOGGER.debug(f"Selected {reader_name}")
            image_file.preferred_reader = reader_name
            return reader_class
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
    assert reader_name in ALL_READERS, f"Reader {reader_name} not found"
    if reader_name not in AVAILABLE_READERS:
        LOGGER.warning(f"Requested reader {reader_name} which is disabled by config."
                       f"CellProfiler will use this reader anyway.")
    return ALL_READERS[reader_name]
