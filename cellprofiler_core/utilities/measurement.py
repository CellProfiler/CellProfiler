import os
import re
import sys
import tempfile

from .hdf5_dict import get_top_level_group
from .hdf5_dict import VERSION


def get_length_from_varchar(x):
    """Retrieve the length of a varchar column from its coltype def"""
    m = re.match(r"^varchar\(([0-9]+)\)$", x)
    if m is None:
        return None
    return int(m.groups()[0])


def make_temporary_file():
    """Make a temporary file to use for backing measurements data

    returns a file descriptor (that should be closed when done) and a
    file name.
    """
    from ..preferences import get_temporary_directory

    temporary_directory = get_temporary_directory()
    if not (
        os.path.exists(temporary_directory) and os.access(temporary_directory, os.W_OK)
    ):
        temporary_directory = None
    return tempfile.mkstemp(
        prefix="Cpmeasurements", suffix=".hdf5", dir=temporary_directory
    )


def load_measurements(filename, dest_file=None, run_name=None, image_numbers=None):
    """Load measurements from an HDF5 file

    filename - path to file containing the measurements or file-like object
               if .mat

    dest_file - path to file to be created. This file is used as the backing
                store for the measurements.

    run_name - name of the run (an HDF file can contain measurements
               from multiple runs). By default, takes the last.

    returns a Measurements object
    """
    from ..measurement import Measurements

    HDF5_HEADER = (
        chr(137) + chr(72) + chr(68) + chr(70) + chr(13) + chr(10) + chr(26) + chr(10)
    )
    if hasattr(filename, "seek"):
        filename.seek(0)
        header = filename.read(len(HDF5_HEADER))
        filename.seek(0)
    else:
        fd = open(filename, "rb")
        header = fd.read(len(HDF5_HEADER))
        fd.close()

    if header.decode("unicode_escape") == HDF5_HEADER:
        f, top_level = get_top_level_group(filename)
        try:
            if VERSION in list(f.keys()):
                if run_name is not None:
                    top_level = top_level[run_name]
                else:
                    # Assume that the user wants the last one
                    last_key = sorted(top_level.keys())[-1]
                    top_level = top_level[last_key]
            m = Measurements(
                filename=dest_file, copy=top_level, image_numbers=image_numbers
            )
            return m
        finally:
            f.close()
    else:
        # FIXME - add clearer exception
        raise ValueError("Received HDF5 file header was invalid")


def load_measurements_from_buffer(buf):
    from ..preferences import get_default_output_directory

    dirtgt = get_default_output_directory()
    if not (os.path.exists(dirtgt) and os.access(dirtgt, os.W_OK)):
        dirtgt = None
    fd, filename = tempfile.mkstemp(prefix="Cpmeasurements", suffix=".hdf5", dir=dirtgt)
    if sys.platform.startswith("win"):
        # Change file descriptor mode to binary
        import msvcrt

        msvcrt.setmode(fd, os.O_BINARY)
    os.write(fd, buf)
    os.close(fd)
    try:
        return load_measurements(filename)
    finally:
        os.unlink(filename)


def find_metadata_tokens(pattern):
    """Return a list of strings which are the metadata token names in a pattern

    pattern - a regexp-like pattern that specifies how to find
              metadata in a string. Each token has the form:
              "(?<METADATA_TAG>...match-exp...)" (matlab-style) or
              "\g<METADATA_TAG>" (Python-style replace)
              "(?P<METADATA_TAG>...match-exp..)" (Python-style search)
    """
    result = []
    while True:
        m = re.search("\\(\\?[<](.+?)[>]", pattern)
        if not m:
            m = re.search("\\\\g[<](.+?)[>]", pattern)
            if not m:
                m = re.search("\\(\\?P[<](.+?)[>]", pattern)
                if not m:
                    break
        result.append(m.groups()[0])
        pattern = pattern[m.end() :]
    return result


def extract_metadata(pattern, text):
    """Return a dictionary of metadata extracted from the text

    pattern - a regexp that specifies how to find
              metadata in a string. Each token has the form:
              "\(?<METADATA_TAG>...match-exp...\)" (matlab-style) or
              "\(?P<METADATA_TAG>...match-exp...\)" (Python-style)
    text - text to be searched

    We do a little fixup in here to change Matlab searches to Python ones
    before executing.
    """
    # Convert Matlab to Python
    orig_pattern = pattern
    pattern = re.sub("(\\(\\?)([<].+?[>])", "\\1P\\2", pattern)
    match = re.search(pattern, text)
    if match:
        return match.groupdict()
    else:
        raise ValueError(
            "Metadata extraction failed: regexp '%s' does not match '%s'"
            % (orig_pattern, text)
        )


def is_well_row_token(x):
    """True if the string represents a well row metadata tag"""
    return x.lower() in ("wellrow", "well_row", "row")


def is_well_column_token(x):
    """true if the string represents a well column metadata tag"""
    return x.lower() in (
        "wellcol",
        "well_col",
        "wellcolumn",
        "well_column",
        "column",
        "col",
    )


def get_agg_measurement_name(agg, object_name, feature):
    """Return the name of an aggregate measurement

    agg - one of the names in AGG_NAMES, like AGG_MEAN
    object_name - the name of the object that we're aggregating
    feature - the name of the object's measurement
    """
    return "%s_%s_%s" % (agg, object_name, feature)


def agg_ignore_feature(feature_name):
    """Return True if the feature is one to be ignored when aggregating"""
    if feature_name.startswith("Description_"):
        return True
    if feature_name.startswith("ModuleError_"):
        return True
    if feature_name.startswith("TimeElapsed_"):
        return True
    if feature_name == "Number_Object_Number":
        return True
    return False
