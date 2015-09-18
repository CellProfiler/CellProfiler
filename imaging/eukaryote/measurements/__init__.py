"""
Measurements.py - storage for image and object measurements
"""

from __future__ import with_statement

import h5py
import json
import logging

logger = logging.getLogger(__name__)
import numpy as np
import re
from scipy.io.matlab import loadmat
from itertools import repeat
import cellprofiler.preferences as cpprefs
from cellprofiler.utilities.hdf5_dict import HDF5Dict, get_top_level_group
from cellprofiler.utilities.hdf5_dict import VERSION, HDFCSV, VStringArray
from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
from cellprofiler.utilities.hdf5_dict import NullLock
import tempfile
import numpy as np
import warnings
import os
import os.path
import mmap
import urllib
import sys

AGG_MEAN = "Mean"
AGG_STD_DEV = "StDev"
AGG_MEDIAN = "Median"
AGG_NAMES = [AGG_MEAN, AGG_MEDIAN, AGG_STD_DEV]

"""The per-image measurement category"""
IMAGE = "Image"

"""The per-experiment measurement category"""
EXPERIMENT = "Experiment"

"""The relationship measurement category"""
RELATIONSHIP = "Relationship"

"""The neighbor association measurement category"""
NEIGHBORS = "Neighbors"

"""The per-object "category" (if anyone needs the word, "Object")"""
OBJECT = "Object"

disallowed_object_names = [IMAGE, EXPERIMENT, RELATIONSHIP]

COLTYPE_INTEGER = "integer"
COLTYPE_FLOAT = "float"
'''16bit Binary Large Object. This object can fit 64K of raw data.
Currently used for storing image thumbnails as 200 x 200px (max) 8-bit pngs.
Should NOT be used for storing larger than 256 x 256px 8-bit pngs.'''
COLTYPE_BLOB = "blob"
'''24bit Binary Large Object. This object can fit 16M of raw data.
Not currently used'''
COLTYPE_MEDIUMBLOB = "mediumblob"
'''32bit Binary Large Object. This object can fit 4GB of raw data.
Not currently used'''
COLTYPE_LONGBLOB = "longblob"
'''SQL format for a varchar column

To get a varchar column of width X: COLTYPE_VARCHAR_FORMAT % X
'''
COLTYPE_VARCHAR_FORMAT = "varchar(%d)"
COLTYPE_VARCHAR = "varchar"
'''# of characters reserved for path name in the database'''
PATH_NAME_LENGTH = 256
'''# of characters reserved for file name in the database'''
FILE_NAME_LENGTH = 128
COLTYPE_VARCHAR_FILE_NAME = COLTYPE_VARCHAR_FORMAT % FILE_NAME_LENGTH
COLTYPE_VARCHAR_PATH_NAME = COLTYPE_VARCHAR_FORMAT % PATH_NAME_LENGTH

'''Column attribute: available after each cycle'''
MCA_AVAILABLE_EACH_CYCLE = "AvailableEachCycle"

'''Column attribute: only available after post_group is run (True / False)'''
MCA_AVAILABLE_POST_GROUP = "AvailablePostGroup"

'''Column attribute: only available after post_run is run'''
MCA_AVAILABLE_POST_RUN = "AvailablePostRun"

'''The name of the metadata category'''
C_METADATA = "Metadata"

'''The name of the site metadata feature'''
FTR_SITE = "Site"

'''The name of the well metadata feature'''
FTR_WELL = "Well"

'''The name of the row metadata feature'''
FTR_ROW = "Row"

'''The name of the column metadata feature'''
FTR_COLUMN = "Column"

'''The name of the plate metadata feature'''
FTR_PLATE = "Plate"

M_SITE, M_WELL, M_ROW, M_COLUMN, M_PLATE = \
    ['_'.join((C_METADATA, x))
     for x in (FTR_SITE, FTR_WELL, FTR_ROW, FTR_COLUMN, FTR_PLATE)]

MEASUREMENTS_GROUP_NAME = "Measurements"
IMAGE_NUMBER = "ImageNumber"
OBJECT_NUMBER = "ObjectNumber"
GROUP_NUMBER = "Group_Number"  # 1-based group index
GROUP_INDEX = "Group_Index"  # 1-based index within group

"""The image number of the first object in the relationship"""
R_FIRST_IMAGE_NUMBER = IMAGE_NUMBER + "_" + "First"

"""The object number of the first object in the relationship"""
R_FIRST_OBJECT_NUMBER = OBJECT_NUMBER + "_" + "First"

"""The image number of the second object in the relationship"""
R_SECOND_IMAGE_NUMBER = IMAGE_NUMBER + "_" + "Second"

"""The object number of the first object in the relationship"""
R_SECOND_OBJECT_NUMBER = OBJECT_NUMBER + "_" + "Second"

"""Indicates """

'''The FileName measurement category'''
C_FILE_NAME = "FileName"

'''The PathName measurement category'''
C_PATH_NAME = "PathName"

'''The URL measurement category'''
C_URL = "URL"

'''The series of an image file'''
C_SERIES = "Series"

'''The frame of a movie file'''
C_FRAME = "Frame"

'''For 3-D images (e.g. 3 color planes), the indexes of the planes'''
C_FRAMES = "Frames"

'''The channel # of a color image plane'''
C_CHANNEL = "Channel"

'''The FileName measurement category when loading objects'''
C_OBJECTS_FILE_NAME = "ObjectsFileName"

'''The PathName measurement category when loading objects'''
C_OBJECTS_PATH_NAME = "ObjectsPathName"

'''The URL category when loading objects'''
C_OBJECTS_URL = "ObjectsURL"

'''The series of an image file'''
C_OBJECTS_SERIES = "ObjectsSeries"

'''The index of an image file'''
C_OBJECTS_FRAME = "ObjectsFrame"

'''The channel # of a color image plane'''
C_OBJECTS_CHANNEL = "ObjectsChannel"

'''The ChannelType experiment measurement category'''
C_CHANNEL_TYPE = "ChannelType"

'''The automatically-collected file location (as a URL)'''
C_FILE_LOCATION = "File_Location"

'''The experiment feature name used to store the image set's metadata tags'''
M_METADATA_TAGS = "_".join((C_METADATA, "Tags"))

'''The experiment feature name used to store the image set's grouping tags'''
M_GROUPING_TAGS = "_".join((C_METADATA, "GroupingTags"))

'''Tags that are reserved for automatic population of metadata'''
RESERVED_METADATA_TAGS = ("C", "T", "Z", "ColorFormat", "ChannelName",
                          C_SERIES, C_FRAME, C_FILE_LOCATION)

'''A JSON-encoding of the local/remote path mappings'''
M_PATH_MAPPINGS = "Path_Mappings"

'''Case-sensitive comparison flag in M_PATH_MAPPINGS'''
K_CASE_SENSITIVE = "CaseSensitive"

'''Path-mappings sequence of two-tuple key in M_PATH_MAPPINGS'''
K_PATH_MAPPINGS = "PathMappings"

'''Local path separator as stored in M_PATH_MAPPINGS'''
K_LOCAL_SEPARATOR = "LocalSeparator"

'''Source of local url2pathname function for M_PATH_MAPPINGS'''
K_URL2PATHNAME_PACKAGE_NAME = "Url2PathnamePackageName"

'''Name of the batch data file'''
F_BATCH_DATA = 'Batch_data.mat'

'''Name of the .h5 batch data file'''
F_BATCH_DATA_H5 = 'Batch_data.h5'


def get_length_from_varchar(x):
    '''Retrieve the length of a varchar column from its coltype def'''
    m = re.match(r'^varchar\(([0-9]+)\)$', x)
    if m is None:
        return None
    return int(m.groups()[0])


def make_temporary_file():
    '''Make a temporary file to use for backing measurements data
    
    returns a file descriptor (that should be closed when done) and a
    file name.
    '''
    dir = cpprefs.get_temporary_directory()
    if not (os.path.exists(dir) and os.access(dir, os.W_OK)):
        dir = None
    return tempfile.mkstemp(
        prefix='Cpmeasurements', suffix='.hdf5', dir=dir)


def load_measurements_from_buffer(buf):
    dir = cpprefs.get_default_output_directory()
    if not (os.path.exists(dir) and os.access(dir, os.W_OK)):
        dir = None
    fd, filename = tempfile.mkstemp(prefix='Cpmeasurements', suffix='.hdf5', dir=dir)
    if sys.platform.startswith('win'):
        # Change file descriptor mode to binary
        import msvcrt
        msvcrt.setmode(fd, os.O_BINARY)
    os.write(fd, buf)
    os.close(fd)
    try:
        return load_measurements(filename)
    finally:
        os.unlink(filename)


def load_measurements(filename, dest_file=None, can_overwrite=False, run_name=None, image_numbers=None):
    '''Load measurements from an HDF5 file
    
    filename - path to file containing the measurements or file-like object
               if .mat
    
    dest_file - path to file to be created. This file is used as the backing
                store for the measurements.
                
    can_overwrite - True to allow overwriting of existing measurements (not
                    supported any longer)
                    
    run_name - name of the run (an HDF file can contain measurements
               from multiple runs). By default, takes the last.
    
    returns a Measurements object
    '''
    HDF5_HEADER = (chr(137) + chr(72) + chr(68) + chr(70) + chr(13) + chr(10) +
                   chr(26) + chr(10))
    if hasattr(filename, "seek"):
        filename.seek(0)
        header = filename.read(len(HDF5_HEADER))
        filename.seek(0)
    else:
        fd = open(filename, "rb")
        header = fd.read(len(HDF5_HEADER))
        fd.close()

    if header == HDF5_HEADER:
        f, top_level = get_top_level_group(filename)
        try:
            if VERSION in f.keys():
                if run_name is not None:
                    top_level = top_level[run_name]
                else:
                    # Assume that the user wants the last one
                    last_key = sorted(top_level.keys())[-1]
                    top_level = top_level[last_key]
            m = Measurements(filename=dest_file, copy=top_level,
                             image_numbers=image_numbers)
            return m
        except:
            logger.error("Error loading HDF5 %s", filename, exc_info=True)
        finally:
            f.close()
    else:
        m = Measurements(filename=dest_file)
        m.load(filename)
        return m


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
        m = re.search('\\(\\?[<](.+?)[>]', pattern)
        if not m:
            m = re.search('\\\\g[<](.+?)[>]', pattern)
            if not m:
                m = re.search('\\(\\?P[<](.+?)[>]', pattern)
                if not m:
                    break
        result.append(m.groups()[0])
        pattern = pattern[m.end():]
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
    pattern = re.sub('(\\(\\?)([<].+?[>])', '\\1P\\2', pattern)
    match = re.search(pattern, text)
    if match:
        return match.groupdict()
    else:
        raise ValueError("Metadata extraction failed: regexp '%s' does not match '%s'" % (orig_pattern, text))


def is_well_row_token(x):
    '''True if the string represents a well row metadata tag'''
    return x.lower() in ("wellrow", "well_row", "row")


def is_well_column_token(x):
    '''true if the string represents a well column metadata tag'''
    return x.lower() in ("wellcol", "well_col", "wellcolumn", "well_column",
                         "column", "col")


def get_agg_measurement_name(agg, object_name, feature):
    '''Return the name of an aggregate measurement

    agg - one of the names in AGG_NAMES, like AGG_MEAN
    object_name - the name of the object that we're aggregating
    feature - the name of the object's measurement
    '''
    return "%s_%s_%s" % (agg, object_name, feature)


def agg_ignore_feature(feature_name):
    '''Return True if the feature is one to be ignored when aggregating'''
    if feature_name.startswith('Description_'):
        return True
    if feature_name.startswith('ModuleError_'):
        return True
    if feature_name.startswith('TimeElapsed_'):
        return True
    if feature_name == "Number_Object_Number":
        return True
    return False
