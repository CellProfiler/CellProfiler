"""Measurements.py - storage for image and object measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
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
MCA_AVAILABLE_POST_GROUP  = "AvailablePostGroup"

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
RESERVED_METADATA_TAGS = ( "C", "T", "Z", "ColorFormat", "ChannelName",
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
    
class Measurements(object):
    """Represents measurements made on images and objects
    """
    def __init__(self,
                 can_overwrite=False,
                 image_set_start=None,
                 filename = None,
                 copy = None,
                 mode = "w",
                 image_numbers = None,
                 multithread = True):
        """Create a new measurements collection

        can_overwrite - DEPRECATED and has no effect
        image_set_start - the index of the first image set in the image set list
                          or None to start at the beginning
        filename - store the measurement in an HDF5 file with this name
        copy - initialize by copying measurements from here, either an HDF5Dict
               or an H5py group or file.
        mode - open mode for the HDF5 file. 
               "r" for read-only access to an existing measurements file, 
               "w" to open a new file or truncate an old file, 
               "w-" to open a new file and fail if the file exists,
               "w+" to create a new measurements instance in an existing file,
               "a" to create a new file or open an existing file as read/write
               "r+" to open an existing file as read/write
               "memory" to create an HDF5 memory-backed File
        multithread - True if this measurements structure is used in a
               multithreading context, False to disable locking.
        """
        # XXX - allow saving of partial results
        if mode == "memory" and sys.platform == "darwin":
            # Core driver doesn't work on Mac
            # http://code.google.com/p/h5py/issues/detail?id=215
            filename = None
            mode = "w"
        if mode == "memory":
            filename = None
            mode = "w"
            is_temporary = False
        elif filename is None:
            fd, filename = make_temporary_file()
            is_temporary = True
            import traceback
            logger.debug("Created temporary file %s" % filename)
            for frame in traceback.extract_stack():
                logger.debug("%s: (%d %s): %s" % frame)
        else:
            is_temporary = False
        if isinstance(copy, Measurements):
            with copy.hdf5_dict.lock:
                self.hdf5_dict = HDF5Dict(
                    filename, 
                    is_temporary = is_temporary,
                    copy = copy.hdf5_dict.top_group,
                    mode = mode,
                    image_numbers=image_numbers)
        elif hasattr(copy, '__getitem__') and hasattr(copy, 'keys'):
            self.hdf5_dict = HDF5Dict(
                filename,
                is_temporary = is_temporary,
                copy = copy,
                mode = mode,
                image_numbers=image_numbers)
            if not multithread:
                self.hdf5_dict.lock = NullLock
        elif copy is not None:
            raise ValueError('Copy source for measurments is neither a Measurements or HDF5 group.')
        else:
            self.hdf5_dict = HDF5Dict(filename, 
                                      is_temporary = is_temporary,
                                      mode = mode)
        if is_temporary:
            os.close(fd)

        self.image_set_number = image_set_start or 1
        self.image_set_start = image_set_start

        self.__is_first_image = True
        self.__initialized_explicitly = False
        self.__relationships = set()
        self.__images = {}
        self.__image_providers = []
        self.__images = {}
        self.__image_providers = []
        self.__image_number_relationships = {}
        self.__image_cache_file = None
        if RELATIONSHIP in self.hdf5_dict.top_group:
            rgroup = self.hdf5_dict.top_group[RELATIONSHIP]
            for module_number in rgroup:
                try:
                    mnum = int(module_number)
                except:
                    continue
                mrgroup = rgroup[module_number]
                if not isinstance(mrgroup, h5py.Group):
                    continue
                for rname in mrgroup:
                    rmrgroup = mrgroup[rname]
                    if not isinstance(rmrgroup, h5py.Group):
                        continue
                    for o1_name in rmrgroup:
                        rmro1group = rmrgroup[o1_name]
                        if not isinstance(rmro1group, h5py.Group):
                            continue
                        for o2_name in rmro1group:
                            if not isinstance(rmro1group[o2_name], h5py.Group):
                                continue
                            self.__relationships.add(
                                (mnum, rname, o1_name, o2_name))

    def __del__(self):
        if hasattr(self, "hdf5_dict"):
            self.close()

    def close(self):
        if hasattr(self, "hdf5_dict"):
            self.hdf5_dict.close()
            del self.hdf5_dict
        if self.__image_cache_file is not None:
            self.__image_cache_file.close()
            self.__image_cache_file = None
            os.remove(self.__image_cache_path)
            del self.__image_cache_path
        
    def __getitem__(self, key):
        # we support slicing the last dimension for the limited case of [..., :]
        if (len(key) == 3 and 
            isinstance(key[2], slice) and
            key[2] == slice(None, None, None)):
            return self.get_all_measurements(*key[:2])
        return self.get_measurement(*key)

    def __setitem__(self, key, value):
        assert 2 <= len(key) <= 4
        if len(key) == 2:
            self.add_measurement(key[0], key[1], value)
        elif len(key) == 3:
            self.add_measurement(key[0], key[1], value, image_set_number=key[2])
        else:
            self.add_measurement(key[0], key[1], value,
                                 image_set_number = key[2],
                                 data_type = key[3])

    def flush(self):
        if self.hdf5_dict is not None:
            self.hdf5_dict.flush()

    def file_contents(self):
        return self.hdf5_dict.file_contents()

    def initialize(self, measurement_columns):
        '''Initialize the measurements with a list of objects and features

        This explicitly initializes the measurements with a list of
        object/feature pairs as would be returned by
        get_measurement_columns()

        measurement_columns - list of 3-tuples: object name, feature, type
        '''
        # clear the old data, if any
        self.hdf5_dict.clear()

        def fix_type(t):
            if t == 'integer':
                return np.int
            if t.startswith('varchar'):
                len = t.split('(')[1][:-1]
                return np.dtype('a' + len)
            return t

        for object_name, feature, coltype in measurement_columns:
            coltype = fix_type(coltype)
            if object_name == EXPERIMENT:
                dims = 0
            elif object_name == IMAGE:
                dims = 1
            else:
                dims = 2
            self.hdf5_dict.add_object(object_name)
            self.hdf5_dict.add_feature(object_name, feature)
        self.__initialized_explicitly = True

    def next_image_set(self, explicit_image_set_number=None):
        assert explicit_image_set_number is None or explicit_image_set_number > 0
        if explicit_image_set_number is None:
            self.image_set_number += 1
        else:
            self.image_set_number = explicit_image_set_number
        self.__is_first_image = False
        self.__images = {}
        self.__image_providers = []

    @property
    def image_set_count(self):
        '''The number of complete image sets measured'''
        # XXX - question for Lee: should this return the minimum number
        # of non-null values across columns in the the Image table?
        try:
            return len(self.hdf5_dict.get_indices('Image', 'ImageNumber'))
        except KeyError:
            return 0

    def get_is_first_image(self):
        '''True if this is the first image in the set'''
        return self.__is_first_image

    def set_is_first_image(self, value):
        if not value:
            raise ValueError("Can only reset to be first image")
        self.__is_first_image = True
        self.image_set_number = self.image_set_start_number

    is_first_image = property(get_is_first_image, set_is_first_image)

    @property
    def image_set_start_number(self):
        '''The first image set (one-based) processed by the pipeline'''
        if self.image_set_start is None:
            return 1
        return self.image_set_start

    @property
    def has_image_set_start(self):
        '''True if the image set has an explicit start'''
        return self.image_set_start is not None

    def load(self, measurements_file_name):
        '''Load measurements from a matlab file'''
        handles = loadmat(measurements_file_name, struct_as_record=True)
        self.create_from_handles(handles)

    def create_from_handles(self, handles):
        '''Load measurements from a handles structure'''
        m = handles["handles"][0, 0][MEASUREMENTS_GROUP_NAME][0, 0]
        for object_name in m.dtype.fields.keys():
            omeas = m[object_name][0, 0]
            object_counts = np.zeros(0, int)
            for feature_name in omeas.dtype.fields.keys():
                if object_name == IMAGE:
                    values = [None if len(x) == 0 else x.flatten()[0] 
                              for x in omeas[feature_name][0]]
                elif object_name == EXPERIMENT:
                    value = omeas[feature_name][0, 0].flatten()[0]
                    self.add_experiment_measurement(feature_name, value)
                    continue
                else:
                    values = [x.flatten()
                              for x in omeas[feature_name][0].tolist()]
                    #
                    # Keep track of # of objects
                    #
                    if len(object_counts) < len(values):
                        temp, object_counts = object_counts, np.zeros(len(values), int)
                        if len(temp) > 0:
                            object_counts[:len(temp)] = temp
                    object_counts[:len(values)] = np.maximum(
                        object_counts[:len(values)], 
                        np.array([len(x) for x in values]))
                self.add_all_measurements(object_name,
                                          feature_name,
                                          values)
            if object_name not in (EXPERIMENT, IMAGE) and not self.has_feature(
                object_name, OBJECT_NUMBER):
                self.add_all_measurements(
                    object_name, OBJECT_NUMBER,
                    [np.arange(1, x+1) for x in object_counts])
        #
        # Set the image set number to beyond the last in the handles
        #
        self.image_set_number = self.image_set_count + 1

    def add_image_measurement(self, feature_name, data, can_overwrite = False):
        """Add a measurement to the "Image" category

        """
        self.add_measurement(IMAGE, feature_name, data)

    def add_experiment_measurement(self, feature_name, data):
        """Add an experiment measurement to the measurement

        Experiment measurements have one value per experiment
        """
        if isinstance(data, basestring):
            data = unicode(data).encode('unicode_escape')
        self.hdf5_dict.add_all(EXPERIMENT, feature_name, [data], [0])

    def get_group_number(self):
        '''The number of the group currently being processed'''
        return self.get_current_image_measurement(GROUP_NUMBER)

    def set_group_number(self, group_number, can_overwrite=False):
        self.add_image_measurement(GROUP_NUMBER, group_number)

    group_number = property(get_group_number, set_group_number)

    def get_group_index(self):
        '''The within-group index of the current image set'''
        return self.get_current_image_measurement(GROUP_INDEX)

    def set_group_index(self, group_index):
        self.add_image_measurement(GROUP_INDEX, group_index)

    group_index = property(get_group_index, set_group_index)
    
    def get_groupings(self, features):
        '''Return groupings of image sets based on feature values
        
        features - a sequence of feature names
                   
        returns groupings suitable for return from CPModule.get_groupings.
        
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple is a dictionary giving
                     the metadata values for the metadata keys
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Metadata_Row':'A','Metadata_Column':'01'}, [1,97,193]),
          ({'Metadata_Row':'A','Metadata_Column':'02'), [2,98,194]),... ]
        '''
        d = {}
        image_numbers = self.get_image_numbers()
        values = [[unicode(x) for x in self.get_measurement(IMAGE, feature, image_numbers)]
                  for feature in features]
        for i, image_number in enumerate(image_numbers):
            key = tuple([(k, v[i]) for k, v in zip(features, values)])
            if not d.has_key(key):
                d[key] = []
            d[key].append(image_number)
        return [ (dict(k), d[k]) for k in sorted(d.keys()) ]
            

    def get_relationship_hdf5_group(self, module_number, relationship,
                                    object_name1, object_name2):
        '''Return the HDF5 group for a relationship'''
        return self.hdf5_dict.top_group \
                .require_group(RELATIONSHIP)\
                .require_group(str(module_number)) \
                .require_group(relationship) \
                .require_group(object_name1) \
                .require_group(object_name2)
    
    def add_relate_measurement(
        self, module_number,
        relationship,
        object_name1, object_name2,
        image_numbers1, object_numbers1,
        image_numbers2, object_numbers2):
        '''Add object relationships to the measurements

        module_number - the module that generated the relationship

        relationship - the relationship of the two objects, for instance,
                       "Parent" means object # 1 is the parent of object # 2

        object_name1, object_name2 - the name of the segmentation for the first and second objects

        image_numbers1, image_numbers2 - the image number of each object

        object_numbers1, object_numbers2 - for each object, the object number
                                           in the object's object set

        This method lets the caller store any sort of arbitrary relationship
        between objects as long as they are in the same group. To record
        all neighbors within a particular segmentation, call with the same
        object name for object_name1 and object_name2 and the same group
        index - that of the current image. Relating would have different object
        names and TrackObjects would have different group indices.
        
        The structure in the HDF file:
        Measurements / <date> / Relationship / <module #> /
           <relationship-name> / <object-name-1> / <object-name-2> /
           [ImageNumber_First, ObjectNumber_First, 
            ImageNumber_Second, ObjectNumber_Second]
        
        The leaves are vector datasets.
        '''
        if len(image_numbers1) == 0:
            return
        with self.hdf5_dict.lock:
            rgroup = self.get_relationship_hdf5_group(
                module_number, relationship, object_name1, object_name2)
            
            for name, values in ((R_FIRST_IMAGE_NUMBER, image_numbers1),
                                 (R_FIRST_OBJECT_NUMBER, object_numbers1),
                                 (R_SECOND_IMAGE_NUMBER, image_numbers2),
                                 (R_SECOND_OBJECT_NUMBER, object_numbers2)):
                if name not in rgroup:
                    current_size = 0
                    rgroup.create_dataset(name, data=values, 
                                          dtype='int32', chunks=(1024,), 
                                          maxshape=(None,))
                else:
                    dset = rgroup[name]
                    current_size = dset.shape[0]
                    dset.resize((current_size + len(values),))
                    dset[current_size:] = values
            key = (module_number, relationship, 
                   object_name1, object_name2)
            self.__relationships.add(key)
            if key not in self.__image_number_relationships:
                self.__image_number_relationships[key] = \
                    self.init_image_number_relationships(rgroup)
            else:
                d = self.__image_number_relationships[key]
                for image_numbers in (image_numbers1, image_numbers2):
                    self.update_image_number_relationships(
                        image_numbers, current_size, d)

    def get_relationship_groups(self):
        '''Return the keys of each of the relationship groupings.

        The value returned is a list composed of objects with the following
        attributes:
        module_number - the module number of the module used to generate the relationship
        relationship - the relationship of the two objects
        object_name1 - the object name of the first object in the relationship
        object_name2 - the object name of the second object in the relationship
        '''

        return [RelationshipKey(module_number, relationship, obj1, obj2) for
                (module_number, relationship, obj1, obj2) in self.__relationships]

    def get_relationships(self, module_number, relationship, 
                          object_name1, object_name2,
                          image_numbers = None):
        '''Get the relationships recorded by a particular module
        
        module_number - # of module recording the relationship
        
        relationship - the name of the relationship, e.g. "Parent" for
                       object # 1 is parent of object # 2
                       
        object_name1, object_name2 - the names of the two objects
        
        image_numbers - if defined, only return relationships with first or
                        second objects in these image numbers.
        
        returns a recarray with the following fields:
        R_FIRST_IMAGE_NUMBER, R_SECOND_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER,
        R_SECOND_OBJECT_NUMBER
        '''
        features = (R_FIRST_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER,
                            R_SECOND_IMAGE_NUMBER, R_SECOND_OBJECT_NUMBER)
        dt = np.dtype([(feature, np.int32, 1) for feature in features])
        if not (module_number, relationship, object_name1, object_name2) \
           in self.__relationships:
            return np.zeros(0, dt).view(np.recarray)
        with self.hdf5_dict.lock:
            grp = self.get_relationship_hdf5_group(
                module_number, relationship, object_name1, object_name2)
            n_records = grp[R_FIRST_IMAGE_NUMBER].shape[0]
            if n_records == 0:
                return np.zeros(0, dt).view(np.recarray)
            if image_numbers is None:
                temp = np.zeros(n_records, dt)
                for feature in features:
                    temp[feature] = grp[feature]
            else:
                image_numbers = np.atleast_1d(image_numbers)
                k = (module_number, relationship, object_name1, object_name2)
                d = self.__image_number_relationships.get(k, None)
                if d is None:
                    d = self.__image_number_relationships[k] = \
                        self.init_image_number_relationships(grp)
                #
                # Find the slice of the hdf5 array that contains all records
                # for the desired image numbers
                #
                t_min = sys.maxint
                t_max = 0
                for image_number in image_numbers:
                    i_min, i_max = d.get(image_number, (t_min, t_max-1))
                    t_min = min(i_min, t_min)
                    t_max = max(i_max+1, t_max)
                if t_min >= t_max:
                    return np.zeros(0, dt).view(np.recarray)
                #
                # Construct a mask, offset by the minimum index to be addressed
                # of the image numbers to keep in the slice
                #
                in_min = np.min(image_numbers)
                in_max = np.max(image_numbers)
                to_keep = np.zeros(in_max - in_min + 1, bool)
                to_keep[image_numbers - in_min] = True
                mask = np.zeros(t_max - t_min, bool)
                for a in grp[R_FIRST_IMAGE_NUMBER][t_min:t_max],\
                    grp[R_SECOND_IMAGE_NUMBER][t_min:t_max]:
                    m1 = (a >= in_min) & (a <= in_max)
                    mask[m1] = mask[m1] | to_keep[a[m1] - in_min]
                #
                # Apply the mask to slices for all of the features
                #
                n_records = np.sum(mask)
                temp = np.zeros(n_records, dt)
                for feature in features:
                    temp[feature] = grp[feature][t_min:t_max][mask]
            return temp.view(np.recarray)
    
    @staticmethod    
    def init_image_number_relationships(grp):
        '''Create a dictionary of where to find image numbers in a relationship
        
        grp - the HDF5 group of the relationship
        
        returns a dictionary whose key is image number and whose value
        is a pair of the minimum and maximum position in the array of that
        image number.
        '''
        d = {}
        chunk_size = 1000000
        for imgnums in (grp[R_FIRST_IMAGE_NUMBER], 
                        grp[R_SECOND_IMAGE_NUMBER]):
            for i in range(0, imgnums.shape[0], chunk_size):
                limit = min(imgnums.shape[0], i+chunk_size)
                Measurements.update_image_number_relationships(
                    imgnums[i:limit], i, d)
        return d
    
    @staticmethod
    def update_image_number_relationships(imgnums, offset, d):
        '''Update an image number indexing dictionary with new image numbers
        
        imgnums - a vector of image numbers
        
        offset - the offset of this chunk within the relationships records
        
        d - the dictionary to update
        '''
    
        offsets = offset+np.arange(len(imgnums))
        order = np.lexsort((offsets, imgnums))
        imgnums = imgnums[order]
        offsets = offsets[order]
        firsts = np.hstack(([True], imgnums[:-1] != imgnums[1:]))
        lasts = np.hstack((firsts[1:], [True]))
        for i, f, l in zip(
            imgnums[firsts], offsets[firsts], offsets[lasts]):
            old_f, old_l = d.get(i, (sys.maxint, 0))
            d[i] = (min(old_f, f), max(old_l, l))
        
    def copy_relationships(self, src):
        '''Copy the relationships from another measurements file
        
        src - a Measurements possibly having relationships.
        '''
        for rk in src.get_relationship_groups():
            r = src.get_relationships(
                rk.module_number, rk.relationship, 
                rk.object_name1, rk.object_name2)
            self.add_relate_measurement(
                rk.module_number, rk.relationship, 
                rk.object_name1, rk.object_name2,
                r[R_FIRST_IMAGE_NUMBER], r[R_FIRST_OBJECT_NUMBER],
                r[R_SECOND_IMAGE_NUMBER], r[R_SECOND_OBJECT_NUMBER])

    def add_measurement(self, object_name, feature_name, data, 
                        can_overwrite=False, image_set_number=None,
                        data_type =  None):
        """Add a measurement or, for objects, an array of measurements to the set

        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        can_overwrite - legacy / ignored
        image_set_number - write the measurement to this image set or if
                           a sequence of image sets, write the sequence of
                           data values to the sequence of image sets
        data_type - an explicit data type to use when storing the measurements.
        """
        if image_set_number is None:
            image_set_number = self.image_set_number

        # some code adds ImageNumber and ObjectNumber measurements explicitly
        if feature_name in (IMAGE_NUMBER, OBJECT_NUMBER):
            return

        if object_name == EXPERIMENT:
            if not np.isscalar(data) and data is not None and data_type is None:
                data = data[0]
            if data is None:
                data = []
            self.hdf5_dict[EXPERIMENT, feature_name, 0, data_type] = \
                Measurements.wrap_string(data)
        elif object_name == IMAGE:
            if np.isscalar(image_set_number):
                image_set_number = [image_set_number]
                data = [data]
            data = [d if d is None or d is np.NaN
                    else Measurements.wrap_string(d) if np.isscalar(d)
                    else Measurements.wrap_string(d[0]) if data_type is None
                    else d
                    for d in data]
            self.hdf5_dict[IMAGE, feature_name, image_set_number, data_type] = data
            for n in image_set_number:
                if not self.hdf5_dict.has_data(object_name, IMAGE_NUMBER, n):
                    self.hdf5_dict[IMAGE, IMAGE_NUMBER, n] = n
        else:
            self.hdf5_dict[
                object_name, feature_name, image_set_number, data_type] = data
            for n, d in (((image_set_number,data), ) if np.isscalar(image_set_number)
                         else zip(image_set_number, data)):
                if not self.hdf5_dict.has_data(IMAGE, IMAGE_NUMBER, n):
                    self.hdf5_dict[IMAGE, IMAGE_NUMBER, n] = n
                if ((not self.hdf5_dict.has_data(
                    object_name, OBJECT_NUMBER, n)) and 
                    (d is not None)):
                    self.hdf5_dict[object_name, IMAGE_NUMBER, n] = [n] * len(d)
                self.hdf5_dict[object_name, OBJECT_NUMBER, n] =\
                    np.arange(1, len(d) + 1)
                
    def remove_measurement(self, object_name, feature_name, image_number=None):
        '''Remove the measurement for the given image number
        
        object_name - the measurement's object. If other than Image or Experiment,
                      will remove measurements for all objects
        feature_name - name of the measurement feature
        image_number - the image set's image number
        '''
        if image_number is None:
            del self.hdf5_dict[object_name, feature_name]
        else:
            del self.hdf5_dict[object_name, feature_name, image_number]
        
    def clear(self):
        '''Remove all measurements'''
        self.hdf5_dict.clear()

    def get_object_names(self):
        """The list of object names (including Image) that have measurements
        """
        return [x for x in self.hdf5_dict.top_level_names()
                if x != RELATIONSHIP]

    object_names = property(get_object_names)

    def get_feature_names(self, object_name):
        """The list of feature names (measurements) for an object
        """
        return [name for name in self.hdf5_dict.second_level_names(object_name) if name not in ('ImageNumber', 'ObjectNumber')]
    
    def get_image_numbers(self):
        '''Return the image numbers from the Image table'''
        image_numbers = np.array(
            self.hdf5_dict.get_indices(IMAGE, IMAGE_NUMBER).keys(), int)
        image_numbers.sort()
        return image_numbers
    
    def reorder_image_measurements(self, new_image_numbers):
        '''Assign all image measurements to new image numbers
        
        new_image_numbers - a zero-based array that maps old image number
                            to new image number, e.g. if 
                            new_image_numbers = [ 0, 3, 1, 2], then
                            the measurements for old image number 1 will
                            be the measurements for new image number 3, etc.
                            
        Note that this does not handle any image numbers that might be stored
        in the measurements themselves. It is intended for use in
        prepare_run when it is necessary to reorder image numbers because
        of regrouping.
        '''
        for feature in self.get_feature_names(IMAGE):
            self.hdf5_dict.reorder(IMAGE, feature, new_image_numbers)

    def has_feature(self, object_name, feature_name):
        return self.hdf5_dict.has_feature(object_name, feature_name)

    def get_current_image_measurement(self, feature_name):
        '''Return the value for the named image measurement

        feature_name - the name of the measurement feature to be returned
        '''
        return self.get_current_measurement(IMAGE, feature_name)

    def get_current_measurement(self, object_name, feature_name):
        """Return the value for the named measurement for the current image set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned
        """
        return self.get_measurement(object_name, feature_name, self.image_set_number)

    @staticmethod
    def wrap_string(v):
        if isinstance(v, basestring):
            if getattr(v, "__class__") == str:
                v = v.decode("utf-8")
            return v.encode('unicode_escape')
        return v

        
    @staticmethod
    def unwrap_string(v):
        # hdf5 returns string columns as a wrapped type
        # Strings are (sometimes?) returned as numpy.object_ and bizarrely,
        # type(v) == numpy.object_, but v.__class__==str. Additionally,
        # builtin type like number has a __class__ attribute but that can't be
        # referenced with the dot syntax.
        #
        if getattr(v, "__class__") == str:
            return v.decode('unicode_escape')
        return v
    
    def get_measurement(self, object_name, feature_name, image_set_number=None):
        """Return the value for the named measurement and indicated image set
        
        object_name - the name of one of the objects or one of the generic
                      names such as Image or Experiment
                      
        feature_name - the name of the feature to retrieve 
        
        image_set_number - the current image set by default, a single 
                           image set number to get measurements for one
                           image set or a sequence of image numbers to
                           return measurements for each of the image sets
                           listed.
        """
        if object_name == EXPERIMENT:
            result = self.hdf5_dict[EXPERIMENT, feature_name, 0]
            if len(result) == 1:
                result = result[0]
            return Measurements.unwrap_string(result)
        if image_set_number is None:
            image_set_number = self.image_set_number
        vals = self.hdf5_dict[object_name, feature_name, image_set_number]
        if object_name == IMAGE:
            if np.isscalar(image_set_number):
                if vals is None or len(vals) == 0:
                    return None
                if len(vals) == 1:
                    return Measurements.unwrap_string(vals[0]) 
                return vals
            else:
                measurement_dtype = self.hdf5_dict.get_feature_dtype(
                    object_name, feature_name)
                if h5py.check_dtype(vlen = measurement_dtype) == str:
                    result = [ Measurements.unwrap_string(v[0]) 
                               if v is not None else None
                               for v in vals]
                elif measurement_dtype == np.uint8:
                    #
                    # Blobs - just pass them through as an array.
                    #
                    result = vals
                else:
                    # numeric expect as numpy array, text as list (or possibly
                    # array of object in order to handle np.NaN
                    #
                    # A missing result is assumed to be "unable to calculate
                    # in this case and we substitute NaN for it.
                    #
                    result = np.array(
                        [np.NaN if v is None or len(v) == 0
                         else v[0] if len(v) == 1
                         else v for v in vals])
                return result
        if np.isscalar(image_set_number):
            return np.array([]) if vals is None else vals.flatten()
        return [np.array([]) if v is None else v.flatten() for v in vals]
    
    def get_measurement_columns(self):
        '''Return the measurement columns for the current measurements
        
        This returns the measurement columns in the style of
        pipeline.get_measurement_columns. It can be used for cases where
        the measurements are loaded from a file and do not reflect
        current module functionality.
        
        Note that this doesn't correctly differentiate string data and blob
        data.
        '''
        result = []
        for object_name in self.get_object_names():
            for feature_name in self.get_feature_names(object_name):
                dtype = self.hdf5_dict.get_feature_dtype(object_name, feature_name)
                if dtype.kind in ['O', 'S', 'U']:
                    result.append((object_name, feature_name, COLTYPE_VARCHAR))
                elif np.issubdtype(dtype, float):
                    result.append((object_name, feature_name, COLTYPE_FLOAT))
                else:
                    result.append((object_name, feature_name, COLTYPE_INTEGER))
        return result

    def has_measurements(self, object_name, feature_name, image_set_number):
        if object_name == EXPERIMENT:
            return self.hdf5_dict.has_data(EXPERIMENT, feature_name, 0)
        return self.hdf5_dict.has_data(object_name, feature_name, image_set_number)

    def has_current_measurements(self, object_name, feature_name):
        return self.has_measurements(object_name, feature_name, self.image_set_number)

    def get_all_measurements(self, object_name, feature_name):
        warnings.warn("get_all_measurements is deprecated. Please use "
                      "get_measurements with an array of image numbers instead",
                      DeprecationWarning)
        return self.get_measurement(object_name, feature_name,
                                    self.get_image_numbers())

    def add_all_measurements(self, object_name, feature_name, values,
                             data_type = None):
        '''Add a list of measurements for all image sets

        object_name - name of object or Images
        feature_name - feature to add
        values - list of either values or arrays of values
        '''
        values = [[] if value is None
                  else [Measurements.wrap_string(value)] if np.isscalar(value)
                  else value
                  for value in values]
        if ((not self.hdf5_dict.has_feature(IMAGE, IMAGE_NUMBER)) or
            (np.max(self.get_image_numbers()) < len(values))):
            image_numbers = np.arange(1, len(values)+1)
            self.hdf5_dict.add_all(
                IMAGE, IMAGE_NUMBER, image_numbers)
        else:
            image_numbers = self.get_image_numbers()
        self.hdf5_dict.add_all(object_name, feature_name, values, 
                               image_numbers, data_type=data_type)

    def get_experiment_measurement(self, feature_name):
        """Retrieve an experiment-wide measurement
        """
        result = self.get_measurement(EXPERIMENT, feature_name)
        return 'N/A' if result is None else result
    
    def apply_metadata(self, pattern, image_set_number=None):
        """Apply metadata from the current measurements to a pattern

        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        image_name - name of image associated with the metadata (or None
                     if metadata is not associated with an image)
        image_set_number - # of image set to use to retrieve data.
                           None for current.
        
        C_SERIES and C_FRAME are special cases. We look at the series/frame
        values for all images in the image set and take the one that's the
        highest - assuming that there may be a stack + a single image like
        one used for background subtraction. Admittedly a heuristic, but
        there it is.
        
        returns a string with the metadata tags replaced by the metadata
        """
        if image_set_number == None:
            image_set_number = self.image_set_number
        result_pieces = []
        double_backquote = "\\\\"
        single_backquote = "\\"
        for piece in pattern.split(double_backquote):
            # Replace tags in piece
            result = ''
            while(True):
                # Replace one tag
                m = re.search('\\(\\?[<](.+?)[>]\\)', piece)
                if not m:
                    m = re.search('\\\\g[<](.+?)[>]', piece)
                    if not m:
                        result += piece
                        break
                result += piece[:m.start()]
                feature = m.groups()[0]
                if feature in (C_SERIES, C_FRAME):
                    max_value = 0
                    for mname in self.get_feature_names(IMAGE):
                        if mname.startswith(feature + "_"):
                            value = self[IMAGE, mname, image_set_number]
                            if value > max_value:
                                max_value = value
                    result += str(max_value)
                else:
                    measurement = '%s_%s' % (C_METADATA, feature)
                    result += str(self.get_measurement("Image", measurement,
                                                       image_set_number))
                piece = piece[m.end():]
            result_pieces.append(result)
        return single_backquote.join(result_pieces)

    def has_groups(self):
        '''Return True if there is more than one group in the image sets
        
        Note - this works the dumb way now: it fetches all of the group numbers
               and sees if there is a single unique group number. It involves
               fetching the whole column and it doesn't cache, so it could
               be expensive. Alternatively, this could be an experiment
               measurement, populated after prepare_run.
        '''
        if self.has_feature(IMAGE, GROUP_NUMBER):
            image_numbers = self.get_image_numbers()
            if len(image_numbers) > 0:
                group_numbers = self.get_measurement(
                    IMAGE, GROUP_NUMBER,
                    image_set_number = image_numbers)
                return len(np.unique(group_numbers)) > 1
        return False

    def group_by_metadata(self, tags):
        """Return groupings of image sets with matching metadata tags

        tags - a sequence of tags to match.

        Returns a sequence of MetadataGroup objects. Each one represents
        a set of values for the metadata tags along with the image numbers of
        the image sets that match the values
        """
        if len(tags) == 0:
            # if there are no tags, all image sets match each other
            return [MetadataGroup({}, self.get_image_numbers())]

        #
        # The flat_dictionary has a row of tag values as a key
        #
        flat_dictionary = {}
        image_numbers = self.get_image_numbers()
        values = [self.get_measurement(
            IMAGE, "%s_%s" % (C_METADATA, tag), image_numbers)
                  for tag in tags]
        for i, image_number in enumerate(image_numbers):
            key = tuple([(k, v[i]) for k, v in zip(tags, values)])
            if not flat_dictionary.has_key(key):
                flat_dictionary[key] = []
            flat_dictionary[key].append(image_number)
        result = []
        for row in flat_dictionary.keys():
            tag_dictionary = dict(row)
            result.append(MetadataGroup(tag_dictionary, flat_dictionary[row]))
        return result
    
    def match_metadata(self, features, values):
        '''Match vectors of metadata values to existing measurements
        
        This method finds the image sets that match each row in a vector
        of metadata values. Imagine being given an image set with metadata
        values of plate, well and site and annotations for each well
        with metadata values of plate and well and annotation. You'd like
        to match each annotation with all of the sites for it's well. This
        method will return the image numbers that match.
        
        The method can also be used to match images, for instance when
        different illumination correction functions need to be matched
        against plates or sites.
        
        features - the measurement names for the incoming metadata
        
        values - a sequence of vectors, one per feature, giving the
                 metadata values to be matched.

        returns a sequence of vectors of image numbers of equal length
        to the values. An exception is thrown if the metadata for more
        than one row in the values matches the same image set unless the number
        of values in each vector equals the number of image sets - in that case,
        the vectors are assumed to be arranged in the correct order already.
        '''
        #
        # Get image features populated by previous modules. If there are any,
        # then we launch the desperate heuristics that attempt to match
        # to them, either by order or by common metadata
        #
        image_set_count = len(self.get_image_numbers())
        by_order = [[i+1] for i in range(len(values[0]))]
        if image_set_count == 0:
            return by_order
        
        image_features = self.get_feature_names(IMAGE)
        metadata_features = [x for x in image_features
                             if x.startswith(C_METADATA + "_")]
        common_features = [x for x in metadata_features
                           if x in features]
        if len(common_features) == 0:
            if image_set_count > len(values[0]):
                raise ValueError(
                    "The measurements and data have no metadata in common")
            return by_order
        #
        # This reduces numberlike things to integers so that they can be
        # more loosely matched.
        #
        def cast(x):
            if isinstance(x,basestring) and x.isdigit():
                return int(x)
            return x
        
        common_tags = [f[(len(C_METADATA)+1):] for f in common_features]
        groupings = self.group_by_metadata(common_tags)
        groupings = dict([(tuple([cast(d[f]) for f in common_tags]), 
                           d.image_numbers)
                          for d in groupings])
        if image_set_count == len(values[0]):
            #
            # Test whether the common features uniquely identify
            # all image sets. If so, then we can match by metadata
            # and that will be correct, even when the user wants to
            # match by order (assuming the user really did match
            # the metadata)
            #
            if any([len(v) != 1 for v in groupings.values()]):
                return by_order
        #
        # Create a list of values that matches the common_features
        #
        result = []
        vv = [values[features.index(c)] for c in common_features]
        for i in range(len(values[0])):
            key = tuple([cast(vvv[i]) for vvv in vv])
            if not groupings.has_key(key):
                raise ValueError(
                    ("There was no image set whose metadata matched row %d.\n" % (i+1)) +
                    "Metadata values: " +
                    ", ".join(["%s = %s" % (k, v)
                               for k,v in zip(common_features, key)]))
            result.append(groupings[key])
        return result
        
    def agg_ignore_object(self, object_name):
        """Ignore objects (other than 'Image') if this returns true"""
        if object_name in (EXPERIMENT, NEIGHBORS):
            return True

    def agg_ignore_feature(self, object_name, feature_name):
        """Return true if we should ignore a feature during aggregation"""

        if self.agg_ignore_object(object_name):
            return True
        if self.hdf5_dict.has_feature(object_name, "SubObjectFlag"):
            return True
        return agg_ignore_feature(feature_name)

    def compute_aggregate_measurements(self, image_set_number,
                                       aggs=AGG_NAMES):
        """Compute aggregate measurements for a given image set

        returns a dictionary whose key is the aggregate measurement name and
        whose value is the aggregate measurement value
        """
        d = {}
        if len(aggs) == 0:
            return d
        for object_name in self.get_object_names():
            if object_name == 'Image':
                continue
            for feature in self.get_feature_names(object_name):
                if self.agg_ignore_feature(object_name, feature):
                    continue
                feature_name = "%s_%s" % (object_name, feature)
                values = self.get_measurement(object_name, feature,
                                              image_set_number)
                if values is not None:
                    values = values[np.isfinite(values)]
                #
                # Compute the mean and standard deviation
                #
                if AGG_MEAN in aggs:
                    mean_feature_name = get_agg_measurement_name(
                        AGG_MEAN, object_name, feature)
                    mean = values.mean() if values is not None else np.NaN
                    d[mean_feature_name] = mean
                if AGG_MEDIAN in aggs:
                    median_feature_name = get_agg_measurement_name(
                        AGG_MEDIAN, object_name, feature)
                    median = np.median(values) if values is not None else np.NaN
                    d[median_feature_name] = median
                if AGG_STD_DEV in aggs:
                    stdev_feature_name = get_agg_measurement_name(
                        AGG_STD_DEV, object_name, feature)
                    stdev = values.std() if values is not None else np.NaN
                    d[stdev_feature_name] = stdev
        return d
    
    def load_image_sets(self, fd_or_file, start=None, stop=None):
        '''Load image sets from a .csv file into a measurements file
        
        fd_or_file - either the path name of the .csv file or a file-like object
        
        start - the 1-based image set number to start the loading. For instance,
                for start = 2, we skip the first line and write image
                measurements starting at line 2 into image set # 2
                
        stop - stop loading when this line is reached.
        '''
        if isinstance(fd_or_file, basestring):
            with open(fd_or_file, "r") as fd:
                return self.load_image_sets(fd, start, stop)
        import csv
        reader = csv.reader(fd_or_file)
        header = [x.decode('utf-8') for x in reader.next()]
        columns = [[] for _ in range(len(header))]
        column_is_all_none = np.ones(len(header), bool)
        last_image_number = 0
        for i, fields in enumerate(reader):
            fields = [x.decode('utf-8') for x in fields]
            image_number = i + 1
            if start is not None and start < image_number:
                continue
            if stop is not None and image_number == stop:
                break
            for j, (field, column) in enumerate(zip(fields, columns)):
                if field == "None" or len(field) == 0:
                    field = None
                else:
                    column_is_all_none[j] = False
                column.append(field)
            last_image_number = image_number
        if last_image_number == 0:
            logger.warn("No image sets were loaded")
            return
        if start is None:
            image_numbers = list(range(1, last_image_number + 1))
        else:
            image_numbers = list(range(start, last_image_number + 1))
        self.hdf5_dict.add_all(IMAGE, IMAGE_NUMBER, image_numbers, image_numbers)
        for feature, column, all_none in zip(header, columns, column_is_all_none):
            if not all_none:
                # try to convert to an integer, then float, then leave as string
                column = np.array(column, object)
                try:
                    column = column.astype(int)
                except:
                    try:
                        column = column.astype(float)
                    except:
                        column = np.array(
                            [Measurements.wrap_string(x) for x in column], 
                            object)
                self.hdf5_dict.add_all(IMAGE, feature, column, image_numbers)
                
    def write_image_sets(self, fd_or_file, start = None, stop = None):
        if isinstance(fd_or_file, basestring):
            with open(fd_or_file, "w") as fd:
                return self.write_image_sets(fd, start, stop)
        
        fd = fd_or_file
        
        to_save = [ GROUP_NUMBER, GROUP_INDEX]
        to_save_prefixes = [
            C_URL, C_PATH_NAME, C_FILE_NAME, C_SERIES, C_FRAME,
            C_CHANNEL, C_OBJECTS_URL, C_OBJECTS_PATH_NAME,
            C_OBJECTS_FILE_NAME, C_OBJECTS_SERIES, C_OBJECTS_FRAME,
            C_OBJECTS_CHANNEL, C_METADATA]
        
        keys = []
        image_features = self.get_feature_names(IMAGE)
        for feature in to_save:
            if feature in image_features:
                keys.append(feature)
        for prefix in to_save_prefixes:
            for feature in image_features:
                if feature.startswith(prefix) and feature not in keys:
                    keys.append(feature)
        header = "\""+"\",\"".join(keys) + "\"\n"
        fd.write(header)
        image_numbers = self.get_image_numbers()
        if start is not None:
            image_numbers = [x for x in image_numbers if x >= start]
        if stop is not None:
            image_numbers = [x for x in image_numbers if x <= stop]
            
        if len(image_numbers) == 0:
            return
        
        columns = [self.get_measurement(IMAGE, feature_name, 
                                        image_set_number = image_numbers)
                   for feature_name in keys]
        for i, image_number in enumerate(image_numbers):
            for j, column in enumerate(columns):
                field = column[i]
                if field is np.NaN or field is None:
                    field = ""
                if isinstance(field, basestring):
                    if isinstance(field, unicode):
                        field = field.encode("utf-8")
                    field = "\"" + field.replace("\"", "\"\"") + "\""
                else:
                    field = str(field)
                if j > 0:
                    fd.write(","+field)
                else:
                    fd.write(field)
            fd.write("\n")
            
    def alter_path_for_create_batch(self, name, is_image, fn_alter_path):
        '''Alter the path of image location measurements for CreateBatchFiles
        
        name - name of the image or objects
        is_image - True to load as an image, False to load as objects
        fn_later_path - call this function to alter the path for batch processing
        '''
        from cellprofiler.modules.loadimages import url2pathname, pathname2url
        if is_image:
            path_feature = C_PATH_NAME
            file_feature = C_FILE_NAME
            url_feature = C_URL
        else:
            path_feature = C_OBJECTS_PATH_NAME
            file_feature = C_OBJECTS_FILE_NAME
            url_feature = C_OBJECTS_URL
        path_feature, file_feature, url_feature = [
            "_".join((f, name))
            for f in (path_feature, file_feature, url_feature)]
        
        all_image_numbers = self.get_image_numbers()
        urls = self.get_measurement(IMAGE, url_feature, 
                                    image_set_number=all_image_numbers)

        new_urls = []
        for url in urls:
            if url.lower().startswith("file:"):
                full_name = url2pathname(url.encode("utf-8"))
                full_name = fn_alter_path(full_name)
                new_url = pathname2url(full_name)
            else:
                new_url = url
            new_urls.append(new_url)
        if any([url != new_url for url, new_url in zip(urls, new_urls)]):
            self.add_all_measurements(IMAGE, url_feature, new_urls)
            
        paths = self.get_measurement(IMAGE, path_feature,
                                     image_set_number = all_image_numbers)
        new_paths = [fn_alter_path(path) for path in paths]
        if any([path != new_path for path, new_path in zip(paths, new_paths)]):
            self.add_all_measurements(IMAGE, path_feature, new_paths)
            
        filenames = self.get_measurement(IMAGE, file_feature,
                                         image_set_number = all_image_numbers)
        new_filenames = [fn_alter_path(filename) for filename in filenames]
        if any([filename != new_filename
                for filename, new_filename in zip(filenames, new_filenames)]):
            self.add_all_measurements(IMAGE, file_feature, new_filenames)
    
    def write_path_mappings(self, mappings):
        '''Write the mappings of local/remote dirs as an experiment measurement
        
        This records the mappings of local and remote directories entered
        by the CreateBatchFiles module.
        
        mappings - a sequence of two-tuples. The first tuple is the local
                   path and the second is the remote path (on the target
                   machine for the run)
        '''
        d = {
            K_CASE_SENSITIVE: (os.path.normcase("A") != os.path.normcase("a")),
            K_LOCAL_SEPARATOR: os.path.sep,
            K_PATH_MAPPINGS: tuple([tuple(m) for m in mappings]),
            K_URL2PATHNAME_PACKAGE_NAME: urllib.url2pathname.__module__
            }
        s = json.dumps(d)
        self.add_experiment_measurement(M_PATH_MAPPINGS, s)
        
    def alter_url_post_create_batch(self, url):
        '''Apply CreateBatchFiles path mappings to an unmapped URL
        
        This method can be run on the measurements output by CreateBatchFiles
        to map the paths of any URL that wasn't mapped by the alter-paths
        mechanism (e.g. URLs encoded in blobs)
        
        url - the url to map
        
        returns - a possibly mapped URL
        '''
        if not url.lower().startswith("file:"):
            return url
        if not self.has_feature(EXPERIMENT, M_PATH_MAPPINGS):
            return url
        d = json.loads(self.get_experiment_measurement(M_PATH_MAPPINGS))
        os_url2pathname = __import__(d[K_URL2PATHNAME_PACKAGE_NAME]).url2pathname
        full_name = os_url2pathname(url[5:].encode("utf-8"))
        full_name_c = full_name if d[K_CASE_SENSITIVE] else full_name.lower()
        if d[K_LOCAL_SEPARATOR] != os.path.sep:
            full_name = full_name.replace(d[K_LOCAL_SEPARATOR], os.path.sep)
        for local_directory, remote_directory in d[K_PATH_MAPPINGS]:
            if d[K_CASE_SENSITIVE]:
                if full_name_c.startswith(local_directory):
                    full_name = \
                        remote_directory + full_name[len(local_directory):]
            else:
                if full_name_c.startswith(local_directory.lower()):
                    full_name = \
                        remote_directory + full_name[len(local_directory):]
        url = "file:" + urllib.pathname2url(full_name)
        if isinstance(url, unicode):
            url = url.encode("utf-8")
        return url
        
    ###########################################################
    #
    # Ducktyping measurements as image sets
    #
    ###########################################################
    
    @property
    def image_number(self):
        '''The image number of the current image'''
        return self.image_set_number
    
    @property
    def get_keys(self):
        '''The keys that uniquely identify the image set
        
        Return key/value pairs for the metadata that specifies the site
        for the image set, for instance, plate / well / site. If image set
        was created by matching images by order, the image number will be
        returned.
        '''
        #
        # XXX (leek) - save the metadata tags used for matching in the HDF
        #              then use it to look up the values per image set
        #              and cache.
        #
        return { IMAGE_NUMBER: str(self.image_number) }
    
    def get_grouping_keys(self):
        '''Get a key, value dictionary that uniquely defines the group
        
        returns a dictionary for the current image set's group where the
        key is the image feature name and the value is the value to match
        in the image measurements.
        
        Note: this is somewhat legacy, from before GROUP_NUMBER was defined
              and the only way to determine which images were in a group
              was to get the metadata colums used to define groups and scan
              them for matches. Now, we just return { GROUP_NUMBER: value }
        '''
        return { GROUP_NUMBER: 
                 self.get_current_image_measurement(GROUP_NUMBER) }
    
    def get_image(self, name, 
                  must_be_binary = False,
                  must_be_color = False,
                  must_be_grayscale = False,
                  must_be_rgb = False,
                  cache = True):
        """Return the image associated with the given name
        
        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        must_be_rgb - raise an exception if 2-d or if # channels not 3 or 4,
                      discard alpha channel.
        """
        from .modules.loadimages import LoadImagesImageProviderURL
        from .cpimage import GrayscaleImage, RGBImage
        name = str(name)
        if self.__images.has_key(name):
            image  = self.__images[name]
        else:
            matching_providers = [p for p in self.__image_providers
                                  if p.get_name() == name]
            if len(matching_providers) == 0:
                #
                # Try looking up the URL in measurements
                #
                url_feature_name = "_".join((C_URL, name))
                series_feature_name = "_".join((C_SERIES, name))
                index_feature_name = "_".join((C_FRAME, name))
                if not self.has_feature(IMAGE, url_feature_name):
                    raise ValueError("The %s image is missing from the pipeline."%(name))
                # URL should be ASCII only
                url = str(self.get_current_image_measurement(url_feature_name))
                if self.has_feature(IMAGE, series_feature_name):
                    series = self.get_current_image_measurement(
                        series_feature_name)
                else:
                    series = None
                if self.has_feature(IMAGE, index_feature_name):
                    index = self.get_current_image_measurement(
                        index_feature_name)
                else:
                    index = None
                #
                # XXX (leek): Rescale needs to be bubbled up into 
                #             NamesAndTypes and needs to be harvested
                #             from LoadImages etc.
                #             and stored in the measurements.
                #
                rescale = True
                provider = LoadImagesImageProviderURL(
                    name, url, rescale, series, index)
                self.__image_providers.append(provider)
                matching_providers.append(provider)
            image = matching_providers[0].provide_image(self)
            if cache:
                self.__images[name] = image
        if must_be_binary and image.pixel_data.ndim == 3:
            raise ValueError("Image must be binary, but it was color")
        if must_be_binary and image.pixel_data.dtype != np.bool:
            raise ValueError("Image was not binary")
        if must_be_color and image.pixel_data.ndim != 3:
            raise ValueError("Image must be color, but it was grayscale")
        if (must_be_grayscale and 
            (image.pixel_data.ndim != 2)):
            pd = image.pixel_data
            if pd.shape[2] >= 3 and\
               np.all(pd[:,:,0]==pd[:,:,1]) and\
               np.all(pd[:,:,0]==pd[:,:,2]):
                return GrayscaleImage(image)
            raise ValueError("Image must be grayscale, but it was color")
        if must_be_grayscale and image.pixel_data.dtype.kind == 'b':
            return GrayscaleImage(image)
        if must_be_rgb:
            if image.pixel_data.ndim != 3:
                raise ValueError("Image must be RGB, but it was grayscale")
            elif image.pixel_data.shape[2] not in (3,4):
                raise ValueError("Image must be RGB, but it had %d channels" %
                                 image.pixel_data.shape[2])
            elif image.pixel_data.shape[2] == 4:
                logger.warning("Discarding alpha channel.")
                return RGBImage(image)
        return image
    
    def get_providers(self):
        """The list of providers (populated during the image discovery phase)"""
        return self.__image_providers
    
    providers = property(get_providers)
    
    def get_image_provider(self, name):
        """Get a named image provider
        
        name - return the image provider with this name
        """
        providers = filter(lambda x: x.name == name, self.__image_providers)
        assert len(providers)>0, "No provider of the %s image"%(name)
        assert len(providers)==1, "More than one provider of the %s image"%(name)
        return providers[0]
    
    def remove_image_provider(self, name):
        """Remove a named image provider
        
        name - the name of the provider to remove
        """
        self.__image_providers = filter(lambda x: x.name != name, 
                                        self.__image_providers)
        
    def clear_image(self, name):
        '''Remove the image memory associated with a provider
        
        name - the name of the provider
        '''
        self.get_image_provider(name).release_memory()
        if self.__images.has_key(name):
            del self.__images[name]
            
    def cache(self):
        '''Move all uncached images to an HDF5 backing-store'''
        if self.__image_cache_file is None:
            h, self.__image_cache_path = tempfile.mkstemp(
                suffix=".h5", prefix="CellProfilerImageCache")
            self.__image_cache_file = h5py.File(
                self.__image_cache_path, "w")
            os.close(h)
        for name, image in self.__images.items():
            image.cache(name, self.__image_cache_file)
            
    def clear_cache(self):
        '''Remove all of the cached images'''
        self.__images.clear()
    
    def get_names(self):
        """Get the image provider names
        """
        return [provider.name for provider in self.providers]
    
    names = property(get_names)
    
    def add(self, name, image):
        from .cpimage import VanillaImageProvider
        old_providers = [provider for provider in self.providers
                         if provider.name == name]
        if len(old_providers) > 0:
            self.clear_image(name)
        for provider in old_providers:
            self.providers.remove(provider)
        provider = VanillaImageProvider(name, image)
        self.providers.append(provider)
        self.__images[name] = image
        
    def set_channel_descriptors(self, channel_descriptors):
        '''Write the names and data types of the channel descriptors
        
        channel_descriptors - pipeline channel descriptors describing the
                              channels in the image set.
        '''
        for iscd in channel_descriptors:
            feature = "_".join((C_CHANNEL_TYPE, iscd.name))
            self.add_experiment_measurement(feature, iscd.channel_type)
        
    def get_channel_descriptors(self):
        '''Read the channel descriptors
        
        Returns pipeline.ImageSetChannelDescriptor instances for each
        channel descriptor specified in the experiment measurements.
        '''
        from cellprofiler.pipeline import Pipeline
        ImageSetChannelDescriptor = Pipeline.ImageSetChannelDescriptor
        iscds = []
        for feature_name in self.get_feature_names(EXPERIMENT):
            if feature_name.startswith(C_CHANNEL_TYPE):
                channel_name = feature_name[(len(C_CHANNEL_TYPE)+1):]
                channel_type = self.get_experiment_measurement(feature_name)
                if channel_type == ImageSetChannelDescriptor.CT_OBJECTS:
                    url_feature = "_".join([C_OBJECTS_URL, channel_name])
                else:
                    url_feature = "_".join([C_URL, channel_name])
                if url_feature not in self.get_feature_names(IMAGE):
                    continue
                iscds.append(ImageSetChannelDescriptor(channel_name, channel_type))
        return iscds
    
    def get_channel_descriptor(self, name):
        '''Return the channel descriptor with the given name'''
        for iscd in self.get_channel_descriptors():
            if iscd.name == name:
                return iscd
        return None
    
    def set_metadata_tags(self, metadata_tags):
        '''Write the metadata tags that are used to make an image set
        
        metadata_tags - image feature names of the metadata tags that uniquely
                        define an image set. If metadata matching wasn't used,
                        write the image number feature name.
        '''
        data = json.dumps(metadata_tags)
        self.add_experiment_measurement(M_METADATA_TAGS, data)
        
    def get_metadata_tags(self):
        '''Read the metadata tags that are used to make an image set
        
        returns a list of metadata tags
        '''
        if M_METADATA_TAGS not in self.get_feature_names(EXPERIMENT):
            return [ IMAGE_NUMBER ]
        return json.loads(self.get_experiment_measurement(M_METADATA_TAGS))
    
    def set_grouping_tags(self, grouping_tags):
        '''Write the metadata tags that are used to group an image set
        
        grouping_tags - image feature names of the metadata tags that
                        uniquely define a group.
        '''
        data = json.dumps(grouping_tags)
        self.add_experiment_measurement(M_GROUPING_TAGS, data)
        
    def get_grouping_tags(self):
        '''Get the metadata tags that were used to group the image set
        
        '''
        if not self.has_feature(EXPERIMENT, M_GROUPING_TAGS):
            return self.get_metadata_tags()
        
        return json.loads(self.get_experiment_measurement(M_GROUPING_TAGS))

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

def load_measurements(filename, dest_file = None, can_overwrite = False,
                      run_name = None,
                      image_numbers = None):
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
                   chr (26) + chr(10))
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
            m = Measurements(filename=dest_file, copy = top_level,
                             image_numbers = image_numbers)
            return m
        except:
            logger.error("Error loading HDF5 %s", filename, exc_info=True)
        finally:
            f.close()
    else:
        m = Measurements(filename = dest_file)
        m.load(filename)
        return m

class MetadataGroup(dict):
    """A set of metadata tag values and the image set indexes that match

    The MetadataGroup object represents a group of image sets that
    have the same values for a given set of tags. For instance, if an
    experiment has metadata tags of "Plate", "Well" and "Site" and
    we form a metadata group of "Plate" and "Well", then each metadata
    group will have image set indexes of the images taken of a particular
    well
    """
    def __init__(self, tag_dictionary, image_numbers):
        super(MetadataGroup, self).__init__(tag_dictionary)
        self.__image_numbers = image_numbers

    @property
    def image_numbers(self):
        return self.__image_numbers

    def __setitem__(self, tag, value):
        raise NotImplementedError("The dictionary is read-only")

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

class RelationshipKey:
    def __init__(self, module_number,  relationship,
                 object_name1, object_name2):
        self.module_number = module_number
        self.relationship = relationship
        self.object_name1 = object_name1
        self.object_name2 = object_name2

class ImageSetCache(object):
    '''An ImageSetCache holds the computed results from an image set calculation
    
    The cache remembers the following things:
    
    * Metadata key names used to match images/objects in the image set.
    
    * Image / object names in the image sets.
    
    * Metadata values used in matching.
    
    * URLs, series #, frame #, channel # for each image
    
    * Rows that have errors: the row, image or object name in error.
    
    The image set cache holds the same sort of information as the image
    URL and metadata measurements. It's purpose is to be a more informal
    collection of image set results for cases where the configuration is
    incomplete or might contain errors.
    '''
    # The name of the group hosting the image set cache
    IMAGE_SET_CACHE_GROUP = "ImageSetCache"
    # The version # of the structure.
    IMAGE_SET_CACHE_VERSION = 1
    # Store the version # in the Version attribute of the top-level group
    VERSION_ATTRIBUTE = "Version"
    # The table that holds the image set rows: an HDFCSV
    IMAGE_SET_TABLE = "ImageSetTable"
    # An Nx2 dataset giving the row and column of errors in the image set table
    ERROR_ROW_AND_COLUMN_DATASET = "ErrorRowAndColumn"
    # A descriptive error message for each error in ErrorRowAndColumn
    ERROR_MESSAGES = "ErrorMessages"
    # All metadata key names - an attribute on the image set cache group
    METADATA_KEYS = "MetadataKeys"
    # All image names - an attribute on the image set cache group
    IMAGE_NAMES = "ImageNames"
    # For each image name, either IMAGE or OBJECT depending on whether
    # the name is a name of images or objects
    IMAGE_OR_OBJECT = "ImageOrObject"
    # The row index field in the errors dataset
    ROW_INDEX = "RowIndex"
    # The index of the image name in the ImageNames attribute or the index
    # of the object name in the ObjectNames attribute
    IO_INDEX = "ImageOrObjectIndex"
    # The number of image sets in the cache
    IMAGE_SET_COUNT = "ImageSetCount"
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        assert isinstance(hdf5_file, h5py.Group)
        if self.IMAGE_SET_CACHE_GROUP in hdf5_file:
            self.image_set_cache_group = hdf5_file[self.IMAGE_SET_CACHE_GROUP]
            version = self.image_set_cache_group.attrs[self.VERSION_ATTRIBUTE]
            if version != self.IMAGE_SET_CACHE_VERSION:
                # It's a cache... sorry, if version is not correct, destroy
                del self.hdf5_file[self.IMAGE_SET_CACHE_GROUP]
                self.__has_cache = False
            else:
                self.__has_cache = True
        else:
            self.__has_cache = False
        if self.__has_cache:
            self.image_set_count = \
                self.image_set_cache_group.attrs[self.IMAGE_SET_COUNT]
            self.metadata_keys = self.load_strings_attr(self.METADATA_KEYS)
            if len(self.metadata_keys) == 0:
                self.metadata_keys = None
            self.image_names = self.load_strings_attr(self.IMAGE_NAMES)
            self.image_or_object = self.load_strings_attr(self.IMAGE_OR_OBJECT)
            self.image_set_table = HDFCSV(self.image_set_cache_group,
                                          self.IMAGE_SET_TABLE)
            self.error_row_and_column_dataset = \
                self.image_set_cache_group[self.ERROR_ROW_AND_COLUMN_DATASET]
            self.error_messages = \
                VStringArray(self.image_set_cache_group[self.ERROR_MESSAGES])
        
    def store_strings_attr(self, name, strings):
        '''Store a list of unicode strings in an attribute of the top group
        
        name - name of the attribute
        
        strings - a sequence of strings or unicode strings
        '''
        if len(strings) > 0:
            strings = [ unicode(x).encode("utf-8") for x in strings]
            self.image_set_cache_group.attrs.create(name, strings)
        elif name in self.image_set_cache_group.attrs:
            del self.image_set_cache_group.attrs[name]
        
    def load_strings_attr(self, name):
        '''Load a list of unicode strings from an attribute in the top group
        
        name - name of attribute
        
        returns a sequence of strings if the attribute exists, otherwise
        an empty sequence
        '''
        return [x.decode("utf-8") 
                for x in self.image_set_cache_group.attrs.get(name, [])]
    
    @property
    def has_cache(self):
        '''True if there are image sets in the cache'''
        return self.__has_cache
    
    class ImageSetData(object):
        '''Represents the information in one image set:
        
        key - the unique metadata values that distinguish this image set
        
        ipds - the image plane descriptors of the images in the set.
        
        errors - a sequence of errors, each of which is a two-tuple
                 of the index of the column causing the error and
                 the error message.
        '''
        def __init__(self, key, ipds, errors):
            self.key = key
            self.ipds = ipds
            self.errors = errors
            
    class ImageData(object):
        '''An alternative ducktype to IPDs'''
        def __init__(self, url, series, index, channel):
            self.url = url
            self.series = series
            self.index = index
            self.channel = channel
            
    def cache_image_set(self, 
                        image_names,
                        image_set_data,
                        metadata_keys = None):
        '''Cache the current image set
        
        image_names - a sequence two tuples of the image or object name and
                      either IMAGE or OBJECT to distinguish between the
                      two uses.
        
        image_set_data - a sequence of ImageSetData objects
        
        metadata_keys - the names of the columns that provide the unique keys
                        for each image set. Index by order if None.
        '''
        if not self.__has_cache:
            self.image_set_cache_group = self.hdf5_file.require_group(
                self.IMAGE_SET_CACHE_GROUP)
            self.image_set_cache_group.attrs[self.VERSION_ATTRIBUTE] = \
                self.IMAGE_SET_CACHE_VERSION
        self.image_names = [n for n,io in image_names]
        self.image_or_object = [io for n, io in image_names]
        self.store_strings_attr(self.IMAGE_NAMES, self.image_names)
        self.store_strings_attr(self.IMAGE_OR_OBJECT, self.image_or_object)
        if metadata_keys is None:
            self.store_strings_attr(self.METADATA_KEYS, [])
            self.metadata_keys = None
        else:
            self.store_strings_attr(self.METADATA_KEYS, metadata_keys)
            self.metadata_keys = list(metadata_keys)
        self.image_set_table = HDFCSV(self.image_set_cache_group,
                                      self.IMAGE_SET_TABLE)
        if metadata_keys is not None:
            for i, name in enumerate(metadata_keys):
                self.image_set_table.add_column(
                    name,
                    [isd.key[i] for isd in image_set_data])
        for i, name in enumerate(self.image_names):
            self.image_set_table.add_column(
                C_URL + "_" + name,
                [isd.ipds[i].url for isd in image_set_data])
            for feature, values in (
                (C_SERIES, [isd.ipds[i].series for isd in image_set_data]),
                (C_FRAME, [isd.ipds[i].index for isd in image_set_data]),
                (C_CHANNEL, [isd.ipds[i].channel for isd in image_set_data])):
                self.image_set_table.add_column(
                    feature + "_" + name, 
                    [str(value) if value is not None else "" 
                     for value in values])
        
        errors = sum(
            [[(i, idx, msg) for idx, msg in isd.errors]
             for i, isd in enumerate(image_set_data)], [])
        if self.ERROR_ROW_AND_COLUMN_DATASET in self.image_set_cache_group:
            self.error_row_and_column_dataset = \
                self.image_set_cache_group[self.ERROR_ROW_AND_COLUMN_DATASET]
            self.error_row_and_column_dataset,resize((len(errors),))
        else:
            self.error_row_and_column_dataset = \
                self.image_set_cache_group.create_dataset(
                    self.ERROR_ROW_AND_COLUMN_DATASET,
                    dtype = np.dtype([(self.ROW_INDEX, np.uint32, 1),
                                      (self.IO_INDEX, np.uint8, 1)]),
                    shape = (len(errors), ),
                    chunks = (256, ),
                    maxshape = (None, ))
            self.error_messages = VStringArray(
                self.image_set_cache_group.require_group(self.ERROR_MESSAGES))
                    
        for i, (image_set_row_number, index, msg) in enumerate(errors):
            self.error_row_and_column_dataset[i] = (
                image_set_row_number, index)
            self.error_messages[i] = msg
        self.image_set_count = len(image_set_data)
        self.image_set_cache_group.attrs[self.IMAGE_SET_COUNT] = \
            self.image_set_count        
        self.__has_cache = True
            
    def get_error_rows(self):
        '''Return the indices of rows with errors
        
        Precondition: some image set must be cached.
        '''
        if self.error_row_and_column_dataset.shape[0] == 0:
            return np.zeros(0, int)
        return np.unique(
            self.error_row_and_column_dataset[self.ROW_INDEX])
            
    def get_errors(self, idx):
        '''Get errors in the idx'th row of a data set
        
        idx - the index of the data set that might have an error
        
        returns a list of two tuples of the form:
        image_index - index of the name of the image in self.image_names
        msg - descriptive error message
        '''
        if self.error_row_and_column_dataset.shape[0] == 0:
            return []
        errors = np.where(
            self.error_row_and_column_dataset[self.ROW_INDEX] == idx)
        idxs = self.error_row_and_column_dataset[self.IO_INDEX][:]
        return [ (idxs[i], 
                  self.error_messages[i])
                 for i in errors]
    
    def get_image_set_data(self, idx):
        '''Get an ImageSetData item for the indexed image set'''
        errors = self.get_errors(idx)
        if self.metadata_keys is not None:
            key = tuple([self.image_set_table[k][idx] 
                         for k in self.metadata_keys])
        else:
            key = []
        ipds = []
        for name in self.image_names:
            url = self.image_set_table[C_URL + "_" + name][idx]
            series, index, channel = [
                None if len(value) == 0 else int(value)
                for value in [self.image_set_table[ftr + "_" + name][idx]
                              for ftr in (C_SERIES, C_FRAME, C_CHANNEL)]]
            ipds.append(self.ImageData(url, series, index, channel))
        return self.ImageSetData(key, ipds, errors)
            
            
