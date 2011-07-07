"""Measurements.py - storage for image and object measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
from __future__ import with_statement

__version__ = "$Revision$"

import numpy as np
import re
from scipy.io.matlab import loadmat
from itertools import repeat
import cellprofiler.preferences as cpprefs
from cellprofiler.utilities.hdf5_dict import HDF5Dict
import tempfile
import numpy as np
import warnings
import os
import os.path

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

'''Column attribute: only available after post_group is run (True / False)'''
MCA_AVAILABLE_POST_GROUP  = "AvailablePostGroup"

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
GROUP_NUMBER = "Group_Number"
GROUP_INDEX = "Group_Index"

def get_length_from_varchar(x):
    '''Retrieve the length of a varchar column from its coltype def'''
    m = re.match(r'^varchar\(([0-9]+)\)$', x)
    if m is None:
        return None
    return int(m.groups()[0])

class Measurements(object):
    """Represents measurements made on images and objects
    """
    def __init__(self,
                 can_overwrite=False,
                 image_set_start=None):
        """Create a new measurements collection

        can_overwrite - if True, overwriting measurements during operation
                        is allowed. We turn this on for debugging.
        image_set_start - the index of the first image set in the image set list
                          or None to start at the beginning
        """
        # XXX - clean up on destruction of measurements, allow saving of partial results
        dir = cpprefs.get_default_output_directory()
        if not os.path.exists(dir):
            dir = None
        fd, filename = tempfile.mkstemp(prefix='Cpmeasurements', suffix='.hdf5', dir=dir)
        self.hdf5_dict = HDF5Dict(filename)
        os.close(fd)

        self.image_set_number = image_set_start or 1
        self.image_set_start = image_set_start

        self.__can_overwrite = can_overwrite
        self.__is_first_image = True
        self.__initialized_explicitly = False
        self.__relationships = set()
        self.__relationship_names = set()

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

    def next_image_set(self, explicit_image_set_number=None, erase=False):
        assert explicit_image_set_number is None or explicit_image_set_number > 0
        if explicit_image_set_number is None:
            self.image_set_number += 1
        else:
            self.image_set_number = explicit_image_set_number
        self.__is_first_image = False

        if erase:
            # HDF5 doesn't have an easy way to remove rows, so we
            # just overwrite this imagenumber with 0 in all the
            # tables.
            #
            # The primary use case for overwriting is for testing, so
            # I don't think we need to pack the data, ever. -Ray
            for object_name in self.get_object_names():
                if object_name != EXPERIMENT:
                    self.erase(object_name, self.image_set_number)

    def erase(self, object_name, image_set_number):
        for feature_name in self.get_feature_names(object_name):
            del self.hdf5_dict[object_name, feature_name, image_set_number]
        for feature_name in 'ImageNumber', 'ObjectNumber':
            if self.hdf5_dict.has_data(object_name, feature_name, image_set_number):
                del self.hdf5_dict[object_name, feature_name, image_set_number]

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
            for feature_name in omeas.dtype.fields.keys():
                if object_name == IMAGE:
                    values = [x.flatten()[0] for x in omeas[feature_name][0]]
                elif object_name == EXPERIMENT:
                    value = omeas[feature_name][0, 0].flatten()[0]
                    self.add_experiment_measurement(feature_name, value)
                    continue
                else:
                    values = [x.flatten()
                              for x in omeas[feature_name][0].tolist()]
                self.add_all_measurements(object_name,
                                          feature_name,
                                          values)
        #
        # Set the image set number to beyond the last in the handles
        #
        self.image_set_number = self.image_set_count + 1

    def add_image_measurement(self, feature_name, data, can_overwrite = False):
        """Add a measurement to the "Image" category

        """
        self.add_measurement(IMAGE, feature_name, data, can_overwrite)

    def add_experiment_measurement(self, feature_name, data):
        """Add an experiment measurement to the measurement

        Experiment measurements have one value per experiment
        """
        self.add_measurement(EXPERIMENT, feature_name, data)

    def get_group_number(self):
        '''The number of the group currently being processed'''
        return self.get_current_image_measurement(GROUP_NUMBER)

    def set_group_number(self, group_number, can_overwrite=False):
        self.add_image_measurement(GROUP_NUMBER, group_number, can_overwrite)

    group_number = property(get_group_number, set_group_number)

    def get_group_index(self):
        '''The within-group index of the current image set'''
        return self.get_current_image_measurement(GROUP_INDEX)

    def set_group_index(self, group_index):
        self.add_image_measurement(GROUP_INDEX, group_index)

    group_index = property(get_group_index, set_group_index)

    def add_relate_measurement(
        self, module_number,
        relationship,
        object_name1, object_name2,
        group_indexes1, object_numbers1,
        group_indexes2, object_numbers2):
        '''Add object relationships to the measurements

        module_number - the module that generated the relationship

        relationship - the relationship of the two objects, for instance,
                       "Parent" means object # 1 is the parent of object # 2

        object_name1, object_name2 - the name of the segmentation for the first and second objects

        group_indexes1, group_indexes2 - for each object, the group index of
                                         that object's image set.
                                         (MUST NOT BE A SCALAR)

        object_numbers1, object_numbers2 - for each object, the object number
                                           in the object's object set

        This method lets the caller store any sort of arbitrary relationship
        between objects as long as they are in the same group. To record
        all neighbors within a particular segmentation, call with the same
        object name for object_name1 and object_name2 and the same group
        index - that of the current image. Relating would have different object
        names and TrackObjects would have different group indices.
        '''

        # XXX - check overwrite?
        # XXX - Should group number be moved out of the measurement name?
        group_number = self.group_number
        with self.hdf5_dict.lock:
            self.hdf5_dict.top_group.require_group(RELATIONSHIP)
            relationship_group = self.hdf5_dict.top_group.require_group('%s/%02d_%d_%s_%s_%s' % (RELATIONSHIP, module_number, group_number, relationship, object_name1, object_name2))
            features = ["group_number", "group_index1", "group_index2", "object_number1", "object_number2"]
            if "group_number" not in relationship_group:
                for name in features:
                    relationship_group.create_dataset(name, (0,), dtype='int32', chunks=(1024,), maxshape=(None,))
            current_size = relationship_group['group_number'].shape[0]
            for name in features:
                relationship_group[name].resize((current_size + len(group_indexes1),))
            relationship_group['group_number'][current_size:] = group_number
            relationship_group['group_index1'][current_size:] = group_indexes1
            relationship_group['group_index2'][current_size:] = group_indexes2
            relationship_group['object_number1'][current_size:] = object_numbers1
            relationship_group['object_number2'][current_size:] = object_numbers2
            self.__relationships.add((module_number, group_number, relationship, object_name1, object_name2))
            self.__relationship_names.add(relationship_group.name)

    def get_relationship_groups(self):
        '''Return the keys of each of the relationship groupings.

        The value returned is a list composed of objects with the following
        attributes:
        module_number - the module number of the module used to generate the relationship
        group_number - the group number of the relationship
        relationship - the relationship of the two objects
        object_name1 - the object name of the first object in the relationship
        object_name2 - the object name of the second object in the relationship
        '''

        return [RelationshipKey(module_number, group_number, relationship, obj1, obj2) for
                (module_number, group_number, relationship, obj1, obj2) in self.__relationships]

    def get_relationships(self, module_number, relationship, object_name1, object_name2, group_number):
        if not (module_number, group_number, relationship, object_name1, object_name2) in self.__relationships:
            return np.zeros(0, [("group_index1", int, 1),
                                ("object_number1", int, 1),
                                ("group_index2", int, 1),
                                ("object_number2", int, 1)]).view(np.recarray)
        with self.hdf5_dict.lock:
            grp = self.hdf5_dict.top_group['%s/%02d_%d_%s_%s_%s' % (RELATIONSHIP, module_number, group_number, relationship, object_name1, object_name2)]
            dt = np.dtype([("group_index1", np.int, 1),
                           ("object_number1", np.int, 1),
                           ("group_index2", np.int, 1),
                           ("object_number2", np.int, 1)])
            temp = np.zeros(grp['group_index1'].shape, dt)
            temp['group_index1'] = grp['group_index1']
            temp['object_number1'] = grp['object_number1']
            temp['group_index2'] = grp['group_index2']
            temp['object_number2'] = grp['object_number2']
            return temp.view(np.recarray)

    def add_measurement(self, object_name, feature_name, data, can_overwrite=False, image_set_number=None):
        """Add a measurement or, for objects, an array of measurements to the set

        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        """
        can_overwrite = can_overwrite or self.__can_overwrite
        if image_set_number is None:
            image_set_number = self.image_set_number

        # some code adds ImageNumber and ObjectNumber measurements explicitly
        if feature_name in (IMAGE_NUMBER, OBJECT_NUMBER):
            return

        fail_if_missing = not (can_overwrite or (self.is_first_image and not self.__initialized_explicitly) or (object_name == EXPERIMENT))
        if fail_if_missing:
            assert object_name in self.get_object_names(), \
                ("Object %s requested for the first time in image set number %d" % \
                     (object_name, self.image_set_number))
            assert feature_name in self.get_feature_names(object_name), \
                ("Feature %s.%s added for the first time in image set # %d (%s %s %s)" % \
                     (object_name, feature_name, self.image_set_number, can_overwrite, self.is_first_image, self.__initialized_explicitly))

        # XXX - Why is this here?  To allow repeated columns in Metadata?
        if (not can_overwrite and
            object_name == IMAGE and feature_name.startswith(C_METADATA)
            and self.has_measurements(object_name, feature_name, image_set_number)):
            assert self.get_measurement(IMAGE, feature_name, image_set_number) == data, '%s != %s at img # %d' % \
                (self.get_measurement(IMAGE, feature_name, image_set_number), data, image_set_number)
        else:
            assert (can_overwrite or not
                    self.has_measurements(object_name, feature_name, image_set_number)),\
                    ("Feature %s.%s has already been set for image cycle %d" %
                     (object_name, feature_name, image_set_number))

        def wrap_string(v):
            if isinstance(v, basestring):
                return unicode(v).encode('unicode_escape')
            return v

        if object_name == EXPERIMENT:
            if not np.isscalar(data) and data is not None:
                data = data[0]
            if data is None:
                data = []
            self.hdf5_dict[EXPERIMENT, feature_name, 0] = wrap_string(data)
        elif object_name == IMAGE:
            if not np.isscalar(data) and data is not None:
                data = data[0]
            if data is None:
                data = []
            self.hdf5_dict[IMAGE, feature_name, image_set_number] = wrap_string(data)
            if not self.hdf5_dict.has_data(object_name, 'ImageNumber', image_set_number):
                self.hdf5_dict[IMAGE, 'ImageNumber', image_set_number] = image_set_number
        else:
            self.hdf5_dict[object_name, feature_name, image_set_number] = data
            if not self.hdf5_dict.has_data(object_name, 'ObjectNumber', image_set_number):
                self.hdf5_dict[object_name, 'ImageNumber', image_set_number] = [image_set_number] * len(data)
                self.hdf5_dict[object_name, 'ObjectNumber', image_set_number] = np.arange(1, len(data) + 1)

    def get_object_names(self):
        """The list of object names (including Image) that have measurements
        """
        return self.hdf5_dict.top_level_names()

    object_names = property(get_object_names)

    def get_feature_names(self, object_name):
        """The list of feature names (measurements) for an object
        """
        return [name for name in self.hdf5_dict.second_level_names(object_name) if name not in ('ImageNumber', 'ObjectNumber')]
    
    def get_image_numbers(self):
        '''Return the image numbers from the Image table'''
        image_numbers = np.array(
            self.hdf5_dict.get_indices(IMAGE, IMAGE_NUMBER), int)
        image_numbers.sort()
        return image_numbers

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

    def get_measurement(self, object_name, feature_name, image_set_number=None):
        """Return the value for the named measurement and indicated image set"""
        def unwrap_string(v):
            # hdf5 returns string columns as a wrapped type
            if isinstance(v, str):
                return unicode(str(v)).decode('unicode_escape')
            return v
        if object_name == EXPERIMENT:
            return unwrap_string(self.hdf5_dict[EXPERIMENT, feature_name, 0][0])
        if object_name == IMAGE:
            return unwrap_string(self.hdf5_dict[IMAGE, feature_name, image_set_number][0])
        vals = self.hdf5_dict[object_name, feature_name, image_set_number]
        if vals is None:
            return np.array([])
        return vals.flatten()

    def has_measurements(self, object_name, feature_name, image_set_number):
        if object_name == EXPERIMENT:
            return self.hdf5_dict.has_data(EXPERIMENT, feature_name, 0)
        return self.hdf5_dict.has_data(object_name, feature_name, image_set_number)

    def has_current_measurements(self, object_name, feature_name):
        return self.has_measurements(object_name, feature_name, self.image_set_number)

    def get_all_measurements(self, object_name, feature_name):
        warnings.warn("get_all_measurements() is deprecated.  Use get_measurement()", DeprecationWarning, stacklevel=2)

        assert self.hdf5_dict.has_feature(object_name, feature_name) or feature_name in ('ImageNumber', 'ObjectNumber')
        if object_name == EXPERIMENT:
            return self.hdf5_dict[EXPERIMENT, feature_name, 0][0]
        vals = [self.get_measurement(object_name, feature_name, idx) for idx in self.hdf5_dict.get_indices(object_name, 'ImageNumber')]
        if object_name == IMAGE:
            return np.array(vals)
        return vals

    def add_all_measurements(self, object_name, feature_name, values):
        '''Add a list of measurements for all image sets

        object_name - name of object or Images
        feature_name - feature to add
        values - list of either values or arrays of values
        '''
        for idx, val in zip(xrange(len(values)), values):
            if val is None:
                if self.hdf5_dict.has_feature(object_name, feature_name):
                    del self.hdf5_dict[object_name, feature_name, idx + 1]
            else:
                self.add_measurement(object_name, feature_name, val, can_overwrite=True, image_set_number=idx + 1)

    def get_experiment_measurement(self, feature_name):
        """Retrieve an experiment-wide measurement
        """
        return self.get_measurement(EXPERIMENT, feature_name) or 'N/A'

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
                measurement = '%s_%s' % (C_METADATA, m.groups()[0])
                result += str(self.get_measurement("Image", measurement,
                                                   image_set_number))
                piece = piece[m.end():]
            result_pieces.append(result)
        return single_backquote.join(result_pieces)

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
        for image_number in self.get_image_numbers():
            values = [self.get_measurement(
                IMAGE, "%s_%s" % (C_METADATA, tag), image_number)
                      for tag in tags]
            values = [ x if np.isscalar(x) else x[0] for x in values]
            key = tuple(values)
            if not flat_dictionary.has_key(key):
                flat_dictionary[key] = []
            flat_dictionary[key].append(image_number)
        result = []
        for row in flat_dictionary.keys():
            tag_dictionary = {}
            tag_dictionary.update(zip(tags, row))
            result.append(MetadataGroup(tag_dictionary, flat_dictionary[row]))
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

def load_measurements(measurements_file_name):
    '''Load measurements from a .mat file'''

    m = Measurements()
    m.load(measurements_file_name)
    return m

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
    def __init__(self, module_number, group_number, relationship,
                 object_name1, object_name2):
        self.module_number = module_number
        self.group_number = group_number
        self.relationship = relationship
        self.object_name1 = object_name1
        self.object_name2 = object_name2
