"""Measurements.py - storage for image and object measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import re
from scipy.io.matlab import loadmat

AGG_MEAN = "Mean"
AGG_STD_DEV= "StDev"
AGG_MEDIAN = "Median"
AGG_NAMES = [AGG_MEAN, AGG_MEDIAN, AGG_STD_DEV]

"""The per-image measurement category"""
IMAGE = "Image"

"""The per-experiment measurement category"""
EXPERIMENT = "Experiment"

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
MCA_AVAILABLE_POST_GROUP ="AvailablePostGroup"

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
                 can_overwrite = False, 
                 image_set_start = None):
        """Create a new measurements collection
        
        can_overwrite - if True, overwriting measurements during operation
                        is allowed. We turn this on for debugging.
        image_set_start - the index of the first image set in the image set list
                          or None to start at the beginning
        """
        self.__dictionary = {}
        self.__image_set_number = (1 if image_set_start is None
                                   else image_set_start+1)
        self.__image_set_start = image_set_start
        self.__can_overwrite = can_overwrite
        self.__is_first_image = True
        self.__image_set_index = 0
        self.__initialized_explicitly = False
    
    def initialize(self, measurement_columns):
        '''Initialize the measurements dictionary with a list of columns
        
        This explicitly initializes the measurements dictionary with
        a list of columns as would be returned by get_measurement_columns()
        
        measurement_columns - list of 3-tuples: object name, feature, type
        '''
        self.__dictionary = {}
        for column in measurement_columns:
            object_name, feature, coltype = column[:3]
            if not self.__dictionary.has_key(object_name):
                self.__dictionary[object_name] = {}
            self.__dictionary[object_name][feature] = [ None ]
        self.__initialized_explicitly = True
                
        
    def next_image_set(self, explicit_image_set_number = None, erase=False):
        if explicit_image_set_number is None:
            self.__image_set_number+=1
        else:
            self.__image_set_number = explicit_image_set_number
        self.__image_set_index += 1
        self.__is_first_image = False
        for object_name, object_features in self.__dictionary.iteritems():
            if object_name in (EXPERIMENT, NEIGHBORS):
                continue
            for measurements in object_features.values():
                mlen = len(measurements)
                if mlen <= self.__image_set_index:
                    measurements += [None] * (self.__image_set_index + 1 - mlen)
                elif erase:
                    measurements[self.__image_set_index] = None
    
    @property
    def image_set_count(self):
        '''The number of complete image sets measured'''
        if len(self.__dictionary) == 0:
            return 0
        if not self.__dictionary.has_key(IMAGE):
            return 0
        min_count = np.iinfo(int).max
        for measurements in self.__dictionary[IMAGE].values():
            min_count = min(min_count, len(measurements))
        return min_count
    
    
    def get_is_first_image(self):
        '''True if this is the first image in the set'''
        return self.__is_first_image
    
    def set_is_first_image(self, value):
        if not value:
            raise ValueError("Can only reset to be first image")
        self.__is_first_image = value
        self.__image_set_index = 0
        self.__image_set_number = self.image_set_start_number
        
    is_first_image = property(get_is_first_image, set_is_first_image)
    
    @property
    def image_set_start_number(self):
        '''The first image set (one-based) processed by the pipeline'''
        if self.__image_set_start is None:
            return 1
        return self.__image_set_start+1
    
    @property
    def has_image_set_start(self):
        '''True if the image set has an explicit start'''
        return not self.__image_set_start is None
    
    def get_image_set_number(self):
        """The image set number ties a bunch of measurements to a particular image set
        """
        return self.__image_set_number
    
    def set_image_set_number(self, number):
        self.__image_set_index = number -1
        self.__image_set_number = number
    
    image_set_number = property(get_image_set_number, set_image_set_number)
    
    @property
    def image_set_index(self):
        '''Return the index into the measurements for the current measurement'''
        return self.__image_set_index
    
    def load(self, measurements_file_name):
        '''Load measurements from a matlab file'''
        handles = loadmat(measurements_file_name, struct_as_record=True)
        self.create_from_handles(handles)
        
    def create_from_handles(self, handles):
        '''Load measurements from a handles structure'''
        m = handles["handles"][0,0]["Measurements"][0,0]
        for object_name in m.dtype.fields.keys():
            omeas = m[object_name][0,0]
            for feature_name in omeas.dtype.fields.keys():
                if object_name == IMAGE:
                    values = [x.flatten()[0] for x in omeas[feature_name][0]]
                elif object_name == EXPERIMENT:
                    value = omeas[feature_name][0,0].flatten()[0]
                    self.add_experiment_measurement(feature_name, value)
                    continue
                else:
                    values = [x.flatten() 
                              for x in omeas[feature_name][0].tolist()]
                self.add_all_measurements(object_name,
                                          feature_name,
                                          values)
        #
        # Set the image set index to the end
        #
        self.__image_set_index = self.image_set_count - 1
    
    def add_image_measurement(self, feature_name, data):
        """Add a measurement to the "Image" category
        
        """
        self.add_measurement(IMAGE, feature_name, data)
    
    def add_experiment_measurement(self, feature_name, data):
        """Add an experiment measurement to the measurement
        
        Experiment measurements have one value per experiment
        """
        if not self.__dictionary.has_key(EXPERIMENT):
            self.__dictionary[EXPERIMENT] = {}
        self.__dictionary[EXPERIMENT][feature_name] = data
        
    def add_measurement(self, object_name, feature_name, data, can_overwrite=False):
        """Add a measurement or, for objects, an array of measurements to the set
        
        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        """
        can_overwrite = can_overwrite or self.__can_overwrite
        if self.is_first_image and not self.__initialized_explicitly:
            if not self.__dictionary.has_key(object_name):
                self.__dictionary[object_name] = {}
            object_dict = self.__dictionary[object_name]
            if not object_dict.has_key(feature_name):
                object_dict[feature_name] = [data]
            elif can_overwrite:
                object_dict[feature_name] = [data]
            elif (object_name == IMAGE and 
                  feature_name.startswith(C_METADATA)):
                if object_dict[feature_name][0] != data:
                    raise ValueError(
                        "Metadata feature %s has conficting values: %s and %s" %
                        (feature_name, object_dict[feature_name][0], data))
            else:
                assert False,"Adding a feature for a second time: %s.%s"%(object_name,feature_name)
        else:
            if can_overwrite:
                if not self.__dictionary.has_key(object_name):
                    self.__dictionary[object_name] = {}
                object_dict = self.__dictionary[object_name]
                if not object_dict.has_key(feature_name):
                    object_dict[feature_name] = [ None] * (self.image_set_index+1)
                
            assert self.__dictionary.has_key(object_name),\
                   ("Object %s requested for the first time on pass # %d" %
                    (object_name,self.image_set_index))
            assert self.__dictionary[object_name].has_key(feature_name),\
                   ("Feature %s.%s added for the first time on pass # %d" %
                    (object_name,feature_name,self.image_set_index))
            if (object_name == IMAGE and feature_name.startswith(C_METADATA)
                and self.has_current_measurements(object_name, feature_name)):
                assert self.get_current_image_measurement(feature_name) == data
            else:
                assert (can_overwrite or not
                        self.has_current_measurements(object_name, feature_name)),\
                       ("Feature %s.%s has already been set for this image cycle" %
                        (object_name,feature_name))
            #
            # These are for convenience - wrap measurement in an numpy array to make it a cell
            #
            if isinstance(data,unicode):
                data = str(data)
            if isinstance(data,str):
                a = np.ndarray((1,1),dtype='S%d'%(max(len(data),1)))
                a[0,0]=data
            self.__dictionary[object_name][feature_name][self.image_set_index] = data
    
    def get_object_names(self):
        """The list of object names (including Image) that have measurements
        """
        return self.__dictionary.keys()
    
    object_names = property(get_object_names)
    
    def get_feature_names(self,object_name):
        """The list of feature names (measurements) for an object
        """
        if self.__dictionary.has_key(object_name):
            return self.__dictionary[object_name].keys()
        return []
    
    def has_feature(self, object_name, feature_name):
        """Return true if a particular object has a particular feature"""
        if not self.__dictionary.has_key(object_name):
            return False
        return self.__dictionary[object_name].has_key(feature_name)
    
    def get_current_image_measurement(self, feature_name):
        '''Return the value for the named image measurement
        
        feature_name - the name of the measurement feature to be returned
        '''
        return self.get_current_measurement(IMAGE, feature_name)
    
    def get_current_measurement(self,object_name,feature_name):
        """Return the value for the named measurement for the current image set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        return self.get_all_measurements(object_name,feature_name)[self.image_set_index]
    
    def get_measurement(self,object_name,feature_name,image_set_index):
        """Return the value for the named measurement and indicated image set"""
        return self.get_all_measurements(object_name,feature_name)[image_set_index]
    
    def has_current_measurements(self,object_name,feature_name):
        """Return true if the value for the named measurement for the current image set has been set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        if not self.__dictionary.has_key(object_name):
            return False
        if not self.__dictionary[object_name].has_key(feature_name):
            return False
        return self.__dictionary[object_name][feature_name][self.image_set_index] != None
    
    def get_all_measurements(self,object_name,feature_name):
        assert self.__dictionary.has_key(object_name),"No measurements for %s"%(object_name)
        assert self.__dictionary[object_name].has_key(feature_name),"No measurements for %s.%s"%(object_name,feature_name)
        return self.__dictionary[object_name][feature_name]
    
    def add_all_measurements(self, object_name, feature_name, values):
        '''Add a list of measurements for all image sets
        
        object_name - name of object or Images
        feature_name - feature to add
        values - list of either values or arrays of values
        '''
        if not self.__dictionary.has_key(object_name):
            self.__dictionary[object_name] = {}
        self.__dictionary[object_name][feature_name] = values
    
    def get_experiment_measurement(self, feature_name):
        """Retrieve an experiment-wide measurement
        """
        return self.get_all_measurements(EXPERIMENT, feature_name)
    
    def apply_metadata(self, pattern, image_set_index=None):
        """Apply metadata from the current measurements to a pattern
        
        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        image_name - name of image associated with the metadata (or None
                     if metadata is not associated with an image)
        image_set_index - # of image set to use to retrieve data. 
                           None for current.
        returns a string with the metadata tags replaced by the metadata
        """
        if image_set_index == None:
            image_set_index = self.image_set_index
        result = ''
        while True:
            # Replace double \\ with \
            m = re.search('\\\\\\\\', pattern)
            if m:
                result += pattern[:m.start()]+'\\'
                pattern = pattern[m.end():]
                continue
            # Replace a tag
            m = re.search('\\(\\?[<](.+?)[>]\\)', pattern)
            if not m:
                m = re.search('\\\\g[<](.+?)[>]', pattern)
                if not m:
                    break
            result += pattern[:m.start()]
            measurement = 'Metadata_'+m.groups()[0]
            result += str(self.get_measurement("Image", measurement, 
                                               image_set_index))
            pattern = pattern[m.end():]
        result += pattern
        return result
    
    def group_by_metadata(self, tags):
        """Return groupings of image sets with matching metadata tags
        
        tags - a sequence of tags to match.
        
        Returns a sequence of MetadataGroup objects. Each one represents
        a set of values for the metadata tags along with the indexes of
        the image sets that match the values
        """
        if len(tags) == 0:
            # if there are no tags, all image sets match each other
            return [MetadataGroup({}, range(self.image_set_index+1))]
            
        #
        # The flat_dictionary has a row of tag values as a key
        #
        flat_dictionary = {}
        values = [self.get_all_measurements('Image', "Metadata_%s"%tag)
                  for tag in tags]
        for index in range(self.image_set_index+1):
            row = tuple([value[index] for value in values])
            if flat_dictionary.has_key(row):
                flat_dictionary[row].append(index)
            else:
                flat_dictionary[row] = [index]
        result = []
        for row in flat_dictionary.keys():
            tag_dictionary = {}
            tag_dictionary.update(zip(tags,row))
            result.append(MetadataGroup(tag_dictionary, flat_dictionary[row]))
        return result

    def agg_ignore_object(self,object_name):
        """Ignore objects (other than 'Image') if this returns true"""
        if object_name in (EXPERIMENT,NEIGHBORS):
            return True
        
    def agg_ignore_feature(self, object_name, feature_name):
        """Return true if we should ignore a feature during aggregation"""
        
        if self.agg_ignore_object(object_name):
            return True
        if self.has_feature(object_name, "SubObjectFlag"):
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
                feature_name = "%s_%s"%(object_name, feature)
                values = self.get_measurement(object_name, feature, 
                                              image_set_number)
                if values is not None:
                    values = values[np.isfinite(values)] 
                #
                # Compute the mean and standard deviation
                #
                if AGG_MEAN in aggs:
                    mean_feature_name = get_agg_measurement_name(
                        AGG_MEAN, object_name,feature)
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
    def __init__(self, tag_dictionary, indexes):
        super(MetadataGroup,self).__init__(tag_dictionary)
        self.__indexes = indexes
    
    @property
    def indexes(self):
        return self.__indexes
    
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
    pattern = re.sub('(\\(\\?)([<].+?[>])','\\1P\\2',pattern)
    match = re.search(pattern, text)
    if match:
        return match.groupdict()
    else:
        raise ValueError("Metadata extraction failed: regexp '%s' does not match '%s'"%(orig_pattern, text))

def is_well_row_token(x):
    '''True if the string represents a well row metadata tag'''
    return x.lower() in ("wellrow","well_row", "row")

def is_well_column_token(x):
    '''true if the string represents a well column metadata tag'''
    return x.lower() in ("wellcol","well_col","wellcolumn","well_column",
                         "column","col")

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
    