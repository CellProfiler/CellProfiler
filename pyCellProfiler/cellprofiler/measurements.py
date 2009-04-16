"""Measurements.py - storage for image and object measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import re

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

class Measurements(object):
    """Represents measurements made on images and objects
    """
    def __init__(self, can_overwrite = False):
        """Create a new measurements collection
        
        can_overwrite - if True, overwriting measurements during operation
                        is allowed. We turn this on for debugging.
        """
        self.__dictionary = {}
        self.__image_set_number = 0
        self.__can_overwrite = can_overwrite
    
    def next_image_set(self):
        self.__image_set_number+=1
        for object_features in self.__dictionary.values():
            for measurements in object_features.values():
                measurements.append(None)
    
    def get_image_set_number(self):
        """The image set number ties a bunch of measurements to a particular image set
        """
        return self.__image_set_number
    
    image_set_number = property(get_image_set_number)
    
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
        
    def add_measurement(self, object_name, feature_name, data):
        """Add a measurement or, for objects, an array of measurements to the set
        
        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        """
        if self.image_set_number == 0:
            if not self.__dictionary.has_key(object_name):
                self.__dictionary[object_name] = {}
            object_dict = self.__dictionary[object_name]
            if not object_dict.has_key(feature_name):
                object_dict[feature_name] = [data]
            elif self.__can_overwrite:
                object_dict[feature_name] = [data]
            else:
                assert False,"Adding a feature for a second time: %s.%s"%(object_name,feature_name)
        else:
            assert self.__dictionary.has_key(object_name),"Object %s requested for the first time on pass # %d"%(object_name,self.image_set_number)
            assert self.__dictionary[object_name].has_key(feature_name),"Feature %s.%s added for the first time on pass # %d"%(object_name,feature_name,self.image_set_number)
            assert not self.has_current_measurements(object_name, feature_name), "Feature %s.%s has already been set for this image set"%(object_name,feature_name)
            #
            # These are for convenience - wrap measurement in an numpy array to make it a cell
            #
            if isinstance(data,unicode):
                data = str(data)
            if isinstance(data,str):
                a = np.ndarray((1,1),dtype='S%d'%(max(len(data),1)))
                a[0,0]=data
            self.__dictionary[object_name][feature_name][self.image_set_number] = data
    
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
        return self.__dictionary[object_name].has_key(feature_name)
    
    def get_current_measurement(self,object_name,feature_name):
        """Return the value for the named measurement for the current image set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        return self.get_all_measurements(object_name,feature_name)[self.image_set_number]
    
    def get_measurement(self,object_name,feature_name,image_set_number):
        """Return the value for the named measurement and indicated image set"""
        return self.get_all_measurements(object_name,feature_name)[image_set_number]
    
    def has_current_measurements(self,object_name,feature_name):
        """Return true if the value for the named measurement for the current image set has been set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        if not self.__dictionary.has_key(object_name):
            return False
        if not self.__dictionary[object_name].has_key(feature_name):
            return False
        return self.__dictionary[object_name][feature_name][self.image_set_number] != None
    
    def get_all_measurements(self,object_name,feature_name):
        assert self.__dictionary.has_key(object_name),"No measurements for %s"%(object_name)
        assert self.__dictionary[object_name].has_key(feature_name),"No measurements for %s.%s"%(object_name,feature_name)
        return self.__dictionary[object_name][feature_name]
    
    def get_experiment_measurement(self, feature_name):
        """Retrieve an experiment-wide measurement
        """
        return self.get_all_measurements(EXPERIMENT, feature_name)
    
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
        result = ''
        while True:
            m = re.search('\\(\\?[<](.+?)[>]\\)', pattern)
            if not m:
                m = re.search('\\\\g[<](.+?)[>]', pattern)
                if not m:
                    break
            result += pattern[:m.start()]
            measurement = 'Metadata_'+m.groups()[0]
            result += self.get_measurement("Image", measurement, 
                                           image_set_number)
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
            return [MetadataGroup({}, range(self.image_set_number+1))]
            
        #
        # The flat_dictionary has a row of tag values as a key
        #
        flat_dictionary = {}
        values = [self.get_all_measurements('Image', "Metadata_%s"%tag)
                  for tag in tags]
        for index in range(self.image_set_number+1):
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
        if feature_name.startswith('Description_'):
            return True
        if feature_name.startswith('ModuleError_'):
            return True
        if feature_name.startswith('TimeElapsed_'):
            return True
        return False
    

    def compute_aggregate_measurements(self, image_set_number):
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
                values[np.logical_not(np.isfinite(values))] = 0
                #
                # Compute the mean and standard deviation
                #
                mean_feature_name = '%s_%s_%s'%(AGG_MEAN, object_name,
                                                 feature)
                mean = values.mean()
                d[mean_feature_name] = mean
                median_feature_name = '%s_%s_%s'%(AGG_MEDIAN, 
                                                  object_name, feature)
                median = np.median(values)
                d[median_feature_name] = median
                stdev_feature_name = '%s_%s_%s'%(AGG_STD_DEV,
                                                 object_name, feature)
                stdev = values.std()
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
    pattern = re.sub('(\\(\\?)([<].+?[>])','\\1P\\2',pattern)
    match = re.search(pattern, text)
    if match:
        return match.groupdict()
    else:
        return {}
