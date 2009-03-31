"""Measurements.py - storage for image and object measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy
import re

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
                a = numpy.ndarray((1,1),dtype='S%d'%(max(len(data),1)))
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
    
    def apply_metadata(self, pattern):
        """Apply metadata from the current measurements to a pattern
        
        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        image_name - name of image associated with the metadata (or None
                     if metadata is not associated with an image)
        returns a string with the metadata tags replaced by the metadata
        """
        result = ''
        while True:
            m = re.search('\\(\\?[<](.+?)[>]\\)', pattern)
            if not m:
                m = re.search('\\g[<](.+?)[>]', pattern)
                if not m:
                    break
            result += pattern[:m.start()]
            measurement = 'Metadata_'+m.groups()[0]
            result += self.get_current_measurement("Image", measurement)
            pattern = pattern[m.end():]
        result += pattern
        return result

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
