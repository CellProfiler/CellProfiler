"""Measurements.py - storage for image and object measurements

"""
import numpy

class Measurements(object):
    """Represents measurements made on images and objects
    """
    def __init__(self):
        self.__dictionary = {}
        self.__image_set_number = 0
    
    def NextImageSet(self):
        self.__image_set_number+=1
        for object_features in self.__dictionary.values():
            for measurements in object_features.values():
                measurements.append(None)
    
    def GetImageSetNumber(self):
        """The image set number ties a bunch of measurements to a particular image set
        """
        return self.__image_set_number
    
    ImageSetNumber = property(GetImageSetNumber)
    
    def AddMeasurement(self, ObjectName, FeatureName, Data):
        """Add a measurement or, for objects, an array of measurements to the set
        
        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        """
        if self.ImageSetNumber == 0:
            if not self.__dictionary.has_key(ObjectName):
                self.__dictionary[ObjectName] = {}
            object_dict = self.__dictionary[ObjectName]
            if not object_dict.has_key(FeatureName):
                object_dict[FeatureName] = [Data]
            else:
                assert False,"Adding a feature for a second time: %s.%s"%(ObjectName,FeatureName)
        else:
            assert self.__dictionary.has_key(ObjectName),"Object %s requested for the first time on pass # %d"%(ObjectName,self.ImageSetNumber)
            assert self.__dictionary[ObjectName].has_key(FeatureName),"Feature %s.%s added for the first time on pass # %d"%(ObjectName,FeatureName,self.ImageSetNumber)
            assert not self.HasCurrentMeasurements(ObjectName, FeatureName), "Feature %s.%s has already been set for this image set"%(ObjectName,FeatureName)
            #
            # These are for convenience - wrap measurement in an numpy array to make it a cell
            #
            if isinstance(Data,unicode):
                Data = str(Data)
            if isinstance(Data,str):
                a = numpy.ndarray((1,1),dtype='S%d'%(len(Data)))
                a[0,0]=Data
            self.__dictionary[ObjectName][FeatureName][self.ImageSetNumber] = Data
    
    def GetObjectNames(self):
        """The list of object names (including Image) that have measurements
        """
        return self.__dictionary.keys()
    
    ObjectNames = property(GetObjectNames)
    
    def GetFeatureNames(self,object_name):
        """The list of feature names (measurements) for an object
        """
        if self.__dictionary.has_key(object_name):
            return self.__dictionary[object_name].keys()
        return []
    
    def GetCurrentMeasurement(self,object_name,feature_name):
        """Return the value for the named measurement for the current image set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        return self.GetAllMeasurements(object_name,feature_name)[self.ImageSetNumber]
    
    def HasCurrentMeasurements(self,object_name,feature_name):
        """Return true if the value for the named measurement for the current image set has been set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned 
        """
        if not self.__dictionary.has_key(object_name):
            return False
        if not self.__dictionary[object_name].has_key(feature_name):
            return False
        return self.__dictionary[object_name][feature_name][self.ImageSetNumber] != None
    
    def GetAllMeasurements(self,object_name,feature_name):
        assert self.__dictionary.has_key(object_name),"No measurements for %s"%(object_name)
        assert self.__dictionary[object_name].has_key(feature_name),"No measurements for %s.%s"%(object_name,feature_name)
        return self.__dictionary[object_name][feature_name]
    