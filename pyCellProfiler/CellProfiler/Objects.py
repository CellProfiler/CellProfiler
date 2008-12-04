""" CellProfiler.Objects.py - represents a labelling of objects in an image

"""
import numpy

class Objects(object):
    """The labelling of an image with object #s
    """
    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
    
    def GetSegmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix of object #s
        """
        return self.__segmented
    
    def SetSegmented(self,labels):
        check_consistency(labels, self.__unedited_segmented, self.__small_removed_segmented)
        self.__segmented = labels
    
    Segmented = property(GetSegmented,SetSegmented)
    
    def HasUneditedSegmented(self):
        """Return true if there is an unedited segmented matrix
        """
        return self.__unedited_segmented != None
    
    def GetUneditedSegmented(self):
        """Get the segmentation of the image into objects including junk that should be ignored: a matrix of object #s
        
        The default, if no unedited matrix is available, is the segmented labeling
        """
        if self.__segmented != None:
            if self.__unedited_segmented != None:
                return self.__unedited_segmented
            return self.__segmented
        return self.__unedited_segmented
    
    def SetUneditedSegmented(self,labels):
        check_consistency(self.__segmented, labels, self.__small_removed_segmented)
        self.__unedited_segmented = labels
    
    UneditedSegmented = property(GetUneditedSegmented,SetUneditedSegmented)
    
    def HasSmallRemovedSegmented(self):
        """Return true if there is a junk object matrix
        """
        return self.__small_removed_segmented != None
    
    def GetSmallRemovedSegmented(self):
        """Get the junk objects only: a matrix of object #s
        
        The default, if no unedited matrix is available, is a matrix
        """
        if self.__small_removed_segmented != None:
            return self.__small_removed_segmented
        if self.__segmented != None:
            if self.__unedited_segmented != None:
                # The small removed are the unedited minus the official segmentation
                result = self.__unedited_segmented.copy()
                result[self.__unedited_segmented == self.__segmented] = 0
                return result
            # if there's only a segmented, then there is no junk
            return numpy.zeros(shape=self.__segmented.shape,dtype=self.__segmented.dtype)
        return self.__small_removed_segmented
    
    def SetSmallRemovedSegmented(self,labels):
        check_consistency(self.__segmented, self.__unedited_segmented, labels)
        self.__small_removed_segmented = labels
    
    SmallRemovedSegmented = property(GetSmallRemovedSegmented, SetSmallRemovedSegmented)

def check_consistency(segmented, unedited_segmented, small_removed_segmented):
    """Check the three components of Objects to make sure they are consistent
    """
    assert segmented == None or segmented.ndim == 2, "Segmented label matrix must have two dimensions, has %d"%(segmented.ndim)
    assert unedited_segmented == None or unedited_segmented.ndim == 2, "Unedited segmented label matrix must have two dimensions, has %d"%(unedited_segmented.ndim)
    assert small_removed_segmented == None or small_removed_segmented.ndim == 2, "Small removed segmented label matrix must have two dimensions, has %d"%(small_removed_segmented.ndim)
    assert segmented == None or unedited_segmented == None or segmented.shape == unedited_segmented.shape, "Segmented %s and unedited segmented %s shapes differ"%(repr(segmented.shape),repr(unedited_segmented.shape))
    assert segmented == None or small_removed_segmented == None or segmented.shape == small_removed_segmented.shape, "Segmented %s and small removed segmented %s shapes differ"%(repr(segmented.shape),repr(small_removed_segmented.shape))
    assert segmented == None or \
           unedited_segmented == None or \
           numpy.logical_or(segmented == 0,segmented==unedited_segmented).all(), \
           "Segmented must be a subset of unedited segmented & numbered the same"
    assert small_removed_segmented == None or \
           unedited_segmented == None or \
           numpy.logical_or(small_removed_segmented == 0,small_removed_segmented==unedited_segmented).all(), \
           "Small removed segmented must be a subset of unedited segmented & numbered the same"
    

class ObjectSet(object):
    """The set of objects associated with some image set
    
    The idea here is to be able to refer to the available objects by name or to be able to iterate over them
    """
    
    def __init__(self):
        self.__objects_by_name = {}
    
    def AddObjects(self, objects, name):
        assert isinstance(objects,Objects), "objects must be an instance of CellProfiler.Objects"
        assert not self.__objects_by_name.has_key(name), "The object, %s, is already in the object set"%(name)
        self.__objects_by_name[name] = objects
    
    def GetObjectNames(self):
        """Return the names of all of the objects
        """
        return self.__objects_by_name.keys()
    
    ObjectNames = property(GetObjectNames)
    
    def GetObjects(self,name):
        """Return the objects instance with the given name
        """
        return self.__objects_by_name[name]
    
    def GetAllObjects(self):
        """Return a list of name / objects tuples
        """
        return self.__objects_by_name.items()
    
    AllObjects = property(GetAllObjects)