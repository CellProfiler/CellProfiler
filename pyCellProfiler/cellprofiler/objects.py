""" CellProfiler.Objects.py - represents a labelling of objects in an image

"""
import numpy
import scipy.sparse

class Objects(object):
    """The labelling of an image with object #s
    """
    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None
    
    def get_segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix of object #s
        """
        return self.__segmented
    
    def set_segmented(self,labels):
        check_consistency(labels, self.__unedited_segmented, self.__small_removed_segmented)
        self.__segmented = labels
    
    segmented = property(get_segmented,set_segmented)
    
    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix
        """
        return self.__unedited_segmented != None
    
    def get_unedited_segmented(self):
        """Get the segmentation of the image into objects including junk that should be ignored: a matrix of object #s
        
        The default, if no unedited matrix is available, is the segmented labeling
        """
        if self.__segmented != None:
            if self.__unedited_segmented != None:
                return self.__unedited_segmented
            return self.__segmented
        return self.__unedited_segmented
    
    def set_unedited_segmented(self,labels):
        check_consistency(self.__segmented, labels, self.__small_removed_segmented)
        self.__unedited_segmented = labels
    
    unedited_segmented = property(get_unedited_segmented,set_unedited_segmented)
    
    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix
        """
        return self.__small_removed_segmented != None
    
    def get_small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed
        
        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
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
    
    def set_small_removed_segmented(self,labels):
        check_consistency(self.__segmented, self.__unedited_segmented, labels)
        self.__small_removed_segmented = labels
    
    small_removed_segmented = property(get_small_removed_segmented, set_small_removed_segmented)
    
    def get_parent_image(self):
        """The image that was analyzed to yield the objects
        
        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image
    
    def set_parent_image(self, parent_image):
        self.__parent_image = parent_image
        
    parent_image = property(get_parent_image, set_parent_image)
    
    def crop_image_similarly(self, image):
        """Crop an image similarly to the way the parent image of these objects were cropped
        
        """
        if image.shape == self.segmented.shape:
            return image
        if self.parent_image == None:
            raise ValueError("Images are of different size and no parent image")
        return self.parent_image.crop_image_similarly(image)
    
    def relate_children(self, children):
        """Relate the object numbers in one label to the object numbers in another
        
        children - another "objects" instance: the labels of children within
                   the parent which is "self"
        
        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        """
        parent_labels = self.segmented
        child_labels = children.segmented
        # Apply cropping to the parent if done to the child
        parent_labels  = children.crop_image_similarly(parent_labels)
        #
        # Only look at points that are labeled in parent and child
        #
        not_zero = numpy.logical_and(parent_labels > 0,
                                     child_labels > 0)
        not_zero_count = numpy.sum(not_zero)
        max_parent = numpy.max(parent_labels)
        max_child  = numpy.max(child_labels)
        histogram = scipy.sparse.coo_matrix((numpy.ones((not_zero_count,)),
                                             (parent_labels[not_zero],
                                              child_labels[not_zero])),
                                             shape=(max_parent+1,max_child+1))
        #
        # each row (axis = 0) is a parent
        # each column (axis = 1) is a child
        #
        histogram = histogram.toarray()
        parents_of_children = numpy.argmax(histogram,axis=0)
        #
        # Create a histogram of # of children per parent
        poc_histogram = scipy.sparse.coo_matrix((numpy.ones((max_child+1,)),
                                                 (parents_of_children,
                                                  numpy.zeros((max_child+1,),int))),
                                                 shape=(max_parent+1,1))
        children_per_parent = poc_histogram.toarray().flatten()
        return children_per_parent, parents_of_children
        

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
           numpy.logical_or(segmented == 0,unedited_segmented!=0).all(), \
           "Unedited segmented must be labeled if segmented is labeled"
    assert small_removed_segmented == None or \
           unedited_segmented == None or \
           numpy.logical_or(small_removed_segmented == 0,unedited_segmented!=0).all(), \
           "Unedited segmented must be labeled if small_removed_segmented is labeled"
    

class ObjectSet(object):
    """The set of objects associated with some image set
    
    The idea here is to be able to refer to the available objects by name or to be able to iterate over them
    """
    
    def __init__(self):
        self.__objects_by_name = {}
    
    def add_objects(self, objects, name):
        assert isinstance(objects,Objects), "objects must be an instance of CellProfiler.Objects"
        assert not self.__objects_by_name.has_key(name), "The object, %s, is already in the object set"%(name)
        self.__objects_by_name[name] = objects
    
    def get_object_names(self):
        """Return the names of all of the objects
        """
        return self.__objects_by_name.keys()
    
    object_names = property(get_object_names)
    
    def get_objects(self,name):
        """Return the objects instance with the given name
        """
        return self.__objects_by_name[name]
    
    def get_all_objects(self):
        """Return a list of name / objects tuples
        """
        return self.__objects_by_name.items()
    
    all_objects = property(get_all_objects)
