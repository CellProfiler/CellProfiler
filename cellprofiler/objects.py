""" CellProfiler.Objects.py - represents a labelling of objects in an image

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import decorator
import numpy as np
from scipy.sparse import coo_matrix

from cellprofiler.cpmath.cpmorphology import all_connected_components
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.index import Indexes, all_pairs

OBJECT_TYPE_NAME = "objects"

@decorator.decorator
def memoize_method(function, *args):
    """Cache the result of a method in that class's dictionary
    
    The dictionary is indexed by function name and the values of that
    dictionary are themselves dictionaries with args[1:] as the keys
    and the result of applying function to args[1:] as the values.
    """
    sself = args[0]
    d = getattr(sself, "memoize_method_dictionary", False)
    if not d:
        d = {}
        setattr(sself,"memoize_method_dictionary",d)
    if not d.has_key(function):
        d[function] = {}
    if not d[function].has_key(args[1:]):
        d[function][args[1:]] = function(*args)
    return d[function][args[1:]]

        
class Objects(object):
    """Represents a segmentation of an image.

    IdentityPrimAutomatic produces three variants of its segmentation
    result. This object contains all three.
    """
    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None
        self.__ijv = None
        self.__shape = None
    
    def get_segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix 
        of object numbers.
        """
        if self.__segmented is None:
            label_stack = [l for l,c in self.get_labels()]
            assert len(label_stack) == 1, "Operation failed because objects overlapped. Please try with non-overlapping objects"
            return label_stack[0]
        return self.__segmented
    
    def set_segmented(self,labels):
        check_consistency(labels, self.__unedited_segmented, 
                          self.__small_removed_segmented)
        self.__segmented = downsample_labels(labels)
        # Clear all cached results.
        if getattr(self, "memoize_method_dictionary", False):
            self.memoize_method_dictionary = {}
    
    segmented = property(get_segmented,set_segmented)
    
    def set_ijv(self, ijv, shape=None):
        '''Set the segmentation to an IJV object format
        
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        self.__ijv = ijv
        self.__shape = shape
        
    def get_ijv(self):
        '''Get the segmentation in IJV object format
        
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        if self.__ijv is None and self.__segmented is not None:
            i,j = np.argwhere(self.__segmented > 0).transpose()
            #
            # Sort so that same "i" coordinates are close together
            # so that memory accesses are localized.
            #
            order = np.lexsort((j,i))
            i,j = i[order],j[order]
            self.__ijv = np.column_stack((i,j,self.__segmented[i,j]))
        return self.__ijv
    
    ijv = property(get_ijv, set_ijv)
    
    @property
    def has_ijv(self):
        '''Return true if there is an IJV formulation for the object'''
        return self.__ijv is not None
    
    @property
    def shape(self):
        '''The i and j extents of the labels'''
        if self.__shape is not None:
            return self.__shape
        if self.__segmented is not None:
            return self.__segmented.shape
        if self.has_parent_image:
            return self.parent_image.pixel_data.shape
        if self.__ijv.shape[0] == 0:
            return (0, 0)
        return tuple([np.max(self.__ijv[:, i]+1) for i in range(2)])
    
    def get_labels(self, shape = None):
        '''Get a set of labels matrices consisting of non-overlapping labels
        
        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.
        
        returns a list of label matrixes and the indexes in each
        '''
        if self.__segmented is not None:
            return [(self.__segmented, self.indices)]
        elif self.__ijv is not None:
            if shape is None:
                shape = self.__shape
            def ijv_to_segmented(ijv, shape=shape):
                if shape is not None:
                    pass
                elif self.has_parent_image:
                    shape = self.parent_image.pixel_data.shape
                elif len(ijv) == 0:
                    # degenerate case, no parent info and no labels
                    shape = (1,1)
                else:
                    shape = np.max(ijv[:,:2], 0) + 2 # add a border of "0" to the right
                labels = np.zeros(shape, np.int16)
                if ijv.shape[0] > 0:
                    labels[ijv[:,0],ijv[:,1]] = ijv[:,2]
                return labels
            
            if len(self.__ijv) == 0:
                return [(ijv_to_segmented(self.__ijv), self.indices)]
            
            sort_order = np.lexsort((self.__ijv[:,2],
                                     self.__ijv[:,1], 
                                     self.__ijv[:,0]))
            sijv = self.__ijv[sort_order]
            #
            # Locations in sorted array where i,j are same consecutively
            # are locations that have an overlap.
            #
            overlap = np.all(sijv[:-1,:2] == sijv[1:,:2],1)
            #
            # Find the # at each location by finding the index of the
            # first example of a location, then subtracting successive indexes
            #
            firsts = np.argwhere(np.hstack(([True], ~overlap, [True]))).flatten()
            counts = firsts[1:] - firsts[:-1]
            indexer = Indexes(counts)
            #
            # Eliminate the locations that are singly labeled
            #
            sijv = sijv[counts[indexer.rev_idx] > 1, :]
            counts = counts[counts > 1]
            if len(counts) == 0:
                return [(ijv_to_segmented(self.__ijv), self.indices)]
            #
            # There are n * n-1 pairs for each coordinate (n = # labels)
            # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
            #
            pairs = all_pairs(np.max(counts))
            pair_counts = counts * (counts - 1)
            #
            # Create an indexer for the inputs (sijv) and for the outputs
            # (first and second of the pairs)
            #
            input_indexer = Indexes(counts)
            output_indexer = Indexes(pair_counts)
            first = sijv[input_indexer.fwd_idx[output_indexer.rev_idx] +
                         pairs[output_indexer.idx[0], 0], 2]
            second = sijv[input_indexer.fwd_idx[output_indexer.rev_idx] +
                          pairs[output_indexer.idx[0], 1], 2]
            #
            # And sort these so that we get consecutive lists for each
            #
            sort_order = np.lexsort((second, first))
            first = first[sort_order]
            second = second[sort_order]
            #
            # Eliminate dupes
            #
            to_keep = np.hstack(([True], 
                                 (first[1:] != first[:-1]) |
                                 (second[1:] != second[:-1])))
            first = first[to_keep]
            second = second[to_keep]
            #
            # Bincount each label so we can find the ones that have the
            # most overlap. See cpmorphology.color_labels and
            # Welsh, "An upper bound for the chromatic number of a graph and
            # its application to timetabling problems", The Computer Journal, 10(1)
            # p 85 (1967)
            #
            overlap_counts = np.bincount(first)
            nlabels = len(self.indices)
            if len(overlap_counts) < nlabels + 1:
                overlap_counts = np.hstack(
                    (overlap_counts, [0] * (nlabels - len(overlap_counts) + 1)))
            #
            # The index to the i'th label's stuff
            #
            indexes = np.cumsum(overlap_counts) - overlap_counts
            #
            # A vector of a current color per label
            #
            v_color = np.zeros(len(overlap_counts), int)
            #
            # Assign all non-overlapping to color 1
            #
            v_color[overlap_counts == 0] = 1
            #
            # Assign all absent objects to color -1
            #
            v_color[1:][self.areas == 0] = -1
            #
            # The processing order is from most overlapping to least
            #
            processing_order = np.lexsort((np.arange(len(overlap_counts)), overlap_counts))
            processing_order = processing_order[overlap_counts[processing_order] > 0]
            
            for index in processing_order:
                neighbors = second[indexes[index]:indexes[index] + overlap_counts[index]]
                colors = np.unique(v_color[neighbors])
                if colors[0] == 0:
                    if len(colors) == 1:
                        # all unassigned - put self in group 1
                        v_color[index] = 1
                        continue
                    else:
                        # otherwise, ignore the unprocessed group and continue
                        colors = colors[1:]
                # Match a range against the colors array - the first place
                # they don't match is the first color we can use
                crange = np.arange(1, len(colors)+1)
                misses = crange[colors != crange]
                if len(misses):
                    color = misses[0]
                else:
                    max_color = len(colors) + 1
                    color = max_color
                v_color[index] = color
            #
            # Now, get ijv groups by color
            #
            result = []
            for color in np.unique(v_color):
                if color == -1:
                    continue
                ijv = self.__ijv[v_color[self.__ijv[:,2]] == color]
                indices = np.arange(1, len(v_color))[v_color[1:] == color]
                result.append((ijv_to_segmented(ijv), indices))
            return result
        else:
            return []
    
    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented != None
    
    def get_unedited_segmented(self):
        """Get the segmentation of the image into objects, including junk that 
        should be ignored: a matrix of object numbers.
        
        The default, if no unedited matrix is available, is the
        segmented labeling.
        """
        if self.__unedited_segmented != None:
            return self.__unedited_segmented
        return self.segmented
    
    def set_unedited_segmented(self,labels):
        check_consistency(self.__segmented, labels, 
                          self.__small_removed_segmented)
        self.__unedited_segmented = downsample_labels(labels)
    
    unedited_segmented = property(get_unedited_segmented, 
                                  set_unedited_segmented)
    
    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented != None
    
    def get_small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed
        
        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
        """
        if self.__small_removed_segmented != None:
            return self.__small_removed_segmented
        return self.unedited_segmented
    
    def set_small_removed_segmented(self,labels):
        check_consistency(self.__segmented, self.__unedited_segmented, labels)
        self.__small_removed_segmented = downsample_labels(labels)
    
    small_removed_segmented = property(get_small_removed_segmented, 
                                       set_small_removed_segmented)
    
    
    def get_parent_image(self):
        """The image that was analyzed to yield the objects.
        
        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image
    
    def set_parent_image(self, parent_image):
        self.__parent_image = parent_image
        
    parent_image = property(get_parent_image, set_parent_image)
    
    def get_has_parent_image(self):
        """True if the objects were derived from a parent image
        
        """
        return self.__parent_image != None
    has_parent_image = property(get_has_parent_image)
    
    def crop_image_similarly(self, image):
        """Crop an image in the same way as the parent image was cropped."""
        if image.shape == self.segmented.shape:
            return image
        if self.parent_image == None:
            raise ValueError("Images are of different size and no parent image")
        return self.parent_image.crop_image_similarly(image)

    def make_ijv_outlines(self, colors):
        '''Make ijv-style color outlines
        
        Make outlines, coloring each object differently to distinguish between
        objects that might overlap.
        
        colors: a N x 3 color map to be used to color the outlines
        '''
        #
        # Get planes of non-overlapping objects. The idea here is to use
        # the most similar colors in the color space for objects that
        # don't overlap.
        #
        all_labels = [(outline(label), indexes) for label, indexes in self.get_labels()]
        image = np.zeros(list(all_labels[0][0].shape) + [3], np.float32)
        #
        # Find out how many unique labels in each
        #
        counts = [np.sum(np.unique(l) != 0) for l, _ in all_labels]
        if len(counts) == 1 and counts[0] == 0:
            return image
        
        if len(colors) < len(all_labels):
            # Have to color 2 planes using the same color!
            # There's some chance that overlapping objects will get
            # the same color. Give me more colors to work with please.
            colors = np.vstack([colors] * (1 + len(all_labels) / len(colors)))
        r = np.random.mtrand.RandomState()
        alpha = np.zeros(all_labels[0][0].shape, np.float32)
        order = np.lexsort([counts])
        label_colors = []
        for idx,i in enumerate(order):
            max_available = len(colors) / (len(all_labels) - idx)
            ncolors = min(counts[i], max_available)
            my_colors = colors[:ncolors]
            colors = colors[ncolors:]
            my_colors = my_colors[r.permutation(np.arange(ncolors))]
            my_labels, indexes = all_labels[i]
            color_idx = np.zeros(np.max(indexes) + 1, int)
            color_idx[indexes] = np.arange(len(indexes)) % ncolors
            image[my_labels != 0,:] += \
                 my_colors[color_idx[my_labels[my_labels != 0]],:]
            alpha[my_labels != 0] += 1
        image[alpha > 0, :] /= alpha[alpha > 0][:, np.newaxis]
        return image

    def relate_children(self, children):
        """Relate the object numbers in one label to the object numbers in another

        children - another "objects" instance: the labels of children within
                   the parent which is "self"

        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        """
        if self.has_ijv or children.has_ijv:
            histogram = self.histogram_from_ijv(self.ijv, children.ijv)
        else:
            histogram = self.histogram_from_labels(self.segmented, children.segmented)
        return self.relate_histogram(histogram)
    
    def relate_labels(self, parent_labels, child_labels):
        '''relate the object numbers in one label to those in another
        
        parent_labels - 2d label matrix of parent labels
        
        child_labels - 2d label matrix of child labels
        
        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        '''
        histogram = self.histogram_from_labels(parent_labels, child_labels)
        return self.relate_histogram(histogram)

    def relate_histogram(self, histogram):
        '''Return child counts and parents of children given a histogram
        
        histogram - histogram from histogram_from_ijv or histogram_from_labels
        '''
        parent_count = histogram.shape[0] - 1
        child_count = histogram.shape[1] - 1

        parents_of_children = np.argmax(histogram, axis=0)
        #
        # Create a histogram of # of children per parent
        children_per_parent = np.histogram(parents_of_children[1:], np.arange(parent_count + 2))[0][1:]

        #
        # Make sure to remove the background elements at index 0
        #
        return children_per_parent, parents_of_children[1:]

    @staticmethod
    def histogram_from_labels(parent_labels, child_labels):
        """Find per pixel overlap of parent labels and child labels

        parent_labels - the parents which contain the children
        child_labels - the children to be mapped to a parent

        Returns a 2d array of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.

        """
        parent_count = np.max(parent_labels)
        child_count = np.max(child_labels)
        #
        # If the labels are different shapes, crop to shared shape.
        #
        common_shape = np.minimum(parent_labels.shape, child_labels.shape)
        parent_labels = parent_labels[0:common_shape[0], 0:common_shape[1]]
        child_labels = child_labels[0:common_shape[0], 0:common_shape[1]]
        #
        # Only look at points that are labeled in parent and child
        #
        not_zero = (parent_labels > 0) & (child_labels > 0)
        not_zero_count = np.sum(not_zero)
        #
        # each row (axis = 0) is a parent
        # each column (axis = 1) is a child
        #
        return coo_matrix((np.ones((not_zero_count,)),
                           (parent_labels[not_zero],
                            child_labels[not_zero])),
                          shape=(parent_count + 1, child_count + 1)).toarray()

    @staticmethod
    def histogram_from_ijv(parent_ijv, child_ijv):
        """Find per pixel overlap of parent labels and child labels,
        stored in ijv format.

        parent_ijv - the parents which contain the children
        child_ijv - the children to be mapped to a parent

        Returns a 2d array of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.

        """
        parent_count = 0 if (parent_ijv.shape[0] == 0) else np.max(parent_ijv[:, 2])
        child_count = 0 if (child_ijv.shape[0] == 0) else np.max(child_ijv[:, 2])

        if parent_count == 0 or child_count == 0:
            return np.zeros((parent_count + 1, child_count + 1), int)

        dim_i = max(np.max(parent_ijv[:, 0]), np.max(child_ijv[:, 0])) + 1
        dim_j = max(np.max(parent_ijv[:, 1]), np.max(child_ijv[:, 1])) + 1
        parent_linear_ij = parent_ijv[:, 0] + dim_i * parent_ijv[:, 1]
        child_linear_ij = child_ijv[:, 0] + dim_i * child_ijv[:, 1]

        parent_matrix = coo_matrix((np.ones((parent_ijv.shape[0],)),
                                    (parent_ijv[:, 2], parent_linear_ij)),
                                   shape=(parent_count + 1, dim_i * dim_j))
        child_matrix = coo_matrix((np.ones((child_ijv.shape[0],)),
                                   (child_linear_ij, child_ijv[:, 2])),
                                  shape=(dim_i * dim_j, child_count + 1))
        # I surely do not understand the sparse code.  Converting both
        # arrays to csc gives the best peformance... Why not p.csr and
        # c.csc?
        return (parent_matrix.tocsc() * child_matrix.tocsc()).toarray()

    @memoize_method
    def get_indices(self):
        """Get the indices for a scipy.ndimage-style function from the segmented labels
        
        """
        if self.__ijv is not None:
            if len(self.__ijv) == 0:
                max_label = 0
            else:
                max_label = np.max(self.__ijv[:,2])
        else:
            max_label = np.max(self.segmented)
        return np.arange(max_label).astype(np.int32) + 1
    
    indices = property(get_indices)
    
    @property
    def count(self):
        """The number of objects labeled"""
        return len(self.indices)
    
    @memoize_method
    def get_areas(self):
        """The area of each object"""
        if len(self.indices) == 0:
            return np.zeros(0, int)
        if self.__ijv is not None:
            return np.bincount(self.__ijv[:,2])[self.indices]
        else:
            return np.bincount(self.segmented.ravel())[self.indices]
     
    areas = property(get_areas)
    @memoize_method
    def fn_of_label(self, function):
        """Call a function taking just a label matrix
        
        function - should have a signature like
            labels - label_matrix
    """
        return function(self.segmented)
    
    @memoize_method
    def fn_of_label_and_index(self, function):
        """Call a function taking a label matrix with the segmented labels
        
        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return function(self.segmented,self.indices)
    
    @memoize_method
    def fn_of_ones_label_and_index(self,function):
        """Call a function taking an image, a label matrix and an index with an image of all ones
        
        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        Pass this function an "image" of all ones, for instance to compute
        a center or an area
        """
    
        return function(np.ones(self.segmented.shape),
                        self.segmented,
                        self.indices)
    
    @memoize_method
    def fn_of_image_label_and_index(self, function, image):
        """Call a function taking an image, a label matrix and an index
        
        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return function(image,
                self.segmented,
                self.indices)

def check_consistency(segmented, unedited_segmented, small_removed_segmented):
    """Check the three components of Objects to make sure they are consistent
    """
    assert segmented == None or np.all(segmented >= 0)
    assert unedited_segmented == None or np.all(unedited_segmented >= 0)
    assert small_removed_segmented == None or np.all(small_removed_segmented >= 0)
    assert segmented == None or segmented.ndim == 2, "Segmented label matrix must have two dimensions, has %d"%(segmented.ndim)
    assert unedited_segmented == None or unedited_segmented.ndim == 2, "Unedited segmented label matrix must have two dimensions, has %d"%(unedited_segmented.ndim)
    assert small_removed_segmented == None or small_removed_segmented.ndim == 2, "Small removed segmented label matrix must have two dimensions, has %d"%(small_removed_segmented.ndim)
    assert segmented == None or unedited_segmented == None or segmented.shape == unedited_segmented.shape, "Segmented %s and unedited segmented %s shapes differ"%(repr(segmented.shape),repr(unedited_segmented.shape))
    assert segmented == None or small_removed_segmented == None or segmented.shape == small_removed_segmented.shape, "Segmented %s and small removed segmented %s shapes differ"%(repr(segmented.shape),repr(small_removed_segmented.shape))
   

class ObjectSet(object):
    """A set of objects.Objects instances.
    
    This class allows you to either refer to an object by name or
    iterate over all available objects.
    """
    
    def __init__(self, can_overwrite = False):
        """Initialize the object set
        
        can_overwrite - True to allow overwriting of a new copy of objects
                        over an old one of the same name (for debugging)
        """
        self.__can_overwrite = can_overwrite
        self.__types_and_instances = {OBJECT_TYPE_NAME:{} }
        
    @property
    def __objects_by_name(self):
        return self.__types_and_instances[OBJECT_TYPE_NAME]
    
    def add_objects(self, objects, name):
        assert isinstance(objects,Objects), "objects must be an instance of CellProfiler.Objects"
        assert ((not self.__objects_by_name.has_key(name)) or
                self.__can_overwrite), "The object, %s, is already in the object set"%(name)
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
    
    def get_types(self):
        '''Get then names of types of per-image set "things"
        
        The object set can store arbitrary types of things other than objects,
        for instance ImageJ data tables. This function returns the thing types
        defined in the object set at this stage of the pipeline.
        '''
        return self.__types_and_instances.keys()
    
    def add_type_instance(self, type_name, instance_name, instance):
        '''Add a named instance of a type
        
        A thing of a given type can be stored in the object set so that
        it can be retrieved by name later in the pipeline. This function adds
        an instance of a type to the object set.
        
        type_name - the name of the instance's type
        instance_name - the name of the instance
        instance - the instance itself
        '''
        if type_name not in self.__types_and_instances:
            self.__types_and_instances[type_name] = {}
        self.__types_and_instances[type_name][instance_name] = instance
        
    def get_type_instance(self, type_name, instance_name):
        '''Get an named instance of a type
        
        type_name - the name of the type of instance
        instance_name - the name of the instance to retrieve
        '''
        if (type_name not in self.__types_and_instance or
            instance_name not in self.__types_and_instances[type_name]):
            return None
        return self.__types_and_instances[type_name][instance_name]

def downsample_labels(labels):
    '''Convert a labels matrix to the smallest possible integer format'''
    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)

def crop_labels_and_image(labels, image):
    '''Crop a labels matrix and an image to the lowest common size
    
    labels - a n x m labels matrix
    image - a 2-d or 3-d image
    
    Assumes that points outside of the common boundary should be masked.
    '''
    min_height = min(labels.shape[0], image.shape[0])
    min_width = min(labels.shape[1], image.shape[1])
    if image.ndim == 2:
        return (labels[:min_height, :min_width], 
                image[:min_height, :min_width])
    else:
        return (labels[:min_height, :min_width], 
                image[:min_height, :min_width,:])
    
def size_similarly(labels, secondary):
    '''Size the secondary matrix similarly to the labels matrix
    
    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).
    
    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    '''
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if (labels.shape[0] <= secondary.shape[0] and
        labels.shape[1] <= secondary.shape[1]):
        if secondary.ndim == 2:
            return (secondary[:labels.shape[0], :labels.shape[1]],
                    np.ones(labels.shape, bool))
        else:
            return (secondary[:labels.shape[0], :labels.shape[1],:],
                    np.ones(labels.shape, bool))
            
    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = np.zeros(list(labels.shape) + list(secondary.shape[2:]),
                      secondary.dtype)
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max,:j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max,:j_max,:] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask

