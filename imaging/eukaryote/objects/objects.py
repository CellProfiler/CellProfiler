class Objects(object):
    """Represents a segmentation of an image.

    IdentityPrimAutomatic produces three variants of its segmentation
    result. This object contains all three.
    
    There are three formats for segmentation, two of which support
    overlapping objects:
    
    get/set_segmented - legacy, a single plane of labels that does not
                        support overlapping objects
    get/set_labels - supports overlapping objects, returns one or more planes
                     along with indices. A typical usage is to perform an
                     operation per-plane as if the objects did not overlap.
    get/set_ijv    - supports overlapping objects, returns a sparse
                     representation in which the first two columns are the
                     coordinates and the last is the object number. This
                     is efficient for doing things like calculating intensity
                     per-object.
                     
    You can set one of the types and then get any of the types (except that
    get_segmented will raise an exception if objects overlap).
    """
    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None
    
    def get_segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix 
        of object numbers.
        """
        assert isinstance(self.__segmented, Segmentation), \
               "Operation failed because objects were not initialized"
        dense, indices = self.__segmented.get_dense()
        assert len(dense) == 1, "Operation failed because objects overlapped. Please try with non-overlapping objects"
        assert np.all(np.array(dense.shape[1:-2]) == 1), \
               "Operation failed because the segmentation was not 2D"
        return dense.reshape(dense.shape[-2:])
    
    def set_segmented(self,labels):
        dense = downsample_labels(labels)
        dense = dense.reshape((1, 1, 1, 1, dense.shape[0], dense.shape[1]))
        self.__segmented = Segmentation(dense=dense)
            
        # Clear all cached results.
        if getattr(self, "memoize_method_dictionary", False):
            self.memoize_method_dictionary = {}
    
    segmented = property(get_segmented,set_segmented)
    
    def set_ijv(self, ijv, shape=None):
        '''Set the segmentation to an IJV object format
        
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        sparse = np.core.records.fromarrays(
            (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
            [(HDF5ObjectSet.AXIS_Y, ijv.dtype, 1),
             (HDF5ObjectSet.AXIS_X, ijv.dtype, 1),
             (HDF5ObjectSet.AXIS_LABELS, ijv.dtype, 1)])
        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])
        self.__segmented = Segmentation(sparse=sparse, shape=shape)
        
    def get_ijv(self):
        '''Get the segmentation in IJV object format
        
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        sparse = self.__segmented.get_sparse()
        return np.column_stack(
            [sparse[axis] for axis in 
             HDF5ObjectSet.AXIS_Y, HDF5ObjectSet.AXIS_X, 
             HDF5ObjectSet.AXIS_LABELS])
    
    ijv = property(get_ijv, set_ijv)
    
    @property
    def shape(self):
        '''The i and j extents of the labels'''
        return self.__segmented.get_shape()[-2:]
    
    def get_labels(self, shape = None):
        '''Get a set of labels matrices consisting of non-overlapping labels
        
        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.
        
        returns a list of label matrixes and the indexes in each
        '''
        dense, indices = self.__segmented.get_dense()
        return [
            (dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]
    
    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented is not None
    
    def get_unedited_segmented(self):
        """Get the segmentation of the image into objects, including junk that 
        should be ignored: a matrix of object numbers.
        
        The default, if no unedited matrix is available, is the
        segmented labeling.
        """
        if self.__unedited_segmented is not None:
            dense, indices = self.__unedited_segmented.get_dense()
            return dense[0, 0, 0, 0]
        return self.segmented
    
    def set_unedited_segmented(self,labels):
        dense = downsample_labels(labels).reshape(
            (1, 1, 1, 1, labels.shape[0], labels.shape[1]))
        self.__unedited_segmented = Segmentation(dense=dense)
        
    
    unedited_segmented = property(get_unedited_segmented, 
                                  set_unedited_segmented)
    
    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented is not None
    
    def get_small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed
        
        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
        """
        if self.__small_removed_segmented is not None:
            dense, indices = self.__small_removed_segmented.get_dense()
            return dense[0, 0, 0, 0]
        return self.unedited_segmented
    
    def set_small_removed_segmented(self,labels):
        dense = downsample_labels(labels).reshape(
            (1, 1, 1, 1, labels.shape[0], labels.shape[1]))
        self.__small_removed_segmented = Segmentation(dense=dense)
    
    small_removed_segmented = property(get_small_removed_segmented, 
                                       set_small_removed_segmented)
    
    def cache(self, hdf5_object_set, objects_name):
        '''Move the segmentations out of memory and into HDF5
        
        hdf5_object_set - an HDF5ObjectSet attached to an HDF5 file
        objects_name - name of the objects
        '''
        for segmentation, segmentation_name in (
            (self.__segmented, Segmentation.SEGMENTED),
            (self.__unedited_segmented, Segmentation.UNEDITED_SEGMENTED),
            (self.__small_removed_segmented, Segmentation.SMALL_REMOVED_SEGMENTED)):
            if segmentation is not None:
                segmentation.cache(
                    hdf5_object_set, objects_name, segmentation_name)
            
    def get_parent_image(self):
        """The image that was analyzed to yield the objects.
        
        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image
    
    def set_parent_image(self, parent_image):
        self.__parent_image = parent_image
        for segmentation in self.__segmented, self.__small_removed_segmented,\
            self.__unedited_segmented:
            if segmentation is not None and not segmentation.has_shape():
                shape = (1, 1, 1, 
                         parent_image.pixel_data.shape[0],
                         parent_image.pixel_data.shape[1])
                segmentation.set_shape(shape)
        
    parent_image = property(get_parent_image, set_parent_image)
    
    def get_has_parent_image(self):
        """True if the objects were derived from a parent image
        
        """
        return self.__parent_image is not None
    has_parent_image = property(get_has_parent_image)
    
    def crop_image_similarly(self, image):
        """Crop an image in the same way as the parent image was cropped."""
        if image.shape == self.segmented.shape:
            return image
        if self.parent_image is None:
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
        histogram = self.histogram_from_ijv(self.ijv, children.ijv)
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
        parent_linear_ij = parent_ijv[:, 0] +\
            dim_i * parent_ijv[:, 1].astype(np.uint64)
        child_linear_ij = child_ijv[:, 0] +\
            dim_i * child_ijv[:, 1].astype(np.uint64)

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
        if len(self.ijv) == 0:
            return np.zeros(0, np.int32)
        max_label = np.max(self.ijv[:, 2])
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
        return np.bincount(self.ijv[:,2])[self.indices]
     
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
