import centrosome.index
import centrosome.outline
import numpy
import numpy.random
import scipy.sparse
import cellprofiler.segmentation

OBJECT_TYPE_NAME = "objects"


# TODO: remove getter/setter on ivj and instead make method coordinates which converts label matrix to/from coordinate (x, y, label) or (x, y, z, label).
# TODO: there is no set_labels. Maybe make segmented (or labels) have the same functionality (support overlapping objects).
# TODO: remove the concept of shaping. Modules should shape then set. This will make for cleaner setters.
# TODO: remove hdf5
class Region(object):
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

    @property
    def segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix
        of object numbers.
        """
        assert isinstance(self.__segmented, cellprofiler.segmentation.Segmentation), "Operation failed because objects were not initialized"

        dense, indices = self.__segmented.dense

        assert len(dense) == 1, "Operation failed because objects overlapped. Please try with non-overlapping objects"

        assert numpy.all(numpy.array(dense.shape[1:-2]) == 1), "Operation failed because the segmentation was not 2D"

        return dense.reshape(dense.shape[-2:])

    @segmented.setter
    def segmented(self, labels):
        dense = downsample_labels(labels)

        dense = dense.reshape((1, 1, 1, 1, dense.shape[0], dense.shape[1]))

        self.__segmented = cellprofiler.segmentation.Segmentation(dense=dense)

    def set_ijv(self, ijv, shape=None):
        '''Set the segmentation to an IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        sparse = numpy.core.records.fromarrays(
                (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
                [(HDF5ObjectSet.AXIS_Y, ijv.dtype, 1),
                 (HDF5ObjectSet.AXIS_X, ijv.dtype, 1),
                 (HDF5ObjectSet.AXIS_LABELS, ijv.dtype, 1)])
        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])
        self.__segmented = cellprofiler.segmentation.Segmentation(sparse=sparse, shape=shape)

    def get_ijv(self):
        '''Get the segmentation in IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        sparse = self.__segmented.sparse
        return numpy.column_stack(
                [sparse[axis] for axis in
                 HDF5ObjectSet.AXIS_Y, HDF5ObjectSet.AXIS_X,
                 HDF5ObjectSet.AXIS_LABELS])

    ijv = property(get_ijv, set_ijv)

    @property
    def shape(self):
        '''The i and j extents of the labels'''
        return self.__segmented.shape()[-2:]

    # TODO: Use or remove shape parameter
    # TODO: Make property?
    def labels(self, shape=None):
        """Get a set of labels matrices consisting of non-overlapping labels

        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.

        returns a list of label matrices and the indexes in each
        """
        dense, indices = self.__segmented.dense
        return [(dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]

    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented is not None

    @property
    def unedited_segmented(self):
        """Get the segmentation of the image into objects, including junk that
        should be ignored: a matrix of object numbers.

        The default, if no unedited matrix is available, is the
        segmented labeling.
        """
        if self.__unedited_segmented is not None:
            dense, indices = self.__unedited_segmented.dense
            return dense[0, 0, 0, 0]
        return self.segmented

    @unedited_segmented.setter
    def unedited_segmented(self, labels):
        dense = downsample_labels(labels).reshape(
                (1, 1, 1, 1, labels.shape[0], labels.shape[1]))
        self.__unedited_segmented = cellprofiler.segmentation.Segmentation(dense=dense)

    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented is not None

    @property
    def small_removed_segmented(self):
        """Get the matrix of segmented objects with the small objects removed

        This should be the same as the unedited_segmented label matrix with
        the small objects removed, but objects touching the sides of the image
        or the image mask still present.
        """
        if self.__small_removed_segmented is not None:
            dense, indices = self.__small_removed_segmented.dense
            return dense[0, 0, 0, 0]
        return self.unedited_segmented

    @small_removed_segmented.setter
    def small_removed_segmented(self, labels):
        dense = downsample_labels(labels).reshape(
                (1, 1, 1, 1, labels.shape[0], labels.shape[1]))
        self.__small_removed_segmented = cellprofiler.segmentation.Segmentation(dense=dense)

    @property
    def parent_image(self):
        """The image that was analyzed to yield the objects.

        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image

    @parent_image.setter
    def parent_image(self, parent_image):
        self.__parent_image = parent_image
        for segmentation in self.__segmented, self.__small_removed_segmented, self.__unedited_segmented:
            if segmentation is not None and not segmentation.has_shape():
                shape = (1, 1, 1,
                         parent_image.pixel_data.shape[0],
                         parent_image.pixel_data.shape[1])
                segmentation.shape(shape)

    @property
    def has_parent_image(self):
        """True if the objects were derived from a parent image

        """
        return self.__parent_image is not None

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
        all_labels = [(centrosome.outline.outline(label), indexes) for label, indexes in self.labels()]
        image = numpy.zeros(list(all_labels[0][0].shape) + [3], numpy.float32)
        #
        # Find out how many unique labels in each
        #
        counts = [numpy.sum(numpy.unique(l) != 0) for l, _ in all_labels]
        if len(counts) == 1 and counts[0] == 0:
            return image

        if len(colors) < len(all_labels):
            # Have to color 2 planes using the same color!
            # There's some chance that overlapping objects will get
            # the same color. Give me more colors to work with please.
            colors = numpy.vstack([colors] * (1 + len(all_labels) / len(colors)))

        r = numpy.random.mtrand.RandomState()
        alpha = numpy.zeros(all_labels[0][0].shape, numpy.float32)
        order = numpy.lexsort([counts])

        for idx, i in enumerate(order):
            max_available = len(colors) / (len(all_labels) - idx)
            ncolors = min(counts[i], max_available)
            my_colors = colors[:ncolors]
            colors = colors[ncolors:]
            my_colors = my_colors[r.permutation(numpy.arange(ncolors))]
            my_labels, indexes = all_labels[i]
            color_idx = numpy.zeros(numpy.max(indexes) + 1, int)
            color_idx[indexes] = numpy.arange(len(indexes)) % ncolors
            image[my_labels != 0, :] += \
                my_colors[color_idx[my_labels[my_labels != 0]], :]
            alpha[my_labels != 0] += 1

        image[alpha > 0, :] /= alpha[alpha > 0][:, numpy.newaxis]

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

        parents_of_children = numpy.argmax(histogram, axis=0)
        #
        # Create a histogram of # of children per parent
        children_per_parent = numpy.histogram(parents_of_children[1:], numpy.arange(parent_count + 2))[0][1:]

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
        parent_count = numpy.max(parent_labels)
        child_count = numpy.max(child_labels)
        #
        # If the labels are different shapes, crop to shared shape.
        #
        common_shape = numpy.minimum(parent_labels.shape, child_labels.shape)
        parent_labels = parent_labels[0:common_shape[0], 0:common_shape[1]]
        child_labels = child_labels[0:common_shape[0], 0:common_shape[1]]
        #
        # Only look at points that are labeled in parent and child
        #
        not_zero = (parent_labels > 0) & (child_labels > 0)
        not_zero_count = numpy.sum(not_zero)
        #
        # each row (axis = 0) is a parent
        # each column (axis = 1) is a child
        #
        return scipy.sparse.coo_matrix((numpy.ones((not_zero_count,)),
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
        parent_count = 0 if (parent_ijv.shape[0] == 0) else numpy.max(parent_ijv[:, 2])
        child_count = 0 if (child_ijv.shape[0] == 0) else numpy.max(child_ijv[:, 2])

        if parent_count == 0 or child_count == 0:
            return numpy.zeros((parent_count + 1, child_count + 1), int)

        dim_i = max(numpy.max(parent_ijv[:, 0]), numpy.max(child_ijv[:, 0])) + 1
        dim_j = max(numpy.max(parent_ijv[:, 1]), numpy.max(child_ijv[:, 1])) + 1
        parent_linear_ij = parent_ijv[:, 0] + \
                           dim_i * parent_ijv[:, 1].astype(numpy.uint64)
        child_linear_ij = child_ijv[:, 0] + \
                          dim_i * child_ijv[:, 1].astype(numpy.uint64)

        parent_matrix = scipy.sparse.coo_matrix((numpy.ones((parent_ijv.shape[0],)),
                                                 (parent_ijv[:, 2], parent_linear_ij)),
                                                shape=(parent_count + 1, dim_i * dim_j))
        child_matrix = scipy.sparse.coo_matrix((numpy.ones((child_ijv.shape[0],)),
                                                (child_linear_ij, child_ijv[:, 2])),
                                               shape=(dim_i * dim_j, child_count + 1))
        # I surely do not understand the sparse code.  Converting both
        # arrays to csc gives the best peformance... Why not p.csr and
        # c.csc?
        return (parent_matrix.tocsc() * child_matrix.tocsc()).toarray()

    @property
    def indices(self):
        """Get the indices for a scipy.ndimage-style function from the segmented labels

        """
        if len(self.ijv) == 0:
            return numpy.zeros(0, numpy.int32)
        max_label = numpy.max(self.ijv[:, 2])
        return numpy.arange(max_label).astype(numpy.int32) + 1

    @property
    def count(self):
        """The number of objects labeled"""
        return len(self.indices)

    @property
    def areas(self):
        """The area of each object"""
        if len(self.indices) == 0:
            return numpy.zeros(0, int)
        return numpy.bincount(self.ijv[:, 2])[self.indices]

    def fn_of_label(self, function):
        """Call a function taking just a label matrix

        function - should have a signature like
            labels - label_matrix
        """
        return function(self.segmented)

    def fn_of_label_and_index(self, function):
        """Call a function taking a label matrix with the segmented labels

        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return function(self.segmented, self.indices)

    def fn_of_ones_label_and_index(self, function):
        """Call a function taking an image, a label matrix and an index with an image of all ones

        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        Pass this function an "image" of all ones, for instance to compute
        a center or an area
        """

        return function(numpy.ones(self.segmented.shape),
                        self.segmented,
                        self.indices)

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


class Set(object):
    """A set of objects.Objects instances.

    This class allows you to either refer to an object by name or
    iterate over all available objects.
    """

    def __init__(self, can_overwrite=False):
        """Initialize the object set

        can_overwrite - True to allow overwriting of a new copy of objects
                        over an old one of the same name (for debugging)
        """
        self.__can_overwrite = can_overwrite
        self.__types_and_instances = {OBJECT_TYPE_NAME: {}}

    @property
    def __objects_by_name(self):
        return self.__types_and_instances[OBJECT_TYPE_NAME]

    def add_objects(self, objects, name):
        assert isinstance(objects, Region), "objects must be an instance of CellProfiler.Objects"
        assert ((not self.__objects_by_name.has_key(name)) or self.__can_overwrite), "The object, %s, is already in the object set" % name
        self.__objects_by_name[name] = objects

    @property
    def object_names(self):
        """Return the names of all of the objects
        """
        return self.__objects_by_name.keys()

    def get_objects(self, name):
        """Return the objects instance with the given name
        """
        return self.__objects_by_name[name]

    @property
    def all_objects(self):
        """Return a list of name / objects tuples
        """
        return self.__objects_by_name.items()

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
    labels_max = numpy.max(labels)
    if labels_max < 128:
        return labels.astype(numpy.int8)
    elif labels_max < 32768:
        return labels.astype(numpy.int16)
    return labels.astype(numpy.int32)


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
                image[:min_height, :min_width, :])


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
        return secondary, numpy.ones(secondary.shape, bool)
    if (labels.shape[0] <= secondary.shape[0] and
                labels.shape[1] <= secondary.shape[1]):
        if secondary.ndim == 2:
            return (secondary[:labels.shape[0], :labels.shape[1]],
                    numpy.ones(labels.shape, bool))
        else:
            return (secondary[:labels.shape[0], :labels.shape[1], :],
                    numpy.ones(labels.shape, bool))

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = numpy.zeros(list(labels.shape) + list(secondary.shape[2:]),
                         secondary.dtype)
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = numpy.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask
