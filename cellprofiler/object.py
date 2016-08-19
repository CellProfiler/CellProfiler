""" CellProfiler.Objects.py - represents a labelling of objects in an image
"""

import centrosome.index
import centrosome.outline
import numpy
import scipy.sparse

OBJECT_TYPE_NAME = "objects"


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
        self.parent_image = None
        self.__ijv = (None, None)

    @property
    def segmented(self):
        if self.__segmented is not None:
            return self.__segmented

        labels = self.get_labels()

        assert(len(labels) == 1, "Overlapping labels!!!")  # this is why ijv should not set segmented!

        return labels[0][0]

    @segmented.setter
    def segmented(self, labels):
        self.__segmented = self.__downsample_labels(labels)

    # TODO: Can this be removed?
    @property
    def unedited_segmented(self):
        if self.__unedited_segmented is not None:
            return self.__unedited_segmented

        return self.segmented

    @unedited_segmented.setter
    def unedited_segmented(self, labels):
        self.__unedited_segmented = self.__downsample_labels(labels)

    # TODO: Can this be removed?
    @property
    def small_removed_segmented(self):
        if self.__small_removed_segmented is not None:
            return self.__small_removed_segmented
        elif self.__unedited_segmented is not None:
            return self.__unedited_segmented
        else:
            return self.segmented

    @small_removed_segmented.setter
    def small_removed_segmented(self, labels):
        self.__small_removed_segmented = self.__downsample_labels(labels)

    # TODO: rename to cellprofiler.object.labels_to_coordinates (or similar)
    def get_ijv(self):
        '''Get the segmentation in IJV object format
        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''

        if self.__ijv[0] is not None:
            return self.__ijv[0]
        elif numpy.all(self.__segmented == 0):
            return numpy.zeros((0, 3), dtype=numpy.uint16)

        x, y = numpy.nonzero(self.__segmented)
        values = self.__segmented[x, y]

        return numpy.asarray(zip(x, y, values))

    # TODO: rename to cellprofiler.object.coordinates_to_labels (or similar)
    def set_ijv(self, ijv, shape=None):
        '''Set the segmentation to an IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        '''
        self.__ijv = (ijv, shape)

    ijv = property(get_ijv, set_ijv)

    @property
    def shape(self):
        '''The i and j extents of the labels'''
        if self.__segmented is not None:
            return self.__segmented.shape

        return self.__shape_from_ijv()

    def get_labels(self, shape=None):
        '''Get a set of labels matrices consisting of non-overlapping labels

        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.

        returns a list of label matrixes and the indexes in each
        '''
        if self.__segmented is not None:
            return [(self.__segmented, numpy.unique(self.__segmented))]

        # Convert IJV to label matrix
        ijv, ijv_shape = self.__ijv
        x = ijv[:, 0]
        y = ijv[:, 1]

        if shape is None:
            shape = self.__shape_from_ijv()

        # TODO: vectorize me
        # The tests expect the minimum number of labeling matrices to be returned. Can be simplified if we store
        # matrices once a collision is detected and start on a new one (i.e., don't iterate through a list of labeling
        # matrices to find one this label fits in).
        labeling_matrices = [numpy.zeros(shape)]
        for label in numpy.unique(ijv[:, 2]):
            # Get all points with that label.
            points = ijv[ijv[:, 2] == label]
            px = points[:, 0]
            py = points[:, 1]

            # Determine if any points overlap with already labeled indices.
            overlapping = True
            for labeling_matrix in labeling_matrices:
                if numpy.all(labeling_matrix[px, py] == 0):
                    # Add the label
                    labeling_matrix[px, py] = label

                    # This label does not overlap. Go on to next label.
                    overlapping = False
                    break

            if overlapping:
                labeling_matrix = numpy.zeros(shape)
                labeling_matrix[px, py] = label
                labeling_matrices.append(labeling_matrix)

        # TODO: Can we return only the list of labeling matrices?
        result = [(
                      self.__downsample_labels(labeling_matrix),
                      numpy.asarray(numpy.unique(labeling_matrix)[1:], dtype=int)
                  ) for labeling_matrix in labeling_matrices]

        return result

    def has_unedited_segmented(self):
        """Return true if there is an unedited segmented matrix."""
        return self.__unedited_segmented is not None

    def has_small_removed_segmented(self):
        """Return true if there is a junk object matrix."""
        return self.__small_removed_segmented is not None

    @property
    def has_parent_image(self):
        """True if the objects were derived from a parent image

        """
        return self.parent_image is not None

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
        all_labels = [(centrosome.outline.outline(label), indexes) for label, indexes in self.get_labels()]

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

            image[my_labels != 0, :] += my_colors[color_idx[my_labels[my_labels != 0]], :]

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

    def fn_of_label_and_index(self, function):
        """Call a function taking a label matrix with the segmented labels

        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return function(self.__segmented, self.indices)

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
        return function(numpy.ones(self.__segmented.shape), self.__segmented, self.indices)

    def __downsample_labels(self, labels):
        if labels.size == 0:
            return labels

        '''Convert a labels matrix to the smallest possible integer format'''
        labels_max = numpy.max(labels)
        if labels_max < 128:
            return labels.astype(numpy.int8)
        elif labels_max < 32768:
            return labels.astype(numpy.int16)
        return labels.astype(numpy.int32)

    def __shape_from_ijv(self):
        ijv, shape = self.__ijv

        if shape is not None:
            return shape
        elif self.has_parent_image:
            return self.parent_image.pixel_data.shape
        elif numpy.all(numpy.array(ijv) == 0):
            return 1, 1

        x = ijv[:, 0]
        y = ijv[:, 1]

        return numpy.max(x) + 2, numpy.max(y) + 2  # whyyyyy???

def check_consistency(segmented, unedited_segmented, small_removed_segmented):
    """Check the three components of Objects to make sure they are consistent
    """
    assert segmented is None or numpy.all(segmented >= 0)
    assert unedited_segmented is None or numpy.all(unedited_segmented >= 0)
    assert small_removed_segmented is None or numpy.all(small_removed_segmented >= 0)
    assert segmented is None or segmented.ndim == 2, "Segmented label matrix must have two dimensions, has %d" % (
        segmented.ndim)
    assert unedited_segmented is None or unedited_segmented.ndim == 2, "Unedited segmented label matrix must have two dimensions, has %d" % (
        unedited_segmented.ndim)
    assert small_removed_segmented is None or small_removed_segmented.ndim == 2, "Small removed segmented label matrix must have two dimensions, has %d" % (
        small_removed_segmented.ndim)
    assert segmented is None or unedited_segmented is None or segmented.shape == unedited_segmented.shape, "Segmented %s and unedited segmented %s shapes differ" % (
        repr(segmented.shape), repr(unedited_segmented.shape))
    assert segmented is None or small_removed_segmented is None or segmented.shape == small_removed_segmented.shape, "Segmented %s and small removed segmented %s shapes differ" % (
        repr(segmented.shape), repr(small_removed_segmented.shape))


class ObjectSet(object):
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
        assert isinstance(objects, Objects), "objects must be an instance of CellProfiler.Objects"
        assert ((not self.__objects_by_name.has_key(name)) or
                self.__can_overwrite), "The object, %s, is already in the object set" % name
        self.__objects_by_name[name] = objects

    def get_object_names(self):
        """Return the names of all of the objects
        """
        return self.__objects_by_name.keys()

    object_names = property(get_object_names)

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
