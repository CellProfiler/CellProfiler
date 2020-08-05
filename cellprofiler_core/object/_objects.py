import centrosome.index
import centrosome.outline
import numpy
import scipy.ndimage
import scipy.sparse
from numpy.random.mtrand import RandomState

from ._segmentation import Segmentation
from ..utilities.core.object import downsample_labels


class Objects:
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
    def dimensions(self):
        if self.__parent_image:
            return self.__parent_image.dimensions

        shape = self.shape

        return len(shape)

    @property
    def volumetric(self):
        return self.dimensions == 3

    @property
    def masked(self):
        mask = self.parent_image.mask

        return numpy.logical_and(self.segmented, mask)

    @property
    def shape(self):
        dense, _ = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return dense.shape[-2:]

        return dense.shape[-3:]

    @property
    def segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix
        of object numbers.
        """
        return self.__segmentation_to_labels(self.__segmented)

    @segmented.setter
    def segmented(self, labels):
        self.__segmented = self.__labels_to_segmentation(labels)

    @staticmethod
    def __labels_to_segmentation(labels):
        dense = downsample_labels(labels)

        if dense.ndim == 3:
            z, x, y = dense.shape
        else:
            x, y = dense.shape
            z = 1

        dense = dense.reshape((1, 1, 1, z, x, y))

        return Segmentation(dense=dense)

    @staticmethod
    def __segmentation_to_labels(segmentation):
        assert isinstance(
            segmentation, Segmentation
        ), "Operation failed because objects were not initialized"

        dense, indices = segmentation.get_dense()

        assert (
            len(dense) == 1
        ), "Operation failed because objects overlapped. Please try with non-overlapping objects"

        if dense.shape[3] == 1:
            return dense.reshape(dense.shape[-2:])

        return dense.reshape(dense.shape[-3:])

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
        return len(self.indices)

    @property
    def areas(self):
        """The area of each object"""
        if len(self.indices) == 0:
            return numpy.zeros(0, int)

        return numpy.bincount(self.ijv[:, 2])[self.indices]

    def set_ijv(self, ijv, shape=None):
        """Set the segmentation to an IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = numpy.core.records.fromarrays(
            (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
            [("y", ijv.dtype), ("x", ijv.dtype), ("label", ijv.dtype)],
        )
        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])
        self.__segmented = Segmentation(sparse=sparse, shape=shape)

    def get_ijv(self):
        """Get the segmentation in IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = self.__segmented.sparse
        return numpy.column_stack([sparse[axis] for axis in ("y", "x", "label")])

    ijv = property(get_ijv, set_ijv)

    def get_labels(self):
        """Get a set of labels matrices consisting of non-overlapping labels

        In IJV format, a single pixel might have multiple labels. If you
        want to use a labels matrix, you have an ambiguous situation and the
        resolution is to process separate labels matrices consisting of
        non-overlapping labels.

        returns a list of label matrixes and the indexes in each
        """
        dense, indices = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return [(dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]

        return [(dense[i, 0, 0], indices[i]) for i in range(dense.shape[0])]

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
            return self.__segmentation_to_labels(self.__unedited_segmented)

        return self.segmented

    @unedited_segmented.setter
    def unedited_segmented(self, labels):
        self.__unedited_segmented = self.__labels_to_segmentation(labels)

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
            return self.__segmentation_to_labels(self.__small_removed_segmented)

        return self.unedited_segmented

    @small_removed_segmented.setter
    def small_removed_segmented(self, labels):
        self.__small_removed_segmented = self.__labels_to_segmentation(labels)

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
        for segmentation in (
            self.__segmented,
            self.__small_removed_segmented,
            self.__unedited_segmented,
        ):
            if segmentation is not None and not segmentation.has_shape():
                shape = (
                    1,
                    1,
                    1,
                    parent_image.pixel_data.shape[0],
                    parent_image.pixel_data.shape[1],
                )
                segmentation.shape = shape

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
        """Make ijv-style color outlines

        Make outlines, coloring each object differently to distinguish between
        objects that might overlap.

        colors: a N x 3 color map to be used to color the outlines
        """
        #
        # Get planes of non-overlapping objects. The idea here is to use
        # the most similar colors in the color space for objects that
        # don't overlap.
        #
        all_labels = [
            (centrosome.outline.outline(label), indexes)
            for label, indexes in self.get_labels()
        ]
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
            colors = numpy.vstack([colors] * (1 + len(all_labels) // len(colors)))
        r = RandomState()
        alpha = numpy.zeros(all_labels[0][0].shape, numpy.float32)
        order = numpy.lexsort([counts])
        label_colors = []
        for idx, i in enumerate(order):
            max_available = len(colors) / (len(all_labels) - idx)
            ncolors = min(counts[i], max_available)
            my_colors = colors[:ncolors]
            colors = colors[ncolors:]
            my_colors = my_colors[r.permutation(numpy.arange(ncolors))]
            my_labels, indexes = all_labels[i]
            color_idx = numpy.zeros(numpy.max(indexes) + 1, int)
            color_idx[indexes] = numpy.arange(len(indexes)) % ncolors
            image[my_labels != 0, :] += my_colors[
                color_idx[my_labels[my_labels != 0]], :
            ]
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
        if self.volumetric:
            histogram = self.histogram_from_labels(self.segmented, children.segmented)
        else:
            histogram = self.histogram_from_ijv(self.ijv, children.ijv)

        return self.relate_histogram(histogram)

    def relate_labels(self, parent_labels, child_labels):
        """relate the object numbers in one label to those in another

        parent_labels - 2d label matrix of parent labels

        child_labels - 2d label matrix of child labels

        Returns two 1-d arrays. The first gives the number of children within
        each parent. The second gives the mapping of each child to its parent's
        object number.
        """
        histogram = self.histogram_from_labels(parent_labels, child_labels)
        return self.relate_histogram(histogram)

    @staticmethod
    def relate_histogram(histogram):
        """Return child counts and parents of children given a histogram

        histogram - histogram from histogram_from_ijv or histogram_from_labels
        """
        parent_count = histogram.shape[0] - 1

        parents_of_children = numpy.argmax(histogram, axis=0)
        #
        # Create a histogram of # of children per parent
        children_per_parent = numpy.histogram(
            parents_of_children[1:], numpy.arange(parent_count + 2)
        )[0][1:]

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

        if parent_labels.ndim == 3:
            parent_labels = parent_labels[
                0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]
            ]
            child_labels = child_labels[
                0 : common_shape[0], 0 : common_shape[1], 0 : common_shape[2]
            ]
        else:
            parent_labels = parent_labels[0 : common_shape[0], 0 : common_shape[1]]
            child_labels = child_labels[0 : common_shape[0], 0 : common_shape[1]]

        #
        # Only look at points that are labeled in parent and child
        #
        not_zero = (parent_labels > 0) & (child_labels > 0)
        not_zero_count = numpy.sum(not_zero)

        #
        # each row (axis = 0) is a parent
        # each column (axis = 1) is a child
        #
        return scipy.sparse.coo_matrix(
            (
                numpy.ones((not_zero_count,)),
                (parent_labels[not_zero], child_labels[not_zero]),
            ),
            shape=(parent_count + 1, child_count + 1),
        ).toarray()

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
        parent_linear_ij = parent_ijv[:, 0] + dim_i * parent_ijv[:, 1].astype(
            numpy.uint64
        )
        child_linear_ij = child_ijv[:, 0] + dim_i * child_ijv[:, 1].astype(numpy.uint64)

        parent_matrix = scipy.sparse.coo_matrix(
            (numpy.ones((parent_ijv.shape[0],)), (parent_ijv[:, 2], parent_linear_ij)),
            shape=(parent_count + 1, dim_i * dim_j),
        )
        child_matrix = scipy.sparse.coo_matrix(
            (numpy.ones((child_ijv.shape[0],)), (child_linear_ij, child_ijv[:, 2])),
            shape=(dim_i * dim_j, child_count + 1),
        )
        # I surely do not understand the sparse code.  Converting both
        # arrays to csc gives the best peformance... Why not p.csr and
        # c.csc?
        return (parent_matrix.tocsc() * child_matrix.tocsc()).toarray()

    def fn_of_label_and_index(self, func):
        """Call a function taking a label matrix with the segmented labels

        function - should have signature like
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        """
        return func(self.segmented, self.indices)

    def fn_of_ones_label_and_index(self, func):
        """Call a function taking an image, a label matrix and an index with an image of all ones

        function - should have signature like
                   image  - image with same dimensions as labels
                   labels - label matrix
                   index  - sequence of label indices documenting which
                            label indices are of interest
        Pass this function an "image" of all ones, for instance to compute
        a center or an area
        """
        return func(numpy.ones(self.segmented.shape), self.segmented, self.indices)

    def center_of_mass(self):
        labels = self.segmented

        index = numpy.unique(labels)

        if index[0] == 0:
            index = index[1:]

        return numpy.array(
            scipy.ndimage.center_of_mass(numpy.ones_like(labels), labels, index)
        )

    def overlapping(self):
        if not isinstance(self.__segmented, Segmentation):
            return False
        dense, indices = self.__segmented.get_dense()
        return len(dense) != 1
