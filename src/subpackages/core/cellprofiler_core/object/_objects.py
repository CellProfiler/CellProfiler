import numpy as np

from cellprofiler_library.functions.segmentation import convert_dense_to_label_set
from cellprofiler_library.functions.segmentation import convert_sparse_to_ijv
from cellprofiler_library.functions.segmentation import indices_from_ijv
from cellprofiler_library.functions.segmentation import count_from_ijv
from cellprofiler_library.functions.segmentation import areas_from_ijv
from cellprofiler_library.functions.segmentation import convert_labels_to_dense
from cellprofiler_library.functions.segmentation import convert_ijv_to_sparse
from cellprofiler_library.functions.segmentation import make_rgb_outlines
from cellprofiler_library.functions.segmentation import find_label_overlaps
from cellprofiler_library.functions.segmentation import find_ijv_overlaps
from cellprofiler_library.functions.segmentation import center_of_labels_mass

from ._segmentation import Segmentation


class Objects:
    """Represents a segmentation of an image.

    IdentityPrimAutomatic produces three variants of its segmentation
    result. This object contains all three.

    There are three formats for segmentation, two of which support
    overlapping objects:

    get/set_segmented - Legacy, a single plane of labels that does not
                        support overlapping objects.

    (see cellprofiler_library.functions.segmentation._validate_labels)


    get_labels        - Supports overlapping objects, returns one or more planes
                        along with indices. A typical usage is to perform an
                        operation per-plane as if the objects did not overlap.

    (see cellprofiler_library.functions.segmentation.convert_dense_to_label_set)


    get/set_ijv       - Supports overlapping objects, returns a sparse
                        representation in which the first two columns are the
                        coordinates and the last is the object number. This
                        is efficient for doing things like calculating intensity
                        per-object.

    (see cellprofiler_library.functions.segmentation._validate_ijv)


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
        parent_image = self.parent_image

        assert parent_image is not None, "No parent image"

        return np.logical_and(self.segmented, parent_image.mask)

    @property
    def shape(self):
        dense, _ = self.__segmented.get_dense()

        if dense.shape[3] == 1:
            return dense.shape[-2:]

        return dense.shape[-3:]

    def get_segmented(self):
        """Get the de-facto segmentation of the image into objects: a matrix
        of object numbers.
        """
        return self.__segmentation_to_labels(self.__segmented)

    def set_segmented(self, labels):
        self.__segmented = self.__labels_to_segmentation(labels)

    segmented = property(get_segmented, set_segmented)

    @staticmethod
    def __labels_to_segmentation(labels):
        dense = convert_labels_to_dense(labels)
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
        return indices_from_ijv(self.ijv, validate=False)

    @property
    def count(self):
        return count_from_ijv(self.ijv, validate=False)

    @property
    def areas(self):
        return areas_from_ijv(self.ijv, validate=False)

    def set_ijv(self, ijv, shape=None):
        """Set the segmentation to an IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = convert_ijv_to_sparse(ijv)

        if shape is not None:
            shape = (1, 1, 1, shape[0], shape[1])

        self.__segmented = Segmentation(sparse=sparse, shape=shape)

    def get_ijv(self):
        """Get the segmentation in IJV object format

        The ijv format is a list of i,j coordinates in slots 0 and 1
        and the label at the pixel in slot 2.
        """
        sparse = self.__segmented.sparse
        return convert_sparse_to_ijv(sparse, validate=False)

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

        return convert_dense_to_label_set(dense, indices=indices, validate=False)

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
        return make_rgb_outlines(self.get_labels(), colors, validate=True)

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

        parents_of_children = np.asarray(histogram.argmax(axis=0))
        if len(parents_of_children.shape) == 2:
            parents_of_children = np.squeeze(parents_of_children, axis=0)
        #
        # Create a histogram of # of children per parent
        children_per_parent = np.histogram(
            parents_of_children[1:], np.arange(parent_count + 2)
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

        Returns a sparse matrix of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        return find_label_overlaps(parent_labels, child_labels, validate=True)

    @staticmethod
    def histogram_from_ijv(parent_ijv, child_ijv):
        """Find per pixel overlap of parent labels and child labels,
        stored in ijv format.

        parent_ijv - the parents which contain the children
        child_ijv - the children to be mapped to a parent

        Returns a sparse matrix of overlap between each parent and child.
        Note that the first row and column are empty, as these
        correspond to parent and child labels of 0.
        """
        return find_ijv_overlaps(parent_ijv, child_ijv, validate=True)

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
        return func(np.ones(self.segmented.shape), self.segmented, self.indices)

    def center_of_mass(self):
        return center_of_labels_mass(self.segmented, validate=False)

    def overlapping(self):
        if not isinstance(self.__segmented, Segmentation):
            return False
        dense, indices = self.__segmented.get_dense()
        return len(dense) != 1
