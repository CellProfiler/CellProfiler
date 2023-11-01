from cellprofiler_library.functions.segmentation import convert_dense_to_sparse
from cellprofiler_library.functions.segmentation import convert_sparse_to_dense
from cellprofiler_library.functions.segmentation import indices_from_dense
from cellprofiler_library.functions.segmentation import dense_shape_from_sparse
from cellprofiler_library.functions.segmentation import _validate_dense
from cellprofiler_library.functions.segmentation import _validate_sparse
from cellprofiler_library.functions.segmentation import _validate_dense_shape


class Segmentation:
    """A segmentation of a space into labeled objects

    Supports overlapping objects and cacheing. Retrieval can be as a
    single plane (legacy), as multiple planes and as sparse ijv.
    """

    def __init__(self, dense=None, sparse=None, shape=None):
        """Initialize the segmentation with either a dense or sparse labeling

        dense - A 6-d array composed of one or more 5-d integer labelings of
        each hyper-voxel. The dimension order is labeling, c, t, z, y, x.
        Typically, a 2-D non-overlapping segmentation has dimensions of
        1, 1, 1, 1, y, x.

        sparse - a labeling stored in a record data type with each column
                 having a name of "c", "t", "z", "y", "x" or "label".
                 The "label" column is the object number, starting with 1.
                 When "c", "t", "z" are absent, this is the interoperable with
                 an ijv labeling of the pixels, or in the COOrdinate format
                 (as in scipy.sparse.coo_matrix).

        shape - the 5-D shape of the imaging site if sparse. If this is absent
                the shape is inferred from the given coordinates of the sparse
                labeling, which is a size capable of containing an equivilent
                dense representation, but may not be exactly equal to the
                original shape of the imaging site.
        """
        if dense is not None:
            _validate_dense(dense)
        if sparse is not None:
            _validate_sparse(sparse)
        if shape is not None:
            _validate_dense_shape(shape)

        self.__dense = dense
        self.__sparse = sparse
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = None
            self.__explicit_shape = False

        if dense is not None:
            self.__indices = indices_from_dense(dense)

    @property
    def shape(self):
        """Get or estimate the shape of the segmentation matrix
        This is the dense shape, ('c', 't', 'z', 'y', 'x')

        Order of precedence:
        Shape supplied in the constructor
        Shape of the dense representation
        maximum extent of the sparse representation + 1
        """
        if self.__shape is not None:
            return self.__shape
        if self.has_dense():
            self.__shape = self.get_dense()[0].shape[1:]
        else:
            sparse = self.sparse
            if len(sparse) == 0:
                self.__shape = (1, 1, 1, 1, 1)
            else:
                self.__shape = dense_shape_from_sparse(sparse, validate=False)
        return self.__shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of the segmentation array

        shape - the 5D shape of the array

        This fixes the shape of the 5D array for sparse representations
        """
        self.__shape = shape
        self.__explicit_shape = True

    def has_dense(self):
        return self.__dense is not None

    def has_sparse(self):
        return self.__sparse is not None

    def has_shape(self):
        if self.__explicit_shape:
            return True

        return self.has_dense()

    @property
    def sparse(self):
        """Get the sparse representation of the segmentation

        returns a Numpy record array where every row represents
        the labeling of a pixel. The dtype record names are taken from
        HDF5ObjectSet.AXIS_[C,T,Z,Y,X] and AXIS_LABELS for the object
        numbers.
        """
        if self.__sparse is not None:
            return self.__sparse

        if not self.has_dense():
            raise ValueError("Can't find object dense segmentation.")

        return self.__convert_dense_to_sparse()

    def get_dense(self):
        """Get the dense representation of the segmentation

        return the segmentation as a 6-D array and a sequence of arrays of the
        object numbers in each 5-D hyperplane of the segmentation. The first
        axis of the segmentation allows us to assign multiple labels to
        individual pixels. Given a 5-D algorithm, the code typically iterates
        over the first axis:

        for labels in self.get_dense():
            # do something

        The remaining axes are in the order, c, t, z, y and x
        """
        if self.__dense is not None:
            return self.__dense, self.__indices

        if not self.has_sparse():
            raise ValueError("Can't find object sparse segmentation.")

        return self.__convert_sparse_to_dense()

    def __convert_dense_to_sparse(self):
        dense, _ = self.get_dense()
        sparse = convert_dense_to_sparse(dense, validate=False)
        self.__sparse = sparse
        return sparse

    def __set_dense(self, dense, indices=None):
        self.__dense = dense

        if indices is not None:
            self.__indices = indices
        else:
            self.__indices = indices_from_dense(dense)

        return dense, self.__indices

    def __convert_sparse_to_dense(self):
        dense, indices = convert_sparse_to_dense(
            self.sparse, self.shape, validate=False)

        return self.__set_dense(dense, indices)
