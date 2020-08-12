import centrosome.index
import centrosome.outline
import numpy


class Segmentation:
    """A segmentation of a space into labeled objects

    Supports overlapping objects and cacheing. Retrieval can be as a
    single plane (legacy), as multiple planes and as sparse ijv.
    """

    SEGMENTED = "segmented"
    UNEDITED_SEGMENTED = "unedited segmented"
    SMALL_REMOVED_SEGMENTED = "small removed segmented"

    def __init__(self, dense=None, sparse=None, shape=None):
        """Initialize the segmentation with either a dense or sparse labeling

        dense - a 6-D labeling with the first axis allowing for alternative
                labelings of the same hyper-voxel.
        sparse - the sparse labeling as a record array with axes from
                 cellprofiler_core.utilities.hdf_dict.HDF5ObjectSet
        shape - the 5-D shape of the imaging site if sparse.
        """

        self.__dense = dense
        self.__sparse = sparse
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = None
            self.__explicit_shape = False

        if dense is not None:
            self.__indices = [numpy.unique(d) for d in dense]
            self.__indices = [idx[1:] if idx[0] == 0 else idx for idx in self.__indices]

    @property
    def shape(self):
        """Get or estimate the shape of the segmentation matrix

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
                self.__shape = tuple(
                    [
                        numpy.max(sparse[axis]) + 2
                        if axis in list(sparse.dtype.fields.keys())
                        else 1
                        for axis in ("c", "t", "z", "y", "x")
                    ]
                )
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
        HDF5ObjectSet.AXIS_[X,Y,Z,C,T] and AXIS_LABELS for the object
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

        The remaining axes are in the order, C, T, Z, Y and X
        """
        if self.__dense is not None:
            return self.__dense, self.__indices

        if not self.has_sparse():
            raise ValueError("Can't find object sparse segmentation.")

        return self.__convert_sparse_to_dense()

    def __convert_dense_to_sparse(self):
        dense, indices = self.get_dense()
        axes = list(("c", "t", "z", "y", "x"))
        axes, shape = [
            [a for a, s in zip(aa, self.shape) if s > 1] for aa in (axes, self.shape)
        ]
        #
        # dense.shape[0] is the overlap-axis - it's usually 1
        # except if there are multiply-labeled pixels and overlapping
        # objects. When collecting the coords, we can discard this axis.
        #
        dense = dense.reshape([dense.shape[0]] + shape)
        coords = numpy.where(dense != 0)
        plane, coords = coords[0], coords[1:]
        if numpy.max(shape) < 2 ** 16:
            coords_dtype = numpy.uint16
        else:
            coords_dtype = numpy.uint32
        if len(plane) > 0:
            labels = dense[tuple([plane] + list(coords))]
            max_label = numpy.max(indices)
            if max_label < 2 ** 8:
                labels_dtype = numpy.uint8
            elif max_label < 2 ** 16:
                labels_dtype = numpy.uint16
            else:
                labels_dtype = numpy.uint32
        else:
            labels = numpy.zeros(0, dense.dtype)
            labels_dtype = numpy.uint8
        dtype = [(axis, coords_dtype) for axis in axes]
        dtype.append(("label", labels_dtype))
        sparse = numpy.core.records.fromarrays(list(coords) + [labels], dtype=dtype)
        self.__sparse = sparse
        return sparse

    def __set_dense(self, dense, indices=None):
        self.__dense = dense
        if indices is not None:
            self.__indices = indices
        else:
            self.__indices = [numpy.unique(d) for d in dense]
            self.__indices = [idx[1:] if idx[0] == 0 else idx for idx in self.__indices]
        return dense, self.__indices

    def __convert_sparse_to_dense(self):
        sparse = self.sparse
        if len(sparse) == 0:
            return self.__set_dense(numpy.zeros([1] + list(self.shape), numpy.uint16))

        #
        # The code below assigns a "color" to each label so that no
        # two labels have the same color
        #
        positional_columns = []
        available_columns = []
        lexsort_columns = []
        for axis in ("c", "t", "z", "y", "x"):
            if axis in list(sparse.dtype.fields.keys()):
                positional_columns.append(sparse[axis])
                available_columns.append(sparse[axis])
                lexsort_columns.insert(0, sparse[axis])
            else:
                positional_columns.append(0)
        labels = sparse["label"]
        lexsort_columns.insert(0, labels)

        sort_order = numpy.lexsort(lexsort_columns)
        n_labels = numpy.max(labels)
        #
        # Find the first of a run that's different from the rest
        #
        mask = (
            available_columns[0][sort_order[:-1]]
            != available_columns[0][sort_order[1:]]
        )
        for column in available_columns[1:]:
            mask = mask | (column[sort_order[:-1]] != column[sort_order[1:]])
        breaks = numpy.hstack(([0], numpy.where(mask)[0] + 1, [len(labels)]))
        firsts = breaks[:-1]
        counts = breaks[1:] - firsts
        #
        # Eliminate the locations that are singly labeled
        #
        mask = counts > 1
        firsts = firsts[mask]
        counts = counts[mask]
        if len(counts) == 0:
            dense = numpy.zeros([1] + list(self.shape), labels.dtype)
            dense[tuple([0] + positional_columns)] = labels
            return self.__set_dense(dense)
        #
        # There are n * n-1 pairs for each coordinate (n = # labels)
        # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
        #
        pairs = centrosome.index.all_pairs(numpy.max(counts))
        pair_counts = counts * (counts - 1)
        #
        # Create an indexer for the inputs (indexes) and for the outputs
        # (first and second of the pairs)
        #
        # Remember idx points into sort_order which points into labels
        # to get the nth label, grouped into consecutive positions.
        #
        output_indexer = centrosome.index.Indexes(pair_counts)
        #
        # The start of the run of overlaps and the offsets
        #
        run_starts = firsts[output_indexer.rev_idx]
        offs = pairs[output_indexer.idx[0], :]
        first = labels[sort_order[run_starts + offs[:, 0]]]
        second = labels[sort_order[run_starts + offs[:, 1]]]
        #
        # And sort these so that we get consecutive lists for each
        #
        pair_sort_order = numpy.lexsort((second, first))
        #
        # Eliminate dupes
        #
        to_keep = numpy.hstack(
            ([True], (first[1:] != first[:-1]) | (second[1:] != second[:-1]))
        )
        to_keep = to_keep & (first != second)
        pair_idx = pair_sort_order[to_keep]
        first = first[pair_idx]
        second = second[pair_idx]
        #
        # Bincount each label so we can find the ones that have the
        # most overlap. See cpmorphology.color_labels and
        # Welsh, "An upper bound for the chromatic number of a graph and
        # its application to timetabling problems", The Computer Journal, 10(1)
        # p 85 (1967)
        #
        overlap_counts = numpy.bincount(first.astype(numpy.int32))
        #
        # The index to the i'th label's stuff
        #
        indexes = numpy.cumsum(overlap_counts) - overlap_counts
        #
        # A vector of a current color per label. All non-overlapping
        # objects are assigned to plane 1
        #
        v_color = numpy.ones(n_labels + 1, int)
        v_color[0] = 0
        #
        # Clear all overlapping objects
        #
        v_color[numpy.unique(first)] = 0
        #
        # The processing order is from most overlapping to least
        #
        ol_labels = numpy.where(overlap_counts > 0)[0]
        processing_order = numpy.lexsort((ol_labels, overlap_counts[ol_labels]))

        for index in ol_labels[processing_order]:
            neighbors = second[indexes[index] : indexes[index] + overlap_counts[index]]
            colors = numpy.unique(v_color[neighbors])
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
            crange = numpy.arange(1, len(colors) + 1)
            misses = crange[colors != crange]
            if len(misses):
                color = misses[0]
            else:
                max_color = len(colors) + 1
                color = max_color
            v_color[index] = color
        #
        # Create the dense matrix by using the color to address the
        # 5-d hyperplane into which we place each label
        #
        dense = numpy.zeros([numpy.max(v_color)] + list(self.shape), labels.dtype)
        slices = tuple([v_color[labels] - 1] + positional_columns)
        dense[slices] = labels
        indices = [numpy.where(v_color == i)[0] for i in range(1, dense.shape[0] + 1)]

        return self.__set_dense(dense, indices)
