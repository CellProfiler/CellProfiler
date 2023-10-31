import numpy as np
import centrosome.index

def _validate_dense(dense):
    """
    A 'dense' matrix is a 6 dimensional array with axis order:
    (label_idx, c, t, z, y, x)

    When the 'label_idx' dim = 1, it hosts zero or more non-overlapping labels
    When the 'label_idx' dim > 1, each index hosts one or more non-overlapping
    labels (within that index)
    In other words, while labels within an index of 'label_idx' are never
    overlapping, labels between indices of 'label_idx' would overlap
    i.e. 'dense.sum(axis=0)' is invalid, producing innaccurate labels

    A 'dense' matrix is usually paired with an array of indices specifying
    which label values are present in which index of the 'label_idx' dim
    (see 'indices_from_dense' for more details)
    """
    assert type(dense == np.ndarray), "dense must be ndarray"
    assert dense.ndim == 6, \
    "dense must be 6-dimensional - (label_idx, c, t, z, y, x)"

def _validate_dense_shape(dense_shape):
    """
    'dense_shape', as opposed to 'dense.shape', is the shape of the 'dense'
    matrix sans the 'label_idx' axis, i.e.
    (c, t, z, y, z)
    """
    assert (dense_shape is None or
            len(dense_shape) == 5
    ), "dense_shape must be length 5, omitting 'label_idx' dim"

def _validate_labels(labels):
    """
    A 'labels' matrix is another, more constrained, dense representation

    It is strictly 2- or 3-dimensional, of shape: (y, x) or (z, y, x)
    It does not allow for overlapping labels

    It is essentially a 'dense' of shape (1, 1, 1, 1, y, x), but squeezed
    such that the ('label_idx', 'c', 't', 'z') axes are removed
    """
    assert type(labels) == np.ndarray, "labels must be ndarray"
    assert (
        labels.ndim == 2 or
        labels.ndim == 3
    ), "labels must be 2- or 3-dimensional"

def _validate_sparse(sparse):
    """
    'sparse' is a sparse representation of labelings
    It's either a numpy recarray, or castable as such via
    'arr.view(np.recarray)'
    where the data types are typed fields who's names are a subset of:
    set('label', 'c', 't', 'z', 'y', 'x')
    and where the data is a 1-dimensional array of tuples, matching the fields

    e.g.
    rec.array([(0, 0, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1),
               (0, 1, 0, 2), (0, 1, 1, 2), (1, 1, 0, 2), (1, 1, 1, 2)],
              dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')])

    Note that each field may have its own type, the tuple order matches the
    field order, each tuple is unique by at least one element, and there is
    no need (although permissable) to specify a field if it will be the same
    among all the data values (i.e. 'c' and 't' not specified above as they
    would all be '0')

    Note also there is no tuple who's 'label' value is '0' because including
    all '0' values would make 'sparse' equivilent in memory to 'dense',
    and including any '0' values at all would be including a non-label
    (i.e. background)
    """
    assert (
        type(sparse) == np.ndarray or
        type(sparse) == np.recarray
    ), "sparse must be ndarray or recarray"

    assert sparse.ndim == 1, "sparse mut be 1-dimensional"

    axes = sparse.dtype.names

    assert axes is not None, "sparse must have dtype fields"

    axes_set = set(axes)
    full_set = set(('label', 'c', 't', 'z', 'y', 'x'))

    assert len(axes) == len(axes_set), "duplicate dtype fields in sparse"
    assert axes_set.issubset(full_set), "sparse has unknown dtype fields"

def _validate_ijv(ijv):
    """
    'ijv' is another, more constrained, sparse representation

    It is a 2-dimensional array of triplets, of shape: (num_coords, 3)
    It also allows for overlapping labels

    Unlike 'sparse', it is strictly an ndarray with a single dtype
    for all values, and is strictly in triplet form, where
    [i, j, v] = [y_coord, x_coord, label_value]
    It cannot host values for 'c', 't' or 'z'

    e.g. this 'sparse' record:
    rec.array([(0, 0, 0, 1), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 1, 2),],
             dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')])

    would equate to this 'ijv' matrix:
    array([[0, 0, 1],
           [1, 0, 1],
           [1, 0, 2],
           [1, 1, 2]],
          dtype=uint16)
    """
    assert type(ijv) == np.ndarray, "ijv must be ndarray"
    assert ijv.ndim == 2, "ijv must be 2-dimensional"
    assert ijv.shape[1] == 3, "ijv must have 3 columns"

def indices_from_dense(dense):
    """
    Retrieve indices of a 'dense' matrix

    The return value is a python list of 1-dimensional ndarrays
    whose lengths may or may not be equal (non-homogeneous)

    A given index of the 'indices' list corresponds to the same index in the
    'label_idx' axis of the 'dense' matrix
    An element of the 'indices' list is a 1-dimensional ndarray of labels that
    are present in the 'dense' matrix, at that index

    e.g. the following 'indices' list:
    [array([1, 2, 4], dtype=uint8), array([3, 5], dtype=uint8)]

    specifies that its corresponding 'dense' matrix has a 'label_idx' dim = 2,
    i.e. a shape of: (2, c, t, z, y, x),
    where the labels '1', '2', '4' are present at index '0', and the labels '3'
    and '4' are present at index '1'
    """
    indices = [np.unique(d) for d in dense]
    indices = [idx[1:] if idx[0] == 0 else idx for idx in indices]
    return indices

def dense_shape_from_sparse(sparse):
    return tuple([
        np.max(sparse[axis]) + 2
        if axis in list(sparse.dtype.fields.keys())
        else 1
        for axis in ('c', 't', 'z', 'y', 'x')
    ])

def indices_from_ijv(ijv):
    """
    TODO - 4758: not sure 'indices' is the correct name,
    it just returns the set of labels

    Get the indices for a scipy.ndimage-style function from the 'ijv' formatted
    segmented labels
    """
    _validate_ijv(ijv)

    if len(ijv) == 0:
        return np.zeros(0, np.int32)

    max_label = np.max(ijv[:, 2])

    return np.arange(max_label).astype(np.int32) + 1

def count_from_ijv(ijv, indices=None):
    """
    The number of labels in an 'ijv' formatted label matrix
    """
    if indices is None:
        indices = indices_from_ijv(ijv)

    return len(indices)

def areas_from_ijv(ijv, indices=None):
    """
    The area of each object in an 'ijv' formatted label matrix

    Because of the discrete nature of the matrix, this is simply equal to the
    occurrence count of each label in the matrix
    """
    if indices is None:
        indices = indices_from_ijv(ijv)

    if len(indices) == 0:
        return np.zeros(0, int)

    return np.bincount(ijv[:, 2])[indices]

def downsample_labels(labels):
    """
    Convert a 'labels' matrix to the smallest possible integer format
    """
    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)

def convert_dense_to_label_set(dense, indices=None):
    """
    Convert a 'dense' matrix into a list of 2-tuples,
    where the number of tuples corresponds to the number of unique labels
    in the 'dense' matrix, the tuple's first element is a 'labels' matrix,
    and the tuple's second element is the single label which the 'labels' matrix
    represents
    """
    _validate_dense(dense)

    if indices is None:
        indices = indices_from_dense(dense)

    # z = 1 => 2-D
    if dense.shape[3] == 1:
        return [(dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]

    return [(dense[i, 0, 0], indices[i]) for i in range(dense.shape[0])]

def convert_labels_to_dense(labels):
    """
    Convert a 'labels' matrix (e.g. scipy.ndimage.label) to 'dense' matrix
    """
    _validate_labels(labels)

    dense = downsample_labels(labels)

    if dense.ndim == 3:
        z, y, x = dense.shape
    else:
        y, x = dense.shape
        z = 1

    return dense.reshape((1, 1, 1, z, y, x))


def convert_dense_to_sparse(dense):
    _validate_dense(dense)

    label_dim = dense.shape[0]
    dense_shape = dense.shape[1:]

    axes_labels = np.array(('c', 't', 'z', 'y', 'x'))
    axes = axes_labels[np.where(np.array(dense_shape) > 1)]

    compact = np.squeeze(dense)
    if label_dim == 1:
        compact = np.expand_dims(compact, axis=0)

    coords = np.where(compact != 0)
    labels = compact[coords]
    # no longer need the labels dim
    coords = coords[1:]

    if np.max(compact.shape) < 2 ** 16:
        coords_dtype = np.uint16
    else:
        coords_dtype = np.uint32

    if len(labels) > 0:
        max_label = np.max(labels)
        if max_label < 2 ** 8:
            labels_dtype = np.uint8
        elif max_label < 2 ** 16:
            labels_dtype = np.uint16
        else:
            labels_dtype = np.uint32
    else:
        labels_dtype = np.uint8

    dtype = [(axis, coords_dtype) for axis in axes]
    dtype.append(('label', labels_dtype))
    sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)

    return sparse

def convert_ijv_to_sparse(ijv):
    _validate_ijv(ijv)

    return np.core.records.fromarrays(
        (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
        [('y', ijv.dtype), ('x', ijv.dtype), ('label', ijv.dtype)],
    )

def convert_sparse_to_ijv(sparse):
    _validate_sparse(sparse)

    return np.column_stack([sparse[axis] for axis in ('y', 'x', 'label')])

def convert_sparse_to_dense(sparse, dense_shape=None):
    _validate_sparse(sparse)
    _validate_dense_shape(dense_shape)

    if len(sparse) == 0:
        if dense_shape is None:
            full_dense_shape = (1,1,1,1,1,1)
        else:
            full_dense_shape = (1,) + dense_shape

        dense = np.zeros(full_dense_shape, np.uint8)
        return dense, indices_from_dense(dense)

    if dense_shape is None:
        # len 5
        dense_shape = dense_shape_from_sparse(sparse)

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
    labels = sparse['label']
    lexsort_columns.insert(0, labels)

    sort_order = np.lexsort(lexsort_columns)
    n_labels = np.max(labels)
    #
    # Find the first of a run that's different from the rest
    #
    mask = (
        available_columns[0][sort_order[:-1]]
        != available_columns[0][sort_order[1:]]
    )
    for column in available_columns[1:]:
        mask = mask | (column[sort_order[:-1]] != column[sort_order[1:]])
    breaks = np.hstack(([0], np.where(mask)[0] + 1, [len(labels)]))
    firsts = breaks[:-1]
    counts = breaks[1:] - firsts
    #
    # Eliminate the locations that are singly labeled
    #
    mask = counts > 1
    firsts = firsts[mask]
    counts = counts[mask]
    if len(counts) == 0:
        dense = np.zeros([1] + list(dense_shape), labels.dtype)
        dense[tuple([0] + positional_columns)] = labels
        return dense, indices_from_dense(dense)
    #
    # There are n * n-1 pairs for each coordinate (n = # labels)
    # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
    #
    pairs = centrosome.index.all_pairs(np.max(counts))
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
    pair_sort_order = np.lexsort((second, first))
    #
    # Eliminate dupes
    #
    to_keep = np.hstack(
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
    overlap_counts = np.bincount(first.astype(np.int32))
    #
    # The index to the i'th label's stuff
    #
    indexes = np.cumsum(overlap_counts) - overlap_counts
    #
    # A vector of a current color per label. All non-overlapping
    # objects are assigned to plane 1
    #
    v_color = np.ones(n_labels + 1, int)
    v_color[0] = 0
    #
    # Clear all overlapping objects
    #
    v_color[np.unique(first)] = 0
    #
    # The processing order is from most overlapping to least
    #
    ol_labels = np.where(overlap_counts > 0)[0]
    processing_order = np.lexsort((ol_labels, overlap_counts[ol_labels]))

    for index in ol_labels[processing_order]:
        neighbors = second[indexes[index] : indexes[index] + overlap_counts[index]]
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
        crange = np.arange(1, len(colors) + 1)
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
    dense = np.zeros([np.max(v_color)] + list(dense_shape), labels.dtype)
    slices = tuple([v_color[labels] - 1] + positional_columns)
    dense[slices] = labels
    indices = [np.where(v_color == i)[0] for i in range(1, dense.shape[0] + 1)]

    return dense, indices
