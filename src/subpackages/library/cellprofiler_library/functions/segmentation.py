from enum import Enum
import numpy as np
from numpy.random.mtrand import RandomState
import scipy.sparse
import centrosome.index

class SPARSE_FIELD(Enum):
    label = "label"
    c = "c"
    t = "t"
    z = "z"
    y = "y"
    x = "x"
    
class DENSE_AXIS(Enum):
    label_idx = 0
    c = 1
    t = 2
    z = 3
    y = 4
    x = 5

SPARSE_FIELDS = tuple([mem.value for mem in SPARSE_FIELD])
SPARSE_AXES_FIELDS = SPARSE_FIELDS[1:]
DENSE_AXIS_NAMES = tuple([mem.name for mem in DENSE_AXIS])
DENSE_SHAPE_NAMES = DENSE_AXIS_NAMES[1:]

# ------ Functions for validating segmentation formats ------

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
    ndim = len(DENSE_AXIS_NAMES)
    assert type(dense) == np.ndarray, "dense must be ndarray"
    assert dense.ndim == ndim, \
    f"dense must be {ndim}-dimensional - f{DENSE_AXIS_NAMES}"

def _validate_dense_shape(dense_shape):
    """
    'dense_shape', as opposed to 'dense.shape', is the shape of the 'dense'
    matrix sans the 'label_idx' axis, i.e.
    (c, t, z, y, z)
    """
    ndim = len(DENSE_SHAPE_NAMES)
    assert (dense_shape is None or
            len(dense_shape) == ndim
    ), f"dense_shape must be length {ndim}, omitting '{DENSE_AXIS.label_idx.name}' dim"

def _validate_labels(labels):
    """
    A 'labels' matrix is another, more constrained, dense representation

    It is strictly 2- or 3-dimensional, of shape: (y, x) or (z, y, x)
    A single 'labels' matrix does not allow for overlapping labels within it

    It is essentially a 'dense' of shape (1, 1, 1, 1, y, x), but squeezed
    such that the ('label_idx', 'c', 't', 'z') axes are removed

    For a 'dense' with shape (2+, 1, 1, 1, y, x), a 'label_set' can be
    constructed (see 'convert_dense_to_label_set' for more details)
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
    full_set = set(SPARSE_FIELDS)

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

# ------ Functions converting between segmentation formats ------

def indices_from_dense(dense, validate=True):
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
    if validate:
        _validate_dense(dense)

    indices = [np.unique(d) for d in dense]
    indices = [idx[1:] if idx[0] == 0 else idx for idx in indices]
    return indices

def dense_shape_from_sparse(sparse, validate=True):
    if validate:
        _validate_sparse(sparse)

    return tuple([
        np.max(sparse[axis]) + 2
        if axis in list(sparse.dtype.fields.keys())
        else 1
        for axis in SPARSE_AXES_FIELDS
    ])

def indices_from_ijv(ijv, validate=True):
    """
    Get the indices for a scipy.ndimage-style function from the 'ijv' formatted
    segmented labels
    """
    if validate:
        _validate_ijv(ijv)

    if len(ijv) == 0:
        return np.zeros(0, np.int32)

    max_label = np.max(ijv[:, 2])

    return np.arange(max_label).astype(np.int32) + 1

def count_from_ijv(ijv, indices=None, validate=True):
    """
    The number of labels in an 'ijv' formatted label matrix
    """
    if validate:
        _validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    return len(indices)

def areas_from_ijv(ijv, indices=None, validate=True):
    """
    The area of each object in an 'ijv' formatted label matrix

    Because of the discrete nature of the matrix, this is simply equal to the
    occurrence count of each label in the matrix
    """
    if validate:
        _validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    if len(indices) == 0:
        return np.zeros(0, int)

    return np.bincount(ijv[:, 2])[indices]

def downsample_labels(labels, validate=True):
    """
    Convert a 'labels' matrix to the smallest possible integer format
    """
    if validate:
        _validate_labels(labels)

    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)

def convert_dense_to_label_set(dense, indices=None, validate=True):
    """
    Convert a 'dense' matrix into a list of 2-tuples,
    where the number of tuples corresponds to the 'label_idx' dim of the
    'dense' matrix (see '_validate_dense' for details),
    the tuple's first element is a 'labels' matrix
    (see '_validate_labels' for details),
    and the tuple's second element is the 1-d ndarray of labels in the matrix
    (see 'indices_from_dense' for details)
    """
    if validate:
        _validate_dense(dense)
        assert(
            dense.shape[DENSE_AXIS.c.value] == 1 and 
            dense.shape[DENSE_AXIS.t.value] == 1
        ), f"dense must have shape where f{DENSE_AXIS.c.name} = 1 and f{DENSE_AXIS.t.name} = 1"

    if indices is None:
        indices = indices_from_dense(dense, validate=False)

    label_set_len = dense.shape[DENSE_AXIS.label_idx.value]
    squeezed_dense = dense.squeeze()

    if label_set_len == 1:
        return [(squeezed_dense, indices[0])]
    
    return [(squeezed_dense[i], indices[i]) for i in range(label_set_len)]

def indices_from_labels(labels, validate=True):
    if validate:
        _validate_labels(labels)

    return np.unique(labels[labels != 0])

def cast_labels_to_label_set(labels, validate=True):
    """
    Takes in a 'labels' matrix and casts it into a 1-element 'label_set'
    """
    if validate:
        _validate_labels(labels)

    return [(labels, indices_from_labels(labels, validate=False))]

def convert_labels_to_dense(labels, validate=True):
    """
    Convert a 'labels' matrix (e.g. scipy.ndimage.label) to 'dense' matrix
    """
    if validate:
        _validate_labels(labels)

    typed_labels = downsample_labels(labels, validate=False)

    if labels.ndim == 3:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value
        )
    else:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value,
            DENSE_AXIS.z.value
        )

    return np.expand_dims(typed_labels, axis=expand_axes)

def convert_dense_to_sparse(dense, validate=True):
    if validate:
        _validate_dense(dense)

    full_shape = dense.shape
    label_dim = full_shape[DENSE_AXIS.label_idx.value]
    dense_shape = tuple(
        [full_shape[DENSE_AXIS[n].value] for n in DENSE_SHAPE_NAMES]
    )

    axes_labels = np.array(SPARSE_AXES_FIELDS)
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
    dtype.append((SPARSE_FIELD.label.value, labels_dtype))
    sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)

    return sparse

def convert_ijv_to_sparse(ijv, validate=True):
    if validate:
        _validate_ijv(ijv)

    return np.core.records.fromarrays(
        (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
        [
            (SPARSE_FIELD.y.value, ijv.dtype),
            (SPARSE_FIELD.x.value, ijv.dtype),
            (SPARSE_FIELD.label.value, ijv.dtype)
        ],
    )

def convert_sparse_to_ijv(sparse, validate=True):
    if validate:
        _validate_sparse(sparse)

    return np.column_stack([sparse[axis] for axis in (
        SPARSE_FIELD.y.value, SPARSE_FIELD.x.value, SPARSE_FIELD.label.value)
    ])

def convert_labels_to_ijv(labels, validate=True):
    if validate:
        _validate_labels(labels)

    dense = convert_labels_to_dense(labels, validate=False)
    sparse = convert_dense_to_sparse(dense, validate=False)
    ijv = convert_sparse_to_ijv(sparse, validate=False)

    return ijv

def convert_ijv_to_label_set(ijv, dense_shape=None, validate=True):
    if validate:
        _validate_ijv(ijv)

    sparse = convert_ijv_to_sparse(ijv, validate=False)

    if dense_shape is None:
        dense_shape = dense_shape_from_sparse(sparse)

    dense, indices = convert_sparse_to_dense(
        sparse,
        dense_shape=dense_shape,
        validate=False
    )

    label_set = convert_dense_to_label_set(
        dense,
        indices=indices,
        validate=False
    )

    return label_set

def convert_label_set_to_ijv(label_set, validate=True):
    return np.concatenate(
        [convert_labels_to_ijv(l[0], validate) for l in label_set],
        axis=0
    )

def convert_sparse_to_dense(sparse, dense_shape=None, validate=True):
    """
    Convert 'sparse' representation to 'dense' matrix

    Returns 'dense' matrix and corresponding 'indices'
    """
    if validate:
        _validate_sparse(sparse)
        _validate_dense_shape(dense_shape)

    if len(sparse) == 0:
        if dense_shape is None:
            dense_shape = tuple([1 for _ in range(len(DENSE_SHAPE_NAMES))])

        dense = np.expand_dims(
            np.zeros(dense_shape, np.uint8),
            axis=DENSE_AXIS.label_idx.value
        )

        return dense, indices_from_dense(dense, validate=False)

    if dense_shape is None:
        dense_shape = dense_shape_from_sparse(sparse, validate=False)

    #
    # The code below assigns a "color" to each label so that no
    # two labels have the same color
    #
    positional_columns = []
    available_columns = []
    lexsort_columns = []
    for axis in SPARSE_AXES_FIELDS:
        if axis in list(sparse.dtype.fields.keys()):
            positional_columns.append(sparse[axis])
            available_columns.append(sparse[axis])
            lexsort_columns.insert(0, sparse[axis])
        else:
            positional_columns.append(0)
    labels = sparse[SPARSE_FIELD.label.value]
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
        return dense, indices_from_dense(dense, validate=False)
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

# ------ Functions for operating on segmentation formats ------

def make_rgb_outlines(label_set, colors, random_seed=None, validate=True):
    """
    Assign rgb colors to outlines of labels in 'label_set`

    Make outlines, coloring each object differently to distinguish between
    objects that might overlap.

    'label_set': see 'convert_dense_to_label_set'

    'colors': a N x 3 color map to be used to color the outlines
    where N in dim 0 should match the number of unique labels in the
    `label_set`, and values are R, G, and B values normalized to [0, 1]

    'random_seed' when provided, will seed the RNG for permuting colors
    between 'labels' matrices in the 'label_set'
    """
    if validate:
        assert type(colors) == np.ndarray, "'colors' must be ndarray"
        assert (
            colors.ndim == 2 and
            colors.shape[1] == 3
        ), "'colors' must be of shape (N, 3)"
        indices = [i for _, idxs in label_set for i in idxs]
        # >= because technically you can have superflous colors (but don't)
        assert colors.shape[0] >= len(indices), \
            "axis 1 of 'colors' must be equal to the number of unique labels in 'label_set'"
    #
    # Get planes of non-overlapping objects. The idea here is to use
    # the most similar colors in the color space for objects that
    # don't overlap.
    #
    label_outline_set = [
        (centrosome.outline.outline(label), indexes)
        for label, indexes in label_set
    ]
    rgb_image = np.zeros(list(label_outline_set[0][0].shape) + [3], np.float32)
    #
    # Find out how many unique labels in each
    #
    counts = [np.sum(np.unique(l) != 0) for l, _ in label_outline_set]
    if len(counts) == 1 and counts[0] == 0:
        return rgb_image

    if len(colors) < len(label_outline_set):
        # Have to color 2 planes using the same color!
        # There's some chance that overlapping objects will get
        # the same color. Give me more colors to work with please.
        colors = np.vstack([colors] * (1 + len(label_outline_set) // len(colors)))
    r = RandomState()
    r.seed(random_seed)
    alpha = np.zeros(label_outline_set[0][0].shape, np.float32)
    order = np.lexsort([counts])

    for idx, i in enumerate(order):
        max_available = len(colors) / (len(label_outline_set) - idx)
        ncolors = min(counts[i], max_available)
        my_colors = colors[:ncolors]
        colors = colors[ncolors:]
        my_colors = my_colors[r.permutation(np.arange(ncolors))]
        my_labels, indexes = label_outline_set[i]
        color_idx = np.zeros(np.max(indexes) + 1, int)
        color_idx[indexes] = np.arange(len(indexes)) % ncolors
        rgb_image[my_labels != 0, :] += my_colors[
            color_idx[my_labels[my_labels != 0]], :
        ]
        alpha[my_labels != 0] += 1
    rgb_image[alpha > 0, :] /= alpha[alpha > 0][:, np.newaxis]

    return rgb_image

# needs library tests
def find_label_overlaps(parent_labels, child_labels, validate=True):
    """
    Find per pixel overlap of parent labels and child labels

    'parent_labels' - the parents which contain the children in 'labels' format
    'child_labels' - the children to be mapped to a parent in 'labels' format

    Returns a sparse 'coo_matrix' of overlap between each parent and child.
    Note that the first row and column are empty, as these
    correspond to parent and child labels of 0.
    """
    if validate:
        _validate_labels(parent_labels)
        _validate_labels(child_labels)

    parent_count = np.max(parent_labels)
    child_count = np.max(child_labels)
    #
    # If the labels are different shapes, crop to shared shape.
    #
    common_shape = np.minimum(parent_labels.shape, child_labels.shape)

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
    not_zero_count = np.sum(not_zero)

    #
    # each row (axis = 0) is a parent
    # each column (axis = 1) is a child
    #
    return scipy.sparse.coo_matrix(
        (
            np.ones((not_zero_count,)),
            (parent_labels[not_zero], child_labels[not_zero]),
        ),
        shape=(parent_count + 1, child_count + 1),
    )

# needs library tests
def find_ijv_overlaps(parent_ijv, child_ijv, validate=True):
    """
    Find per pixel overlap of parent labels and child labels

    'parent_ijv' - the parents which contain the children, in 'ijv' format
    'child_ijv' - the children to be mapped to a parent, in 'ijv' format

    Returns a sparse 'csc_matrix' of overlap between each parent and child.
    Note that the first row and column are empty, as these
    correspond to parent and child labels of 0.
    """
    if validate:
        _validate_ijv(parent_ijv)
        _validate_ijv(child_ijv)

    parent_count = 0 if (parent_ijv.shape[0] == 0) else np.max(parent_ijv[:, 2])
    child_count = 0 if (child_ijv.shape[0] == 0) else np.max(child_ijv[:, 2])

    if parent_count == 0 or child_count == 0:
        return np.zeros((parent_count + 1, child_count + 1), int)

    dim_i = max(np.max(parent_ijv[:, 0]), np.max(child_ijv[:, 0])) + 1
    dim_j = max(np.max(parent_ijv[:, 1]), np.max(child_ijv[:, 1])) + 1
    parent_linear_ij = parent_ijv[:, 0] + dim_i * parent_ijv[:, 1].astype(
        np.uint64
    )
    child_linear_ij = child_ijv[:, 0] + dim_i * child_ijv[:, 1].astype(np.uint64)

    parent_matrix = scipy.sparse.coo_matrix(
        (np.ones((parent_ijv.shape[0],)), (parent_ijv[:, 2], parent_linear_ij)),
        shape=(parent_count + 1, dim_i * dim_j),
    )
    child_matrix = scipy.sparse.coo_matrix(
        (np.ones((child_ijv.shape[0],)), (child_linear_ij, child_ijv[:, 2])),
        shape=(dim_i * dim_j, child_count + 1),
    )
    # I surely do not understand the sparse code.  Converting both
    # arrays to csc gives the best peformance... Why not p.csr and
    # c.csc?
    return parent_matrix.tocsc() * child_matrix.tocsc()

def center_of_labels_mass(labels, validate=True):
    if validate:
        _validate_labels(labels)

    indices = indices_from_labels(labels)
    return np.array(
        scipy.ndimage.center_of_mass(np.ones_like(labels), labels, indices)
    )
