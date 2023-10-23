import numpy as np
import centrosome.index

def _validate_dense(dense):
    assert dense.ndim == 6, "dense must be 6-dimensional - (labels, c, t, z, y, x)"

def indices_from_dense(dense):
    indices = [np.unique(d) for d in dense]
    indices = [idx[1:] if idx[0] == 0 else idx for idx in indices]
    return np.array(indices)

def shape_from_sparse(sparse):
    return tuple([
        np.max(sparse[axis]) + 2
        if axis in list(sparse.dtype.fields.keys())
        else 1
        for axis in ("c", "t", "z", "y", "x")
    ])

def label_set_from_dense(dense, indices=None):
    _validate_dense(dense)

    if indices is None:
        indices = indices_from_dense(dense)

    if dense.shape[3] == 1:
        return [(dense[i, 0, 0, 0], indices[i]) for i in range(dense.shape[0])]

    return [(dense[i, 0, 0], indices[i]) for i in range(dense.shape[0])]

def ijv_from_sparse(sparse):
    return np.column_stack([sparse[axis] for axis in ("y", "x", "label")])

def indices_from_ijv(ijv):
    """
    Get the indices for a scipy.ndimage-style function from the segmented labels
    """
    if len(ijv) == 0:
        return np.zeros(0, np.int32)

    max_label = np.max(ijv[:, 2])

    return np.arange(max_label).astype(np.int32) + 1

def count_from_ijv(ijv, indices=None):
    """The number of labels in an object"""
    if indices is None:
        indices = indices_from_ijv(ijv)

    return len(indices)

def areas_from_ijv(ijv, indices=None):
    """The area of each object"""
    if indices is None:
        indices = indices_from_ijv(ijv)

    if len(indices) == 0:
        return np.zeros(0, int)

    return np.bincount(ijv[:, 2])[indices]

def downsample_labels(labels):
    """
    Convert a labels matrix to the smallest possible integer format
    """
    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)

def convert_labels_to_dense(labels):
    """
    Convert a labels matrix (e.g. scipy.ndimage.label) to dense labeling
    """
    dense = downsample_labels(labels)

    if dense.ndim == 3:
        z, y, x = dense.shape
    else:
        y, x = dense.shape
        z = 1

    return dense.reshape((1, 1, 1, z, y, x))


def convert_dense_to_sparse(dense, indices=None):
    _validate_dense(dense)

    if indices is None:
        indices = indices_from_dense(dense)

    label_dim = dense.shape[0]
    dense_shape = dense.shape[1:]

    axes_labels = np.array(("c", "t", "z", "y", "x"))
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
        max_label = np.max(indices)
        if max_label < 2 ** 8:
            labels_dtype = np.uint8
        elif max_label < 2 ** 16:
            labels_dtype = np.uint16
        else:
            labels_dtype = np.uint32
    else:
        labels_dtype = np.uint8

    dtype = [(axis, coords_dtype) for axis in axes]
    dtype.append(("label", labels_dtype))
    sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)

    return sparse

def convert_sparse_to_dense(sparse, dense_shape=None):
    if len(sparse) == 0:
        if dense_shape is None:
            dense_shape = (1,1,1,1,1,1)
        # normally the labels dim is ommitted for dense_shape
        elif len(dense_shape) == 5:
            dense_shape = (1,) + dense_shape

        dense = np.zeros(dense_shape, np.uint8)
        return dense, indices_from_dense(dense)

    if dense_shape is None:
        dense_shape = shape_from_sparse(sparse)

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
