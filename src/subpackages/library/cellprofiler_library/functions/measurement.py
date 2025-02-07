import numpy as np
import scipy
import centrosome
import centrosome.cpmorphology
import centrosome.filter
import centrosome.propagate
import centrosome.fastemd
from sklearn.cluster import KMeans

from cellprofiler_library.opts import measureimageoverlap as mio
from cellprofiler_library.functions.segmentation import convert_labels_to_ijv
from cellprofiler_library.functions.segmentation import indices_from_ijv
from cellprofiler_library.functions.segmentation import count_from_ijv
from cellprofiler_library.functions.segmentation import areas_from_ijv
from cellprofiler_library.functions.segmentation import cast_labels_to_label_set


def measure_image_overlap_statistics(
    ground_truth_image,
    test_image,
    mask=None,
):
    # Check that the inputs are binary
    if not np.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not np.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = np.ones_like(ground_truth_image, bool)

    orig_shape = ground_truth_image.shape
    
    # Covert 3D image to 2D long
    if ground_truth_image.ndim > 2:
        
        ground_truth_image = ground_truth_image.reshape(
            -1, ground_truth_image.shape[-1]
        )
        test_image = test_image.reshape(-1, test_image.shape[-1])

        mask = mask.reshape(-1, mask.shape[-1])

    false_positives = test_image & ~ground_truth_image

    false_positives[~mask] = False

    false_negatives = (~test_image) & ground_truth_image

    false_negatives[~mask] = False

    true_positives = test_image & ground_truth_image

    true_positives[~mask] = False

    true_negatives = (~test_image) & (~ground_truth_image)

    true_negatives[~mask] = False

    false_positive_count = np.sum(false_positives)

    true_positive_count = np.sum(true_positives)

    false_negative_count = np.sum(false_negatives)

    true_negative_count = np.sum(true_negatives)

    labeled_pixel_count = true_positive_count + false_positive_count

    true_count = true_positive_count + false_negative_count

    if labeled_pixel_count == 0:
        precision = 1.0
    else:
        precision = float(true_positive_count) / float(labeled_pixel_count)

    if true_count == 0:
        recall = 1.0
    else:
        recall = float(true_positive_count) / float(true_count)

    if (precision + recall) == 0:
        f_factor = 0.0  # From http://en.wikipedia.org/wiki/F1_score
    else:
        f_factor = 2.0 * precision * recall / (precision + recall)

    negative_count = false_positive_count + true_negative_count

    if negative_count == 0:
        false_positive_rate = 0.0

        true_negative_rate = 1.0
    else:
        false_positive_rate = float(false_positive_count) / float(negative_count)

        true_negative_rate = float(true_negative_count) / float(negative_count)
    if true_count == 0:
        false_negative_rate = 0.0

        true_positive_rate = 1.0
    else:
        false_negative_rate = float(false_negative_count) / float(true_count)

        true_positive_rate = float(true_positive_count) / float(true_count)

    ground_truth_labels, ground_truth_count = scipy.ndimage.label(
        ground_truth_image & mask, np.ones((3, 3), bool)
    )

    test_labels, test_count = scipy.ndimage.label(
        test_image & mask, np.ones((3, 3), bool)
    )

    rand_index, adjusted_rand_index = compute_rand_index(
        test_labels, ground_truth_labels, mask
    )

    data = {
        "true_positives": true_positives.reshape(orig_shape),
        "true_negatives": true_negatives.reshape(orig_shape),
        "false_positives": false_positives.reshape(orig_shape),
        "false_negatives": false_negatives.reshape(orig_shape),
        "Ffactor": f_factor,
        "Precision": precision,
        "Recall": recall,
        "TruePosRate": true_positive_rate,
        "FalsePosRate": false_positive_rate,
        "FalseNegRate": false_negative_rate,
        "TrueNegRate": true_negative_rate,
        "RandIndex": rand_index,
        "AdjustedRandIndex": adjusted_rand_index,
    }

    return data


def compute_rand_index(test_labels, ground_truth_labels, mask):
    """Calculate the Rand Index

    http://en.wikipedia.org/wiki/Rand_index

    Given a set of N elements and two partitions of that set, X and Y

    A = the number of pairs of elements in S that are in the same set in
        X and in the same set in Y
    B = the number of pairs of elements in S that are in different sets
        in X and different sets in Y
    C = the number of pairs of elements in S that are in the same set in
        X and different sets in Y
    D = the number of pairs of elements in S that are in different sets
        in X and the same set in Y

    The rand index is:   A + B
                            -----
                        A+B+C+D


    The adjusted rand index is the rand index adjusted for chance
    so as not to penalize situations with many segmentations.

    Jorge M. Santos, Mark Embrechts, "On the Use of the Adjusted Rand
    Index as a Metric for Evaluating Supervised Classification",
    Lecture Notes in Computer Science,
    Springer, Vol. 5769, pp. 175-184, 2009. Eqn # 6

    ExpectedIndex = best possible score

    ExpectedIndex = sum(N_i choose 2) * sum(N_j choose 2)

    MaxIndex = worst possible score = 1/2 (sum(N_i choose 2) + sum(N_j choose 2)) * total

    A * total - ExpectedIndex
    -------------------------
    MaxIndex - ExpectedIndex

    returns a tuple of the Rand Index and the adjusted Rand Index
    """
    ground_truth_labels = ground_truth_labels[mask].astype(np.uint32)
    test_labels = test_labels[mask].astype(np.uint32)
    if len(test_labels) > 0:
        #
        # Create a sparse matrix of the pixel labels in each of the sets
        #
        # The matrix, N(i,j) gives the counts of all of the pixels that were
        # labeled with label I in the ground truth and label J in the
        # test set.
        #
        N_ij = scipy.sparse.coo_matrix(
            (np.ones(len(test_labels)), (ground_truth_labels, test_labels))
        ).toarray()

        def choose2(x):
            """Compute # of pairs of x things = x * (x-1) / 2"""
            return x * (x - 1) / 2

        #
        # Each cell in the matrix is a count of a grouping of pixels whose
        # pixel pairs are in the same set in both groups. The number of
        # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
        #
        A = np.sum(choose2(N_ij))
        #
        # B is the sum of pixels that were classified differently by both
        # sets. But the easier calculation is to find A, C and D and get
        # B by subtracting A, C and D from the N * (N - 1), the total
        # number of pairs.
        #
        # For C, we take the number of pixels classified as "i" and for each
        # "j", subtract N(i,j) from N(i) to get the number of pixels in
        # N(i,j) that are in some other set = (N(i) - N(i,j)) * N(i,j)
        #
        # We do the similar calculation for D
        #
        N_i = np.sum(N_ij, 1)
        N_j = np.sum(N_ij, 0)
        C = np.sum((N_i[:, np.newaxis] - N_ij) * N_ij) / 2
        D = np.sum((N_j[np.newaxis, :] - N_ij) * N_ij) / 2
        total = choose2(len(test_labels))
        # an astute observer would say, why bother computing A and B
        # when all we need is A+B and C, D and the total can be used to do
        # that. The calculations aren't too expensive, though, so I do them.
        B = total - A - C - D
        rand_index = (A + B) / total
        #
        # Compute adjusted Rand Index
        #
        expected_index = np.sum(choose2(N_i)) * np.sum(choose2(N_j))
        max_index = (np.sum(choose2(N_i)) + np.sum(choose2(N_j))) * total / 2

        adjusted_rand_index = (A * total - expected_index) / (
            max_index - expected_index
        )
    else:
        rand_index = adjusted_rand_index = np.nan
    return rand_index, adjusted_rand_index


def compute_earth_movers_distance(
    ground_truth_image,
    test_image,
    mask=None,
    decimation_method: mio.DM = mio.DM.KMEANS,
    max_distance: int = 250,
    max_points: int = 250,
    penalize_missing: bool = False,
):
    """Compute the earthmovers distance between two sets of objects

    src_objects - move pixels from these objects

    dest_objects - move pixels to these objects

    returns the earth mover's distance
    """

    # Check that the inputs are binary
    if not np.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not np.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = np.ones_like(ground_truth_image, bool)

    # Covert 3D image to 2D long
    if ground_truth_image.ndim > 2:
        ground_truth_image = ground_truth_image.reshape(
            -1, ground_truth_image.shape[-1]
        )

        test_image = test_image.reshape(-1, test_image.shape[-1])

        mask = mask.reshape(-1, mask.shape[-1])

    # ground truth labels
    dest_labels = scipy.ndimage.label(
        ground_truth_image & mask, np.ones((3, 3), bool)
    )[0]
    dest_labelset = cast_labels_to_label_set(dest_labels)
    dest_ijv = convert_labels_to_ijv(dest_labels, validate=False)
    dest_ijv_indices = indices_from_ijv(dest_ijv, validate=False)
    dest_count = count_from_ijv(
        dest_ijv, indices=dest_ijv_indices, validate=False)
    dest_areas = areas_from_ijv(
        dest_ijv, indices=dest_ijv_indices, validate=False)

    # test labels
    src_labels = scipy.ndimage.label(
        test_image & mask, np.ones((3, 3), bool)
    )[0]
    src_labelset = cast_labels_to_label_set(src_labels)
    src_ijv = convert_labels_to_ijv(src_labels, validate=False)
    src_ijv_indices = indices_from_ijv(src_ijv, validate=False)
    src_count = count_from_ijv(
        src_ijv, indices=src_ijv_indices, validate=False)
    src_areas = areas_from_ijv(
        src_ijv, indices=src_ijv_indices, validate=False)

    #
    # if either foreground set is empty, the emd is the penalty.
    #
    for lef_count, right_areas in (
        (src_count, dest_areas),
        (dest_count, src_areas),
    ):
        if lef_count == 0:
            if penalize_missing:
                return np.sum(right_areas) * max_distance
            else:
                return 0
    if decimation_method == mio.DM.KMEANS:
        isrc, jsrc = get_kmeans_points(src_ijv, dest_ijv, max_points)
        idest, jdest = isrc, jsrc
    elif decimation_method == mio.DM.SKELETON:
        isrc, jsrc = get_skeleton_points(src_labelset, src_labels.shape, max_points)
        idest, jdest = get_skeleton_points(dest_labelset, dest_labels.shape, max_points)
    else:
        raise TypeError("Unknown type for decimation method: %s" % decimation_method)
    src_weights, dest_weights = [
        get_weights(i, j, get_labels_mask(labelset, shape))
        for i, j, labelset, shape in (
            (isrc, jsrc, src_labelset, src_labels.shape),
            (idest, jdest, dest_labelset, dest_labels.shape),
        )
    ]
    ioff, joff = [
        src[:, np.newaxis] - dest[np.newaxis, :]
        for src, dest in ((isrc, idest), (jsrc, jdest))
    ]
    c = np.sqrt(ioff * ioff + joff * joff).astype(np.int32)
    c[c > max_distance] = max_distance
    extra_mass_penalty = max_distance if penalize_missing else 0

    emd = centrosome.fastemd.emd_hat_int32(
        src_weights.astype(np.int32),
        dest_weights.astype(np.int32),
        c,
        extra_mass_penalty=extra_mass_penalty,
    )
    return emd


def get_labels_mask(labelset, shape):
    labels_mask = np.zeros(shape, bool)
    for labels, indexes in labelset:
        labels_mask = labels_mask | labels > 0
    return labels_mask


def get_skeleton_points(labelset, shape, max_points):
    """Get points by skeletonizing the objects and decimating"""
    total_skel = np.zeros(shape, bool)

    for labels, indexes in labelset:
        colors = centrosome.cpmorphology.color_labels(labels)
        for color in range(1, np.max(colors) + 1):
            labels_mask = colors == color
            skel = centrosome.cpmorphology.skeletonize(
                labels_mask,
                ordering=scipy.ndimage.distance_transform_edt(labels_mask)
                * centrosome.filter.poisson_equation(labels_mask),
            )
            total_skel = total_skel | skel

    n_pts = np.sum(total_skel)

    if n_pts == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)

    i, j = np.where(total_skel)

    if n_pts > max_points:
        #
        # Decimate the skeleton by finding the branchpoints in the
        # skeleton and propagating from those.
        #
        markers = np.zeros(total_skel.shape, np.int32)
        branchpoints = centrosome.cpmorphology.branchpoints(
            total_skel
        ) | centrosome.cpmorphology.endpoints(total_skel)
        markers[branchpoints] = np.arange(np.sum(branchpoints)) + 1
        #
        # We compute the propagation distance to that point, then impose
        # a slightly arbitrary order to get an unambiguous ordering
        # which should number the pixels in a skeleton branch monotonically
        #
        ts_labels, distances = centrosome.propagate.propagate(
            np.zeros(markers.shape), markers, total_skel, 1
        )
        order = np.lexsort((j, i, distances[i, j], ts_labels[i, j]))
        #
        # Get a linear space of self.max_points elements with bounds at
        # 0 and len(order)-1 and use that to select the points.
        #
        order = order[np.linspace(0, len(order) - 1, max_points).astype(int)]
        return i[order], j[order]

    return i, j


def get_kmeans_points(src_ijv, dest_ijv, max_points):
    """Get representative points in the objects using K means

    src_ijv - get some of the foreground points from the source ijv labeling
    dest_ijv - get the rest of the foreground points from the ijv labeling
                objects

    returns a vector of i coordinates of representatives and a vector
            of j coordinates
    """

    ijv = np.vstack((src_ijv, dest_ijv))
    if len(ijv) <= max_points:
        return ijv[:, 0], ijv[:, 1]
    random_state = np.random.RandomState()
    random_state.seed(ijv.astype(int).flatten())
    kmeans = KMeans(n_clusters=max_points, tol=2, random_state=random_state)
    kmeans.fit(ijv[:, :2])
    return (
        kmeans.cluster_centers_[:, 0].astype(np.uint32),
        kmeans.cluster_centers_[:, 1].astype(np.uint32),
    )


def get_weights(i, j, labels_mask):
    """Return the weights to assign each i,j point

    Assign each pixel in the labels mask to the nearest i,j and return
    the number of pixels assigned to each i,j
    """
    #
    # Create a mapping of chosen points to their index in the i,j array
    #
    total_skel = np.zeros(labels_mask.shape, int)
    total_skel[i, j] = np.arange(1, len(i) + 1)
    #
    # Compute the distance from each chosen point to all others in image,
    # return the nearest point.
    #
    ii, jj = scipy.ndimage.distance_transform_edt(
        total_skel == 0, return_indices=True, return_distances=False
    )
    #
    # Filter out all unmasked points
    #
    ii, jj = [x[labels_mask] for x in (ii, jj)]
    if len(ii) == 0:
        return np.zeros(0, np.int32)
    #
    # Use total_skel to look up the indices of the chosen points and
    # bincount the indices.
    #
    result = np.zeros(len(i), np.int32)
    bc = np.bincount(total_skel[ii, jj])[1:]
    result[: len(bc)] = bc
    return result
