import numpy
import scipy
import centrosome
import centrosome.cpmorphology
import centrosome.filter
import centrosome.propagate
import centrosome.fastemd
from sklearn.cluster import KMeans

from cellprofiler_library.object import Objects
from cellprofiler_library.opts import measureimageoverlap as mio


def measure_image_overlap_statistics(
    ground_truth_image,
    test_image,
    mask=None,
):
    # Check that the inputs are binary
    if not numpy.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not numpy.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = numpy.ones_like(ground_truth_image, bool)

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

    false_positive_count = numpy.sum(false_positives)

    true_positive_count = numpy.sum(true_positives)

    false_negative_count = numpy.sum(false_negatives)

    true_negative_count = numpy.sum(true_negatives)

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
        ground_truth_image & mask, numpy.ones((3, 3), bool)
    )

    test_labels, test_count = scipy.ndimage.label(
        test_image & mask, numpy.ones((3, 3), bool)
    )

    rand_index, adjusted_rand_index = compute_rand_index(
        test_labels, ground_truth_labels, mask
    )

    data = {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
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
    ground_truth_labels = ground_truth_labels[mask].astype(numpy.uint32)
    test_labels = test_labels[mask].astype(numpy.uint32)
    if len(test_labels) > 0:
        #
        # Create a sparse matrix of the pixel labels in each of the sets
        #
        # The matrix, N(i,j) gives the counts of all of the pixels that were
        # labeled with label I in the ground truth and label J in the
        # test set.
        #
        N_ij = scipy.sparse.coo_matrix(
            (numpy.ones(len(test_labels)), (ground_truth_labels, test_labels))
        ).toarray()

        def choose2(x):
            """Compute # of pairs of x things = x * (x-1) / 2"""
            return x * (x - 1) / 2

        #
        # Each cell in the matrix is a count of a grouping of pixels whose
        # pixel pairs are in the same set in both groups. The number of
        # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
        #
        A = numpy.sum(choose2(N_ij))
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
        N_i = numpy.sum(N_ij, 1)
        N_j = numpy.sum(N_ij, 0)
        C = numpy.sum((N_i[:, numpy.newaxis] - N_ij) * N_ij) / 2
        D = numpy.sum((N_j[numpy.newaxis, :] - N_ij) * N_ij) / 2
        total = choose2(len(test_labels))
        # an astute observer would say, why bother computing A and B
        # when all we need is A+B and C, D and the total can be used to do
        # that. The calculations aren't too expensive, though, so I do them.
        B = total - A - C - D
        rand_index = (A + B) / total
        #
        # Compute adjusted Rand Index
        #
        expected_index = numpy.sum(choose2(N_i)) * numpy.sum(choose2(N_j))
        max_index = (numpy.sum(choose2(N_i)) + numpy.sum(choose2(N_j))) * total / 2

        adjusted_rand_index = (A * total - expected_index) / (
            max_index - expected_index
        )
    else:
        rand_index = adjusted_rand_index = numpy.nan
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
    if not numpy.array_equal(ground_truth_image, ground_truth_image.astype(bool)):
        raise ValueError("ground_truth_image is not a binary image")
    
    if not numpy.array_equal(test_image, test_image.astype(bool)):
        raise ValueError("test_image is not a binary image")

    if mask is None:
        mask = numpy.ones_like(ground_truth_image, bool)

    # Covert 3D image to 2D long
    if ground_truth_image.ndim > 2:
        ground_truth_image = ground_truth_image.reshape(
            -1, ground_truth_image.shape[-1]
        )

        test_image = test_image.reshape(-1, test_image.shape[-1])

        mask = mask.reshape(-1, mask.shape[-1])

    ground_truth_labels = scipy.ndimage.label(
        ground_truth_image & mask, numpy.ones((3, 3), bool)
    )[0]

    test_labels = scipy.ndimage.label(test_image & mask, numpy.ones((3, 3), bool))[0]

    # Convert labelmaps into CellProfiler objects
    dest_objects = Objects()
    dest_objects.segmented = ground_truth_labels

    src_objects = Objects()
    src_objects.segmented = test_labels

    #
    # if either foreground set is empty, the emd is the penalty.
    #
    for angels, demons in (
        (src_objects, dest_objects),
        (dest_objects, src_objects),
    ):
        if angels.count == 0:
            if penalize_missing:
                return numpy.sum(demons.areas) * max_distance
            else:
                return 0
    if decimation_method == mio.DM.KMEANS:
        isrc, jsrc = get_kmeans_points(src_objects, dest_objects, max_points)
        idest, jdest = isrc, jsrc
    elif decimation_method == mio.DM.SKELETON:
        isrc, jsrc = get_skeleton_points(src_objects, max_points)
        idest, jdest = get_skeleton_points(dest_objects, max_points)
    else:
        raise TypeError("Unknown type for decimation method: %s" % decimation_method)
    src_weights, dest_weights = [
        get_weights(i, j, get_labels_mask(objects))
        for i, j, objects in (
            (isrc, jsrc, src_objects),
            (idest, jdest, dest_objects),
        )
    ]
    ioff, joff = [
        src[:, numpy.newaxis] - dest[numpy.newaxis, :]
        for src, dest in ((isrc, idest), (jsrc, jdest))
    ]
    c = numpy.sqrt(ioff * ioff + joff * joff).astype(numpy.int32)
    c[c > max_distance] = max_distance
    extra_mass_penalty = max_distance if penalize_missing else 0

    emd = centrosome.fastemd.emd_hat_int32(
        src_weights.astype(numpy.int32),
        dest_weights.astype(numpy.int32),
        c,
        extra_mass_penalty=extra_mass_penalty,
    )
    return emd


def get_labels_mask(obj):
    labels_mask = numpy.zeros(obj.shape, bool)
    for labels, indexes in obj.get_labels():
        labels_mask = labels_mask | labels > 0
    return labels_mask


def get_skeleton_points(obj, max_points):
    """Get points by skeletonizing the objects and decimating"""
    ii = []
    jj = []
    total_skel = numpy.zeros(obj.shape, bool)
    for labels, indexes in obj.get_labels():
        colors = centrosome.cpmorphology.color_labels(labels)
        for color in range(1, numpy.max(colors) + 1):
            labels_mask = colors == color
            skel = centrosome.cpmorphology.skeletonize(
                labels_mask,
                ordering=scipy.ndimage.distance_transform_edt(labels_mask)
                * centrosome.filter.poisson_equation(labels_mask),
            )
            total_skel = total_skel | skel
    n_pts = numpy.sum(total_skel)
    if n_pts == 0:
        return numpy.zeros(0, numpy.int32), numpy.zeros(0, numpy.int32)
    i, j = numpy.where(total_skel)
    if n_pts > max_points:
        #
        # Decimate the skeleton by finding the branchpoints in the
        # skeleton and propagating from those.
        #
        markers = numpy.zeros(total_skel.shape, numpy.int32)
        branchpoints = centrosome.cpmorphology.branchpoints(
            total_skel
        ) | centrosome.cpmorphology.endpoints(total_skel)
        markers[branchpoints] = numpy.arange(numpy.sum(branchpoints)) + 1
        #
        # We compute the propagation distance to that point, then impose
        # a slightly arbitrary order to get an unambiguous ordering
        # which should number the pixels in a skeleton branch monotonically
        #
        ts_labels, distances = centrosome.propagate.propagate(
            numpy.zeros(markers.shape), markers, total_skel, 1
        )
        order = numpy.lexsort((j, i, distances[i, j], ts_labels[i, j]))
        #
        # Get a linear space of self.max_points elements with bounds at
        # 0 and len(order)-1 and use that to select the points.
        #
        order = order[numpy.linspace(0, len(order) - 1, max_points).astype(int)]
        return i[order], j[order]
    return i, j


def get_kmeans_points(src_obj, dest_obj, max_points):
    """Get representative points in the objects using K means

    src_obj - get some of the foreground points from the source objects
    dest_obj - get the rest of the foreground points from the destination
                objects

    returns a vector of i coordinates of representatives and a vector
            of j coordinates
    """

    # Add objects to Objects class if not already
    if not isinstance(src_obj, Objects):
        src = Objects()
        src.segmented = src_obj
        src_obj = src

        dst = Objects()
        dst.segmented = dest_obj
        dest_obj = dst

    ijv = numpy.vstack((src_obj.ijv, dest_obj.ijv))
    if len(ijv) <= max_points:
        return ijv[:, 0], ijv[:, 1]
    random_state = numpy.random.RandomState()
    random_state.seed(ijv.astype(int).flatten())
    kmeans = KMeans(n_clusters=max_points, tol=2, random_state=random_state)
    kmeans.fit(ijv[:, :2])
    return (
        kmeans.cluster_centers_[:, 0].astype(numpy.uint32),
        kmeans.cluster_centers_[:, 1].astype(numpy.uint32),
    )


def get_weights(i, j, labels_mask):
    """Return the weights to assign each i,j point

    Assign each pixel in the labels mask to the nearest i,j and return
    the number of pixels assigned to each i,j
    """
    #
    # Create a mapping of chosen points to their index in the i,j array
    #
    total_skel = numpy.zeros(labels_mask.shape, int)
    total_skel[i, j] = numpy.arange(1, len(i) + 1)
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
        return numpy.zeros(0, numpy.int32)
    #
    # Use total_skel to look up the indices of the chosen points and
    # bincount the indices.
    #
    result = numpy.zeros(len(i), numpy.int32)
    bc = numpy.bincount(total_skel[ii, jj])[1:]
    result[: len(bc)] = bc
    return result
