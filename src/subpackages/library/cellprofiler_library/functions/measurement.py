import numpy as np
import numpy
import scipy
import scipy.ndimage
import skimage
import centrosome
import centrosome.cpmorphology
import centrosome.filter
import centrosome.propagate
import centrosome.fastemd

from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Dict, Callable, List
from scipy.linalg import lstsq
from numpy.typing import NDArray

from cellprofiler_library.opts import measureimageoverlap as mio
from cellprofiler_library.functions.segmentation import convert_labels_to_ijv
from cellprofiler_library.functions.segmentation import indices_from_ijv
from cellprofiler_library.functions.segmentation import count_from_ijv
from cellprofiler_library.functions.segmentation import areas_from_ijv
from cellprofiler_library.functions.segmentation import cast_labels_to_label_set
from cellprofiler_library.functions.image_processing import masked_erode, restore_scale, get_morphology_footprint

from cellprofiler_library.opts.objectsizeshapefeatures import ObjectSizeShapeFeatures
from cellprofiler_library.types import Pixel, ObjectLabel, ImageGrayscale, ImageGrayscaleMask, ObjectSegmentation
from cellprofiler_library.opts.measurecolocalization import CostesMethod


#
# For each object, build a little record
#
class ObjectRecord(object):
    def __init__(
            self, 
            name: str, 
            segmented: ObjectSegmentation, 
            im_mask: ImageGrayscaleMask, 
            im_pixel_data: ImageGrayscale
            ):
        self.name = name
        self.labels = segmented
        self.nobjects = np.max(self.labels)
        if self.nobjects != 0:
            self.range = np.arange(1, np.max(self.labels) + 1)
            self.labels = self.labels.copy()
            self.labels[~im_mask] = 0
            self.current_mean = fix(
                scipy.ndimage.mean(im_pixel_data, self.labels, self.range)
            )
            self.start_mean = np.maximum(
                self.current_mean, np.finfo(float).eps
            )

            
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


def measure_object_size_shape(
    labels,
    desired_properties,
    calculate_zernikes: bool = True,
    calculate_advanced: bool = True,
    spacing: Tuple = None
):
    label_indices = numpy.unique(labels[labels != 0])
    nobjects = len(label_indices)
    
    if spacing is None:
        spacing = (1.0,) * labels.ndim

    if len(labels.shape) == 2:
        # 2D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)

        formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
        denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
        compactness = props["perimeter"] ** 2 / denom

        max_radius = numpy.zeros(nobjects)
        median_radius = numpy.zeros(nobjects)
        mean_radius = numpy.zeros(nobjects)
        min_feret_diameter = numpy.zeros(nobjects)
        max_feret_diameter = numpy.zeros(nobjects)
        zernike_numbers = centrosome.zernike.get_zernike_indexes(ObjectSizeShapeFeatures.ZERNIKE_N.value + 1)

        zf = {}
        for n, m in zernike_numbers:
            zf[(n, m)] = numpy.zeros(nobjects)

        for index, mini_image in enumerate(props["image"]):
            # Pad image to assist distance tranform
            mini_image = numpy.pad(mini_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.maximum(distances, mini_image)
            )
            mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(distances, mini_image)
            )
            median_radius[index] = centrosome.cpmorphology.median_of_labels(
                distances, mini_image.astype("int"), [1]
            )

        #
        # Zernike features
        #
        if calculate_zernikes:
            zf_l = centrosome.zernike.zernike(zernike_numbers, labels, label_indices)
            for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                zf[(n, m)] = z

        if nobjects > 0:
            chulls, chull_counts = centrosome.cpmorphology.convex_hull(
                labels, label_indices
            )
            #
            # Feret diameter
            #
            (
                min_feret_diameter,
                max_feret_diameter,
            ) = centrosome.cpmorphology.feret_diameter(
                chulls, chull_counts, label_indices
            )

            features_to_record = {
                ObjectSizeShapeFeatures.F_AREA.value: props["area"],
                ObjectSizeShapeFeatures.F_PERIMETER.value: props["perimeter"],
                ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
                ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
                ObjectSizeShapeFeatures.F_ECCENTRICITY.value: props["eccentricity"],
                ObjectSizeShapeFeatures.F_ORIENTATION.value: props["orientation"] * (180 / numpy.pi),
                ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-1"],
                ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-0"],
                ObjectSizeShapeFeatures.F_BBOX_AREA.value: props["bbox_area"],
                ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-1"],
                ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-3"],
                ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-0"],
                ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-2"],
                ObjectSizeShapeFeatures.F_FORM_FACTOR.value: formfactor,
                ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
                ObjectSizeShapeFeatures.F_SOLIDITY.value: props["solidity"],
                ObjectSizeShapeFeatures.F_COMPACTNESS.value: compactness,
                ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
                ObjectSizeShapeFeatures.F_MAXIMUM_RADIUS.value: max_radius,
                ObjectSizeShapeFeatures.F_MEAN_RADIUS.value: mean_radius,
                ObjectSizeShapeFeatures.F_MEDIAN_RADIUS.value: median_radius,
                ObjectSizeShapeFeatures.F_CONVEX_AREA.value: props["convex_area"],
                ObjectSizeShapeFeatures.F_MIN_FERET_DIAMETER.value: min_feret_diameter,
                ObjectSizeShapeFeatures.F_MAX_FERET_DIAMETER.value: max_feret_diameter,
                ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
            }
            if calculate_advanced:
                features_to_record.update(
                    {
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_0.value: props["moments-0-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_1.value: props["moments-0-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_2.value: props["moments-0-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_3.value: props["moments-0-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_0.value: props["moments-1-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_1.value: props["moments-1-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_2.value: props["moments-1-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_3.value: props["moments-1-3"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_0.value: props["moments-2-0"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_1.value: props["moments-2-1"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_2.value: props["moments-2-2"],
                        ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_3.value: props["moments-2-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_0.value: props["moments_central-0-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_1.value: props["moments_central-0-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_2.value: props["moments_central-0-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_3.value: props["moments_central-0-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_0.value: props["moments_central-1-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_1.value: props["moments_central-1-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_2.value: props["moments_central-1-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_3.value: props["moments_central-1-3"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_0.value: props["moments_central-2-0"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_1.value: props["moments_central-2-1"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_2.value: props["moments_central-2-2"],
                        ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_3.value: props["moments_central-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_0.value: props["moments_normalized-0-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_1.value: props["moments_normalized-0-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_2.value: props["moments_normalized-0-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_3.value: props["moments_normalized-0-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_0.value: props["moments_normalized-1-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_1.value: props["moments_normalized-1-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_2.value: props["moments_normalized-1-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_3.value: props["moments_normalized-1-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_0.value: props["moments_normalized-2-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_1.value: props["moments_normalized-2-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_2.value: props["moments_normalized-2-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_3.value: props["moments_normalized-2-3"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_0.value: props["moments_normalized-3-0"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_1.value: props["moments_normalized-3-1"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_2.value: props["moments_normalized-3-2"],
                        ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_3.value: props["moments_normalized-3-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_0.value: props["moments_hu-0"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_1.value: props["moments_hu-1"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_2.value: props["moments_hu-2"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_3.value: props["moments_hu-3"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_4.value: props["moments_hu-4"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_5.value: props["moments_hu-5"],
                        ObjectSizeShapeFeatures.F_HU_MOMENT_6.value: props["moments_hu-6"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_0.value: props["inertia_tensor-0-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_1.value: props["inertia_tensor-0-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_0.value: props["inertia_tensor-1-0"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_1.value: props["inertia_tensor-1-1"],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_0.value: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_1.value: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if calculate_zernikes:
                features_to_record.update(
                    {f"Zernike_{n}_{m}": zf[(n, m)] for n, m in zernike_numbers}
                )

    else:
        # 3D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)
        # SurfaceArea
        surface_areas = numpy.zeros(len(props["label"]))
        for index, label in enumerate(props["label"]):
            # this seems less elegant than you might wish, given that regionprops returns a slice,
            # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
            volume = labels[
                max(props["bbox-0"][index] - 1, 0) : min(
                    props["bbox-3"][index] + 1, labels.shape[0]
                ),
                max(props["bbox-1"][index] - 1, 0) : min(
                    props["bbox-4"][index] + 1, labels.shape[1]
                ),
                max(props["bbox-2"][index] - 1, 0) : min(
                    props["bbox-5"][index] + 1, labels.shape[2]
                ),
            ]
            volume = volume == label
            verts, faces, _normals, _values = skimage.measure.marching_cubes(
                volume,
                method="lewiner",
                spacing=spacing,
                level=0,
            )
            surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

        features_to_record = {
            ObjectSizeShapeFeatures.F_VOLUME.value: props["area"],
            ObjectSizeShapeFeatures.F_SURFACE_AREA.value: surface_areas,
            ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
            ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
            ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-2"],
            ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-1"],
            ObjectSizeShapeFeatures.F_CENTER_Z.value: props["centroid-0"],
            ObjectSizeShapeFeatures.F_BBOX_VOLUME.value: props["bbox_area"],
            ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-2"],
            ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-5"],
            ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-1"],
            ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-4"],
            ObjectSizeShapeFeatures.F_MIN_Z.value: props["bbox-0"],
            ObjectSizeShapeFeatures.F_MAX_Z.value: props["bbox-3"],
            ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
            ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
            ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
        }
        if calculate_advanced:
            features_to_record[ObjectSizeShapeFeatures.F_SOLIDITY.value] = props["solidity"]
    return features_to_record, props["label"], nobjects


########################################################
# MeasureColocalization
########################################################

def get_sum_per_object(
    im_pixels:  NDArray[Pixel], 
    mask:       NDArray[np.bool_], 
    labels:     NDArray[ObjectLabel], 
    lrange:     NDArray[np.int32]
    ) -> NDArray[np.float64]:
    """Computes the sum of the pixel internsities for each object in the lrange, uses object numbers from labels to group objects

    Args:
        im_pixels (NDArray[Pixel]): Input image pixels
        mask (NDArray[np.bool_]): Mask of where the summation should be performed
        labels (NDArray[np.int32]): Object labels for pixels in `im_pixels`
        lrange (NDArray[np.int32]): Range over which the summation should be performed

    Returns:
        NDArray[np.float64]: Pixel intensity totals for each object in `lrange`
    """
    S = np.array(
        scipy.ndimage.sum(
            im_pixels, labels[mask], lrange
        )
    ).astype(np.float64)
    return S

def get_threshold_values_for_objects(
    image_threshold_percentage: float, 
    pixels: NDArray[Pixel], 
    labels: NDArray[ObjectLabel],
    lrange: Optional[NDArray[np.int32]] = None
    ) -> NDArray[np.float64]:
    """Finds threshold values as a percentage of the maximum intensity for each object

    Args:
        image_threshold_percentage (float): Percentage value to use when finding threshold values
        pixels (NDArray[Pixel]): Array of pixel intensities
        labels (NDArray[ObjectLabel]): Segmentation labels for each pixel
        lrange (Optional[NDArray[np.int32]], optional): Range of labels over which to find the maximum. Defaults to None.

    Returns:
        NDArray[np.float64]: Returns an array of threshold values for each object. Same length as `lrange`.
    """
    lrange = lrange if lrange is not None else numpy.arange(labels.max(), dtype=numpy.int32) + 1
    object_threshold_values = (image_threshold_percentage / 100) * fix(
        scipy.ndimage.maximum(pixels, labels, lrange)
    )
    return object_threshold_values

def get_thresholded_sum(
        pixels: NDArray[Pixel],
        object_threshold_values: NDArray[np.float64],
        labels: NDArray[ObjectLabel],
        lrange: Optional[NDArray[np.int32]] = None
        ) -> NDArray[np.float64]:
    """Gets the sum of the pixels that are above the threshold value for each object grouped by label

    Args:
        pixels (NDArray[Pixel]): Array of pixel intensities
        object_threshold_values (NDArray[np.float64]): Array of threshold values for each object
        labels (NDArray[ObjectLabel]): Segmentation labels for each pixel
        lrange (Optional[NDArray[np.int32]], optional): Range of labels over which to find the sum. Defaults to None.

    Returns:
        NDArray[np.float64]: Returns an array of sums of the pixel intensities that are above the threshold value for each object. Same length as `lrange`.
    """
    lrange = lrange if lrange is not None else numpy.arange(labels.max(), dtype=numpy.int32) + 1
    return scipy.ndimage.sum(
        pixels[pixels >= object_threshold_values[labels - 1]],
        labels[pixels >= object_threshold_values[labels - 1]],
        lrange,
    ).astype(np.float64)

def get_threshold_sum_and_mask(
        image_threshold_percentage: float,
        pixels: NDArray[Pixel],
        labels: NDArray[ObjectLabel],
        lrange: Optional[NDArray[np.int32]] = None
        ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Gets the sum of the thresholds and a mask of the pixels that are above the threshold value for each object

    Args:
        image_threshold_percentage (float): Percentage value to use when finding threshold values
        pixels (NDArray[Pixel]): Array of pixel intensities
        labels (NDArray[ObjectLabel]): Segmentation labels for each pixel
        lrange (Optional[NDArray[np.int32]], optional): Range of labels over which to threshold. Defaults to None.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.bool_]]: Returns a tuple of the sum of the thresholds and a mask of the pixels that are above the threshold value for each object. 
    """
    lrange = lrange if lrange is not None else numpy.arange(labels.max(), dtype=numpy.int32) + 1
    object_threshold_values = get_threshold_values_for_objects(image_threshold_percentage, pixels, labels, lrange) # tff / tss
    thresholded_sum = get_thresholded_sum(pixels, object_threshold_values, labels, lrange) # tot_fi_thr / tot_si_thr
    threshold_mask = (pixels >= object_threshold_values[labels - 1])
    return thresholded_sum, threshold_mask

def get_thresholded_images_and_counts(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    im1_thr_percent: float,
    im2_thr_percent: float,
    labels: Optional[NDArray[ObjectLabel]] = None
) -> Tuple[
    NDArray[np.float64], 
    NDArray[np.float64], 
    NDArray[np.bool_]
    ]:
    """Finds the threshold sums and the intersection of the masks after thresholding each image

    Args:
        im1_pixels (NDArray[Pixel]): Array of pixel intensities for the first image
        im2_pixels (NDArray[Pixel]): Array of pixel intensities for the second image
        im1_thr_percent (float): Percentage value to use when finding threshold values for the first image
        im2_thr_percent (float): Percentage value to use when finding threshold values for the second image
        labels (Optional[NDArray[ObjectLabel]], optional): Segmentation labels for each pixel. Defaults to None.

    Returns:
        Tuple[ NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_] ]: Returns a tuple of the sum of the thresholds for each image and the intersection of the masks after thresholding each image.
    """

    if labels is None:
        labels = numpy.ones(im1_pixels.shape, int)
    lrange = numpy.arange(labels.max(), dtype=numpy.int32) + 1

    im1_thr_sum, im1_thr_mask = get_threshold_sum_and_mask(im1_thr_percent, im1_pixels, labels, lrange)
    im2_thr_sum, im2_thr_mask = get_threshold_sum_and_mask(im2_thr_percent, im2_pixels, labels, lrange)

    thr_mask_intersection = (im1_thr_mask) & (im2_thr_mask)

    return (
        im1_thr_sum, 
        im2_thr_sum,
        thr_mask_intersection
    )

#
# Correlation and Slope
#
def measure_correlation_and_slope(
        im1_pixels: NDArray[Pixel], 
        im2_pixels: NDArray[Pixel],
    ) -> Tuple[np.float64, np.float64]:
    #
    # Perform the correlation, which returns:
    # [ [ii, ij],
    #   [ji, jj] ]
    #
    corr = np.corrcoef((im1_pixels, im2_pixels))[1, 0]
    #
    # Find the slope as a linear regression to
    # A * i1 + B = i2
    #
    least_squares_solution = lstsq(
        np.array((im1_pixels, np.ones_like(im1_pixels))).transpose(), 
        im2_pixels)
    assert least_squares_solution is not None
    coeffs = least_squares_solution[0]
    slope = coeffs[0]
    assert slope is not None

    return corr, slope

def measure_correlation_and_slope_from_objects(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    labels: NDArray[ObjectLabel],
    lrange: NDArray[np.int32]
) -> NDArray[np.float64]:
    #
    # The correlation is sum((x-mean(x))(y-mean(y)) /
    #                         ((n-1) * std(x) *std(y)))
    #

    mean1 = fix(scipy.ndimage.mean(im1_pixels, labels, lrange))
    mean2 = fix(scipy.ndimage.mean(im2_pixels, labels, lrange))
    #
    # Calculate the standard deviation times the population.
    #
    std1 = numpy.sqrt(
        fix(
            scipy.ndimage.sum(
                (im1_pixels - mean1[labels - 1]) ** 2, labels, lrange
            )
        )
    )
    std2 = numpy.sqrt(
        fix(
            scipy.ndimage.sum(
                (im2_pixels - mean2[labels - 1]) ** 2, labels, lrange
            )
        )
    )
    x = im1_pixels - mean1[labels - 1]  # x - mean(x)
    y = im2_pixels - mean2[labels - 1]  # y - mean(y)
    corr = fix(
        scipy.ndimage.sum(
            x * y / (std1[labels - 1] * std2[labels - 1]), labels, lrange
        )
    )
    # Explicitly set the correlation to NaN for masked objects
    corr[scipy.ndimage.sum(1, labels, lrange) == 0] = numpy.NaN
    return corr

#
# Mander's Coefficient
#
def get_manders_coefficient(
        im_thr_common_pixels: NDArray[Pixel],
        thr_mask_intersection: NDArray[np.bool_],
        im_thr_sum: NDArray[np.float64],
        labels: NDArray[ObjectLabel],
        lrange: NDArray[np.int32]
        ):
    M = get_sum_per_object(im_thr_common_pixels, thr_mask_intersection, labels, lrange)
    M = M / im_thr_sum
    return M

def measure_manders_coefficient(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    im1_thr_sum: np.float64,
    im2_thr_sum: np.float64,
    thr_mask_intersection: NDArray[np.bool_],
    ) -> Tuple[np.float64, np.float64]:

    im1_thr_common_pixels = im1_pixels[thr_mask_intersection] # fi_thresh
    im2_thr_common_pixels = im2_pixels[thr_mask_intersection] # si_thresh
    # Manders Coefficient
    M1 = 0
    M2 = 0
    M1 = im1_thr_common_pixels.sum() / im1_thr_sum
    M2 = im2_thr_common_pixels.sum() / im2_thr_sum

    return M1, M2

def measure_manders_coefficient_from_objects(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    im1_thr_sum: NDArray[np.float64],
    im2_thr_sum: NDArray[np.float64],
    thr_mask_intersection: NDArray[np.bool_],
    labels: NDArray[ObjectLabel],
    lrange: NDArray[np.int32],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    # TODO: Many methods use this. Should it be cached?
    im1_thr_common_pixels = im1_pixels[thr_mask_intersection]
    im2_thr_common_pixels = im2_pixels[thr_mask_intersection]
    # Manders Coefficient
    M1 = numpy.zeros(len(lrange))
    M2 = numpy.zeros(len(lrange))

    if thr_mask_intersection is not None and numpy.any(thr_mask_intersection):
        M1 = get_manders_coefficient(im1_thr_common_pixels, thr_mask_intersection, im1_thr_sum, labels, lrange)
        M2 = get_manders_coefficient(im2_thr_common_pixels, thr_mask_intersection, im2_thr_sum, labels, lrange)

    return M1, M2

#
# Rank Weighted Coefficient
#
def get_image_rank(
        im_pixels: NDArray[Pixel], 
        labels: Optional[NDArray[ObjectLabel]]=None
    ) -> NDArray[np.int32]:

    if labels is None:
        Rank = np.lexsort([im_pixels])
    else:
        [Rank] = np.lexsort(([labels], [im_pixels]))

    Rank_U = np.hstack([[False], im_pixels[Rank[:-1]] != im_pixels[Rank[1:]]])
    Rank_S = np.cumsum(Rank_U)
    Rank_im = np.zeros(im_pixels.shape, dtype=int)
    Rank_im[Rank] = Rank_S
    return Rank_im

def calculate_rank_weight(
        Rank_im1: NDArray[np.int32], 
        Rank_im2: NDArray[np.int32]
    ) -> NDArray[np.float64]:

    R = max(Rank_im1.max(), Rank_im2.max()) + 1
    Di = abs(Rank_im1 - Rank_im2)
    weight = ((R - Di) * 1.0) / R
    return weight
    
def get_rwc_coefficient(
        im1_thr_common_pixels: NDArray[Pixel],
        weight_thresh: NDArray[np.float64],
        thr_mask_intersection: NDArray[np.bool_],
        im1_thr_sum: NDArray[np.float64],
        labels: NDArray[ObjectLabel],
        lrange: NDArray[np.int32]
        ):
    weighted_pixels = im1_thr_common_pixels * weight_thresh
    RWC = get_sum_per_object(weighted_pixels, thr_mask_intersection, labels, lrange)
    RWC = RWC / np.array(im1_thr_sum)
    return RWC

def measure_rwc_coefficient(
        im1_pixels: NDArray[Pixel], 
        im2_pixels: NDArray[Pixel],
        im1_thr_sum: np.float64,
        im2_thr_sum: np.float64,
        thr_mask_intersection: NDArray[np.bool_]
    ) -> Tuple[np.float64, np.float64]:

    im1_thr_common_pixels = im1_pixels[thr_mask_intersection]
    im2_thr_common_pixels = im2_pixels[thr_mask_intersection]
    # RWC Coefficient
    RWC1 = 0
    RWC2 = 0
    Rank_im1 = get_image_rank(im1_pixels)
    Rank_im2 = get_image_rank(im2_pixels)
    weight = calculate_rank_weight(Rank_im1, Rank_im2)
    weight_thresh = weight[thr_mask_intersection]
    RWC1 = (im1_thr_common_pixels * weight_thresh).sum() / im1_thr_sum
    RWC2 = (im2_thr_common_pixels * weight_thresh).sum() / im2_thr_sum

    return RWC1, RWC2

def measure_rwc_coefficient_from_objects(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    im1_thr_sum: NDArray[np.float64],
    im2_thr_sum: NDArray[np.float64],
    thr_mask_intersection: NDArray[np.bool_],
    labels: NDArray[ObjectLabel],
    lrange: NDArray[np.int32],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # RWC Coefficient
    RWC1 = np.zeros(len(lrange))
    RWC2 = np.zeros(len(lrange))

    im1_thr_common_pixels = im1_pixels[thr_mask_intersection]
    im2_thr_common_pixels = im2_pixels[thr_mask_intersection]
    
    Rank_im1 = get_image_rank(im1_pixels, labels)
    Rank_im2 = get_image_rank(im2_pixels, labels)
    weight = calculate_rank_weight(Rank_im1, Rank_im2)
    weight_thresh = weight[thr_mask_intersection]

    if thr_mask_intersection is not None and np.any(thr_mask_intersection):
        RWC1 = get_rwc_coefficient(im1_thr_common_pixels, weight_thresh, thr_mask_intersection, im1_thr_sum, labels, lrange)
        RWC2 = get_rwc_coefficient(im2_thr_common_pixels, weight_thresh, thr_mask_intersection, im2_thr_sum, labels, lrange)
    
    return RWC1, RWC2

#
# Overlap
#
def measure_overlap_coefficient(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    thr_mask_intersection: NDArray[np.bool_],
    ) -> Tuple[np.float64, np.float64, np.float64]:

    im1_thr_common_pixels = im1_pixels[thr_mask_intersection]
    im2_thr_common_pixels = im2_pixels[thr_mask_intersection]
    # Overlap Coefficient
    overlap = 0
    overlap = (im1_thr_common_pixels * im2_thr_common_pixels).sum() / np.sqrt(
        (im1_thr_common_pixels ** 2).sum() * (im2_thr_common_pixels ** 2).sum()
    )
    K1 = (im1_thr_common_pixels * im2_thr_common_pixels).sum() / (im1_thr_common_pixels ** 2).sum()
    K2 = (im1_thr_common_pixels * im2_thr_common_pixels).sum() / (im2_thr_common_pixels ** 2).sum()

    return overlap, K1, K2

def measure_overlap_coefficient_from_objects(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    thr_mask_intersection: Optional[NDArray[np.bool_]],
    labels: NDArray[ObjectLabel],
    lrange: NDArray[np.int32]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Overlap Coefficient
    if thr_mask_intersection is not None and numpy.any(thr_mask_intersection):
        im1_pixels_sq = im1_pixels[thr_mask_intersection] ** 2
        im2_pixels_sq = im2_pixels[thr_mask_intersection] ** 2

        fpsq = get_sum_per_object(im1_pixels_sq, thr_mask_intersection, labels, lrange)
        spsq = get_sum_per_object(im2_pixels_sq, thr_mask_intersection, labels, lrange)
        pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
        
        image_pixels_product = im1_pixels[thr_mask_intersection] * im2_pixels[thr_mask_intersection]
        sum_of_intenseties_per_object = get_sum_per_object(image_pixels_product, thr_mask_intersection, labels, lrange)
        
        overlap = fix(sum_of_intenseties_per_object / pdt)
        K1 = fix(sum_of_intenseties_per_object / numpy.array(fpsq))
        K2 = fix(sum_of_intenseties_per_object / numpy.array(spsq))

    else:
        overlap = K1 = K2 = numpy.zeros(len(lrange))


    return overlap, K1, K2

#
# Costes Thresholded Mander's Coefficient
#
def get_costes_thresholded_pixels_and_mask(
        im1_pixels: NDArray[Pixel],
        im2_pixels: NDArray[Pixel],
        im1_costes_pixels: NDArray[Pixel],
        im2_costes_pixels: NDArray[Pixel],
        im1_scale: Optional[np.float64],
        im2_scale: Optional[np.float64],
        costes_method: CostesMethod
        ) -> Tuple[
            NDArray[np.bool_], 
            NDArray[np.bool_], 
            NDArray[np.bool_],
            NDArray[Pixel],
            NDArray[Pixel]
            ]:
    # Orthogonal Regression for Costes' automated threshold
    scale = get_scale_for_costes_threshold(im1_scale, im2_scale)
    costes_function: Dict[CostesMethod, Callable[[NDArray[Pixel], NDArray[Pixel], np.float64], Tuple[np.float64, np.float64]]] = {
        CostesMethod.FASTER: bisection_costes,
        CostesMethod.ACCURATE: linear_costes,
        CostesMethod.FAST: linear_costes,
    }
    im1_costes_thr_val, im2_costes_thr_val = costes_function[costes_method](im1_costes_pixels, im2_costes_pixels, scale)
    # Costes' thershold for entire image is applied to each object
    im1_costes_thr_mask = im1_pixels >= im1_costes_thr_val
    im2_costes_thr_mask = im2_pixels >= im2_costes_thr_val
    combined_thresh_c = im1_costes_thr_mask & im2_costes_thr_mask
    im1_costes_thr_common_pixels = im1_pixels[combined_thresh_c]
    im2_costes_thr_common_pixels = im2_pixels[combined_thresh_c]
    return im1_costes_thr_mask, im2_costes_thr_mask, combined_thresh_c, im1_costes_thr_common_pixels, im2_costes_thr_common_pixels

def get_costes_threshold_pixel_sum(
        im_costes_thr_mask: NDArray[np.bool_],
        im_pixels: NDArray[Pixel],
        labels: NDArray[ObjectLabel],
        lrange: NDArray[np.int32]
        ) -> NDArray[np.float64]:
    if numpy.any(im_costes_thr_mask):
        im_costes_thr_sum = scipy.ndimage.sum(
            im_pixels[im_costes_thr_mask],
            labels[im_costes_thr_mask],
            lrange,
        ).astype(np.float64)
    else:
        im_costes_thr_sum = numpy.zeros(len(lrange))

    return im_costes_thr_sum

def get_costes_coefficient(
        im_costes_thr_common_pixels: NDArray[Pixel],
        combined_thresh_c: NDArray[np.bool_],
        im_costes_thr_sum: NDArray[np.float64],
        labels: NDArray[ObjectLabel],
        lrange: NDArray[np.int32]
        ):
    C = get_sum_per_object(im_costes_thr_common_pixels, combined_thresh_c, labels, lrange) 
    C = C / np.array(im_costes_thr_sum)
    return C

def measure_costes_coefficient(
        im1_pixels: NDArray[Pixel], 
        im2_pixels: NDArray[Pixel],
        im1_scale: Optional[float] = None,
        im2_scale: Optional[float] = None,
        costes_method: CostesMethod = CostesMethod.FAST,
    ) -> Tuple[ np.float64, np.float64]:
    #
    # Find the Costes threshold for each image
    #
    (
        im1_costes_thr_mask, 
        im2_costes_thr_mask,
        combined_thresh_c, 
        im1_costes_thr_common_pixels, 
        im2_costes_thr_common_pixels  
    ) = get_costes_thresholded_pixels_and_mask(
        im1_pixels, 
        im2_pixels, 
        im1_pixels, 
        im2_pixels, 
        im1_scale, 
        im2_scale, 
        costes_method
    )
    im1_costes_thr_sum = im1_pixels[(im1_costes_thr_mask)].sum()
    im2_costes_thr_sum = im2_pixels[(im2_costes_thr_mask)].sum()

    # Costes' Automated Threshold
    C1 = 0
    C2 = 0
    C1 = im1_costes_thr_common_pixels.sum() / im1_costes_thr_sum
    C2 = im2_costes_thr_common_pixels.sum() / im2_costes_thr_sum


    return C1, C2

def measure_costes_coefficient_from_objects(
    im1_pixels: NDArray[Pixel],
    im2_pixels: NDArray[Pixel],
    im1_costes_pixels: NDArray[Pixel],
    im2_costes_pixels: NDArray[Pixel],
    labels: NDArray[ObjectLabel],
    lrange: NDArray[np.int32],
    im1_scale: numpy.float64,
    im2_scale: numpy.float64,
    costes_method: CostesMethod = CostesMethod.FAST,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    (
        im1_costes_thr_mask, 
        im2_costes_thr_mask, 
        combined_thresh_c, 
        im1_costes_thr_common_pixels, 
        im2_costes_thr_common_pixels 
    ) = get_costes_thresholded_pixels_and_mask (
        im1_pixels = im1_pixels, 
        im2_pixels = im2_pixels, 
        im1_costes_pixels = im1_costes_pixels, 
        im2_costes_pixels = im2_costes_pixels, 
        im1_scale = im1_scale, 
        im2_scale = im2_scale, 
        costes_method = costes_method
    )
    im1_costes_thr_sum = get_costes_threshold_pixel_sum(im1_costes_thr_mask, im1_pixels, labels, lrange)
    im2_costes_thr_sum = get_costes_threshold_pixel_sum(im2_costes_thr_mask, im2_pixels, labels, lrange)

    # Costes Automated Threshold
    C1 = numpy.zeros(len(lrange))
    C2 = numpy.zeros(len(lrange))
    if numpy.any(combined_thresh_c):
        C1 = get_costes_coefficient(im1_costes_thr_common_pixels, combined_thresh_c, im1_costes_thr_sum, labels, lrange)
        C2 = get_costes_coefficient(im2_costes_thr_common_pixels, combined_thresh_c, im2_costes_thr_sum, labels, lrange)


    return C1, C2

def get_scale_for_costes_threshold(scale_1: Optional[np.float64], scale_2: Optional[np.float64]) -> np.float64:
    if scale_1 is not None and scale_2 is not None:
        return max(scale_1, scale_2)
    elif scale_1 is not None:
        return scale_1
    elif scale_2 is not None:
        return scale_2
    else:
        return np.float64(255)

def bisection_costes(
        im1_costes_pixels: NDArray[Pixel], 
        im2_costes_pixels: NDArray[Pixel], 
        scale_max:np.float64=np.float64(255)
        ) -> Tuple[np.float64, np.float64]:
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point below 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (im1_costes_pixels > 0) | (im2_costes_pixels > 0)
    xvar = np.var(im1_costes_pixels[non_zero], axis=0, ddof=1)
    yvar = np.var(im2_costes_pixels[non_zero], axis=0, ddof=1)

    xmean = np.mean(im1_costes_pixels[non_zero], axis=0)
    ymean = np.mean(im2_costes_pixels[non_zero], axis=0)

    z = im1_costes_pixels[non_zero] + im2_costes_pixels[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6/5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_fi_c = mid / scale_max
        thr_si_c = (a * thr_fi_c) + b
        combt: NDArray[np.bool_] = (im1_costes_pixels < thr_fi_c) | (im2_costes_pixels < thr_si_c)
        if np.count_nonzero(combt) <= 2:
            # Can't run pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(im1_costes_pixels[combt], im2_costes_pixels[combt])
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_fi_c = (valid - 1) / scale_max
    thr_si_c = (a * thr_fi_c) + b

    return thr_fi_c, thr_si_c

def linear_costes(
        im1_costes_pixels: NDArray[Pixel], 
        im2_costes_pixels: NDArray[Pixel], 
        scale_max:np.float64=np.float64(255), 
        costes_method: CostesMethod = CostesMethod.FAST
        ) -> Tuple[np.float64, np.float64]:
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0.
    """
    i_step = 1 / scale_max
    non_zero = (im1_costes_pixels > 0) | (im2_costes_pixels > 0)
    xvar = np.var(im1_costes_pixels[non_zero], axis=0, ddof=1)
    yvar = np.var(im2_costes_pixels[non_zero], axis=0, ddof=1)

    xmean = np.mean(im1_costes_pixels[non_zero], axis=0)
    ymean = np.mean(im2_costes_pixels[non_zero], axis=0)

    z = im1_costes_pixels[non_zero] + im2_costes_pixels[non_zero]
    zvar = np.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + np.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(im1_costes_pixels.max(), im2_costes_pixels.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    fi_max = im1_costes_pixels.max()
    si_max = im2_costes_pixels.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(im1_costes_pixels, im2_costes_pixels)
    thr_fi_c = i
    thr_si_c = (a * i) + b
    while i > fi_max and (a * i) + b > si_max:
        i -= i_step
    while i > i_step:
        thr_fi_c = i
        thr_si_c = (a * i) + b
        combt = (im1_costes_pixels < thr_fi_c) | (im2_costes_pixels < thr_si_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := np.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(im1_costes_pixels[combt], im2_costes_pixels[combt])
                num_true = positives

            if costReg <= 0:
                break
            elif costes_method == CostesMethod.ACCURATE or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_fi_c, thr_si_c


###############################################################################
# Measure Granularity
###############################################################################
def get_granularity_measurements(
        im_pixel_data: ImageGrayscale,
        pixels: ImageGrayscale, 
        mask: ImageGrayscaleMask, 
        new_shape: NDArray[numpy.float64],
        granular_spectrum_length: int,
        dimensions: int,
        object_records: List[ObjectRecord]
        ):
    # TODO: PR #5034 update the return types of this funciton
    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    footprint = get_morphology_footprint(1, dimensions)
    ng = granular_spectrum_length
    startmean = np.mean(pixels[mask])
    ero = pixels.copy()
    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    #
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, np.finfo(float).eps)
    measurements_arr = []
    image_measurements_arr = []

    statistics = []
    for i in range(1, ng + 1):
        prevmean = currentmean
        ero = masked_erode(ero, mask, footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = np.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        statistics += ["%.2f" % gs]
        # feature = C_GRANULARITY % (i, image_name)
        image_measurements_arr += [gs]
        # measurements.add_image_measurement(feature, gs)
        #
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        #
        orig_shape = im_pixel_data.shape
        rec = restore_scale(dimensions, orig_shape, new_shape, rec)

        #
        # Calculate the means for the objects
        #
        obj_measurements=[]
        for object_record in object_records:
            assert isinstance(object_record, ObjectRecord)
            if object_record.nobjects > 0:
                new_mean = fix(
                    scipy.ndimage.mean(
                        rec, object_record.labels, object_record.range
                    )
                )
                gss = (
                    (object_record.current_mean - new_mean)
                    * 100
                    / object_record.start_mean
                )
                object_record.current_mean = new_mean
            else:
                gss = np.zeros((0,))
            obj_measurements += [[object_record.name, gss]]
        measurements_arr += [obj_measurements]
    return measurements_arr, image_measurements_arr, statistics

#
# For each object, build a little record
#
class ObjectRecord(object):
    def __init__(self, name, segmented, im_mask, im_pixel_data):
        self.name = name
        self.labels = segmented
        self.nobjects = np.max(self.labels)
        if self.nobjects != 0:
            self.range = np.arange(1, np.max(self.labels) + 1)
            self.labels = self.labels.copy()
            self.labels[~im_mask] = 0
            self.current_mean = fix(
                scipy.ndimage.mean(im_pixel_data, self.labels, self.range)
            )
            self.start_mean = np.maximum(
                self.current_mean, np.finfo(float).eps
            )
            