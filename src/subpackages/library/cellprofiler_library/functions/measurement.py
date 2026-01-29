import numpy as np
import numpy
import scipy
import scipy.ndimage
import scipy.sparse
import skimage
import centrosome
import centrosome.cpmorphology
import centrosome.filter
import centrosome.propagate
import centrosome.fastemd
import centrosome.index
import centrosome.threshold
import centrosome.haralick
import centrosome.radial_power_spectrum
from centrosome.cpmorphology import strel_disk, centers_of_labels
from centrosome.outline import outline
from functools import reduce
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from sklearn.cluster import KMeans
from pydantic import validate_call, ConfigDict
from typing import Tuple, Optional, Dict, Callable, List, Union, Any, Sequence
from scipy.linalg import lstsq
from scipy.ndimage import grey_dilation, grey_erosion
from numpy.typing import NDArray


from cellprofiler_library.opts import measureimageoverlap as mio
from cellprofiler_library.functions.segmentation import count_from_ijv
from cellprofiler_library.functions.segmentation import areas_from_ijv
from cellprofiler_library.functions.segmentation import cast_labels_to_label_set
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv
from cellprofiler_library.functions.image_processing import masked_erode, restore_scale, get_morphology_footprint
from cellprofiler_library.functions.segmentation import relate_labels

from cellprofiler_library.types import Pixel, ObjectLabel, ImageGrayscale, ImageGrayscaleMask, ImageAny, ImageBinary, ImageBinaryMask, ObjectSegmentation, ObjectLabelsDense, ObjectLabelSet, ObjectSegmentationIJV, Image2DBinary, Image2DColor, Image2DGrayscale
from cellprofiler_library.opts.objectsizeshapefeatures import ObjectSizeShapeFeatures, ZERNIKE_N
from cellprofiler_library.opts.measurecolocalization import CostesMethod
from cellprofiler_library.opts.measureobjectoverlap import DecimationMethod as ObjectDecimationMethod
from cellprofiler_library.opts.measureobjectskeleton import VF_I, VF_J, VF_LABELS, VF_KIND, EF_V1, EF_V2, EF_LENGTH, EF_TOTAL_INTENSITY
from cellprofiler_library.opts.measureobjectneighbors import DistanceMethod as NeighborsDistanceMethod
from cellprofiler_library.opts.measureobjectneighbors import Measurement as NeighborsMeasurement
from cellprofiler_library.opts.measureobjectneighbors import MeasurementScale as NeighborsMeasurementScale

###############################################################################
# MeasureImageOverlap
###############################################################################

def measure_image_overlap_statistics(
    ground_truth_image: ImageBinary,
    test_image: ImageBinary,
    mask: Optional[ImageBinaryMask]=None,
) -> Dict[str, numpy.float_]:
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


def compute_rand_index(
        test_labels: ObjectSegmentation, 
        ground_truth_labels: ObjectSegmentation, 
        mask: Optional[ImageBinaryMask]=None
        ) -> Tuple[numpy.float_, numpy.float_]:
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
    ground_truth_image: ImageBinary,
    test_image: ImageBinary,
    mask: Optional[ImageBinary]=None,
    decimation_method: mio.DM = mio.DM.KMEANS,
    max_distance: int = 250,
    max_points: int = 250,
    penalize_missing: bool = False,
) -> numpy.float_:
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
    dest_labels = scipy.ndimage.label(ground_truth_image & mask, np.ones((3, 3), bool))[0]
    src_labels = scipy.ndimage.label(test_image & mask, np.ones((3, 3), bool))[0]
    dest_labelset = cast_labels_to_label_set(dest_labels)
    src_labelset = cast_labels_to_label_set(src_labels)
    emd = compute_earth_movers_distance_objects(
        src_objects_label_set=src_labelset,
        dest_objects_label_set=dest_labelset,
        penalize_missing=penalize_missing,
        max_distance=max_distance,
        max_points=max_points,
        decimation_method=decimation_method,
    )
    return emd
    

def get_labels_mask(labelset, shape):
    labels_mask = np.zeros(shape, bool)
    for labels, indexes in labelset:
        labels_mask = labels_mask | labels > 0
    return labels_mask


def get_skeleton_points(
        labelset: ObjectLabelSet, 
        shape: Tuple[int, int], 
        max_points: int=250,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
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


def get_kmeans_points(
        src_ijv: ObjectSegmentationIJV, 
        dest_ijv: ObjectSegmentationIJV, 
        max_points: int=250
    ) -> Tuple[NDArray[numpy.int_], NDArray[numpy.int_]]:
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
    return ( # TODO: why uint32 below?
        kmeans.cluster_centers_[:, 0].astype(np.uint32).astype(np.int64),
        kmeans.cluster_centers_[:, 1].astype(np.uint32).astype(np.int64),
    )


def get_weights(
        i: NDArray[numpy.int_],
        j: NDArray[numpy.int_],
        labels_mask: NDArray[numpy.bool_],
        ) -> NDArray[numpy.int_]:
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


################################################################################
# MeasureObjectSizeShape
################################################################################

def get_object_derived_shape_features(
    area: NDArray[np.float64],
    perimeter: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute derived shape features from area and perimeter.
    
    Args:
        area: Array of object areas
        perimeter: Array of object perimeters
        
    Returns:
        Tuple of (formfactor, compactness) arrays
    """
    formfactor = 4.0 * numpy.pi * area / perimeter ** 2
    denom = [max(x, 1) for x in 4.0 * numpy.pi * area]
    compactness = perimeter ** 2 / denom
    return formfactor, compactness


def get_object_radius_features(
    images: List[NDArray[np.bool_]]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute radius features for objects from their binary images.
    
    Args:
        images: List of binary images for each object
        
    Returns:
        Tuple of (max_radius, mean_radius, median_radius) arrays
    """
    nobjects = len(images)
    max_radius = numpy.zeros(nobjects)
    median_radius = numpy.zeros(nobjects)
    mean_radius = numpy.zeros(nobjects)
    
    for index, mini_image in enumerate(images):
        # Pad image to assist distance transform
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
    
    return max_radius, mean_radius, median_radius


def get_object_zernike_features(
    labels: NDArray[ObjectLabel],
    label_indices: NDArray[np.int32],
    zernike_numbers: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], NDArray[np.float64]]:
    """Compute Zernike features for objects.
    
    Args:
        labels: Label matrix
        label_indices: Array of object label indices
        zernike_numbers: List of (N, M) Zernike polynomial indices
        
    Returns:
        Dictionary mapping (N, M) tuples to arrays of Zernike coefficients
    """
    zf = {}
    zf_l = centrosome.zernike.zernike(zernike_numbers, labels, label_indices)
    for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
        zf[(n, m)] = z
    return zf


def get_object_feret_features(
    labels: NDArray[ObjectLabel],
    label_indices: NDArray[np.int32]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Feret diameter features for objects.
    
    Args:
        labels: Label matrix
        label_indices: Array of object label indices
        
    Returns:
        Tuple of (min_feret_diameter, max_feret_diameter) arrays
    """
    if len(label_indices) == 0:
        return numpy.zeros(0), numpy.zeros(0)
    
    chulls, chull_counts = centrosome.cpmorphology.convex_hull(
        labels, label_indices
    )
    min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(
        chulls, chull_counts, label_indices
    )
    return min_feret_diameter, max_feret_diameter


def measure_object_size_shape_2d(
    labels: NDArray[ObjectLabel],
    desired_properties: List[str],
    calculate_zernikes: bool,
    spacing: Tuple[float, ...]
) -> Tuple[
    Dict[str, Any], # props
    Tuple[NDArray[np.float64], NDArray[np.float64]], # formfactor, compactness
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], # max, mean, median radius
    Dict[Tuple[int, int], NDArray[np.float64]], # zernike
    Tuple[NDArray[np.float64], NDArray[np.float64]] # feret
]:
    """Compute 2D object size and shape measurements.
    
    Args:
        labels: 2D label matrix
        desired_properties: List of property names to compute
        calculate_zernikes: Whether to compute Zernike features
        spacing: Pixel spacing tuple
        
    Returns:
        Tuple of (props, (formfactor, compactness), (max_r, mean_r, median_r), 
                  zernike_features, (min_feret, max_feret))
    """
    label_indices = numpy.unique(labels[labels != 0])
    
    props = skimage.measure.regionprops_table(labels, properties=desired_properties)
    
    formfactor, compactness = get_object_derived_shape_features(props["area"], props["perimeter"])
    
    max_r, mean_r, median_r = get_object_radius_features(props["image"])
    
    zernike_features = {}
    if calculate_zernikes:
        zernike_numbers = centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
        zernike_features = get_object_zernike_features(labels, label_indices, zernike_numbers)
        
    min_feret, max_feret = get_object_feret_features(labels, label_indices)
    
    return props, (formfactor, compactness), (max_r, mean_r, median_r), zernike_features, (min_feret, max_feret)

def measure_object_size_shape_3d(
    labels: NDArray[ObjectLabel],
    desired_properties: List[str],
    spacing: Tuple[float, ...]
) -> Tuple[
    Dict[str, Any], # props
    NDArray[np.float64] # surface_areas
]:
    props = skimage.measure.regionprops_table(labels, properties=desired_properties)
    
    # SurfaceArea
    surface_areas = numpy.zeros(len(props["label"]))
    for index, label in enumerate(props["label"]):
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
        
    return props, surface_areas





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

class ObjectRecord(object):
    def __init__(
            self, 
            name: str, 
            segmented: ObjectSegmentation, 
            im_mask: Optional[ImageGrayscaleMask], 
            im_pixel_data: Optional[ImageGrayscale]
            ):
        self.name = name
        self.labels = segmented
        self.nobjects = np.max(self.labels)
        if self.nobjects != 0:
            assert im_mask is not None
            assert im_pixel_data is not None
            self.range = np.arange(1, np.max(self.labels) + 1)
            self.labels = self.labels.copy()
            self.labels[~im_mask] = 0
            self.current_mean = fix(
                scipy.ndimage.mean(im_pixel_data, self.labels, self.range)
            )
            self.start_mean = np.maximum(
                self.current_mean, np.finfo(float).eps
            )


def get_granularity_measurements(
        im_pixel_data: ImageGrayscale,
        pixels: ImageGrayscale, 
        mask: ImageGrayscaleMask, 
        new_shape: NDArray[numpy.float64],
        granular_spectrum_length: int,
        dimensions: int,
        object_records: List[ObjectRecord]
        ) -> Tuple[List[List[Any]], List[float]]:
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

    for i in range(1, ng + 1):
        prevmean = currentmean
        ero = masked_erode(ero, mask, footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = np.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
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
    return measurements_arr, image_measurements_arr


################################################################################
# MeasureImageAreaOccupied
################################################################################

def measure_surface_area(
        label_image: Union[ImageBinary, ObjectLabelsDense], 
        spacing: Optional[Tuple[float, ...]]=None, 
        index: Optional[NDArray[np.int32]]=None,
        ) -> NDArray[np.float64]:
    if spacing is None:
        spacing = (1.0,) * label_image.ndim

    if index is None:
        verts, faces, _, _ = skimage.measure.marching_cubes(
            label_image, spacing=spacing, level=0, method="lorensen"
        )

        return skimage.measure.mesh_surface_area(verts, faces)

    return np.sum(
        [
            np.round(measure_label_surface_area(label_image, label, spacing))
            for label in index
        ]
    )


def measure_perimeter(im_pixel_data: ImageBinary, im_volumetric: bool, im_spacing: Optional[Tuple[float, ...]] = None) -> np.float_:
    if im_volumetric:
        perimeter = measure_surface_area(im_pixel_data > 0, spacing=im_spacing)
    else:
        perimeter = skimage.measure.perimeter(im_pixel_data > 0)
    return perimeter

def measure_area_occupied(im_pixel_data: ImageBinary) -> np.float_:
    return np.sum(im_pixel_data > 0)

def measure_total_area(im_pixel_data: ImageBinary) -> np.int_:
    return np.prod(np.shape(im_pixel_data))


def measure_object_perimeter(
        label_image: ObjectLabelsDense,
        mask: Optional[ImageAny] = None,
        regionprops: Optional[List[Any]] = None,
        volumetric: bool = False,
        spacing: Optional[Tuple[float, ...]] = None
        ) -> np.float_:
    """ Uses skimage.measure.regionprops to calculate the perimeter for 2D. Uses skimage.measure.mesh_surface_area to calculate the perimeter for 3D"""
    if regionprops is None:
        if mask is not None:
            label_image[~mask] = 0
        regionprops = skimage.measure.regionprops(label_image)
    if volumetric:
        labels = np.unique(label_image)
        if labels[0] == 0:
            labels = labels[1:]
        
    if volumetric:
        perimeter = measure_surface_area(label_image, spacing=spacing, index=labels)
    else:
        perimeter = np.sum(
            [np.round(region["perimeter"]) for region in regionprops]
        )
    return perimeter

def measure_objects_area_occupied(
    label_image: Optional[ObjectLabelsDense],
    mask: Optional[ImageAny] = None,
    regionprops: Optional[List[Any]] = None,
    ) -> np.float_:
    """ Area occupied can either be calculated from the label image (with/without mask) or if regionprops are already computed, they can be passed in"""
    if regionprops is None:
        assert label_image is not None, "Either label_image or region_properties must be provided"
        if mask is not None:
            label_image[~mask] = 0
        regionprops = skimage.measure.regionprops(label_image)

    return np.sum([region["area"] for region in regionprops])

def measure_objects_total_area(
    label_image: Optional[ObjectLabelsDense],
    mask: Optional[ImageAny] = None,
    ) -> np.int_:
    """ Total area can either be calculated from the label image or from the mask"""
    if mask is not None:
        total_area = np.sum(mask)
    else:
        total_area = np.product(label_image.shape)
    return total_area



def measure_label_surface_area(
        label_image: ObjectLabelsDense, 
        label: int, 
        spacing: Tuple[float, ...]
        ) -> float:
    verts, faces, _, _ = skimage.measure.marching_cubes(
        label_image == label, spacing=spacing, level=0, method="lorensen"
    )

    return skimage.measure.mesh_surface_area(verts, faces)


###############################################################################
# MeasureImageIntensity
###############################################################################

def measure_image_intensities(
        pixels: NDArray[np.float32], 
        percentiles: List[int]=[]
    ) -> Tuple[List[float], Dict[int, float]]:
    pixel_count = numpy.product(pixels.shape)
    percentile_measures = {}
    if pixel_count == 0:
        pixel_sum = 0
        pixel_mean = 0
        pixel_std = 0
        pixel_mad = 0
        pixel_median = 0
        pixel_min = 0
        pixel_max = 0
        pixel_pct_max = 0
        pixel_lower_qrt = 0
        pixel_upper_qrt = 0
        if percentiles:
            for percentile in percentiles:
                percentile_measures[percentile] = 0
    else:
        pixels = pixels.flatten()
        pixels = pixels[
            numpy.nonzero(numpy.isfinite(pixels))[0]
        ]  # Ignore NaNs, Infs
        pixel_count = numpy.product(pixels.shape)

        pixel_sum = numpy.sum(pixels)
        pixel_mean = pixel_sum / float(pixel_count)
        pixel_std = numpy.std(pixels)
        pixel_median = numpy.median(pixels)
        pixel_mad = numpy.median(numpy.abs(pixels - pixel_median))
        pixel_min = numpy.min(pixels)
        pixel_max = numpy.max(pixels)
        pixel_pct_max = (
            100.0 * float(numpy.sum(pixels == pixel_max)) / float(pixel_count)
        )
        pixel_lower_qrt, pixel_upper_qrt = numpy.percentile(pixels, [25, 75])

        if percentiles:
            percentile_results = numpy.percentile(pixels, percentiles)
            for percentile, res in zip(percentiles, percentile_results):
                percentile_measures[percentile] = res
    return (
        pixel_sum,
        pixel_mean,
        pixel_median,
        pixel_std,
        pixel_mad,
        pixel_max,
        pixel_min,
        pixel_count,
        pixel_pct_max,
        pixel_lower_qrt,
        pixel_upper_qrt,
    ), percentile_measures


###############################################################################
# MeasureObjectIntensity
###############################################################################


def measure_object_area_occupied(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.int_]:
    return centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(numpy.ones(len(limg)), llabels, lindexes))

def measure_integrated_intensity(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
    return centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(limg, llabels, lindexes))

def measure_mean_intensity(integrated_intensity: NDArray[numpy.float_], lcount: NDArray[numpy.int_]) -> NDArray[numpy.float_]:   
    return integrated_intensity / lcount

def measure_std_intensity(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_], mean_intensity: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
    #
    # This function takes in mean_intensity as an array where each element is the mean intensity of the corresponding label in lindexes
    # which is then converted into a 1D array of pixels where each pixel is the mean intensity of the corresponding label in llabels
    # It's done this way as it makes the code more readable in the main function
    #
    mean_intensity_per_label = numpy.zeros((max(lindexes),))
    mean_intensity_per_label[lindexes - 1] = mean_intensity
    # mean_intensity[llabels - 1] replaces label numbers with mean intensity creating a 1D array of intensities
    mean_intensity_pixels = mean_intensity_per_label[llabels - 1]
    return numpy.sqrt(
        centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(
                (limg - mean_intensity_pixels) ** 2,
                llabels,
                lindexes,
            )
        )
    )
def measure_min_intensity(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
    return centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.minimum(limg, llabels, lindexes)
    )

def measure_max_intensity(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
    return centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.maximum(limg, llabels, lindexes)
    )

def measure_max_position(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
    # Compute the position of the intensity maximum
    max_position = numpy.array(
        centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.maximum_position(limg, llabels, lindexes)
        ),
        dtype=int,
    )
    max_position = numpy.reshape(
        max_position, (max_position.shape[0],)
    )
    return max_position

def measure_center_of_mass_binary(coordinates: NDArray[numpy.int_], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
    cm = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.mean(coordinates, llabels, lindexes)
        )
    return cm

def measure_center_of_mass_intensity(coordinates_mesh: NDArray[numpy.int_], limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_], integrated_intensity: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
    coordinate_scaled_pixels = centrosome.cpmorphology.fixup_scipy_ndimage_result(
        scipy.ndimage.sum(coordinates_mesh * limg, llabels, lindexes)
    )
    center_of_mass_intensity = coordinate_scaled_pixels / integrated_intensity
    return center_of_mass_intensity

def measure_mass_displacement(center_of_mass_binary: Tuple[NDArray[numpy.float_], NDArray[numpy.float_], NDArray[numpy.float_]], center_of_mass_intensity: Tuple[NDArray[numpy.float_], NDArray[numpy.float_], NDArray[numpy.float_]]) -> NDArray[numpy.float_]:
    diff_x = center_of_mass_binary[0] - center_of_mass_intensity[0]
    diff_y = center_of_mass_binary[1] - center_of_mass_intensity[1]
    diff_z = center_of_mass_binary[2] - center_of_mass_intensity[2]
    mass_displacement = numpy.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
    return mass_displacement

def measure_quartile_intensity(indices:NDArray[numpy.int_], areas: NDArray[numpy.int_], fraction: float, limg: NDArray[Pixel], order):
    qindex = indices.astype(float) + areas * fraction
    qfraction = qindex - numpy.floor(qindex)
    qindex = qindex.astype(int)
    qmask = qindex < indices + areas - 1
    qi = qindex[qmask]
    qf = qfraction[qmask]
    _dest = (limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf)
    #
    # In some situations (e.g., only 3 points), there may
    # not be an upper bound.
    #
    qmask_no_upper = (~qmask) & (areas > 0)
    dest_no_upper = limg[order[qindex[qmask_no_upper]]]
    return qmask, _dest, qmask_no_upper, dest_no_upper

###############################################################################
# MeasureObjectOverlap
###############################################################################

def calculate_overlap_measurements(
    objects_GT_labelset: ObjectLabelSet,
    objects_ID_labelset: ObjectLabelSet,
    objects_GT_shape: Tuple[int, int],
    objects_ID_shape: Tuple[int, int],
    ) -> Tuple[
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        Union[numpy.float_, float],
        NDArray[numpy.float64],
        NDArray[numpy.float64],
        int,
        int,
    ]:
    objects_GT_ijv = convert_label_set_to_ijv(objects_GT_labelset)
    objects_ID_ijv = convert_label_set_to_ijv(objects_ID_labelset)
    gt_areas: NDArray[numpy.int_] = areas_from_ijv(objects_GT_ijv)
    id_areas: NDArray[numpy.int_] = areas_from_ijv(objects_ID_ijv)

    iGT, jGT, lGT = objects_GT_ijv.transpose()
    iID, jID, lID = objects_ID_ijv.transpose()

    ID_obj = 0 if len(lID) == 0 else max(lID)
    GT_obj = 0 if len(lGT) == 0 else max(lGT)

    xGT, yGT = objects_GT_shape
    xID, yID = objects_ID_shape
    GT_pixels = numpy.zeros((xGT, yGT))
    ID_pixels = numpy.zeros((xID, yID))
    total_pixels = xGT * yGT

    GT_pixels[iGT, jGT] = 1
    ID_pixels[iID, jID] = 1

    GT_tot_area = len(iGT)

    #
    # Intersect matrix [i, j] = number of pixels that are common among object number i from the ground truth and object number j from the test
    #
    intersect_matrix = get_intersect_matrix(iGT, jGT, lGT, iID, jID, lID, ID_obj, GT_obj)
    FN_area = gt_areas[numpy.newaxis, :] - intersect_matrix
    
    #
    # for each object in the ground truth, find the object in the test that has the highest overlap
    #
    dom_ID = get_dominating_ID_objects(ID_obj, lID, iID, jID, ID_pixels, intersect_matrix)
    
    for i in range(0, len(intersect_matrix.T)):
        if len(numpy.where(dom_ID == i)[0]) > 1:
            final_id = numpy.where(intersect_matrix.T[i] == max(intersect_matrix.T[i]))
            final_id = final_id[0][0]
            all_id = numpy.where(dom_ID == i)[0]
            nonfinal = [x for x in all_id if x != final_id]
            for (n) in nonfinal:  # these others cannot be candidates for the corr ID now
                intersect_matrix.T[i][n] = 0
        else:
            continue
        
    TP, FN, FP, TN = object_overlap_confusion_matrix(dom_ID, id_areas, intersect_matrix, FN_area, total_pixels)

    def nan_divide(numerator, denominator):
        if denominator == 0:
            return numpy.nan
        return numpy.float64(float(numerator) / float(denominator))
    recall = nan_divide(TP, GT_tot_area)
    precision = nan_divide(TP, (TP + FP))
    F_factor = nan_divide(2 * (precision * recall), (precision + recall))
    true_positive_rate = nan_divide(TP, (FN + TP))
    false_positive_rate = nan_divide(FP, (FP + TN))
    false_negative_rate = nan_divide(FN, (FN + TP))
    true_negative_rate = nan_divide(TN, (FP + TN))
    shape = numpy.maximum(
        numpy.maximum(numpy.array(objects_GT_shape), numpy.array(objects_ID_shape)),
        numpy.ones(2, int),
    )
    rand_index, adjusted_rand_index = compute_rand_index_ijv(
        objects_GT_ijv, objects_ID_ijv, shape
    )
    return (
    F_factor,
    precision,
    recall,
    true_positive_rate,
    false_positive_rate,
    true_negative_rate,
    false_negative_rate,
    rand_index,
    adjusted_rand_index,
    GT_pixels,
    ID_pixels,
    xGT,
    yGT
    )




def object_overlap_confusion_matrix(
        dom_ID: NDArray[numpy.int_],
        id_areas: NDArray[numpy.int_],
        intersect_matrix: NDArray[Union[numpy.int_, numpy.float_]],
        FN_area: NDArray[numpy.float_],
        total_pixels: int
        ) -> Tuple[
            int,
            int,
            int,
            int,
            ]:
    TP = 0
    FN = 0
    FP = 0
    for i in range(0, len(dom_ID)):
        d = dom_ID[i]
        if d == -1:
            tp = 0
            fn = id_areas[i]
            fp = 0
        else:
            fp = numpy.sum(intersect_matrix[i][0:d]) + numpy.sum(intersect_matrix[i][(d + 1) : :])
            tp = intersect_matrix[i][d]
            fn = FN_area[i][d]
        TP += tp
        FN += fn
        FP += fp

    TN = max(0, total_pixels - TP - FN - FP)
    return TP, FN, FP, TN


def get_dominating_ID_objects(
        ID_obj: int, 
        lID: NDArray[numpy.int_],
        iID: NDArray[numpy.int_],
        jID: NDArray[numpy.int_],
        ID_pixels: NDArray[numpy.float_],
        intersect_matrix: NDArray[Union[numpy.int_, numpy.float_]]
        ) -> NDArray[numpy.int_]:
    dom_ID = []

    for i in range(0, ID_obj):
        indices_jj = numpy.nonzero(lID == i)
        indices_jj = indices_jj[0]
        id_i = iID[indices_jj]
        id_j = jID[indices_jj]
        ID_pixels[id_i, id_j] = 1

    for i in intersect_matrix:  # loop through the GT objects first
        if len(i) == 0 or max(i) == 0:
            id = -1  # we missed the object; arbitrarily assign -1 index
        else:
            id = numpy.where(i == max(i))[0][0]  # what is the ID of the max pixels?
        dom_ID += [id]  # for ea GT object, which is the dominating ID?

    dom_ID = numpy.array(dom_ID)

    
    return dom_ID

def get_intersect_matrix(
        iGT: NDArray[numpy.int_],
        jGT: NDArray[numpy.int_],
        lGT: NDArray[numpy.int_],
        iID: NDArray[numpy.int_],
        jID: NDArray[numpy.int_],
        lID: NDArray[numpy.int_],
        ID_obj: int, 
        GT_obj: int
        ) -> NDArray[Union[numpy.int_, numpy.float_]]:
    if len(iGT) == 0 and len(iID) == 0:
        intersect_matrix = numpy.zeros((0, 0), int)
    else:
        #
        # Build a matrix with rows of i, j, label and a GT/ID flag
        #
        all_ijv = numpy.column_stack(
            (
                numpy.hstack((iGT, iID)),
                numpy.hstack((jGT, jID)),
                numpy.hstack((lGT, lID)),
                numpy.hstack((numpy.zeros(len(iGT)), numpy.ones(len(iID)))),
            )
        )
        #
        # Order it so that runs of the same i, j are consecutive
        #
        order = numpy.lexsort((all_ijv[:, -1], all_ijv[:, 0], all_ijv[:, 1]))
        all_ijv = all_ijv[order, :]
        # Mark the first at each i, j != previous i, j
        first = numpy.where(
            numpy.hstack(
                ([True], ~numpy.all(all_ijv[:-1, :2] == all_ijv[1:, :2], 1), [True])
            )
        )[0]
        # Count # at each i, j
        count = first[1:] - first[:-1]
        # First indexer - mapping from i,j to index in all_ijv
        all_ijv_map = centrosome.index.Indexes([count])
        # Bincount to get the # of ID pixels per i,j
        id_count = numpy.bincount(all_ijv_map.rev_idx, all_ijv[:, -1]).astype(int)
        gt_count = count - id_count
        # Now we can create an indexer that has NxM elements per i,j
        # where N is the number of GT pixels at that i,j and M is
        # the number of ID pixels. We can then use the indexer to pull
        # out the label values for each to populate a sparse array.
        #
        cross_map = centrosome.index.Indexes([id_count, gt_count])
        off_gt = all_ijv_map.fwd_idx[cross_map.rev_idx] + cross_map.idx[0]
        off_id = (
            all_ijv_map.fwd_idx[cross_map.rev_idx]
            + cross_map.idx[1]
            + id_count[cross_map.rev_idx]
        )
        intersect_matrix = scipy.sparse.coo_matrix(
            (numpy.ones(len(off_gt)), (all_ijv[off_id, 2], all_ijv[off_gt, 2])),
            shape=(ID_obj + 1, GT_obj + 1),
        ).toarray()[1:, 1:]
    return intersect_matrix


def compute_rand_index_ijv(
        gt_ijv: ObjectSegmentationIJV,
        test_ijv: ObjectSegmentationIJV,
        shape: Union[Tuple[int, int], NDArray[numpy.int_]]
    ) -> Tuple[numpy.float_, numpy.float_]:
    """Compute the Rand Index for an IJV matrix

    This is in part based on the Omega Index:
    Collins, "Omega: A General Formulation of the Rand Index of Cluster
    Recovery Suitable for Non-disjoint Solutions", Multivariate Behavioral
    Research, 1988, 23, 231-242

    The basic idea of the paper is that a pair should be judged to
    agree only if the number of clusters in which they appear together
    is the same.
    """
    #
    # The idea here is to assign a label to every pixel position based
    # on the set of labels given to that position by both the ground
    # truth and the test set. We then assess each pair of labels
    # as agreeing or disagreeing as to the number of matches.
    #
    # First, add the backgrounds to the IJV with a label of zero
    #
    gt_bkgd = numpy.ones(shape, bool)
    gt_bkgd[gt_ijv[:, 0], gt_ijv[:, 1]] = False
    test_bkgd = numpy.ones(shape, bool)
    test_bkgd[test_ijv[:, 0], test_ijv[:, 1]] = False
    gt_ijv = numpy.vstack(
        [
            gt_ijv,
            numpy.column_stack(
                [
                    numpy.argwhere(gt_bkgd),
                    numpy.zeros(numpy.sum(gt_bkgd), gt_bkgd.dtype),
                ]
            ),
        ]
    )
    test_ijv = numpy.vstack(
        [
            test_ijv,
            numpy.column_stack(
                [
                    numpy.argwhere(test_bkgd),
                    numpy.zeros(numpy.sum(test_bkgd), test_bkgd.dtype),
                ]
            ),
        ]
    )
    #
    # Create a unified structure for the pixels where a fourth column
    # tells you whether the pixels came from the ground-truth or test
    #
    u = numpy.vstack(
        [
            numpy.column_stack(
                [gt_ijv, numpy.zeros(gt_ijv.shape[0], gt_ijv.dtype)]
            ),
            numpy.column_stack(
                [test_ijv, numpy.ones(test_ijv.shape[0], test_ijv.dtype)]
            ),
        ]
    )
    #
    # Sort by coordinates, then by identity
    #
    order = numpy.lexsort([u[:, 2], u[:, 3], u[:, 0], u[:, 1]])
    u = u[order, :]
    # Get rid of any duplicate labellings (same point labeled twice with
    # same label.
    #
    first = numpy.hstack([[True], numpy.any(u[:-1, :] != u[1:, :], 1)])
    u = u[first, :]
    #
    # Create a 1-d indexer to point at each unique coordinate.
    #
    first_coord_idxs = numpy.hstack(
        [
            [0],
            numpy.argwhere(
                (u[:-1, 0] != u[1:, 0]) | (u[:-1, 1] != u[1:, 1])
            ).flatten()
            + 1,
            [u.shape[0]],
        ]
    )
    first_coord_counts = first_coord_idxs[1:] - first_coord_idxs[:-1]
    indexes = centrosome.index.Indexes([first_coord_counts])
    #
    # Count the number of labels at each point for both gt and test
    #
    count_test = numpy.bincount(indexes.rev_idx, u[:, 3]).astype(numpy.int64)
    count_gt = first_coord_counts - count_test
    #
    # For each # of labels, pull out the coordinates that have
    # that many labels. Count the number of similarly labeled coordinates
    # and record the count and labels for that group.
    #
    labels = []
    for i in range(1, numpy.max(count_test) + 1):
        for j in range(1, numpy.max(count_gt) + 1):
            match = (count_test[indexes.rev_idx] == i) & (
                count_gt[indexes.rev_idx] == j
            )
            if not numpy.any(match):
                continue
            #
            # Arrange into an array where the rows are coordinates
            # and the columns are the labels for that coordinate
            #
            lm = u[match, 2].reshape(numpy.sum(match) // (i + j), i + j)
            #
            # Sort by label.
            #
            order = numpy.lexsort(lm.transpose())
            lm = lm[order, :]
            #
            # Find indices of unique and # of each
            #
            lm_first = numpy.hstack(
                [
                    [0],
                    numpy.argwhere(numpy.any(lm[:-1, :] != lm[1:, :], 1)).flatten()
                    + 1,
                    [lm.shape[0]],
                ]
            )
            lm_count = lm_first[1:] - lm_first[:-1]
            for idx, count in zip(lm_first[:-1], lm_count):
                labels.append((count, lm[idx, :j], lm[idx, j:]))
    #
    # We now have our sets partitioned. Do each against each to get
    # the number of true positive and negative pairs.
    #
    max_t_labels = reduce(max, [len(t) for c, t, g in labels], 0)
    max_g_labels = reduce(max, [len(g) for c, t, g in labels], 0)
    #
    # tbl is the contingency table from Table 4 of the Collins paper
    # It's a table of the number of pairs which fall into M sets
    # in the ground truth case and N in the test case.
    #
    tbl = numpy.zeros(((max_t_labels + 1), (max_g_labels + 1)))
    for i, (c1, tobject_numbers1, gobject_numbers1) in enumerate(labels):
        for j, (c2, tobject_numbers2, gobject_numbers2) in enumerate(labels[i:]):
            nhits_test = numpy.sum(
                tobject_numbers1[:, numpy.newaxis]
                == tobject_numbers2[numpy.newaxis, :]
            )
            nhits_gt = numpy.sum(
                gobject_numbers1[:, numpy.newaxis]
                == gobject_numbers2[numpy.newaxis, :]
            )
            if j == 0:
                N = c1 * (c1 - 1) / 2
            else:
                N = c1 * c2
            tbl[nhits_test, nhits_gt] += N

    N = numpy.sum(tbl)
    #
    # Equation 13 from the paper
    #
    min_JK = min(max_t_labels, max_g_labels) + 1
    rand_index = numpy.sum(tbl[:min_JK, :min_JK] * numpy.identity(min_JK)) / N
    #
    # Equation 15 from the paper, the expected index
    #
    e_omega = (
        numpy.sum(
            numpy.sum(tbl[:min_JK, :min_JK], 0)
            * numpy.sum(tbl[:min_JK, :min_JK], 1)
        )
        / N ** 2
    )
    #
    # Equation 16 is the adjusted index
    #
    adjusted_rand_index = (rand_index - e_omega) / (1 - e_omega)
    return rand_index, adjusted_rand_index


def compute_earth_movers_distance_objects(
        src_objects_label_set: ObjectLabelSet,
        dest_objects_label_set: ObjectLabelSet,
        decimation_method: Union[ObjectDecimationMethod, mio.DM]=ObjectDecimationMethod.KMEANS,
        max_points: int=250,
        max_distance: int=250,
        penalize_missing: bool=False,
        ) -> np.float_:
    src_objects_shape = src_objects_label_set[0][0].shape
    dest_objects_shape = dest_objects_label_set[0][0].shape

    src_obj_ijv = convert_label_set_to_ijv(src_objects_label_set, validate=True)
    src_count = count_from_ijv(src_obj_ijv, validate=False)
    src_areas = areas_from_ijv(src_obj_ijv, validate=False)

    dest_obj_ijv = convert_label_set_to_ijv(dest_objects_label_set, validate=True)
    dest_count = count_from_ijv(dest_obj_ijv, validate=False)
    dest_areas = areas_from_ijv(dest_obj_ijv, validate=False)

    """Compute the earthmovers distance between two sets of objects

    src_objects - move pixels from these objects

    dest_objects - move pixels to these objects

    returns the earth mover's distance
    """
    #
    # if either foreground set is empty, the emd is the penalty.
    #

    for left_count, right_areas in (
        (src_count, dest_areas),
        (dest_count, src_areas),
        ):
        if left_count == 0:
            if penalize_missing:
                return np.sum(right_areas) * max_distance
            else:
                return np.float64(0)
    
    if decimation_method in (ObjectDecimationMethod.KMEANS, mio.DM.KMEANS):
        isrc, jsrc = get_kmeans_points(src_obj_ijv, dest_obj_ijv, max_points)
        idest, jdest = isrc, jsrc

    elif decimation_method in (ObjectDecimationMethod.SKELETON, mio.DM.SKELETON):
        assert src_objects_label_set is not None, "src_objects_labels must be provided for Skeleton decimation method"
        assert dest_objects_label_set is not None, "dest_objects_labels must be provided for Skeleton decimation method"
        assert src_objects_shape is not None, "src_objects_shape must be provided for Skeleton decimation method"
        assert dest_objects_shape is not None, "dest_objects_shape must be provided for Skeleton decimation method"
        isrc, jsrc = get_skeleton_points(src_objects_label_set, src_objects_shape, max_points)
        idest, jdest = get_skeleton_points(dest_objects_label_set, dest_objects_shape, max_points)
    else:
        raise TypeError("Unknown type for decimation method: %s" % decimation_method)
    src_labels_mask = get_labels_mask(src_objects_label_set, src_objects_shape)
    dest_labels_mask = get_labels_mask(dest_objects_label_set, dest_objects_shape)
            
    src_weights = get_weights(isrc, jsrc, src_labels_mask)
    dest_weights = get_weights(idest, jdest, dest_labels_mask)

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


###############################################################################
# MeasureImageQuality
###############################################################################

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_focus_score_for_scale_group(
        scale_groups: Sequence[int],
        pixel_data: ImageGrayscale,
        dimensions: int,
        image_mask: Optional[ImageGrayscaleMask]=None,
) -> Tuple[
    NDArray[numpy.float_],
    int,
    Sequence[numpy.float_]
]:
    if image_mask is not None:
            masked_pixel_data = pixel_data[image_mask]
    else:
        masked_pixel_data = pixel_data
    shape = pixel_data.shape
    local_focus_score = []
    focus_score = 0
    for scale in scale_groups:
        focus_score = 0
        if len(masked_pixel_data):
            mean_image_value = numpy.mean(masked_pixel_data)
            squared_normalized_image = (masked_pixel_data - mean_image_value) ** 2
            if mean_image_value > 0:
                focus_score = numpy.sum(squared_normalized_image) / (
                    numpy.product(masked_pixel_data.shape) * mean_image_value
                )
        #
        # Create a labels matrix that grids the image to the dimensions
        # of the window size
        #
        if dimensions == 2:
            i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
            m, n = (numpy.array(shape) + scale - 1) // scale
            i = (i * float(m) / float(shape[0])).astype(int)
            j = (j * float(n) / float(shape[1])).astype(int)
            grid = i * n + j + 1
            grid_range = numpy.arange(0, m * n + 1, dtype=numpy.int32)
        elif dimensions == 3:
            k, i, j = numpy.mgrid[
                0 : shape[0], 0 : shape[1], 0 : shape[2]
            ].astype(float)
            o, m, n = (numpy.array(shape) + scale - 1) // scale
            k = (k * float(o) / float(shape[0])).astype(int)
            i = (i * float(m) / float(shape[1])).astype(int)
            j = (j * float(n) / float(shape[2])).astype(int)
            grid = k * o + i * n + j + 1  # hmm
            grid_range = numpy.arange(0, m * n * o + 1, dtype=numpy.int32)
        else:
            raise ValueError("Image dimensions must be 2 or 3")
        if image_mask is not None:
            grid[numpy.logical_not(image_mask)] = 0
        
        #
        # Do the math per label
        #
        local_means = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(pixel_data, grid, grid_range)
        )
        local_squared_normalized_image = (
            pixel_data - local_means[grid]
        ) ** 2
        #
        # Compute the sum of local_squared_normalized_image values for each
        # grid for means > 0. Exclude grid label = 0 because that's masked
        #
        grid_mask = (local_means != 0) & ~numpy.isnan(local_means)
        nz_grid_range = grid_range[grid_mask]
        if len(nz_grid_range) and nz_grid_range[0] == 0:
            nz_grid_range = nz_grid_range[1:]
            local_means = local_means[1:]
            grid_mask = grid_mask[1:]
        local_focus_score += [
            0
        ]  # assume the worst - that we can't calculate it
        if len(nz_grid_range):
            sums = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(
                    local_squared_normalized_image, grid, nz_grid_range
                )
            )
            pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(numpy.ones(shape), grid, nz_grid_range)
            )
            local_norm_var = sums / (pixel_counts * local_means[grid_mask])
            local_norm_median = numpy.median(local_norm_var)
            if numpy.isfinite(local_norm_median) and local_norm_median > 0:
                local_focus_score[-1] = (
                    numpy.var(local_norm_var) / local_norm_median
                )
    return focus_score, scale, local_focus_score

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_correlation_for_scale_group(
        pixel_data: ImageGrayscale, 
        scale_groups: Sequence[int], 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Dict[int, numpy.float_]:
    # Compute Haralick's correlation texture for the given scales
    image_labels = numpy.ones(pixel_data.shape, int)
    if image_mask is not None:
        image_labels[~image_mask] = 0
    scale_measurements = {}
    for scale in scale_groups:
        scale_measurements[scale] = get_correlation_for_scale(pixel_data, image_labels, scale)
    return scale_measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_correlation_for_scale(
        pixel_data: ImageGrayscale, 
        image_labels: NDArray[numpy.int_], 
        scale: int
    ) -> float:
    # Compute Haralick's correlation texture for the given scale
    value = centrosome.haralick.Haralick(pixel_data, image_labels, 0, scale).H3()

    if len(value) != 1 or not numpy.isfinite(value[0]):
        value = 0.0
    else:
        value = float(value)
    return value


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_saturation_value(
        pixel_data: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Tuple[
        float,
        float
    ]:
    if image_mask is not None:
        pixel_data = pixel_data[image_mask]
    pixel_count = numpy.product(pixel_data.shape)
    if pixel_count == 0:
        percent_maximal = 0
        percent_minimal = 0
    else:
        number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))
        number_pixels_minimal = numpy.sum(pixel_data == numpy.min(pixel_data))
        percent_maximal = (
            100.0 * float(number_pixels_maximal) / float(pixel_count)
        )
        percent_minimal = (
            100.0 * float(number_pixels_minimal) / float(pixel_count)
        )
    return percent_minimal, percent_maximal


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_intensity_measurement_values(
        pixels: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Tuple[
        numpy.int_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_
    ]:
    if image_mask is not None:
        pixels = pixels[image_mask]
    
    pixel_count = numpy.product(pixels.shape)
    if pixel_count == 0:
        pixel_sum = 0
        pixel_mean = 0
        pixel_std = 0
        pixel_mad = 0
        pixel_median = 0
        pixel_min = 0
        pixel_max = 0
    else:
        pixel_sum = numpy.sum(pixels)
        pixel_mean = pixel_sum / float(pixel_count)
        pixel_std = numpy.std(pixels)
        pixel_median = numpy.median(pixels)
        pixel_mad = numpy.median(numpy.abs(pixels - pixel_median))
        pixel_min = numpy.min(pixels)
        pixel_max = numpy.max(pixels)
    return pixel_count, pixel_sum, pixel_mean, pixel_std, pixel_mad, pixel_median, pixel_min, pixel_max


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_power_spectrum_measurement_values(
        pixel_data: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None, 
        dimensions: int = 2
    ) -> float:
    if dimensions == 3:
        raise NotImplementedError("Power spectrum calculation for volumes is not implemented")

    if image_mask is not None:
        pixel_data = numpy.array(pixel_data)  # make a copy
        masked_pixels = pixel_data[image_mask]
        pixel_count = numpy.product(masked_pixels.shape)
        if pixel_count > 0:
            pixel_data[~image_mask] = numpy.mean(masked_pixels)
        else:
            pixel_data[~image_mask] = 0

    radii, magnitude, power = centrosome.radial_power_spectrum.rps(pixel_data)
    if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = numpy.isfinite(numpy.log(power))
            powerslope = scipy.linalg.basic.lstsq(
                numpy.hstack(
                    (
                        numpy.log(radii)[idx][:, numpy.newaxis],
                        numpy.ones(radii.shape)[idx][:, numpy.newaxis],
                    )
                ),
                numpy.log(power)[idx][:, numpy.newaxis],
            )[0][0]
        else:
            powerslope = 0
    else:
        powerslope = 0
    return powerslope


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_threshold_for_threshold_group(
        image_pixel_data: ImageGrayscale = numpy.ones((100, 100)),
        mask: Optional[ImageGrayscaleMask] = None,
        algorithm: str = centrosome.threshold.TM_OTSU,
        object_fraction: float = 0.1,
        two_class_otsu: bool = True,
        use_weighted_variance: bool = True,
        assign_middle_to_foreground: bool = True,
) -> float:
    if mask is not None:
        _, global_threshold = centrosome.threshold.get_threshold(
            algorithm,
            centrosome.threshold.TM_GLOBAL,
            image_pixel_data,
            mask = mask,
            object_fraction=object_fraction,
            two_class_otsu=two_class_otsu,
            use_weighted_variance=use_weighted_variance,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
    else:
        _, global_threshold = centrosome.threshold.get_threshold(
            algorithm,
            centrosome.threshold.TM_GLOBAL,
            image_pixel_data,
            object_fraction=object_fraction,
            two_class_otsu=two_class_otsu,
            use_weighted_variance=use_weighted_variance,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
    return global_threshold
###############################################################################
# Measure Image Skeleton
###############################################################################

def neighbors(image: ImageBinary):
    """

    Counts the neighbor pixels for each pixel of an image:

            x = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]

            _neighbors(x)

            [
                [0, 3, 0],
                [3, 4, 3],
                [0, 3, 0]
            ]

    :type image: numpy.ndarray

    :param image: A two-or-three dimensional image

    :return: neighbor pixels for each pixel of an image

    """
    padding = numpy.pad(image, 1, "constant")
    mask = padding > 0

    padding = padding.astype(float)

    if image.ndim == 2:
        response = 3 ** 2 * scipy.ndimage.uniform_filter(padding) - 1
        labels = (response * mask)[1:-1, 1:-1]

        return labels.astype(numpy.uint16)
    elif image.ndim == 3:
        response = 3 ** 3 * scipy.ndimage.uniform_filter(padding) - 1
        labels = (response * mask)[1:-1, 1:-1, 1:-1]

        return labels.astype(numpy.uint16)
    else:
        raise ValueError("Only 2D and 3D images are supported")

def branches(image: ImageBinary):
    return neighbors(image) > 2
def endpoints(image: ImageBinary):
    return neighbors(image) == 1

###############################################################################
# MeasureObjectSkeleton
###############################################################################

def get_ring_with_trunks(
        skeleton: Image2DBinary, 
        labels: ObjectSegmentation
        ) -> Tuple[
            ObjectSegmentation,
            Image2DBinary,
            Image2DBinary
        ]:
    #
    # The following code makes a ring around the seed objects with
    # the skeleton trunks sticking out of it.
    #
    # Create a new skeleton with holes at the seed objects
    # First combine the seed objects with the skeleton so
    # that the skeleton trunks come out of the seed objects.
    #
    # Erode the labels once so that all of the trunk branchpoints
    # will be within the labels
    #
    #
    # Dilate the objects, then subtract them to make a ring
    #
    my_disk = centrosome.cpmorphology.strel_disk(1.5).astype(int)
    dilated_labels = grey_dilation(labels, footprint=my_disk)
    seed_mask = dilated_labels > 0
    combined_skel = skeleton | seed_mask

    closed_labels = grey_erosion(dilated_labels, footprint=my_disk)
    seed_center = closed_labels > 0
    combined_skel = combined_skel & (~seed_center)
    return dilated_labels, combined_skel, seed_center


def calculate_object_skeleton(
        skeleton: Image2DBinary, 
        cropped_labels: ObjectSegmentation, 
        labels_count: numpy.integer[Any],
        fill_small_holes: bool, 
        max_hole_size: Optional[int],
        wants_objskeleton_graph: bool,
        intensity_image_pixel_data: Optional[Image2DGrayscale],
        wants_branchpoint_image: bool
        ) -> Tuple[
            NDArray[numpy.float_], # trunk counts
            NDArray[numpy.float_], # branch counts
            NDArray[numpy.float_], # end counts
            NDArray[numpy.float_], # total distance
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Image2DColor],
        ]:
    #
    # The following code makes a ring around the seed objects with
    # the skeleton trunks sticking out of it.
    #
    dilated_labels, combined_skel, seed_center = get_ring_with_trunks(skeleton, cropped_labels)
    
    #
    # Fill in single holes (but not a one-pixel hole made by
    # a one-pixel image)
    #
    if fill_small_holes:
        assert max_hole_size is not None, "max_hole_size must be provided for filling small holes"
        def size_fn(area, is_object):
            return (~is_object) and (area <= max_hole_size)

        combined_skel = centrosome.cpmorphology.fill_labeled_holes(
            combined_skel, ~seed_center, size_fn
        )
    #
    # Reskeletonize to make true branchpoints at the ring boundaries
    #
    combined_skel = centrosome.cpmorphology.skeletonize(combined_skel)
    #
    # The skeleton outside of the labels
    #
    outside_skel = combined_skel & (dilated_labels == 0)
    #
    # Associate all skeleton points with seed objects
    #
    dlabels, distance_map = centrosome.propagate.propagate(
        numpy.zeros(cropped_labels.shape), dilated_labels, combined_skel, 1
    )
    #
    # Get rid of any branchpoints not connected to seeds
    #
    combined_skel[dlabels == 0] = False
    #
    # Find the branchpoints
    #
    branch_points = centrosome.cpmorphology.branchpoints(combined_skel)

    #
    # Odd case: when four branches meet like this, branchpoints are not
    # assigned because they are arbitrary. So assign them.
    #
    # .  .
    #  B.
    #  .B
    # .  .
    #
    odd_case = (
        combined_skel[:-1, :-1]
        & combined_skel[1:, :-1]
        & combined_skel[:-1, 1:]
        & combined_skel[1, 1]
    )
    branch_points[:-1, :-1][odd_case] = True
    branch_points[1:, 1:][odd_case] = True
    #
    # Find the branching counts for the trunks (# of extra branches
    # emanating from a point other than the line it might be on).
    #
    branching_counts = centrosome.cpmorphology.branchings(combined_skel)
    branching_counts = numpy.array([0, 0, 0, 1, 2])[branching_counts]
    #
    # Only take branches within 1 of the outside skeleton
    #
    dilated_skel = scipy.ndimage.binary_dilation(
        outside_skel, centrosome.cpmorphology.eight_connect
    )
    branching_counts[~dilated_skel] = 0
    #
    # Find the endpoints
    #
    end_points = centrosome.cpmorphology.endpoints(combined_skel)
    #
    # We use two ranges for classification here:
    # * anything within one pixel of the dilated image is a trunk
    # * anything outside of that range is a branch
    #
    nearby_labels = dlabels.copy()
    nearby_labels[distance_map > 1.5] = 0

    outside_labels = dlabels.copy()
    outside_labels[nearby_labels > 0] = 0
    #
    # The trunks are the branchpoints that lie within one pixel of
    # the dilated image.
    #
    label_range = numpy.arange(labels_count, dtype=numpy.int32) + 1
    if labels_count > 0:
        trunk_counts = fix(
            scipy.ndimage.sum(branching_counts, nearby_labels, label_range)
        ).astype(int)
    else:
        trunk_counts = numpy.zeros((0,), int)
    #
    # The branches are the branchpoints that lie outside the seed objects
    #
    if labels_count > 0:
        branch_counts = fix(
            scipy.ndimage.sum(branch_points, outside_labels, label_range)
        )
    else:
        branch_counts = numpy.zeros((0,), int)
    #
    # Save the endpoints
    #
    if labels_count > 0:
        end_counts = fix(scipy.ndimage.sum(end_points, outside_labels, label_range))
    else:
        end_counts = numpy.zeros((0,), int)
    #
    # Calculate the distances
    #
    total_distance = centrosome.cpmorphology.skeleton_length(
        dlabels * outside_skel, label_range
    ).astype(numpy.float64)
    edge_graph = None
    vertex_graph = None
    
    branchpoint_image = None
    if wants_objskeleton_graph or wants_branchpoint_image:
        trunk_mask = (branching_counts > 0) & (nearby_labels != 0)
        
        if wants_objskeleton_graph:
            assert intensity_image_pixel_data is not None
            edge_graph, vertex_graph = make_objskeleton_graph(
                combined_skel,
                dlabels,
                trunk_mask,
                branch_points & ~trunk_mask,
                end_points,
                intensity_image_pixel_data,
            )
        if wants_branchpoint_image:
            branchpoint_image = numpy.zeros((skeleton.shape[0], skeleton.shape[1], 3))
            branch_mask = branch_points & (outside_labels != 0)
            end_mask = end_points & (outside_labels != 0)
            branchpoint_image[outside_skel, :] = 1
            branchpoint_image[trunk_mask | branch_mask | end_mask, :] = 0
            branchpoint_image[trunk_mask, 0] = 1
            branchpoint_image[branch_mask, 1] = 1
            branchpoint_image[end_mask, 2] = 1
            branchpoint_image[dilated_labels != 0, :] *= 0.875
            branchpoint_image[dilated_labels != 0, :] += 0.1

    return (
        trunk_counts, 
        branch_counts,
        end_counts,
        total_distance,
        edge_graph, 
        vertex_graph,
        branchpoint_image
    )

def make_objskeleton_graph(
    skeleton: Image2DBinary, 
    skeleton_labels: ObjectSegmentation, 
    trunks: Image2DBinary, 
    branchpoints: Image2DBinary, 
    endpoints: Image2DBinary, 
    image: Image2DGrayscale
    ) -> Tuple[
        Dict[str, NDArray[Union[numpy.float_, numpy.int_]]],
        Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]
    ]:
    """Make a table that captures the graph relationship of the skeleton

    skeleton - binary skeleton image + outline of seed objects
    skeleton_labels - labels matrix of skeleton
    trunks - binary image with trunk points as 1
    branchpoints - binary image with branchpoints as 1
    endpoints - binary image with endpoints as 1
    image - image for intensity measurement

    returns two tables.
    Table 1: edge table
    The edge table is a numpy record array with the following named
    columns in the following order:
    v1: index into vertex table of first vertex of edge
    v2: index into vertex table of second vertex of edge
    length: # of intermediate pixels + 2 (for two vertices)
    total_intensity: sum of intensities along the edge

    Table 2: vertex table
    The vertex table is a numpy record array:
    i: I coordinate of the vertex
    j: J coordinate of the vertex
    label: the vertex's label
    kind: kind of vertex = "T" for trunk, "B" for branchpoint or "E" for endpoint.
    """
    i, j = numpy.mgrid[0 : skeleton.shape[0], 0 : skeleton.shape[1]]
    #
    # Give each point of interest a unique number
    #
    points_of_interest = trunks | branchpoints | endpoints
    number_of_points = numpy.sum(points_of_interest)
    #
    # Make up the vertex table
    #
    tbe = numpy.zeros(points_of_interest.shape, "|S1")
    tbe[trunks] = "T"
    tbe[branchpoints] = "B"
    tbe[endpoints] = "E"
    i_idx = i[points_of_interest]
    j_idx = j[points_of_interest]
    poe_labels = skeleton_labels[points_of_interest]
    tbe = tbe[points_of_interest]
    vertex_table = {
        VF_I: i_idx,
        VF_J: j_idx,
        VF_LABELS: poe_labels,
        VF_KIND: tbe,
    }
    #
    # First, break the skeleton by removing the branchpoints, endpoints
    # and trunks
    #
    broken_skeleton = skeleton & (~points_of_interest)
    #
    # Label the broken skeleton: this labels each edge differently
    #
    edge_labels, nlabels = centrosome.cpmorphology.label_skeleton(skeleton)
    #
    # Reindex after removing the points of interest
    #
    edge_labels[points_of_interest] = 0
    if nlabels > 0:
        indexer = numpy.arange(nlabels + 1)
        unique_labels = numpy.sort(numpy.unique(edge_labels))
        nlabels = len(unique_labels) - 1
        indexer[unique_labels] = numpy.arange(len(unique_labels))
        edge_labels = indexer[edge_labels]
        #
        # find magnitudes and lengths for all edges
        #
        magnitudes = fix(
            scipy.ndimage.sum(
                image, edge_labels, numpy.arange(1, nlabels + 1, dtype=numpy.int32)
            )
        )
        lengths = fix(
            scipy.ndimage.sum(
                numpy.ones(edge_labels.shape),
                edge_labels,
                numpy.arange(1, nlabels + 1, dtype=numpy.int32),
            )
        ).astype(int)
    else:
        magnitudes = numpy.zeros(0)
        lengths = numpy.zeros(0, int)
    #
    # combine the edge labels and indexes of points of interest with padding
    #
    edge_mask = edge_labels != 0
    all_labels = numpy.zeros(numpy.array(edge_labels.shape) + 2, int)
    all_labels[1:-1, 1:-1][edge_mask] = edge_labels[edge_mask] + number_of_points
    all_labels[i_idx + 1, j_idx + 1] = numpy.arange(1, number_of_points + 1)
    #
    # Collect all 8 neighbors for each point of interest
    #
    p1 = numpy.zeros(0, int)
    p2 = numpy.zeros(0, int)
    for i_off, j_off in (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ):
        p1 = numpy.hstack((p1, numpy.arange(1, number_of_points + 1)))
        p2 = numpy.hstack((p2, all_labels[i_idx + i_off, j_idx + j_off]))
    #
    # Get rid of zeros which are background
    #
    p1 = p1[p2 != 0]
    p2 = p2[p2 != 0]
    #
    # Find point_of_interest -> point_of_interest connections.
    #
    p1_poi = p1[(p2 <= number_of_points) & (p1 < p2)]
    p2_poi = p2[(p2 <= number_of_points) & (p1 < p2)]
    #
    # Make sure matches are labeled the same
    #
    same_labels = (
        skeleton_labels[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
        == skeleton_labels[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
    )
    p1_poi = p1_poi[same_labels]
    p2_poi = p2_poi[same_labels]
    #
    # Find point_of_interest -> edge
    #
    p1_edge = p1[p2 > number_of_points]
    edge = p2[p2 > number_of_points]
    #
    # Now, each value that p2_edge takes forms a group and all
    # p1_edge whose p2_edge are connected together by the edge.
    # Possibly they touch each other without the edge, but we will
    # take the minimum distance connecting each pair to throw out
    # the edge.
    #
    edge, p1_edge, p2_edge = centrosome.cpmorphology.pairwise_permutations(
        edge, p1_edge
    )
    indexer = edge - number_of_points - 1
    lengths = lengths[indexer]
    magnitudes = magnitudes[indexer]
    #
    # OK, now we make the edge table. First poi<->poi. Length = 2,
    # magnitude = magnitude at each point
    #
    poi_length = numpy.ones(len(p1_poi)) * 2
    poi_magnitude = (
        image[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
        + image[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
    )
    #
    # Now the edges...
    #
    poi_edge_length = lengths + 2
    poi_edge_magnitude = (
        image[i_idx[p1_edge - 1], j_idx[p1_edge - 1]]
        + image[i_idx[p2_edge - 1], j_idx[p2_edge - 1]]
        + magnitudes
    )
    #
    # Put together the columns
    #
    v1 = numpy.hstack((p1_poi, p1_edge))
    v2 = numpy.hstack((p2_poi, p2_edge))
    lengths = numpy.hstack((poi_length, poi_edge_length))
    magnitudes = numpy.hstack((poi_magnitude, poi_edge_magnitude))
    #
    # Sort by p1, p2 and length in order to pick the shortest length
    #
    indexer = numpy.lexsort((lengths, v1, v2))
    v1 = v1[indexer]
    v2 = v2[indexer]
    lengths = lengths[indexer]
    magnitudes = magnitudes[indexer]
    if len(v1) > 0:
        to_keep = numpy.hstack(([True], (v1[1:] != v1[:-1]) | (v2[1:] != v2[:-1])))
        v1 = v1[to_keep]
        v2 = v2[to_keep]
        lengths = lengths[to_keep]
        magnitudes = magnitudes[to_keep]
    #
    # Put it all together into a table
    #
    edge_table = {
        EF_V1: v1,
        EF_V2: v2,
        EF_LENGTH: lengths,
        EF_TOTAL_INTENSITY: magnitudes,
    }
    return edge_table, vertex_table


################################################################################
# MeasureObjectNeighbors
################################################################################

def get_distance_and_labels(
        labels: ObjectSegmentation, 
        neighbor_labels: ObjectSegmentation, 
        neighbors_are_objects: bool,
        distance_method: NeighborsDistanceMethod,
        distance: int,
        dimensions: int
    ) -> Tuple[
        int, 
        Union[str, NeighborsDistanceMethod, NeighborsMeasurementScale],
        ObjectSegmentation,
        Optional[ObjectSegmentation],
        ObjectSegmentation

    ]:
    expanded_labels = None
    if distance_method == NeighborsDistanceMethod.EXPAND:
        # Find the i,j coordinates of the nearest foreground point
        # to every background point
        if dimensions == 2:
            i, j = scipy.ndimage.distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            # Assign each background pixel to the label of its nearest
            # foreground pixel. Assign label to label for foreground.
            labels = labels[i, j]
        else:
            k, i, j = scipy.ndimage.distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            labels = labels[k, i, j]
        expanded_labels = labels  # for display
        distance = 1  # dilate once to make touching edges overlap
        scale = NeighborsMeasurementScale.EXPANDED
        if neighbors_are_objects:
            neighbor_labels = labels.copy()
    elif distance_method == NeighborsDistanceMethod.WITHIN:
        distance = distance
        scale = str(distance)
    elif distance_method == NeighborsDistanceMethod.ADJACENT:
        distance = 1
        scale = NeighborsMeasurementScale.ADJACENT
    else:
        raise ValueError("Unknown distance method: %s" % distance_method)
    return distance, scale, labels, expanded_labels, neighbor_labels

def get_structuring_elements(
        distance: int, 
        dimensions: int
    ) -> Tuple[
        NDArray[numpy.float_],
        NDArray[numpy.float_]
    ]:
    # Make the structuring element for dilation
    if dimensions == 2:
        strel = strel_disk(distance)
    else:
        strel = skimage.morphology.ball(distance)
    #
    # A little bigger one to enter into the border with a structure
    # that mimics the one used to create the outline
    #
    if dimensions == 2:
        strel_touching = strel_disk(distance + 0.5)
    else:
        strel_touching = skimage.morphology.ball(distance + 0.5)

    return strel, strel_touching

def get_mins_and_maxs(
        idx: NDArray[numpy.int_], 
        labels: ObjectSegmentation, 
        object_indexes: NDArray[numpy.int_], 
        distance: int, 
        max_limit: int
    ) -> Tuple[
        NDArray[numpy.int_],
        NDArray[numpy.int_]
    ]:
    minimums_i, maximums_i, _, _ = scipy.ndimage.extrema(idx, labels, object_indexes)
    minimums_i = numpy.maximum(fix(minimums_i) - distance, 0).astype(int)
    maximums_i = numpy.minimum(fix(maximums_i) + distance + 1, max_limit).astype(int)
    return minimums_i, maximums_i

def get_extents(
        labels: ObjectSegmentation, 
        object_indexes: NDArray[numpy.int_], 
        distance: int, 
        dimensions: int
    ) -> Sequence[
        Tuple[
            Optional[NDArray[numpy.int_]], 
            Optional[NDArray[numpy.int_]], 
            ]
        ]:
    #
    # Get the extents for each object and calculate the patch
    # that excises the part of the image that is "distance"
    # away
    minimums_and_maximums: List[Tuple[Optional[NDArray[numpy.int_]], Optional[NDArray[numpy.int_]]]] = [
        (None, None),
        (None, None),
        (None, None)
    ]
    if dimensions == 2:
        i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
        minimums_and_maximums[0] = get_mins_and_maxs(i, labels, object_indexes, distance, labels.shape[0])
        minimums_and_maximums[1] = get_mins_and_maxs(j, labels, object_indexes, distance, labels.shape[1])
    else:
        k, i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1], 0 : labels.shape[2]]
        minimums_and_maximums[2] = get_mins_and_maxs(k, labels, object_indexes, distance, labels.shape[2])
        minimums_and_maximums[0] = get_mins_and_maxs(i, labels, object_indexes, distance, labels.shape[0])
        minimums_and_maximums[1] = get_mins_and_maxs(j ,labels, object_indexes, distance, labels.shape[1])

    return minimums_and_maximums

def get_patches_from_extents(
        ijk_extents: Sequence[Tuple[Optional[NDArray[numpy.int_]], Optional[NDArray[numpy.int_]]]],
        labels: NDArray[ObjectLabel], 
        neighbor_labels: NDArray[ObjectLabel], 
        index: int, # this is actual labels minus one
        dimensions: int,
    ) -> Tuple[
        NDArray[ObjectLabel],
        NDArray[ObjectLabel]
    ]:
    (
            (minimums_i, maximums_i), 
            (minimums_j, maximums_j), 
            (minimums_k, maximums_k),
    ) = ijk_extents
    assert minimums_i is not None, "Unexpected error: minimums_i extent value is None"
    assert maximums_i is not None, "Unexpected error: maximums_i extent value is None"
    assert minimums_j is not None, "Unexpected error: minimums_j extent value is None"
    assert maximums_j is not None, "Unexpected error: maximums_j extent value is None"

    if dimensions == 2:

        patch = labels[
            minimums_i[index] : maximums_i[index],
            minimums_j[index] : maximums_j[index],
            ]
        npatch = neighbor_labels[
            minimums_i[index] : maximums_i[index],
            minimums_j[index] : maximums_j[index],
            ]
    else:
        assert minimums_k is not None, "Unexpected error: minimums_k extent value is None"
        assert maximums_k is not None, "Unexpected error: maximums_k extent value is None"

        patch = labels[
            minimums_k[index] : maximums_k[index],
            minimums_i[index] : maximums_i[index],
            minimums_j[index] : maximums_j[index],
            ]
        npatch = neighbor_labels[
            minimums_k[index] : maximums_k[index],
            minimums_i[index] : maximums_i[index],
            minimums_j[index] : maximums_j[index],
            ]
    return patch, npatch

def get_outline_patch(
        ijk_extents: Sequence[Tuple[Optional[NDArray[numpy.int_]], Optional[NDArray[numpy.int_]]]], 
        perimeter_outlines: ObjectSegmentation, 
        object_number, 
        index, 
        dimensions
    ):
    (
            (minimums_i, maximums_i), 
            (minimums_j, maximums_j), 
            (minimums_k, maximums_k),
    ) = ijk_extents
    assert minimums_i is not None, "Unexpected error: minimums_i extent value is None"
    assert maximums_i is not None, "Unexpected error: maximums_i extent value is None"
    assert minimums_j is not None, "Unexpected error: minimums_j extent value is None"
    assert maximums_j is not None, "Unexpected error: maximums_j extent value is None"  
    if dimensions == 2:
        outline_patch = (
            perimeter_outlines[
                minimums_i[index] : maximums_i[index],
                minimums_j[index] : maximums_j[index],
            ]
            == object_number
        )
    else:
        assert minimums_k is not None, "Unexpected error: minimums_k extent value is None"
        assert maximums_k is not None, "Unexpected error: maximums_k extent value is None"
        outline_patch = (
            perimeter_outlines[
                minimums_k[index] : maximums_k[index],
                minimums_i[index] : maximums_i[index],
                minimums_j[index] : maximums_j[index],
            ]
            == object_number
        )
    return outline_patch

def renumber_labels(
        _objects: Sequence[NDArray[ObjectLabel]], 
        object_numbers: NDArray[ObjectLabel]
    ) -> NDArray[ObjectLabel]:
    #
    # Renumbers labels to be contiguous and start at 1
    #
    objects = numpy.hstack(_objects)
    reverse_numbers = numpy.zeros(
        max(numpy.max(object_numbers), numpy.max(objects)) + 1, int
    )
    reverse_numbers[object_numbers] = (
        numpy.arange(len(object_numbers)) + 1
    )
    objects = reverse_numbers[objects]
    return objects

def get_first_and_second_objects(
        first_objects: Sequence[NDArray[numpy.int_]], 
        second_objects: Sequence[NDArray[numpy.int_]], 
        object_numbers: NDArray[ObjectLabel], 
        neighbor_numbers: NDArray[ObjectLabel]
    ) -> Tuple[
        NDArray[ObjectLabel],
        NDArray[ObjectLabel]
    ]:
    if sum([len(x) for x in first_objects]) > 0:
        _first_objects = renumber_labels(first_objects, object_numbers)
        _second_objects = renumber_labels(second_objects, neighbor_numbers)

        to_keep = (_first_objects > 0) & (_second_objects > 0)
        _first_objects = _first_objects[to_keep]
        _second_objects = _second_objects[to_keep]
    else:
        _first_objects = numpy.zeros(0, int)
        _second_objects = numpy.zeros(0, int)
    return _first_objects, _second_objects

def get_first_and_second_object_numbers(
        nkept_objects: int, 
        ocenters: NDArray[numpy.int_], 
        ncenters: NDArray[numpy.int_], 
        has_pixels: NDArray[numpy.bool_], 
        neighbor_has_pixels: NDArray[numpy.bool_], 
        object_indexes: NDArray[numpy.int_], 
        neighbor_indexes, 
        neighbors_are_objects
    ) -> Tuple[NDArray[numpy.int_], NDArray[numpy.int_]]:
    #
    # Have to recompute nearest
    #
    first_object_number = numpy.zeros(nkept_objects, int)
    second_object_number = numpy.zeros(nkept_objects, int)
    if nkept_objects > (1 if neighbors_are_objects else 0):
        di = (
            ocenters[object_indexes[:, numpy.newaxis], 0]
            - ncenters[neighbor_indexes[numpy.newaxis, :], 0]
        )
        dj = (
            ocenters[object_indexes[:, numpy.newaxis], 1]
            - ncenters[neighbor_indexes[numpy.newaxis, :], 1]
        )
        distance_matrix = numpy.sqrt(di * di + dj * dj)
        distance_matrix[~has_pixels, :] = numpy.inf
        distance_matrix[:, ~neighbor_has_pixels] = numpy.inf
        #
        # order[:,0] should be arange(nobjects)
        # order[:,1] should be the nearest neighbor
        # order[:,2] should be the next nearest neighbor
        #
        order = numpy.lexsort([distance_matrix]).astype(
            first_object_number.dtype
        )
        if neighbors_are_objects:
            first_object_number[has_pixels] = order[has_pixels, 1] + 1
            if nkept_objects > 2:
                second_object_number[has_pixels] = order[has_pixels, 2] + 1
        else:
            first_object_number[has_pixels] = order[has_pixels, 0] + 1
            if order.shape[1] > 1:
                second_object_number[has_pixels] = order[has_pixels, 1] + 1
    return first_object_number, second_object_number

def get_first_and_second_x_y_vectors_and_angle(
        nobjects: int, 
        nneighbors: int, 
        neighbors_are_objects: bool,
        ocenters: NDArray[numpy.int_], 
        ncenters: NDArray[numpy.int_], 
        object_indexes: NDArray[numpy.int_]
    ) -> Tuple[
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.float_]
    ]:
    angle = numpy.zeros((nobjects,))
    first_x_vector = numpy.zeros((nobjects,))
    second_x_vector = numpy.zeros((nobjects,))
    first_y_vector = numpy.zeros((nobjects,))
    second_y_vector = numpy.zeros((nobjects,))
    #
    # order[:,0] should be arange(nobjects)
    # order[:,1] should be the nearest neighbor
    # order[:,2] should be the next nearest neighbor
    #
    order = numpy.zeros((nobjects, min(nneighbors, 3)), dtype=numpy.uint32)
    j = numpy.arange(nneighbors)
    # (0, 1, 2) unless there are less than 3 neighbors
    partition_keys = tuple(range(min(nneighbors, 3)))
    for i in range(nobjects):
        dr = numpy.sqrt((ocenters[i, 0] - ncenters[j, 0])**2 + (ocenters[i, 1] - ncenters[j, 1])**2)
        order[i, :] = numpy.argpartition(dr, partition_keys)[:3]

    first_neighbor = 1 if neighbors_are_objects else 0
    first_object_index = order[:, first_neighbor]
    first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
    first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
    if nneighbors > first_neighbor + 1:
        second_neighbor = first_neighbor + 1
        second_object_index = order[:, second_neighbor]
        second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
        second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
        v1 = numpy.array((first_x_vector, first_y_vector))
        v2 = numpy.array((second_x_vector, second_y_vector))
        #
        # Project the unit vector v1 against the unit vector v2
        #
        dot = numpy.sum(v1 * v2, 0) / numpy.sqrt(
            numpy.sum(v1 ** 2, 0) * numpy.sum(v2 ** 2, 0)
        )
        angle = numpy.arccos(dot) * 180.0 / numpy.pi
    first_x_vector = first_x_vector[object_indexes]
    second_x_vector = second_x_vector[object_indexes]
    first_y_vector = first_y_vector[object_indexes]
    second_y_vector = second_y_vector[object_indexes]
    angle = angle[object_indexes]
    first_closest_distance = numpy.sqrt(first_x_vector ** 2 + first_y_vector ** 2)
    second_closest_distance = numpy.sqrt(second_x_vector ** 2 + second_y_vector ** 2)
    return (
        first_closest_distance,
        second_closest_distance,
        angle,
    )

def get_extended_dilated_patch(
        patch_mask: NDArray[numpy.bool_], 
        strel:  NDArray[numpy.float_], 
        distance: int
    ):
    if distance <= 5:
        extended = scipy.ndimage.binary_dilation(patch_mask, strel)
    else:
        extended = (scipy.signal.fftconvolve(patch_mask, strel, mode="same") > 0.5)
    return extended

def measure_object_neighbors(
        objects_small_removed_segmented: ObjectSegmentation, 
        kept_labels: ObjectSegmentation,
        neighbor_small_removed_segmented: ObjectSegmentation, 
        neighbor_kept_labels: ObjectSegmentation,
        neighbors_are_objects: bool,
        dimensions: int, 
        distance_value:int, 
        distance_method: NeighborsDistanceMethod, 
        kept_label_has_pixels: NDArray[numpy.bool_],
        nkept_objects: int,
        wants_excluded_objects: bool=True,
        ) -> Tuple[
        NDArray[numpy.float_],
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        Optional[NDArray[numpy.int_]],
    ]:
    labels: ObjectSegmentation = objects_small_removed_segmented.copy()
    neighbor_labels: ObjectSegmentation = neighbor_small_removed_segmented.copy()

    if not wants_excluded_objects:
        # Remove labels not present in kept segmentation while preserving object IDs.
        mask = neighbor_kept_labels > 0
        neighbor_labels[~mask] = 0

    nneighbors = numpy.max(neighbor_labels)
    nobjects = numpy.max(labels)

    _, object_numbers = relate_labels(labels, kept_labels)
    if neighbors_are_objects:
        neighbor_numbers = object_numbers
        neighbor_has_pixels = kept_label_has_pixels
    else:
        _, neighbor_numbers = relate_labels(neighbor_labels, neighbor_kept_labels)
        neighbor_has_pixels = numpy.bincount(neighbor_kept_labels.ravel())[1:] > 0

    neighbor_count = numpy.zeros((nobjects,))
    pixel_count = numpy.zeros((nobjects,))

    distance, scale, labels, expanded_labels, neighbor_labels = get_distance_and_labels(labels, neighbor_labels, neighbors_are_objects, distance_method, distance_value, dimensions)
    
    if nneighbors > (1 if neighbors_are_objects else 0):
        first_objects = []
        second_objects = []
        object_indexes = numpy.arange(nobjects, dtype=numpy.int32) + 1
        #
        # First, compute the first and second nearest neighbors,
        # and the angles between self and the first and second
        # nearest neighbors
        #
        ocenters = centers_of_labels(objects_small_removed_segmented).transpose()
        ncenters = centers_of_labels(neighbor_small_removed_segmented).transpose()
        first_closest_distance, second_closest_distance, angle = get_first_and_second_x_y_vectors_and_angle(nobjects.astype(int), nneighbors.astype(int), neighbors_are_objects, ocenters, ncenters, object_numbers - 1)

        perimeter_outlines: ObjectSegmentation = outline(labels)
        perimeters = fix(scipy.ndimage.sum(numpy.ones(labels.shape), perimeter_outlines, object_indexes))

        strel, strel_touching = get_structuring_elements(distance, dimensions)
        
        ijk_extents = get_extents(labels, object_indexes, distance, dimensions)

        #
        # Loop over all objects
        # Calculate which ones overlap "index"
        # Calculate how much overlap there is of others to "index"
        #
        for object_number in object_numbers:
            if object_number == 0:
                #
                # No corresponding object in small-removed. This means
                # that the object has no pixels, e.g., not renumbered.
                #
                continue
            index = object_number - 1

            patch, npatch = get_patches_from_extents(ijk_extents, labels, neighbor_labels, index, dimensions)

            #
            # Find the neighbors
            #
            patch_mask = patch == (index + 1)

            extended = get_extended_dilated_patch(patch_mask, strel, distance)
            neighbors = numpy.unique(npatch[extended])
            neighbors = neighbors[neighbors != 0]
            if neighbors_are_objects:
                neighbors = neighbors[neighbors != object_number]
            nc = len(neighbors)
            neighbor_count[index] = nc
            if nc > 0:
                first_objects.append(numpy.ones(nc, int) * object_number)
                second_objects.append(neighbors)
            #
            # Find the # of overlapping pixels. Dilate the neighbors
            # and see how many pixels overlap our image. Use a 3x3
            # structuring element to expand the overlapping edge
            # into the perimeter.
            #
            outline_patch = get_outline_patch(ijk_extents, perimeter_outlines, object_number, index, dimensions)

            if neighbors_are_objects:
                extendme = (patch != 0) & (patch != object_number)
            else:
                extendme = (npatch != 0)
            
            extended = get_extended_dilated_patch(extendme, strel_touching, distance)
            overlap = numpy.sum(outline_patch & extended)
            pixel_count[index] = overlap

        first_objects, second_objects = get_first_and_second_objects(first_objects, second_objects, object_numbers, neighbor_numbers)
        percent_touching = pixel_count * 100 / perimeters
        object_indexes = object_numbers - 1
        neighbor_indexes = neighbor_numbers - 1
        #
        # Have to recompute nearest
        #
        first_object_number, second_object_number = get_first_and_second_object_numbers(nkept_objects, ocenters, ncenters, kept_label_has_pixels, neighbor_has_pixels, object_indexes, neighbor_indexes, neighbors_are_objects)

    else:
        
        first_x_vector = numpy.zeros((nobjects,))
        second_x_vector = numpy.zeros((nobjects,))
        first_y_vector = numpy.zeros((nobjects,))
        second_y_vector = numpy.zeros((nobjects,))
        first_closest_distance = 0
        second_closest_distance = 0
        first_object_number = numpy.zeros((nobjects,), int)
        second_object_number = numpy.zeros((nobjects,), int)
        percent_touching = numpy.zeros((nobjects,))

        angle = numpy.zeros((nobjects,))
        object_indexes = object_numbers - 1
        neighbor_indexes = neighbor_numbers - 1
        first_objects = numpy.zeros(0, int)
        second_objects = numpy.zeros(0, int)
        first_x_vector = first_x_vector[object_numbers-1]
        second_x_vector = second_x_vector[object_numbers-1]
        first_y_vector = first_y_vector[object_numbers-1]
        second_y_vector = second_y_vector[object_numbers-1]
        angle = angle[object_numbers-1]
        first_closest_distance = numpy.sqrt(first_x_vector ** 2 + first_y_vector ** 2)
        second_closest_distance = numpy.sqrt(second_x_vector ** 2 + second_y_vector ** 2)

    #
    # Now convert all measurements from the small-removed to
    # the final number set.
    #
    neighbor_count = neighbor_count[object_indexes]
    neighbor_count[~kept_label_has_pixels] = 0
    percent_touching = percent_touching[object_indexes]
    percent_touching[~kept_label_has_pixels] = 0

    return (
        neighbor_count,
        first_object_number,
        second_object_number,
        first_closest_distance,
        second_closest_distance,
        angle,
        percent_touching,
        first_objects,
        second_objects,
        expanded_labels,
    )
