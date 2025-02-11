from typing import Literal
import numpy
import scipy
import skimage
from cellprofiler_library.functions.object_processing import smooth_image
from cellprofiler_library.modules import threshold

def identifyprimaryobjects3d(
    image,
    mask=None,
    threshold_method: Literal[
        "minimum_cross_entropy", "otsu", "multiotsu", "robust_background"
    ] = "minimum_cross_entropy",
    threshold_scope: Literal["global", "adaptive"] = "global",
    assign_middle_to_foreground: Literal["foreground", "background"] = "background",
    log_transform: bool = False,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    window_size: int = 50,
    threshold_smoothing: float = 0.0,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: Literal["mean", "median", "mode"] = "mean",
    variance_method: Literal[
        "standard_deviation", "median_absolute_deviation"
    ] = "standard_deviation",
    number_of_deviations: int = 2,
    automatic: bool = False,
    exclude_size: bool = True,
    min_size: int = 10,
    max_size: int = 40,
    exclude_border_objects: bool = True,
    unclump_method: Literal["intensity", "shape", "none"] = "intensity",
    watershed_method: Literal["intensity", "shape", "propagate", "none"] = "intensity",
    fill_holes_method: Literal["never", "thresholding", "declumping"] = "thresholding",
    smoothing_filter_size: int = None,
    automatic_suppression: bool = True,
    maxima_suppression_size: float = 7,
    low_res_maxima: bool = True,
    maximum_object_count: int = None,
    predefined_threshold: float = None,
    return_cp_output: bool = False
):
    if automatic:
        return identifyprimaryobjects3d(
            image,
            mask=mask,
            automatic=False,  # Since this call sets up automatic settings
            exclude_size=exclude_size,
            min_size=min_size,
            max_size=max_size,
            exclude_border_objects=exclude_border_objects,
            unclump_method="intensity",
            watershed_method="intensity",
            fill_holes_method="thresholding",
            declump_smoothing=None,
            low_res_maxima=True if min_size > 10 else False,
            automatic_suppression=True,
            return_cp_output=return_cp_output,
        )
    
    (final_threshold, orig_threshold, guide_threshold, binary_image, sigma) = threshold(
        image=image,
        mask=mask,
        threshold_scope=threshold_scope,
        threshold_method=threshold_method,
        assign_middle_to_foreground=assign_middle_to_foreground,
        log_transform=log_transform,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        window_size=window_size,
        smoothing=threshold_smoothing,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
        volumetric=True,
        automatic=automatic,
        predefined_threshold=predefined_threshold,
    )

    global_threshold = numpy.mean(numpy.atleast_1d(final_threshold))
    # Label the thresholded image
    labeled_image = scipy.ndimage.label(binary_image, numpy.ones((3, 3, 3), bool))[0]

    if smoothing_filter_size is None:
        declump_smoothing_filter_size = 2.35 * min_size / 3.5
    else:
        declump_smoothing_filter_size = smoothing_filter_size

        # If no declumping is selected, a maxima image is not returned
    if return_cp_output:
        (
            labeled_image,
            labeled_maxima,
            maxima_suppression_size,
        ) = separate_neighboring_objects_3d(
            image,
            labeled_image=labeled_image,
            mask=mask,
            unclump_method=unclump_method,
            watershed_method=watershed_method,
            fill_holes_method=fill_holes_method,
            filter_size=declump_smoothing_filter_size,
            min_size=min_size,
            max_size=max_size,
            low_res_maxima=low_res_maxima,
            automatic_suppression=automatic_suppression,
            maxima_suppression_size=maxima_suppression_size,
            return_cp_output=True,
        )
    else:
        labeled_image = separate_neighboring_objects_3d(
            image,
            labeled_image=labeled_image,
            mask=mask,
            unclump_method=unclump_method,
            watershed_method=watershed_method,
            fill_holes_method=fill_holes_method,
            filter_size=declump_smoothing_filter_size,
            min_size=min_size,
            max_size=max_size,
            low_res_maxima=low_res_maxima,
            maxima_suppression_size=maxima_suppression_size,
            automatic_suppression=automatic_suppression,
        )

    unedited_labels = labeled_image.copy()

    # Filter out objects touching the border or mask
    border_excluded_labeled_image = labeled_image.copy()
    if exclude_border_objects:
        labeled_image = skimage.segmentation.clear_border(labeled_image, mask=mask)
    border_excluded_labeled_image[labeled_image > 0] = 0
    
    # Filter out small and large objects
    size_excluded_labeled_image = labeled_image.copy()

    # If requested, remove cells that are outside the size range
    if exclude_size:
        labeled_image, small_removed_labels = filter_on_size_3d(
            labeled_image, min_size, max_size, return_only_small=True
        )
    else:
        labeled_image, small_removed_labels = labeled_image, labeled_image.copy()

    size_excluded_labeled_image[labeled_image > 0] = 0

    #
    # Fill holes again after watershed
    #
    if fill_holes_method.casefold() != "never":
        labeled_image = skimage.morphology.remove_small_holes(labeled_image.astype(bool))
        # remove_small_holes returns bool, so relabel
        labeled_image = scipy.ndimage.label(labeled_image, numpy.ones((3, 3, 3), bool))[0]

    # Relabel the image
    labeled_image = skimage.segmentation.relabel_sequential(labeled_image)[0]

    object_count = numpy.max(labeled_image)

    if maximum_object_count is not None:
        if object_count > maximum_object_count:
            labeled_image = numpy.zeros(labeled_image.shape, int)
            border_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
            size_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
            object_count = 0

    if return_cp_output:
        return (
            labeled_image,
            unedited_labels,
            small_removed_labels,
            size_excluded_labeled_image,
            border_excluded_labeled_image,
            labeled_maxima,
            maxima_suppression_size,
            object_count,
            final_threshold,
            orig_threshold,
            guide_threshold,
            binary_image,
            global_threshold,
            sigma,
        )
    else:
        return labeled_image


def separate_neighboring_objects_3d(
    image,
    labeled_image,
    mask=None,
    unclump_method: Literal["intensity", "shape", "none"] = "intensity",
    watershed_method: Literal["intensity", "shape", "propagate", "none"] = "intensity",
    fill_holes_method: Literal["never", "thresholding", "declumping"] = "thresholding",
    filter_size=None,
    min_size=10,
    max_size=40,
    low_res_maxima=False,
    maxima_suppression_size=7,
    automatic_suppression=False,
    return_cp_output=False,
):
    if unclump_method.casefold() == "none" or watershed_method.casefold() == "none":
        if return_cp_output:
            return labeled_image, numpy.zeros_like(labeled_image), 7
        else:
            return labeled_image
        
    blurred_image = smooth_image(image, mask, filter_size, min_size)

    # For image resizing, the min_size must be larger than 10
    if min_size > 10 and low_res_maxima:
        image_resize_factor = 10.0 / float(min_size)
        if automatic_suppression:
            maxima_suppression_size = 7
        else:
            maxima_suppression_size = (
                maxima_suppression_size * image_resize_factor + 0.5
            )
    else:
        image_resize_factor = 1.0
        if automatic_suppression:
            maxima_suppression_size = min_size / 1.5
        else:
            maxima_suppression_size = maxima_suppression_size

    footprint = skimage.morphology.ball(
        max(1, maxima_suppression_size - 0.5)
    )

    distance_transformed_image = None

    if unclump_method.casefold() == "intensity":
        # Find maxima in masked regions
        masked_blurred_image = numpy.where(labeled_image > 0, blurred_image, 0)
        maxima_coords = skimage.feature.peak_local_max(
            masked_blurred_image, footprint=footprint
        )
        maxima_image = numpy.zeros(labeled_image.shape, dtype=bool)
        maxima_image[tuple(maxima_coords.T)] = True
        maxima_image = scipy.ndimage.label(maxima_image)[0]

        # Erode blobs of touching maxima to a single point
        # NOT IMPLEMENTED FOR 3D
        # maxima_image = centrosome.cpmorphology.binary_shrink(maxima_image)

    elif unclump_method.casefold() == "shape":
        if fill_holes_method.casefold() == "never":
            foreground = skimage.morphology.remove_small_holes(labeled_image) > 0
        else:
            foreground = labeled_image > 0
        distance_transformed_image = scipy.ndimage.distance_transform_edt(foreground)

        # randomize the distance slightly to get unique maxima
        # numpy.random.seed(0)
        # distance_transformed_image += numpy.random.uniform(
        #     0, 0.001, distance_transformed_image.shape
        # )
        masked_distance_transformed_image = numpy.where(labeled_image > 0, distance_transformed_image, 0)

        maxima_coords = skimage.feature.peak_local_max(
                masked_distance_transformed_image, footprint=footprint
            )
        maxima_image = numpy.zeros(labeled_image.shape, dtype=bool)
        maxima_image[tuple(maxima_coords.T)] = True
        maxima_image = scipy.ndimage.label(maxima_image)[0]
    else:
        raise ValueError(f"Unsupported unclump method: {unclump_method}")


    # Create the image for watershed
    if watershed_method.casefold() == "intensity":
        # use the reverse of the image to get valleys at peaks
        watershed_image = 1 - image
    elif watershed_method.casefold() == "shape":
        if distance_transformed_image is None:
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                labeled_image > 0
            )
        watershed_image = -distance_transformed_image
        watershed_image = watershed_image - numpy.min(watershed_image)
    elif watershed_method.casefold() == "propagate":
        raise NotImplementedError(f"{watershed_method} not supported in 3D")
    else:
        raise ValueError(f"Unsupported watershed method: {watershed_method}")
    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = scipy.ndimage.label(
        maxima_image
    )
    markers_dtype = (
        numpy.int16 if object_count < numpy.iinfo(numpy.int16).max else numpy.int32
    )
    markers = numpy.zeros(watershed_image.shape, markers_dtype)
    markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]

    #
    # Some labels have only one maker in them, some have multiple and
    # will be split up.
    #

    watershed_boundaries = skimage.segmentation.watershed(
        connectivity=1,
        image=watershed_image,
        markers=markers,
        mask=labeled_image != 0,
    )

    watershed_boundaries = -watershed_boundaries

    if return_cp_output:
        return watershed_boundaries, labeled_maxima, maxima_suppression_size
    else:
        return watershed_boundaries



def filter_on_size_3d(labeled_image, min_size, max_size, return_only_small=False):
    """Filter the labeled image based on the size range

    labeled_image - pixel image labels (z, x, y)
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed
    """
    labeled_image = labeled_image.copy()

    # Take the max since objects may have been removed, but their label number
    # has not been adjusted accordingly. eg. array [2, 1, 0, 3] has label 2
    # removed due to being on the border, so the array is [0, 1, 0, 3].
    # Object numbers/indices will be used for slicing in areas[labeled_image]
    object_count = numpy.max(labeled_image)
    # Check if there are no labelled objects
    if object_count > 0:
        areas = scipy.ndimage.measurements.sum(
            numpy.ones(labeled_image.shape),
            labeled_image,
            numpy.array(list(range(0, object_count + 1)), dtype=numpy.int32),
        )
        areas = numpy.array(areas, dtype=int)
        depth = labeled_image.shape[0]
        # Calculate volume as a sphere
        # min_allowed_area = numpy.pi * (min_size**3) / 6 
        # max_allowed_area = numpy.pi * (max_size**3) / 6
        # Calculate volume as an ellipsoid with depth as the image z
        # This strategy is imperfect, since there may be scenarios in which
        # the depth of the image is greater than the depth of the cell
        # Thus, depth should likely be user defined as it's difficult to define
        # otherwise
        min_allowed_area = (numpy.pi / 6) * min_size * min_size * depth
        max_allowed_area = (numpy.pi / 6) * max_size * max_size * depth

        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        if return_only_small:
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
            return labeled_image, small_removed_labels
        else:
            labeled_image[area_image > max_allowed_area] = 0
            return labeled_image
    else:
        if return_only_small:
            small_removed_labels = labeled_image.copy()
            return labeled_image, small_removed_labels
        else:
            return labeled_image