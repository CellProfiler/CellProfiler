# What IDPrimary does:
# Threshold, smooth,

from . import threshold
from typing import Literal
from cellprofiler_library.functions.object_processing import (
    get_maxima,
    smooth_image,
    filter_on_size,
    filter_on_border,
    separate_neighboring_objects,
)
import centrosome
import scipy
import numpy


def identifyprimaryobjects(
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
    exclude_size: bool = False,
    min_size: int = 10,
    max_size: int = 40,
    exclude_border: bool = False,
    unclump_method: Literal["intensity", "shape", "none"] = "intensity",
    watershed_method: Literal["intensity", "shape", "propagate", "none"] = "intensity",
    fill_holes_method: Literal["never", "thresholding", "declumping"] = "thresholding",
    declump_smoothing: float = None,
    low_res_maxima: bool = False,
    maxima_suppression_size: int = 7,
    automatic_suppression: bool = False,
    maximum_object_count: int = None,
    predefined_threshold: float = None,
    return_cp_output: bool = False
    # **kwargs
):
    """
    if automatic == True, the following settings are used:
    {
        fill_holes_method = "thresholding"
        if min_size > 10:
            low_res_maxima = True
        else:
            low_res_maxima = False
        automatic_suppression = True
        unclump_method = "intensity"
        watershed_method = "intensity"
    }
    automatic is passed to threshold, which results
    in the following settings being used for threshold:
    {
    log_transform = False
    threshold_smoothing = 1
    threshold_scope = "global" (simply called smoothing in threshold)
    threshold_method = "minimum_cross_entropy"
    }
    """
    # Define automatic settings
    if automatic:
        if return_cp_output:
            return identifyprimaryobjects(
                    image,
                    mask=mask,
                    automatic=False, # Since this call sets up automatic settings
                    exclude_size=exclude_size,
                    min_size=min_size,
                    max_size=max_size,
                    exclude_border=exclude_border,
                    unclump_method="intensity",
                    watershed_method="intensity",
                    fill_holes_method="thresholding",
                    declump_smoothing=None,
                    low_res_maxima=True if min_size > 10 else False,
                    automatic_suppression=True,
                    return_cp_output=return_cp_output
            )
        else:
            return identifyprimaryobjects(
                image,
                mask=mask,
                automatic=False,
                exclude_size=exclude_size,
                min_size=min_size,
                max_size=max_size,
                exclude_border=exclude_border,
                unclump_method="intensity",
                watershed_method="intensity",
                fill_holes_method="thresholding",
                declump_smoothing=None,
                low_res_maxima=True if min_size > 10 else False,
                automatic_suppression=True,
                return_cp_output=return_cp_output
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
        volumetric=False,  # IDPrimary does not support 3D
        automatic=automatic,
        predefined_threshold=predefined_threshold
    )

    global_threshold = numpy.mean(numpy.atleast_1d(final_threshold))

    if fill_holes_method.casefold() == "thresholding":
        binary_image = centrosome.cpmorphology.fill_labeled_holes(
            binary_image, size_fn=lambda size, is_foreground: size < max_size * max_size
        )

    # Label the thresholded image
    labeled_image = scipy.ndimage.label(binary_image, numpy.ones((3, 3), bool))[0]

    if declump_smoothing is None:
        declump_smoothing_filter_size = 2.35 * min_size / 3.5
    else:
        declump_smoothing_filter_size = declump_smoothing

    if return_cp_output:
        labeled_image, maxima_suppression_size = separate_neighboring_objects(
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
            return_suppression_size=True,
        )
    else:
        labeled_image = separate_neighboring_objects(
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
    if exclude_border:
        labeled_image = filter_on_border(labeled_image, mask)
        border_excluded_labeled_image[labeled_image > 0] = 0

    # Filter out small and large objects
    size_excluded_labeled_image = labeled_image.copy()

    # If requested, remove cells that are outside the size range
    if exclude_size:
        labeled_image, small_removed_labels = filter_on_size(
            labeled_image, min_size, max_size, return_only_small=True
        )
    else:
        labeled_image, small_removed_labels = labeled_image, labeled_image.copy()

    size_excluded_labeled_image[labeled_image > 0] = 0

    #
    # Fill holes again after watershed
    #
    if fill_holes_method.casefold() != "never":
        labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

    # Relabel the image
    labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

    if maximum_object_count is not None:
        if object_count > maximum_object_count:
            labeled_image = numpy.zeros(labeled_image.shape, int)
            border_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
            size_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
            object_count = 0

    if return_cp_output:
        return labeled_image, unedited_labels, small_removed_labels, size_excluded_labeled_image, border_excluded_labeled_image, maxima_suppression_size, object_count, global_threshold, sigma
    else:
        return labeled_image
