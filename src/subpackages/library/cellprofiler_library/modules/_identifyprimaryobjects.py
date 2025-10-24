from centrosome import cpmorphology
import numpy
import scipy
from typing import Annotated, Optional, Union, Tuple
# from cellprofiler_library.functions.object_processing import identify_primary_objects
from cellprofiler_library.modules import threshold
from cellprofiler_library.functions.object_processing import (
    separate_neighboring_objects,
    filter_on_border,
    filter_on_size
)
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.types import Image2DGrayscale, Image2DGrayscaleMask, ObjectSegmentation, ImageGrayscale, ImageGrayscaleMask
import cellprofiler_library.opts.threshold as ThresholdOpts
from cellprofiler_library.opts.identifyprimaryobjects import UnclumpMethod, WatershedMethod, FillHolesMethod

CP_Output_Type = Tuple[
    ObjectSegmentation,
    ObjectSegmentation,
    ObjectSegmentation,
    ObjectSegmentation,
    ObjectSegmentation,
    ObjectSegmentation,
    float,
    int,
    Union[float, ImageGrayscale],
    Union[float, ImageGrayscale],
    Optional[float],
    ImageGrayscaleMask,
    float,
    float
]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def identifyprimaryobjects(
    image:                      Annotated[Image2DGrayscale, Field(description="Input image")],
    mask:                       Annotated[Optional[Image2DGrayscaleMask], Field(description="Input mask")] = None,      
    threshold_method:           Annotated[ThresholdOpts.Method, Field(description="Thresholding method")] = ThresholdOpts.Method.MINIMUM_CROSS_ENTROPY,
    threshold_scope:            Annotated[ThresholdOpts.Scope, Field(description="Thresholding scope")] = ThresholdOpts.Scope.GLOBAL,
    assign_middle_to_foreground:Annotated[ThresholdOpts.Assignment, Field(description="Assign middle to foreground")] = ThresholdOpts.Assignment.BACKGROUND,
    log_transform:              Annotated[bool, Field(description="Apply log transform to image before thresholding")] = False,
    threshold_correction_factor:Annotated[float, Field(description="Multiply threshold by this factor")] = 1.0,
    threshold_min:              Annotated[float, Field(description="Minimum threshold value")] = 0.0,
    threshold_max:              Annotated[float, Field(description="Maximum threshold value")] = 1.0,
    window_size:                Annotated[int, Field(description="Size of window for thresholding")] = 50,
    threshold_smoothing:        Annotated[float, Field(description="Smoothing factor for thresholding")] = 0.0,
    lower_outlier_fraction:     Annotated[float, Field(description="Fraction of pixels to use for lower outlier detection")] = 0.05,
    upper_outlier_fraction:     Annotated[float, Field(description="Fraction of pixels to use for upper outlier detection")] = 0.05,
    averaging_method:           Annotated[ThresholdOpts.AveragingMethod, Field(description="Averaging method for thresholding")] = ThresholdOpts.AveragingMethod.MEAN,
    variance_method:            Annotated[ThresholdOpts.VarianceMethod, Field(description="Variance method for thresholding")] = ThresholdOpts.VarianceMethod.STANDARD_DEVIATION,
    number_of_deviations:       Annotated[int, Field(description="Number of deviations for thresholding")] = 2,
    automatic:                  Annotated[bool, Field(description="Automatically determine thresholding parameters")] = False,
    exclude_size:               Annotated[bool, Field(description="Exclude objects smaller than this size")] = True,
    min_size:                   Annotated[int, Field(description="Minimum object size")] = 10,
    max_size:                   Annotated[int, Field(description="Maximum object size")] = 40,
    exclude_border_objects:     Annotated[bool, Field(description="Exclude objects touching the border")] = True,
    unclump_method:             Annotated[UnclumpMethod, Field(description="Unclump method for thresholding")] = UnclumpMethod.INTENSITY,
    watershed_method:           Annotated[WatershedMethod, Field(description="Watershed method for thresholding")] = WatershedMethod.INTENSITY,
    fill_holes_method:          Annotated[FillHolesMethod, Field(description="Fill holes method for thresholding")] = FillHolesMethod.THRESHOLDING,
    smoothing_filter_size:      Annotated[Optional[int], Field(description="Smoothing filter size for thresholding")] = None,
    automatic_suppression:      Annotated[bool, Field(description="Automatically calculate suppression size")] = True,
    maxima_suppression_size:    Annotated[float, Field(description="Suppression size for thresholding")] = 7.0,
    low_res_maxima:             Annotated[bool, Field(description="Use low-resolution maxima for thresholding")] = True,
    maximum_object_count:       Annotated[Optional[int], Field(description="Maximum number of objects to threshold")] = None,
    predefined_threshold:       Annotated[Optional[float], Field(description="Predefined threshold value")] = None,
    return_cp_output:           Annotated[bool, Field(description="Return CellProfiler output")] = False
) -> Union[ObjectSegmentation, CP_Output_Type]:
    
    if automatic:
        # Define automatic settings by recursively calling identifyprimaryobjects and defining 
        # the automatic settings
        return identifyprimaryobjects(
            image,
            mask=mask,
            threshold_smoothing=1,
            log_transform=False,
            threshold_scope=ThresholdOpts.Scope.GLOBAL,
            threshold_method=ThresholdOpts.Method.MINIMUM_CROSS_ENTROPY,
            automatic=False,  # False, since this recursive call defines what the automatic settings are
            exclude_size=exclude_size,
            min_size=min_size,
            max_size=max_size,
            exclude_border_objects=exclude_border_objects,
            unclump_method=UnclumpMethod.INTENSITY,
            watershed_method=WatershedMethod.INTENSITY,
            fill_holes_method=FillHolesMethod.THRESHOLDING,
            smoothing_filter_size=None,
            low_res_maxima=True if min_size > 10 else False,
            automatic_suppression=True,
            return_cp_output=return_cp_output,
        )

    final_threshold, orig_threshold, guide_threshold, binary_image, sigma = threshold(
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
        volumetric=False,  # IdentifyPrimaryObjects does not support 3D
        predefined_threshold=predefined_threshold,
    )

    global_threshold = numpy.mean(numpy.atleast_1d(final_threshold))

    if fill_holes_method == FillHolesMethod.THRESHOLDING.value:
        binary_image = cpmorphology.fill_labeled_holes(
            binary_image, size_fn=lambda size, is_foreground: size < max_size * max_size
        )

    # Label the thresholded image
    labeled_image = scipy.ndimage.label(binary_image, numpy.ones((3, 3), bool))[0]

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
        ) = separate_neighboring_objects(
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
    # Maxima image will be retuened
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
    if exclude_border_objects:
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
    if fill_holes_method != FillHolesMethod.NEVER:
        labeled_image = cpmorphology.fill_labeled_holes(labeled_image)

    # Relabel the image
    labeled_image, object_count = cpmorphology.relabel(labeled_image)

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

