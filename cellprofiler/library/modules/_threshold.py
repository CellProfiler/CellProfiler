import centrosome
import numpy
import skimage
from cellprofiler.library.functions.image_processing import get_adaptive_threshold, get_global_threshold, apply_threshold

### Add to functions ###

def threshold(
    image,
    mask=None,
    threshold_scope="global",
    threshold_method="otsu",
    assign_middle_to_foreground="foreground",
    log_transform=False,
    threshold_correction_factor=1,
    threshold_min=0,
    threshold_max=1,
    window_size=50,
    smoothing=0,
    lower_outlier_fraction=0.05,
    upper_outlier_fraction=0.05,
    averaging_method="mean",
    variance_method="standard_deviation",
    number_of_deviations=2,
    volumetric=False,
    **kwargs,
):
    """
    Returns three threshold values:

    Final threshold: Threshold following application of the 
    threshold_correction_factor and clipping to min/max threshold
    
    orig_threshold: The threshold following either adaptive or global 
    thresholding strategies, prior to correction
    
    guide_threshold: Only produced by adaptive threshold, otherwise None. 
    This is the global threshold that constrains the adaptive threshold 
    within a certain range, as defined by global_limits (default [0.7, 1.5])
    """
    print(
        threshold_scope,
        threshold_method,
        assign_middle_to_foreground,
        log_transform,
        threshold_correction_factor,
        threshold_min,
        threshold_max,
        window_size,
        smoothing,
        lower_outlier_fraction,
        upper_outlier_fraction,
        averaging_method,
        variance_method,
        number_of_deviations,
        volumetric,
        **kwargs,
    )

# How to account for "default" values that are not required
# For example, if a user selects Otsu thresholding but there exists defaults for
# variance_method, will this be passed as kwargs? Probably not

    # Mask image
    # This is an addition, as previously masks were only applied when smoothing
    # That is, thresholds were calculated on image.pixel_data without a mask
    if mask is not None:
        image = numpy.where(mask, image, True)

    # Only pass robust_background kwargs when selected as the threshold_method
    if threshold_method.casefold() == "robust_background":
        kwargs = {
            "lower_outlier_fraction": lower_outlier_fraction,
            "upper_outlier_fraction": upper_outlier_fraction,
            "averaging_method": averaging_method,
            "variance_method": variance_method,
            "number_of_deviations": number_of_deviations,            
        }
    # else:
    #     kwargs=kwargs
    if threshold_scope.casefold() == "adaptive":
        final_threshold = get_adaptive_threshold(
            image,
            threshold_method=threshold_method,
            window_size=window_size,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_correction_factor=threshold_correction_factor,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric,
            **kwargs,
            # lower_outlier_fraction=lower_outlier_fraction,
            # upper_outlier_fraction=upper_outlier_fraction,
            # averaging_method=averaging_method,
            # variance_method=variance_method,
            # number_of_deviations=number_of_deviations,
        )
        orig_threshold = get_adaptive_threshold(
            image,
            threshold_method=threshold_method,
            window_size=window_size,
            threshold_min=0,
            threshold_max=1,
            threshold_correction_factor=1,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric,
            **kwargs,
            # lower_outlier_fraction=lower_outlier_fraction,
            # upper_outlier_fraction=upper_outlier_fraction,
            # averaging_method=averaging_method,
            # variance_method=variance_method,
            # number_of_deviations=number_of_deviations,
        )

        guide_threshold = get_global_threshold(
            image,
            threshold_method,
            threshold_min,
            threshold_max,
            threshold_correction_factor,
            assign_middle_to_foreground,
            log_transform,
            volumetric,
            **kwargs,
            )

        binary_image = apply_threshold(
            image,
            threshold=final_threshold,
            mask=mask,
            smoothing=smoothing,
        )

        return final_threshold, orig_threshold, guide_threshold, binary_image

    elif threshold_scope.casefold() == "global":
        final_threshold = get_global_threshold(
            image,
            threshold_method=threshold_method,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_correction_factor=threshold_correction_factor,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric,
            )

        orig_threshold = get_global_threshold(
            image,
            threshold_method=threshold_method,
            threshold_min=0,
            threshold_max=1,
            threshold_correction_factor=1,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric
            )
        guide_threshold = None

        binary_image = apply_threshold(
            image,
            threshold=final_threshold,
            mask=mask,
            smoothing=smoothing,
        )

        return final_threshold, orig_threshold, guide_threshold, binary_image

# apply_threshold(image, threshold, smoothing=0)