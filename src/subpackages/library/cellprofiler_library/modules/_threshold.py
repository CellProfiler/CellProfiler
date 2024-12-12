from ..functions.image_processing import (
    get_adaptive_threshold,
    get_global_threshold,
    apply_threshold,
)

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
    predefined_threshold=None,
    volumetric=False,
    automatic=False,
    **kwargs,
):
    """
    Returns three threshold values and a binary image.
    Thresholds returned are:

    Final threshold: Threshold following application of the
    threshold_correction_factor and clipping to min/max threshold

    orig_threshold: The threshold following either adaptive or global
    thresholding strategies, prior to correction

    guide_threshold: Only produced by adaptive threshold, otherwise None.
    This is the global threshold that constrains the adaptive threshold
    within a certain range, as defined by global_limits (default [0.7, 1.5])
    """

    # A predefined threshold has been requested (ie. a manual or measurement one)
    if predefined_threshold is not None:
        final_threshold = predefined_threshold
        final_threshold *= threshold_correction_factor
        # For manual thresholds in the GUI, min/max filtering is not applied
        if threshold_min is not None and threshold_max is not None:
            final_threshold = min(max(final_threshold, threshold_min), threshold_max)
        orig_threshold = predefined_threshold
        guide_threshold = None
        binary_image, sigma = apply_threshold(
            image=image,
            threshold=final_threshold,
            mask=mask,
            smoothing=smoothing
            )
        return final_threshold, orig_threshold, guide_threshold, binary_image, sigma

    if automatic:
        # Use automatic settings
        smoothing = 1
        log_transform = False
        threshold_scope = "global"
        threshold_method = "minimum_cross_entropy"

    # Only pass robust_background kwargs when selected as the threshold_method
    if threshold_method.casefold() == "robust_background":
        kwargs = {
            "lower_outlier_fraction": lower_outlier_fraction,
            "upper_outlier_fraction": upper_outlier_fraction,
            "averaging_method": averaging_method,
            "variance_method": variance_method,
            "number_of_deviations": number_of_deviations,
        }

    if threshold_scope.casefold() == "adaptive":
        final_threshold = get_adaptive_threshold(
            image,
            mask=mask,
            threshold_method=threshold_method,
            window_size=window_size,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_correction_factor=threshold_correction_factor,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric,
            **kwargs,
        )
        orig_threshold = get_adaptive_threshold(
            image,
            mask=mask,
            threshold_method=threshold_method,
            window_size=window_size,
            # If automatic=True, do not correct the threshold
            threshold_min=threshold_min if automatic else 0,
            threshold_max=threshold_max if automatic else 1,
            threshold_correction_factor=threshold_correction_factor if automatic else 1,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            volumetric=volumetric,
            **kwargs,
        )

        guide_threshold = get_global_threshold(
            image,
            mask=mask,
            threshold_method=threshold_method,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_correction_factor=threshold_correction_factor,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            **kwargs,
        )

        binary_image, sigma = apply_threshold(
            image,
            threshold=final_threshold,
            mask=mask,
            smoothing=smoothing,
        )

        return final_threshold, orig_threshold, guide_threshold, binary_image, sigma

    elif threshold_scope.casefold() == "global":
        final_threshold = get_global_threshold(
            image,
            mask=mask,
            threshold_method=threshold_method,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_correction_factor=threshold_correction_factor,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            **kwargs,
        )
        orig_threshold = get_global_threshold(
            image,
            mask=mask,
            threshold_method=threshold_method,
            # If automatic=True, do not correct the threshold
            threshold_min=threshold_min if automatic else 0,
            threshold_max=threshold_max if automatic else 1,
            threshold_correction_factor=threshold_correction_factor if automatic else 1,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            **kwargs,
        )
        guide_threshold = None
        binary_image, sigma = apply_threshold(
            image,
            threshold=final_threshold,
            mask=mask,
            smoothing=smoothing,
        )
        return final_threshold, orig_threshold, guide_threshold, binary_image, sigma
