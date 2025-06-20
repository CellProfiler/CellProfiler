from ..functions.image_processing import (
    get_adaptive_threshold,
    get_global_threshold,
    apply_threshold,
)
from ..opts.threshold import (
    Scope,
    Method,
    Assignment,
    AveragingMethod,
    VarianceMethod,
)

from ..types import ImageGrayscale, ImageGrayscaleMask

from pydantic import BaseModel, Field, BeforeValidator, validate_call, ConfigDict
from typing import Optional, Tuple, Annotated, Dict, Any, Union

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def threshold(
    image:                      Annotated[ImageGrayscale, Field(description="Image to threshold")],
    mask:                       Annotated[Optional[ImageGrayscaleMask], Field(description="Mask to apply to the image")] = None,
    threshold_scope:            Annotated[Scope, Field(description="Thresholding scope"), BeforeValidator(str.casefold)] = Field(default=Scope.GLOBAL),
    threshold_method:           Annotated[Method, Field(description="Thresholding method"), BeforeValidator(str.casefold)] = Field(default=Method.OTSU),
    assign_middle_to_foreground:Annotated[Assignment, Field(description="Assign middle to foreground"), BeforeValidator(str.casefold)] = Field(default=Assignment.FOREGROUND),
    log_transform:              Annotated[bool, Field(description="Log transform")] = Field(default=False),
    threshold_correction_factor:Annotated[float, Field(description="Threshold correction factor")] = Field(default=1),
    threshold_min:              Annotated[float, Field(description="Minimum threshold")] = Field(default=0),
    threshold_max:              Annotated[float, Field(description="Maximum threshold")] = Field(default=1),
    window_size:                Annotated[int, Field(description="Window size for adaptive thresholding")] = Field(default=50),
    smoothing:                  Annotated[float, Field(description="Smoothing factor")] = Field(default=0),
    lower_outlier_fraction:     Annotated[float, Field(description="Lower outlier fraction")] = Field(default=0.05),
    upper_outlier_fraction:     Annotated[float, Field(description="Upper outlier fraction")] = Field(default=0.05),
    averaging_method:           Annotated[AveragingMethod, Field(description="Averaging method"), BeforeValidator(str.casefold)] = Field(default=AveragingMethod.MEAN),
    variance_method:            Annotated[VarianceMethod, Field(description="Variance method"), BeforeValidator(str.casefold)] = Field(default=VarianceMethod.STANDARD_DEVIATION),
    number_of_deviations:       Annotated[int, Field(description="Number of deviations")] = Field(default=2),
    volumetric:                 Annotated[bool, Field(description="Volumetric thresholding")] = Field(default=False),
    automatic:                  Annotated[bool, Field(description="Automatic thresholding")] = Field(default=False),
    **kwargs:                   Annotated[Any, Field(description="Additional keyword arguments")]
) -> Tuple[
    Annotated[Union[Any, float, int], Field(description="Final threshold")],
    Annotated[Union[Any, float, int], Field(description="Original threshold")],
    Annotated[Union[Any, float, int], Field(description="Guide threshold")],
    Annotated[ImageGrayscaleMask, Field(description="Binary image")],
    Annotated[float, Field(description="Sigma value")],

]:
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

    if automatic:
        # Use automatic settings
        smoothing = 1
        log_transform = False
        threshold_scope = Scope.GLOBAL
        threshold_method = Method.MINIMUM_CROSS_ENTROPY

    # Only pass robust_background kwargs when selected as the threshold_method
    if threshold_method == Method.ROBUST_BACKGROUND:
        kwargs = {
            "lower_outlier_fraction": lower_outlier_fraction,
            "upper_outlier_fraction": upper_outlier_fraction,
            "averaging_method": averaging_method,
            "variance_method": variance_method,
            "number_of_deviations": number_of_deviations,
        }

    if threshold_scope == Scope.ADAPTIVE:
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

    elif threshold_scope == Scope.GLOBAL:
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
