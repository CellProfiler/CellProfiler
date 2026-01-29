import numpy
from typing import Optional, List, Sequence, Union, Tuple
from pydantic import BaseModel, Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.measureimagequality import (
    C_IMAGE_QUALITY, Feature, INTENSITY_FEATURES, SATURATION_FEATURES, C_SCALING
)
from cellprofiler_library.functions.measurement import (
    get_focus_score_for_scale_group,
    get_correlation_for_scale_group,
    get_saturation_value,
    get_intensity_measurement_values,
    get_power_spectrum_measurement_values,
    calculate_threshold_for_threshold_group
)
import centrosome.threshold

class ThresholdConfig(BaseModel):
    algorithm: str
    object_fraction: float = 0.1
    two_class_otsu: bool = True
    use_weighted_variance: bool = True
    assign_middle_to_foreground: bool = True

    def get_scale_string(self) -> Optional[str]:
        if self.algorithm == centrosome.threshold.TM_OTSU:
            scale = "2" if self.two_class_otsu else "3"
            if not self.two_class_otsu:
                scale += "F" if self.assign_middle_to_foreground else "B"
            scale += "W" if self.use_weighted_variance else "S"
            return scale
        elif self.algorithm == centrosome.threshold.TM_MOG:
            return str(int(self.object_fraction * 100))
        return None

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_quality(
    image: ImageGrayscale,
    image_name: str,
    mask: Optional[ImageGrayscaleMask] = None,
    volumetric: bool = False,
    include_image_scalings: bool = False,
    image_scale_value: float = numpy.nan,
    check_blur: bool = False,
    blur_scales: Optional[Sequence[int]] = None,
    check_saturation: bool = False,
    check_intensity: bool = False,
    calculate_threshold: bool = False,
    threshold_groups: Optional[List[ThresholdConfig]] = None
) -> LibraryMeasurements:
    """
    Measure image quality metrics.

    Args:
        image: Input image.
        image_name: Name of the image.
        mask: Optional mask for the image.
        volumetric: Whether the image is volumetric (3D).
        include_image_scalings: Whether to record image scaling.
        image_scale_value: The scaling value of the image.
        check_blur: Whether to calculate blur metrics.
        blur_scales: List of scales (window sizes) for blur metrics.
        check_saturation: Whether to calculate saturation metrics.
        check_intensity: Whether to calculate intensity metrics.
        calculate_threshold: Whether to calculate threshold metrics.
        threshold_groups: List of threshold configurations.

    Returns:
        LibraryMeasurements object containing the measurements.
    """
    measurements = LibraryMeasurements()

    dimensions = image.ndim

    # Image Scalings
    if include_image_scalings:
        feature = f"{C_IMAGE_QUALITY}_{C_SCALING}_{image_name}"
        measurements.add_image_measurement(feature, image_scale_value)

    # Blur Measurements
    if check_blur and blur_scales is not None:
        # Focus Score
        focus_score, scale, local_focus_score = get_focus_score_for_scale_group(
            blur_scales,
            image,
            dimensions,
            mask,
        )

        focus_score_name = f"{C_IMAGE_QUALITY}_{Feature.FOCUS_SCORE.value}_{image_name}"
        measurements.add_image_measurement(focus_score_name, focus_score)

        for idx, s in enumerate(blur_scales):
            local_focus_score_name = f"{C_IMAGE_QUALITY}_{Feature.LOCAL_FOCUS_SCORE.value}_{image_name}_{s}"
            measurements.add_image_measurement(local_focus_score_name, local_focus_score[idx])

        # Correlation
        scale_measurements = get_correlation_for_scale_group(image, blur_scales, mask)
        for s in blur_scales:
            measurements.add_image_measurement(
                f"{C_IMAGE_QUALITY}_{Feature.CORRELATION.value}_{image_name}_{s}",
                scale_measurements[s]
            )

        # Power Spectrum
        if dimensions == 2:
            powerslope = get_power_spectrum_measurement_values(image, mask, dimensions)
            measurements.add_image_measurement(
                f"{C_IMAGE_QUALITY}_{Feature.POWER_SPECTRUM_SLOPE.value}_{image_name}",
                powerslope
            )

    # Saturation Measurements
    if check_saturation:
        percent_minimal, percent_maximal = get_saturation_value(image, mask)
        
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.PERCENT_MAXIMAL.value}_{image_name}",
            percent_maximal
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.PERCENT_MINIMAL.value}_{image_name}",
            percent_minimal
        )

    # Intensity Measurements
    if check_intensity:
        area_feature = Feature.TOTAL_VOLUME.value if volumetric else Feature.TOTAL_AREA.value
        
        (
            pixel_count,
            pixel_sum,
            pixel_mean,
            pixel_std,
            pixel_mad,
            pixel_median,
            pixel_min,
            pixel_max,
        ) = get_intensity_measurement_values(image, mask)

        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{area_feature}_{image_name}", pixel_count
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.TOTAL_INTENSITY.value}_{image_name}", pixel_sum
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.MEAN_INTENSITY.value}_{image_name}", pixel_mean
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.MEDIAN_INTENSITY.value}_{image_name}", pixel_median
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.STD_INTENSITY.value}_{image_name}", pixel_std
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.MAD_INTENSITY.value}_{image_name}", pixel_mad
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.MAX_INTENSITY.value}_{image_name}", pixel_max
        )
        measurements.add_image_measurement(
            f"{C_IMAGE_QUALITY}_{Feature.MIN_INTENSITY.value}_{image_name}", pixel_min
        )

    # Threshold Measurements
    if calculate_threshold and threshold_groups is not None:
        # We need to cast pixel data to float32 for threshold calculations as in original code
        # "pixel_data = image.pixel_data.astype(numpy.float32)"
        pixel_data_float = image.astype(numpy.float32)
        
        for config in threshold_groups:
            global_threshold = calculate_threshold_for_threshold_group(
                image_pixel_data=pixel_data_float,
                mask=mask,
                algorithm=config.algorithm,
                object_fraction=config.object_fraction,
                two_class_otsu=config.two_class_otsu,
                use_weighted_variance=config.use_weighted_variance,
                assign_middle_to_foreground=config.assign_middle_to_foreground,
            )
            
            scale = config.get_scale_string()
            
            hdr = Feature.THRESHOLD.value
            algo = config.algorithm.split(" ")[0] # The original code uses split(" ")[0] for algo name in key
            
            if scale is None:
                feature_name = f"{C_IMAGE_QUALITY}_{hdr}{algo}_{image_name}"
            else:
                feature_name = f"{C_IMAGE_QUALITY}_{hdr}{algo}_{image_name}_{scale}"
                
            measurements.add_image_measurement(feature_name, global_threshold)

    return measurements
    return global_threshold
    
    