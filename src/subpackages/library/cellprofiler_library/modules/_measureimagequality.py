import numpy
from numpy.typing import NDArray
from typing import Optional, List, Sequence, Annotated, Union, Tuple, Any
from pydantic import BaseModel, Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.measureimagequality import (
     Feature, ScaledThresholdMethod, TemplateMeasurementFormat
)
from cellprofiler_library.functions.measurement import (
    get_focus_score_for_scale_group,
    get_correlation_for_scale_group,
    get_saturation_value,
    get_intensity_measurement_values,
    get_power_spectrum_measurement_values,
    calculate_threshold_for_threshold_group
)

ImageQualityMesVal = Union[
    # image scaling, blur powerslope, saturation percent_minimal|percent_maximal, threshold global_threshold
    float, 
    # blur focus_score
    NDArray[numpy.float_],
    # intensity pixel_count
    numpy.int_,
    # blur local_focus_score
    # intensity
    #  pixel_sum|pixel_mean|pixel_std|pixel_mad|pixel_median|pixel_min|pixel_max
    numpy.float_,
    # converted stat vals
    str
]
ImageQualityStatistics = List[Tuple[str, ImageQualityMesVal]]

class ImageQualityDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    statistics: ImageQualityStatistics

class ThresholdConfig(BaseModel):
    algorithm: str
    object_fraction: float = 0.1
    two_class_otsu: bool = True
    use_weighted_variance: bool = True
    assign_middle_to_foreground: bool = True

    def get_scale_string(self) -> Optional[str]:
        if self.algorithm == ScaledThresholdMethod.OTSU.value:
            scale = "2" if self.two_class_otsu else "3"
            if not self.two_class_otsu:
                scale += "F" if self.assign_middle_to_foreground else "B"
            scale += "W" if self.use_weighted_variance else "S"
            return scale
        elif self.algorithm == ScaledThresholdMethod.MOG.value:
            return str(int(self.object_fraction * 100))
        return None

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_quality(
    image:                  Annotated[ImageGrayscale, Field(description="The image to measure.")],
    image_name:             Annotated[str, Field(description="The name of the image.")],
    image_mask:             Annotated[Optional[ImageGrayscaleMask], Field(description="The mask for the image.")] = None,
    volumetric:             Annotated[bool, Field(description="Whether the image is volumetric (3D).")] = False,
    include_image_scalings: Annotated[bool, Field(description="Whether to include image scalings in the measurements.")] = False,
    image_scale_value:      Annotated[float, Field(description="The scaling value of the image.")] = numpy.nan,
    check_blur:             Annotated[bool, Field(description="Whether to calculate blur metrics.")] = False,
    blur_scales:            Annotated[Optional[Sequence[int]], Field(description="List of scales (window sizes) for blur metrics.")] = None,
    check_saturation:       Annotated[bool, Field(description="Whether to calculate saturation metrics.")] = False,
    check_intensity:        Annotated[bool, Field(description="Whether to calculate intensity metrics.")] = False,
    calculate_threshold:    Annotated[bool, Field(description="Whether to calculate threshold metrics.")] = False,
    threshold_groups:       Annotated[Optional[List[ThresholdConfig]], Field(description="List of threshold configurations.")] = None,
    return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, ImageQualityDisplayData]]:
    """
    Measure image quality metrics.

    Args:
        image: Input image.
        image_name: Name of the image.
        image_mask: Optional mask for the image.
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
    statistics: ImageQualityStatistics = []

    dimensions = image.ndim

    feat_stat_name_map = {
        TemplateMeasurementFormat.IQ_SCALING: f"{image_name} scaling",
        TemplateMeasurementFormat.IQ_POWER_SPECTRUM_SLOPE: f"{image_name} {Feature.POWER_SPECTRUM_SLOPE.value}",
        TemplateMeasurementFormat.IQ_PERCENT_MAXIMAL: f"{image_name} maximal",
        TemplateMeasurementFormat.IQ_PERCENT_MINIMAL: f"{image_name} minimal",
        TemplateMeasurementFormat.IQ_TOTAL_AREA: f"{image_name} Total Area",
        TemplateMeasurementFormat.IQ_TOTAL_VOLUME: f"{image_name} Total Volume",
        TemplateMeasurementFormat.IQ_TOTAL_INTENSITY: f"{image_name} Total intensity",
        TemplateMeasurementFormat.IQ_MEAN_INTENSITY: f"{image_name} Mean intensity",
        TemplateMeasurementFormat.IQ_MEDIAN_INTENSITY: f"{image_name} Median intensity",
        TemplateMeasurementFormat.IQ_STD_INTENSITY: f"{image_name} Std intensity",
        TemplateMeasurementFormat.IQ_MAD_INTENSITY: f"{image_name} MAD intensity",
        TemplateMeasurementFormat.IQ_MIN_INTENSITY: f"{image_name} Min intensity",
        TemplateMeasurementFormat.IQ_MAX_INTENSITY: f"{image_name} Max intensity",
        # template stat names
        TemplateMeasurementFormat.IQ_FOCUS_SCORE: f"{image_name} focus score @%d",
        TemplateMeasurementFormat.IQ_LOCAL_FOCUS_SCORE: f"{image_name} local focus score @%d",
        TemplateMeasurementFormat.IQ_CORR: f"{image_name} {Feature.CORRELATION.value} @%d",
        TemplateMeasurementFormat.IQ_THRESHOLD: f"{image_name} %s",
    }
    def add_measurement(
            feature_template: str,
            template_val: str,
            feat_val: ImageQualityMesVal,
            stat_template_val: Optional[Any]=None,
            feat_val_template: Optional[str]=None
        ):
        feature = feature_template % template_val
        measurements.add_image_measurement(feature, feat_val)
        if return_visualization_data:
            stat_name = feat_stat_name_map.get(feature_template, None)
            if stat_name is None:
                return # TODO: 5114 - handle this
            if stat_template_val is not None:
                stat_name = stat_name % stat_template_val
            if feat_val_template is not None:
                feat_val = feat_val_template % feat_val
            statistics.append((stat_name, feat_val))

    # Image Scalings
    if include_image_scalings:
        add_measurement(TemplateMeasurementFormat.IQ_SCALING, image_name, image_scale_value)

    # Blur Measurements
    if check_blur and blur_scales is not None:
        # Focus Score
        focus_score, scale, local_focus_score = get_focus_score_for_scale_group(
            blur_scales,
            image,
            dimensions,
            image_mask,
        )
        # Correlation
        scale_measurements = get_correlation_for_scale_group(image, blur_scales, image_mask)

        # Focus Score
        add_measurement(
            TemplateMeasurementFormat.IQ_FOCUS_SCORE,
            image_name,
            focus_score,
            stat_template_val=blur_scales[-1]
        )

        for idx, scale in enumerate(blur_scales):
            # Focus Score
            add_measurement(
                TemplateMeasurementFormat.IQ_LOCAL_FOCUS_SCORE,
                f"{image_name}_{scale}",
                local_focus_score[idx],
                stat_template_val=scale,
            )
            # Correlation
            add_measurement(
                TemplateMeasurementFormat.IQ_CORR,
                f"{image_name}_{scale}",
                scale_measurements[scale],
                stat_template_val=scale,
                feat_val_template="%.2f",
            )

        # Power Spectrum
        if dimensions == 2:
            powerslope = get_power_spectrum_measurement_values(image, image_mask, dimensions)
            add_measurement(
                TemplateMeasurementFormat.IQ_POWER_SPECTRUM_SLOPE,
                image_name,
                powerslope,
                feat_val_template="%.1f",
            )

    # Saturation Measurements
    if check_saturation:
        percent_minimal, percent_maximal = get_saturation_value(image, image_mask)

        add_measurement(
            TemplateMeasurementFormat.IQ_PERCENT_MAXIMAL,
            image_name,
            percent_maximal,
            feat_val_template="%.1f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_PERCENT_MINIMAL,
            image_name,
            percent_minimal,
            feat_val_template="%.1f",
        )

    # Intensity Measurements
    if check_intensity:
        (
            pixel_count,
            pixel_sum,
            pixel_mean,
            pixel_std,
            pixel_mad,
            pixel_median,
            pixel_min,
            pixel_max,
        ) = get_intensity_measurement_values(image, image_mask)

        add_measurement(
            TemplateMeasurementFormat.IQ_TOTAL_VOLUME if volumetric else TemplateMeasurementFormat.IQ_TOTAL_AREA,
            image_name,
            pixel_count,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_TOTAL_INTENSITY,
            image_name,
            pixel_sum,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_MEAN_INTENSITY,
            image_name,
            pixel_mean,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_MEDIAN_INTENSITY,
            image_name,
            pixel_median,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_STD_INTENSITY,
            image_name,
            pixel_std,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_MAD_INTENSITY,
            image_name,
            pixel_mad,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_MAX_INTENSITY,
            image_name,
            pixel_max,
            feat_val_template="%.2f",
        )
        add_measurement(
            TemplateMeasurementFormat.IQ_MIN_INTENSITY,
            image_name,
            pixel_min,
            feat_val_template="%.2f",
        )

    # Threshold Measurements
    if calculate_threshold and threshold_groups is not None:
        # We need to cast pixel data to float32 for threshold calculations as in original code
        # "pixel_data = image.pixel_data.astype(numpy.float32)"
        pixel_data_float = image.astype(numpy.float32)

        for config in threshold_groups:
            global_threshold = calculate_threshold_for_threshold_group(
                image_pixel_data=pixel_data_float,
                mask=image_mask,
                algorithm=config.algorithm,
                object_fraction=config.object_fraction,
                two_class_otsu=config.two_class_otsu,
                use_weighted_variance=config.use_weighted_variance,
                assign_middle_to_foreground=config.assign_middle_to_foreground,
            )

            scale = config.get_scale_string()
            algo = config.algorithm

            add_measurement(
                TemplateMeasurementFormat.IQ_THRESHOLD,
                f"{algo}_{image_name}_{scale}" if scale else f"{algo}_{image_name}",
                global_threshold,
                stat_template_val=f"{algo}" + (f" {scale}" if scale else "")
            )

    if return_visualization_data:
        return measurements, ImageQualityDisplayData(
            statistics=statistics,
        )
    return measurements
