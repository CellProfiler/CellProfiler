import numpy as np
from numpy.typing import NDArray
from typing import List, Annotated, Optional, Tuple, Union
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import measure_image_intensities
from cellprofiler_library.opts.measureimageintensity import TemplateMeasurementFormat, Feature, FORMATED_FEATURE_NAMES, FORMATED_PERCENTILE_TEMPLATE 
from cellprofiler_library.measurement_model import LibraryMeasurements

IntensityStatistics = List[Union[List[str], Tuple[str,float]]]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_intensity(
        pixels:         Annotated[NDArray[np.float32], Field(description="Image pixel data")],
        image_name:     Annotated[str, Field(description="Name of the image")],
        object_name:    Annotated[Optional[str], Field(description="Name of the object set (if any)")] = None,
        percentiles:    Annotated[Optional[List[int]], Field(description="Percentiles to measure")]=[],
        ) -> Tuple[LibraryMeasurements, IntensityStatistics]:
    
    if percentiles is None:
        percentiles = []
        
    # Construct measurement name suffix
    measurement_name = image_name
    if object_name:
        measurement_name += "_" + object_name
        
    (
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
    ), percentile_measures = measure_image_intensities(pixels, percentiles)

    measurements = LibraryMeasurements()
    statistics: IntensityStatistics = []

    def add_statistic(feature_name: str, feature_value: Union[int, float]):
        statistics.append([
            image_name,
            object_name if object_name else "",
            feature_name,
            str(feature_value),
        ])

    # Add measurements
    measurements.add_image_measurement(TemplateMeasurementFormat.TOTAL_INTENSITY % measurement_name, pixel_sum)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.TOTAL_INTENSITY.value], pixel_sum)
    measurements.add_image_measurement(TemplateMeasurementFormat.MEAN_INTENSITY % measurement_name, pixel_mean)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.MEAN_INTENSITY.value], pixel_mean)
    measurements.add_image_measurement(TemplateMeasurementFormat.MEDIAN_INTENSITY % measurement_name, pixel_median)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.MEDIAN_INTENSITY.value], pixel_median)
    measurements.add_image_measurement(TemplateMeasurementFormat.STD_INTENSITY % measurement_name, pixel_std)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.STD_INTENSITY.value], pixel_std)
    measurements.add_image_measurement(TemplateMeasurementFormat.MAD_INTENSITY % measurement_name, pixel_mad)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.MAD_INTENSITY.value], pixel_mad)
    measurements.add_image_measurement(TemplateMeasurementFormat.MAX_INTENSITY % measurement_name, pixel_max)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.MAX_INTENSITY.value], pixel_max)
    measurements.add_image_measurement(TemplateMeasurementFormat.MIN_INTENSITY % measurement_name, pixel_min)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.MIN_INTENSITY.value], pixel_min)
    measurements.add_image_measurement(TemplateMeasurementFormat.TOTAL_AREA % measurement_name, pixel_count)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.TOTAL_AREA.value], pixel_count)
    measurements.add_image_measurement(TemplateMeasurementFormat.PERCENT_MAXIMAL % measurement_name, pixel_pct_max)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.PERCENT_MAXIMAL.value], pixel_pct_max)
    measurements.add_image_measurement(TemplateMeasurementFormat.LOWER_QUARTILE % measurement_name, pixel_lower_qrt)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.LOWER_QUARTILE.value], pixel_lower_qrt)
    measurements.add_image_measurement(TemplateMeasurementFormat.UPPER_QUARTILE % measurement_name, pixel_upper_qrt)
    add_statistic(FORMATED_FEATURE_NAMES[Feature.UPPER_QUARTILE.value], pixel_upper_qrt)
    
    percentile_stats: List[Tuple[int,float]] = []
    for percentile, value in percentile_measures.items():
        key = TemplateMeasurementFormat.PERCENTILE % (percentile, measurement_name)
        measurements.add_image_measurement(key, value)
        percentile_stats.append((percentile, value))

    percentile_stats.sort(key = lambda p: p[0])
    statistics += [(FORMATED_PERCENTILE_TEMPLATE % p[0], p[1]) for p in percentile_stats]

    return measurements, statistics
