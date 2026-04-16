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

    def add_measurement(feature_name: str, fmt_template: str, feature_value: Union[int, float]):
        measurements.add_image_measurement(fmt_template % measurement_name, feature_value)

        statistics.append([
            image_name,
            object_name if object_name else "",
            feature_name,
            str(feature_value),
        ])

    # Add measurements
    add_measurement(FORMATED_FEATURE_NAMES[Feature.TOTAL_INTENSITY.value], TemplateMeasurementFormat.TOTAL_INTENSITY, pixel_sum)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.MEAN_INTENSITY.value], TemplateMeasurementFormat.MEAN_INTENSITY, pixel_mean)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.MEDIAN_INTENSITY.value], TemplateMeasurementFormat.MEDIAN_INTENSITY, pixel_median)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.STD_INTENSITY.value], TemplateMeasurementFormat.STD_INTENSITY, pixel_std)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.MAD_INTENSITY.value], TemplateMeasurementFormat.MAD_INTENSITY, pixel_mad)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.MAX_INTENSITY.value], TemplateMeasurementFormat.MAX_INTENSITY, pixel_max)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.MIN_INTENSITY.value], TemplateMeasurementFormat.MIN_INTENSITY, pixel_min)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.TOTAL_AREA.value], TemplateMeasurementFormat.TOTAL_AREA, pixel_count)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.PERCENT_MAXIMAL.value], TemplateMeasurementFormat.PERCENT_MAXIMAL, pixel_pct_max)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.LOWER_QUARTILE.value], TemplateMeasurementFormat.LOWER_QUARTILE, pixel_lower_qrt)
    add_measurement(FORMATED_FEATURE_NAMES[Feature.UPPER_QUARTILE.value], TemplateMeasurementFormat.UPPER_QUARTILE, pixel_upper_qrt)
    
    percentile_stats: List[Tuple[int,float]] = []
    for percentile, value in percentile_measures.items():
        key = TemplateMeasurementFormat.PERCENTILE % (percentile, measurement_name)
        measurements.add_image_measurement(key, value)
        percentile_stats.append((percentile, value))

    percentile_stats.sort(key = lambda p: p[0])
    statistics += [(FORMATED_PERCENTILE_TEMPLATE % p[0], p[1]) for p in percentile_stats]

    return measurements, statistics
