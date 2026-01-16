import numpy as np
from numpy.typing import NDArray
from typing import List, Annotated, Optional
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import measure_image_intensities
from cellprofiler_library.opts.measureimageintensity import IntensityMeasurementFormat
from cellprofiler_library.measurement_model import LibraryMeasurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_intensity(
        pixels:         Annotated[NDArray[np.float32], Field(description="Image pixel data")],
        image_name:     Annotated[str, Field(description="Name of the image")],
        object_name:    Annotated[Optional[str], Field(description="Name of the object set (if any)")] = None,
        percentiles:    Annotated[Optional[List[int]], Field(description="Percentiles to measure")]=[],
        ) -> LibraryMeasurements:
    
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
    
    # Add measurements
    measurements.add_image_measurement(IntensityMeasurementFormat.TOTAL_INTENSITY % measurement_name, pixel_sum)
    measurements.add_image_measurement(IntensityMeasurementFormat.MEAN_INTENSITY % measurement_name, pixel_mean)
    measurements.add_image_measurement(IntensityMeasurementFormat.MEDIAN_INTENSITY % measurement_name, pixel_median)
    measurements.add_image_measurement(IntensityMeasurementFormat.STD_INTENSITY % measurement_name, pixel_std)
    measurements.add_image_measurement(IntensityMeasurementFormat.MAD_INTENSITY % measurement_name, pixel_mad)
    measurements.add_image_measurement(IntensityMeasurementFormat.MAX_INTENSITY % measurement_name, pixel_max)
    measurements.add_image_measurement(IntensityMeasurementFormat.MIN_INTENSITY % measurement_name, pixel_min)
    measurements.add_image_measurement(IntensityMeasurementFormat.TOTAL_AREA % measurement_name, pixel_count)
    measurements.add_image_measurement(IntensityMeasurementFormat.PERCENT_MAXIMAL % measurement_name, pixel_pct_max)
    measurements.add_image_measurement(IntensityMeasurementFormat.LOWER_QUARTILE % measurement_name, pixel_lower_qrt)
    measurements.add_image_measurement(IntensityMeasurementFormat.UPPER_QUARTILE % measurement_name, pixel_upper_qrt)
    
    for percentile, value in percentile_measures.items():
        measurements.add_image_measurement(f"Intensity_Percentile_{percentile}_{measurement_name}", value)
        
    return measurements
