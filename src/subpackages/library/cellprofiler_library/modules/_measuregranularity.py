from pydantic import validate_call, ConfigDict, Field
from typing import Annotated, List, Dict, Any, Tuple
from enum import Enum
import numpy as np

from cellprofiler_library.functions.measurement import get_granularity_measurements, ObjectRecord
from cellprofiler_library.functions.image_processing import apply_grayscale_tophat_filter, downsample_image_and_mask
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask

class GranularityMeasurementFormat(str, Enum):
    GRANULARITY = "Granularity_%d_%s"

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_granularity(
        image_name:                 Annotated[str, Field(description="Name of the image")],
        im_pixel_data:              Annotated[ImageGrayscale, Field(description="Pixel values of the image")],
        im_mask:                    Annotated[ImageGrayscaleMask, Field(description="Boolean mask of where the measurements should be made")],
        subsample_size:             Annotated[float, Field(description="Subsampling factor for granularity measurements")],
        image_sample_size:          Annotated[float, Field(description="Subsampling factor for background reduction")],
        element_size:               Annotated[int, Field(description="Radius of structuring element")],
        object_records:             Annotated[List[ObjectRecord], Field(description="Object records")],
        granular_spectrum_length:   Annotated[int, Field(description="Range of the granular spectrum")],
        dimensions:                 Annotated[int, Field(description="Dimensionality of the image")] = 2,
        ) -> Tuple[Dict[str, Any], List[str]]:
    #
    # Downsample the image and mask
    #
    pixels, mask, new_shape = downsample_image_and_mask(im_pixel_data, im_mask, dimensions, subsample_size)
    
    #
    # Remove background pixels using a greyscale tophat filter
    #
    pixels = apply_grayscale_tophat_filter(pixels, mask, dimensions, image_sample_size, element_size, new_shape)
    
    #
    # Compute measurements
    #
    measurements_arr, image_measurements_arr, stats_strings = get_granularity_measurements(
        im_pixel_data,
        pixels, 
        mask, 
        new_shape,
        granular_spectrum_length,
        dimensions,
        object_records
    )

    measurements = {
        "Image": {},
        "Object": {}
    }

    # Process Image Measurements
    for i, gs_value in enumerate(image_measurements_arr):
        spectrum_index = i + 1
        feature_name = GranularityMeasurementFormat.GRANULARITY % (spectrum_index, image_name)
        measurements["Image"][feature_name] = gs_value

    # Process Object Measurements
    # measurements_arr is list of lists. Outer list index i corresponds to spectrum_index i+1.
    for i, obj_measurements_list in enumerate(measurements_arr):
        spectrum_index = i + 1
        feature_name = GranularityMeasurementFormat.GRANULARITY % (spectrum_index, image_name)
        
        for obj_name, obj_gss in obj_measurements_list:
            if obj_name not in measurements["Object"]:
                measurements["Object"][obj_name] = {}
            
            measurements["Object"][obj_name][feature_name] = obj_gss
            
            # Calculate stats for this object measurement and add to Image measurements
            stat_name_base = f"{feature_name}_{obj_name}"
            
            measurements["Image"][f"Mean_{stat_name_base}"] = np.mean(obj_gss) if len(obj_gss) > 0 else 0.0
            measurements["Image"][f"Median_{stat_name_base}"] = np.median(obj_gss) if len(obj_gss) > 0 else 0.0
            measurements["Image"][f"Max_{stat_name_base}"] = np.max(obj_gss) if len(obj_gss) > 0 else 0.0
            measurements["Image"][f"Min_{stat_name_base}"] = np.min(obj_gss) if len(obj_gss) > 0 else 0.0
            measurements["Image"][f"StDev_{stat_name_base}"] = np.std(obj_gss) if len(obj_gss) > 0 else 0.0

    # Summary for UI display
    summary = [image_name] + stats_strings
    
    return measurements, summary
