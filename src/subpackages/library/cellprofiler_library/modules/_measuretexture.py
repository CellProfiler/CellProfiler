import numpy
from numpy.typing import NDArray
from typing import List, Dict, Union, Optional, Annotated
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, ObjectLabel, ObjectSegmentation
from pydantic import validate_call, Field, ConfigDict
from cellprofiler_library.functions.measurement import measure_haralick_features_image, measure_haralick_features_objects
from cellprofiler_library.opts.measuretexture import F_HARALICK, TEXTURE
from cellprofiler_library.measurement_model import LibraryMeasurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_texture(
        pixel_data: Annotated[ImageGrayscale, Field(description="Input image to perform texture measurements on")],
        gray_levels: Annotated[int, Field(description="Enter the number of gray levels (ie, total possible values of intensity) you want to measure texture at")], 
        scale: Annotated[int, Field(description="You can specify the scale of texture to be measured, in pixel units; the texture scale is the distance between correlated intensities in the image")], 
        image_name: Annotated[str, Field(description="Name to be assigned in measurements")]
    ) -> LibraryMeasurements:
    
    raw_data = measure_haralick_features_image(pixel_data, gray_levels, scale)
    measurements = LibraryMeasurements()
    
    for direction in range(raw_data.shape[0]):
        scale_str = "{:d}_{:02d}".format(scale, direction)
        gray_str = "{:d}".format(gray_levels)
        
        for feature_idx, feature_enum in enumerate(F_HARALICK):
            feature_name = feature_enum.value
            full_name = f"{TEXTURE}_{feature_name}_{image_name}_{scale_str}_{gray_str}"
            val = raw_data[direction, feature_idx]
            if not numpy.isfinite(val):
                val = 0.0
            measurements.add_image_measurement(full_name, val)
            
    return measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_texture(
        object_name: Annotated[str, Field(description="Object name to be assigned in measurements")], 
        labels: Annotated[ObjectSegmentation, Field(description="Segmentation labels for object")],
        image_name: Annotated[str, Field(description="Name of the image to assign in measurements")],
        pixel_data: Annotated[ImageGrayscale, Field(description="Image pixel data to measure texture on")],
        mask: Annotated[Optional[ImageGrayscaleMask], Field(description="Image mask if any")],
        gray_levels: Annotated[int, Field(description="Enter the number of gray levels (ie, total possible values of intensity) you want to measure texture at")], 
        unique_labels: Annotated[NDArray[ObjectLabel], Field(description="The unique labels in the object segmentation prior to any masking")], # objects.indices
        scale: Annotated[int, Field(description="You can specify the scale of texture to be measured, in pixel units; the texture scale is the distance between correlated intensities in the image")], 
        volumetric: Annotated[bool, Field(description="Is the input image or objects 3D?")],
    ) -> LibraryMeasurements:
    
    max_label = 0
    if len(unique_labels) > 0:
        max_label = int(numpy.max(unique_labels))
        
    raw_data = measure_haralick_features_objects(labels, pixel_data, mask, gray_levels, max_label, scale, volumetric)
    
    measurements = LibraryMeasurements()
    
    n_directions = 13 if volumetric else 4
    
    if raw_data.size > 0:
        for direction in range(raw_data.shape[1]):
            scale_str = "{:d}_{:02d}".format(scale, direction)
            gray_str = "{:d}".format(gray_levels)
            
            for feature_idx, feature_enum in enumerate(F_HARALICK):
                feature_name = feature_enum.value
                full_name = f"{TEXTURE}_{feature_name}_{image_name}_{scale_str}_{gray_str}"
                
                values = raw_data[:, direction, feature_idx]
                
                # Calculate stats on valid (finite) values, mirroring original logic
                valid_values = values[numpy.isfinite(values)]
                
                # Clean values for storage (replace non-finite with 0), mirroring original frontend logic
                clean_values = values.copy()
                clean_values[~numpy.isfinite(clean_values)] = 0
                
                measurements.add_measurement(object_name, full_name, clean_values)
                
                stats_map = {
                    "Mean": numpy.mean,
                    "Median": numpy.median,
                    "Min": numpy.min,
                    "Max": numpy.max,
                    "StDev": numpy.std
                }
                
                if len(valid_values) > 0:
                    for stat_name, func in stats_map.items():
                        measurements.add_image_measurement(f"{stat_name}_{full_name}", func(valid_values))
                else:
                    for stat_name in stats_map.keys():
                        measurements.add_image_measurement(f"{stat_name}_{full_name}", 0.0)
    else:
        # Handle empty objects case
        for direction in range(n_directions):
            scale_str = "{:d}_{:02d}".format(scale, direction)
            gray_str = "{:d}".format(gray_levels)
            
            for feature_enum in F_HARALICK:
                feature_name = feature_enum.value
                full_name = f"{TEXTURE}_{feature_name}_{image_name}_{scale_str}_{gray_str}"
                
                measurements.add_measurement(object_name, full_name, numpy.zeros((0,)))
                
                stats_map_keys = ["Mean", "Median", "Min", "Max", "StDev"]
                for stat_name in stats_map_keys:
                    measurements.add_image_measurement(f"{stat_name}_{full_name}", 0.0)

    return measurements
    