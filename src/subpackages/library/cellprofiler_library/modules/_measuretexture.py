import numpy
from numpy.typing import NDArray
from typing import List, Union, Optional, Annotated, Tuple
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, ObjectLabel, ObjectSegmentation
from pydantic import validate_call, Field, ConfigDict, BaseModel
from cellprofiler_library.functions.measurement import measure_haralick_features_image, measure_haralick_features_objects
from cellprofiler_library.opts.measuretexture import F_HARALICK, TEXTURE
from cellprofiler_library.measurement_model import LibraryMeasurements


MeasureTextureStatistics = List[Tuple[
    str, # iamge name
    str, # object name
    str, # feature name
    str, # scale
    str  # stat val
]]

class MeasureTextureDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    statistics: MeasureTextureStatistics

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_texture(
        pixel_data: Annotated[ImageGrayscale, Field(description="Input image to perform texture measurements on")],
        gray_levels: Annotated[int, Field(description="Enter the number of gray levels (ie, total possible values of intensity) you want to measure texture at")], 
        scale: Annotated[int, Field(description="You can specify the scale of texture to be measured, in pixel units; the texture scale is the distance between correlated intensities in the image")], 
        image_name: Annotated[str, Field(description="Name to be assigned in measurements")],
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
    ) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, MeasureTextureDisplayData]]:
    
    haralick_feats = measure_haralick_features_image(pixel_data, gray_levels, scale)
    measurements = LibraryMeasurements()
    statistics: MeasureTextureStatistics = []
    
    for direction in range(haralick_feats.shape[0]):
        scale_str = "{:d}_{:02d}".format(scale, direction)
        gray_str = "{:d}".format(gray_levels)
        
        for feature_idx, feature_enum in enumerate(F_HARALICK):
            feature_name = feature_enum.value
            full_name = f"{TEXTURE}_{feature_name}_{image_name}_{scale_str}_{gray_str}"
            val = haralick_feats[direction, feature_idx]
            if not numpy.isfinite(val):
                val = 0.0
            measurements.add_image_measurement(full_name, val)

            if return_visualization_data:
                statistics.append((
                    image_name,
                    "-",
                    f"{feature_name}",
                    scale_str,
                    "{:.2f}".format(float(val))
                ))
            
    if return_visualization_data:
        return measurements, MeasureTextureDisplayData(statistics=statistics)
    else:
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
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
    ) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, MeasureTextureDisplayData]]:
    
    max_label = 0
    if len(unique_labels) > 0:
        max_label = int(numpy.max(unique_labels))
        
    haralick_feats = measure_haralick_features_objects(labels, pixel_data, mask, gray_levels, max_label, scale, volumetric)
    
    measurements = LibraryMeasurements()
    statistics: MeasureTextureStatistics = []

    if haralick_feats.size > 0:
        for direction in range(haralick_feats.shape[1]):
            scale_str = "{:d}_{:02d}".format(scale, direction)
            gray_str = "{:d}".format(gray_levels)
            
            for feature_idx, feature_enum in enumerate(F_HARALICK):
                feature_name = feature_enum.value
                full_name = f"{TEXTURE}_{feature_name}_{image_name}_{scale_str}_{gray_str}"
                
                values = haralick_feats[:, direction, feature_idx]
                
                # Calculate stats on valid (finite) values, mirroring original logic
                valid_values = values[numpy.isfinite(values)]
                
                # Clean values for storage (replace non-finite with 0), mirroring original frontend logic
                clean_values = values.copy()
                clean_values[~numpy.isfinite(clean_values)] = 0
                
                measurements.add_measurement(object_name, full_name, clean_values)
                
                stats = [
                    ("Mean", "mean", numpy.mean),
                    ("Median", "median", numpy.median),
                    ("Min", "min", numpy.min),
                    ("Max", "max", numpy.max),
                    ("StDev", "std dev", numpy.std),
                ]
                
                if len(valid_values) > 0:
                    for stat_name, display_name, func in stats:
                        stat_val = func(valid_values)
                        measurements.add_image_measurement(f"{stat_name}_{full_name}", stat_val)

                        if return_visualization_data:
                            statistics.append((
                                image_name,
                                object_name,
                                f"{display_name} {feature_name}",
                                scale_str,
                                f"{float(stat_val):.2f}" if len(clean_values) > 0 else "-",
                            ))
                else:
                    for stat_name, _, _ in stats:
                        measurements.add_image_measurement(f"{stat_name}_{full_name}", 0.0)
    else:
        # Handle empty objects case
        n_directions = 13 if volumetric else 4
    
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

    if return_visualization_data:
        return measurements, MeasureTextureDisplayData(statistics=statistics)
    else:
        return measurements
    
