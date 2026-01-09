import numpy
from typing import Tuple, Optional, Any, Dict, Union, Annotated, List
from numpy.typing import NDArray
from cellprofiler_library.types import Image2DBinary, ObjectSegmentation, Image2DColor, Image2DGrayscale
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import calculate_object_skeleton
from cellprofiler_library.opts.measureobjectskeleton import SkeletonMeasurements, C_OBJSKELETON

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_skeleton(
        object_name:                Annotated[str, Field(description="Name of the object being measured")],
        image_name:                 Annotated[str, Field(description="Name of the image being measured")],
        skeleton:                   Annotated[Image2DBinary, Field(description="Input image")],
        cropped_labels:             Annotated[ObjectSegmentation, Field(description="Input labels")],
        labels_count:               Annotated[numpy.integer[Any], Field(description="Number of labels in the original segmentation prior to cropping")],
        fill_small_holes:           Annotated[bool, Field(description="Fill small holes?")],
        max_hole_size:              Annotated[Optional[int], Field(description="Maximum hole size", ge=1)]=None,
        wants_objskeleton_graph:    Annotated[bool, Field(description="Return the skeleton graph relationships along with the measurements?")]=False,
        intensity_image_pixel_data: Annotated[Optional[Image2DGrayscale], Field(description="Intensity image used for graph relationships")]=None,
        wants_branchpoint_image:    Annotated[bool, Field(description="Return the branchpoint image?")]=False,
        ) -> Tuple[
            Dict[str, Any],
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Image2DColor],
        ]:
    
    (
        trunk_counts,
        branch_counts,
        end_counts,
        total_distance,
        edge_graph,
        vertex_graph,
        branchpoint_image
    ) = calculate_object_skeleton(
        skeleton, 
        cropped_labels, 
        labels_count, 
        fill_small_holes, 
        max_hole_size, 
        wants_objskeleton_graph, 
        intensity_image_pixel_data,
        wants_branchpoint_image
    )

    measurements = {
        "Image": {},
        "Object": {
            object_name: {}
        }
    }
    
    # Per-object measurements
    feature_map = {
        SkeletonMeasurements.NUMBER_TRUNKS: trunk_counts,
        SkeletonMeasurements.NUMBER_NON_TRUNK_BRANCHES: branch_counts,
        SkeletonMeasurements.NUMBER_BRANCH_ENDS: end_counts,
        SkeletonMeasurements.TOTAL_OBJSKELETON_LENGTH: total_distance,
    }

    for feature_name, values in feature_map.items():
        # Feature name format: ObjectSkeleton_Feature_Image
        full_feature_name = f"{C_OBJSKELETON}_{feature_name}_{image_name}"
        measurements["Object"][object_name][full_feature_name] = values
        
        # Calculate summary statistics
        if len(values) > 0:
            measurements["Image"][f"Mean_{full_feature_name}"] = numpy.mean(values)
            measurements["Image"][f"Median_{full_feature_name}"] = numpy.median(values)
            measurements["Image"][f"StDev_{full_feature_name}"] = numpy.std(values)
            measurements["Image"][f"Min_{full_feature_name}"] = numpy.min(values)
            measurements["Image"][f"Max_{full_feature_name}"] = numpy.max(values)
        else:
            measurements["Image"][f"Mean_{full_feature_name}"] = 0.0
            measurements["Image"][f"Median_{full_feature_name}"] = 0.0
            measurements["Image"][f"StDev_{full_feature_name}"] = 0.0
            measurements["Image"][f"Min_{full_feature_name}"] = 0.0
            measurements["Image"][f"Max_{full_feature_name}"] = 0.0

    return (
        measurements,
        edge_graph,
        vertex_graph,
        branchpoint_image
    )

