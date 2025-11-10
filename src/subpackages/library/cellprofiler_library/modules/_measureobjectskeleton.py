import numpy
from typing import Tuple, Optional, Any, Dict, Union, Annotated
from numpy.typing import NDArray
from cellprofiler_library.types import Image2DBinary, ObjectSegmentation, Image2DColor, Image2DGrayscale
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import calculate_object_skeleton

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_skeleton(
        skeleton:                   Annotated[Image2DBinary, Field(description="Input image")],
        cropped_labels:             Annotated[ObjectSegmentation, Field(description="Input labels")],
        labels_count:               Annotated[numpy.integer[Any], Field(description="Number of labels in the original segmentation prior to cropping")],
        fill_small_holes:           Annotated[bool, Field(description="Fill small holes?")],
        max_hole_size:              Annotated[Optional[int], Field(description="Maximum hole size", ge=1)]=None,
        wants_objskeleton_graph:    Annotated[bool, Field(description="Return the skeleton graph relationships along with the measurements?")]=False,
        intensity_image_pixel_data: Annotated[Optional[Image2DGrayscale], Field(description="Intensity image used for graph relationships")]=None,
        wants_branchpoint_image:    Annotated[bool, Field(description="Return the branchpoint image?")]=False,
        ) -> Tuple[
            NDArray[numpy.float_],
            NDArray[numpy.float_],
            NDArray[numpy.float_],
            NDArray[numpy.float_],
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
        ) =  calculate_object_skeleton(
        skeleton, 
        cropped_labels, 
        labels_count, 
        fill_small_holes, 
        max_hole_size, 
        wants_objskeleton_graph, 
        intensity_image_pixel_data,
        wants_branchpoint_image
        )
        return (
            trunk_counts,
            branch_counts,
            end_counts,
            total_distance,
            edge_graph,
            vertex_graph,
            branchpoint_image
        )

