# coding=utf-8

"""
ErodeObjects module for the CellProfiler library.

This module contains the core algorithms for object erosion operations.
"""

from pydantic import validate_call, ConfigDict, Field
from typing import Union, Tuple, Annotated
from cellprofiler_library.types import StructuringElement, ObjectSegmentation
from cellprofiler_library.functions.object_processing import erode_objects_with_structuring_element
from cellprofiler_library.functions.image_processing import get_structuring_element
from cellprofiler_library.opts.structuring_elements import StructuringElementShape2D, StructuringElementShape3D

StructuringElementSize = Annotated[int, Field(description="Size of structuring element", gt=0)]
StructuringElementParameters = Tuple[Union[StructuringElementShape2D, StructuringElementShape3D], StructuringElementSize]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def erode_objects(
    labels: Annotated[ObjectSegmentation, Field(description="Input object segmentations")],
    structuring_element: Annotated[Union[StructuringElement, StructuringElementParameters], Field(description="Structuring element for erosion operation as either an NDArray or a tuple of (StructuringElement[N]D, size)")],
    preserve_midpoints: Annotated[bool, Field(description="If set to True, the central pixels for each object will not be eroded. This ensures that objects are not lost.")] = False,
    relabel_objects: Annotated[bool, Field(description="Selecting True will assign new label numbers to resulting objects")] = False
) -> ObjectSegmentation:
    """Erode objects based on the structuring element provided.
    
    This function is similar to the "Shrink" function of ExpandOrShrinkObjects,
    with two major distinctions:
    1. ErodeObjects supports 3D objects, unlike ExpandOrShrinkObjects.
    2. An object smaller than the structuring element will be removed entirely
       unless preserve_midpoints is enabled.
    
    Args:
        labels: Input labeled objects array
        structuring_element: Structuring element for erosion operation
        preserve_midpoints: If set to True, the central pixels for each object will not be eroded. This ensures that objects are not lost.
        relabel_objects: If set to True, the resulting objects will be relabeled with new label numbers
        
    Returns:
        Eroded objects array with same dimensions as input
    """
    if isinstance(structuring_element, tuple):
        structuring_element = get_structuring_element(structuring_element[0], structuring_element[1])
    return erode_objects_with_structuring_element(
        labels=labels,
        structuring_element=structuring_element,
        preserve_midpoints=preserve_midpoints,
        relabel_objects=relabel_objects
    )
