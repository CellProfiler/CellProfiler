# coding=utf-8

"""
DilateObjects module for the CellProfiler library.

This module contains the core algorithms for object dilation operations.
"""

from pydantic import validate_call, ConfigDict, Field
from typing import Union, Tuple, Annotated
from cellprofiler_library.types import StructuringElement, ObjectSegmentation
from cellprofiler_library.functions.object_processing import dilate_objects_with_structuring_element
from cellprofiler_library.functions.image_processing import get_structuring_element
from cellprofiler_library.opts.structuring_elements import StructuringElementShape2D, StructuringElementShape3D

StructuringElementSize = Annotated[int, Field(description="Size of structuring element", gt=0)]
StructuringElementParameters = Tuple[Union[StructuringElementShape2D, StructuringElementShape3D], StructuringElementSize]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dilate_objects(
    labels: Annotated[ObjectSegmentation, Field(description="Input object segmentations")],
    structuring_element: Annotated[Union[StructuringElement, StructuringElementParameters], Field(description="Structuring element for dilation operation as either an NDArray or a tuple of (StructuringElement[N]D, size)")],
) -> ObjectSegmentation:
    """Dilate objects based on the structuring element provided.
    
    This function is similar to the "Expand" function of ExpandOrShrinkObjects,
    with two major distinctions:
    1. DilateObjects supports 3D objects, unlike ExpandOrShrinkObjects.
    2. In ExpandOrShrinkObjects, two objects closer than the expansion distance
       will expand until they meet and then stop there. In this module, the object with
       the larger object number (the object that is lower in the image) will be expanded
       on top of the object with the smaller object number.
    
    Args:
        labels: Input labeled objects array
        structuring_element: Structuring element for dilation operation
        
    Returns:
        Dilated objects array with same dimensions as input
    """
    if isinstance(structuring_element, tuple):
        structuring_element = get_structuring_element(structuring_element[0], structuring_element[1])
    return dilate_objects_with_structuring_element(
        labels=labels,
        structuring_element=structuring_element
    )
