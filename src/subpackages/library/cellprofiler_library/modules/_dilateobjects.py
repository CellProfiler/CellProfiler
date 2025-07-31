# coding=utf-8

"""
DilateObjects module for the CellProfiler library.

This module contains the core algorithms for object dilation operations.
"""

from pydantic import validate_call, ConfigDict
import numpy as np
import numpy.typing as npt

from ..types import ImageAny
from ..functions.object_processing import dilate_objects_with_structuring_element


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dilate_objects(
    labels: npt.NDArray,
    structuring_element: np.ndarray
) -> npt.NDArray:
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
    return dilate_objects_with_structuring_element(
        labels=labels,
        structuring_element=structuring_element
    )
