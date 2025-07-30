# coding=utf-8

"""
ErodeObjects module for the CellProfiler library.

This module contains the core algorithms for object erosion operations.
"""

from pydantic import validate_call, ConfigDict
import numpy as np
import numpy.typing as npt

from ..types import ImageAny
from ..opts.erodeobjects import PreserveMidpoints, RelabelObjects
from ..functions.object_processing import erode_objects_with_structuring_element


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def erode_objects(
    labels: npt.NDArray,
    structuring_element: np.ndarray,
    preserve_midpoints: PreserveMidpoints = PreserveMidpoints.PREVENT_REMOVAL,
    relabel_objects: RelabelObjects = RelabelObjects.KEEP_ORIGINAL
) -> npt.NDArray:
    """Erode objects based on the structuring element provided.
    
    This function is similar to the "Shrink" function of ExpandOrShrinkObjects,
    with two major distinctions:
    1. ErodeObjects supports 3D objects, unlike ExpandOrShrinkObjects.
    2. An object smaller than the structuring element will be removed entirely
       unless preserve_midpoints is enabled.
    
    Args:
        labels: Input labeled objects array
        structuring_element: Structuring element for erosion operation
        preserve_midpoints: Whether to prevent object removal by preserving midpoints
        relabel_objects: Whether to assign new label numbers to resulting objects
        
    Returns:
        Eroded objects array with same dimensions as input
    """
    preserve_flag = preserve_midpoints == PreserveMidpoints.PREVENT_REMOVAL
    relabel_flag = relabel_objects == RelabelObjects.RELABEL
    
    return erode_objects_with_structuring_element(
        labels=labels,
        structuring_element=structuring_element,
        preserve_midpoints=preserve_flag,
        relabel_objects=relabel_flag
    )
