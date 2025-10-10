# coding=utf-8

"""
DilateImage module for the CellProfiler library.

This module contains the core algorithms for morphological dilation operations.
"""

from pydantic import validate_call, ConfigDict
from typing import Union, Tuple
from cellprofiler_library.types import ImageAny, StructuringElement
from cellprofiler_library.functions.image_processing import morphology_dilation, get_structuring_element
from cellprofiler_library.opts.structuring_elements import StructuringElementShape2D, StructuringElementShape3D


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dilate_image(
    image: ImageAny,
    structuring_element: Union[StructuringElement, Tuple[Union[StructuringElementShape2D, StructuringElementShape3D], int]]
) -> ImageAny:
    """Apply morphological dilation to an image.
    
    Args:
        image: Input image (2D or 3D grayscale)
        structuring_element: Structuring element for dilation operation as an NDArray or a tuple of (StructuringElement[N]D, size)
        
    Returns:
        Dilated image with same dimensions and type as input
        
    Raises:
        NotImplementedError: If trying to apply 3D structuring element to 2D image
    """
    if isinstance(structuring_element, tuple):
        structuring_element = get_structuring_element(structuring_element[0], structuring_element[1])
    return morphology_dilation(image, structuring_element)
