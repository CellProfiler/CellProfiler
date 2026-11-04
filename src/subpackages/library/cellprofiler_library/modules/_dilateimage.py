# coding=utf-8

"""
DilateImage module for the CellProfiler library.

This module contains the core algorithms for morphological dilation operations.
"""

from pydantic import validate_call, ConfigDict, Field
from typing import Union, Tuple, Annotated
from cellprofiler_library.types import ImageAny, StructuringElement
from cellprofiler_library.functions.image_processing import morphology_dilation, get_structuring_element
from cellprofiler_library.opts.structuring_elements import StructuringElementShape2D, StructuringElementShape3D

StructuringElementSize = Annotated[int, Field(description="Size of structuring element", gt=0)]
StructuringElementParameters = Tuple[Union[StructuringElementShape2D, StructuringElementShape3D], StructuringElementSize]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dilate_image(
    image: Annotated[ImageAny, Field(description="Input image to perform dilation on")],
    structuring_element: Annotated[Union[StructuringElement, StructuringElementParameters], Field(description="Structuring element for dilation operation as either an NDArray or a tuple of (StructuringElement[N]D, size)")]
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
