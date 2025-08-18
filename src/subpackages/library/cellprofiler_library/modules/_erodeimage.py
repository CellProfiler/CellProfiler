# coding=utf-8

"""
ErodeImage module for the CellProfiler library.

This module contains the core algorithms for morphological erosion operations.
"""

from pydantic import validate_call, ConfigDict
from typing import Union
import numpy as np

from ..types import Image2D, Image2DGrayscale, Image3DGrayscale, ImageGrayscale
from ..functions.image_processing import morphology_erosion


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def erode_image(
    image: ImageGrayscale,
    structuring_element: np.ndarray
) -> ImageGrayscale:
    """Apply morphological erosion to an image.
    
    Args:
        image: Input image (2D or 3D grayscale)
        structuring_element: Structuring element for erosion operation
        
    Returns:
        Eroded image with same dimensions and type as input
        
    Raises:
        NotImplementedError: If trying to apply 3D structuring element to 2D image
    """
    return morphology_erosion(image, structuring_element)
