"""Resize module core functionality.

This module contains the extracted resize functions that implement
the core image resizing algorithms.
"""

from typing import Annotated, Optional, Union, Any
from pydantic import Field, validate_call, ConfigDict
from numpy.typing import NDArray
import numpy

from ..types import ImageGrayscale, ImageGrayscaleMask, ImageAny
from ..functions.image_processing import resized_shape, spline_order, apply_resize
from ..opts.resize import ResizingMethod, DimensionMethod, InterpolationMethod


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def resize_image(
    im_pixel_data:          Annotated[ImageAny, Field(description="Input image pixel data array")],
    im_mask:                Annotated[ImageGrayscaleMask, Field(description="Input image mask")],
    im_dimensions:          Annotated[int, Field(description="Number of spatial dimensions in the image", ge=2, le=3)],
    im_crop_mask:           Annotated[Optional[NDArray[numpy.bool_]], Field(description="Image crop mask")],
    size_method:            Annotated[ResizingMethod, Field(description="Method for determining resize dimensions")],
    resizing_factor_x:      Annotated[float, Field(description="X-axis scaling factor", gt=0)],
    resizing_factor_y:      Annotated[float, Field(description="Y-axis scaling factor", gt=0)],
    resizing_factor_z:      Annotated[Optional[float], Field(description="Z-axis scaling factor", gt=0)],
    use_manual_or_image:    Annotated[DimensionMethod, Field(description="Method for specifying dimensions")],
    specific_width:         Annotated[Optional[int], Field(description="Specific width in pixels", ge=1)],
    specific_height:        Annotated[Optional[int], Field(description="Specific height in pixels", ge=1)],
    specific_planes:        Annotated[Optional[int], Field(description="Specific number of planes", ge=1)],
    reference_image_shape:  Annotated[Optional[tuple], Field(description="Shape of reference image for dimensions")],
    interpolation_method:   Annotated[InterpolationMethod, Field(description="Interpolation method for resizing")],
) -> tuple[ImageAny, ImageGrayscaleMask, Optional[NDArray[numpy.bool_]]]:
    """High-level dispatcher function for unified resize API.
    
    Centralizes all resize functionality with flattened parameters,
    providing a single entry point for all resize operations.
    """
    # Apply the resize operation using the library function with enum integration
    return apply_resize(
        im_pixel_data,
        im_mask,
        im_dimensions,
        im_crop_mask,
        size_method,
        resizing_factor_x,
        resizing_factor_y,
        resizing_factor_z,
        use_manual_or_image,
        specific_width,
        specific_height,
        specific_planes,
        reference_image_shape,
        interpolation_method,
    )


