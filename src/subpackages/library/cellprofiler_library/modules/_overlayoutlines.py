from typing import Annotated, Optional, Tuple, List, Union
from pydantic import Field, validate_call, ConfigDict
from numpy.typing import NDArray
import numpy as np

from ..types import ImageAny, ObjectLabelSet, ImageColor
from ..opts.overlayoutlines import BrightnessMode, LineMode
from ..functions.image_processing import (
    create_overlay_base_image,
    overlay_outlines_grayscale,
    overlay_outlines_color,
)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def overlay_outlines(
    brightness_mode:    Annotated[BrightnessMode, Field(description="Brightness control mode for grayscale outline processing", default=BrightnessMode.MAX_POSSIBLE)], 
    line_mode:          Annotated[LineMode, Field(description="Line style mode for drawing outlines", default=LineMode.INNER)],
    obj_shape:          Annotated[Optional[Tuple[int, ...]], Field(description="Object dimensions for creating blank base image when no input image provided", default=None)], 
    obj_dimensions:     Annotated[Optional[int], Field(description="Number of dimensions for objects (2D or 3D) when creating blank image", default=None)], 
    im_pixel_data:      Annotated[Optional[ImageAny], Field(description="Input image pixel data to overlay outlines on (None to create blank image)", default=None)], 
    im_multichannel:    Annotated[bool, Field(description="Whether input image has multiple color channels (False if overlaying on blank image)", default=False)], 
    im_dimensions:      Annotated[Optional[int], Field(description="Number of spatial dimensions in input image (2D or 3D), None if overlaying on blank image", default=None)],
    object_labels_list: Annotated[List[ObjectLabelSet], Field(description="List of object label sets containing segmented object data for outline generation", default_factory=list)],
    colors_list:        Annotated[Optional[List[Tuple[int, int, int]]], Field(description="List of RGB color tuples (0-255) for colored outline mode (None for grayscale)", default=None)],
    is_volumetric:      Annotated[bool, Field(description="Whether objects are 3D volumetric data requiring plane-wise processing", default=False)]
) -> Tuple[ImageAny, ImageColor, Optional[int]]:
    """Dispatcher function that centralizes all overlay outlines business logic.
    
    This function orchestrates the overlay outlines workflow by creating a base image
    (either blank or from input) and then applying either color or grayscale outline
    processing based on the presence of a colors list.
    
    Args:
        brightness_mode: Brightness control mode determining outline intensity in grayscale mode
        line_mode: Line style mode controlling how outlines are drawn (inner, outer, thick)
        obj_shape: Object dimensions for creating blank base image when no input provided
        obj_dimensions: Spatial dimensionality (2 or 3) for objects when creating blank image
        im_pixel_data: Input image data to overlay outlines on (None creates blank image)
        im_multichannel: Whether input image contains multiple color channels (False if overlaying on blank image)
        im_dimensions: Spatial dimensionality (2 or 3) of input image, None if overlaying on blank image
        object_labels_list: Object label sets containing segmented regions for outline generation
        colors_list: RGB color values for colored outlines (None triggers grayscale mode)
        is_volumetric: Whether objects require 3D plane-wise processing
        
    Returns:
        Tuple of (processed_pixel_data, base_image, dimensions) for the final result
    """
    # Create base image (blank or from input)
    base_image, dimensions = create_overlay_base_image(
        obj_shape, obj_dimensions, 
        im_pixel_data, im_multichannel, im_dimensions
    )
    
    if colors_list is not None:
        # Color outline processing
        pixel_data = overlay_outlines_color(base_image.copy(), object_labels_list, colors_list, line_mode, is_volumetric)
    else:
        # Black and white outline processing
        pixel_data = overlay_outlines_grayscale(
            base_image, brightness_mode, 
            object_labels_list, line_mode, is_volumetric
        )
        
    return pixel_data, base_image, dimensions
