from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Optional, Tuple, Dict, Callable

from cellprofiler_library.functions.image_processing import flip_image, rotate_image
from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.opts.flipandrotate import FlipDirection, RotateDirection, RotationCycle, RotationCoordinateAlignmnet

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def flip_and_rotate(
        pixel_data:                 Annotated[Image2D, Field(description="Pixel data of image to be flipped and/or rotated")],
        mask:                       Annotated[Image2DMask, Field(description="Mask of the image to be flipped and/or rotated")],
        flip_choice:                Annotated[FlipDirection, Field(description="Direction to flip the image")],
        rotate_choice:              Annotated[RotateDirection, Field(description="Rotation method")],
        rotate_angle:               Annotated[Optional[float], Field(description="Angle to rotate the image")],
        rotate_point_1:             Annotated[Optional[Tuple[float, float]], Field(description="Point to rotate the image around")],
        rotate_point_2:             Annotated[Optional[Tuple[float, float]], Field(description="Second point to rotate the image around")],
        rotate_coordinate_alignment: Annotated[Optional[RotationCoordinateAlignmnet], Field(description="Alignment of the rotated points. The points you select will be aligned horizontally or vertically after the rotation is complete.")],
        state_dict_for_mouse_mode:  Annotated[Optional[Dict[str, Optional[float]]], Field(description="State dictionary for mouse mode")],
        mouse_mode_cycle:           Annotated[Optional[RotationCycle], Field(description="Mouse mode cycle")],
        mouse_interaction_handler:  Annotated[Optional[Callable[[Image2D], float]], Field(description="Mouse interaction handler")],
        wants_crop:                 Annotated[bool, Field(description="Whether to crop the image")],
        ) -> Tuple[Image2D, Image2DMask, Optional[Image2DMask], float]:
    #
    # Perform flip
    #
    pixel_data, mask = flip_image(pixel_data, mask, flip_choice) 
    
    #
    # Perform rotation
    #
    pixel_data, mask, crop, angle = rotate_image(pixel_data, mask, rotate_choice, rotate_angle, rotate_point_1, rotate_point_2, rotate_coordinate_alignment, state_dict_for_mouse_mode, mouse_mode_cycle, mouse_interaction_handler, wants_crop)
    
    return pixel_data, mask, crop, angle
