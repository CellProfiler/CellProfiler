import numpy
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Optional, Tuple, Dict, Callable
from numpy.typing import NDArray
from cellprofiler_library.functions.image_processing import flip_image_both, flip_image_left_to_right, flip_image_top_to_bottom, rotate_image_angle, rotate_image_coordinates
from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.opts.flipandrotate import FlipDirection, RotateMethod, RotationCycle, RotationCoordinateAlignmnet

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def flip_and_rotate(
        pixel_data:                 Annotated[Image2D, Field(description="Pixel data of image to be flipped and/or rotated")],
        mask:                       Annotated[Image2DMask, Field(description="Mask of the image to be flipped and/or rotated")],
        flip_choice:                Annotated[FlipDirection, Field(description="Direction to flip the image")],
        rotate_choice:              Annotated[RotateMethod, Field(description="Rotation method")],
        rotate_angle:               Annotated[Optional[float], Field(description="Angle to rotate the image")],
        rotate_point_1:             Annotated[Optional[Tuple[float, float]], Field(description="Point to rotate the image around")],
        rotate_point_2:             Annotated[Optional[Tuple[float, float]], Field(description="Second point to rotate the image around")],
        rotate_coordinate_alignment: Annotated[Optional[RotationCoordinateAlignmnet], Field(description="Alignment of the rotated points. The points you select will be aligned horizontally or vertically after the rotation is complete.")],
        wants_crop:                 Annotated[bool, Field(description="Whether to crop the image")],
        ) -> Tuple[Image2D, Image2DMask, Optional[Image2DMask], float]:
    #
    # Perform flip
    #
    pixel_data, mask = flip_image(pixel_data, mask, flip_choice) 
    
    #
    # Perform rotation
    #
    pixel_data, mask, crop, angle = rotate_image(pixel_data, mask, rotate_choice, rotate_angle, rotate_point_1, rotate_point_2, rotate_coordinate_alignment, wants_crop)
    
    return pixel_data, mask, crop, angle


def flip_image(
        pixel_data: Image2D, 
        mask: Image2DMask, 
        flip_choice: FlipDirection
        ) -> Tuple[Image2D, Image2DMask]:
    flip_dispatch: Dict[FlipDirection, Callable[[Image2D],  NDArray[numpy.int_]]] = {
        FlipDirection.LEFT_TO_RIGHT: flip_image_left_to_right,
        FlipDirection.TOP_TO_BOTTOM: flip_image_top_to_bottom,
        FlipDirection.BOTH: flip_image_both,
    }
    if flip_choice != FlipDirection.NONE:
        if flip_choice in flip_dispatch.keys():
            i, j = flip_dispatch[flip_choice](pixel_data)
        else:
            raise NotImplementedError(
                "Unknown flipping operation: %s" % flip_choice.value
            )
        mask = mask[i, j]
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[i, j]
        else:
            pixel_data = pixel_data[i, j, :]
    return pixel_data, mask


def rotate_image(
        pixel_data: Image2D, 
        mask: Image2DMask, 
        rotate_choice: RotateMethod, 
        rotate_angle: Optional[float],
        rotate_point_1: Optional[Tuple[float, float]],
        rotate_point_2: Optional[Tuple[float, float]],
        rotate_coordinate_alignment: Optional[RotationCoordinateAlignmnet],
        wants_crop: bool,
        ) -> Tuple[Image2D, Image2DMask, Optional[Image2DMask], float]:
    if rotate_choice != RotateMethod.NONE:
        if rotate_choice == RotateMethod.ANGLE:
            assert rotate_angle is not None, "rotate_angle must be provided for rotate_choice == RotateMethod.ANGLE"
            angle = rotate_angle
        elif rotate_choice == RotateMethod.COORDINATES:
            assert rotate_point_1 is not None, "rotate_point_1 must be provided for rotate_choice == RotateMethod.COORDINATES"
            assert rotate_point_2 is not None, "rotate_point_2 must be provided for rotate_choice == RotateMethod.COORDINATES"
            assert rotate_coordinate_alignment is not None, "rotate_coordinate_alignment must be provided for rotate_choice == RotateMethod.COORDINATES"
            angle = rotate_image_coordinates(pixel_data, mask, rotate_point_1, rotate_point_2, rotate_coordinate_alignment)

        else:
            raise NotImplementedError(
                "Unknown rotation method: %s" % rotate_choice.value
            )
        # rangle = angle * numpy.pi / 180.0
        pixel_data, mask, crop = rotate_image_angle(pixel_data, mask, angle, wants_crop)

    else:
        crop = None
        angle = 0.0
    return pixel_data, mask, crop, angle