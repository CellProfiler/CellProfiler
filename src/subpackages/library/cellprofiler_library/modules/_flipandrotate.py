import numpy
import scipy.ndimage
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Optional, Tuple, Dict, Callable

from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.opts.flipandrotate import FlipDirection, RotateDirection, RotationCycle, RotationCoordinateAlignmnet, D_ANGLE #, M_ROTATION_CATEGORY, M_ROTATION_F, FLIP_ALL, ROTATE_ALL, IO_ALL, C_ALL

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
        state_dict_for_mouse_mode:  Annotated[Dict[str, Optional[float]], Field(description="State dictionary for mouse mode")],
        mouse_mode_cycle:           Annotated[RotationCycle, Field(description="Mouse mode cycle")],
        mouse_interaction_handler:  Annotated[Callable[[Image2D], float], Field(description="Mouse interaction handler")],
        wants_crop:                 Annotated[bool, Field(description="Whether to crop the image")],
        ) -> Tuple[Image2D, Image2DMask, Optional[Image2DMask], float]:
    if flip_choice != FlipDirection.NONE:
        if flip_choice == FlipDirection.LEFT_TO_RIGHT:
            i, j = numpy.mgrid[
                0 : pixel_data.shape[0], pixel_data.shape[1] - 1 : -1 : -1
            ]
        elif flip_choice == FlipDirection.TOP_TO_BOTTOM:
            i, j = numpy.mgrid[
                pixel_data.shape[0] - 1 : -1 : -1, 0 : pixel_data.shape[1]
            ]
        elif flip_choice == FlipDirection.BOTH:
            i, j = numpy.mgrid[
                pixel_data.shape[0] - 1 : -1 : -1, pixel_data.shape[1] - 1 : -1 : -1
            ]
        else:
            raise NotImplementedError(
                "Unknown flipping operation: %s" % flip_choice.value
            )
        mask = mask[i, j]
        if pixel_data.ndim == 2:
            pixel_data = pixel_data[i, j]
        else:
            pixel_data = pixel_data[i, j, :]

    if rotate_choice != RotateDirection.NONE:
        if rotate_choice == RotateDirection.ANGLE:
            assert rotate_angle is not None, "rotate_angle must be provided for rotate_choice == RotateDirection.ANGLE"
            angle = rotate_angle
        elif rotate_choice == RotateDirection.COORDINATES:
            assert rotate_point_1 is not None, "rotate_point_1 must be provided for rotate_choice == RotateDirection.COORDINATES"
            assert rotate_point_2 is not None, "rotate_point_2 must be provided for rotate_choice == RotateDirection.COORDINATES"
            assert rotate_coordinate_alignment is not None, "rotate_coordinate_alignment must be provided for rotate_choice == RotateDirection.COORDINATES"
            xdiff = rotate_point_2[0] - rotate_point_1[0]
            ydiff = rotate_point_2[1] - rotate_point_1[1]

            if rotate_coordinate_alignment == RotationCoordinateAlignmnet.VERTICALLY:
                angle = -numpy.arctan2(ydiff, xdiff) * 180.0 / numpy.pi
            elif rotate_coordinate_alignment == RotationCoordinateAlignmnet.HORIZONTALLY:
                angle = numpy.arctan2(xdiff, ydiff) * 180.0 / numpy.pi
            else:
                raise NotImplementedError(
                    "Unknown axis: %s" % rotate_coordinate_alignment.value
                )
        elif rotate_choice == RotateDirection.MOUSE:
            # state_dict_for_mouse_mode = self.get_dictionary()
            if (
                mouse_mode_cycle == RotationCycle.ONCE
                and D_ANGLE in state_dict_for_mouse_mode
                and state_dict_for_mouse_mode[D_ANGLE] is not None
            ):
                angle = state_dict_for_mouse_mode[D_ANGLE]
            else:
                angle = mouse_interaction_handler(pixel_data)
            if mouse_mode_cycle == RotationCycle.ONCE:
                state_dict_for_mouse_mode[D_ANGLE] = angle
        else:
            raise NotImplementedError(
                "Unknown rotation method: %s" % rotate_choice.value
            )
        # rangle = angle * numpy.pi / 180.0
        mask = scipy.ndimage.rotate(mask.astype(float), angle, reshape=True) > 0.50
        crop = (
            scipy.ndimage.rotate(
                numpy.ones(pixel_data.shape[:2]), angle, reshape=True
            )
            > 0.50
        )
        mask = mask & crop
        pixel_data = scipy.ndimage.rotate(pixel_data, angle, reshape=True)
        if wants_crop:
            #
            # We want to find the largest rectangle that fits inside
            # the crop. The cumulative sum in the i and j direction gives
            # the length of the rectangle in each direction and
            # multiplying them gives you the area.
            #
            # The left and right halves are symmetric, so we compute
            # on just two of the quadrants.
            #
            half = (numpy.array(crop.shape) / 2).astype(int)
            #
            # Operate on the lower right
            #
            quartercrop = crop[half[0] :, half[1] :]
            ci = numpy.cumsum(quartercrop, 0)
            cj = numpy.cumsum(quartercrop, 1)
            carea_d = ci * cj
            carea_d[quartercrop == 0] = 0
            #
            # Operate on the upper right by flipping I
            #
            quartercrop = crop[crop.shape[0] - half[0] - 1 :: -1, half[1] :]
            ci = numpy.cumsum(quartercrop, 0)
            cj = numpy.cumsum(quartercrop, 1)
            carea_u = ci * cj
            carea_u[quartercrop == 0] = 0
            carea = carea_d + carea_u
            max_carea = numpy.max(carea)
            max_area = numpy.argwhere(carea == max_carea)[0] + half
            min_i = max(crop.shape[0] - max_area[0] - 1, 0)
            max_i = max_area[0] + 1
            min_j = max(crop.shape[1] - max_area[1] - 1, 0)
            max_j = max_area[1] + 1
            ii = numpy.index_exp[min_i:max_i, min_j:max_j]
            crop = numpy.zeros(pixel_data.shape, bool)
            crop[ii] = True
            mask = mask[ii]
            pixel_data = pixel_data[ii]
        else:
            crop = None
    else:
        crop = None
        angle = 0
    return pixel_data, mask, crop, angle

def flip_image():
    pass

def flip_left_to_right():
    pass

def flip_right_to_left():
    pass

def rotate_image():
    pass
