from typing import Optional, Annotated
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.functions.image_processing import get_3d_adjacent_after_erosion, process_all_connected_components, find_adjacent_by_distance
from cellprofiler_library.types import Image2DBinary, Image2DBinaryMask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def identify_dead_worms(
        pixel_data:         Annotated[Image2DBinary, Field(description="Input binary image")],
        image_mask:         Annotated[Optional[Image2DBinaryMask], Field(description="Input binary image mask")],
        automatic_distance: Annotated[bool, Field(default=True, description="Whether to calculate distance parameters automatically")],
        worm_width:         Annotated[Optional[int], Field(default=100, ge=1, description="This is the width (the short axis), measured in pixels, of the diamond used as a template when matching against the worm. It should be less than the width of a worm.")],
        worm_length:        Annotated[Optional[int], Field(default=10, ge=1, description="This is the length (the long axis), measured in pixels, of the diamond used as a template when matching against the worm. It should be less than the length of a worm")],
        angle_count:        Annotated[int, Field(description="Number of different angles at which the template will betried", ge=1)] = 32,
        space_distance:     Annotated[Optional[float], Field(default=5, ge=1, description="Used only if not automatically calculating distance parameters Enter the distance for calculating the worm centers, in units of pixels. The worm centers must be at least many pixels apart for the centers to be considered two separate worms.")]=5,
        angular_distance:   Annotated[Optional[float], Field(default=30, ge=1, description="Used only if automatically calculating distance parameters IdentifyDeadWorms calculates the worm centers at different angles. Two worm centers are considered to represent different worms if their angular distance is larger than this number. The number is measured in degrees.")]=30,
    ):

    mask = pixel_data
    if image_mask is not None:
        mask = mask & image_mask
    #
    # We collect the i,j and angle of pairs of points that
    # are 3-d adjacent after erosion.
    #
    i_center, j_center, angular_orientation = get_3d_adjacent_after_erosion(mask, angle_count, worm_width, worm_length)

    #
    # Find connections based on distances, not adjacency
    #
    first, second = find_adjacent_by_distance(
        i_center, 
        j_center, 
        angular_orientation, 
        automatic_distance,
        worm_width, 
        worm_length,
        angle_count,
        space_distance,
        angular_distance
        )
        
    #
    # Do all connected components.
    #
    center_x, center_y, angles, nlabels, label_indexes, labels = process_all_connected_components(first, second, i_center, j_center, angular_orientation, mask)
    return center_x, center_y, angles, nlabels, label_indexes, labels
