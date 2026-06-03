import numpy

from numpy.typing import NDArray
from typing import Optional, Annotated, Union, Tuple
from pydantic import Field, validate_call, ConfigDict, BaseModel

from cellprofiler_library.types import Image2DBinary, Image2DBinaryMask, ObjectLabel
from cellprofiler_library.functions.image_processing import get_3d_adjacent_after_erosion, process_all_connected_components, find_adjacent_by_distance
from cellprofiler_library.measurement_model import LibraryMeasurements

from cellprofiler_library.opts.identifydeadworms import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, M_ANGLE, M_NUMBER_OBJECT_NUMBER, TemplateMeasurementFormat


class IdentifyDeadWormsDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    center_x: NDArray[numpy.int_]
    center_y: NDArray[numpy.int_]
    angles: NDArray[numpy.float_]
    mask: Image2DBinaryMask
    nlabels: int


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def identify_dead_worms(
        pixel_data:         Annotated[Image2DBinary, Field(description="Input binary image")],
        image_mask:         Annotated[Optional[Image2DBinaryMask], Field(description="Input binary image mask")],
        automatic_distance: Annotated[bool, Field(default=True, description="Whether to calculate distance parameters automatically")],
        worm_width:         Annotated[int, Field(default=100, ge=1, description="This is the width (the short axis), measured in pixels, of the diamond used as a template when matching against the worm. It should be less than the width of a worm.")],
        worm_length:        Annotated[int, Field(default=10, ge=1, description="This is the length (the long axis), measured in pixels, of the diamond used as a template when matching against the worm. It should be less than the length of a worm")],
        angle_count:        Annotated[int, Field(description="Number of different angles at which the template will betried", ge=1)] = 32,
        space_distance:     Annotated[Optional[float], Field(default=5, ge=1, description="Used only if not automatically calculating distance parameters Enter the distance for calculating the worm centers, in units of pixels. The worm centers must be at least many pixels apart for the centers to be considered two separate worms.")]=5,
        angular_distance:   Annotated[Optional[float], Field(default=30, ge=1, description="Used only if automatically calculating distance parameters IdentifyDeadWorms calculates the worm centers at different angles. Two worm centers are considered to represent different worms if their angular distance is larger than this number. The number is measured in degrees.")]=30,
        object_name:        Annotated[str, Field(description="Name for the dead worm object")]="dead worm",
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
) -> Union[
        Tuple[NDArray[ObjectLabel], LibraryMeasurements],
        Tuple[NDArray[ObjectLabel], LibraryMeasurements, IdentifyDeadWormsDisplayData]
    ]:

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

    #
    # Make measurements
    #
    measurements = LibraryMeasurements()

    measurements.add_measurement(object_name, M_LOCATION_CENTER_X, center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y, center_y)
    measurements.add_measurement(object_name, M_ANGLE, angles * 180 / numpy.pi)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER, label_indexes)
    measurements.add_image_measurement(TemplateMeasurementFormat.FF_COUNT % object_name, nlabels)

    if return_visualization_data:
        return labels, measurements, IdentifyDeadWormsDisplayData(
            mask=mask,
            center_x=center_x,
            center_y=center_y,
            angles=angles,
            nlabels=nlabels,
        )
    return labels, measurements
