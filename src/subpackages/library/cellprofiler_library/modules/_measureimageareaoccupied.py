import skimage.measure
import numpy as np
from typing import Annotated, Optional, Tuple
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageBinary, ObjectSegmentation, ImageAny
from cellprofiler_library.functions.measurement import measure_area_occupied, measure_total_area, measure_perimeter, measure_object_perimeter, measure_objects_area_occupied, measure_objects_total_area


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_area_perimeter(
    im_pixel_data:  Annotated[ImageBinary, Field(description="Binary image pixel data")],
    im_volumetric:  Annotated[bool, Field(description="Image is volumetric")],
    im_spacing:     Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None
    ) -> Tuple[np.float_, np.float_, np.int_]:

    area_occupied = measure_area_occupied(im_pixel_data)
    perimeter = measure_perimeter(im_pixel_data, im_volumetric, im_spacing)  if area_occupied > 0 else np.float64(0.0)
    total_area = measure_total_area(im_pixel_data)
 
    return area_occupied, perimeter, total_area

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_objects_area_perimeter(
    label_image:    Annotated[ObjectSegmentation, Field(description="Object labels to measure")],
    mask:           Annotated[Optional[ImageAny], Field(description="Mask of the image")] = None,
    volumetric:     Annotated[bool, Field(description="True if the image is volumetric")] = False,
    spacing:        Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None
    ) -> Tuple[np.float_, np.float_, np.int_]:
    if mask is not None:
        label_image[~mask] = 0
    regionprops = skimage.measure.regionprops(label_image)

    total_area = measure_objects_total_area(label_image, mask)
    area_occupied = measure_objects_area_occupied(None, regionprops=regionprops)
    perimeter = measure_object_perimeter(label_image, regionprops=regionprops, volumetric=volumetric, spacing=spacing) if area_occupied > 0 else np.float64(0.0)

    return area_occupied, perimeter, total_area
