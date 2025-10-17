from typing import Annotated, Optional, Tuple
from pydantic import Field, validate_call, ConfigDict
import skimage.measure
import numpy as np
from ..types import ImageBinary, ObjectLabelsDense, ImageAny
from ..functions.measurement import measure_surface_area


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_area_perimeter(
    im_pixel_data: Annotated[ImageBinary, Field(description="Binary image pixel data")],
    im_volumetric: Annotated[bool, Field(description="Image is volumetric")],
    im_spacing: Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None
    ) -> Tuple[float, float, float]:

    area_occupied = np.sum(im_pixel_data > 0)

    if area_occupied > 0:
        if im_volumetric:
            perimeter = measure_surface_area(im_pixel_data > 0, spacing=im_spacing)
        else:
            perimeter = skimage.measure.perimeter(im_pixel_data > 0)
    else:
        perimeter = 0
        
    total_area = np.prod(np.shape(im_pixel_data))
 
    return area_occupied, perimeter, total_area



def measure_objects_area_perimeter(
    label_image: ObjectLabelsDense, 
    mask: Optional[ImageAny], 
    volumetric: bool, 
    spacing: Optional[Tuple[float, ...]] = None
    ) -> Tuple[float, float, float]:
    if mask is not None:
        label_image[~mask] = 0
        total_area = np.sum(mask)

    else:
        total_area = np.product(label_image.shape)

    region_properties = skimage.measure.regionprops(label_image)
    area_occupied = np.sum([region["area"] for region in region_properties])
    
    if area_occupied > 0:
        if volumetric:
            labels = np.unique(label_image)
            if labels[0] == 0:
                labels = labels[1:]
            
        if volumetric:
            perimeter = measure_surface_area(label_image, spacing=spacing, index=labels)
        else:
            perimeter = np.sum(
                [np.round(region["perimeter"]) for region in region_properties]
            )
    else:
        perimeter = 0
    return area_occupied, perimeter, total_area
