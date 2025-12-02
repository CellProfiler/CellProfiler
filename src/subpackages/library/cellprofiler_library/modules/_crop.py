from typing import Annotated, Optional, Tuple, List
from pydantic import Field, validate_call, ConfigDict
import numpy
from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.functions.image_processing import get_cropped_mask, get_cropped_image_mask, get_cropped_image_pixels
from cellprofiler_library.opts.crop import RemovalMethod, Measurement

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def crop(
        orig_image_pixels:  Annotated[Image2D, Field(description="Pixel values of the original image")],
        cropping:           Annotated[Image2DMask, Field(description="The region of interest to be kept. 1 for pixels to keep, 0 for pixels to remove")],
        mask:               Annotated[Optional[Image2DMask], Field(description="Previous cropping's mask")],
        orig_image_mask:    Annotated[Optional[Image2DMask], Field(description="Mask that may have been set on the original image")],
        removal_method:     Annotated[RemovalMethod, Field(description="Removal method")],
        ) -> Tuple[Image2D, Image2DMask, Image2DMask]:
    #
    # Crop the mask
    #
    mask = get_cropped_mask(cropping, mask, removal_method)

    #
    # Crop the image_mask 
    image_mask = get_cropped_image_mask(cropping, mask, orig_image_mask, removal_method)

    #
    # Crop the image
    #
    cropped_pixel_data = get_cropped_image_pixels(orig_image_pixels, cropping, mask, removal_method)

    return cropped_pixel_data, mask, image_mask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_area_retained_after_cropping(cropping: Image2DMask) -> int:
    return numpy.sum(cropping.astype(float))

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_original_image_area(orig_image_pixels: Image2D) -> int:
    return numpy.product(orig_image_pixels.shape)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_measurements(cropping: Image2DMask, orig_image_pixels:Image2D, cropped_image_name: str = "CroppedImage") -> List[Tuple[str, str, int]]:
    orig_image_area = measure_original_image_area(orig_image_pixels)
    area_retained_after_cropping = measure_area_retained_after_cropping(cropping)
    return [("Image", str(Measurement.ORIGINAL_AREA % cropped_image_name), orig_image_area),
            ("Image", str(Measurement.AREA_RETAINED % cropped_image_name), area_retained_after_cropping)]
