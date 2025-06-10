from typing import Annotated, Any, Optional, Tuple, Callable, Union
from pydantic import Field, validate_call, BeforeValidator, ConfigDict
from cellprofiler_library.opts.correctilluminationapply import Method
from ..types import Image2D, Image2DGrayscale
from ..functions.image_processing import fix_illumination_function, apply_divide, apply_subtract, clip_low, clip_high
import numpy


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def correct_illumination_apply(
        image_pixels: Annotated[Image2D, Field(description="Pixel data of image to apply the illumination function to")],
        illum_function_pixel_data: Annotated[Image2D, Field(description="Pixel data of illumination function")],
        method_divide_or_subtract: Annotated[Method, Field(description="Method to apply the illumination function")],
        truncate_low: Annotated[Optional[bool], Field(description="Set output image values less than 0 equal to 0?")],
        truncate_high: Annotated[Optional[bool], Field(description="Set output image values greater than 1 equal to 1?")],
        ) -> Annotated[Image2DGrayscale, Field(description="Pixel data of image with illumination function applied")]:
    """
    Perform illumination according to the parameters of one image setting group
    """

    illum_function_pixel_data = fix_illumination_function(image_pixels.shape, illum_function_pixel_data)
    #
    # Either divide or subtract the illumination image from the original
    #
    if method_divide_or_subtract == Method.DIVIDE:
        output_pixels = apply_divide(image_pixels, illum_function_pixel_data)
    elif method_divide_or_subtract == Method.SUBTRACT:
        output_pixels = apply_subtract(image_pixels, illum_function_pixel_data)
    else:
        raise ValueError(
            "Unhandled option for divide or subtract: %s"
            % image.divide_or_subtract.value
        )
    #
    # Optionally, clip high and low values
    #
    if truncate_low:
        output_pixels = clip_low(output_pixels)
    if truncate_high:
        output_pixels = clip_high(output_pixels) 
    
    return output_pixels