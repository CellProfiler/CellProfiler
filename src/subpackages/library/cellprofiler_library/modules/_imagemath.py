import numpy
from typing import Callable, Optional, List, Tuple, cast, Union, Annotated
from numpy import isscalar as _isscalar
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.opts.imagemath import Operator, BINARY_OUTPUT_OPS, UNARY_INPUT_OPS, BINARY_INPUT_OPS
from cellprofiler_library.types import ImageAny, ImageAnyMask
from cellprofiler_library.functions.image_processing import imagemath_apply_binary_operation, imagemath_apply_unary_operation
isscalar = cast(Callable[[Optional[ImageAny]], bool], _isscalar)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def image_math(
        opval:              Annotated[Operator, Field(description=" Operator to apply to the images or measurements")],
        operands:           Annotated[List[Union[ImageAny, float]], Field(description=" List of images and/or measurements to apply the operator to")],
        masks:              Annotated[Optional[List[Optional[Union[ImageAnyMask, bool]]]], Field(description=" List of masks for the images. Use `True` for measurement operands")],
        output_pixel_data:  Annotated[ImageAny, Field(description=" Output image or measurement")],
        output_mask:        Annotated[Optional[ImageAnyMask], Field(description=" Output mask")],
        ignore_mask:        Annotated[bool, Field(description=" Ignore the masks")],
        image_factors:      Annotated[Optional[List[float]], Field(description=" List of image factors to multiply the images by")],
        exponent:           Annotated[Optional[float], Field(description=" Exponent to raise the images by")],
        after_factor:       Annotated[Optional[float], Field(description=" Multiply the images by this value after the operation")],
        addend:             Annotated[Optional[float], Field(description=" Add this value to the images")],
        truncate_low:       Annotated[Optional[bool], Field(description=" Truncate the images below 0")],
        truncate_high:      Annotated[Optional[bool], Field(description=" Truncate the images above 0")],
        replace_nan:        Annotated[Optional[bool], Field(description=" Replace invalid values with 0")],
        ) -> Tuple[
            ImageAny, Optional[ImageAnyMask]
        ]:
    if opval in BINARY_INPUT_OPS:
        output_pixel_data, output_mask = imagemath_apply_binary_operation(opval, operands, masks, output_pixel_data, output_mask, image_factors, ignore_mask)

    elif opval in UNARY_INPUT_OPS:
        output_pixel_data, output_mask = imagemath_apply_unary_operation(opval, operands, masks, output_pixel_data, output_mask, ignore_mask)

    else:
        raise NotImplementedError(
            "The operation %s has not been implemented" % opval
        )

    # Check to see if there was a measurement & image w/o mask. If so
    # set mask to none
    if isscalar(output_mask):
        output_mask = None
    if opval not in BINARY_OUTPUT_OPS:
        #
        # Post-processing: exponent, multiply, add
        #
        if exponent is not None and exponent != 1:
            output_pixel_data **= exponent
        if after_factor is not None and after_factor != 1:
            output_pixel_data *= after_factor
        if addend is not None and addend != 0:
            output_pixel_data += addend

        #
        # truncate values
        #
        if truncate_low:
            output_pixel_data[output_pixel_data < 0] = 0
        if truncate_high:
            output_pixel_data[output_pixel_data > 1] = 1
        if replace_nan:
            output_pixel_data[numpy.isnan(output_pixel_data)] = 0
    return output_pixel_data, output_mask