import numpy
from typing import Callable, Optional, List, Tuple, cast
from cellprofiler_library.opts.imagemath import Operator, BINARY_OUTPUT_OPS, UNARY_INPUT_OPS, BINARY_INPUT_OPS
from ..types import ImageAny, ImageMaskAny
from numpy import isscalar as _isscalar
from cellprofiler_library.functions.image_processing import imagemath_apply_binary_operation, imagemath_apply_unary_operation
isscalar = cast(Callable[[Optional[ImageAny]], bool], _isscalar)



def image_math(
        opval: Operator, 
        pixel_data: List[ImageAny], 
        masks: Optional[List[Optional[ImageMaskAny]]], 
        output_pixel_data: ImageAny, 
        output_mask: Optional[ImageMaskAny],
        ignore_mask: bool = False,
        image_factors: Optional[List[float]] = None,
        exponent: Optional[float] = None,
        after_factor: Optional[float] = None,
        addend: Optional[float] = None,
        truncate_low: Optional[bool] = None,
        truncate_high: Optional[bool] = None,
        replace_nan: Optional[bool] = None,
        ) -> Tuple[
            ImageAny, Optional[ImageMaskAny]
        ]:
    if opval in BINARY_INPUT_OPS:
        output_pixel_data, output_mask = imagemath_apply_binary_operation(opval, pixel_data, masks, output_pixel_data, output_mask, image_factors, ignore_mask)

    elif opval in UNARY_INPUT_OPS:
        output_pixel_data, output_mask = imagemath_apply_unary_operation(opval, pixel_data, masks, output_pixel_data, output_mask, ignore_mask)

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