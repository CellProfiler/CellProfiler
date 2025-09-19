import numpy
from typing import Callable, Optional, List, Tuple, cast, Dict
from skimage.util import invert as _invert
from cellprofiler_library.opts.imagemath import Operator, BINARY_OUTPUT_OPS, UNARY_INPUT_OPS, BINARY_INPUT_OPS
from ..types import ImageAny, ImageMaskAny
from numpy import isscalar as _isscalar
invert = cast(Callable[[ImageAny], ImageAny], _invert)
isscalar = cast(Callable[[Optional[ImageAny]], bool], _isscalar)


def apply_on_image(
        output_pixel_data: ImageAny, 
        pd: ImageAny, 
        comparitor: ImageAny, 
        op: Callable[[ImageAny, ImageAny], ImageAny], 
        opval: Operator,
        ) -> ImageAny:
    assert isinstance(output_pixel_data, numpy.ndarray), "output_pixel_data must be a numpy array" # Pylance needs to understand this is a numpy array
    if not isscalar(pd) and output_pixel_data.ndim != pd.ndim:
        if output_pixel_data.ndim == 2:
            output_pixel_data = output_pixel_data[:, :, numpy.newaxis]
            if opval == Operator.EQUALS and not isscalar(comparitor):
                comparitor = comparitor[:, :, numpy.newaxis]
        if pd.ndim == 2:
            pd = pd[:, :, numpy.newaxis]
    if opval == Operator.EQUALS:
        output_pixel_data = output_pixel_data & (comparitor == pd)
    else:
        output_pixel_data = op(output_pixel_data, pd)
    return output_pixel_data


def logical_subtract(output_pixel_data: ImageAny, pd: ImageAny) -> ImageAny:
    output_pixel_data[pd] = False
    return output_pixel_data


def apply_binary_operation(
    opval: Operator, 
    pixel_data: List[ImageAny], 
    masks: Optional[List[Optional[ImageMaskAny]]], 
    output_pixel_data: ImageAny, 
    output_mask: Optional[ImageMaskAny],
    image_factors: Optional[List[float]], # TODO: can this be removed?
    ignore_mask: bool, # TODO: can this be removed?
    ) -> Tuple[
        ImageAny, 
        Optional[ImageMaskAny], 
    ]:
    #
    # Helper function to determine if logical operations should be used
    #
    def use_logical_operation(pixel_data: List[ImageAny]) -> bool:
        return all(
            [pd.dtype == bool for pd in pixel_data if not isscalar(pd)]
        )
    # initialize op to return martix of ones by default
    comparitor = pixel_data[0] # fix pylance error
    use_logical = use_logical_operation(pixel_data)
    op_fn_dispatch: Dict[Operator, Callable[[ImageAny, ImageAny], ImageAny]] = {
        Operator.ADD: numpy.add,
        Operator.SUBTRACT: logical_subtract if use_logical else numpy.subtract,
        Operator.DIFFERENCE: numpy.logical_xor if use_logical else lambda x, y: numpy.abs(numpy.subtract(x, y)),
        Operator.MULTIPLY: numpy.logical_and if use_logical else numpy.multiply,
        Operator.MINIMUM: numpy.minimum,
        Operator.MAXIMUM: numpy.maximum,
        Operator.AVERAGE: numpy.add,
        Operator.MAXIMUM: numpy.maximum,
        Operator.AND: numpy.logical_and,
        Operator.OR: numpy.logical_or,
        Operator.EQUALS: numpy.equal,
        Operator.NONE: lambda x, y: x,
        Operator.DIVIDE: numpy.divide,
    }
    if opval not in op_fn_dispatch:
        raise NotImplementedError(f"Unimplemented operation: {opval}")
    #
    # Binary operations
    #
    op = op_fn_dispatch[opval]

    #
    # Equals and Subtract operations need additional handling
    #
    if opval == Operator.EQUALS:
        output_pixel_data = numpy.ones(pixel_data[0].shape, bool)
        comparitor = pixel_data[0]
    elif opval == Operator.SUBTRACT and use_logical:
        output_pixel_data = pixel_data[0].copy()

    
    # _masks is a list of Nones if masks is None. Fixes type warnings.
    if masks is None:
        masks = [None for _ in pixel_data]

    #
    # Apply the operation to each image in the list
    #
    for pd, mask in zip(pixel_data[1:], masks[1:]):
        output_pixel_data = apply_on_image(output_pixel_data, pd, comparitor, op, opval)
        if not ignore_mask:
            if output_mask is None:
                output_mask = mask
            elif mask is not None:
                output_mask = output_mask & mask
    #
    # Average operation needs additional handling
    #
    if opval == Operator.AVERAGE:
        if not use_logical:
            assert image_factors is not None, "image_factors must be provided for average operation"
            output_pixel_data /= sum(image_factors)
    return output_pixel_data, output_mask


def apply_unary_operation(
    opval: Operator, 
    pixel_data: List[ImageAny], 
    masks: Optional[List[Optional[ImageMaskAny]]], 
    output_pixel_data: ImageAny,
    output_mask: Optional[ImageMaskAny],
    ignore_mask: bool,
    ) -> Tuple[
        ImageAny, 
        Optional[ImageMaskAny],
    ]:
    if opval == Operator.STDEV:
        pixel_array = numpy.array(pixel_data)
        output_pixel_data = numpy.std(pixel_array,axis=0)
        if not ignore_mask:
            mask_array = numpy.array(masks)
            output_mask = mask_array.all(axis=0) 
    elif opval == Operator.INVERT:
        output_pixel_data = invert(output_pixel_data)
    elif opval == Operator.NOT:
        output_pixel_data = numpy.logical_not(output_pixel_data)
    elif opval == Operator.LOG_TRANSFORM:
        output_pixel_data = numpy.log2(output_pixel_data + 1)
    elif opval == Operator.LOG_TRANSFORM_LEGACY:
        output_pixel_data = numpy.log2(output_pixel_data)
    elif opval == Operator.NONE:
        output_pixel_data = output_pixel_data.copy()
    else:
        raise NotImplementedError(
            "The operation %s has not been implemented" % opval
        )
    return output_pixel_data, output_mask


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
        output_pixel_data, output_mask = apply_binary_operation(opval, pixel_data, masks, output_pixel_data, output_mask, image_factors, ignore_mask)

    elif opval in UNARY_INPUT_OPS:
        output_pixel_data, output_mask = apply_unary_operation(opval, pixel_data, masks, output_pixel_data, output_mask, ignore_mask)

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