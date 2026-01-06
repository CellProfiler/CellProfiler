import numpy
import skimage.exposure
from typing import Tuple, Optional, Union
from pydantic import validate_call, ConfigDict
from cellprofiler_library.opts.rescaleintensity import RescaleMethod, MinimumIntensityMethod, MaximumIntensityMethod, M_ALL, LOW_ALL, HIGH_ALL
from cellprofiler_library.types import ImageAny, ImageAnyMask, ImageGrayscale, ImageGrayscaleMask, ImageUInt

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def rescale(
        image_pixel_data: Union[ImageAny, ImageUInt], 
        in_range: Tuple[float, float],
        out_range: Tuple[float, float] = (0.0, 1.0)
    ):
    data = 1.0 * image_pixel_data

    rescaled = skimage.exposure.rescale_intensity(
        data, in_range=in_range, out_range=out_range
    )

    return rescaled

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def stretch(
        data: Union[Union[ImageAny, ImageUInt], ImageUInt], 
        mask: ImageAnyMask, 
        multichannel: bool = False
    ):
    if multichannel:
        splitaxis = data.ndim - 1
        singlechannels = numpy.split(data, data.shape[-1], splitaxis)
        newchannels = []
        for channel in singlechannels:
            channel = numpy.squeeze(channel, axis=splitaxis)
            if (masked_channel := channel[mask]).size == 0:
                in_range = (0, 1)
            else:
                in_range = (min(masked_channel), max(masked_channel))

            rescaled = rescale(channel, in_range)

            newchannels.append(rescaled)
        full_rescaled = numpy.stack(newchannels, axis=-1)
        return full_rescaled
    if (masked_data := data[mask]).size == 0:
        in_range = (0, 1)
    else:
        in_range = (min(masked_data), max(masked_data))
    return rescale(data, in_range)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def manual_input_range(
        data: Union[ImageAny, ImageUInt], 
        mask: ImageAnyMask, 
        source_high: float, 
        source_low: float, 
        source_scale_min: float, 
        source_scale_max: float, 
        auto_high: MaximumIntensityMethod, 
        auto_low: MinimumIntensityMethod, 
        shared_dict, # do not type annotate as it has a bad interation with Pydantic
    ):
    return manual_io_range(
        data, 
        mask, 
        source_high, 
        source_low, 
        source_scale_min, 
        source_scale_max, 
        auto_high, 
        auto_low, 
        shared_dict
    )

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def manual_io_range(
        data: Union[ImageAny, ImageUInt], 
        mask: ImageAnyMask, 
        source_high: float, 
        source_low: float, 
        source_scale_min: float, 
        source_scale_max: float, 
        auto_high: MaximumIntensityMethod, 
        auto_low: MinimumIntensityMethod, 
        shared_dict, # do not type annotate as it has a bad interation with Pydantic
        dest_scale_min: Optional[float]=None, 
        dest_scale_max: Optional[float]=None
    ):
    in_range = get_source_range(data, mask, source_high, source_low, source_scale_min, source_scale_max, auto_high, auto_low, shared_dict)
    if dest_scale_min is None and dest_scale_max is None:
        return rescale(data, in_range)
    else:
        out_range = (dest_scale_min, dest_scale_max)
        return rescale(data, in_range, out_range)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def divide(data: Union[ImageAny, ImageUInt], value):
    if value == 0.0:
        raise ZeroDivisionError("Cannot divide pixel intensity by 0.")

    return data / float(value)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def divide_by_image_minimum(data: Union[ImageAny, ImageUInt], mask: ImageAnyMask):
    if (masked_data := data[mask]).size == 0:
        src_min = 0
    else:
        src_min = numpy.min(masked_data)

    return divide(data, src_min)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def divide_by_image_maximum(data: Union[ImageAny, ImageUInt], mask: ImageAnyMask):
    if (masked_data := data[mask]).size == 0:
        src_max = 1
    else:
        src_max = numpy.max(masked_data)

    return divide(data, src_max)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def divide_by_value(data: Union[ImageAny, ImageUInt], divisor_value):
    return divide(data, divisor_value)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def scale_by_image_maximum(data: Union[ImageAny, ImageUInt], mask: ImageAnyMask, reference_data: Union[ImageGrayscale, ImageUInt], reference_mask: ImageGrayscaleMask):
    ###
    # Scale the image by the maximum of another image
    #
    # Find the maximum value within the unmasked region of the input
    # and reference image. Multiply by the reference maximum, divide
    # by the input maximum to scale the input image to the same
    # range as the reference image
    ###
    if (masked_input := data[mask]).size == 0:
        return data
    else:
        image_max = numpy.max(masked_input)

    if image_max == 0:
        return data


    if (masked_ref := reference_data[reference_mask]).size == 0:
        reference_max = 1
    else:
        reference_max = numpy.max(masked_ref)

    return divide(data * reference_max, image_max)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_source_range(
        data: Union[ImageAny, ImageUInt], 
        mask: ImageAnyMask, 
        source_high: float, 
        source_low: float, 
        source_scale_min: float, 
        source_scale_max: float, 
        auto_high: MaximumIntensityMethod, 
        auto_low: MinimumIntensityMethod, 
        shared_dict, # do not type annotate as it has a bad interation with Pydantic
    ):
    """Get the source range, accounting for automatically computed values"""
    input_pixels = None
    if (
        auto_high == MaximumIntensityMethod.CUSTOM_VALUE.value
        and auto_low == MinimumIntensityMethod.CUSTOM_VALUE.value
    ):
        return source_scale_min, source_scale_max

    if (
        auto_low == MinimumIntensityMethod.EACH_IMAGE.value
        or auto_high == MaximumIntensityMethod.EACH_IMAGE.value
    ):
        input_pixels = data
        if mask is not None:
            input_pixels = input_pixels[mask]
            if input_pixels.size == 0:
                return 0, 1

    if auto_low == MinimumIntensityMethod.ALL_IMAGES.value:
        src_min = shared_dict[MinimumIntensityMethod.ALL_IMAGES.value]
    elif auto_low == MinimumIntensityMethod.EACH_IMAGE.value:
        assert input_pixels is not None, "Invalid settings for automatic minimum, please check your settings and data"
        src_min = numpy.min(input_pixels)
    else:
        src_min = source_low

    if auto_high == MaximumIntensityMethod.ALL_IMAGES.value:
        src_max = shared_dict[MaximumIntensityMethod.ALL_IMAGES.value]
    elif auto_high == MaximumIntensityMethod.EACH_IMAGE.value:
        assert input_pixels is not None, "Invalid settings for automatic maximum, please check your settings and data"
        src_max = numpy.max(input_pixels)
    else:
        src_max = source_high
        
    return src_min, src_max
