from typing import Optional, Union, Annotated
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.opts.rescaleintensity import RescaleMethod, MinimumIntensityMethod, MaximumIntensityMethod, M_ALL, LOW_ALL, HIGH_ALL
from cellprofiler_library.types import ImageAny, ImageAnyMask, ImageGrayscale, ImageGrayscaleMask
from cellprofiler_library.functions.image_processing import stretch, manual_input_range, manual_io_range, divide_by_image_maximum, divide_by_image_minimum, divide_by_value, scale_by_image_maximum

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def rescale_intensity(
        rescale_method:         Annotated[RescaleMethod, Field(description="Method to rescale the image")],
        in_pixel_data:          Annotated[ImageAny, Field(description="Input image data")],
        in_mask:                Annotated[ImageAnyMask, Field(description="Input image mask")],
        input_image_has_mask:   Annotated[bool, Field(description="Whether the input image has a mask")],
        in_multichannel:        Annotated[bool, Field(description="Whether the input image is multichannel")],
        divisor_value:          Annotated[float, Field(description="Value to divide the input image by (for divide by value)")],
        auto_high:              Annotated[MaximumIntensityMethod, Field(description="Method to determine the maximum intensity")],
        auto_low:               Annotated[MinimumIntensityMethod, Field(description="Method to determine the minimum intensity")],
        source_high:            Annotated[float, Field(description="Upper intensity limit for the input image")],
        source_low:             Annotated[float, Field(description="Lower intensity limit for the input image")],
        source_scale_min:       Annotated[float, Field(description="Intensity range for the input image - minimum")],
        source_scale_max:       Annotated[float, Field(description="Intensity range for the input image - maximum")],
        shared_dict, # do not type annotate as it has a bad interation with Pydantic TODO #5088: Discuss
        dest_scale_min:         Annotated[Optional[float], Field(description="Intensity range for the output image - minimum")],
        dest_scale_max:         Annotated[Optional[float], Field(description="Intensity range for the output image - maximum")],
        reference_image_pixel_data: Annotated[Optional[ImageAny], Field(description="Reference image data - for scale by image maximum")],
        reference_image_mask:   Annotated[Optional[ImageGrayscaleMask], Field(description="Reference image mask - for scale by image maximum")],
    ) -> ImageAny:
    if rescale_method == RescaleMethod.STRETCH.value:
        output_image = stretch(in_pixel_data, in_mask, in_multichannel)
    elif (rescale_method == RescaleMethod.MANUAL_INPUT_RANGE.value) or (rescale_method == RescaleMethod.MANUAL_IO_RANGE.value):
        in_mask_manual_input = None if input_image_has_mask else in_mask # for this method, mask needs to be None if not present
        if rescale_method == RescaleMethod.MANUAL_INPUT_RANGE.value:
            output_image = manual_input_range(
                in_pixel_data,
                in_mask_manual_input,
                source_high, source_low, source_scale_min, source_scale_max, auto_high, auto_low, shared_dict
            )
        else:
            output_image = manual_io_range(
                in_pixel_data,
                in_mask_manual_input,
                source_high, source_low, source_scale_min, source_scale_max, auto_high, auto_low, shared_dict, dest_scale_min, dest_scale_max
                )
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM.value:
        output_image = divide_by_image_minimum(in_pixel_data, in_mask)
    elif rescale_method == RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM.value:
        output_image = divide_by_image_maximum(in_pixel_data, in_mask)
    elif rescale_method == RescaleMethod.DIVIDE_BY_VALUE.value:
        output_image = divide_by_value(in_pixel_data, divisor_value)
    elif rescale_method == RescaleMethod.DIVIDE_BY_MEASUREMENT.value:
        # TODO #5088 update this once measurement format is finalized
        raise NotImplementedError("Rescale by measurement not implemented in library yet")
    elif rescale_method == RescaleMethod.SCALE_BY_IMAGE_MAXIMUM.value:
        assert reference_image_pixel_data is not None, "Reference image is required for scale by image maximum"
        assert reference_image_mask is not None, "Reference image mask is required for scale by image maximum"
        output_image = scale_by_image_maximum(in_pixel_data, in_mask, reference_image_pixel_data, reference_image_mask)
    else:
        raise ValueError(f"Invalid rescale method: {rescale_method.value}")    
    return output_image

