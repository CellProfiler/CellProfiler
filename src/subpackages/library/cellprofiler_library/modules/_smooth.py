import numpy
from typing import Annotated, Optional
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.functions.image_processing import smoothing_gaussian, smoothing_median, smoothing_keeping_edges, smoothing_fit_polynomial, smoothing_circular_average, smoothing_smooth_to_average
from cellprofiler_library.opts.smooth import SmoothingMethod
from cellprofiler_library.types import Image2D, Image2DMask


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def smooth(
        pixel_data:         Annotated[Image2D, Field(description="Pixel data of image to be smoothed")],
        mask:               Annotated[Optional[Image2DMask], Field(description="Mask of the pixel data")],
        multichannel:       Annotated[bool, Field(description="Set to true if image is multichannel")],
        object_size:        Annotated[Optional[float], Field(description="Object size")],
        smoothing_method:   Annotated[SmoothingMethod, Field(description="Smoothing method")],
        sigma_range:        Annotated[Optional[float], Field(description="Sigma range")],
        clip:               Annotated[Optional[bool], Field(description="Clip intensities to 0 and 1")] = True,
) -> Image2D:
    if object_size is not None:
        obj_size = float(object_size)
    else:
        obj_size = float(min(30, max(1, numpy.mean(pixel_data.shape) / 40)))
    sigma = obj_size / 2.35

    if smoothing_method == SmoothingMethod.GAUSSIAN_FILTER:
        output_pixels = smoothing_gaussian(pixel_data, mask, sigma)
    elif smoothing_method == SmoothingMethod.MEDIAN_FILTER:
        output_pixels = smoothing_median(pixel_data, mask, obj_size)
    elif smoothing_method == SmoothingMethod.SMOOTH_KEEPING_EDGES:
        assert sigma_range is not None, "sigma_range must be provided for smooth_keeping_edges"
        output_pixels = smoothing_keeping_edges(pixel_data, multichannel, sigma_range, sigma)
    elif smoothing_method == SmoothingMethod.FIT_POLYNOMIAL:
        assert clip is not None, "clip must be provided for fit_polynomial"
        output_pixels = smoothing_fit_polynomial(pixel_data, mask, clip)
    elif smoothing_method == SmoothingMethod.CIRCULAR_AVERAGE_FILTER:
        output_pixels = smoothing_circular_average(pixel_data, mask, obj_size)
    elif smoothing_method == SmoothingMethod.SM_TO_AVERAGE:
        output_pixels = smoothing_smooth_to_average(pixel_data, mask)
    else:
        raise ValueError(
            "Unsupported smoothing method: %s" % smoothing_method
        )
    return output_pixels