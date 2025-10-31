import numpy
import scipy.ndimage
from typing import Annotated, Optional, cast, Callable
from pydantic import Field, validate_call, ConfigDict
from skimage.restoration import denoise_bilateral
from centrosome.filter import median_filter as _median_filter
from centrosome.filter import circular_average_filter as _circular_average_filter
from centrosome.smooth import fit_polynomial as _fit_polynomial
from centrosome.smooth import smooth_with_function_and_mask as _smooth_with_function_and_mask

from cellprofiler_library.opts.smooth import SmoothingMethod
from cellprofiler_library.types import Image2D, Image2DMask

median_filter = cast(Callable[[Image2D, Optional[Image2DMask], float], Image2D], _median_filter)
circular_average_filter = cast(Callable[[Image2D, float, Optional[Image2DMask]], Image2D], _circular_average_filter)
smooth_with_function_and_mask = cast(Callable[[Image2D, Callable[[Image2D], Image2D], Optional[Image2DMask]], Image2D], _smooth_with_function_and_mask)
fit_polynomial = cast(Callable[[Image2D, Optional[Image2DMask], bool], Image2D], _fit_polynomial)

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

        def fn(image: Image2D) -> Image2D:
            return scipy.ndimage.gaussian_filter(
                image, sigma, mode="constant", cval=0
            )

        output_pixels = smooth_with_function_and_mask(pixel_data, fn, mask)
    elif smoothing_method == SmoothingMethod.MEDIAN_FILTER:
        output_pixels = median_filter(pixel_data, mask, obj_size / 2 + 1)
    elif smoothing_method == SmoothingMethod.SMOOTH_KEEPING_EDGES:
        assert sigma_range is not None, "sigma_range must be provided for smooth_keeping_edges"
        output_pixels = denoise_bilateral(
            image=pixel_data.astype(float),
            channel_axis=2 if multichannel else None,
            sigma_color=sigma_range,
            sigma_spatial=sigma,
        )
    elif smoothing_method == SmoothingMethod.FIT_POLYNOMIAL:
        assert clip is not None, "clip must be provided for fit_polynomial"
        output_pixels = fit_polynomial(pixel_data, mask, clip)
    elif smoothing_method == SmoothingMethod.CIRCULAR_AVERAGE_FILTER:
        output_pixels = circular_average_filter(
            pixel_data, obj_size / 2 + 1, mask
        )
    elif smoothing_method == SmoothingMethod.SM_TO_AVERAGE:
        if mask is not None:
            mean = numpy.mean(pixel_data[mask])
        else:
            mean = numpy.mean(pixel_data)
        output_pixels = numpy.ones(pixel_data.shape, pixel_data.dtype) * mean
    else:
        raise ValueError(
            "Unsupported smoothing method: %s" % smoothing_method
        )
    return output_pixels