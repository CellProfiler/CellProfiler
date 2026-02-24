import numpy
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from pydantic import validate_call, ConfigDict
from cellprofiler_library.opts.graytocolor import Scheme
from cellprofiler_library.types import Image2DGrayscale, Image2DColor
from cellprofiler_library.functions.image_processing import gray_to_rgb, gray_to_cmyk, gray_to_composite_color, gray_to_stacked_color

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def gray_to_color(
    pixel_data_arr: List[Optional[Image2DGrayscale]],
    scheme: Scheme,
    color_array: Optional[List[Tuple[int, int, int]]] = None,
    weight_array: Optional[List[float]] = None,
    adjustment_factor_array: Optional[List[float]] = None,
    intensities: List[Tuple[float, ...]]=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    wants_rescale: bool=True,
) -> Image2DColor:
    if scheme == Scheme.RGB:
        assert adjustment_factor_array is not None, "adjustment_factor_array must be provided for RGB mode"
        assert intensities is not None, "intensities must be provided for RGB mode"
        return gray_to_rgb(
            pixel_data_arr = pixel_data_arr,
            adjustment_factor_array = adjustment_factor_array,
            intensities = intensities,
            wants_rescale = wants_rescale,
        )
    elif scheme == Scheme.CMYK:
        assert adjustment_factor_array is not None, "adjustment_factor_array must be provided for CMYK mode"
        assert intensities is not None, "intensities must be provided for CMYK mode"
        return gray_to_cmyk(
            pixel_data_arr = pixel_data_arr,
            adjustment_factor_array = adjustment_factor_array,
            intensities = intensities,
            wants_rescale = wants_rescale,
        )
    elif scheme == Scheme.COMPOSITE:
        assert color_array is not None, "color_array must be provided for composite mode"
        assert weight_array is not None, "weight_array must be provided for composite mode"
        return gray_to_composite_color(
            pixel_data_arr = pixel_data_arr,
            color_array = color_array,
            weight_array = weight_array,
            wants_rescale = wants_rescale,
        )
    elif scheme == Scheme.STACK:
        assert pixel_data_arr is not None, "pixel_data_arr must be provided for stack mode"
        return gray_to_stacked_color(
            pixel_data_arr = pixel_data_arr,
        )
    else:
        raise ValueError(f"Unimplemented scheme: {scheme}")

