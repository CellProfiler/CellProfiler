import centrosome.smooth
import centrosome.cpmorphology
import scipy.ndimage
import numpy
from typing import Any, Dict, Optional, Tuple, Annotated
from pydantic import validate_call, ConfigDict
from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.opts.correctilluminationcalculate import (
    SmoothingFilterSize,
    IntensityChoice,
    SmoothingMethod,
    SplineBackgroundMode,
    RescaleIlluminationFunction,
    StateKey,
)

from cellprofiler_library.functions.image_processing import smooth_plane

ROBUST_FACTOR = 0.02  # For rescaling, take 2nd percentile value

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def apply_dilation(
        pixel_data: Image2D,
        mask: Image2DMask,
        object_dilation_radius: int,
    ) -> Image2D:
    """Return pixel data dilated with a circular Gaussian kernel.

    This filter spreads the boundaries of cells, effectively "dilating" them.

    Args:
        pixel_data: 2-D or 3-D image array (H, W) or (H, W, C).
        mask: Boolean mask array (H, W). True = valid pixel.
        object_dilation_radius: Radius for the circular Gaussian kernel.

    Returns:
        Dilated pixel data array of the same shape as pixel_data.
    """
    kernel = centrosome.smooth.circular_gaussian_kernel(
        object_dilation_radius, object_dilation_radius * 3
    )

    def fn(image):
        return scipy.ndimage.convolve(image, kernel, mode="constant", cval=0)

    if pixel_data.ndim == 2:
        dilated_pixels = centrosome.smooth.smooth_with_function_and_mask(
            pixel_data, fn, mask
        )
    else:
        dilated_pixels = numpy.dstack(
            [
                centrosome.smooth.smooth_with_function_and_mask(
                    x, fn, mask
                )
                for x in pixel_data.transpose(2, 0, 1)
            ]
        )
    return dilated_pixels

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def preprocess_image_for_averaging(
        pixel_data: Image2D,
        mask: Optional[Image2DMask],
        intensity_choice: IntensityChoice,
        smoothing_method: SmoothingMethod,
        block_size: int,
    ) -> Image2D:
    """Create a version of the image appropriate for averaging.

    For Regular or Splines mode: zeros out masked pixels (if any) and
    returns the result. For Background mode: finds the minimum pixel
    intensity within blocks and returns a block-minimum image.

    Args:
        pixel_data: Image array (H, W) or (H, W, C).
        has_mask: Whether a mask is present.
        mask: Boolean mask array (H, W); True = valid. Required when
            has_mask is True.
        intensity_choice: Regular or Background illumination method.
        smoothing_method: The smoothing method selected (affects whether
            the Splines branch is used).
        block_size: Side length in pixels of each background block
            (Background mode only).

    Returns:
        Preprocessed image array suitable for accumulation.
    """
    if intensity_choice == IntensityChoice.REGULAR.value or smoothing_method == SmoothingMethod.SPLINES.value:
        if mask is not None:
            if pixel_data.ndim == 2:
                pixel_data[~mask] = 0
            else:
                pixel_data[~mask, :] = 0
            return pixel_data
        else:
            return pixel_data
    else:
        # For background, we create a labels image using the block
        # size and find the minimum within each block.
        labels, indexes = centrosome.cpmorphology.block(
            pixel_data.shape[:2], (block_size, block_size)
        )
        if mask is not None:
            labels[~mask] = -1

        min_block = numpy.zeros(pixel_data.shape)
        if pixel_data.ndim == 2:
            minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.minimum(pixel_data, labels, indexes)
            )
            min_block[labels != -1] = minima[labels[labels != -1]]
        else:
            for i in range(pixel_data.shape[2]):
                minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.minimum(pixel_data[:, :, i], labels, indexes)
                )
                min_block[labels != -1, i] = minima[labels[labels != -1]]
        return min_block

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def initialize_illumination_accumulation(
        preprocessed_pixel_data: numpy.ndarray,
        mask: Optional[numpy.ndarray],
) -> Dict[str, Any]:
    """Initialize the illumination accumulation state from the first image.

    Creates a zeroed image_sum and mask_count, then accumulates the
    first (already preprocessed) image into them.

    Args:
        preprocessed_pixel_data: Output of preprocess_image_for_averaging
            for the first image, shape (H, W) or (H, W, C).
        has_mask: Whether the image has a mask.
        mask: Boolean mask (H, W); True = valid. Required when has_mask
            is True.

    Returns:
        Initial accumulation state dict with StateKey.IMAGE_SUM and
        StateKey.MASK_COUNT entries.
    """
    state: Dict[str, Any] = {
        StateKey.IMAGE_SUM.value: numpy.zeros(
            preprocessed_pixel_data.shape, preprocessed_pixel_data.dtype
        ),
        StateKey.MASK_COUNT.value: numpy.zeros(
            preprocessed_pixel_data.shape[:2], numpy.int32
        ),
    }
    accumulate_illumination_image(preprocessed_pixel_data, mask, state)
    return state

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def accumulate_illumination_image(
        preprocessed_pixel_data: numpy.ndarray,
        mask: Optional[numpy.ndarray],
        state: Dict[str, Any],
) -> Dict[str, Any]:
    """Accumulate a preprocessed image into the illumination state.

    Args:
        preprocessed_pixel_data: Output of preprocess_image_for_averaging,
            shape (H, W) or (H, W, C).
        has_mask: Whether the image has a mask.
        mask: Boolean mask (H, W); True = valid. Required when has_mask
            is True.
        state: Existing accumulation state dict (mutated in place).

    Returns:
        The updated state dict (same object as input).
    """
    image_sum = state[StateKey.IMAGE_SUM.value]
    mask_count = state[StateKey.MASK_COUNT.value]
    if mask is not None:
        if image_sum.ndim == 2:
            image_sum[mask] += preprocessed_pixel_data[mask]
        else:
            image_sum[mask, :] += preprocessed_pixel_data[mask, :]
        mask_count[mask] += 1
    else:
        image_sum += preprocessed_pixel_data
        mask_count += 1
    return state

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_average_from_state(
        state: Dict[str, Any],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Compute the average illumination image from the accumulated state.

    Args:
        state: Accumulation state dict produced by
            initialize_illumination_accumulation /
            accumulate_illumination_image.

    Returns:
        Tuple of (avg_pixel_data, mask) where avg_pixel_data contains
        the per-pixel mean values and mask is a boolean array
        (True where at least one image contributed).
    """
    image_sum = state[StateKey.IMAGE_SUM.value]
    mask_count = state[StateKey.MASK_COUNT.value]
    pixel_data = numpy.zeros(image_sum.shape, image_sum.dtype)
    mask = mask_count > 0
    if pixel_data.ndim == 2:
        pixel_data[mask] = image_sum[mask] / mask_count[mask]
    else:
        for i in range(pixel_data.shape[2]):
            pixel_data[mask, i] = image_sum[mask, i] / mask_count[mask]
    return pixel_data, mask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def apply_smoothing(
        image_pixel_data: Image2D,
        image_mask: Image2DMask,
        smoothing_method: SmoothingMethod,
        automatic_object_width: Optional[SmoothingFilterSize], 
        size_of_smoothing_filter: Optional[int], # default to 10
        object_width: Optional[int], # default to 10 
        image_shape: Optional[Tuple[int, ...]],
        automatic_splines: bool,
        spline_bg_mode: Optional[SplineBackgroundMode],
        spline_points: Optional[int],
        spline_threshold: Optional[float],
        spline_convergence: Optional[float],
        spline_maximum_iterations: Optional[int],
        spline_rescale: Optional[float],
    ):
    """Return an image that is smoothed according to the settings

    image - an instance of cpimage.Image containing the pixels to analyze
    orig_image - the ancestor source image or None if ambiguous
    returns another instance of cpimage.Image
    """
    pixel_data = image_pixel_data
    if pixel_data.ndim == 3:
        output_pixels = numpy.zeros(pixel_data.shape, pixel_data.dtype)
        for i in range(pixel_data.shape[2]):
            output_pixels[:, :, i] = smooth_plane(
                pixel_data = pixel_data[:, :, i], 
                mask = image_mask,
                smoothing_method = smoothing_method,
                automatic_object_width = automatic_object_width, 
                size_of_smoothing_filter = size_of_smoothing_filter, 
                object_width = object_width, 
                image_shape = image_shape,
                automatic_splines = automatic_splines,
                spline_bg_mode = spline_bg_mode,
                spline_points = spline_points,
                spline_threshold = spline_threshold,
                spline_convergence = spline_convergence,
                spline_maximum_iterations = spline_maximum_iterations,
                spline_rescale = spline_rescale,
            )
    else:
        output_pixels = smooth_plane(
            pixel_data = pixel_data, 
            mask = image_mask,
            smoothing_method = smoothing_method,
            automatic_object_width = automatic_object_width, 
            size_of_smoothing_filter = size_of_smoothing_filter, 
            object_width = object_width, 
            image_shape = image_shape,
            automatic_splines = automatic_splines,
            spline_bg_mode = spline_bg_mode,
            spline_points = spline_points,
            spline_threshold = spline_threshold,
            spline_convergence = spline_convergence,
            spline_maximum_iterations = spline_maximum_iterations,
            spline_rescale = spline_rescale,
        )
    return output_pixels

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def apply_scaling(
        image_pixel_data,
        image_mask, 
        rescale_option: RescaleIlluminationFunction,
    ):
    """Return an image that is rescaled according to the settings

    image - an instance of cpimage.Image
    returns another instance of cpimage.Image
    """
    if rescale_option == RescaleIlluminationFunction.NO.value:
        return image_pixel_data

    def scaling_fn_2d(pixel_data):
        if image_mask is not None:
            sorted_pixel_data = pixel_data[(pixel_data > 0) & image_mask]
        else:
            sorted_pixel_data = pixel_data[pixel_data > 0]
        if sorted_pixel_data.shape[0] == 0:
            return pixel_data
        sorted_pixel_data.sort()
        if rescale_option == RescaleIlluminationFunction.YES.value:
            idx = int(sorted_pixel_data.shape[0] * ROBUST_FACTOR)
            robust_minimum = sorted_pixel_data[idx]
            pixel_data = pixel_data.copy()
            pixel_data[pixel_data < robust_minimum] = robust_minimum
        elif rescale_option == RescaleIlluminationFunction.MEDIAN.value:
            idx = int(sorted_pixel_data.shape[0] / 2)
            robust_minimum = sorted_pixel_data[idx]
        else: 
            raise ValueError(f"Unknown rescale option: {rescale_option}")
        if robust_minimum == 0:
            return pixel_data
        return pixel_data / robust_minimum

    if image_pixel_data.ndim == 2:
        output_pixels = scaling_fn_2d(image_pixel_data)
    else:
        output_pixels = numpy.dstack(
            [scaling_fn_2d(x) for x in image_pixel_data.transpose(2, 0, 1)]
        )
    return output_pixels
