import centrosome.smooth
import centrosome.cpmorphology
import scipy.ndimage
import numpy
from typing import Any, Dict, Optional, Tuple, Annotated
from pydantic import Field, validate_call, ConfigDict
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
        pixel_data:             Annotated[Image2D, Field(description="Input image for dilation")],
        mask:                   Annotated[Image2DMask, Field(description="Input image mask")],
        object_dilation_radius: Annotated[int, Field(description="Radius for the circular Gaussian dilation kernel")],
    ) -> Image2D:
    """Return pixel data dilated with a circular Gaussian kernel.

    This filter spreads the boundaries of cells, effectively "dilating" them.

    Args:
        pixel_data: Input image for dilation.
        mask: Input image mask.
        object_dilation_radius: Radius for the circular Gaussian dilation kernel.

    Returns:
        Dilated pixel data of same shape as input.
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
        pixel_data:         Annotated[Image2D, Field(description="Input image for averaging")],
        mask:               Annotated[Optional[Image2DMask], Field(description="Input image mask, or None if no mask")],
        intensity_choice:   Annotated[IntensityChoice, Field(description="'Regular' uses per-pixel intensity; 'Background' finds block minima")],
        smoothing_method:   Annotated[SmoothingMethod, Field(description="Smoothing method; Splines triggers the Regular code path")],
        block_size:         Annotated[int, Field(description="Block side length in pixels for Background mode")],
    ) -> Image2D:
    """Create a version of the image appropriate for averaging.

    For Regular or Splines mode: zeros out masked pixels (if any) and
    returns the result. For Background mode: finds the minimum pixel
    intensity within blocks and returns a block-minimum image.

    Args:
        pixel_data: Input image for averaging.
        mask: Input image mask, or None if no mask.
        intensity_choice: 'Regular' uses per-pixel intensity; 'Background'
            finds block minima.
        smoothing_method: Smoothing method; Splines triggers the Regular
            code path.
        block_size: Block side length in pixels for Background mode.

    Returns:
        Preprocessed image suitable for accumulation.
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
        preprocessed_pixel_data:    Annotated[numpy.ndarray, Field(description="Preprocessed image from preprocess_image_for_averaging, shape (H, W) or (H, W, C)")],
        mask:                       Annotated[Optional[numpy.ndarray], Field(description="Input image mask, or None if no mask")],
) -> Dict[str, Any]:
    """Initialize the illumination accumulation state from the first image.

    Creates a zeroed image_sum and mask_count, then accumulates the
    first (already preprocessed) image into them.

    Args:
        preprocessed_pixel_data: Preprocessed image from
            preprocess_image_for_averaging, shape (H, W) or (H, W, C).
        mask: Input image mask, or None if no mask.

    Returns:
        Initial accumulation state with image_sum and mask_count.
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
        preprocessed_pixel_data: Annotated[numpy.ndarray, Field(description="Preprocessed image from preprocess_image_for_averaging, shape (H, W) or (H, W, C)")],
        mask:                   Annotated[Optional[numpy.ndarray], Field(description="Input image mask, or None if no mask")],
        state:                  Annotated[Dict[str, Any], Field(description="Existing accumulation state dict (mutated in place)")],
) -> Dict[str, Any]:
    """Accumulate a preprocessed image into the illumination state.

    Args:
        preprocessed_pixel_data: Preprocessed image from
            preprocess_image_for_averaging, shape (H, W) or (H, W, C).
        mask: Input image mask, or None if no mask.
        state: Existing accumulation state dict (mutated in place).

    Returns:
        Updated accumulation state (same object as input).
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
        state: Annotated[Dict[str, Any], Field(description="Accumulation state from initialize/accumulate functions")],
) -> Tuple[Image2D, Image2DMask]:
    """Compute the average illumination image from the accumulated state.

    Args:
        state: Accumulation state from initialize/accumulate functions.

    Returns:
        Tuple of (average pixel data, boolean mask where at least one
        image contributed).
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
        image_pixel_data: Annotated[Image2D, Field(description="Pixel data of image to smooth")],
        image_mask: Annotated[Image2DMask, Field(description="Input image mask; True = valid pixel")],
        smoothing_method: Annotated[SmoothingMethod, Field(description="Smoothing method to apply")],
        automatic_object_width: Annotated[Optional[SmoothingFilterSize], Field(description="Method to calculate smoothing filter size (Automatic, Object size, or Manually)")],
        size_of_smoothing_filter: Annotated[Optional[int], Field(description="Manual smoothing filter size in pixels")],
        object_width: Annotated[Optional[int], Field(description="Approximate object diameter in pixels for filter size calculation")],
        image_shape: Annotated[Optional[Tuple[int, ...]], Field(description="Shape of the original image (H, W)")],
        automatic_splines: Annotated[bool, Field(description="Whether to automatically calculate spline parameters")],
        spline_bg_mode: Annotated[Optional[SplineBackgroundMode], Field(description="Background mode for spline fitting (auto, dark, bright, or gray)")],
        spline_points: Annotated[Optional[int], Field(description="Number of spline control points in the grid")],
        spline_threshold: Annotated[Optional[float], Field(description="Std-dev cutoff for background pixel classification")],
        spline_convergence: Annotated[Optional[float], Field(description="Residual value fraction for convergence criterion")],
        spline_maximum_iterations: Annotated[Optional[int], Field(description="Maximum number of spline fitting iterations")],
        spline_rescale: Annotated[Optional[float], Field(description="Image resampling factor for spline computation")],
    ) -> Image2D:
    """Return an image that is smoothed according to the settings.

    Args:
        image_pixel_data: Pixel data of image to smooth.
        image_mask: Input image mask; True = valid pixel.
        smoothing_method: Smoothing method to apply.
        automatic_object_width: Method to calculate smoothing filter size
            (Automatic, Object size, or Manually).
        size_of_smoothing_filter: Manual smoothing filter size in pixels.
        object_width: Approximate object diameter in pixels for filter
            size calculation.
        image_shape: Shape of the original image (H, W).
        automatic_splines: Whether to automatically calculate spline
            parameters.
        spline_bg_mode: Background mode for spline fitting (auto, dark,
            bright, or gray).
        spline_points: Number of spline control points in the grid.
        spline_threshold: Std-dev cutoff for background pixel
            classification.
        spline_convergence: Residual value fraction for convergence
            criterion.
        spline_maximum_iterations: Maximum number of spline fitting
            iterations.
        spline_rescale: Image resampling factor for spline computation.

    Returns:
        Smoothed pixel data.
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
        image_pixel_data:   Annotated[Image2D, Field(description="Pixel data of the illumination function to rescale")],
        image_mask:         Annotated[Optional[Image2DMask], Field(description="Input image mask, or None if no mask")],
        rescale_option:     Annotated[RescaleIlluminationFunction, Field(description="Rescaling method: Yes (robust minimum), No (skip), or Median")],
    ) -> Image2D:
    """Return an image that is rescaled according to the settings.

    Args:
        image_pixel_data: Pixel data of the illumination function to
            rescale.
        image_mask: Input image mask, or None if no mask.
        rescale_option: Rescaling method: Yes (robust minimum), No
            (skip), or Median.

    Returns:
        Rescaled pixel data.
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
