from functools import partial
from collections.abc import Callable
import centrosome.smooth
import centrosome.cpmorphology
import scipy.ndimage
import numpy
from numpy.typing import NDArray
from typing import Optional, Tuple, Annotated
from typing_extensions import TypeAlias
from pydantic import Field, validate_call, ConfigDict, BaseModel
from cellprofiler_library.types import Image2D, Image2DMask
from cellprofiler_library.opts.correctilluminationcalculate import (
    SmoothingFilterSize,
    IntensityChoice,
    SmoothingMethod,
    SplineBackgroundMode,
    RescaleIlluminationFunction,
)

from cellprofiler_library.functions.image_processing import smooth_plane

ROBUST_FACTOR = 0.02  # For rescaling, take 2nd percentile value

IlluminationAccumulate: TypeAlias = Callable[
    [
        Image2D, # image
        Optional[Image2DMask], # mask
    ],
    "IlluminationAccumulator"
]
IlluminationFinalize: TypeAlias = Callable[
    [],
    Tuple[Image2D, Image2D, Image2D, Image2DMask] # (output, dilated, averaged, mask)
]

class IlluminationAccumulator(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    accumulate: IlluminationAccumulate
    finalize: IlluminationFinalize


# Field descriptions shared by correctilluminationcalculate() and _accumulate_illumination_image(),
# which both need to re-thread the same settings into the IlluminationAccumulator they return.
_D_INTENSITY_CHOICE = "'Regular' uses per-pixel intensity; 'Background' finds block minima"
_D_SMOOTHING_METHOD = "Smoothing method; also used while preprocessing when Splines is selected"
_D_BLOCK_SIZE = "Block side length in pixels for Background mode"
_D_DILATE_OBJECTS = "Whether to dilate the averaged image with a circular Gaussian kernel"
_D_OBJECT_DILATION_RADIUS = "Radius for the circular Gaussian dilation kernel"
_D_AUTOMATIC_OBJECT_WIDTH = "Method to calculate smoothing filter size (Automatic, Object size, or Manually)"
_D_SIZE_OF_SMOOTHING_FILTER = "Manual smoothing filter size in pixels"
_D_OBJECT_WIDTH = "Approximate object diameter in pixels for filter size calculation"
_D_AUTOMATIC_SPLINES = "Whether to automatically calculate spline parameters"
_D_SPLINE_BG_MODE = "Background mode for spline fitting (auto, dark, bright, or gray)"
_D_SPLINE_POINTS = "Number of spline control points in the grid"
_D_SPLINE_THRESHOLD = "Std-dev cutoff for background pixel classification"
_D_SPLINE_CONVERGENCE = "Residual value fraction for convergence criterion"
_D_SPLINE_MAXIMUM_ITERATIONS = "Maximum number of spline fitting iterations"
_D_SPLINE_RESCALE = "Image resampling factor for spline computation"
_D_RESCALE_OPTION = "Rescaling method: Yes (robust minimum), No (skip), or Median"

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def correctilluminationcalculate(
        image:                      Annotated[Image2D, Field(description="The pixel data of the first image to accumulate")],
        mask:                       Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")],
        intensity_choice:           Annotated[IntensityChoice, Field(description=_D_INTENSITY_CHOICE)],
        smoothing_method:           Annotated[SmoothingMethod, Field(description=_D_SMOOTHING_METHOD)],
        block_size:                 Annotated[int, Field(description=_D_BLOCK_SIZE)],
        dilate_objects:             Annotated[bool, Field(description=_D_DILATE_OBJECTS)],
        object_dilation_radius:     Annotated[int, Field(description=_D_OBJECT_DILATION_RADIUS)],
        automatic_object_width:     Annotated[Optional[SmoothingFilterSize], Field(description=_D_AUTOMATIC_OBJECT_WIDTH)],
        size_of_smoothing_filter:   Annotated[Optional[int], Field(description=_D_SIZE_OF_SMOOTHING_FILTER)],
        object_width:               Annotated[Optional[int], Field(description=_D_OBJECT_WIDTH)],
        automatic_splines:          Annotated[bool, Field(description=_D_AUTOMATIC_SPLINES)],
        spline_bg_mode:             Annotated[Optional[SplineBackgroundMode], Field(description=_D_SPLINE_BG_MODE)],
        spline_points:              Annotated[Optional[int], Field(description=_D_SPLINE_POINTS)],
        spline_threshold:           Annotated[Optional[float], Field(description=_D_SPLINE_THRESHOLD)],
        spline_convergence:         Annotated[Optional[float], Field(description=_D_SPLINE_CONVERGENCE)],
        spline_maximum_iterations:  Annotated[Optional[int], Field(description=_D_SPLINE_MAXIMUM_ITERATIONS)],
        spline_rescale:             Annotated[Optional[float], Field(description=_D_SPLINE_RESCALE)],
        rescale_option:             Annotated[RescaleIlluminationFunction, Field(description=_D_RESCALE_OPTION)],
    ) -> IlluminationAccumulator:
    """
    Preprocesses and accumulates the first image, then returns an
    IlluminationAccumulator whose `.accumulate(image, mask)` folds in
    further images and whose `.finalize()` runs the full
    average -> dilate -> smooth -> rescale pipeline and returns
    (output, dilated, averaged, mask).

    Args:
        image: The pixel data of the first image to accumulate.
        mask: The mask of the image (True = valid). If None, all pixels
            are valid.
        intensity_choice, smoothing_method, block_size: see
            _preprocess_image_for_averaging.
        dilate_objects, object_dilation_radius: see _apply_dilation.
        automatic_object_width, size_of_smoothing_filter, object_width,
            automatic_splines, spline_bg_mode, spline_points,
            spline_threshold, spline_convergence,
            spline_maximum_iterations, spline_rescale: see _apply_smoothing.
        rescale_option: see _apply_scaling.

    Returns:
        An IlluminationAccumulator wrapping the initial accumulation.
    """
    preprocessed = _preprocess_image_for_averaging(image, mask, intensity_choice, smoothing_method, block_size)
    image_sum = numpy.zeros(preprocessed.shape, preprocessed.dtype)
    mask_count = numpy.zeros(preprocessed.shape[:2], numpy.int32)
    _mut_accumulate(preprocessed, mask, image_sum, mask_count)

    return IlluminationAccumulator(
        accumulate = partial(
            _accumulate_illumination_image,
            intensity_choice=intensity_choice,
            smoothing_method=smoothing_method,
            block_size=block_size,
            image_sum=image_sum,
            mask_count=mask_count,
            dilate_objects=dilate_objects,
            object_dilation_radius=object_dilation_radius,
            automatic_object_width=automatic_object_width,
            size_of_smoothing_filter=size_of_smoothing_filter,
            object_width=object_width,
            automatic_splines=automatic_splines,
            spline_bg_mode=spline_bg_mode,
            spline_points=spline_points,
            spline_threshold=spline_threshold,
            spline_convergence=spline_convergence,
            spline_maximum_iterations=spline_maximum_iterations,
            spline_rescale=spline_rescale,
            rescale_option=rescale_option,
        ),
        finalize = partial(
            _calculate_illumination_images,
            image_sum=image_sum,
            mask_count=mask_count,
            dilate_objects=dilate_objects,
            object_dilation_radius=object_dilation_radius,
            smoothing_method=smoothing_method,
            automatic_object_width=automatic_object_width,
            size_of_smoothing_filter=size_of_smoothing_filter,
            object_width=object_width,
            automatic_splines=automatic_splines,
            spline_bg_mode=spline_bg_mode,
            spline_points=spline_points,
            spline_threshold=spline_threshold,
            spline_convergence=spline_convergence,
            spline_maximum_iterations=spline_maximum_iterations,
            spline_rescale=spline_rescale,
            rescale_option=rescale_option,
        ),
    )

def _accumulate_illumination_image(
        image:      Annotated[Image2D, Field(description="The pixel data of the image to accumulate")],
        mask:       Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")],
        *,
        intensity_choice:           Annotated[IntensityChoice, Field(description=_D_INTENSITY_CHOICE)],
        smoothing_method:           Annotated[SmoothingMethod, Field(description=_D_SMOOTHING_METHOD)],
        block_size:                 Annotated[int, Field(description=_D_BLOCK_SIZE)],
        image_sum:                  Annotated[Image2D, Field(description="Running sum of preprocessed images (mutated in place)")],
        mask_count:                 Annotated[NDArray[numpy.int_], Field(description="Running per-pixel count of contributing images (mutated in place)")],
        dilate_objects:             Annotated[bool, Field(description=_D_DILATE_OBJECTS)],
        object_dilation_radius:     Annotated[int, Field(description=_D_OBJECT_DILATION_RADIUS)],
        automatic_object_width:     Annotated[Optional[SmoothingFilterSize], Field(description=_D_AUTOMATIC_OBJECT_WIDTH)],
        size_of_smoothing_filter:   Annotated[Optional[int], Field(description=_D_SIZE_OF_SMOOTHING_FILTER)],
        object_width:               Annotated[Optional[int], Field(description=_D_OBJECT_WIDTH)],
        automatic_splines:          Annotated[bool, Field(description=_D_AUTOMATIC_SPLINES)],
        spline_bg_mode:             Annotated[Optional[SplineBackgroundMode], Field(description=_D_SPLINE_BG_MODE)],
        spline_points:              Annotated[Optional[int], Field(description=_D_SPLINE_POINTS)],
        spline_threshold:           Annotated[Optional[float], Field(description=_D_SPLINE_THRESHOLD)],
        spline_convergence:         Annotated[Optional[float], Field(description=_D_SPLINE_CONVERGENCE)],
        spline_maximum_iterations:  Annotated[Optional[int], Field(description=_D_SPLINE_MAXIMUM_ITERATIONS)],
        spline_rescale:             Annotated[Optional[float], Field(description=_D_SPLINE_RESCALE)],
        rescale_option:             Annotated[RescaleIlluminationFunction, Field(description=_D_RESCALE_OPTION)],
    ) -> IlluminationAccumulator:
    """
    Accumulate an image into the illumination sum.

    Only intensity_choice/smoothing_method/block_size/image_sum/mask_count
    are used directly here (to preprocess and fold in `image`); the
    remaining settings are unused by this step but re-threaded into the
    IlluminationAccumulator this function returns, so that `.finalize()`
    keeps working after any number of `.accumulate()` calls.

    Returns:
        An IlluminationAccumulator wrapping the updated accumulation.
    """
    preprocessed = _preprocess_image_for_averaging(image, mask, intensity_choice, smoothing_method, block_size)
    _mut_accumulate(preprocessed, mask, image_sum, mask_count)

    return IlluminationAccumulator(
        accumulate = partial(
            _accumulate_illumination_image,
            intensity_choice=intensity_choice,
            smoothing_method=smoothing_method,
            block_size=block_size,
            image_sum=image_sum,
            mask_count=mask_count,
            dilate_objects=dilate_objects,
            object_dilation_radius=object_dilation_radius,
            automatic_object_width=automatic_object_width,
            size_of_smoothing_filter=size_of_smoothing_filter,
            object_width=object_width,
            automatic_splines=automatic_splines,
            spline_bg_mode=spline_bg_mode,
            spline_points=spline_points,
            spline_threshold=spline_threshold,
            spline_convergence=spline_convergence,
            spline_maximum_iterations=spline_maximum_iterations,
            spline_rescale=spline_rescale,
            rescale_option=rescale_option,
        ),
        finalize = partial(
            _calculate_illumination_images,
            image_sum=image_sum,
            mask_count=mask_count,
            dilate_objects=dilate_objects,
            object_dilation_radius=object_dilation_radius,
            smoothing_method=smoothing_method,
            automatic_object_width=automatic_object_width,
            size_of_smoothing_filter=size_of_smoothing_filter,
            object_width=object_width,
            automatic_splines=automatic_splines,
            spline_bg_mode=spline_bg_mode,
            spline_points=spline_points,
            spline_threshold=spline_threshold,
            spline_convergence=spline_convergence,
            spline_maximum_iterations=spline_maximum_iterations,
            spline_rescale=spline_rescale,
            rescale_option=rescale_option,
        ),
    )

def _calculate_illumination_images(
        *,
        image_sum:                  Annotated[Image2D, Field(description="Running sum of preprocessed images")],
        mask_count:                 Annotated[NDArray[numpy.int_], Field(description="Running per-pixel count of contributing images")],
        dilate_objects:             Annotated[bool, Field(description=_D_DILATE_OBJECTS)],
        object_dilation_radius:     Annotated[int, Field(description=_D_OBJECT_DILATION_RADIUS)],
        smoothing_method:           Annotated[SmoothingMethod, Field(description=_D_SMOOTHING_METHOD)],
        automatic_object_width:     Annotated[Optional[SmoothingFilterSize], Field(description=_D_AUTOMATIC_OBJECT_WIDTH)],
        size_of_smoothing_filter:   Annotated[Optional[int], Field(description=_D_SIZE_OF_SMOOTHING_FILTER)],
        object_width:               Annotated[Optional[int], Field(description=_D_OBJECT_WIDTH)],
        automatic_splines:          Annotated[bool, Field(description=_D_AUTOMATIC_SPLINES)],
        spline_bg_mode:             Annotated[Optional[SplineBackgroundMode], Field(description=_D_SPLINE_BG_MODE)],
        spline_points:              Annotated[Optional[int], Field(description=_D_SPLINE_POINTS)],
        spline_threshold:           Annotated[Optional[float], Field(description=_D_SPLINE_THRESHOLD)],
        spline_convergence:         Annotated[Optional[float], Field(description=_D_SPLINE_CONVERGENCE)],
        spline_maximum_iterations:  Annotated[Optional[int], Field(description=_D_SPLINE_MAXIMUM_ITERATIONS)],
        spline_rescale:             Annotated[Optional[float], Field(description=_D_SPLINE_RESCALE)],
        rescale_option:             Annotated[RescaleIlluminationFunction, Field(description=_D_RESCALE_OPTION)],
    ) -> Tuple[Image2D, Image2D, Image2D, Image2DMask]:
    """Run the full average -> dilate -> smooth -> rescale pipeline.

    Returns:
        Tuple of (output pixel data, dilated pixel data, averaged pixel
        data, mask). All three images share the same mask: none of
        dilation, smoothing, or rescaling change which pixels are valid.
    """
    # TODO: 5129 - consider skipping average on EACH (as in cp4)
    avg_pixel_data, mask = _calculate_average_image(image_sum, mask_count)

    if dilate_objects:
        dilated_pixel_data = _apply_dilation(avg_pixel_data, mask, object_dilation_radius)
    else:
        dilated_pixel_data = avg_pixel_data

    if smoothing_method != SmoothingMethod.NONE.value:
        smoothed_pixel_data = _apply_smoothing(
            image_pixel_data=dilated_pixel_data,
            image_mask=mask,
            smoothing_method=smoothing_method,
            automatic_object_width=automatic_object_width,
            size_of_smoothing_filter=size_of_smoothing_filter,
            object_width=object_width,
            image_shape=dilated_pixel_data.shape[:2],
            automatic_splines=automatic_splines,
            spline_bg_mode=spline_bg_mode,
            spline_points=spline_points,
            spline_threshold=spline_threshold,
            spline_convergence=spline_convergence,
            spline_maximum_iterations=spline_maximum_iterations,
            spline_rescale=spline_rescale,
        )
    else:
        smoothed_pixel_data = dilated_pixel_data

    if rescale_option != RescaleIlluminationFunction.NO.value:
        output_pixel_data = _apply_scaling(
            image_pixel_data=smoothed_pixel_data,
            image_mask=mask,
            rescale_option=rescale_option,
        )
    else:
        output_pixel_data = smoothed_pixel_data

    return output_pixel_data, dilated_pixel_data, avg_pixel_data, mask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _preprocess_image_for_averaging(
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

def _calculate_average_image(
        image_sum:  Annotated[Image2D, Field(description="Running sum of preprocessed images")],
        mask_count: Annotated[NDArray[numpy.int_], Field(description="Running per-pixel count of contributing images")],
    ) -> Tuple[Image2D, Image2DMask]:
    """Compute the average illumination image from the accumulated sums.

    Returns:
        Tuple of (average pixel data, boolean mask where at least one
        image contributed).
    """
    pixel_data = numpy.zeros(image_sum.shape, image_sum.dtype)
    mask = mask_count > 0
    if pixel_data.ndim == 2:
        pixel_data[mask] = image_sum[mask] / mask_count[mask]
    else:
        for i in range(pixel_data.shape[2]):
            pixel_data[mask, i] = image_sum[mask, i] / mask_count[mask]
    return pixel_data, mask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _apply_scaling(
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

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def _apply_smoothing(
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
def _apply_dilation(
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
    # scipy.ndimage.convolve upconverts to float64 regardless of input dtype;
    # downstream smoothing methods (e.g. Splines) are sensitive to the
    # resulting sub-epsilon floating point noise, so match the input's
    # precision rather than silently widening it.
    return dilated_pixels.astype(pixel_data.dtype)

# NOTE: _mut prefix means the function mutates input numpy arrays, rather than returning new arrays
def _mut_accumulate(
        preprocessed_pixel_data: Image2D,
        mask: Optional[Image2DMask],
        image_sum: Image2D, # mutated
        mask_count: NDArray[numpy.int_], # mutated
    ):
    mask_count[mask] += 1
    if mask is not None:
        if image_sum.ndim == 2:
            image_sum[mask] += preprocessed_pixel_data[mask]
        else:
            image_sum[mask, :] += preprocessed_pixel_data[mask, :]
    else:
        image_sum += preprocessed_pixel_data
