import centrosome.smooth
import centrosome.cpmorphology
import scipy.ndimage
import numpy
import skimage.filters
import centrosome.bg_compensate
import centrosome.filter
from typing import Optional, Tuple

from cellprofiler_library.opts.correctilluminationcalculate import (
    SmoothingFilterSize,
    IntensityChoice,
    SmoothingMethod,
    SplineBackgroundMode,
    RescaleIlluminationFunction
)

ROBUST_FACTOR = 0.02  # For rescaling, take 2nd percentile value

def apply_dilation(
        image, 
        object_dilation_radius,
    ):
    """Return an image that is dilated according to the settings

    image - an instance of cpimage.Image

    returns another instance of cpimage.Image
    """
    #
    # This filter is designed to spread the boundaries of cells
    # and this "dilates" the cells
    #
    kernel = centrosome.smooth.circular_gaussian_kernel(
        object_dilation_radius, object_dilation_radius * 3
    )

    def fn(image):
        return scipy.ndimage.convolve(image, kernel, mode="constant", cval=0)

    if image.pixel_data.ndim == 2:
        dilated_pixels = centrosome.smooth.smooth_with_function_and_mask(
            image.pixel_data, fn, image.mask
        )
    else:
        dilated_pixels = numpy.dstack(
            [
                centrosome.smooth.smooth_with_function_and_mask(
                    x, fn, image.mask
                )
                for x in image.pixel_data.transpose(2, 0, 1)
            ]
        )
    return dilated_pixels

def get_smoothing_filter_size(
        automatic_object_width: SmoothingFilterSize, 
        smoothing_filter_size: Optional[int], # default to 10
        object_width: Optional[int], # default to 10 
        image_shape: Optional[Tuple[int, ...]]
    ) -> float:
    """Return the smoothing filter size based on the settings and image size"""
    filter_size = None
    if automatic_object_width == SmoothingFilterSize.MANUALLY.value:
        assert smoothing_filter_size is not None, "Manual smoothing filter size must be provided"
        # Convert from full-width at half-maximum to standard deviation
        # (or so says CPsmooth.m)
        filter_size = smoothing_filter_size
    elif automatic_object_width == SmoothingFilterSize.OBJECT_SIZE.value:
        assert object_width is not None, "Object width must be provided"
        filter_size =  object_width * 2.35 / 3.5
    elif automatic_object_width == SmoothingFilterSize.AUTOMATIC.value:
        assert image_shape is not None, "Image shape must be provided"
        filter_size = min(30, float(numpy.max(image_shape)) / 40.0)
    else:
        raise ValueError(f"Unknown smoothing filter size: {automatic_object_width}")
    return filter_size

# intensity_choice = self.intensity_choice.value
# smoothing_method = self.smoothing_method.value
# block_size = self.block_size.value
def preprocess_image_for_averaging(
        orig_image,
        intensity_choice: IntensityChoice,
        smoothing_method: SmoothingMethod,
        block_size: int,
    ):
    """Create a version of the image appropriate for averaging

    """
    pixels = orig_image.pixel_data
    if intensity_choice == IntensityChoice.REGULAR.value or smoothing_method == SmoothingMethod.SPLINES.value:
        if orig_image.has_mask:
            if pixels.ndim == 2:
                pixels[~orig_image.mask] = 0
            else:
                pixels[~orig_image.mask, :] = 0
            avg_image = pixels
        else:
            avg_image = orig_image
    else:
        # For background, we create a labels image using the block
        # size and find the minimum within each block.
        labels, indexes = centrosome.cpmorphology.block(
            pixels.shape[:2], (block_size, block_size)
        )
        if orig_image.has_mask:
            labels[~orig_image.mask] = -1

        min_block = numpy.zeros(pixels.shape)
        if pixels.ndim == 2:
            minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.minimum(pixels, labels, indexes)
            )
            min_block[labels != -1] = minima[labels[labels != -1]]
        else:
            for i in range(pixels.shape[2]):
                minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.minimum(pixels[:, :, i], labels, indexes)
                )
                min_block[labels != -1, i] = minima[labels[labels != -1]]
        avg_image = min_block
    return avg_image

# image_pixel_data = image.pixel_data
# image_mask = image.mask
def apply_smoothing(
        image_pixel_data,
        image_mask,
        # smooth filter args
        smoothing_method: SmoothingMethod,
        # smooth filter size args
        automatic_object_width: SmoothingFilterSize, 
        size_of_smoothing_filter: Optional[int], # default to 10
        object_width: Optional[int], # default to 10 
        image_shape: Optional[Tuple[int, ...]],
        # spline args
        automatic_splines: bool,
        spline_bg_mode: SplineBackgroundMode,
        spline_points: int,
        spline_threshold: float,
        spline_convergence: float,
        spline_maximum_iterations: int,
        spline_rescale: float,
    ):
    """Return an image that is smoothed according to the settings

    image - an instance of cpimage.Image containing the pixels to analyze
    orig_image - the ancestor source image or None if ambiguous
    returns another instance of cpimage.Image
    """
    # if self.smoothing_method == SmoothingMethod.NONE.value:
    #     return image
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
    # output_image = Image(output_pixels, parent_image=orig_image)
    return output_pixels

def smooth_plane(
        pixel_data, 
        mask,
        smoothing_method: SmoothingMethod,
        automatic_object_width: SmoothingFilterSize, 
        size_of_smoothing_filter: Optional[int], # default to 10
        object_width: Optional[int], # default to 10 
        image_shape: Optional[Tuple[int, ...]],
        automatic_splines: bool,
        spline_bg_mode: SplineBackgroundMode,
        spline_points: int,
        spline_threshold: float,
        spline_convergence: float,
        spline_maximum_iterations: int,
        spline_rescale: float,
    ):
    """Smooth one 2-d color plane of an image"""

    sigma = get_smoothing_filter_size(automatic_object_width, size_of_smoothing_filter, object_width, image_shape) / 2.35 # What's up with 2.35?
    if smoothing_method == SmoothingMethod.FIT_POLYNOMIAL.value:
        output_pixels = centrosome.smooth.fit_polynomial(pixel_data, mask)
    elif smoothing_method == SmoothingMethod.GAUSSIAN_FILTER.value:
        #
        # Smoothing with the mask is good, even if there's no mask
        # because the mechanism undoes the edge effects that are introduced
        # by any choice of how to deal with border effects.
        #
        def fn(image):
            return scipy.ndimage.gaussian_filter(
                image, sigma, mode="constant", cval=0
            )

        output_pixels = centrosome.smooth.smooth_with_function_and_mask(
            pixel_data, fn, mask
        )
    elif smoothing_method == SmoothingMethod.MEDIAN_FILTER.value:
        filter_sigma = max(1, int(sigma + 0.5))
        strel = centrosome.cpmorphology.strel_disk(filter_sigma)
        rescaled_pixel_data = pixel_data * 65535
        rescaled_pixel_data = rescaled_pixel_data.astype(numpy.uint16)
        rescaled_pixel_data *= mask
        output_pixels = skimage.filters.median(rescaled_pixel_data, strel, behavior="rank")
    elif smoothing_method == SmoothingMethod.TO_AVERAGE.value:
        mean = numpy.mean(pixel_data[mask])
        output_pixels = numpy.ones(pixel_data.shape, pixel_data.dtype) * mean
    elif smoothing_method == SmoothingMethod.SPLINES.value:
        output_pixels = smooth_with_splines(
            pixel_data, 
            mask,
            automatic_splines,
            spline_bg_mode,
            spline_points,
            spline_threshold,
            spline_convergence,
            spline_maximum_iterations,
            spline_rescale,
        )
    elif smoothing_method == SmoothingMethod.CONVEX_HULL.value:
        output_pixels = smooth_with_convex_hull(pixel_data, mask)
    else:
        raise ValueError(
            "Unimplemented smoothing method: %s:" % smoothing_method.value
        )
    return output_pixels

def smooth_with_convex_hull(pixel_data, mask):
    """Use the convex hull transform to smooth the image"""
    #
    # Apply an erosion, then the transform, then a dilation, heuristically
    # to ignore little spikey noisy things.
    #
    image = centrosome.cpmorphology.grey_erosion(pixel_data, 2, mask)
    image = centrosome.filter.convex_hull_transform(image, mask=mask)
    image = centrosome.cpmorphology.grey_dilation(image, 2, mask)
    return image

# automatic_splines = self.automatic_splines
# spline_bg_mode = self.spline_bg_mode.value
# spline_points = self.spline_points.value
# spline_threshold = self.spline_threshold.value
# spline_convergence = self.spline_convergence.value
# spline_maximum_iterations = self.spline_maximum_iterations.value
# spline_rescale = self.spline_rescale.value
def smooth_with_splines(
        pixel_data, 
        mask,
        automatic_splines: bool,
        # centrosome.bg_compensate args
        spline_bg_mode: SplineBackgroundMode,
        spline_points: int,
        spline_threshold: float,
        spline_convergence: float,
        spline_maximum_iterations: int,
        spline_rescale: float,
    ):
    if automatic_splines:
        # Make the image 200 pixels long on its shortest side
        shortest_side = min(pixel_data.shape)
        if shortest_side < 200:
            scale = 1
        else:
            scale = float(shortest_side) / 200
        result = centrosome.bg_compensate.backgr(pixel_data, mask, scale=scale)
    else:
        mode = spline_bg_mode
        spline_points = spline_points
        threshold = spline_threshold
        convergence = spline_convergence
        iterations = spline_maximum_iterations
        rescale = spline_rescale
        result = centrosome.bg_compensate.backgr(
            pixel_data,
            mask,
            mode=mode,
            thresh=threshold, # TODO: #5129 centrosome expects int but CellProfiler sends a float
            splinepoints=spline_points,
            scale=rescale, # TODO: #5129 centrosome expects int but CellProfiler sends a float
            maxiter=iterations,
            convergence=convergence,
        )
    #
    # The result is a fit to the background intensity, but we
    # want to normalize the intensity by subtraction, leaving
    # the mean intensity alone.
    #
    mean_intensity = numpy.mean(result[mask])
    result[mask] -= mean_intensity
    return result

# rescale_option = self.rescale_option.value
def apply_scaling(
        image, 
        rescale_option: RescaleIlluminationFunction,
        orig_image=None
    ):
    """Return an image that is rescaled according to the settings

    image - an instance of cpimage.Image
    returns another instance of cpimage.Image
    """
    if rescale_option == "No":
        return image

    def scaling_fn_2d(pixel_data):
        if image.has_mask:
            sorted_pixel_data = pixel_data[(pixel_data > 0) & image.mask]
        else:
            sorted_pixel_data = pixel_data[pixel_data > 0]
        if sorted_pixel_data.shape[0] == 0:
            return pixel_data
        sorted_pixel_data.sort()
        if rescale_option == "Yes":
            idx = int(sorted_pixel_data.shape[0] * ROBUST_FACTOR)
            robust_minimum = sorted_pixel_data[idx]
            pixel_data = pixel_data.copy()
            pixel_data[pixel_data < robust_minimum] = robust_minimum
        elif rescale_option == RescaleIlluminationFunction.MEDIAN.value:
            idx = int(sorted_pixel_data.shape[0] / 2)
            robust_minimum = sorted_pixel_data[idx]
        if robust_minimum == 0:
            return pixel_data
        return pixel_data / robust_minimum

    if image.pixel_data.ndim == 2:
        output_pixels = scaling_fn_2d(image.pixel_data)
    else:
        output_pixels = numpy.dstack(
            [scaling_fn_2d(x) for x in image.pixel_data.transpose(2, 0, 1)]
        )
    return output_pixels