import numpy
from numpy import isscalar as _isscalar
import skimage.color
import skimage.morphology
import skimage.segmentation
import skimage.util
from skimage.util import invert as _invert
import skimage.transform
import skimage
import skimage.restoration
import centrosome
import centrosome.threshold
import centrosome.filter
import scipy
import scipy.interpolate
import matplotlib
import math
from numpy.typing import NDArray
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from typing import Any, Optional, Tuple, Callable, Union, List, cast, Dict, TypeVar
from numpy.typing import NDArray
from typing import Any, Optional, Tuple, Callable, Union, List, TypeVar
from skimage.restoration import denoise_bilateral
from centrosome.filter import median_filter as _median_filter
from centrosome.filter import circular_average_filter as _circular_average_filter
from centrosome.smooth import fit_polynomial as _fit_polynomial
from centrosome.smooth import smooth_with_function_and_mask as _smooth_with_function_and_mask
from centrosome.cpmorphology import get_line_pts, all_connected_components
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.ndimage import mean as mean_of_labels
import scipy.ndimage as scind
import scipy.sparse
from centrosome.filter import stretch
from scipy.fftpack import fft2, ifft2
from cellprofiler_library.types import (
    ImageGrayscale, 
    ImageGrayscaleMask, 
    Image2DColor, 
    Image2DGrayscale, Image2DGrayscaleMask, 
    ImageAny, ImageAnyMask, 
    ObjectSegmentation, 
    Image2D, 
    Image2DMask, StructuringElement, ObjectLabelSet, ImageColor, ImageBinaryMask, 
    ImageAnyMask, Image2DBinary, ObjectLabel, ImageBinary, Pixel
)
from cellprofiler_library.opts import threshold as Threshold
from cellprofiler_library.opts.enhanceorsuppressfeatures import SpeckleAccuracy, NeuriteMethod
from cellprofiler_library.opts.overlayoutlines import BrightnessMode
from cellprofiler_library.opts.crop import RemovalMethod
from cellprofiler_library.opts.structuring_elements import StructuringElementShape2D, StructuringElementShape3D
from cellprofiler_library.opts.resize import ResizingMethod, DimensionMethod, InterpolationMethod
from cellprofiler_library.opts.imagemath import Operator
from cellprofiler_library.opts.flipandrotate import RotationCoordinateAlignmnet
from cellprofiler_library.opts.enhanceedges import EdgeDirection

invert = cast(Callable[[ImageAny], ImageAny], _invert)
isscalar = cast(Callable[[Optional[ImageAny]], bool], _isscalar)
median_filter_centrosome = cast(Callable[[Image2D, Optional[Image2DMask], float], Image2D], _median_filter)
circular_average_filter = cast(Callable[[Image2D, float, Optional[Image2DMask]], Image2D], _circular_average_filter)
smooth_with_function_and_mask = cast(Callable[[Image2D, Callable[[Image2D], Image2D], Optional[Image2DMask]], Image2D], _smooth_with_function_and_mask)
fit_polynomial = cast(Callable[[Image2D, Optional[Image2DMask], bool], Image2D], _fit_polynomial)

T = TypeVar("T", bound=ImageAny)
MorphImageT = TypeVar("Union[ImageGrayscale, ImageGrayscaleMask]", bound=Union[ImageGrayscale, ImageGrayscaleMask])

def rgb_to_greyscale(image):
    if image.shape[-1] == 4:
        output = skimage.color.rgba2rgb(image)
        return skimage.color.rgb2gray(output)
    else:
        return skimage.color.rgb2gray(image)


def medial_axis(image):
    if image.ndim > 2 and image.shape[-1] in (3, 4):
        raise ValueError("Convert image to grayscale or use medialaxis module")
    if image.ndim > 2 and image.shape[-1] not in (3, 4):
        raise ValueError("Process 3D images plane-wise or use the medialaxis module")
    return skimage.morphology.medial_axis(image)


def enhance_edges_sobel(
        image: Image2DGrayscale, 
        mask: Optional[Image2DGrayscaleMask]=None, 
        direction: EdgeDirection=EdgeDirection.ALL
        ):
    if direction == EdgeDirection.ALL:
        output_pixels = centrosome.filter.sobel(image, mask)
    elif direction == EdgeDirection.HORIZONTAL:
        output_pixels = centrosome.filter.hsobel(image, mask)
    elif direction == EdgeDirection.VERTICAL:
        output_pixels = centrosome.filter.vsobel(image, mask)
    else:
        raise NotImplementedError(f"Unimplemented direction for Sobel: {direction}")
    return output_pixels


def enhance_edges_log(
        image: Image2DGrayscale, 
        mask: Optional[Image2DGrayscaleMask]=None, 
        sigma: float=2.0
        ) -> Image2DGrayscale:
    size = int(sigma * 4) + 1
    output_pixels = centrosome.filter.laplacian_of_gaussian(image, mask, size, sigma)
    return output_pixels


def enhance_edges_prewitt(
        image: Image2DGrayscale, 
        mask: Optional[Image2DGrayscaleMask]=None, 
        direction: EdgeDirection=EdgeDirection.ALL
        ) -> Image2DGrayscale:
    if direction == EdgeDirection.ALL:
        output_pixels = centrosome.filter.prewitt(image, mask)
    elif direction == EdgeDirection.HORIZONTAL:
        output_pixels = centrosome.filter.hprewitt(image, mask)
    elif direction == EdgeDirection.VERTICAL:
        output_pixels = centrosome.filter.vprewitt(image, mask)
    else:
        raise NotImplementedError(f"Unimplemented direction for Prewitt: {direction}")
    return output_pixels


def enhance_edges_canny(
    image: Image2DGrayscale,
    mask: Optional[Image2DGrayscaleMask] = None,
    auto_threshold: bool = True,
    auto_low_threshold: bool = True,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    manual_threshold: float = 0.2,
    threshold_adjustment_factor: float = 1.0,
    ) -> Image2DGrayscale:

    if auto_threshold or auto_low_threshold:
        sobel_image = centrosome.filter.sobel(image)
        low, high = centrosome.otsu.otsu3(sobel_image[mask])
        if auto_threshold:
            high_th = high * threshold_adjustment_factor
        if auto_low_threshold:
            low_th = low * threshold_adjustment_factor
    else:
        low_th = low_threshold
        high_th = manual_threshold

    output_pixels = centrosome.filter.canny(image, mask, sigma, low_th, high_th)
    return output_pixels


def morphology_closing(image, structuring_element=skimage.morphology.disk(1)):
    if structuring_element.ndim == 3 and image.ndim == 2:
        raise ValueError("Cannot apply a 3D structuring element to a 2D image")
    # Check if a 2D structuring element will be applied to a 3D image planewise
    planewise = structuring_element.ndim == 2 and image.ndim == 3
    if planewise:
        output = numpy.zeros_like(image)
        for index, plane in enumerate(image):
            output[index] = skimage.morphology.closing(plane, structuring_element)
        return output
    else:
        return skimage.morphology.closing(image, structuring_element)


def morphology_opening(image, structuring_element=skimage.morphology.disk(1)):
    if structuring_element.ndim == 3 and image.ndim == 2:
        raise ValueError("Cannot apply a 3D structuring element to a 2D image")
    # Check if a 2D structuring element will be applied to a 3D image planewise
    planewise = structuring_element.ndim == 2 and image.ndim == 3
    if planewise:
        output = numpy.zeros_like(image)
        for index, plane in enumerate(image):
            output[index] = skimage.morphology.opening(plane, structuring_element)
        return output
    else:
        return skimage.morphology.opening(image, structuring_element)


def morphological_skeleton_2d(image):
    return skimage.morphology.skeletonize(image)


def morphological_skeleton_3d(image):
    return skimage.morphology.skeletonize_3d(image)


################################################################################
# Morphological Operations Helpers
################################################################################

def get_structuring_element(shape: Union[StructuringElementShape2D, StructuringElementShape3D], size: int) -> StructuringElement:
    return getattr(skimage.morphology, shape.value.lower())(size)

################################################################################
# ErodeImage
################################################################################

def morphology_erosion(image: ImageAny, structuring_element: StructuringElement) -> ImageAny:
    """Apply morphological erosion to an image.
    
    Args:
        image: Input image (2D or 3D)
        structuring_element: Structuring element for erosion
        
    Returns:
        Eroded image with same dimensions as input
    """
    is_strel_2d = structuring_element.ndim == 2
    is_img_2d = image.ndim == 2
    
    if is_strel_2d and not is_img_2d:
        # Apply 2D structuring element to 3D image planewise
        y_data = numpy.zeros_like(image)
        for index, plane in enumerate(image):
            y_data[index] = skimage.morphology.erosion(plane, structuring_element)
        return y_data
    
    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )
    
    # Apply erosion directly for matching dimensions
    y_data = skimage.morphology.erosion(image, structuring_element)
    return y_data


################################################################################
# DilateImage
################################################################################

def morphology_dilation(image: ImageAny, structuring_element: StructuringElement) -> ImageAny:
    """Apply morphological dilation to an image.
    
    Args:
        image: Input image (2D or 3D)
        structuring_element: Structuring element for dilation
        
    Returns:
        Dilated image with same dimensions as input
    """
    is_strel_2d = structuring_element.ndim == 2
    is_img_2d = image.ndim == 2
    
    if is_strel_2d and not is_img_2d:
        # Apply 2D structuring element to 3D image planewise
        y_data = numpy.zeros_like(image)
        for index, plane in enumerate(image):
            y_data[index] = skimage.morphology.dilation(plane, structuring_element)
        return y_data
    
    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )
    
    # Apply dilation directly for matching dimensions
    y_data = skimage.morphology.dilation(image, structuring_element)
    return y_data

def median_filter(image, window_size, mode):
    return scipy.ndimage.median_filter(image, size=window_size, mode=mode)


def reduce_noise(image, patch_size, patch_distance, cutoff_distance, channel_axis=None):
    denoised = skimage.restoration.denoise_nl_means(
        image=image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=cutoff_distance,
        channel_axis=channel_axis,
        fast_mode=True,
    )
    return denoised


################################################################################
# Resize Functions
################################################################################

def resized_shape(
    im_pixel_data: ImageAny,
    im_dimensions: int,
    size_method: str,
    resizing_factor_x: float,
    resizing_factor_y: float,
    resizing_factor_z: Optional[float],
    use_manual_or_image: DimensionMethod,
    specific_width: Optional[int],
    specific_height: Optional[int],
    specific_planes: Optional[int],
    reference_image_shape: Optional[Tuple[int, ...]] = None,
) -> NDArray[numpy.int_]:
    """Calculate target dimensions based on resize method."""
    im_volumetric = True if im_dimensions == 3 else False
    im_multichannel = True if im_pixel_data.ndim > im_dimensions else False
    
    
    shape = numpy.array(im_pixel_data.shape).astype(float)

    if size_method == ResizingMethod.BY_FACTOR:
        factor_x = resizing_factor_x
        factor_y = resizing_factor_y

        if im_volumetric:
            factor_z = resizing_factor_z
            height, width = shape[1:3]
            planes = shape[0]
            planes = numpy.round(planes * factor_z)
        else:
            height, width = shape[:2]

        height = numpy.round(height * factor_y)
        width = numpy.round(width * factor_x)

    else:
        if use_manual_or_image == DimensionMethod.MANUAL:
            height = specific_height
            width = specific_width
            if im_volumetric:
                planes = specific_planes
        else:
            if reference_image_shape is None:
                raise ValueError("Reference image shape must be provided when using image-based dimensions")

            if im_volumetric:
                planes, height, width = reference_image_shape[:3]
            else:
                height, width = reference_image_shape[:2]

    new_shape = []

    if im_volumetric:
        new_shape += [planes]

    new_shape += [height, width]

    if im_multichannel:
        new_shape += [shape[-1]]

    return numpy.asarray(new_shape)


def spline_order(interpolation_method: str) -> int:
    """Determine interpolation order from method."""
    
    if interpolation_method == InterpolationMethod.NEAREST_NEIGHBOR:
        return 0

    if interpolation_method == InterpolationMethod.BILINEAR:
        return 1

    return 3


def apply_resize(
        im_pixel_data: ImageAny,
        im_mask: ImageGrayscaleMask,
        im_dimensions: int,
        im_crop_mask: Optional[ImageGrayscaleMask],
        size_method: str,
        resizing_factor_x: float,
        resizing_factor_y: float,
        resizing_factor_z: Optional[float],
        use_manual_or_image: DimensionMethod,
        specific_width: Optional[int],
        specific_height: Optional[int],
        specific_planes: Optional[int],
        reference_image_shape: Optional[Tuple[int, ...]] = None,
        interpolation_method: str = "bilinear",
    ) -> Tuple[ImageAny, ImageGrayscaleMask, Optional[ImageBinaryMask]]:
    
    new_shape = resized_shape(
        im_pixel_data,
        im_dimensions,
        size_method,
        resizing_factor_x,
        resizing_factor_y,
        resizing_factor_z,
        use_manual_or_image,
        specific_width,
        specific_height,
        specific_planes,
        reference_image_shape,
    )

    order = spline_order(interpolation_method)
    im_volumetric = True if im_dimensions == 3 else False
    im_multichannel = True if im_pixel_data.ndim > im_dimensions else False
    if im_volumetric and im_multichannel:
        output_pixels = numpy.zeros(new_shape.astype(int), dtype=im_pixel_data.dtype)

        for idx in range(int(new_shape[-1])):
            output_pixels[:, :, :, idx] = skimage.transform.resize(
                im_pixel_data[:, :, :, idx],
                new_shape[:-1],
                order=order,
                mode="symmetric",
            )
    else:
        output_pixels = skimage.transform.resize(
            im_pixel_data, new_shape, order=order, mode="symmetric"
        )

    if im_multichannel and len(new_shape) > im_dimensions:
        new_shape = new_shape[:-1]

    mask = skimage.transform.resize(im_mask, new_shape, order=0, mode="constant")

    mask = skimage.img_as_bool(mask)

    if im_crop_mask is not None:
        cropping = skimage.transform.resize(
            im_crop_mask, new_shape, order=0, mode="constant"
        )

        cropping = skimage.img_as_bool(cropping)
    else:
        cropping = None

    return output_pixels, mask, cropping


def get_threshold_robust_background(
    image:                  ImageGrayscale,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method:       Threshold.AveragingMethod = Threshold.AveragingMethod.MEAN,
    variance_method:        Threshold.VarianceMethod = Threshold.VarianceMethod.STANDARD_DEVIATION,
    number_of_deviations:   int = 2,
) -> float:
    """Calculate threshold based on mean & standard deviation.
    The threshold is calculated by trimming the top and bottom 5% of
    pixels off the image, then calculating the mean and standard deviation
    of the remaining image. The threshold is then set at 2 (empirical
    value) standard deviations above the mean.


    lower_outlier_fraction - after ordering the pixels by intensity, remove
        the pixels from 0 to len(image) * lower_outlier_fraction from
        the threshold calculation (default = 0.05).
    upper_outlier_fraction - remove the pixels from
        len(image) * (1 - upper_outlier_fraction) to len(image) from
        consideration (default = 0.05).
    averaging_method - Determines how the intensity midpoint is determined
        after discarding outliers. (default "Mean". Options: "Mean", "Median",
        "Mode").
    variance_method - Method to calculate variance (default =
        "Standard deviation". Options: "Standard deviation",
        "Median absolute deviation")
    number_of_deviations - Following calculation of the standard deviation
        or MAD, multiply this number and add to the average to get the final
        threshold (default = 2)
    average_fn - function used to calculate the average intensity (e.g.
        np.mean, np.median or some sort of mode function). Default = np.mean
    variance_fn - function used to calculate the amount of variance.
                    Default = np.sd
    """
    averaging_method_map = {
        Threshold.AveragingMethod.MEAN: numpy.mean,
        Threshold.AveragingMethod.MEDIAN: numpy.median,
        Threshold.AveragingMethod.MODE: centrosome.threshold.binned_mode,
    }
    variance_method_map = {
        Threshold.VarianceMethod.STANDARD_DEVIATION: numpy.std,
        Threshold.VarianceMethod.MEDIAN_ABSOLUTE_DEVIATION: centrosome.threshold.mad,
    }
    # Check if the averaging method is valid
    if averaging_method not in averaging_method_map:
        raise ValueError(
            f"{averaging_method} not in {', '.join([e.value for e in Threshold.AveragingMethod])}. "
        )
    # Check if the variance method is valid
    if variance_method not in variance_method_map:
        raise ValueError(
            f"{variance_method} not in {', '.join([e.value for e in Threshold.VarianceMethod])}. "
        )
    
    average_fn = averaging_method_map[averaging_method]
    variance_fn = variance_method_map[variance_method]


    flat_image = image.flatten()
    n_pixels = len(flat_image)
    if n_pixels < 3:
        return 0

    flat_image.sort()
    if flat_image[0] == flat_image[-1]:
        return flat_image[0]
    low_chop = int(round(n_pixels * lower_outlier_fraction))
    hi_chop = n_pixels - int(round(n_pixels * upper_outlier_fraction))
    im = flat_image if low_chop == 0 else flat_image[low_chop:hi_chop]
    mean = average_fn(im)
    sd = variance_fn(im)
    return mean + sd * number_of_deviations

# Helper function for get_adaptive_threshold()
def __apply_threshold_function(
        image:              ImageGrayscale,
        window_size:        int,
        threshold_method:   Threshold.Method,
        threshold_fn:       Callable[[Any], Any],
        bin_wanted:         int,
        **kwargs:           Any,
)   -> ImageGrayscale:
    image_size = numpy.array(image.shape[:2], dtype=int)
    nblocks = image_size // window_size
    if any(n < 2 for n in nblocks):
        raise ValueError(
            "Adaptive window cannot exceed 50%% of an image dimension.\n"
            "Window of %dpx is too large for a %sx%s image"
            % (window_size, image_size[1], image_size[0])
        )
    #
    # Use a floating point block size to apportion the roundoff
    # roughly equally to each block
    #
    increment = numpy.array(image_size, dtype=float) / numpy.array(
        nblocks, dtype=float
    )
    #
    # Put the answer here
    #
    thresh_out = numpy.zeros(image_size, image.dtype)
    #
    # Loop once per block, computing the "global" threshold within the
    # block.
    #
    block_threshold = numpy.zeros([nblocks[0], nblocks[1]])
    for i in range(nblocks[0]):
        i0 = int(i * increment[0])
        i1 = int((i + 1) * increment[0])
        for j in range(nblocks[1]):
            j0 = int(j * increment[1])
            j1 = int((j + 1) * increment[1])
            block = image[i0:i1, j0:j1]
            block = block[~numpy.logical_not(block)]
            if len(block) == 0:
                threshold_out = 0.0
            elif numpy.all(block == block[0]):
                # Don't compute blocks with only 1 value.
                threshold_out = block[0]
            elif threshold_method == Threshold.Method.MULTI_OTSU and len(numpy.unique(block)) < 3:
                # Region within window has only 2 values.
                # Can't run 3-class otsu on only 2 values.
                threshold_out = skimage.filters.threshold_otsu(block)
            else:
                try:
                    threshold_out = threshold_fn(block, **kwargs)
                except ValueError:
                    # Drop nbins kwarg when multi-otsu fails. See issue #6324 scikit-image
                    threshold_out = threshold_fn(block)
            if isinstance(threshold_out, numpy.ndarray):
                # Select correct bin if running multiotsu
                threshold_out = threshold_out[bin_wanted]
            block_threshold[i, j] = threshold_out
    #
    # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
    #
    spline_order = min(3, numpy.min(nblocks) - 1)
    xStart = int(increment[0] / 2)
    xEnd = int((nblocks[0] - 0.5) * increment[0])
    yStart = int(increment[1] / 2)
    yEnd = int((nblocks[1] - 0.5) * increment[1])
    xtStart = 0.5
    xtEnd = image.shape[0] - 0.5
    ytStart = 0.5
    ytEnd = image.shape[1] - 0.5
    block_x_coords = numpy.linspace(xStart, xEnd, nblocks[0])
    block_y_coords = numpy.linspace(yStart, yEnd, nblocks[1])
    adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
        block_x_coords,
        block_y_coords,
        block_threshold,
        bbox=(xtStart, xtEnd, ytStart, ytEnd),
        kx=spline_order,
        ky=spline_order,
    )
    thresh_out_x_coords = numpy.linspace(
        0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
    )
    thresh_out_y_coords = numpy.linspace(
        0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
    )
    # Smooth out the "blocky" adaptive threshold
    thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)
    return thresh_out

def get_adaptive_threshold(
    image:                          ImageGrayscale,
    mask:                           Optional[ImageGrayscaleMask] = None,
    threshold_method:               Threshold.Method = Threshold.Method.OTSU,
    window_size:                    int = 50,
    threshold_min:                  float = 0,
    threshold_max:                  float = 1,
    threshold_correction_factor:    float = 1,
    assign_middle_to_foreground:    Threshold.Assignment = Threshold.Assignment.FOREGROUND,
    global_limits:                  Tuple[float, float] = (0.7, 1.5),
    log_transform:                  bool = False,
    volumetric:                     bool = False,
    **kwargs:                       Any,
) -> ImageGrayscale:

    if mask is not None:
        # Apply mask and preserve image shape
        image = numpy.where(mask, image, False)

    if volumetric:
        # Array to store threshold values
        thresh_out = numpy.zeros(image.shape)
        for z in range(image.shape[0]):
            thresh_out[z, :, :] = get_adaptive_threshold(
                image[z, :, :],
                mask=None,  # Mask has already been applied
                threshold_method=threshold_method,
                window_size=window_size,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_correction_factor=threshold_correction_factor,
                assign_middle_to_foreground=assign_middle_to_foreground,
                global_limits=global_limits,
                log_transform=log_transform,
                volumetric=False,  # Processing a single plane, so volumetric=False
                **kwargs,
            )
        return thresh_out
    conversion_dict = None
    if log_transform:
        image, conversion_dict = centrosome.threshold.log_transform(image)
    bin_wanted = 0 if assign_middle_to_foreground == Threshold.Assignment.FOREGROUND else 1

    thresh_out = None
    threshold_fn = lambda x: None

    if len(image) == 0 or numpy.all(image == numpy.nan):
        thresh_out = numpy.zeros_like(image)

    elif numpy.all(image == image.ravel()[0]):
        thresh_out = numpy.full_like(image, image.ravel()[0])

    # Define the threshold method to be run in each adaptive window
    elif threshold_method == Threshold.Method.OTSU:
        threshold_fn = skimage.filters.threshold_otsu

    elif threshold_method == Threshold.Method.MULTI_OTSU:
        threshold_fn = skimage.filters.threshold_multiotsu
        # If nbins not set in kwargs, use default 128
        kwargs["nbins"] = kwargs.get("nbins", 128)

    elif threshold_method == Threshold.Method.MINIMUM_CROSS_ENTROPY:
        tol = max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
        kwargs["tolerance"] = tol
        threshold_fn = skimage.filters.threshold_li

    elif threshold_method == Threshold.Method.ROBUST_BACKGROUND:
        threshold_fn = get_threshold_robust_background
        kwargs["lower_outlier_fraction"] = kwargs.get("lower_outlier_fraction", 0.05)
        kwargs["upper_outlier_fraction"] = kwargs.get("upper_outlier_fraction", 0.05)
        kwargs["averaging_method"] = kwargs.get("averaging_method", Threshold.AveragingMethod.MEAN)
        kwargs["variance_method"] = kwargs.get("variance_method", Threshold.VarianceMethod.STANDARD_DEVIATION)
        kwargs["number_of_deviations"] = kwargs.get("number_of_deviations", 2)
        
    elif threshold_method == Threshold.Method.SAUVOLA:
        if window_size % 2 == 0:
            window_size += 1
        thresh_out = skimage.filters.threshold_sauvola(image, window_size)
        
    else:
        raise NotImplementedError(f"Threshold method {threshold_method} not supported.")

    if thresh_out is None:
        thresh_out = __apply_threshold_function(
            image,
            window_size,
            threshold_method,
            threshold_fn,
            bin_wanted,
            **kwargs,
        )
        
    # Get global threshold
    global_threshold = get_global_threshold(
        image,
        mask,
        threshold_method,
        threshold_min,
        threshold_max,
        threshold_correction_factor,
        assign_middle_to_foreground,
        log_transform=log_transform,
    )

    if log_transform:
        # Revert the log transformation
        thresh_out = centrosome.threshold.inverse_log_transform(
            thresh_out, conversion_dict
        )
        global_threshold = centrosome.threshold.inverse_log_transform(
            global_threshold, conversion_dict
        )

    # Apply threshold_correction
    thresh_out *= threshold_correction_factor

    t_min = max(threshold_min, global_threshold * global_limits[0])
    t_max = min(threshold_max, global_threshold * global_limits[1])
    thresh_out[thresh_out < t_min] = t_min
    thresh_out[thresh_out > t_max] = t_max
    return thresh_out


def get_global_threshold(
    image:                       ImageGrayscale,
    mask:                        Optional[ImageGrayscaleMask] = None,
    threshold_method:            Threshold.Method = Threshold.Method.OTSU,
    threshold_min:               float = 0,
    threshold_max:               float = 1,
    threshold_correction_factor: float = 1,
    assign_middle_to_foreground: Threshold.Assignment = Threshold.Assignment.FOREGROUND,
    log_transform:               bool = False,
    max_intensity_percentage:    float = 100,
    **kwargs:                    Any,
) -> float:
    conversion_dict = None
    if log_transform:
        image, conversion_dict = centrosome.threshold.log_transform(image)

    if mask is not None:
        # Apply mask and discard masked pixels
        image = image[mask]

    # Shortcuts - Check if image array is empty or all pixels are the same value.
    if len(image) == 0:
        threshold = 0.0
    elif numpy.all(image == image.ravel()[0]):
        # All pixels are the same value
        threshold = image.ravel()[0]

    elif threshold_method in (Threshold.Method.MINIMUM_CROSS_ENTROPY, Threshold.Method.SAUVOLA):
        tol = max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
        threshold = skimage.filters.threshold_li(image, tolerance=tol)
    elif threshold_method == Threshold.Method.ROBUST_BACKGROUND:
        threshold = get_threshold_robust_background(image, **kwargs)
    elif threshold_method == Threshold.Method.OTSU:
        threshold = skimage.filters.threshold_otsu(image)
    elif threshold_method == Threshold.Method.MULTI_OTSU:
        bin_wanted = 0 if assign_middle_to_foreground == Threshold.Assignment.FOREGROUND else 1
        kwargs["nbins"] = kwargs.get("nbins", 128)
        threshold = skimage.filters.threshold_multiotsu(image, **kwargs)
        threshold = threshold[bin_wanted]
    elif threshold_method.casefold() == Threshold.Method.MAX_INTENSITY_PERCENTAGE:
        threshold = max_intensity_percentage * numpy.max(image) / 100
    else:
        raise NotImplementedError(f"Threshold method {threshold_method} not supported.")

    if log_transform:
        threshold = centrosome.threshold.inverse_log_transform(
            threshold, conversion_dict
        )

    threshold *= threshold_correction_factor
    threshold = min(max(threshold, threshold_min), threshold_max)
    return threshold


def apply_threshold(
        image: ImageGrayscale,
        threshold: Union[float, ImageGrayscale],
        mask: Optional[ImageGrayscaleMask] = None,
        smoothing: float = 0,
        ) -> Tuple[ImageGrayscaleMask,
                   float]:
    if mask is None:
        # Create a fake mask if one isn't provided
        mask = numpy.full(image.shape, True)
    if smoothing == 0:
        return (image >= threshold) & mask, 0
    else:
        # Convert from a scale into a sigma. What I've done here
        # is to structure the Gaussian so that 1/2 of the smoothed
        # intensity is contributed from within the smoothing diameter
        # and 1/2 is contributed from outside.
        sigma = smoothing / 0.6744 / 2.0

    blurred_image = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
        mask,
    )
    return (blurred_image >= threshold) & mask, sigma


def overlay_objects(image, labels, opacity=0.3, max_label=None, seed=None, colormap="jet"):
    cmap = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap(colormap))

    colors = cmap.to_rgba(
        numpy.arange(labels.max() if max_label is None else max_label)
    )[:, :3]

    if seed is not None:
        # Resetting the random seed helps keep object label colors consistent in displays
        # where consistency is important, like RelateObjects.
        numpy.random.seed(seed)

    numpy.random.shuffle(colors)

    if labels.ndim == 3:
        overlay = numpy.zeros(labels.shape + (3,), dtype=numpy.float32)

        for index, plane in enumerate(image):
            unique_labels = numpy.unique(labels[index])

            if unique_labels[0] == 0:
                unique_labels = unique_labels[1:]

            overlay[index] = skimage.color.label2rgb(
                labels[index],
                alpha=opacity,
                bg_color=[0, 0, 0],
                bg_label=0,
                colors=colors[unique_labels - 1],
                image=plane,
            )

        return overlay

    return skimage.color.label2rgb(
        labels,
        alpha=opacity,
        bg_color=[0, 0, 0],
        bg_label=0,
        colors=colors,
        image=image,
    )

def gaussian_filter(image, sigma):
    '''
    GaussianFilter will blur an image and remove noise, and can be helpful where the foreground signal is noisy or near the noise floor.
    image=input image, y_data=output image
    Sigma is the standard deviation of the kernel to be used for blurring, larger sigmas induce more blurring. 
    '''
    # this replicates "automatic channel detection" present in skimage < 0.22, which was removed in 0.22
    # only relevant for ndim < len(sigma), e.g. multichannel images
    # the channel dim being last, and being equal to 3, is an assumption that should likely be revisited
    # but that was how skimage did it, and therefore is in keeping with prior behavior
    if image.ndim == 3 and image.shape[-1] == 3:
        channel_axis = -1
    else:
        channel_axis = None
    y_data = skimage.filters.gaussian(image, sigma=sigma, channel_axis=channel_axis)
    return y_data


################################################################################
# ColorToGray
################################################################################

def combine_colortogray(
    image:          Image2DColor,
    channels:       List[int],
    contributions:  List[float],
    ) -> Image2DGrayscale:
    denominator = sum(contributions)
    _channels = numpy.array(channels, int)
    _contributions = numpy.array(contributions) / denominator

    output_image = numpy.sum(
        image[:, :, _channels]
        * _contributions[numpy.newaxis, numpy.newaxis, :],
        2
    )
    return output_image

def split_hsv(
        input_image: Image2DColor,
) -> List[Image2DGrayscale]:
     output_image = matplotlib.colors.rgb_to_hsv(input_image)
     return [i for i in output_image.transpose(2, 0, 1)]

def split_rgb(input_image: Image2DColor) -> List[Image2DGrayscale]:
     return [i for i in input_image.transpose(2, 0, 1)]

def split_multichannel(input_image: Image2DColor) -> List[Image2DGrayscale]:
     return split_rgb(input_image)


################################################################################
# ConvertImageToObjects
################################################################################

def image_to_objects(
        data:           ImageAny, 
        cast_to_bool:   bool,
        preserve_label: bool,
        background:     int,
        connectivity:   Union[int, None],
        ) -> ObjectSegmentation:
    # Compatibility with skimage
    connectivity = None if connectivity == 0 else connectivity

    caster = skimage.img_as_bool if cast_to_bool else skimage.img_as_uint
    data = caster(data)

    # If preservation is desired, just return the original labels
    if preserve_label and not cast_to_bool:
        return data

    return skimage.measure.label(data, background=background, connectivity=connectivity)

###########################################################################
# CorrectIlluminationApply
###########################################################################

def apply_divide(image_pixels: Image2D, illum_function_pixel_data: Image2D) -> Image2D:
    return image_pixels / illum_function_pixel_data

def apply_subtract(image_pixels: Image2D, illum_function_pixel_data: Image2D) -> Image2D:
    output_image = image_pixels - illum_function_pixel_data
    output_image[output_image < 0] = 0
    return output_image

def clip_low(output_pixels: Image2D) -> Image2D:
    return numpy.where(output_pixels < 0, 0, output_pixels)

def clip_high(output_pixels: Image2D) -> Image2D:
    return numpy.where(output_pixels > 1, 1, output_pixels)

################################################################################
# Crop
################################################################################

def get_ellipse_cropping(
        orig_image_pixels:  Image2D,
        ellipse_center:     Tuple[float, float],
        ellipse_radius:     Tuple[float, float]
    ) -> Image2DMask:
    x_center, y_center = ellipse_center
    x_radius, y_radius = ellipse_radius
    x_max = orig_image_pixels.shape[1]
    y_max = orig_image_pixels.shape[0]
    if x_radius > y_radius:
        dist_x = math.sqrt(x_radius ** 2 - y_radius ** 2)
        dist_y = 0
        major_radius = x_radius
    else:
        dist_x = 0
        dist_y = math.sqrt(y_radius ** 2 - x_radius ** 2)
        major_radius = y_radius

    focus_1_x, focus_1_y = (x_center - dist_x, y_center - dist_y)
    focus_2_x, focus_2_y = (x_center + dist_x, y_center + dist_y)
    y, x = numpy.mgrid[0:y_max, 0:x_max]
    d1 = numpy.sqrt((x - focus_1_x) ** 2 + (y - focus_1_y) ** 2)
    d2 = numpy.sqrt((x - focus_2_x) ** 2 + (y - focus_2_y) ** 2)
    cropping = d1 + d2 <= major_radius * 2
    return cropping


def get_rectangle_cropping(
    orig_image_pixels:      Image2D,
    bounding_box:           Tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
    validate_boundaries:    bool = True
) -> Image2DMask:
    cropping = numpy.ones(orig_image_pixels.shape[:2], bool)
    left, right, top, bottom = bounding_box
    if validate_boundaries:
        if left and left > 0:
            cropping[:, :left] = False
        if right and right < cropping.shape[1]:
            cropping[:, right:] = False
        if top and top > 0:
            cropping[:top, :] = False
        if bottom and bottom < cropping.shape[0]:
            cropping[bottom:, :] = False
    else:
        cropping[:, :left] = False
        cropping[:, right:] = False
        cropping[:top, :] = False
        cropping[bottom:, :] = False
    return cropping


def crop_image(
        image:          Union[ImageAny, ImageAnyMask],
        crop_mask:      ImageAnyMask,
        crop_internal:  Optional[bool]=False
    ) -> Union[ImageAny, ImageAnyMask]:
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum = numpy.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum = numpy.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return numpy.zeros((0, 0), dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = numpy.argwhere(i_histogram > 0)
        j_keep = numpy.argwhere(j_histogram > 0)
        #
        # Then slice the array by I, then by J to get what's not blank
        #
        return image[i_keep.flatten(), :][:, j_keep.flatten()].copy()
    else:
        #
        # The first non-blank row and column are where the cumsum is 1
        # The last are at the first where the cumsum is it's max (meaning
        # what came after was all zeros and added nothing)
        #
        i_first = numpy.argwhere(i_cumsum == 1)[0]
        i_last = numpy.argwhere(i_cumsum == i_cumsum.max())[0]
        i_end = i_last + 1
        j_first = numpy.argwhere(j_cumsum == 1)[0]
        j_last = numpy.argwhere(j_cumsum == j_cumsum.max())[0]
        j_end = j_last + 1

        if image.ndim == 3:
            return image[i_first[0] : i_end[0], j_first[0] : j_end[0], :].copy()

        return image[i_first[0] : i_end[0], j_first[0] : j_end[0]].copy()


def get_cropped_mask(
        cropping:           Image2DMask,
        mask:               Optional[Image2DMask],
        removal_method:     RemovalMethod = RemovalMethod.NO,
) -> Image2DMask:
    if removal_method == RemovalMethod.NO:
        #
        # Check for previous cropping's mask. If it doesn't exist, set it to the current cropping
        #
        if mask is None:
            mask = cropping
    elif removal_method in (RemovalMethod.EDGES, RemovalMethod.ALL):
        crop_internal = removal_method == RemovalMethod.ALL
        #
        # Check for previous cropping's mask. If it doesn't exist, set it to the region of interest specified
        # by the cropping. The final mask output size could be smaller as the crop_image function removes
        # edges by default.
        #
        if mask is None:
            mask = crop_image(cropping, cropping, crop_internal)
    else:
        raise NotImplementedError(f"Unimplemented removal method: {removal_method}")
    assert mask is not None
    return mask


def get_cropped_image_mask(
        cropping:           Image2DMask,
        mask:               Optional[Image2DMask],
        orig_image_mask:    Optional[Image2DMask] = None,
        removal_method:     RemovalMethod = RemovalMethod.NO,
) -> Image2DMask:
    if mask is None:
        mask = get_cropped_mask(cropping, mask, removal_method)
    if removal_method == RemovalMethod.NO:
        #
        # Check if a mask has been set on the original image. If not, set it to the current mask
        # This is a mask that could have been set by another module and this module "respects masks".
        #
        if orig_image_mask is not None:
            # Image mask is the region of interest indicator for the final image object.
            image_mask = orig_image_mask & mask
        else:
            image_mask = mask

        return image_mask
    elif removal_method in (RemovalMethod.EDGES, RemovalMethod.ALL):
        crop_internal = removal_method == RemovalMethod.ALL
        #
        # Check if a mask has been set on the original image. If not, set it to the current mask
        # This is a mask that could have been set by another module and this module "respects masks".
        # The final mask output size could be smaller as the crop_image function removes edges by default.
        #
        if orig_image_mask is not None:
            # Image mask is the region of interest indicator for the final image object.
            image_mask = crop_image(orig_image_mask, cropping, crop_internal) & mask
        else:
            image_mask = mask
    else:
        raise NotImplementedError(f"Unimplemented removal method: {removal_method}")

    return image_mask


def get_cropped_image_pixels(
        orig_image_pixels:  Image2D,
        cropping:           Image2DMask,
        mask:               Optional[Image2DMask],
        removal_method:     RemovalMethod = RemovalMethod.NO,
) -> Image2D:
    if removal_method == RemovalMethod.NO:
        cropped_pixel_data = apply_crop_keep_rows_and_columns(orig_image_pixels, cropping)
    elif removal_method in (RemovalMethod.EDGES, RemovalMethod.ALL):
        cropped_pixel_data = apply_crop_remove_rows_and_columns(orig_image_pixels, cropping, mask, removal_method)
    else:
        raise NotImplementedError(f"Unimplemented removal method: {removal_method}")
    return cropped_pixel_data


def apply_crop_keep_rows_and_columns(
    orig_image_pixels:  Image2D,
    final_cropping:     Image2DMask,
) -> Image2D:
    cropped_pixel_data = orig_image_pixels.copy()
    cropped_pixel_data = erase_pixels(cropped_pixel_data, final_cropping)
    return cropped_pixel_data


def apply_crop_remove_rows_and_columns(
        orig_image_pixels:  Image2D,
        final_cropping:     Image2DMask,
        mask:               Optional[Image2DMask],
        removal_method: RemovalMethod = RemovalMethod.EDGES,
) -> Image2D:
    if mask is None:
        mask = get_cropped_mask(final_cropping, mask, removal_method)
    # Apply first level of cropping to get the region of interest that matches the original image
    cropped_pixel_data = crop_image(orig_image_pixels, final_cropping, removal_method==RemovalMethod.ALL)
    cropped_pixel_data = erase_pixels(cropped_pixel_data, mask)
    return cropped_pixel_data


def erase_pixels(
        cropped_pixel_data: Image2D,
        crop:               Image2DMask
        ) -> Image2D:
    #
    # Apply crop to all channels automatically for color images
    #
    if cropped_pixel_data.ndim == 3:
        cropped_pixel_data[~crop, :] = 0
    else:
        cropped_pixel_data[~crop] = 0
    return cropped_pixel_data


###############################################################################
# EnhanceOrSuppressFeatures
###############################################################################

def __mask(
        pixel_data: T,
        mask:       ImageAnyMask,
        ) -> T:
    data = numpy.zeros_like(pixel_data)
    data[mask] = pixel_data[mask]
    return data

def __unmask(
        data:       T,
        pixel_data: T, 
        mask:       ImageAnyMask,
        ) -> T:
    data[~mask] = pixel_data[~mask]
    return data

def __structuring_element(
        radius, 
        volumetric
        ) -> NDArray[numpy.uint8]:
    if volumetric:
        return skimage.morphology.ball(radius)

    return skimage.morphology.disk(radius)


def enhance_speckles(
        im_pixel_data:  ImageGrayscale,
        im_mask:        ImageGrayscaleMask,
        im_volumetric:  bool,
        radius:         float,
        accuracy:       SpeckleAccuracy,
        ) -> ImageGrayscale:
    data = __mask(im_pixel_data, im_mask)
    footprint = __structuring_element(radius, im_volumetric)

    if accuracy == SpeckleAccuracy.SLOW or radius <= 3:
        result = skimage.morphology.white_tophat(data, footprint=footprint)
    else:
        #
        # white_tophat = img - opening
        #              = img - dilate(erode)
        #              = img - maximum_filter(minimum_filter)
        #
        minimum = scipy.ndimage.filters.minimum_filter(data, footprint=footprint)
        maximum = scipy.ndimage.filters.maximum_filter(minimum, footprint=footprint)
        result = data - maximum
        
    return __unmask(result, im_pixel_data, im_mask)


def enhance_neurites(
        im_pixel_data:      ImageGrayscale,
        im_mask:            ImageGrayscaleMask,
        im_volumetric:      bool,
        im_spacing:         Tuple[float, ...],
        smoothing_value:    float,
        radius:             float,
        method:             NeuriteMethod,
        neurite_rescale:    bool,
        ) -> ImageGrayscale:
    data = __mask(im_pixel_data, im_mask)

    if method == NeuriteMethod.GRADIENT:
        # desired effect = img + white_tophat - black_tophat
        footprint = __structuring_element(radius, im_volumetric)
        white = skimage.morphology.white_tophat(data, footprint=footprint)
        black = skimage.morphology.black_tophat(data, footprint=footprint)
        result = data + white - black
        result[result > 1] = 1
        result[result < 0] = 0
    else:
        sigma = smoothing_value
        smoothed = scipy.ndimage.gaussian_filter(data, numpy.divide(sigma, im_spacing))

        if im_volumetric:
            result = numpy.zeros_like(smoothed)
            for index, plane in enumerate(smoothed):
                hessian = centrosome.filter.hessian(plane, return_hessian=False, return_eigenvectors=False)
                result[index] = (-hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2))
        else:
            hessian = centrosome.filter.hessian(smoothed, return_hessian=False, return_eigenvectors=False)
            #
            # The positive values are darker pixels with lighter
            # neighbors. The original ImageJ code scales the result
            # by sigma squared - I have a feeling this might be
            # a first-order correction for e**(-2*sigma), possibly
            # because the hessian is taken from one pixel away
            # and the gradient is less as sigma gets larger.
            #
            result = -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)
            
    result = __unmask(result, im_pixel_data, im_mask)
    if neurite_rescale:
        result = skimage.exposure.rescale_intensity(result)
    return result


def enhance_circles(
        im_pixel_data:  ImageGrayscale,
        im_mask:        ImageGrayscaleMask,
        im_volumetric:  bool,
        radius:         float,
        ) -> ImageGrayscale:
    data = __mask(im_pixel_data, im_mask)
    if im_volumetric:
        result = numpy.zeros_like(data)
        for index, plane in enumerate(data):
            result[index] = skimage.transform.hough_circle(plane, radius)[0]
    else:
        result = skimage.transform.hough_circle(data, radius)[0]
    return __unmask(result, im_pixel_data, im_mask)


def enhance_texture(
        im_pixel_data:  ImageGrayscale,
        im_mask:        ImageGrayscaleMask,
        sigma:          float,
        ) -> ImageGrayscale:
    mask = im_mask
    data = __mask(im_pixel_data, mask)
    gmask = skimage.filters.gaussian(mask.astype(float), sigma, mode="constant")
    img_mean = (skimage.filters.gaussian(data, sigma, mode="constant") / gmask)
    img_squared = (skimage.filters.gaussian(data ** 2, sigma, mode="constant")/ gmask)
    result = img_squared - img_mean ** 2
    return __unmask(result, im_pixel_data, mask)


def enhance_dark_holes(
        im_pixel_data:          ImageGrayscale,
        im_mask:                ImageGrayscaleMask,
        im_volumetric:          bool,
        dark_hole_radius_min:   int,
        dark_hole_radius_max:   int,
        min_radius:             Optional[int] = None,
        max_radius:             Optional[int] = None,
        ) -> ImageGrayscale:
    if min_radius is None:
        min_radius = max(1, int(dark_hole_radius_min / 2))
    if max_radius is None:
        max_radius = int((dark_hole_radius_max + 1) / 2)

    pixel_data = im_pixel_data
    mask = im_mask
    se = __structuring_element(1, im_volumetric)
    inverted_image = pixel_data.max() - pixel_data
    previous_reconstructed_image = inverted_image
    eroded_image = inverted_image
    smoothed_image = numpy.zeros(pixel_data.shape)

    for i in range(max_radius + 1):
        eroded_image = skimage.morphology.erosion(eroded_image, se)
        if mask is not None:
            eroded_image *= mask
        reconstructed_image = skimage.morphology.reconstruction(eroded_image, inverted_image, "dilation", se)
        output_image = previous_reconstructed_image - reconstructed_image
        if i >= min_radius:
            smoothed_image = numpy.maximum(smoothed_image, output_image)
        previous_reconstructed_image = reconstructed_image
    return smoothed_image


def enhance_dic(
        im_pixel_data:  ImageGrayscale,
        im_volumetric:  bool,
        angle:          float,
        decay:          float,
        smoothing:      float,
        ) -> ImageGrayscale:
    pixel_data = im_pixel_data

    if im_volumetric:
        result = numpy.zeros_like(pixel_data).astype(numpy.float64)
        for index, plane in enumerate(pixel_data):
            result[index] = centrosome.filter.line_integration(plane, angle, decay, smoothing)
        return result

    if smoothing == 0:
        smoothing = float(numpy.finfo(float).eps)

    return centrosome.filter.line_integration(pixel_data, angle, decay, smoothing)


def suppress(
        im_pixel_data:  ImageGrayscale, 
        im_mask:        ImageGrayscaleMask,
        im_volumetric:  bool,
        radius:         float,
        ) -> ImageGrayscale:
    data = __mask(im_pixel_data, im_mask)
    footprint = __structuring_element(radius, im_volumetric)
    result = skimage.morphology.opening(data, footprint)
    return __unmask(result, im_pixel_data, im_mask)


################################################################################
# Morphological Operations (from Morph module)
################################################################################


def apply_branchpoints(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        ) -> MorphImageT:
    """Apply branchpoints morphological operation.
    
    Removes all pixels except those that are the branchpoints of a skeleton.
    This operation should be applied to an image after skeletonizing.
    """
    return centrosome.cpmorphology.branchpoints(pixel_data, mask)

def apply_bridge(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> MorphImageT:
    """Apply bridge morphological operation.
    
    Sets a pixel to 1 if it has two non-zero neighbors that are on
    opposite sides of this pixel.
    """
    return centrosome.cpmorphology.bridge(pixel_data, mask, count)

def apply_clean(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply clean morphological operation.
    
    Removes isolated pixels.
    """
    return centrosome.cpmorphology.clean(pixel_data, mask, count)

def apply_convex_hull(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        ) -> ImageGrayscale:
    """Apply convex hull morphological operation.
    
    Finds the convex hull of a binary image.
    """
    if mask is None:
        return centrosome.cpmorphology.convex_hull_image(pixel_data)
    else:
        return centrosome.cpmorphology.convex_hull_image(pixel_data & mask)

def apply_diag(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply diag morphological operation.
    
    Fills in pixels whose neighbors are diagonally connected to 4-connect
    pixels that are 8-connected.
    """
    return centrosome.cpmorphology.diag(pixel_data, mask, count)

def apply_distance(
        pixel_data: MorphImageT,
        rescale_values: bool,
        ) -> ImageGrayscale:
    """Apply distance transform morphological operation.
    
    Computes the distance transform of a binary image.
    """
    image = scipy.ndimage.distance_transform_edt(pixel_data)
    if rescale_values:
        image = image / numpy.max(image)
    return image

def apply_endpoints(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        ) -> ImageGrayscale:
    """Apply endpoints morphological operation.
    
    Removes all pixels except the ones that are at the end of a skeleton.
    """
    return centrosome.cpmorphology.endpoints(pixel_data, mask)

def apply_fill(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply fill morphological operation.
    
    Sets a pixel to 1 if all of its neighbors are 1.
    """
    return centrosome.cpmorphology.fill(pixel_data, mask, count)

def apply_hbreak(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply hbreak morphological operation.
    
    Removes pixels that form vertical bridges between horizontal lines.
    """
    return centrosome.cpmorphology.hbreak(pixel_data, mask, count)

def apply_majority(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply majority morphological operation.
    
    Each pixel takes on the value of the majority that surround it.
    """
    return centrosome.cpmorphology.majority(pixel_data, mask, count)

def apply_openlines(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        linelength: int,
        ) -> ImageGrayscale:
    """Apply openlines morphological operation.
    
    Performs an erosion followed by a dilation using rotating linear structural
    elements.
    """
    return centrosome.cpmorphology.openlines(pixel_data, linelength=linelength, mask=mask)

def apply_remove(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply remove morphological operation.
    
    Removes pixels that are otherwise surrounded by others (4 connected).
    """
    return centrosome.cpmorphology.remove(pixel_data, mask, count)

def apply_shrink(
        pixel_data: MorphImageT,
        count: int,
        ) -> ImageGrayscale:
    """Apply shrink morphological operation.
    
    Performs a thinning operation that erodes unless that operation would change
    the image's Euler number.
    """
    return centrosome.cpmorphology.binary_shrink(pixel_data, count)

def apply_skelpe(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        ) -> ImageGrayscale:
    """Apply skelpe morphological operation.
    
    Performs a skeletonizing operation using the metric, PE * D to control the
    erosion order.
    """
    return centrosome.cpmorphology.skeletonize(
        pixel_data,
        mask,
        scipy.ndimage.distance_transform_edt(pixel_data)
        * centrosome.filter.poisson_equation(pixel_data),
    )

def apply_spur(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply spur morphological operation.
    
    Removes spur pixels, i.e., pixels that have exactly one 8-connected neighbor.
    """
    return centrosome.cpmorphology.spur(pixel_data, mask, count)

def apply_thicken(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply thicken morphological operation.
    
    Dilates the exteriors of objects where that dilation does not 8-connect the
    object with another.
    """
    return centrosome.cpmorphology.thicken(pixel_data, mask, count)

def apply_thin(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        count: int,
        ) -> ImageGrayscale:
    """Apply thin morphological operation.
    
    Thin lines preserving the Euler number using the thinning algorithm.
    """
    return centrosome.cpmorphology.thin(pixel_data, mask, count)

def apply_vbreak(
        pixel_data: MorphImageT,
        mask: Optional[ImageGrayscaleMask],
        ) -> ImageGrayscale:
    """Apply vbreak morphological operation.
    
    Removes pixels that form horizontal bridges between vertical lines.
    """
    return centrosome.cpmorphology.vbreak(pixel_data, mask)


################################################################################
# OverlayOutlines
################################################################################

def create_overlay_base_image(
    obj_shape: Optional[Tuple[int, ...]] = None,
    obj_dimensions: Optional[int] = None,
    im_pixel_data: Optional[ImageAny] = None,
    im_multichannel: bool = False,
    im_dimensions: Optional[int] = None
) -> Tuple[ImageColor, Optional[int]]:
    """Creates base image for overlay outlines processing.
    
    This function creates the foundation image for outline overlay operations.
    When no input image is provided, it creates a blank RGB image using object
    dimensions. When an input image is provided, it converts grayscale images
    to RGB format while preserving existing RGB images.
    
    Args:
        obj_shape: Object spatial dimensions for creating blank RGB image when no input provided
        obj_dimensions: Number of spatial dimensions (2D or 3D) for objects in blank image mode
        im_pixel_data: Input image data to use as base (triggers blank image creation if None)
        im_multichannel: Whether input image already contains multiple color channels
        im_dimensions: Number of spatial dimensions (2D or 3D) in input image
    
    Returns:
        Tuple of (rgb_pixel_data, dimensions) where rgb_pixel_data is always RGB format
    """
    if im_pixel_data is None:
        return numpy.zeros(obj_shape + (3,)), obj_dimensions

    pixel_data = skimage.img_as_float(im_pixel_data)

    if im_multichannel:
        return pixel_data, im_dimensions

    return skimage.color.gray2rgb(pixel_data), im_dimensions


def overlay_outlines_grayscale(
    pixel_data: Optional[ImageColor], 
    brightness_mode: BrightnessMode, 
    object_labels_list: List[ObjectLabelSet], 
    line_mode_value: str, 
    is_volumetric: bool, 
) -> ImageGrayscale:
    """Overlay outlines on RGB image and convert to grayscale with brightness control.
    
    Args:
        pixel_data: RGB base image data to overlay outlines on
        brightness_mode: Brightness control determining outline intensity calculation
        object_labels_list: Object label sets containing segmented object data
        line_mode_value: Line drawing mode string for outline appearance control
        is_volumetric: Whether objects require 3D plane-wise outline processing
    
    Returns:
        Grayscale image with object outlines overlaid
    """
    if brightness_mode == BrightnessMode.MAX_POSSIBLE:
        color = 1.0
    else:
        color = numpy.max(pixel_data)

    for obj_labels in object_labels_list:
        pixel_data = overlay_outlines_on_image(pixel_data, obj_labels, is_volumetric, color, line_mode_value)

    return skimage.color.rgb2gray(pixel_data)


def overlay_outlines_color(
    pixel_data: ImageColor, 
    object_labels_list: List[ObjectLabelSet], 
    colors_list: List[Tuple[int, int, int]], 
    line_mode_value: str, 
    is_volumetric: bool, 
) -> ImageColor:
    """Overlay colored outlines on RGB image.
    
    Args:
        pixel_data: RGB base image
        object_labels_list: Object label sets
        colors_list: RGB colors (0-255) for each object set
        line_mode_value: Line drawing mode
        is_volumetric: Whether objects are 3D
    
    Returns:
        RGB image with colored outlines
    """
    for obj_labels, color_rgb in zip(object_labels_list, colors_list):
        color = tuple(c / 255.0 for c in color_rgb)
        pixel_data = overlay_outlines_on_image(pixel_data, obj_labels, is_volumetric, color, line_mode_value)

    return pixel_data


def overlay_outlines_on_image(
    pixel_data: ImageAny, 
    obj_labels_list: ObjectLabelSet, 
    obj_volumetric: bool, 
    color: Union[float, Tuple[float, float, float]], 
    line_mode_value: str, 
) -> ImageColor:
    """Draw object outlines on image.
    
    Args:
        pixel_data: Image to draw outlines on
        obj_labels_list: Object label set
        obj_volumetric: Whether objects are 3D
        color: Outline color
        line_mode_value: Line drawing mode
    
    Returns:
        Image with outlines drawn
    """
    for labels, _ in obj_labels_list:
        resized_labels = resize_labels_for_overlay(pixel_data, labels)

        if obj_volumetric:
            for index, plane in enumerate(resized_labels):
                pixel_data[index] = skimage.segmentation.mark_boundaries(
                    pixel_data[index],
                    plane,
                    color=color,
                    mode=line_mode_value.lower(),
                )
        else:
            pixel_data = skimage.segmentation.mark_boundaries(
                pixel_data,
                resized_labels,
                color=color,
                mode=line_mode_value.lower(),
            )

    return pixel_data


def resize_labels_for_overlay(
    pixel_data: ImageAny, 
    labels: ObjectSegmentation, 
) -> ObjectSegmentation:
    """Resize labels to match image dimensions.
    
    Args:
        pixel_data: Target image with desired dimensions
        labels: Object labels to resize
    
    Returns:
        Resized labels matching image dimensions
    """
    initial_shape = labels.shape

    final_shape = pixel_data.shape

    if pixel_data.ndim > labels.ndim:  # multichannel
        final_shape = final_shape[:-1]

    adjust = numpy.subtract(final_shape, initial_shape)

    cropped = skimage.util.crop(
        labels,
        [
            (0, dim_adjust)
            for dim_adjust in numpy.abs(
                numpy.minimum(adjust, numpy.zeros_like(adjust))
            )
        ],
    )

    return numpy.pad(
        cropped,
        [
            (0, dim_adjust)
            for dim_adjust in numpy.maximum(adjust, numpy.zeros_like(adjust))
        ],
        mode="constant",
        constant_values=0,
    )


###############################################################################
# RemoveHoles
###############################################################################

def fill_holes(image: ImageAny, diameter: float) -> ImageAny:
    radius = diameter / 2.0

    if image.dtype.kind == "f":
        image = skimage.img_as_bool(image)

    if image.ndim == 2 or image.shape[-1] in (3, 4):
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    size = numpy.pi * factor

    return skimage.morphology.remove_small_holes(image, size)

################################################################################
# ImageMath
################################################################################

def imagemath_apply_on_image(
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



def imagemath_apply_binary_operation(
    opval: Operator, 
    operands: List[Union[ImageAny, float]], 
    masks: Optional[List[Optional[Union[ImageAnyMask, bool]]]], 
    output_pixel_data: ImageAny, 
    output_mask: Optional[ImageAnyMask],
    image_factors: Optional[List[float]], # TODO: can this be removed?
    ignore_mask: bool, # TODO: can this be removed?
    ) -> Tuple[
        ImageAny, 
        Optional[ImageAnyMask], 
    ]:
    #
    # Helper function to determine if logical operations should be used
    #
    def use_logical_operation(pixel_data: List[ImageAny]) -> bool:
        return all(
            [pd.dtype == bool for pd in pixel_data if not isscalar(pd)]
        )
    
    #
    # Helper function for logical subtraction
    #

    def logical_subtract(output_pixel_data: ImageAny, pd: ImageAny) -> ImageAny:
        output_pixel_data[pd] = False
        return output_pixel_data

    comparitor = operands[0] # fix pylance error
    use_logical = use_logical_operation(operands)
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
        output_pixel_data = numpy.ones(operands[0].shape, bool)
        comparitor = operands[0]
    elif opval == Operator.SUBTRACT and use_logical:
        output_pixel_data = operands[0].copy()

    
    # _masks is a list of Nones if masks is None. Fixes type warnings.
    if masks is None:
        masks = [None for _ in operands]

    #
    # Apply the operation to each image in the list
    #
    for pd, mask in zip(operands[1:], masks[1:]):
        output_pixel_data = imagemath_apply_on_image(output_pixel_data, pd, comparitor, op, opval)
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


def imagemath_apply_unary_operation(
    opval: Operator, 
    operands: List[Union[ImageAny, float]], 
    masks: Optional[List[Optional[Union[ImageAnyMask, bool]]]], 
    output_pixel_data: ImageAny,
    output_mask: Optional[ImageAnyMask],
    ignore_mask: bool,
    ) -> Tuple[
        ImageAny, 
        Optional[ImageAnyMask],
    ]:
    if opval == Operator.STDEV:
        pixel_array = numpy.array(operands)
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


###############################################################################
# GrayToColor
###############################################################################

def gray_to_rgb(
        pixel_data_arr: List[Optional[Image2DGrayscale]],
        adjustment_factor_array: List[float],
        intensities: List[Tuple[float, ...]]=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        wants_rescale: bool=True,
        ) -> Image2DColor:
    assert(len(pixel_data_arr) == len(adjustment_factor_array)), f"pixel_data_arr and adjustment_factor_array must be the same length. pixel_data_arr has {len(pixel_data_arr)} elements, adjustment_factor_array has {len(adjustment_factor_array)} elements."
    assert(len(pixel_data_arr) == len(intensities)), f"pixel_data_arr and intensities must be the same length. pixel_data_arr has {len(pixel_data_arr)} elements, intensities has {len(intensities)} elements."

    parent_image = None
    rgb_pixel_data = None
    for pixel_data, adjustment_factor, intensity_triplet in zip(pixel_data_arr, adjustment_factor_array, intensities):
        if pixel_data is None:
            continue
        multiplier = numpy.array(intensity_triplet) * adjustment_factor
        if wants_rescale:
            pixel_data = pixel_data /numpy.max(pixel_data)
        if parent_image is not None:
            if parent_image.shape != pixel_data.shape:
                raise ValueError(
                    "The input images have different sizes (%s vs %s)"
                    % (
                        parent_image.shape, 
                        pixel_data.shape,
                    )
                )
            rgb_pixel_data += numpy.dstack([pixel_data] * 3) * multiplier
        else:
            parent_image = pixel_data
            rgb_pixel_data = numpy.dstack([pixel_data] * 3) * multiplier
    return rgb_pixel_data

def gray_to_cmyk(*args, **kwargs):
    # gray to cmyk has the same implementation as gray to rgb but with different intensities
    return gray_to_rgb(*args, **kwargs)


def gray_to_composite_color(
        pixel_data_arr: List[Optional[Image2DGrayscale]],
        color_array: List[Tuple[int, int, int]],
        weight_array: List[float],
        wants_rescale: bool,
) -> Image2DColor:
    source_channels = pixel_data_arr
    parent_image = pixel_data_arr[0]
    for idx, pd in enumerate(source_channels):
        if pd is None:
            continue
        if pd.shape != source_channels[0].shape:
            raise ValueError(
                "The input images have different sizes (%s vs %s)"
                % (
                    source_channels[0].shape,
                    pd.shape,
                )
            )

    colors: List[NDArray[numpy.float32]] = []
    pixel_data = parent_image
    if wants_rescale:
        pixel_data = pixel_data / numpy.max(pixel_data)
    for color_tuple, weight in zip(color_array, weight_array):
        color = (weight * numpy.array(color_tuple).astype(pixel_data.dtype) / 255)
        colors += [color[numpy.newaxis, numpy.newaxis, :]]
    rgb_pixel_data = pixel_data[:, :, numpy.newaxis] * colors[0]
    for image, color in zip(source_channels[1:], colors[1:]):
        if wants_rescale:
            image = image / numpy.max(image)
        rgb_pixel_data = rgb_pixel_data + image[:, :, numpy.newaxis] * color

    return rgb_pixel_data
    

def gray_to_stacked_color(
    pixel_data_arr: List[Optional[Image2DGrayscale]]
) -> Image2DColor:
    source_channels = pixel_data_arr
    rgb_pixel_data = numpy.dstack(source_channels)
    return rgb_pixel_data


################################################################################
# FlipAndRotate
################################################################################

def flip_image_left_to_right(pixel_data: Image2D) -> NDArray[numpy.int_]:
    i, j = numpy.mgrid[
        0 : pixel_data.shape[0], pixel_data.shape[1] - 1 : -1 : -1
    ]
    return i, j

def flip_image_top_to_bottom(pixel_data: Image2D) ->  NDArray[numpy.int_]:
    i, j = numpy.mgrid[
        pixel_data.shape[0] - 1 : -1 : -1, 0 : pixel_data.shape[1]
    ]
    return i, j

def flip_image_both(pixel_data: Image2D) ->  NDArray[numpy.int_]:
    i, j = numpy.mgrid[
        pixel_data.shape[0] - 1 : -1 : -1, pixel_data.shape[1] - 1 : -1 : -1
    ]
    return i, j


def rotate_image_angle(pixel_data: Image2D, mask: Image2DMask, rotate_angle: float, wants_crop: bool) -> Tuple[Image2D, Image2DMask, Optional[Image2DMask]]:
    angle = rotate_angle
    mask = scipy.ndimage.rotate(mask.astype(float), angle, reshape=True) > 0.50
    crop = (
        scipy.ndimage.rotate(
            numpy.ones(pixel_data.shape[:2]), angle, reshape=True
        )
        > 0.50
    )
    mask = mask & crop
    pixel_data = scipy.ndimage.rotate(pixel_data, angle, reshape=True)
    if wants_crop:
        #
        # We want to find the largest rectangle that fits inside
        # the crop. The cumulative sum in the i and j direction gives
        # the length of the rectangle in each direction and
        # multiplying them gives you the area.
        #
        # The left and right halves are symmetric, so we compute
        # on just two of the quadrants.
        #
        half = (numpy.array(crop.shape) / 2).astype(int)
        #
        # Operate on the lower right
        #
        quartercrop = crop[half[0] :, half[1] :]
        ci = numpy.cumsum(quartercrop, 0)
        cj = numpy.cumsum(quartercrop, 1)
        carea_d = ci * cj
        carea_d[quartercrop == 0] = 0
        #
        # Operate on the upper right by flipping I
        #
        quartercrop = crop[crop.shape[0] - half[0] - 1 :: -1, half[1] :]
        ci = numpy.cumsum(quartercrop, 0)
        cj = numpy.cumsum(quartercrop, 1)
        carea_u = ci * cj
        carea_u[quartercrop == 0] = 0
        carea = carea_d + carea_u
        max_carea = numpy.max(carea)
        max_area = numpy.argwhere(carea == max_carea)[0] + half
        min_i = max(crop.shape[0] - max_area[0] - 1, 0)
        max_i = max_area[0] + 1
        min_j = max(crop.shape[1] - max_area[1] - 1, 0)
        max_j = max_area[1] + 1
        ii = numpy.index_exp[min_i:max_i, min_j:max_j]
        crop = numpy.zeros(pixel_data.shape, bool)
        crop[ii] = True
        mask = mask[ii]
        pixel_data = pixel_data[ii]
    else:
        crop = None
    return pixel_data, mask, crop


def rotate_image_coordinates(pixel_data: Image2D, mask: Image2DMask, rotate_point_1: Tuple[float, float], rotate_point_2: Tuple[float, float], rotate_coordinate_alignment: RotationCoordinateAlignmnet) -> float:
    xdiff = rotate_point_2[0] - rotate_point_1[0]
    ydiff = rotate_point_2[1] - rotate_point_1[1]

    if rotate_coordinate_alignment == RotationCoordinateAlignmnet.VERTICALLY:
        angle = -numpy.arctan2(ydiff, xdiff) * 180.0 / numpy.pi
    elif rotate_coordinate_alignment == RotationCoordinateAlignmnet.HORIZONTALLY:
        angle = numpy.arctan2(xdiff, ydiff) * 180.0 / numpy.pi
    else:
        raise NotImplementedError(
            "Unknown axis: %s" % rotate_coordinate_alignment.value
        )
    return angle


###############################################################################
# Smoothing
###############################################################################

def smoothing_gaussian(pixel_data: Image2D, mask: Optional[Image2DMask], sigma: float) -> Image2D:
    def fn(image: Image2D) -> Image2D:
            return scipy.ndimage.gaussian_filter(
                image, sigma, mode="constant", cval=0
            )
    return smooth_with_function_and_mask(pixel_data, fn, mask)
     

def smoothing_median(pixel_data: Image2D, mask: Optional[Image2DMask], obj_size: float) -> Image2D:
    return median_filter_centrosome(pixel_data, mask, obj_size / 2 + 1)


def smoothing_keeping_edges(pixel_data: Image2D, multichannel: bool, sigma_range: float, sigma: float) -> Image2D:
    assert sigma_range is not None, "sigma_range must be provided for smooth_keeping_edges"
    return denoise_bilateral(
        image=pixel_data.astype(float),
        channel_axis=2 if multichannel else None,
        sigma_color=sigma_range,
        sigma_spatial=sigma,
    )


def smoothing_fit_polynomial(pixel_data: Image2D, mask: Optional[Image2DMask], clip: bool) -> Image2D:
    return fit_polynomial(pixel_data, mask, clip)


def smoothing_circular_average(pixel_data: Image2D, mask: Optional[Image2DMask], obj_size: float) -> Image2D:
    return circular_average_filter(pixel_data, obj_size / 2 + 1, mask)

def smoothing_smooth_to_average(pixel_data: Image2D, mask: Optional[Image2DMask]) -> Image2D:
    if mask is not None:
        mean = numpy.mean(pixel_data[mask])
    else:
        mean = numpy.mean(pixel_data)
    return numpy.ones(pixel_data.shape, pixel_data.dtype) * mean


################################################################################
# MeasureColocalization
################################################################################

def crop_image_similarly(
        this_image: ImageAny, 
        other_image: ImageAny,
        this_crop_mask: Optional[ImageAny] = None,
    ):
    """Crop a 2-d or 3-d image (other_image) using this image's crop mask
    crop mask is the binary image used to crop the parent image to the
    dimensions of the child (this) image. The crop_mask is the same size as
    the parent image.
    image - a np.ndarray to be cropped (of any type)
    """
    if other_image.shape[:2] == this_image.shape[:2]:
        # Same size - no cropping needed
        return other_image
    if any(
        [
            my_size > other_size
            for my_size, other_size in zip(this_image.shape, other_image.shape)
        ]
    ):
        raise ValueError(
            "Image to be cropped is smaller: %s vs %s"
            % (repr(other_image.shape), repr(this_image.shape))
        )
    if this_crop_mask is None:
        raise RuntimeError(
            "Images are of different size and no crop mask available.\n"
            "Use the Crop and Align modules to match images of different sizes."
        )
    cropped_image = crop_image(other_image, this_crop_mask)
    if cropped_image.shape[0:2] != this_image.shape[0:2]:
        raise ValueError(
            "Cropped image is not the same size as the reference image: %s vs %s"
            % (repr(cropped_image.shape), repr(this_image.shape))
        )
    return cropped_image

def apply_threshold_to_objects(
        image:              ImageGrayscale,
        segmented:          ObjectSegmentation,
        threshold_value:    float,
        mask:               Optional[ImageGrayscaleMask] = None,
        ) -> ImageGrayscaleMask:
    output_image_arr = numpy.zeros_like(image)
    if mask is None:
        # Create a fake mask if one isn't provided
        mask = numpy.full(segmented.shape, True)
    assert (image.shape == segmented.shape)
    mask = (segmented > 0) & mask & (~numpy.isnan(image))
    segmented = segmented.copy()
    segmented = segmented[mask]
    n_objects = len(numpy.unique(segmented))
    if (not (n_objects == 0)) and (not (numpy.where(mask)[0].__len__() == 0)):
        #
        # First get the maximum intensity of each object and create
        # a 1d array of floats representing the threshold for each object
        #
        lrange = numpy.arange(n_objects, dtype=numpy.int32) + 1
        # Threshold as percentage of maximum intensity of objects in each channel
        scaled_image = (threshold_value / 100) * fix(
            scipy.ndimage.maximum(image, segmented, lrange)
        )

        #
        # Apply the threshold to the image
        # Use the mask to apply to specific pixels
        #
        output_image_arr[mask] = (image >= scaled_image[segmented - 1])        

    return output_image_arr


################################################################################
# MeasureGranularity
################################################################################

def rescale_pixel_data_and_mask(
    new_shape: Union[Tuple[int, ...], NDArray[numpy.float64]], 
    subsample_size: float, 
    im_pixel_data: ImageGrayscale, 
    im_mask: ImageGrayscaleMask, 
    dimensions: int
    ) -> Tuple[ImageGrayscale, ImageGrayscaleMask]:
    if dimensions == 2:
        i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(im_pixel_data, (i, j), order=1)
        mask = (
            scipy.ndimage.map_coordinates(im_mask.astype(float), (i, j)).astype(float) > 0.9
        )
    else:
        k, i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
            / subsample_size
        )
        pixels = scipy.ndimage.map_coordinates(im_pixel_data, (k, i, j), order=1)
        mask = (
            scipy.ndimage.map_coordinates(im_mask.astype(float), (k, i, j)).astype(float) > 0.9
            )
    return pixels, mask


def restore_scale(
    dimensions: int, 
    orig_shape: NDArray[numpy.float64], 
    scaled_shape: NDArray[numpy.float64], 
    scaled_pixels: ImageGrayscale
    ) -> ImageGrayscale:
    if dimensions == 2:
        i, j = numpy.mgrid[0 : orig_shape[0], 0 : orig_shape[1]].astype(float)
        #
        # Make sure the mapping only references the index range of
        # back_pixels.
        #
        i *= float(scaled_shape[0] - 1) / float(orig_shape[0] - 1)
        j *= float(scaled_shape[1] - 1) / float(orig_shape[1] - 1)
        scaled_pixels = scipy.ndimage.map_coordinates(scaled_pixels, (i, j), order=1).astype(float)
    else:
        k, i, j = numpy.mgrid[
            0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
        ].astype(float)
        k *= float(scaled_shape[0] - 1) / float(orig_shape[0] - 1)
        i *= float(scaled_shape[1] - 1) / float(orig_shape[1] - 1)
        j *= float(scaled_shape[2] - 1) / float(orig_shape[2] - 1)
        scaled_pixels = scipy.ndimage.map_coordinates(scaled_pixels, (k, i, j), order=1).astype(float)
    return scaled_pixels

def downsample_image_and_mask(
    im_pixel_data: ImageGrayscale, 
    im_mask: ImageGrayscaleMask, 
    dimensions: int, 
    subsample_size: float
    ) -> Tuple[ImageGrayscale, ImageGrayscaleMask, NDArray[numpy.float64]]:
    #
    # Downsample the image and mask
    #
    new_shape = numpy.array(im_pixel_data.shape)
    if subsample_size < 1:
        new_shape = new_shape * subsample_size
        pixels, mask = rescale_pixel_data_and_mask(new_shape, subsample_size, im_pixel_data, im_mask, dimensions)
    else:
        pixels = im_pixel_data.copy()
        mask = im_mask.copy()
    return pixels, mask, new_shape

def apply_grayscale_tophat_filter(
    pixels: ImageGrayscale, 
    mask: ImageGrayscaleMask, 
    dimensions: int, 
    image_sample_size: float, 
    radius: int, 
    new_shape: NDArray[numpy.float64]
    ) -> ImageGrayscale:
    back_pixels, back_mask, back_shape = downsample_image_and_mask(pixels, mask, dimensions, image_sample_size)
    # radius = element_size
    footprint = get_morphology_footprint(radius, dimensions)
    back_pixels = masked_erode(back_pixels, back_mask, footprint)
    back_pixels = masked_dilate(back_pixels, back_mask, footprint)
    if image_sample_size < 1:
        back_pixels = restore_scale(dimensions, new_shape, back_shape, back_pixels)
    pixels -= back_pixels
    pixels[pixels < 0] = 0
    return pixels

def masked_dilate(im, mask, footprint):
    im_mask = numpy.zeros_like(im)
    im_mask[mask == True] = im[mask == True]
    im = morphology_dilation(im_mask, footprint)
    return im

def masked_erode(im, mask, footprint):
    im_mask = numpy.zeros_like(im)
    im_mask[mask == True] = im[mask == True]
    im = morphology_erosion(im_mask, footprint)
    return im

def get_morphology_footprint(radius, dimensions):
    if dimensions == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    return footprint
    
###############################################################################
# IdentifyDeadWorms
###############################################################################


def get_3d_adjacent_after_erosion(
        mask: Image2DBinary,
        angle_count: int = 32,
        worm_width: int = 100,
        worm_length: int = 10,
    ) -> Tuple[
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        NDArray[numpy.int_],
    ]:
    #
    # We collect the i,j and angle of pairs of points that
    # are 3-d adjacent after erosion.
    #
    # i - the i coordinate of each point found after erosion
    # j - the j coordinate of each point found after erosion
    # a - the angle of the structuring element for each point found
    #
    i = numpy.zeros(0, int)
    j = numpy.zeros(0, int)
    a = numpy.zeros(0, int)

    ig, jg = numpy.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    for angle_number in range(angle_count):
        angle = float(angle_number) * numpy.pi / float(angle_count)
        strel = get_diamond(angle, worm_width, worm_length)
        erosion = binary_erosion(mask, strel)
        #
        # Accumulate the count, i, j and angle for all foreground points
        # in the erosion
        #
        this_count = numpy.sum(erosion)
        i = numpy.hstack((i, ig[erosion]))
        j = numpy.hstack((j, jg[erosion]))
        a = numpy.hstack((a, numpy.ones(this_count, float) * angle))

    return i, j, a

def process_all_connected_components(
        first: NDArray[numpy.int_], 
        second: NDArray[numpy.int_], 
        i_center: NDArray[numpy.int_], 
        j_center: NDArray[numpy.int_], 
        angular_orientation: NDArray[numpy.int_], 
        mask: NDArray[ObjectLabel]
    ) -> Tuple[
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        NDArray[numpy.float_],
        int,
        NDArray[numpy.int_],
        NDArray[ObjectLabel],
    ]:
    #
    # Do all connected components.
    #
    if len(first) > 0:
        ij_labels = all_connected_components(first, second) + 1
        nlabels = numpy.max(ij_labels)
        label_indexes = numpy.arange(1, nlabels + 1)
        #
        # Compute the measurements
        #
        center_x = fix(mean_of_labels(j_center, ij_labels, label_indexes))
        center_y = fix(mean_of_labels(i_center, ij_labels, label_indexes))
        #
        # The angles are wierdly complicated because of the wrap-around.
        # You can imagine some horrible cases, like a circular patch of
        # "worm" in which all angles are represented or a gentle "U"
        # curve.
        #
        # For now, I'm going to use the following heuristic:
        #
        # Compute two different "angles". The angles of one go
        # from 0 to 180 and the angles of the other go from -90 to 90.
        # Take the variance of these from the mean and
        # choose the representation with the lowest variance.
        #
        # An alternative would be to compute the variance at each possible
        # dividing point. Another alternative would be to actually trace through
        # the connected components - both overkill for such an inconsequential
        # measurement I hope.
        #
        angles = fix(mean_of_labels(angular_orientation, ij_labels, label_indexes))
        angular_orientation_variance = (angular_orientation - angles[ij_labels - 1]) ** 2

        vangles = fix(mean_of_labels(angular_orientation_variance, ij_labels, label_indexes))
        
        aa = angular_orientation.copy()
        aa[angular_orientation > numpy.pi / 2] -= numpy.pi
        
        aangles = fix(mean_of_labels(aa, ij_labels, label_indexes))
        aangular_orientation_variance = (aa - aangles[ij_labels - 1]) ** 2
        
        vaangles = fix(mean_of_labels(aangular_orientation_variance, ij_labels, label_indexes))
        
        aangles[aangles < 0] += numpy.pi
        angles[vaangles < vangles] = aangles[vaangles < vangles]
        #
        # Squish the labels to 2-d. The labels for overlaps are arbitrary.
        #
        labels = numpy.zeros(mask.shape, int)
        labels[i_center, j_center] = ij_labels
    else:
        center_x = numpy.zeros(0, int)
        center_y = numpy.zeros(0, int)
        angles = numpy.zeros(0)
        nlabels = 0
        label_indexes = numpy.zeros(0, int)
        labels = numpy.zeros(mask.shape, int)

    return center_x, center_y, angles, nlabels, label_indexes, labels


def get_diamond(
        angle: float, 
        worm_width: int, 
        worm_length: int
    ) -> StructuringElement:
    """Get a diamond-shaped structuring element

    angle - angle at which to tilt the diamond

    returns a binary array that can be used as a footprint for
    the erosion
    """
    #
    # The shape:
    #
    #                   + x1,y1
    #
    # x0,y0 +                          + x2, y2
    #
    #                   + x3,y3
    #
    x0 = int(numpy.sin(angle) * worm_length / 2)
    x1 = int(numpy.cos(angle) * worm_width / 2)
    x2 = -x0
    x3 = -x1
    y2 = int(numpy.cos(angle) * worm_length / 2)
    y1 = int(numpy.sin(angle) * worm_width / 2)
    y0 = -y2
    y3 = -y1
    xmax = numpy.max(numpy.abs([x0, x1, x2, x3]))
    ymax = numpy.max(numpy.abs([y0, y1, y2, y3]))
    strel = numpy.zeros((ymax * 2 + 1, xmax * 2 + 1), bool)
    index, count, i, j = get_line_pts(
        numpy.array([y0, y1, y2, y3]) + ymax,
        numpy.array([x0, x1, x2, x3]) + xmax,
        numpy.array([y1, y2, y3, y0]) + ymax,
        numpy.array([x1, x2, x3, x0]) + xmax,
    )
    strel[i, j] = True
    strel = binary_fill_holes(strel)
    return strel


def find_adjacent_by_distance(
        i_center: NDArray[numpy.int_], 
        j_center: NDArray[numpy.int_], 
        angular_orientation: NDArray[numpy.int_],
        wants_automatic_distance: bool = True,
        worm_width: Optional[int] = 100,
        worm_length: Optional[int] = 10,
        angle_count: Optional[int] = 32,
        space_distance: Optional[float] = 5,
        angular_distance: Optional[float] = 30,
    ) -> Tuple[
        NDArray[numpy.int_],
        NDArray[numpy.int_],
    ]:
    """Return pairs of worm centers that are deemed adjacent by distance

    i - i-centers of worms
    j - j-centers of worms
    a - angular orientation of worms

    Returns two vectors giving the indices of the first and second
    centers that are connected.
    """
    if len(i_center) < 2:
        return numpy.zeros(len(i_center), int), numpy.zeros(len(i_center), int)
    if wants_automatic_distance:
        assert worm_width is not None and worm_length is not None, "worm_width and worm_length must be provided if wants_automatic_distance is True"
        space_distance = worm_width
        angle_distance = numpy.arctan2(
            worm_width, worm_length
        )
        angle_distance += numpy.pi / angle_count
    else:
        assert space_distance is not None and angular_distance is not None, "space_distance and angular_distance must be provided if wants_automatic_distance is False"
        space_distance = space_distance
        angle_distance = angular_distance * numpy.pi / 180
    #
    # Sort by i and break the sorted vector into chunks where
    # consecutive locations are separated by more than space_distance
    #
    order = numpy.lexsort((angular_orientation, j_center, i_center))
    i_center = i_center[order]
    j_center = j_center[order]
    angular_orientation = angular_orientation[order]
    breakpoint = numpy.hstack(([False], i_center[1:] - i_center[:-1] > space_distance))
    if numpy.all(~breakpoint):
        # No easy win - cross all with all
        first, second = numpy.mgrid[0 : len(i_center), 0 : len(i_center)]
    else:
        # The segment that each belongs to
        segment_number = numpy.cumsum(breakpoint)
        # The number of elements in each segment
        member_count = numpy.bincount(segment_number)
        # The index of the first element in the segment
        member_idx = numpy.hstack(([0], numpy.cumsum(member_count[:-1])))
        # The index of the first element, for every element in the segment
        segment_start = member_idx[segment_number]
        #
        # Develop the cross-products for each segment. Each segment has
        # member_count * member_count crosses.
        #
        # # of (first,second) pairs in each segment
        cross_size = member_count ** 2
        # Index in final array of first element of each segment
        segment_idx = numpy.cumsum(cross_size)
        # relative location of first "first"
        first_start_idx = numpy.cumsum(member_count[segment_number[:-1]])
        first = numpy.zeros(segment_idx[-1], int)
        first[first_start_idx] = 1
        # The "firsts" array
        first = numpy.cumsum(first)
        first_start_idx = numpy.hstack(([0], first_start_idx))
        second = (
            numpy.arange(len(first)) - first_start_idx[first] + segment_start[first]
        )
    mask = (
        numpy.abs((i_center[first] - i_center[second]) ** 2 + (j_center[first] - j_center[second]) ** 2)
        <= space_distance ** 2
    ) & (
        (numpy.abs(angular_orientation[first] - angular_orientation[second]) <= angle_distance)
        | (angular_orientation[first] + numpy.pi - angular_orientation[second] <= angle_distance)
        | (angular_orientation[second] + numpy.pi - angular_orientation[first] <= angle_distance)
    )
    return order[first[mask]], order[second[mask]]

################################################################################
# Align
################################################################################


def align_cross_correlation(
        pixels1: Image2D, 
        pixels2: Image2D
    ) -> Tuple[
        int,
        int
    ]:
    """Align the second image with the first using max cross-correlation

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    Many of the ideas here are based on the paper, "Fast Normalized
    Cross-Correlation" by J.P. Lewis
    (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
    which is frequently cited when addressing this problem.
    """
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        pixels1 = numpy.mean(pixels1, 2)
    if pixels2.ndim == 3:
        pixels2 = numpy.mean(pixels2, 2)
    #
    # We double the size of the image to get a field of zeros
    # for the parts of one image that don't overlap the displaced
    # second image.
    #
    # Since we're going into the frequency domain, if the images are of
    # different sizes, we can make the FFT shape large enough to capture
    # the period of the largest image - the smaller just will have zero
    # amplitude at that frequency.
    #
    s = numpy.maximum(pixels1.shape, pixels2.shape)
    fshape = s * 2
    #
    # Calculate the # of pixels at a particular point
    #
    i, j = numpy.mgrid[-s[0]: s[0], -s[1]: s[1]]
    unit = numpy.abs(i * j).astype(float)
    unit[unit < 1] = 1  # keeps from dividing by zero in some places
    #
    # Normalize the pixel values around zero which does not affect the
    # correlation, keeps some of the sums of multiplications from
    # losing precision and precomputes t(x-u,y-v) - t_mean
    #
    pixels1 = pixels1 - numpy.mean(pixels1)
    pixels2 = pixels2 - numpy.mean(pixels2)
    #
    # Lewis uses an image, f and a template t. He derives a normalized
    # cross correlation, ncc(u,v) =
    # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
    # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
    #
    # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
    # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
    # by t-t_mean.
    #
    fp1 = fft2(pixels1, fshape.tolist())
    fp2 = fft2(pixels2, fshape.tolist())
    corr12 = ifft2(fp1 * fp2.conj()).real

    #
    # Use the trick of Lewis here - compute the cumulative sums
    # in a fashion that accounts for the parts that are off the
    # edge of the template.
    #
    # We do this in quadrants:
    # q0 q1
    # q2 q3
    # For the first,
    # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
    # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
    # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
    # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
    #
    # The second is done as above but reflected lr and ud
    #
    p1_si = pixels1.shape[0]
    p1_sj = pixels1.shape[1]
    p1_sum = numpy.zeros(fshape)
    p1_sum[:p1_si, :p1_sj] = cumsum_quadrant(pixels1, False, False)
    p1_sum[:p1_si, -p1_sj:] = cumsum_quadrant(pixels1, False, True)
    p1_sum[-p1_si:, :p1_sj] = cumsum_quadrant(pixels1, True, False)
    p1_sum[-p1_si:, -p1_sj:] = cumsum_quadrant(pixels1, True, True)
    #
    # Divide the sum over the # of elements summed-over
    #
    p1_mean = p1_sum / unit

    p2_si = pixels2.shape[0]
    p2_sj = pixels2.shape[1]
    p2_sum = numpy.zeros(fshape)
    p2_sum[:p2_si, :p2_sj] = cumsum_quadrant(pixels2, False, False)
    p2_sum[:p2_si, -p2_sj:] = cumsum_quadrant(pixels2, False, True)
    p2_sum[-p2_si:, :p2_sj] = cumsum_quadrant(pixels2, True, False)
    p2_sum[-p2_si:, -p2_sj:] = cumsum_quadrant(pixels2, True, True)
    p2_sum = numpy.fliplr(numpy.flipud(p2_sum))
    p2_mean = p2_sum / unit
    #
    # Once we have the means for u,v, we can calculate the
    # variance-like parts of the equation. We have to multiply
    # the mean^2 by the # of elements being summed-over
    # to account for the mean being summed that many times.
    #
    p1sd = numpy.sum(pixels1 ** 2) - p1_mean ** 2 * numpy.product(s)
    p2sd = numpy.sum(pixels2 ** 2) - p2_mean ** 2 * numpy.product(s)
    #
    # There's always chance of roundoff error for a zero value
    # resulting in a negative sd, so limit the sds here
    #
    sd = numpy.sqrt(numpy.maximum(p1sd * p2sd, 0))
    corrnorm = corr12 / sd
    #
    # There's not much information for points where the standard
    # deviation is less than 1/100 of the maximum. We exclude these
    # from consideration.
    #
    corrnorm[(unit < numpy.product(s) / 2) & (sd < numpy.mean(sd) / 100)] = 0
    i, j = numpy.unravel_index(numpy.argmax(corrnorm), fshape)
    #
    # Reflect values that fall into the second half
    #
    if i > pixels1.shape[0]:
        i = i - fshape[0]
    if j > pixels1.shape[1]:
        j = j - fshape[1]
    return int(j), int(i)

def align_mutual_information(
        pixels1: Image2D,
        pixels2: Image2D,
        mask1: Image2DMask,
        mask2: Image2DMask
    ) -> Tuple[
        int,
        int
    ]:
    """Align the second image with the first using mutual information

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    The algorithm computes the mutual information content of the two
    images, offset by one in each direction (including diagonal) and
    then picks the direction in which there is the most mutual information.
    From there, it tries all offsets again and so on until it reaches
    a local maximum.
    """
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        _pixels1: Image2DGrayscale = numpy.mean(pixels1, 2)
    else:
        _pixels1: Image2DGrayscale = pixels1
    if pixels2.ndim == 3:
        _pixels2: Image2DGrayscale = numpy.mean(pixels2, 2)
    else:
        _pixels2: Image2DGrayscale = pixels2

    def mutualinf(x: Image2DGrayscale, y: Image2DGrayscale, maskx: Image2DMask, masky: Image2DMask) -> float:
        _x = x[maskx & masky]
        _y = y[maskx & masky]
        return entropy(_x) + entropy(_y) - entropy2(_x, _y)
    maxshape = tuple(numpy.maximum(_pixels1.shape, _pixels2.shape))
    _pixels1 = reshape_image(_pixels1, maxshape)
    _pixels2 = reshape_image(_pixels2, maxshape)
    mask1 = reshape_image(mask1, maxshape)
    mask2 = reshape_image(mask2, maxshape)

    best = mutualinf(_pixels1, _pixels2, mask1, mask2)
    i = 0
    j = 0
    while True:
        last_i = i
        last_j = j
        for new_i in range(last_i - 1, last_i + 2):
            for new_j in range(last_j - 1, last_j + 2):
                if new_i == 0 and new_j == 0:
                    continue
                p2, p1 = offset_slice(_pixels2, _pixels1, new_i, new_j)
                m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                info = mutualinf(p1, p2, m1, m2)
                if info > best:
                    best = info
                    i = new_i
                    j = new_j
        if i == last_i and j == last_j:
            return j, i
        
def offset_slice(
        pixels1: Union[Image2D, ImageBinary], # Union to support crop_mask (3 channel binary NDArray)
        pixels2: Image2D, 
        i: int, 
        j: int
    ) -> Tuple[
        Union[Image2D, ImageBinary],
        Image2D
    ]:
    """Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.

    """
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2


def cumsum_quadrant(
        x: Image2D, 
        i_forwards: bool, 
        j_forwards: bool
    ) -> NDArray[numpy.float64]:
    """Return the cumulative sum going in the i, then j direction

    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    """
    if i_forwards:
        x = x.cumsum(0)
    else:
        x = numpy.flipud(numpy.flipud(x).cumsum(0))
    if j_forwards:
        return x.cumsum(1)
    else:
        return numpy.fliplr(numpy.fliplr(x).cumsum(1))


def entropy(x: NDArray[Pixel]) -> float:
    """The entropy of x as if x is a probability distribution"""
    histogram = scind.histogram(x.astype(float), numpy.min(x), numpy.max(x), 256)
    n = numpy.sum(histogram)
    if n > 0 and numpy.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return numpy.log2(n) - numpy.sum(histogram * numpy.log2(histogram)) / n
    else:
        return 0


def entropy2(x: NDArray[Pixel], y: NDArray[Pixel]) -> float:
    """Joint entropy of paired samples X and Y"""
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix(
        (numpy.ones(xy.shape, dtype=numpy.int32), (xy, numpy.zeros(xy.shape, dtype=numpy.int32)))
    )
    histogram = sparse.toarray()
    n = numpy.sum(histogram)
    if n > 0 and numpy.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return numpy.log2(n) - numpy.sum(histogram * numpy.log2(histogram)) / n
    else:
        return 0

ReshapeImageInput = TypeVar("ReshapeImageInput", bound=Union[Image2D, Image2DMask])
def reshape_image(
        source: ReshapeImageInput,  # Union to support crop_mask (3 channel binary NDArray)
        new_shape: Tuple[int, int]
    ) -> ReshapeImageInput:
    """Reshape an image to a larger shape, padding with zeros"""
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = numpy.zeros(new_shape, source.dtype)
    result[: source.shape[0], : source.shape[1]] = source
    return result

