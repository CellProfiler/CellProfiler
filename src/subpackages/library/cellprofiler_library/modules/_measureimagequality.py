import numpy
from numpy.typing import NDArray
import scipy.linalg.basic
import scipy.ndimage
import centrosome.cpmorphology
import centrosome.haralick
import centrosome.radial_power_spectrum
import centrosome.threshold
from pydantic import Field, ConfigDict, validate_call
from typing import Optional, Sequence, Tuple, Dict
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask
    
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_focus_score_for_scale_group(
        scale_groups: Sequence[int],
        pixel_data: ImageGrayscale,
        dimensions: int,
        image_mask: Optional[ImageGrayscaleMask]=None,
) -> Tuple[
    NDArray[numpy.float_],
    int,
    Sequence[numpy.float_]
]:
    if image_mask is not None:
            masked_pixel_data = pixel_data[image_mask]
    else:
        masked_pixel_data = pixel_data
    shape = pixel_data.shape
    local_focus_score = []
    focus_score = 0
    for scale in scale_groups:
        focus_score = 0
        if len(masked_pixel_data):
            mean_image_value = numpy.mean(masked_pixel_data)
            squared_normalized_image = (masked_pixel_data - mean_image_value) ** 2
            if mean_image_value > 0:
                focus_score = numpy.sum(squared_normalized_image) / (
                    numpy.product(masked_pixel_data.shape) * mean_image_value
                )
        #
        # Create a labels matrix that grids the image to the dimensions
        # of the window size
        #
        if dimensions == 2:
            i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
            m, n = (numpy.array(shape) + scale - 1) // scale
            i = (i * float(m) / float(shape[0])).astype(int)
            j = (j * float(n) / float(shape[1])).astype(int)
            grid = i * n + j + 1
            grid_range = numpy.arange(0, m * n + 1, dtype=numpy.int32)
        elif dimensions == 3:
            k, i, j = numpy.mgrid[
                0 : shape[0], 0 : shape[1], 0 : shape[2]
            ].astype(float)
            o, m, n = (numpy.array(shape) + scale - 1) // scale
            k = (k * float(o) / float(shape[0])).astype(int)
            i = (i * float(m) / float(shape[1])).astype(int)
            j = (j * float(n) / float(shape[2])).astype(int)
            grid = k * o + i * n + j + 1  # hmm
            grid_range = numpy.arange(0, m * n * o + 1, dtype=numpy.int32)
        else:
            raise ValueError("Image dimensions must be 2 or 3")
        if image_mask is not None:
            grid[numpy.logical_not(image_mask)] = 0
        
        #
        # Do the math per label
        #
        local_means = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(pixel_data, grid, grid_range)
        )
        local_squared_normalized_image = (
            pixel_data - local_means[grid]
        ) ** 2
        #
        # Compute the sum of local_squared_normalized_image values for each
        # grid for means > 0. Exclude grid label = 0 because that's masked
        #
        grid_mask = (local_means != 0) & ~numpy.isnan(local_means)
        nz_grid_range = grid_range[grid_mask]
        if len(nz_grid_range) and nz_grid_range[0] == 0:
            nz_grid_range = nz_grid_range[1:]
            local_means = local_means[1:]
            grid_mask = grid_mask[1:]
        local_focus_score += [
            0
        ]  # assume the worst - that we can't calculate it
        if len(nz_grid_range):
            sums = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(
                    local_squared_normalized_image, grid, nz_grid_range
                )
            )
            pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.sum(numpy.ones(shape), grid, nz_grid_range)
            )
            local_norm_var = sums / (pixel_counts * local_means[grid_mask])
            local_norm_median = numpy.median(local_norm_var)
            if numpy.isfinite(local_norm_median) and local_norm_median > 0:
                local_focus_score[-1] = (
                    numpy.var(local_norm_var) / local_norm_median
                )
    return focus_score, scale, local_focus_score

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_correlation_for_scale_group(
        pixel_data: ImageGrayscale, 
        scale_groups: Sequence[int], 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Dict[int, numpy.float_]:
    # Compute Haralick's correlation texture for the given scales
    image_labels = numpy.ones(pixel_data.shape, int)
    if image_mask is not None:
        image_labels[~image_mask] = 0
    scale_measurements = {}
    for scale in scale_groups:
        scale_measurements[scale] = get_correlation_for_scale(pixel_data, image_labels, scale)
    return scale_measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_correlation_for_scale(
        pixel_data: ImageGrayscale, 
        image_labels: NDArray[numpy.int_], 
        scale: int
    ) -> float:
    # Compute Haralick's correlation texture for the given scale
    value = centrosome.haralick.Haralick(pixel_data, image_labels, 0, scale).H3()

    if len(value) != 1 or not numpy.isfinite(value[0]):
        value = 0.0
    else:
        value = float(value)
    return value


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_saturation_value(
        pixel_data: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Tuple[
        float,
        float
    ]:
    if image_mask is not None:
        pixel_data = pixel_data[image_mask]
    pixel_count = numpy.product(pixel_data.shape)
    if pixel_count == 0:
        percent_maximal = 0
        percent_minimal = 0
    else:
        number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))
        number_pixels_minimal = numpy.sum(pixel_data == numpy.min(pixel_data))
        percent_maximal = (
            100.0 * float(number_pixels_maximal) / float(pixel_count)
        )
        percent_minimal = (
            100.0 * float(number_pixels_minimal) / float(pixel_count)
        )
    return percent_minimal, percent_maximal


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_intensity_measurement_values(
        pixels: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None
    ) -> Tuple[
        numpy.int_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_,
        numpy.float_
    ]:
    if image_mask is not None:
        pixels = pixels[image_mask]
    
    pixel_count = numpy.product(pixels.shape)
    if pixel_count == 0:
        pixel_sum = 0
        pixel_mean = 0
        pixel_std = 0
        pixel_mad = 0
        pixel_median = 0
        pixel_min = 0
        pixel_max = 0
    else:
        pixel_sum = numpy.sum(pixels)
        pixel_mean = pixel_sum / float(pixel_count)
        pixel_std = numpy.std(pixels)
        pixel_median = numpy.median(pixels)
        pixel_mad = numpy.median(numpy.abs(pixels - pixel_median))
        pixel_min = numpy.min(pixels)
        pixel_max = numpy.max(pixels)
    return pixel_count, pixel_sum, pixel_mean, pixel_std, pixel_mad, pixel_median, pixel_min, pixel_max


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_power_spectrum_measurement_values(
        pixel_data: ImageGrayscale, 
        image_mask: Optional[ImageGrayscaleMask]=None, 
        dimensions: int = 2
    ) -> float:
    if dimensions == 3:
        raise NotImplementedError("Power spectrum calculation for volumes is not implemented")

    if image_mask is not None:
        pixel_data = numpy.array(pixel_data)  # make a copy
        masked_pixels = pixel_data[image_mask]
        pixel_count = numpy.product(masked_pixels.shape)
        if pixel_count > 0:
            pixel_data[~image_mask] = numpy.mean(masked_pixels)
        else:
            pixel_data[~image_mask] = 0

    radii, magnitude, power = centrosome.radial_power_spectrum.rps(pixel_data)
    if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
        valid = magnitude > 0
        radii = radii[valid].reshape((-1, 1))
        power = power[valid].reshape((-1, 1))
        if radii.shape[0] > 1:
            idx = numpy.isfinite(numpy.log(power))
            powerslope = scipy.linalg.basic.lstsq(
                numpy.hstack(
                    (
                        numpy.log(radii)[idx][:, numpy.newaxis],
                        numpy.ones(radii.shape)[idx][:, numpy.newaxis],
                    )
                ),
                numpy.log(power)[idx][:, numpy.newaxis],
            )[0][0]
        else:
            powerslope = 0
    else:
        powerslope = 0
    return powerslope


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_threshold_for_threshold_group(
        image_pixel_data: ImageGrayscale = numpy.ones((100, 100)),
        mask: Optional[ImageGrayscaleMask] = None,
        algorithm: str = centrosome.threshold.TM_OTSU,
        object_fraction: float = 0.1,
        two_class_otsu: bool = True,
        use_weighted_variance: bool = True,
        assign_middle_to_foreground: bool = True,
) -> float:
    if mask is not None:
        _, global_threshold = centrosome.threshold.get_threshold(
            algorithm,
            centrosome.threshold.TM_GLOBAL,
            image_pixel_data,
            mask = mask,
            object_fraction=object_fraction,
            two_class_otsu=two_class_otsu,
            use_weighted_variance=use_weighted_variance,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
    else:
        _, global_threshold = centrosome.threshold.get_threshold(
            algorithm,
            centrosome.threshold.TM_GLOBAL,
            image_pixel_data,
            object_fraction=object_fraction,
            two_class_otsu=two_class_otsu,
            use_weighted_variance=use_weighted_variance,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
    return global_threshold
    
    