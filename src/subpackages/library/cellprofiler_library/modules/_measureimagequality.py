import numpy
from numpy.typing import NDArray
from pydantic import Field, ConfigDict, validate_call
from typing import Optional, Sequence, Tuple, Dict, List, Any
import cellprofiler_core.constants.image
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask
from cellprofiler_library.measurement_model import Measurement, MeasurementResult, MeasurementTarget
from cellprofiler_library.opts.measureimagequality import C_IMAGE_QUALITY, Feature, INTENSITY_FEATURES, SATURATION_FEATURES
from cellprofiler_library.functions.measurement import get_focus_score_for_scale_group, get_correlation_for_scale_group, get_correlation_for_scale, get_saturation_value, get_intensity_measurement_values, get_power_spectrum_measurement_values, calculate_threshold_for_threshold_group

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_quality(
    image_name: str,
    pixel_data: ImageGrayscale,
    image_mask: Optional[ImageGrayscaleMask] = None,
    dimensions: int = 2,
    check_blur: bool = False,
    check_saturation: bool = False,
    check_intensity: bool = False,
    calculate_threshold: bool = False,
    scale_groups: List[int] = [],
    include_image_scalings: bool = False,
    image_scale: Optional[float] = None,
    threshold_groups: List[Dict[str, Any]] = [],
    volumetric: bool = False
) -> MeasurementResult:
    result = MeasurementResult()
    
    if image_mask is not None:
        masked_pixel_data = pixel_data[image_mask]
    else:
        masked_pixel_data = pixel_data

    # Image Scaling
    if include_image_scalings:
        value = image_scale if image_scale is not None else numpy.nan
        
        result.add_measurement(Measurement(
            category=C_IMAGE_QUALITY,
            feature_name=cellprofiler_core.constants.image.C_SCALING,
            parameters={'image': image_name},
            value=value
        ))
        
        result.add_statistic(
            f"{image_name} scaling",
            str(value)
        )

    # Blur Metrics
    if check_blur:
        # Focus Score
        focus_score, scale, local_focus_score = get_focus_score_for_scale_group(
            scale_groups,
            pixel_data,
            dimensions,
            image_mask,
        )
        
        result.add_measurement(Measurement(
            category=C_IMAGE_QUALITY,
            feature_name=Feature.FOCUS_SCORE.value,
            parameters={'image': image_name},
            value=focus_score
        ))
        
        result.add_statistic(
            f"{image_name} focus score @{scale}",
            str(focus_score)
        )

        # Local Focus Score
        for idx, s in enumerate(scale_groups):
             result.add_measurement(Measurement(
                category=C_IMAGE_QUALITY,
                feature_name=Feature.LOCAL_FOCUS_SCORE.value,
                parameters={'image': image_name, 'scale': s},
                value=local_focus_score[idx]
            ))
             result.add_statistic(
                f"{image_name} local focus score @{s}",
                str(local_focus_score[idx])
             )

        # Correlation
        scale_measurements = get_correlation_for_scale_group(pixel_data, scale_groups, image_mask)
        for s in scale_groups:
            result.add_measurement(Measurement(
                category=C_IMAGE_QUALITY,
                feature_name=Feature.CORRELATION.value,
                parameters={'image': image_name, 'scale': s},
                value=scale_measurements[s]
            ))
            result.add_statistic(
                f"{image_name} {Feature.CORRELATION.value} @{s}",
                "{:.2f}".format(scale_measurements[s])
            )

        # Power Spectrum (2D only)
        if dimensions == 2:
            powerslope = get_power_spectrum_measurement_values(pixel_data, image_mask, dimensions)
            result.add_measurement(Measurement(
                category=C_IMAGE_QUALITY,
                feature_name=Feature.POWER_SPECTRUM_SLOPE.value,
                parameters={'image': image_name},
                value=powerslope
            ))
            result.add_statistic(
                f"{image_name} {Feature.POWER_SPECTRUM_SLOPE.value}",
                "{:.1f}".format(float(powerslope))
            )

    # Saturation Metrics
    if check_saturation:
        percent_minimal, percent_maximal = get_saturation_value(pixel_data, image_mask)
        
        result.add_measurement(Measurement(
            category=C_IMAGE_QUALITY,
            feature_name=Feature.PERCENT_MAXIMAL.value,
            parameters={'image': image_name},
            value=percent_maximal
        ))
        
        result.add_measurement(Measurement(
            category=C_IMAGE_QUALITY,
            feature_name=Feature.PERCENT_MINIMAL.value,
            parameters={'image': image_name},
            value=percent_minimal
        ))
        
        result.add_statistic(f"{image_name} maximal", "{:.1f} %".format(percent_maximal))
        result.add_statistic(f"{image_name} minimal", "{:.1f} %".format(percent_minimal))

    # Intensity Metrics
    if check_intensity:
        area_text = "Volume" if volumetric else "Area"
        area_feature = Feature.TOTAL_VOLUME.value if volumetric else Feature.TOTAL_AREA.value
        
        (
            pixel_count,
            pixel_sum,
            pixel_mean,
            pixel_std,
            pixel_mad,
            pixel_median,
            pixel_min,
            pixel_max,
        ) = get_intensity_measurement_values(pixel_data, image_mask)

        measurements_data = [
            (area_feature, pixel_count, f"Total {area_text}"),
            (Feature.TOTAL_INTENSITY.value, pixel_sum, "Total intensity"),
            (Feature.MEAN_INTENSITY.value, pixel_mean, "Mean intensity"),
            (Feature.MEDIAN_INTENSITY.value, pixel_median, "Median intensity"),
            (Feature.STD_INTENSITY.value, pixel_std, "Std intensity"),
            (Feature.MAD_INTENSITY.value, pixel_mad, "MAD intensity"),
            (Feature.MAX_INTENSITY.value, pixel_max, "Max intensity"),
            (Feature.MIN_INTENSITY.value, pixel_min, "Min intensity"),
        ]
        
        for feature, value, stat_name in measurements_data:
            result.add_measurement(Measurement(
                category=C_IMAGE_QUALITY,
                feature_name=feature,
                parameters={'image': image_name},
                value=value
            ))
            result.add_statistic(f"{image_name} {stat_name}", "{:.2f}".format(value))

    # Threshold Metrics
    if calculate_threshold:
        # Note: calculate_threshold_for_threshold_group expects image_pixel_data, mask, etc.
        # The caller (frontend) iterates over threshold groups and prepares params.
        # Here we accept a list of threshold_groups, where each item is a dict of parameters 
        # INCLUDING 'threshold_algorithm' and 'threshold_scale' (for naming)
        # and the arguments for calculate_threshold_for_threshold_group.
        
        for tg in threshold_groups:
            params = tg['params']
            algo = tg['algorithm']
            scale = tg['scale']
            
            calc_params = params.copy()
            calc_params['image_pixel_data'] = pixel_data.astype(numpy.float32) # Logic from frontend
            calc_params['mask'] = image_mask
            
            global_threshold = calculate_threshold_for_threshold_group(**calc_params)
            
            feature_name = Feature.THRESHOLD.value + algo
            
            measurement_params = {'image': image_name}
            if scale:
                measurement_params['scale'] = scale
            
            scale_text = (" " + scale) if scale is not None else ""
            threshold_description = algo + scale_text

            result.add_measurement(Measurement(
                category=C_IMAGE_QUALITY,
                feature_name=feature_name, # e.g. ThresholdOtsu
                parameters=measurement_params,
                value=global_threshold
            ))
            
            result.add_statistic(
                 f"{image_name} {threshold_description} threshold",
                 str(global_threshold)
            )

    return result

# def get_focus_score_for_scale_group(
#         scale_groups: Sequence[int],
#         pixel_data: ImageGrayscale,
#         dimensions: int,
#         image_mask: Optional[ImageGrayscaleMask]=None,
# ) -> Tuple[
#     NDArray[numpy.float_],
#     int,
#     Sequence[numpy.float_]
# ]:
#     if image_mask is not None:
#             masked_pixel_data = pixel_data[image_mask]
#     else:
#         masked_pixel_data = pixel_data
#     shape = pixel_data.shape
#     local_focus_score = []
#     focus_score = 0
#     for scale in scale_groups:
#         focus_score = 0
#         if len(masked_pixel_data):
#             mean_image_value = numpy.mean(masked_pixel_data)
#             squared_normalized_image = (masked_pixel_data - mean_image_value) ** 2
#             if mean_image_value > 0:
#                 focus_score = numpy.sum(squared_normalized_image) / (
#                     numpy.product(masked_pixel_data.shape) * mean_image_value
#                 )
#         #
#         # Create a labels matrix that grids the image to the dimensions
#         # of the window size
#         #
#         if dimensions == 2:
#             i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
#             m, n = (numpy.array(shape) + scale - 1) // scale
#             i = (i * float(m) / float(shape[0])).astype(int)
#             j = (j * float(n) / float(shape[1])).astype(int)
#             grid = i * n + j + 1
#             grid_range = numpy.arange(0, m * n + 1, dtype=numpy.int32)
#         elif dimensions == 3:
#             k, i, j = numpy.mgrid[
#                 0 : shape[0], 0 : shape[1], 0 : shape[2]
#             ].astype(float)
#             o, m, n = (numpy.array(shape) + scale - 1) // scale
#             k = (k * float(o) / float(shape[0])).astype(int)
#             i = (i * float(m) / float(shape[1])).astype(int)
#             j = (j * float(n) / float(shape[2])).astype(int)
#             grid = k * o + i * n + j + 1  # hmm
#             grid_range = numpy.arange(0, m * n * o + 1, dtype=numpy.int32)
#         else:
#             raise ValueError("Image dimensions must be 2 or 3")
#         if image_mask is not None:
#             grid[numpy.logical_not(image_mask)] = 0
        
#         #
#         # Do the math per label
#         #
#         local_means = centrosome.cpmorphology.fixup_scipy_ndimage_result(
#             scipy.ndimage.mean(pixel_data, grid, grid_range)
#         )
#         local_squared_normalized_image = (
#             pixel_data - local_means[grid]
#         ) ** 2
#         #
#         # Compute the sum of local_squared_normalized_image values for each
#         # grid for means > 0. Exclude grid label = 0 because that's masked
#         #
#         grid_mask = (local_means != 0) & ~numpy.isnan(local_means)
#         nz_grid_range = grid_range[grid_mask]
#         if len(nz_grid_range) and nz_grid_range[0] == 0:
#             nz_grid_range = nz_grid_range[1:]
#             local_means = local_means[1:]
#             grid_mask = grid_mask[1:]
#         local_focus_score += [
#             0
#         ]  # assume the worst - that we can't calculate it
#         if len(nz_grid_range):
#             sums = centrosome.cpmorphology.fixup_scipy_ndimage_result(
#                 scipy.ndimage.sum(
#                     local_squared_normalized_image, grid, nz_grid_range
#                 )
#             )
#             pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
#                 scipy.ndimage.sum(numpy.ones(shape), grid, nz_grid_range)
#             )
#             local_norm_var = sums / (pixel_counts * local_means[grid_mask])
#             local_norm_median = numpy.median(local_norm_var)
#             if numpy.isfinite(local_norm_median) and local_norm_median > 0:
#                 local_focus_score[-1] = (
#                     numpy.var(local_norm_var) / local_norm_median
#                 )
#     return focus_score, scale, local_focus_score

# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def get_correlation_for_scale_group(
#         pixel_data: ImageGrayscale, 
#         scale_groups: Sequence[int], 
#         image_mask: Optional[ImageGrayscaleMask]=None
#     ) -> Dict[int, numpy.float_]:
#     # Compute Haralick's correlation texture for the given scales
#     image_labels = numpy.ones(pixel_data.shape, int)
#     if image_mask is not None:
#         image_labels[~image_mask] = 0
#     scale_measurements = {}
#     for scale in scale_groups:
#         scale_measurements[scale] = get_correlation_for_scale(pixel_data, image_labels, scale)
#     return scale_measurements

# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def get_correlation_for_scale(
#         pixel_data: ImageGrayscale, 
#         image_labels: NDArray[numpy.int_], 
#         scale: int
#     ) -> float:
#     # Compute Haralick's correlation texture for the given scale
#     value = centrosome.haralick.Haralick(pixel_data, image_labels, 0, scale).H3()

#     if len(value) != 1 or not numpy.isfinite(value[0]):
#         value = 0.0
#     else:
#         value = float(value)
#     return value


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def get_saturation_value(
#         pixel_data: ImageGrayscale, 
#         image_mask: Optional[ImageGrayscaleMask]=None
#     ) -> Tuple[
#         float,
#         float
#     ]:
#     if image_mask is not None:
#         pixel_data = pixel_data[image_mask]
#     pixel_count = numpy.product(pixel_data.shape)
#     if pixel_count == 0:
#         percent_maximal = 0
#         percent_minimal = 0
#     else:
#         number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))
#         number_pixels_minimal = numpy.sum(pixel_data == numpy.min(pixel_data))
#         percent_maximal = (
#             100.0 * float(number_pixels_maximal) / float(pixel_count)
#         )
#         percent_minimal = (
#             100.0 * float(number_pixels_minimal) / float(pixel_count)
#         )
#     return percent_minimal, percent_maximal


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def get_intensity_measurement_values(
#         pixels: ImageGrayscale, 
#         image_mask: Optional[ImageGrayscaleMask]=None
#     ) -> Tuple[
#         numpy.int_,
#         numpy.float_,
#         numpy.float_,
#         numpy.float_,
#         numpy.float_,
#         numpy.float_,
#         numpy.float_,
#         numpy.float_
#     ]:
#     if image_mask is not None:
#         pixels = pixels[image_mask]
    
#     pixel_count = numpy.product(pixels.shape)
#     if pixel_count == 0:
#         pixel_sum = 0
#         pixel_mean = 0
#         pixel_std = 0
#         pixel_mad = 0
#         pixel_median = 0
#         pixel_min = 0
#         pixel_max = 0
#     else:
#         pixel_sum = numpy.sum(pixels)
#         pixel_mean = pixel_sum / float(pixel_count)
#         pixel_std = numpy.std(pixels)
#         pixel_median = numpy.median(pixels)
#         pixel_mad = numpy.median(numpy.abs(pixels - pixel_median))
#         pixel_min = numpy.min(pixels)
#         pixel_max = numpy.max(pixels)
#     return pixel_count, pixel_sum, pixel_mean, pixel_std, pixel_mad, pixel_median, pixel_min, pixel_max


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def get_power_spectrum_measurement_values(
#         pixel_data: ImageGrayscale, 
#         image_mask: Optional[ImageGrayscaleMask]=None, 
#         dimensions: int = 2
#     ) -> float:
#     if dimensions == 3:
#         raise NotImplementedError("Power spectrum calculation for volumes is not implemented")

#     if image_mask is not None:
#         pixel_data = numpy.array(pixel_data)  # make a copy
#         masked_pixels = pixel_data[image_mask]
#         pixel_count = numpy.product(masked_pixels.shape)
#         if pixel_count > 0:
#             pixel_data[~image_mask] = numpy.mean(masked_pixels)
#         else:
#             pixel_data[~image_mask] = 0

#     radii, magnitude, power = centrosome.radial_power_spectrum.rps(pixel_data)
#     if sum(magnitude) > 0 and len(numpy.unique(pixel_data)) > 1:
#         valid = magnitude > 0
#         radii = radii[valid].reshape((-1, 1))
#         power = power[valid].reshape((-1, 1))
#         if radii.shape[0] > 1:
#             idx = numpy.isfinite(numpy.log(power))
#             powerslope = scipy.linalg.basic.lstsq(
#                 numpy.hstack(
#                     (
#                         numpy.log(radii)[idx][:, numpy.newaxis],
#                         numpy.ones(radii.shape)[idx][:, numpy.newaxis],
#                     )
#                 ),
#                 numpy.log(power)[idx][:, numpy.newaxis],
#             )[0][0]
#         else:
#             powerslope = 0
#     else:
#         powerslope = 0
#     return powerslope


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
# def calculate_threshold_for_threshold_group(
#         image_pixel_data: ImageGrayscale = numpy.ones((100, 100)),
#         mask: Optional[ImageGrayscaleMask] = None,
#         algorithm: str = centrosome.threshold.TM_OTSU,
#         object_fraction: float = 0.1,
#         two_class_otsu: bool = True,
#         use_weighted_variance: bool = True,
#         assign_middle_to_foreground: bool = True,
# ) -> float:
#     if mask is not None:
#         _, global_threshold = centrosome.threshold.get_threshold(
#             algorithm,
#             centrosome.threshold.TM_GLOBAL,
#             image_pixel_data,
#             mask = mask,
#             object_fraction=object_fraction,
#             two_class_otsu=two_class_otsu,
#             use_weighted_variance=use_weighted_variance,
#             assign_middle_to_foreground=assign_middle_to_foreground,
#         )
#     else:
#         _, global_threshold = centrosome.threshold.get_threshold(
#             algorithm,
#             centrosome.threshold.TM_GLOBAL,
#             image_pixel_data,
#             object_fraction=object_fraction,
#             two_class_otsu=two_class_otsu,
#             use_weighted_variance=use_weighted_variance,
#             assign_middle_to_foreground=assign_middle_to_foreground,
#         )
#     return global_threshold
    
    