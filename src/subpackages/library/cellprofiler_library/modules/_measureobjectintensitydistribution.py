import numpy
import centrosome.zernike
from pydantic import Field, validate_call, ConfigDict
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, List, Union, Any
from cellprofiler_library.types import ObjectSegmentation, ObjectLabelSet, Image2DGrayscale, Image2DGrayscaleMask, ObjectLabel, ObjectSegmentationIJV
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv, indices_from_ijv
from cellprofiler_library.opts.measureobjectintensitydistribution import CenterChoice, IntensityZernike, MeasurementFeature, OverflowFeature
from cellprofiler_library.functions.measurement import get_normalized_distance_centers_and_good_mask, get_bin_measurements, get_radial_index, get_radial_cv, get_positions_within_unit_circle, calculate_zernikes_for_image, get_zernike_magnitude_name, get_zernike_phase_name

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_object_intensity_zernikes(
        objects_names_and_label_sets: List[Tuple[str, ObjectLabelSet]], 
        zernike_degree: int,
        image_name_data_mask_list: List[Tuple[str, Image2DGrayscale, Image2DGrayscaleMask]], 
        zernike_opts: IntensityZernike
    ):
    measurements_dict = {}
    zernike_indexes = centrosome.zernike.get_zernike_indexes(
            zernike_degree + 1
        )
    for object_name, objects_labels in objects_names_and_label_sets:
        measurements_dict_for_image = {} if object_name not in measurements_dict else measurements_dict[object_name]
        objects_ijv = convert_label_set_to_ijv(objects_labels)
        objects_indices = indices_from_ijv(objects_ijv)
        
        l, yx, z = get_positions_within_unit_circle(objects_labels, zernike_indexes)
        for image_name, image_pixel_data, image_mask in image_name_data_mask_list:
            measurements_dict_for_image = calculate_zernikes_for_image(
                image_pixel_data, 
                image_mask, 
                image_name,
                objects_ijv, 
                objects_indices,
                zernike_opts,
                zernike_indexes,
                l, 
                yx, 
                z, 
            )
            measurements_dict[object_name] = measurements_dict_for_image
    return measurements_dict

HeatMapDictType = Dict[Union[int, str], Union[NDArray[numpy.float64], Any]]
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_object_intensity_distribution_measurements(
        object_name: str, 
        center_object_name: Optional[str], 
        heatmap_dict, # cannot use HeatMapDictType because pydantic is creating a copy of the dict when an annotation is specified
        labels: ObjectSegmentation, # these are obtained after `cropping similarly` to the image
        center_object_segmented: Optional[ObjectSegmentation], 
        center_choice: CenterChoice,
        wants_scaled: bool, 
        maximum_radius: int, # default 100, minimum 1
        bin_count: int, # default 4, minval 2
        pixel_data: Image2DGrayscale,
        nobjects: int,
        image_name: str,
        heatmaps,
        objects_indices: Optional[NDArray[ObjectLabel]]
    ):
    name = (
        object_name
        if center_object_name is None
        else "{}_{}".format(object_name, center_object_name)
    )
    if objects_indices is None:
        objects_ijv = convert_label_set_to_ijv(labels)
        objects_indices = indices_from_ijv(objects_ijv)

    if name in heatmap_dict:
        normalized_distance, i_center, j_center, good_mask = heatmap_dict[name]
    else:
        if center_object_name is not None:
            center_object_segmentations = center_object_segmented
        else:
            center_object_segmentations = None

        normalized_distance, i_center, j_center, good_mask = get_normalized_distance_centers_and_good_mask(labels, center_object_segmentations, objects_indices, center_choice, wants_scaled, maximum_radius)
        heatmap_dict[name] = [normalized_distance, i_center, j_center, good_mask]
    (
        bin_indexes, 
        fraction_at_distance, 
        mean_pixel_fraction, 
        masked_fraction_at_distance, 
        masked_mean_pixel_fraction, 
    ) = get_bin_measurements(good_mask, labels, normalized_distance, bin_count, pixel_data, nobjects)
    
    radial_index = get_radial_index(labels, good_mask, i_center, j_center)

    statistics = []
    measurements = []

    for bin in range(bin_count + (0 if wants_scaled else 1)):
        (
            bin_mask, 
            bin_labels, 
            radial_cv, 
            mask
        ) = get_radial_cv(good_mask, bin_indexes, labels, radial_index, pixel_data, nobjects, bin)

        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin], MeasurementFeature.FRAC_AT_D.value, OverflowFeature.FRAC_AT_D.value),
            (mean_pixel_fraction[:, bin], MeasurementFeature.MEAN_FRAC.value, OverflowFeature.MEAN_FRAC.value),
            (numpy.array(radial_cv), MeasurementFeature.RADIAL_CV.value, OverflowFeature.RADIAL_CV.value),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature % image_name
            else:
                measurement_name = feature % (image_name, bin + 1, bin_count)

            measurements.append([object_name, measurement_name, measurement])

            if feature in heatmaps:
                heatmaps[feature][bin_mask] = measurement[bin_labels - 1]

        radial_cv.mask = numpy.sum(~mask, 1) == 0

        bin_name = str(bin + 1) if bin < bin_count else "Overflow"

        statistics += [
            (
                image_name,
                object_name,
                bin_name,
                str(bin_count),
                numpy.round(numpy.mean(masked_fraction_at_distance[:, bin]), 4),
                numpy.round(numpy.mean(masked_mean_pixel_fraction[:, bin]), 4),
                numpy.round(numpy.mean(radial_cv), 4),
            )
        ]

    return statistics, measurements
