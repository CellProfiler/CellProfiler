from cellprofiler_library.measurement_model import LibraryMeasurements

import numpy
import centrosome.zernike
from pydantic import Field, validate_call, ConfigDict
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, List, Union, Any, Annotated
from cellprofiler_library.types import ObjectSegmentation, ObjectLabelSet, Image2DGrayscale, Image2DGrayscaleMask, ObjectLabel, ObjectSegmentationIJV
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv, indices_from_ijv
from cellprofiler_library.opts.measureobjectintensitydistribution import CenterChoice, IntensityZernike, MeasurementFeature, OverflowFeature
from cellprofiler_library.functions.measurement import get_normalized_distance_centers_and_good_mask, get_bin_measurements, get_radial_index, get_radial_cv, get_positions_within_unit_circle, calculate_zernikes_for_image, get_zernike_magnitude_name, get_zernike_phase_name

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def compute_zernike_geometry(
    labels: Annotated[ObjectLabelSet, Field(description="Segmentation of objects")],
    zernike_degree: int
) -> Tuple[Any, Any, Any, Any]:
    """Compute geometry needed for Zernike calculations."""
    zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    l, yx, z = get_positions_within_unit_circle(labels, zernike_indexes)
    return l, yx, z, zernike_indexes

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_intensity_zernikes(
    image: Image2DGrayscale,
    image_mask: Image2DGrayscaleMask,
    image_name: str,
    object_name: str,
    labels: ObjectLabelSet,
    zernike_degree: int,
    zernike_opts: IntensityZernike,
    geometry: Optional[Tuple] = None,
    objects_indices: Optional[NDArray[ObjectLabel]] = None
) -> LibraryMeasurements:
    """Compute intensity Zernike moments for a single object-image pair."""
    measurements = LibraryMeasurements()
    
    if geometry is None:
        l, yx, z, zernike_indexes = compute_zernike_geometry(labels, zernike_degree)
    else:
        l, yx, z, zernike_indexes = geometry

    if objects_indices is None:
        objects_ijv = convert_label_set_to_ijv(labels)
        objects_indices = indices_from_ijv(objects_ijv)
    else:
        # We need ijv for calculate_zernikes_for_image
        objects_ijv = convert_label_set_to_ijv(labels)
    
    results = calculate_zernikes_for_image(
        image, 
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
    
    for feature_name, values in results.items():
         measurements.add_measurement(object_name, feature_name, values)
         
    return measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def compute_radial_distribution_geometry(
    labels: ObjectSegmentation,
    center_object_segmentations: Optional[ObjectSegmentation],
    objects_indices: Optional[NDArray[ObjectLabel]],
    center_choice: CenterChoice,
    wants_scaled: bool,
    maximum_radius: int
) -> Tuple[Any, Any, Any, Any]:
     """Compute geometry needed for radial distribution measurements."""
     return get_normalized_distance_centers_and_good_mask(
        labels, center_object_segmentations, objects_indices, center_choice, wants_scaled, maximum_radius
     )

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_intensity_distribution(
    image: Image2DGrayscale,
    image_name: str,
    object_name: str,
    labels: ObjectSegmentation,
    nobjects: int,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    center_choice: CenterChoice,
    center_object_name: Optional[str] = None,
    center_object_labels: Optional[ObjectSegmentation] = None,
    geometry: Optional[Tuple] = None,
    objects_indices: Optional[NDArray[ObjectLabel]] = None,
    heatmap_features: Optional[List[str]] = None, 
    return_visualization_data: bool = False
) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, List[Tuple], Dict[str, NDArray]]]:
    """Compute radial intensity distribution measurements for a single object-image pair."""
    
    measurements = LibraryMeasurements()
    
    if geometry is None:
        if objects_indices is None:
            if nobjects > 0:
                objects_ijv = convert_label_set_to_ijv(labels)
                objects_indices = indices_from_ijv(objects_ijv)
            else:
                objects_indices = numpy.zeros(0, dtype=numpy.int32)
            
        if nobjects > 0:
            normalized_distance, i_center, j_center, good_mask = compute_radial_distribution_geometry(
                labels, center_object_labels, objects_indices, center_choice, wants_scaled, maximum_radius
            )
        else:
            normalized_distance = numpy.zeros(0)
            i_center = numpy.zeros(0, int)
            j_center = numpy.zeros(0, int)
            good_mask = numpy.zeros(labels.shape, bool)
    else:
        normalized_distance, i_center, j_center, good_mask = geometry

    if nobjects > 0:
        (
            bin_indexes, 
            fraction_at_distance, 
            mean_pixel_fraction, 
            masked_fraction_at_distance, 
            masked_mean_pixel_fraction, 
        ) = get_bin_measurements(good_mask, labels, normalized_distance, bin_count, image, nobjects)
    else:
        # Return empty arrays with correct shapes
        # fraction_at_distance: (nobjects, bin_count) -> (0, bin_count)
        # mean_pixel_fraction: (nobjects, bin_count)
        # masked_...: (nobjects, bin_count)
        # bin_indexes: shape of good_mask? No, it's 1D array of values for good pixels.
        # But if nobjects=0, good_mask is all False.
        bin_indexes = numpy.zeros(0, dtype=numpy.int32)
        shape = (0, bin_count)
        fraction_at_distance = numpy.zeros(shape)
        mean_pixel_fraction = numpy.zeros(shape)
        masked_fraction_at_distance = numpy.zeros(shape)
        masked_mean_pixel_fraction = numpy.zeros(shape)
        
    if nobjects > 0:
        radial_index = get_radial_index(labels, good_mask, i_center, j_center)
    else:
        radial_index = numpy.zeros(0, dtype=numpy.int32)

    statistics = []
    heatmap_data = {}
    
    # Initialize heatmap arrays if needed
    if heatmap_features:
        for feature in heatmap_features:
            heatmap_data[feature] = numpy.zeros(labels.shape)

    for bin in range(bin_count + (0 if wants_scaled else 1)):
        if nobjects > 0:
            (
                bin_mask, 
                bin_labels, 
                radial_cv, 
                mask
            ) = get_radial_cv(good_mask, bin_indexes, labels, radial_index, image, nobjects, bin)
        else:
            bin_mask = numpy.zeros(labels.shape, bool)
            bin_labels = numpy.zeros(0, dtype=numpy.int32)
            radial_cv = numpy.zeros(0)
            mask = numpy.zeros((0, 8), bool) # get_radial_cv returns mask for 8 wedges?
            # radial_cv should be array of length nobjects?
            # yes.
        
        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin], MeasurementFeature.FRAC_AT_D.value, OverflowFeature.FRAC_AT_D.value),
            (mean_pixel_fraction[:, bin], MeasurementFeature.MEAN_FRAC.value, OverflowFeature.MEAN_FRAC.value),
            (numpy.array(radial_cv), MeasurementFeature.RADIAL_CV.value, OverflowFeature.RADIAL_CV.value),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature % image_name
            else:
                measurement_name = feature % (image_name, bin + 1, bin_count)

            measurements.add_measurement(object_name, measurement_name, measurement)

            if heatmap_features and feature in heatmap_features:
                 heatmap_data[feature][bin_mask] = measurement[bin_labels - 1]

        if nobjects > 0:
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
        
    if return_visualization_data:
        return measurements, statistics, heatmap_data
    else:
        return measurements
