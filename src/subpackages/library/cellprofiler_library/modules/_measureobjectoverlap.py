import numpy
from numpy.typing import NDArray
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Tuple, Union, Optional
from cellprofiler_library.opts.measureobjectoverlap import DecimationMethod, Feature, C_IMAGE_OVERLAP
from cellprofiler_library.types import ObjectLabelSet
from cellprofiler_library.functions.measurement import calculate_overlap_measurements, compute_earth_movers_distance_objects, get_labels_mask
from cellprofiler_library.measurement_model import LibraryMeasurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_overlap(
        objects_GT_labelset: Annotated[ObjectLabelSet, Field(description="Source objects segmentation")],
        objects_ID_labelset: Annotated[ObjectLabelSet, Field(description="Destination objects segmentation")],
        objects_GT_shape:   Annotated[Tuple[int, int], Field(description="Shape of the ground truth segmentation")], # Shape cannot be inferred from ijv
        objects_ID_shape:   Annotated[Tuple[int, int], Field(description="Shape of the test segmentation")], # Shape cannot be inferred from ijv
        object_name_GT:     Annotated[str, Field(description="Name of the ground truth objects")],
        object_name_ID:     Annotated[str, Field(description="Name of the test objects")],
        calcualte_emd:      Annotated[bool, Field(description="Calculate Earth Movers Distance")] = False,
        decimation_method:  Annotated[Optional[DecimationMethod], Field(description="Decimation method")] = DecimationMethod.KMEANS,
        max_distance:       Annotated[Optional[int], Field(description="Maximum distance")] = 250,
        penalize_missing:   Annotated[Optional[bool], Field(description="Penalize missing pixels")] = False,
        max_points:         Annotated[Optional[int], Field(description="Maximum # of points")] = 250,
) -> LibraryMeasurements:

    measurements = LibraryMeasurements()

    (
        F_factor,
        precision,
        recall,
        true_positive_rate,
        false_positive_rate,
        true_negative_rate,
        false_negative_rate,
        rand_index,
        adjusted_rand_index
    ) = calculate_overlap_measurements(
        objects_GT_labelset,
        objects_ID_labelset,
        objects_GT_shape,
        objects_ID_shape,
    )
    
    def get_measurement_name(feature_name: str) -> str:
        return f"{C_IMAGE_OVERLAP}_{feature_name}_{object_name_GT}_{object_name_ID}"

    measurements.add_image_measurement(get_measurement_name(Feature.F_FACTOR), F_factor)
    measurements.add_image_measurement(get_measurement_name(Feature.PRECISION), precision)
    measurements.add_image_measurement(get_measurement_name(Feature.RECALL), recall)
    measurements.add_image_measurement(get_measurement_name(Feature.TRUE_POS_RATE), true_positive_rate)
    measurements.add_image_measurement(get_measurement_name(Feature.FALSE_POS_RATE), false_positive_rate)
    measurements.add_image_measurement(get_measurement_name(Feature.TRUE_NEG_RATE), true_negative_rate)
    measurements.add_image_measurement(get_measurement_name(Feature.FALSE_NEG_RATE), false_negative_rate)
    measurements.add_image_measurement(get_measurement_name(Feature.RAND_INDEX), rand_index)
    measurements.add_image_measurement(get_measurement_name(Feature.ADJUSTED_RAND_INDEX), adjusted_rand_index)

    if calcualte_emd:
        assert decimation_method is not None, "Decimation method must be provided for Earth Movers Distance calculation"
        assert max_distance is not None, "Maximum distance must be provided for Earth Movers Distance calculation"
        assert max_points is not None, "Maximum points must be provided for Earth Movers Distance calculation"
        assert penalize_missing is not None, "Penalize missing must be provided for Earth Movers Distance calculation"
        emd = compute_earth_movers_distance_objects(
            src_objects_label_set=objects_ID_labelset,
            dest_objects_label_set=objects_GT_labelset,
            decimation_method=decimation_method,
            max_distance=max_distance,
            max_points=max_points,
            penalize_missing=penalize_missing,
        )
        measurements.add_image_measurement(get_measurement_name(Feature.EARTH_MOVERS_DISTANCE), emd)
    
    return measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_overlap_masks(
    objects_GT_labelset: Annotated[ObjectLabelSet, Field(description="Source objects segmentation")],
    objects_ID_labelset: Annotated[ObjectLabelSet, Field(description="Destination objects segmentation")],
    objects_GT_shape:   Annotated[Tuple[int, int], Field(description="Shape of the ground truth segmentation")], 
    objects_ID_shape:   Annotated[Tuple[int, int], Field(description="Shape of the test segmentation")], 
) -> Tuple[NDArray[numpy.bool_], NDArray[numpy.bool_]]:
    """Helper to reconstruct binary masks for visualization"""
    gt_mask = get_labels_mask(objects_GT_labelset, objects_GT_shape)
    id_mask = get_labels_mask(objects_ID_labelset, objects_ID_shape)
    return gt_mask, id_mask
