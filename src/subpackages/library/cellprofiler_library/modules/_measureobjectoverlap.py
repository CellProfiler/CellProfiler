import numpy
from numpy.typing import NDArray
from pydantic import Field, validate_call, ConfigDict, BaseModel
from typing import Annotated, Tuple, Union, Optional, List, Any
from cellprofiler_library.opts.measureobjectoverlap import DecimationMethod, Feature, C_IMAGE_OVERLAP
from cellprofiler_library.types import ObjectLabelSet
from cellprofiler_library.functions.measurement import calculate_overlap_measurements, compute_earth_movers_distance_objects
from cellprofiler_library.measurement_model import LibraryMeasurements

ObjectOverlapStatistics = List[Tuple[str, float]]

class ObjectOverlapDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    statistics: ObjectOverlapStatistics
    true_positives: float
    true_negatives: float
    false_positives: float
    false_negatives: float


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
        return_visualization_data: Annotated[bool, Field(description="Return GT_pixels and ID_pixels for visualization")] = False,
) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, ObjectOverlapDisplayData]]:

    measurements = LibraryMeasurements()
    statistics: ObjectOverlapStatistics = []

    (
        F_factor,
        precision,
        recall,
        true_positive_rate,
        false_positive_rate,
        true_negative_rate,
        false_negative_rate,
        rand_index,
        adjusted_rand_index,
        GT_pixels,
        ID_pixels,
        xGT, 
        yGT,
    ) = calculate_overlap_measurements(
        objects_GT_labelset,
        objects_ID_labelset,
        objects_GT_shape,
        objects_ID_shape,
    )
    
    # Helper to construct measurement names
    def get_measurement_name(feature_name: str) -> str:
        return f"{C_IMAGE_OVERLAP}_{feature_name}_{object_name_GT}_{object_name_ID}"
    
    def add_measurement(feature_name: str, measurement_val: float):
        measurements.add_image_measurement(get_measurement_name(feature_name), measurement_val)
        if return_visualization_data:
            statistics.append((feature_name, measurement_val))

    add_measurement(Feature.F_FACTOR.value, float(F_factor))
    add_measurement(Feature.PRECISION.value, float(precision))
    add_measurement(Feature.RECALL.value, float(recall))
    add_measurement(Feature.TRUE_POS_RATE.value, float(true_positive_rate))
    add_measurement(Feature.FALSE_POS_RATE.value, float(false_positive_rate))
    add_measurement(Feature.TRUE_NEG_RATE.value, float(true_negative_rate))
    add_measurement(Feature.FALSE_NEG_RATE.value, float(false_negative_rate))
    add_measurement(Feature.RAND_INDEX.value, float(rand_index))
    add_measurement(Feature.ADJUSTED_RAND_INDEX.value, float(adjusted_rand_index))

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
        add_measurement(Feature.EARTH_MOVERS_DISTANCE, float(emd))
    
    if return_visualization_data:
        def subscripts(condition1: int, condition2: int):
            x1, y1 = numpy.where(GT_pixels == condition1)
            x2, y2 = numpy.where(ID_pixels == condition2)
            mask = set(zip(x1, y1)) & set(zip(x2, y2))
            return list(mask)

        TP_mask = subscripts(1, 1)
        FN_mask = subscripts(1, 0)
        FP_mask = subscripts(0, 1)
        TN_mask = subscripts(0, 0)

        TP_pixels = numpy.zeros((xGT, yGT))
        FN_pixels = numpy.zeros((xGT, yGT))
        FP_pixels = numpy.zeros((xGT, yGT))
        TN_pixels = numpy.zeros((xGT, yGT))

        def maskimg(mask: List[Tuple[Any, Any]], img: NDArray[numpy.float_]):
            for ea in mask:
                img[ea] = 1
            return img

        TP_pixels = maskimg(TP_mask, TP_pixels)
        FN_pixels = maskimg(FN_mask, FN_pixels)
        FP_pixels = maskimg(FP_mask, FP_pixels)
        TN_pixels = maskimg(TN_mask, TN_pixels)
        
        display_data = ObjectOverlapDisplayData(
            statistics = statistics,
            true_positives = float(TP_pixels),
            true_negatives = float(TN_pixels),
            false_positives = float(FP_pixels),
            false_negatives = float(FN_pixels),
        )
        return measurements, display_data
    else:
        return measurements
