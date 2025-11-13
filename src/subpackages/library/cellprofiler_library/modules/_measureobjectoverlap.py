import numpy
from numpy.typing import NDArray
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Tuple, Union, Optional
from cellprofiler_library.opts.measureobjectoverlap import DecimationMethod
from cellprofiler_library.types import ObjectLabelSet
from cellprofiler_library.functions.measurement import calculate_overlap_measurements, compute_earth_movers_distance_objects

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_overlap(
        objects_GT_labelset: Annotated[ObjectLabelSet, Field(description="Source objects segmentation")],
        objects_ID_labelset: Annotated[ObjectLabelSet, Field(description="Destination objects segmentation")],
        objects_GT_shape:   Annotated[Tuple[int, int], Field(description="Shape of the ground truth segmentation")], # Shape cannot be inferred from ijv
        objects_ID_shape:   Annotated[Tuple[int, int], Field(description="Shape of the test segmentation")], # Shape cannot be inferred from ijv
        calcualte_emd:      Annotated[bool, Field(description="Calculate Earth Movers Distance")] = False,
        decimation_method:  Annotated[Optional[DecimationMethod], Field(description="Decimation method")] = DecimationMethod.KMEANS,
        max_distance:       Annotated[Optional[int], Field(description="Maximum distance")] = 250,
        penalize_missing:   Annotated[Optional[bool], Field(description="Penalize missing pixels")] = False,
        max_points:         Annotated[Optional[int], Field(description="Maximum # of points")] = 250,
) -> Tuple[
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    Union[numpy.float_, float],
    NDArray[numpy.float64],
    NDArray[numpy.float64],
    int,
    int,
    Optional[numpy.float_],
    ]:

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
    emd = None
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
    return (
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
    emd
    )
