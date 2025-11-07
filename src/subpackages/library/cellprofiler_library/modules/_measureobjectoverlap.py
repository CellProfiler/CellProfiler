import numpy
from numpy.typing import NDArray
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Tuple, Union
from cellprofiler_library.opts.measureobjectoverlap import DecimationMethod
from cellprofiler_library.types import ObjectSegmentationIJV, ObjectLabelSet
from cellprofiler_library.functions.segmentation import areas_from_ijv, areas_from_ijv
from cellprofiler_library.functions.measurement import object_overlap_confusion_matrix, get_dominating_ID_objects, get_intersect_matrix, compute_rand_index_ijv, compute_earth_movers_distance_objects

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_overlap(
        objects_GT_ijv:     Annotated[ObjectSegmentationIJV, Field(description="ijv representation of the ground truth segmentation")],
        objects_ID_ijv:     Annotated[ObjectSegmentationIJV, Field(description="ijv representation of the test segmentation")],
        objects_GT_shape:   Annotated[Tuple[int, int], Field(description="shape of the ground truth segmentation")], # Shape cannot be inferred from ijv
        objects_ID_shape:   Annotated[Tuple[int, int], Field(description="shape of the test segmentation")], # Shape cannot be inferred from ijv
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
    ]:
    gt_areas: NDArray[numpy.int_] = areas_from_ijv(objects_GT_ijv)
    id_areas: NDArray[numpy.int_] = areas_from_ijv(objects_ID_ijv)

    iGT, jGT, lGT = objects_GT_ijv.transpose()
    iID, jID, lID = objects_ID_ijv.transpose()

    ID_obj = 0 if len(lID) == 0 else max(lID)
    GT_obj = 0 if len(lGT) == 0 else max(lGT)

    xGT, yGT = objects_GT_shape
    xID, yID = objects_ID_shape
    GT_pixels = numpy.zeros((xGT, yGT))
    ID_pixels = numpy.zeros((xID, yID))
    total_pixels = xGT * yGT

    GT_pixels[iGT, jGT] = 1
    ID_pixels[iID, jID] = 1

    GT_tot_area = len(iGT)

    #
    # Intersect matrix [i, j] = number of pixels that are common among object number i from the ground truth and object number j from the test
    #
    intersect_matrix = get_intersect_matrix(iGT, jGT, lGT, iID, jID, lID, ID_obj, GT_obj)
    FN_area = gt_areas[numpy.newaxis, :] - intersect_matrix
    
    #
    # for each object in the ground truth, find the object in the test that has the highest overlap
    #
    dom_ID = get_dominating_ID_objects(ID_obj, lID, iID, jID, ID_pixels, intersect_matrix)
    
    for i in range(0, len(intersect_matrix.T)):
        if len(numpy.where(dom_ID == i)[0]) > 1:
            final_id = numpy.where(intersect_matrix.T[i] == max(intersect_matrix.T[i]))
            final_id = final_id[0][0]
            all_id = numpy.where(dom_ID == i)[0]
            nonfinal = [x for x in all_id if x != final_id]
            for (n) in nonfinal:  # these others cannot be candidates for the corr ID now
                intersect_matrix.T[i][n] = 0
        else:
            continue
        
    TP, FN, FP, TN = object_overlap_confusion_matrix(dom_ID, id_areas, intersect_matrix, FN_area, total_pixels)

    def nan_divide(numerator, denominator):
        if denominator == 0:
            return numpy.nan
        return numpy.float64(float(numerator) / float(denominator))
    recall = nan_divide(TP, GT_tot_area)
    precision = nan_divide(TP, (TP + FP))
    F_factor = nan_divide(2 * (precision * recall), (precision + recall))
    true_positive_rate = nan_divide(TP, (FN + TP))
    false_positive_rate = nan_divide(FP, (FP + TN))
    false_negative_rate = nan_divide(FN, (FN + TP))
    true_negative_rate = nan_divide(TN, (FP + TN))
    shape = numpy.maximum(
        numpy.maximum(numpy.array(objects_GT_shape), numpy.array(objects_ID_shape)),
        numpy.ones(2, int),
    )
    rand_index, adjusted_rand_index = compute_rand_index_ijv(
        objects_GT_ijv, objects_ID_ijv, shape
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
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def compute_emd(
        src_objects_labels: Annotated[ObjectLabelSet, Field(description="Source objects segmentation")],
        dest_objects_labels: Annotated[ObjectLabelSet, Field(description="Destination objects segmentation")],
        decimation_method: Annotated[DecimationMethod, Field(description="Decimation method")]=DecimationMethod.KMEANS,
        max_distance: Annotated[int, Field(description="Maximum distance")]=250,
        max_points: Annotated[int, Field(description="Maximum # of points")]=250,
        penalize_missing: Annotated[bool, Field(description="Penalize missing pixels")]=False,
        ) -> numpy.float_:
    return compute_earth_movers_distance_objects(
        src_objects_labels=src_objects_labels,
        dest_objects_labels=dest_objects_labels,
        decimation_method=decimation_method,
        max_distance=max_distance,
        max_points=max_points,
        penalize_missing=penalize_missing,
    )