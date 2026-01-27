import numpy
from numpy.typing import NDArray
from typing import Optional, Tuple, Annotated, Dict, List, Any
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.types import ObjectSegmentation, ObjectLabelSet
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.measureobjectneighbors import (
    DistanceMethod as NeighborsDistanceMethod,
    Measurement as NeighborsMeasurement,
    MeasurementScale as NeighborsMeasurementScale,
    C_NEIGHBORS
)
from cellprofiler_library.functions.measurement import measure_object_neighbors as _measure_object_neighbors
from cellprofiler_library.functions.segmentation import cast_labels_to_label_set, convert_label_set_to_ijv, indices_from_ijv, areas_from_ijv

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_neighbors(
        labels:                 Annotated[ObjectSegmentation, Field(description="Input labeled objects. Remember to include any small objects or objects that are on the edge")],
        kept_labels:            Annotated[ObjectSegmentation, Field(description="Input labels of interest. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        neighbor_labels:        Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        neighbor_kept_labels:   Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects of interest. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        object_name:            Annotated[str, Field(description="Name of the objects being measured")],
        neighbors_name:         Annotated[str, Field(description="Name of the neighbor objects")],
        neighbors_are_objects:  Annotated[bool, Field(description="Set to True if the neighbors are taken from the same object set as the input objects")],
        dimensions:             Annotated[int, Field(description="Use 2 for 2D and 3 for 3D")],
        distance_value:         Annotated[int, Field(description="Neighbor distance")],
        distance_method:        Annotated[NeighborsDistanceMethod, Field(description="Method to determine neighbors")], 
        kept_label_has_pixels:  Annotated[NDArray[numpy.bool_], Field(description="Array of booleans indicating whether each object has any pixels. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        nkept_objects:          Annotated[int, Field(description="Number of objects in the segmentation that are of interest (excluding objects that have been discarded for touching image border)")],
        wants_excluded_objects: Annotated[bool, Field(description="Consider objects discarded for touching image border?")]=True,
    ) -> Tuple[
        LibraryMeasurements,
        Tuple[NDArray[numpy.int_], NDArray[numpy.int_]],
        Optional[NDArray[numpy.int_]],
    ]:
    (
        neighbor_count,
        first_object_number,
        second_object_number,
        first_closest_distance,
        second_closest_distance,
        angle,
        percent_touching,
        first_objects,
        second_objects,
        expanded_labels,
    ) = _measure_object_neighbors(
        labels, 
        kept_labels,
        neighbor_labels, 
        neighbor_kept_labels,
        neighbors_are_objects,
        dimensions, 
        distance_value,
        distance_method, 
        kept_label_has_pixels,
        nkept_objects,
        wants_excluded_objects,
    )

    # Determine scale string
    if distance_method == NeighborsDistanceMethod.EXPAND:
        scale = NeighborsMeasurementScale.EXPANDED.value
    elif distance_method == NeighborsDistanceMethod.WITHIN:
        scale = str(distance_value)
    elif distance_method == NeighborsDistanceMethod.ADJACENT:
        scale = NeighborsMeasurementScale.ADJACENT.value
    else:
        raise ValueError(f"Unknown distance method: {distance_method}")

    # Helper to format feature name
    def get_feature_name(feature: str) -> str:
        if neighbors_are_objects:
             return "_".join((C_NEIGHBORS, feature, scale))
        else:
             return "_".join((C_NEIGHBORS, feature, neighbors_name, scale))

    measurements = LibraryMeasurements()
    
    features_and_data = [
        (NeighborsMeasurement.NUMBER_OF_NEIGHBORS.value, neighbor_count),
        (NeighborsMeasurement.FIRST_CLOSEST_OBJECT_NUMBER.value, first_object_number),
        (NeighborsMeasurement.FIRST_CLOSEST_DISTANCE.value, first_closest_distance),
        (NeighborsMeasurement.SECOND_CLOSEST_OBJECT_NUMBER.value, second_object_number),
        (NeighborsMeasurement.SECOND_CLOSEST_DISTANCE.value, second_closest_distance),
        (NeighborsMeasurement.ANGLE_BETWEEN_NEIGHBORS.value, angle),
        (NeighborsMeasurement.PERCENT_TOUCHING.value, percent_touching),
    ]

    for feature_name, data in features_and_data:
        full_name = get_feature_name(feature_name)
        measurements.add_measurement(object_name, full_name, data)

    # Calculate statistics
    for feature_name, data in features_and_data:
        full_name = get_feature_name(feature_name)
        
        # Skip stats for object IDs
        if feature_name in [NeighborsMeasurement.FIRST_CLOSEST_OBJECT_NUMBER.value, NeighborsMeasurement.SECOND_CLOSEST_OBJECT_NUMBER.value]:
            continue

        if len(data) > 0:
            measurements.add_image_measurement(f"Mean_{full_name}", float(numpy.mean(data)))
            measurements.add_image_measurement(f"Median_{full_name}", float(numpy.median(data)))
            measurements.add_image_measurement(f"Max_{full_name}", float(numpy.max(data)))
            measurements.add_image_measurement(f"Min_{full_name}", float(numpy.min(data)))
            measurements.add_image_measurement(f"StDev_{full_name}", float(numpy.std(data)))
        else:
            measurements.add_image_measurement(f"Mean_{full_name}", 0.0)
            measurements.add_image_measurement(f"Median_{full_name}", 0.0)
            measurements.add_image_measurement(f"Max_{full_name}", 0.0)
            measurements.add_image_measurement(f"Min_{full_name}", 0.0)
            measurements.add_image_measurement(f"StDev_{full_name}", 0.0)

    return measurements, (first_objects, second_objects), expanded_labels

def get_nkept_objects(
    kept_labels: Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects")],
    ) -> int:
    """This is a utility/helper function for the module which returns the number of objects in the segmentation"""
    kept_label_set = cast_labels_to_label_set(kept_labels)
    kept_label_ijv = convert_label_set_to_ijv(kept_label_set, validate=False)
    return len(indices_from_ijv(kept_label_ijv, validate=False))

def get_kept_label_has_pixels(
    kept_labels: Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects")],
    ) -> NDArray[numpy.bool_]:
    """This is a utility/helper function for the module which returns an array of booleans indicating whether each object has any pixels"""
    kept_label_set = cast_labels_to_label_set(kept_labels)
    kept_label_ijv = convert_label_set_to_ijv(kept_label_set, validate=False)
    return areas_from_ijv(kept_label_ijv) > 0
