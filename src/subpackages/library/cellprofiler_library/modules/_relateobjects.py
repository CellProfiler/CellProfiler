from typing import Optional, Annotated, List
from pydantic import validate_call, ConfigDict, Field
import numpy as np
import scipy.ndimage

from cellprofiler_library.types import ObjectSegmentation, ObjectSegmentationIJV
from cellprofiler_library.functions.segmentation import relate_children
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.relateobjects import TemplateMeasurementFormat, Relationship, C_MEAN, C_PARENT, M_NUMBER_OBJECT_NUMBER
from cellprofiler_library.functions.measurement import (
    calculate_centroid_distances,
    calculate_minimum_distances,
    find_parents_of
)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_relate_objects(
        parent_labels:          Annotated[ObjectSegmentation, Field(description="Segmentation of parent")],
        child_labels:           Annotated[ObjectSegmentation, Field(description="Segmentation of children")],
        parent_ijv:             Annotated[Optional[ObjectSegmentationIJV], Field(description="Segmentation of parent in IJV format")],
        child_ijv:              Annotated[Optional[ObjectSegmentationIJV], Field(description="Segmentation of children in IJV format")],
        parent_name:            Annotated[str, Field(description="Name of the parent object")] = Relationship.PARENT.value,
        child_name:             Annotated[str, Field(description="Name of the child object")] = Relationship.CHILD.value,
        volumetric:             Annotated[bool, Field(description="Indicates whether objects are 3D")] = False,
        parent_and_step_parent_names:Annotated[List[str], Field(description="List of parents and step-parent names for which to calculate distances")] = [],
        find_centroid:          Annotated[bool, Field(description="Indicates whether centroid-centroid distances should be calculated")] = False,
        find_minimum:           Annotated[bool, Field(description="Indicates whether minimum distances should be calculated")] = False,
        child_dimensions:       Annotated[int, Field(description="Number of dimensions of the child object")] = 2,
        wants_per_parent_means: Annotated[bool, Field(description="Inidicates whether per-parent means should be calculated")] = False,
        measurements:           Annotated[Optional[LibraryMeasurements], Field(description="Measurements object for which per-parent means will be calculated")] = None,
) -> LibraryMeasurements:
    """
    Relate child objects to parent objects, compute basic statistics, and optionally calculate distances between parent and child objects, and optionally calculate per-parent means.
    
    Args:
        parent_labels: Segmentation of primary parent
        child_labels: Segmentation of children
        parent_ijv: Optional IJV for primary parent
        child_ijv: Optional IJV for children
        parent_name: Name of primary parent
        child_name: Name of children
        volumetric: Whether objects are 3D
        parent_and_step_parent_names: List of parents and step-parent names for which to calculate distances
        find_centroid: Indicates whether centroid-centroid distances should be calculated
        find_minimum: Indicates whether minimum distances should be calculated
        child_dimensions: Number of dimensions of the child object
        wants_per_parent_means: Inidicates whether per-parent means should be calculated
        measurements: Measurements object for which per-parent means will be calculated
    """

    # lib_measurements is the object that will be returned
    lib_measurements = LibraryMeasurements()

    # RelateObject module produces per-parent means for all child measurements. The measurements over which
    # the means are calculated come from other modules. We accept a LibraryMeasurements object as an argument
    # so that the user can pass in said measurements.
    if measurements is None:
        measurements = LibraryMeasurements()
    
    # Calculate relationship for primary parent
    child_count, parents_of = relate_children(
        parent_labels, 
        child_labels, 
        parent_ijv, 
        child_ijv, 
        volumetric
    )
    #
    # Note: Later in the code, you will see add_measurements calls to both lib_measurements and measurements.
    # The reason for this is that the RelateObjects module optionally procuces per-parent means (aka aggregation) on the measurements
    # passed in. The optional `measurements` argument should not be merged with the measurements returned by RelateObjects (i.e. lib_measurements) as
    # the output of this module should be restricted to the measurements produced by RelateObjects.
    #
    lib_measurements.add_measurement(child_name, TemplateMeasurementFormat.FF_PARENT % parent_name, parents_of)
    # No need to add the folloiwng measurement to measurements as it is not an aggregate measurement (see __should_aggregate_feature() below)
    # measurements.add_measurement(child_name, FF_PARENT % parent_name, parents_of)
    lib_measurements.add_measurement(parent_name, TemplateMeasurementFormat.FF_CHILDREN_COUNT % child_name, child_count)
    # No need to add the folloiwng measurement to measurements as it is not an aggregate measurement (see __should_aggregate_feature() below)
    # measurements.add_measurement(parent_name, FF_CHILDREN_COUNT % child_name, child_count)

    good_parents = parents_of[parents_of != 0]

    good_children = np.argwhere(parents_of != 0).flatten() + 1

    if np.any(good_parents):
        lib_measurements.add_relate_measurement(
            Relationship.PARENT.value,
            parent_name,
            child_name,
            good_parents,
            good_children,
        )
        # No need to add relate measurements to measurements as aggregation cannot be performed on relationships

        lib_measurements.add_relate_measurement(
            Relationship.CHILD.value,
            child_name,
            parent_name,
            good_children,
            good_parents,
        )
        # No need to add relate measurements to measurements as aggregation cannot be performed on relationships
    
    for parent_step_parent_name in parent_and_step_parent_names:
        merged_measurements = measurements.merge(lib_measurements)
        parents_of = find_parents_of(parent_step_parent_name, parent_name, child_name, merged_measurements)
        if find_centroid:
            dist = calculate_centroid_distances(parent_labels, child_labels, parents_of)
            lib_measurements.add_measurement(child_name, TemplateMeasurementFormat.FF_CENTROID % parent_step_parent_name, dist)
            # Also add the measurement to the measurements object passed in. This is needed for the
            # RelateObjects module to calculate the per-parent means on these measurements.
            # Not the most elegant solution, but this is good enough
            measurements.add_measurement(child_name, TemplateMeasurementFormat.FF_CENTROID % parent_step_parent_name, dist)
        if find_minimum:
            dist = calculate_minimum_distances(parent_labels, child_labels, child_dimensions, parents_of)
            lib_measurements.add_measurement(child_name, TemplateMeasurementFormat.FF_MINIMUM % parent_step_parent_name, dist)
            # Also add the measurement to the measurements object passed in. This is needed for the
            # RelateObjects module to calculate the per-parent means on these measurements.
            # Perfect is the enemy of good enough
            measurements.add_measurement(child_name, TemplateMeasurementFormat.FF_MINIMUM % parent_step_parent_name, dist)

    if wants_per_parent_means:
        parent_indexes = np.arange(np.max(parent_labels)) + 1
        # Notice the for loop below iterates over measurements, not lib_measurements. This is because
        # the per-parent means are calculated on measurements processed by other modules (and 
        # made available to RelateObjects via the measurements argument).
        for feature_name in measurements.get_feature_names(child_name):
            if not __should_aggregate_feature(feature_name):
                continue
            # Notice the statement below uses measurements, not lib_measurements. See previous comment for explanation.
            data = measurements.get_measurement(child_name, feature_name) # changed get_current_measurement to get_measurement assuming that when get_meaurements is called they actually correspond to the correct measurements

            if data is not None and len(data) > 0:
                if len(parents_of) > 0:
                    means = scipy.ndimage.mean(
                        data.astype(float), parents_of, parent_indexes
                    )
                else:
                    means = np.zeros((0,))
            else:
                # No child measurements - all NaN
                means = np.ones(len(parents_of)) * np.nan

            mean_feature_name = TemplateMeasurementFormat.FF_MEAN % (child_name, feature_name)

            lib_measurements.add_measurement(parent_name, mean_feature_name, means)

    return lib_measurements

def __should_aggregate_feature(feature_name: str) -> bool:
    """Return True if aggregate measurements should be made on a feature

    feature_name - name of a measurement, such as Location_Center_X
    """
    if feature_name.startswith(C_MEAN):
        return False

    if feature_name.startswith(C_PARENT):
        return False

    if feature_name in set(M_NUMBER_OBJECT_NUMBER):
        return False

    return True