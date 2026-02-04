from typing import Optional, Tuple, List, Union, Dict
from pydantic import validate_call, ConfigDict
import numpy as np
from numpy.typing import NDArray
import skimage.measure
import scipy.ndimage

from cellprofiler_library.types import ObjectSegmentation, ObjectSegmentationIJV
from cellprofiler_library.functions.segmentation import relate_children
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.relateobjects import TemplateMeasurementFormat
from cellprofiler_library.functions.measurement import (
    calculate_centroid_distances,
    calculate_minimum_distances
)
from cellprofiler_core.constants.measurement import (
    C_PARENT,
    FF_PARENT,
    FF_CHILDREN_COUNT,
    R_PARENT,
    R_CHILD,
    M_NUMBER_OBJECT_NUMBER,
)
R_PARENT = "Parent" # TODO: move this to opts/relateobjects.py
R_CHILD = "Children" # TODO: move this to opts/relateobjects.py
C_MEAN = "Mean"

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_relate_objects(
        parent_labels: ObjectSegmentation,
        child_labels: ObjectSegmentation,
        parent_ijv: Optional[ObjectSegmentationIJV],
        child_ijv: Optional[ObjectSegmentationIJV],
        parent_name: str,
        child_name: str,
        volumetric: bool = False,
        image_set_number: int = 1,
        module_num: int = 1,
        x_name: str = R_PARENT,
        y_name: str = R_CHILD,
        step_parent_names = [],
        wants_step_parent_disatnces: bool = False,
        find_centroid = False,
        find_minimum = False,
        child_dimensions: int = 2,
        wants_per_parent_means: bool = False,
        measurements: Optional[LibraryMeasurements] = None
) -> LibraryMeasurements:
    """
    Relate child objects to parent objects and compute basic statistics.
    
    Args:
        parent_labels: Segmentation of primary parent
        child_labels: Segmentation of children
        parent_ijv: Optional IJV for primary parent
        child_ijv: Optional IJV for children
        parent_name: Name of primary parent
        child_name: Name of children
        volumetric: Whether objects are 3D
    """

    def get_parent_names(x_name, wants_step_parent_distances, step_parent_names):
        parent_names = [x_name]

        if wants_step_parent_distances:
            parent_names += [
                group.step_parent_name.value for group in step_parent_names
            ]

        return parent_names
    
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
    lib_measurements.add_measurement(y_name, FF_PARENT % x_name, parents_of)
    lib_measurements.add_measurement(x_name, FF_CHILDREN_COUNT % y_name, child_count)

    good_parents = parents_of[parents_of != 0]

    image_numbers = np.ones(len(good_parents), int) * image_set_number
    good_children = np.argwhere(parents_of != 0).flatten() + 1

    if np.any(good_parents):
        lib_measurements.add_relate_measurement(
            module_num,
            R_PARENT,
            x_name,
            y_name,
            image_numbers,
            good_parents,
            image_numbers,
            good_children,
        )

        lib_measurements.add_relate_measurement(
            module_num,
            R_CHILD,
            y_name,
            x_name,
            image_numbers,
            good_children,
            image_numbers,
            good_parents,
        )
    parent_names = get_parent_names(x_name, wants_step_parent_disatnces, step_parent_names)

    for parent_name in parent_names:
        if find_centroid:
            dist = calculate_centroid_distances(parent_labels, child_labels, parents_of)
            lib_measurements.add_measurement(y_name, TemplateMeasurementFormat.FF_CENTROID % parent_name, dist)
            # Also add the measurement to the measurements object passed in. This is needed for the
            # RelateObjects module to calculate the per-parent means on these measurements.
            measurements.add_measurement(y_name, TemplateMeasurementFormat.FF_CENTROID % parent_name, dist)
        if find_minimum:
            dist = calculate_minimum_distances(parent_labels, child_labels, child_dimensions, parents_of)
            lib_measurements.add_measurement(y_name, TemplateMeasurementFormat.FF_MINIMUM % parent_name, dist)
            # Also add the measurement to the measurements object passed in. This is needed for the
            # RelateObjects module to calculate the per-parent means on these measurements.
            measurements.add_measurement(y_name, TemplateMeasurementFormat.FF_MINIMUM % parent_name, dist)

    def should_aggregate_feature(feature_name):
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
    if wants_per_parent_means:
        parent_indexes = np.arange(np.max(parent_labels)) + 1
        # Notice the for loop here iterates over measurements, not lib_measurements. This is because
        # the per-parent means are calculated on measurements processed by other modules (and 
        # made available to RelateObjects via the measurements argument).
        for feature_name in measurements.get_feature_names(y_name):
            if not should_aggregate_feature(feature_name):
                continue
            # Notice the for loop here iterates over measurements, not lib_measurements. See previous comment for explanation.
            data = measurements.get_measurement(y_name, feature_name) # changed get_current_measurement to get_measurement assuming that when get_meaurements is called they actually correspond to the correct measurements

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

            mean_feature_name = TemplateMeasurementFormat.FF_MEAN % (y_name, feature_name)

            lib_measurements.add_measurement(x_name, mean_feature_name, means)

    
    return lib_measurements
