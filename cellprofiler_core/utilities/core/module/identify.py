import numpy
import scipy.ndimage

from cellprofiler_core.constants.measurement import (
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    M_NUMBER_OBJECT_NUMBER,
    FF_COUNT,
    COLTYPE_FLOAT,
    COLTYPE_INTEGER,
    IMAGE,
    M_LOCATION_CENTER_Z,
)


def add_object_location_measurements(
    measurements, object_name, labels, object_count=None
):
    """Add the X and Y centers of mass to the measurements

    measurements - the measurements container
    object_name  - the name of the objects being measured
    labels       - the label matrix
    object_count - (optional) the object count if known, otherwise
                   takes the maximum value in the labels matrix which is
                   usually correct.
    """
    if object_count is None:
        object_count = numpy.max(labels)
    #
    # Get the centers of each object - center_of_mass <- list of two-tuples.
    #
    if object_count:
        centers = scipy.ndimage.center_of_mass(
            numpy.ones(labels.shape), labels, list(range(1, object_count + 1))
        )
        centers = numpy.array(centers)
        centers = centers.reshape((object_count, len(labels.shape)))
        if centers.shape[1] != 3:
            location_center_y = centers[:, 0]
            location_center_x = centers[:, 1]
        else:
            location_center_z = centers[:, 0]
            location_center_y = centers[:, 1]
            location_center_x = centers[:, 2]
        number = numpy.arange(1, object_count + 1)
    else:
        location_center_z = numpy.zeros((0,), dtype=float)
        location_center_y = numpy.zeros((0,), dtype=float)
        location_center_x = numpy.zeros((0,), dtype=float)
        number = numpy.zeros((0,), dtype=int)
    measurements.add_measurement(
        object_name, M_LOCATION_CENTER_X, location_center_x,
    )
    measurements.add_measurement(
        object_name, M_LOCATION_CENTER_Y, location_center_y,
    )
    if len(labels.shape) > 2:
        measurements.add_measurement(
            object_name, M_LOCATION_CENTER_Z, location_center_z,
        )

    measurements.add_measurement(
        object_name, M_NUMBER_OBJECT_NUMBER, number,
    )


def add_object_location_measurements_ijv(
    measurements, object_name, ijv, object_count=None
):
    """Add object location measurements for IJV-style objects"""
    if object_count is None:
        object_count = 0 if ijv.shape[0] == 0 else numpy.max(ijv[:, 2])
    if object_count == 0:
        center_x = numpy.zeros(0)
        center_y = numpy.zeros(0)
    else:
        areas = numpy.zeros(object_count, int)
        areas_bc = numpy.bincount(ijv[:, 2])[1:]
        areas[: len(areas_bc)] = areas_bc
        center_x = numpy.bincount(ijv[:, 2], ijv[:, 1])[1:] / areas
        center_y = numpy.bincount(ijv[:, 2], ijv[:, 0])[1:] / areas
    measurements.add_measurement(
        object_name, M_LOCATION_CENTER_X, center_x,
    )
    measurements.add_measurement(
        object_name, M_LOCATION_CENTER_Y, center_y,
    )
    measurements.add_measurement(
        object_name, M_NUMBER_OBJECT_NUMBER, numpy.arange(1, object_count + 1),
    )


def add_object_count_measurements(measurements, object_name, object_count):
    """Add the # of objects to the measurements"""
    measurements.add_measurement(
        "Image", FF_COUNT % object_name, numpy.array([object_count], dtype=float),
    )


def get_object_measurement_columns(object_name):
    """Get the column definitions for measurements made by identify modules

    Identify modules can use this call when implementing
    CPModule.get_measurement_columns to get the column definitions for
    the measurements made by add_object_location_measurements and
    add_object_count_measurements.
    """
    return [
        (object_name, M_LOCATION_CENTER_X, COLTYPE_FLOAT,),
        (object_name, M_LOCATION_CENTER_Y, COLTYPE_FLOAT,),
        (object_name, M_NUMBER_OBJECT_NUMBER, COLTYPE_INTEGER,),
        (IMAGE, FF_COUNT % object_name, COLTYPE_INTEGER,),
    ]
