import numpy
import scipy.ndimage
from typing import Annotated, Optional
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.types import ObjectSegmentation, ObjectSegmentationIJV
from cellprofiler_library.functions.segmentation import center_of_labels_mass, relate_children
from cellprofiler_library.opts.measurement import (
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    M_LOCATION_CENTER_Z,
    M_NUMBER_OBJECT_NUMBER,
    FF_COUNT,
    FF_PARENT,
    FF_CHILDREN_COUNT,
)
from cellprofiler_library.measurements.measurement_model import LibraryMeasurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def wrap_object_location_measurements(
    object_name: Annotated[str, Field(description="Name of the objects being measured")],
    labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the objects")],
    object_count: Annotated[
        Optional[int],
        Field(description="Number of objects in labels, if known. Otherwise taken as the max label value."),
    ] = None,
) -> LibraryMeasurements:
    """Return the X/Y(/Z) centers of mass and object numbers for the given objects"""
    measurements = LibraryMeasurements()
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
    return measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def wrap_object_count_measurements(
    object_name: Annotated[str, Field(description="Name of the objects being counted")],
    object_count: Annotated[int, Field(description="Number of objects")],
) -> LibraryMeasurements:
    """Return the image-level object count measurement"""
    lib_measurements = LibraryMeasurements()
    lib_measurements.add_image_measurement(
        FF_COUNT % object_name, numpy.array([object_count], dtype=float),
    )
    return lib_measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def wrap_relate_object_measurements(
    object_labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the child objects")],
    object_volumetric: Annotated[bool, Field(description="Whether the objects are volumetric (3D)")],
    object_name: Annotated[str, Field(description="Name of the child objects")],
    object_ijv: Annotated[ObjectSegmentationIJV, Field(description="Child object segmentation in IJV format")],
    parent_object_name: Annotated[str, Field(description="Name of the parent objects")],
    parent_object_labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the parent objects")],
    parent_object_ijv: Annotated[ObjectSegmentationIJV, Field(description="Parent object segmentation in IJV format")],
) -> LibraryMeasurements:
    """Return parent/child relate measurements: children-per-parent count and parent-of-child"""
    lib_measurements = LibraryMeasurements()
    children_per_parent, parents_of_children = relate_children(
        parent_object_labels,
        object_labels,
        parent_object_ijv,
        object_ijv,
        volumetric=object_volumetric,
    )
    lib_measurements.add_measurement(
        parent_object_name,
        FF_CHILDREN_COUNT % object_name,
        children_per_parent,
    )

    lib_measurements.add_measurement(
        object_name, FF_PARENT % parent_object_name, parents_of_children,
    )
    return lib_measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def wrap_image_segmentation_measurements(
    object_labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the objects")],
    object_volumetric: Annotated[bool, Field(description="Whether the objects are volumetric (3D)")],
    object_count: Annotated[int, Field(description="Number of objects")],
    object_name: Annotated[str, Field(description="Name of the objects being measured")],
) -> LibraryMeasurements:
    """Return the location and count measurements produced by ImageSegmentation.add_measurements"""
    lib_measurements = LibraryMeasurements()
    centers = center_of_labels_mass(object_labels, validate=False)

    if len(centers) == 0:
        center_z, center_y, center_x = [], [], []
    else:
        if object_volumetric:
            center_z, center_y, center_x = centers.transpose()
        else:
            center_z = [0] * len(centers)

            center_y, center_x = centers.transpose()

    lib_measurements.add_measurement(
        object_name, M_LOCATION_CENTER_X, center_x,
    )

    lib_measurements.add_measurement(
        object_name, M_LOCATION_CENTER_Y, center_y,
    )

    lib_measurements.add_measurement(
        object_name, M_LOCATION_CENTER_Z, center_z,
    )

    lib_measurements.add_measurement(
        object_name, M_NUMBER_OBJECT_NUMBER, numpy.arange(1, object_count + 1),
    )

    lib_measurements = lib_measurements.merge(
        wrap_object_count_measurements(object_name, object_count)
    )
    return lib_measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def wrap_object_processing_measurements(
    object_labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the output objects")],
    object_volumetric: Annotated[bool, Field(description="Whether the output objects are volumetric (3D)")],
    object_count: Annotated[int, Field(description="Number of output objects")],
    object_name: Annotated[str, Field(description="Name of the output objects")],
    object_ijv: Annotated[ObjectSegmentationIJV, Field(description="Output object segmentation in IJV format")],
    parent_object_name: Annotated[str, Field(description="Name of the input (parent) objects")],
    parent_object_labels: Annotated[ObjectSegmentation, Field(description="Dense label matrix of the input objects")],
    parent_object_ijv: Annotated[ObjectSegmentationIJV, Field(description="Input object segmentation in IJV format")],
) -> LibraryMeasurements:
    """Return the measurements produced by ObjectProcessing.add_measurements"""
    lib_measurements = wrap_image_segmentation_measurements(
        object_labels, object_volumetric, object_count, object_name
    )
    lib_measurements_relate = wrap_relate_object_measurements(
        object_labels, object_volumetric, object_name, object_ijv,
        parent_object_name, parent_object_labels, parent_object_ijv,
    )
    lib_measurements = lib_measurements.merge(lib_measurements_relate)

    return lib_measurements
