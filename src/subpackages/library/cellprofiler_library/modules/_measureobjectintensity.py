import numpy
import skimage.segmentation

from numpy.typing import NDArray
from typing import Tuple, Annotated, Optional, List, Any, Union
from pydantic import Field, validate_call, ConfigDict, BaseModel
from cellprofiler_core.utilities.core.object import crop_labels_and_image
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, ObjectLabelSet, Pixel, ObjectLabel
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.functions.measurement import measure_object_area_occupied, measure_integrated_intensity, measure_mean_intensity, measure_std_intensity, measure_min_intensity, measure_max_intensity, measure_max_position, measure_center_of_mass_binary, measure_center_of_mass_intensity, measure_mass_displacement, measure_quartile_intensity
from cellprofiler_library.opts.measureobjectintensity import TemplateMeasurementFormat, IntensityFeature


ObjectIntensityStatistics = List[Tuple[str,str,str,Any,Any,Any]]

class ObjectIntensityDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    statistics: ObjectIntensityStatistics

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_location_measurements(
        masked_image_shape: Tuple[int, ...],
        lmask: NDArray[numpy.bool_],
        limg: NDArray[Pixel],
        llabels: NDArray[ObjectLabel],
        lindexes: NDArray[numpy.int_],
        integrated_intensity: NDArray[numpy.float_],
):
    mesh_z, mesh_y, mesh_x = numpy.mgrid[
        0 : masked_image_shape[0],
        0 : masked_image_shape[1],
        0 : masked_image_shape[2],
    ]

    mesh_x = mesh_x[lmask]
    mesh_y = mesh_y[lmask]
    mesh_z = mesh_z[lmask]
    # Compute the position of the intensity maximum
    max_position = measure_max_position(limg, llabels, lindexes)

    # Get the coordinates of the maximum intensity
    _max_x = mesh_x[max_position]
    _max_y = mesh_y[max_position]
    _max_z = mesh_z[max_position]

    # The mass displacement is the distance between the center
    # of mass of the binary image and of the intensity image. The
    # center of mass is the average X or Y for the binary image
    # and the sum of X or Y * intensity / integrated intensity
    cm_x = measure_center_of_mass_binary(mesh_x, llabels, lindexes)
    cm_y = measure_center_of_mass_binary(mesh_y, llabels, lindexes)
    cm_z = measure_center_of_mass_binary(mesh_z, llabels, lindexes)

    _cmi_x = measure_center_of_mass_intensity(mesh_x, limg, llabels, lindexes, integrated_intensity)
    _cmi_y = measure_center_of_mass_intensity(mesh_y, limg, llabels, lindexes, integrated_intensity)
    _cmi_z = measure_center_of_mass_intensity(mesh_z, limg, llabels, lindexes, integrated_intensity)
    _mass_displacement = measure_mass_displacement((cm_x, cm_y, cm_z), (_cmi_x, _cmi_y, _cmi_z))
    return (
        (_max_x, _max_y, _max_z),
        (_cmi_x, _cmi_y, _cmi_z),
        _mass_displacement,
    )


def get_intensity_measurements(limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> Tuple[NDArray[numpy.float_], ...]:
    lcount = measure_object_area_occupied(limg, llabels, lindexes)
    _integrated_intensity = measure_integrated_intensity(limg, llabels, lindexes)
    _mean_intensity = measure_mean_intensity(_integrated_intensity, lcount)
    _std_intensity = measure_std_intensity(limg, llabels, lindexes, _mean_intensity)
    _min_intensity = measure_min_intensity(limg, llabels, lindexes)
    _max_intensity = measure_max_intensity(limg, llabels, lindexes)
    return _integrated_intensity, _mean_intensity, _std_intensity, _min_intensity, _max_intensity

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_intensity(
        img:                Annotated[ImageGrayscale, Field(description="Image to measure")],
        image_name:         Annotated[str, Field(description="Name of the image")],
        object_name:        Annotated[str, Field(description="Name of the object set")],
        image_mask:         Annotated[Optional[ImageGrayscaleMask], Field(description="Mask of the image")],
        object_labels:      Annotated[ObjectLabelSet, Field(description="Object labels")],
        nobjects:           Annotated[int, Field(description="Number of objects in object_labels")],
        image_dimensions:   Annotated[int, Field(description="For 2D images, this is 2. For 3D images, this is 3")],
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
        ) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, ObjectIntensityDisplayData]]:
    """
    Compute intensity measurements for objects in an image.
    
    Args:
        img: Input image pixel data
        image_name: Name of the image (for measurement key construction)
        object_name: Name of the object set (for measurement key construction)
        image_mask: Mask of the image (Optional)
        object_labels: Object label matrix
        nobjects: Number of objects
        image_dimensions: 2 for 2D images, 3 for 3D images
        
    Returns:
        LibraryMeasurements instance with all computed intensity measurements
    """
    image_has_mask = image_mask is not None
    if image_has_mask:
        masked_image = img.copy()
        masked_image[~image_mask] = 0
    else:
        masked_image = img
        image_mask = numpy.ones_like(img, dtype=bool)

    if image_dimensions == 2:
        img = img.reshape(1, *img.shape)
        masked_image = masked_image.reshape(1, *masked_image.shape)
        image_mask = image_mask.reshape(1, *image_mask.shape)

    integrated_intensity = numpy.zeros((nobjects,))
    integrated_intensity_edge = numpy.zeros((nobjects,))
    mean_intensity = numpy.zeros((nobjects,))
    mean_intensity_edge = numpy.zeros((nobjects,))
    std_intensity = numpy.zeros((nobjects,))
    std_intensity_edge = numpy.zeros((nobjects,))
    min_intensity = numpy.zeros((nobjects,))
    min_intensity_edge = numpy.zeros((nobjects,))
    max_intensity = numpy.zeros((nobjects,))
    max_intensity_edge = numpy.zeros((nobjects,))
    mass_displacement = numpy.zeros((nobjects,))
    lower_quartile_intensity = numpy.zeros((nobjects,))
    median_intensity = numpy.zeros((nobjects,))
    mad_intensity = numpy.zeros((nobjects,))
    upper_quartile_intensity = numpy.zeros((nobjects,))
    cmi_x = numpy.zeros((nobjects,))
    cmi_y = numpy.zeros((nobjects,))
    cmi_z = numpy.zeros((nobjects,))
    max_x = numpy.zeros((nobjects,))
    max_y = numpy.zeros((nobjects,))
    max_z = numpy.zeros((nobjects,))
    for labels, lindexes in object_labels:
        lindexes = lindexes[lindexes != 0]

        if image_dimensions == 2:
            labels = labels.reshape(1, *labels.shape)

        labels, img = crop_labels_and_image(labels, img)
        _, masked_image = crop_labels_and_image(labels, masked_image)
        outlines = skimage.segmentation.find_boundaries(
            labels, mode="inner"
        )

        if image_has_mask:
            _, mask = crop_labels_and_image(labels, image_mask)
            masked_labels = labels.copy()
            masked_labels[~mask] = 0
            masked_outlines = outlines.copy()
            masked_outlines[~mask] = 0
        else:
            masked_labels = labels
            masked_outlines = outlines

        lmask = masked_labels > 0 & numpy.isfinite(img)  # Ignore NaNs, Infs
        has_objects = numpy.any(lmask)
        if has_objects:
            limg: NDArray[Pixel] = img[lmask] # This is a 1D array of pixels
            llabels: NDArray[ObjectLabel] = labels[lmask] # This is a 1D array of labels

            (
                _integrated_intensity,
                _mean_intensity,
                _std_intensity,
                _min_intensity,
                _max_intensity
            ) = get_intensity_measurements(limg, llabels, lindexes)

            (
                _max_positions,
                _center_of_mass_intensities,
                _mass_displacement,
            ) = get_location_measurements(
                masked_image.shape,
                lmask,
                limg,
                llabels,
                lindexes,
                _integrated_intensity,
            )

            integrated_intensity[lindexes - 1] = _integrated_intensity
            mean_intensity[lindexes - 1] = _mean_intensity
            std_intensity[lindexes - 1] = _std_intensity
            min_intensity[lindexes - 1] = _min_intensity
            max_intensity[lindexes - 1] = _max_intensity
            max_x[lindexes - 1] = _max_positions[0]
            max_y[lindexes - 1] = _max_positions[1]
            max_z[lindexes - 1] = _max_positions[2]
            cmi_x[lindexes - 1] = _center_of_mass_intensities[0]
            cmi_y[lindexes - 1] = _center_of_mass_intensities[1]
            cmi_z[lindexes - 1] = _center_of_mass_intensities[2]
            mass_displacement[lindexes - 1] = _mass_displacement

            #
            # Sort the intensities by label, then intensity.
            # For each label, find the index above and below
            # the 25%, 50% and 75% mark and take the weighted
            # average.
            #
            areas = measure_object_area_occupied(limg, llabels, lindexes).astype(int)
            indices = numpy.cumsum(areas) - areas
            order = numpy.lexsort((limg, llabels))
            for dest, fraction in (
                (lower_quartile_intensity, 1.0 / 4.0),
                (median_intensity, 1.0 / 2.0),
                (upper_quartile_intensity, 3.0 / 4.0),
            ):
                qmask, _dest, qmask_no_upper, _dest_no_upper = measure_quartile_intensity(indices, areas, fraction, limg, order)
                dest[lindexes[qmask] - 1] = _dest
                dest[lindexes[qmask_no_upper] - 1] = _dest_no_upper

            #
            # Once again, for the MAD
            #
            fraction = 1.0/ image_dimensions
            madimg = numpy.abs(limg - median_intensity[llabels - 1])
            order = numpy.lexsort((madimg, llabels))

            qmask, _mad_intensity, qmask_no_upper, _mad_intensity_no_upper = measure_quartile_intensity(indices, areas, fraction, madimg, order)
            mad_intensity[lindexes[qmask] - 1] = _mad_intensity
            mad_intensity[lindexes[qmask_no_upper] - 1] = _mad_intensity_no_upper

        emask = masked_outlines > 0
        eimg = img[emask]
        elabels = labels[emask]
        has_edge = len(eimg) > 0

        if has_edge:
            (
                _integrated_intensity_edge,
                _mean_intensity_edge,                        
                _std_intensity_edge,
                _min_intensity_edge,
                _max_intensity_edge,
            ) = get_intensity_measurements(eimg, elabels, lindexes)

            integrated_intensity_edge[lindexes - 1] = _integrated_intensity_edge
            mean_intensity_edge[lindexes - 1] = _mean_intensity_edge
            std_intensity_edge[lindexes - 1] = _std_intensity_edge
            min_intensity_edge[lindexes - 1] = _min_intensity_edge
            max_intensity_edge[lindexes - 1] = _max_intensity_edge

    measurements = LibraryMeasurements()
    statistics: ObjectIntensityStatistics = []

    def add_measurement(feature_name: str, frmt_template: str, measurement_val: NDArray[numpy.float_]):
        measurements.add_measurement(object_name, frmt_template % image_name, measurement_val)

        statistics.append((
            image_name,
            object_name,
            feature_name,
            numpy.round(numpy.mean(measurement_val), 3),
            numpy.round(numpy.median(measurement_val), 3),
            numpy.round(numpy.std(measurement_val), 3),
        ))

    # Intensity measurements
    add_measurement(IntensityFeature.INTEGRATED_INTENSITY.value, TemplateMeasurementFormat.INTEGRATED_INTENSITY, integrated_intensity)
    add_measurement(IntensityFeature.MEAN_INTENSITY.value, TemplateMeasurementFormat.MEAN_INTENSITY, mean_intensity)
    add_measurement(IntensityFeature.STD_INTENSITY.value, TemplateMeasurementFormat.STD_INTENSITY, std_intensity)
    add_measurement(IntensityFeature.MIN_INTENSITY.value, TemplateMeasurementFormat.MIN_INTENSITY, min_intensity)
    add_measurement(IntensityFeature.MAX_INTENSITY.value, TemplateMeasurementFormat.MAX_INTENSITY, max_intensity)
    add_measurement(IntensityFeature.INTEGRATED_INTENSITY_EDGE.value, TemplateMeasurementFormat.INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge)
    add_measurement(IntensityFeature.MEAN_INTENSITY_EDGE.value, TemplateMeasurementFormat.MEAN_INTENSITY_EDGE, mean_intensity_edge)
    add_measurement(IntensityFeature.STD_INTENSITY_EDGE.value, TemplateMeasurementFormat.STD_INTENSITY_EDGE, std_intensity_edge)
    add_measurement(IntensityFeature.MIN_INTENSITY_EDGE.value, TemplateMeasurementFormat.MIN_INTENSITY_EDGE, min_intensity_edge)
    add_measurement(IntensityFeature.MAX_INTENSITY_EDGE.value, TemplateMeasurementFormat.MAX_INTENSITY_EDGE, max_intensity_edge)
    add_measurement(IntensityFeature.MASS_DISPLACEMENT.value, TemplateMeasurementFormat.MASS_DISPLACEMENT, mass_displacement)
    add_measurement(IntensityFeature.LOWER_QUARTILE_INTENSITY.value, TemplateMeasurementFormat.LOWER_QUARTILE_INTENSITY, lower_quartile_intensity)
    add_measurement(IntensityFeature.MEDIAN_INTENSITY.value, TemplateMeasurementFormat.MEDIAN_INTENSITY, median_intensity)
    add_measurement(IntensityFeature.MAD_INTENSITY.value, TemplateMeasurementFormat.MAD_INTENSITY, mad_intensity)
    add_measurement(IntensityFeature.UPPER_QUARTILE_INTENSITY.value, TemplateMeasurementFormat.UPPER_QUARTILE_INTENSITY, upper_quartile_intensity)

    # Location measurements (Using the updated Location prefix)
    add_measurement(IntensityFeature.LOC_CMI_X.value, TemplateMeasurementFormat.LOC_CMI_X, cmi_x)
    add_measurement(IntensityFeature.LOC_CMI_Y.value, TemplateMeasurementFormat.LOC_CMI_Y, cmi_y)
    add_measurement(IntensityFeature.LOC_CMI_Z.value, TemplateMeasurementFormat.LOC_CMI_Z, cmi_z)
    add_measurement(IntensityFeature.LOC_MAX_X.value, TemplateMeasurementFormat.LOC_MAX_X, max_x)
    add_measurement(IntensityFeature.LOC_MAX_Y.value, TemplateMeasurementFormat.LOC_MAX_Y, max_y)
    add_measurement(IntensityFeature.LOC_MAX_Z.value, TemplateMeasurementFormat.LOC_MAX_Z, max_z)

    if return_visualization_data:
        return measurements, ObjectIntensityDisplayData(statistics=statistics)
    return measurements
