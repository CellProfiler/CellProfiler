import centrosome.cpmorphology
import centrosome.filter
import centrosome.outline
import numpy
import scipy.ndimage
import skimage.segmentation
from cellprofiler_core.constants.measurement import C_LOCATION, COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, ValidationError
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)
from cellprofiler_core.utilities.core.object import crop_labels_and_image
from typing import Tuple
from cellprofiler.modules import _help
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, ObjectLabelSet, Pixel, ObjectLabel
from numpy.typing import NDArray

__doc__ = """
MeasureObjectIntensity
======================

**MeasureObjectIntensity** measures several intensity features for
identified objects.

Given an image with objects identified (e.g., nuclei or cells), this
module extracts intensity features for each object based on one or more
corresponding grayscale images. Measurements are recorded for each
object.

Intensity measurements are made for all combinations of the images and
objects entered. If you want only specific image/object measurements,
you can use multiple MeasureObjectIntensity modules for each group of
measurements desired.

{HELP_ON_MEASURING_INTENSITIES}

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **NamesAndTypes**, **MeasureImageIntensity**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *IntegratedIntensity:* The sum of the pixel intensities within an
   object.
-  *MeanIntensity:* The average pixel intensity within an object.
-  *StdIntensity:* The standard deviation of the pixel intensities
   within an object.
-  *MaxIntensity:* The maximal pixel intensity within an object.
-  *MinIntensity:* The minimal pixel intensity within an object.
-  *IntegratedIntensityEdge:* The sum of the edge pixel intensities of
   an object.
-  *MeanIntensityEdge:* The average edge pixel intensity of an object.
-  *StdIntensityEdge:* The standard deviation of the edge pixel
   intensities of an object.
-  *MaxIntensityEdge:* The maximal edge pixel intensity of an object.
-  *MinIntensityEdge:* The minimal edge pixel intensity of an object.
-  *MassDisplacement:* The distance between the centers of gravity in
   the gray-level representation of the object and the binary
   representation of the object.
-  *LowerQuartileIntensity:* The intensity value of the pixel for which
   25% of the pixels in the object have lower values.
-  *MedianIntensity:* The median intensity value within the object.
-  *MADIntensity:* The median absolute deviation (MAD) value of the
   intensities within the object. The MAD is defined as the
   median(\|x\ :sub:`i` - median(x)\|).
-  *UpperQuartileIntensity:* The intensity value of the pixel for which
   75% of the pixels in the object have lower values.
-  *Location\_CenterMassIntensity\_X, Location\_CenterMassIntensity\_Y:*
   The (X,Y) coordinates of the intensity weighted centroid (=
   center of mass = first moment) of all pixels within the object.
-  *Location\_MaxIntensity\_X, Location\_MaxIntensity\_Y:* The
   (X,Y) coordinates of the pixel with the maximum intensity within the
   object.

""".format(
    **{"HELP_ON_MEASURING_INTENSITIES": _help.HELP_ON_MEASURING_INTENSITIES}
)

INTENSITY = "Intensity"
INTEGRATED_INTENSITY = "IntegratedIntensity"
MEAN_INTENSITY = "MeanIntensity"
STD_INTENSITY = "StdIntensity"
MIN_INTENSITY = "MinIntensity"
MAX_INTENSITY = "MaxIntensity"
INTEGRATED_INTENSITY_EDGE = "IntegratedIntensityEdge"
MEAN_INTENSITY_EDGE = "MeanIntensityEdge"
STD_INTENSITY_EDGE = "StdIntensityEdge"
MIN_INTENSITY_EDGE = "MinIntensityEdge"
MAX_INTENSITY_EDGE = "MaxIntensityEdge"
MASS_DISPLACEMENT = "MassDisplacement"
LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"
MEDIAN_INTENSITY = "MedianIntensity"
MAD_INTENSITY = "MADIntensity"
UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
LOC_CMI_X = "CenterMassIntensity_X"
LOC_CMI_Y = "CenterMassIntensity_Y"
LOC_CMI_Z = "CenterMassIntensity_Z"
LOC_MAX_X = "MaxIntensity_X"
LOC_MAX_Y = "MaxIntensity_Y"
LOC_MAX_Z = "MaxIntensity_Z"

ALL_MEASUREMENTS = [
    INTEGRATED_INTENSITY,
    MEAN_INTENSITY,
    STD_INTENSITY,
    MIN_INTENSITY,
    MAX_INTENSITY,
    INTEGRATED_INTENSITY_EDGE,
    MEAN_INTENSITY_EDGE,
    STD_INTENSITY_EDGE,
    MIN_INTENSITY_EDGE,
    MAX_INTENSITY_EDGE,
    MASS_DISPLACEMENT,
    LOWER_QUARTILE_INTENSITY,
    MEDIAN_INTENSITY,
    MAD_INTENSITY,
    UPPER_QUARTILE_INTENSITY,
]
ALL_LOCATION_MEASUREMENTS = [
    LOC_CMI_X,
    LOC_CMI_Y,
    LOC_CMI_Z,
    LOC_MAX_X,
    LOC_MAX_Y,
    LOC_MAX_Z,
]


class MeasureObjectIntensity(Module):
    module_name = "MeasureObjectIntensity"
    variable_revision_number = 4
    category = "Measurement"

    def create_settings(self):
        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select the grayscale images whose intensity you want to measure.""",
        )
        self.divider = Divider()
        self.objects_list = LabelListSubscriber(
            "Select objects to measure",
            [],
            doc="""Select the object sets whose intensity you want to measure.""",
        )

    def settings(self):
        result = [self.images_list, self.objects_list]
        return result

    def visible_settings(self):
        result = [self.images_list, self.divider, self.objects_list]
        return result

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 2:
            num_imgs = setting_values.index("Do not use")
            setting_values = (
                [str(num_imgs)]
                + setting_values[:num_imgs]
                + setting_values[num_imgs + 1 :]
            )
            variable_revision_number = 3
        if variable_revision_number == 3:
            num_imgs = int(setting_values[0])
            images_list = setting_values[1 : num_imgs + 1]
            objects_list = setting_values[num_imgs + 1 :]
            setting_values = [
                ", ".join(map(str, images_list)),
                ", ".join(map(str, objects_list)),
            ]
            variable_revision_number = 4
        return setting_values, variable_revision_number

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        images = set()
        if len(self.images_list.value) == 0:
            raise ValidationError("No images selected", self.images_list)
        elif len(self.objects_list.value) == 0:
            raise ValidationError("No objects selected", self.objects_list)
        for image_name in self.images_list.value:
            if image_name in images:
                raise ValidationError(
                    "%s has already been selected" % image_name, image_name
                )
            images.add(image_name)

        objects = set()
        for object_name in self.objects_list.value:
            if object_name in objects:
                raise ValidationError(
                    "%s has already been selected" % object_name, object_name
                )
            objects.add(object_name)

    def get_measurement_columns(self, pipeline):
        """Return the column definitions for measurements made by this module"""
        columns = []
        for image_name in self.images_list.value:
            for object_name in self.objects_list.value:
                for category, features in (
                    (INTENSITY, ALL_MEASUREMENTS),
                    (C_LOCATION, ALL_LOCATION_MEASUREMENTS,),
                ):
                    for feature in features:
                        columns.append(
                            (
                                object_name,
                                "%s_%s_%s" % (category, feature, image_name),
                                COLTYPE_FLOAT,
                            )
                        )

        return columns

    def get_categories(self, pipeline, object_name):
        """Get the categories of measurements supplied for the given object name

        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        for object_set in self.objects_list.value:
            if object_set == object_name:
                return [INTENSITY, C_LOCATION]
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Get the measurements made on the given object in the given category"""
        if category == C_LOCATION:
            all_measurements = ALL_LOCATION_MEASUREMENTS
        elif category == INTENSITY:
            all_measurements = ALL_MEASUREMENTS
        else:
            return []
        for object_set in self.objects_list.value:
            if object_set == object_name:
                return all_measurements
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Get the images used to make the given measurement in the given category on the given object"""
        if category == INTENSITY:
            if measurement not in ALL_MEASUREMENTS:
                return []
        elif category == C_LOCATION:
            if measurement not in ALL_LOCATION_MEASUREMENTS:
                return []
        else:
            return []
        for object_set in self.objects_list.value:
            if object_set == object_name:
                return self.images_list.value
        return []
    
    ###############################################################################
    # Helpers
    ###############################################################################
    def get_count(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.int_]:
        return centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(numpy.ones(len(limg)), llabels, lindexes))

    def get_integrated_intensity(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
        return centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(limg, llabels, lindexes))
    
    def get_mean_intensity(self, integrated_intensity: NDArray[numpy.float_], lcount: NDArray[numpy.int_]) -> NDArray[numpy.float_]:   
        return integrated_intensity / lcount
    
    def get_std_intensity(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_], mean_intensity: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
        #
        # This function takes in mean_intensity as an array where each element is the mean intensity of the corresponding label in lindexes
        # which is then converted into a 1D array of pixels where each pixel is the mean intensity of the corresponding label in llabels
        # It's done this way as it makes the code more readable in the main function
        #
        mean_intensity_per_label = numpy.zeros((max(lindexes),))
        mean_intensity_per_label[lindexes - 1] = mean_intensity
        # mean_intensity[llabels - 1] replaces label numbers with mean intensity creating a 1D array of intensities
        mean_intensity_pixels = mean_intensity_per_label[llabels - 1]
        return numpy.sqrt(
            centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(
                    (limg - mean_intensity_pixels) ** 2,
                    llabels,
                    lindexes,
                )
            )
        )
    def get_min_intensity(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
        return centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.minimum(limg, llabels, lindexes)
        )
    
    def get_max_intensity(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
        return centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.maximum(limg, llabels, lindexes)
        )
    
    def get_max_position(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
        # Compute the position of the intensity maximum
        max_position = numpy.array(
            centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.maximum_position(limg, llabels, lindexes)
            ),
            dtype=int,
        )
        max_position = numpy.reshape(
            max_position, (max_position.shape[0],)
        )
        return max_position
    
    def get_center_of_mass_binary(self, coordinates: NDArray[numpy.int_], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> NDArray[numpy.float_]:
        cm = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.mean(coordinates, llabels, lindexes)
            )
        return cm
    
    def get_center_of_mass_intensity(self, coordinates_mesh: NDArray[numpy.int_], limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_], integrated_intensity: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
        coordinate_scaled_pixels = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.sum(coordinates_mesh * limg, llabels, lindexes)
        )
        center_of_mass_intensity = coordinate_scaled_pixels / integrated_intensity
        return center_of_mass_intensity
    
    def get_mass_displacement(self, center_of_mass_binary: Tuple[NDArray[numpy.float_], NDArray[numpy.float_], NDArray[numpy.float_]], center_of_mass_intensity: Tuple[NDArray[numpy.float_], NDArray[numpy.float_], NDArray[numpy.float_]]) -> NDArray[numpy.float_]:
        diff_x = center_of_mass_binary[0] - center_of_mass_intensity[0]
        diff_y = center_of_mass_binary[1] - center_of_mass_intensity[1]
        diff_z = center_of_mass_binary[2] - center_of_mass_intensity[2]
        mass_displacement = numpy.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
        return mass_displacement
    
    # TODO: what to name this function?
    def foo(self, indices:NDArray[numpy.int_], areas: NDArray[numpy.int_], fraction: float, limg: NDArray[Pixel], order):
        qindex = indices.astype(float) + areas * fraction
        qfraction = qindex - numpy.floor(qindex)
        qindex = qindex.astype(int)
        qmask = qindex < indices + areas - 1
        qi = qindex[qmask]
        qf = qfraction[qmask]
        _dest = (limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf)
        #
        # In some situations (e.g., only 3 points), there may
        # not be an upper bound.
        #
        qmask_no_upper = (~qmask) & (areas > 0)
        dest_no_upper = limg[order[qindex[qmask_no_upper]]]
        return qmask, _dest, qmask_no_upper, dest_no_upper
    
    def get_location_measurements(
            self,
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
        max_position = self.get_max_position(limg, llabels, lindexes)

        # Get the coordinates of the maximum intensity
        _max_x = mesh_x[max_position]
        _max_y = mesh_y[max_position]
        _max_z = mesh_z[max_position]

        # The mass displacement is the distance between the center
        # of mass of the binary image and of the intensity image. The
        # center of mass is the average X or Y for the binary image
        # and the sum of X or Y * intensity / integrated intensity
        cm_x = self.get_center_of_mass_binary(mesh_x, llabels, lindexes)
        cm_y = self.get_center_of_mass_binary(mesh_y, llabels, lindexes)
        cm_z = self.get_center_of_mass_binary(mesh_z, llabels, lindexes)

        _cmi_x = self.get_center_of_mass_intensity(mesh_x, limg, llabels, lindexes, integrated_intensity)
        _cmi_y = self.get_center_of_mass_intensity(mesh_y, limg, llabels, lindexes, integrated_intensity)
        _cmi_z = self.get_center_of_mass_intensity(mesh_z, limg, llabels, lindexes, integrated_intensity)
        _mass_displacement = self.get_mass_displacement((cm_x, cm_y, cm_z), (_cmi_x, _cmi_y, _cmi_z))
        return (
            (_max_x, _max_y, _max_z),
            (_cmi_x, _cmi_y, _cmi_z),
            _mass_displacement,
        )


    def get_intensity_measurements(self, limg: NDArray[Pixel], llabels: NDArray[ObjectLabel], lindexes: NDArray[numpy.int_]) -> Tuple[NDArray[numpy.float_], ...]:
        lcount = self.get_count(limg, llabels, lindexes)
        _integrated_intensity = self.get_integrated_intensity(limg, llabels, lindexes)
        _mean_intensity = self.get_mean_intensity(_integrated_intensity, lcount)
        _std_intensity = self.get_std_intensity(limg, llabels, lindexes, _mean_intensity)
        _min_intensity = self.get_min_intensity(limg, llabels, lindexes)
        _max_intensity = self.get_max_intensity(limg, llabels, lindexes)
        return _integrated_intensity, _mean_intensity, _std_intensity, _min_intensity, _max_intensity

    def get_measureobjectintensity_measurements(self, img: ImageGrayscale, image_mask: ImageGrayscaleMask, object_labels: ObjectLabelSet, nobjects: int, image_dimensions: int, image_has_mask: bool=False) -> Tuple[NDArray[numpy.float64], ...]:
        if image_has_mask:
            masked_image = img.copy()
            masked_image[~image_mask] = 0
            image_mask = image_mask # TODO: check if this is needed
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
                ) = self.get_intensity_measurements(limg, llabels, lindexes)

                (
                    _max_positions,
                    _center_of_mass_intensities,
                    _mass_displacement,
                ) = self.get_location_measurements(
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
                areas = self.get_count(limg, llabels, lindexes).astype(int)
                indices = numpy.cumsum(areas) - areas
                order = numpy.lexsort((limg, llabels))
                for dest, fraction in (
                    (lower_quartile_intensity, 1.0 / 4.0),
                    (median_intensity, 1.0 / 2.0),
                    (upper_quartile_intensity, 3.0 / 4.0),
                ):
                    qmask, _dest, qmask_no_upper, _dest_no_upper = self.foo(indices, areas, fraction, limg, order)
                    dest[lindexes[qmask] - 1] = _dest
                    dest[lindexes[qmask_no_upper] - 1] = _dest_no_upper

                #
                # Once again, for the MAD
                #
                fraction = 1.0/ image_dimensions
                madimg = numpy.abs(limg - median_intensity[llabels - 1])
                order = numpy.lexsort((madimg, llabels))

                qmask, _mad_intensity, qmask_no_upper, _mad_intensity_no_upper = self.foo(indices, areas, fraction, madimg, order)
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
                ) = self.get_intensity_measurements(eimg, elabels, lindexes)

                integrated_intensity_edge[lindexes - 1] = _integrated_intensity_edge
                mean_intensity_edge[lindexes - 1] = _mean_intensity_edge
                std_intensity_edge[lindexes - 1] = _std_intensity_edge
                min_intensity_edge[lindexes - 1] = _min_intensity_edge
                max_intensity_edge[lindexes - 1] = _max_intensity_edge

        return (
            integrated_intensity,
            integrated_intensity_edge,
            mean_intensity,
            mean_intensity_edge,
            std_intensity,
            std_intensity_edge,
            min_intensity,
            min_intensity_edge,
            max_intensity,
            max_intensity_edge,
            mass_displacement,
            lower_quartile_intensity,
            median_intensity,
            mad_intensity,
            upper_quartile_intensity,
            cmi_x,
            cmi_y,
            cmi_z,
            max_x,
            max_y,
            max_z,
        )

    ###############################################################################
    # End Helpers
    ###############################################################################
    
    def run(self, workspace):
        if self.show_window:
            workspace.display_data.col_labels = (
                "Image",
                "Object",
                "Feature",
                "Mean",
                "Median",
                "STD",
            )
            workspace.display_data.statistics = statistics = []
        if len(self.images_list.value) == 0 or len(self.objects_list.value) == 0:
            raise ValueError(
                "This module needs at least 1 image and object set selected"
            )
        for image_name in self.images_list.value:
            image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
            for object_name in self.objects_list.value:
                if object_name not in workspace.object_set.object_names:
                    raise ValueError(
                        "The %s objects are missing from the pipeline." % object_name
                    )
                # Need to refresh image after each iteration...
                img = image.pixel_data
                image_has_mask = image.has_mask
                image_mask = image.mask



                objects = workspace.object_set.get_objects(object_name)
                nobjects = objects.count
                (
                    integrated_intensity,
                    integrated_intensity_edge,
                    mean_intensity,
                    mean_intensity_edge,
                    std_intensity,
                    std_intensity_edge,
                    min_intensity,
                    min_intensity_edge,
                    max_intensity,
                    max_intensity_edge,
                    mass_displacement,
                    lower_quartile_intensity,
                    median_intensity,
                    mad_intensity,
                    upper_quartile_intensity,
                    cmi_x,
                    cmi_y,
                    cmi_z,
                    max_x,
                    max_y,
                    max_z
                ) = self.get_measureobjectintensity_measurements(
                    img,
                    image_mask, 
                    objects.get_labels(), 
                    nobjects,
                    image.dimensions, 
                    image.has_mask
                )

                m = workspace.measurements

                for category, feature_name, measurement in (
                    (INTENSITY, INTEGRATED_INTENSITY, integrated_intensity),
                    (INTENSITY, MEAN_INTENSITY, mean_intensity),
                    (INTENSITY, STD_INTENSITY, std_intensity),
                    (INTENSITY, MIN_INTENSITY, min_intensity),
                    (INTENSITY, MAX_INTENSITY, max_intensity),
                    (INTENSITY, INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge),
                    (INTENSITY, MEAN_INTENSITY_EDGE, mean_intensity_edge),
                    (INTENSITY, STD_INTENSITY_EDGE, std_intensity_edge),
                    (INTENSITY, MIN_INTENSITY_EDGE, min_intensity_edge),
                    (INTENSITY, MAX_INTENSITY_EDGE, max_intensity_edge),
                    (INTENSITY, MASS_DISPLACEMENT, mass_displacement),
                    (INTENSITY, LOWER_QUARTILE_INTENSITY, lower_quartile_intensity),
                    (INTENSITY, MEDIAN_INTENSITY, median_intensity),
                    (INTENSITY, MAD_INTENSITY, mad_intensity),
                    (INTENSITY, UPPER_QUARTILE_INTENSITY, upper_quartile_intensity),
                    (C_LOCATION, LOC_CMI_X, cmi_x),
                    (C_LOCATION, LOC_CMI_Y, cmi_y),
                    (C_LOCATION, LOC_CMI_Z, cmi_z),
                    (C_LOCATION, LOC_MAX_X, max_x),
                    (C_LOCATION, LOC_MAX_Y, max_y),
                    (C_LOCATION, LOC_MAX_Z, max_z),
                ):
                    measurement_name = "{}_{}_{}".format(
                        category, feature_name, image_name
                    )
                    m.add_measurement(object_name, measurement_name, measurement)
                    if self.show_window and len(measurement) > 0:
                        statistics.append(
                            (
                                image_name,
                                object_name,
                                feature_name,
                                numpy.round(numpy.mean(measurement), 3),
                                numpy.round(numpy.median(measurement), 3),
                                numpy.round(numpy.std(measurement), 3),
                            )
                        )

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
            title="default",
        )

    def volumetric(self):
        return True
