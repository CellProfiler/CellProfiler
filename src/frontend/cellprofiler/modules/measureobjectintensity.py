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
from cellprofiler_library.modules._measureobjectintensity import measure_object_intensity

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
                
                lib_measurements = measure_object_intensity(
                    img=img,
                    image_name=image_name,
                    object_name=object_name,
                    image_mask=image_mask,
                    object_labels=objects.get_labels(),
                    nobjects=nobjects,
                    image_dimensions=image.dimensions,
                    image_has_mask=image.has_mask
                )

                m = workspace.measurements

                for category, feature_name in (
                    (INTENSITY, INTEGRATED_INTENSITY),
                    (INTENSITY, MEAN_INTENSITY),
                    (INTENSITY, STD_INTENSITY),
                    (INTENSITY, MIN_INTENSITY),
                    (INTENSITY, MAX_INTENSITY),
                    (INTENSITY, INTEGRATED_INTENSITY_EDGE),
                    (INTENSITY, MEAN_INTENSITY_EDGE),
                    (INTENSITY, STD_INTENSITY_EDGE),
                    (INTENSITY, MIN_INTENSITY_EDGE),
                    (INTENSITY, MAX_INTENSITY_EDGE),
                    (INTENSITY, MASS_DISPLACEMENT),
                    (INTENSITY, LOWER_QUARTILE_INTENSITY),
                    (INTENSITY, MEDIAN_INTENSITY),
                    (INTENSITY, MAD_INTENSITY),
                    (INTENSITY, UPPER_QUARTILE_INTENSITY),
                    (C_LOCATION, LOC_CMI_X),
                    (C_LOCATION, LOC_CMI_Y),
                    (C_LOCATION, LOC_CMI_Z),
                    (C_LOCATION, LOC_MAX_X),
                    (C_LOCATION, LOC_MAX_Y),
                    (C_LOCATION, LOC_MAX_Z),
                ):
                    measurement_name = "{}_{}_{}".format(
                        category, feature_name, image_name
                    )
                    
                    # Retrieve measurement from LibraryMeasurements
                    # Note: measure_object_intensity adds them with the same key format
                    if lib_measurements.has_feature(object_name, measurement_name):
                        measurement = lib_measurements.get_measurement(object_name, measurement_name)
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
