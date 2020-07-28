import logging

import numpy
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary, ValidationError, Divider
from cellprofiler_core.setting.subscriber import (
    LabelListSubscriber,
    ImageListSubscriber,
)

from cellprofiler.modules import _help

__doc__ = """
MeasureImageIntensity
=====================

**MeasureImageIntensity** measures several intensity features across an
entire image (excluding masked pixels).

For example, this module will sum all pixel values to measure the total image
intensity. You can choose to measure all pixels in the image or restrict
the measurement to pixels within objects that were identified in a prior
module. If the image has a mask, only unmasked pixels will be measured.

{HELP_ON_MEASURING_INTENSITIES}

As of **CellProfiler 4.0** the settings for this module have been changed to simplify
configuration. All selected images and objects are now analysed together rather
than needing to be matched in pairs.
Pipelines from older versions will be converted to match this format, which may
create extra computational work. Specific pairing can still be achieved by running
multiple copies of this module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES 
============ ============ ===============

See also
^^^^^^^^

See also **MeasureObjectIntensity**, **MaskImage**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *TotalIntensity:* Sum of all pixel intensity values.
-  *MeanIntensity, MedianIntensity:* Mean and median of pixel intensity
   values.
-  *StdIntensity, MADIntensity:* Standard deviation and median absolute
   deviation (MAD) of pixel intensity values. The MAD is defined as the
   median(\|x\ :sub:`i` - median(x)\|).
-  *MinIntensity, MaxIntensity:* Minimum and maximum of pixel intensity
   values.
-  *LowerQuartileIntensity:* The intensity value of the pixel for which
   25% of the pixels in the object have lower values.
-  *UpperQuartileIntensity:* The intensity value of the pixel for which
   75% of the pixels in the object have lower values.
-  *TotalArea:* Number of pixels measured, e.g., the area of the image
   excluding masked regions.

""".format(
    **{"HELP_ON_MEASURING_INTENSITIES": _help.HELP_ON_MEASURING_INTENSITIES}
)

"""Measurement feature name format for the TotalIntensity measurement"""
F_TOTAL_INTENSITY = "Intensity_TotalIntensity_%s"

"""Measurement feature name format for the MeanIntensity measurement"""
F_MEAN_INTENSITY = "Intensity_MeanIntensity_%s"

"""Measurement feature name format for the MeanIntensity measurement"""
F_MEDIAN_INTENSITY = "Intensity_MedianIntensity_%s"

"""Measurement feature name format for the StdIntensity measurement"""
F_STD_INTENSITY = "Intensity_StdIntensity_%s"

"""Measurement feature name format for the MedAbsDevIntensity measurement"""
F_MAD_INTENSITY = "Intensity_MADIntensity_%s"

"""Measurement feature name format for the MaxIntensity measurement"""
F_MAX_INTENSITY = "Intensity_MaxIntensity_%s"

"""Measurement feature name format for the MinIntensity measurement"""
F_MIN_INTENSITY = "Intensity_MinIntensity_%s"

"""Measurement feature name format for the TotalArea measurement"""
F_TOTAL_AREA = "Intensity_TotalArea_%s"

"""Measurement feature name format for the PercentMaximal measurement"""
F_PERCENT_MAXIMAL = "Intensity_PercentMaximal_%s"

"""Measurement feature name format for the Quartile measurements"""
F_UPPER_QUARTILE = "Intensity_UpperQuartileIntensity_%s"
F_LOWER_QUARTILE = "Intensity_LowerQuartileIntensity_%s"

ALL_MEASUREMENTS = [
    "TotalIntensity",
    "MeanIntensity",
    "StdIntensity",
    "MADIntensity",
    "MedianIntensity",
    "MinIntensity",
    "MaxIntensity",
    "TotalArea",
    "PercentMaximal",
    "LowerQuartileIntensity",
    "UpperQuartileIntensity",
]


class MeasureImageIntensity(Module):
    module_name = "MeasureImageIntensity"
    category = "Measurement"
    variable_revision_number = 3

    def create_settings(self):
        """Create the settings & name the module"""
        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select the grayscale images whose intensity you want to measure.""",
        )

        self.divider = Divider(line=False)
        self.wants_objects = Binary(
            "Measure the intensity only from areas enclosed by objects?",
            False,
            doc="""\
        Select *Yes* to measure only those pixels within an object type you
        choose, identified by a prior module. Note that this module will
        aggregate intensities across all objects in the image: to measure each
        object individually, see **MeasureObjectIntensity** instead.
        """,
        )

        self.objects_list = LabelListSubscriber(
            "Select input object sets",
            [],
            doc="""Select the object sets whose intensity you want to measure.""",
        )

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        images = set()
        if len(self.images_list.value) == 0:
            raise ValidationError("No images selected", self.images_list)
        for image_name in self.images_list.value:
            if image_name in images:
                raise ValidationError(
                    "%s has already been selected" % image_name, image_name
                )
            images.add(image_name)
        if self.wants_objects:
            objects = set()
            if len(self.objects_list.value) == 0:
                raise ValidationError("No objects selected", self.objects_list)
            for object_name in self.objects_list.value:
                if object_name in objects:
                    raise ValidationError(
                        "%s has already been selected" % object_name, object_name
                    )
                objects.add(object_name)

    def settings(self):
        result = [self.images_list, self.wants_objects, self.objects_list]
        return result

    def visible_settings(self):
        result = [self.images_list, self.wants_objects]
        if self.wants_objects:
            result += [self.objects_list]
        return result

    def run(self, workspace):
        """Perform the measurements on the image sets"""
        col_labels = ["Image", "Masking object", "Feature", "Value"]
        statistics = []
        for im in self.images_list.value:
            image = workspace.image_set.get_image(im, must_be_grayscale=True)
            input_pixels = image.pixel_data

            measurement_name = im
            if self.wants_objects.value:
                for object_set in self.objects_list.value:
                    measurement_name += "_" + object_set
                    objects = workspace.get_objects(object_set)
                    if objects.shape != input_pixels.shape:
                        raise ValueError(
                            "This module requires that the image and object sets have matching dimensions.\n"
                            "The %s image and %s objects do not (%s vs %s).\n"
                            "If they are paired correctly you may want to use the Resize, ResizeObjects or "
                            "Crop module(s) to make them the same size."
                            % (im, object_set, input_pixels.shape, objects.shape,)
                        )
                    if image.has_mask:
                        pixels = input_pixels[
                            numpy.logical_and(objects.segmented != 0, image.mask)
                        ]
                    else:
                        pixels = input_pixels[objects.segmented != 0]
                    statistics += self.measure(
                        pixels, im, object_set, measurement_name, workspace
                    )
            else:
                if image.has_mask:
                    pixels = input_pixels[image.mask]
                else:
                    pixels = input_pixels
                statistics += self.measure(
                    pixels, im, None, measurement_name, workspace
                )
        workspace.display_data.statistics = statistics
        workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
        )

    def measure(self, pixels, image_name, object_name, measurement_name, workspace):
        """Perform measurements on an array of pixels
        pixels - image pixel data, masked to objects if applicable
        image_name - name of the current input image
        object_name - name of the current object set pixels are masked to
        measurement_name - group title to be used in data tables
        workspace - has all the details for current image set
        """
        pixel_count = numpy.product(pixels.shape)
        if pixel_count == 0:
            pixel_sum = 0
            pixel_mean = 0
            pixel_std = 0
            pixel_mad = 0
            pixel_median = 0
            pixel_min = 0
            pixel_max = 0
            pixel_pct_max = 0
            pixel_lower_qrt = 0
            pixel_upper_qrt = 0
        else:
            pixels = pixels.flatten()
            pixels = pixels[
                numpy.nonzero(numpy.isfinite(pixels))[0]
            ]  # Ignore NaNs, Infs
            pixel_count = numpy.product(pixels.shape)

            pixel_sum = numpy.sum(pixels)
            pixel_mean = pixel_sum / float(pixel_count)
            pixel_std = numpy.std(pixels)
            pixel_median = numpy.median(pixels)
            pixel_mad = numpy.median(numpy.abs(pixels - pixel_median))
            pixel_min = numpy.min(pixels)
            pixel_max = numpy.max(pixels)
            pixel_pct_max = (
                100.0 * float(numpy.sum(pixels == pixel_max)) / float(pixel_count)
            )
            sorted_pixel_data = sorted(pixels)
            pixel_lower_qrt = sorted_pixel_data[int(len(sorted_pixel_data) * 0.25)]
            pixel_upper_qrt = sorted_pixel_data[int(len(sorted_pixel_data) * 0.75)]

        m = workspace.measurements
        m.add_image_measurement(F_TOTAL_INTENSITY % measurement_name, pixel_sum)
        m.add_image_measurement(F_MEAN_INTENSITY % measurement_name, pixel_mean)
        m.add_image_measurement(F_MEDIAN_INTENSITY % measurement_name, pixel_median)
        m.add_image_measurement(F_STD_INTENSITY % measurement_name, pixel_std)
        m.add_image_measurement(F_MAD_INTENSITY % measurement_name, pixel_mad)
        m.add_image_measurement(F_MAX_INTENSITY % measurement_name, pixel_max)
        m.add_image_measurement(F_MIN_INTENSITY % measurement_name, pixel_min)
        m.add_image_measurement(F_TOTAL_AREA % measurement_name, pixel_count)
        m.add_image_measurement(F_PERCENT_MAXIMAL % measurement_name, pixel_pct_max)
        m.add_image_measurement(F_LOWER_QUARTILE % measurement_name, pixel_lower_qrt)
        m.add_image_measurement(F_UPPER_QUARTILE % measurement_name, pixel_upper_qrt)
        return [
            [
                image_name,
                object_name if self.wants_objects.value else "",
                feature_name,
                str(value),
            ]
            for feature_name, value in (
                ("Total intensity", pixel_sum),
                ("Mean intensity", pixel_mean),
                ("Median intensity", pixel_median),
                ("Std intensity", pixel_std),
                ("MAD intensity", pixel_mad),
                ("Min intensity", pixel_min),
                ("Max intensity", pixel_max),
                ("Pct maximal", pixel_pct_max),
                ("Lower quartile", pixel_lower_qrt),
                ("Upper quartile", pixel_upper_qrt),
                ("Total area", pixel_count),
            )
        ]

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements made by this module"""
        columns = []
        for im in self.images_list.value:
            for feature, coltype in (
                (F_TOTAL_INTENSITY, COLTYPE_FLOAT),
                (F_MEAN_INTENSITY, COLTYPE_FLOAT),
                (F_MEDIAN_INTENSITY, COLTYPE_FLOAT),
                (F_STD_INTENSITY, COLTYPE_FLOAT),
                (F_MAD_INTENSITY, COLTYPE_FLOAT),
                (F_MIN_INTENSITY, COLTYPE_FLOAT),
                (F_MAX_INTENSITY, COLTYPE_FLOAT),
                (F_TOTAL_AREA, "integer"),
                (F_PERCENT_MAXIMAL, COLTYPE_FLOAT),
                (F_LOWER_QUARTILE, COLTYPE_FLOAT),
                (F_UPPER_QUARTILE, COLTYPE_FLOAT),
            ):
                if self.wants_objects:
                    for object_set in self.objects_list.value:
                        measurement_name = im + "_" + object_set
                        columns.append(("Image", feature % measurement_name, coltype,))
                else:
                    measurement_name = im
                    columns.append(("Image", feature % measurement_name, coltype,))
        return columns

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return ["Intensity"]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == "Intensity":
            return ALL_MEASUREMENTS
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if (
            object_name == "Image"
            and category == "Intensity"
            and measurement in ALL_MEASUREMENTS
        ):
            result = []
            for im in self.images_list.value:
                image_name = im
                if self.wants_objects:
                    for object_name in self.objects_list.value:
                        image_name += "_" + object_name
                        result += [image_name]
                else:
                    result += [image_name]
            return result
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Convert to new format, warn if settings will be lost.
            images_set, use_objects, objects_set = [
                set(setting_values[i::3]) for i in range(3)
            ]
            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            images_string = ", ".join(map(str, images_set))
            wants_objects = "Yes" if "Yes" in use_objects else "No"
            objects_string = ", ".join(map(str, objects_set))
            setting_values = [images_string, wants_objects, objects_string]
            if len(use_objects) > 1 or len(objects_set) > 1:
                logging.warning(
                    "The pipeline you loaded was converted from an older version of CellProfiler.\n"
                    "The MeasureImageIntensity module no longer uses pairs of images and objects.\n"
                    "Instead, all selected images and objects will be analysed together.\n"
                    "If you want to limit analysis of particular objects or perform both "
                    "whole image and object-restricted analysis you should use a second "
                    "copy of the module.",
                )
            variable_revision_number = 3
        return setting_values, variable_revision_number

    def volumetric(self):
        return True
