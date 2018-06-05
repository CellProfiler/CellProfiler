# coding=utf-8

"""
MeasureImageAreaOccupied
========================

**MeasureImageAreaOccupied** measures the total area in an image that
is occupied by objects.

This module reports the sum of the areas and perimeters of the objects
defined by one of the **Identify** modules, or the area of the
foreground in a binary image. If the input image has a mask (for
example, created by the **MaskImage** module), the measurements made by
this module will take the mask into account by ignoring the pixels
outside the mask.

You can use this module to measure the number of pixels above a given
threshold if you precede it with thresholding performed by
**Threshold**, and then select the binary image output by
**Threshold** to be measured by this module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **IdentifyPrimaryObjects**, **IdentifySecondaryObjects**,
**IdentifyTertiaryObjects**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *AreaOccupied/VolumeOccupied:* The total area (2D) or volume (3D)
   occupied by the input objects or binary image.
-  *Perimeter/SurfaceArea* The total length of the perimeter (2D) or
   surface area (3D) of the input objects/binary image.
-  *TotalArea/TotalVolume:* The total pixel area (2D) or volume (3D)
   of the image that was subjected to measurement, excluding masked
   regions.
"""

import numpy
import skimage.measure

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting

C_AREA_OCCUPIED = "AreaOccupied"

# Measurement feature name format for the AreaOccupied/VolumeOccupied measurement
F_AREA_OCCUPIED = "AreaOccupied"
F_VOLUME_OCCUPIED = "VolumeOccupied"

# Measure feature name format for the Perimeter/SurfaceArea measurement
F_PERIMETER = "Perimeter"
F_SURFACE_AREA = "SurfaceArea"

# Measure feature name format for the TotalArea/TotalVolume measurement
F_TOTAL_AREA = "TotalArea"
F_TOTAL_VOLUME = "TotalVolume"

O_BINARY_IMAGE = "Binary Image"
O_OBJECTS = "Objects"

# The number of settings per image or object group
IMAGE_SETTING_COUNT = 1

OBJECT_SETTING_COUNT = 3


class MeasureImageAreaOccupied(cellprofiler.module.Module):
    module_name = "MeasureImageAreaOccupied"
    category = "Measurement"
    variable_revision_number = 4

    def create_settings(self):
        self.operands = []

        self.count = cellprofiler.setting.HiddenCount(self.operands)

        self.add_operand(can_remove=False)

        self.add_operand_button = cellprofiler.setting.DoSomething(
            "",
            "Add another area",
            self.add_operand
        )

        self.remover = cellprofiler.setting.DoSomething(
            "",
            "Remove this area",
            self.remove
        )

    def add_operand(self, can_remove=True):
        class Operand(object):
            def __init__(self):
                self.__spacer = cellprofiler.setting.Divider(line=True)

                self.__operand_choice = cellprofiler.setting.Choice(
                    "Measure the area occupied in a binary image, or in objects?",
                    [
                        O_BINARY_IMAGE,
                        O_OBJECTS
                    ],
                    doc="""\
The area can be measured in two ways:

-  *{O_BINARY_IMAGE}:* The area occupied by the foreground in a binary (black and white) image.
-  *{O_OBJECTS}:* The area occupied by previously-identified objects.
                    """.format(**{
                        "O_BINARY_IMAGE": O_BINARY_IMAGE,
                        "O_OBJECTS": O_OBJECTS
                    })
                )

                self.__operand_objects = cellprofiler.setting.ObjectNameSubscriber(
                    "Select objects to measure",
                    cellprofiler.setting.NONE,
                    doc="""\
*(Used only if ‘{O_OBJECTS}’ are to be measured)*

Select the previously identified objects you would like to measure.
                    """.format(**{
                        "O_OBJECTS": O_OBJECTS
                    })
                )

                self.__binary_name = cellprofiler.setting.ImageNameSubscriber(
                    "Select a binary image to measure",
                    cellprofiler.setting.NONE,
                    doc="""\
*(Used only if ‘{O_BINARY_IMAGE}’ is to be measured)*

This is a binary image created earlier in the pipeline, where you would
like to measure the area occupied by the foreground in the image.
                    """.format(**{
                        "O_BINARY_IMAGE": O_BINARY_IMAGE
                    })
                )

            @property
            def spacer(self):
                return self.__spacer

            @property
            def operand_choice(self):
                return self.__operand_choice

            @property
            def operand_objects(self):
                return self.__operand_objects

            @property
            def binary_name(self):
                return self.__binary_name

            @property
            def remover(self):
                return self.__remover

            @property
            def object(self):
                if self.operand_choice == O_BINARY_IMAGE:
                    return self.binary_name.value
                else:
                    return self.operand_objects.value

        self.operands += [Operand()]

    def remove(self):
        del self.operands[-1]

        return self.operands

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        settings = {}
        for group in self.operands:
            if (group.operand_choice.value, group.operand_objects.value) in settings:
                if group.operand_choice.value == O_OBJECTS:
                    raise cellprofiler.setting.ValidationError(
                        u"{} has already been selected".format(group.operand_objects.value),
                        group.operand_objects
                    )

            settings[(group.operand_choice.value, group.operand_objects.value)] = True

        settings = {}
        for group in self.operands:
            if (group.operand_choice.value, group.binary_name.value) in settings:
                if group.operand_choice.value == O_BINARY_IMAGE:
                    raise cellprofiler.setting.ValidationError(
                            "%s has already been selected" % group.binary_name.value,
                            group.binary_name)
            settings[(group.operand_choice.value, group.binary_name.value)] = True

    def settings(self):
        result = [self.count]

        for op in self.operands:
            result += [
                op.operand_choice,
                op.operand_objects,
                op.binary_name
            ]

        return result

    def prepare_settings(self, setting_values):
        count = int(setting_values[0])

        sequence = self.operands

        del sequence[count:]

        while len(sequence) < count:
            self.add_operand()

            sequence = self.operands

    def visible_settings(self):
        result = []

        for op in self.operands:
            result += [op.spacer]

            result += [op.operand_choice]

            result += [op.operand_objects] if op.operand_choice == O_OBJECTS else [op.binary_name]

        result.append(self.add_operand_button)

        result.append(self.remover)

        return result

    def run(self, workspace):
        m = workspace.measurements

        statistics = []

        for op in self.operands:
            if op.operand_choice == O_OBJECTS:
                statistics += self.measure_objects(op, workspace)

            if op.operand_choice == O_BINARY_IMAGE:
                statistics += self.measure_images(op, workspace)

        if self.show_window:
            workspace.display_data.statistics = statistics

            workspace.display_data.col_labels = [
                "Objects or Image",
                "Area Occupied",
                "Perimeter",
                "Total Area"
            ]

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, workspace.display_data.statistics, col_labels=workspace.display_data.col_labels)

    def _add_image_measurement(self, name, feature_name, features, measurements):
        measurements.add_image_measurement(
            "{:s}_{:s}_{:s}".format(C_AREA_OCCUPIED, feature_name, name),
            numpy.array([features], dtype=float)
        )

    def measure_objects(self, operand, workspace):
        objects = workspace.get_objects(operand.operand_objects.value)

        label_image = objects.segmented

        if objects.has_parent_image:
            mask = objects.parent_image.mask

            label_image[~mask] = 0

            total_area = numpy.sum(mask)
        else:
            total_area = numpy.product(label_image.shape)

        region_properties = skimage.measure.regionprops(label_image)

        area_occupied = numpy.sum([region["area"] for region in region_properties])

        if objects.volumetric:
            spacing = None

            if objects.has_parent_image:
                spacing = objects.parent_image.spacing

            labels = numpy.unique(label_image)

            if labels[0] == 0:
                labels = labels[1:]

            perimeter = surface_area(label_image, spacing=spacing, index=labels)
        else:
            perimeter = numpy.sum([numpy.round(region["perimeter"]) for region in region_properties])

        measurements = workspace.measurements
        pipeline = workspace.pipeline

        self._add_image_measurement(
            operand.operand_objects.value,
            F_VOLUME_OCCUPIED if pipeline.volumetric() else F_AREA_OCCUPIED,
            area_occupied,
            measurements
        )

        self._add_image_measurement(
            operand.operand_objects.value,
            F_SURFACE_AREA if pipeline.volumetric() else F_PERIMETER,
            perimeter,
            measurements
        )

        self._add_image_measurement(
            operand.operand_objects.value,
            F_TOTAL_VOLUME if pipeline.volumetric() else F_TOTAL_AREA,
            total_area,
            measurements
        )

        return [[
            operand.operand_objects.value,
            str(area_occupied),
            str(perimeter),
            str(total_area)
        ]]

    def measure_images(self, operand, workspace):
        image = workspace.image_set.get_image(operand.binary_name.value, must_be_binary=True)

        area_occupied = numpy.sum(image.pixel_data > 0)

        if image.volumetric:
            perimeter = surface_area(image.pixel_data > 0, spacing=image.spacing)
        else:
            perimeter = skimage.measure.perimeter(image.pixel_data > 0)

        total_area = numpy.prod(numpy.shape(image.pixel_data))

        measurements = workspace.measurements
        pipeline = workspace.pipeline

        self._add_image_measurement(
            operand.binary_name.value,
            F_VOLUME_OCCUPIED if pipeline.volumetric() else F_AREA_OCCUPIED,
            area_occupied,
            measurements
        )

        self._add_image_measurement(
            operand.binary_name.value,
            F_SURFACE_AREA if pipeline.volumetric() else F_PERIMETER,
            perimeter,
            measurements
        )

        self._add_image_measurement(
            operand.binary_name.value,
            F_TOTAL_VOLUME if pipeline.volumetric() else F_TOTAL_AREA,
            total_area,
            measurements
        )

        return [[
            operand.binary_name.value,
            str(area_occupied),
            str(perimeter),
            str(total_area)
        ]]

    def _get_feature_names(self, pipeline):
        if pipeline.volumetric():
            return [F_VOLUME_OCCUPIED, F_SURFACE_AREA, F_TOTAL_VOLUME]

        return [F_AREA_OCCUPIED, F_PERIMETER, F_TOTAL_AREA]

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = []

        for op in self.operands:
            for feature in self._get_feature_names(pipeline):
                columns.append((
                    cellprofiler.measurement.IMAGE,
                    "{:s}_{:s}_{:s}".format(
                        C_AREA_OCCUPIED,
                        feature,
                        op.operand_objects.value if op.operand_choice == O_OBJECTS else op.binary_name.value
                    ),
                    cellprofiler.measurement.COLTYPE_FLOAT
                ))

        return columns

    def get_categories(self, pipeline, object_name):
        if object_name == cellprofiler.measurement.IMAGE:
            return [C_AREA_OCCUPIED]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cellprofiler.measurement.IMAGE and category == C_AREA_OCCUPIED:
            return self._get_feature_names(pipeline)

        return []

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if object_name == "Image" and category == "AreaOccupied" and measurement in self._get_feature_names(pipeline):
            return [op.operand_objects.value for op in self.operands if op.operand_choice == O_OBJECTS]

        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if object_name == "Image" and category == "AreaOccupied" and measurement in self._get_feature_names(pipeline):
            return [op.binary_name.value for op in self.operands if op.operand_choice == O_BINARY_IMAGE]

        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab:
            raise NotImplementedError("The MeasureImageAreaOccupied module has changed substantially. \n"
                                      "You should use this module by either:\n"
                                      "(1) Thresholding your image using an Identify module\n"
                                      "and then measure the resulting objects' area; or\n"
                                      "(2) Create a binary image with Threshold and then measure the\n"
                                      "resulting foreground image area.")
        if variable_revision_number == 1:
            # We added the ability to process multiple objects in v2, but
            # the settings for v1 miraculously map to v2
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Permits choice of binary image or objects to measure from
            count = len(setting_values) / 3

            new_setting_values = [str(count)]

            for i in range(0, count):
                new_setting_values += [
                    'Objects',
                    setting_values[(i * 3)],
                    setting_values[(i * 3) + 1],
                    setting_values[(i * 3) + 2],
                    cellprofiler.setting.NONE
                ]

            setting_values = new_setting_values

            variable_revision_number = 3

        if variable_revision_number == 3:
            n_objects = int(setting_values[0])

            operand_choices = setting_values[1::5][:n_objects]
            operand_objects = setting_values[2::5][:n_objects]
            binary_name = setting_values[5::5][:n_objects]

            object_settings = sum(
                [list(settings) for settings in zip(operand_choices, operand_objects, binary_name)],
                []
            )

            setting_values = [setting_values[0]] + object_settings

            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


def surface_area(label_image, spacing=None, index=None):
    if spacing is None:
        spacing = (1.0,) * label_image.ndim

    if index is None:
        verts, faces = skimage.measure.marching_cubes_classic(label_image, spacing=spacing, level=0)

        return skimage.measure.mesh_surface_area(verts, faces)

    return numpy.sum([numpy.round(_label_surface_area(label_image, label, spacing)) for label in index])


def _label_surface_area(label_image, label, spacing):
    verts, faces = skimage.measure.marching_cubes_classic(label_image == label, spacing=spacing, level=0)

    return skimage.measure.mesh_surface_area(verts, faces)
