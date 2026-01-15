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
from typing import List
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)
from cellprofiler_library.modules._measureimageareaoccupied import measure_image_area_perimeter, measure_objects_area_perimeter
from cellprofiler_library.opts.measureimageareaoccupied import MeasurementType, Target, C_AREA_OCCUPIED

# The number of settings per image or object group
IMAGE_SETTING_COUNT = 1

OBJECT_SETTING_COUNT = 3


class MeasureImageAreaOccupied(Module):
    module_name = "MeasureImageAreaOccupied"
    category = "Measurement"
    variable_revision_number = 5

    def create_settings(self):
        self.operand_choice = Choice(
            "Measure the area occupied by",
            [Target.BINARY_IMAGE.value, Target.OBJECTS.value, Target.BOTH.value],
            doc="""\
Area occupied can be measured in two ways:

-  *{O_BINARY_IMAGE}:* The area occupied by the foreground in a binary (black and white) image.
-  *{O_OBJECTS}:* The area occupied by previously-identified objects.
                    """.format(
                **{"O_BINARY_IMAGE": Target.BINARY_IMAGE.value, "O_OBJECTS": Target.OBJECTS.value}
            ),
        )

        self.divider = Divider()

        self.images_list = ImageListSubscriber(
            "Select binary images to measure",
            [],
            doc="""*(Used only if ‘{O_BINARY_IMAGE}’ is to be measured)*

These should be binary images created earlier in the pipeline, where you would
like to measure the area occupied by the foreground in the image.
                    """.format(
                **{"O_BINARY_IMAGE": Target.BINARY_IMAGE.value}
            ),
        )

        self.objects_list = LabelListSubscriber(
            "Select object sets to measure",
            [],
            doc="""*(Used only if ‘{O_OBJECTS}’ are to be measured)*

Select the previously identified objects you would like to measure.""".format(
                **{"O_OBJECTS": Target.OBJECTS.value}
            ),
        )

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        if self.operand_choice in (Target.BINARY_IMAGE.value, Target.BOTH.value):
            images = set()
            if len(self.images_list.value) == 0:
                raise ValidationError("No images selected", self.images_list)
            for image_name in self.images_list.value:
                if image_name in images:
                    raise ValidationError(
                        "%s has already been selected" % image_name, image_name
                    )
                images.add(image_name)
        if self.operand_choice in (Target.OBJECTS, Target.BOTH):
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
        result = [self.operand_choice, self.images_list, self.objects_list]
        return result

    def visible_settings(self):
        result = [self.operand_choice, self.divider]
        if self.operand_choice.value in (Target.BOTH, Target.BINARY_IMAGE):
            result.append(self.images_list)
        if self.operand_choice.value in (Target.BOTH, Target.OBJECTS):
            result.append(self.objects_list)
        return result

    def run(self, workspace):
        m = workspace.measurements

        statistics = []

        if self.operand_choice.value in (Target.BOTH, Target.BINARY_IMAGE):
            if len(self.images_list.value) == 0:
                raise ValueError("No images were selected for analysis.")
            for binary_image in self.images_list.value:
                statistics += self.measure_images(binary_image, workspace)
        if self.operand_choice.value in (Target.BOTH, Target.OBJECTS):
            if len(self.objects_list.value) == 0:
                raise ValueError("No object sets were selected for analysis.")
            for object_set in self.objects_list.value:
                statistics += self.measure_objects(object_set, workspace)

        if self.show_window:
            workspace.display_data.statistics = statistics

            workspace.display_data.col_labels = [
                "Objects or Image",
                "Area Occupied",
                "Perimeter",
                "Total Area",
            ]

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
        )

    def _add_image_measurement(self, name, feature_name, features, measurements):
        measurements.add_image_measurement(
            "{:s}_{:s}_{:s}".format(C_AREA_OCCUPIED, feature_name, name),
            numpy.array([features], dtype=float),
        )

    def measure_objects(self, object_set: str, workspace):
        objects = workspace.get_objects(object_set)

        label_image = objects.segmented

        mask = None
        if objects.has_parent_image:
            # Image always has a mask. returns all ones if not specified.
            mask = objects.parent_image.mask

        spacing = None
        if objects.volumetric:
            if objects.has_parent_image:
                spacing = objects.parent_image.spacing
        pipeline_volumetric = workspace.pipeline.volumetric()
        lib_measurements = measure_objects_area_perimeter(
            label_image=label_image,
            object_name=object_set,
            mask=mask, 
            volumetric=objects.volumetric, 
            spacing=spacing
        )

        for feature_name, value in lib_measurements.image.items():
            workspace.measurements.add_image_measurement(feature_name, value)

        # Retrieve values for statistics display
        # We reconstruct keys to fetch them from the library measurements
        feature_area_occupied = MeasurementType.VOLUME_OCCUPIED.value if pipeline_volumetric else MeasurementType.AREA_OCCUPIED.value
        feature_perimeter = MeasurementType.SURFACE_AREA.value if pipeline_volumetric else MeasurementType.PERIMETER.value
        feature_total_area = MeasurementType.TOTAL_VOLUME.value if pipeline_volumetric else MeasurementType.TOTAL_AREA.value

        area_occupied = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_area_occupied}_{object_set}"]
        perimeter = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_perimeter}_{object_set}"]
        total_area = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_total_area}_{object_set}"]

        return [[object_set, str(area_occupied), str(perimeter), str(total_area),]]


    def measure_images(self, image_set: str, workspace):
        image = workspace.image_set.get_image(image_set, must_be_binary=True)
        pipeline_volumetric = workspace.pipeline.volumetric()
        
        lib_measurements = measure_image_area_perimeter(
            im_pixel_data=image.pixel_data, 
            image_name=image_set,
            im_volumetric=image.volumetric,
            im_spacing=image.spacing
        )

        for feature_name, value in lib_measurements.image.items():
            workspace.measurements.add_image_measurement(feature_name, value)

        feature_area_occupied = MeasurementType.VOLUME_OCCUPIED.value if pipeline_volumetric else MeasurementType.AREA_OCCUPIED.value
        feature_perimeter = MeasurementType.SURFACE_AREA.value if pipeline_volumetric else MeasurementType.PERIMETER.value
        feature_total_area = MeasurementType.TOTAL_VOLUME.value if pipeline_volumetric else MeasurementType.TOTAL_AREA.value

        area_occupied = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_area_occupied}_{image_set}"]
        perimeter = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_perimeter}_{image_set}"]
        total_area = lib_measurements.image[f"{C_AREA_OCCUPIED}_{feature_total_area}_{image_set}"]

        return [[image_set, str(area_occupied), str(perimeter), str(total_area),]]

    def _get_feature_names(self, pipeline):
        if pipeline.volumetric():
            return [MeasurementType.VOLUME_OCCUPIED.value, MeasurementType.SURFACE_AREA.value, MeasurementType.TOTAL_VOLUME.value]

        return [MeasurementType.AREA_OCCUPIED.value, MeasurementType.PERIMETER.value, MeasurementType.TOTAL_AREA.value]

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements made by this module"""
        columns = []

        if self.operand_choice.value in (Target.BOTH, Target.OBJECTS):
            for object_set in self.objects_list.value:
                for feature in self._get_feature_names(pipeline):
                    columns.append(
                        (
                            "Image",
                            "{:s}_{:s}_{:s}".format(
                                C_AREA_OCCUPIED, feature, object_set,
                            ),
                            COLTYPE_FLOAT,
                        )
                    )
        if self.operand_choice.value in (Target.BOTH, Target.BINARY_IMAGE):
            for image_set in self.images_list.value:
                for feature in self._get_feature_names(pipeline):
                    columns.append(
                        (
                            "Image",
                            "{:s}_{:s}_{:s}".format(
                                C_AREA_OCCUPIED, feature, image_set,
                            ),
                            COLTYPE_FLOAT,
                        )
                    )

        return columns

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return [C_AREA_OCCUPIED]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == C_AREA_OCCUPIED:
            return self._get_feature_names(pipeline)
        return []

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if (
            object_name == "Image"
            and category == "AreaOccupied"
            and measurement in self._get_feature_names(pipeline)
        ):
            return [
                object_name
                for object_name in self.objects_list.value
                if self.operand_choice.value in (Target.OBJECTS, Target.BOTH)
            ]
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if (
            object_name == "Image"
            and category == "AreaOccupied"
            and measurement in self._get_feature_names(pipeline)
        ):
            return [
                image_name
                for image_name in self.images_list.value
                if self.operand_choice.value in (Target.BINARY_IMAGE, Target.BOTH)
            ]
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
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
                    "Objects",
                    setting_values[(i * 3)],
                    setting_values[(i * 3) + 1],
                    setting_values[(i * 3) + 2],
                    "None",
                ]

            setting_values = new_setting_values

            variable_revision_number = 3

        if variable_revision_number == 3:
            n_objects = int(setting_values[0])

            operand_choices = setting_values[1::5][:n_objects]
            operand_objects = setting_values[2::5][:n_objects]
            binary_name = setting_values[5::5][:n_objects]

            object_settings = sum(
                [
                    list(settings)
                    for settings in zip(operand_choices, operand_objects, binary_name)
                ],
                [],
            )

            setting_values = [setting_values[0]] + object_settings

            variable_revision_number = 4
        if variable_revision_number == 4:
            num_sets = setting_values[0]
            setting_values = setting_values[1:]
            images_set = set()
            objects_set = set()
            conditions, names1, names2 = [(setting_values[i::3]) for i in range(3)]
            for condition, name1, name2 in zip(conditions, names1, names2):
                if condition == Target.BINARY_IMAGE.value:
                    images_set.add(name2)
                elif condition == Target.OBJECTS.value:
                    objects_set.add(name1)
            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            if len(images_set) > 0 and len(objects_set) > 0:
                mode = Target.BOTH.value
            elif len(images_set) == 0:
                mode = Target.OBJECTS.value
            else:
                mode = Target.BINARY_IMAGE.value
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))
            setting_values = [mode, images_string, objects_string]
            variable_revision_number = 5
        return setting_values, variable_revision_number

    def volumetric(self):
        return True