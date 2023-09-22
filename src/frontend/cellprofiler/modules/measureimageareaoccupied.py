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
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)

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
O_BOTH = "Both"

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
            [O_BINARY_IMAGE, O_OBJECTS, O_BOTH],
            doc="""\
Area occupied can be measured in two ways:

-  *{O_BINARY_IMAGE}:* The area occupied by the foreground in a binary (black and white) image.
-  *{O_OBJECTS}:* The area occupied by previously-identified objects.
                    """.format(
                **{"O_BINARY_IMAGE": O_BINARY_IMAGE, "O_OBJECTS": O_OBJECTS}
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
                **{"O_BINARY_IMAGE": O_BINARY_IMAGE}
            ),
        )

        self.objects_list = LabelListSubscriber(
            "Select object sets to measure",
            [],
            doc="""*(Used only if ‘{O_OBJECTS}’ are to be measured)*

Select the previously identified objects you would like to measure.""".format(
                **{"O_OBJECTS": O_OBJECTS}
            ),
        )

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        if self.operand_choice in (O_BINARY_IMAGE, O_BOTH):
            images = set()
            if len(self.images_list.value) == 0:
                raise ValidationError("No images selected", self.images_list)
            for image_name in self.images_list.value:
                if image_name in images:
                    raise ValidationError(
                        "%s has already been selected" % image_name, image_name
                    )
                images.add(image_name)
        if self.operand_choice in (O_OBJECTS, O_BOTH):
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
        if self.operand_choice in (O_BOTH, O_BINARY_IMAGE):
            result.append(self.images_list)
        if self.operand_choice in (O_BOTH, O_OBJECTS):
            result.append(self.objects_list)
        return result

    def run(self, workspace):
        m = workspace.measurements

        statistics = []

        if self.operand_choice in (O_BOTH, O_BINARY_IMAGE):
            if len(self.images_list.value) == 0:
                raise ValueError("No images were selected for analysis.")
            for binary_image in self.images_list.value:
                statistics += self.measure_images(binary_image, workspace)
        if self.operand_choice in (O_BOTH, O_OBJECTS):
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

    def measure_objects(self, object_set, workspace):
        objects = workspace.get_objects(object_set)

        label_image = objects.segmented

        if objects.has_parent_image:
            mask = objects.parent_image.mask

            label_image[~mask] = 0

            total_area = numpy.sum(mask)
        else:
            total_area = numpy.product(label_image.shape)

        region_properties = skimage.measure.regionprops(label_image)

        area_occupied = numpy.sum([region["area"] for region in region_properties])
        
        if area_occupied > 0:
            if objects.volumetric:
                spacing = None

                if objects.has_parent_image:
                    spacing = objects.parent_image.spacing

                labels = numpy.unique(label_image)

                if labels[0] == 0:
                    labels = labels[1:]

                perimeter = surface_area(label_image, spacing=spacing, index=labels)
            else:
                perimeter = numpy.sum(
                    [numpy.round(region["perimeter"]) for region in region_properties]
                )
        else:
            perimeter = 0

        measurements = workspace.measurements
        pipeline = workspace.pipeline

        self._add_image_measurement(
            object_set,
            F_VOLUME_OCCUPIED if pipeline.volumetric() else F_AREA_OCCUPIED,
            area_occupied,
            measurements,
        )

        self._add_image_measurement(
            object_set,
            F_SURFACE_AREA if pipeline.volumetric() else F_PERIMETER,
            perimeter,
            measurements,
        )

        self._add_image_measurement(
            object_set,
            F_TOTAL_VOLUME if pipeline.volumetric() else F_TOTAL_AREA,
            total_area,
            measurements,
        )

        return [[object_set, str(area_occupied), str(perimeter), str(total_area),]]

    def measure_images(self, image_set, workspace):
        image = workspace.image_set.get_image(image_set, must_be_binary=True)

        area_occupied = numpy.sum(image.pixel_data > 0)

        if area_occupied > 0:
            if image.volumetric:
                perimeter = surface_area(image.pixel_data > 0, spacing=image.spacing)
            else:
                perimeter = skimage.measure.perimeter(image.pixel_data > 0)
        else:
            perimeter = 0
            
        total_area = numpy.prod(numpy.shape(image.pixel_data))

        measurements = workspace.measurements
        pipeline = workspace.pipeline

        self._add_image_measurement(
            image_set,
            F_VOLUME_OCCUPIED if pipeline.volumetric() else F_AREA_OCCUPIED,
            area_occupied,
            measurements,
        )

        self._add_image_measurement(
            image_set,
            F_SURFACE_AREA if pipeline.volumetric() else F_PERIMETER,
            perimeter,
            measurements,
        )

        self._add_image_measurement(
            image_set,
            F_TOTAL_VOLUME if pipeline.volumetric() else F_TOTAL_AREA,
            total_area,
            measurements,
        )

        return [[image_set, str(area_occupied), str(perimeter), str(total_area),]]

    def _get_feature_names(self, pipeline):
        if pipeline.volumetric():
            return [F_VOLUME_OCCUPIED, F_SURFACE_AREA, F_TOTAL_VOLUME]

        return [F_AREA_OCCUPIED, F_PERIMETER, F_TOTAL_AREA]

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements made by this module"""
        columns = []

        if self.operand_choice in (O_BOTH, O_OBJECTS):
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
        if self.operand_choice in (O_BOTH, O_BINARY_IMAGE):
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
                if self.operand_choice in (O_OBJECTS, O_BOTH)
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
                if self.operand_choice in (O_BINARY_IMAGE, O_BOTH)
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
                if condition == O_BINARY_IMAGE:
                    images_set.add(name2)
                elif condition == O_OBJECTS:
                    objects_set.add(name1)
            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            if len(images_set) > 0 and len(objects_set) > 0:
                mode = O_BOTH
            elif len(images_set) == 0:
                mode = O_OBJECTS
            else:
                mode = O_BINARY_IMAGE
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))
            setting_values = [mode, images_string, objects_string]
            variable_revision_number = 5
        return setting_values, variable_revision_number

    def volumetric(self):
        return True


def surface_area(label_image, spacing=None, index=None):
    if spacing is None:
        spacing = (1.0,) * label_image.ndim

    if index is None:
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            label_image, spacing=spacing, level=0, method="lorensen"
        )

        return skimage.measure.mesh_surface_area(verts, faces)

    return numpy.sum(
        [
            numpy.round(_label_surface_area(label_image, label, spacing))
            for label in index
        ]
    )


def _label_surface_area(label_image, label, spacing):
    verts, faces, _normals, _values = skimage.measure.marching_cubes(
        label_image == label, spacing=spacing, level=0, method="lorensen"
    )

    return skimage.measure.mesh_surface_area(verts, faces)
