# coding=utf-8

"""
DilateObjects
===================

**DilateObjects** removes objects smaller or larger than the specified diameter.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.morphology
import skimage.segmentation

import cellprofiler.object
import cellprofiler.module
import cellprofiler.setting

SQUARE = 'Square'
DIAMOND = 'Diamond'
DISK = 'Disk'


class DilateObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "DilateObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateObjects, self).create_settings()

        self.size = cellprofiler.setting.Integer(
            text="Dilation Radius",
            value=2,
            doc="Radius by which to dilate a single pixel"
        )

        # TODO: MORE DOCUMENTATION HERE
        self.method = cellprofiler.setting.Choice(
            text="Dilation Method",
            choices=[SQUARE, DIAMOND, DISK],
            doc="Method by which to dilate a single pixel"
        )


    def settings(self):
        __settings__ = super(DilateObjects, self).settings()

        return __settings__ + [
            self.size,
            self.method
        ]

    def visible_settings(self):
        __settings__ = super(DilateObjects, self).visible_settings()

        return __settings__ + [
            self.size,
            self.method
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        object_set = workspace.object_set
        images = workspace.image_set

        x = object_set.get_objects(x_name)

        dimensions = x.dimensions
        y_data = x.segmented.copy()

        structure = None
        r = self.size.value

        if self.method == SQUARE:
            structure = get_structure(x, r*2, skimage.morphology.square, skimage.morphology.cube)
        elif self.method == DIAMOND:
            structure = get_structure(x, r, skimage.morphology.diamond, skimage.morphology.octahedron)
        elif self.method == DISK:
            structure = get_structure(x, r, skimage.morphology.disk, skimage.morphology.ball)

        y_data = skimage.morphology.dilation(y_data, structure)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data
        objects.parent_image = x.parent_image

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.segmented

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions


def get_structure(objects, size, func2D, func3D):
    if objects.volumetric:
        return func3D(size)
    return func2D(size)
