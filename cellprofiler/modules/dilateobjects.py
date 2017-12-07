# coding=utf-8

"""
DilateObjects
=============

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


class DilateObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "DilateObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateObjects, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement(allow_planewise=True)

    def settings(self):
        __settings__ = super(DilateObjects, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(DilateObjects, self).visible_settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        object_set = workspace.object_set

        x = object_set.get_objects(x_name)

        is_strel_2d = self.structuring_element.value.ndim == 2

        is_img_2d = x.segmented.ndim == 2

        if is_strel_2d and not is_img_2d:

            self.function = planewise_morphology_dilation

        elif not is_strel_2d and is_img_2d:

            raise NotImplementedError("A 3D structuring element cannot be applied to 2D objects.")

        else:

            self.function = skimage.morphology.dilation

        super(DilateObjects, self).run(workspace)


def planewise_morphology_dilation(x_data, structuring_element):

    y_data = numpy.zeros_like(x_data)

    for index, plane in enumerate(x_data):

        y_data[index] = skimage.morphology.dilation(plane, structuring_element)

    return y_data
