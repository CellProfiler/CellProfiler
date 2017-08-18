# coding=utf-8

"""
Closing
=======

**Closing** is the erosion of the dilation of an image. It’s used to
remove pepper noise.
"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Closing(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Closing"

    variable_revision_number = 1

    def create_settings(self):
        super(Closing, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):

        x = workspace.image_set.get_image(self.x_name.value)
        is_strel_2d = self.structuring_element.value.ndim == 2
        is_img_2d = x.pixel_data.ndim == 2

        if is_strel_2d and not is_img_2d:
            self.function = planewise_morphology_closing
        elif not is_strel_2d and is_img_2d:
            raise NotImplementedError("A 3D structuring element cannot be applied to a 2D image.")
        else:
            if x.pixel_data.dtype == numpy.bool:
                # issue #2837
                # self.function = skimage.morphology.binary_closing
                self.function = skimage.morphology.closing
            else:
                self.function = skimage.morphology.closing

        super(Closing, self).run(workspace)


def planewise_morphology_closing(x_data, structuring_element):
    y_data = numpy.zeros_like(x_data)

    for index, plane in enumerate(x_data):
        if x_data.dtype == numpy.bool:
            # issue #2837
            # y_data[index] = skimage.morphology.binary_closing(plane, structuring_element)
            y_data[index] = skimage.morphology.closing(plane, structuring_element)
        else:
            y_data[index] = skimage.morphology.closing(plane, structuring_element)

    return y_data
