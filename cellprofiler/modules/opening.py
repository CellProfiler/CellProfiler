# coding=utf-8

"""

<strong>Opening</strong> is the dilation of the erosion of an image. It’s used to remove salt noise.

"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Opening(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Opening"

    variable_revision_number = 1

    def create_settings(self):
        super(Opening, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Opening, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Opening, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        x = workspace.image_set.get_image(self.x_name.value)
        is_strel_2d = self.structuring_element.value.ndim == 2
        is_img_2d = x.pixel_data.ndim == 2

        if is_strel_2d and not is_img_2d:
            self.function = planewise_morphology_opening
        elif not is_strel_2d and is_img_2d:
            raise NotImplementedError("A 3D structuring element cannot be applied to a 2D image.")
        else:
            if x.pixel_data.dtype == numpy.bool:
                self.function = skimage.morphology.binary_opening
            else:
                self.function = skimage.morphology.opening

        super(Opening, self).run(workspace)


def planewise_morphology_opening(x_data, structuring_element):
    y_data = numpy.zeros_like(x_data)

    for index, plane in enumerate(x_data):
        if x_data.dtype == numpy.bool:
            y_data[index] = skimage.morphology.binary_opening(plane, structuring_element)
        else:
            y_data[index] = skimage.morphology.opening(plane, structuring_element)

    return y_data
