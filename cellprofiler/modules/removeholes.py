# coding=utf-8

"""
Remove holes
============
"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class RemoveHoles(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "RemoveHoles"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveHoles, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0
        )

    def settings(self):
        __settings__ = super(RemoveHoles, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveHoles, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda image, diameter: fill_holes(image, diameter)

        super(RemoveHoles, self).run(workspace)


def fill_holes(image, diameter):
    radius = diameter / 2.0

    image = skimage.img_as_bool(image)

    if image.ndim == 2 or image.shape[-1] in (3, 4):
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    size = numpy.pi * factor

    return skimage.morphology.remove_small_holes(image, size)
