# coding=utf-8

"""
Remove labeled holes
====================

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class RemoveLabeledHoles(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "RemoveLabeledHoles"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveLabeledHoles, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0
        )

    def settings(self):
        __settings__ = super(RemoveLabeledHoles, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveLabeledHoles, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda image, diameter: remove_objects_and_fill_holes(image, diameter)

        super(RemoveLabeledHoles, self).run(workspace)


def remove_objects_and_fill_holes(image, diameter):
    radius = diameter / 2.0

    image = skimage.img_as_bool(image)

    if image.ndim == 2 or image.shape[-1] in (3, 4):
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    size = numpy.pi * factor

    return skimage.morphology.remove_small_holes(image, size)
