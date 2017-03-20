# -*- coding: utf-8 -*-

"""

Remove objects

"""

import numpy
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class RemoveObjects(cellprofiler.module.ObjectProcessing):
    category = "Mathematical morphology"

    module_name = "Remove objects"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveObjects, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0
        )

    def settings(self):
        __settings__ = super(RemoveObjects, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveObjects, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda labels, diameter: remove_objects(labels, diameter)

        super(RemoveObjects, self).run(workspace)


def remove_objects(labels, diameter):
    radius = diameter / 2.0

    if labels.ndim == 2:
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    size = numpy.pi * factor

    labels = skimage.morphology.remove_small_objects(labels, size)

    return skimage.segmentation.relabel_sequential(labels)[0]
