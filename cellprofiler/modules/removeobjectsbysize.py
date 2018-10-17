# coding=utf-8

"""
RemoveObjectsBySize
===================

**RemoveObjectsBySize** removes objects smaller or larger than the specified diameter.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

"""

import numpy
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class RemoveObjectsBySize(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "RemoveObjectsBySize"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveObjectsBySize, self).create_settings()

        self.size = cellprofiler.setting.FloatRange(
            text="Size",
            value=(0.0, numpy.inf),
            doc="""
            Specify the minimum and maximum diameters of objects (in pixels) to remove. Set the first value to 0 to
            keep small objects. Set the second value to "inf" to keep large objects.
            """
        )

    def settings(self):
        __settings__ = super(RemoveObjectsBySize, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveObjectsBySize, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda labels, diameter: remove_objects(labels, diameter)

        super(RemoveObjectsBySize, self).run(workspace)


def remove_objects(labels, diameter):
    labels = labels.copy()

    radius = numpy.divide(diameter, 2.0)

    if labels.ndim == 2:
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_size, max_size = numpy.pi * factor

    if min_size > 0.0:
        labels = skimage.morphology.remove_small_objects(labels, min_size)

    if max_size < numpy.inf:
        labels ^= skimage.morphology.remove_small_objects(labels, max_size)

    return skimage.segmentation.relabel_sequential(labels)[0]
