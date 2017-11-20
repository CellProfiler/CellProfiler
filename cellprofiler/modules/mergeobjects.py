# coding=utf-8

"""
MergeObjects
===========

**MergeObjects** merges objects below a certain threshold into its most prevalent, adjacent neighbor.

This module works best on integer-labeled images (i.e., the output of **ConvertObjectsToImage**
when the color format is *uint16*).

The output of this module is a labeled image of the same data type as the input.
**MergeObjects** can be run *after* any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**). Labels are preserved and, where possible, small
objects are merged into neighboring objects that constitute a majority of the small object's
border. This is useful for reversing over-segmentation and artifacts that result from seeding
modules.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy as np
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class MergeObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "MergeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(MergeObjects, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Minimum object size",
            value=64.,
            doc="Objects smaller than this diameter will be merged with their most significant neighbor."
        )

    def settings(self):
        __settings__ = super(MergeObjects, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(MergeObjects, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda labels, diameter: \
            fill_object_holes(labels, diameter)

        super(MergeObjects, self).run(workspace)


def fill_object_holes(labels, diameter):
    radius = diameter / 2.0

    if labels.ndim == 2 or labels.shape[-1] in (3, 4):
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = np.pi * factor

    sizes = np.bincount(labels.ravel())
    # Find the indices of all objects below threshold
    mask_sizes = (sizes < min_obj_size) & (sizes != 0)

    merged = np.copy(labels)
    # Iterate through each small object, determine most significant adjacent neighbor,
    # and merge the object into that neighbor
    for n in np.nonzero(mask_sizes)[0]:
        mask = labels == n
        # "Thick" mode ensures the border bleeds into the neighboring objects
        bound = skimage.segmentation.find_boundaries(mask, mode='thick')
        neighbors = np.bincount(labels[bound].ravel())
        # If self is the largest neighbor, the object should be removed
        if len(neighbors) >= n:
            neighbors[n] = 0
        # Background should be set to 0
        neighbors[0] = 0
        max_neighbor = np.argmax(neighbors)
        # Set object value to largest neighbor
        merged[merged == n] = max_neighbor

    return merged
