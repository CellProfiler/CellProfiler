# coding=utf-8

"""
MergeObjects
===========

**MergeObjects** merges objects below a certain threshold into its most prevalent, adjacent neighbor.

The output of this module is a object image of the same data type as the input.
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

        self.slice_wise = cellprofiler.setting.Binary(
            text="Slice wise merge",
            value=False,
            doc="""\
Select "*{YES}*" to merge objects on a per-slice level. 
This will perform the "significant neighbor" merge on 
each slice of a volumetric image, rather than on the 
image as a whole. This may be helpful for removing seed
artifacts that are the result of segmentation.
**Note**: Slice-wise operations will be considerably slower.
""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

    def settings(self):
        __settings__ = super(MergeObjects, self).settings()

        return __settings__ + [
            self.size,
            self.slice_wise
        ]

    def visible_settings(self):
        __settings__ = super(MergeObjects, self).visible_settings()

        return __settings__ + [
            self.size,
            self.slice_wise
        ]

    def run(self, workspace):
        self.function = lambda labels, diameter, slicewise: \
            merge_objects(labels, diameter, slicewise)

        super(MergeObjects, self).run(workspace)


def _merge_neighbors(array, min_obj_size):
    sizes = np.bincount(array.ravel())
    # Find the indices of all objects below threshold
    mask_sizes = (sizes < min_obj_size) & (sizes != 0)

    merged = np.copy(array)
    # Iterate through each small object, determine most significant adjacent neighbor,
    # and merge the object into that neighbor
    for n in np.nonzero(mask_sizes)[0]:
        mask = array == n
        # "Thick" mode ensures the border bleeds into the neighboring objects
        bound = skimage.segmentation.find_boundaries(mask, mode='thick')
        neighbors = np.bincount(array[bound].ravel())
        # If self is the largest neighbor, the object should be removed
        if len(neighbors) >= n:
            neighbors[n] = 0
        # Background should be set to 0
        neighbors[0] = 0
        max_neighbor = np.argmax(neighbors)
        # Set object value to largest neighbor
        merged[merged == n] = max_neighbor
    return merged


def merge_objects(labels, diameter, slicewise):
    radius = diameter / 2.0

    if labels.ndim == 2 or labels.shape[-1] in (3, 4) or slicewise:
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = np.pi * factor

    if not slicewise:
        return _merge_neighbors(labels, min_obj_size)
    return np.array([_merge_neighbors(x, min_obj_size) for x in labels])
