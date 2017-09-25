# coding=utf-8

"""
ResizeObjects
=============

**ResizeObjects** will upsize or downsize an object’s label matrix by a factor or by specifying
the final dimensions in pixels. **ResizeObjects** is similar to **ResizeImage**, but
**ResizeObjects** is specific to CellProfiler objects created by modules such as
**IdentifyPrimaryObjects** or **Watershed**. **ResizeObjects** uses nearest neighbor
interpolation to preserve object labels after the resizing operation.

When resizing 3D data, the height and width will be changed, but
the original depth (or z-dimension) will be kept. This 3D behavior was chosen, because in most
cases the number of slices in a z-stack is much fewer than the number of pixels that define the
x-y dimensions. Otherwise, a significant fraction of z information would be lost during downsizing.

**ResizeObjects** is useful for processing very large or 3D data to reduce computation time. You
might downsize a 3D image with **ResizeImage** to generate a segmentation, then use
**ResizeObjects** to stretch the segmented objects to their original size
before computing measurements with the original 3D image. **ResizeObjects** differs
from **ExpandOrShrinkObjects** and **ShrinkToObjectCenters** in that the overall dimensions
of the object label matrix, or image, are changed. In contrast, **ExpandOrShrinkObjects**
will alter the size of the objects within an image, but it will not change the size of the image
itself.
"""

import numpy
import scipy.ndimage
import skimage.transform

import cellprofiler.module
import cellprofiler.setting


class ResizeObjects(cellprofiler.module.ObjectProcessing):
    module_name = "ResizeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ResizeObjects, self).create_settings()

        self.method = cellprofiler.setting.Choice(
            "Method",
            [
                "Dimensions",
                "Factor"
            ],
            doc="""
            The following options are available:
            <ul>
                <li><i>Dimensions:</i> Enter the new height and width of the resized objects.</li>
                <li><i>Factor:</i> Enter a single value which specifies the scaling.</li>
            </ul>
            """,
            value="Factor"
        )

        self.factor = cellprofiler.setting.Float(
            "Factor",
            0.25,
            minval=0,
            doc="""
            <i>(Used only if resizing by <i>Factor</i>)</i><br />
            Numbers less than 1 will shrink the objects; numbers greater than 1 will enlarge the objects.
            """
        )

        self.width = cellprofiler.setting.Integer(
            "Width",
            100,
            minval=1,
            doc="""
            <i>(Used only if resizing by <i>Dimensions</i>)</i><br />
            Enter the desired width of the final objects, in pixels.
            """
        )

        self.height = cellprofiler.setting.Integer(
            "Height",
            100,
            minval=1,
            doc="""
            <i>(Used only if resizing by <i>Dimensions</i>)</i><br />
            Enter the desired height of the final objects, in pixels.
            """
        )

    def settings(self):
        settings = super(ResizeObjects, self).settings()

        settings += [
            self.method,
            self.factor,
            self.width,
            self.height
        ]

        return settings

    def visible_settings(self):
        visible_settings = super(ResizeObjects, self).visible_settings()

        visible_settings += [
            self.method
        ]

        if self.method.value == "Dimensions":
            visible_settings += [
                self.width,
                self.height
            ]
        else:
            visible_settings += [
                self.factor
            ]

        return visible_settings

    def run(self, workspace):
        self.function = lambda data, method, factor, width, height: \
            resize(data, (height, width)) if method == "Dimensions" else rescale(data, factor)

        super(ResizeObjects, self).run(workspace)

    def add_measurements(self, workspace, input_object_name=None, output_object_name=None):
        super(cellprofiler.module.ObjectProcessing, self).add_measurements(workspace, self.y_name.value)

        labels = workspace.object_set.get_objects(self.y_name.value).segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        workspace.measurements.add_measurement(
            self.x_name.value,
            cellprofiler.measurement.FF_CHILDREN_COUNT % self.y_name.value,
            [1] * len(unique_labels)
        )

        workspace.measurements.add_measurement(
            self.y_name.value,
            cellprofiler.measurement.FF_PARENT % self.x_name.value,
            unique_labels
        )


def resize(data, size):
    if data.ndim == 3:
        size = (data.shape[0],) + size

    return scipy.ndimage.zoom(
        data,
        numpy.divide(numpy.multiply(1.0, size), data.shape),
        order=0,
        mode="nearest"
    )


# [SKIMAGE-14] ND-support for skimage.transform.rescale (https://github.com/scikit-image/scikit-image/pull/2587)
def rescale(data, factor):
    factor = (factor, factor)

    if data.ndim == 3:
        factor = (1,) + factor

    return scipy.ndimage.zoom(
        data,
        factor,
        order=0,
        mode="nearest"
    )
