import numpy
import scipy.ndimage
from cellprofiler_core.constants.measurement import FF_CHILDREN_COUNT, FF_PARENT
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.object import Objects
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float

from cellprofiler.modules import _help

__doc__ = """\
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
will alter the size of the objects within an image, but it will not change the size of the image itself.

See also
^^^^^^^^

{HELP_ON_SAVING_OBJECTS}

""".format(
    **{"HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS}
)


class ResizeObjects(ObjectProcessing):
    module_name = "ResizeObjects"

    variable_revision_number = 2

    def create_settings(self):
        super(ResizeObjects, self).create_settings()

        self.method = Choice(
            "Method",
            ["Dimensions", "Factor", "Match Image"],
            doc="""\
The following options are available:

-  *Dimensions:* Enter the new height and width of the resized objects.
-  *Factor:* Enter a single value which specifies the scaling.""",
            value="Factor",
        )

        self.factor = Float(
            "Factor",
            0.25,
            minval=0,
            doc="""\
*(Used only if resizing by "Factor")*

Numbers less than 1 will shrink the objects; numbers greater than 1 will
enlarge the objects.""",
        )

        self.width = Integer(
            "Width",
            100,
            minval=1,
            doc="""\
*(Used only if resizing by "Dimensions")*

Enter the desired width of the final objects, in pixels.""",
        )

        self.height = Integer(
            "Height",
            100,
            minval=1,
            doc="""\
*(Used only if resizing by "Dimensions")*

Enter the desired height of the final objects, in pixels.""",
        )

        self.specific_image = ImageSubscriber(
            "Select the image with the desired dimensions",
            "None",
            doc="""\
        *(Used only if resizing by specifying desired final dimensions using an image)*

        The input object set will be resized to the dimensions of the specified image.""",
        )

    def settings(self):
        settings = super(ResizeObjects, self).settings()

        settings += [
            self.method,
            self.factor,
            self.width,
            self.height,
            self.specific_image,
        ]

        return settings

    def visible_settings(self):
        visible_settings = super(ResizeObjects, self).visible_settings()

        visible_settings += [self.method]

        if self.method.value == "Dimensions":
            visible_settings += [self.width, self.height]
        elif self.method.value == "Factor":
            visible_settings += [self.factor]
        else:
            visible_settings += [self.specific_image]
        return visible_settings

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        objects = workspace.object_set
        x = objects.get_objects(x_name)
        dimensions = x.dimensions
        x_data = x.segmented

        if self.method.value == "Dimensions":
            y_data = resize(x_data, (self.height.value, self.width.value))
        elif self.method.value == "Match Image":
            target_image = workspace.image_set.get_image(self.specific_image.value)
            if target_image.volumetric:
                tgt_height, tgt_width = target_image.pixel_data.shape[1:3]
            else:
                tgt_height, tgt_width = target_image.pixel_data.shape[:2]
            size = (tgt_height, tgt_width)
            y_data = resize(x_data, size)
        else:
            y_data = rescale(x_data, self.factor.value)

        y = Objects()
        y.segmented = y_data
        objects.add_objects(y, y_name)
        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def add_measurements(
        self, workspace, input_object_name=None, output_object_name=None
    ):
        super(ObjectProcessing, self).add_measurements(workspace, self.y_name.value)

        labels = workspace.object_set.get_objects(self.y_name.value).segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        workspace.measurements.add_measurement(
            self.x_name.value,
            FF_CHILDREN_COUNT % self.y_name.value,
            [1] * len(unique_labels),
        )

        workspace.measurements.add_measurement(
            self.y_name.value, FF_PARENT % self.x_name.value, unique_labels,
        )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values += ["None"]
            variable_revision_number = 2
        return setting_values, variable_revision_number


def resize(data, size):
    if data.ndim == 3:
        size = (data.shape[0],) + size

    return scipy.ndimage.zoom(
        data,
        numpy.divide(numpy.multiply(1.0, size), data.shape),
        order=0,
        mode="nearest",
    )


# [SKIMAGE-14] ND-support for skimage.transform.rescale (https://github.com/scikit-image/scikit-image/pull/2587)
def rescale(data, factor):
    factor = (factor, factor)

    if data.ndim == 3:
        factor = (1,) + factor

    return scipy.ndimage.zoom(data, factor, order=0, mode="nearest")
