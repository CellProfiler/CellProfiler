import skimage
import skimage.measure
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.text import Integer

HELP_BINARY_IMAGE = """\
This module can also convert a grayscale image to binary before converting it to an object.
Connected components of the binary image are assigned to the same object. This feature is 
useful for identifying objects that can be cleanly distinguished using **Threshold**. 
If you wish to distinguish clumped objects, see **Watershed** or the **Identify** modules.

Note that grayscale images provided as input with this setting will be converted to binary 
images. Pixel intensities below or equal to 50% of the input's full intensity range are 
assigned to the background (i.e., assigned the value 0). Pixel intensities above 50% of 
the input's full intensity range are assigned to the foreground (i.e., assigned the
value 1).
"""

__doc__ = """\
ConvertImageToObjects
=====================

**ConvertImageToObjects** converts an image to objects. This module is useful for importing
a previously segmented or labeled image into CellProfiler, as it will preserve the labels
of an integer-labelled input. 

{HELP_BINARY_IMAGE}

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

""".format(
    **{"HELP_BINARY_IMAGE": HELP_BINARY_IMAGE}
)


class ConvertImageToObjects(ImageSegmentation):
    category = "Object Processing"

    module_name = "ConvertImageToObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ConvertImageToObjects, self).create_settings()

        self.cast_to_bool = Binary(
            text="Convert to boolean image", value=True, doc=HELP_BINARY_IMAGE
        )

        self.preserve_labels = Binary(
            text="Preserve original labels",
            value=False,
            doc="""\
By default, this module will re-label the input image.
Setting this to *{YES}* will ensure that the original labels 
(i.e. pixel values of the objects) are preserved.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.background_label = Integer(
            text="Background label",
            value=0,
            doc="""\
Consider all pixels with this value as background pixels, and label them as 0. 
By default, 0-valued pixels are considered as background pixels.
""",
        )

        self.connectivity = Integer(
            text="Connectivity",
            minval=0,
            value=0,
            doc="""\
Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
Accepted values are ranging from 1 to the number of dimensions of the input. 
If set to 0, a full connectivity of the input dimension is used.
""",
        )

    def settings(self):
        __settings__ = super(ConvertImageToObjects, self).settings()

        return __settings__ + [
            self.cast_to_bool,
            self.preserve_labels,
            self.background_label,
            self.connectivity,
        ]

    def visible_settings(self):
        __settings__ = super(ConvertImageToObjects, self).visible_settings()

        __settings__ += [self.cast_to_bool]

        if not self.cast_to_bool.value:
            __settings__ += [self.preserve_labels]

        if not self.preserve_labels.value:
            __settings__ += [self.background_label, self.connectivity]

        return __settings__

    def run(self, workspace):
        self.function = lambda data, cast_to_bool, preserve_label, background, connectivity: convert_to_objects(
            data, cast_to_bool, preserve_label, background, connectivity
        )

        super(ConvertImageToObjects, self).run(workspace)

    def display(self, workspace, figure):
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title=self.x_name.value,
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )


def convert_to_objects(data, cast_to_bool, preserve_label, background, connectivity):
    # Compatibility with skimage
    connectivity = None if connectivity == 0 else connectivity

    caster = skimage.img_as_bool if cast_to_bool else skimage.img_as_uint
    data = caster(data)

    # If preservation is desired, just return the original labels
    if preserve_label and not cast_to_bool:
        return data

    return skimage.measure.label(data, background=background, connectivity=connectivity)
