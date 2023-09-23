"""
Create an RGB image with color-coded labels overlaid on a grayscale image.
"""

from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Float
from cellprofiler_library.modules import overlayobjects
from cellprofiler_core.preferences import get_default_colormap


class OverlayObjects(ImageProcessing):
    module_name = "OverlayObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(OverlayObjects, self).create_settings()

        self.x_name.text = "Input"

        self.x_name.doc = "Objects will be overlaid on this image."

        self.y_name.doc = (
            "An RGB image with color-coded labels overlaid on a grayscale image."
        )

        self.objects = LabelSubscriber(
            text="Objects",
            doc="Color-coded labels of this object will be overlaid on the input image.",
        )

        self.opacity = Float(
            text="Opacity",
            value=0.3,
            minval=0.0,
            maxval=1.0,
            doc="""
            Opacity of overlaid labels. Increase this value to decrease the transparency of the colorized object
            labels.
            """,
        )

    def settings(self):
        settings = super(OverlayObjects, self).settings()

        settings += [self.objects, self.opacity]

        return settings

    def visible_settings(self):
        visible_settings = super(OverlayObjects, self).visible_settings()

        visible_settings += [self.objects, self.opacity]

        return visible_settings

    def run(self, workspace):
        self.function = lambda pixel_data, objects_name, opacity: overlayobjects(
            pixel_data,
            workspace.object_set.get_objects(objects_name).segmented,
            opacity,
            colormap=get_default_colormap()
        )

        super(OverlayObjects, self).run(workspace)

    def display(self, workspace, figure, cmap=None):
        if cmap is None:
            cmap = ["gray", None]
        super(OverlayObjects, self).display(workspace, figure, cmap=["gray", None])
