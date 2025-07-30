"""
DilateImage
===========

**DilateImage** expands bright shapes in an image. See `this tutorial <https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html#dilation>`__ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import StructuringElement

from ._help import HELP_FOR_STREL


class DilateImage(ImageProcessing):
    category = "Advanced"

    module_name = "DilateImage"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateImage, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):
        from cellprofiler_library.modules._dilateimage import dilate_image
        
        x_name = self.x_name.value
        y_name = self.y_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        x_data = x.pixel_data
        
        # Call library function for dilation
        y_data = dilate_image(
            image=x_data,
            structuring_element=self.structuring_element.value
        )
        
        # Create output image and add to workspace
        from cellprofiler_core.image import Image
        y = Image(dimensions=dimensions, image=y_data, parent_image=x, convert=False)
        images.add(y_name, y)
        
        # Handle display data
        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = dimensions
