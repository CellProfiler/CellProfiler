"""
MatchTemplate
=============

The **MatchTemplate** module uses `normalized cross-correlation`_ to
match a template to a single-channel two-or-three dimensional image or
multi-channel two-dimensional image. The output of the module is an
image where each pixel corresponds to the `Pearson product-moment
correlation coefficient`_ between the image and the template. Practically, this 
allows you to crop a single object of interest (i.e., a cell) and predict where
other such objects are in the image. Note that this is not rotation invariant, so
this module will perform best when objects are approximately round or are angled
in a relatively unified direction.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

.. _normalized cross-correlation: http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation
.. _Pearson product-moment correlation coefficient: http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
"""
import imageio
import skimage.feature
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Pathname, ImageName


class MatchTemplate(Module):
    module_name = "MatchTemplate"
    category = "Advanced"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = ImageSubscriber(
            "Image", doc="Select the image you want to use."
        )

        self.template_name = Pathname(
            "Template",
            doc="Specify the location of the cropped image you want to use as a template.",
        )

        self.output_image_name = ImageName(
            "Output",
            doc="Enter the name you want to call the image produced by this module.",
        )

    def settings(self):
        return [self.input_image_name, self.template_name, self.output_image_name]

    def visible_settings(self):
        return [self.input_image_name, self.template_name, self.output_image_name]

    def run(self, workspace):
        input_image_name = self.input_image_name.value

        template_name = self.template_name.value

        output_image_name = self.output_image_name.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name)

        input_pixels = input_image.pixel_data

        template = imageio.imread(template_name)

        output_pixels = skimage.feature.match_template(
            image=input_pixels, template=template, pad_input=True
        )

        output_image = Image(output_pixels, parent_image=input_image)

        image_set.add(output_image_name, output_image)

        if self.show_window:
            workspace.display_data.input_pixels = input_pixels

            workspace.display_data.template = template

            workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        dimensions = (2, 1)

        figure.set_subplots(dimensions)

        figure.subplot_imshow(0, 0, workspace.display_data.input_pixels, "Image")

        figure.subplot_imshow(
            1,
            0,
            workspace.display_data.output_pixels,
            "Correlation coefficient",
            sharexy=figure.subplot(0, 0),
        )
