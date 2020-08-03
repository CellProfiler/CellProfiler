"""
ImageRegistration
=================

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import skimage.feature
import skimage.registration
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName
import numpy
import skimage.transform


class ImageRegistration(Module):
    category = "Advanced"

    module_name = "ImageRegistration"

    variable_revision_number = 1

    def create_settings(self):
        self.reference = ImageSubscriber(
            doc="""
            Reference image
            """,
            text="Reference image",
        )

        self.image = ImageSubscriber(
            doc="""
            Image
            """,
            text="Image",
        )

        self.registration = ImageName(
            doc="""
            Registration
            """,
            text="Registration",
            value="registration",
        )

    def settings(self):
        return [self.reference, self.image, self.registration]

    def run(self, workspace):
        images = workspace.image_set

        reference = images.get_image(self.reference.value)
        image = images.get_image(self.image.value)

        v, u = skimage.registration.optical_flow_tvl1(
            reference.pixel_data, image.pixel_data
        )

        r = numpy.arange(reference.shape[0])
        c = numpy.arange(reference.shape[1])

        r_coordinates, c_coordinates = numpy.meshgrid(r, c, indexing="ij")

        inverse_map = numpy.array([r_coordinates + v, c_coordinates + u])

        registration = skimage.transform.warp(image, inverse_map, mode="nearest")

        registration = Image(registration)

        if self.show_window:
            workspace.display_data.reference = reference.pixel_data
            workspace.display_data.image = image.pixel_data
            workspace.display_data.registration = registration.pixel_data

    def display(self, workspace, figure):
        pass
