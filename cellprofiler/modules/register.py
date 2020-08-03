"""
Register
========

The Register module resolves cross-image rotation, scale, and skew problems by
translating an image’s coordinate system into a reference image’s coordinate
system.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import numpy
import skimage.feature
import skimage.registration
import skimage.transform
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName


class Register(Module):
    category = "Advanced"

    module_name = "Register"

    variable_revision_number = 1

    def create_settings(self):
        self.reference = ImageSubscriber(
            doc="""
            Reference image
            """,
            text="Reference image",
        )

        self.x = ImageSubscriber(
            doc="""
            Input image
            """,
            text="Input image",
        )

        self.y = ImageName(
            doc="""
            Output image name
            """,
            text="Output image name",
            value=self.__class__.__name__,
        )

    def settings(self):
        return [self.reference, self.x, self.y]

    def run(self, workspace):
        images = workspace.image_set

        reference = images.get_image(self.reference.value)

        x_image = images.get_image(self.x.value)

        v, u = skimage.registration.optical_flow_tvl1(
            reference_image=reference.pixel_data, moving_image=x_image.pixel_data
        )

        r = numpy.arange(reference.pixel_data.shape[0])
        c = numpy.arange(reference.pixel_data.shape[1])

        r_coordinates, c_coordinates = numpy.meshgrid(r, c, indexing="ij")

        inverse_map = numpy.array([r_coordinates + v, c_coordinates + u])

        y = skimage.transform.warp(x_image.pixel_data, inverse_map, mode="nearest")

        y_image = Image(y)

        workspace.image_set.add(self.y.value, y_image)

        if self.show_window:
            workspace.display_data.dimensions = x_image.dimensions
            workspace.display_data.reference = reference.pixel_data
            workspace.display_data.u = u
            workspace.display_data.v = v
            workspace.display_data.x = x_image.pixel_data
            workspace.display_data.y = y_image.pixel_data

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions

        subplots = (2, 2)

        figure.set_subplots(dimensions=dimensions, subplots=subplots)

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.reference,
            title=f"Reference ({self.reference.value})",
            x=0,
            y=0,
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x,
            title=f"Image ({self.x.value})",
            sharexy=figure.subplot(0, 0),
            x=1,
            y=0,
        )

        stack = [
            workspace.display_data.x,
            workspace.display_data.v,
            workspace.display_data.u,
        ]

        stack = numpy.stack(stack, -1)

        figure.subplot_imshow(
            image=stack, sharexy=figure.subplot(0, 0), title="Optical flow", x=0, y=1,
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.y,
            sharexy=figure.subplot(0, 0),
            title=f"Registration ({self.y.value})",
            x=1,
            y=1,
        )
