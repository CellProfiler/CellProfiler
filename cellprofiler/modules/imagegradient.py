# coding=utf-8

"""
**Image gradient** computes the local gradient of the image. The image
gradient is useful for finding boundaries of objects. In a gradient
image, pixels at the edges of bright regions of interest have the
brightest intensities. Pixels in the background or in the centers of
regions of interest have zero or dimmer intensity.
"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage
import skimage.filters
import skimage.morphology


class ImageGradient(cellprofiler.module.ImageProcessing):
    module_name = "ImageGradient"

    variable_revision_number = 1

    def create_settings(self):
        super(ImageGradient, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement(
            doc="""Neighborhood in which to compute the local gradient. Select a two-dimensional shape such as "disk"
            for images, and a three-dimensional shape such as "ball" for volumes. A larger size will compute the gradient
            over larger patches of the image and can obscure smaller features."""
        )

    def settings(self):
        __settings__ = super(ImageGradient, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(ImageGradient, self).visible_settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        if x.dimensions == 3 or x.multichannel:
            y_data = numpy.zeros_like(x_data)

            for z, image in enumerate(x_data):
                y_data[z] = skimage.filters.rank.gradient(image, self.__structuring_element())
        else:
            y_data = skimage.filters.rank.gradient(x_data, self.structuring_element.value)

        y = cellprofiler.image.Image(
            image=y_data,
            dimensions=x.dimensions,
            parent_image=x,
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions

    def __structuring_element(self):
        shape = self.structuring_element.shape

        size = self.structuring_element.size

        if shape == "ball":
            return skimage.morphology.disk(size)

        if shape == "cube":
            return skimage.morphology.square(size)

        if shape == "octahedron":
            return skimage.morphology.diamond(size)

        return self.structuring_element.value
