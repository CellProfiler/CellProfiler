# coding=utf-8

"""

Detect edges in an image or volume using the Sobel transform. Multi-channel images are converted to grayscale before
the transform is applied. An edge is a region in which intensity changes dramatically. For example, an edge is the line
between a dark background and a bright foreground.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.color
import skimage.filters


class EdgeDetection(cellprofiler.module.ImageProcessing):
    category = "Feature Detection"

    module_name = "EdgeDetection"

    variable_revision_number = 1

    def create_settings(self):
        super(EdgeDetection, self).create_settings()

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            can_be_blank=True,
            doc="""
            Optional. A binary image the same shape as "Input". Limit application of the edge filter to unmasked
            regions of "Input".
            """
        )

    def settings(self):
        __settings__ = super(EdgeDetection, self).settings()

        return __settings__ + [
            self.mask
        ]

    def visible_settings(self):
        __settings__ = super(EdgeDetection, self).visible_settings()

        return __settings__ + [
            self.mask
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        if x.multichannel:
            x_data = skimage.color.rgb2gray(x_data)

        mask_data = None

        if not self.mask.is_blank:
            mask_name = self.mask.value

            mask = images.get_image(mask_name)

            mask_data = mask.pixel_data

        dimensions = x.dimensions

        if dimensions == 2:
            y_data = skimage.filters.sobel(x_data, mask=mask_data)
        else:
            y_data = numpy.zeros_like(x_data)

            for plane, image in enumerate(x_data):
                plane_mask = None if mask_data is None else mask_data[plane]

                y_data[plane] = skimage.filters.sobel(image, mask=plane_mask)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=dimensions
        )

        y_name = self.y_name.value

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
