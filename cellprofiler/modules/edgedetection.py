# coding=utf-8

"""

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.color
import skimage.filters


class EdgeDetection(cellprofiler.module.Module):
    category = "Feature Detection"

    module_name = "EdgeDetection"

    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input",
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"EdgeDetection"
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            u"Mask",
            can_be_blank=True
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.mask
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
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

        if dimensions is 2:
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

            workspace.dimensions = dimensions

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 1))

        figure.subplot_imshow(0, 0, workspace.display_data.x_data, dimensions=dimensions)

        figure.subplot_imshow(0, 1, workspace.display_data.y_data, dimensions=dimensions)
