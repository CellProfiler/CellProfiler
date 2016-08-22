# coding=utf-8
"""

Edge detection

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.morphology


class EdgeDetection(cellprofiler.module.Module):
    category = "Feature detection"
    module_name = "Edge detection"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input",
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"OutputImage"
        )

        self.operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Canny edge detection",
                u"Prewitt operator",
                u"Roberts’ cross",
                u"Sobel operator"
            ]
        )

        self.sigma = cellprofiler.setting.Float(
            u"Sigma",
            1.0
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            u"Mask",
            can_be_blank=True
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.operation,
            self.sigma,
            self.mask
        ]

    def visible_settings(self):
        settings = [
            self.x_name,
            self.y_name,
            self.operation
        ]

        if self.operation.value == u"Canny edge detection":
            settings = settings + [
                self.sigma,
                self.mask
            ]

        if self.operation.value == u"Prewitt operator":
            settings = settings + [
                self.mask
            ]

        if self.operation.value == u"Roberts’ cross":
            settings = settings + [
                self.mask
            ]

        if self.operation.value == u"Sobel operator":
            settings = settings + [
                self.mask
            ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        y_data = numpy.zeros_like(x_data)

        mask = self.mask.value
        if self.mask.is_blank:
            mask = None

        for plane, image in enumerate(x_data):
            if self.operation.value == u"Canny edge detection":
                y_data[plane] = skimage.feature.canny(x_data[plane], sigma=self.sigma.value, mask=mask)
            elif self.operation.value == u"Prewitt operator":
                y_data[plane] = skimage.filters.prewitt(x_data[plane], mask=mask)
            elif self.operation.value == u"Roberts’ cross":
                y_data[plane] = skimage.filters.roberts(x_data[plane], mask=mask)
            elif self.operation.value == u"Sobel operator":
                y_data[plane] = skimage.filters.sobel(x_data[plane], mask=mask)

        y_data = skimage.exposure.rescale_intensity(y_data * 1.0)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        figure.gridspec((1, 2), (3, 3))

        figure.add_grid(0, workspace.display_data.x_data)

        figure.add_grid(1, workspace.display_data.y_data)
