# coding=utf-8
"""

Thresholding

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters
import skimage.morphology


class Thresholding(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "Thresholding"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
        )

        self.method = cellprofiler.setting.Choice(
            u"Method",
            [
                u"Adaptive",
                u"ISODATA",
                u"Li’s Minimum Cross Entropy",
                u"Otsu’s method",
                u"Yen’s method"
            ]
        )

        self.adaptive_block_size = cellprofiler.setting.Integer(
            u"Block size",
            3
        )

        self.adaptive_method = cellprofiler.setting.Choice(
            u"Adaptive method",
            [
                u"Gaussian",
                u"Generic",
                u"Mean",
                u"Median"
            ],
            u"Gaussian"
        )

        self.adaptive_offset = cellprofiler.setting.Float(
            u"Offset",
            0.0
        )

        self.adaptive_mode = cellprofiler.setting.Choice(
            u"Mode",
            [
                u"Constant",
                u"Mirror",
                u"Nearest",
                u"Reflect",
                u"Wrap"
            ],
            u"Reflect"
        )

        self.isodata_bins = cellprofiler.setting.Integer(
            u"Bins",
            256
        )

        self.otsu_bins = cellprofiler.setting.Integer(
            u"Bins",
            256
        )

        self.yen_bins = cellprofiler.setting.Integer(
            u"Bins",
            256
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.method,
            self.adaptive_block_size,
            self.adaptive_method,
            self.adaptive_offset,
            self.adaptive_mode,
            self.isodata_bins,
            self.otsu_bins,
            self.yen_bins
        ]

    def visible_settings(self):
        settings = [
            self.x_name,
            self.y_name,
            self.method
        ]

        if self.method.value == u"Adaptive":
            settings = settings + [
                self.adaptive_block_size,
                self.adaptive_method,
                self.adaptive_offset,
                self.adaptive_mode
            ]

        if self.method.value == u"ISODATA":
            settings = settings + [
                self.isodata_bins
            ]

        if self.method.value == u"Otsu’s method":
            settings = settings + [
                self.otsu_bins
            ]

        if self.method.value == u"Yen’s method":
            settings = settings + [
                self.yen_bins
            ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        y_data = numpy.zeros_like(x_data)

        for plane, image in enumerate(x_data):
            if self.method.value == u"Adaptive":
                y_data[plane] = skimage.filters.threshold_adaptive(
                    image=image,
                    block_size=self.adaptive_block_size,
                    method=self.adaptive_method,
                    offset=self.adaptive_offset,
                    mode=self.adaptive_mode
                )
            elif self.method.value == u"ISODATA":
                y_data[plane] = skimage.filters.threshold_isodata(
                    image=image,
                    nbins=self.isodata_bins
                )
            elif self.method.value == u"Li’s Minimum Cross Entropy":
                y_data[plane] = skimage.filters.threshold_li(
                    image=image
                )
            elif self.method.value == u"Otsu’s method":
                y_data[plane] = skimage.filters.threshold_otsu(
                    image=image,
                    nbins=self.otsu_bins
                )
            elif self.method.value == u"Yen’s method":
                y_data[plane] = skimage.filters.threshold_yen(
                    image=image,
                    nbins=self.yen_bins
                )

        y_data = skimage.exposure.rescale_intensity(y_data * 1.0)

        y_data = x_data >= y_data

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        dimensions = (2, 1)

        x_data = workspace.display_data.x_data[16]
        y_data = workspace.display_data.y_data[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            x_data,
            colormap="gray"
        )

        figure.subplot_imshow(
            1,
            0,
            y_data,
            colormap="gray"
        )
