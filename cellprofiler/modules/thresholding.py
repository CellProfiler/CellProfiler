# -*- coding: utf-8 -*-

"""

Thresholding

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.filters


class Thresholding(cellprofiler.module.ImageProcessing):
    module_name = "Thresholding"

    variable_revision_number = 1

    def create_settings(self):
        super(Thresholding, self).create_settings()

        self.operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Adaptive",
                u"Iterative selection thresholding",
                u"Manual",
                u"Minimum cross entropy thresholding",
                u"Otsu’s method",
                u"Yen’s method"
            ]
        )

        self.block_size = cellprofiler.setting.OddInteger(
            u"Block size",
            value=3
        )

        self.adaptive_method = cellprofiler.setting.Choice(
            u"Adaptive method",
            [
                u"Gaussian",
                u"Mean",
                u"Median"
            ],
            u"Gaussian"
        )

        self.sigma = cellprofiler.setting.Float(
            u"Sigma",
            1.0,
            minval=0.0
        )

        self.offset = cellprofiler.setting.Float(
            u"Offset",
            value=0.0
        )

        self.bins = cellprofiler.setting.Integer(
            u"Bins",
            value=256
        )

        self.lower = cellprofiler.setting.Float(
            u"Lower",
            value=0.0
        )

        self.upper = cellprofiler.setting.Float(
            u"Upper",
            value=1.0
        )

    def settings(self):
        __settings__ = super(Thresholding, self).settings()

        return __settings__ + [
            self.adaptive_method,
            self.bins,
            self.block_size,
            self.lower,
            self.offset,
            self.operation,
            self.sigma,
            self.upper,
            self.x_name,
            self.y_name
        ]

    def visible_settings(self):
        __settings__ = super(Thresholding, self).visible_settings()

        settings = __settings__ + [
            self.operation
        ]

        if self.operation.value == u"Adaptive":
            settings = settings + [
                self.adaptive_method
            ]

            if self.adaptive_method == u"Gaussian":
                settings = settings + [
                    self.sigma
                ]

            settings = settings + [
                self.block_size,
                self.offset
            ]

        if self.operation.value in [
            u"Iterative selection thresholding",
            u"Otsu’s method",
            u"Yen’s method"
        ]:
            settings = settings + [
                self.bins
            ]

        if self.operation.value == u"Manual":
            settings = settings + [
                self.lower,
                self.upper
            ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        y_data = numpy.zeros_like(x_data)

        if self.operation.value == u"Adaptive":
            y_data = skimage.filters.threshold_adaptive(
                image=x_data,
                block_size=self.block_size.value,
                method=self.adaptive_method.value.lower(),
                offset=self.offset.value,
                param=self.sigma.value
            )
        elif self.operation.value == u"Iterative selection thresholding":
            y_data = skimage.filters.threshold_isodata(
                image=x_data,
                nbins=self.bins.value
            )

            y_data = x_data >= y_data
        elif self.operation.value == u"Manual":
            y_data = x_data > self.lower

            y_data = y_data < self.upper
        elif self.operation.value == u"Minimum cross entropy thresholding":
            y_data = skimage.filters.threshold_li(
                image=x_data
            )

            y_data = x_data >= y_data
        elif self.operation.value == u"Otsu’s method":
            y_data = skimage.filters.threshold_otsu(
                image=x_data,
                nbins=self.bins.value
            )

            y_data = x_data >= y_data
        elif self.operation.value == u"Yen’s method":
            y_data = skimage.filters.threshold_yen(
                image=x_data,
                nbins=self.bins.value
            )

            y_data = x_data >= y_data

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=x.dimensions
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions
