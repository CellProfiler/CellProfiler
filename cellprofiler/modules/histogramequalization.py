# -*- coding: utf-8 -*-

"""

Increase the global contrast of a low-contrast image or volume. A low contrast image has a background and
foreground that are both dark, or both light. Histogram equalization redistributes intensities such that the most
common frequencies are more distinct, increasing contrast.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure


class HistogramEqualization(cellprofiler.module.ImageProcessing):
    module_name = "HistogramEqualization"

    variable_revision_number = 1

    def create_settings(self):
        super(HistogramEqualization, self).create_settings()

        self.nbins = cellprofiler.setting.Integer(
            u"Bins",
            value=256,
            minval=0,
            doc="Number of bins for image histogram."
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            u"Mask",
            can_be_blank=True,
            doc="""
            Optional. Mask image must be the same size as "Input". Only unmasked points of the "Input" image are used
            to compute the equalization, which is applied to the entire "Input" image.
            """
        )

        self.local = cellprofiler.setting.Binary(
            u"Local",
            False
        )

    def settings(self):
        __settings__ = super(HistogramEqualization, self).settings()

        return __settings__ + [
            self.nbins,
            self.mask,
            self.local
        ]

    def visible_settings(self):
        __settings__ = super(HistogramEqualization, self).settings()

        __settings__ += [self.local, self.nbins]

        if not self.local.value:
            __settings__ += [self.mask]

        return __settings__

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        mask_data = None

        if not self.mask.is_blank:
            mask_name = self.mask.value

            mask = images.get_image(mask_name)

            mask_data = mask.pixel_data

        nbins = self.nbins.value

        if self.local.value:
            if x.volumetric:
                y_data = numpy.zeros_like(x_data)

                for index, plane in enumerate(x_data):
                    y_data[index] = skimage.exposure.equalize_adapthist(plane, nbins=nbins)
            else:
                y_data = skimage.exposure.equalize_adapthist(x_data, nbins=nbins)
        else:
            y_data = skimage.exposure.equalize_hist(x_data, nbins=nbins, mask=mask_data)

        y = cellprofiler.image.Image(
            dimensions=dimensions,
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
