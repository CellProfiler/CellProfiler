# -*- coding: utf-8 -*-

"""

Gaussian filter

"""

import cellprofiler.module
import cellprofiler.setting
import skimage.filters
import numpy


class GaussianFilter(cellprofiler.module.ImageProcessing):
    module_name = "GaussianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(GaussianFilter, self).create_settings()

        self.sigma = cellprofiler.setting.Integer(
            text="Sigma",
            value=1
        )

        self.shape = cellprofiler.setting.Choice(
            "Shape of Gaussian",
            choices=[
                "isotropic",
                "anisotropic"
            ],
            value="isotropic",
            doc="""*isotropic* ignores scale and voxel size is uniform. *anisotropic* accounts for voxel size
            specified within *NamesAndTypes*."""
        )

    def run(self, workspace):
        self.function = skimage.filters.gaussian

        super(GaussianFilter, self).run(workspace)
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if x.volumetric:
            if self.shape.value == "isotropic":
                y_data = skimage.filters.gaussian(x_data, sigma=self.sigma.value)
            else:
                spacing = numpy.divide(1,x.spacing)
                spacing = numpy.divide(spacing, spacing[-1])
                sigma = numpy.multiply(self.sigma.value, spacing)
                y_data = skimage.filters.gaussian(x_data, sigma=sigma)
        else:
            y_data = skimage.filters.gaussian(x_data, sigma=self.sigma.value)

        y = cellprofiler.image.Image(
            dimensions=dimensions,
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

    def settings(self):
        __settings__ = super(GaussianFilter, self).settings()

        return __settings__ + [
            self.sigma,
            self.shape
        ]

    def visible_settings(self):
        __settings__ = super(GaussianFilter, self).visible_settings()

        __settings__ = __settings__ + [
            self.sigma,
            self.shape
        ]

        #super(GaussianFilter, self).run(workspace)

        #x_name = self.x_name.value

        #images = workspace.image_set

        #x = images.get_image(x_name)

        #if x.volumetric:
         #   __settings__ = __settings__ + [
          #      self.shape
           # ]

        return __settings__