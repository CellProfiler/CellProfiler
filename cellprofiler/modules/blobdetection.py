# coding=utf-8

"""

Blob detection

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.draw
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.morphology


class BlobDetection(cellprofiler.module.Module):
    category = "Feature detection"
    module_name = "Blob detection"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"OutputImage"
        )

        self.operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Determinant of the Hessian (DoH)",
                u"Difference of Gaussians (DoG)",
                u"Laplacian of Gaussian (LoG)"
            ],
            u"Laplacian of Gaussian (LoG)"
        )

        self.minimum = cellprofiler.setting.Float(
            u"Minimum",
            1.0
        )

        self.maximum = cellprofiler.setting.Float(
            u"Maximum",
            50.0
        )

        self.ratio = cellprofiler.setting.Float(
            u"Ratio",
            1.6
        )

        self.threshold = cellprofiler.setting.Float(
            u"Threshold",
            2.0
        )

        self.overlap = cellprofiler.setting.Float(
            u"Overlap",
            0.5
        )

        self.scale = cellprofiler.setting.Choice(
            u"Scale",
            [
                u"Linear interpolation",
                u"Logarithm"
            ]
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.operation,
            self.minimum,
            self.maximum,
            self.ratio,
            self.threshold,
            self.overlap,
            self.scale
        ]

    def visible_settings(self):
        settings = [
            self.x_name,
            self.y_name,
            self.operation,
            self.minimum,
            self.maximum,
            self.ratio,
            self.threshold,
            self.overlap
        ]

        if self.operation.value in [u"Determinant of the Hessian (DoH)", u"Laplacian of Gaussian (LoG)"]:
            settings = settings + [
                self.scale
            ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        operation = self.operation.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        y_data = numpy.zeros_like(x_data)

        for z, image in enumerate(x_data):
            if operation == u"Determinant of the Hessian (DoH)":
                blobs = skimage.feature.blob_doh(x_data[z])

                blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

                for r, c, radius in blobs:
                    rr, cc = skimage.draw.circle(r, c, radius)

                    y_data[rr, cc, z] = 1
            elif operation == u"Difference of Gaussians (DoG)":
                blobs = skimage.feature.blob_dog(x_data[z])

                blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

                for r, c, radius in blobs:
                    rr, cc = skimage.draw.circle(r, c, radius)

                    y_data[rr, cc, z] = 1
            elif operation == u"Laplacian of Gaussian (LoG)":
                blobs = skimage.feature.blob_log(x_data[z])

                blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

                for r, c, radius in blobs:
                    rr, cc = skimage.draw.circle(r, c, radius)

                    y_data[rr, cc, z] = 1

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
