# coding=utf-8

"""

Find blobs in an image or volume. Blobs are bright on dark or dark on bright regions in an image. The output of this
module is a binary image of white circles or spheres centered around found blobs on a black background.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.color
import skimage.draw
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.morphology


class BlobDetection(cellprofiler.module.Module):
    category = "Feature Detection"

    module_name = "BlobDetection"

    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input",
            doc="""
            A image or volume to detect blobs in.
            """
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"BlobDetection",
            doc="""
            Name the output. The output of this module is a binary image, or volume, with a black (zero-valued)
            background and white (one-valued) circles, or spheres, centered at detected blobs. Each circle, or sphere,
            has diameter approximating blob size.
            """
        )

        self.operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Determinant of the Hessian (DoH)",
                u"Difference of Gaussians (DoG)",
                u"Laplacian of Gaussian (LoG)"
            ],
            u"Laplacian of Gaussian (LoG)",
            doc="""
            Select a method for detecting blobs.
            <ul>
                <li>
                    <i>Determinant of the Hessian (DoH)</i>: This is the fastest method of detecting blobs. The time to
                    detect a blob is independent of its size. Additionally, DoH can detect bright blobs on a dark
                    background or dark blobs on a bright background. Blobs under 3 pixels aren't accurately detected.
                </li>

                <li>
                    <i>Difference of Gaussians (DoG)</i>: This method is slower than DoH but faster than LoG. It works
                    well for detecting small blobs. This method is slow when detecting large blobs. DoG can only detect
                    bright blobs on a dark background.
                </li>

                <li>
                    <i>Laplacian of Gaussian (LoG)</i> (default): The slowest but most accurate approach. This method
                    is very slow when detecting large blobs. LoG can only detect bright blobs on a dark background.
                </li>
            </ul>
            """
        )

        self.minimum = cellprofiler.setting.Float(
            u"Minimum",
            1.0,
            doc="""
            Set this to a low value to detect smaller blobs. When the Operation is "Difference of Gaussians (DoG)" or
            "Laplacian of Gaussian (LoG)" the radius of the smallest detected blob will be approximately <i>sqrt(2) *
            Minimum</i> pixels. When the Operation is "Determinant of the Hessian (DoH)" the radius of the smallest
            blob is approximately <i>Minimum</i>.
            """
        )

        self.maximum = cellprofiler.setting.Float(
            u"Maximum",
            50.0,
            doc="""
            Set this to a high value to detect larger blobs. When the Operation is "Difference of Gaussians (DoG)" or
            "Laplacian of Gaussian (LoG)" the radius of the largest detected blob will be approximately <i>sqrt(2) *
            Maximum</i> pixels. When the Operation is "Determinant of the Hessian (DoH)" the radius of the largest
            blob is approximately <i>Maximum</i>.
            """
        )

        self.ratio = cellprofiler.setting.Float(
            u"Ratio",
            1.6,
            doc="""
            Step size for values between "Minimum" and "Maximum".
            """
        )

        self.count = cellprofiler.setting.Integer(
            u"Count",
            10,
            doc="""
            Number of intermediate values between "Minimum" and "Maximum".
            """
        )

        self.threshold = cellprofiler.setting.Float(
            u"Threshold",
            0.2,
            doc="""
            The absolute lower bound for intensity of detectable blobs. Regions with intensity less than this value are
            ignored. Reduce this value to ignore dimmer blobs.
            """
        )

        self.overlap = cellprofiler.setting.Float(
            u"Overlap",
            0.5,
            minval=0.0,
            maxval=1.0,
            doc="""
            If the area of two blobs overlaps by a fraction greater than this value, the smaller blob is removed.
            """
        )

        self.scale = cellprofiler.setting.Choice(
            u"Scale",
            [
                u"Linear interpolation",
                u"Logarithm"
            ],
            u"Linear interpolation",
            doc="""
            Select the method for determining intermediate values between "Minimum" and "Maximum":
            <ul>
                <li>
                    <i>Linear interpolation</i> (default): Choose "Count" number of intermediate values using linear
                    interpolation.
                </li>

                <li>
                    <i>Logarithm</i>: Choose "Count" number of intermediate values using a base 10 logarithmic scale.
                </li>
            </ul>
            """
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.operation,
            self.minimum,
            self.maximum,
            self.ratio,
            self.count,
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
            self.maximum
        ]

        if self.operation.value in [u"Determinant of the Hessian (DoH)", u"Laplacian of Gaussian (LoG)"]:
            settings = settings + [
                self.count
            ]

        if self.operation.value == u"Difference of Gaussians (DoG)":
            settings = settings + [
                self.ratio
            ]

        settings = settings + [
            self.threshold,
            self.overlap
        ]

        if self.operation.value in [u"Determinant of the Hessian (DoH)", u"Laplacian of Gaussian (LoG)"]:
            settings = settings + [
                self.scale
            ]

        return settings

    def on_setting_changed(self, setting, pipeline):
        if not setting == self.operation:
            return

        if setting.value == u"Determinant of the Hessian (DoH)":
            self.minimum.value = 1.0

            self.maximum.value = 30.0

            self.count.value = 10

            self.threshold.value = 0.01

            self.overlap.value = 0.5

            self.scale.value = u"Linear interpolation"

        if setting.value == u"Difference of Gaussians (DoG)":
            self.minimum.value = 1.0

            self.maximum.value = 50.0

            self.ratio.value = 1.6

            self.threshold.value = 2.0

            self.overlap.value = 0.5

        if setting.value == u"Laplacian of Gaussian (LoG)":
            self.minimum.value = 1.0

            self.maximum.value = 50.0

            self.count.value = 10

            self.threshold.value = 0.2

            self.overlap.value = 0.5

            self.scale.value = u"Linear interpolation"

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        if x.multichannel:
            x_data = skimage.color.rgb2gray(x_data)

        dimensions = x.dimensions

        if dimensions == 2:
            blobs = self.__detect_blobs(x_data)

            y_data = self.__draw_circles(blobs, x_data.shape)
        else:
            y_data = numpy.zeros_like(x_data)

            for z, plane in enumerate(x_data):
                blobs = self.__detect_blobs(plane)

                y_data[z] = self.__draw_circles(blobs, plane.shape)

        y = cellprofiler.image.Image(
            image=y_data,
            dimensions=dimensions,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

            workspace.display_data.colormap = None if x.multichannel else "gray"

    def __detect_blobs(self, data):
        operation = self.operation.value

        if operation == u"Difference of Gaussians (DoG)":
            return skimage.feature.blob_dog(
                data,
                min_sigma=self.minimum.value,
                max_sigma=self.maximum.value,
                sigma_ratio=self.ratio.value,
                threshold=self.threshold.value,
                overlap=self.overlap.value
            )

        log_scale = True if self.scale.value == u"Logarithm" else False

        if operation == u"Determinant of the Hessian (DoH)":
            return skimage.feature.blob_doh(
                data,
                min_sigma=self.minimum.value,
                max_sigma=self.maximum.value,
                num_sigma=self.count.value,
                threshold=self.threshold.value,
                overlap=self.overlap.value,
                log_scale=log_scale
            )

        if operation == u"Laplacian of Gaussian (LoG)":
            return skimage.feature.blob_log(
                data,
                min_sigma=self.minimum.value,
                max_sigma=self.maximum.value,
                num_sigma=self.count.value,
                threshold=self.threshold.value,
                overlap=self.overlap.value,
                log_scale=log_scale
            )

    def __draw_circles(self, blobs, shape):
        result = numpy.zeros(shape)

        if blobs.size == 0:
            return result

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < result.shape[0],
                    cc < result.shape[1]
                ],
                axis=0
            )

            result[rr[in_bounds], cc[in_bounds]] = 1

        return result

    def display(self, workspace, figure):
        x_data = workspace.display_data.x_data

        y_data = workspace.display_data.y_data

        dimensions = workspace.display_data.dimensions

        colormap = workspace.display_data.colormap

        if dimensions == 2:
            overlay = skimage.color.label2rgb(
                y_data,
                image=x_data,
                bg_label=0
            )
        else:
            overlay = numpy.zeros(y_data.shape + (3,))

            for z, plane in enumerate(y_data):
                overlay[z] = skimage.color.label2rgb(
                    plane,
                    image=x_data[z],
                    bg_label=0
                )

        figure.set_subplots((3, 1), dimensions=dimensions)

        figure.subplot_imshow(0, 0, x_data, colormap=colormap, dimensions=dimensions)

        figure.subplot_imshow(1, 0, overlay, dimensions=dimensions)

        figure.subplot_imshow(2, 0, y_data, colormap=colormap, dimensions=dimensions)

    def volumetric(self):
        return True
