# coding=utf-8

"""

Thresholding

Thresholding is used to create a binary image from a grayscale image. The simplest thresholding methods replace in an
image with a black pixel if the image intensity is less than some fixed constant T, or a white pixel if the image
intensity is greater than that constant.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.filters


class Thresholding(cellprofiler.module.Module):
    category = "Image Processing"
    module_name = "Thresholding"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"Thresholding"
        )

        # TODO: Make this friendly. :)
        self.operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Adaptive",
                u"Iterative selection thresholding",
                u"Manual",
                u"Minimum cross entropy thresholding",
                u"Otsu’s method",
                u"Yen’s method"
            ],
            doc="""Select the thresholding operation.
            <ul>
                <li>
                    <i>Adaptive</i>: Applies an adaptive threshold. Also known as local or dynamic thresholding where
                    the threshold value is the weighted mean for the local neighborhood of a pixel subtracted by a
                    constant. Available methods for adaptive threshold are: "gaussian", "mean", or "median". See help
                    for "Adaptive method" for descriptions of these options.
                </li>
                <br>
                <li>
                    <i>Iterative selection thresholding</i>: Histogram-based threshold, known as Ridler-Calvard method
                    or inter-means. Threshold values returned satisfy the following equality:
                    <br>
                    <i>threshold = (image[image <= threshold].mean() + image[image > threshold].mean()) / 2.0</i>
                    <br>
                    <br>
                    That is, returned thresholds are intensities that separate the image into two groups of pixels,
                    where the threshold intensity is midway between the mean intensities of these groups.
                    <br>
                    <br>
                    For integer images, the above equality holds to within one; for floating- point images, the
                    equality holds to within the histogram bin-width.
                </li>
                <br>
                <li>
                    <i>Manual</i>: Threshold the image by assigning a white pixels to the pixels with intensity in range
                    [lower, upper], or a black pixel to pixels with intensity outside that range.
                </li>
                <br>
                <li>
                    <i>Minimum cross entropy thresholding</i>: Li’s Minimum Cross Entropy method. All pixels with an
                    intensity higher than this value are assumed to be foreground
                </li>
                <br>
                <li>
                    <i>Otsu's method</i>: Threshold using Otsu's method. The algorithm assumes that the image contains
                    two classes of pixels following bi-modal histogram (foreground pixels and background pixels), it
                    then calculates the optimum threshold separating the two classes so that their combined spread
                    (intra-class variance) is minimal, or equivalently (because the sum of pairwise squared distances
                    is constant), so that their inter-class variance is maximal.
                </li>
                <br>
                <li>
                    <i>Yen's method</i>: Threshold using Yen's thresholding method from "A new criterion for automatic
                    multilevel thresholding". Yen's method considers the differences between the original image and
                    the thresholded image and the number of bits required to represent the thresholded image to
                    determine the threshold value.
                </li>
            </ul>
            """
        )

        self.block_size = cellprofiler.setting.OddInteger(
            u"Block size",
            value=3,
            doc="Odd size of pixel neighborhood which is used to calculate the threshold value."
        )

        self.adaptive_method = cellprofiler.setting.Choice(
            u"Adaptive method",
            [
                u"Gaussian",
                u"Mean",
                u"Median"
            ],
            u"Gaussian",
            doc="""Method used to determine adaptive threshold for local neighbourhood in weighted mean image.
            <ul>
                <li>
                    <i>Gaussian</i>: Apply Gaussian filter.
                </li>
                <br>
                <li>
                    <i>Mean</i>: Apply arithmetic mean filter.
                </li>
                <br>
                <li>
                    <i>Median</i>: Apply median rank filter.
                </li>
            </ul>
            """
        )

        self.sigma = cellprofiler.setting.Float(
            u"Sigma",
            1.0,
            minval=0.0,
            doc="Sigma used to compute Gaussian."
        )

        self.offset = cellprofiler.setting.Float(
            u"Offset",
            value=0.0,
            doc="Constant subtracted from weighted mean of neighborhood to calculate the local threshold value."
        )

        self.bins = cellprofiler.setting.Integer(
            u"Bins",
            value=256,
            doc="""Number of bins used to calculate the histogram in automatic thresholding. Histogram shape-based
            methods in particular, but also many other thresholding algorithms, make certain assumptions about the
            image intensity probability distribution. The most common thresholding methods work on bimodal
            distributions, but algorithms have also been developed for unimodal distributions, multimodal
            distributions, and circular distributions."""
        )

        self.lower = cellprofiler.setting.Float(
            u"Lower",
            value=0.0,
            doc="""The minimum pixel intensity for thresholding. All pixels with intensities less than the minimum are
            assigned a black pixel. All pixels with intensities above the minimum (but below the maximum) are assigned
            a white pixel."""
        )

        self.upper = cellprofiler.setting.Float(
            u"Upper",
            value=1.0,
            doc="""The maximum pixel intensity for thresholding. All pixels with intensities greater than the minimum are
            assigned a black pixel. All pixels with intensities above the maximum (but above the minimum) are assigned
            a white pixel."""
        )

    def settings(self):
        return [
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
        settings = [
            self.x_name,
            self.y_name,
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

        if self.operation.value == u"Iterative selection thresholding":
            settings = settings + [
                self.bins
            ]

        if self.operation.value == u"Manual":
            settings = settings + [
                self.lower,
                self.upper
            ]

        if self.operation.value == u"Otsu’s method":
            settings = settings + [
                self.bins
            ]

        if self.operation.value == u"Yen’s method":
            settings = settings + [
                self.bins
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

        for z, image in enumerate(x_data):
            if self.operation.value == u"Adaptive":
                y_data[z] = skimage.filters.threshold_adaptive(
                    image=image,
                    block_size=self.block_size.value,
                    method=self.adaptive_method.value.lower(),
                    offset=self.offset.value,
                    param=self.sigma.value
                )
            elif self.operation.value == u"Iterative selection thresholding":
                y_data[z] = skimage.filters.threshold_isodata(
                    image=image,
                    nbins=self.bins.value
                )

                y_data[z] = image >= y_data[z]
            elif self.operation.value == u"Manual":
                y_data[z] = image > self.lower

                y_data[z] = image < self.upper
            elif self.operation.value == u"Minimum cross entropy thresholding":
                y_data[z] = skimage.filters.threshold_li(
                    image=image
                )

                y_data[z] = image >= y_data[z]
            elif self.operation.value == u"Otsu’s method":
                y_data[z] = skimage.filters.threshold_otsu(
                    image=image,
                    nbins=self.bins.value
                )

                y_data[z] = image >= y_data[z]
            elif self.operation.value == u"Yen’s method":
                y_data[z] = skimage.filters.threshold_yen(
                    image=image,
                    nbins=self.bins.value
                )

                y_data[z] = image >= y_data[z]

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

    def display(self, workspace, figure):
        figure.set_subplots((1, 2), dimensions=workspace.display_data.dimensions)

        figure.gridshow(0, 0, workspace.display_data.x_data, dimensions=workspace.display_data.dimensions)

        figure.gridshow(0, 1, workspace.display_data.y_data, dimensions=workspace.display_data.dimensions)
