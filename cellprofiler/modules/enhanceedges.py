# coding=utf-8

"""
EnhanceEdges
============

**EnhanceEdges** enhances or identifies edges in an image, which can
improve object identification or other downstream image processing.

This module enhances the edges (gradients - places where pixel
intensities change dramatically) in a grayscale image. All
methods other than Canny produce a grayscale image that can be used in
an **Identify** module or thresholded using the **Threshold**
module to produce a binary (black/white) mask of edges. The Canny
algorithm produces a binary (black/white) mask image consisting of the
edge pixels.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

"""

import numpy as np
from centrosome.filter import laplacian_of_gaussian
from centrosome.filter import prewitt, hprewitt, vprewitt, stretch
from centrosome.filter import roberts, canny, sobel, hsobel, vsobel
from centrosome.kirsch import kirsch
from centrosome.otsu import otsu3
from scipy.ndimage import convolve

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO

M_SOBEL = "Sobel"
M_PREWITT = "Prewitt"
M_ROBERTS = "Roberts"
M_LOG = "LoG"
M_CANNY = "Canny"
M_KIRSCH = "Kirsch"

O_BINARY = "Binary"
O_GRAYSCALE = "Grayscale"

E_ALL = "All"
E_HORIZONTAL = "Horizontal"
E_VERTICAL = "Vertical"


class EnhanceEdges(cpm.Module):
    module_name = "EnhanceEdges"
    category = "Image Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc='''Select the image whose edges you want to enhance.''')

        self.output_image_name = cps.ImageNameProvider(
                "Name the output image", "EdgedImage", doc='''Enter a name for the resulting image with edges enhanced.''')

        self.method = cps.Choice(
                "Select an edge-finding method",
                [M_SOBEL, M_PREWITT, M_ROBERTS, M_LOG, M_CANNY, M_KIRSCH], doc='''\
There are several methods that can be used to enhance edges. Often, it
is best to test them against each other empirically:

-  *%(M_SOBEL)s:* Finds edges using the %(M_SOBEL)s approximation to
   the derivative. The %(M_SOBEL)s method derives a horizontal and
   vertical gradient measure and returns the square-root of the sum of
   the two squared signals.
-  *%(M_PREWITT)s:* Finds edges using the %(M_PREWITT)s approximation
   to the derivative. It returns edges at those points where the
   gradient of the image is maximum.
-  *%(M_ROBERTS)s:* Finds edges using the Roberts approximation to the
   derivative. The %(M_ROBERTS)s method looks for gradients in the
   diagonal and anti-diagonal directions and returns the square-root of
   the sum of the two squared signals. This method is fast, but it
   creates diagonal artifacts that may need to be removed by smoothing.
-  *%(M_LOG)s:* Applies a Laplacian of Gaussian filter to the image and
   finds zero crossings.
-  *%(M_CANNY)s:* Finds edges by looking for local maxima of the
   gradient of the image. The gradient is calculated using the
   derivative of a Gaussian filter. The method uses two thresholds to
   detect strong and weak edges, and includes the weak edges in the
   output only if they are connected to strong edges. This method is
   therefore less likely than the others to be fooled by noise, and more
   likely to detect true weak edges.
-  *%(M_KIRSCH)s:* Finds edges by calculating the gradient among the 8
   compass points (North, North-east, etc.) and selecting the maximum as
   the pixel’s value.
''' % globals())

        self.wants_automatic_threshold = cps.Binary(
                "Automatically calculate the threshold?", True, doc='''\
*(Used only with the "%(M_CANNY)s" option and automatic thresholding)*

Select *%(YES)s* to automatically calculate the threshold using a
three-category Otsu algorithm performed on the Sobel transform of the
image.

Select *%(NO)s* to manually enter the threshold value.
''' % globals())

        self.manual_threshold = cps.Float(
                "Absolute threshold", 0.2, 0, 1, doc='''\
*(Used only with the "%(M_CANNY)s" option and manual thresholding)*

The upper cutoff for Canny edges. All Sobel-transformed pixels with this
value or higher will be marked as an edge. You can enter a threshold
between 0 and 1.
''' % globals())

        self.threshold_adjustment_factor = cps.Float(
                "Threshold adjustment factor", 1, doc='''\
*(Used only with the "%(M_CANNY)s" option and automatic thresholding)*

This threshold adjustment factor is a multiplier that is applied to both
the lower and upper Canny thresholds if they are calculated
automatically. An adjustment factor of 1 indicates no adjustment. The
adjustment factor has no effect on any threshhold entered manually.
''' % globals())

        self.direction = cps.Choice(
                "Select edge direction to enhance",
                [E_ALL, E_HORIZONTAL, E_VERTICAL], doc='''\
*(Used only with "%(M_PREWITT)s" and "%(M_SOBEL)s" methods)*

Select the direction of the edges you aim to identify in the image
(predominantly horizontal, predominantly vertical, or both).
''' % globals())

        self.wants_automatic_sigma = cps.Binary("Calculate Gaussian's sigma automatically?", True, doc="""\
Select *%(YES)s* to automatically calculate the Gaussian's sigma.

Select *%(NO)s* to manually enter the value.
""" % globals())

        self.sigma = cps.Float("Gaussian's sigma value", 10, doc="""Set a value for Gaussian's sigma.""")

        self.wants_automatic_low_threshold = cps.Binary(
                "Calculate value for low threshold automatically?", True, doc="""\
*(Used only with the "%(M_CANNY)s" option and automatic thresholding)*

Select *%(YES)s* to automatically calculate the low / soft threshold
cutoff for the %(M_CANNY)s method.

Select *%(NO)s* to manually enter the low threshold value.
""" % globals())

        self.low_threshold = cps.Float(
                "Low threshold value", 0.1, 0, 1, doc="""\
*(Used only with the "%(M_CANNY)s" option and manual thresholding)*

Enter the soft threshold cutoff for the %(M_CANNY)s method. The
%(M_CANNY)s method will mark all %(M_SOBEL)s-transformed pixels with
values below this threshold as not being edges.
""" % globals())

    def settings(self):
        return [self.image_name, self.output_image_name,
                self.wants_automatic_threshold, self.manual_threshold,
                self.threshold_adjustment_factor, self.method,
                self.direction, self.wants_automatic_sigma, self.sigma,
                self.wants_automatic_low_threshold, self.low_threshold]

    def help_settings(self):
        return [
            self.image_name,
            self.output_image_name,
            self.method,
            self.direction,
            self.wants_automatic_sigma,
            self.sigma,
            self.wants_automatic_threshold,
            self.manual_threshold,
            self.threshold_adjustment_factor,
            self.wants_automatic_low_threshold,
            self.low_threshold
        ]

    def visible_settings(self):
        settings = [self.image_name, self.output_image_name]
        settings += [self.method]
        if self.method in (M_SOBEL, M_PREWITT):
            settings += [self.direction]
        if self.method in (M_LOG, M_CANNY):
            settings += [self.wants_automatic_sigma]
            if not self.wants_automatic_sigma.value:
                settings += [self.sigma]
        if self.method == M_CANNY:
            settings += [self.wants_automatic_threshold]
            if not self.wants_automatic_threshold.value:
                settings += [self.manual_threshold]
            settings += [self.wants_automatic_low_threshold]
            if not self.wants_automatic_low_threshold.value:
                settings += [self.low_threshold]
            if (self.wants_automatic_threshold or
                    self.wants_automatic_low_threshold):
                settings += [self.threshold_adjustment_factor]
        return settings

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        orig_pixels = image.pixel_data
        if image.has_mask:
            mask = image.mask
        else:
            mask = np.ones(orig_pixels.shape, bool)
        if self.method == M_SOBEL:
            if self.direction == E_ALL:
                output_pixels = sobel(orig_pixels, mask)
            elif self.direction == E_HORIZONTAL:
                output_pixels = hsobel(orig_pixels, mask)
            elif self.direction == E_VERTICAL:
                output_pixels = vsobel(orig_pixels, mask)
            else:
                raise NotImplementedError("Unimplemented direction for Sobel: %s", self.direction.value)
        elif self.method == M_LOG:
            sigma = self.get_sigma()
            size = int(sigma * 4) + 1
            output_pixels = laplacian_of_gaussian(orig_pixels, mask, size, sigma)
        elif self.method == M_PREWITT:
            if self.direction == E_ALL:
                output_pixels = prewitt(orig_pixels)
            elif self.direction == E_HORIZONTAL:
                output_pixels = hprewitt(orig_pixels, mask)
            elif self.direction == E_VERTICAL:
                output_pixels = vprewitt(orig_pixels, mask)
            else:
                raise NotImplementedError("Unimplemented direction for Prewitt: %s", self.direction.value)
        elif self.method == M_CANNY:
            high_threshold = self.manual_threshold.value
            low_threshold = self.low_threshold.value
            if (self.wants_automatic_low_threshold.value or
                    self.wants_automatic_threshold.value):
                sobel_image = sobel(orig_pixels, mask)
                low, high = otsu3(sobel_image[mask])
                if self.wants_automatic_low_threshold.value:
                    low_threshold = low * self.threshold_adjustment_factor.value
                if self.wants_automatic_threshold.value:
                    high_threshold = high * self.threshold_adjustment_factor.value
            output_pixels = canny(orig_pixels, mask, self.get_sigma(),
                                  low_threshold,
                                  high_threshold)
        elif self.method == M_ROBERTS:
            output_pixels = roberts(orig_pixels, mask)
        elif self.method == M_KIRSCH:
            output_pixels = kirsch(orig_pixels)
        else:
            raise NotImplementedError("Unimplemented edge detection method: %s" %
                                      self.method.value)

        output_image = cpi.Image(output_pixels, parent_image=image)
        workspace.image_set.add(self.output_image_name.value, output_image)

        if self.show_window:
            workspace.display_data.orig_pixels = orig_pixels
            workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        orig_pixels = workspace.display_data.orig_pixels
        output_pixels = workspace.display_data.output_pixels

        figure.set_subplots((2, 2))
        figure.subplot_imshow_grayscale(0, 0, orig_pixels,
                                        "Original: %s" %
                                        self.image_name.value)
        if self.method == M_CANNY:
            # Canny is binary
            figure.subplot_imshow_bw(0, 1, output_pixels,
                                     self.output_image_name.value,
                                     sharexy=figure.subplot(0, 0))
        else:
            figure.subplot_imshow_grayscale(0, 1, output_pixels,
                                            self.output_image_name.value,
                                            sharexy=figure.subplot(0, 0))
        color_image = np.zeros((output_pixels.shape[0],
                                output_pixels.shape[1], 3))
        color_image[:, :, 0] = stretch(orig_pixels)
        color_image[:, :, 1] = stretch(output_pixels)
        figure.subplot_imshow(1, 0, color_image, "Composite image",
                              sharexy=figure.subplot(0, 0))

    def get_sigma(self):
        if self.wants_automatic_sigma.value:
            #
            # Constants here taken from FindEdges.m
            #
            if self.method == M_CANNY:
                return 1.0
            elif self.method == M_LOG:
                return 2.0
            else:
                raise NotImplementedError("Automatic sigma not supported for method %s." % self.method.value)
        else:
            return self.sigma.value

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 3:
            setting_values = [
                setting_values[0],  # ImageName
                setting_values[1],  # OutputName
                setting_values[2] == cps.DO_NOT_USE,  # Threshold
                setting_values[2]
                if setting_values[2] != cps.DO_NOT_USE
                else .5,
                setting_values[3],  # Threshold adjustment factor
                setting_values[4],  # Method
                setting_values[5],  # Filter size
                setting_values[8],  # Direction
                setting_values[9] == cps.DO_NOT_USE,  # Sigma
                setting_values[9]
                if setting_values[9] != cps.DO_NOT_USE
                else 5,
                setting_values[10] == cps.DO_NOT_USE,  # Low threshold
                setting_values[10]
                if setting_values[10] != cps.DO_NOT_USE
                else .5]
            from_matlab = False
            variable_revision_number = 1

        if from_matlab == False and variable_revision_number == 1:
            # Ratio removed / filter size removed
            setting_values = setting_values[:6] + setting_values[7:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab


FindEdges = EnhanceEdges
