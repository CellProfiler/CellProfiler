"""
Smooth
======

**Smooth** smooths (i.e., blurs) images.

This module allows you to smooth (blur) images, which can be helpful to
remove small artifacts. Note that smoothing can be a time-consuming process.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also several related modules in the *Advanced* category (e.g.,
**MedianFilter** and **GaussianFilter**).
"""

import numpy
import scipy.ndimage
import skimage.restoration
from cellprofiler_core.constants.module import (
    HELP_ON_MEASURING_DISTANCES,
    HELP_ON_PIXEL_INTENSITIES,
)
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Float
from centrosome.filter import median_filter, circular_average_filter
from centrosome.smooth import fit_polynomial
from centrosome.smooth import smooth_with_function_and_mask

FIT_POLYNOMIAL = "Fit Polynomial"
MEDIAN_FILTER = "Median Filter"
GAUSSIAN_FILTER = "Gaussian Filter"
SMOOTH_KEEPING_EDGES = "Smooth Keeping Edges"
CIRCULAR_AVERAGE_FILTER = "Circular Average Filter"
SM_TO_AVERAGE = "Smooth to Average"


class Smooth(Module):
    module_name = "Smooth"
    category = "Image Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="""Select the image to be smoothed.""",
        )

        self.filtered_image_name = ImageName(
            "Name the output image",
            "FilteredImage",
            doc="""Enter a name for the resulting image.""",
        )

        self.smoothing_method = Choice(
            "Select smoothing method",
            [
                FIT_POLYNOMIAL,
                GAUSSIAN_FILTER,
                MEDIAN_FILTER,
                SMOOTH_KEEPING_EDGES,
                CIRCULAR_AVERAGE_FILTER,
                SM_TO_AVERAGE,
            ],
            doc="""\
This module smooths images using one of several filters. Fitting a
polynomial is fastest but does not allow a very tight fit compared to
the other methods:

-  *%(FIT_POLYNOMIAL)s:* This method is fastest but does not allow
   a very tight “fit” compared to the other methods. Thus, it will usually be less
   accurate. The method treats the intensity of the image
   pixels as a polynomial function of the x and y position of each
   pixel. It fits the intensity to the polynomial, *A x* :sup:`2` *+ B
   y* :sup:`2` *+ C xy + D x + E y + F*. This will produce a smoothed
   image with a single peak or trough of intensity that tapers off
   elsewhere in the image. For many microscopy images (where the
   illumination of the lamp is brightest in the center of field of
   view), this method will produce an image with a bright central region
   and dimmer edges. But, in some cases the peak/trough of the
   polynomial may actually occur outside of the image itself.
-  *%(GAUSSIAN_FILTER)s:* This method convolves the image with a
   Gaussian whose full width at half maximum is the artifact diameter
   entered. Its effect is to blur and obscure features smaller than the
   specified diameter and spread bright or dim features larger than the
   specified diameter.
-  *%(MEDIAN_FILTER)s:* This method finds the median pixel value within
   the diameter you specify. It removes bright or dim features
   that are significantly smaller than the specified diameter.
-  *%(SMOOTH_KEEPING_EDGES)s:* This method uses a bilateral filter
   which limits Gaussian smoothing across an edge while applying
   smoothing perpendicular to an edge. The effect is to respect edges in
   an image while smoothing other features. *%(SMOOTH_KEEPING_EDGES)s*
   will filter an image with reasonable speed for artifact diameters
   greater than 10 and for intensity differences greater than 0.1. The
   algorithm will consume more memory and operate more slowly as you
   lower these numbers.
-  *%(CIRCULAR_AVERAGE_FILTER)s:* This method convolves the image with
   a uniform circular averaging filter whose size is the artifact
   diameter entered. This filter is useful for re-creating an
   out-of-focus blur to an image.
-  *%(SM_TO_AVERAGE)s:* Creates a flat, smooth image where every pixel
   of the image equals the average value of the original image.

*Note, when deciding between %(MEDIAN_FILTER)s and %(GAUSSIAN_FILTER)s
we typically recommend
%(MEDIAN_FILTER)s over %(GAUSSIAN_FILTER)s because the
median is less sensitive to outliers, although the results are also
slightly less smooth and the fact that images are in the range of 0
to 1 means that outliers typically will not dominate too strongly
anyway.*
"""
            % globals(),
        )

        self.wants_automatic_object_size = Binary(
            "Calculate artifact diameter automatically?",
            True,
            doc="""\
*(Used only if “%(GAUSSIAN_FILTER)s”, “%(MEDIAN_FILTER)s”, “%(SMOOTH_KEEPING_EDGES)s” or “%(CIRCULAR_AVERAGE_FILTER)s” is selected)*

Select *Yes* to choose an artifact diameter based on the size of
the image. The minimum size it will choose is 30 pixels, otherwise the
size is 1/40 of the size of the image.

Select *No* to manually enter an artifact diameter.
"""
            % globals(),
        )

        self.object_size = Float(
            "Typical artifact diameter",
            16.0,
            doc="""\
*(Used only if choosing the artifact diameter automatically is set to
“No”)*

Enter the approximate diameter (in pixels) of the features to be blurred
by the smoothing algorithm. This value is used to calculate the size of
the spatial filter. {} For most
smoothing methods, selecting a diameter over ~50 will take substantial
amounts of time to process.
""".format(
                HELP_ON_MEASURING_DISTANCES
            ),
        )

        self.sigma_range = Float(
            "Edge intensity difference",
            0.1,
            doc="""\
*(Used only if “{smooth_help}” is selected)*

Enter the intensity step (which indicates an edge in an image) that you
want to preserve. Edges are locations where the intensity changes
precipitously, so this setting is used to adjust the rough magnitude of
these changes. A lower number will preserve weaker edges. A higher
number will preserve only stronger edges. Values should be between zero
and one. {pixel_help}
""".format(
                smooth_help=SMOOTH_KEEPING_EDGES, pixel_help=HELP_ON_PIXEL_INTENSITIES
            ),
        )

        self.clip = Binary(
            "Clip intensities to 0 and 1?",
            True,
            doc="""\
*(Used only if "{fit}" is selected)*

The *{fit}* method is the only smoothing option that can
yield an output image whose values are outside of the values of the
input image. This setting controls whether to limit the image
intensity to the 0 - 1 range used by CellProfiler.

Select *Yes* to set all output image pixels less than zero to zero
and all pixels greater than one to one.

Select *No* to allow values less than zero and greater than one in
the output image.
""".format(
                fit=FIT_POLYNOMIAL
            ),
        )

    def settings(self):
        return [
            self.image_name,
            self.filtered_image_name,
            self.smoothing_method,
            self.wants_automatic_object_size,
            self.object_size,
            self.sigma_range,
            self.clip,
        ]

    def visible_settings(self):
        result = [self.image_name, self.filtered_image_name, self.smoothing_method]
        if self.smoothing_method.value not in [FIT_POLYNOMIAL, SM_TO_AVERAGE]:
            result.append(self.wants_automatic_object_size)
            if not self.wants_automatic_object_size.value:
                result.append(self.object_size)
            if self.smoothing_method.value == SMOOTH_KEEPING_EDGES:
                result.append(self.sigma_range)
        if self.smoothing_method.value == FIT_POLYNOMIAL:
            result.append(self.clip)
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(
            self.image_name.value, must_be_grayscale=True
        )
        pixel_data = image.pixel_data
        if self.wants_automatic_object_size.value:
            object_size = min(30, max(1, numpy.mean(pixel_data.shape) / 40))
        else:
            object_size = float(self.object_size.value)
        sigma = object_size / 2.35
        if self.smoothing_method.value == GAUSSIAN_FILTER:

            def fn(image):
                return scipy.ndimage.gaussian_filter(
                    image, sigma, mode="constant", cval=0
                )

            output_pixels = smooth_with_function_and_mask(pixel_data, fn, image.mask)
        elif self.smoothing_method.value == MEDIAN_FILTER:
            output_pixels = median_filter(pixel_data, image.mask, object_size / 2 + 1)
        elif self.smoothing_method.value == SMOOTH_KEEPING_EDGES:
            sigma_range = float(self.sigma_range.value)

            output_pixels = skimage.restoration.denoise_bilateral(
                image=pixel_data.astype(float),
                channel_axis=2 if image.multichannel else None,
                sigma_color=sigma_range,
                sigma_spatial=sigma,
            )
        elif self.smoothing_method.value == FIT_POLYNOMIAL:
            output_pixels = fit_polynomial(pixel_data, image.mask, self.clip.value)
        elif self.smoothing_method.value == CIRCULAR_AVERAGE_FILTER:
            output_pixels = circular_average_filter(
                pixel_data, object_size / 2 + 1, image.mask
            )
        elif self.smoothing_method.value == SM_TO_AVERAGE:
            if image.has_mask:
                mean = numpy.mean(pixel_data[image.mask])
            else:
                mean = numpy.mean(pixel_data)
            output_pixels = numpy.ones(pixel_data.shape, pixel_data.dtype) * mean
        else:
            raise ValueError(
                "Unsupported smoothing method: %s" % self.smoothing_method.value
            )
        output_image = Image(output_pixels, parent_image=image)
        workspace.image_set.add(self.filtered_image_name.value, output_image)
        workspace.display_data.pixel_data = pixel_data
        workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))
        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.pixel_data,
            "Original: %s" % self.image_name.value,
        )
        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.output_pixels,
            "Filtered: %s" % self.filtered_image_name.value,
            sharexy=figure.subplot(0, 0),
        )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values = setting_values + ["Yes"]
            variable_revision_number = 2
        return setting_values, variable_revision_number
