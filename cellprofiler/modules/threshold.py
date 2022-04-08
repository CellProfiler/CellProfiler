"""
Threshold
=========

**Threshold** produces a binary, or black and white, image based on a threshold that
can be pre-selected or calculated automatically using one of many
methods. After the threshold value has been determined, the **Threshold** module will
set pixel intensities below the value to zero (black) and above the value to one (white).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

import centrosome.smooth
import centrosome.threshold
import numpy
import scipy.interpolate
import scipy.ndimage
import skimage.filters
import skimage.filters.rank
import skimage.morphology
from cellprofiler_core.constants.measurement import (
    FF_WEIGHTED_VARIANCE,
    FF_FINAL_THRESHOLD,
    FF_ORIG_THRESHOLD,
    FF_GUIDE_THRESHOLD,
    FF_SUM_OF_ENTROPIES,
    COLTYPE_FLOAT,
    C_THRESHOLD,
    FTR_FINAL_THRESHOLD,
    FTR_ORIG_THRESHOLD,
    FTR_GUIDE_THRESHOLD,
    FTR_SUM_OF_ENTROPIES,
    FTR_WEIGHTED_VARIANCE,
)
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Measurement, ValidationError, Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.text import Float, Integer

from cellprofiler.modules import _help

O_TWO_CLASS = "Two classes"
O_THREE_CLASS = "Three classes"

O_FOREGROUND = "Foreground"
O_BACKGROUND = "Background"

RB_MEAN = "Mean"
RB_MEDIAN = "Median"
RB_MODE = "Mode"
RB_SD = "Standard deviation"
RB_MAD = "Median absolute deviation"

TS_GLOBAL = "Global"
TS_ADAPTIVE = "Adaptive"
TM_MANUAL = "Manual"
TM_MEASUREMENT = "Measurement"
TM_LI = "Minimum Cross-Entropy"
TM_OTSU = "Otsu"
TM_ROBUST_BACKGROUND = "Robust Background"
TM_SAUVOLA = "Sauvola"


TS_ALL = [TS_GLOBAL, TS_ADAPTIVE]

PROTIP_RECOMMEND_ICON = "thumb-up.png"
PROTIP_AVOID_ICON = "thumb-down.png"
TECH_NOTE_ICON = "gear.png"


class Threshold(ImageProcessing):
    module_name = "Threshold"

    variable_revision_number = 12

    def create_settings(self):
        super(Threshold, self).create_settings()

        self.threshold_scope = Choice(
            "Threshold strategy",
            TS_ALL,
            value=TS_GLOBAL,
            doc="""\
The thresholding strategy determines the type of input that is used to
calculate the threshold. These options allow you to calculate a
threshold based on the whole image or based on image sub-regions.

The choices for the threshold strategy are:

-  *{TS_GLOBAL}:* Calculates a single threshold value based on the
   unmasked pixels of the input image and use that value to classify
   pixels above the threshold as foreground and below as background.

   |image0| This strategy is fast and robust, especially if the background is
   relatively uniform (for example, after illumination correction).

-  *{TS_ADAPTIVE}:* Calculates a different threshold for each pixel,
   thus adapting to changes in foreground/background intensities
   across the image. For each pixel, the threshold is calculated based
   on the pixels within a given neighborhood (or window) surrounding
   that pixel.

   |image1| This method is slower but can produce better results for
   non-uniform backgrounds. However, for significant illumination
   variation, using the **CorrectIllumination** modules is preferable.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "TS_ADAPTIVE": TS_ADAPTIVE,
                    "TS_GLOBAL": TS_GLOBAL,
                }
            ),
        )

        self.global_operation = Choice(
            "Thresholding method",
            [TM_LI, TM_OTSU, TM_ROBUST_BACKGROUND, TM_MEASUREMENT, TM_MANUAL,],
            value=TM_LI,
            doc="""\
*(Used only if "{TS_GLOBAL}" is selected for thresholding strategy)*

The intensity threshold affects the decision of whether each pixel
will be considered foreground (objects/region(s) of interest) or background. A
higher threshold value will result in only the brightest regions being
identified, whereas a lower threshold value will include dim regions.
You can have the threshold automatically calculated from a choice of
several methods, or you can enter a number manually between 0 and 1
for the threshold.

Both the automatic and manual options have advantages and disadvantages.

|image0| An automatically-calculated threshold adapts to changes in
lighting/staining conditions between images and is usually more
robust/accurate. In the vast majority of cases, an automatic method is
sufficient to achieve the desired thresholding, once the proper method
is selected. In contrast, an advantage of a manually-entered number is
that it treats every image identically, so use this option when you have
a good sense for what the threshold should be across all images. To help
determine the choice of threshold manually, you can inspect the pixel
intensities in an image of your choice.

{HELP_ON_PIXEL_INTENSITIES}

|image1| The manual method is not robust with regard to slight changes
in lighting/staining conditions between images. The automatic methods
may occasionally produce a poor threshold for unusual or artifactual
images. It also takes a small amount of time to calculate, which can add
to processing time for analysis runs on a large number of images.

The threshold that is used for each image is recorded as a per-image
measurement, so if you are surprised by unusual measurements from one of
your images, you might check whether the automatically calculated
threshold was unusually high or low compared to the other images. See
the **FlagImage** module if you would like to flag an image based on the
threshold value.

There are a number of methods for finding thresholds automatically:

-  *{TM_LI}:* The distributions of intensities that define foreground and background are
   used as estimates for probability distributions that produce the intensities of foreground
   and background pixels. For each possible threshold the cross-entropy between the foreground
   and background distributions is calculated and the lowest cross-entropy value is chosen as
   the final threshold. The lowest cross-entropy can be interpreted as the value where the information
   shared between the two probability distributions is the highest. On average, given a pixel of an
   arbitrary intensity, the likelihood it came from the foreground or background would be at its highest.

-  *{TM_OTSU}:* This approach calculates the threshold separating the
   two classes of pixels (foreground and background) by minimizing the
   variance within the each class.

   |image2| This method is a good
   initial approach if you do not know much about the image
   characteristics of all the images in your experiment, especially if
   the percentage of the image covered by foreground varies
   substantially from image to image.

   |image3| Our implementation of
   Otsu’s method allows for assigning the threshold value based on
   splitting the image into either two classes (foreground and
   background) or three classes (foreground, mid-level, and background).
   See the help below for more details.
   
   NOTE that CellProfiler 2 used a non-standard implementation of two-class Otsu
   thresholding; CellProfiler 3.0.0 and onward use the standard implementation. 
   While in most cases the calculated threshold is very similar, pipelines that 
   are adapted from CellProfiler 2 and use two-class Otsu thresholding should be 
   checked when converting to CellProfiler 3 and beyond to make sure that method
   is still the most appropriate. 
   
   NOTE that from CellProfiler 4.0.0 and onwards the standard implementation will
   be used for three-class Otsu thresholding as well. Results with three-class
   Otsu thresholding are likely to be slightly different from older versions, so
   imported pipelines which use these methods should be checked when converting
   to the latest version to ensure that settings are still appropriate.


-  *{TM_ROBUST_BACKGROUND}:* This method assumes that the background
   distribution approximates a Gaussian by trimming the brightest and
   dimmest X% of pixel intensities, where you choose a suitable percentage.
   It then calculates the mean and
   standard deviation of the remaining pixels and calculates the
   threshold as the mean + N times the standard deviation, where again you
   choose the number of standard deviations to suit your images.

   |image4| This thresholding method can be helpful if the majority of the image
   is background. It can also be helpful if your images vary in overall
   brightness, but the objects of interest are consistently *N* times
   brighter than the background level of the image.

-  *{TM_MEASUREMENT}:* Use a prior image measurement as the threshold.
   The measurement should have values between zero and one. This
   strategy can also be used to apply a pre-calculated threshold imported as
   per-image metadata.

-  *{TM_MANUAL}:* Enter a single value between zero and one that
   applies to all images and is thus independent of the input image.

   |image5|  This approach is useful if the input image has a stable or
   negligible background, or if the input image is the probability map
   output of a pixel-based classifier (in which case, a value of
   0.5 should be chosen). If the input image is already binary (i.e.,
   where the foreground is 1 and the background is 0), a manual value of
   0.5 will identify the objects.


**References**

-  Sezgin M, Sankur B (2004) “Survey over image thresholding techniques
   and quantitative performance evaluation.” *Journal of Electronic
   Imaging*, 13(1), 146-165. (`link`_)

.. _link: https://doi.org/10.1117/1.1631315
.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_AVOID_ICON}
.. |image2| image:: {PROTIP_RECOMMEND_ICON}
.. |image3| image:: {TECH_NOTE_ICON}
.. |image4| image:: {PROTIP_RECOMMEND_ICON}
.. |image5| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{
                    "HELP_ON_PIXEL_INTENSITIES": _help.HELP_ON_PIXEL_INTENSITIES,
                    "PROTIP_AVOID_ICON": _help.PROTIP_AVOID_ICON,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "TECH_NOTE_ICON": _help.TECH_NOTE_ICON,
                    "TM_LI": TM_LI,
                    "TM_OTSU": TM_OTSU,
                    "TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND,
                    "TM_MANUAL": TM_MANUAL,
                    "TM_MEASUREMENT": TM_MEASUREMENT,
                    "TS_GLOBAL": TS_GLOBAL,
                }
            ),
        )

        self.local_operation = Choice(
            "Thresholding method",
            [TM_LI, TM_OTSU, TM_ROBUST_BACKGROUND, TM_SAUVOLA,],
            value=TM_LI,
            doc="""\
*(Used only if "{TS_ADAPTIVE}" is selected for thresholding strategy)*

The intensity threshold affects the decision of whether each pixel
will be considered foreground (region(s) of interest) or background. A
higher threshold value will result in only the brightest regions being
identified, whereas a lower threshold value will include dim regions.
When in "Adaptive" mode, the source image is broken into 'blocks' equal
to the size of the "Adaptive Window". A seperate threshold can then be
calculated for each block and blended to create a gradient of different
thresholds for each pixel in the image, determined by local intensity.
A block's threshold can be calculated using many of the methods available
when using the "Global" strategy.

{HELP_ON_PIXEL_INTENSITIES}

The threshold that is used for each image is recorded as a per-image
measurement, so if you are surprised by unusual measurements from one of
your images, you might check whether the automatically calculated
threshold was unusually high or low compared to the other images. See
the **FlagImage** module if you would like to flag an image based on the
threshold value.

-  *{TM_LI}:* The distributions of intensities that define foreground and background are
   used as estimates for probability distributions that produce the intensities of foreground
   and background pixels. For each possible threshold the cross-entropy between the foreground
   and background distributions is calculated and the lowest cross-entropy value is chosen as
   the final threshold. The lowest cross-entropy can be interpreted as the value where the information
   shared between the two probability distributions is the highest. On average, given a pixel of an
   arbitrary intensity, the likelihood it came from the foreground or background would be at its highest.

-  *{TM_OTSU}:* This approach calculates the threshold separating the
   two classes of pixels (foreground and background) by minimizing the
   variance within the each class.

   |image2| This method is a good
   initial approach if you do not know much about the image
   characteristics of all the images in your experiment, especially if
   the percentage of the image covered by foreground varies
   substantially from image to image.

   |image3| Our implementation of
   Otsu’s method allows for assigning the threshold value based on
   splitting the image into either two classes (foreground and
   background) or three classes (foreground, mid-level, and background).
   See the help below for more details.
   
   NOTE that CellProfiler 2 used a non-standard implementation of two-class Otsu
   thresholding; CellProfiler 3.0.0 and onward use the standard implementation. 
   While in most cases the calculated threshold is very similar, pipelines that 
   are adapted from CellProfiler 2 and use two-class Otsu thresholding should be 
   checked when converting to CellProfiler 3 and beyond to make sure that method
   is still the most appropriate. 
   
   NOTE that from CellProfiler 4.0.0 and onwards the standard implementation will
   be used for three-class Otsu thresholding as well. Results with three-class
   Otsu thresholding are likely to be slight different from older versions, so
   imported pipelines which use these methods should be checked when converting
   to the latest version to ensure that settings are still appropriate.
   

-  *{TM_ROBUST_BACKGROUND}:* This method assumes that the background
   distribution approximates a Gaussian by trimming the brightest and
   dimmest X% of pixel intensities, where you choose a suitable percentage.
   It then calculates the mean and
   standard deviation of the remaining pixels and calculates the
   threshold as the mean + N times the standard deviation, where again you
   choose the number of standard deviations to suit your images.

   |image4| This thresholding method can be helpful if the majority of the image
   is background. It can also be helpful if your images vary in overall
   brightness, but the objects of interest are consistently *N* times
   brighter than the background level of the image.

-  *{TM_SAUVOLA}:* This method is a modified variant of Niblack's per-pixel
   thresholding strategy, originally developed for text recognition. A
   threshold is determined for every individual pixel, based on the mean and
   standard deviation of the surrounding pixels within a square window. The
   size of this window is set using the adaptive window parameter.

   |image4| This thresholding method can be helpful when you want to use 
   a very small adaptive window size, which may be useful when trying to
   detect puncti or fine details.
   
   |image3| To improve speed and efficiency, most of these adaptive thresholding
   methods divide the image into blocks, calculate a single threshold for each
   block and interpolate the values between them. In contrast, the simplicity of
   the Sauvola formula allows our implementation to calculate every individual
   pixel seperately (no interpolation) without needing excessive computation
   time.

   |image3| As regions are likely to contain no cells, adaptive thresholds are constrained
   to ensure all pixel thresholds are between 0.7x and 1.5x a global threshold, termed the
   "Guide Threshold". This guide is calculated using the global strategy using the same
   method as selected for adaptive mode. The one exception to this is Sauvola thresholding,
   which uses a Minimum Cross-Entropy global threshold as a guide (since Sauvola is only
   available as a local threshold).

**References**

-  Sezgin M, Sankur B (2004) “Survey over image thresholding techniques
   and quantitative performance evaluation.” *Journal of Electronic
   Imaging*, 13(1), 146-165. (`link`_)

.. _link: https://doi.org/10.1117/1.1631315
.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_AVOID_ICON}
.. |image2| image:: {PROTIP_RECOMMEND_ICON}
.. |image3| image:: {TECH_NOTE_ICON}
.. |image4| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{
                    "HELP_ON_PIXEL_INTENSITIES": _help.HELP_ON_PIXEL_INTENSITIES,
                    "PROTIP_AVOID_ICON": _help.PROTIP_AVOID_ICON,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "TECH_NOTE_ICON": _help.TECH_NOTE_ICON,
                    "TM_OTSU": TM_OTSU,
                    "TM_LI": TM_LI,
                    "TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND,
                    "TM_SAUVOLA": TM_SAUVOLA,
                    "TS_ADAPTIVE": TS_ADAPTIVE,
                }
            ),
        )

        self.threshold_smoothing_scale = Float(
            "Threshold smoothing scale",
            0,
            minval=0,
            doc="""\
This setting controls the scale used to smooth the input image before
the threshold is applied.
The input image can be optionally smoothed before being thresholded.
Smoothing can improve the uniformity of the resulting objects, by
removing holes and jagged edges caused by noise in the acquired image.
Smoothing is most likely *not* appropriate if the input image is binary,
if it has already been smoothed or if it is an output of a pixel-based classifier.
The scale should be approximately the size of the artifacts to be
eliminated by smoothing. A Gaussian is used with a sigma adjusted so
that 1/2 of the Gaussian’s distribution falls within the diameter given
by the scale (sigma = scale / 0.674)
Use a value of 0 for no smoothing. Use a value of 1.3488 for smoothing
with a sigma of 1.
""",
        )

        self.threshold_correction_factor = Float(
            "Threshold correction factor",
            1,
            doc="""\
This setting allows you to adjust the threshold as calculated by the
above method. The value entered here adjusts the threshold either
upwards or downwards, by multiplying it by this value. A value of 1
means no adjustment, 0 to 1 makes the threshold more lenient and > 1
makes the threshold more stringent.

|image0|  When the threshold is
calculated automatically, you may find that the value is consistently
too stringent or too lenient across all images. This setting is helpful
for adjusting the threshold to a value that you empirically determine is
more suitable. For example, the {TM_OTSU} automatic thresholding
inherently assumes that 50% of the image is covered by objects. If a
larger percentage of the image is covered, the Otsu method will give a
slightly biased threshold that may have to be corrected using this
setting.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "TM_OTSU": TM_OTSU,
                }
            ),
        )

        self.threshold_range = FloatRange(
            "Lower and upper bounds on threshold",
            (0, 1),
            minval=0,
            maxval=1,
            doc="""\
Enter the minimum and maximum allowable threshold, a value from 0 to 1.
This is helpful as a safety precaution: when the threshold as calculated
automatically is clearly outside a reasonable range, the min/max allowable
threshold will override the automatic threshold.

|image0| For example, if there are no objects in the field of view, the automatic
threshold might be calculated as unreasonably low; the algorithm will
still attempt to divide the foreground from background (even though
there is no foreground), and you may end up with spurious false positive
foreground regions. In such cases, you can estimate the background pixel
intensity and set the lower bound according to this
empirically-determined value.

{HELP_ON_PIXEL_INTENSITIES}

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
            """.format(
                **{
                    "HELP_ON_PIXEL_INTENSITIES": _help.HELP_ON_PIXEL_INTENSITIES,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

        self.manual_threshold = Float(
            "Manual threshold",
            value=0.0,
            minval=0.0,
            maxval=1.0,
            doc="""\
*(Used only if Manual selected for thresholding method)*

Enter the value that will act as an absolute threshold for the images, a
value from 0 to 1.
""",
        )

        self.thresholding_measurement = Measurement(
            "Select the measurement to threshold with",
            lambda: "Image",
            doc="""\
*(Used only if Measurement is selected for thresholding method)*

Choose the image measurement that will act as an absolute threshold for
the images, for example, the mean intensity calculated from an image in
a prior module.
""",
        )

        self.two_class_otsu = Choice(
            "Two-class or three-class thresholding?",
            [O_TWO_CLASS, O_THREE_CLASS],
            doc="""\
*(Used only for the Otsu thresholding method)*

-  *{O_TWO_CLASS}:* Select this option if the grayscale levels are
   readily distinguishable into only two classes: foreground (i.e.,
   regions of interest) and background.
-  *{O_THREE_CLASS}*: Choose this option if the grayscale levels fall
   instead into three classes: foreground, background and a middle
   intensity between the two. You will then be asked whether the middle
   intensity class should be added to the foreground or background class
   in order to generate the final two-class output.

Note that whether two- or three-class thresholding is chosen, the image
pixels are always finally assigned to only two classes: foreground and
background.

|image0|  As an example, three-class thresholding can be useful for images
in which you have nuclear staining along with low-intensity non-specific
cell staining. In such a case, the background is one class, dim cell
staining is the second class, and bright nucleus staining is the third
class. Depending on your goals, you might wish to identify the nuclei only,
in which case you use three-class thresholding with the middle class
assigned as background. If you want to identify the entire cell, you
use three-class thresholding with the middle class
assigned as foreground.

|image1|  However, in extreme cases where either
there are almost no objects or the entire field of view is covered with
objects, three-class thresholding may perform worse than two-class.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_AVOID_ICON}
""".format(
                **{
                    "O_THREE_CLASS": O_THREE_CLASS,
                    "O_TWO_CLASS": O_TWO_CLASS,
                    "PROTIP_AVOID_ICON": _help.PROTIP_AVOID_ICON,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

        self.assign_middle_to_foreground = Choice(
            "Assign pixels in the middle intensity class to the foreground or the background?",
            [O_FOREGROUND, O_BACKGROUND],
            doc="""\
*(Used only for three-class thresholding)*

Choose whether you want the pixels with middle grayscale intensities to
be assigned to the foreground class or the background class.
""",
        )

        self.lower_outlier_fraction = Float(
            "Lower outlier fraction",
            0.05,
            minval=0,
            maxval=1,
            doc="""\
*(Used only when customizing the "{TM_ROBUST_BACKGROUND}" method)*

Discard this fraction of the pixels in the image starting with those of
the lowest intensity.
""".format(
                **{"TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND}
            ),
        )

        self.upper_outlier_fraction = Float(
            "Upper outlier fraction",
            0.05,
            minval=0,
            maxval=1,
            doc="""\
*(Used only when customizing the "{TM_ROBUST_BACKGROUND}" method)*

Discard this fraction of the pixels in the image starting with those of
the highest intensity.
""".format(
                **{"TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND}
            ),
        )

        self.averaging_method = Choice(
            "Averaging method",
            [RB_MEAN, RB_MEDIAN, RB_MODE],
            doc="""\
*(Used only when customizing the "{TM_ROBUST_BACKGROUND}" method)*

This setting determines how the intensity midpoint is determined.

-  *{RB_MEAN}*: Use the mean of the pixels remaining after discarding
   the outliers. This is a good choice if the cell density is variable
   or high.
-  *{RB_MEDIAN}*: Use the median of the pixels. This is a good choice
   if, for all images, more than half of the pixels are in the
   background after removing outliers.
-  *{RB_MODE}*: Use the most frequently occurring value from among the
   pixel values. The {TM_ROBUST_BACKGROUND} method groups the
   intensities into bins (the number of bins is the square root of the
   number of pixels in the unmasked portion of the image) and chooses
   the intensity associated with the bin with the most pixels.
""".format(
                **{
                    "RB_MEAN": RB_MEAN,
                    "RB_MEDIAN": RB_MEDIAN,
                    "RB_MODE": RB_MODE,
                    "TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND,
                }
            ),
        )

        self.variance_method = Choice(
            "Variance method",
            [RB_SD, RB_MAD],
            doc="""\
*(Used only when customizing the "{TM_ROBUST_BACKGROUND}" method)*

Robust background adds a number of deviations (standard or MAD) to the
average to get the final background. This setting chooses the method
used to assess the variance in the pixels, after removing outliers.
Choose one of *{RB_SD}* or *{RB_MAD}* (the median of the absolute
difference of the pixel intensities from their median).
""".format(
                **{
                    "RB_MAD": RB_MAD,
                    "RB_SD": RB_SD,
                    "TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND,
                }
            ),
        )

        self.number_of_deviations = Float(
            "# of deviations",
            2,
            doc="""\
*(Used only when customizing the "{TM_ROBUST_BACKGROUND}" method)*

Robust background calculates the variance, multiplies it by the value
given by this setting and adds it to the average. Adding several
deviations raises the threshold well above the average.
Use a larger number to be more stringent about identifying foreground pixels.
Use a smaller number to be less stringent. It’s even possible to
use a negative number if you want the threshold to be lower than the average
(e.g., for images that are densely covered by foreground).
""".format(
                **{"TM_ROBUST_BACKGROUND": TM_ROBUST_BACKGROUND}
            ),
        )

        self.adaptive_window_size = Integer(
            "Size of adaptive window",
            50,
            doc="""\
*(Used only if "{TS_ADAPTIVE}" is selected for thresholding strategy)*

Enter the size of the window (in pixels) to be used for the adaptive method.
Often a good choice is some multiple of the largest expected object size.
""".format(
                **{"TS_ADAPTIVE": TS_ADAPTIVE}
            ),
        )
        self.log_transform = Binary(
            "Log transform before thresholding?",
            value=False,
            doc=f"""\
*(Used only with the "{TM_LI}" and "{TM_OTSU}" methods)*

Choose whether to log-transform intensity values before thresholding.
The log transformation is applied before calculating the threshold, and the resulting 
threshold values will be converted back onto a linear scale.

Automatic thresholding is usually performed using histograms of pixel intensities. Areas of similar intensity, 
such as positive staining, form a peak which is used to determine the threshold. Log transformation 
helps to enhance peaks of intensity which are particularly wide. This helps to detect areas of staining 
which have a wide dynamic range.

In practice this tends to increase the sensitivity of the resulting threshold, which is useful when trying to detect 
objects such as cells which are not stained uniformly throughout. You might want to enable this option if you're
trying to detect autofluorescence or to pick up the entire cytoplasm of cells which contain smaller areas of intense 
staining.
""",
        )

    @property
    def threshold_operation(self):
        if self.threshold_scope.value == TS_GLOBAL:
            return self.global_operation.value

        return self.local_operation.value

    def visible_settings(self):
        visible_settings = super(Threshold, self).visible_settings()

        visible_settings += [self.threshold_scope]

        if self.threshold_scope.value == TS_GLOBAL:
            visible_settings += [self.global_operation]
        else:
            visible_settings += [self.local_operation]

        if self.threshold_operation == TM_MANUAL:
            visible_settings += [self.manual_threshold]
        elif self.threshold_operation == TM_MEASUREMENT:
            visible_settings += [self.thresholding_measurement]
        elif self.threshold_operation == TM_OTSU:
            visible_settings += [self.two_class_otsu]

            if self.two_class_otsu == O_THREE_CLASS:
                visible_settings += [self.assign_middle_to_foreground]
        elif self.threshold_operation == TM_ROBUST_BACKGROUND:
            visible_settings += [
                self.lower_outlier_fraction,
                self.upper_outlier_fraction,
                self.averaging_method,
                self.variance_method,
                self.number_of_deviations,
            ]

        visible_settings += [self.threshold_smoothing_scale]

        if self.threshold_operation != TM_MANUAL:
            visible_settings += [self.threshold_correction_factor, self.threshold_range]

        if self.threshold_scope == TS_ADAPTIVE:
            visible_settings += [self.adaptive_window_size]

        if self.threshold_operation in (TM_LI, TM_OTSU):
            visible_settings += [self.log_transform]

        return visible_settings

    def settings(self):
        settings = super(Threshold, self).settings()

        return settings + [
            self.threshold_scope,
            self.global_operation,
            self.threshold_smoothing_scale,
            self.threshold_correction_factor,
            self.threshold_range,
            self.manual_threshold,
            self.thresholding_measurement,
            self.two_class_otsu,
            self.log_transform,
            self.assign_middle_to_foreground,
            self.adaptive_window_size,
            self.lower_outlier_fraction,
            self.upper_outlier_fraction,
            self.averaging_method,
            self.variance_method,
            self.number_of_deviations,
            self.local_operation,
        ]

    def help_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.threshold_scope,
            self.global_operation,
            self.local_operation,
            self.manual_threshold,
            self.thresholding_measurement,
            self.two_class_otsu,
            self.log_transform,
            self.assign_middle_to_foreground,
            self.lower_outlier_fraction,
            self.upper_outlier_fraction,
            self.averaging_method,
            self.variance_method,
            self.number_of_deviations,
            self.adaptive_window_size,
            self.threshold_correction_factor,
            self.threshold_range,
            self.threshold_smoothing_scale,
        ]

    def run(self, workspace):
        input_image = workspace.image_set.get_image(
            self.x_name.value, must_be_grayscale=True
        )
        dimensions = input_image.dimensions
        final_threshold, orig_threshold, guide_threshold = self.get_threshold(
            input_image, workspace, automatic=False,
        )

        self.add_threshold_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        binary_image, _ = self.apply_threshold(input_image, final_threshold)

        self.add_fg_bg_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            input_image,
            binary_image,
        )

        output = Image(binary_image, parent_image=input_image, dimensions=dimensions)

        workspace.image_set.add(self.y_name.value, output)

        if self.show_window:
            workspace.display_data.input_pixel_data = input_image.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            workspace.display_data.dimensions = dimensions
            statistics = workspace.display_data.statistics = []
            workspace.display_data.col_labels = ("Feature", "Value")
            if self.threshold_scope == TS_ADAPTIVE:
                workspace.display_data.threshold_image = final_threshold

            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])
                statistics += [(column[1].split("_")[1], str(value))]

    def apply_threshold(self, image, threshold, automatic=False):
        data = image.pixel_data

        mask = image.mask

        if not automatic and self.threshold_smoothing_scale.value == 0:
            return (data >= threshold) & mask, 0

        if automatic:
            sigma = 1
        else:
            # Convert from a scale into a sigma. What I've done here
            # is to structure the Gaussian so that 1/2 of the smoothed
            # intensity is contributed from within the smoothing diameter
            # and 1/2 is contributed from outside.
            sigma = self.threshold_smoothing_scale.value / 0.6744 / 2.0

        blurred_image = centrosome.smooth.smooth_with_function_and_mask(
            data,
            lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
            mask,
        )

        return (blurred_image >= threshold) & mask, sigma

    def get_threshold(self, image, workspace, automatic=False):
        need_transform = (
                not automatic and
                self.threshold_operation in (TM_LI, TM_OTSU) and
                self.log_transform
        )

        if need_transform:
            image_data, conversion_dict = centrosome.threshold.log_transform(image.pixel_data)
        else:
            image_data = image.pixel_data

        if self.threshold_operation == TM_MANUAL:
            return self.manual_threshold.value, self.manual_threshold.value, None

        elif self.threshold_operation == TM_MEASUREMENT:
            # Thresholds are stored as single element arrays.  Cast to float to extract the value.
            orig_threshold = float(
                workspace.measurements.get_current_image_measurement(
                    self.thresholding_measurement.value
                )
            )
            return self._correct_global_threshold(orig_threshold), orig_threshold, None

        elif self.threshold_scope.value == TS_GLOBAL or automatic:
            th_guide = None
            th_original = self.get_global_threshold(image_data, image.mask, automatic=automatic)

        elif self.threshold_scope.value == TS_ADAPTIVE:
            th_guide = self.get_global_threshold(image_data, image.mask)
            th_original = self.get_local_threshold(image_data, image.mask, image.volumetric)
        else:
            raise ValueError("Invalid thresholding settings")

        if need_transform:
            th_original = centrosome.threshold.inverse_log_transform(th_original, conversion_dict)
            if th_guide is not None:
                th_guide = centrosome.threshold.inverse_log_transform(th_guide, conversion_dict)

        if self.threshold_scope.value == TS_GLOBAL or automatic:
            th_corrected = self._correct_global_threshold(th_original)
        else:
            th_guide = self._correct_global_threshold(th_guide)
            th_corrected = self._correct_local_threshold(th_original, th_guide)

        return th_corrected, th_original, th_guide

    def get_global_threshold(self, image, mask, automatic=False):
        image_data = image[mask]

        # Shortcuts - Check if image array is empty or all pixels are the same value.
        if len(image_data) == 0:
            threshold = 0.0

        elif numpy.all(image_data == image_data[0]):
            threshold = image_data[0]

        elif automatic or self.threshold_operation in (TM_LI, TM_SAUVOLA):
            tol = max(numpy.min(numpy.diff(numpy.unique(image_data))) / 2, 0.5 / 65536)
            threshold = skimage.filters.threshold_li(image_data, tolerance=tol)

        elif self.threshold_operation == TM_ROBUST_BACKGROUND:
            threshold = self.get_threshold_robust_background(image_data)

        elif self.threshold_operation == TM_OTSU:
            if self.two_class_otsu.value == O_TWO_CLASS:
                threshold = skimage.filters.threshold_otsu(image_data)
            elif self.two_class_otsu.value == O_THREE_CLASS:
                bin_wanted = (
                    0 if self.assign_middle_to_foreground.value == "Foreground" else 1
                )
                threshold = skimage.filters.threshold_multiotsu(image_data, nbins=128)
                threshold = threshold[bin_wanted]
        else:
            raise ValueError("Invalid thresholding settings")
        return threshold

    def get_local_threshold(self, image, mask, volumetric):
        image_data = numpy.where(mask, image, numpy.nan)

        if len(image_data) == 0 or numpy.all(image_data == numpy.nan):
            local_threshold = numpy.zeros_like(image_data)

        elif numpy.all(image_data == image_data[0]):
            local_threshold = numpy.full_like(image_data, image_data[0])

        elif self.threshold_operation == TM_LI:
            local_threshold = self._run_local_threshold(
                image_data,
                method=skimage.filters.threshold_li,
                volumetric=volumetric,
                tolerance=max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
            )
        elif self.threshold_operation == TM_OTSU:
            if self.two_class_otsu.value == O_TWO_CLASS:
                local_threshold = self._run_local_threshold(
                    image_data,
                    method=skimage.filters.threshold_otsu,
                    volumetric=volumetric,
                )

            elif self.two_class_otsu.value == O_THREE_CLASS:
                local_threshold = self._run_local_threshold(
                    image_data,
                    method=skimage.filters.threshold_multiotsu,
                    volumetric=volumetric,
                    nbins=128,
                )

        elif self.threshold_operation == TM_ROBUST_BACKGROUND:
            local_threshold = self._run_local_threshold(
                image_data,
                method=self.get_threshold_robust_background,
                volumetric=volumetric,
            )

        elif self.threshold_operation == TM_SAUVOLA:
            image_data = numpy.where(mask, image, 0)
            adaptive_window = self.adaptive_window_size.value
            if adaptive_window % 2 == 0:
                adaptive_window += 1
            local_threshold = skimage.filters.threshold_sauvola(
                image_data, window_size=adaptive_window
            )

        else:
            raise ValueError("Invalid thresholding settings")
        return local_threshold

    def _run_local_threshold(self, image_data, method, volumetric=False, **kwargs):
        if volumetric:
            t_local = numpy.zeros_like(image_data)
            for index, plane in enumerate(image_data):
                t_local[index] = self._get_adaptive_threshold(plane, method, **kwargs)
        else:
            t_local = self._get_adaptive_threshold(image_data, method, **kwargs)
        return skimage.img_as_float(t_local)

    def _get_adaptive_threshold(self, image_data, threshold_method, **kwargs):
        """Given a global threshold, compute a threshold per pixel

        Break the image into blocks, computing the threshold per block.
        Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
        """
        # for the X and Y direction, find the # of blocks, given the
        # size constraints
        if self.threshold_operation == TM_OTSU:
            bin_wanted = (
                0 if self.assign_middle_to_foreground.value == "Foreground" else 1
            )
        image_size = numpy.array(image_data.shape[:2], dtype=int)
        nblocks = image_size // self.adaptive_window_size.value
        if any(n < 2 for n in nblocks):
            raise ValueError(
                "Adaptive window cannot exceed 50%% of an image dimension.\n"
                "Window of %dpx is too large for a %sx%s image"
                % (self.adaptive_window_size.value, image_size[1], image_size[0])
            )
        #
        # Use a floating point block size to apportion the roundoff
        # roughly equally to each block
        #
        increment = numpy.array(image_size, dtype=float) / numpy.array(
            nblocks, dtype=float
        )
        #
        # Put the answer here
        #
        thresh_out = numpy.zeros(image_size, image_data.dtype)
        #
        # Loop once per block, computing the "global" threshold within the
        # block.
        #
        block_threshold = numpy.zeros([nblocks[0], nblocks[1]])
        for i in range(nblocks[0]):
            i0 = int(i * increment[0])
            i1 = int((i + 1) * increment[0])
            for j in range(nblocks[1]):
                j0 = int(j * increment[1])
                j1 = int((j + 1) * increment[1])
                block = image_data[i0:i1, j0:j1]
                block = block[~numpy.isnan(block)]
                if len(block) == 0:
                    threshold_out = 0.0
                elif numpy.all(block == block[0]):
                    # Don't compute blocks with only 1 value.
                    threshold_out = block[0]
                elif (self.threshold_operation == TM_OTSU and
                      self.two_class_otsu.value == O_THREE_CLASS and
                      len(numpy.unique(block)) < 3):
                    # Can't run 3-class otsu on only 2 values.
                    threshold_out = skimage.filters.threshold_otsu(block)
                else:
                    try: 
                        threshold_out = threshold_method(block, **kwargs)
                    except ValueError:
                        threshold_out = threshold_method(block)
                if isinstance(threshold_out, numpy.ndarray):
                    # Select correct bin if running multiotsu
                    threshold_out = threshold_out[bin_wanted]
                block_threshold[i, j] = threshold_out

        #
        # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
        #
        spline_order = min(3, numpy.min(nblocks) - 1)
        xStart = int(increment[0] / 2)
        xEnd = int((nblocks[0] - 0.5) * increment[0])
        yStart = int(increment[1] / 2)
        yEnd = int((nblocks[1] - 0.5) * increment[1])
        xtStart = 0.5
        xtEnd = image_data.shape[0] - 0.5
        ytStart = 0.5
        ytEnd = image_data.shape[1] - 0.5
        block_x_coords = numpy.linspace(xStart, xEnd, nblocks[0])
        block_y_coords = numpy.linspace(yStart, yEnd, nblocks[1])
        adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
            block_x_coords,
            block_y_coords,
            block_threshold,
            bbox=(xtStart, xtEnd, ytStart, ytEnd),
            kx=spline_order,
            ky=spline_order,
        )
        thresh_out_x_coords = numpy.linspace(
            0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
        )
        thresh_out_y_coords = numpy.linspace(
            0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
        )

        thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

        return thresh_out

    def _correct_global_threshold(self, threshold):
        threshold *= self.threshold_correction_factor.value
        return min(max(threshold, self.threshold_range.min), self.threshold_range.max)

    def _correct_local_threshold(self, t_local_orig, t_guide):
        t_local = t_local_orig.copy()
        t_local *= self.threshold_correction_factor.value

        # Constrain the local threshold to be within [0.7, 1.5] * global_threshold. It's for the pretty common case
        # where you have regions of the image with no cells whatsoever that are as large as whatever window you're
        # using. Without a lower bound, you start having crazy threshold s that detect noise blobs. And same for
        # very crowded areas where there is zero background in the window. You want the foreground to be all
        # detected.
        t_min = max(self.threshold_range.min, t_guide * 0.7)
        t_max = min(self.threshold_range.max, t_guide * 1.5)

        t_local[t_local < t_min] = t_min
        t_local[t_local > t_max] = t_max

        return t_local

    def get_threshold_robust_background(self, image_data):
        """Calculate threshold based on mean & standard deviation
           The threshold is calculated by trimming the top and bottom 5% of
           pixels off the image, then calculating the mean and standard deviation
           of the remaining image. The threshold is then set at 2 (empirical
           value) standard deviations above the mean.

           lower_outlier_fraction - after ordering the pixels by intensity, remove
               the pixels from 0 to len(image) * lower_outlier_fraction from
               the threshold calculation (default = .05).
            upper_outlier_fraction - remove the pixels from
               len(image) * (1 - upper_outlier_fraction) to len(image) from
               consideration (default = .05).
            deviations_above_average - calculate the standard deviation or MAD and
               multiply by this number and add to the average to get the final
               threshold (default = 2)
            average_fn - function used to calculate the average intensity (e.g.
               np.mean, np.median or some sort of mode function). Default = np.mean
            variance_fn - function used to calculate the amount of variance.
                          Default = np.sd
        """

        average_fn = {
            RB_MEAN: numpy.mean,
            RB_MEDIAN: numpy.median,
            RB_MODE: centrosome.threshold.binned_mode,
        }.get(self.averaging_method.value, numpy.mean)

        variance_fn = {RB_SD: numpy.std, RB_MAD: centrosome.threshold.mad}.get(
            self.variance_method.value, numpy.std
        )
        flat_image = image_data.flatten()
        n_pixels = len(flat_image)
        if n_pixels < 3:
            return 0

        flat_image.sort()
        if flat_image[0] == flat_image[-1]:
            return flat_image[0]
        low_chop = int(round(n_pixels * self.lower_outlier_fraction.value))
        hi_chop = n_pixels - int(round(n_pixels * self.upper_outlier_fraction.value))
        im = flat_image if low_chop == 0 else flat_image[low_chop:hi_chop]
        mean = average_fn(im)
        sd = variance_fn(im)
        return mean + sd * self.number_of_deviations.value

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 2), dimensions=dimensions)

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.input_pixel_data,
            title="Original image: {}".format(self.x_name.value),
        )

        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.output_pixel_data,
            title="Thresholded image: {}".format(self.y_name.value),
            sharexy=figure.subplot(0, 0),
        )

        if self.threshold_scope == TS_ADAPTIVE:
            figure.subplot_imshow_grayscale(
                0,
                1,
                workspace.display_data.threshold_image,
                title="Local threshold values",
                sharexy=figure.subplot(0, 0),
                vmax=workspace.display_data.input_pixel_data.max(),
                vmin=workspace.display_data.input_pixel_data.min(),
                normalize=False,
            )

        figure.subplot_table(
            1, 1, workspace.display_data.statistics, workspace.display_data.col_labels
        )

    def get_measurement_objects_name(self):
        return self.y_name.value

    def add_threshold_measurements(
        self,
        objname,
        measurements,
        final_threshold,
        orig_threshold,
        guide_threshold=None,
    ):
        ave_final_threshold = numpy.mean(numpy.atleast_1d(final_threshold))
        ave_orig_threshold = numpy.mean(numpy.atleast_1d(orig_threshold))
        measurements.add_measurement(
            "Image", FF_FINAL_THRESHOLD % objname, ave_final_threshold,
        )

        measurements.add_measurement(
            "Image", FF_ORIG_THRESHOLD % objname, ave_orig_threshold,
        )

        if self.threshold_scope == TS_ADAPTIVE:
            measurements.add_measurement(
                "Image", FF_GUIDE_THRESHOLD % objname, guide_threshold,
            )

    def add_fg_bg_measurements(self, objname, measurements, image, binary_image):
        data = image.pixel_data

        mask = image.mask

        wv = centrosome.threshold.weighted_variance(data, mask, binary_image)

        measurements.add_measurement(
            "Image", FF_WEIGHTED_VARIANCE % objname, numpy.array([wv], dtype=float),
        )

        entropies = centrosome.threshold.sum_of_entropies(data, mask, binary_image)

        measurements.add_measurement(
            "Image",
            FF_SUM_OF_ENTROPIES % objname,
            numpy.array([entropies], dtype=float),
        )

    def get_measurement_columns(self, pipeline, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        measures = [
            ("Image", FF_FINAL_THRESHOLD % object_name, COLTYPE_FLOAT,),
            ("Image", FF_ORIG_THRESHOLD % object_name, COLTYPE_FLOAT,),
        ]
        if self.threshold_scope == TS_ADAPTIVE:
            measures += [("Image", FF_GUIDE_THRESHOLD % object_name, COLTYPE_FLOAT,)]
        measures += [
            ("Image", FF_WEIGHTED_VARIANCE % object_name, COLTYPE_FLOAT,),
            ("Image", FF_SUM_OF_ENTROPIES % object_name, COLTYPE_FLOAT,),
        ]
        return measures

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return [C_THRESHOLD]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == C_THRESHOLD:
            measures = [
                FTR_ORIG_THRESHOLD,
                FTR_FINAL_THRESHOLD,
            ]
            if self.threshold_scope == TS_ADAPTIVE:
                measures += [FTR_GUIDE_THRESHOLD]
            measures += [
                FTR_SUM_OF_ENTROPIES,
                FTR_WEIGHTED_VARIANCE,
            ]
            return measures
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.get_measurement_objects_name()]

        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number < 7:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3.0."
            )

        if variable_revision_number == 7:
            setting_values = setting_values[:2] + setting_values[6:]

            setting_values = setting_values[:2] + self.upgrade_threshold_settings(
                setting_values[2:]
            )

            variable_revision_number = 8

        if variable_revision_number == 8:
            setting_values = setting_values[:2] + setting_values[3:]

            variable_revision_number = 9

        if variable_revision_number == 9:
            if setting_values[2] in [TM_MANUAL, TM_MEASUREMENT]:
                setting_values[3] = setting_values[2]

                setting_values[2] = TS_GLOBAL

            if setting_values[2] == TS_ADAPTIVE and setting_values[3] in [
                centrosome.threshold.TM_MCT,
                centrosome.threshold.TM_ROBUST_BACKGROUND,
            ]:
                setting_values[2] = TS_GLOBAL

            if setting_values[3] == centrosome.threshold.TM_MCT:
                setting_values[3] = TM_LI

            if setting_values[2] == TS_ADAPTIVE:
                setting_values += [setting_values[3]]
            else:
                setting_values += [centrosome.threshold.TM_OTSU]
            variable_revision_number = 10
        used_log_otsu = False
        if variable_revision_number == 10:
            # Relabel method names
            if setting_values[3] == "RobustBackground":
                setting_values[3] = TM_ROBUST_BACKGROUND
            elif setting_values[3] == "Minimum cross entropy":
                setting_values[3] = TM_LI
            if (setting_values[2] == TS_GLOBAL and setting_values[3] == TM_OTSU) or (
                    setting_values[2] == TS_ADAPTIVE and setting_values[-1] == TM_OTSU):
                if setting_values[9] == O_THREE_CLASS:
                    used_log_otsu = True
            variable_revision_number = 11
        if variable_revision_number == 11:
            setting_values.insert(10, used_log_otsu)
            variable_revision_number = 12
        return setting_values, variable_revision_number

    def upgrade_threshold_settings(self, setting_values):
        """Upgrade the threshold settings to the current version

        use the first setting which is the version to determine the
        threshold settings version and upgrade as appropriate
        """
        version = int(setting_values[0])

        if version == 1:
            # Added robust background settings
            #
            setting_values = setting_values + [
                "Default",  # Robust background custom choice
                0.05,
                0.05,  # lower and upper outlier fractions
                RB_MEAN,  # averaging method
                RB_SD,  # variance method
                2,
            ]  # of standard deviations
            version = 2

        if version == 2:
            if setting_values[1] in ["Binary image", "Per object"]:
                setting_values[1] = "None"

            if setting_values[1] == "Automatic":
                setting_values[1] = TS_GLOBAL
                setting_values[2] = centrosome.threshold.TM_MCT
                setting_values[3] = "Manual"
                setting_values[4] = "1.3488"
                setting_values[5] = "1"
                setting_values[6] = "(0.0, 1.0)"

            removed_threshold_methods = [
                centrosome.threshold.TM_KAPUR,
                centrosome.threshold.TM_MOG,
                centrosome.threshold.TM_RIDLER_CALVARD,
            ]

            if setting_values[2] in removed_threshold_methods:
                setting_values[2] = "None"

            if setting_values[2] == centrosome.threshold.TM_BACKGROUND:
                setting_values[2] = centrosome.threshold.TM_ROBUST_BACKGROUND
                setting_values[17] = "Custom"
                setting_values[18] = "0.02"
                setting_values[19] = "0.02"
                setting_values[20] = RB_MODE
                setting_values[21] = RB_SD
                setting_values[22] = "0"

                correction_factor = float(setting_values[5])

                if correction_factor == 0:
                    correction_factor = 2
                else:
                    correction_factor *= 2

                setting_values[5] = str(correction_factor)

            if setting_values[3] == "No smoothing":
                setting_values[4] = "0"

            if setting_values[3] == "Automatic":
                setting_values[4] = "1.3488"

            if setting_values[17] == "Default":
                setting_values[18] = "0.05"
                setting_values[19] = "0.05"
                setting_values[20] = RB_MEAN
                setting_values[21] = RB_SD
                setting_values[22] = "2"

            new_setting_values = setting_values[:3]
            new_setting_values += setting_values[4:7]
            new_setting_values += setting_values[8:10]
            new_setting_values += setting_values[12:13]
            new_setting_values += setting_values[14:15]
            new_setting_values += setting_values[16:17]
            new_setting_values += setting_values[18:]

            setting_values = new_setting_values

        return setting_values

    def validate_module(self, pipeline):
        if (
            self.threshold_operation == TM_ROBUST_BACKGROUND
            and self.lower_outlier_fraction.value + self.upper_outlier_fraction.value
            >= 1
        ):
            raise ValidationError(
                """
                The sum of the lower robust background outlier fraction ({0:f}) and the upper fraction ({1:f}) must be
                less than one.
                """.format(
                    self.lower_outlier_fraction.value, self.upper_outlier_fraction.value
                ),
                self.upper_outlier_fraction,
            )
