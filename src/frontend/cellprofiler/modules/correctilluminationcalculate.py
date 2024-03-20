"""
CorrectIlluminationCalculate
============================

**CorrectIlluminationCalculate** calculates an illumination function
that is used to correct uneven illumination/lighting/shading or to
reduce uneven background in images.

This module calculates an illumination function that can either be saved to the
hard drive for later use or immediately applied to images later in the pipeline.
This function will correct for the uneven illumination in images. Use the
**CorrectIlluminationApply** module to apply the function to the image to be
corrected. Use **SaveImages** to export an illumination function to the hard
drive using the "npy" file format.

Warning: illumination correction is a challenge to do properly;
please see the `examples`_ and `tutorials`_ pages on the CellProfiler
website for further advice.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **CorrectIlluminationApply**, **Smooth**, and
**EnhanceOrSuppressFeatures**.

References
^^^^^^^^^^

-  J Lindblad and E Bengtsson (2001) “A comparison of methods for estimation
   of intensity nonuniformities in 2D and 3D microscope images of fluorescence
   stained cells.”, Proceedings of the 12th Scandinavian Conference on Image Analysis
   (SCIA), pp. 264-271

.. _examples: https://cellprofiler.org/examples
.. _tutorials: https://tutorials.cellprofiler.org
"""

import centrosome.bg_compensate
import centrosome.cpmorphology
import centrosome.cpmorphology
import centrosome.filter
import centrosome.smooth
import numpy
import scipy.ndimage
import skimage.filters
from cellprofiler_core.image import AbstractImage
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer

IC_REGULAR = "Regular"
IC_BACKGROUND = "Background"
RE_MEDIAN = "Median"
EA_EACH = "Each"
EA_ALL = "All"
EA_ALL_FIRST = "All: First cycle"
EA_ALL_ACROSS = "All: Across cycles"
SRC_LOAD_IMAGES = "Load Images module"
SRC_PIPELINE = "Pipeline"
SM_NONE = "No smoothing"
SM_CONVEX_HULL = "Convex Hull"
SM_FIT_POLYNOMIAL = "Fit Polynomial"
SM_MEDIAN_FILTER = "Median Filter"
SM_GAUSSIAN_FILTER = "Gaussian Filter"
SM_TO_AVERAGE = "Smooth to Average"
SM_SPLINES = "Splines"

FI_AUTOMATIC = "Automatic"
FI_OBJECT_SIZE = "Object size"
FI_MANUALLY = "Manually"

ROBUST_FACTOR = 0.02  # For rescaling, take 2nd percentile value

OUTPUT_IMAGE = "OutputImage"

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"


class CorrectIlluminationCalculate(Module):
    module_name = "CorrectIlluminationCalculate"
    variable_revision_number = 2
    category = "Image Processing"

    def create_settings(self):
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="Choose the image to be used to calculate the illumination function.",
        )

        self.illumination_image_name = ImageName(
            "Name the output image",
            "IllumBlue",
            doc="""Enter a name for the resultant illumination function.""",
            provided_attributes={"aggregate_image": True, "available_on_last": False,},
        )

        self.intensity_choice = Choice(
            "Select how the illumination function is calculated",
            [IC_REGULAR, IC_BACKGROUND],
            IC_REGULAR,
            doc="""\
Choose which method you want to use to calculate the illumination
function. You may chose from the following options:

-  *{IC_REGULAR}:* If you have objects that are evenly dispersed across
   your image(s) and cover most of the image, the *Regular* method might
   be appropriate. *Regular* makes the illumination function
   based on the intensity at each pixel of the image (or group of images
   if you are in *{EA_ALL}* mode) and is most often rescaled (see
   below) and applied by division using **CorrectIlluminationApply.**
   Note that if you are in *{EA_EACH}* mode or using a small set of
   images with few objects, there will be regions in the average image
   that contain no objects and smoothing by median filtering is unlikely
   to work well. *Note:* it does not make sense to choose
   (*{IC_REGULAR} + {SM_NONE} + {EA_EACH}*) because the illumination
   function would be identical to the original image and applying it
   will yield a blank image. You either need to smooth each image, or
   you need to use *{EA_ALL}* images.
-  *{IC_BACKGROUND}:* If you think that the background (dim regions)
   between objects show the same pattern of illumination as your objects
   of interest, you can choose the *{IC_BACKGROUND}* method. Background
   intensities finds the minimum pixel intensities in blocks across the
   image (or group of images if you are in *{EA_ALL}* mode) and is most
   often applied by subtraction using the **CorrectIlluminationApply**
   module. *Note:* if you will be using the *{DOS_SUBTRACT}* option in
   the **CorrectIlluminationApply** module, you almost certainly do not
   want to rescale the illumination function.

Please note that if a mask was applied to the input image, the pixels
outside of the mask will be excluded from consideration. This is useful,
for instance, in cases where you have masked out the well edge in an
image from a multi-well plate; the dark well edge would distort the
illumination correction function along the interior well edge. Masking
the image beforehand solves this problem.
""".format(
                **{
                    "IC_REGULAR": IC_REGULAR,
                    "EA_ALL": EA_ALL,
                    "EA_EACH": EA_EACH,
                    "SM_NONE": SM_NONE,
                    "IC_BACKGROUND": IC_BACKGROUND,
                    "DOS_SUBTRACT": DOS_SUBTRACT,
                }
            ),
        )

        self.dilate_objects = Binary(
            "Dilate objects in the final averaged image?",
            False,
            doc="""\
*(Used only if the “%(IC_REGULAR)s” method is selected)*

For some applications, the incoming images are binary and each object
should be dilated with a Gaussian filter in the final averaged
(projection) image. This is for a sophisticated method of illumination
correction where model objects are produced. Select *Yes* to dilate
objects for this approach.
"""
            % globals(),
        )

        self.object_dilation_radius = Integer(
            "Dilation radius",
            1,
            0,
            doc="""\
*(Used only if the “%(IC_REGULAR)s” method and dilation is selected)*

This value should be roughly equal to the original radius of the objects.
"""
            % globals(),
        )

        self.block_size = Integer(
            "Block size",
            60,
            1,
            doc="""\
*(Used only if “%(IC_BACKGROUND)s” is selected)*

The block size should be large enough that every square block of pixels
is likely to contain some background pixels, where no objects are
located.
"""
            % globals(),
        )

        self.rescale_option = Choice(
            "Rescale the illumination function?",
            ["Yes", "No", RE_MEDIAN],
            doc="""\
The illumination function can be rescaled so that the pixel intensities
are all equal to or greater than 1. You have the following options:

-  *Yes:* Rescaling is recommended if you plan to use the
   *%(IC_REGULAR)s* method (and hence, the *%(DOS_DIVIDE)s* option in
   **CorrectIlluminationApply**). Rescaling the illumination function to
   >1 ensures that the values in your corrected image will stay between
   0-1 after division.
-  *No:* Rescaling is not recommended if you plan to use the
   *%(IC_BACKGROUND)s* method, which is paired with the
   *%(DOS_SUBTRACT)s* option in **CorrectIlluminationApply**. Because
   rescaling causes the illumination function to have values from 1 to
   infinity, subtracting those values from your image would cause the
   corrected images to be very dark, even negative.
-  %(RE_MEDIAN)s\ *:* This option chooses the median value in the image
   to rescale so that division increases some values and decreases others.
"""
            % globals(),
        )

        self.each_or_all = Choice(
            "Calculate function for each image individually, or based on all images?",
            [EA_EACH, EA_ALL_FIRST, EA_ALL_ACROSS],
            doc="""\
Calculate a separate function for each image, or one for all the
images? You can calculate the illumination function using just the
current image or you can calculate the illumination function using all
of the images in each group (or in the entire experiment). The
illumination function can be calculated in one of the three ways:

-  *%(EA_EACH)s:* Calculate an illumination function for each image
   individually.
-  *%(EA_ALL_FIRST)s:* Calculate an illumination function based on all
   of the images in a group, performing the calculation before
   proceeding to the next module. This means that the illumination
   function will be created in the first cycle (making the first cycle
   longer than subsequent cycles), and lets you use the function in a
   subsequent **CorrectIlluminationApply** module in the same
   pipeline, but also means that you will not have the ability to filter
   out images (e.g., by using **FlagImage**). The input images need to
   be assembled using the **Input** modules; using images produced by
   other modules will yield an error. Thus, typically,
   **CorrectIlluminationCalculate** will be the first module after the
   input modules.
-  *%(EA_ALL_ACROSS)s:* Calculate an illumination function across all
   cycles in each group. This option takes any image as input; however,
   the illumination function will not be completed until the end of the
   last cycle in the group. You can use **SaveImages** to save the
   illumination function after the last cycle in the group and then use
   the resulting image in another pipeline. The option is useful if you
   want to exclude images that are filtered by a prior **FlagImage**
   module.
"""
            % globals(),
        )
        self.smoothing_method = Choice(
            "Smoothing method",
            [
                SM_NONE,
                SM_CONVEX_HULL,
                SM_FIT_POLYNOMIAL,
                SM_MEDIAN_FILTER,
                SM_GAUSSIAN_FILTER,
                SM_TO_AVERAGE,
                SM_SPLINES,
            ],
            doc="""\
If requested, the resulting image is smoothed. If you are using *Each* mode,
smoothing is definitely needed. For *All* modes, you usually also want to
smooth, especially if you have few objects in each image or a small image set.

You should smooth to the point where the illumination function resembles
a believable pattern. For example, if you are trying to correct a lamp
illumination problem, apply smoothing until you obtain a fairly smooth
pattern without sharp bright or dim regions. Note that smoothing is a
time-consuming process, but some methods are faster than others.

-  *%(SM_FIT_POLYNOMIAL)s:* This method is fastest but does not allow
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
-  *%(SM_MEDIAN_FILTER)s* and *%(SM_GAUSSIAN_FILTER)s:*
   We typically recommend
   *%(SM_MEDIAN_FILTER)s* vs. *%(SM_GAUSSIAN_FILTER)s* because the
   median is less sensitive to outliers, although the results are also
   slightly less smooth and the fact that images are in the range of 0
   to 1 means that outliers typically will not dominate too strongly
   anyway. The *%(SM_GAUSSIAN_FILTER)s* convolves the image with a
   Gaussian whose full width at half maximum is the artifact diameter
   entered. Its effect is to blur and obscure features smaller than the
   specified diameter and spread bright or dim features larger than the
   specified diameter. The *%(SM_MEDIAN_FILTER)s* finds the median pixel value within
   the diameter you specify. It removes bright or dim features
   that are significantly smaller than the specified diameter.
-  *%(SM_TO_AVERAGE)s:* A less commonly used option is to completely
   smooth the entire image, which will create a flat, smooth image where
   every pixel of the image is the average of what the illumination
   function would otherwise have been.
-  *%(SM_SPLINES)s:* This method (*Lindblad and Bengtsson, 2001*) fits
   a grid of cubic splines to the background while excluding foreground
   pixels from the calculation. It operates iteratively, classifying
   pixels as background, computing a best fit spline to this background
   and then reclassifying pixels as background until the spline
   converges on its final value. This method is best for backgrounds that
   are highly variable and irregular. Note that the computation time can
   be significant, especially with a large number of control points.
-  *%(SM_CONVEX_HULL)s:* This method can be used on an image whose objects are
   darker than their background and whose illumination intensity
   decreases monotonically from the brightest point. It proceeds as follows:
   
   -  Choose 256 evenly-spaced intensity levels between the minimum and
      maximum intensity for the image
   -  Set the intensity of the output image to the minimum intensity of
      the input image
   -  Iterate over the intensity levels, from lowest to highest
   -  For a given intensity, find all pixels with equal or higher
      intensities
   -  Find the convex hull that encloses those pixels
   -  Set the intensity of the output image within the convex hull to
      the current intensity

   The *%(SM_CONVEX_HULL)s* method is useful for calculating illumination correction
   images in empty brightfield images. It is a good option if the image contains a whole well.
   The edges of the well will be preserved, where there is a sharp transition in
   intensity, because there is no smoothing involved with this method.

**References**
-  J Lindblad and E Bengtsson (2001) “A comparison of methods for estimation
of intensity nonuniformities in 2D and 3D microscope images of fluorescence
stained cells.”, Proceedings of the 12th Scandinavian Conference on Image Analysis
(SCIA), pp. 264-271
"""
            % globals(),
        )

        self.automatic_object_width = Choice(
            "Method to calculate smoothing filter size",
            [FI_AUTOMATIC, FI_OBJECT_SIZE, FI_MANUALLY],
            doc="""\
*(Used only if a smoothing method other than Fit Polynomial is selected)*

Calculate the smoothing filter size. There are three options:

-  *%(FI_AUTOMATIC)s:* The size is computed as 1/40 the size of the
   image or 30 pixels, whichever is smaller.
-  *%(FI_OBJECT_SIZE)s:* The module will calculate the smoothing size
   based on the width of typical objects in your images.
-  *%(FI_MANUALLY)s:* You can enter a value yourself.
"""
            % globals(),
        )

        self.object_width = Integer(
            "Approximate object diameter",
            10,
            doc="""\
*(Used only if %(FI_OBJECT_SIZE)s is selected for smoothing filter size calculation)*

Enter the approximate diameter of typical objects, in pixels.
"""
            % globals(),
        )

        self.size_of_smoothing_filter = Integer(
            "Smoothing filter size",
            10,
            doc="""\
*(Used only if %(FI_MANUALLY)s is selected for smoothing filter size calculation)*

Enter the size of the desired smoothing filter, in pixels.
"""
            % globals(),
        )

        self.save_average_image = Binary(
            "Retain the averaged image?",
            False,
            doc="""\
The averaged image is the illumination function prior to dilation or
smoothing. It is an image produced during the calculations, not
typically needed for downstream modules. It can be helpful to retain it
in case you wish to try several different smoothing methods without
taking the time to recalculate the averaged image each time.

Select *Yes* to retain this averaged image. Use the **SaveImages**
module to save it to your hard drive.
"""
            % globals(),
        )

        self.average_image_name = ImageName(
            "Name the averaged image",
            "IllumBlueAvg",
            doc="""\
*(Used only if the averaged image is to be retained for later use in the pipeline)*

Enter a name that will allow the averaged image to be selected later in the pipeline.""",
        )

        self.save_dilated_image = Binary(
            "Retain the dilated image?",
            False,
            doc="""\
The dilated image is the illumination function after dilation but prior
to smoothing. It is an image produced during the calculations, and is
not typically needed for downstream modules.

Select *Yes* to retain this dilated image. Use the **SaveImages**
module to save it to your hard drive.
"""
            % globals(),
        )

        self.dilated_image_name = ImageName(
            "Name the dilated image",
            "IllumBlueDilated",
            doc="""\
*(Used only if the dilated image is to be retained for later use in the pipeline)*

Enter a name that will allow the dilated image to be selected later in
the pipeline.""",
        )

        self.automatic_splines = Binary(
            "Automatically calculate spline parameters?",
            True,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method)*

Select *Yes* to automatically calculate the parameters for spline
fitting.

Select *No* to specify the background mode, background threshold,
scale, maximum number of iterations and convergence.
"""
            % globals(),
        )

        self.spline_bg_mode = Choice(
            "Background mode",
            [
                centrosome.bg_compensate.MODE_AUTO,
                centrosome.bg_compensate.MODE_DARK,
                centrosome.bg_compensate.MODE_BRIGHT,
                centrosome.bg_compensate.MODE_GRAY,
            ],
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method
and spline parameters are not calculated automatically)*

This setting determines which pixels are background and which are
foreground.

-  *{auto}*: Determine the mode from the image. This will set
   the mode to {dark} if most of the pixels are dark,
   {bright} if most of the pixels are bright and %(MODE_GRAY)s
   if there are relatively few dark and light pixels relative to the
   number of mid-level pixels
-  *{dark}s*: Fit the spline to the darkest pixels in the image,
   excluding brighter pixels from consideration. This may be appropriate
   for a fluorescent image.
-  *{bright}*: Fit the spline to the lightest pixels in the
   image, excluding the darker pixels. This may be appropriate for a
   histologically stained image.
-  *{gray}*: Fit the spline to mid-range pixels, excluding both
   dark and light pixels. This may be appropriate for a brightfield
   image where the objects of interest have light and dark features.
""".format(
                auto=centrosome.bg_compensate.MODE_AUTO,
                bright=centrosome.bg_compensate.MODE_BRIGHT,
                dark=centrosome.bg_compensate.MODE_DARK,
                gray=centrosome.bg_compensate.MODE_GRAY,
            ),
        )

        self.spline_threshold = Float(
            "Background threshold",
            2,
            minval=0.1,
            maxval=5.0,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method
and spline parameters are not calculated automatically)*

This setting determines the cutoff used when excluding foreground
pixels from consideration. On each iteration, the method computes the
standard deviation of background pixels from the computed background.
The number entered in this setting is the number of standard
deviations a pixel can be from the computed background on the last
pass if it is to be considered as background during the next pass.

You should enter a higher number to converge stabily and slowly on a
final background and a lower number to converge more rapidly, but with
lower stability. The default for this parameter is two standard
deviations; this will provide a fairly stable, smooth background estimate.
"""
            % globals(),
        )

        self.spline_points = Integer(
            "Number of spline points",
            5,
            4,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method and
spline parameters are not calculated automatically)*

This is the number of control points for the spline. A value of 5
results in a 5x5 grid of splines across the image and is the value
suggested by the method’s authors. A lower value will give you a more
stable background while a higher one will fit variations in the
background more closely and take more time to compute.
"""
            % globals(),
        )

        self.spline_rescale = Float(
            "Image resampling factor",
            2,
            minval=1,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method and
spline parameters are not calculated automatically)*

This setting controls how the image is resampled to make a smaller
image. Resampling will speed up processing, but may degrade performance
if the resampling factor is larger than the diameter of foreground
objects. The image will be downsampled by the factor you enter. For
instance, a 500x600 image will be downsampled into a 250x300 image if a
factor of 2 is entered.
"""
            % globals(),
        )

        self.spline_maximum_iterations = Integer(
            "Maximum number of iterations",
            40,
            minval=1,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method and
spline parameters are not calculated automatically)*

This setting determines the maximum number of iterations of the
algorithm to be performed. The algorithm will perform fewer iterations
if it converges.
"""
            % globals(),
        )

        self.spline_convergence = Float(
            "Residual value for convergence",
            value=0.001,
            minval=0.00001,
            maxval=0.1,
            doc="""\
*(Used only if %(SM_SPLINES)s are selected for the smoothing method
and spline parameters are not calculated automatically)*

This setting determines the convergence criterion. The software sets
the convergence criterion to the number entered here times the signal
intensity; the convergence you enter is the fraction of the signal
intensity that indicates convergence. The algorithm derives a standard
deviation of the background pixels from the calculated background on
each iteration. The algorithm terminates when the difference between
the standard deviation for the current iteration and the previous
iteration is less than the convergence criterion.

Enter a smaller number for the convergence to calculate a more accurate
background. Enter a larger number to calculate the background using
fewer iterations, but less accuracy.
"""
            % globals(),
        )

    def settings(self):
        return [
            self.image_name,
            self.illumination_image_name,
            self.intensity_choice,
            self.dilate_objects,
            self.object_dilation_radius,
            self.block_size,
            self.rescale_option,
            self.each_or_all,
            self.smoothing_method,
            self.automatic_object_width,
            self.object_width,
            self.size_of_smoothing_filter,
            self.save_average_image,
            self.average_image_name,
            self.save_dilated_image,
            self.dilated_image_name,
            self.automatic_splines,
            self.spline_bg_mode,
            self.spline_points,
            self.spline_threshold,
            self.spline_rescale,
            self.spline_maximum_iterations,
            self.spline_convergence,
        ]

    def visible_settings(self):
        """The settings as seen by the UI

        """
        result = [self.image_name, self.illumination_image_name, self.intensity_choice]
        if self.intensity_choice == IC_REGULAR:
            result += [self.dilate_objects]
            if self.dilate_objects.value:
                result += [self.object_dilation_radius]
        elif self.smoothing_method != SM_SPLINES:
            result += [self.block_size]

        result += [self.rescale_option, self.each_or_all, self.smoothing_method]
        if self.smoothing_method in (SM_GAUSSIAN_FILTER, SM_MEDIAN_FILTER):
            result += [self.automatic_object_width]
            if self.automatic_object_width == FI_OBJECT_SIZE:
                result += [self.object_width]
            elif self.automatic_object_width == FI_MANUALLY:
                result += [self.size_of_smoothing_filter]
        elif self.smoothing_method == SM_SPLINES:
            result += [self.automatic_splines]
            if not self.automatic_splines:
                result += [
                    self.spline_bg_mode,
                    self.spline_points,
                    self.spline_threshold,
                    self.spline_rescale,
                    self.spline_maximum_iterations,
                    self.spline_convergence,
                ]
        result += [self.save_average_image]
        if self.save_average_image.value:
            result += [self.average_image_name]
        result += [self.save_dilated_image]
        if self.save_dilated_image.value:
            result += [self.dilated_image_name]
        return result

    def help_settings(self):
        return [
            self.image_name,
            self.illumination_image_name,
            self.intensity_choice,
            self.dilate_objects,
            self.object_dilation_radius,
            self.block_size,
            self.rescale_option,
            self.each_or_all,
            self.smoothing_method,
            self.automatic_object_width,
            self.object_width,
            self.size_of_smoothing_filter,
            self.automatic_splines,
            self.spline_bg_mode,
            self.spline_points,
            self.spline_threshold,
            self.spline_rescale,
            self.spline_maximum_iterations,
            self.spline_convergence,
            self.save_average_image,
            self.average_image_name,
            self.save_dilated_image,
            self.dilated_image_name,
        ]

    def prepare_group(self, workspace, grouping, image_numbers):
        image_set_list = workspace.image_set_list
        pipeline = workspace.pipeline
        assert isinstance(pipeline, Pipeline)
        m = workspace.measurements
        assert isinstance(m, Measurements)
        if self.each_or_all != EA_EACH and len(image_numbers) > 0:
            title = "#%d: CorrectIlluminationCalculate for %s" % (
                self.module_num,
                self.image_name,
            )
            message = (
                "CorrectIlluminationCalculate is averaging %d images while "
                "preparing for run" % (len(image_numbers))
            )
            output_image_provider = CorrectIlluminationImageProvider(
                self.illumination_image_name.value, self
            )
            d = self.get_dictionary(image_set_list)[OUTPUT_IMAGE] = {}
            if self.each_or_all == EA_ALL_FIRST:
                #
                # Find the module that provides the image we need
                #
                md = workspace.pipeline.get_provider_dictionary(
                    self.image_name.group, self
                )
                src_module, src_setting = md[self.image_name.value][-1]
                modules = list(pipeline.modules())
                idx = modules.index(src_module)
                last_module = modules[idx + 1]
                for w in pipeline.run_group_with_yield(
                    workspace, grouping, image_numbers, last_module, title, message
                ):
                    image = w.image_set.get_image(self.image_name.value, cache=False)
                    output_image_provider.add_image(image)
                    w.image_set.clear_cache()
            output_image_provider.serialize(d)

        return True

    def run(self, workspace):
        if self.each_or_all != EA_EACH:
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.deserialize(
                d, self
            )
            if self.each_or_all == EA_ALL_ACROSS:
                #
                # We are accumulating a pipeline image. Add this image set's
                # image to the output image provider.
                #
                orig_image = workspace.image_set.get_image(self.image_name.value)
                output_image_provider.add_image(orig_image)
                output_image_provider.serialize(d)

            # fetch images for display
            if (
                self.show_window
                or self.save_average_image
                or self.save_dilated_image
                or self.each_or_all == EA_ALL_FIRST
            ):
                avg_image = output_image_provider.provide_avg_image()
                dilated_image = output_image_provider.provide_dilated_image()
                workspace.image_set.add_provider(output_image_provider)
                output_image = output_image_provider.provide_image(workspace.image_set)
            else:
                workspace.image_set.add_provider(output_image_provider)
        else:
            orig_image = workspace.image_set.get_image(self.image_name.value)
            pixels = orig_image.pixel_data
            avg_image = self.preprocess_image_for_averaging(orig_image)
            dilated_image = self.apply_dilation(avg_image, orig_image)
            smoothed_image = self.apply_smoothing(dilated_image, orig_image)
            output_image = self.apply_scaling(smoothed_image, orig_image)
            # for illumination correction, we want the smoothed function to extend beyond the mask.
            output_image.mask = numpy.ones(output_image.pixel_data.shape[:2], bool)
            workspace.image_set.add(self.illumination_image_name.value, output_image)

        if self.save_average_image.value:
            workspace.image_set.add(self.average_image_name.value, avg_image)
        if self.save_dilated_image.value:
            workspace.image_set.add(self.dilated_image_name.value, dilated_image)
        if self.show_window:
            # store images for potential display
            workspace.display_data.avg_image = avg_image.pixel_data
            workspace.display_data.dilated_image = dilated_image.pixel_data
            workspace.display_data.output_image = output_image.pixel_data

    def is_aggregation_module(self):
        """Return True if aggregation is performed within a group"""
        return self.each_or_all != EA_EACH

    def post_group(self, workspace, grouping):
        """Handle tasks to be performed after a group has been processed

        For CorrectIllumninationCalculate, we make sure the current image
        set includes the aggregate image. "run" may not have run if an
        image was filtered out.
        """
        if self.each_or_all != EA_EACH:
            image_set = workspace.image_set
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.deserialize(
                d, self
            )
            assert isinstance(output_image_provider, CorrectIlluminationImageProvider)
            if not self.illumination_image_name.value in image_set.names:
                workspace.image_set.add_provider(output_image_provider)
            if (
                self.save_average_image
                and self.average_image_name.value not in image_set.names
            ):
                workspace.image_set.add(
                    self.average_image_name.value,
                    output_image_provider.provide_avg_image(),
                )
            if (
                self.save_dilated_image
                and self.dilated_image_name.value not in image_set.names
            ):
                workspace.image_set.add(
                    self.dilated_image_name.value,
                    output_image_provider.provide_dilated_image(),
                )

    def display(self, workspace, figure):
        # these are actually just the pixel data
        avg_image = workspace.display_data.avg_image
        dilated_image = workspace.display_data.dilated_image
        output_image = workspace.display_data.output_image

        figure.set_subplots((2, 2))

        def imshow(x, y, image, *args, **kwargs):
            if image.ndim == 2:
                f = figure.subplot_imshow_grayscale
            else:
                f = figure.subplot_imshow_color
            return f(x, y, image, *args, **kwargs)

        imshow(0, 0, avg_image, "Averaged image")
        pixel_data = output_image
        imshow(
            0,
            1,
            output_image,
            "Final illumination function",
            sharexy=figure.subplot(0, 0),
        )
        imshow(1, 0, dilated_image, "Dilated image", sharexy=figure.subplot(0, 0))
        statistics = [
            ["Min value", round(numpy.min(output_image), 2)],
            ["Max value", round(numpy.max(output_image), 2)],
            ["Calculation type", self.intensity_choice.value],
        ]
        if self.intensity_choice == IC_REGULAR:
            statistics.append(["Radius", self.object_dilation_radius.value])
        elif self.smoothing_method != SM_SPLINES:
            statistics.append(["Block size", self.block_size.value])
        statistics.append(["Rescaling?", self.rescale_option.value])
        statistics.append(["Each or all?", self.each_or_all.value])
        statistics.append(["Smoothing method", self.smoothing_method.value])
        statistics.append(
            [
                "Smoothing filter size",
                round(self.smoothing_filter_size(output_image.size), 2),
            ]
        )
        figure.subplot_table(
            1, 1, [[x[1]] for x in statistics], row_labels=[x[0] for x in statistics]
        )

    def apply_dilation(self, image, orig_image=None):
        """Return an image that is dilated according to the settings

        image - an instance of cpimage.Image

        returns another instance of cpimage.Image
        """
        if self.dilate_objects.value:
            #
            # This filter is designed to spread the boundaries of cells
            # and this "dilates" the cells
            #
            kernel = centrosome.smooth.circular_gaussian_kernel(
                self.object_dilation_radius.value, self.object_dilation_radius.value * 3
            )

            def fn(image):
                return scipy.ndimage.convolve(image, kernel, mode="constant", cval=0)

            if image.pixel_data.ndim == 2:
                dilated_pixels = centrosome.smooth.smooth_with_function_and_mask(
                    image.pixel_data, fn, image.mask
                )
            else:
                dilated_pixels = numpy.dstack(
                    [
                        centrosome.smooth.smooth_with_function_and_mask(
                            x, fn, image.mask
                        )
                        for x in image.pixel_data.transpose(2, 0, 1)
                    ]
                )
            return Image(dilated_pixels, parent_image=orig_image)
        else:
            return image

    def smoothing_filter_size(self, image_shape):
        """Return the smoothing filter size based on the settings and image size

        """
        if self.automatic_object_width == FI_MANUALLY:
            # Convert from full-width at half-maximum to standard deviation
            # (or so says CPsmooth.m)
            return self.size_of_smoothing_filter.value
        elif self.automatic_object_width == FI_OBJECT_SIZE:
            return self.object_width.value * 2.35 / 3.5
        elif self.automatic_object_width == FI_AUTOMATIC:
            return min(30, float(numpy.max(image_shape)) / 40.0)

    def preprocess_image_for_averaging(self, orig_image):
        """Create a version of the image appropriate for averaging

        """
        pixels = orig_image.pixel_data
        if self.intensity_choice == IC_REGULAR or self.smoothing_method == SM_SPLINES:
            if orig_image.has_mask:
                if pixels.ndim == 2:
                    pixels[~orig_image.mask] = 0
                else:
                    pixels[~orig_image.mask, :] = 0
                avg_image = Image(pixels, parent_image=orig_image)
            else:
                avg_image = orig_image
        else:
            # For background, we create a labels image using the block
            # size and find the minimum within each block.
            labels, indexes = centrosome.cpmorphology.block(
                pixels.shape[:2], (self.block_size.value, self.block_size.value)
            )
            if orig_image.has_mask:
                labels[~orig_image.mask] = -1

            min_block = numpy.zeros(pixels.shape)
            if pixels.ndim == 2:
                minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.minimum(pixels, labels, indexes)
                )
                min_block[labels != -1] = minima[labels[labels != -1]]
            else:
                for i in range(pixels.shape[2]):
                    minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                        scipy.ndimage.minimum(pixels[:, :, i], labels, indexes)
                    )
                    min_block[labels != -1, i] = minima[labels[labels != -1]]
            avg_image = Image(min_block, parent_image=orig_image)
        return avg_image

    def apply_smoothing(self, image, orig_image=None):
        """Return an image that is smoothed according to the settings

        image - an instance of cpimage.Image containing the pixels to analyze
        orig_image - the ancestor source image or None if ambiguous
        returns another instance of cpimage.Image
        """
        if self.smoothing_method == SM_NONE:
            return image

        pixel_data = image.pixel_data
        if pixel_data.ndim == 3:
            output_pixels = numpy.zeros(pixel_data.shape, pixel_data.dtype)
            for i in range(pixel_data.shape[2]):
                output_pixels[:, :, i] = self.smooth_plane(
                    pixel_data[:, :, i], image.mask
                )
        else:
            output_pixels = self.smooth_plane(pixel_data, image.mask)
        output_image = Image(output_pixels, parent_image=orig_image)
        return output_image

    def smooth_plane(self, pixel_data, mask):
        """Smooth one 2-d color plane of an image"""

        sigma = self.smoothing_filter_size(pixel_data.shape) / 2.35
        if self.smoothing_method == SM_FIT_POLYNOMIAL:
            output_pixels = centrosome.smooth.fit_polynomial(pixel_data, mask)
        elif self.smoothing_method == SM_GAUSSIAN_FILTER:
            #
            # Smoothing with the mask is good, even if there's no mask
            # because the mechanism undoes the edge effects that are introduced
            # by any choice of how to deal with border effects.
            #
            def fn(image):
                return scipy.ndimage.gaussian_filter(
                    image, sigma, mode="constant", cval=0
                )

            output_pixels = centrosome.smooth.smooth_with_function_and_mask(
                pixel_data, fn, mask
            )
        elif self.smoothing_method == SM_MEDIAN_FILTER:
            filter_sigma = max(1, int(sigma + 0.5))
            strel = centrosome.cpmorphology.strel_disk(filter_sigma)
            rescaled_pixel_data = pixel_data * 65535
            rescaled_pixel_data = rescaled_pixel_data.astype(numpy.uint16)
            rescaled_pixel_data *= mask
            output_pixels = skimage.filters.median(rescaled_pixel_data, strel, behavior="rank")
        elif self.smoothing_method == SM_TO_AVERAGE:
            mean = numpy.mean(pixel_data[mask])
            output_pixels = numpy.ones(pixel_data.shape, pixel_data.dtype) * mean
        elif self.smoothing_method == SM_SPLINES:
            output_pixels = self.smooth_with_splines(pixel_data, mask)
        elif self.smoothing_method == SM_CONVEX_HULL:
            output_pixels = self.smooth_with_convex_hull(pixel_data, mask)
        else:
            raise ValueError(
                "Unimplemented smoothing method: %s:" % self.smoothing_method.value
            )
        return output_pixels

    def smooth_with_convex_hull(self, pixel_data, mask):
        """Use the convex hull transform to smooth the image"""
        #
        # Apply an erosion, then the transform, then a dilation, heuristically
        # to ignore little spikey noisy things.
        #
        image = centrosome.cpmorphology.grey_erosion(pixel_data, 2, mask)
        image = centrosome.filter.convex_hull_transform(image, mask=mask)
        image = centrosome.cpmorphology.grey_dilation(image, 2, mask)
        return image

    def smooth_with_splines(self, pixel_data, mask):
        if self.automatic_splines:
            # Make the image 200 pixels long on its shortest side
            shortest_side = min(pixel_data.shape)
            if shortest_side < 200:
                scale = 1
            else:
                scale = float(shortest_side) / 200
            result = centrosome.bg_compensate.backgr(pixel_data, mask, scale=scale)
        else:
            mode = self.spline_bg_mode.value
            spline_points = self.spline_points.value
            threshold = self.spline_threshold.value
            convergence = self.spline_convergence.value
            iterations = self.spline_maximum_iterations.value
            rescale = self.spline_rescale.value
            result = centrosome.bg_compensate.backgr(
                pixel_data,
                mask,
                mode=mode,
                thresh=threshold,
                splinepoints=spline_points,
                scale=rescale,
                maxiter=iterations,
                convergence=convergence,
            )
        #
        # The result is a fit to the background intensity, but we
        # want to normalize the intensity by subtraction, leaving
        # the mean intensity alone.
        #
        mean_intensity = numpy.mean(result[mask])
        result[mask] -= mean_intensity
        return result

    def apply_scaling(self, image, orig_image=None):
        """Return an image that is rescaled according to the settings

        image - an instance of cpimage.Image
        returns another instance of cpimage.Image
        """
        if self.rescale_option == "No":
            return image

        def scaling_fn_2d(pixel_data):
            if image.has_mask:
                sorted_pixel_data = pixel_data[(pixel_data > 0) & image.mask]
            else:
                sorted_pixel_data = pixel_data[pixel_data > 0]
            if sorted_pixel_data.shape[0] == 0:
                return pixel_data
            sorted_pixel_data.sort()
            if self.rescale_option == "Yes":
                idx = int(sorted_pixel_data.shape[0] * ROBUST_FACTOR)
                robust_minimum = sorted_pixel_data[idx]
                pixel_data = pixel_data.copy()
                pixel_data[pixel_data < robust_minimum] = robust_minimum
            elif self.rescale_option == RE_MEDIAN:
                idx = int(sorted_pixel_data.shape[0] / 2)
                robust_minimum = sorted_pixel_data[idx]
            if robust_minimum == 0:
                return pixel_data
            return pixel_data / robust_minimum

        if image.pixel_data.ndim == 2:
            output_pixels = scaling_fn_2d(image.pixel_data)
        else:
            output_pixels = numpy.dstack(
                [scaling_fn_2d(x) for x in image.pixel_data.transpose(2, 0, 1)]
            )
        output_image = Image(output_pixels, parent_image=orig_image)
        return output_image

    def validate_module(self, pipeline):
        """Produce error if 'All:First' is selected and input image is not provided by the file image provider."""
        if (
            not pipeline.is_image_from_file(self.image_name.value)
            and self.each_or_all == EA_ALL_FIRST
        ):
            raise ValidationError(
                "All: First cycle requires that the input image be provided by the Input modules, or LoadImages/LoadData.",
                self.each_or_all,
            )

        """Modify the image provider attributes based on other setttings"""
        d = self.illumination_image_name.provided_attributes
        if self.each_or_all == EA_ALL_ACROSS:
            d["available_on_last"] = True
        elif "available_on_last" in d:
            del d["available_on_last"]

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if self.each_or_all == EA_ALL_FIRST:
            raise ValidationError(
                "Pre-calculation of the illumination function is time-intensive, especially for Test Mode. The analysis will proceed, but consider using '%s' instead."
                % EA_ALL_ACROSS,
                self.each_or_all,
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting values of old versions

        setting_values - sequence of strings that are the values for our settings
        variable_revision_number - settings were saved by module with this
                                   variable revision number
        module_name - name of module that did the saving
        returns upgraded setting values and upgraded variable revision number
        pyCellProfiler variable revision number 1 supported.
        """

        if variable_revision_number == 1:
            # Added spline parameters
            setting_values = setting_values + [
                "Yes",  # automatic_splines
                centrosome.bg_compensate.MODE_AUTO,  # spline_bg_mode
                "5",  # spline points
                "2",  # spline threshold
                "2",  # spline rescale
                "40",  # spline maximum iterations
                "0.001",
            ]  # spline convergence
            variable_revision_number = 2

        return setting_values, variable_revision_number

    def post_pipeline_load(self, pipeline):
        """After loading, set each_or_all appropriately

        This function handles the legacy EA_ALL which guessed the user's
        intent: processing before the first cycle or not. We look for
        the image provider and see if it is a file image provider.
        """
        if self.each_or_all == EA_ALL:
            if pipeline.is_image_from_file(self.image_name.value):
                self.each_or_all.value = EA_ALL_FIRST
            else:
                self.each_or_all.value = EA_ALL_ACROSS


class CorrectIlluminationImageProvider(AbstractImage):
    """CorrectIlluminationImageProvider provides the illumination correction image

    This class accumulates the image data from successive images and
    calculates the illumination correction image when asked.
    """

    def __init__(self, name, module):
        super(CorrectIlluminationImageProvider, self).__init__()
        self.__name = name
        self.__module = module
        self.__dirty = False
        self.__image_sum = None
        self.__mask_count = None
        self.__cached_image = None
        self.__cached_avg_image = None
        self.__cached_dilated_image = None
        self.__cached_mask_count = None

    D_NAME = "name"
    D_IMAGE_SUM = "image_sum"
    D_MASK_COUNT = "mask_count"

    def serialize(self, d):
        """Save the internal state of the provider to a dictionary

        d - save to this dictionary, numpy arrays and json serializable only
        """
        d[self.D_NAME] = self.__name
        d[self.D_IMAGE_SUM] = self.__image_sum
        d[self.D_MASK_COUNT] = self.__mask_count

    @staticmethod
    def deserialize(d, module):
        """Restore a state saved by serialize

        d - dictionary containing the state
        module - the module providing details on how to perform the correction

        returns a provider set up with the restored state
        """
        provider = CorrectIlluminationImageProvider(
            d[CorrectIlluminationImageProvider.D_NAME], module
        )
        provider.__dirty = True
        provider.__image_sum = d[CorrectIlluminationImageProvider.D_IMAGE_SUM]
        provider.__mask_count = d[CorrectIlluminationImageProvider.D_MASK_COUNT]
        return provider

    def add_image(self, image):
        """Accumulate the data from the given image

        image - an instance of cellprofiler.cpimage.Image, including
                image data and a mask
        """
        self.__dirty = True
        pimage = self.__module.preprocess_image_for_averaging(image)
        pixel_data = pimage.pixel_data
        if self.__image_sum is None:
            self.__image_sum = numpy.zeros(pixel_data.shape, pixel_data.dtype)
            self.__mask_count = numpy.zeros(pixel_data.shape[:2], numpy.int32)
        if image.has_mask:
            mask = image.mask
            if self.__image_sum.ndim == 2:
                self.__image_sum[mask] = self.__image_sum[mask] + pixel_data[mask]
            else:
                self.__image_sum[mask, :] = (
                    self.__image_sum[mask, :] + pixel_data[mask, :]
                )
            self.__mask_count[mask] = self.__mask_count[mask] + 1
        else:
            self.__image_sum = self.__image_sum + pixel_data
            self.__mask_count = self.__mask_count + 1

    def reset(self):
        """Reset the image sum at the start of a group"""
        self.__image_sum = None
        self.__cached_image = None
        self.__cached_avg_image = None
        self.__cached_dilated_image = None
        self.__cached_mask_count = None

    def provide_image(self, image_set):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_image

    def get_name(self):
        return self.__name

    def provide_avg_image(self):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_avg_image

    def provide_dilated_image(self):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_dilated_image

    def calculate_image(self):
        pixel_data = numpy.zeros(self.__image_sum.shape, self.__image_sum.dtype)
        mask = self.__mask_count > 0
        if pixel_data.ndim == 2:
            pixel_data[mask] = self.__image_sum[mask] / self.__mask_count[mask]
        else:
            for i in range(pixel_data.shape[2]):
                pixel_data[mask, i] = (
                    self.__image_sum[mask, i] / self.__mask_count[mask]
                )
        self.__cached_avg_image = Image(pixel_data, mask)
        self.__cached_dilated_image = self.__module.apply_dilation(
            self.__cached_avg_image
        )
        smoothed_image = self.__module.apply_smoothing(self.__cached_dilated_image)
        self.__cached_image = self.__module.apply_scaling(smoothed_image)
        self.__dirty = False

    def release_memory(self):
        # Memory is released during reset(), so this is a no-op
        pass


class CorrectIlluminationAvgImageProvider(AbstractImage):
    """Provide the image after averaging but before dilation and smoothing"""

    def __init__(self, name, ci_provider):
        """Construct using a parent provider that does the real work

        name - name of the image provided
        ci_provider - a CorrectIlluminationProvider that does the actual
                      accumulation and calculation
        """
        super(CorrectIlluminationAvgImageProvider, self).__init__()
        self.__name = name
        self.__ci_provider = ci_provider

    def provide_image(self, image_set):
        return self.__ci_provider.provide_avg_image()

    def get_name(self):
        return self.__name


class CorrectIlluminationDilatedImageProvider(AbstractImage):
    """Provide the image after averaging but before dilation and smoothing"""

    def __init__(self, name, ci_provider):
        """Construct using a parent provider that does the real work

        name - name of the image provided
        ci_provider - a CorrectIlluminationProvider that does the actual
                      accumulation and calculation
        """
        super(CorrectIlluminationDilatedImageProvider, self).__init__()
        self.__name = name
        self.__ci_provider = ci_provider

    def provide_image(self, image_set):
        return self.__ci_provider.provide_dilated_image()

    def get_name(self):
        return self.__name
