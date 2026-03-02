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

import numpy
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
from cellprofiler_library.opts.correctilluminationapply import Method as CorrectIlluminationApplyMethod
from cellprofiler_library.opts.correctilluminationcalculate import (
    IntensityChoice,
    RescaleIlluminationFunction,
    CalculateFunctionTarget,
    SmoothingMethod,
    SmoothingFilterSize,
    SplineBackgroundMode,
)
from cellprofiler_library.functions.image_processing import get_smoothing_filter_size
from cellprofiler_library.modules._correctilluminationcalculate import (
    apply_smoothing,
    apply_dilation,
    apply_scaling,
    preprocess_image_for_averaging,
    initialize_illumination_accumulation,
    accumulate_illumination_image,
    calculate_average_from_state,
)

EA_ALL = "All"

OUTPUT_IMAGE = "OutputImage"

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
            [IntensityChoice.REGULAR.value, IntensityChoice.BACKGROUND.value],
            IntensityChoice.REGULAR.value,
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
                    "IC_REGULAR": IntensityChoice.REGULAR.value,
                    "EA_ALL": EA_ALL,
                    "EA_EACH": CalculateFunctionTarget.EACH.value,
                    "SM_NONE": SmoothingMethod.NONE.value,
                    "IC_BACKGROUND": IntensityChoice.BACKGROUND.value,
                    "DOS_SUBTRACT": CorrectIlluminationApplyMethod.SUBTRACT.value,
                }
            ),
        )

        self.dilate_objects = Binary(
            "Dilate objects in the final averaged image?",
            False,
            doc="""\
*(Used only if the “{IC_REGULAR}” method is selected)*

For some applications, the incoming images are binary and each object
should be dilated with a Gaussian filter in the final averaged
(projection) image. This is for a sophisticated method of illumination
correction where model objects are produced. Select *Yes* to dilate
objects for this approach.
""".format(
                **{"IC_REGULAR": IntensityChoice.REGULAR.value}
),
        )

        self.object_dilation_radius = Integer(
            "Dilation radius",
            1,
            0,
            doc="""\
*(Used only if the “{IC_REGULAR}” method and dilation is selected)*

This value should be roughly equal to the original radius of the objects.
""".format(
                **{"IC_REGULAR": IntensityChoice.REGULAR.value}
),
        )

        self.block_size = Integer(
            "Block size",
            60,
            1,
            doc="""\
*(Used only if “{IC_BACKGROUND}” is selected)*

The block size should be large enough that every square block of pixels
is likely to contain some background pixels, where no objects are
located.
""".format(
                **{"IC_BACKGROUND": IntensityChoice.BACKGROUND.value}
),
        )

        self.rescale_option = Choice(
            "Rescale the illumination function?",
            ["Yes", "No", RescaleIlluminationFunction.MEDIAN.value],
            doc="""\
The illumination function can be rescaled so that the pixel intensities
are all equal to or greater than 1. You have the following options:

-  *Yes:* Rescaling is recommended if you plan to use the
   *{IC_REGULAR}* method (and hence, the *{DOS_DIVIDE}* option in
   **CorrectIlluminationApply**). Rescaling the illumination function to
   >1 ensures that the values in your corrected image will stay between
   0-1 after division.
-  *No:* Rescaling is not recommended if you plan to use the
   *{IC_BACKGROUND}* method, which is paired with the
   *{DOS_SUBTRACT}* option in **CorrectIlluminationApply**. Because
   rescaling causes the illumination function to have values from 1 to
   infinity, subtracting those values from your image would cause the
   corrected images to be very dark, even negative.
-  {RE_MEDIAN}\ *:* This option chooses the median value in the image
   to rescale so that division increases some values and decreases others.
""".format(
                **{
                    "IC_REGULAR": IntensityChoice.REGULAR.value, 
                    "IC_BACKGROUND": IntensityChoice.BACKGROUND.value,
                    "DOS_SUBTRACT": CorrectIlluminationApplyMethod.SUBTRACT.value,
                    "DOS_DIVIDE": CorrectIlluminationApplyMethod.DIVIDE.value,
                    "RE_MEDIAN": RescaleIlluminationFunction.MEDIAN.value,
                    }  
),
        )

        self.each_or_all = Choice(
            "Calculate function for each image individually, or based on all images?",
            [CalculateFunctionTarget.EACH.value, CalculateFunctionTarget.ALL_FIRST.value, CalculateFunctionTarget.ALL_ACROSS.value],
            doc="""\
Calculate a separate function for each image, or one for all the
images? You can calculate the illumination function using just the
current image or you can calculate the illumination function using all
of the images in each group (or in the entire experiment). The
illumination function can be calculated in one of the three ways:

-  *{EA_EACH}:* Calculate an illumination function for each image
   individually.
-  *{EA_ALL_FIRST}:* Calculate an illumination function based on all
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
-  *{EA_ALL_ACROSS}:* Calculate an illumination function across all
   cycles in each group. This option takes any image as input; however,
   the illumination function will not be completed until the end of the
   last cycle in the group. You can use **SaveImages** to save the
   illumination function after the last cycle in the group and then use
   the resulting image in another pipeline. The option is useful if you
   want to exclude images that are filtered by a prior **FlagImage**
   module.
""".format(
                **{
                    "EA_EACH": CalculateFunctionTarget.EACH.value, 
                    "EA_ALL_FIRST": CalculateFunctionTarget.ALL_FIRST.value, 
                    "EA_ALL_ACROSS": CalculateFunctionTarget.ALL_ACROSS.value
                   }
),
        )
        self.smoothing_method = Choice(
            "Smoothing method",
            [
                SmoothingMethod.NONE.value,
                SmoothingMethod.CONVEX_HULL.value,
                SmoothingMethod.FIT_POLYNOMIAL.value,
                SmoothingMethod.MEDIAN_FILTER.value,
                SmoothingMethod.GAUSSIAN_FILTER.value,
                SmoothingMethod.TO_AVERAGE.value,
                SmoothingMethod.SPLINES.value,
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

-  *{SM_FIT_POLYNOMIAL}:* This method is fastest but does not allow
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
-  *{SM_MEDIAN_FILTER}* and *{SM_GAUSSIAN_FILTER}:*
   We typically recommend
   *{SM_MEDIAN_FILTER}* vs. *{SM_GAUSSIAN_FILTER}* because the
   median is less sensitive to outliers, although the results are also
   slightly less smooth and the fact that images are in the range of 0
   to 1 means that outliers typically will not dominate too strongly
   anyway. The *{SM_GAUSSIAN_FILTER}* convolves the image with a
   Gaussian whose full width at half maximum is the artifact diameter
   entered. Its effect is to blur and obscure features smaller than the
   specified diameter and spread bright or dim features larger than the
   specified diameter. The *{SM_MEDIAN_FILTER}* finds the median pixel value within
   the diameter you specify. It removes bright or dim features
   that are significantly smaller than the specified diameter.
-  *{SM_TO_AVERAGE}:* A less commonly used option is to completely
   smooth the entire image, which will create a flat, smooth image where
   every pixel of the image is the average of what the illumination
   function would otherwise have been.
-  *{SM_SPLINES}:* This method (*Lindblad and Bengtsson, 2001*) fits
   a grid of cubic splines to the background while excluding foreground
   pixels from the calculation. It operates iteratively, classifying
   pixels as background, computing a best fit spline to this background
   and then reclassifying pixels as background until the spline
   converges on its final value. This method is best for backgrounds that
   are highly variable and irregular. Note that the computation time can
   be significant, especially with a large number of control points.
-  *{SM_CONVEX_HULL}:* This method can be used on an image whose objects are
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

   The *{SM_CONVEX_HULL}* method is useful for calculating illumination correction
   images in empty brightfield images. It is a good option if the image contains a whole well.
   The edges of the well will be preserved, where there is a sharp transition in
   intensity, because there is no smoothing involved with this method.

**References**
-  J Lindblad and E Bengtsson (2001) “A comparison of methods for estimation
of intensity nonuniformities in 2D and 3D microscope images of fluorescence
stained cells.”, Proceedings of the 12th Scandinavian Conference on Image Analysis
(SCIA), pp. 264-271
""".format(
                **{
                    "SM_FIT_POLYNOMIAL": SmoothingMethod.FIT_POLYNOMIAL.value,
                    "SM_MEDIAN_FILTER": SmoothingMethod.MEDIAN_FILTER.value,
                    "SM_GAUSSIAN_FILTER": SmoothingMethod.GAUSSIAN_FILTER.value,
                    "SM_TO_AVERAGE": SmoothingMethod.TO_AVERAGE.value,
                    "SM_SPLINES": SmoothingMethod.SPLINES.value,
                    "SM_CONVEX_HULL": SmoothingMethod.CONVEX_HULL.value,
                }
),
        )

        self.automatic_object_width = Choice(
            "Method to calculate smoothing filter size",
            [SmoothingFilterSize.AUTOMATIC.value, SmoothingFilterSize.OBJECT_SIZE.value, SmoothingFilterSize.MANUALLY.value],
            doc="""\
*(Used only if a smoothing method other than Fit Polynomial is selected)*

Calculate the smoothing filter size. There are three options:

-  *{FI_AUTOMATIC}:* The size is computed as 1/40 the size of the
   image or 30 pixels, whichever is smaller.
-  *{FI_OBJECT_SIZE}:* The module will calculate the smoothing size
   based on the width of typical objects in your images.
-  *{FI_MANUALLY}:* You can enter a value yourself.
""".format(
                **{
                    "FI_AUTOMATIC": SmoothingFilterSize.AUTOMATIC.value,
                    "FI_OBJECT_SIZE": SmoothingFilterSize.OBJECT_SIZE.value,
                    "FI_MANUALLY": SmoothingFilterSize.MANUALLY.value,
                }
),
        )

        self.object_width = Integer(
            "Approximate object diameter",
            10,
            doc="""\
*(Used only if {FI_OBJECT_SIZE} is selected for smoothing filter size calculation)*

Enter the approximate diameter of typical objects, in pixels.
""".format(
                **{"FI_OBJECT_SIZE": SmoothingFilterSize.OBJECT_SIZE.value}
),
        )

        self.size_of_smoothing_filter = Integer(
            "Smoothing filter size",
            10,
            doc="""\
*(Used only if {FI_MANUALLY} is selected for smoothing filter size calculation)*

Enter the size of the desired smoothing filter, in pixels.
""".format(
                **{"FI_MANUALLY": SmoothingFilterSize.MANUALLY.value}
),
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
""",
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
""",
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
*(Used only if {SM_SPLINES} are selected for the smoothing method)*

Select *Yes* to automatically calculate the parameters for spline
fitting.

Select *No* to specify the background mode, background threshold,
scale, maximum number of iterations and convergence.
""".format(
                **{"SM_SPLINES": SmoothingMethod.SPLINES.value,}
),
        )

        self.spline_bg_mode = Choice(
            "Background mode",
            [
                SplineBackgroundMode.AUTO.value,
                SplineBackgroundMode.DARK.value,
                SplineBackgroundMode.BRIGHT.value,
                SplineBackgroundMode.GRAY.value,
            ],
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method
and spline parameters are not calculated automatically)*

This setting determines which pixels are background and which are
foreground.

-  *{MODE_AUTO}*: Determine the mode from the image. This will set
   the mode to {MODE_DARK} if most of the pixels are dark,
   {MODE_BRIGHT} if most of the pixels are bright and {MODE_GRAY}
   if there are relatively few dark and light pixels relative to the
   number of mid-level pixels
-  *{MODE_DARK}s*: Fit the spline to the darkest pixels in the image,
   excluding brighter pixels from consideration. This may be appropriate
   for a fluorescent image.
-  *{MODE_BRIGHT}*: Fit the spline to the lightest pixels in the
   image, excluding the darker pixels. This may be appropriate for a
   histologically stained image.
-  *{MODE_GRAY}*: Fit the spline to mid-range pixels, excluding both
   dark and light pixels. This may be appropriate for a brightfield
   image where the objects of interest have light and dark features.
""".format(
                **{
                    "SM_SPLINES": SmoothingMethod.SPLINES.value,
                    "MODE_AUTO": SplineBackgroundMode.AUTO.value, 
                    "MODE_BRIGHT": SplineBackgroundMode.BRIGHT.value, 
                    "MODE_DARK": SplineBackgroundMode.DARK.value, 
                    "MODE_GRAY": SplineBackgroundMode.GRAY.value
                    }
                
            ),
        )

        self.spline_threshold = Float(
            "Background threshold",
            2,
            minval=0.1,
            maxval=5.0,
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method
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
""".format(
            **{"SM_SPLINES": SmoothingMethod.SPLINES.value}
    ),
        )

        self.spline_points = Integer(
            "Number of spline points",
            5,
            4,
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method and
spline parameters are not calculated automatically)*

This is the number of control points for the spline. A value of 5
results in a 5x5 grid of splines across the image and is the value
suggested by the method’s authors. A lower value will give you a more
stable background while a higher one will fit variations in the
background more closely and take more time to compute.
""".format(
                **{"SM_SPLINES": SmoothingMethod.SPLINES.value}
),
        )

        self.spline_rescale = Float(
            "Image resampling factor",
            2,
            minval=1,
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method and
spline parameters are not calculated automatically)*

This setting controls how the image is resampled to make a smaller
image. Resampling will speed up processing, but may degrade performance
if the resampling factor is larger than the diameter of foreground
objects. The image will be downsampled by the factor you enter. For
instance, a 500x600 image will be downsampled into a 250x300 image if a
factor of 2 is entered.
""".format(
                **{"SM_SPLINES": SmoothingMethod.SPLINES.value}
),
        )

        self.spline_maximum_iterations = Integer(
            "Maximum number of iterations",
            40,
            minval=1,
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method and
spline parameters are not calculated automatically)*

This setting determines the maximum number of iterations of the
algorithm to be performed. The algorithm will perform fewer iterations
if it converges.
""".format(
                **{"SM_SPLINES": SmoothingMethod.SPLINES.value}
),
        )

        self.spline_convergence = Float(
            "Residual value for convergence",
            value=0.001,
            minval=0.00001,
            maxval=0.1,
            doc="""\
*(Used only if {SM_SPLINES} are selected for the smoothing method
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
""".format(
                **{"SM_SPLINES": SmoothingMethod.SPLINES.value}
),
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
        if self.intensity_choice == IntensityChoice.REGULAR.value:
            result += [self.dilate_objects]
            if self.dilate_objects.value:
                result += [self.object_dilation_radius]
        elif self.smoothing_method != SmoothingMethod.SPLINES.value:
            result += [self.block_size]

        result += [self.rescale_option, self.each_or_all, self.smoothing_method]
        if self.smoothing_method in (SmoothingMethod.GAUSSIAN_FILTER.value, SmoothingMethod.MEDIAN_FILTER.value):
            result += [self.automatic_object_width]
            if self.automatic_object_width == SmoothingFilterSize.OBJECT_SIZE.value:
                result += [self.object_width]
            elif self.automatic_object_width == SmoothingFilterSize.MANUALLY.value:
                result += [self.size_of_smoothing_filter]
        elif self.smoothing_method == SmoothingMethod.SPLINES.value:
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
        if self.each_or_all != CalculateFunctionTarget.EACH.value and len(image_numbers) > 0:
            title = "#%d: CorrectIlluminationCalculate for %s" % (
                self.module_num,
                self.image_name,
            )
            message = (
                "CorrectIlluminationCalculate is averaging %d images while "
                "preparing for run" % (len(image_numbers))
            )
            output_image_provider = CorrectIlluminationImageProvider.create(
                self.illumination_image_name.value, self
            )
            d = self.get_dictionary(image_set_list)[OUTPUT_IMAGE] = {}
            if self.each_or_all == CalculateFunctionTarget.ALL_FIRST.value:
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
                    if not output_image_provider.has_image:
                        output_image_provider.set_image(image)
                    else:
                        output_image_provider.accumulate_image(image)
                    w.image_set.clear_cache()
            output_image_provider.save_state(d)

        return True

    def run(self, workspace):
        if self.each_or_all != CalculateFunctionTarget.EACH.value:
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.restore_from_state(d)
            if self.each_or_all == CalculateFunctionTarget.ALL_ACROSS.value:
                #
                # We are accumulating a pipeline image. Add this image set's
                # image to the output image provider.
                #
                orig_image = workspace.image_set.get_image(self.image_name.value)
                if not output_image_provider.has_image:
                    output_image_provider.set_image(orig_image)
                else:
                    output_image_provider.accumulate_image(orig_image)
                output_image_provider.save_state(d)

            # fetch images for display
            if (
                self.show_window
                or self.save_average_image
                or self.save_dilated_image
                or self.each_or_all == CalculateFunctionTarget.ALL_FIRST.value
            ):
                avg_image = output_image_provider.provide_avg_image()
                dilated_image = output_image_provider.provide_dilated_image()
                workspace.image_set.add_provider(output_image_provider)
                output_image = output_image_provider.provide_image(workspace.image_set)
            else:
                workspace.image_set.add_provider(output_image_provider)
        else:
            orig_image = workspace.image_set.get_image(self.image_name.value)
            output_image_provider = CorrectIlluminationImageProvider.create(
                self.illumination_image_name.value, self
            )
            output_image_provider.set_image(orig_image)
            avg_image = output_image_provider.provide_avg_image()
            dilated_image = output_image_provider.provide_dilated_image()
            output_image = output_image_provider.provide_image(workspace.image_set)
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
        return self.each_or_all != CalculateFunctionTarget.EACH.value

    def post_group(self, workspace, grouping):
        """Handle tasks to be performed after a group has been processed

        For CorrectIllumninationCalculate, we make sure the current image
        set includes the aggregate image. "run" may not have run if an
        image was filtered out.
        """
        if self.each_or_all != CalculateFunctionTarget.EACH.value:
            image_set = workspace.image_set
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.restore_from_state(d)
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
        if self.intensity_choice == IntensityChoice.REGULAR.value:
            statistics.append(["Radius", self.object_dilation_radius.value])
        elif self.smoothing_method != SmoothingMethod.SPLINES.value:
            statistics.append(["Block size", self.block_size.value])
        statistics.append(["Rescaling?", self.rescale_option.value])
        statistics.append(["Each or all?", self.each_or_all.value])
        statistics.append(["Smoothing method", self.smoothing_method.value])
        statistics.append(
            [
                "Smoothing filter size",
                round(
                    get_smoothing_filter_size(
                        self.automatic_object_width.value,
                        self.size_of_smoothing_filter.value,
                        self.object_width.value,
                        output_image.size,
                    ),
                    2,
                ),
            ]
        )
        figure.subplot_table(
            1, 1, [[x[1]] for x in statistics], row_labels=[x[0] for x in statistics]
        )

    def validate_module(self, pipeline):
        """Produce error if 'All:First' is selected and input image is not provided by the file image provider."""
        if (
            not pipeline.is_image_from_file(self.image_name.value)
            and self.each_or_all == CalculateFunctionTarget.ALL_FIRST.value
        ):
            raise ValidationError(
                "All: First cycle requires that the input image be provided by the Input modules, or LoadImages/LoadData.",
                self.each_or_all,
            )

        """Modify the image provider attributes based on other setttings"""
        d = self.illumination_image_name.provided_attributes
        if self.each_or_all == CalculateFunctionTarget.ALL_ACROSS.value:
            d["available_on_last"] = True
        elif "available_on_last" in d:
            del d["available_on_last"]

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if self.each_or_all == CalculateFunctionTarget.ALL_FIRST.value:
            raise ValidationError(
                "Pre-calculation of the illumination function is time-intensive, especially for Test Mode. The analysis will proceed, but consider using '%s' instead."
                % CalculateFunctionTarget.ALL_ACROSS.value,
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
                SplineBackgroundMode.AUTO,  # spline_bg_mode
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
                self.each_or_all.value = CalculateFunctionTarget.ALL_FIRST.value
            else:
                self.each_or_all.value = CalculateFunctionTarget.ALL_ACROSS.value


class CorrectIlluminationImageProvider(AbstractImage):
    """Provides the illumination correction image.

    Accumulates image data from successive images and calculates the
    illumination correction image on demand.  Follows the same style as
    MakeProjection.ImageProvider – self-contained, no back-reference to
    the module.
    """
    D_NAME = "name"
    D_LIBRARY_STATE = "library_state"
    D_INTENSITY_CHOICE = "intensity_choice"
    D_DILATE_OBJECTS = "dilate_objects"
    D_OBJECT_DILATION_RADIUS = "object_dilation_radius"
    D_BLOCK_SIZE = "block_size"
    D_RESCALE_OPTION = "rescale_option"
    D_SMOOTHING_METHOD = "smoothing_method"
    D_AUTOMATIC_OBJECT_WIDTH = "automatic_object_width"
    D_SIZE_OF_SMOOTHING_FILTER = "size_of_smoothing_filter"
    D_OBJECT_WIDTH = "object_width"
    D_AUTOMATIC_SPLINES = "automatic_splines"
    D_SPLINE_BG_MODE = "spline_bg_mode"
    D_SPLINE_POINTS = "spline_points"
    D_SPLINE_THRESHOLD = "spline_threshold"
    D_SPLINE_CONVERGENCE = "spline_convergence"
    D_SPLINE_MAXIMUM_ITERATIONS = "spline_maximum_iterations"
    D_SPLINE_RESCALE = "spline_rescale"

    def __init__(
        self,
        name,
        intensity_choice,
        dilate_objects,
        object_dilation_radius,
        block_size,
        rescale_option,
        smoothing_method,
        automatic_object_width,
        size_of_smoothing_filter,
        object_width,
        automatic_splines,
        spline_bg_mode,
        spline_points,
        spline_threshold,
        spline_convergence,
        spline_maximum_iterations,
        spline_rescale,
    ):
        super(CorrectIlluminationImageProvider, self).__init__()
        # TODO: Do we need so many state variables in addition to the library_state? Why not place these inside of it?
        self._name = name
        self.intensity_choice = intensity_choice
        self.dilate_objects = dilate_objects
        self.object_dilation_radius = object_dilation_radius
        self.block_size = block_size
        self.rescale_option = rescale_option
        self.smoothing_method = smoothing_method
        self.automatic_object_width = automatic_object_width
        self.size_of_smoothing_filter = size_of_smoothing_filter
        self.object_width = object_width
        self.automatic_splines = automatic_splines
        self.spline_bg_mode = spline_bg_mode
        self.spline_points = spline_points
        self.spline_threshold = spline_threshold
        self.spline_convergence = spline_convergence
        self.spline_maximum_iterations = spline_maximum_iterations
        self.spline_rescale = spline_rescale
        self.library_state = {}
        self._cached_image = None
        self._cached_avg_image = None
        self._cached_dilated_image = None

    @staticmethod
    def create(name, module):
        """Factory method: extract settings from the module.

        Args:
            name: Name of the output illumination image.
            module: The CorrectIlluminationCalculate module instance.

        Returns:
            A new CorrectIlluminationImageProvider configured from module
            settings.
        """
        # TODO: look into splitting this into smaller providers with independent, single-purpose functions.
        return CorrectIlluminationImageProvider(
            name=name,
            intensity_choice=module.intensity_choice.value,
            dilate_objects=module.dilate_objects.value,
            object_dilation_radius=module.object_dilation_radius.value,
            block_size=module.block_size.value,
            rescale_option=module.rescale_option.value,
            smoothing_method=module.smoothing_method.value,
            automatic_object_width=module.automatic_object_width.value,
            size_of_smoothing_filter=module.size_of_smoothing_filter.value,
            object_width=module.object_width.value,
            automatic_splines=module.automatic_splines.value,
            spline_bg_mode=module.spline_bg_mode.value,
            spline_points=module.spline_points.value,
            spline_threshold=module.spline_threshold.value,
            spline_convergence=module.spline_convergence.value,
            spline_maximum_iterations=module.spline_maximum_iterations.value,
            spline_rescale=module.spline_rescale.value,
        )

    def save_state(self, d):
        """Save provider state to a dictionary for cross-cycle persistence.

        Args:
            d: Dictionary to write state into (numpy arrays and JSON-serializable
               values only).
        """
        # TODO: So many args needed? Look into splitting this into smaller providers with
        # TODO: independent, single-purpose functions.
        d[CorrectIlluminationImageProvider.D_NAME] = self._name
        d[CorrectIlluminationImageProvider.D_LIBRARY_STATE] = self.library_state
        d[CorrectIlluminationImageProvider.D_INTENSITY_CHOICE] = self.intensity_choice
        d[CorrectIlluminationImageProvider.D_DILATE_OBJECTS] = self.dilate_objects
        d[CorrectIlluminationImageProvider.D_OBJECT_DILATION_RADIUS] = self.object_dilation_radius
        d[CorrectIlluminationImageProvider.D_BLOCK_SIZE] = self.block_size
        d[CorrectIlluminationImageProvider.D_RESCALE_OPTION] = self.rescale_option
        d[CorrectIlluminationImageProvider.D_SMOOTHING_METHOD] = self.smoothing_method
        d[CorrectIlluminationImageProvider.D_AUTOMATIC_OBJECT_WIDTH] = self.automatic_object_width
        d[CorrectIlluminationImageProvider.D_SIZE_OF_SMOOTHING_FILTER] = self.size_of_smoothing_filter
        d[CorrectIlluminationImageProvider.D_OBJECT_WIDTH] = self.object_width
        d[CorrectIlluminationImageProvider.D_AUTOMATIC_SPLINES] = self.automatic_splines
        d[CorrectIlluminationImageProvider.D_SPLINE_BG_MODE] = self.spline_bg_mode
        d[CorrectIlluminationImageProvider.D_SPLINE_POINTS] = self.spline_points
        d[CorrectIlluminationImageProvider.D_SPLINE_THRESHOLD] = self.spline_threshold
        d[CorrectIlluminationImageProvider.D_SPLINE_CONVERGENCE] = self.spline_convergence
        d[CorrectIlluminationImageProvider.D_SPLINE_MAXIMUM_ITERATIONS] = self.spline_maximum_iterations
        d[CorrectIlluminationImageProvider.D_SPLINE_RESCALE] = self.spline_rescale

    @staticmethod
    def restore_from_state(d):
        """Reconstruct a provider from a previously saved state dictionary.

        Args:
            d: Dictionary from a prior call to save_state.

        Returns:
            A CorrectIlluminationImageProvider restored from d.
        """
        P = CorrectIlluminationImageProvider
        # TODO: So many args needed? Look into splitting this into smaller providers with 
        # TODO: independent, single-purpose functions.
        provider = P(
            name=d[P.D_NAME],
            intensity_choice=d[P.D_INTENSITY_CHOICE],
            dilate_objects=d[P.D_DILATE_OBJECTS],
            object_dilation_radius=d[P.D_OBJECT_DILATION_RADIUS],
            block_size=d[P.D_BLOCK_SIZE],
            rescale_option=d[P.D_RESCALE_OPTION],
            smoothing_method=d[P.D_SMOOTHING_METHOD],
            automatic_object_width=d[P.D_AUTOMATIC_OBJECT_WIDTH],
            size_of_smoothing_filter=d[P.D_SIZE_OF_SMOOTHING_FILTER],
            object_width=d[P.D_OBJECT_WIDTH],
            automatic_splines=d[P.D_AUTOMATIC_SPLINES],
            spline_bg_mode=d[P.D_SPLINE_BG_MODE],
            spline_points=d[P.D_SPLINE_POINTS],
            spline_threshold=d[P.D_SPLINE_THRESHOLD],
            spline_convergence=d[P.D_SPLINE_CONVERGENCE],
            spline_maximum_iterations=d[P.D_SPLINE_MAXIMUM_ITERATIONS],
            spline_rescale=d[P.D_SPLINE_RESCALE],
        )
        provider.library_state = d.get(P.D_LIBRARY_STATE, {})
        return provider

    def reset(self):
        """Reset accumulation at the start of a group."""
        self.library_state = {}
        self._cached_image = None
        self._cached_avg_image = None
        self._cached_dilated_image = None

    @property
    def has_image(self):
        """True when at least one image has been accumulated."""
        return len(self.library_state) > 0

    def set_image(self, image):
        """Initialize accumulation from the first image.

        Args:
            image: A cellprofiler_core.image.Image instance.
        """
        self._cached_image = None
        self._cached_avg_image = None
        self._cached_dilated_image = None
        mask = image.mask if image.has_mask else None
        preprocessed = preprocess_image_for_averaging(
            image.pixel_data,
            mask,
            self.intensity_choice,
            self.smoothing_method,
            self.block_size,
        )
        self.library_state = initialize_illumination_accumulation(
            preprocessed, mask
        )

    def accumulate_image(self, image):
        """Accumulate a subsequent image into the running state.

        Args:
            image: A cellprofiler_core.image.Image instance.
        """
        self._cached_image = None
        self._cached_avg_image = None
        self._cached_dilated_image = None
        mask = image.mask if image.has_mask else None
        preprocessed = preprocess_image_for_averaging(
            image.pixel_data,
            mask,
            self.intensity_choice,
            self.smoothing_method,
            self.block_size,
        )
        accumulate_illumination_image(
            preprocessed, mask, self.library_state
        )

    def provide_image(self, image_set):
        # TODO: Currently provide image is updating all caches. Is this necessary?
        """Return the final illumination correction Image.

        Computes (and caches) the full average -> dilation -> smoothing ->
        scaling pipeline on the first call; returns the cached result on
        subsequent calls.

        Args:
            image_set: The current image set (may be None when called
                internally to pre-compute intermediate images).

        Returns:
            cellprofiler_core.image.Image containing the illumination
            function pixel data.
        """
        if self._cached_image is not None:
            return self._cached_image

        # --- Step 1: compute average ---
        avg_pixel_data, avg_mask = calculate_average_from_state(self.library_state)
        self._cached_avg_image = Image(avg_pixel_data, avg_mask)

        # --- Step 2: apply dilation ---
        if self.dilate_objects:
            dilated_pixels = apply_dilation(
                avg_pixel_data, avg_mask, self.object_dilation_radius
            )
            self._cached_dilated_image = Image(
                dilated_pixels, parent_image=self._cached_avg_image
            )
        else:
            self._cached_dilated_image = self._cached_avg_image

        # --- Step 3: apply smoothing ---
        if self.smoothing_method != SmoothingMethod.NONE.value:
            dilated_image = self._cached_dilated_image
            smoothed_pixels = apply_smoothing(
                image_pixel_data=dilated_image.pixel_data,
                image_mask = dilated_image.mask if dilated_image.has_mask else None,
                smoothing_method=self.smoothing_method,
                automatic_object_width=self.automatic_object_width,
                size_of_smoothing_filter=self.size_of_smoothing_filter,
                object_width=self.object_width,
                image_shape=dilated_image.pixel_data.shape[:2],
                automatic_splines=self.automatic_splines,
                spline_bg_mode=self.spline_bg_mode,
                spline_points=self.spline_points,
                spline_threshold=self.spline_threshold,
                spline_convergence=self.spline_convergence,
                spline_maximum_iterations=self.spline_maximum_iterations,
                spline_rescale=self.spline_rescale,
            )
            smoothed_image = Image(smoothed_pixels, parent_image=self._cached_avg_image)
        else:
            smoothed_image = self._cached_dilated_image

        # --- Step 4: apply scaling ---
        if self.rescale_option != "No":
            output_pixels = apply_scaling(
                image_pixel_data=smoothed_image.pixel_data,
                image_mask=smoothed_image.mask if smoothed_image.has_mask else None,
                rescale_option=self.rescale_option,
            )
            self._cached_image = Image(output_pixels, parent_image=self._cached_avg_image)
        else:
            self._cached_image = smoothed_image

        return self._cached_image

    def provide_avg_image(self):
        """Return the averaged image (before dilation and smoothing)."""
        if self._cached_avg_image is None:
            # TODO: Function call below updates dilate as well and other caches too. Original code did not do this.
            self.provide_image(None)
        return self._cached_avg_image

    def provide_dilated_image(self):
        """Return the dilated image (after dilation, before smoothing)."""
        if self._cached_dilated_image is None:
            # TODO: Function call below updates avg as well and other caches too. Original code did not do this.
            self.provide_image(None)
        return self._cached_dilated_image

    def get_name(self):
        """Return the name of the output image."""
        return self._name

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
