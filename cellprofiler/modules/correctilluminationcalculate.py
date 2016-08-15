'''<b>Correct Illumination - Calculate</b> calculates an illumination function that is used to correct uneven
illumination/lighting/shading or to reduce uneven background in images.
<hr>
This module calculates an illumination function that can either be saved to the
hard drive for later use or immediately applied to images later in the
pipeline. This function will correct for the uneven illumination in images.
If saving, select <i>.mat</i> format in <b>SaveImages</b>.
Use the <b>CorrectIlluminationApply</b> module to apply the
function to the image to be corrected.

Illumination correction is a challenge to do properly; please see the
<a href="http://www.cellprofiler.org/examples.html">examples</a> and
<a href="http://cellprofiler.org/tutorials.html">tutorials</a> pages
on the CellProfiler website for further advice.

See also <b>CorrectIlluminationApply</b>, <b>EnhanceOrSuppressFeatures</b>.
'''

import centrosome.cpmorphology as cpmm
import numpy as np
import scipy.linalg
import scipy.ndimage as scind
import skimage.filters
from centrosome.bg_compensate import MODE_DARK, MODE_GRAY
from centrosome.bg_compensate import backgr, MODE_AUTO, MODE_BRIGHT
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import grey_erosion, grey_dilation, strel_disk
from centrosome.filter import convex_hull_transform
from centrosome.rankorder import rank_order
from centrosome.smooth import circular_gaussian_kernel
from centrosome.smooth import fit_polynomial
from centrosome.smooth import smooth_with_function_and_mask

import cellprofiler.image  as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
from cellprofiler.setting import YES, NO

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

ROBUST_FACTOR = .02  # For rescaling, take 2nd percentile value

OUTPUT_IMAGE = "OutputImage"

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"


class CorrectIlluminationCalculate(cpm.Module):
    module_name = "CorrectIlluminationCalculate"
    variable_revision_number = 2
    category = "Image Processing"

    def create_settings(self):
        """Create the setting variables
        """
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc='''
            Choose the image to be used to calculate the illumination function.''')

        self.illumination_image_name = cps.ImageNameProvider(
                "Name the output image", "IllumBlue", doc='''
            Enter a name for the resultant illumination function.''',
                provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                     cps.AVAILABLE_ON_LAST_ATTRIBUTE: False})

        self.intensity_choice = cps.Choice(
                "Select how the illumination function is calculated",
                [IC_REGULAR, IC_BACKGROUND],
                IC_REGULAR, doc='''
            Choose which method you want to use to calculate the illumination function. You may chose
            from the following options:
            <ul>
            <li><i>%(IC_REGULAR)s:</i> If you have objects that are evenly dispersed across your image(s) and
             cover most of the image, the <i>Regular</i> method might be appropriate. Regular
             intensities makes the illumination function based on the intensity at
             each pixel of the image (or group of images if you are in <i>%(EA_ALL)s</i> mode) and
             is most often rescaled (see below) and applied by division using
             <b>CorrectIlluminationApply.</b> Note that if you are in <i>%(EA_EACH)s</i> mode or using a
             small set of images with few objects, there will be regions in the
             average image that contain no objects and smoothing by median filtering
             is unlikely to work well.
             <i>Note:</i> it does not make sense to choose (<i>%(IC_REGULAR)s + %(SM_NONE)s + %(EA_EACH)s</i>)
             because the illumination function would be identical to the original
             image and applying it will yield a blank image. You either need to smooth
             each image, or you need to use <i>%(EA_ALL)s</i> images.</li>
             <li><i>%(IC_BACKGROUND)s:</i>
             If you think that the background (dim points) between objects show the
             same pattern of illumination as your objects of interest, you can choose the
             <i>%(IC_BACKGROUND)s</i> method. Background intensities finds the minimum pixel
             intensities in blocks across the image (or group of images if you are in
             <i>%(EA_ALL)s</i> mode) and is most often applied by subtraction using the
             <b>CorrectIlluminationApply</b> module.
             <i>Note:</i> if you will be using the <i>%(DOS_SUBTRACT)s</i> option in the
             <b>CorrectIlluminationApply</b> module, you almost certainly do not want to
             rescale the illumination function. </li>
             </ul>
             <p>Please note that if a mask was applied to the input image, the pixels outside of the
             mask will be excluded from consideration. This is useful, for instance, in cases where
             you have masked out the well edge in an image from a multi-well plate; the dark well
             edge would distort the illumination correction function along the interior well edge.
             Masking the image beforehand solves this problem.</p>''' % globals())

        self.dilate_objects = cps.Binary(
                "Dilate objects in the final averaged image?", False, doc='''
            <i>(Used only if the Regular method is selected)</i><br>
            For some applications, the incoming images are binary and each object
            should be dilated with a Gaussian filter in the final averaged
            (projection) image. This is for a sophisticated method of illumination
            correction where model objects are produced.
            Select <i>%(YES)s</i> to dilate objects for this approach.
            ''' % globals())

        self.object_dilation_radius = cps.Integer(
                "Dilation radius", 1, 0, doc='''
            <i>(Used only if the "%(IC_REGULAR)s" method and dilation is selected)</i><br>
            This value should be roughly equal to the original radius of the objects''' % globals())

        self.block_size = cps.Integer(
                "Block size", 60, 1, doc='''
            <i>(Used only if "%(IC_BACKGROUND)s" is selected)</i><br>
            The block size should be large enough that every square block of pixels is likely
            to contain some background pixels, where no objects are located.''' % globals())

        self.rescale_option = cps.Choice(
                "Rescale the illumination function?",
                [cps.YES, cps.NO, RE_MEDIAN], doc='''
            The illumination function can be rescaled so that the pixel intensities
            are all equal to or greater than 1. You have the following options:
            <ul>
            <li><i>%(YES)s:</i> Rescaling is recommended if you plan to
            use the <i>%(IC_REGULAR)s</i> method (and hence, the <i>%(DOS_DIVIDE)s</i> option in
            <b>CorrectIlluminationApply</b>) so that the corrected images are in the
            range 0 to 1.</li>
            <li><i>%(NO)s:</i> Rescaling is not recommended if you plan to use the <i>%(IC_BACKGROUND)s</i>
            method, which is paired with the <i>%(DOS_SUBTRACT)s</i> option in <b>CorrectIlluminationApply</b>.
            Note that as a result of the illumination function being rescaled from 1 to
            infinity, the rescaling of each image might be dramatic if there is substantial
            variation across the field of view, causing the corrected images
            to be very dark. </li>
            <li>%(RE_MEDIAN)s<i>:</i> This option chooses the median value in the
            image to rescale so that division increases some values and decreases others.</li>
            </ul>''' % globals())

        self.each_or_all = cps.Choice(
                "Calculate function for each image individually, or based on all images?",
                [EA_EACH, EA_ALL_FIRST, EA_ALL_ACROSS], doc='''
            Calculate a separate function for each image, or one for all the images?
            You can calculate the illumination function using just the current
            image or you can calculate the illumination function using all of
            the images in each group.
            The illumination function can be calculated in one of the three ways:
            <ul>
            <li><i>%(EA_EACH)s:</i> Calculate an illumination function for each image
            individually. </li>
            <li><i>%(EA_ALL_FIRST)s:</i> Calculate an illumination
            function based on all of the images in a group, performing the
            calculation before proceeding to the next module. This means that the
            illumination function will be created in the first cycle (making the first
            cycle longer than subsequent cycles), and lets you use the function in a subsequent
            <b>CorrectIllumination_Apply</b> module in the same pipeline, but also
            means that you will not have the ability to filter out images (e.g., by using
            <b>FlagImage</b>). The input images need to be assembled using the <b>Input</b>
            modules; using images produced by other modules will yield an error.</li>
            <li><i>%(EA_ALL_ACROSS)s:</i> Calculate an illumination function
            across all cycles in each group. This option takes any image
            as input; however, the illumination function
            will not be completed until the end of the last cycle in the group.
            You can use <b>SaveImages</b> to save the illumination function
            after the last cycle in the group and then use the resulting
            image in another pipeline. The option is useful if you want to exclude
            images that are filtered by a prior <b>FlagImage</b> module.</li>
            </ul>''' % globals())

        self.smoothing_method = cps.Choice(
                "Smoothing method",
                [SM_NONE,
                 SM_CONVEX_HULL,
                 SM_FIT_POLYNOMIAL,
                 SM_MEDIAN_FILTER,
                 SM_GAUSSIAN_FILTER,
                 SM_TO_AVERAGE,
                 SM_SPLINES], doc='''
             If requested, the resulting image is smoothed. See the
             <b>EnhanceOrSuppressFeatures</b> module help for more details. If you are using <i>Each</i> mode,
             this is almost certainly necessary. If you have few objects in each image or a
             small image set, you may want to smooth.
             <p>You should smooth to the point where the illumination function resembles a believable pattern.
             For example, if you are trying to correct a lamp illumination problem,
             apply smoothing until you obtain a fairly smooth pattern
             without sharp bright or dim regions.  Note that smoothing is a
             time-consuming process, but some methods are faster than others.
             <ul>
             <li><i>%(SM_FIT_POLYNOMIAL)s:</i> This methdod is fastest but does not
             allow a very tight fit compared to the slower median and Gaussian
             filtering methods. </li>
             <li>%(SM_MEDIAN_FILTER)s, <i>%(SM_GAUSSIAN_FILTER)s:</i> Use a
             median or Gaussian filter, respectively. We typically recommend <i>%(SM_MEDIAN_FILTER)s</i>
             vs. <i>%(SM_GAUSSIAN_FILTER)s</i> because the median
             is less sensitive to outliers, although the results are also slightly
             less smooth and the fact that images are in the range of 0 to 1 means that
             outliers typically will not dominate too strongly anyway.</li>
             <li><i>%(SM_TO_AVERAGE)s:</i> A less commonly
             used option is to completely smooth the entire image, which will create a flat,
             smooth image where every pixel of the image is the average of what the illumination
             function would otherwise have been.</li>
             <li><i>%(SM_SPLINES)s:</i> This method (<i>Lindblad and Bengtsson, 2001</i>)
             fits a grid of cubic splines to the background while
             excluding foreground pixels from the calculation. It operates
             iteratively, classifying pixels as background, computing a best
             fit spline to this background and then reclassifying pixels
             as background until the spline converges on its final value.</li>
             <li><i>%(SM_CONVEX_HULL)s:</i> This method algorithm proceeds as follows:
             <ul><li>Choose 256 evenly-spaced intensity levels between the
             minimum and maximum intensity for the image</li>
             <li>Set the intensity of the output image to the minimum intensity
             of the input image</li>
             <li>Iterate over the intensity levels, from lowest to highest
             <ul>
             <li>For a given intensity, find all pixels with
             equal or higher intensities</li>
             <li>Find the convex hull that encloses those pixels</li>
             <li>Set the intensity of the output image within the convex hull
             to the current intensity</li>
             </ul></li>
             </ul>
             The Convex Hull method can be used on an image whose objects
             are darker than their background and whose illumination
             intensity decreases monotonically from the brightest point.</li>
             </ul>

             <b>References</b>
             <ul><li>J Lindblad and E Bengtsson (2001) "A comparison of methods for estimation of
             intensity nonuniformities in 2D and 3D microscope images of fluorescence
             stained cells.", Proceedings of the 12th Scandinavian Conference on Image
             Analysis (SCIA), pp. 264-271</li>
             </ul>
             ''' % globals())

        self.automatic_object_width = cps.Choice(
                "Method to calculate smoothing filter size",
                [FI_AUTOMATIC, FI_OBJECT_SIZE, FI_MANUALLY], doc='''
            <i>(Used only if a smoothing method other than Fit Polynomial is selected)</i><br>
            Calculate the smoothing filter size. There are three options:
            <ul>
            <li><i>%(FI_AUTOMATIC)s:</i> The size is computed as 1/40 the size of the image or
            30 pixels, whichever is smaller.</li>
            <li><i>%(FI_OBJECT_SIZE)s:</i> The size is obtained relative to the width
            of artifacts to be smoothed.</li>
            <li><i>%(FI_MANUALLY)s:</i> Use a manually entered value.</li>
            </ul>''' % globals())

        self.object_width = cps.Integer(
                "Approximate object size", 10, doc='''
            <i>(Used only if %(FI_AUTOMATIC)s is selected for smoothing filter size calculation)</i><br>
            Enter the approximate width of the artifacts to be smoothed, in pixels.''' % globals())

        self.size_of_smoothing_filter = cps.Integer(
                "Smoothing filter size", 10, doc='''
            <i>(Used only if %(FI_MANUALLY)s is selected for smoothing filter size calculation)</i><br>
            Enter the size of the desired smoothing filter, in pixels.''' % globals())

        self.save_average_image = cps.Binary(
                "Retain the averaged image?", False, doc='''
            The averaged image is the illumination function
            prior to dilation or smoothing. It is an image produced during the calculations, not typically
            needed for downstream modules. It can be helpful to retain it in case you wish to try several
            different smoothing methods without taking the time to recalculate the averaged image each time.
            <p>Select <i>%(YES)s</i> to retain this averaged image. Use the <b>SaveImages</b> module to save
            it to your hard drive.</p>''' % globals())

        self.average_image_name = cps.ImageNameProvider(
                "Name the averaged image", "IllumBlueAvg", doc='''
            <i>(Used only if the averaged image is to be retained for later use in the pipeline)</i><br>
            Enter a name that will allow the averaged image to be selected later in the pipeline.''')

        self.save_dilated_image = cps.Binary(
                "Retain the dilated image?", False, doc='''
            The dilated image is the illumination function after dilation but prior to smoothing.
            It is an image produced during the calculations, and is not typically needed for downstream modules.
            <p>Select <i>%(YES)s</i> to retain this dilated image. Use the <b>SaveImages</b> module to save it
            to your hard drive.</p>''' % globals())

        self.dilated_image_name = cps.ImageNameProvider(
                "Name the dilated image", "IllumBlueDilated", doc='''
            <i>(Used only if the dilated image is to be retained for later use in the pipeline)</i><br>
            Enter a name that will allow the dilated image to be selected later in the pipeline.''')

        self.automatic_splines = cps.Binary(
                "Automatically calculate spline parameters?", True, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method)</i><br>
            Select <i>%(YES)s</i> to automatically calculate
            the parameters for spline fitting.
            <p>Select <i>%(NO)s</i> to specify the background mode, background threshold, scale,
            maximum number of iterations and convergence.</p>""" % globals())

        self.spline_bg_mode = cps.Choice(
                "Background mode",
                [MODE_AUTO, MODE_DARK, MODE_BRIGHT, MODE_GRAY], doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This setting determines which pixels are background
            and which are foreground.<br>
            <ul><li><i>%(MODE_AUTO)s</i>: Determine the mode from the image.
            This will set the mode to %(MODE_DARK)s if most of the pixels are
            dark, %(MODE_BRIGHT)s if most of the pixels are bright and
            %(MODE_GRAY)s if there are relatively few dark and light pixels
            relative to the number of mid-level pixels</li>
            <li><i>%(MODE_DARK)s</i>: Fit the spline to the darkest pixels
            in the image, excluding brighter pixels from consideration.
            This may be appropriate for a fluorescent image.
            </li>
            <li><i>%(MODE_BRIGHT)s</i>: Fit the spline to the lightest pixels
            in the image, excluding the darker pixels. This may be appropriate
            for a histologically stained image.</li>
            <li><i>%(MODE_GRAY)s</i>: Fit the spline to mid-range pixels,
            excluding both dark and light pixels. This may be appropriate
            for a brightfield image where the objects of interest have
            light and dark features.</li></ul>""" % globals())

        self.spline_threshold = cps.Float(
                "Background threshold", 2, minval=.1, maxval=5.0, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This setting determines the cutoff used when excluding
            foreground pixels from consideration. On each iteration,
            the method computes the standard deviation of background
            pixels from the computed background. The number entered in this
            setting is the number of standard deviations a pixel can be
            from the computed background on the last pass if it is to
            be considered as background during the next pass.
            <p>
            You should enter a higher number to converge stabily and slowly
            on a final background and a lower number to converge more
            rapidly, but with lower stability. The default for this
            parameter is two standard deviations; this will provide a fairly
            stable background estimate.""" % globals())

        self.spline_points = cps.Integer(
                "Number of spline points", 5, 4, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This is the number of control points for the spline.
            A value of 5 results in a 5x5 grid of splines across the image and
            is the value suggested by the method's authors. A lower value
            will give you a more stable background while a higher one will
            fit variations in the background more closely and take more time
            to compute.""" % globals())

        self.spline_rescale = cps.Float(
                "Image resampling factor", 2, minval=1, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This setting controls how the image is resampled to
            make a smaller image. Resampling will speed up processing,
            but may degrade performance if the resampling factor is larger
            than the diameter of foreground objects. The image will
            be downsampled by the factor you enter. For instance, a 500x600
            image will be downsampled into a 250x300 image if a factor of 2
            is entered.""" % globals())

        self.spline_maximum_iterations = cps.Integer(
                "Maximum number of iterations", 40, minval=1, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This setting determines the maximum number of iterations
            of the algorithm to be performed. The algorithm will perform
            fewer iterations if it converges.""" % globals())

        self.spline_convergence = cps.Float(
                "Residual value for convergence", value=.001, minval=.00001, maxval=.1, doc="""
            <i>(Used only if %(SM_SPLINES)s are selected for the smoothing method and
            spline parameters are not calculated automatically)</i><br>
            This setting determines the convergence criterion.
            The software sets the convergence criterion to the number entered
            here times the signal intensity; the convergence you enter is the
            fraction of the signal intensity that indicates convergence.
            The algorithm derives a standard deviation of the background
            pixels from the calculated background on each iteration. The
            algorithm terminates when the difference between the standard
            deviation for the current iteration and the previous iteration
            is less than the convergence criterion.
            <p>Enter a smaller number for the convergence to calculate a
            more accurate background. Enter a larger number to calculate
            the background using fewer iterations, but less accuracy.""" % globals())

    def settings(self):
        return [self.image_name, self.illumination_image_name,
                self.intensity_choice, self.dilate_objects,
                self.object_dilation_radius, self.block_size,
                self.rescale_option, self.each_or_all, self.smoothing_method,
                self.automatic_object_width, self.object_width,
                self.size_of_smoothing_filter, self.save_average_image,
                self.average_image_name, self.save_dilated_image,
                self.dilated_image_name,
                self.automatic_splines, self.spline_bg_mode,
                self.spline_points, self.spline_threshold, self.spline_rescale,
                self.spline_maximum_iterations, self.spline_convergence]

    def visible_settings(self):
        """The settings as seen by the UI

        """
        result = [self.image_name, self.illumination_image_name,
                  self.intensity_choice]
        if self.intensity_choice == IC_REGULAR:
            result += [self.dilate_objects]
            if self.dilate_objects.value:
                result += [self.object_dilation_radius]
        elif self.smoothing_method != SM_SPLINES:
            result += [self.block_size]

        result += [self.rescale_option, self.each_or_all,
                   self.smoothing_method]
        if self.smoothing_method in (SM_GAUSSIAN_FILTER, SM_MEDIAN_FILTER):
            result += [self.automatic_object_width]
            if self.automatic_object_width == FI_OBJECT_SIZE:
                result += [self.object_width]
            elif self.automatic_object_width == FI_MANUALLY:
                result += [self.size_of_smoothing_filter]
        elif self.smoothing_method == SM_SPLINES:
            result += [self.automatic_splines]
            if not self.automatic_splines:
                result += [self.spline_bg_mode, self.spline_points,
                           self.spline_threshold,
                           self.spline_rescale, self.spline_maximum_iterations,
                           self.spline_convergence]
        result += [self.save_average_image]
        if self.save_average_image.value:
            result += [self.average_image_name]
        result += [self.save_dilated_image]
        if self.save_dilated_image.value:
            result += [self.dilated_image_name]
        return result

    def help_settings(self):
        return [self.image_name, self.illumination_image_name,
                self.intensity_choice, self.dilate_objects,
                self.object_dilation_radius, self.block_size,
                self.rescale_option, self.each_or_all, self.smoothing_method,
                self.automatic_object_width, self.object_width,
                self.size_of_smoothing_filter,
                self.automatic_splines, self.spline_bg_mode,
                self.spline_points, self.spline_threshold, self.spline_rescale,
                self.spline_maximum_iterations, self.spline_convergence,
                self.save_average_image,
                self.average_image_name, self.save_dilated_image,
                self.dilated_image_name]

    def prepare_group(self, workspace, grouping, image_numbers):
        image_set_list = workspace.image_set_list
        pipeline = workspace.pipeline
        assert isinstance(pipeline, cpp.Pipeline)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        if self.each_or_all != EA_EACH and len(image_numbers) > 0:
            title = "#%d: CorrectIlluminationCalculate for %s" % (
                self.module_num, self.image_name)
            message = ("CorrectIlluminationCalculate is averaging %d images while "
                       "preparing for run" % (len(image_numbers)))
            output_image_provider = CorrectIlluminationImageProvider(
                    self.illumination_image_name.value, self)
            d = self.get_dictionary(image_set_list)[OUTPUT_IMAGE] = {}
            if self.each_or_all == EA_ALL_FIRST:
                #
                # Find the module that provides the image we need
                #
                md = workspace.pipeline.get_provider_dictionary(
                        self.image_name.group, self)
                src_module, src_setting = md[self.image_name.value][-1]
                modules = list(pipeline.modules())
                idx = modules.index(src_module)
                last_module = modules[idx + 1]
                for w in pipeline.run_group_with_yield(
                        workspace, grouping, image_numbers,
                        last_module, title, message):
                    image = w.image_set.get_image(self.image_name.value,
                                                  cache=False)
                    output_image_provider.add_image(image)
                    w.image_set.clear_cache()
            output_image_provider.serialize(d)

        return True

    def run(self, workspace):
        if self.each_or_all != EA_EACH:
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.deserialize(
                    d, self)
            if self.each_or_all == EA_ALL_ACROSS:
                #
                # We are accumulating a pipeline image. Add this image set's
                # image to the output image provider.
                #
                orig_image = workspace.image_set.get_image(self.image_name.value)
                output_image_provider.add_image(orig_image)
                output_image_provider.serialize(d)

            # fetch images for display
            if (self.show_window or self.save_average_image or
                    self.save_dilated_image or self.each_or_all == EA_ALL_FIRST):
                avg_image = output_image_provider.provide_avg_image()
                dilated_image = output_image_provider.provide_dilated_image()
                workspace.image_set.providers.append(output_image_provider)
                output_image = output_image_provider.provide_image(workspace.image_set)
            else:
                workspace.image_set.providers.append(output_image_provider)
        else:
            orig_image = workspace.image_set.get_image(self.image_name.value)
            pixels = orig_image.pixel_data
            avg_image = self.preprocess_image_for_averaging(orig_image)
            dilated_image = self.apply_dilation(avg_image, orig_image)
            smoothed_image = self.apply_smoothing(dilated_image, orig_image)
            output_image = self.apply_scaling(smoothed_image, orig_image)
            # for illumination correction, we want the smoothed function to extend beyond the mask.
            output_image.mask = np.ones(output_image.pixel_data.shape[:2], bool)
            workspace.image_set.add(self.illumination_image_name.value,
                                    output_image)

        if self.save_average_image.value:
            workspace.image_set.add(self.average_image_name.value,
                                    avg_image)
        if self.save_dilated_image.value:
            workspace.image_set.add(self.dilated_image_name.value,
                                    dilated_image)
        if self.show_window:
            # store images for potential display
            workspace.display_data.avg_image = avg_image.pixel_data
            workspace.display_data.dilated_image = dilated_image.pixel_data
            workspace.display_data.output_image = output_image.pixel_data

    def is_aggregation_module(self):
        '''Return True if aggregation is performed within a group'''
        return self.each_or_all != EA_EACH

    def post_group(self, workspace, grouping):
        '''Handle tasks to be performed after a group has been processed

        For CorrectIllumninationCalculate, we make sure the current image
        set includes the aggregate image. "run" may not have run if an
        image was filtered out.
        '''
        if self.each_or_all != EA_EACH:
            image_set = workspace.image_set
            d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
            output_image_provider = CorrectIlluminationImageProvider.deserialize(
                    d, self)
            assert isinstance(output_image_provider, CorrectIlluminationImageProvider)
            if not self.illumination_image_name.value in image_set.names:
                workspace.image_set.providers.append(output_image_provider)
            if (self.save_average_image and
                        self.average_image_name.value not in image_set.names):
                workspace.image_set.add(
                        self.average_image_name.value,
                        output_image_provider.provide_avg_image())
            if (self.save_dilated_image and
                        self.dilated_image_name.value not in image_set.names):
                workspace.image_set.add(
                        self.dilated_image_name.value,
                        output_image_provider.provide_dilated_image())

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
        imshow(0, 1, output_image,
               "Final illumination function",
               sharexy=figure.subplot(0, 0))
        imshow(1, 0, dilated_image,
               "Dilated image",
               sharexy=figure.subplot(0, 0))
        statistics = [["Min value", round(np.min(output_image), 2)],
                      ["Max value", round(np.max(output_image), 2)],
                      ["Calculation type", self.intensity_choice.value]
                      ]
        if self.intensity_choice == IC_REGULAR:
            statistics.append(["Radius", self.object_dilation_radius.value])
        elif self.smoothing_method != SM_SPLINES:
            statistics.append(["Block size", self.block_size.value])
        statistics.append(["Rescaling?", self.rescale_option.value])
        statistics.append(["Each or all?", self.each_or_all.value])
        statistics.append(["Smoothing method", self.smoothing_method.value])
        statistics.append(["Smoothing filter size",
                           round(self.smoothing_filter_size(output_image.size), 2)])
        figure.subplot_table(1, 1,
                             [[x[1]] for x in statistics],
                             row_labels=[x[0] for x in statistics])

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
            kernel = circular_gaussian_kernel(self.object_dilation_radius.value,
                                              self.object_dilation_radius.value * 3)

            def fn(image):
                return scind.convolve(image, kernel, mode='constant', cval=0)

            if image.pixel_data.ndim == 2:
                dilated_pixels = smooth_with_function_and_mask(
                        image.pixel_data, fn, image.mask)
            else:
                dilated_pixels = np.dstack([
                                               smooth_with_function_and_mask(x, fn, image.mask)
                                               for x in image.pixel_data.transpose(2, 0, 1)])
            return cpi.Image(dilated_pixels, parent_image=orig_image)
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
            return min(30, float(np.max(image_shape)) / 40.0)

    def preprocess_image_for_averaging(self, orig_image):
        """Create a version of the image appropriate for averaging

        """
        pixels = orig_image.pixel_data
        if (self.intensity_choice == IC_REGULAR or
                    self.smoothing_method == SM_SPLINES):
            if orig_image.has_mask:
                if pixels.ndim == 2:
                    pixels[~ orig_image.mask] = 0
                else:
                    pixels[~ orig_image.mask, :] = 0
                avg_image = cpi.Image(pixels, parent_image=orig_image)
            else:
                avg_image = orig_image
        else:
            # For background, we create a labels image using the block
            # size and find the minimum within each block.
            labels, indexes = cpmm.block(pixels.shape[:2],
                                         (self.block_size.value,
                                          self.block_size.value))
            if orig_image.has_mask:
                labels[~ orig_image.mask] = -1

            min_block = np.zeros(pixels.shape)
            if pixels.ndim == 2:
                minima = fix(scind.minimum(pixels, labels, indexes))
                min_block[labels != -1] = minima[labels[labels != -1]]
            else:
                for i in range(pixels.shape[2]):
                    minima = fix(scind.minimum(pixels[:, :, i], labels, indexes))
                    min_block[labels != -1, i] = minima[labels[labels != -1]]
            avg_image = cpi.Image(min_block, parent_image=orig_image)
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
            output_pixels = np.zeros(pixel_data.shape, pixel_data.dtype)
            for i in range(pixel_data.shape[2]):
                output_pixels[:, :, i] = self.smooth_plane(pixel_data[:, :, i],
                                                           image.mask)
        else:
            output_pixels = self.smooth_plane(pixel_data, image.mask)
        output_image = cpi.Image(output_pixels, parent_image=orig_image)
        return output_image

    def smooth_plane(self, pixel_data, mask):
        '''Smooth one 2-d color plane of an image'''

        sigma = self.smoothing_filter_size(pixel_data.shape) / 2.35
        if self.smoothing_method == SM_FIT_POLYNOMIAL:
            output_pixels = fit_polynomial(pixel_data, mask)
        elif self.smoothing_method == SM_GAUSSIAN_FILTER:
            #
            # Smoothing with the mask is good, even if there's no mask
            # because the mechanism undoes the edge effects that are introduced
            # by any choice of how to deal with border effects.
            #
            def fn(image):
                return scind.gaussian_filter(image, sigma,
                                             mode='constant', cval=0)

            output_pixels = smooth_with_function_and_mask(pixel_data, fn,
                                                          mask)
        elif self.smoothing_method == SM_MEDIAN_FILTER:
            filter_sigma = max(1, int(sigma+.5))
            strel = strel_disk(filter_sigma)
            indices, values = rank_order(pixel_data, 65535)
            indices = indices.astype(np.uint16)
            output_pixels = skimage.filters.median(pixel_data, strel, mask=mask)
        elif self.smoothing_method == SM_TO_AVERAGE:
            mean = np.mean(pixel_data[mask])
            output_pixels = np.ones(pixel_data.shape, pixel_data.dtype) * mean
        elif self.smoothing_method == SM_SPLINES:
            output_pixels = self.smooth_with_splines(pixel_data, mask)
        elif self.smoothing_method == SM_CONVEX_HULL:
            output_pixels = self.smooth_with_convex_hull(pixel_data, mask)
        else:
            raise ValueError("Unimplemented smoothing method: %s:" % self.smoothing_method.value)
        return output_pixels

    def smooth_with_convex_hull(self, pixel_data, mask):
        '''Use the convex hull transform to smooth the image'''
        #
        # Apply an erosion, then the transform, then a dilation, heuristically
        # to ignore little spikey noisy things.
        #
        image = grey_erosion(pixel_data, 2, mask)
        image = convex_hull_transform(image, mask=mask)
        image = grey_dilation(image, 2, mask)
        return image

    def smooth_with_splines(self, pixel_data, mask):
        if self.automatic_splines:
            # Make the image 200 pixels long on its shortest side
            shortest_side = min(pixel_data.shape)
            if shortest_side < 200:
                scale = 1
            else:
                scale = float(shortest_side) / 200
            result = backgr(pixel_data, mask, scale=scale)
        else:
            mode = self.spline_bg_mode.value
            spline_points = self.spline_points.value
            threshold = self.spline_threshold.value
            convergence = self.spline_convergence.value
            iterations = self.spline_maximum_iterations.value
            rescale = self.spline_rescale.value
            result = backgr(pixel_data, mask, mode=mode, thresh=threshold,
                            splinepoints=spline_points, scale=rescale,
                            maxiter=iterations, convergence=convergence)
        #
        # The result is a fit to the background intensity, but we
        # want to normalize the intensity by subtraction, leaving
        # the mean intensity alone.
        #
        mean_intensity = np.mean(result[mask])
        result[mask] -= mean_intensity
        return result

    def apply_scaling(self, image, orig_image=None):
        """Return an image that is rescaled according to the settings

        image - an instance of cpimage.Image
        returns another instance of cpimage.Image
        """
        if self.rescale_option == cps.NO:
            return image

        def scaling_fn_2d(pixel_data):
            if image.has_mask:
                sorted_pixel_data = pixel_data[(pixel_data > 0) & image.mask]
            else:
                sorted_pixel_data = pixel_data[pixel_data > 0]
            if sorted_pixel_data.shape[0] == 0:
                return pixel_data
            sorted_pixel_data.sort()
            if self.rescale_option == cps.YES:
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
            output_pixels = np.dstack([
                                          scaling_fn_2d(x) for x in image.pixel_data.transpose(2, 0, 1)])
        output_image = cpi.Image(output_pixels, parent_image=orig_image)
        return output_image

    def validate_module(self, pipeline):
        '''Produce error if 'All:First' is selected and input image is not provided by the file image provider.'''
        if not pipeline.is_image_from_file(self.image_name.value) and self.each_or_all == EA_ALL_FIRST:
            raise cps.ValidationError(
                    "All: First cycle requires that the input image be provided by the Input modules, or LoadImages/LoadData.",
                    self.each_or_all)

        '''Modify the image provider attributes based on other setttings'''
        d = self.illumination_image_name.provided_attributes
        if self.each_or_all == EA_ALL_ACROSS:
            d[cps.AVAILABLE_ON_LAST_ATTRIBUTE] = True
        elif d.has_key(cps.AVAILABLE_ON_LAST_ATTRIBUTE):
            del d[cps.AVAILABLE_ON_LAST_ATTRIBUTE]

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if self.each_or_all == EA_ALL_FIRST:
            raise cps.ValidationError(
                    "Pre-calculation of the illumination function is time-intensive, especially for Test Mode. The analysis will proceed, but consider using '%s' instead." % EA_ALL_ACROSS,
                    self.each_or_all)

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        """Adjust the setting values of old versions

        setting_values - sequence of strings that are the values for our settings
        variable_revision_number - settings were saved by module with this
                                   variable revision number
        module_name - name of module that did the saving
        from_matlab - True if it was the Matlab version that did the saving

        returns upgraded setting values, upgraded variable revision number
                and from_matlab flag

        Matlab variable revision numbers 6 and 7 supported.
        pyCellProfiler variable revision number 1 supported.
        """

        if from_matlab and variable_revision_number == 6:
            # Smoothing could be sum of squares or square of sums in 6,
            # could be Gaussian in 7 - arbitrarily, I've translated
            # the obsolete ones to be Gaussian
            new_setting_values = list(setting_values)
            if new_setting_values[8] in ("Sum of Squares", "Square of Sum"):
                new_setting_values[8] = SM_GAUSSIAN_FILTER
            setting_values = new_setting_values

        if from_matlab and variable_revision_number == 7:
            # Convert Matlab variable order to ours
            new_setting_values = list(setting_values[:3])
            #
            # If object_dilation_radius is 0, then set self.dilate_objects
            # to false, otherwise true
            #
            if setting_values[3] == "0":
                new_setting_values.append(cps.NO)
            else:
                new_setting_values.append(cps.YES)
            #
            # We determine whether the input image is loaded from a file
            # or generated by the pipeline. In Matlab, setting # 8 (our 7)
            # made the user answer this question.
            #
            new_setting_values.extend(setting_values[3:7])
            new_setting_values.append(setting_values[8])
            #
            # set self.automatic_object_width based on settings 9 (the old
            # ObjectWidth setting) and 10 (the old SizeOfSmoothingFilter)
            #
            if setting_values[9] == FI_AUTOMATIC:
                new_setting_values.extend([FI_AUTOMATIC, "10", "10"])
            elif (setting_values[10] == cps.DO_NOT_USE or
                          setting_values[10] == "/"):
                new_setting_values.extend([FI_OBJECT_SIZE, setting_values[9], "10"])
            else:
                new_setting_values.extend([FI_MANUALLY, setting_values[9],
                                           setting_values[10]])
            #
            # The optional output images: were "Do not use" if the user
            # didn't want them. Now it's two settings each.
            #
            for setting, name in zip(setting_values[11:],
                                     ("IllumBlueAvg", "IllumBlueDilated")):
                if setting == cps.DO_NOT_USE:
                    new_setting_values.extend([cps.NO, name])
                else:
                    new_setting_values.extend([cps.YES, setting])
            setting_values = new_setting_values
            variable_revision_number = 1
            from_matlab = False

        if (not from_matlab) and variable_revision_number == 1:
            # Added spline parameters
            setting_values = setting_values + [
                cps.YES,  # automatic_splines
                MODE_AUTO,  # spline_bg_mode
                "5",  # spline points
                "2",  # spline threshold
                "2",  # spline rescale
                "40",  # spline maximum iterations
                "0.001"]  # spline convergence
            variable_revision_number = 2

        return setting_values, variable_revision_number, from_matlab

    def post_pipeline_load(self, pipeline):
        '''After loading, set each_or_all appropriately

        This function handles the legacy EA_ALL which guessed the user's
        intent: processing before the first cycle or not. We look for
        the image provider and see if it is a file image provider.
        '''
        if self.each_or_all == EA_ALL:
            if pipeline.is_image_from_file(self.image_name.value):
                self.each_or_all.value = EA_ALL_FIRST
            else:
                self.each_or_all.value = EA_ALL_ACROSS


class CorrectIlluminationImageProvider(cpi.AbstractImageProvider):
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
        '''Save the internal state of the provider to a dictionary

        d - save to this dictionary, numpy arrays and json serializable only
        '''
        d[self.D_NAME] = self.__name
        d[self.D_IMAGE_SUM] = self.__image_sum
        d[self.D_MASK_COUNT] = self.__mask_count

    @staticmethod
    def deserialize(d, module):
        '''Restore a state saved by serialize

        d - dictionary containing the state
        module - the module providing details on how to perform the correction

        returns a provider set up with the restored state
        '''
        provider = CorrectIlluminationImageProvider(
                d[CorrectIlluminationImageProvider.D_NAME],
                module)
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
            self.__image_sum = np.zeros(pixel_data.shape,
                                        pixel_data.dtype)
            self.__mask_count = np.zeros(pixel_data.shape[:2],
                                         np.int32)
        if image.has_mask:
            mask = image.mask
            if self.__image_sum.ndim == 2:
                self.__image_sum[mask] = \
                    self.__image_sum[mask] + pixel_data[mask]
            else:
                self.__image_sum[mask, :] = \
                    self.__image_sum[mask, :] + pixel_data[mask, :]
            self.__mask_count[mask] = self.__mask_count[mask] + 1
        else:
            self.__image_sum = self.__image_sum + pixel_data
            self.__mask_count = self.__mask_count + 1

    def reset(self):
        '''Reset the image sum at the start of a group'''
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
        pixel_data = np.zeros(self.__image_sum.shape,
                              self.__image_sum.dtype)
        mask = self.__mask_count > 0
        if pixel_data.ndim == 2:
            pixel_data[mask] = self.__image_sum[mask] / self.__mask_count[mask]
        else:
            for i in range(pixel_data.shape[2]):
                pixel_data[mask, i] = \
                    self.__image_sum[mask, i] / self.__mask_count[mask]
        self.__cached_avg_image = cpi.Image(pixel_data, mask)
        self.__cached_dilated_image = \
            self.__module.apply_dilation(self.__cached_avg_image)
        smoothed_image = \
            self.__module.apply_smoothing(self.__cached_dilated_image)
        self.__cached_image = self.__module.apply_scaling(smoothed_image)
        self.__dirty = False

    def release_memory(self):
        # Memory is released during reset(), so this is a no-op
        pass


class CorrectIlluminationAvgImageProvider(cpi.AbstractImageProvider):
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


class CorrectIlluminationDilatedImageProvider(cpi.AbstractImageProvider):
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
