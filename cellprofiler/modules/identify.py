"""identify.py - a base class for common functionality for identify modules

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""


import math
import scipy.ndimage
import scipy.sparse
import numpy as np
import scipy.stats

import cellprofiler.cpmodule
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.cpmath.outline
import cellprofiler.objects
from cellprofiler.cpmath.smooth import smooth_with_noise
from cellprofiler.cpmath.smooth import smooth_with_function_and_mask
from cellprofiler.cpmath.threshold import TM_MANUAL, TM_MEASUREMENT, TM_METHODS, get_threshold
from cellprofiler.cpmath.threshold import TM_GLOBAL, TM_ADAPTIVE, TM_BINARY_IMAGE
from cellprofiler.cpmath.threshold import TM_PER_OBJECT, TM_OTSU, TM_MOG, TM_MCT
from cellprofiler.cpmath.threshold import weighted_variance, sum_of_entropies
from cellprofiler.gui.help import HELP_ON_PIXEL_INTENSITIES

O_TWO_CLASS = 'Two classes'
O_THREE_CLASS = 'Three classes'

O_WEIGHTED_VARIANCE = 'Weighted variance'
O_ENTROPY = 'Entropy'

O_FOREGROUND = 'Foreground'
O_BACKGROUND = 'Background'

FI_IMAGE_SIZE    = 'Image size'
FI_CUSTOM        = 'Custom'

'''The location measurement category'''
C_LOCATION = "Location"

'''The number category (e.g. Number_Object_Number)'''
C_NUMBER = "Number"

'''The count category (e.g. Count_Nuclei)'''
C_COUNT = "Count"

'''The threshold category (e.g. Threshold_FinalThreshold_DNA)'''
C_THRESHOLD = "Threshold"

'''The parent category (e.g. Parent_Nuclei)'''
C_PARENT = "Parent"

'''The parent relationship'''
R_PARENT = "Parent"

'''The children category (e.g. Children_Cells_Count)'''
C_CHILDREN = "Children"

'''The child relationship'''
R_CHILD = "Child"

FTR_CENTER_X = "Center_X"
'''The centroid X coordinate measurement feature name'''
M_LOCATION_CENTER_X = '%s_%s'%(C_LOCATION, FTR_CENTER_X)

FTR_CENTER_Y = "Center_Y"
'''The centroid Y coordinate measurement feature name'''
M_LOCATION_CENTER_Y = '%s_%s'%(C_LOCATION, FTR_CENTER_Y)

FTR_OBJECT_NUMBER = "Object_Number"
'''The object number - an index from 1 to however many objects'''
M_NUMBER_OBJECT_NUMBER = '%s_%s' % (C_NUMBER, FTR_OBJECT_NUMBER)

'''The format for the object count image measurement'''
FF_COUNT = '%s_%%s' % C_COUNT

FTR_FINAL_THRESHOLD = "FinalThreshold"

'''Format string for the FinalThreshold feature name'''
FF_FINAL_THRESHOLD = '%s_%s_%%s' %(C_THRESHOLD, FTR_FINAL_THRESHOLD)

FTR_ORIG_THRESHOLD = "OrigThreshold"

'''Format string for the OrigThreshold feature name'''
FF_ORIG_THRESHOLD = '%s_%s_%%s' % (C_THRESHOLD,FTR_ORIG_THRESHOLD)

FTR_WEIGHTED_VARIANCE = "WeightedVariance"

'''Format string for the WeightedVariance feature name'''
FF_WEIGHTED_VARIANCE = '%s_%s_%%s' % (C_THRESHOLD, FTR_WEIGHTED_VARIANCE)

FTR_SUM_OF_ENTROPIES = "SumOfEntropies"

'''Format string for the SumOfEntropies feature name'''
FF_SUM_OF_ENTROPIES = '%s_%s_%%s' % (C_THRESHOLD, FTR_SUM_OF_ENTROPIES)

'''Format string for # of children per parent feature name'''
FF_CHILDREN_COUNT = "%s_%%s_Count" % C_CHILDREN

'''Format string for parent of child feature name'''
FF_PARENT = "%s_%%s" % C_PARENT

'''Threshold scope = automatic - use defaults of global + MCT, no adjustments'''
TS_AUTOMATIC ="Automatic"

'''Threshold scope = global - one threshold per image'''
TS_GLOBAL = "Global"

'''Threshold scope = adaptive - threshold locally'''
TS_ADAPTIVE = "Adaptive"

'''Threshold scope = per-object - one threshold per controlling object'''
TS_PER_OBJECT = "Per object"

'''Threshold scope = manual - choose one threshold for all'''
TS_MANUAL = "Manual"

'''Threshold scope = binary mask - use a binary mask to determine threshold'''
TS_BINARY_IMAGE = "Binary image"

'''Threshold scope = measurement - use a measurement value as the threshold'''
TS_MEASUREMENT = "Measurement"

TS_ALL = [TS_AUTOMATIC, TS_GLOBAL, TS_ADAPTIVE, TS_PER_OBJECT, TS_MANUAL, 
          TS_BINARY_IMAGE, TS_MEASUREMENT]

'''The legacy choice of object in per-object measurements

Legacy pipelines required MaskImage to be used to mask an image with objects
in order to do per-object thresholding. We support legacy pipelines by
including this among the choices.
'''
O_FROM_IMAGE = "From image"

'''Do not smooth image before thresholding'''
TSM_NONE = "No smoothing"

'''Use a gaussian with sigma = 1 - the legacy value for IdentifyPrimary'''
TSM_AUTOMATIC = "Automatic"

'''Allow the user to enter a smoothing factor'''
TSM_MANUAL = "Manual"

class Identify(cellprofiler.cpmodule.CPModule):
    threshold_setting_version = 1
    def create_threshold_settings(self, methods = TM_METHODS):
        '''Create settings related to thresholding'''
        # The threshold setting version is invisible to the user
        self.threshold_setting_version = cps.Integer(
            "Threshold setting version", 
            value = self.threshold_setting_version)
        self.threshold_scope = cps.Choice(
            'Threshold strategy',
            TS_ALL,
            doc = """
            The thresholding strategy determines whether the module calculates
            a threshold from the input image's intensity or whether it
            uses a threshold taken from user input, from a previous measurement
            or from a binary mask. If a threshold is calculated, the threshold
            algorithm can be performed on the whole image or can be performed
            on tiles or objects supplied by a prior module.
            <br>
            The choices for the threshold strategy are:
            <br><ul>
            <li><i>%(TS_AUTOMATIC)s:</i> use the default settings for
            thresholding. This strategy calculates the threshold using MCT
            on the whole image and applies the threshold to the image, smoothed
            with a Gaussian with sigma of 1. This is
            fairly robust, but does not allow you to select the threshold
            algorithm and does not allow you to apply corrections to the
            threshold.
            </li>
            <li><i>%(TS_GLOBAL)s:</i> calculate a single threshold based on
            the unmasked pixels of the input image and use the single value
            to classify pixels above the threshold as foreground and below
            as background. This strategy is fast and robust, especially if
            the background is uniformly illuminated.
            </li>
            <li><i>%(TS_ADAPTIVE)s:</i> partition the input image into tiles
            and calculate thresholds for each tile. Apply the threshold for
            a tile to the pixels within that tile. This method is slower but
            can produce better results for non-uniform backgrounds.</li>
            <li><i>%(TS_PER_OBJECT)s:</i> use objects from a prior module
            such as <i>IdentifyPrimaryObjects</i> to partition the input image.
            Calculate a separate threshold per object and apply that threshold
            to pixels within the object. Pixels outside of objects will be
            classified as background. This method can be useful for identifying
            particles or single-molecule probes within cells if the cells
            contribute varying amounts of signal to the background through
            autofluorescence or other mechanisms.</li>
            <li><i>%(TS_MANUAL)s:</i> Enter a single value between zero and
            one that applies to all cycles and is independent of the input
            image. This is useful if the input image has a stable or
            negligible background or if the input image is the probability
            map output of the <i>ClassifyPixels</i> module (in which case, .5
            is an appropriate value).</li>
            <li><i>%(TS_BINARY_IMAGE)s:</i> use a binary image to classify
            pixels as foreground or background. Pixel values other than zero
            will be foreground and pixel values that are zero will be
            background. This method can be used to import a ground-truth
            segmentation into CellProfiler or to import a segmentation from
            another program. It can also be used to create objects from
            a mask produced by other CellProfiler modules such as <i>Morph</i></li>
            <li><i>%(TS_MEASUREMENT)s:</i> use an image measurement as the
            threshold. The measurement should have values between zero and one.
            This strategy can be used to apply a pre-calculated threshold
            imported as image metadata via <i>LoadData</i> or <i>Metadata</i>.
            The threshold measurement from a prior <i>ApplyThreshold</i>,
            <i>IdentifyPrimaryObjects</i> or <i>IdentifySecondaryObjects</i>
            module can be used to apply the same threshold to subsequent
            alternate segmentations.</li>
            </ul>
	    """ % globals())
	    
        self.threshold_method = cps.Choice(
            'Threshold method',
            methods, doc="""
            The intensity threshold affects the decision of whether each pixel
            will be considered foreground (regions of interest) or background.
            A stringent threshold will result in only 
            bright regions being identified, with tight lines around them, whereas 
            a lenient threshold will include dim regions and the lines between regions 
            and background will be more loose. You can have the threshold automatically calculated 
            using several methods, or you can enter an absolute number between 0 
            and 1 for the threshold. To help determine the choice of threshold manually, you
            can inspect the pixel intensities in an image of your choice. 
            %(HELP_ON_PIXEL_INTENSITIES)s"""%globals() + """ Both options have advantages. 
            An absolute number treats every image identically, but is not robust 
            with regard to slight changes in lighting/staining conditions between images. An
            automatically calculated threshold adapts to changes in
            lighting/staining conditions between images and is usually more
            robust/accurate, but it can occasionally produce a poor threshold for
            unusual/artifactual images. It also takes a small amount of time to
            calculate.
            
            <p>The threshold that is used for each image is recorded as a
            measurement in the output file, so if you are surprised by unusual measurements from
            one of your images, you might check whether the automatically calculated
            threshold was unusually high or low compared to the other images.
            
            <p>There are seven methods for finding thresholds automatically:
            <ul><li><i>Otsu:</i> This method is probably best if you are not able 
            to make certain assumptions about every images in your experiment, 
            especially if the percentage of the image covered by regions 
            of interest varies substantially from image to image. Our implementation 
            takes into account the maximum and minimum values in the image and log-transforming the
            image prior to calculating the threshold. For this reason, please note
            that negative-valued pixels are ignored in this computation, so caution should be
            used in using image offsets (such as by using <b>ImageMath</b>).
            <p>If you know that the percentage of 
            each image that is foreground does not vary much from image
            to image, the MoG method can be better, especially if the foreground percentage is
            not near 50%.</li>
            <li><i>Mixture of Gaussian (MoG):</i>This function assumes that the 
            pixels in the image belong to either a background class or a foreground
            class, using an initial guess of the fraction of the image that is 
            covered by foreground. This method is our own version of a Mixture of Gaussians
            algorithm (<i>O. Friman, unpublished</i>). Essentially, there are two steps:
            <ol><li>First, a number of Gaussian distributions are estimated to 
            match the distribution of pixel intensities in the image. Currently 
            three Gaussian distributions are fitted, one corresponding to a 
            background class, one corresponding to a foreground class, and one 
            distribution for an intermediate class. The distributions are fitted
            using the Expectation-Maximization algorithm, a procedure referred 
            to as Mixture of Gaussians modeling. </li>
            <li>When the three Gaussian distributions have been fitted, a decision 
            is made whether the intermediate class more closely models the background pixels 
            or foreground pixels, based on the estimated fraction provided by the user.</li></ol></li>
            <li><i>Background:</i> This method is simple and appropriate for images in 
            which most of the image is background. It finds the mode of the 
            histogram of the image, which is assumed to be the background of the 
            image, and chooses a threshold at twice that value (which you can 
            adjust with a Threshold Correction Factor; see below).  The calculation 
	    includes those pixels between 2% and 98% of the intensity range. This thresholding method 
	    can be helpful if your images vary in overall brightness, but the objects of 
            interest are consistently N times brighter than the background level of the image. </li>
            <li><i>Robust background:</i> Much like the Background method, this method is 
	    also simple and assumes that the background distribution
	    approximates a Gaussian by trimming the brightest and dimmest 5% of pixel 
	    intensities. It then calculates the mean and standard deviation of the 
            remaining pixels and calculates the threshold as the mean + 2 times 
            the standard deviation. This thresholding method can be helpful if the majority
	    of the image is background, and the results are often comparable or better than the
	    Background method.</li>
            <li><i>Ridler-Calvard:</i> This method is simple and its results are
            often very similar to Otsu's. According to
            Sezgin and Sankur's paper (<i>Journal of Electronic Imaging</i>, 2004), Otsu's 
            overall quality on testing 40 nondestructive testing images is slightly 
            better than Ridler's (average error: Otsu, 0.318; Ridler, 0.401). 
            Ridler-Calvard chooses an initial threshold and then iteratively calculates the next 
            one by taking the mean of the average intensities of the background and 
            foreground pixels determined by the first threshold, repeating this until 
            the threshold converges.</li>
            <li><i>Kapur:</i> This method computes the threshold of an image by
            log-transforming its values, then searching for the threshold that
            maximizes the sum of entropies of the foreground and background
            pixel values, when treated as separate distributions.</li>
            <li><i>Maximum correlation:</i>This is an implementation of the
            method described in Padmanabhan et al, 2010. It computes the maximum
            correlation between the binary mask created by thresholding and
            the thresholded image and is somewhat similar mathematically to
            Otsu. The authors claim superior results when thresholding images
            of neurites and other images that have sparse foreground densities.</li>
            </ul>
            
            <p>You can also choose between <i>Global</i>, <i>Adaptive</i>, and 
            <i>Per-object</i> thresholding for the automatic methods:
            <ul>
            <li><i>Global:</i> One threshold is calculated for the entire image (fast)</li>
            <li><i>Adaptive:</i> The calculated threshold varies across the image. This method 
            is a bit slower but may be more accurate near edges of regions of interest, 
            or where illumination variation is significant (though in the latter case, 
            using the <b>CorrectIllumination</b> modules is preferable).</li>
            <li><i>Per-object:</i> If you are using this module to find child objects located
            <i>within</i> parent objects, the per-object method will calculate a distinct
            threshold for each parent object. This is especially helpful, for
            example, when the background brightness varies substantially among the
            parent objects. 
            <br><i>Important:</i> the per-object method requires that you run an
            <b>IdentifyPrimaryObjects</b> module to identify the parent objects upstream in the
            pipeline. After the parent objects are identified in the pipeline, you
            must then also run a <b>Crop</b> module with the following inputs: 
            <ul>
            <li>The input image is the image containing the sub-objects to be identified.</li>
            <li>Select <i>Objects</i> as the shape to crop into.</li>
            <li>Select the parent objects (e.g., <i>Nuclei</i>) as the objects to use as a cropping mask.</li>
            </ul>
            Finally, in the <b>IdentifyPrimaryObjects</b> module, select the cropped image as input image.</li></ul>
            
            <p>Selecting <i>manual thresholding</i> allows you to enter a single value between 0 and 1
            as the threshold value. This setting can be useful when you are certain what the
            cutoff should be and it does not vary from image to image in the experiment. If you are 
            using this module to find objects in an image that is already binary (where the foreground is 1 and 
            the background is 0), a manual value of 0.5 will identify the objects.
            
            <p>Selecting thresholding via a <i>binary image</i> will use a selected binary image as a mask for the
            input image. The most typical approach to produce a binary image is to use the <b>ApplyThreshold</b> module 
            (image as input, image as output) or the <b>ConvertObjectsToImage</b> module (objects as input, image 
            as output); both have options to produce a binary image. Note that unlike <b>MaskImage</b>, the binary 
            image will not be stored permanently as a mask. Also, even though no algorithm is actually used to 
            find the threshold in this case, the final threshold value is reported as the Otsu threshold 
            calculated for the foreground region.
            
            <p>Selecting thresholding via <i>measurement</i> will use an image measurement previously calculated
            in order to threshold the image. Like manual thresholding, this setting can be useful when you are 
            certain what the cutoff should be. The difference in this case is that the desired threshold does 
            vary from image to image in the experiment but can be measured using a Measurement module.</p>
            
            <p>References
            <ul>
            <li>Sezgin M, Sankur B (2004) "Survey over image thresholding techniques and quantitative 
            performance evaluation." <i>Journal of Electronic Imaging</i>, 13(1), 146-165</li>
            <li>Padmanabhan K, Eddy WF, Crowley JC (2010) "A novel algorithm for
            optimal image thresholding of biological data" <i>Journal of 
            Neuroscience Methods</i> 193, 380-384.</li>
            </ul></p>
            """)
        
        self.threshold_smoothing_choice = cps.Choice(
            "Smoothing for threshold", [TSM_AUTOMATIC, TSM_MANUAL, TSM_NONE],
            doc = 
            """<i>(Only for strategies other than %(TS_AUTOMATIC)s and
            %(TS_BINARY_IMAGE)s)</i><br>
            The input image can be optionally smoothed before being thresholded.
            Smoothing can regularize the objects, removing holes and jagged
            edges caused by noise in the acquired image. Smoothing is most
            likely not appropriate if the input image is binary, if it
            has already been smoothed or if it is an output of the
            <i>ClassifyPixels</i> module. The choices are:
            <br><ul>
            <li><i>%(TSM_AUTOMATIC)s</i>: Smooth the image with a Gaussian
            with a sigma of one pixel before thresholding.</li>
            <li><i>%(TSM_MANUAL)s</i>: Smooth the image with a Gaussian with
            user-controlled scale</li>
            <li><i>%(TSM_NONE)s</i>: apply the threshold to the unsmoothed
            image</li>
            </ul>""" % globals())
        self.threshold_smoothing_scale = cps.Float(
            "Threshold smoothing scale", 1.0, minval = 0,
            doc =
            """<i>(Only if smoothing for threshold is %(TSM_MANUAL)s)</i>
            <br>This setting controls the scale used to smooth the input image
            before the threshold is applied. The scale should be approximately
            the size of the artifacts to be eliminated by smoothing. A Gaussian
            is used with a sigma adjusted so that 1/2 of the Gaussian's
            distribution falls within the diameter given by the scale
            (sigma = scale / .674)""")

        self.threshold_correction_factor = cps.Float('Threshold correction factor', 1,
                                                doc="""\
            When the threshold is calculated automatically, it may consistently be
            too stringent or too lenient. You may need to enter an adjustment factor
            that you empirically determine is suitable for your images. The number 1
            means no adjustment, 0 to 1 makes the threshold more lenient and greater
            than 1 (e.g., 1.3) makes the threshold more stringent. For example, the
            Otsu automatic thresholding inherently assumes that 50% of the image is
            covered by objects. If a larger percentage of the image is covered, the
            Otsu method will give a slightly biased threshold that may have to be
            corrected using this setting.""")
        
        self.threshold_range = cps.FloatRange('Lower and upper bounds on threshold', (0,1), minval=0,
                                         maxval=1, doc="""\
            Enter the minimum and maximum allowable threshold, in the range [0,1].  
            This is helpful as a safety precaution when the threshold is calculated
            automatically. For example, if there are no objects in the field of view,
            the automatic threshold might be calculated as unreasonably low. In such cases, the
            lower bound you enter here will override the automatic threshold.""")
        
        self.object_fraction = cps.CustomChoice(
            'Approximate fraction of image covered by objects?', 
            ['0.01','0.1','0.2','0.3', '0.4','0.5','0.6','0.7', '0.8','0.9',
             '0.99'], doc="""\
            <i>(Used only when applying the MoG thresholding method)</i><br>
            Enter an estimate of how much of the image is covered with objects, which
            is used to estimate the distribution of pixel intensities.""")
        
        self.manual_threshold = cps.Float("Manual threshold", 
                                          value=0.0, minval=0.0, maxval=1.0,doc="""\
            <i>(Used only if Manual selected for thresholding method)</i><br>
            Enter the value that will act as an absolute threshold for the images, in the range of [0,1].""")
        
        self.thresholding_measurement = cps.Measurement("Select the measurement to threshold with",
            lambda : cpmeas.IMAGE, doc = """
            <i>(Used only if Measurement is selected for thresholding method)</i><br>
            Choose the image measurement that will act as an absolute threshold for the images.""")
        
        self.binary_image = cps.ImageNameSubscriber(
            "Select binary image", "None", doc = """
            <i>(Used only if Binary image selected for thresholding method)</i><br>
            What is the binary thresholding image?""")
        
        self.masking_objects = MaskObjectNameSubscriber(
            "Masking objects", "None",
            doc = """<i>(Used only if %(TS_PER_OBJECT)s is selected for the
            thresholding strategy)</i><br>A threshold will be calculated for
            each object and applied to the pixels inside that object. Pixels
            outside of any object will be assigned to the background.""")
        
        self.two_class_otsu = cps.Choice(
            'Two-class or three-class thresholding?',
            [O_TWO_CLASS, O_THREE_CLASS],doc="""
            <i>(Used only for the Otsu thresholding method)</i> <br>
            Select <i>Two</i> if the grayscale levels are readily distinguishable 
            into only two classes: foreground 
            (i.e., objects) and background. Select <i>Three</i> if the grayscale 
            levels fall instead into three classes. You will then be asked whether 
            the middle intensity class should be added to the foreground or background 
            class in order to generate the final two-class output.  Note that whether 
            two- or three-class thresholding is chosen, the image pixels are always 
            finally assigned two classes: foreground and background.
            <p>For example, three-class thresholding may
            be useful for images in which you have nuclear staining along with 
            low-intensity non-specific cell staining. Where two-class thresholding
            might incorrectly assign this intermediate staining to the nuclei 
            objects for some cells, three-class thresholding allows you to assign it to the 
            foreground or background as desired. However, in extreme cases where either 
            there are almost no objects or the entire field of view is covered with 
            objects, three-class thresholding may perform worse than two-class.""")
        
        self.use_weighted_variance = cps.Choice(
            'Minimize the weighted variance or the entropy?',
            [O_WEIGHTED_VARIANCE, O_ENTROPY])
        
        self.assign_middle_to_foreground = cps.Choice(
            'Assign pixels in the middle intensity class to the foreground '
            'or the background?', [O_FOREGROUND, O_BACKGROUND],doc="""
            <i>(Used only for three-class thresholding)</i><br>
            Choose whether you want the pixels with middle grayscale intensities to be assigned 
            to the foreground class or the background class.""")
        
        self.adaptive_window_method = cps.Choice(
            "Method to calculate adaptive window size",
            [FI_IMAGE_SIZE, FI_CUSTOM], doc="""
            <i>(Used only if an adaptive thresholding method is used)</i><br>
            The adaptive method breaks the image into blocks, computing the threshold 
            for each block. There are two ways to compute the block size:
            <ul>
            <li><i>%(FI_IMAGE_SIZE)s:</i> The block size is one-tenth of the image dimensions,
            or 50 x 50 pixels, whichever is bigger.</li>
            <li><i>%(FI_CUSTOM)s:</i> The block size is specified by the user.</li>
            </ul>"""%globals())
                                            
        self.adaptive_window_size = cps.Integer(
            'Size of adaptive window', 10, doc="""
            <i>(Used only if an adaptive thresholding method with a %(FI_CUSTOM)s window size 
            are selected)</i><br>
            Enter the window for the adaptive method. For example,
            you may want to use a multiple of the largest expected object size."""%globals())
        
    def get_threshold_settings(self):
        '''Return the threshold settings to be saved in the pipeline'''
        return [self.threshold_setting_version,
                self.threshold_scope, self.threshold_method, 
                self.threshold_smoothing_choice, 
                self.threshold_smoothing_scale, 
                self.threshold_correction_factor,
                self.threshold_range,
                self.object_fraction,
                self.manual_threshold,
                self.thresholding_measurement,
                self.binary_image,
                self.masking_objects,
                self.two_class_otsu,
                self.use_weighted_variance,
                self.assign_middle_to_foreground,
                self.adaptive_window_method,
                self.adaptive_window_size]
    
    def get_threshold_help_settings(self):
        '''Return the threshold settings to be displayed in help'''
        return [self.threshold_scope,
                self.threshold_method,
                self.binary_image,
                self.manual_threshold,
                self.thresholding_measurement,
                self.two_class_otsu,
                self.use_weighted_variance,
                self.assign_middle_to_foreground,
                self.object_fraction,
                self.adaptive_window_method,
                self.adaptive_window_size,
                self.threshold_correction_factor,
                self.threshold_range,
                self.threshold_smoothing_choice,
                self.threshold_smoothing_scale]
    
    def upgrade_legacy_threshold_settings(
        self, threshold_method, threshold_smoothing_choice, threshold_correction_factor,
        threshold_range, object_fraction, manual_threshold,
        thresholding_measurement, binary_image, 
        two_class_otsu, use_weighted_variance, assign_middle_to_foreground,
        adaptive_window_method, adaptive_window_size,
        masking_objects = O_FROM_IMAGE):
        '''Return threshold setting strings built from the legacy elements
        
        IdentifyPrimaryObjects, IdentifySecondaryObjects and ApplyThreshold
        used to store their settings independently. This method creates the
        unified settings block from the values in their separate arrangements.
        
        All parameters should be self-explanatory except for
        threshold_smoothing_choice which should be TSM_AUTOMATIC if
        smoothing was applied to the image before thresholding or TSM_NONE
        if it wasn't'''
        if threshold_method == TM_BINARY_IMAGE:
            threshold_scope = TS_BINARY_IMAGE
            threshold_method = TM_OTSU
        elif threshold_method == TM_MANUAL:
            threshold_scope = TS_MANUAL
            threshold_method = TM_OTSU
        elif threshold_method == TM_MEASUREMENT:
            threshold_scope = TS_MEASUREMENT
            threshold_method = TM_OTSU
        else:
            threshold_method, threshold_scope = threshold_method.rsplit(" ", 1)
            if threshold_scope == TM_GLOBAL:
                threshold_scope = TS_GLOBAL
            elif threshold_scope == TM_PER_OBJECT:
                threshold_scope = TS_PER_OBJECT
            elif threshold_scope == TM_ADAPTIVE:
                threshold_scope = TS_ADAPTIVE
        setting_values = [ 
            "1", threshold_scope, threshold_method, threshold_smoothing_choice,
            "1", threshold_correction_factor, threshold_range,
            object_fraction, manual_threshold, thresholding_measurement, 
            binary_image, masking_objects, two_class_otsu,
            use_weighted_variance, assign_middle_to_foreground, 
            adaptive_window_method, adaptive_window_size]
        return setting_values
        
    def upgrade_threshold_settings(self, setting_values):
        '''Upgrade the threshold settings to the current version
        
        use the first setting which is the version to determine the
        threshold settings version and upgrade as appropriate
        '''
        version = int(setting_values[0])
        if version > self.threshold_setting_version:
            raise ValueError("Unsupported pipeline version: threshold setting version = %d" % version)
        return setting_values
    
    def get_threshold_visible_settings(self):
        '''Return visible settings related to thresholding'''
        vv = [self.threshold_scope]
        if self.threshold_scope == TS_AUTOMATIC:
            return vv
        if self.threshold_scope == TS_MANUAL:
            vv += [self.manual_threshold]
        elif self.threshold_scope == TS_MEASUREMENT:
            vv += [self.thresholding_measurement]
        elif self.threshold_scope == TS_BINARY_IMAGE:
            vv += [self.binary_image]
        elif self.threshold_scope in (TS_GLOBAL, TS_ADAPTIVE, TS_PER_OBJECT):
            vv += [self.threshold_method]
            if self.threshold_scope == TS_PER_OBJECT:
                vv += [self.masking_objects]
            if self.threshold_method == TM_OTSU:
                vv += [self.two_class_otsu, self.use_weighted_variance]
                if self.two_class_otsu == O_THREE_CLASS:
                    vv.append(self.assign_middle_to_foreground)
            if self.threshold_method == TM_MOG:
                vv += [self.object_fraction]
        if self.threshold_scope != TM_BINARY_IMAGE:
            vv += [ self.threshold_smoothing_choice]
            if self.threshold_smoothing_choice == TSM_MANUAL:
                vv += [self.threshold_smoothing_scale]
        if not self.threshold_scope in (TM_MANUAL, TM_BINARY_IMAGE):
            vv += [ self.threshold_correction_factor, self.threshold_range]
        if self.threshold_scope == TM_ADAPTIVE:
            vv += [ self.adaptive_window_method ]
            if self.adaptive_window_method == FI_CUSTOM:
                vv += [ self.adaptive_window_size ]
            
        return vv
        
    def threshold_image(self, image_name, workspace, 
                        wants_local_threshold=False):
        """Compute the threshold using whichever algorithm was selected by the user
        
        image_name - name of the image to use for thresholding
        
        workspace - get any measurements / objects / images from the workspace
        
        returns: thresholded binary image
        """
        #
        # Retrieve the relevant image and mask
        #
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale = True)
        img = image.pixel_data
        mask = image.mask
        if self.threshold_scope == TS_BINARY_IMAGE:
            binary_image = workspace.image_set.get_image(
                self.binary_image.value, must_be_binary = True).pixel_data
            self.add_fg_bg_measurements(
                workspace.measurements, img, mask, binary_image)
            if wants_local_threshold:
                return binary_image, None
            return binary_image
        local_threshold, global_threshold = self.get_threshold(
            img, mask, workspace)
    
        if self.threshold_smoothing_choice == TSM_NONE:
            blurred_image = img
        else:
            if self.threshold_smoothing_choice == TSM_AUTOMATIC:
                sigma = 1
            else:
                # Convert from a scale into a sigma. What I've done here
                # is to structure the Gaussian so that 1/2 of the smoothed
                # intensity is contributed from within the smoothing diameter
                # and 1/2 is contributed from outside.
                sigma = self.threshold_smoothing_scale.value / 0.6744 / 2.0
            def fn(img, sigma=sigma):
                return scipy.ndimage.gaussian_filter(
                    img, sigma, mode='constant', cval=0)
            blurred_image = smooth_with_function_and_mask(img, fn, mask)
            
        binary_image = (blurred_image >= local_threshold) & mask
        self.add_fg_bg_measurements(
            workspace.measurements, img, mask, binary_image)
        if wants_local_threshold:
            return binary_image, local_threshold
        return binary_image
        
    def get_threshold(self, img, mask, workspace):
        '''Calculate a local and global threshold
        
        img - base the threshold on this image's intensity
        
        mask - use this mask to define the pixels of interest
        
        workspace - get objects and measurements from this workspace and
                    add threshold measurements to this workspace's measurements.
        '''
        if self.threshold_scope == TM_MANUAL:
            local_threshold = global_threshold = self.manual_threshold.value
        else:
            if self.threshold_scope == TM_MEASUREMENT:
                m = workspace.measurements
                # Thresholds are stored as single element arrays.  Cast to
                # float to extract the value.
                value = float(m.get_current_image_measurement(
                    self.thresholding_measurement.value))
                value *= self.threshold_correction_factor.value
                if not self.threshold_range.min is None:
                    value = max(value, self.threshold_range.min)
                if not self.threshold_range.max is None:
                    value = min(value, self.threshold_range.max)
                local_threshold = global_threshold = value
            else:
                if self.threshold_scope == TS_PER_OBJECT:
                    if self.masking_objects == O_FROM_IMAGE:
                        labels = image.labels
                    else:
                        masking_objects = workspace.object_set.get_objects(
                            self.masking_objects.value)
                        label_planes = masking_objects.get_labels(img.shape[:2])
                        if len(label_planes) == 1:
                            labels = label_planes[0][0]
                        else:
                            # For overlaps, we arbitrarily assign a pixel to
                            # the first label it appears in. Alternate would be
                            # to average, seems like it's too fine a point
                            # to deal with it. A third possibility would be to 
                            # treat overlaps as distinct entities since the overlapping
                            # areas will likely be different than either object.
                            labels = np.zeros(label_planes[0][0].shape,
                                              label_planes[0][0].dtype)
                            for label_plane, indices in label_planes:
                                labels[labels == 0] = label_plane[labels == 0]
                else:
                    labels = None
                if self.threshold_scope == TS_ADAPTIVE:
                    if self.adaptive_window_method == FI_IMAGE_SIZE:
                         # The original behavior
                        image_size = np.array(img.shape[:2], dtype=int)
                        block_size = image_size / 10
                        block_size[block_size<50] = 50
                    elif self.adaptive_window_method == FI_CUSTOM:
                        block_size = self.adaptive_window_size.value * \
                            np.array([1,1])
                else:
                    block_size = None
                object_fraction = self.object_fraction.value
                if object_fraction.endswith("%"):
                    object_fraction = float(object_fraction[:-1])/100.0
                else:
                    object_fraction = float(object_fraction)
                local_threshold, global_threshold = get_threshold(
                    self.threshold_algorithm,
                    self.threshold_modifier,
                    img, 
                    mask = mask,
                    labels = labels,
                    threshold_range_min = self.threshold_range.min,
                    threshold_range_max = self.threshold_range.max,
                    threshold_correction_factor = self.threshold_correction_factor.value,
                    object_fraction = object_fraction,
                    two_class_otsu = self.two_class_otsu.value == O_TWO_CLASS,
                    use_weighted_variance = self.use_weighted_variance.value == O_WEIGHTED_VARIANCE,
                    assign_middle_to_foreground = self.assign_middle_to_foreground.value == O_FOREGROUND,
                    adaptive_window_size = block_size)
        self.add_threshold_measurements(workspace.measurements,
                                        local_threshold, global_threshold)
        if hasattr(workspace.display_data, "statistics"):
            workspace.display_data.statistics.append(
                ["Threshold","%0.3f"%(global_threshold)])
            
        return local_threshold, global_threshold
    
    def get_measurement_objects_name(self):
        '''Return the name of the measurement objects
        
        Identify modules and ApplyThreshold store measurements in the Image
        table and append an object name or, for ApplyThreshold, an image
        name to distinguish between different thresholds in the same pipeline.
        '''
        raise NotImplementedError(
            "Please implement get_measurement_objects_name() for this module")
    
    def add_threshold_measurements(self, measurements,
                                   local_threshold, global_threshold):
        '''Compute and add threshold statistics measurements 
        
        measurements - add the measurements here
        local_threshold - either a per-pixel threshold (a matrix) or a 
                          copy of the global threshold (a scalar)
        global_threshold - the globally-calculated threshold
        '''
        objname = self.get_measurement_objects_name()
        ave_threshold = np.mean(np.atleast_1d(local_threshold))
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_FINAL_THRESHOLD%(objname),
                                     ave_threshold)
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_ORIG_THRESHOLD%(objname),
                                     global_threshold)
        
    def add_fg_bg_measurements(self, measurements, image, mask, binary_image):
        '''Add statistical measures of the within class variance and entropy
        
        measurements - store measurements here
        
        image - assess the variance and entropy on this intensity image
        
        mask - mask of pixels to be considered
        
        binary_image - the foreground / background segmentation of the image
        '''
        objname = self.get_measurement_objects_name()
        wv = weighted_variance(image, mask, binary_image)
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_WEIGHTED_VARIANCE%(objname),
                                     np.array([wv],dtype=float))
        entropies = sum_of_entropies(image, mask, binary_image)
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_SUM_OF_ENTROPIES%(objname),
                                     np.array([entropies],dtype=float))
        
    def validate_module(self, pipeline):
        if hasattr(self, "object_fraction"):
            try:
                if self.object_fraction.value.endswith("%"):
                    float(self.object_fraction.value[:-1])
                else:
                    float(self.object_fraction.value)
            except ValueError:
                raise cps.ValidationError("%s is not a floating point value"%
                                          self.object_fraction.value,
                                          self.object_fraction)

    def get_threshold_modifier(self):
        """The threshold algorithm modifier
        
        TM_GLOBAL                       = "Global"
        TM_ADAPTIVE                     = "Adaptive"
        TM_PER_OBJECT                   = "PerObject"
        """
        if self.threshold_scope.value in (TS_AUTOMATIC,
            TS_GLOBAL, TS_BINARY_IMAGE, TS_MANUAL, TS_MEASUREMENT):
            return TM_GLOBAL 
        elif self.threshold_scope.value == TS_PER_OBJECT:
            return TM_PER_OBJECT
        return TM_ADAPTIVE
    
    threshold_modifier = property(get_threshold_modifier)
    
    def get_threshold_algorithm(self):
        """The thresholding algorithm, for instance TM_OTSU"""
        if self.threshold_scope == TS_AUTOMATIC:
            return TM_MCT
        return self.threshold_method.value
    
    threshold_algorithm = property(get_threshold_algorithm)
    
    def get_threshold_measurement_columns(self, pipeline):
        '''Return the measurement columns for the threshold measurements'''
        features = [FF_SUM_OF_ENTROPIES, FF_WEIGHTED_VARIANCE]
        if self.threshold_scope != TS_BINARY_IMAGE:
            features += [FF_ORIG_THRESHOLD, FF_FINAL_THRESHOLD]
        return [(cpmeas.IMAGE, 
                 ftr % self.get_measurement_objects_name(),
                 cpmeas.COLTYPE_FLOAT) for ftr in features]
    
    def get_threshold_categories(self, pipeline, object_name):
        '''Get categories related to thresholding'''
        if object_name == cpmeas.IMAGE:
            return [C_THRESHOLD]
        return []
    
    def get_threshold_measurements(self, pipeline, object_name, category):
        '''Return a list of threshold measurements for a given category
        
        object_name - either "Image" or an object name
        category - must be "Threshold" to get anything back
        '''
        if object_name == cpmeas.IMAGE and category == C_THRESHOLD:
            if self.threshold_scope == TS_BINARY_IMAGE:
                return [FTR_SUM_OF_ENTROPIES, FTR_WEIGHTED_VARIANCE]
            return [FTR_ORIG_THRESHOLD, FTR_FINAL_THRESHOLD,
                    FTR_SUM_OF_ENTROPIES, FTR_WEIGHTED_VARIANCE]
        return []
    
    def get_threshold_measurement_objects(self, pipeline, object_name, category, 
                                          measurement):
        '''Get the measurement objects for a threshold measurement
        
        pipeline - not used
        object_name - either "Image" or an object name. (must be "Image")
        category - the measurement category. (must be "Threshold")
        measurement - the feature being measured
        '''
        if measurement in self.get_threshold_measurements(
            pipeline, object_name, category):
            return [self.get_measurement_objects_name()]
        else:
            return []
        
    def get_object_categories(self, pipeline, object_name, 
                              object_dictionary):
        '''Get categories related to creating new children
        
        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of
                            an object created by this module and each
                            value is a list of names of parents.
        '''
        if object_name == cpmeas.IMAGE:
            return [C_COUNT]
        result = []
        if object_dictionary.has_key(object_name):
            result += [C_LOCATION, C_NUMBER]
            if len(object_dictionary[object_name]) > 0:
                result += [C_PARENT]
        if object_name in reduce(lambda x,y: x+y, object_dictionary.values()):
            result += [C_CHILDREN]
        return result
    
    def get_object_measurements(self, pipleline, object_name, category,
                                object_dictionary):
        '''Get measurements related to creating new children
        
        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of 
                            an object created by this module and each
                            value is a list of names of parents.
        '''
        if object_name == cpmeas.IMAGE and category == C_COUNT:
            return list(object_dictionary.keys())
        
        if object_dictionary.has_key(object_name):
            if category == C_LOCATION:
                return [FTR_CENTER_X, FTR_CENTER_Y]
            elif category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_PARENT:
                return list(object_dictionary[object_name])
        if category == C_CHILDREN:
            result = []
            for child_object_name in object_dictionary.keys():
                if object_name in object_dictionary[child_object_name]:
                    result += ["%s_Count" % child_object_name]
            return result
        return []
    
def add_object_location_measurements(measurements, 
                                     object_name,
                                     labels, object_count = None):
    """Add the X and Y centers of mass to the measurements
    
    measurements - the measurements container
    object_name  - the name of the objects being measured
    labels       - the label matrix
    object_count - (optional) the object count if known, otherwise
                   takes the maximum value in the labels matrix which is
                   usually correct.
    """
    if object_count is None:
        object_count = np.max(labels)
    #
    # Get the centers of each object - center_of_mass <- list of two-tuples.
    #
    if object_count:
        centers = scipy.ndimage.center_of_mass(np.ones(labels.shape), 
                                               labels, 
                                               range(1,object_count+1))
        centers = np.array(centers)
        centers = centers.reshape((object_count,2))
        location_center_y = centers[:,0]
        location_center_x = centers[:,1]
        number = np.arange(1,object_count+1)
    else:
        location_center_y = np.zeros((0,),dtype=float)
        location_center_x = np.zeros((0,),dtype=float)
        number = np.zeros((0,),dtype=int)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X,
                                 location_center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y,
                                 location_center_y)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER, number)
    
def add_object_location_measurements_ijv(measurements,
                                         object_name,
                                         ijv, object_count = None):
    '''Add object location measurements for IJV-style objects'''
    if object_count is None:
        object_count = 0 if ijv.shape[0] == 0 else np.max(ijv[:, 2])
    if object_count == 0:
        center_x = np.zeros(0)
        center_y = np.zeros(0)
    else:
        areas = np.zeros(object_count, int)
        areas_bc = np.bincount(ijv[:, 2])[1:]
        areas[:len(areas_bc)] = areas_bc
        center_x = np.bincount(ijv[:, 2], ijv[:, 1])[1:] / areas
        center_y = np.bincount(ijv[:, 2], ijv[:, 0])[1:] / areas
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X, center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y, center_y)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER, 
                                 np.arange(1, object_count+1))
    

def add_object_count_measurements(measurements, object_name, object_count):
    """Add the # of objects to the measurements"""
    measurements.add_measurement('Image',
                                 FF_COUNT%(object_name),
                                 np.array([object_count],
                                             dtype=float))

def get_object_measurement_columns(object_name):
    '''Get the column definitions for measurements made by identify modules
    
    Identify modules can use this call when implementing
    CPModule.get_measurement_columns to get the column definitions for
    the measurements made by add_object_location_measurements and
    add_object_count_measurements.
    '''
    return [(object_name, M_LOCATION_CENTER_X, cpmeas.COLTYPE_FLOAT),
            (object_name, M_LOCATION_CENTER_Y, cpmeas.COLTYPE_FLOAT),
            (object_name, M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
            (cpmeas.IMAGE, FF_COUNT%object_name, cpmeas.COLTYPE_INTEGER)]

def get_threshold_measurement_columns(image_name):
    '''Get the column definitions for threshold measurements, if made
    
    image_name - name of the image
    '''
    return [(cpmeas.IMAGE, FF_FINAL_THRESHOLD%image_name, cpmeas.COLTYPE_FLOAT),
            (cpmeas.IMAGE, FF_ORIG_THRESHOLD%image_name, cpmeas.COLTYPE_FLOAT),
            (cpmeas.IMAGE, FF_WEIGHTED_VARIANCE%image_name, cpmeas.COLTYPE_FLOAT),
            (cpmeas.IMAGE, FF_SUM_OF_ENTROPIES%image_name, cpmeas.COLTYPE_FLOAT)]

def draw_outline(img, outline, color):
    '''Draw the given outline on the given image in the given color'''
    red = float(color.Red()) / 255.0
    green = float(color.Green()) / 255.0
    blue = float(color.Blue()) / 255.0
    img[outline != 0, 0] = red
    img[outline != 0, 1] = green
    img[outline != 0, 2] = blue
                
class MaskObjectNameSubscriber(cps.ObjectNameSubscriber):
    '''This class allows the legacy "From Image" choice'''
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        
    def test_valid(self, pipeline):
        if self.value == O_FROM_IMAGE:
            return
        super(self.__class__, self).test_valid(pipeline)
