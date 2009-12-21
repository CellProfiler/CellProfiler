"""identify.py - a base class for common functionality for identify modules

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision$"

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
from cellprofiler.cpmath.threshold import TM_MANUAL, TM_METHODS, get_threshold
from cellprofiler.cpmath.threshold import TM_GLOBAL, TM_BINARY_IMAGE, TM_MOG
from cellprofiler.cpmath.threshold import TM_OTSU

O_TWO_CLASS = 'Two classes'
O_THREE_CLASS = 'Three classes'

O_WEIGHTED_VARIANCE = 'Weighted variance'
O_ENTROPY = 'Entropy'

O_FOREGROUND = 'Foreground'
O_BACKGROUND = 'Background'

'''The centroid X coordinate measurement feature name'''
M_LOCATION_CENTER_X = 'Location_Center_X'

'''The centroid Y coordinate measurement feature name'''
M_LOCATION_CENTER_Y = 'Location_Center_Y'

'''The object number - an index from 1 to however many objects'''
M_NUMBER_OBJECT_NUMBER = 'Number_Object_Number'

'''The format for the object count image measurement'''
FF_COUNT = 'Count_%s' 

'''Format string for the FinalThreshold feature name'''
FF_FINAL_THRESHOLD = 'Threshold_FinalThreshold_%s'

'''Format string for the OrigThreshold feature name'''
FF_ORIG_THRESHOLD = 'Threshold_OrigThreshold_%s'

'''Format string for the WeightedVariance feature name'''
FF_WEIGHTED_VARIANCE = 'Threshold_WeightedVariance_%s'

'''Format string for the SumOfEntropies feature name'''
FF_SUM_OF_ENTROPIES = 'Threshold_SumOfEntropies_%s'

'''Format string for # of children per parent feature name'''
FF_CHILDREN_COUNT = "Children_%s_Count"

'''Format string for parent of child feature name'''
FF_PARENT = "Parent_%s"

class Identify(cellprofiler.cpmodule.CPModule):
    def create_threshold_settings(self, methods = TM_METHODS):
        '''Create settings related to thresholding'''
        self.threshold_method = cps.Choice(
            'Select the thresholding method',
            methods, doc='''\
            The threshold affects the stringency of the lines between the objects 
            and the background. You can have the threshold automatically calculated 
            using several methods, or you can enter an absolute number between 0 
            and 1 for the threshold (to see the pixel intensities for your images 
            in the appropriate range of 0 to 1, use <i>Tools &gt; Show pixel data</i> 
            in a window showing your image). There are advantages either way. 
            An absolute number treats every image identically, but is not robust 
            to slight changes in lighting/staining conditions between images. An
            automatically calculated threshold adapts to changes in
            lighting/staining conditions between images and is usually more
            robust/accurate, but it can occasionally produce a poor threshold for
            unusual/artifactual images. It also takes a small amount of time to
            calculate.
            
            <p>The threshold which is used for each image is recorded as a
            measurement in the output file, so if you find unusual measurements from
            one of your images, you might check whether the automatically calculated
            threshold was unusually high or low compared to the other images.
            
            <p>There are six methods for finding thresholds automatically:
            <ul><li><i>Otsu:</i> This method is probably best if you don't know 
            anything about the image, or if the percent of the image covered by 
            objects varies substantially from image to image. Our implementation 
            takes into account the max and min values in the image and log-transforming the
            image prior to calculating the threshold. If you know that the object 
            coverage percentage does not vary much from image
            to image, the MoG method can be better, especially if the coverage percentage is
            not near 50%. Note, however, that the MoG function is experimental and
            has not been thoroughly validated. </li>
            <li><i>Mixture of Gaussian (MoG):</i>This function assumes that the 
            pixels in the image belong to either a background class or an object
            class, using an initial guess of the fraction of the image that is 
            covered by objects. Essentially, there are two steps:
            <ol><li>First, a number of Gaussian distributions are estimated to 
            match the distribution of pixel intensities in the image. Currently 
            three Gaussian distributions are fitted, one corresponding to a 
            background class, one corresponding to an object class, and one 
            distribution for an intermediate class. The distributions are fitted
            using the Expectation-Maximization algorithm, a procedure referred 
            to as Mixture of Gaussians modeling. </li>
            <li>When the three Gaussian distributions have been fitted, a decsion 
            is made whether the intermediate class models the background pixels 
            or object pixels based on the fraction provided by the user.</li></ol>
            <li><i>Background:</i> This method is simple and appropriate for images in 
            which most of the image is background. It finds the mode of the 
            histogram of the image, which is assumed to be the background of the 
            image, and chooses a threshold at twice that value (which you can 
            adjust with a Threshold Correction Factor,
            see below).  Note that the mode is protected from a high number of 
            saturated pixels by only counting pixels < 0.95. This can be very helpful,
            for example, if your images vary in overall brightness but the objects of 
            interest are always twice (or actually, any constant) as bright as the 
            background of the image. </li>
            <li><i>Robust background:</i> This method trims the brightest and 
            dimmest 5% of pixel intensities in the hopes that the remaining pixels 
            represent a gaussian of intensity values that are mostly background 
            pixels. It then calculates the mean and standard deviation of the 
            remaining pixels and calculates the threshold as the mean + 2 times 
            the standard deviation.</li>
            <li><i>Ridler-Calvard:</i> This method is simple and its results are
            often very similar to Otsu's - according to
            Sezgin and Sankur's paper (<i>Journal of Electronic Imaging</i>, 2004), Otsu's 
            overall quality on testing 40 nondestructive testing images is slightly 
            better than Ridler's (Average error - Otsu: 0.318, Ridler: 0.401). 
            It chooses an initial threshold, and then iteratively calculates the next 
            one by taking the mean of the average intensities of the background and 
            foreground pixels determined by the first threshold, repeating this until 
            the threshold converges.</li>
            <li><i>Kapur:</i> This method computes the threshold of an image by
            log-transforming its values, then searching for the threshold that
            maximizes the sum of entropies of the foreground and background
            pixel values, when treated as separate distributions.</li>
            </ul>
            
            <p>You can also choose between <i>Global</i>, <i>Adaptive</i>, and 
            <i>Per-Object</i> thresholding for the automatic methods:
            <ul>
            <li><i>Global:</i> One threshold is used for the entire image (fast)</li>
            <li><i>Adaptive:</i> The threshold varies across the image - a bit slower but
            provides more accurate edge determination which may help to separate
            clumps, especially if you are not using a clump-separation method </li>
            <li><i>Per-Object:</i> If you are using this module to find child objects located
            <i>within</i> parent objects, the per object method will calculate a distinct
            threshold for each parent object. This is especially helpful, for
            example, when the background brightness varies substantially among the
            parent objects. 
            <p>Important: the per-object method requires that you run an
            IdentifyPrimAutomatic module to identify the parent objects upstream in the
            pipeline. After the parent objects are identified in the pipeline, you
            must then also run a <b>Crop</b> module with the following inputs: 
            <ul>
            <li>The input image is the image containing the sub-objects to be identified.</li>
            <li>Select <i>Objects</i> as the shape to crop into.</li>
            <li>Select the parent objects (e.g., Nuclei) as the objects to use as a cropping mask.</li>
            </ul>
            Finally, in the IdentifyPrimAutomatic module, select the cropped image as input image.</ul>
            
            <p>Selecting manual thresholding allows you to enter a single value between 0 and 1
            as the threshold value. This setting can be useful when you are certain what the
            cutoff should be. Also, in the case of a binary image (where the foreground is 1 and 
            the background is 0), a manual value of 0.5 will identify the objects.
            
            <p>Selecting a binary image will essentially use the binary image as a mask for the
            input image.''')

        self.threshold_correction_factor = cps.Float('Threshold correction factor', 1,
                                                doc="""\
            When the threshold is calculated automatically, it may consistently be
            too stringent or too lenient. You may need to enter an adjustment factor
            which you empirically determine is suitable for your images. The number 1
            means no adjustment, 0 to 1 makes the threshold more lenient and greater
            than 1 (e.g. 1.3) makes the threshold more stringent. For example, the
            Otsu automatic thresholding inherently assumes that 50% of the image is
            covered by objects. If a larger percentage of the image is covered, the
            Otsu method will give a slightly biased threshold that may have to be
            corrected using this setting.""")
        
        self.threshold_range = cps.FloatRange('Lower and upper bounds on threshold:', (0,1), minval=0,
                                         maxval=1, doc="""\
            In the range [0,1].  May be used as a safety precaution when the threshold is calculated
            automatically. For example, if there are no objects in the field of view,
            the automatic threshold will be unreasonably low. In such cases, the
            lower bound you enter here will override the automatic threshold.""")
        
        self.object_fraction = cps.CustomChoice(
            'Approximate fraction of image covered by objects?', 
            ['0.01','0.1','0.2','0.3', '0.4','0.5','0.6','0.7', '0.8','0.9',
             '0.99'], doc="""\
            <i>(Only used when applying the Mixture of Gaussian thresholding method)</i>
            <p>An estimate of how much of the image is covered with objects, which
            is used to estimate the distribution of pixel intensities.""")
        
        self.manual_threshold = cps.Float("Enter manual threshold:", 
                                          value=0.0, minval=0.0, maxval=1.0,doc="""\
            <i>(Only used if Manual selected for thresholding method)</i>
            <p>Enter the value that will act as an absolute threshold for the image""")
        
        self.binary_image = cps.ImageNameSubscriber(
            "Select binary image:", "None", doc = """What is the binary thresholding image?""")
        
        self.two_class_otsu = cps.Choice(
            'Two-class or three-class thresholding?',
            [O_TWO_CLASS, O_THREE_CLASS],doc="""
            <i>(Only used for the Otsu thresholding method)</i> 
            <p>Select <i>Two</i> if the grayscale levels are readily distinguishable into foregound 
            (i.e., objects) and background. Select <i>Three</i> if there is an 
            middle set of grayscale levels which belong to neither the
            foreground nor background. 
            <p>For example, three-class thresholding may
            be useful for images in which you have nuclear staining along with a
            low-intensity non-specific cell staining. Where two-class thresholding
            might incorrectly assign this intemediate staining to the nuclei 
            objects, three-class thresholding allows you to assign it to the 
            foreground or background as desired. However, in extreme cases where either 
            there are almost no objects or the entire field of view is covered with 
            objects, three-class thresholding may perform worse than two-class.""")
        
        self.use_weighted_variance = cps.Choice(
            'Minimize the weighted variance or the entropy?',
            [O_WEIGHTED_VARIANCE, O_ENTROPY])
        
        self.assign_middle_to_foreground = cps.Choice(
            'Assign pixels in the middle intensity class to the foreground '
            'or the background?', [O_FOREGROUND, O_BACKGROUND],doc="""
            <i>Only used for the Otsu thresholding method with three-class thresholding)</i>
            <p>Select whether you want the middle grayscale intensities to be assigned 
            to the foreground pixels or the background pixels.""")
    
    def get_threshold_visible_settings(self):
        '''Return visible settings related to thresholding'''
        vv = [self.threshold_method]
        if self.threshold_method == TM_MANUAL:
            vv += [self.manual_threshold]
        elif self.threshold_method == TM_BINARY_IMAGE:
            vv += [self.binary_image]
        if self.threshold_algorithm == TM_OTSU:
            vv += [self.two_class_otsu, self.use_weighted_variance]
            if self.two_class_otsu == O_THREE_CLASS:
                vv.append(self.assign_middle_to_foreground)
        if self.threshold_algorithm == TM_MOG:
            vv += [self.object_fraction]
        if not self.threshold_method in (TM_MANUAL, TM_BINARY_IMAGE):
            vv += [ self.threshold_correction_factor, self.threshold_range]
        return vv
        
    def get_threshold(self, image, mask, labels):
        """Compute the threshold using whichever algorithm was selected by the user
        image - image to threshold
        mask  - ignore pixels whose mask value is false
        labels - labels matrix that restricts thresholding to within the object boundary
        returns: threshold to use (possibly an array) and global threshold
        """
        if self.threshold_method == TM_MANUAL:
            return self.manual_threshold.value, self.manual_threshold.value
        object_fraction = self.object_fraction.value
        if object_fraction.endswith("%"):
            object_fraction = float(object_fraction[:-1])/100.0
        else:
            object_fraction = float(object_fraction)
        return get_threshold(
            self.threshold_algorithm,
            self.threshold_modifier,
            image, 
            mask = mask,
            labels = labels,
            threshold_range_min = self.threshold_range.min,
            threshold_range_max = self.threshold_range.max,
            threshold_correction_factor = self.threshold_correction_factor.value,
            object_fraction = object_fraction,
            two_class_otsu = self.two_class_otsu.value == O_TWO_CLASS,
            use_weighted_variance = self.use_weighted_variance.value == O_WEIGHTED_VARIANCE,
            assign_middle_to_foreground = self.assign_middle_to_foreground.value == O_FOREGROUND)
    
    def validate_module(self, pipeline):
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
        if self.threshold_method.value == TM_MANUAL:
            return TM_GLOBAL 
        parts = self.threshold_method.value.split(' ')
        return parts[1]
    
    threshold_modifier = property(get_threshold_modifier)
    
    def get_threshold_algorithm(self):
        """The thresholding algorithm, for instance TM_OTSU"""
        parts = self.threshold_method.value.split(' ')
        return parts[0]
    
    threshold_algorithm = property(get_threshold_algorithm)


def add_object_location_measurements(measurements, 
                                     object_name,
                                     labels):
    """Add the X and Y centers of mass to the measurements
    
    measurements - the measurements container
    object_name  - the name of the objects being measured
    labels       - the label matrix
    """
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

