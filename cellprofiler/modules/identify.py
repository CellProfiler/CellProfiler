"""identify.py - a base class for common functionality for identify modules

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

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
from cellprofiler.cpmath.threshold import TM_MANUAL, TM_MEASUREMENT, TM_METHODS, get_threshold
from cellprofiler.cpmath.threshold import TM_GLOBAL, TM_BINARY_IMAGE, TM_MOG
from cellprofiler.cpmath.threshold import TM_OTSU
from cellprofiler.cpmath.threshold import weighted_variance, sum_of_entropies
from cellprofiler.gui.help import HELP_ON_PIXEL_INTENSITIES

O_TWO_CLASS = 'Two classes'
O_THREE_CLASS = 'Three classes'

O_WEIGHTED_VARIANCE = 'Weighted variance'
O_ENTROPY = 'Entropy'

O_FOREGROUND = 'Foreground'
O_BACKGROUND = 'Background'

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

'''The children category (e.g. Children_Cells_Count)'''
C_CHILDREN = "Children"

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

class Identify(cellprofiler.cpmodule.CPModule):
    def create_threshold_settings(self, methods = TM_METHODS):
        '''Create settings related to thresholding'''
        self.threshold_method = cps.Choice(
            'Select the thresholding method',
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
            
            <p>There are six methods for finding thresholds automatically:
            <ul><li><i>Otsu:</i> This method is probably best if you are not able 
            to make certain assumptions about every images in your experiment, 
            especially if the percentage of the image covered by regions 
            of interest varies substantially from image to image. Our implementation 
            takes into account the maximum and minimum values in the image and log-transforming the
            image prior to calculating the threshold. If you know that the percentage of 
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
            or foreground pixels, based on the estimated fraction provided by the user.</li></ol>
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
            Finally, in the <b>IdentifyPrimaryObjects</b> module, select the cropped image as input image.</ul>
            
            <p>Selecting <i>manual thresholding</i> allows you to enter a single value between 0 and 1
            as the threshold value. This setting can be useful when you are certain what the
            cutoff should be and it does not vary from image to image in the experiment. If you are 
            using this module to find objects in an image that is already binary (where the foreground is 1 and 
            the background is 0), a manual value of 0.5 will identify the objects.
            
            <p>Selecting thresholding via a <i>binary image</i> will use the binary image as a mask for the
            input image. Note that unlike <b>MaskImage</b>, the binary image will not be stored permanently
            as a mask. Also, even though no algorithm is actually used to find the threshold in this case, the 
            final threshold value is reported as the Otsu threshold calculated for the foreground region.
            
            <p>Selecting thresholding via <i>measurement</i> will use an image measurement previously calculated
            in order to threshold the image. Like manual thresholding, this setting can be useful when you are 
            certain what the cutoff should be. The difference in this case is that the desired threshold does 
            vary from image to image in the experiment but can be measured using a Measurement module. 
            """)

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
    
    def get_threshold_visible_settings(self):
        '''Return visible settings related to thresholding'''
        vv = [self.threshold_method]
        if self.threshold_method == TM_MANUAL:
            vv += [self.manual_threshold]
        elif self.threshold_method == TM_MEASUREMENT:
            vv += [self.thresholding_measurement]
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
        
    def get_threshold(self, image, mask, labels, workspace=None):
        """Compute the threshold using whichever algorithm was selected by the user
        image - image to threshold
        mask  - ignore pixels whose mask value is false
        labels - labels matrix that restricts thresholding to within the object boundary
        workspace - contains measurements (measurements for this run)
        returns: threshold to use (possibly an array) and global threshold
        """
        if self.threshold_method == TM_MANUAL:
            return self.manual_threshold.value, self.manual_threshold.value
        if self.threshold_method == TM_MEASUREMENT:
            m = workspace.measurements
            value = m.get_current_image_measurement(self.thresholding_measurement.value)
            value *= self.threshold_correction_factor.value
            if not self.threshold_range.min is None:
                value = max(value, self.threshold_range.min)
            if not self.threshold_range.max is None:
                value = min(value, self.threshold_range.max)
            return value, value
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
    
    def add_threshold_measurements(self, measurements, image, mask, 
                                   local_threshold, global_threshold,
                                   objname):
        '''Compute and add threshold statistics measurements 
        
        measurements - add the measurements here
        image - the image that was thresholded
        mask - mask of significant pixels
        local_threshold - either a per-pixel threshold (a matrix) or a 
                          copy of the global threshold (a scalar)
        global_threshold - the globally-calculated threshold
        objname - either the name of the objects that were created or,
                  for ApplyThreshold, the image created.
        '''
        if self.threshold_modifier == TM_GLOBAL:
            # The local threshold is a single number
            assert(not isinstance(local_threshold,np.ndarray))
            ave_threshold = local_threshold
        else:
            # The local threshold is an array
            ave_threshold = local_threshold.mean()
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_FINAL_THRESHOLD%(objname),
                                     np.array([ave_threshold],
                                                 dtype=float))
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_ORIG_THRESHOLD%(objname),
                                     np.array([global_threshold],
                                                  dtype=float))
        wv = weighted_variance(image, mask, local_threshold)
        measurements.add_measurement(cpmeas.IMAGE,
                                     FF_WEIGHTED_VARIANCE%(objname),
                                     np.array([wv],dtype=float))
        entropies = sum_of_entropies(image, mask, local_threshold)
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
        if self.threshold_method.value == TM_MANUAL or self.threshold_method.value == TM_MEASUREMENT:
            return TM_GLOBAL 
        parts = self.threshold_method.value.split(' ')
        return parts[1]
    
    threshold_modifier = property(get_threshold_modifier)
    
    def get_threshold_algorithm(self):
        """The thresholding algorithm, for instance TM_OTSU"""
        parts = self.threshold_method.value.split(' ')
        return parts[0]
    
    threshold_algorithm = property(get_threshold_algorithm)
    
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
            return [FTR_ORIG_THRESHOLD, FTR_FINAL_THRESHOLD,
                    FTR_SUM_OF_ENTROPIES, FTR_WEIGHTED_VARIANCE]
        return []
    
    def get_threshold_measurement_objects(self, pipeline, object_name, category, 
                                          measurement, child_object_name):
        '''Get the measurement objects for a threshold measurement
        
        pipeline - not used
        object_name - either "Image" or an object name. (must be "Image")
        category - the measurement category. (must be "Threshold")
        measurement - the feature being measured
        child_object_name - the name of the objects that are created by this
                            module.
        '''
        if (object_name == cpmeas.IMAGE and category == C_THRESHOLD and
            measurement in (FTR_ORIG_THRESHOLD, FTR_FINAL_THRESHOLD,
                            FTR_SUM_OF_ENTROPIES, FTR_WEIGHTED_VARIANCE)):
            return [child_object_name]
        else:
            return []
        
    def get_threshold_measurement_images(self, pipeline, object_name, category, 
                                          measurement, image_name):
        '''Get the measurement objects for a threshold measurement
        
        pipeline - not used
        object_name - either "Image" or an object name. (must be "Image")
        category - the measurement category. (must be "Threshold")
        measurement - the feature being measured
        child_object_name - the name of the image used during thresholding
        '''
        if (object_name == cpmeas.IMAGE and category == C_THRESHOLD and
            measurement in (FTR_ORIG_THRESHOLD, FTR_FINAL_THRESHOLD,
                            FTR_SUM_OF_ENTROPIES, FTR_WEIGHTED_VARIANCE)):
            return [image_name]
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
                
    
