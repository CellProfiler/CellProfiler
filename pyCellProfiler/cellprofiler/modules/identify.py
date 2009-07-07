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
import numpy
import scipy.stats

import cellprofiler.cpmodule
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.cpmath.outline
import cellprofiler.objects
from cellprofiler.cpmath.smooth import smooth_with_noise
from cellprofiler.cpmath.threshold import TM_MANUAL, TM_METHODS, get_threshold
from cellprofiler.cpmath.threshold import TM_GLOBAL

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
    def get_threshold(self, image, mask, labels):
        """Compute the threshold using whichever algorithm was selected by the user
        image - image to threshold
        mask  - ignore pixels whose mask value is false
        labels - labels matrix that restricts thresholding to within the object boundary
        returns: threshold to use (possibly an array) and global threshold
        """
        if self.threshold_method == TM_MANUAL:
            return self.manual_threshold.value, self.manual_threshold.value
        return get_threshold(
            self.threshold_algorithm,
            self.threshold_modifier,
            image, 
            mask = mask,
            labels = labels,
            threshold_range_min = self.threshold_range.min,
            threshold_range_max = self.threshold_range.max,
            threshold_correction_factor = self.threshold_correction_factor.value,
            object_fraction = self.object_fraction.value,
            two_class_otsu = self.two_class_otsu.value == O_TWO_CLASS,
            use_weighted_variance = self.use_weighted_variance.value == O_WEIGHTED_VARIANCE,
            assign_middle_to_foreground = self.assign_middle_to_foreground.value == O_FOREGROUND)
    

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
    object_count = numpy.max(labels)
    #
    # Get the centers of each object - center_of_mass <- list of two-tuples.
    #
    if object_count:
        centers = scipy.ndimage.center_of_mass(numpy.ones(labels.shape), 
                                               labels, 
                                               range(1,object_count+1))
        centers = numpy.array(centers)
        centers = centers.reshape((object_count,2))
        location_center_y = centers[:,0]
        location_center_x = centers[:,1]
    else:
        location_center_y = numpy.zeros((0,),dtype=float)
        location_center_x = numpy.zeros((0,),dtype=float)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X,
                                 location_center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y,
                                 location_center_y)

def add_object_count_measurements(measurements, object_name, object_count):
    """Add the # of objects to the measurements"""
    measurements.add_measurement('Image',
                                 FF_COUNT%(object_name),
                                 numpy.array([object_count],
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
            (cpmeas.IMAGE, FF_COUNT%object_name, cpmeas.COLTYPE_INTEGER)]
