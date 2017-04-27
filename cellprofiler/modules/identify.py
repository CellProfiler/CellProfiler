import numpy
import scipy.ndimage

import cellprofiler.measurement
import cellprofiler.module

O_TWO_CLASS = 'Two classes'
O_THREE_CLASS = 'Three classes'

O_WEIGHTED_VARIANCE = 'Weighted variance'
O_ENTROPY = 'Entropy'

O_FOREGROUND = 'Foreground'
O_BACKGROUND = 'Background'

FI_IMAGE_SIZE = 'Image size'
FI_CUSTOM = 'Custom'

RB_DEFAULT = "Default"
RB_CUSTOM = "Custom"
RB_MEAN = "Mean"
RB_MEDIAN = "Median"
RB_MODE = "Mode"
RB_SD = "Standard deviation"
RB_MAD = "Median absolute deviation"

'''Threshold scope = automatic - use defaults of global + MCT, no adjustments'''
TS_AUTOMATIC = "Automatic"

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

TS_ALL = [TS_GLOBAL, TS_ADAPTIVE, TS_MANUAL, TS_MEASUREMENT]

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

PROTIP_RECOMEND_ICON = "thumb-up.png"
PROTIP_AVOID_ICON = "thumb-down.png"
TECH_NOTE_ICON = "gear.png"


class Identify(cellprofiler.module.Module):
    def get_object_categories(self, pipeline, object_name, object_dictionary):
        '''Get categories related to creating new children

        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of
                            an object created by this module and each
                            value is a list of names of parents.
        '''
        if object_name == cellprofiler.measurement.IMAGE:
            return [cellprofiler.measurement.C_COUNT]
        result = []
        if object_dictionary.has_key(object_name):
            result += [cellprofiler.measurement.C_LOCATION, cellprofiler.measurement.C_NUMBER]
            if len(object_dictionary[object_name]) > 0:
                result += [cellprofiler.measurement.C_PARENT]
        if object_name in reduce(lambda x, y: x + y, object_dictionary.values()):
            result += [cellprofiler.measurement.C_CHILDREN]
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
        if object_name == cellprofiler.measurement.IMAGE and category == cellprofiler.measurement.C_COUNT:
            return list(object_dictionary.keys())

        if object_dictionary.has_key(object_name):
            if category == cellprofiler.measurement.C_LOCATION:
                return [cellprofiler.measurement.FTR_CENTER_X, cellprofiler.measurement.FTR_CENTER_Y]
            elif category == cellprofiler.measurement.C_NUMBER:
                return [cellprofiler.measurement.FTR_OBJECT_NUMBER]
            elif category == cellprofiler.measurement.C_PARENT:
                return list(object_dictionary[object_name])
        if category == cellprofiler.measurement.C_CHILDREN:
            result = []
            for child_object_name in object_dictionary.keys():
                if object_name in object_dictionary[child_object_name]:
                    result += ["%s_Count" % child_object_name]
            return result
        return []


def add_object_location_measurements(measurements,
                                     object_name,
                                     labels, object_count=None):
    """Add the X and Y centers of mass to the measurements

    measurements - the measurements container
    object_name  - the name of the objects being measured
    labels       - the label matrix
    object_count - (optional) the object count if known, otherwise
                   takes the maximum value in the labels matrix which is
                   usually correct.
    """
    if object_count is None:
        object_count = numpy.max(labels)
    #
    # Get the centers of each object - center_of_mass <- list of two-tuples.
    #
    if object_count:
        centers = scipy.ndimage.center_of_mass(numpy.ones(labels.shape),
                                               labels,
                                               range(1, object_count + 1))
        centers = numpy.array(centers)
        centers = centers.reshape((object_count, 2))
        location_center_y = centers[:, 0]
        location_center_x = centers[:, 1]
        number = numpy.arange(1, object_count + 1)
    else:
        location_center_y = numpy.zeros((0,), dtype=float)
        location_center_x = numpy.zeros((0,), dtype=float)
        number = numpy.zeros((0,), dtype=int)
    measurements.add_measurement(object_name, cellprofiler.measurement.M_LOCATION_CENTER_X,
                                 location_center_x)
    measurements.add_measurement(object_name, cellprofiler.measurement.M_LOCATION_CENTER_Y,
                                 location_center_y)
    measurements.add_measurement(object_name, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER, number)


def add_object_location_measurements_ijv(measurements,
                                         object_name,
                                         ijv, object_count=None):
    '''Add object location measurements for IJV-style objects'''
    if object_count is None:
        object_count = 0 if ijv.shape[0] == 0 else numpy.max(ijv[:, 2])
    if object_count == 0:
        center_x = numpy.zeros(0)
        center_y = numpy.zeros(0)
    else:
        areas = numpy.zeros(object_count, int)
        areas_bc = numpy.bincount(ijv[:, 2])[1:]
        areas[:len(areas_bc)] = areas_bc
        center_x = numpy.bincount(ijv[:, 2], ijv[:, 1])[1:] / areas
        center_y = numpy.bincount(ijv[:, 2], ijv[:, 0])[1:] / areas
    measurements.add_measurement(object_name, cellprofiler.measurement.M_LOCATION_CENTER_X, center_x)
    measurements.add_measurement(object_name, cellprofiler.measurement.M_LOCATION_CENTER_Y, center_y)
    measurements.add_measurement(object_name, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                                 numpy.arange(1, object_count + 1))


def add_object_count_measurements(measurements, object_name, object_count):
    """Add the # of objects to the measurements"""
    measurements.add_measurement('Image',
                                 cellprofiler.measurement.FF_COUNT % object_name,
                                 numpy.array([object_count],
                                             dtype=float))


def get_object_measurement_columns(object_name):
    '''Get the column definitions for measurements made by identify modules

    Identify modules can use this call when implementing
    CPModule.get_measurement_columns to get the column definitions for
    the measurements made by add_object_location_measurements and
    add_object_count_measurements.
    '''
    return [(object_name, cellprofiler.measurement.M_LOCATION_CENTER_X, cellprofiler.measurement.COLTYPE_FLOAT),
            (object_name, cellprofiler.measurement.M_LOCATION_CENTER_Y, cellprofiler.measurement.COLTYPE_FLOAT),
            (object_name, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER, cellprofiler.measurement.COLTYPE_INTEGER),
            (cellprofiler.measurement.IMAGE, cellprofiler.measurement.FF_COUNT % object_name, cellprofiler.measurement.COLTYPE_INTEGER)]
