import centrosome.smooth
import centrosome.threshold
import numpy
import scipy.ndimage

import applythreshold
import cellprofiler.gui.help
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting

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
M_LOCATION_CENTER_X = '%s_%s' % (C_LOCATION, FTR_CENTER_X)

FTR_CENTER_Y = "Center_Y"
'''The centroid Y coordinate measurement feature name'''
M_LOCATION_CENTER_Y = '%s_%s' % (C_LOCATION, FTR_CENTER_Y)

FTR_OBJECT_NUMBER = "Object_Number"
'''The object number - an index from 1 to however many objects'''
M_NUMBER_OBJECT_NUMBER = '%s_%s' % (C_NUMBER, FTR_OBJECT_NUMBER)

'''The format for the object count image measurement'''
FF_COUNT = '%s_%%s' % C_COUNT

FTR_FINAL_THRESHOLD = "FinalThreshold"

'''Format string for the FinalThreshold feature name'''
FF_FINAL_THRESHOLD = '%s_%s_%%s' % (C_THRESHOLD, FTR_FINAL_THRESHOLD)

FTR_ORIG_THRESHOLD = "OrigThreshold"

'''Format string for the OrigThreshold feature name'''
FF_ORIG_THRESHOLD = '%s_%s_%%s' % (C_THRESHOLD, FTR_ORIG_THRESHOLD)

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
    def __init__(self):
        self.apply_threshold = applythreshold.ApplyThreshold()

        super(Identify, self).__init__()

    def create_settings(self):
        self.threshold_setting_version = cellprofiler.setting.Integer(
            "Threshold setting version",
            value=self.apply_threshold.variable_revision_number
        )

        for setting_name, setting in vars(self.apply_threshold).iteritems():
            if isinstance(setting, cellprofiler.setting.Setting):
                if isinstance(setting, cellprofiler.setting.ImageNameProvider) or \
                        isinstance(setting, cellprofiler.setting.ImageNameSubscriber):
                    continue

                setattr(self, setting_name, setting)

        self.threshold_smoothing_scale.value = 1.3488  # sigma = 1

    def settings(self):
        settings = [self.threshold_setting_version]

        settings += self.apply_threshold.settings()[2:]

        return settings

    def help_settings(self):
        return self.apply_threshold.help_settings()[2:]

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        upgrade_settings = self.apply_threshold.upgrade_threshold_settings(setting_values)

        upgrade_settings[0] = str(self.apply_threshold.variable_revision_number)

        return upgrade_settings, self.apply_threshold.variable_revision_number, False

    def visible_settings(self):
        return self.apply_threshold.visible_settings()[2:]

    def threshold_image(self, image_name, workspace, automatic=False):
        input = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        data = input.pixel_data

        mask = input.mask

        local_threshold, global_threshold = self.apply_threshold.get_threshold(data, mask, workspace, automatic)

        self.apply_threshold.add_threshold_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            local_threshold,
            global_threshold
        )

        binary_image, sigma = self.apply_threshold.apply_threshold(data, mask, local_threshold, automatic)

        self.apply_threshold.add_fg_bg_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            data,
            mask,
            binary_image
        )

        if hasattr(workspace, "display_data"):
            workspace.display_data.threshold_sigma = sigma

            if hasattr(workspace.display_data, "statistics"):
                workspace.display_data.statistics.append(["Threshold", "%0.3g" % global_threshold])

        return binary_image

    def get_measurement_objects_name(self):
        '''Return the name of the measurement objects

        Identify modules store measurements in the Image table and append an object name to distinguish between
        different thresholds in the same pipeline.
        '''
        raise NotImplementedError("Please implement get_measurement_objects_name() for this module")

    def get_measurement_columns(self, pipeline):
        return applythreshold.image_measurement_columns(self.get_measurement_objects_name())

    def get_categories(self, pipeline, object_name):
        return self.apply_threshold.get_categories(pipeline, object_name)

    def get_measurements(self, pipeline, object_name, category):
        return self.apply_threshold.get_measurements(pipeline, object_name, category)

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.get_measurement_objects_name()]

        return []

    def get_object_categories(self, pipeline, object_name, object_dictionary):
        '''Get categories related to creating new children

        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of
                            an object created by this module and each
                            value is a list of names of parents.
        '''
        if object_name == cellprofiler.measurement.IMAGE:
            return [C_COUNT]
        result = []
        if object_dictionary.has_key(object_name):
            result += [C_LOCATION, C_NUMBER]
            if len(object_dictionary[object_name]) > 0:
                result += [C_PARENT]
        if object_name in reduce(lambda x, y: x + y, object_dictionary.values()):
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
        if object_name == cellprofiler.measurement.IMAGE and category == C_COUNT:
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
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X,
                                 location_center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y,
                                 location_center_y)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER, number)


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
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X, center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y, center_y)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER,
                                 numpy.arange(1, object_count + 1))


def add_object_count_measurements(measurements, object_name, object_count):
    """Add the # of objects to the measurements"""
    measurements.add_measurement('Image',
                                 FF_COUNT % object_name,
                                 numpy.array([object_count],
                                             dtype=float))


def get_object_measurement_columns(object_name):
    '''Get the column definitions for measurements made by identify modules

    Identify modules can use this call when implementing
    CPModule.get_measurement_columns to get the column definitions for
    the measurements made by add_object_location_measurements and
    add_object_count_measurements.
    '''
    return [(object_name, M_LOCATION_CENTER_X, cellprofiler.measurement.COLTYPE_FLOAT),
            (object_name, M_LOCATION_CENTER_Y, cellprofiler.measurement.COLTYPE_FLOAT),
            (object_name, M_NUMBER_OBJECT_NUMBER, cellprofiler.measurement.COLTYPE_INTEGER),
            (cellprofiler.measurement.IMAGE, FF_COUNT % object_name, cellprofiler.measurement.COLTYPE_INTEGER)]
