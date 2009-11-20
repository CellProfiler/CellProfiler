'''<b>FlagImage</b>: This module allows you to flag an image if it fails some quality control
measurement you specify. 
<hr>

This module allows the user to assign a flag if
an image fails some quality control measurement the user specifies.  The
value of the measurement is '1' if the image has failed QC, and '0' if it
has passed. The flag can be used in post-processing to filter out images
the user does not want to analyze in CPAnalyst. Additionally, you can
use ExportToExcel to generate a file that includes the measurement as metadata
associated with the images. This file can then be used by the LoadText 
module to put images that pass QC into one group and the images that fail 
into another. If you plan to use a flag in LoadText, give it a category of
"Metadata" so that it can be used in grouping.

The flag is stored in a measurement whose name is a combination of the
flag's category and feature name. For instance, the default category is
"Metadata" and the default feature name is "QCFlag". The default
measurement name is "Metadata_QCFlag".

A flag can be based on one or more measurements. If you create a flag based
on more than one measurement, you'll have to choose between setting the
flag if all measurements are outside the bounds or if one of the measurements
is outside of the bounds.

This module requires the measurement modules be placed prior to this
module in the pipeline.

'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
import sys
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.gui.cpfigure as cpf

C_ANY = "Flag if any fail"
C_ALL = "Flag if all fail"

S_IMAGE = "Image"
S_AVERAGE_OBJECT = "Average for objects"
S_ALL_OBJECTS = "All objects"
S_ALL = [S_IMAGE, S_AVERAGE_OBJECT, S_ALL_OBJECTS]

'''Number of settings in the module, aside from those in the flags'''
N_FIXED_SETTINGS = 1
'''Number of settings in each flag, aside from those in the measurements'''
N_FIXED_SETTINGS_PER_FLAG = 4
'''Number of settings per measurement'''
N_SETTINGS_PER_MEASUREMENT = 7

class FlagImage(cpm.CPModule):
   
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "FlagImage"
    
    def create_settings(self):
        self.flags = []
        self.flag_count = cps.HiddenCount(self.flags)
        self.add_flag_button = cps.DoSomething("Add another QC flag",
                                               "Add flag",
                                               self.add_flag)
        self.add_flag(False)
        
    def add_flag(self, can_delete=True):
        self.flags.append(FlagSettings(self.flags, can_delete))
    
    def settings(self):
        result = [self.flag_count]
        for flag in self.flags:
            result += flag.settings()
        return result
    
    def prepare_settings(self, setting_values):
        '''Construct the correct number of flags'''
        flag_count = int(setting_values[0])
        while len(self.flags) > flag_count:
            del self.flags[-1]
        while len(self.flags) < flag_count:
            self.add_flag()
            
        setting_values = setting_values[N_FIXED_SETTINGS:]
        for flag in self.flags:
            assert isinstance(flag, FlagSettings)
            setting_values = flag.prepare_settings(setting_values)
    
    def visible_settings(self):
        result = []
        for flag in self.flags:
            result += flag.visible_settings()
        result += [self.add_flag_button]
        return result
    
    def run(self, workspace):
        statistics = [ ("Flag", "Source", "Measurement", "Value","Pass/Fail")]
        for flag in self.flags:
            statistics += self.run_flag(workspace, flag)
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            assert isinstance(figure, cpf.CPFigureFrame)
            figure.subplot_table(0,0, statistics,
                                 (.25,.25,.25,.125,.125))

    def run_flag(self, workspace, flag):
        assert isinstance(flag, FlagSettings)
        ok, stats = self.eval_measurement(workspace, 
                                          flag.measurement_settings[0])
        statistics = [tuple([flag.measurement_name] + list(stats))]
        for measurement_setting in flag.measurement_settings[1:]:
            ok_1, stats = self.eval_measurement(workspace, measurement_setting)
            statistics += [tuple([flag.measurement_name] + list(stats))]
            if flag.combination_choice == C_ALL:
                ok = ok or ok_1
            elif flag.combination_choice == C_ANY:
                ok = ok and ok_1
            else:
                raise NotImplementedError("Unimplemented combination choice: %s" %
                                          flag.combination_choice.value)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_image_measurement(flag.measurement_name, 0 if ok else 1)
        return statistics
        
    def eval_measurement(self, workspace, ms):
        '''Evaluate a measurement
        
        workspace - holds the measurements to be evaluated
        ms - the MeasurementSettings indicating how to evaluate
        
        returns a tuple
           first tuple element is True = pass, False = Fail
           second tuple element has all of the statistics except for the
                        flag name
        '''
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        assert isinstance(ms, MeasurementSettings)
        if ms.source_choice == S_IMAGE:
            value = m.get_current_image_measurement(ms.measurement.value)
            min_value = max_value = value
            display_value = str(round(value,3))
            source = cpmeas.IMAGE
        elif ms.source_choice == S_AVERAGE_OBJECT:
            data = m.get_current_measurement(ms.object_name.value,
                                             ms.measurement.value)
            min_value = max_value = np.mean(data)
            display_value = str(round(min_value, 3))
            source = "Ave. %s"%ms.object_name.value
        elif ms.source_choice == S_ALL_OBJECTS:
            data = m.get_current_measurement(ms.object_name.value,
                                             ms.measurement.value)
            source = ms.object_name.value
            min_value = np.min(data)
            max_value = np.max(data)
            if min_value == max_value:
                display_value = str(min_value)
            else:
                display_value = "%.3f - %.3f"%(min_value,max_value)
        else:
            raise NotImplementedError("Source choice of %s not implemented" %
                                      ms.source_choice)
        fail = ((ms.wants_minimum.value and 
                 min_value < ms.minimum_value.value) or
                (ms.wants_maximum.value and
                 max_value > ms.maximum_value.value))
        return ((not fail), (source, ms.measurement.value, display_value, 
                             "Fail" if fail else "Pass"))
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for each flag mesurment in the module'''
        return [(cpmeas.IMAGE, flag.measurement_name, cpmeas.COLTYPE_INTEGER)
                for flag in self.flags]
    
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [flag.category.value for flag in self.flags]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name != cpmeas.IMAGE:
            return []
        return [flag.feature_name.value for flag in self.flags
                if flag.category.value == category]

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            image_name, category, feature_num_or_name, min_value, max_value, \
                      new_or_append, new_name, old_name = setting_values
            measurement_name = '_'.join((category, feature_num_or_name,
                                         image_name))
            if min_value == 'No minimum':
                wants_minimum = cps.NO
                min_value = "0"
            else:
                wants_minimum = cps.YES
            if max_value == 'No maximum':
                wants_maximum = cps.NO
                max_value = "1"
            else:
                wants_maximum = cps.YES
            if new_or_append == "Append existing flag":
                sys.stderr.write("WARNING: CellProfiler 2.0 can't combine flags from multiple FlagImageForQC modules imported from version 1.0\n")
            
            new_name_split = new_name.find('_')
            if new_name_split == -1:
                flag_category = 'Metadata'
                flag_feature = new_name
            else:
                flag_category = new_name[:new_name_split]
                flag_feature = new_name[new_name_split+1:]
            setting_values = ["1", # of flags in module
                              "1", # of measurements in the flag
                              flag_category,
                              flag_feature,
                              C_ANY, # combination choice
                              S_IMAGE, # measurement source
                              "None", # object name
                              measurement_name,
                              wants_minimum,
                              min_value,
                              wants_maximum,
                              max_value]
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab
    
class MeasurementSettings(object):
    # XXX needs to use cps.SettingsGroup
    '''Represents the settings for one flag measurement '''
    def __init__(self, measurements, can_delete = True):
        self.can_delete = can_delete
        self.key = uuid.uuid4()
        def remove(measurements = measurements):
            index = [x.key for x in measurements].index(self.key)
            del measurements[index]
        self.source_choice = cps.Choice(
            "Do you want to filter on an image measurement, "
            "on the average value of an object measurement, or "
            "on the values of all objects in the image?", S_ALL, doc = '''<ul><li>Image: This will flag an image based
            on a per-image measurement, such as intensity or granularity.</li><li>Average for objects: This will flag
            an image based on the average of all object measurements in an image.</li>
            <li>All objects: This will flag an image based on all the object measurements in an image, without averaging.
             </li></ul>''')
        self.object_name = cps.ObjectNameSubscriber(
            "Select the object to filter by",
            "None", doc = '''What did you call the objects whose measurements you want to filter by?''')
        def object_fn():
            if self.source_choice == S_IMAGE:
                return cpmeas.IMAGE
            return self.object_name.value
        
        self.measurement = cps.Measurement("What measurement do you want to use?",
                                           object_fn)
        self.wants_minimum = cps.Binary("Do you want to flag images based on low values?",
                                        True, doc = '''Low values: Images with measurements below this cutoff will be flagged.''')
        self.minimum_value = cps.Float("What is the minimum value for the measurement?", 0)
        self.wants_maximum = cps.Binary("Do you want to flag images based on high values?",
                                        True, doc = '''High values: Images with measurements above this cutoff will be flagged.''')
        self.maximum_value = cps.Float("What is the maximum value for the measurement?", 1)
        
        if self.can_delete:
            self.remove_button = cps.DoSomething("Remove this measurement",
                                                 "Remove measurement",
                                                 remove)
    def settings(self):
        '''Return the settings to save or load from a pipeline'''
        return [self.source_choice, self.object_name, self.measurement,
                self.wants_minimum, self.minimum_value,
                self.wants_maximum, self.maximum_value]
    
    def visible_settings(self):
        '''Return the settings to show the user'''
        result = [self.source_choice]
        if self.source_choice != S_IMAGE:
            result += [self.object_name]
        result += [self.measurement, self.wants_minimum]
        if self.wants_minimum.value:
            result += [self.minimum_value]
        result += [self.wants_maximum]
        if self.wants_maximum.value:
            result += [self.maximum_value]
        if self.can_delete:
            result += [self.remove_button]
        return result
    
class FlagSettings(object):
    '''Represents the settings for a QC flag in the FlagImages module'''
    def __init__(self, flags, can_delete = True):
        self.measurement_settings = []
        self.can_delete = can_delete
        self.key = uuid.uuid4()
        def remove(flags=flags):
            index = [x.key for x in flags].index(self.key)
            del flags[index]
        self.measurement_count = cps.HiddenCount(self.measurement_settings)
        self.category = cps.Text("What is the flag's measurement category?",
                                 "Metadata", doc = '''The default is 'Metadata', which allows you to group images
                                 by quality if loading the QCFlag via LoadText.  Otherwise, the flag can be stored
                                 in the 'Image' category.''')
        self.feature_name = cps.Text("What is the flag's feature name ?"
                                     ,"QCFlag", doc = "The default name of the flag's measurement is "
                                     "Metadata_QCFlag.")
        self.combination_choice = cps.Choice(
            "Do you want to set the flag if any measurement fails to meet the criteria or if all measurements fail to meet the criteria?",
            [ C_ANY, C_ALL], doc = '''<ul><li>Any: An image will be assigned a flag if any of its measurements fail. This can be useful
            for capturing images possessing varied QC flaws; for example, you can flag all bright images and all out of focus images with one flag.</li>
            <li>All: A flag will only be assigned if all measurements fail.  This can be useful for capturing images that possess only a combination
            of QC flaws; for example, you can flag only images that are both bright and out of focus.</li></ul>''')
        self.add_measurement_button = cps.DoSomething("Add another measurement",
                                                      "Add measurement",
                                                      self.add_measurement)
        self.add_measurement(False)
        if can_delete:
            self.remove_button = cps.DoSomething("Remove this flag",
                                                 "Remove flag",
                                                 remove)
    @property
    def measurement_name(self):
        '''The name to use when storing the flag in measurements'''
        return "_".join((self.category.value, self.feature_name.value))
    
    def add_measurement(self, can_delete = True):
        self.measurement_settings.append(
            MeasurementSettings(self.measurement_settings, can_delete))
        
    def settings(self):
        result = [self.measurement_count, self.category, self.feature_name, 
                  self.combination_choice]
        for measurement_setting in self.measurement_settings:
            result += measurement_setting.settings()
        return result
    
    def prepare_settings(self, setting_values):
        '''Create the appropriate number of measurements
        
        setting_values - the setting values, starting from those for this flag
        
        returns a sequence of setting values not consumed by this flag
        '''
        count = int(setting_values[0])
        while len(self.measurement_settings) > count:
            del self.measurement_settings[-1]
        while len(self.measurement_settings) < count:
            self.add_measurement()
        return setting_values[N_FIXED_SETTINGS_PER_FLAG +
                              count * N_SETTINGS_PER_MEASUREMENT:]
    
    def visible_settings(self):
        result = [self.category, self.feature_name]
        if len(self.measurement_settings) > 1:
            result += [self.combination_choice]
        for measurement_setting in self.measurement_settings:
            result += measurement_setting.visible_settings()
        result += [self.add_measurement_button]
        if self.can_delete:
            result += [self.remove_button]
        return result
