'''<b>Flag Image</b> allows you to flag an image based on properties that you specify, for example, quality control measurements
<hr>
This module allows the user to assign a flag if
an image meets certain measurement criteria that the user specifies (for example, if the image fails a quality control measurement).  The
value of the flag is '1' if the image meets the selected criteria (for example, if it fails QC), and '0' if it
does not meet the criteria (if it passes QC). The flag can be used in post-processing to filter out images
the user does not want to analyze, e.g. in CellProfiler Analyst. Additionally, you can
use ExportToExcel to generate a file that includes the measurement as metadata
associated with the images. This file can then be used by the LoadText 
module to put images that pass QC into one group and images that fail 
into another. If you plan to use a flag in LoadText, give it a category of
"Metadata" so that it can be used in grouping.

The flag is stored as a Per-image measurement whose name is a combination of the
flag's category and feature name, underscore delimited. 
For instance, if the measurement category is
"Metadata" and the feature name is "QCFlag", then the default
measurement name would be "Metadata_QCFlag".

A flag can be based on one or more measurements. If you create a flag based
on more than one measurement, you'll have to choose between setting the
flag if all measurements are outside the bounds or if one of the measurements
is outside of the bounds.

This module must be placed in the pipeline after the relevant measurement 
modules upon which the flags are based.
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

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.gui.cpfigure as cpf

C_ANY = "Flag if any fail"
C_ALL = "Flag if all fail"

S_IMAGE = "Whole-image measurement"
S_AVERAGE_OBJECT = "Average measurement for all objects in each image"
S_ALL_OBJECTS = "Measurement for all objects in each image"
S_ALL = [S_IMAGE, S_AVERAGE_OBJECT, S_ALL_OBJECTS]

'''Number of settings in the module, aside from those in the flags'''
N_FIXED_SETTINGS = 1
'''Number of settings in each flag, aside from those in the measurements'''
N_FIXED_SETTINGS_PER_FLAG = 4
'''Number of settings per measurement'''
N_SETTINGS_PER_MEASUREMENT = 7

# XXX - needs some Dividers to space things out

class FlagImage(cpm.CPModule):
   
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "FlagImage"
    
    def create_settings(self):
        self.flags = []
        self.flag_count = cps.HiddenCount(self.flags)
        self.add_flag_button = cps.DoSomething("", "Add another flag",
                                               self.add_flag)
        self.add_flag(False)
        
    def add_flag(self, can_delete=True):
        group = cps.SettingsGroup()
        group.append("measurement_settings", [])
        group.append("measurement_count", cps.HiddenCount(group.measurement_settings))
        group.append("category", cps.Text("Flag's measurement category?",
                                 "Metadata", doc = '''Choose the measurement category to flag.  The default is 'Metadata', which allows you to group images
                                 by quality if loading the QCFlag via LoadText.  Otherwise, the flag can be stored
                                 in the 'Image' category.'''))
        group.append("feature_name", cps.Text("Name the flag"
                                     ,"QCFlag", doc = "Choose the measurement category to flag. "
                                     "The default name of the flag's measurement is 'QCFlag'."))
        group.append("combination_choice",
                     cps.Choice(
                "Flag if any, or all, measurement(s) fails to meet the criteria?",
                [ C_ANY, C_ALL], doc = '''
                <ul>
                <li><i>Any:</i> An image will be assigned a flag if any of its measurements fail. This can be useful
                for flagging images possessing varied QC flaws; for example, you can flag all bright images and all out of focus images with one flag.</li>
                <li><i>All:</i> A flag will only be assigned if all measurements fail.  This can be useful for flagging  images that possess only a combination
                of QC flaws; for example, you can flag only images that are both bright and out of focus.</li>
                </ul>'''))
        group.append("add_measurement_button", 
                     cps.DoSomething("",
                                     "Add another measurement",
                                     self.add_measurement, group))
        self.add_measurement(group, False)
        if can_delete:
            group.append("remover", cps.RemoveSettingButton("", "Remove this flag", self.flags, group))
        self.flags.append(group)

    def add_measurement(self, flag_settings, can_delete=True):
        measurement_settings = flag_settings.measurement_settings

        group = cps.SettingsGroup()
        group.append("source_choice",
                     cps.Choice(
                "Flag is based on", S_ALL, doc = '''
                <ul>
                <li><i>Image:</i> This will flag an image based
                on a per-image measurement, such as intensity or granularity.</li>
                <li><i>Average for objects:</i> This will flag
                an image based on the average of all object measurements in an image.</li>
                <li><i>All objects:</i> This will flag an image based on all the 
                object measurements in an image, without averaging.</li>
                </ul>'''))
        group.append("object_name",
                     cps.ObjectNameSubscriber(
                "Select the object whose measurements will be used to flag",
                "None", doc = '''What did you call the objects whose measurements you want to use for flagging?'''))

        def object_fn():
            if group.source_choice == S_IMAGE:
                return cpmeas.IMAGE
            return group.object_name.value

        group.append("measurement", cps.Measurement("Which measurement?",
                                                    object_fn))
        group.append("wants_minimum",
                     cps.Binary("Flag images based on low values?",
                                True, doc = '''Images with measurements below this cutoff will be flagged.'''))
        group.append("minimum_value", cps.Float("Minimum value", 0))
        group.append("wants_maximum",
                     cps.Binary("Flag images based on high values?",
                                True, doc = '''Images with measurements above this cutoff will be flagged.'''))
        group.append("maximum_value", cps.Float("Maximum value", 1))
        
        if can_delete:
            group.append("remover", cps.RemoveSettingButton("", "Remove this measurement", measurement_settings, group))

        measurement_settings.append(group)

    
    def settings(self):
        result = [self.flag_count]
        for flag in self.flags:
            result += [flag.measurement_count, flag.category, flag.feature_name, 
                       flag.combination_choice]
            for mg in flag.measurement_settings:
                result += [mg.source_choice, mg.object_name, mg.measurement,
                           mg.wants_minimum, mg.minimum_value,
                           mg.wants_maximum, mg.maximum_value]
        return result
    
    def prepare_settings(self, setting_values):
        '''Construct the correct number of flags'''
        flag_count = int(setting_values[0])
        del self.flags[:]
        self.add_flag(can_delete=False)
        while len(self.flags) < flag_count:
            self.add_flag()
            
        setting_values = setting_values[N_FIXED_SETTINGS:]
        for flag in self.flags:
            count = int(setting_values[0])
            # Adding a flag adds the first measurement automatically
            while len(flag.measurement_settings) < count:
                self.add_measurement(flag, can_delete=True)
            setting_values = setting_values[N_FIXED_SETTINGS_PER_FLAG +
                                            count * N_SETTINGS_PER_MEASUREMENT:]
    
    def visible_settings(self):
        def measurement_visibles(m_g):
            result = [m_g.source_choice]
            if m_g.source_choice != S_IMAGE:
                result += [m_g.object_name]
            result += [m_g.measurement, m_g.wants_minimum]
            if m_g.wants_minimum.value:
                result += [m_g.minimum_value]
            result += [m_g.wants_maximum]
            if m_g.wants_maximum.value:
                result += [m_g.maximum_value]
            if hasattr(m_g, "remover"):
                result += [m_g.remover]
            return result

        def flag_visibles(flag):
            result = [flag.category, flag.feature_name]
            if len(flag.measurement_settings) > 1:
                result += [flag.combination_choice]
            for measurement_settings in flag.measurement_settings:
                result += measurement_visibles(measurement_settings)
            result += [flag.add_measurement_button]
            if hasattr(flag, "remover"):
                result += [flag.remover]
            return result

        result = []
        for flag in self.flags:
            result += flag_visibles(flag)

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

    def measurement_name(self, flag):
        return "_".join((flag.category.value, flag.feature_name.value))

    def run_flag(self, workspace, flag):
        ok, stats = self.eval_measurement(workspace, 
                                          flag.measurement_settings[0])
        statistics = [tuple([self.measurement_name(flag)] + list(stats))]
        for measurement_setting in flag.measurement_settings[1:]:
            ok_1, stats = self.eval_measurement(workspace, measurement_setting)
            statistics += [tuple([self.measurement_name(flag)] + list(stats))]
            if flag.combination_choice == C_ALL:
                ok = ok or ok_1
            elif flag.combination_choice == C_ANY:
                ok = ok and ok_1
            else:
                raise NotImplementedError("Unimplemented combination choice: %s" %
                                          flag.combination_choice.value)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_image_measurement(self.measurement_name(flag), 0 if ok else 1)
        return statistics
        
    def eval_measurement(self, workspace, ms):
        '''Evaluate a measurement
        
        workspace - holds the measurements to be evaluated
        ms - the measurement settings indicating how to evaluate
        
        returns a tuple
           first tuple element is True = pass, False = Fail
           second tuple element has all of the statistics except for the
                        flag name
        '''
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        fail = False
        if ms.source_choice == S_IMAGE:
            value = m.get_current_image_measurement(ms.measurement.value)
            min_value = max_value = value
            display_value = str(round(value,3))
            source = cpmeas.IMAGE
        elif ms.source_choice == S_AVERAGE_OBJECT:
            data = m.get_current_measurement(ms.object_name.value,
                                             ms.measurement.value)
            if len(data) == 0:
                min_value = max_value = np.NaN
                fail = True
                display_value = "No objects"
            else:
                min_value = max_value = np.mean(data)
                display_value = str(round(min_value, 3))
            source = "Ave. %s"%ms.object_name.value
        elif ms.source_choice == S_ALL_OBJECTS:
            data = m.get_current_measurement(ms.object_name.value,
                                             ms.measurement.value)
            source = ms.object_name.value
            if len(data) == 0:
                min_value = max_value = np.NaN
                fail = True
                display_value = "No objects"
            else:
                min_value = np.min(data)
                max_value = np.max(data)
                if min_value == max_value:
                    display_value = str(min_value)
                else:
                    display_value = "%.3f - %.3f"%(min_value,max_value)
        else:
            raise NotImplementedError("Source choice of %s not implemented" %
                                      ms.source_choice)
        fail = (fail or
                (ms.wants_minimum.value and 
                 min_value < ms.minimum_value.value) or
                (ms.wants_maximum.value and
                 max_value > ms.maximum_value.value))
        return ((not fail), (source, ms.measurement.value, display_value, 
                             "Fail" if fail else "Pass"))
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for each flag mesurment in the module'''
        return [(cpmeas.IMAGE, self.measurement_name(flag), cpmeas.COLTYPE_INTEGER)
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
        if from_matlab and (variable_revision_number == 1 or variable_revision_number == 2):
            
            if variable_revision_number == 1:
                image_name, category, feature_num_or_name, min_value, max_value, \
                      new_or_append, new_name, old_name = setting_values
            elif variable_revision_number == 2:
                image_name, category, feature_num_or_name, scale, min_value, max_value, \
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
            
        return setting_values, variable_revision_number, from_matlab
    
