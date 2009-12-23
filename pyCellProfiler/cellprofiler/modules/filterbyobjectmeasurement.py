'''<b>Filter By Object Measurement</b> eliminates objects based on their measurements (e.g. area, shape,
texture, intensity).
<hr>
This module removes objects based on their measurements produced by another module (e.g. 
MeasureObjectAreaShape, MeasureObjectIntensity, MeasureTexture, etc). All objects that do not satisty  
the specified parameters will be discarded.

See also: Any of the <b>MeasureObject*</b> modules, <b>MeasureTexture</b>,
<b>MeasureCorrelation</b> and <b>CalculateRatios</b>.
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

__version__ = "$Revision$"

import numpy as np
import os
import scipy.ndimage as scind
import traceback

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.utilities.rules as cprules
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns

'''Minimal filter - pick a single object per image by minimum measured value'''
FI_MINIMAL = "Minimal"

'''Maximal filter - pick a single object per image by maximum measured value'''
FI_MAXIMAL = "Maximal"

'''Pick one object per containing object by minimum measured value'''
FI_MINIMAL_PER_OBJECT = "Minimal per object"

'''Pick one object per containing object by maximum measured value'''
FI_MAXIMAL_PER_OBJECT = "Maximal per object"

'''Keep all objects whose values fall between set limits'''
FI_LIMITS = "Limits"

FI_ALL = [ FI_MINIMAL, FI_MAXIMAL, FI_MINIMAL_PER_OBJECT,
          FI_MAXIMAL_PER_OBJECT, FI_LIMITS ]

'''The number of settings for this module in the pipeline if no additional objects'''
FIXED_SETTING_COUNT = 14

'''The number of settings per additional object'''
ADDITIONAL_OBJECT_SETTING_COUNT = 4

FF_PARENT = "Parent_%s"

ROM_RULES = "Rules"
ROM_MEASUREMENTS = "Measurements"

DIR_DEFAULT_INPUT = "Default input folder"
DIR_DEFAULT_OUTPUT = "Default output folder"
DIR_CUSTOM = "Custom folder"

class FilterByObjectMeasurement(cpm.CPModule):

    module_name = 'FilterByObjectMeasurement'
    category = "Object Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        '''Create the initial settings and name the module'''
        self.target_name = cps.ObjectNameProvider('Name the output objects','FilteredBlue',doc = """
                                What do you want to call the filtered objects? This will be the name for the collection of objects that meet the filter
                                criteria.""")
        
        self.object_name = cps.ObjectNameSubscriber('Select the object to filter by','None', doc = """
                                What object would you like to filter by, or if using a Ratio, what is the numerator object?
                                This setting controls which objects will be filtered to generate the
                                filtered objects. It also controls the measurement choices for filtering:
                                you can only filter on measurements made on these objects. The values
                                for ratio measurements are assigned to the numerator object, so you have
                                to select the numerator object to access a ratio measurement.""")
        
        self.spacer_1 = cps.Divider(line=False)
        
        self.rules_or_measurement = cps.Choice(
            'Do you want to filter using classifier rules or measurements?',
            [ROM_MEASUREMENTS, ROM_RULES],
            doc = """You can either pick a measurement made on the objects or
            a rules file as produced by CellProfiler Analyst. If you choose
            "Rules", you will have to ensure that this pipeline makes every
            measurement in that rules file.""")
        self.measurement = cps.Measurement('Select the measurement to filter by', 
                                self.object_name.get_value, "AreaShape_Area", doc = """
                                See the help of the Measurements modules
                                for more information on the features measured.""")
        
        self.spacer_2 = cps.Divider(line=False)
        
        self.filter_choice = cps.Choice("Select the filtering method", FI_ALL, FI_LIMITS, doc = """
                                There are five different ways to filter objects:
                                <ul>
                                <li><i>Maximal:</i> Keep the object with the maximum value for the measurement
                                of interest. If multiple objects share a maximal value, retain one object 
                                selected arbirtraily per image.</li>
                                <li><i>Minimal:</i> Keep the object with the minimum value for the measurement
                                of interest. If multiple objects share a minimal value, retain one object 
                                selected arbirtraily per image.</li>
                                <li><i>Maximal per object:</i> This option requires a choice of a set of container
                                objects. The container objects might contain several objects of
                                choice (for instance, mitotic spindles within a cell or FISH
                                probe spots within a nucleus). This option will keep only the
                                object with the maximum value for the measurement among the
                                set of objects within the container objects.</li>
                                <li><i>Minimal per object:</i> Same as Maximal per object, except use minimum to filter.</li>
                                <li><i>Limits:</i> Keep an object if its measurement value falls between a minimal
                                and maximal limit.</li>
                                </ul>""")
        
        self.wants_minimum = cps.Binary('Filter using a minimum measurement value?', True, doc = """
                                <i>(Used if Limits is selected for filtering method)</i><br>
                                Check this box to filter the objects based on a minimum acceptable object
                                measurement value. Objects which are greater than or equal to this value
                                will be retained.""")
        
        self.min_limit = cps.Float('Minimum value',0)
        
        self.wants_maximum = cps.Binary('Filter using a maximum measurement value?', True, doc = """
                                <i>(Used if Limits is selected for filtering method)</i><br>
                                Check this box to filter the objects based on a maxmium acceptable object
                                measurement value. Objects which are less than or equal to this value
                                will be retained.""")
        
        self.max_limit = cps.Float('Maximum value',1)
        
        self.enclosing_object_name = cps.ObjectNameSubscriber('What did you call the objects that contain the filtered objects?','None', doc = """
                                <i>(Used if a Per-Object filtering method is selected)</i><br>
                                This setting selects the container (i.e, parent) objects for the <i>Maximal per object</i> 
                                and <i>Minimal per object</i> filtering choices.""")
        
        self.rules_file_name = cps.Text(
            "Rules file name:","rules.txt",
            doc="""The filename of the file holding the rules. Each line of
            this file should be a rule, naming a measurement to be made
            on the object you selected. For instance, a rule might be:
            <br><tt>
            IF (Nuclei_AreaShape_Area < 351.3, [0.79, -0.79], [-0.94, 0.94])
            </tt><br>
            The above rule will score +.79 for the positive category and -0.94
            for the negative category for nuclei whose area is less than 351.3 
            pixels and will score the opposite for nuclei whose area is larger.
            The filter adds positive and negative and keeps only objects whose
            positive score is higher than the negative score""")
        self.rules_directory_choice = cps.Choice(
            "Where is the rules file?",
            [DIR_DEFAULT_INPUT, DIR_DEFAULT_OUTPUT, DIR_CUSTOM],
            doc = """The location of the rules file. Choose "Default input
            folder" if the rules file is in the default input folder, 
            "Default output folder" if the rules file is in the default output
            folder or "Custom folder" if you want to enter a folder name other
            than the default input or output folder.""")
        self.rules_directory = cps.Text(
            "Rules folder name:",".",
            doc="""Enter the path to the folder containing the rules file. You
            can use "." for a path name that's relative to the default input
            directory and "&amp;" for a path that's relative to the default 
            output directory.""")
        
        self.wants_outlines = cps.Binary('Save outlines of filtered objects?', False)
        
        self.outlines_name = cps.ImageNameProvider('Name the outline image','FilteredBlue')
        
        self.additional_objects = []
        self.spacer_3 = cps.Divider(line=False)
        
        self.additional_object_button = cps.DoSomething('Add an object to be relabeled like the filtered object',
                                'Add', self.add_additional_object, doc = """
                                Click this button to add an object to receive the same post-filtering labels as
                                the filtered object. This is useful in making sure that labeling is maintained 
                                between related objects (e.g., primary and secondary objects) after filtering.""")
    
    def add_additional_object(self):
        group = cps.SettingsGroup()
        group.append("object_name",
                     cps.ObjectNameSubscriber('Select additional object to relabel',
                                              'None'))
        group.append("target_name",
                     cps.ObjectNameProvider('Name the relabeled objects','FilteredGreen'))
        
        group.append("wants_outlines",
                     cps.Binary('Save outlines of relabeled objects?', False))
        
        group.append("outlines_name",
                     cps.ImageNameProvider('Name the outline image','OutlinesFilteredGreen'))
        
        group.append("remover", cps.RemoveSettingButton("", "Remove above object", self.additional_objects, group))
        group.append("divider", cps.Divider(line=False))
        self.additional_objects.append(group)

    def prepare_settings(self, setting_values):
        '''Make sure the # of slots for additional objects matches 
           the anticipated number of additional objects'''
        setting_count = len(setting_values)
        assert ((setting_count - FIXED_SETTING_COUNT) % 
                ADDITIONAL_OBJECT_SETTING_COUNT) == 0
        additional_object_count = ((setting_count - FIXED_SETTING_COUNT) /
                                   ADDITIONAL_OBJECT_SETTING_COUNT)
        while len(self.additional_objects) > additional_object_count:
            self.remove_additional_object(self.additional_objects[-1].key)
        while len(self.additional_objects) < additional_object_count:
            self.add_additional_object()

    def settings(self):
        result =[self.target_name, self.object_name, self.measurement,
                 self.filter_choice, self.enclosing_object_name,
                 self.wants_minimum, self.min_limit,
                 self.wants_maximum, self.max_limit,
                  self.wants_outlines, self.outlines_name,
                  self.rules_or_measurement, self.rules_directory_choice,
                  self.rules_directory]
        for x in self.additional_objects:
            result += [x.object_name, x.target_name, x.wants_outlines, x.outlines_name]
        return result

    def visible_settings(self):
        result =[self.target_name, self.object_name, 
                 self.spacer_2, self.rules_or_measurement]
        if self.rules_or_measurement == ROM_RULES:
            result += [self.rules_file_name, self.rules_directory_choice]
            if self.rules_directory_choice == DIR_CUSTOM:
                result += [self.rules_directory]
        else:
            result += [self.spacer_1, self.measurement, 
                       self.filter_choice]
            if self.filter_choice.value in (FI_MINIMAL_PER_OBJECT, 
                                            FI_MAXIMAL_PER_OBJECT):
                result.append(self.enclosing_object_name)
            elif self.filter_choice == FI_LIMITS:
                result.append(self.wants_minimum)
                if self.wants_minimum.value:
                    result.append(self.min_limit)
                result.append(self.wants_maximum)
                if self.wants_maximum.value:
                    result.append(self.max_limit)
        result.append(self.wants_outlines)
        if self.wants_outlines.value:
            result.append(self.outlines_name)
        result.append(self.spacer_3)
        for x in self.additional_objects:
            temp = x.unpack_group()
            if not x.wants_outlines.value:
                del temp[temp.index(x.wants_outlines) + 1]
            result += temp
        result += [self.additional_object_button]
        return result

    def validate_module(self, pipeline):
        '''Make sure that the user has selected some limits when filtering'''
        if (self.rules_or_measurement == ROM_MEASUREMENTS and
            self.filter_choice == FI_LIMITS and
            self.wants_minimum.value == False and
            self.wants_maximum.value == False):
            raise cps.ValidationError('Please enter a minimum and/or maximum limit for your measurement',
                                      self.wants_minimum)
        if self.rules_or_measurement == ROM_RULES:
            try:
                self.get_rules()
            except Exception, instance:
                traceback.print_exc()
                raise cps.ValidationError(str(instance),
                                          self.rules_file_name)

    def run(self, workspace):
        '''Filter objects for this image set, display results'''
        src_objects = workspace.get_objects(self.object_name.value)
        if self.rules_or_measurement == ROM_RULES:
            indexes = self.keep_by_rules(workspace, src_objects)
        elif self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
            indexes = self.keep_one(workspace, src_objects)
        elif self.filter_choice in (FI_MINIMAL_PER_OBJECT, 
                                    FI_MAXIMAL_PER_OBJECT):
            indexes = self.keep_per_object(workspace, src_objects)
        elif self.filter_choice == FI_LIMITS:
            indexes = self.keep_within_limits(workspace, src_objects)
        else:
            raise ValueError("Unknown filter choice: %s"%
                             self.filter_choice.value)
        
        #
        # Create an array that maps label indexes to their new values
        # All labels to be deleted have a value in this array of zero
        #
        new_object_count = len(indexes)
        max_label = np.max(src_objects.segmented)
        label_indexes = np.zeros((max_label+1,),int)
        label_indexes[indexes] = np.arange(1,new_object_count+1)
        #
        # Loop over both the primary and additional objects
        #
        object_list = ([(self.object_name.value, self.target_name.value,
                         self.wants_outlines.value, self.outlines_name.value)] + 
                       [(x.object_name.value, x.target_name.value, 
                         x.wants_outlines.value, x.outlines_name.value)
                         for x in self.additional_objects])
        m = workspace.measurements
        for src_name, target_name, wants_outlines, outlines_name in object_list:
            src_objects = workspace.get_objects(src_name)
            target_labels = src_objects.segmented.copy()
            #
            # Reindex the labels of the old source image
            #
            target_labels[target_labels > max_label] = 0
            target_labels = label_indexes[target_labels]
            #
            # Make a new set of objects - retain the old set's unedited
            # segmentation for the new and generally try to copy stuff
            # from the old to the new.
            #
            target_objects = cpo.Objects()
            target_objects.segmented = target_labels
            target_objects.unedited_segmented = src_objects.unedited_segmented
            if src_objects.has_parent_image:
                target_objects.parent_image = src_objects.parent_image
            workspace.object_set.add_objects(target_objects, target_name)
            #
            # Add measurements for the new objects
            add_object_count_measurements(m, target_name, new_object_count)
            add_object_location_measurements(m, target_name, target_labels)
            #
            # Relate the old numbering to the new numbering
            #
            m.add_measurement(target_name,
                              FF_PARENT%(src_name),
                              np.array(indexes))
            #
            # Add an outline if asked to do so
            #
            if wants_outlines:
                outline_image = cpi.Image(outline(target_labels) > 0,
                                          parent_image = target_objects.parent_image)
                workspace.image_set.add(outlines_name, outline_image)

    def is_interactive(self):
        return False

    def display(self, workspace):
        '''Display what was filtered'''
        src_name = self.object_name.value
        src_objects = workspace.get_objects(src_name)
        target_name = self.target_name.value
        target_objects = workspace.get_objects(target_name)
        image = None
        image_name = self.measurement.get_image_name(workspace.pipeline)
        if image_name is None:
            # Measurement isn't image-based
            if src_objects.has_parent_image:
                image = src_objects.parent_image
        else:
            image = workspace.image_set.get_image(image_name)
        if image is None:
            # Oh so sad - no image, just display the old and new labels
            figure = workspace.create_or_find_figure(subplots=(2,1))
            figure.subplot_imshow_labels(0,0,src_objects.segmented,
                                         title="Original: %s"%src_name)
            figure.subplot_imshow_labels(0,1,target_objects.segmented,
                                         title="Filtered: %s"%
                                         target_name)
        else:
            figure = workspace.create_or_find_figure(subplots=(2,2))
            figure.subplot_imshow_labels(0,0,src_objects.segmented,
                                         title="Original: %s"%src_name)
            figure.subplot_imshow_labels(0,1,target_objects.segmented,
                                         title="Filtered: %s"%
                                         target_name)
            figure.subplot_imshow_grayscale(1,0,image.pixel_data,
                                            "Input image #%d"%
                                            (workspace.measurements.image_set_number+1))
            outs = outline(target_objects.segmented) > 0
            pixel_data = image.pixel_data
            maxpix = np.max(pixel_data)
            if maxpix == 0:
                maxpix = 1.0
            if len(pixel_data.shape) == 3:
                picture = pixel_data.copy()
            else:
                picture = np.dstack((pixel_data,pixel_data,pixel_data))
            red_channel = picture[:,:,0]
            red_channel[outs] = maxpix
            figure.subplot_imshow_color(1,1,picture,
                                        "Outlines")
    
    def keep_one(self, workspace, src_objects):
        '''Return an array containing the single object to keep
        
        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        measurement = self.measurement.value
        src_name = self.object_name.value
        values = workspace.measurements.get_current_measurement(src_name,
                                                                measurement)
        if len(values) == 0:
            return np.array([], int)
        best_idx = (np.argmax(values) if self.filter_choice == FI_MAXIMAL
                    else np.argmin(values)) + 1
        return np.array([best_idx], int)

    def keep_per_object(self, workspace, src_objects):
        '''Return an array containing the best object per enclosing object
        
        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        measurement = self.measurement.value
        src_name = self.object_name.value
        enclosing_name = self.enclosing_object_name.value
        src_objects = workspace.get_objects(src_name)
        enclosing_labels = workspace.get_objects(enclosing_name).segmented
        enclosing_max = np.max(enclosing_labels)
        if enclosing_max == 0:
            return np.array([],int)
        enclosing_range = np.arange(1, enclosing_max+1)
        #
        # Make a vector of the value of the measurement per label index.
        # We can then label each pixel in the image with the measurement
        # value for the object at that pixel.
        # For unlabeled pixels, put the minimum value if looking for the
        # maximum value and vice-versa
        #
        values = workspace.measurements.get_current_measurement(src_name,
                                                                measurement)
        tricky_values = np.zeros((len(values)+1,))
        tricky_values[1:]=values
        wants_max = self.filter_choice == FI_MAXIMAL_PER_OBJECT
        if wants_max:
            tricky_values[0] = -np.Inf
        else:
            tricky_values[0] = np.Inf
        src_labels = src_objects.segmented
        src_values = tricky_values[src_labels]
        #
        # Now find the location of the best for each of the enclosing objects
        #
        fn = scind.maximum_position if wants_max else scind.minimum_position
        best_pos = fn(src_values, enclosing_labels, enclosing_range)
        best_pos = np.array((best_pos,) if isinstance(best_pos, tuple)
                            else best_pos)
        best_pos = best_pos.astype(np.uint32)
        #
        # Get the label of the pixel at each location
        #
        indexes = src_labels[best_pos[:,0], best_pos[:,1]]
        indexes = set(indexes)
        indexes = list(indexes)
        indexes.sort()
        return indexes[1:] if len(indexes)>0 and indexes[0] == 0 else indexes
    
    def keep_within_limits(self, workspace, src_objects):
        '''Return an array containing the single object to keep
        
        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        measurement = self.measurement.value
        src_name = self.object_name.value
        values = workspace.measurements.get_current_measurement(src_name,
                                                                measurement)
        low_limit = self.min_limit.value
        high_limit = self.max_limit.value
        if self.wants_minimum.value:
            if self.wants_maximum.value:
                hits = np.logical_and(values >= low_limit,
                                      values <= high_limit)
            else:
                hits = values >= low_limit
        elif self.wants_maximum.value:
            hits = values <= high_limit
        else:
            hits = np.ones(values.shape,bool)
        indexes = np.argwhere(hits)[:,0] 
        indexes = indexes + 1
        return indexes

    def get_rules(self):
        '''Read the rules from a file'''
        rules_file = self.rules_file_name.value
        if self.rules_directory_choice == DIR_DEFAULT_INPUT:
            rules_directory = cpprefs.get_default_image_directory()
        elif self.rules_directory_choice == DIR_DEFAULT_OUTPUT:
            rules_directory = cpprefs.get_default_output_directory()
        elif self.rules_directory_choice == DIR_CUSTOM:
            rules_directory = self.rules_directory.value
            rules_directory = cpprefs.get_absolute_path(rules_directory)
        else:
            raise NotImplementedError("Unknown directory choice: %s"%
                                      self.rules_directory_choice.value)
        path = os.path.join(rules_directory, rules_file)
        rules = cprules.Rules()
        rules.parse(path)
        return rules
        
    def keep_by_rules(self, workspace, src_objects):
        '''Keep objects according to rules
        
        workspace - workspace holding the measurements for the rules
        src_objects - filter these objects (uses measurement indexes instead)
        
        Open the rules file indicated by the settings and score the
        objects by the rules. Return the indexes of the objects that pass.
        '''
        rules = self.get_rules()
        scores = rules.score(workspace.measurements)
        #
        # NaN positive scores get - infinity. NaN negative scores get
        # infinity. This means all NaN cells get rejected.
        #
        scores[np.isnan(scores[:,0]),0] = -np.Infinity
        scores[np.isnan(scores[:,1]),1] = np.Infinity
        hits = scores[:,0] > scores[:,1]
        indexes = np.argwhere(hits)[:,0] + 1
        return indexes
    
    def get_measurement_columns(self, pipeline):
        '''Return measurement column defs for the parent/child measurement'''
        object_list = ([(self.object_name.value, self.target_name.value)] + 
                       [(x.object_name.value, x.target_name.value)
                         for x in self.additional_objects])
        columns = []
        for src_name, target_name in object_list:
            columns.append((target_name, 
                            FF_PARENT%src_name, 
                            cpmeas.COLTYPE_INTEGER))
            columns += get_object_measurement_columns(target_name)
        return columns
    
    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        categories = []
        if object_name == cpmeas.IMAGE:
            categories += ["Count"]
        elif object_name == self.object_name:
            categories.append("Children")
        if object_name == self.target_name.value:
            categories += ("Parent", "Location","Number")
        return categories
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []
        
        if object_name == cpmeas.IMAGE:
            if category == "Count":
                result += [self.target_name.value]
        if object_name == self.object_name and category == "Children":
            result += ["%s_Count" % self.target_name.value]
        if object_name == self.target_name:
            if category == "Location":
                result += [ "Center_X","Center_Y"]
            elif category == "Parent":
                result += [ self.object_name.value]
            elif category == "Number":
                result += ["Object_Number"]
        return result
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        '''Account for old save formats
        
        setting_values - the strings for the settings as saved in the pipeline
        variable_revision_number - the variable revision number at the time
                                   of saving
        module_name - this is either FilterByObjectMeasurement for pyCP
                      and Matlab's FilterByObjectMeasurement module or
                      it is KeepLargestObject for Matlab's module of that
                      name.
        from_matlab - true if file was saved by Matlab CP
        '''
        if (module_name == 'KeepLargestObject' and from_matlab
            and variable_revision_number == 1):
            #
            # This is a specialized case:
            # The filtering method is FI_MAXIMAL_PER_OBJECT to pick out
            # the largest. The measurement is AreaShape_Area.
            # The slots are as follows:
            # 0 - the source objects name
            # 1 - the enclosing objects name
            # 2 - the target objects name 
            setting_values = [ setting_values[1],
                              setting_values[2],
                              "AreaShape_Area",
                              FI_MAXIMAL_PER_OBJECT,
                              setting_values[0],
                              cps.YES, "0", cps.YES, "1",
                              cps.NO, "None" ]
            from_matlab = False
            variable_revision_number = 1
            module_name = self.module_name
        if (module_name == 'FilterByObjectMeasurement' and from_matlab and
            variable_revision_number == 6):
            # The measurement may not be correct here - it will display
            # as an error, though
            measurement = '_'.join((setting_values[2],setting_values[3]))
            if setting_values[6] == 'No minimum':
                wants_minimum = cps.NO
                min_limit = "0"
            else:
                wants_minimum = cps.YES
                min_limit = setting_values[6]
            if setting_values[7] == 'No maximum':
                wants_maximum = cps.NO
                max_limit = "1"
            else:
                wants_maximum = cps.YES
                max_limit = setting_values[7]
            if setting_values[8] == cps.DO_NOT_USE:
                wants_outlines = cps.NO
                outlines_name = "None"
            else:
                wants_outlines = cps.YES
                outlines_name = setting_values[8]
                
            setting_values = [setting_values[0], setting_values[1],
                              measurement, FI_LIMITS, "None", 
                              wants_minimum, min_limit,
                              wants_maximum, max_limit,
                              wants_outlines, outlines_name]
            from_matlab = False
            variable_revision_number = 1 
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added CPA rules
            #
            setting_values = (setting_values[:11] + 
                              [ROM_MEASUREMENTS, DIR_DEFAULT_INPUT, "."] +
                              setting_values[11:])
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

