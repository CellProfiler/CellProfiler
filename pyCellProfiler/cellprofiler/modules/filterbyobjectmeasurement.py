'''filterbyobjectmeasurement.py - Filter objects by measurements

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version = "$Revision$"

import numpy as np
import scipy.ndimage as scind
import uuid

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
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
FIXED_SETTING_COUNT = 11

'''The number of settings per additional object'''
ADDITIONAL_OBJECT_SETTING_COUNT = 4

FF_PARENT = "Parent_%s"

class FilterByObjectMeasurement(cpm.CPModule):
    '''SHORT DESCRIPTION:
Eliminates objects based on their measurements (e.g. area, shape,
texture, intensity).
*************************************************************************

This module removes objects based on their measurements produced by
another module (e.g. MeasureObjectAreaShape, MeasureObjectIntensity,
MeasureTexture). All objects outside of the specified parameters will be
discarded.

Settings:
What do you want to call the filtered objects?
This will be the name for the collection of objects that meet the filter
criteria.

Which object do you want to filter by, or if using a Ratio, what is the
numerator object?
This setting controls which objects will be filtered to generate the
filtered objects. It also controls the measurement choices for filtering:
you can only filter on measurements made on these objects. The values
for ratio measurements are assigned to the numerator object, so you have
to select the numerator object to access a ratio measurement.

What category of measurement do you want to use?
This choice box contains a list of available categories of measurements
for your chosen object. Examples are, "AreaShape" for measurements made
by the MeasureObjectAreaShape module or "Intensity" for measurements made
by the MeasureObjectIntensity module. See the help for your measurement of 
choice to find the appropriate category.

What feature do you want to use?
Generally, the feature is the kind of measurement taken or algorithm used
to calculate the measurement. For instance, "MeanIntensity", is the feature
name used when finding the average intensity within an object.

What image do you want to use?
Some features are calculated on the pixels in an image, for instance
"MeanIntensity" might be calculated on both a nuclear and cytoplasm stain
image. This setting allows you to pick the image for the measurement in
cases where there might be ambiguity.

What scale do you want to use?
Some features may be calculated at multiple scales (for instance, texture).
This setting lets you choose one of the measured scales in cases where there
might be ambiguity.

How do you want to filter objects?
The choices are:
* Maximal - only keep the object with the maximum value for the measurement
            of interest. Keep one object per image with an arbitrary choice
            on ties.
* Minimal - only keep the object with the minimum value for the measurement
            of interest. Keep one object per image with an arbitrary choice
            on ties.
* Maximal per object - This option requires a choice of a set of container
            objects. The container objects might contain several objects of
            choice (for instance, mitotic spindles within a cell or FISH
            probe spots within a nucleus). This option will keep only the
            object with the maximum value for the measurement among the
            set of objects within the container objects.
* Minimal per object - same as Maximal per object, except use minimum to filter.
* Limits - keep an object if its measurement value falls between a minimum
           and maximum limit.

What are the objects that enclose the objects to be filtered?
This setting chooses the container objects for the "Maximal per object" and
"Minimal per object" filtering choices.

What additional object do you want to receive the same labels as the filtered
objects?
This setting lets you propagate the filtering of your objects to the objects
that your objects were derived from and to the objects that derived from your
object.

What do you want to rename the relabeled objects?
Enter the name to be given to the relabeled objects after filtering and
relabeling.

Remove above object:
Press this button to remove the object above from the list of ones to be
renumbered.

Add an object to be relabeled similarly to the filtered object:
Press this button to add another object to the list of ones to be relabeled. 

See also MeasureObjectAreaShape, MeasureObjectIntensity, MeasureTexture,
MeasureCorrelation, CalculateRatios, and MeasureObjectNeighbors modules.
    '''

    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the initial settings and name the module'''
        self.module_name = 'FilterByObjectMeasurement'
        self.target_name = cps.ObjectNameProvider('What do you want to call the filtered objects?','FilteredBlue')
        self.object_name = cps.ObjectNameSubscriber('What object would you like to filter by, or if using a Ratio, what is the numerator object?','None')
        self.measurement = cps.Measurement('What measurement do you want to use?',
                                           self.object_name.get_value,
                                           "AreaShape_Area")
        self.filter_choice = cps.Choice('How do you want to filter objects?',
                                        FI_ALL, FI_LIMITS)
        self.wants_minimum = cps.Binary('Do you want a minimum acceptable value for the measurement?', True)
        self.min_limit = cps.Float('Enter the minimum acceptable value for the measurement:',0)
        self.wants_maximum = cps.Binary('Do you want a maximum acceptable value for the measurement?', True)
        self.max_limit = cps.Float('Enter the maximum acceptable value for the measurement:',1)
        self.enclosing_object_name = cps.ObjectNameSubscriber('What did you call the objects that contain the filtered objects?',
                                                              'None')
        self.wants_outlines = cps.Binary('Do you want to save outlines for the filtered image?', False)
        self.outlines_name = cps.ImageNameProvider('What do you want to call the outline image?','FilteredBlue')
        self.additional_objects = []
        self.additional_object_button = cps.DoSomething('Add an object to be relabeled similarly to the filtered object:',
                                                        'Add',
                                                        self.add_additional_object)
    
    def add_additional_object(self):
        class AdditionalObject(object):
            '''An object related to the one being filtered that should be relabeled'''
            def __init__(self, remove_fn):
                self.__key = uuid.uuid4()
                self.__object_name = cps.ObjectNameSubscriber('What additional object do you want to receive the same labels as the filtered objects?',
                                                              'None')
                self.__target_name = cps.ObjectNameProvider('What do you want to rename the relabeled objects?','FilteredGreen')
                self.__wants_outlines = cps.Binary('Do you want to save outline images for the relabeled objects?', False)
                self.__outlines_name = cps.ImageNameProvider('What is the name for the outline image?','OutlinesFilteredGreen')
                self.__remove_button = cps.DoSomething("Remove above object:",
                                                       "Remove",
                                                       remove_fn,
                                                       self.__key)
            @property
            def key(self):
                '''The key that identifies this in the additional_objects list'''
                return self.__key
            
            @property
            def object_name(self):
                '''The name of the object to be relabeled'''
                return self.__object_name
            
            @property
            def target_name(self):
                '''The name of the object list generated by relabeling'''
                return self.__target_name
            
            @property
            def wants_outlines(self):
                '''True if the user wants to save outlines for the relabeled objects'''
                return self.__wants_outlines
            
            @property
            def outlines_name(self):
                '''The name of the outline image'''
                return self.__outlines_name
            
            def settings(self):
                '''These settings should be saved and loaded in the pipeline'''
                return [self.object_name, self.target_name, 
                        self.wants_outlines, self.outlines_name]
            
            def visible_settings(self):
                '''These settings should be displayed'''
                result = [self.object_name, self.target_name, 
                          self.wants_outlines]
                if self.wants_outlines.value:
                    result.append(self.outlines_name)
                return result
        additional_object = AdditionalObject(self.remove_additional_object)
        self.additional_objects.append(additional_object)
    
    def remove_additional_object(self, key):
        '''Remove the additional object with the given key from the list'''
        idx = [x.key for x in self.additional_objects].index(key)
        del self.additional_objects[idx]

    def prepare_to_set_values(self, setting_values):
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

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
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
                                  
        return setting_values, variable_revision_number, from_matlab

    def settings(self):
        result =[self.target_name, self.object_name, self.measurement,
                 self.filter_choice, self.enclosing_object_name,
                 self.wants_minimum, self.min_limit,
                 self.wants_maximum, self.max_limit,
                  self.wants_outlines, self.outlines_name]
        for x in self.additional_objects:
            result += x.settings()
        return result

    def visible_settings(self):
        result =[self.target_name, self.object_name, self.measurement,
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
        for x in self.additional_objects:
            result += x.visible_settings()
        result += [self.additional_object_button]
        return result

    def test_valid(self, pipeline):
        '''Make sure that the user has selected some limits when filtering'''
        if (self.filter_choice == FI_LIMITS and
            self.wants_minimum.value == False and
            self.wants_maximum.value == False):
            raise cps.ValidationError('Please enter a minimum and/or maximum limit for your measurement',
                                      self.wants_minimum)
        super(FilterByObjectMeasurement,self).test_valid(pipeline)

    def run(self, workspace):
        '''Filter objects for this image set, display results'''
        src_objects = workspace.get_objects(self.object_name.value)
        if self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
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
                outline_image = cpi.Image(outline(target_labels),
                                          parent_image = target_objects.parent_image)
                workspace.image_set.add(outlines_name, outline_image)

        if not workspace.frame is None:
            self.display(workspace)
    
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
            image = workspace.get_objects(image_name)
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
