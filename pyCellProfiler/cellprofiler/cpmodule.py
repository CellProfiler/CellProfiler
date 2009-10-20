"""Module.py - represents a CellProfiler pipeline module
    
    TO-DO: capture and save module revision #s in the handles

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import re
import os
import sys

import numpy as np

import cellprofiler.settings as cps
import cellprofiler.cpimage
import cellprofiler.objects
import cellprofiler.measurements
import pipeline as cpp

class CPModule(object):
    """ Derive from the abstract module class to create your own module in Python
    
    You need to implement the following in the derived class:
    create_settings - fill in the module_name and create the settings that
               configure the module.
    settings - return the settings that will be loaded or saved from/to the
               pipeline.
    visible_settings - return the settings that will be displayed on the UI
    upgrade_settings - adjusts settings while loading to account for
               old revisions.
    run - to run the module, producing measurements, etc.
    
    Implement these if you produce measurements:
    get_categories - The category of measurement produced, for instance AreaShape
    get_measurements - The measurements produced by a category
    get_measurement_images - The images measured for a particular measurement
    get_measurement_scales - the scale at which a measurement was taken
    get_measurement_columns - the measurements stored in the database
    
    The pipeline calls hooks in the module before and after runs and groups.
    If your module requires state across image_sets, think of storing that 
    state in the image_set_list's legacy_fields dictionary instead
    of the module. 
    
    The hooks are:
    prepare_run - before run: useful for setting up image sets
    prepare_group - before group: useful for initializing aggregation
    post_group - after group: useful for calculating final aggregation steps
                 and for writing out results.
    post_run - use this to perform operations on the results of the experiment,
               for instance on all measurements
    
    """
    
    def __init__(self):
        if self.__doc__ is None:
            self.__doc__ = sys.modules[self.__module__].__doc__
        self.__module_num = -1
        self.__settings = []
        self.__notes = []
        self.__variable_revision_number = 0
        self.__annotation_dict = None
        self.__show_frame = True
        self.batch_state = np.zeros((0,),np.uint8)
        # Set the name of the module based on the class name.  A
        # subclass can override this either by declaring a module_name
        # attribute in the class definition or by assigning to it in
        # the create_settings method.
        if 'module_name' not in self.__dict__:
            self.module_name = self.__class__.__name__
        self.create_settings()
        
    def create_settings(self):
        """Create your settings by subclassing this function
        
        create_settings is called at the end of initialization. You should
        name your module in this routine:
        
            # Set the name that will appear in the "AddModules" window
            self.module_name = "My module"
        
        You should also create the setting variables for your module:
            # Ask the user for the input image
            self.image_name = cellprofiler.settings.ImageNameSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = cellprofiler.settings.ImageNameProvider(...)
            # Ask the user for a parameter
            self.smoothing_size = cellprofiler.settings.Float(...)
        """
        pass
    
    def create_from_handles(self,handles,module_num):
        """Fill a module with the information stored in the handles structure for module # ModuleNum 
        
        Returns a module with the settings decanted from the handles.
        If the revision is old, a different and compatible module can be returned.
        """
        self.__module_num = module_num
        idx = module_num-1
        settings = handles[cpp.SETTINGS][0,0]
        setting_values = []
        if settings.dtype.fields.has_key(cpp.MODULE_NOTES):
            n=settings[cpp.MODULE_NOTES][0,idx]
            self.__notes = [str(n[i,0][0]) for i in range(0,n.size)]
        else:
            self.__notes = []
        if settings.dtype.fields.has_key(cpp.SHOW_FRAME):
            self.__show_frame = settings[cpp.SHOW_FRAME][0,idx] != 0
        if settings.dtype.fields.has_key(cpp.BATCH_STATE):
            self.batch_state = settings[cpp.BATCH_STATE][0,idx]
        setting_count=settings[cpp.NUMBERS_OF_VARIABLES][0,idx]
        variable_revision_number = settings[cpp.VARIABLE_REVISION_NUMBERS][0,idx]
        module_name = settings[cpp.MODULE_NAMES][0,idx][0]
        for i in range(0,setting_count):
            value_cell = settings[cpp.VARIABLE_VALUES][idx,i]
            if isinstance(value_cell,np.ndarray):
                if np.product(value_cell.shape) == 0:
                    setting_values.append('')
                else:
                    setting_values.append(str(value_cell[0]))
            else:
                setting_values.append(value_cell)
        self.set_setting_values(setting_values, variable_revision_number, 
                                 module_name)
    
    def prepare_to_set_values(self,setting_values):
        """Do any sort of adjustment to the settings required for the given values
        
        setting_values - the values for the settings just prior to mapping
                         as done by set_setting_values
        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        
        See cellprofiler.modules.measureobjectareashape for an example.
        """
        pass
    
    def set_setting_values(self, setting_values, variable_revision_number, 
                            module_name):
        """Set the settings in a module, given a list of values
        
        The default implementation gets all the settings and then
        sets their values using the string passed. A more modern
        module may want to tailor the particular settings set to
        whatever values are in the list or however many values
        are in the list.
        """
        setting_values, variable_revision_number, from_matlab =\
            self.upgrade_settings(setting_values,
                                  variable_revision_number,
                                  module_name,
                                  not '.' in module_name)
        # we can't handle matlab settings anymore
        assert not from_matlab, "Module %s's upgrade_settings returned from_matlab==True"%(module_name)
        self.prepare_to_set_values(setting_values)
        for v,value in zip(self.settings(),setting_values):
            v.value = value
        self.upgrade_module_from_revision(variable_revision_number)
    
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        return setting_values, variable_revision_number, from_matlab
    
    def create_from_annotations(self):
        """Create the settings based on what you can discern from the annotations
        """
        annotation_dict = {}
        value_dict = {}
        max_setting = 0
        
        for annotation in self.annotations():
            vn = annotation.setting_number
            if not annotation_dict.has_key(vn):
                annotation_dict[vn] = {}
            if not annotation_dict[vn].has_key(annotation.kind):
                annotation_dict[vn][annotation.kind] = []
            annotation_dict[vn][annotation.kind].append(annotation.value)
            if annotation.kind == 'default':
                value_dict[vn] = annotation.value
            elif annotation.kind == 'choice' and not value_dict.has_key(vn):
                value_dict[vn] = annotation.value
            if vn > max_setting:
                max_setting = vn
        
        settings = []
        for i in range(1,max_setting+1):
            assert annotation_dict.has_key(i), 'There are no annotations for setting # %d'%(i)
            setting_dict = annotation_dict[i]
            if setting_dict.has_key('text'):
                text = setting_dict['text'][0]
            elif setting_dict.has_key('pathnametext'):
                text = setting_dict['pathnametext'][0]
            elif setting_dict.has_key('filenametext'):
                text = setting_dict['filenametext'][0]
            else:
                text = ''
            if setting_dict.has_key('infotype'):
                default_value = value_dict.get(i,cps.DO_NOT_USE)
                parts = setting_dict['infotype'][0].split(' ')
                if parts[-1] == 'indep':
                    v = cps.NameProvider(text,parts[0], default_value)
                else:
                    v = cps.NameSubscriber(text,parts[0],default_value)
            elif setting_dict.has_key('inputtype') and \
                 setting_dict['inputtype'][0] == 'popupmenu':
                choices = setting_dict['choice']
                if len(choices) == 2 and all([choice in [cps.YES,cps.NO] for choice in choices]):
                    v = cps.Binary(text,choices[0]==cps.YES)
                else:
                    default_value = value_dict.get(i,choices[0])
                    v = cps.Choice(text,choices,default_value)
            elif setting_dict.has_key('inputtype') and \
                 setting_dict['inputtype'][0] == 'popupmenu custom':
                choices = setting_dict['choice']
                default_value = value_dict.get(i,choices[0])
                v = cps.CustomChoice(text,choices,default_value)
            elif setting_dict.has_key('inputtype') and \
                 setting_dict['inputtype'][0] == 'pathnametext':
                v = cps.PathnameText(text,value_dict.get(i,cps.DO_NOT_USE))
            elif setting_dict.has_key('inputtype') and \
                 setting_dict['inputtype'][0] == 'filenametext':
                v = cps.FilenameText(text,value_dict.get(i,cps.DO_NOT_USE))
            else:
                v = cps.Text(text,value_dict.get(i,"n/a"))
            settings.append(v)
                     
        self.set_settings(settings)
    
    def on_post_load(self, pipeline):
        """This is a convenient place to do things to your module after the 
           settings have been loaded or initialized"""
        pass

    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the settings in the module to upgrade it to its 
        current revision number.
        
        Most modules use BackwardsCompatibilize instead of this.
        """
        if variable_revision_number != self.variable_revision_number:
            raise NotImplementedError(
                "Please implement upgrade_module_from_revision or "
                "backwards_compatiblize to upgrade module %s from "
                "revision %d to revision %d"%(self.module_name, 
                                              variable_revision_number, 
                                              self.variable_revision_number))
    
    def get_help(self):
        """Return help text for the module
        
        The default help is taken from your modules docstring and from
        the settings.
        """
        doc = self.__doc__.replace("\r","").replace("\n\n","<p>")
        doc = doc.replace("\n"," ")
        result = "<html><body><h1>%s</h1>" % self.module_name + doc
        first_setting_doc = True
        seen_setting_docs = set()
        for setting in self.settings():
            if setting.doc is not None:
                key = (setting.text, setting.doc)
                if key not in seen_setting_docs:
                    seen_setting_docs.add(key)
                    if first_setting_doc:
                        result = result + "</div><div><h2>Settings:</h2>"
                        first_setting_doc = False
                    result = (result + "<h4>" + setting.text + "</h4><div>" +
                              setting.doc + "</div>")
        if not first_setting_doc:
            result += "</div>"
        result += "</body></html>"
        return result
            
    def save_to_handles(self,handles):
        module_idx = self.module_num-1
        setting = handles[cpp.SETTINGS][0,0]
        setting[cpp.MODULE_NAMES][0,module_idx] = unicode(self.module_class())
        setting[cpp.MODULE_NOTES][0,module_idx] = np.ndarray(shape=(len(self.notes()),1),dtype='object')
        for i in range(0,len(self.notes())):
            setting[cpp.MODULE_NOTES][0,module_idx][i,0]=self.notes()[i]
        setting[cpp.NUMBERS_OF_VARIABLES][0,module_idx] = len(self.settings())
        for i in range(0,len(self.settings())):
            variable = self.settings()[i]
            if len(str(variable)) > 0:
                setting[cpp.VARIABLE_VALUES][module_idx,i] = unicode(str(variable))
            try: # matlab & old-style through annotations
                annotations = self.setting_annotations(variable.key())
                if annotations.has_key('infotype'):
                    setting[cpp.VARIABLE_INFO_TYPES][module_idx,i] = unicode(annotations['infotype'][0].value)
            except:
                pass
            if isinstance(variable,cps.NameProvider):
                setting[cpp.VARIABLE_INFO_TYPES][module_idx,i] = unicode("%s indep"%(variable.group))
            elif isinstance(variable,cps.NameSubscriber):
                setting[cpp.VARIABLE_INFO_TYPES][module_idx,i] = unicode(variable.group)
        setting[cpp.VARIABLE_REVISION_NUMBERS][0,module_idx] = self.variable_revision_number
        setting[cpp.MODULE_REVISION_NUMBERS][0,module_idx] = 0
        setting[cpp.SHOW_FRAME][0,module_idx] = 1 if self.show_frame else 0
        setting[cpp.BATCH_STATE][0,module_idx] = self.batch_state
    
    def in_batch_mode(self):
        '''Return True if the module knows that the pipeline is in batch mode'''
        return None
    
    def check_for_prepare_run_setting(self, setting):
        '''Check to see if changing the given setting means you have to restart
        
        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to do "prepare_run".
        '''
        return False
    
    def turn_off_batch_mode(self):
        '''Reset the module to an editable state if batch mode is on
        
        A module is allowed to create hidden information that it uses
        to turn batch mode on or to save state to be used in batch mode.
        This call signals that the pipeline has been opened for editing,
        even if it is a batch pipeline; all modules should be restored
        to a state that's appropriate for creating a batch file, not
        for running a batch file
        '''
        pass
    
    def test_valid(self,pipeline):
        """Test to see if the module is in a valid state to run
        
        Throw a ValidationError exception with an explanation if a module is not valid.
        """
        for setting in self.visible_settings():
            setting.test_valid(pipeline)
        self.validate_module(pipeline)
    
    def validate_module(self,pipeline):
        pass
    
    def get_name_providers(self, group):
        '''Return a list of name providers supplied by the module for this group
        
        group - a group supported by a subclass of NameProvider
        
        This routine returns additional providers beyond those that
        are provided by the module's provider settings.
        '''
        return []
    
    def setting_annotations(self,key):
        """Return annotations for the setting with the given number
        
        """
        if not self.__annotation_dict:
            self.__annotation_dict = cps.get_annotations_as_dictionary(self.annotations())
        if self.__annotation_dict.has_key(key):
            return self.__annotation_dict[key]
        indexes = [index+1 for setting,index in zip(self.settings(),range(len(self.settings()))) if setting.key()==key]
        if len(indexes):
            alt_key = indexes[0]
            if self.__annotation_dict.has_key(alt_key):
                return self.__annotation_dict[alt_key]
        return {}
    
    def get_module_num(self):
        """Get the module's index number
        
        The module's index number or ModuleNum is a one-based index of its
        execution position in the pipeline. It can be used to predict what
        modules have been run (creating whatever images and measurements
        those modules create) previous to a given module.
        """
        if self.__module_num == -1:
            raise(Exception('Module has not been created'))
        return self.__module_num
    
    def set_module_num(self,module_num):
        """Change the module's one-based index number in the pipeline
        
        """
        self.__module_num = module_num
    
    module_num = property(get_module_num, set_module_num)
    
    def get_module_name(self):
        """The name shown to the user in the Add Modules box.
        Deprecated in favor of accessing the attribute directly."""
        if self.__module_name is None:
            return re.sub('([^A-Z])([A-Z])', '\\1 \\2', 
                          self.__class__.__name__)
        else:
            return self.__module_name
    
    def set_module_name(self, module_name):
        """Deprecated in favor of setting the attribute directly.  Can
        be removed once all modules have been updated."""
        self.module_name = module_name
    
    def module_class(self):
        """The class to instantiate, except for the special case of matlab modules.
        
        """
        return self.__module__+'.'+self.module_name
    
    def get_variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        raise NotImplementedError("Please implement SettingRevisionNumber in the derived class")
    
    def __internal_get_variable_revision_number(self):
        """The revision number for the setting format for this module"""
        return self.get_variable_revision_number()
    variable_revision_number = property(__internal_get_variable_revision_number)
    
    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline
        
        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return self.__settings

    def setting(self,setting_num):
        """Reference a setting by its one-based setting number
        """
        return self.settings()[setting_num-1]
    
    def set_settings(self,settings):
        self.__settings = settings
        
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        return self.settings()
    
    def get_show_frame(self):
        '''True if the user wants to see the figure for this module'''
        return self.__show_frame
    
    def set_show_frame(self, show_frame):
        self.__show_frame = show_frame
    
    show_frame = property(get_show_frame, set_show_frame)

    def annotations(self):
        """Return the setting annotations, as read out of the module file.
        
        Return the setting annotations, as read out of the module file.
        Each annotation is an instance of the cps.Annotation
        class.
        """
        raise NotImplementedError("Please implement Annotations in your derived class")
    
    def delete(self):
        """Delete the module, notifying listeners that it's going away
        
        """
    
    def notes(self):
        """The user-entered notes for a module
        """
        return self.__notes
    
    def set_notes(self,Notes):
        """Give the module new user-entered notes
        
        """
        return self.__notes
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
        pass
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """
        pass
    
    def prepare_run(self, pipeline, image_set_list, frame):
        """Prepare the image set list for a run (& whatever else you want to do)
        
        pipeline - the pipeline being run
        image_set_list - add any image sets to the image set list
        frame - parent frame of application if GUI enabled, None if GUI
                disabled
        
        return True if operation completed, False if aborted 
        """
        return True
    
    def run(self,workspace):
        """Run the module (abstract method)
        
        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        pass
    
    def post_run(self, workspace):
        """Do post-processing after the run completes
        
        workspace - the workspace at the end of the run
        """
        pass
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''Prepare to create a batch file
        
        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.
        
        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        '''
        return True
    
    def get_groupings(self, image_set_list):
        '''Return the image groupings of the image sets in an image set list
        
        get_groupings is called after prepare_run
        
        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple has the values for
                     the key_names for this group.
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ (('A','01'), [0,96,192]),
          (('A','02'), [1,97,193]),... ]
        
        Returns None to indicate that the module does not contribute any
        groupings.
        '''
        return None
    
    def prepare_group(self, pipeline, image_set_list, grouping,
                      image_numbers):
        '''Prepare to start processing a new grouping
        
        pipeline - the pipeline being run
        image_set_list - the image_set_list for the experiment. Add image
                         providers to the image set list here.
        grouping - a dictionary that describes the key for the grouping.
                   For instance, { 'Metadata_Row':'A','Metadata_Column':'01'}
        image_numbers - a sequence of the image numbers within the
                   group (image sets can be retreved as
                   image_set_list.get_image_set(image_numbers[i]-1)
        
        prepare_group is called once after prepare_run if there are no
        groups.
        '''
        pass
    
    
    def post_group(self, workspace, grouping):
        '''Do post-processing after a group completes
        
        workspace - the workspace at the end of the group
        grouping - the group that's being run
        '''
        pass
    
    def get_measurement_columns(self, pipeline):
        '''Return a sequence describing the measurement columns needed by this module
        
        This call should return one element per image or object measurement
        made by the module during image set analysis. The element itself
        is a 3-tuple:
        first entry: either one of the predefined measurement categories,
                     {"Image", "Experiment" or "Neighbors" or the name of one
                     of the objects.}
        second entry: the measurement name (as would be used in a call 
                      to add_measurement)
        third entry: the column data type (for instance, "varchar(255)" or
                     "float")
        '''
        return []

    def get_dictionary(self, image_set_list):
        '''Get the dictionary for this module
        
        image_set_list - get the dictionary from the legacy fields
        '''
        key = "%s:%d"%(self.module_name, self.module_num)
        if not image_set_list.legacy_fields.has_key(key):
            image_set_list.legacy_fields[key] = {}
        return image_set_list.legacy_fields[key]
    
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return []
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        return []
    
    def get_measurement_images(self,pipeline,object_name,category,measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []
    
    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """Return a list of secondary object names used as a basis for a particular measure
        
        object_name - either "Image" or the name of the primary object
        category - the category being measured, for instance "Threshold"
        measurement - the name of the measurement being done
        
        Some modules output image-wide aggregate measurements in addition to
        object measurements. These must be stored using the "Image" object name
        in order to save a single value. A module can override
        get_measurement_objects to tell the user about the object name in
        those situations.
        
        In addition, some modules may make use of two segmentations, for instance
        when measuring the total value of secondary objects related to primary
        ones. This mechanism can be used to identify the secondary objects used.
        """ 
        return []
    
    def get_measurement_scales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
    
    def needs_matlab(self):
        return False

    def is_source_loaded(self, image_name):
        """Return True if this module loads this image name from file."""
        for setting in self.settings():
            if (isinstance(setting, cps.FileImageNameProvider) and
                setting.value == image_name):
                return True
        return False
    
    def should_stop_writing_measurements(self):
        '''Returns True if measurements should not be taken after this module
        
        The ExportToDatabase and ExportToExcel modules expect that no
        measurements will be recorded in latter modules. This function
        returns False in the default, indicating that measurements should
        keep being made, but returns True for these modules, indicating
        that any subsequent modules will lose their measurements and should
        not write any.
        '''
        return False
    
