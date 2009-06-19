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

import numpy

import cellprofiler.settings as cps
import cellprofiler.cpimage
import cellprofiler.objects
import cellprofiler.measurements
import cellprofiler.pipeline
import cellprofiler.matlab.cputils

class CPModule(object):
    """ Derive from the abstract module class to create your own module in Python
    
    You need to implement the following in the derived class:
    UpgradeModuleFromRevision - to modify a module's variables after loading to match the current revision number
    GetHelp - to return help for the module
    VariableRevisionNumber - to return the current variable revision number
    Annotations - to return the variable annotations (see cps.Annotation).
                  These are the annotations in the .M file (like choiceVAR05 = Yes)
    Run - to run the module, producing measurements, etc.
    
    Implement these if you produce measurements:
    GetCategories - The category of measurement produced, for instance AreaShape
    GetMeasurements - The measurements produced by a category
    GetMeasurementImages - The images measured for a particular measurement
    GetMeasurementScales - the scale at which a measurement was taken
    """
    
    def __init__(self):
        self.__module_num = -1
        self.__settings = []
        self.__notes = []
        self.__variable_revision_number = 0
        self.__module_name = 'unknown'
        self.__annotation_dict = None
        self.create_settings()
        
    def create_settings(self):
        """Create your settings by subclassing this function"""
        pass
    
    def create_from_handles(self,handles,module_num):
        """Fill a module with the information stored in the handles structure for module # ModuleNum 
        
        Returns a module with the settings decanted from the handles.
        If the revision is old, a different and compatible module can be returned.
        """
        self.__module_num = module_num
        idx = module_num-1
        settings = handles['Settings'][0,0]
        setting_values = []
        if settings.dtype.fields.has_key('ModuleNotes'):
            n=settings['ModuleNotes'][0,idx]
            self.__notes = [str(n[i,0][0]) for i in range(0,n.size)]
        else:
            self.__notes = []
        setting_count=settings['NumbersOfVariables'][0,idx]
        variable_revision_number = settings['VariableRevisionNumbers'][0,idx]
        module_name = settings['ModuleNames'][0,idx][0]
        for i in range(0,setting_count):
            value_cell = settings['VariableValues'][idx,i]
            if isinstance(value_cell,numpy.ndarray):
                if numpy.product(value_cell.shape) == 0:
                    setting_values.append('')
                else:
                    setting_values.append(str(value_cell[0]))
            else:
                setting_values.append(value_cell)
        self.set_setting_values(setting_values, variable_revision_number, 
                                 module_name)
        self.on_post_load()
    
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
            self.backwards_compatibilize(setting_values,
                                         variable_revision_number,
                                         module_name,
                                         not '.' in module_name)
        self.prepare_to_set_values(setting_values)
        for v,value in zip(self.settings(),setting_values):
            v.value = value
        self.upgrade_module_from_revision(variable_revision_number)
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
                                module_name,from_matlab):
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
        self.on_post_load()
    
    def on_post_load(self):
        """This is a convenient place to do things to your module after the settings have been loaded or initialized"""
        pass

    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the settings in the module to upgrade it to its current revision number
        
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
        
        """
        return self.__doc__
            
    def save_to_handles(self,handles):
        module_idx = self.module_num-1
        setting = handles[cellprofiler.pipeline.SETTINGS][0,0]
        setting[cellprofiler.pipeline.MODULE_NAMES][0,module_idx] = unicode(self.module_class())
        setting[cellprofiler.pipeline.MODULE_NOTES][0,module_idx] = numpy.ndarray(shape=(len(self.notes()),1),dtype='object')
        for i in range(0,len(self.notes())):
            setting[cellprofiler.pipeline.MODULE_NOTES][0,module_idx][i,0]=self.notes()[i]
        setting[cellprofiler.pipeline.NUMBERS_OF_VARIABLES][0,module_idx] = len(self.settings())
        for i in range(0,len(self.settings())):
            variable = self.settings()[i]
            if len(str(variable)) > 0:
                setting[cellprofiler.pipeline.VARIABLE_VALUES][module_idx,i] = unicode(str(variable))
            try: # matlab & old-style through annotations
                annotations = self.setting_annotations(variable.key())
                if annotations.has_key('infotype'):
                    setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode(annotations['infotype'][0].value)
            except:
                pass
            if isinstance(variable,cps.NameProvider):
                setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode("%s indep"%(variable.group))
            elif isinstance(variable,cps.NameSubscriber):
                setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode(variable.group)
        setting[cellprofiler.pipeline.VARIABLE_REVISION_NUMBERS][0,module_idx] = self.variable_revision_number
        setting[cellprofiler.pipeline.MODULE_REVISION_NUMBERS][0,module_idx] = 0
    
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
        """The name shown to the user in the Add Modules box"""
        return self.__module_name
    
    def set_module_name(self, module_name):
        self.__module_name = module_name
        
    module_name = property(get_module_name,set_module_name)
    
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
        """A module's settings
        
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
        image_set_list - add any image providers for this module to this
                         image set list
        frame - parent frame of application if GUI enabled, None if GUI
                disabled
        
        return True if operation completed, False if aborted 
        """
        return True
    
    def run(self,workspace):
        """Run the module (abstract method)
        
        workspace    - The workspace contains
            pipeline     - instance of CellProfiler.Pipeline for this run
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
    
class MatlabModule(CPModule):
    """A matlab module, as from a .m file
    
    """
    def __init__(self):
        super(MatlabModule,self).__init__()
        self.__annotations = None
        self.__filename = None
        self.__help = None
        self.__annotations = None
        self.__features = None
        self.__target_revision_number = None
        
    def create_from_handles(self,handles,module_num):
        settings = handles['Settings'][0,0]
        idx = module_num-1
        module_name = str(settings['ModuleNames'][0,idx][0])
        self.set_module_name(module_name)
        self.__filename = os.path.join(cellprofiler.preferences.module_directory(),
                                       module_name+cellprofiler.preferences.module_extension())
        self.create_from_annotations()
        return super(MatlabModule,self).create_from_handles(handles, module_num)

    def create_from_file(self,file_path,module_num):
        """Parse a file to get the default settings for a module
        """
        self.set_module_num(module_num)
        self.set_module_name(os.path.splitext(os.path.split(file_path)[1])[0])
        self.__filename = file_path
        self.load_annotations()
        self.create_from_annotations()
        self.__variable_revision_number = self.target_variable_revision_number()

    def load_annotations(self):
        """Load the annotations for the module
        
        """
        file = open(self.__filename)
        try:
            (self.__annotations, self.__target_variable_revision_number,self.__help,self.__features) = self.__read_annotations(file)
        finally:
            file.close()
        
    def run(self,workspace):
        """Run the module in Matlab
        
        """
        pipeline = workspace.pipeline
        image_set = workspace.image_set
        object_set = workspace.object_set
        measurements = workspace.measurements
        matlab = cellprofiler.matlab.cputils.get_matlab_instance()
        handles = pipeline.load_pipeline_into_matlab(image_set,object_set,measurements)
        handles.Current.CurrentModuleNumber = str(self.module_num)
        figure_field = 'FigureNumberForModule%d'%(self.module_num)
        if measurements.image_set_number == 0:
            if handles.Preferences.DisplayWindows[self.module_num-1] == 0:
                # Make up a fake figure for the module if we're not displaying its window
                self.__figure = math.ceil(max(matlab.findobj()))+1 
            else:
                self.__figure = matlab.CPfigure(handles,'','Name','%s Display, cycle # '%(self.module_name))
        handles.Current = matlab.setfield(handles.Current, figure_field, self.__figure)
            
        handles = matlab.feval(self.module_name,handles)
        cellprofiler.pipeline.add_matlab_images(handles, image_set)
        cellprofiler.pipeline.add_matlab_objects(handles, object_set)
        cellprofiler.pipeline.add_matlab_measurements(handles, measurements)

    def __read_annotations(self,file):
        """Read and return the annotations and setting revision # from a file
        
        """
        annotations = []
        variable_revision_number = 0
        before_help = True
        after_help = False
        help = []
        features = []
        for line in file:
            if before_help and line[0]=='%':
                before_help = False
            if (not before_help) and (not after_help):
                if line[0]=='%':
                    help.append(line[2:-1])
                    continue
                else:
                    after_help=True
            try:
                annotations.append(cps.Annotation(line))
            except:
                # Might be something else...
                match = re.match('^%%%VariableRevisionNumber = ([0-9]+)',line)
                if match:
                    variable_revision_number = int(match.groups()[0]) 
                    break
                match = re.match('^%feature:([a-zA-Z]+)',line)
                if match:
                    features.append(match.groups()[0])
        return annotations,variable_revision_number,'\n'.join(help),features

    def module_class(self):
        """The class to instantiate, except for the special case of matlab modules.
        
        """
        return self.module_name

    def upgrade_module_from_revision(self,variable_revision_number):
        """Rewrite the settings to upgrade the module from the given revision number.
        
        """
        if variable_revision_number != self.target_variable_revision_number():
            raise RuntimeError("Module #%d (%s) was saved at revision #%d but current version is %d and no rewrite rules exist"%(self.module_num,self.module_name,variable_revision_number,self.target_variable_revision_number()))
        self.__variable_revision_number = variable_revision_number
        return self
    
    def get_variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        return self.__variable_revision_number
    variable_revision_number = property(get_variable_revision_number)
    
    def target_variable_revision_number(self):
        """The setting revision number we need in order to run the module
        
        """
        if not self.__target_revision_number:
            self.load_annotations()
        return self.__target_variable_revision_number
    
    def annotations(self):
        """Return the setting annotations, as read out of the module file.
        
        Return the setting annotations, as read out of the module file.
        Each annotation is an instance of the cps.Annotation
        class.
        """
        if not self.__annotations:
            self.load_annotations()
        return self.__annotations

    def features(self):
        """Return the features that this module supports (e.g. categories & measurements)
        
        """
        if not self.__features:
            self.load_annotations()
        return self.__features
    
    def get_help(self):
        """Return help text for the module
        
        """
        if not self.__help:
            self.LoadAnnotations()
        return self.__help
    
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if 'measurements' in self.features():
            handles = pipeline.load_pipeline_into_matlab()
            handles.Current.CurrentModuleNumber = str(self.ModuleNum())
            matlab=cellprofiler.matlab.utils.get_matlab_instance()
            measurements = matlab.feval(self.module_name,handles,'measurements',str(object_name),str(category))
            count=matlab.eval('length(%s)'%(measurements._name))
            result=[]
            for i in range(0,count):
                result.append(matlab.eval('%s{%d}'%(measurements._name,i+1)))
            return result
        else:
            return []
            
    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if 'categories' in self.features():
            handles = pipeline.load_pipeline_into_matlab()
            handles.Current.CurrentModuleNumber = str(self.module_num)
            matlab=cellprofiler.matlab.utils.get_matlab_instance()
            categories = matlab.feval(self.module_name,handles,'categories',str(object_name))
            count=matlab.eval('length(%s)'%(categories._name))
            result=[]
            for i in range(0,count):
                result.append(matlab.eval('%s{%d}'%(categories._name,i+1)))
            return result
        else:
            return []
    
    def needs_matlab(self):
        return True
