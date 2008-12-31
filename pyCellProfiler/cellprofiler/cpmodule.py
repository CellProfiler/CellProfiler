"""Module.py - represents a CellProfiler pipeline module
    
    $Revision$
    
    TO-DO: capture and save module revision #s in the handles
    """
import re
import os

import numpy

import variable as cpv
import cellprofiler.cpimage
import cellprofiler.objects
import cellprofiler.measurements
import cellprofiler.pipeline
import cellprofiler.matlab.cputils

class AbstractModule(object):
    """ Derive from the abstract module class to create your own module in Python
    
    You need to implement the following in the derived class:
    UpgradeModuleFromRevision - to modify a module's variables after loading to match the current revision number
    GetHelp - to return help for the module
    VariableRevisionNumber - to return the current variable revision number
    Annotations - to return the variable annotations (see cpv.Annotation).
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
        self.__variables = []
        self.__notes = []
        self.__variable_revision_number = 0
        self.__module_name = 'unknown'
        self.__annotation_dict = None
    
    def create_from_handles(self,handles,module_num):
        """Fill a module with the information stored in the handles structure for module # ModuleNum 
        
        Returns a module with the variables decanted from the handles.
        If the revision is old, a different and compatible module can be returned.
        """
        self.create_from_annotations()
        self.__module_num = module_num
        idx = module_num-1
        settings = handles['Settings'][0,0]
        variable_values = []
        if settings.dtype.fields.has_key('ModuleNotes'):
            n=settings['ModuleNotes'][0,idx]
            self.__notes = [str(n[i,0][0]) for i in range(0,n.size)]
        else:
            self.__notes = []
        variable_count=settings['NumbersOfVariables'][0,idx]
        variable_revision_number = settings['VariableRevisionNumbers'][0,idx]
        for i in range(0,variable_count):
            value_cell = settings['VariableValues'][idx,i]
            if isinstance(value_cell,numpy.ndarray):
                if numpy.product(value_cell.shape) == 0:
                    variable_values.append('')
                else:
                    variable_values.append(str(value_cell[0]))
            else:
                variable_values.append(value_cell)
        for v,value in zip(self.variables(),variable_values):
            v.value = value
        self.upgrade_module_from_revision(variable_revision_number)
        self.on_post_load()
    
    def create_from_annotations(self):
        """Create the variables based on what you can discern from the annotations
        """
        annotation_dict = {}
        value_dict = {}
        max_variable = 0
        
        for annotation in self.annotations():
            vn = annotation.variable_number
            if not annotation_dict.has_key(vn):
                annotation_dict[vn] = {}
            if not annotation_dict[vn].has_key(annotation.kind):
                annotation_dict[vn][annotation.kind] = []
            annotation_dict[vn][annotation.kind].append(annotation.value)
            if annotation.kind == 'default':
                value_dict[vn] = annotation.value
            elif annotation.kind == 'choice' and not value_dict.has_key(vn):
                value_dict[vn] = annotation.value
            if vn > max_variable:
                max_variable = vn
        
        variables = []
        for i in range(1,max_variable+1):
            assert annotation_dict.has_key(i), 'There are no annotations for variable # %d'%(i)
            variable_dict = annotation_dict[i]
            if variable_dict.has_key('text'):
                text = variable_dict['text'][0]
            elif variable_dict.has_key('pathnametext'):
                text = variable_dict['pathnametext'][0]
            elif variable_dict.has_key('filenametext'):
                text = variable_dict['filenametext'][0]
            else:
                text = ''
            if variable_dict.has_key('infotype'):
                default_value = value_dict.get(i,cpv.DO_NOT_USE)
                parts = variable_dict['infotype'][0].split(' ')
                if parts[-1] == 'indep':
                    v = cpv.NameProvider(text,parts[0], default_value)
                else:
                    v = cpv.NameSubscriber(text,parts[0],default_value)
            elif variable_dict.has_key('inputtype') and \
                 variable_dict['inputtype'][0] == 'popupmenu':
                choices = variable_dict['choice']
                if len(choices) == 2 and all([choice in [cpv.YES,cpv.NO] for choice in choices]):
                    v = cpv.Binary(text,choices[0]==cpv.YES)
                else:
                    default_value = value_dict.get(i,choices[0])
                    v = cpv.Choice(text,choices,default_value)
            elif variable_dict.has_key('inputtype') and \
                 variable_dict['inputtype'][0] == 'popupmenu custom':
                choices = variable_dict['choice']
                default_value = value_dict.get(i,choices[0])
                v = cpv.CustomChoice(text,choices,default_value)
            elif variable_dict.has_key('inputtype') and \
                 variable_dict['inputtype'][0] == 'pathnametext':
                v = cpv.PathnameText(text,value_dict.get(i,cpv.DO_NOT_USE))
            elif variable_dict.has_key('inputtype') and \
                 variable_dict['inputtype'][0] == 'filenametext':
                v = cpv.FilenameText(text,value_dict.get(i,cpv.DO_NOT_USE))
            else:
                v = cpv.Text(text,value_dict.get(i,"n/a"))
            variables.append(v)
                     
        self.set_variables(variables)
        self.on_post_load()
    
    def on_post_load(self):
        """This is a convenient place to do things to your module after the variables have been loaded or initialized"""
        pass

    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the variables in the module to upgrade it to its current revision number
        
        """
        raise NotImplementedError("Please implement UpgradeModuleFromRevision")
    
    def get_help(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    def save_to_handles(self,handles):
        module_idx = self.module_num-1
        setting = handles[cellprofiler.pipeline.SETTINGS][0,0]
        setting[cellprofiler.pipeline.MODULE_NAMES][0,module_idx] = unicode(self.module_class())
        setting[cellprofiler.pipeline.MODULE_NOTES][0,module_idx] = numpy.ndarray(shape=(len(self.notes()),1),dtype='object')
        for i in range(0,len(self.notes())):
            setting[cellprofiler.pipeline.MODULE_NOTES][0,module_idx][i,0]=self.notes()[i]
        setting[cellprofiler.pipeline.NUMBERS_OF_VARIABLES][0,module_idx] = len(self.variables())
        for i in range(0,len(self.variables())):
            variable = self.variables()[i]
            if len(str(variable)) > 0:
                setting[cellprofiler.pipeline.VARIABLE_VALUES][module_idx,i] = unicode(str(variable))
            try: # matlab & old-style through annotations
                annotations = self.variable_annotations(variable.key())
                if annotations.has_key('infotype'):
                    setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode(annotations['infotype'][0].value)
            except:
                pass
            if isinstance(variable,cpv.NameProvider):
                setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode("%s indep"%(variable.group))
            elif isinstance(variable,cpv.NameSubscriber):
                setting[cellprofiler.pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode(variable.group)
        setting[cellprofiler.pipeline.VARIABLE_REVISION_NUMBERS][0,module_idx] = self.variable_revision_number
        setting[cellprofiler.pipeline.MODULE_REVISION_NUMBERS][0,module_idx] = 0
    
    def variable_annotations(self,key):
        """Return annotations for the variable with the given number
        
        """
        if not self.__annotation_dict:
            self.__annotation_dict = cpv.get_annotations_as_dictionary(self.annotations())
        if self.__annotation_dict.has_key(key):
            return self.__annotation_dict[key]
        indexes = [index+1 for variable,index in zip(self.variables(),range(len(self.variables()))) if variable.key()==key]
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
    
    def variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        raise NotImplementedError("Please implement VariableRevisionNumber in the derived class")
    
    def variables(self):
        """A module's variables
        
        """
        return self.__variables

    def variable(self,variable_num):
        """Reference a variable by its one-based variable number
        """
        return self.variables()[variable_num-1]
    
    def set_variables(self,variables):
        self.__variables = variables
        
    def visible_variables(self):
        """The variables that are visible in the UI
        """
        return self.variables()

    def annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the cpv.Annotation
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
    
    def prepare_run(self, pipeline, image_set_list):
        """Prepare the image set list for a run (& whatever else you want to do)
        """
        pass
    
    def run(self,pipeline,image_set,object_set,measurements,frame=None):
        """Run the module (abstract method)
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        raise(NotImplementedError("Please implement the Run method to do whatever your module does, or use the MatlabModule class for Matlab modules"));

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
    
    def get_measurement_scales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
    
    def category(self):
        raise(NotImplementedError("Please implement the Category method to return the category for the module in the AddModules page"));

class TemplateModule(AbstractModule):
    """Cut and paste this in order to get started writing a module
    """
    def __init__(self):
        AbstractModule.__init__(self)
        self.SetModuleName("Template")
    
    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the variables in the module to upgrade it to its current revision number
        
        """
        raise NotImplementedError("Please implement UpgradeModuleFromRevision")
    
    def get_help(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    def variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        raise NotImplementedError("Please implement VariableRevisionNumber in the derived class")
    
    def annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the cpv.Annotation
        class.
        """
        raise("Please implement Annotations in your derived class")
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """
        
    def run(self,pipeline,image_set,object_set,measurements,frame=None):
        """Run the module (abstract method)
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        raise(NotImplementedError("Please implement the Run method to do whatever your module does, or use the MatlabModule class for Matlab modules"));

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
    
    def get_measurement_scales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
        
class MatlabModule(AbstractModule):
    """A matlab module, as from a .m file
    
    """
    def __init__(self):
        AbstractModule.__init__(self)
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
        return AbstractModule.create_from_handles(self, handles, module_num)

    def create_from_file(self,file_path,module_num):
        """Parse a file to get the default variables for a module
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
        
    def run(self,pipeline,image_set,object_set,measurements,frame=None):
        """Run the module in Matlab
        
        """
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
        """Read and return the annotations and variable revision # from a file
        
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
                annotations.append(cpv.Annotation(line))
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
        """Rewrite the variables to upgrade the module from the given revision number.
        
        """
        if variable_revision_number != self.target_variable_revision_number():
            raise RuntimeError("Module #%d (%s) was saved at revision #%d but current version is %d and no rewrite rules exist"%(self.module_num,self.module_name,variable_revision_number,self.target_variable_revision_number()))
        self.__variable_revision_number = variable_revision_number
        return self
    
    def variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        return self.__variable_revision_number
    
    def target_variable_revision_number(self):
        """The variable revision number we need in order to run the module
        
        """
        if not self.__target_revision_number:
            self.load_annotations()
        return self.__target_variable_revision_number
    
    def annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the cpv.Annotation
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
