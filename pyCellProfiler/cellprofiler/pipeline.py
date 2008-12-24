"""Pipeline.py - an ordered set of modules to be executed

"""
__version = "$Revision$"

import numpy
import scipy.io.matlab
import os
import sys
import tempfile
import datetime
import traceback
import cellprofiler.cpmodule
import cellprofiler.preferences
from cellprofiler.matlab.utils import new_string_cell_array,get_matlab_instance
from cellprofiler.matlab.utils import s_cell_fun,make_cell_struct_dtype
from cellprofiler.matlab.utils import load_into_matlab,get_int_from_matlab
from cellprofiler.matlab.utils import encapsulate_strings_in_arrays
import cellprofiler.variablechoices
import cellprofiler.cpimage
import cellprofiler.measurements
import cellprofiler.objects

CURRENT = 'Current'
NUMBER_OF_IMAGE_SETS     = 'NumberOfImageSets'
NUMBER_OF_MODULES        = 'NumberOfModules'
SET_BEING_ANALYZED       = 'SetBeingAnalyzed'
SAVE_OUTPUT_HOW_OFTEN    = 'SaveOutputHowOften'
TIME_STARTED             = 'TimeStarted'
STARTING_IMAGE_SET       = 'StartingImageSet'
STARTUP_DIRECTORY        = 'StartupDirectory'
DEFAULT_MODULE_DIRECTORY = 'DefaultModuleDirectory'
DEFAULT_IMAGE_DIRECTORY  = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'
IMAGE_TOOLS_FILENAMES    = 'ImageToolsFilenames'
IMAGE_TOOL_HELP          = 'ImageToolHelp'
PREFERENCES              = 'Preferences'
PIXEL_SIZE               = 'PixelSize'
SKIP_ERRORS              = 'SkipErrors'
INTENSITY_COLOR_MAP      = 'IntensityColorMap'
LABEL_COLOR_MAP          = 'LabelColorMap'
STRIP_PIPELINE           = 'StripPipeline'
DISPLAY_MODE_VALUE       = 'DisplayModeValue'
DISPLAY_WINDOWS          = 'DisplayWindows'
FONT_SIZE                = 'FontSize'
IMAGES                   = 'Images'
MEASUREMENTS             = 'Measurements'
PIPELINE                 = 'Pipeline'    
SETTINGS                  = 'Settings'
VARIABLE_VALUES           = 'VariableValues'
VARIABLE_INFO_TYPES       = 'VariableInfoTypes'
MODULE_NAMES              = 'ModuleNames'
PIXEL_SIZE                = 'PixelSize'
NUMBERS_OF_VARIABLES      = 'NumbersOfVariables'
VARIABLE_REVISION_NUMBERS = 'VariableRevisionNumbers'
MODULE_REVISION_NUMBERS   = 'ModuleRevisionNumbers'
MODULE_NOTES              = 'ModuleNotes'
CURRENT_MODULE_NUMBER     = 'CurrentModuleNumber'
SETTINGS_DTYPE = numpy.dtype([(VARIABLE_VALUES, '|O4'), 
                            (VARIABLE_INFO_TYPES, '|O4'), 
                            (MODULE_NAMES, '|O4'), 
                            (NUMBERS_OF_VARIABLES, '|O4'), 
                            (PIXEL_SIZE, '|O4'), 
                            (VARIABLE_REVISION_NUMBERS, '|O4'), 
                            (MODULE_REVISION_NUMBERS, '|O4'), 
                            (MODULE_NOTES, '|O4')])
CURRENT_DTYPE = make_cell_struct_dtype([ NUMBER_OF_IMAGE_SETS,SET_BEING_ANALYZED,NUMBER_OF_MODULES, 
                                     SAVE_OUTPUT_HOW_OFTEN,TIME_STARTED, STARTING_IMAGE_SET,
                                     STARTUP_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, DEFAULT_IMAGE_DIRECTORY, 
                                     IMAGE_TOOLS_FILENAMES, IMAGE_TOOL_HELP])
PREFERENCES_DTYPE = make_cell_struct_dtype([PIXEL_SIZE, DEFAULT_MODULE_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, 
                                         DEFAULT_IMAGE_DIRECTORY, INTENSITY_COLOR_MAP, LABEL_COLOR_MAP,
                                         STRIP_PIPELINE, SKIP_ERRORS, DISPLAY_MODE_VALUE, FONT_SIZE,
                                         DISPLAY_WINDOWS])
 
def add_matlab_images(handles,image_set):
    """Add any images from the handles to the image set
    Generally, the handles have images added as they get returned from a Matlab module.
    You can use this to update the image set and capture them.
    """
    matlab = get_matlab_instance()
    pipeline_fields = matlab.fields(handles.Pipeline)
    provider_set = set([x.name for x in image_set.providers])
    image_fields = set()
    crop_fields = set()
    for i in range(0,int(matlab.length(pipeline_fields)[0,0])):
        field = matlab.cell2mat(pipeline_fields[i])
        if field.startswith('CropMask'):
            crop_fields.add(field)
        elif field.startswith('Segmented') or field.startswith('UneditedSegmented') or field.startswith('SmallRemovedSegmented'):
            continue
        elif field.startswith('Pathname') or field.startswith('FileList') or field.startswith('Filename'):
            if not image_set.legacy_fields.has_key(field):
                value = matlab.getfield(handles.Pipeline,field)
                if not (isinstance(value,str) or isinstance(value,unicode)):
                    # The two supported types: string/unicode or cell array of strings
                    count = matlab.length(value)
                    new_value = numpy.ndarray((1,count),dtype='object')
                    for j in range(0,count):
                        new_value[0,j] = matlab.cell2mat(value[j])
                    value = new_value 
                image_set.legacy_fields[field] = value
        elif not field in provider_set:
            image_fields.add(field)
    for field in image_fields:
        image = cellprofiler.image.Image()
        image.Image = matlab.getfield(handles.Pipeline,field)
        crop_field = 'CropMask'+field
        if crop_field in crop_fields:
            image.Mask = matlab.getfield(handles.Pipeline,crop_field)
        image_set.providers.append(cellprofiler.cpimage.VanillaImageProvider(field,image))
    number_of_image_sets = get_int_from_matlab(handles.Current.NumberOfImageSets)
    if (not image_set.legacy_fields.has_key(NUMBER_OF_IMAGE_SETS)) or number_of_image_sets < image_set.legacy_fields[NUMBER_OF_IMAGE_SETS]:
        image_set.legacy_fields[NUMBER_OF_IMAGE_SETS] = number_of_image_sets

def add_matlab_objects(handles,object_set):
    """Add any objects from the handles to the object set
    You can use this to update the object set after calling a matlab module
    """
    matlab = get_matlab_instance()
    pipeline_fields = matlab.fields(handles.Pipeline)
    objects_names = set(object_set.get_object_names())
    segmented_fields = set()
    unedited_segmented_fields = set()
    small_removed_segmented_fields = set()
    for i in range(0,int(matlab.length(pipeline_fields)[0,0])):
        field = matlab.cell2mat(pipeline_fields[i])
        if field.startswith('Segmented'):
            segmented_fields.add(field)
        elif field.startswith('UneditedSegmented'):
            unedited_segmented_fields.add(field)
        elif field.startswith('SmallRemovedSegmented'):
            small_removed_segmented_fields.add(field)
    for field in segmented_fields:
        object_name = field.replace('Segmented','')
        if object_name in object_set.get_object_names():
            continue
        objects = cellprofiler.objects.Objects()
        objects.segmented = matlab.getfield(handles.Pipeline,field)
        unedited_field ='Unedited'+field
        small_removed_segmented_field = 'SmallRemoved'+field 
        if unedited_field in unedited_segmented_fields:
            objects.unedited_segmented = matlab.getfield(handles.Pipeline,unedited_field)
        if small_removed_segmented_field in small_removed_segmented_fields:
            objects.SmallRemovedSegmented = matlab.getfield(handles.Pipeline,small_removed_segmented_field)
        object_set.add_objects(objects,object_name)

def add_matlab_measurements(handles, measurements):
    """Get measurements made by Matlab and put them into our Python measurements object
    """
    matlab = get_matlab_instance()
    measurement_fields = matlab.fields(handles.Measurements)
    for i in range(0,int(matlab.length(measurement_fields)[0,0])):
        field = matlab.cell2mat(measurement_fields[i])
        object_measurements = matlab.getfield(handles.Measurements,field)
        object_fields = matlab.fields(object_measurements)
        for j in range(0,int(matlab.length(object_fields)[0,0])):
            feature = matlab.cell2mat(object_fields[j])
            if not measurements.has_current_measurements(field,feature):
                value = matlab.cell2mat(matlab.getfield(object_measurements,feature)[measurements.image_set_number])
                if not isinstance(value,numpy.ndarray) or numpy.product(value.shape) > 0:
                    # It's either not a numpy array (it's a string) or it's not the empty numpy array
                    # so add it to the measurements
                    measurements.add_measurement(field,feature,value)

def add_all_images(handles,image_set, object_set):
    """ Add all images to the handles structure passed
    
    Add images to the handles structure, for example in the Python sandwich.
    """
    images = {}
    for provider in image_set.providers:
        name = provider.name()
        image = image_set.get_image(name)
        images[name] = image.image
        if image.has_mask:
            images['CropMask'+name] = image.mask
    
    for object_name in object_set.object_names:
        objects = object_set.get_objects(object_name)
        images['Segmented'+object_name] = objects.segmented
        if objects.has_unedited_segmented():
            images['UneditedSegmented'+object_name] = objects.unedited_segmented
        if objects.has_small_removed_segmented():
            images['SmallRemovedSegmented'+object_name] = objects.small_removed_segmented
    
    npy_images = numpy.ndarray((1,1),dtype=make_cell_struct_dtype(images.keys()))
    for key,image in images.iteritems():
        npy_images[key][0,0] = image
    handles[PIPELINE]=npy_images

def add_all_measurements(handles, measurements):
    """Add all measurements from our measurements object into the numpy structure passed
    
    """
    measurements_dtype = make_cell_struct_dtype(measurements.get_object_names())
    npy_measurements = numpy.ndarray((1,1),dtype=measurements_dtype)
    handles[MEASUREMENTS]=npy_measurements
    for object_name in measurements.get_object_names():
        object_dtype = make_cell_struct_dtype(measurements.get_feature_names(object_name))
        object_measurements = numpy.ndarray((1,1),dtype=object_dtype)
        npy_measurements[object_name][0,0] = object_measurements
        for feature_name in measurements.get_feature_names(object_name):
            feature_measurements = numpy.ndarray((1,measurements.image_set_number+1),dtype='object')
            object_measurements[feature_name][0,0] = feature_measurements
            for i in range(0,measurements.image_set_number+1):
                data = measurements.get_current_measurement(object_name,feature_name)
                if data != None:
                    feature_measurements[0,i] = data

class Pipeline:
    """A pipeline represents the modules that a user has put together
    to analyze their images.
    
    """
    def __init__(self):
        self.__modules = [];
        self.__listeners = [];
        self.__infogroups = {};
        self.__variable_choices = {}
    
    def create_from_handles(self,handles):
        """Read a pipeline's modules out of the handles structure
        
        """
        self.__modules = [];
        self.__variable_choices = {}
        settings = handles[SETTINGS][0,0]
        module_names = settings[MODULE_NAMES]
        module_count = module_names.shape[1]
        for module_num in range(1,module_count+1):
            idx = module_num-1
            module_name = module_names[0,idx][0]
            module = self.instantiate_module(module_name)
            module.create_from_handles(handles, module_num)
            self.__modules.append(module)
            self.__hook_module_variables(module)
        self.notify_listeners(PipelineLoadedEvent())
    
    def instantiate_module(self,module_name):
        if module_name.find('.') != -1:
            parts     = module_name.split('.')
            pkg_name  = '.'.join(parts[:-1])
            pkg       = __import__(pkg_name)
            module    = eval("%s()"%(module_name))
        else:
            module = cellprofiler.cpmodule.MatlabModule()
        return module
        
    def save_to_handles(self):
        """Create a numpy array representing this pipeline
        
        """
        settings = numpy.ndarray(shape=[1,1],dtype=SETTINGS_DTYPE)
        handles = {SETTINGS:settings }
        setting = settings[0,0]
        # The variables are a (modules,max # of variables) array of cells (objects)
        # where an empty cell is a (1,0) array of float64
        variable_count = max([len(module.variables()) for module in self.modules()])
        module_count = len(self.modules())
        setting[VARIABLE_VALUES] =          new_string_cell_array((module_count,variable_count))
        # The variable info types are similarly shaped
        setting[VARIABLE_INFO_TYPES] =      new_string_cell_array((module_count,variable_count))
        setting[MODULE_NAMES] =             new_string_cell_array((1,module_count))
        setting[NUMBERS_OF_VARIABLES] =     numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint8'))
        setting[PIXEL_SIZE] =               cellprofiler.preferences.get_pixel_size() 
        setting[VARIABLE_REVISION_NUMBERS] =numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint8'))
        setting[MODULE_REVISION_NUMBERS] =  numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint16'))
        setting[MODULE_NOTES] =             new_string_cell_array((1,module_count))
        for module in self.modules():
            module.save_to_handles(handles)
        return handles
    
    def save_measurements(self,filename, measurements):
        """Save the measurements and the pipeline settings in a Matlab file
        
        filename     - name of file to create
        measurements - measurements structure that is the result of running the pipeline
        """
        handles = self.build_matlab_handles()
        add_all_measurements(handles, measurements)
        handles[CURRENT][NUMBER_OF_IMAGE_SETS][0,0] = float(measurements.image_set_number+1)
        handles[CURRENT][SET_BEING_ANALYZED][0,0] = float(measurements.image_set_number+1)
        #
        # For the output file, you have to bury it a little deeper - the root has to have
        # a single field named "handles"
        #
        root = {'handles':numpy.ndarray((1,1),dtype=cellprofiler.matlab.utils.make_cell_struct_dtype(handles.keys()))}
        for key,value in handles.iteritems():
            root['handles'][key][0,0]=value
        scipy.io.matlab.mio.savemat(filename,root,format='5',long_field_names=True)
        
    
    def load_pipeline_into_matlab(self, image_set=None, object_set=None, measurements=None):
        """Load the pipeline into the Matlab singleton and return the handles structure
        
        The handles structure has all of the goodies needed to run the pipeline including
        * Settings
        * Current (set up to run the first image with the first module
        * Measurements - filled in from measurements (TO_DO)
        * Pipeline - filled in from the image set (TO_DO)
        Returns the handles proxy
        """
        handles = self.build_matlab_handles(image_set, object_set, measurements)
        mat_handles = load_into_matlab(handles)
        
        matlab = cellprofiler.matlab.utils.get_matlab_instance()
        if not handles.has_key(MEASUREMENTS):
            mat_handles.Measurements = matlab.struct()
        if not handles.has_key(PIPELINE):
            mat_handles.Pipeline = matlab.struct()
        return mat_handles
    
    def build_matlab_handles(self, image_set = None, object_set = None, measurements=None):
        handles = self.save_to_handles()
        image_tools_dir = os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'ImageTools')
        image_tools = [str(os.path.split(os.path.splitext(filename)[0])[1])
                       for filename in os.listdir(image_tools_dir)
                       if os.path.splitext(filename)[1] == '.m']
        image_tools.insert(0,'Image tools')
        npy_image_tools = numpy.ndarray((1,len(image_tools)),dtype=numpy.dtype('object'))
        for tool,idx in zip(image_tools,range(0,len(image_tools))):
            npy_image_tools[0,idx] = tool
            
        current = numpy.ndarray(shape=[1,1],dtype=CURRENT_DTYPE)
        handles[CURRENT]=current
        current[NUMBER_OF_IMAGE_SETS][0,0]     = [(image_set != None and image_set.legacy_fields.has_key(NUMBER_OF_IMAGE_SETS) and image_set.legacy_fields[NUMBER_OF_IMAGE_SETS]) or 1]
        current[SET_BEING_ANALYZED][0,0]       = [(measurements and measurements.image_set_number + 1) or 1]
        current[NUMBER_OF_MODULES][0,0]        = [len(self.__modules)]
        current[SAVE_OUTPUT_HOW_OFTEN][0,0]    = [1]
        current[TIME_STARTED][0,0]             = str(datetime.datetime.now())
        current[STARTING_IMAGE_SET][0,0]       = [1]
        current[STARTUP_DIRECTORY][0,0]        = cellprofiler.preferences.cell_profiler_root_directory()
        current[DEFAULT_OUTPUT_DIRECTORY][0,0] = cellprofiler.preferences.get_default_output_directory()
        current[DEFAULT_IMAGE_DIRECTORY][0,0]  = cellprofiler.preferences.get_default_image_directory()
        current[IMAGE_TOOLS_FILENAMES][0,0]    = npy_image_tools
        current[IMAGE_TOOL_HELP][0,0]          = []

        preferences = numpy.ndarray(shape=(1,1),dtype=PREFERENCES_DTYPE)
        handles[PREFERENCES] = preferences
        preferences[PIXEL_SIZE][0,0]               = cellprofiler.preferences.get_pixel_size()
        preferences[DEFAULT_MODULE_DIRECTORY][0,0] = cellprofiler.preferences.module_directory()
        preferences[DEFAULT_OUTPUT_DIRECTORY][0,0] = cellprofiler.preferences.get_default_output_directory()
        preferences[DEFAULT_IMAGE_DIRECTORY][0,0]  = cellprofiler.preferences.get_default_image_directory()
        preferences[INTENSITY_COLOR_MAP][0,0]      = 'gray'
        preferences[LABEL_COLOR_MAP][0,0]          = 'jet'
        preferences[STRIP_PIPELINE][0,0]           = 'Yes'                  # TODO - get from preferences
        preferences[SKIP_ERRORS][0,0]              = 'No'                   # TODO - get from preferences
        preferences[DISPLAY_MODE_VALUE][0,0]       = [1]                    # TODO - get from preferences
        preferences[FONT_SIZE][0,0]                = [10]                   # TODO - get from preferences
        preferences[DISPLAY_WINDOWS][0,0]          = [1 for module in self.__modules] # TODO - UI allowing user to choose whether to display a window
        
        images = {}
        if image_set:
            for provider in image_set.providers:
                image = image_set.get_image(provider.name)
                if image.image != None:
                    images[provider.name]=image.image
                if image.mask != None:
                    images['CropMask'+provider.name]=image.mask
            for key,value in image_set.legacy_fields.iteritems():
                if key != NUMBER_OF_IMAGE_SETS:
                    images[key]=value
                
        if object_set:
            for name,objects in object_set.all_objects:
                images['Segmented'+name]=objects.segmented
                if objects.has_unedited_segmented():
                    images['UneditedSegmented'+name] = objects.unedited_segmented
                if objects.has_small_removed_segmented():
                    images['SmallRemovedSegmented'+name] = objects.small_removed_segmented
                    
        if len(images):
            pipeline_dtype = make_cell_struct_dtype(images.keys())
            pipeline = numpy.ndarray((1,1),dtype=pipeline_dtype)
            handles[PIPELINE] = pipeline
            for name,image in images.items():
                pipeline[name][0,0] = images[name]

        no_measurements = (measurements == None or len(measurements.get_object_names())==0)
        if not no_measurements:
            measurements_dtype = make_cell_struct_dtype(measurements.get_object_names())
            npy_measurements = numpy.ndarray((1,1),dtype=measurements_dtype)
            handles['Measurements']=npy_measurements
            for object_name in measurements.get_object_names():
                object_dtype = make_cell_struct_dtype(measurements.get_feature_names(object_name))
                object_measurements = numpy.ndarray((1,1),dtype=object_dtype)
                npy_measurements[object_name][0,0] = object_measurements
                for feature_name in measurements.get_feature_names(object_name):
                    feature_measurements = numpy.ndarray((1,measurements.image_set_number+1),dtype='object')
                    object_measurements[feature_name][0,0] = feature_measurements
                    data = measurements.get_current_measurement(object_name,feature_name)
                    feature_measurements.fill(numpy.ndarray((0,),dtype=numpy.float64))
                    if data != None:
                        feature_measurements[0,measurements.image_set_number] = data
        return handles
    
    def run(self,frame = None):
        """Run the pipeline
        
        Run the pipeline, returning the measurements made
        """
        matlab = cellprofiler.matlab.utils.get_matlab_instance()
        self.set_matlab_path()
        display_size = (1024,768)
        image_set_list = cellprofiler.cpimage.ImageSetList()
        measurements = cellprofiler.measurements.Measurements()
        
        for module in self.modules():
            try:
                module.prepare_run(self, image_set_list)
            except Exception,instance:
                traceback.print_exc()
                event = RunExceptionEvent(instance,module)
                self.notify_listeners(event)
                if event.cancel_run:
                    return None
            
        first_set = True
        while first_set or \
            image_set_list.count()>measurements.image_set_number+1 or \
            (image_set_list.legacy_fields.has_key(NUMBER_OF_IMAGE_SETS) and
             image_set_list.legacy_fields[NUMBER_OF_IMAGE_SETS] > measurements.image_set_number+1):
            if not first_set:
                measurements.next_image_set()
            numberof_windows = 0;
            slot_number = 0
            object_set = cellprofiler.objects.ObjectSet()
            image_set = image_set_list.get_image_set(measurements.image_set_number)
            for module in self.modules():
                module_error_measurement = 'ModuleError_%02d%s'%(module.module_num,module.module_name)
                failure = 1
                try:
                    module.run(self,image_set,object_set,measurements, frame)
                    failure = 0
                except Exception,instance:
                    traceback.print_exc()
                    event = RunExceptionEvent(instance,module)
                    self.notify_listeners(event)
                    if event.cancel_run:
                        return None
                if module.module_name != 'Restart':
                    measurements.add_measurement('Image',module_error_measurement,failure);
            first_set = False
        return measurements

    def experimental_run(self,frame = None):
        """Run the pipeline - experimental, uses yield
        
        Run the pipeline, returning the measurements made
        """
        matlab = cellprofiler.matlab.utils.get_matlab_instance()
        self.set_matlab_path()
        display_size = (1024,768)
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurements.Measurements()
        
        for module in self.modules():
            try:
                module.prepare_run(self, image_set_list)
            except Exception,instance:
                traceback.print_exc()
                event = RunExceptionEvent(instance,module)
                self.notify_listeners(event)
                if event.cancel_run:
                    return
            
        first_set = True
        while first_set or \
            image_set_list.count()>measurements.image_set_number+1 or \
            (image_set_list.legacy_fields.has_key(NUMBER_OF_IMAGE_SETS) and
             image_set_list.legacy_fields[NUMBER_OF_IMAGE_SETS] > measurements.image_set_number+1):
            if not first_set:
                measurements.next_image_set()
            numberof_windows = 0;
            slot_number = 0
            object_set = cellprofiler.objects.ObjectSet()
            image_set = image_set_list.get_image_set(measurements.image_set_number)
            for module in self.modules():
                module_error_measurement = 'ModuleError_%02d%s'%(module.module_num,module.module_name)
                failure = 1
                try:
                    module.run(self,image_set,object_set,measurements, frame)
                    failure = 0
                except Exception,instance:
                    traceback.print_exc()
                    event = RunExceptionEvent(instance,module)
                    self.notify_listeners(event)
                    if event.cancel_run:
                        return
                if module.module_name != 'Restart':
                    measurements.add_measurement('Image',module_error_measurement,failure);
                yield measurements
            first_set = False

    def set_matlab_path(self):
        matlab = cellprofiler.matlab.utils.get_matlab_instance()
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'DataTools'),matlab.path())
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'ImageTools'),matlab.path())
        matlab.path(os.path.join(cellprofiler.preferences.cell_profiler_root_directory(),'CPsubfunctions'),matlab.path())
        matlab.path(cellprofiler.preferences.module_directory(),matlab.path())
        matlab.path(cellprofiler.preferences.cell_profiler_root_directory(),matlab.path())

    def clear(self):
        old_modules = self.__modules
        self.__modules = []
        self.__variable_choices = {}
        for module in old_modules:
            module.delete()
        self.notify_listeners(PipelineClearedEvent())
    
    def move_module(self,module_num,direction):
        """Move module # ModuleNum either DIRECTION_UP or DIRECTION_DOWN in the list
        
        Move the 1-indexed module either up one or down one in the list, displacing
        the other modules in the list
        """
        idx=module_num-1
        if direction == DIRECTION_DOWN:
            if module_num >= len(self.__modules):
                raise ValueError('%(ModuleNum)d is at or after the last module in the pipeline and can''t move down'%(locals()))
            module = self.__modules[idx]
            new_module_num = module_num+1
            module.set_module_num(module_num+1)
            next_module = self.__modules[idx+1]
            next_module.set_module_num(module_num)
            self.__modules[idx]=next_module
            self.__modules[idx+1]=module
        elif direction == DIRECTION_UP:
            if module_num <= 1:
                raise ValueError('The module is at the top of the pipeline and can''t move up')
            module = self.__modules[idx]
            prev_module = self.__modules[idx-1]
            new_module_num = prev_module.module_num
            module.module_num = new_module_num
            prev_module.module_num = module_num
            self.__modules[idx]=self.__modules[idx-1]
            self.__modules[idx-1]=module
        else:
            raise ValueError('Unknown direction: %s'%(direction))    
        self.notify_listeners(ModuleMovedPipelineEvent(new_module_num,direction))
        
    def modules(self):
        return self.__modules
    
    def module(self,module_num):
        module = self.__modules[module_num-1]
        assert module.module_num==module_num,'Misnumbered module. Expected %d, got %d'%(module_num,module.module_num)
        return module
    
    def add_module(self,new_module):
        """Insert a module into the pipeline with the given module #
        
        Insert a module into the pipeline with the given module #. 
        'file_name' - the path to the file containing the variables for the module.
        ModuleNum - the one-based index for the placement of the module in the pipeline
        """
        module_num = new_module.module_num
        idx = module_num-1
        self.__modules = self.__modules[:idx]+[new_module]+self.__modules[idx:]
        for module,mn in zip(self.__modules[idx+1:],range(module_num+1,len(self.__modules)+1)):
            module.module_num = mn
        self.__hook_module_variables(new_module)
        self.notify_listeners(ModuleAddedPipelineEvent(module_num))
    
    def remove_module(self,module_num):
        """Remove a module from the pipeline
        
        Remove a module from the pipeline
        ModuleNum - the one-based index of the module
        """
        idx =module_num-1
        module = self.__modules[idx]
        self.__modules = self.__modules[:idx]+self.__modules[idx+1:]
        for variable in module.variables():
            if self.__variable_choices.has_key(variable.key()):
                self.__variable_choices.pop(variable.key())
        module.delete()
        for module in self.__modules[idx:]:
            module.module_num = module.module_num-1
        self.notify_listeners(ModuleRemovedPipelineEvent(module_num))
    
    def __hook_module_variables(self,module):
        """Create whatever VariableChoices are needed
        to represent variable dependencies, groups, etc.
        
        """
        all_variable_notes = []
        for variable_number,variable in zip(range(1,len(module.variables())+1), module.variables()):
            annotations = module.variable_annotations(variable_number)
            # variable_notes stores things we find out about variables as we
            # go along so we can refer back to them for subsequent variables
            variable_notes = {'dependency':None, 'popuptype':None, 'variable':variable }
            if annotations.has_key('inputtype'):
                split = annotations['inputtype'][0].value.split(' ')
                if split[0] == 'popupmenu' and len(split) > 1:
                    variable_notes['popuptype'] = split[1]
            # Handle both info type producers and consumers
            if annotations.has_key('infotype'):
                info = annotations['infotype'][0].value.split(' ')
                if not self.__infogroups.has_key(info[0]):
                    self.__infogroups[info[0]] = cellprofiler.variablechoices.InfoGroupVariableChoices(self)
                if len(info) > 1 and info[-1] == 'indep':
                    self.__infogroups[info[0]].add_indep_variable(variable)
                else:
                    variable_notes['dependency'] = info[0]
            elif (variable_notes['popuptype'] == 'category' and
                  len(all_variable_notes) > 0 and
                  all_variable_notes[-1]['dependency'] == 'objectgroup'):
                # A category popup with an objectgroup ahead of it.
                # We guess here that we're looking for categories
                # of measurements on the selected object
                vc = cellprofiler.variablechoices.CategoryVariableChoices(self, all_variable_notes[-1]['variable'])
                self.__variable_choices[variable.Key()] = vc
            elif (variable_notes['popuptype'] == 'measurement' and
                  len(all_variable_notes) > 1 and
                  all_variable_notes[-1]['popuptype'] == 'category' and
                  all_variable_notes[-2]['dependency'] == 'objectgroup'):
                # A measurement popup that follows an objectgroup variable and a category variable
                vc = cellprofiler.variablechoices.MeasurementVariableChoices(self,
                                                                             all_variable_notes[-2]['variable'],
                                                                             all_variable_notes[-1]['variable'])
                self.__variable_choices[variable.Key()] = vc
            all_variable_notes.append(variable_notes)
    
    def get_variable_choices(self,variable):
        """Get the variable choices instance that provides choices for this variable. Return None if not a choices variable
        """
        module = variable.module()
        annotations = module.variable_annotations(variable.variable_number())
        if annotations.has_key('infotype'):
            info = annotations['infotype'][0].value.split(' ')
            if info[-1] != 'indep':
                return self.__infogroups[info[0]]
        elif annotations.has_key('choice'):
            choices = [annotation.Value for annotation in annotations['choice']]
            return cellprofiler.variablechoices.StaticVariableChoices(choices)
        elif self.__variable_choices.has_key(variable.Key()):
            return self.__variable_choices[variable.Key()]
        
    def notify_listeners(self,event):
        """Notify listeners of an event that happened to this pipeline
        
        """
        for listener in self.__listeners:
            listener(self,event)
    
    def add_listener(self,listener):
        self.__listeners.append(listener)
        
    def remove_listener(self,listener):
        self.__listeners.remove(listener)

class AbstractPipelineEvent:
    """Something that happened to the pipeline and was indicated to the listeners
    """
    def event_type(self):
        raise NotImplementedError("AbstractPipelineEvent does not implement an event type")

class PipelineLoadedEvent(AbstractPipelineEvent):
    """Indicates that the pipeline has been (re)loaded
    
    """
    def event_type(self):
        return "PipelineLoaded"

class PipelineClearedEvent(AbstractPipelineEvent):
    """Indicates that all modules have been removed from the pipeline
    
    """
    def event_type(self):
        return "PipelineCleared"

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
class ModuleMovedPipelineEvent(AbstractPipelineEvent):
    """A module moved up or down
    
    """
    def __init__(self,module_num, direction):
        self.module_num = module_num
        self.direction = direction
    
    def event_type(self):
        return "Module moved"

class ModuleAddedPipelineEvent(AbstractPipelineEvent):
    """A module was added to the pipeline
    
    """
    def __init__(self,module_num):
        self.module_num = module_num
    
    def event_type(self):
        return "Module Added"
    
class ModuleRemovedPipelineEvent(AbstractPipelineEvent):
    """A module was removed from the pipeline
    
    """
    def __init__(self,module_num):
        self.module_num = module_num
        
    def event_type(self):
        return "Module deleted"

class RunExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during a pipeline run
    
    """
    def __init__(self,error,module):
        self.error     = error
        self.cancel_run = True
        self.module    = module
    
    def event_type(self):
        return "Pipeline run exception"

def AddHandlesImages(handles,image_set):
    """Add any images from the handles to the image set
    Generally, the handles have images added as they get returned from a Matlab module.
    You can use this to update the image set and capture them.
    """
    hpipeline = handles['Pipeline'][0,0]
    pipeline_fields = hpipeline.dtype.fields.keys()
    provider_set = set([x.name for x in image_set.providers])
    image_fields = set()
    crop_fields = set()
    for field in pipeline_fields:
        if field.startswith('CropMask'):
            crop_fields.add(field)
        elif field.startswith('Segmented') or field.startswith('UneditedSegmented') or field.startswith('SmallRemovedSegmented'):
            continue
        elif field.startswith('Pathname') or field.startswith('FileList') or field.startswith('Filename'):
            if not image_set.LegacyFields.has_key(field):
                value = hpipeline[field]
                if value.dtype.kind in ['U','S']:
                    image_set.legacy_fields[field] = value[0]
                else:
                    image_set.legacy_fields[field] = value
        elif not field in provider_set:
            image_fields.add(field)
    for field in image_fields:
        image = cellprofiler.image.Image()
        image.Image = hpipeline[field]
        crop_field = 'CropMask'+field
        if crop_field in crop_fields:
            image.Mask = hpipeline[crop_field]
        image_set.providers.append(cellprofiler.image.VanillaImageProvider(field,image))
    number_of_image_sets = int(handles[CURRENT][0,0][NUMBER_OF_IMAGE_SETS][0,0])
    if (not image_set.legacy_fields.has_key(NUMBER_OF_IMAGE_SETS)) or \
           number_of_image_sets < image_set.legacy_fields[NUMBER_OF_IMAGE_SETS]:
        image_set.legacy_fields[NUMBER_OF_IMAGE_SETS] = number_of_image_sets

def add_handles_objects(handles,object_set):
    """Add any objects from the handles to the object set
    You can use this to update the object set after calling a matlab module
    """
    hpipeline = handles['Pipeline'][0,0]
    pipeline_fields = hpipeline.dtype.fields.keys()
    objects_names = set(object_set.get_object_names())
    segmented_fields = set()
    unedited_segmented_fields = set()
    small_removed_segmented_fields = set()
    for field in pipeline_fields:
        if field.startswith('Segmented'):
            segmented_fields.add(field)
        elif field.startswith('UneditedSegmented'):
            unedited_segmented_fields.add(field)
        elif field.startswith('SmallRemovedSegmented'):
            small_removed_segmented_fields.add(field)
    for field in segmented_fields:
        object_name = field.replace('Segmented','')
        if object_name in object_set.get_object_names():
            continue
        objects = cellprofiler.objects.Objects()
        objects.segmented = hpipeline[field]
        unedited_field ='Unedited'+field
        small_removed_segmented_field = 'SmallRemoved'+field 
        if unedited_field in unedited_segmented_fields:
            objects.unedited_segmented = hpipeline[unedited_field]
        if small_removed_segmented_field in small_removed_segmented_fields:
            objects.small_removed_segmented = hpipeline[small_removed_segmented_field]
        object_set.add_objects(objects,object_name)

def add_handles_measurements(handles, measurements):
    """Get measurements made by Matlab and put them into our Python measurements object
    """
    measurement_fields = handles[MEASUREMENTS].dtype.fields.keys()
    set_being_analyzed = handles[CURRENT][0,0][SET_BEING_ANALYZED][0,0]
    for field in measurement_fields:
        object_measurements = handles[MEASUREMENTS][0,0][field][0,0]
        object_fields = object_measurements.dtype.fields.keys()
        for feature in object_fields:
            if not measurements.has_current_measurements(field,feature):
                value = object_measurements[feature][0,set_being_analyzed-1]
                if not isinstance(value,numpy.ndarray) or numpy.product(value.shape) > 0:
                    # It's either not a numpy array (it's a string) or it's not the empty numpy array
                    # so add it to the measurements
                    measurements.add_measurement(field,feature,value)

debug_matlab_run = None

def debug_matlab_run(value):
    global debug_matlab_run
    debug_matlab_run = value
     
def matlab_run(handles):
    """Run a Python module, given a Matlab handles structure
    """
    if debug_matlab_run:
        import wx.py
        import wx
        class MyPyCrustApp(wx.App):
            locals = {}
            def OnInit(self):
                wx.InitAllImageHandlers()
                frame = wx.Frame(None,-1,"MatlabRun explorer")
                sizer = wx.BoxSizer()
                frame.SetSizer(sizer)
                crust = wx.py.crust.Crust(frame,-1,locals=self.locals);
                sizer.Add(crust,1,wx.EXPAND)
                frame.Fit()
                self.SetTopWindow(frame)
                frame.Show()
                return 1

    if debug_matlab_run == u"init":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
        
    encapsulate_strings_in_arrays(handles)
    if debug_matlab_run == u"enc":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
    orig_handles = handles
    handles = handles[0,0]
    #
    # Get all the pieces you need to run a module:
    # pipeline, image set and set list, measurements and object_set
    #
    pipeline = Pipeline()
    pipeline.create_from_handles(handles)
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    measurements = cellprofiler.measurements.Measurements()
    object_set = cellprofiler.objects.ObjectSet()
    #
    # Get the values for the current image_set, making believe this is the first image set
    #
    add_handles_images(handles, image_set)
    add_handles_objects(handles,object_set)
    add_handles_measurements(handles, measurements)
    current_module = int(handles[CURRENT][0,0][CURRENT_MODULE_NUMBER][0])
    #
    # Get and run the module
    #
    module = pipeline.module(current_module)
    if debug_matlab_run == u"ready":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
    module.run(pipeline, image_set, object_set, measurements)
    #
    # Add everything to the handles
    #
    add_all_images(handles, image_set, object_set)
    add_all_measurements(handles, measurements)
    if debug_matlab_run == u"run":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()

    return orig_handles
    
if __name__ == "__main__":
    handles = scipy.io.matlab.loadmat('c:\\temp\\mh.mat',struct_as_record=True)['handles']
    handles[0,0][CURRENT][0,0][CURRENT_MODULE_NUMBER][0] = str(int(handles[0,0][CURRENT][0,0][CURRENT_MODULE_NUMBER][0])+1)
    matlab_run(handles)
