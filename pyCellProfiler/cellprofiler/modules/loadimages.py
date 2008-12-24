""" LoadImages.py - module to load images from files
"""
__version__="$Revision: 1$"
import os
import re
import numpy

import PIL.Image
import matplotlib.image

import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.preferences
import cellprofiler.variable

MATCH_STYLE_VAR = 1
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGULAR_EXPRESSIONS = 'Text-Regular expressions'
MS_ORDER = 'Order'

FIRST_IMAGE_VAR = 2
MAX_IMAGE_COUNT = 4
IMAGES_PER_SET_VAR = 10
TEXT_TO_EXCLUDE_VAR = 11
FILE_FORMAT_VAR = 12
FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi movies'
FF_OTHER_MOVIES = 'tif,tiff,flex movies'
ANALYZE_SUB_DIR_VAR = 13
PATHNAME_VAR = 14
SAVE_AS_BINARY_VAR = 15
SB_GRAYSCALE = 'grayscale'
SB_BINARY = 'binary'

class LoadImages(cellprofiler.cpmodule.AbstractModule):
    """Load images from files
    """
    def __init__(self):
        cellprofiler.cpmodule.AbstractModule.__init__(self)
        self.set_module_name("LoadImages")
        self.__annotations = None
    
    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the variables in the module to upgrade it to its current revision number
        
        """
        if variable_revision_number != self.variable_revision_number():
            raise NotImplementedError("Cannot read version %d of LoadImages"%(variable_revision_number))
    
    def get_help(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    def variable_revision_number(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        return 4
    
    def annotations(self):
        """Return the variable annotations.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the CellProfiler.Variable.Annotation
        class.
        """ 
        if not self.__annotations:
            annotations = cellprofiler.variable.choice_popup_annotation(MATCH_STYLE_VAR, 
                                                                      'How do you want to load these files?', 
                                                                      [MS_EXACT_MATCH, MS_REGULAR_EXPRESSIONS,MS_ORDER])
            annotations += cellprofiler.variable.edit_box_annotation(FIRST_IMAGE_VAR, 'Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)','DAPI')
            annotations += cellprofiler.variable.indep_group_annotation(FIRST_IMAGE_VAR+1, 'What do you want to call these images within CellProfiler?', 'imagegroup','OrigBlue')
            for i in range(1,MAX_IMAGE_COUNT):
                text_to_find_var = i*2+FIRST_IMAGE_VAR
                image_name_var   = text_to_find_var + 1
                annotations += cellprofiler.variable.edit_box_annotation(text_to_find_var, 'Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option). Type "Do not use" to ignore:')
                annotations += cellprofiler.variable.indep_group_annotation(image_name_var, 'What do you want to call these images within CellProfiler? (Type "Do not use" to ignore)', 'imagegroup')
            
            annotations += cellprofiler.variable.edit_box_annotation(IMAGES_PER_SET_VAR, 'If using ORDER, how many images are there in each group (i.e. each field of view)?','3')
            annotations += cellprofiler.variable.edit_box_annotation(TEXT_TO_EXCLUDE_VAR, 'If you want to exclude files, type the text that the excluded images have in common (for TEXT options). Type "Do not use" to ignore.')
            annotations += cellprofiler.variable.choice_popup_annotation(FILE_FORMAT_VAR, 'What type of files are you loading?',
                                                                       [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES,FF_AVI_MOVIES,FF_OTHER_MOVIES])
            annotations += cellprofiler.variable.checkbox_annotation(ANALYZE_SUB_DIR_VAR, 'Analyze all subfolders within the selected folder?')
            annotations += cellprofiler.variable.edit_box_annotation(PATHNAME_VAR, 'Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder or ampersand (&) for default output folder.', '.')
            annotations += cellprofiler.variable.choice_popup_annotation(SAVE_AS_BINARY_VAR,'If the images you are loading are binary (black/white only), in what format do you want to store them?', [SB_GRAYSCALE,SB_BINARY])
            self.__annotations = annotations
        return self.__annotations
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """

    def prepare_run(self, pipeline, image_set_list):
        """Set up all of the image providers inside the image_set_list
        """
        if self.load_movies():
            raise NotImplementedError("Movies aren't implemented yet.")
        
        files = self.collect_files()
        if len(files) == 0:
            raise ValueError("there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well)")
        
        #Deal out the image filenames to a list of lists.
        image_names = self.image_name_vars()
        list_of_lists = [[] for x in image_names]
        for x in files:
            list_of_lists[x[1]].append(x[0])
        
        image_set_count = len(list_of_lists[0])
        for x,name in zip(list_of_lists[1:],image_names):
            if len(x) != image_set_count:
                raise RuntimeError("Image %s has %d files, but image %s has %d files"%(image_names[0],image_set_count,name,len(x)))
        list_of_lists = numpy.array(list_of_lists)
        root = self.image_directory()
        for i in range(0,image_set_count):
            image_set = image_set_list.get_image_set(i)
            providers = [LoadImagesImageProvider(name,root,file) for name,file in zip(image_names, list_of_lists[:,i])]
            image_set.providers.extend(providers)
        
    def run(self,pipeline,image_set,object_set,measurements, frame):
        """Run the module - add the measurements
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        for provider in image_set.providers:
            if isinstance(provider,LoadImagesImageProvider):
                filename = provider.get_filename()
                path = provider.get_pathname()
                name = provider.name
                measurements.add_measurement('Image','FileName_'+name, filename)
                measurements.add_measurement('Image','PathName_'+name, path)

    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return ['Image']
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image':
            return ['FileName','PathName']
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
        return "File Processing"

    
    def load_images(self):
        """Return true if we're loading images
        """
        return self.variables()[FILE_FORMAT_VAR-1].value==FF_INDIVIDUAL_IMAGES
    
    def load_movies(self):
        """Return true if we're loading movies
        """
        return self.variables()[FILE_FORMAT_VAR-1].value !=FF_INDIVIDUAL_IMAGES
    
    def load_choice(self):
        """Return the way to match against files: MS_EXACT_MATCH, MS_REGULAR_EXPRESSIONS or MS_ORDER
        """
        return self.variables()[MATCH_STYLE_VAR-1].value
    
    def analyze_sub_dirs(self):
        """Return True if we should analyze subdirectories in addition to the root image directory
        """
        return self.variables()[ANALYZE_SUB_DIR_VAR-1].is_yes
    
    def collect_files(self, dirs=[]):
        """Collect the files that match the filter criteria
        
        Collect the files that match the filter criteria, starting at the image directory
        and descending downward if AnalyzeSubDirs allows it.
        dirs - a list of subdirectories connecting the image directory to the
               directory currently being searched
        Returns a list of three-tuples where the first element of the tuple is the path
        from the root directory, including the file name, the second element is the
        index within the image variables (e.g. ImageNameVars).
        """
        path = reduce(os.path.join, dirs, self.image_directory() )
        files = os.listdir(path)
        files.sort()
        isdir = lambda x: os.path.isdir(os.path.join(path,x))
        isfile = lambda x: os.path.isfile(os.path.join(path,x))
        subdirs = filter(isdir, files)
        files = filter(isfile,files)
        path_to = (len(dirs) and reduce(os.path.join, dirs)) or ''
        files = [(os.path.join(path_to,file), self.filter_filename(file)) for file in files]
        files = filter(lambda x: x[1] != None,files)
        if self.analyze_sub_dirs():
            for dir in subdirs:
                files += self.collect_files(dirs + [dir])
        return files
        
    def image_directory(self):
        """Return the image directory
        """
        pathname = self.variables()[PATHNAME_VAR-1].value
        if pathname[0] == '.':
            if len(pathname) == 1:
                return cellprofiler.preferences.get_default_image_directory()
            else:
                #% If the pathname start with '.', interpret it relative to
                #% the default image dir.
                return os.path.join(cellprofiler.preferences.get_default_image_directory(),pathname[2:])
        elif pathname == '&':
            if length(pathname) == 1:
                return cellprofiler.preferences.get_default_output_directory()
            else:
                #% If the pathname start with '&', interpret it relative to
                #% the default output directory
                return os.path.join(cellprofiler.preferences.get_default_output_directory(),pathname[2:])
        return Pathname
    
    def image_name_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        result = [self.variables()[FIRST_IMAGE_VAR].value ]
        for i in range(1,MAX_IMAGE_COUNT):
            value = self.variables()[FIRST_IMAGE_VAR+i*2].value
            if value == cellprofiler.variable.DO_NOT_USE:
                break
            result += [value]
        return result
        
    def text_to_find_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        result = [self.variables()[FIRST_IMAGE_VAR-1].value ]
        for i in range(1,MAX_IMAGE_COUNT):
            value = self.variables()[FIRST_IMAGE_VAR+i*2-1].value
            if value == cellprofiler.variable.DO_NOT_USE:
                break
            result += [value]
        return result
    
    def text_to_exclude(self):
        """Return the text to match against the file name to exclude it from the set
        """
        return self.variables()[TEXT_TO_EXCLUDE_VAR-1].value
    
    def filter_filename(self, filename):
        """Returns either None or the index of the match variable
        """
        if self.text_to_exclude() != cellprofiler.variable.DO_NOT_USE and \
            filename.find(self.text_to_exclude()) >=0:
            return None
        if self.load_choice() == MS_EXACT_MATCH:
            ttfs = self.text_to_find_vars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if filename.find(ttf) >=0:
                    return i
        elif self.load_choice() == MS_REGULAR_EXPRESSIONS:
            ttfs = self.text_to_find_vars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if re.search(ttf, filename):
                    return i
        else:
            raise NotImplementedError("Load by order not implemented")
        return None

class LoadImagesImageProvider(cellprofiler.cpimage.AbstractImageProvider):
    """Provide an image by filename, loading the file as it is requested
    """
    def __init__(self,name,pathname,filename):
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
    
    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        img = PIL.Image.open(self.get_full_name())
        img = matplotlib.image.pil_to_array(img)
        return cellprofiler.cpimage.Image(img)
    
    def name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return self.__filename
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.get_filename())
