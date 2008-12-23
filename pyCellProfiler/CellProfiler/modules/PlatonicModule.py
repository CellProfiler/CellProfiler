""" PlatonicModules.py - module to load images from files
"""

import os
import re

import PIL.Image
import numpy
import matplotlib.image

from CellProfiler import cpmodule, cpimage, preferences, variable

# strings for choice variables
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGEXP = 'Text-Regular expressions'
MS_ORDER = 'Order'

FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi movies'
FF_OTHER_MOVIES = 'tif,tiff,flex movies'

DIR_DEFAULT_IMAGE = 'Default Image Directory'
DIR_DEFAULT_OUTPUT = 'Default Output Directory'
DIR_OTHER = 'Elsewhere...'

SB_GRAYSCALE = 'grayscale'
SB_BINARY = 'binary'

def default_cpimage_name(index):
    # the usual suspects
    names = ['DNA', 'Actin', 'Protein']
    if index < len(names):
        return names[index]
    return 'Channel%d'%(index+1)

class LoadImages(cpmodule.AbstractModule):
    """Load images from files.  This is the help text that will be displayed
       to the user.
    """
    def __init__(self):
        super(LoadImages, self).__init__()
        self.module_name = "LoadImages"
        
        # Settings
        self.file_types = variable.Choice('What type of files are you loading?',[FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES])
        self.match_method = variable.Choice('How do you want to load these files?', [MS_EXACT_MATCH, MS_REGEXP, MS_ORDER])
        self.match_exclude = variable.Text('If you want to exclude certain files, type the text that the excluded images have in common', '')
        self.order_group_size = variable.Integer('How many images are there in each group?', 3)
        self.descend_subdirectories = variable.Binary('Analyze all subfolders within the selected folder?', False)
     
        # Settings for each CPimage
        self.images_common_text = [variable.Text('Type the text that these images have in common', 'DAPI')]
        self.images_order_position = [variable.Integer('What is the position of this image in each group', 1)]
        self.image_names = [variable.ImageName('What do you want to call this image in CellProfiler?', default_cpimage_names(0))]
        self.remove_images = [variable.DoSomething('Remove this image...', self.remove_imagecb, 0)]
        
        # Add another image
        self.add_image = variable.DoSometings('Add another image...', self.add_imagecb)
        
        # Location settings
        self.location = variables.Text('Where are the images located?',
                                        [DIR_DEFAULT_IMAGE, DIR_DEFAULT_OUTPUT, DIR_OTHER])
        self.location_other = variables.DirectoryPath("Where are the images located?", '')

    def add_imagecb(self):
            'Adds another image to the variables'
            img_index = len(self.images_order_position)
            self.images_common_text += [variable.Text('Type the text that these images have in common', '')]
            self.images_order_position += [variable.Integer('What is the position of this image in each group', img_index+1)]
            self.image_names += [variable.ImageName('What do you want to call this image in CellProfiler?', default_cpimage_name(img_index))]
            self.remove_images += [variable.DoSomething('Remove this image...', self.remove_imagecb, img_index)]

    def remove_imagecb(self, index):
            'Remove an image from the variables'
            del self.images_common_text[index]
            del self.images_order_position[index]
            del self.image_names[index]
            del self.remove_images[index]

    def visible_variables(self):
        varlist = [self.file_types, self.match_method]
        if self.match_method == MS_EXACT_MATCH:
            varlist += [self.match_exclude, self.images_common_text]
        elif self.match_method == MS_REGEXP:
            varlist += [self.order_group_size]
        varlist += [self.descend_subdirectories]
        
        # per image settings
        if self.match_method == MS_EXACT_MATCH:
            for ctext, imname, rm in zip(self.images_common_text, self.image_names, self.remove_images):
                varlist += [ctext, imname, rm]
        else:
            for pos, imname, rm in zip(self.images_order_position, self.image_names, self.remove_images):
                varlist += [pos, imname, rm]
                
        varlist += [self.add_image]
        varlist += [self.location]
        if self.location == DIR_OTHER:
            varlist += [self.location_other]
        return varlist
        
    # Move this down somehere
    def directory_path(self):
        if self.location == 'Default image folder':
            return xxx_the_default_image_folder
        elif self.location == 'Default output filter':
            return xxx_the_default_output_folder
        else:
            return self.location_other

    def upgrade(self,variable_revision_number):
        """Rewrite the variables in the module to upgrade it to its
           current revision number
        """
        if variable_revision_number != self.variable_revision_number:
            raise NotImplementedError("Cannot read version %d of %s"%(
                variable_revision_number, self.module_name))
    
    variable_revision_number = 4
    
    def WriteToHandles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def WriteToText(self,file):
        """Write the module's state, informally, to a text file
        """

    def PrepareRun(self, pipeline, image_set_list):
        """Set up all of the image providers inside the image_set_list
        """
        if self.LoadMovies():
            raise NotImplementedError("Movies aren't implemented yet.")
        
        files = self.CollectFiles()
        if len(files) == 0:
            raise ValueError("there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well)")
        
        #Deal out the image filenames to a list of lists.
        image_names = self.ImageNameVars()
        list_of_lists = [[] for x in image_names]
        for x in files:
            list_of_lists[x[1]].append(x[0])
        
        image_set_count = len(list_of_lists[0])
        for x,name in zip(list_of_lists[1:],image_names):
            if len(x) != image_set_count:
                raise RuntimeError("Image %s has %d files, but image %s has %d files"%(image_names[0],image_set_count,name,len(x)))
        list_of_lists = numpy.array(list_of_lists)
        root = self.ImageDirectory()
        for i in range(0,image_set_count):
            image_set = image_set_list.GetImageSet(i)
            providers = [LoadImagesImageProvider(name,root,file) for name,file in zip(image_names, list_of_lists[:,i])]
            image_set.Providers.extend(providers)
        
    def Run(self,pipeline,image_set,object_set,measurements, frame):
        """Run the module - add the measurements
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        for provider in image_set.Providers:
            if isinstance(provider,LoadImagesImageProvider):
                filename = provider.GetFilename()
                path = provider.GetPathname()
                name = provider.Name()
                measurements.AddMeasurement('Image','FileName_'+name, filename)
                measurements.AddMeasurement('Image','PathName_'+name, path)

    def GetCategories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return ['Image']
      
    def GetMeasurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image':
            return ['FileName','PathName']
        return []
    
    def GetMeasurementImages(self,pipeline,object_name,category,measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []
    
    def GetMeasurementScales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
    
    def Category(self):
        return "File Processing"

    
    def LoadImages(self):
        """Return true if we're loading images
        """
        return self.Variables()[FILE_FORMAT_VAR-1].Value==FF_INDIVIDUAL_IMAGES
    
    def LoadMovies(self):
        """Return true if we're loading movies
        """
        return self.Variables()[FILE_FORMAT_VAR-1].Value !=FF_INDIVIDUAL_IMAGES
    
    def LoadChoice(self):
        """Return the way to match against files: MS_EXACT_MATCH, MS_REGULAR_EXPRESSIONS or MS_ORDER
        """
        return self.Variables()[MATCH_STYLE_VAR-1].Value
    
    def AnalyzeSubDirs(self):
        """Return True if we should analyze subdirectories in addition to the root image directory
        """
        return self.Variables()[ANALYZE_SUB_DIR_VAR-1].IsYes
    
    def CollectFiles(self, dirs=[]):
        """Collect the files that match the filter criteria
        
        Collect the files that match the filter criteria, starting at the image directory
        and descending downward if AnalyzeSubDirs allows it.
        dirs - a list of subdirectories connecting the image directory to the
               directory currently being searched
        Returns a list of three-tuples where the first element of the tuple is the path
        from the root directory, including the file name, the second element is the
        index within the image variables (e.g. ImageNameVars).
        """
        path = reduce(os.path.join, dirs, self.ImageDirectory() )
        files = os.listdir(path)
        files.sort()
        isdir = lambda x: os.path.isdir(os.path.join(path,x))
        isfile = lambda x: os.path.isfile(os.path.join(path,x))
        subdirs = filter(isdir, files)
        files = filter(isfile,files)
        path_to = (len(dirs) and reduce(os.path.join, dirs)) or ''
        files = [(os.path.join(path_to,file), self.FilterFilename(file)) for file in files]
        files = filter(lambda x: x[1] != None,files)
        if self.AnalyzeSubDirs():
            for dir in subdirs:
                files += self.CollectFiles(dirs + [dir])
        return files
        
    def ImageDirectory(self):
        """Return the image directory
        """
        Pathname = self.Variables()[PATHNAME_VAR-1].Value
        if Pathname[0] == '.':
            if len(Pathname) == 1:
                return CellProfiler.Preferences.GetDefaultImageDirectory()
            else:
                #% If the pathname start with '.', interpret it relative to
                #% the default image dir.
                return os.path.join(CellProfiler.Preferences.GetDefaultImageDirectory(),Pathname[2:])
        elif Pathname == '&':
            if length(Pathname) == 1:
                return CellProfiler.Preferences.GetDefaultOutputDirectory()
            else:
                #% If the pathname start with '&', interpret it relative to
                #% the default output directory
                return os.path.join(CellProfiler.Preferences.GetDefaultOutputDirectory(),Pathname[2:])
        return Pathname
    
    def ImageNameVars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        result = [self.Variables()[FIRST_IMAGE_VAR].Value ]
        for i in range(1,MAX_IMAGE_COUNT):
            value = self.Variables()[FIRST_IMAGE_VAR+i*2].Value
            if value == CellProfiler.Variable.DO_NOT_USE:
                break
            result += [value]
        return result
        
    def TextToFindVars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        result = [self.Variables()[FIRST_IMAGE_VAR-1].Value ]
        for i in range(1,MAX_IMAGE_COUNT):
            value = self.Variables()[FIRST_IMAGE_VAR+i*2-1].Value
            if value == CellProfiler.Variable.DO_NOT_USE:
                break
            result += [value]
        return result
    
    def TextToExclude(self):
        """Return the text to match against the file name to exclude it from the set
        """
        return self.Variables()[TEXT_TO_EXCLUDE_VAR-1].Value
    
    def FilterFilename(self, filename):
        """Returns either None or the index of the match variable
        """
        if self.TextToExclude() != CellProfiler.Variable.DO_NOT_USE and \
            filename.find(self.TextToExclude()) >=0:
            return None
        if self.LoadChoice() == MS_EXACT_MATCH:
            ttfs = self.TextToFindVars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if filename.find(ttf) >=0:
                    return i
        elif self.LoadChoice() == MS_REGULAR_EXPRESSIONS:
            ttfs = self.TextToFindVars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if re.search(ttf, filename):
                    return i
        else:
            raise NotImplementedError("Load by order not implemented")
        return None

class LoadImagesImageProvider(cpimage.AbstractImageProvider):
    """Provide an image by filename, loading the file as it is requested
    """
    def __init__(self,name,pathname,filename):
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
    
    def ProvideImage(self, image_set):
        """Load an image from a pathname
        """
        img = PIL.Image.open(self.GetFullName())
        img = matplotlib.image.pil_to_array(img)
        return CellProfiler.Image.Image(img)
    
    def Name(self):
        return self.__name
    
    def GetPathname(self):
        return self.__pathname
    
    def GetFilename(self):
        return self.__filename
    
    def GetFullName(self):
        return os.path.join(self.GetPathname(),self.GetFilename())
