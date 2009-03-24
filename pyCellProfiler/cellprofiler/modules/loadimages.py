""" LoadImages.py - module to load images from files

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import re

import PIL.Image
import numpy
import matplotlib.image
import scipy.io.mio
import uuid

import cellprofiler.cpmodule as cpmodule
import cellprofiler.cpimage as cpimage
import cellprofiler.preferences as preferences
import cellprofiler.settings as cps

# strings for choice variables
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGEXP = 'Text-Regular expressions'
MS_ORDER = 'Order'

FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi movies'
FF_OTHER_MOVIES = 'tif,tiff,flex movies'
try:
    import ffmpeg
    FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
except ImportError:
    FF = [FF_INDIVIDUAL_IMAGES]

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

class LoadImages(cpmodule.CPModule):
    """Load images from files.  This is the help text that will be displayed
       to the user.
    """
    def create_settings(self):
        self.module_name = "LoadImages"
        
        # Settings
        self.file_types = cps.Choice('What type of files are you loading?', FF)
        self.match_method = cps.Choice('How do you want to load these files?', [MS_EXACT_MATCH, MS_REGEXP, MS_ORDER])
        self.match_exclude = cps.Text('If you want to exclude certain files, type the text that the excluded images have in common', cps.DO_NOT_USE)
        self.order_group_size = cps.Integer('How many images are there in each group?', 3)
        self.descend_subdirectories = cps.Binary('Analyze all subfolders within the selected folder?', False)
        self.check_images = cps.Binary('Do you want to check image sets for missing or duplicate files?',True)
     
        # Settings for each CPimage
        self.image_keys = [ uuid.uuid1() ]
        self.images_common_text = [cps.Text('Type the text that these images have in common', 'DAPI')]
        self.images_order_position = [cps.Integer('What is the position of this image in each group', 1)]
        self.image_names = [cps.FileImageNameProvider('What do you want to call this image in CellProfiler?', default_cpimage_name(0))]
        self.remove_images = [cps.DoSomething('Remove this image...','Remove', self.remove_imagecb, self.image_keys[0])]
        
        # Add another image
        self.add_image = cps.DoSomething('Add another image...','Add', self.add_imagecb)
        
        # Location settings
        self.location = cps.CustomChoice('Where are the images located?',
                                        [DIR_DEFAULT_IMAGE, DIR_DEFAULT_OUTPUT, DIR_OTHER])
        self.location_other = cps.DirectoryPath("Where are the images located?", '')

    def add_imagecb(self):
            'Adds another image to the settings'
            img_index = len(self.images_order_position)
            new_uuid = uuid.uuid1()
            self.image_keys += [ new_uuid ]
            self.images_common_text += [cps.Text('Type the text that these images have in common', '')]
            self.images_order_position += [cps.Integer('What is the position of this image in each group', img_index+1)]
            self.image_names += [cps.FileImageNameProvider('What do you want to call this image in CellProfiler?', default_cpimage_name(img_index))]
            self.remove_images += [cps.DoSomething('Remove this image...', 'Remove',self.remove_imagecb, new_uuid)]

    def remove_imagecb(self, id):
            'Remove an image from the settings'
            index = self.image_keys.index(id)
            del self.image_keys[index]
            del self.images_common_text[index]
            del self.images_order_position[index]
            del self.image_names[index]
            del self.remove_images[index]

    def visible_settings(self):
        varlist = [self.file_types, self.match_method]
        if self.match_method == MS_EXACT_MATCH:
            varlist += [self.match_exclude]
        elif self.match_method == MS_ORDER:
            varlist += [self.order_group_size]
        varlist += [self.descend_subdirectories]
        
        if len(self.images_common_text) > 1:
            varlist += [self.check_images]
        
        # per image settings
        if self.match_method != MS_ORDER:
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
    
    #
    # Slots for storing settings in the array
    #
    SLOT_FILE_TYPE = 0
    SLOT_MATCH_METHOD = 1
    SLOT_ORDER_GROUP_SIZE = 2
    SLOT_MATCH_EXCLUDE = 3
    SLOT_DESCEND_SUBDIRECTORIES = 4
    SLOT_LOCATION = 5
    SLOT_LOCATION_OTHER = 6
    SLOT_CHECK_IMAGES = 7
    SLOT_FIRST_IMAGE = 8
    SLOT_OFFSET_COMMON_TEXT = 0
    SLOT_OFFSET_IMAGE_NAME = 1
    SLOT_OFFSET_ORDER_POSITION = 2
    SLOT_IMAGE_FIELD_COUNT = 3
    def settings(self):
        """Return the settings array in a consistent order"""
        varlist = range(self.SLOT_FIRST_IMAGE + \
                        self.SLOT_IMAGE_FIELD_COUNT * len(self.image_names))
        varlist[self.SLOT_FILE_TYPE]              = self.file_types
        varlist[self.SLOT_MATCH_METHOD]           = self.match_method
        varlist[self.SLOT_ORDER_GROUP_SIZE]       = self.order_group_size
        varlist[self.SLOT_MATCH_EXCLUDE]          = self.match_exclude
        varlist[self.SLOT_DESCEND_SUBDIRECTORIES] = self.descend_subdirectories
        varlist[self.SLOT_LOCATION]               = self.location
        varlist[self.SLOT_LOCATION_OTHER]         = self.location_other
        varlist[self.SLOT_CHECK_IMAGES]           = self.check_images
        for i in range(len(self.image_names)):
            ioff = i*self.SLOT_IMAGE_FIELD_COUNT + self.SLOT_FIRST_IMAGE
            varlist[ioff+self.SLOT_OFFSET_COMMON_TEXT] = \
                self.images_common_text[i]
            varlist[ioff+self.SLOT_OFFSET_IMAGE_NAME] = \
                self.image_names[i]
            varlist[ioff+self.SLOT_OFFSET_ORDER_POSITION] = \
                self.images_order_position[i]
        return varlist
    
    def set_setting_values(self,setting_values,variable_revision_number,module_name):
        """Interpret the setting values as saved by the given revision number
        """
        if variable_revision_number == 1 and module_name == 'LoadImages':
            setting_values,variable_revision_number = self.upgrade_1_to_2(setting_values)
        if variable_revision_number == 2 and module_name == 'LoadImages':
            setting_values,variable_revision_number = self.upgrade_2_to_3(setting_values)
        if variable_revision_number == 3 and module_name == 'LoadImages':
            setting_values,variable_revision_number = self.upgrade_3_to_4(setting_values)
        if variable_revision_number == 4 and module_name == 'LoadImages':
            setting_values,variable_revision_number = self.upgrade_4_to_5(setting_values)
        if variable_revision_number == 5 and module_name == 'LoadImages':
            setting_values,variable_revision_number = self.upgrade_5_to_new_1(setting_values)
            module_name = self.module_class()

        if variable_revision_number != self.variable_revision_number or \
           module_name != self.module_class():
            raise NotImplementedError("Cannot read version %d of %s"%(
                variable_revision_number, self.module_name))
        #
        # Figure out how many images are in the saved settings - make sure
        # the array size matches the incoming #
        #
        assert (len(setting_values) - self.SLOT_FIRST_IMAGE) % self.SLOT_IMAGE_FIELD_COUNT == 0
        image_count = (len(setting_values) - self.SLOT_FIRST_IMAGE) / self.SLOT_IMAGE_FIELD_COUNT
        while len(self.image_names) > image_count:
            self.remove_imagecb(self.image_keys[0])
        while len(self.image_names) < image_count:
            self.add_imagecb()
        super(LoadImages,self).set_setting_values(setting_values, variable_revision_number, module_name)
    
    def upgrade_1_to_2(self, setting_values):
        """Upgrade rev 1 LoadImages to rev 2
        
        Handle movie formats new to rev 2
        """
        new_values = list(setting_values[:10])
        image_or_movie =  setting_values[10]
        if image_or_movie == 'Image':
            new_values.append('individual images')
        elif setting_values[11] == 'avi':
            new_values.append('avi movies')
        elif setting_values[11] == 'stk':
            new_values.append('stk movies')
        else:
            raise ValueError('Unhandled movie type: %s'%(setting_values[11]))
        new_values.extend(setting_values[11:])
        return (new_values,2)
    
    def upgrade_2_to_3(self, setting_values):
        """Added binary/grayscale question"""
        new_values = list(setting_values)
        new_values.append('grayscale')
        new_values.append('')
        return (new_values,3)
    
    def upgrade_3_to_4(self, setting_values):
        """Added text exclusion at slot # 10"""
        new_values = list(setting_values)
        new_values.insert(10,cps.DO_NOT_USE)
        return (new_values,4)
    
    def upgrade_4_to_5(self, setting_values):
        new_values = list(setting_values)
        new_values.append(cps.NO)
        return (new_values,5)
    
    def upgrade_5_to_new_1(self,setting_values):
        """Take the old LoadImages values and put them in the correct slots"""
        new_values = range(self.SLOT_FIRST_IMAGE)
        new_values[self.SLOT_FILE_TYPE]              = setting_values[11]
        new_values[self.SLOT_MATCH_METHOD]           = setting_values[0]
        new_values[self.SLOT_ORDER_GROUP_SIZE]       = setting_values[9]
        new_values[self.SLOT_MATCH_EXCLUDE]          = setting_values[10]
        new_values[self.SLOT_DESCEND_SUBDIRECTORIES] = setting_values[12]
        new_values[self.SLOT_CHECK_IMAGES]           = setting_values[16]
        loc = setting_values[13]
        if loc == '.':
            new_values[self.SLOT_LOCATION]           = DIR_DEFAULT_IMAGE
        elif loc == '&':
            new_values[self.SLOT_LOCATION]           = DIR_DEFAULT_OUTPUT
        else:
            new_values[self.SLOT_LOCATION]           = DIR_OTHER 
        new_values[self.SLOT_LOCATION_OTHER]         = loc 
        for i in range(0,4):
            text_to_find = setting_values[i*2+1]
            image_name = setting_values[i*2+2]
            if text_to_find == cps.DO_NOT_USE or \
               image_name == cps.DO_NOT_USE or\
               text_to_find == '/' or\
               image_name == '/' or\
               text_to_find == '\\' or\
               image_name == '\\':
                break
            new_values.extend([text_to_find,image_name,text_to_find])
        return (new_values,1)
    
    variable_revision_number = 1
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """

    def prepare_run(self, pipeline, image_set_list, frame):
        """Set up all of the image providers inside the image_set_list
        """
        if self.load_movies():
            self.prepare_run_of_movies(pipeline,image_set_list)
        else:
            self.prepare_run_of_images(pipeline, image_set_list)
        return True
    
    def prepare_run_of_images(self, pipeline, image_set_list):
        """Set up image providers for image files"""
        files = self.collect_files()
        if len(files) == 0:
            raise ValueError("there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well)")
        
        #Deal out the image filenames to a list of lists.
        image_names = self.image_name_vars()
        list_of_lists = [[] for x in image_names]
        for pathname,image_index in files:
            list_of_lists[image_index].append(pathname)
        
        image_set_count = len(list_of_lists[0])
        for x,name in zip(list_of_lists[1:],image_names):
            if len(x) != image_set_count:
                raise RuntimeError("Image %s has %d files, but image %s has %d files"%(image_names[0],image_set_count,name.value,len(x)))
        list_of_lists = numpy.array(list_of_lists)
        root = self.image_directory()
        for i in range(0,image_set_count):
            image_set = image_set_list.get_image_set(i)
            providers = [LoadImagesImageProvider(name.value,root,file) for name,file in zip(image_names, list_of_lists[:,i])]
            image_set.providers.extend(providers)
        for name in image_names:
            image_set_list.legacy_fields['Pathname%s'%(name.value)]=root
    
    def prepare_run_of_movies(self, pipeline, image_set_list):
        """Set up image providers for movie files"""
        files = self.collect_files()
        if len(files) == 0:
            raise ValueError("there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well)")
        
        root = self.image_directory()
        image_names = self.image_name_vars()
        #
        # The list of lists has one list per image type. Each per-image type
        # list is composed of tuples of pathname and frame #
        #
        list_of_lists = [[] for x in image_names]
        for pathname,image_index in files:
            frame_count = self.get_frame_count(os.path.join(root,pathname))
            # the video stream to be used when reading the movie
            video_stream = pyffmpeg.VideoStream()
            for i in range(frame_count):
                list_of_lists[image_index].append((pathname,i,video_stream))
        image_set_count = len(list_of_lists[0])
        for x,name in zip(list_of_lists[1:],image_names):
            if len(x) != image_set_count:
                raise RuntimeError("Image %s has %d frames, but image %s has %d frames"%(image_names[0],image_set_count,name.value,len(x)))
        list_of_lists = numpy.array(list_of_lists,dtype=object)
        for i in range(0,image_set_count):
            image_set = image_set_list.get_image_set(i)
            providers = [LoadImagesMovieFrameProvider(name.value, root, file, int(frame), video_stream)
                         for name,(file,frame,video_stream) in zip(image_names,list_of_lists[:,i])]
            image_set.providers.extend(providers)
        for name in image_names:
            image_set_list.legacy_fields['Pathname%s'%(name.value)]=root
        
    def run(self,workspace):
        """Run the module - add the measurements
        
        """
        for provider in workspace.image_set.providers:
            if isinstance(provider,LoadImagesImageProvider):
                filename = provider.get_filename()
                path = provider.get_pathname()
                name = provider.name
                workspace.measurements.add_measurement('Image','FileName_'+name, filename)
                workspace.measurements.add_measurement('Image','PathName_'+name, path)

    def get_frame_count(self, pathname):
        """Return the # of frames in a movie"""
        if self.file_types in (FF_AVI_MOVIES,FF_OTHER_MOVIES):
            print "opening %s"%(pathname)
            video_stream = pyffmpeg.VideoStream()
            video_stream.open("file:%s"%(pathname))
            frame_count = video_stream.frame_count()
            video_stream.close()
            print "Found %d frames"%(frame_count)
            return frame_count
        raise NotImplementedError("get_frame_count not implemented for %s"%(self.file_types))
            
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
    
    category = "File Processing"

    
    def load_images(self):
        """Return true if we're loading images
        """
        return self.file_types == FF_INDIVIDUAL_IMAGES
    
    def load_movies(self):
        """Return true if we're loading movies
        """
        return self.file_types != FF_INDIVIDUAL_IMAGES
    
    def load_choice(self):
        """Return the way to match against files: MS_EXACT_MATCH, MS_REGULAR_EXPRESSIONS or MS_ORDER
        """
        return self.match_method.value
    
    def analyze_sub_dirs(self):
        """Return True if we should analyze subdirectories in addition to the root image directory
        """
        return self.descend_subdirectories.value
    
    def collect_files(self, dirs=[]):
        """Collect the files that match the filter criteria
        
        Collect the files that match the filter criteria, starting at the image directory
        and descending downward if AnalyzeSubDirs allows it.
        dirs - a list of subdirectories connecting the image directory to the
               directory currently being searched
        Returns a list of two-tuples where the first element of the tuple is the path
        from the root directory, including the file name, the second element is the
        index within the image settings (e.g. ImageNameVars).
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
        if self.location == DIR_DEFAULT_IMAGE:
            return preferences.get_default_image_directory()
        elif self.location == DIR_DEFAULT_OUTPUT:
            return preferences.get_default_output_directory()
        elif self.location_other.value[0] == '.':
            return os.path.join(preferences.get_default_image_directory(),self.location_other.value[1:])
        return self.location_other.value
    
    def image_name_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        return self.image_names
        
    def text_to_find_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        return self.images_common_text
    
    def text_to_exclude(self):
        """Return the text to match against the file name to exclude it from the set
        """
        return self.match_exclude.value
    
    def filter_filename(self, filename):
        """Returns either None or the index of the match setting
        """
        if self.text_to_exclude() != cps.DO_NOT_USE and \
            filename.find(self.text_to_exclude()) >=0:
            return None
        if self.load_choice() == MS_EXACT_MATCH:
            ttfs = self.text_to_find_vars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if filename.find(ttf.value) >=0:
                    return i
        elif self.load_choice() == MS_REGEXP:
            ttfs = self.text_to_find_vars()
            for i,ttf in zip(range(0,len(ttfs)),ttfs):
                if re.search(ttf.value, filename):
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
    
    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        if self.__filename.endswith(".mat"):
            imgdata = scipy.io.mio.loadmat(self.get_full_name())
            return cpimage.Image(imgdata["Image"])
        img = PIL.Image.open(self.get_full_name())
        # There's an apparent bug in the PIL library that causes
        # images to be loaded upside-down. At best, load and save have opposite
        # orientations; in other words, if you load an image and then save it
        # the resulting saved image will be upside-down
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if img.mode=='I;16':
            # 16-bit image
            imgdata = numpy.array(img.getdata(),numpy.uint16)
            img = imgdata.reshape(img.size)
            img = img.astype(float) / 65535.0
        else:
            img = matplotlib.image.pil_to_array(img)
        return cpimage.Image(img)
    
    def get_name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return self.__filename
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.get_filename())

class LoadImagesMovieFrameProvider(cpimage.AbstractImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """
    def __init__(self,name,pathname,filename,frame,video_stream):
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__frame    = frame
        self.__video_stream = video_stream
    
    def provide_image(self, image_set):
        """Load an image from a movie frame
        """
        if (self.__frame == 0):
            # open the stream on the first frame
            pathname = self.get_full_name()
            self.__video_stream.open("file:%s"%(pathname))
        img = self.__video_stream.GetNextFrame()
        if self.__frame + 1 == self.__video_stream.frame_count():
            # Close the stream on the last frame
            # self.__video_stream.close()
            pass

        # There's an apparent bug in the PIL library that causes
        # images to be loaded upside-down. At best, load and save have opposite
        # orientations; in other words, if you load an image and then save it
        # the resulting saved image will be upside-down
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if img.mode=='I;16':
            # 16-bit image
            imgdata = numpy.array(img.getdata(),numpy.uint16)
            img = imgdata.reshape(img.size)
            img = img.astype(float) / 65535.0
        else:
            img = matplotlib.image.pil_to_array(img)
        return cpimage.Image(img)
    
    def get_name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return self.__filename
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.get_filename())
