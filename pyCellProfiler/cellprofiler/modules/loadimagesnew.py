'''<b>Load Images</b> allows you to specify which images or movies are to be loaded and in
which order. Groups of images will be loaded per cycle of CellProfiler processing.
<hr>
Tells CellProfiler where to retrieve images and gives each image a
meaningful name for the other modules to access. When used in combination
with a <b>SaveImages</b> module, you can load images in one file format and
save in another file format, making CellProfiler work as a file format
converter.

See also <b>LoadSingleImage</b>,<b>SaveImages</b>
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
import cgi
import hashlib
import os
import re
import sys
import wx
import wx.html

try:
    import bioformats.formatreader as formatreader
    env = formatreader.get_env()
    FormatTools = formatreader.make_format_tools_class(env)
    ImageReader = formatreader.make_image_reader_class(env)
    ChannelSeparator = formatreader.make_reader_wrapper_class(
        env,"loci/formats/ChannelSeparator")
    has_bioformats = True
except:
    has_bioformats = False
import Image as PILImage
import TiffImagePlugin as TIFF
import cellprofiler.dib
import matplotlib.image
import scipy.io.matlab.mio
import uuid

import cellprofiler.cpmodule as cpmodule
import cellprofiler.cpimage as cpimage
import cellprofiler.measurements as cpm
import cellprofiler.preferences as preferences
import cellprofiler.settings as cps

PILImage.init()

FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi movies'
FF_OTHER_MOVIES = 'tif,tiff,flex movies'
if has_bioformats:
    FF_CHOICES = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
else:
    FF_CHOICES = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES]

# The metadata choices:
# M_NONE - don't extract metadata
# M_FILE_NAME - extract metadata from the file name
# M_PATH_NAME - extract metadata from the subdirectory path
# M_BOTH      - extract metadata from both the file name and path
M_NONE      = "None"
M_FILE_NAME = "File name"
M_PATH      = "Path"
M_BOTH      = "Both"

'''The provider name for the image file image provider'''
P_IMAGES = "LoadImagesImageProvider"
'''The version number for the __init__ method of the image file image provider'''
V_IMAGES = 1

'''The provider name for the movie file image provider'''
P_MOVIES = "LoadImagesMovieProvider"
'''The version number for the __init__ method of the movie file image provider'''
V_MOVIES = 1

class LoadImagesNew(cpmodule.CPModule):

    module_name = "LoadImagesNew"
    variable_revision_number = 1
    category = "File Processing"

    def create_settings(self):
        # Settings
        self.file_types = cps.Choice('What type of files are you loading?', FF_CHOICES, doc="""
                The following image file types are permissible for input into CellProfiler:
                <ul>
                <li><i>Individual images:</i>Each file represents a single image. 
                Some methods of file compression sacrifice image quality ("lossy") and should be avoided for automated image analysis 
                if at all possible (e.g., .jpg). Other file compression formats retain exactly the original image information but in 
                a smaller file ("lossless") so they are perfectly acceptable for image analysis (e.g., .png, .tif, .gif). 
                Uncompressed file formats are also fine for image analysis (e.g., .bmp)</li>
                <li><i>AVI movies:</i>An AVI (Audio Video Interleave) file is a type of movie file. Only uncompressed AVIs are supported.
                Files are opened as a stack of images.</li>
                <li><i>TIF,TIFF,FLEX movies:</i>A TIF/TIFF movie is a file in which a series of images are contained as individual frames. 
                The same is true for the FLEX file format (used by Evotec Opera automated microscopes). Files are opened as a stack of images.</li>
                <li><i>STK movies:</i> STKs are a proprietary image format used by MetaMorph (Molecular Devices). It is typically
                used to encode 3D image data, e.g. from confocal microscopy, and is a special version of the TIF format. </li>
                </ul>
                For the movie formats, the files are opened as a stack of images and each image is processed individually.""")
        
        self.descend_subdirectories = cps.Binary('Descend into subdirectories?', False, doc="""
                If this box is checked, all the subfolders under the image directory location that you specify will be
                searched for images.""")
        
        self.top_spacer = cps.Divider()

        # Images settings
        self.exclude = cps.Binary('Do you want to exclude some files?', False,doc="""
                <p>This settings allows you exclude certain files (such as thumbnails) from further consideration.""")
        
        # should this be a regexp?
        self.match_exclude_text = cps.Text('Substring in files to be excluded', cps.DO_NOT_USE,doc="""
                <i>(Only used if file exclusion is selected)</i> 
                <p>Here you can specify substring that marks files for exclusion.</p>""")


        self.image_spacer_1 = cps.Divider(line=False)

        self.images = []
        self.add_image()

        self.add_image_button = cps.DoSomething('Add another image...','Add', self.add_image)

        self.image_spacer_bottom = cps.Divider()


        # Metadata settings
        self.metadata_label = cps.Divider("Metadata")

        

        # extraction settings
        self.grouping_label = cps.Divider("Grouping")

    def add_image(self):
        def default_cpimage_name(index):
            # the usual suspects
            names = ['DNA', 'Actin', 'Protein']
            if index < len(names):
                return names[index]
            return 'Channel%d'%(index+1)

        group = cps.SettingsGroup()
        group.append("image_name", 
                     cps.FileImageNameProvider('What do you want to call this image in CellProfiler?', 
                                               default_cpimage_name(len(self.images)), doc="""
                        Give your images a meaningful name that you will use when referring to
                        these images in later modules.  Keep the following points in mind when deciding 
                        on an image name:
                        <ul>
                        <li>Image names must begin with a letter, which may be followed by any 
                        combination of letters, digits, and underscores. The following names are all invalid:
                        <ul>
                        <li>My.Cells</li>
                        <li>1stCells</li>
                        <li>1+1=3</li>
                        <li>@MyCell</li>
                        </ul>
                        </li>
                        <li>Names are not case senstive. Therefore, <i>OrigBlue</i>, <i>origblue</i>, and <i>ORIGBLUE</i>
                        will correspond to the same name, and unexpected results may ensue.</li>
                        <li>Although the names can be of any length in CellProfiler, you may want to avoid 
                        making the name too long especially if you are uploading to a database. The name is used
                        to generate the column header for a given measurement, and in MySQL, the total bytes used
                        for the column headers cannot exceed 64K. A warning will be generated later if this limit
                        has been exceeded.</li>
                        </ul>"""))
        group.append("image_specifier", cps.Text("What substring is common to these images?", ""))
        group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.images, group))
        group.append("divider", cps.Divider(line=False))
        self.images.append(group)

    def visible_settings(self):
        result = [self.file_types, self.descend_subdirectories, self.top_spacer]
        result += [self.exclude]
        if self.exclude.value:
            result += [self.match_exclude_text]
            
        # images
        result += [self.image_spacer_1]
        for im in self.images:
            result += im.unpack_group()
        result += [self.add_image_button, self.image_spacer_bottom]
        
        # metadata
        result += [self.metadata_label]
        
        # grouping
        result += [self.grouping_label]
        return result

    def settings(self):
        result = [self.file_types, self.descend_subdirectories]
        result += [self.exclude, self.match_exclude_text]
        
        # images
        for im in self.images:
            result += [im.image_name, im.image_specifier]
        
        # metadata
        
        # grouping
        return result

    def prepare_run(self, pipeline, image_set_list, frame):
        """Set up all of the image providers inside the image_set_list
        """
        if pipeline.in_batch_mode():
            # Don't set up if we're going to retrieve the image set list
            # from batch mode
            return True
        if self.load_movies():
            self.prepare_run_of_movies(pipeline,image_set_list)
        else:
            self.prepare_run_of_images(pipeline, image_set_list, frame)
        return True
    
    def prepare_run_of_images(self, pipeline, image_set_list, frame):
        """Set up image providers for image files"""
        return

    def get_imageset_dictionary(self, image_set):
        '''Get the module's legacy fields dictionary for this image set'''
        d = self.get_dictionary()
        if not d.has_key(image_set.number):
            d[image_set.number] = {}
        return d[image_set.number]
    
    def run(self,workspace):
        """Run the module - add the measurements
        
        """
        pass

    def display(self, workspace):
        pass

    def get_filename_metadata(self, fd, filename, path):
        """Get the filename and path metadata for a given image
        
        fd - file/image dictionary
        filename - filename to be parsed
        path - path to be parsed
        """
        metadata = {}
        if fd[FD_METADATA_CHOICE].value in (M_BOTH, M_FILE_NAME):
            metadata.update(cpm.extract_metadata(fd[FD_FILE_METADATA].value,
                                                 filename))
        if fd[FD_METADATA_CHOICE].value in (M_BOTH, M_PATH):
            metadata.update(cpm.extract_metadata(fd[FD_PATH_METADATA].value,
                                                 path))
        return metadata
        
    def get_frame_count(self, pathname):
        """Return the # of frames in a movie"""
        if self.file_types in (FF_AVI_MOVIES,FF_OTHER_MOVIES,FF_STK_MOVIES):
            rdr = ImageReader()
            rdr.setId(pathname)
            if self.file_types == FF_STK_MOVIES:
                return rdr.getSizeZ()
            else:
                return rdr.getSizeT()
            
        raise NotImplementedError("get_frame_count not implemented for %s"%(self.file_types))
    
    def get_metadata_tags(self, fd=None):
        """Find the metadata tags for the indexed image

        fd - an image file directory from self.images
        """
        if not fd:
            s = set()
            for fd in self.images:
                s.update(self.get_metadata_tags(fd))
            tags = list(s)
            tags.sort()
            return tags
        
        tags = []
        if fd[FD_METADATA_CHOICE] in (M_FILE_NAME, M_BOTH):
            tags += cpm.find_metadata_tokens(fd[FD_FILE_METADATA].value)
        if fd[FD_METADATA_CHOICE] in (M_PATH, M_BOTH):
            tags += cpm.find_metadata_tokens(fd[FD_PATH_METADATA].value)
        return tags
    
    def get_groupings(self, image_set_list):
        '''Return the groupings as indicated by the metadata_fields setting'''
        if self.group_by_metadata.value:
            keys = self.metadata_fields.selections
            if len(keys) == 0 or image_set_list is None:
                return None
            return image_set_list.get_groupings(keys)
        else:
            return None
    
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
        else:
            return preferences.get_absolute_path(self.location_other.value)
    
    def image_name_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        return [fd[FD_IMAGE_NAME] for fd in self.images]
        
    def text_to_find_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        return [fd[FD_COMMON_TEXT] for fd in self.images]
    
    def text_to_exclude(self):
        """Return the text to match against the file name to exclude it from the set
        """
        return self.match_exclude.value
    
    def filter_filename(self, filename):
        """Returns either None or the index of the match setting
        """
        if not is_image(filename):
            return None
        if ((True and is_movie(filename)) != 
            (True and (self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES, 
                                           FF_OTHER_MOVIES)))):
            return None
                                                    
        if self.text_to_exclude() != cps.DO_NOT_USE and \
            filename.find(self.text_to_exclude()) >=0:
            return None
        if self.load_choice() == MS_EXACT_MATCH:
            ttfs = self.text_to_find_vars()
            for i,ttf in enumerate(ttfs):
                if filename.find(ttf.value) >=0:
                    return i
        elif self.load_choice() == MS_REGEXP:
            ttfs = self.text_to_find_vars()
            for i,ttf in enumerate(ttfs):
                if re.search(ttf.value, filename):
                    return i
        else:
            raise NotImplementedError("Load by order not implemented")
        return None

    def get_measurement_columns(self, pipeline):
        '''Return a sequence describing the measurement columns needed by this module 
        '''
        cols = []
        for fd in self.images:
            name = fd[FD_IMAGE_NAME].value
            cols += [('Image','FileName_'+name, cpm.COLTYPE_VARCHAR_FILE_NAME)]
            cols += [('Image','PathName_'+name, cpm.COLTYPE_VARCHAR_PATH_NAME)]
            cols += [('Image','MD5Digest_'+name, cpm.COLTYPE_VARCHAR_FORMAT%32)]
        
        fd = self.images[0]    
        if fd[FD_METADATA_CHOICE]==M_FILE_NAME or fd[FD_METADATA_CHOICE]==M_BOTH:
            tokens = cpm.find_metadata_tokens(fd[FD_FILE_METADATA].value)
            cols += [('Image', 'Metadata_'+token, cpm.COLTYPE_VARCHAR_FILE_NAME) for token in tokens]
        
        if fd[FD_METADATA_CHOICE]==M_PATH or fd[FD_METADATA_CHOICE]==M_BOTH:
            tokens = cpm.find_metadata_tokens(fd[FD_PATH_METADATA].value)
            cols += [('Image', 'Metadata_'+token, cpm.COLTYPE_VARCHAR_PATH_NAME) for token in tokens]
        
        return cols
    
    def change_causes_prepare_run(self, setting):
        '''Check to see if changing the given setting means you have to restart
        
        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to do "prepare_run".
        '''
        #
        # It's safest to say that any change in loadimages requires a restart
        #
        return True
            
            
def is_image(filename):
    '''Determine if a filename is a potential image file based on extension'''
    ext = os.path.splitext(filename)[1].lower()
    if PILImage.EXTENSION.has_key(ext):
        return True
    return ext in ('.avi', '.mpeg', '.mat', '.stk')

def is_movie(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ('.avi', '.mpeg', '.stk')


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
        if self.__filename.lower().endswith(".mat"):
            imgdata = scipy.io.matlab.mio.loadmat(self.get_full_name(),
                                                  struct_as_record=True)
            return cpimage.Image(imgdata["Image"])
        elif has_bioformats:
            img = load_using_bioformats(self.get_full_name())
        elif self.__filename.lower().endswith(".dib"):
            img = cpimage.readc01(self.get_full_name())
        else:
            img = load_using_PIL(self.get_full_name())
        return cpimage.Image(img,
                             path_name = self.get_pathname(),
                             file_name = self.get_filename())
    
    def get_name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return self.__filename
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.get_filename())
    
    def release_memory(self):
        '''Release any image memory
        
        The image is either loaded every time or cached so this is a no-op'''
        pass

def load_using_PIL(path, index=0, seekfn=None):
    '''Get the pixel data for an image using PIL
    
    path - path to file
    index - index of the image if stacked image format such as TIFF
    seekfn - a function for seeking to a given image in a stack
    '''
    img = PILImage.open(path)
    if seekfn is None:
        img.seek(index)
    else:
        seekfn(img, index)
    if img.mode=='I;16':
        # 16-bit image
        # deal with the endianness explicitly... I'm not sure
        # why PIL doesn't get this right.
        imgdata = np.fromstring(img.tostring(),np.uint8)
        imgdata.shape=(int(imgdata.shape[0]/2),2)
        imgdata = imgdata.astype(np.uint16)
        hi,lo = (0,1) if img.tag.prefix == 'MM' else (1,0)
        imgdata = imgdata[:,hi]*256 + imgdata[:,lo]
        img_size = list(img.size)
        img_size.reverse()
        new_img = imgdata.reshape(img_size)
        # The magic # for maximum sample value is 281
        if img.tag.has_key(281):
            img = new_img.astype(float) / img.tag[281][0]
        elif np.max(new_img) < 4096:
            img = new_img.astype(float) / 4095.
        else:
            img = new_img.astype(float) / 65535.
    else:
        # There's an apparent bug in the PIL library that causes
        # images to be loaded upside-down. At best, load and save have opposite
        # orientations; in other words, if you load an image and then save it
        # the resulting saved image will be upside-down
        img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        img = matplotlib.image.pil_to_array(img)
    return img

def load_using_bioformats(path, z=0, t=0):
    '''Load the given image file using the Bioformats library
    
    path: path to the file
    z: the frame index in the z (depth) dimension.
    t: the frame index in the time dimension.
    
    Returns either a 2-d (grayscale) or 3-d (2-d + 3 RGB planes) image
    '''
    rdr = ImageReader()
    rdr.setId(path)
    width = rdr.getSizeX()
    height = rdr.getSizeY()
    pixel_type = rdr.getPixelType()
    little_endian = rdr.isLittleEndian()
    if pixel_type == FormatTools.INT8:
        dtype = np.char
        scale = 255
    elif pixel_type == FormatTools.UINT8:
        dtype = np.uint8
        scale = 255
    elif pixel_type == FormatTools.UINT16:
        dtype = '<u2' if little_endian else '>u2'
        scale = 65536
    elif pixel_type == FormatTools.INT16:
        dtype = '<i2' if little_endian else '>i2'
        scale = 65536
    elif pixel_type == FormatTools.UINT32:
        dtype = '<u4' if little_endian else '>u4'
        scale = 2**32
    elif pixel_type == FormatTools.INT32:
        dtype = '<i4' if little_endian else '>i4'
        scale = 2**32
    elif pixel_type == FormatTools.FLOAT:
        dtype = '<f4' if little_endian else '>f4'
        scale = 1
    elif pixel_type == FormatTools.DOUBLE:
        dtype = '<f8' if little_endian else '>f8'
        scale = 1
    max_sample_value = rdr.getMetadataValue('MaxSampleValue')
    if max_sample_value is not None:
        try:
            scale = formatreader.jutil.call(env, max_sample_value, 
                                            'intValue', '()I')
        except:
            sys.stderr.write("WARNING: failed to get MaxSampleValue for image. Intensities may be improperly scaled\n")
    if rdr.getRGBChannelCount() > 1:
        rdr.close()
        rdr = ChannelSeparator(ImageReader())
        rdr.setId(path)
        red_image, green_image, blue_image = [
            np.frombuffer(rdr.openBytes(rdr.getIndex(z,i,t)),dtype)
            for i in range(3)]
        image = np.dstack((red_image, green_image, blue_image))
        image.shape=(height,width,3)
    else:
        index = rdr.getIndex(z,0,t)
        image = np.frombuffer(rdr.openBytes(index),dtype)
        image.shape = (height,width)
    rdr.close()
    image = image.astype(float) / float(scale)
    return image
    
class LoadImagesMovieFrameProvider(cpimage.AbstractImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """
    def __init__(self,name,pathname,filename,frame):
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__frame    = frame
    
    def provide_image(self, image_set):
        """Load an image from a movie frame
        """
        pixel_data = load_using_bioformats(self.get_full_name(), 0, 
                                           self.__frame)
        image = cpimage.Image(pixel_data, path_name = self.get_pathname(),
                              file_name = self.get_filename())
        return image
    
    def get_name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return self.__filename
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.get_filename())

class LoadImagesSTKFrameProvider(cpimage.AbstractImageProvider):
    """Provide an image by filename:frame from an STK file"""
    def __init__(self, name, pathname, filename, frame):
        '''Initialize the provider
        
        name - name of the provider for access from image set
        pathname - path to the file
        filename - name of the file
        frame - # of the frame to provide
        '''
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__frame    = frame
        
    def provide_image(self, image_set):
        if has_bioformats:
            img = load_using_bioformats(self.get_full_name(),
                                         z=self.__frame)
        else:
            def seekfn(img, index):
                '''Seek in an STK file to a given stack frame
                
                The stack frames are of constant size and follow each other.
                The tiles contain offsets which need to be incremented by
                the size of a stack frame. The following is from 
                Molecular Devices' STK file format document:
                StripOffsets
                The strips for all the planes of the stack are stored 
                contiguously at this location. The following pseudocode fragment 
                shows how to find the offset of a specified plane planeNum.
                LONG	planeOffset = planeNum *
                    (stripOffsets[stripsPerImage - 1] +
                    stripByteCounts[stripsPerImage - 1] - stripOffsets[0]);
                Note that the planeOffset must be added to the stripOffset[0]
                to find the image data for the specific plane in the file.
                '''
                plane_offset = long(index) * (img.ifd[TIFF.STRIPOFFSETS][-1] +
                                              img.ifd[TIFF.STRIPBYTECOUNTS][-1] -
                                              img.ifd[TIFF.STRIPOFFSETS][0])
                img.tile = [(coding, location, offset+plane_offset, format)
                            for coding, location, offset, format in img.tile]
                
            img = load_using_PIL(self.get_full_name(), self.__frame, seekfn)
        return cpimage.Image(img,
                             path_name = self.get_pathname(),
                             file_name = self.get_filename())
    def get_name(self):
        return self.__name
    
    def get_pathname(self):
        return self.__pathname
    
    def get_filename(self):
        return "%s:%d" % (self.__filename, self.__frame)
    
    def get_full_name(self):
        return os.path.join(self.get_pathname(),self.__filename)

    def release_memory(self):
        '''Release any image memory
        
        The image is either loaded every time or cached so this is a no-op'''
        pass
    
