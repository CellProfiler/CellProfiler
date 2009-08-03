""" LoadImages.py - module to load images from files

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import cgi
import os
import re
import wx
import wx.html

import Image as PILImage
import numpy
import matplotlib.image
import scipy.io.matlab.mio
import uuid

import cellprofiler.cpmodule as cpmodule
import cellprofiler.cpimage as cpimage
import cellprofiler.measurements as cpm
import cellprofiler.preferences as preferences
import cellprofiler.settings as cps

PILImage.init()

# strings for choice variables
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGEXP = 'Text-Regular expressions'
MS_ORDER = 'Order'

FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi movies'
FF_OTHER_MOVIES = 'tif,tiff,flex movies'
try:
    import cellprofiler.ffmpeg.ffmpeg as ffmpeg
    FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
except ImportError:
    FF = [FF_INDIVIDUAL_IMAGES]

DIR_DEFAULT_IMAGE = 'Default Image Directory'
DIR_DEFAULT_OUTPUT = 'Default Output Directory'
DIR_OTHER = 'Elsewhere...'

SB_GRAYSCALE = 'grayscale'
SB_BINARY = 'binary'

FD_KEY = "Key"
FD_COMMON_TEXT = "CommonText"
FD_ORDER_POSITION = "OrderPosition"
FD_IMAGE_NAME = "ImageName"
FD_REMOVE_IMAGE = "RemoveImage"
FD_METADATA_CHOICE = "MetadataChoice"
FD_FILE_METADATA = "FileMetadata"
FD_PATH_METADATA = "PathMetadata"

# The metadata choices:
# M_NONE - don't extract metadata
# M_FILE_NAME - extract metadata from the file name
# M_PATH_NAME - extract metadata from the subdirectory path
# M_BOTH      - extract metadata from both the file name and path
M_NONE      = "None"
M_FILE_NAME = "File name"
M_PATH      = "Path"
M_BOTH      = "Both"

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
        self.exclude = cps.Binary('Do you want to exclude certain files?', False)
        self.match_exclude = cps.Text('Type the text that the excluded images have in common', cps.DO_NOT_USE)
        self.order_group_size = cps.Integer('How many images are there in each group?', 3)
        self.descend_subdirectories = cps.Binary('Analyze all subfolders within the selected folder?', False)
        self.check_images = cps.Binary('Do you want to check image sets for missing or duplicate files?',True)
        self.group_by_metadata = cps.Binary('Do you want to group image sets by metadata?',True)
        # Add the first image to the images list
        self.images = []
        self.add_imagecb()
        # Add another image
        self.add_image = cps.DoSomething('Add another image...','Add', self.add_imagecb)
        
        # Location settings
        self.location = cps.CustomChoice('Where are the images located?',
                                        [DIR_DEFAULT_IMAGE, DIR_DEFAULT_OUTPUT, DIR_OTHER])
        self.location_other = cps.DirectoryPath("Where are the images located?", '')

    def add_imagecb(self):
        'Adds another image to the settings'
        img_index = len(self.images)
        new_uuid = uuid.uuid1()
        fd = { FD_KEY:new_uuid,
               FD_COMMON_TEXT:cps.Text('Type the text that these images have in common', ''),
               FD_ORDER_POSITION:cps.Integer('What is the position of this image in each group', img_index+1),
               FD_IMAGE_NAME:cps.FileImageNameProvider('What do you want to call this image in CellProfiler?', 
                                                       default_cpimage_name(img_index)),
               FD_METADATA_CHOICE:cps.Choice('Do you want to extract metadata from the file name, the subdirectory path or both?',
                                             [M_NONE, M_FILE_NAME, 
                                              M_PATH, M_BOTH]),
               FD_FILE_METADATA: cps.RegexpText('Type the regular expression that finds metadata in the file name:',
                                                '^(?P<Plate>.+)_(?P<WellRow>[A-P])(?P<WellColumn>[0-9]{1,2})_(?P<Site>[0-9])'),
               FD_PATH_METADATA: cps.RegexpText('Type the regular expression that finds metadata in the subdirectory path:',
                                          '(?P<Year>[0-9]{4})-(?P<Month>[0-9]{2})-(?P<Day>[0-9]{2})'),
               FD_REMOVE_IMAGE:cps.DoSomething('Remove this image...', 'Remove',self.remove_imagecb, new_uuid)
               }
        self.images.append(fd)

    def remove_imagecb(self, id):
        'Remove an image from the settings'
        index = [fd[FD_KEY] for fd in self.images].index(id)
        del self.images[index]

    def visible_settings(self):
        varlist = [self.file_types, self.match_method]
        
        if self.match_method == MS_EXACT_MATCH:
            varlist += [self.exclude]
            if self.exclude.value:
                varlist += [self.match_exclude]
        elif self.match_method == MS_ORDER:
            varlist += [self.order_group_size]
        varlist += [self.descend_subdirectories]
        
        if len(self.images) > 1:
            varlist += [self.check_images]
            varlist += [self.group_by_metadata]
        
        # per image settings
        if self.match_method != MS_ORDER:
            file_kwd = FD_COMMON_TEXT
        else:
            file_kwd = FD_ORDER_POSITION
        
        for fd in self.images:
            varlist += [fd[file_kwd], 
                        fd[FD_IMAGE_NAME],
                        fd[FD_METADATA_CHOICE]]
            if fd[FD_METADATA_CHOICE].value in (M_FILE_NAME, M_BOTH):
                varlist.append(fd[FD_FILE_METADATA])
            if fd[FD_METADATA_CHOICE].value in (M_PATH, M_BOTH):
                varlist.append(fd[FD_PATH_METADATA])
            varlist.append(fd[FD_REMOVE_IMAGE])
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
    SLOT_FIRST_IMAGE_V1 = 8
    SLOT_GROUP_BY_METADATA = 8
    SLOT_EXCLUDE = 9
    SLOT_FIRST_IMAGE_V2 = 9
    SLOT_FIRST_IMAGE = 10
    
    SLOT_OFFSET_COMMON_TEXT = 0
    SLOT_OFFSET_IMAGE_NAME = 1
    SLOT_OFFSET_ORDER_POSITION = 2
    SLOT_OFFSET_METADATA_CHOICE = 3
    SLOT_OFFSET_FILE_METADATA = 4
    SLOT_OFFSET_PATH_METADATA = 5
    SLOT_IMAGE_FIELD_COUNT_V1 = 3
    SLOT_IMAGE_FIELD_COUNT = 6
    
    def settings(self):
        """Return the settings array in a consistent order"""
        varlist = range(self.SLOT_FIRST_IMAGE + \
                        self.SLOT_IMAGE_FIELD_COUNT * len(self.images))
        varlist[self.SLOT_FILE_TYPE]              = self.file_types
        varlist[self.SLOT_MATCH_METHOD]           = self.match_method
        varlist[self.SLOT_ORDER_GROUP_SIZE]       = self.order_group_size
        varlist[self.SLOT_EXCLUDE]                = self.exclude
        varlist[self.SLOT_MATCH_EXCLUDE]          = self.match_exclude
        varlist[self.SLOT_DESCEND_SUBDIRECTORIES] = self.descend_subdirectories
        varlist[self.SLOT_LOCATION]               = self.location
        varlist[self.SLOT_LOCATION_OTHER]         = self.location_other
        varlist[self.SLOT_CHECK_IMAGES]           = self.check_images
        varlist[self.SLOT_GROUP_BY_METADATA]      = self.group_by_metadata
        for i in range(len(self.images)):
            ioff = i*self.SLOT_IMAGE_FIELD_COUNT + self.SLOT_FIRST_IMAGE
            varlist[ioff+self.SLOT_OFFSET_COMMON_TEXT] = \
                self.images[i][FD_COMMON_TEXT]
            varlist[ioff+self.SLOT_OFFSET_IMAGE_NAME] = \
                self.images[i][FD_IMAGE_NAME]
            varlist[ioff+self.SLOT_OFFSET_ORDER_POSITION] = \
                self.images[i][FD_ORDER_POSITION]
            varlist[ioff+self.SLOT_OFFSET_METADATA_CHOICE] = \
                self.images[i][FD_METADATA_CHOICE]
            varlist[ioff+self.SLOT_OFFSET_FILE_METADATA] =\
                self.images[i][FD_FILE_METADATA]
            varlist[ioff+self.SLOT_OFFSET_PATH_METADATA] =\
                self.images[i][FD_PATH_METADATA]
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
        
        if (variable_revision_number == 1 and module_name == self.module_class()):
            setting_values, variable_revision_number = self.upgrade_new_1_to_2(setting_values)
        if (variable_revision_number == 2 and module_name == self.module_class()):
            setting_values, variable_revision_number = self.upgrade_new_2_to_3(setting_values)

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
        while len(self.images) > image_count:
            self.remove_imagecb(self.image_keys[0])
        while len(self.images) < image_count:
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
    
    def upgrade_5_to_new_1(self, setting_values):
        """Take the old LoadImages values and put them in the correct slots"""
        new_values = range(self.SLOT_FIRST_IMAGE_V1)
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
    
    def upgrade_new_1_to_2(self, setting_values):
        """Add the metadata slots to the images"""
        new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V1])
        new_values.append(cps.NO) # Group by metadata is off
        for i in range((len(setting_values)-self.SLOT_FIRST_IMAGE_V1) / self.SLOT_IMAGE_FIELD_COUNT_V1):
            off = self.SLOT_FIRST_IMAGE_V1 + i * self.SLOT_IMAGE_FIELD_COUNT_V1
            new_values.extend([setting_values[off],
                               setting_values[off+1],
                               setting_values[off+2],
                               M_NONE,
                               "None",
                               "None"])
        return (new_values, 2)

    def upgrade_new_2_to_3(self, setting_values):
        """Add the checkbox for excluding certain files"""
        new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V2])
        if setting_values[self.SLOT_MATCH_EXCLUDE] == cps.DO_NOT_USE:
            new_values += [cps.NO]
        else:
            new_values += [cps.YES]
        for i in range((len(setting_values)-self.SLOT_FIRST_IMAGE_V2) / self.SLOT_IMAGE_FIELD_COUNT):
            off = self.SLOT_FIRST_IMAGE_V2 + i * self.SLOT_IMAGE_FIELD_COUNT
            new_values.extend([setting_values[off],
                               setting_values[off+1],
                               setting_values[off+2],
                               M_NONE,
                               "None",
                               "None"])
        return (new_values, 3)

    variable_revision_number = 3
    
    
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
            self.prepare_run_of_images(pipeline, image_set_list, frame)
        return True
    
    def prepare_run_of_images(self, pipeline, image_set_list, frame):
        """Set up image providers for image files"""
        files = self.collect_files()
        if len(files) == 0:
            raise ValueError("there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well)")
        
        if self.group_by_metadata.value and len(self.get_metadata_tags()):
            self.organize_by_metadata(pipeline, image_set_list, files, frame)
        else:
            self.organize_by_order(pipeline, image_set_list, files)
        for name in self.image_name_vars():
            image_set_list.legacy_fields['Pathname%s'%(name.value)]=self.image_directory()

    def organize_by_order(self, pipeline, image_set_list, files):
        """Organize each kind of file by their lexical order
        
        """
        image_names = self.image_name_vars()
        list_of_lists = [[] for x in image_names]
        for pathname,image_index in files:
            list_of_lists[image_index].append(pathname)
        
        image_set_count = len(list_of_lists[0])
        for x,name in zip(list_of_lists[1:],image_names[1:]):
            if len(x) != image_set_count:
                raise RuntimeError("Image %s has %d files, but image %s has %d files" %
                                   (image_names[0], image_set_count,
                                    name.value, len(x)))
        list_of_lists = numpy.array(list_of_lists)
        root = self.image_directory()
        for i in range(0,image_set_count):
            image_set = image_set_list.get_image_set(i)
            providers = [LoadImagesImageProvider(name.value,root,file) for name,file in zip(image_names, list_of_lists[:,i])]
            image_set.providers.extend(providers)
    
    def organize_by_metadata(self, pipeline, image_set_list, files, frame):
        """Organize each kind of file by metadata
        
        """
        #
        # Distribute files according to metadata tags. Each image_name
        # potentially has a subset of the total list of tags. For images
        # without the complete list of tags, we give the image a wildcard
        # for the metadata item that matches everything.
        #
        tags = self.get_metadata_tags()
        #
        # files_by_image_name holds one list of files for each image name index
        #
        files_by_image_name = [[] for i in range(len(self.images))]
        for file,i in files:
            files_by_image_name[i].append(os.path.split(file))
        #
        # Create a monster dictionary tree for each image describing the
        # metadata for each tag. Give a file a metadata value of None
        # if it has no value for a given metadata tag
        #
        d = [{} for i in range(len(self.images))]
        conflicts = []
        for i in range(len(self.images)):
            fd = self.images[i]
            for path, filename in files_by_image_name[i]:
                metadata = self.get_filename_metadata(fd, filename, path)
                parent = d[i]
                for tag in tags[:-1]:
                    value = metadata.get(tag)
                    if parent.has_key(value):
                        child = parent[value]
                    else:
                        child = {}
                        parent[value] = child
                    parent = child
                    last_value = value
                tag = tags[-1]
                value = metadata.get(tag)
                if parent.has_key(value):
                    # There's already a match to this metadata
                    conflict = [fd, parent[value], (path,filename)]
                    conflicts.append(conflict)
                else:
                    parent[value] = (path, filename)
        image_sets = self.get_image_sets(d)
        missing_images = [image_set for image_set in image_sets
                          if None in image_set[1]]
        image_sets = [image_set for image_set in image_sets
                      if not None in image_set[1]]
        #
        # Handle errors, raising an exception if the user wants to check images
        #
        if len(conflicts) or len(missing_images):
            if frame:
                self.report_errors(conflicts, missing_images, frame)
            if self.check_images.value:
                message=""
                if len(conflicts):
                    message +="Conflicts found:\n" 
                    for conflict in conflicts:
                        metadata = self.get_filename_metadata(conflict[0], 
                                                              conflict[1][1], 
                                                              conflict[1][0])
                        message+=("Metadata: %s, First path: %s, First filename: %s, Second path: %s, Second filename: %s\n"%
                                  (str(metadata), conflict[1][0],conflict[1][1],
                                   conflict[2][0],conflict[2][1]))
                if len(missing_images):
                    message += "Missing images:\n"
                    for mi in missing_images:
                        for i in range(len(self.images)):
                            fd = self.images[i]
                            if mi[1][i] is None:
                                message += ("%s: missing " %
                                            (fd[FD_IMAGE_NAME].value))
                            else:
                                message += ("%s: path=%s, file=%s" %
                                            (fd[FD_IMAGE_NAME].value,
                                             mi[1][i][0],mi[1][i][1]))
                raise ValueError(message)
        root = self.image_directory()
        for image_set in image_sets:
            keys = {}
            for tag,value in zip(tags,image_set[0]):
                keys[tag] = value
            cpimageset = image_set_list.get_image_set(keys)
            for i in range(len(self.images)):
                path = os.path.join(image_set[1][i][0],image_set[1][i][1])
                provider = LoadImagesImageProvider(self.images[i][FD_IMAGE_NAME].value,
                                                   root,path)
                cpimageset.providers.append(provider)
    
    def get_image_sets(self, d):
        """Get image sets from a dictionary tree
        
        d - one dictionary tree per image.
        returns a list of tuples. The first element contains a tuple of
        metadata values and the second element contains a list of
        (path,filename) tuples for each image (or are None for errors where 
        there is no metadata match).
        """
        
        if not any([isinstance(dd,dict) for dd in d]):
            # This is a leaf if no elements are dictionaries
            return [(tuple(), d)]
        #
        # Get each represented value for this slot
        #
        values = set()
        for dd in d:
            if not dd is None:
                values.update(dd.keys())
        result = []
        for value in sorted(values):
            subgroup = tuple((dd and dd.has_key(None) and dd[None]) or # wildcard metadata
                             (dd and dd.get(value)) or                 # fetch subvalue or None if missing
                             None                                      # metadata is missing
                             for dd in d)
            subsets = self.get_image_sets(subgroup)
            for subset in subsets:
                # prepend the current key to the tuple of keys and 
                # duplicate the filename tuple
                subset = ((value,)+subset[0], subset[1])
                result.append(subset)
        return result

    def report_errors(self, conflicts, missing_images, frame):
        """Create an error report window
        
        conflicts: 3-tuple of file dictionary, first path/file and second path/file
                   for two images with the same metadata
        missing_images: two-tuple of metadata keys and the file tuples
                        where at least one of the file tuples is None indicating
                        a missing image
        frame: the parent for the error report
        """
        my_frame = wx.Frame(frame, title="Load images: Error report",
                            size=(600,800),
                            pos =(frame.Position[0],
                                  frame.Position[1]+frame.Size[1]))
        my_frame.Icon = frame.Icon
        panel = wx.Panel(my_frame)
        uber_sizer = wx.BoxSizer()
        my_frame.SetSizer(uber_sizer)
        uber_sizer.Add(panel,1,wx.EXPAND)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        font = wx.Font(16,wx.FONTFAMILY_SWISS,wx.FONTSTYLE_NORMAL,
                       wx.FONTWEIGHT_BOLD)
        tags = self.get_metadata_tags()
        tag_ct = len(tags)
        tables = []
        if len(conflicts):
            title = wx.StaticText(panel,
                                  label="Conflicts (two files with same metadata)")
            title.Font = font
            sizer.Add(title, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL|wx.ALL,3)
            table = wx.ListCtrl(panel,style=wx.LC_REPORT)
            tables.append(table)
            sizer.Add(table, 1, wx.EXPAND|wx.ALL,3)
            for tag, index in zip(tags,range(tag_ct)):
                table.InsertColumn(index,tag)
            table.InsertColumn(index+1,"First path")
            table.InsertColumn(index+2,"First file name")
            table.InsertColumn(index+3,"Second path")
            table.InsertColumn(index+4,"Second file name")
            for conflict in conflicts:
                metadata = self.get_filename_metadata(conflict[0], 
                                                      conflict[1][1], 
                                                      conflict[1][0])
                row = [metadata.get(tag) or "-" for tag in tags]
                row.extend([conflict[1][0],conflict[1][1],
                            conflict[2][0],conflict[2][1]])
                table.Append(row)

        if len(missing_images):
            title = wx.StaticText(panel,
                                  label="Missing images")
            title.Font = font
            sizer.Add(title, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL|wx.ALL,3)
            table = wx.ListCtrl(panel,style=wx.LC_REPORT)
            tables.append(table)
            sizer.Add(table, 1, wx.EXPAND|wx.ALL,3)
            for tag, index in zip(tags,range(tag_ct)):
                table.InsertColumn(index,tag)
            for fd,index in zip(self.images,range(len(self.images))):
                table.InsertColumn(index*2+tag_ct,
                                   "%s path"%(fd[FD_IMAGE_NAME].value))
                table.InsertColumn(index*2+1+tag_ct,
                                   "%s filename"%(fd[FD_IMAGE_NAME].value))
            for metadata,files_and_paths in missing_images:
                row = list(metadata)
                for file_and_path in files_and_paths:
                    if file_and_path is None:
                        row.extend(["missing","missing"])
                    else:
                        row.extend(file_and_path)
                table.Append(row)
        best_total_width = 0
        for table in tables:
            total_width = 0
            dc=wx.ClientDC(table)
            dc.Font = table.Font
            try:
                for col in range(table.ColumnCount):
                    text = table.GetColumn(col).Text
                    width = dc.GetTextExtent(text)[0]
                    for row in range(table.ItemCount):
                        text = table.GetItem(row,col).Text
                        width = max(width, dc.GetTextExtent(text)[0])
                    width += 16
                    table.SetColumnWidth(col, width)
                    total_width += width+4
            finally:
                dc.Destroy()
            best_total_width = max(best_total_width, total_width)
        table.SetMinSize((best_total_width, table.GetMinHeight()))
        my_frame.Fit()
        my_frame.Show()
        
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
            pathname = os.path.join(self.image_directory(), pathname)
            video_stream = ffmpeg.VideoStream(pathname)
            frame_count = video_stream.frame_count
            if frame_count == 0:
                print "Warning - no frame count detected"
                frame_count = 256
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
        header = ["Image name","Path","Filename"]
        ratio = [1.0,3.0,2.0]
        tags = self.get_metadata_tags()
        ratio += [1.0 for tag in tags]
        ratio = [x / sum(ratio) for x in ratio]
        header += tags 
        statistics = [header]
        m = workspace.measurements
        for fd in self.images:
            provider = workspace.image_set.get_image_provider(fd[FD_IMAGE_NAME].value)
            path, filename = os.path.split(provider.get_filename())
            name = provider.name
            row = [name, path, filename]
            metadata = self.get_filename_metadata(fd, filename, path)
            m.add_measurement('Image','FileName_'+name, filename)
            full_path = os.path.join(self.image_directory(),path)
            m.add_measurement('Image','PathName_'+name, full_path)
            for key in metadata:
                measurement = 'Metadata_%s'%(key)
                if not m.has_current_measurements('Image',measurement):
                    m.add_measurement('Image',measurement,metadata[key])
                elif metadata[key] != m.get_current_measurement('Image',measurement):
                    raise ValueError("Image set has conflicting %s metadata: %s vs %s"%
                                     (key, metadata[key], 
                                      m.get_current_measurement('Image',measurement)))
            for tag in tags:
                if metadata.has_key(tag):
                    row.append(metadata[tag])
                else:
                    row.append("")
            statistics.append(row)
        if workspace.frame:
            figure = workspace.create_or_find_figure(title="Load images, image set #%d"%(workspace.measurements.image_set_number+1),
                                                     subplots=(1,1))
            figure.subplot_table(0,0,statistics,ratio=ratio)

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
        if self.file_types in (FF_AVI_MOVIES,FF_OTHER_MOVIES):
            f = ffmpeg.open(path)
            index = f.get_frame_types().index["video"]
            if index == 0:
                raise ValueError("No video stream in %s"%pathname)
            frame_count = f.get_frame_count(0)
            return frame_count
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
        
        fd = self.images[0]    
        if fd[FD_METADATA_CHOICE]==M_FILE_NAME or fd[FD_METADATA_CHOICE]==M_BOTH:
            tokens = cpm.find_metadata_tokens(fd[FD_FILE_METADATA].value)
            cols += [('Image', 'Metadata_'+token, cpm.COLTYPE_VARCHAR_FILE_NAME) for token in tokens]
        
        if fd[FD_METADATA_CHOICE]==M_PATH or fd[FD_METADATA_CHOICE]==M_BOTH:
            tokens = cpm.find_metadata_tokens(fd[FD_PATH_METADATA].value)
            cols += [('Image', 'Metadata_'+token, cpm.COLTYPE_VARCHAR_PATH_NAME) for token in tokens]
        
        return cols
            
            
            
def is_image(filename):
    '''Determine if a filename is a potential image file based on extension'''
    ext = os.path.splitext(filename)[1].lower()
    if PILImage.EXTENSION.has_key(ext):
        return True
    return ext in ('.avi', '.mpeg', '.mat')
    


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
            imgdata = scipy.io.matlab.mio.loadmat(self.get_full_name(),
                                                  struct_as_record=True)
            return cpimage.Image(imgdata["Image"])
        img = PILImage.open(self.get_full_name())
        if img.mode=='I;16':
            # 16-bit image
            # deal with the endianness explicitly... I'm not sure
            # why PIL doesn't get this right.
            imgdata = numpy.fromstring(img.tostring(),numpy.uint8)
            imgdata.shape=(int(imgdata.shape[0]/2),2)
            imgdata = imgdata.astype(numpy.uint16)
            hi,lo = (0,1) if img.tag.prefix == 'MM' else (1,0)
            imgdata = imgdata[:,hi]*256 + imgdata[:,lo]
            img_size = list(img.size)
            img_size.reverse()
            new_img = imgdata.reshape(img_size)
            # The magic # for maximum sample value is 281
            if img.tag.has_key(281):
                img = new_img.astype(float) / img.tag[281][0]
            elif numpy.max(new_img) < 4096:
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
        pixel_data = self.__video_stream.read_rgb8().astype(float)/255.
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
