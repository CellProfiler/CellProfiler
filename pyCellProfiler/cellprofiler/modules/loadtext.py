'''<b>LoadText</b> loads metadata to be associated with image sets.
<hr>
The LoadText module loads a CSV file that supplies metadata values for
image sets, e.g. the plate name, well or treatment associated
with a given image, and possibly even the image filename itself.

A CSV (or comma-separated value) file is organized into rows and
columns. The lines of the file represent the rows (technically, each row
is terminated by a newline character: ASCII 10). Each field in a row is
separated by a comma. Text values may be optionally enclosed by double
quotes.

The LoadText module uses the first row of the file as a header. The fields
in this row provide the labels for each metadata column. Subsequent rows
provide the values for the image set. Certain fields have special connotations
that affect downstream processing:

<b>Labels that start with Image_FileName.</b>
A label that starts with "Image_FileName" supplies the file name of an image.
The image's name within CellProfiler appears afterward. For instance,
"Image_FileName_CY3", would supply the file name for the CY3-stained image.

<b>Labels that start with Image_PathName</b>.
A label that starts with "Image_PathName" supplies the path name of an image
relative to the base directory. The image's name within CellProfiler appears
afterward. For instance, "Image_PathName_CY3", would supply the path name
for the CY3-stained image.

<b>Labels that start with Metadata</b>.
A label that starts with "Metadata" can be used to match files loaded by
LoadImages. For instance, an experiment might require images created on
the same day to use an illumination correction image calculated from all
images from that day. Each row would have a "Metadata_Date" field and the
LoadImages module might extract the Metadata_Date field from the image
file name. The pipeline will match image sets with corresponding illumination
correction images based on matching Metadata_Date fields.

An example CSV file:
Image_FileName_FITC, Image_PathName_FITC, Metadata_Plate, Titration_NaCl_uM
"04923_d1.tif","2009-07-08","P-12345",750
"51265_d1.tif","2009-07-09","P-12345",2750

The first row loads the file, "2009-07-08/04923_d1.tif" for the FITC image.
The plate metadata is 'P-12345' and the NaCl titration used in the well
is 750 uM.
The second row has values, "2009-07-09/51265_d1.tif", 'P-12345' and 2750 uM.

The NaCl titration for the image will be recorded in the database and is
available for modules that use numeric metadata, such as CalculateStatistics.

See also <b>LoadImages</b>
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

__version = "$Revision$"

import csv
import hashlib
import numpy as np
import os
import sys
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.loadimages import LoadImagesImageProvider

DIR_DEFAULT_IMAGE = 'Default Image Directory'
DIR_DEFAULT_OUTPUT = 'Default Output Directory'
DIR_OTHER = 'Elsewhere...'
DIR_ALL = [DIR_DEFAULT_IMAGE, DIR_DEFAULT_OUTPUT,DIR_OTHER]

PATH_NAME = 'PathName'
FILE_NAME = 'FileName'
'''Reserve extra space in pathnames for batch processing name rewrites'''
PATH_PADDING = 20

###################################################################
#
# Helper functions for the header columns, Image_FileName_<image-name>
# and Image_PathName_<image-name>
#
# These need to be converted to FileName_<image-name> and
# PathName_<image-name> internally.
###################################################################

def header_to_column(field):
    '''Convert the field name in the header to a column name
    
    This function converts Image_FileName to FileName and 
    Image_PathName to PathName so that the output column names
    in the database will be Image_FileName and Image_PathName
    '''
    for name in (PATH_NAME, FILE_NAME):
        if field.startswith(cpmeas.IMAGE+'_'+name+'_'):
            return field[len(cpmeas.IMAGE)+1:]
    return field

def is_path_name_feature(feature):
    '''Return true if the feature name is a path name'''
    return feature.startswith(PATH_NAME+'_')

def is_file_name_feature(feature):
    '''Return true if the feature name is a file name'''
    return feature.startswith(FILE_NAME+'_')

def get_image_name(feature):
    '''Extract the image name from a feature name'''
    if is_path_name_feature(feature):
        return feature[len(PATH_NAME+'_'):]
    if is_file_name_feature(feature):
        return feature[len(FILE_NAME+'_'):]
    raise ValueError('"%s" is not a path feature or file name feature'%feature)

def make_path_name_feature(image):
    '''Return the path name feature, given an image name

    The path name feature is the name of the measurement that stores
    the image's path name.
    '''
    return PATH_NAME+'_'+image

def make_file_name_feature(image):
    '''Return the file name feature, given an image name
    
    The file name feature is the name of the measurement that stores
    the image's file name.
    '''
    return FILE_NAME+'_'+image
    
class LoadText(cpm.CPModule):
    
    module_name = "LoadText"
    category = 'File Processing'
    variable_revision_number = 2

    def create_settings(self):
        self.uuid = uuid.uuid4()
        self.csv_directory_choice = cps.Choice("CSV file location:", DIR_ALL, doc="""
            This is the folder that contains the CSV file. Choose "Default Image Directory"
            if the CSV file is in the default image directory. Choose "Default Output
            Directory" if the CSV file is in the default output directory. Choose
            "Elsewhere..." to specify a custom directory name. 
            
            Custom directory names that start with "." are relative to the default image directory. Names that
            start with "&" are relative to the default output directory.
            For example '&/../My_directory' looks in a directory called 'My_directory'
            at the same level as the output directory""")
        self.csv_custom_directory = cps.DirectoryPath("Enter the path to the CSV file:",
                                                      ".", doc = 
                                                      """What is the name of the CSV file's directory?""")
        self.csv_file_name = cps.FilenameText("Enter the name of the CSV file:",
                                              "None",doc="""
            This is the file name of the CSV file containing the data.""")
        self.wants_images = cps.Binary("Load images from CSV data?", True, doc="""
            Check this box to have LoadText load images using the Image_FileName field and,
            if it appears in the CSV file, the Image_PathName fields.""")
        self.image_directory_choice = cps.Choice("Image directory location:",
                                                 DIR_ALL, doc="""
            This is the base directory used for paths to images. Path names to image
            files are relative to this directory. Choose "Default Image Directory" to
            make the default image directory the base directory. Choose "Default Output
            Directory" to make the default output directory the base directory. Choose
            "Elsewhere..." to specify a custom directory name.
            
            Custom directory names that start with "." are relative to the default image directory. Names that
            start with "&" are relative to the default output directory.
            For example '&/../My_directory' looks in a directory called 'My_directory'
            at the same level as the output directory""")
        self.image_custom_directory = cps.DirectoryPath("Enter the path to the images:",
                                                        ".", doc = 
                                                        """What is the name of the image directory?""")
        self.wants_image_groupings = cps.Binary("Group images by metadata?", False)
        self.metadata_fields = cps.MultiChoice("Select metadata fields for grouping:", None)
        self.wants_rows = cps.Binary("Process just a range of rows?",
                                     False, doc="""
            Check this box if you want to process a subset of the rows in the CSV file.
            Rows are numbered starting at 1 (but do not count the header line). 
            LoadText will process up to and including the end row.
            This option can be used to break the image sets in an experiment into groups
            that can be processed by different nodes on a computing cluster.""")
        self.row_range = cps.IntegerRange("Rows to process:",
                                          (1,100000),1, doc = 
                                          """Enter the row numbers of the first and last row to be processed.""")

    def settings(self):
        return [self.csv_directory_choice, self.csv_custom_directory,
                self.csv_file_name, self.wants_images, self.image_directory_choice,
                self.image_custom_directory, self.wants_rows,
                self.row_range, self.wants_image_groupings, 
                self.metadata_fields]

    def validate_module(self, pipeline):
        csv_path = self.csv_path
        if not os.path.isfile(csv_path):
            raise cps.ValidationError("No such CSV file: %s"%csv_path,
                                      self.csv_file_name)
        else:
            try:
                self.get_header()
            except:
                raise cps.ValidationError("The CSV file, %s, is not in the proper format. See this module's help for details on CSV format." %
                                          self.csv_path, self.csv_file_name)

    def visible_settings(self):
        result = [self.csv_directory_choice]
        if self.csv_directory_choice == DIR_OTHER:
            result += [self.csv_custom_directory]
        result += [self.csv_file_name, self.wants_images]
        if self.wants_images.value:
            result += [self.image_directory_choice]
            if self.image_directory_choice == DIR_OTHER:
                result += [self.image_custom_directory]
            result += [self.wants_image_groupings]
            if self.wants_image_groupings.value:
                result += [self.metadata_fields]
                if os.path.isfile(self.csv_path):
                    fields = [field[len("Metadata_"):] 
                              for field in self.get_header()
                              if field.startswith("Metadata_")]
                    self.metadata_fields.choices = fields
                else:
                    self.metadata_fields.choices = [ "No CSV file"]
                
        result += [self.wants_rows]
        if self.wants_rows.value:
            result += [self.row_range]
        return result

    @property
    def csv_path(self):
        '''The path and file name of the CSV file to be loaded'''
        if self.csv_directory_choice == DIR_DEFAULT_IMAGE:
            path = cpprefs.get_default_image_directory()
        elif self.csv_directory_choice == DIR_DEFAULT_OUTPUT:
            path = cpprefs.get_default_output_directory()
        else:
            path = cpprefs.get_absolute_path(self.csv_custom_directory.value)
        return os.path.join(path, self.csv_file_name.value)
    
    @property
    def image_path(self):
        if self.image_directory_choice == DIR_DEFAULT_IMAGE:
            path = cpprefs.get_default_image_directory()
        elif self.image_directory_choice == DIR_DEFAULT_OUTPUT:
            path = cpprefs.get_default_output_directory()
        else:
            path = cpprefs.get_absolute_path(self.image_custom_directory.value)
        return path
    
    @property
    def legacy_field_key(self):
        '''The key to use to retrieve the metadata from the image set list'''
        return 'LoadTextMetadata_%d'%self.module_num
    
    def get_header(self):
        '''Read the header fields from the csv file
        
        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        '''
        fd = open(self.csv_path, 'rb')
        reader = csv.reader(fd)
        header = reader.next()
        fd.close()
        return [header_to_column(column) for column in header]
        
    def other_providers(self, group):
        '''Get name providers from the CSV header'''
        if group=='imagegroup' and self.wants_images.value:
            try:
                header = self.get_header()
                return [get_image_name(field)
                        for field in header
                        if is_file_name_feature(field)]
            except Exception,e:
                return []
        return []
    
    def prepare_run(self, pipeline, image_set_list, frame):
        '''Load the CSV file at the outset and populate the image set list'''
        if pipeline.in_batch_mode():
            return True
        fd = open(self.csv_path, 'rb')
        reader = csv.reader(fd)
        header = [header_to_column(column) for column in reader.next()]
        if self.wants_rows.value:
            # skip initial rows
            n_to_skip = self.row_range.min-1
            i=0
            for i in range(n_to_skip):
                reader.next()
            i += 1
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError("Row # %d has the wrong number of elements: %d. Expected %d"%
                                     (i,len(row),len(header)))
                rows.append(row)
                if i == self.row_range.max - 1:
                    break
                i += 1
        else:
            rows = [row for row in reader]
        fd.close()
        #
        # Arrange the metadata in columns
        #
        dictionary = {}
        metadata = {}
        images = {}
        for i in range(len(header)):
            column = [row[i] for row in rows]
            if header[i].startswith('Metadata_'):
                key = header[i][len('Metadata_'):]
                column = np.array(column)
                dictionary[header[i]] = best_cast(column)
                metadata[key] = dictionary[header[i]]
            elif (self.wants_images.value and
                  is_file_name_feature(header[i])):
                column = np.array(column)
                image = get_image_name(header[i])
                if not images.has_key(image):
                    images[image] = {}
                images[image][FILE_NAME] = column
                dictionary[header[i]] = column
            elif (self.wants_images.value and
                  is_path_name_feature(header[i])):
                column = np.array(column)
                image = get_image_name(header[i])
                if not images.has_key(image):
                    images[image] = {}
                images[image][PATH_NAME] = column
                dictionary[header[i]] = column
            else:
                dictionary[header[i]] = best_cast(column)
        
        for image in images.keys():
            if not images[image].has_key(FILE_NAME):
                raise ValueError('The CSV file has an Image_PathName_%s metadata column without a corresponding Image_FileName_%s column'%
                                 (image,image))
        #
        # Populate the image set list 
        #
        use_key = (image_set_list.associating_by_key != False)
        for i in range(len(rows)):
            if len(metadata) and use_key:
                key = {}
                for k in metadata.keys():
                    md = metadata[k][i]
                    if hasattr(md, "dtype"):
                        if md.dtype.name.startswith('string'):
                            md = str(md)
                        elif md.dtype.name.startswith('int'):
                            md = int(md)
                        elif md.dtype.name.startswith('float'):
                            md = float(md)
                    key[k] = md
                image_set = image_set_list.get_image_set(key)
            else:
                image_set = image_set_list.get_image_set(i)
        #
        # Hide the measurements in the image_set_list
        #
        image_set_list.legacy_fields[self.legacy_field_key] = dictionary
        return True
    
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
        dictionary = image_set_list.legacy_fields[self.legacy_field_key]
        path_keys = [key for key in dictionary.keys()
                     if is_path_name_feature(key)]
        for key in path_keys:
            dictionary[key] = np.array([fn_alter_path(path) 
                                        for path in dictionary[key]])
        
        if self.csv_directory_choice == DIR_DEFAULT_IMAGE:
            self.csv_directory_choice.value = DIR_OTHER
            self.csv_custom_directory.value = cpprefs.get_default_image_directory()
        elif self.csv_directory_choice == DIR_DEFAULT_OUTPUT:
            self.csv_directory_choice.value = DIR_OTHER
            self.csv_custom_directory.value = cpprefs.get_default_output_directory()
        self.csv_custom_directory.value = fn_alter_path(self.csv_custom_directory.value)
        self.image_custom_directory.value = \
            fn_alter_path(self.image_custom_directory.value)
        return True
    
    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        dictionary = image_set_list.legacy_fields[self.legacy_field_key]
        path_base = self.image_path
        image_names = self.other_providers('imagegroup')
        if self.wants_images.value:
            for image_number in image_numbers:
                index = image_number -1
                image_set = image_set_list.get_image_set(index)
                for image_name in image_names:
                    path_name_feature = make_path_name_feature(image_name)
                    if dictionary.has_key(path_name_feature):
                        path = os.path.join(path_base, 
                                            dictionary[path_name_feature][index])
                    else:
                        path = path_base
                    file_name_feature = make_file_name_feature(image_name)
                    filename = dictionary[file_name_feature][index]
                    ip = LoadImagesImageProvider(image_name, path, filename)
                    image_set.providers.append(ip)
            
    def run(self, workspace):
        '''Populate the image measurements on each run iteration'''
        dictionary = workspace.image_set_list.legacy_fields[self.legacy_field_key]
        statistics = []
        image_set_keys = workspace.image_set.keys
        if (len(image_set_keys.keys()) > 1 or
            image_set_keys.keys()[0]!= 'number'):
            # Match keys against each dictionary entry
            for index in range(len(dictionary.values()[0])):
                failure = False
                for key in image_set_keys.keys():
                    md_key = "Metadata_%s"%(key)
                    column = dictionary[md_key]
                    if column[index] != image_set_keys[key]:
                        failure = True
                        break
                if not failure:
                    break
        else:
            index = workspace.measurements.image_set_number-1
        for feature_name in dictionary.keys():
            value = dictionary[feature_name][index]
            workspace.measurements.add_image_measurement(feature_name, value)
            statistics += [[feature_name, value]]
        #
        # Calculate the MD5 hash of every image
        #
        for image_name in self.other_providers('imagegroup'):
            md5 = hashlib.md5()
            pixel_data = workspace.image_set.get_image(image_name).pixel_data
            md5.update(np.ascontiguousarray(pixel_data).data)
            workspace.measurements.add_image_measurement(
                'MD5Digest_'+image_name,
                md5.hexdigest())
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics,[.3,.7])
    
    def get_groupings(self, image_set_list):
        '''Return the image groupings of the image sets

        See CPModule for documentation
        '''
        if (self.wants_images.value and 
            self.wants_image_groupings.value and
            len(self.metadata_fields.selections) > 0):
            keys = self.metadata_fields.selections
            if len(keys) == 0:
                return None
            return image_set_list.get_groupings(keys)
        return None

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements output by this module'''
        try:
            fd = open(self.csv_path, 'rb')
            reader = csv.reader(fd)
            header = [header_to_column(x) for x in reader.next()]
        except:
            return []
        coltypes = [cpmeas.COLTYPE_INTEGER]*len(header)
        collen = [0]*len(header)
        for row in reader:
            for field,index in zip(row,range(len(row))):
                len_field = len(field)
                if field.startswith(PATH_NAME) and self.wants_images:
                    # Account for possible rewrite of the pathname
                    # in batch data
                    len_field = max(cpmeas.PATH_NAME_LENGTH, 
                                    len_field + PATH_PADDING)
                if coltypes[index] == cpmeas.COLTYPE_INTEGER:
                    try:
                        int(field)
                        continue
                    except ValueError:
                        coltypes[index] = cpmeas.COLTYPE_FLOAT
                if coltypes[index] == cpmeas.COLTYPE_FLOAT:
                    try:
                        float(field)
                        continue
                    except ValueError:
                        coltypes[index] = cpmeas.COLTYPE_VARCHAR_FORMAT%len(field)
                if collen[index] < len(field):
                    collen[index] = len(field)
                    coltypes[index] = cpmeas.COLTYPE_VARCHAR_FORMAT%len(field)
        image_names = self.other_providers('imagegroup')
        return ([(cpmeas.IMAGE, colname, coltype)
                 for colname, coltype in zip(header, coltypes)] +
                [(cpmeas.IMAGE, 'MD5Digest_'+image_name,
                  cpmeas.COLTYPE_VARCHAR_FORMAT % 32)
                 for image_name in image_names])

    def get_categories(self, pipeline, object_name):
        if object_name != cpmeas.IMAGE:
            return []
        try:
            header = self.get_header()
            return [x.split('_')[0] for x in header]
        except:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name != cpmeas.IMAGE:
            return []
        try:
            header = self.get_header()
            return ['_'.join(x.split('_')[1:])
                    for x in header
                    if x.split('_')[0] == category]
        except:
            return []
        
    def change_causes_prepare_run(self, setting):
        '''Check to see if changing the given setting means you have to restart
        
        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to do "prepare_run".
        '''
        if self.wants_images or setting == self.wants_images:
            return True
        return False

    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            sys.stderr.write("Warning: the format and purpose of LoadText has changed substantially\n")
            text_file_name = setting_values[0]
            field_name = setting_values[1]
            path_name = setting_values[2]
            if path_name=='.':
                path_choice = DIR_DEFAULT_IMAGE
            elif path_name == '&':
                path_choice = DIR_DEFAULT_OUTPUT
            else:
                path_choice = DIR_OTHER
            setting_values = [path_choice, path_name, text_file_name,
                              cps.NO, DIR_DEFAULT_IMAGE, '.',
                              cps.NO, "1,100000"]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = setting_values + [cps.NO, ""]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab 

def best_cast(sequence):
    '''Return the best cast (integer, float or string) of the sequence
    
    sequence - a sequence of strings
    
    Try casting all elements to integer and float, returning a numpy
    array of values. If all fail, return a numpy array of strings.
    '''
    
    try:
        return np.array([int(x) for x in sequence])
    except ValueError:
        try:
            return np.array([float(x) for x in sequence])
        except ValueError:
            return np.array(sequence)
