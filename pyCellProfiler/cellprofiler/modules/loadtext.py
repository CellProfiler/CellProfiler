'''loadtext.py - The LoadText module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
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

class LoadText(cpm.CPModule):
    '''Short description:
The LoadText module loads metadata to be associated with image sets.
*********************************************************************
The LoadText module loads a CSV file that supplies metadata values for
image sets, for instance, the plate name, well or treatment associated
with a given image.

A CSV (or comma-separated value) file is a file organized into rows and
columns. The lines of the file represent the rows (technically, each row
is terminated by a newline character: ASCII 10). Each field in a row is
separated by a comma. Text values may be optionally enclosed by double
quotes.

The LoadText module uses the first row of the file as a header. The fields
in this row provide the labels for each metadata column. Subsequent rows
provide the values for the image set. Certain fields have special connotations
that affect downstream processing:

Labels that start with Image_FileName.
A label that starts with "Image_FileName" supplies the file name of an image.
The image's name within CellProfiler appears afterward. For instance,
"Image_FileName_CY3", would supply the file name for the CY3-stained image.

Labels that start with Image_PathName.
A label that starts with "Image_PathName" supplies the path name of an image
relative to the base directory. The image's name within CellProfiler appears
afterward. For instance, "Image_PathName_CY3", would supply the path name
for the CY3-stained image.

Labels that start with Metadata.
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

Settings:
Where is the CSV file located?
This is the folder that contains the CSV file. Choose "Default Image Directory"
if the CSV file is in the default image directory. Choose "Default Output
Directory" if the CSV file is in the default output directory. Choose
"Elsewhere..." to specify a custom directory name. Custom directory names
that start with "." are relative to the default image directory. Names that
start with "&" are relative to the default output directory.

What is the name of the CSV file?
This is the file name of the CSV file containing the data.

Load images from CSV data?
Check this box to have LoadText load images using the Image_FileName and,
if it appears in the CSV file, the Image_PathName fields.

Where are the images located?
This is the base directory used for paths to images. Path names to image
files are relative to this directory. Choose "Default Image Directory" to
make the default image directory the base directory. Choose "Default Output
Directory" to make the default output directory the base directory. Choose
"Elsewhere..." to specify a custom directory name.

Do you want to specify a range of rows to be processed?
Check this box if you want to process only some of the rows in the CSV file.
Rows are numbered starting at 1. LoadText will process up to and including
the end row.
This option can be used to break the image sets in an experiment into groups
that can be processed by different nodes in a cluster.
'''

    category = 'File Processing'
    variable_revision_number = 2

    def create_settings(self):
        self.module_name = "LoadText"
        self.uuid = uuid.uuid4()
        self.csv_directory_choice = cps.Choice("Where is the CSV file located?",
                                               DIR_ALL)
        self.csv_custom_directory = cps.DirectoryPath("What is the name of the CSV file's directory?",
                                                      ".")
        self.csv_file_name = cps.FilenameText("What is the name of the CSV file?",
                                              "None")
        self.wants_images = cps.Binary("Load images from CSV data?", True)
        self.image_directory_choice = cps.Choice("Where area the images located?",
                                                 DIR_ALL)
        self.image_custom_directory = cps.DirectoryPath("What is the name of the image directory?",
                                                        ".")
        self.wants_image_groupings = cps.Binary("Do you want to group images by metadata?", False)
        self.metadata_fields = cps.MultiChoice("What metadata fields should be used to group?", None)
        self.wants_rows = cps.Binary("Do you want to specify a range of rows to be processed?",
                                     False)
        self.row_range = cps.IntegerRange("Enter the row numbers of the first and last row to be processed",
                                          (1,100000),1)

    def settings(self):
        return [self.csv_directory_choice, self.csv_custom_directory,
                self.csv_file_name, self.wants_images, self.image_directory_choice,
                self.image_custom_directory, self.wants_rows,
                self.row_range, self.wants_image_groupings, 
                self.metadata_fields]

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
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

    def test_valid(self, pipeline):
        super(LoadText, self).test_valid(pipeline)
        csv_path = self.csv_path
        if not os.path.isfile(csv_path):
            raise cps.ValidationError("No such CSV file: %s"%csv_path,
                                      self.csv_file_name) 

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
                fields = [field[len("Metadata_"):] 
                          for field in self.get_header()
                          if field.startswith("Metadata_")]
                self.metadata_fields.choices = fields
                
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
        return header
        
    def get_name_providers(self, group):
        '''Get name providers from the CSV header'''
        if group=='imagegroup' and self.wants_images.value:
            try:
                header = self.get_header()
                return [field[len('Image_FileName_'):]
                        for field in header
                        if field.startswith('Image_FileName_')]
            except Exception,e:
                return []
        return []
    
    def prepare_run(self, pipeline, image_set_list, frame):
        '''Load the CSV file at the outset and populate the image set list'''
        if pipeline.in_batch_mode():
            return True
        fd = open(self.csv_path, 'rb')
        reader = csv.reader(fd)
        header = reader.next()
        if self.wants_rows.value:
            # skip initial rows
            n_to_skip = self.row_range.min-1
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
                metadata[key] = column
                dictionary[header[i]] = column
            elif (self.wants_images.value and
                  header[i].startswith('Image_FileName_')):
                column = np.array(column)
                image = header[i][len('Image_FileName_'):]
                if not images.has_key(image):
                    images[image] = {}
                images[image][FILE_NAME] = column
                dictionary[header[i]] = column
            elif (self.wants_images.value and
                  header[i].startswith('Image_PathName_')):
                column = np.array(column)                
                image = header[i][len('Image_PathName_'):]
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
        for i in range(len(rows)):
            if len(metadata):
                key = {}
                for k in metadata.keys():
                    key[k] = metadata[k][i]
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
                     if key.startswith('Image_PathName_')]
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
        image_names = self.get_name_providers('imagegroup')
        if self.wants_images.value:
            for image_number in image_numbers:
                index = image_number -1
                image_set = image_set_list.get_image_set(index)
                for image_name in image_names:
                    path_name_feature = 'Image_PathName_%s'%image_name
                    if dictionary.has_key(path_name_feature):
                        path = os.path.join(path_base, 
                                            dictionary[path_name_feature][index])
                    else:
                        path = path_base
                    filename = dictionary['Image_FileName_%s'%image_name][index]
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
        for image_name in self.get_name_providers('imagegroup'):
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
        fd = open(self.csv_path, 'rb')
        reader = csv.reader(fd)
        header = reader.next()
        coltypes = [cpmeas.COLTYPE_INTEGER]*len(header)
        collen = [0]*len(header)
        for row in reader:
            for field,index in zip(row,range(len(row))):
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
        image_names = self.get_name_providers('imagegroup')
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
        
    def check_for_prepare_run_setting(self, setting):
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
