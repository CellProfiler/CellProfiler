'''<b>Load Data</b> loads text or numerical data to be associated with images, and 
can also load images specified by file names
<hr>
This module loads a file that supplies text or numerical data associated with the images to be processed, e.g., sample names, plate names, well 
identifiers, or even a list of image filenames to be processed in the analysis run.

The module currently reads files in CSV (comma-separated values) format. 
These files can be produced by spreadsheet programs and are organized into rows and
columns. The lines of the file represent the rows. (Technically, each row
is terminated by the newline character ASCII 10.) Each field in a row is
separated by a comma. Text values may be optionally enclosed by double
quotes. The <b>LoadData</b> module uses the first row of the file as a header. The fields
in this row provide the labels for each column of data. Subsequent rows
provide the values for each image cycle. 
<br><br>
There are many reasons why you might want to prepare a CSV file and load it
via <b>LoadData</b>; using particular names for columns allows special 
functionality for some downstream modules:

<ul>

<li>
<i>Columns with any name</i>. Any data loaded via <b>LoadData</b> will be exported 
as a per-image measurement along with CellProfiler-calculated data. This is a
convenient way for you to add data from your own sources to the files exported by
CellProfiler.
</li>

<li>
<i>Columns whose name begins with Image_FileName.</i>
A column whose name begins with "Image_FileName" can be used to supply the file 
name of an image that you want to load.
The image's name within CellProfiler appears afterward. For instance,
"Image_FileName_CY3" would supply the file name for the CY3-stained image, and
choosing the <i>Load images based on this data?</i> option allows the CY3 images to be 
selected later in the pipeline.
</li>

<li>
<i>Columns whose name begins with Image_PathName</i>.
A column whose name begins with "Image_PathName" can be used to supply the 
path name of an image that you want to load (relative to the base folder). 
The image's name within CellProfiler appears
afterward. For instance, "Image_PathName_CY3" would supply the path names
for the CY3-stained images. This is optional; if all image files are in the base 
folder, this column is not needed. 
</li>

<li>
<i>Columns whose name begins with Metadata</i>. A column whose name begins with 
"Metadata" can be used to group or associate files loaded by <b>LoadImages</b>.
<p>For instance, an experiment might require that images created on the same day 
use an illumination correction function calculated from all images from that day, 
and furthermore, that the date be captured in the file names for the individual image 
sets and in a .csv file specifying the illumination correction functions. 
<p>In this case, if the illumination correction images are loaded with the 
<b>LoadData</b> module, the file should have a "Metadata_Date" 
column which contains the date identifiers. Similarly, if the individual images 
are loaded using the <b>LoadImages</b> module, <b>LoadImages</b> should be set to extract the 
<Date> metadata field from the file names (see <b>LoadImages</b> for more details 
on how to do so). The pipeline will then match the individual image sets with 
their corresponding illumination correction functions based on matching 
Metadata_Date fields.
</li>

<li>
<i>Columns that contain dose-response or positive/negative control information</i>. 
The <b>CalculateStatistics</b> module can calculate metrics of assay quality for 
an experiment if provided with information about which images represent positive
and negative controls and/or what dose of treatment has been used for which images.
This information is provided to <b>CalculateStatistics</b> via the <b>LoadData</b> 
module, using particular formats described in the help for <b>CalculateStatistics</b>. 
</li>
</ul>

<h3>Example CSV file:</h3><br>
<tt><table border="0">
<tr><td>Image_FileName_FITC,</td><td>Image_PathName_FITC,</td><td>Metadata_Plate,</td><td>Titration_NaCl_uM</td></tr><br>
<tr><td>"04923_d1.tif",</td><td>"2009-07-08",</td><td>"P-12345",</td><td>750</td></tr>
<tr><td>"51265_d1.tif",</td><td>"2009-07-09",</td><td>"P-12345",</td><td>2750</td></tr>
</table></tt>

After the first row of header information (the column names), the first 
image-specific row specifies the file, "2009-07-08/04923_d1.tif" for the FITC 
image (2009-07-08 is the name of the subfolder that contains the image, 
relative to the default input folder). The plate metadata is "P-12345" and 
the NaCl titration used in the well is 750 uM. The second image-specific row 
has the values "2009-07-09/51265_d1.tif", "P-12345'\" and 2750 uM. The NaCl 
titration for the image is available for modules that use numeric metadata, 
such as <b>CalculateStatistics</b>.

See also <b>LoadImages</b> and <b>CalculateStatistics</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version = "$Revision$"

import csv
import hashlib
import numpy as np
import os
import sys

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.loadimages import LoadImagesImageProvider

DIR_DEFAULT_IMAGE = 'Default Input Folder'
DIR_DEFAULT_OUTPUT = 'Default Output Folder'
DIR_NONE = 'None'
DIR_OTHER = 'Elsewhere...'
DIR_ALL = [DIR_DEFAULT_IMAGE, DIR_DEFAULT_OUTPUT, DIR_NONE, DIR_OTHER]

PATH_NAME = 'PathName'
FILE_NAME = 'FileName'
'''Reserve extra space in pathnames for batch processing name rewrites'''
PATH_PADDING = 20

'''Cache of header columns for files'''
header_cache = {}
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
    
class LoadData(cpm.CPModule):
    
    module_name = "LoadData"
    category = 'File Processing'
    variable_revision_number = 3

    def create_settings(self):
        self.csv_directory_choice = cps.Choice("File location", DIR_ALL, doc="""
            The folder that contains the CSV file. You can choose among the following options:
            <ul><li><i>Default Input Folder</i>: 
            The CSV file is in the default input folder.</li>
            <li><i>Default Output
            Folder:</i> The CSV file is in the default output folder.</li>
            <li><i>Elsewhere...</i>: You can enter a custom folder name.</li></ul> 
            
            <p>Custom folder names that start with "." are relative to the default input folder. Names that
            start with "&" are relative to the default output folder. Two periods ".." specify to go 
            up one folder level. For example, "./CSVfiles" looks for a folder called <i>CSVfiles</i> that is 
            contained within the default input folder,
            and <i>&/../My_folder" looks in a folder called "My_folder"
            at the same level as the output folder.""")
        
        self.csv_custom_directory = cps.DirectoryPath("Custom file location",
            ".", doc = 
            """<i>(Used only if the file location is specified as Elsewhere)</i><br>
            Enter the pathname to the CSV file to be loaded. The pathname can referenced 
            with respect to the default output folder with a period (".") or 
            the default input folder with an ampersand ("&") as the root folder.""")
        
        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.csv_path
        
        def set_directory_fn(path):
            if os.path.abspath(path) == cpprefs.get_default_image_directory():
                self.csv_directory_choice.value = DIR_DEFAULT_IMAGE
            elif os.path.abspath(path) == cpprefs.get_default_output_directory():
                self.csv_directory_choice.value = DIR_DEFAULT_OUTPUT
            else:
                self.csv_directory_choice.value = DIR_OTHER
                self.csv_custom_directory.value = path
                
        self.csv_file_name = cps.FilenameText(
            "Name of the file",
            "None",
            doc="""Provide the file name of the CSV file containing the data.""",
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            browse_msg = "Choose CSV file",
            exts = [("Data file (*.csv)","*.csv"),("All files (*.*)","*.*")]
        )
        
        self.wants_images = cps.Binary("Load images based on this data?", True, doc="""
            Check this box to have <b>LoadData</b> load images using the <i>Image_FileName</i> field and the 
            <i>Image_PathName</i> fields (the latter is optional).""")
        
        self.image_directory_choice = cps.Choice("Base image location",
                                                 DIR_ALL, doc="""
            The parent (base) folder where images are located. If images are 
            contained in subfolders, then the file you load with this module should 
            contain a column with path names relative to the base image folder (see 
            the general help for this module for more details). Again, you can choose among the following options:
            <ul><li><i>Default Input Folder:</i> Make the default input folder the base folder.</li>
            <li><i>Default Output Folder:</i> Make the default output folder the base folder.</li>
            <li><i>Elsewhere...:</i> You can enter a custom folder name.</li>
            <li><i>None:</i> You have an Image_PathName field that supplies an absolute path.</li></ul>
            
            <p>Custom folder names that start with "." are relative to the default input folder. Names that
            start with "&" are relative to the default output folder. Two periods ".." specify to go 
            up one folder level. For example, "./CSVfiles" looks for a folder called <i>CSVfiles</i> that is 
            contained within the default input folder,
            and "&/../My_folder" looks in a folder called <i>My_folder</i>
            at the same level as the output folder.""")
        
        self.image_custom_directory = cps.DirectoryPath("Custom base image location",
                                                        ".", doc = 
            """<i>(Used only if the base image location is specified as Elsewhere)</i><br>
            Enter the pathname to the images to be loaded. The pathname can referenced 
            with respect to the default output folder with a period (".") or 
            the default input folder with an ampersand ("&") as the root folder.""")
        
        self.wants_image_groupings = cps.Binary("Group images by metadata?", False,doc = """
            Use this option to break the image sets in an experiment into groups
            that can be processed by different nodes on a computing cluster. Each set of
            files that share your selected metadata tags will be processed
            together. See <b>CreateBatchFiles</b> for details on submitting a 
            CellProfiler pipeline to a computing cluster for processing.""")
        
        self.metadata_fields = cps.MultiChoice("Select metadata fields for grouping", None,doc="""
            <i>(Used only if images are to be grouped by metadata)</i><br>
            Select the fields that you want group the image files by here. Multiple tags may be selected. For
            example, if a set of images had metadata for <i>Run</i>,<i>Plate</i>,<i>Well</i> and
            <i>Site</i>, selecting <i>Run</i> and <i>Plate</i> will create groups containing 
            images that share the same [<i>Run</i>,<i>Plate</i>] pair of fields.""")
        
        self.wants_rows = cps.Binary("Process just a range of rows?",
                                     False, doc="""
            Check this box if you want to process a subset of the rows in the CSV file.
            Rows are numbered starting at 1 (but do not count the header line). 
            <b>LoadData</b> will process up to and including the end row.""")
        self.row_range = cps.IntegerRange("Rows to process",
                                          (1,100000),1, doc = 
                                          """<i>(Used only if a range of rows is to be specified)</i><br>Enter the row numbers of the first and last row to be processed.""")

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
        if cpprefs.get_data_file() is not None:
            return cpprefs.get_data_file()
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
        elif self.image_directory_choice == DIR_NONE:
            path = ""
        else:
            path = cpprefs.get_absolute_path(self.image_custom_directory.value)
        return path
    
    @property
    def legacy_field_key(self):
        '''The key to use to retrieve the metadata from the image set list'''
        return 'LoadTextMetadata_%d'%self.module_num

    def get_cache_info(self):
        '''Get the cached information for the data file'''
        global header_cache
        entry = header_cache.get(self.csv_path, dict(ctime=0))
        ctime = os.stat(self.csv_path).st_ctime
        if ctime > entry["ctime"]:
            entry = header_cache[self.csv_path] = {}
            entry["ctime"] = ctime
        return entry
        
    def get_header(self):
        '''Read the header fields from the csv file
        
        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        '''
        entry = self.get_cache_info()
        if entry.has_key("header"):
            return entry["header"]
            
        fd = open(self.csv_path, 'rb')
        reader = csv.reader(fd)
        header = reader.next()
        fd.close()
        entry["header"] = [header_to_column(column) for column in header]
        return entry["header"]
        
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
            if os.path.exists(self.csv_path):
                return True
            raise ValueError(('''Can't find the CSV file, "%s". ''' 
                              '''Please check that the name matches exactly, '''
                              '''including the case''') % self.csv_path)
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
        # Check for correct # of columns
        #
        for i, row in enumerate(rows):
            if len(row) != len(header):
                text = ('Error on line %d of %s.\n'
                        '\n"%s"\n'
                        '%d rows found, expected %d') % (
                            i+2, self.csv_file_name.value,
                            ','.join(row),
                            len(row), len(header))
                raise ValueError(text)
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
        if self.wants_images:
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
        else:
            self.csv_custom_directory.value = cpprefs.get_absolute_path(
                self.csv_custom_directory.value)
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
                        path = dictionary[path_name_feature][index]
                        if self.image_directory_choice != DIR_NONE:
                            path = os.path.join(path_base, path)
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
                    if dictionary.has_key(md_key):
                        column_value = dictionary[md_key][index]
                        isk = image_set_keys[key]
                        if isinstance(column_value, (int, float)):
                            try:
                                if float(isk) == column_value:
                                    continue
                            except:
                                pass
                        if column_value != isk:
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
        # Add a metadata well measurement if only row and column exist
        #
        tokens = [feature for category, feature in
                  [x.split('_',1) for x in dictionary.keys()]
                  if category == cpmeas.C_METADATA]
        if cpmeas.FTR_WELL not in tokens:
            row_tokens = [x for x in tokens if cpmeas.is_well_row_token(x)]
            col_tokens = [x for x in tokens if cpmeas.is_well_column_token(x)]
            if len(row_tokens) > 0 and len(col_tokens) > 0:
                md_well = '_'.join((cpmeas.C_METADATA, cpmeas.FTR_WELL))
                row = dictionary['_'.join((cpmeas.C_METADATA, row_tokens[0]))]
                col = dictionary['_'.join((cpmeas.C_METADATA, col_tokens[0]))]
                row = row[index]
                col = col[index]
                if isinstance(col, int):
                    col = "%02d" % col
                well = row + col
                workspace.measurements.add_image_measurement(md_well, well)
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
        entry = None
        try:
            entry = self.get_cache_info()
            if entry.has_key("measurement_columns"):
                return entry["measurement_columns"]
            fd = open(self.csv_path, 'rb')
            reader = csv.reader(fd)
            header = [header_to_column(x) for x in reader.next()]
        except:
            if entry is not None:
                entry["measurement_columns"] = []
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
                        if isinstance(int(field), long):
                            # "integers" that don't fit are saved as strings
                            coltypes[index] = cpmeas.COLTYPE_VARCHAR_FORMAT % len(field)
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
        result = ([(cpmeas.IMAGE, colname, coltype)
                   for colname, coltype in zip(header, coltypes)] +
                  [(cpmeas.IMAGE, 'MD5Digest_'+image_name,
                    cpmeas.COLTYPE_VARCHAR_FORMAT % 32)
                   for image_name in image_names])
        #
        # Try to make a well column out of well row and well column
        #
        well_column = None
        well_row_column = None
        well_col_column = None
        for column in result:
            category, feature = column[1].split('_',1)
            if category == cpmeas.C_METADATA:
                if cpmeas.is_well_column_token(feature):
                    well_col_column = column
                elif cpmeas.is_well_row_token(feature):
                    well_row_column = column
                elif feature.lower() == cpmeas.FTR_WELL.lower():
                    well_column = column
        if (well_column is None and well_row_column is not None and
            well_col_column is not None):
            length = cpmeas.get_length_from_varchar(well_row_column[2])
            if well_col_column[2] == cpmeas.COLTYPE_INTEGER:
                length += 2
            else:
                length += cpmeas.get_length_from_varchar(well_col_column[2])
            result += [(cpmeas.IMAGE, 
                        '_'.join((cpmeas.C_METADATA, cpmeas.FTR_WELL)),
                        cpmeas.COLTYPE_VARCHAR_FORMAT % length)]
        entry["measurement_columns"] = result
        return result

    def get_categories(self, pipeline, object_name):
        if object_name != cpmeas.IMAGE:
            return []
        try:
            columns = self.get_measurement_columns(pipeline)
            result = set([column[1].split('_')[0] for column in columns])
            return list(result)
        except:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name != cpmeas.IMAGE:
            return []
        try:
            columns = self.get_measurement_columns(pipeline)
            result = [feature for c, feature in
                      [column[1].split('_',1) for column in columns]
                      if c == category]
            return result
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
            module_name = self.module_name
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = setting_values + [cps.NO, ""]
            variable_revision_number = 2
        if variable_revision_number == 2 and (not from_matlab):
            if setting_values[0].startswith("Default Image"):
                setting_values = [DIR_DEFAULT_IMAGE] + setting_values[1:]
            elif setting_values[0].startswith("Default Output"):
                setting_values = [DIR_DEFAULT_OUTPUT] + setting_values[1:]
            if setting_values[4].startswith("Default Image"):
                setting_values = (setting_values[:4] + [DIR_DEFAULT_IMAGE] + 
                                  setting_values[5:])
            elif setting_values[4].startswith("Default Output"):
                setting_values = (setting_values[:4] + [DIR_DEFAULT_OUTPUT] + 
                                  setting_values[5:])
            variable_revision_number = 3
        if variable_revision_number == 3 and (not from_matlab):
            module_name = self.module_name
        return setting_values, variable_revision_number, from_matlab 

LoadText = LoadData

def best_cast(sequence):
    '''Return the best cast (integer, float or string) of the sequence
    
    sequence - a sequence of strings
    
    Try casting all elements to integer and float, returning a numpy
    array of values. If all fail, return a numpy array of strings.
    '''
    
    try:
        if any([isinstance(int(x), long)
                for x in sequence]):
            # Cast very long "integers" as strings
            return np.array(sequence)
        return np.array([int(x) for x in sequence])
    except ValueError:
        try:
            return np.array([float(x) for x in sequence])
        except ValueError:
            return np.array(sequence)
