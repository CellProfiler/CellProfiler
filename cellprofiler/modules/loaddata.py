'''<b>Load Data</b> loads text or numerical data to be associated with images, and
can also load images specified by file names.
<hr>
This module loads a file that supplies text or numerical data associated with
the images to be processed, e.g., sample names, plate names, well
identifiers, or even a list of image filenames to be processed in the analysis run.

<p><i>Disclaimer:</i> Please note that the Input modues (i.e., <b>Images</b>, <b>Metadata</b>, <b>NamesAndTypes</b>
and <b>Groups</b>) largely supercedes this module. However, old pipelines loaded into
CellProfiler that contain this module will provide the option of preserving them;
these pipelines will operate exactly as before.</p>

<p>The module currently reads files in CSV (comma-separated values) format.
These files can be produced by saving a spreadsheet from Excel as
"Windows Comma Separated Values" file format.
 The lines of the file represent the rows, and each field in a row is
separated by a comma. Text values may be optionally enclosed by double
quotes. The <b>LoadData</b> module uses the first row of the file as a header.
The fields in this row provide the labels for each column of data. Subsequent rows
provide the values for each image cycle.<p>

<p>There are many reasons why you might want to prepare a CSV file and load it
via <b>LoadData</b>. Below, we describe how the column nomenclature allows for special
functionality for some downstream modules:
<ul>
<li><i>Columns with any name:</i> Any data loaded via <b>LoadData</b> will be exported
as a per-image measurement along with CellProfiler-calculated data. This is a
convenient way for you to add data from your own sources to the files exported by
CellProfiler.</li>

<li><i>Columns whose name begins with Image_FileName or Image_PathName:</i>
A column whose name begins with "Image_FileName" or "Image_PathName" can be used to
supply the file name and path name (relative to the base folder) of an image that you want to load.
The image's name within CellProfiler appears afterward. For instance,
"Image_FileName_CY3" would supply the file name for the CY3-stained image, and
choosing the <i>Load images based on this data?</i> option allows the CY3 images to be
selected later in the pipeline. "Image_PathName_CY3" would supply the path names
for the CY3-stained images. The path name column is optional; if all image files are in the base
folder, this column is not needed.</li>

<li><i>Columns whose name begins with Image_ObjectsFileName or Image_ObjectsPathName:</i>
The behavior of these columns is identical to that of "Image_FileName" or "Image_PathName"
except that it is used to specify an image that you want to load as objects. </li>

<li><i>Columns whose name begins with Metadata:</i> A column whose name begins with
"Metadata" can be used to group or associate files loaded by <b>LoadData</b>.
<p>For instance, an experiment might require that images created on the same day
use an illumination correction function calculated from all images from that day,
and furthermore, that the date be captured in the file names for the individual image
sets and in a CSV file specifying the illumination correction functions.
<p>In this case, if the illumination correction images are loaded with the
<b>LoadData</b> module, the file should have a "Metadata_Date"
column which contains the date metadata tags. Similarly, if the individual images
are loaded using the <b>LoadImages</b> module, <b>LoadImages</b> should be set to extract the
<Date> metadata tag from the file names (see <b>LoadImages</b> for more details
on how to do so). The pipeline will then match the individual image with
their corresponding illumination correction functions based on matching
"Metadata_Date" tags. This is useful if the same data is associated with several
images (for example, multiple images obtained from a single well).</li>

<li><i>Columns whose name begins with Series or Frame:</i> A columns whose name begins
with "Series" or "Frame" refers to CSVs containing information about image stacks or movies.
The name of the image within CellProfiler appears afterward an underscore character. For
example, "Frame_DNA" would supply the frame number for the movie/image stack file specified
by the "Image_FileName_DNA" and "Image_PathName_DNA" columns.
<p>Using a CSV for loading frames and/or series from an movie/image stack allows you more
flexibility in assembling image sets for operations that would difficult or impossible
using the Input modules alone. For example, if you wanted to analyze a movie of 1,000 frames
by computing the difference between frames, you could create two
image columns in a CSV, one for loading frames 1,2,...,999, and the other for loading
frames 2,3,...,1000. In this case, CellProfiler would load the frame and its predecessor
for each cycle and <b>ImageMath</b> could be used to create the differece image for downstream use.</p>
</li>

<li><i>Columns that contain dose-response or positive/negative control information:</i>
The <b>CalculateStatistics</b> module can calculate metrics of assay quality for
an experiment if provided with information about which images represent positive
and negative controls and/or what dose of treatment has been used for which images.
This information is provided to <b>CalculateStatistics</b> via the <b>LoadData</b>
module, using particular formats described in the help for <b>CalculateStatistics</b>.
Again, using <b>LoadData</b> is useful if the same data is associated with several
images (for example, multiple images obtained from a single well).</li>
</ul>

<h5>Example CSV file:</h5>
<tt><table border="0">
<tr><td>Image_FileName_FITC,</td><td>Image_PathName_FITC,</td><td>Metadata_Plate,</td><td>Titration_NaCl_uM</td></tr><br>
<tr><td>"04923_d1.tif",</td><td>"2009-07-08",</td><td>"P-12345",</td><td>750</td></tr>
<tr><td>"51265_d1.tif",</td><td>"2009-07-09",</td><td>"P-12345",</td><td>2750</td></tr>
</table></tt>

After the first row of header information (the column names), the first
image-specific row specifies the file, "2009-07-08/04923_d1.tif" for the FITC
image (2009-07-08 is the name of the subfolder that contains the image,
relative to the Default Input Folder). The plate metadata is "P-12345" and
the NaCl titration used in the well is 750 uM. The second image-specific row
has the values "2009-07-09/51265_d1.tif", "P-12345" and 2750 uM. The NaCl
titration for the image is available for modules that use numeric metadata,
such as <b>CalculateStatistics</b>; "Titration" will be the category and "NaCl_uM"
will be the measurement.

<h5>Using metadata in LoadData</h5>

<p>If you would like to use the metadata-specific settings, please see <i>Help > General help > Using
metadata in CellProfiler</i> for more details on metadata usage and syntax. Briefly, <b>LoadData</b> can
use metadata provided by the input CSV file for grouping similar images together for the
analysis run and for metadata-specfic options in other modules; see the settings help for
<i>Group images by metadata</i> and, if that setting is selected, <i>Select metadata tags for grouping</i>
for details.</p>

<h6>Using MetaXpress-acquired images in CellProfiler</h6>

<p>To produce a CSV file containing image location and metadata from a <a href=
"http://www.moleculardevices.com/Products/Software/High-Content-Analysis/MetaXpress.html">MetaXpress</a>
imaging run, do the following:
<ul>
<li>Collect image locations from all files that match the string <i>.tif</i> in the desired image folder,
one row per image.</li>
<li>Split up the image pathname and filename into separate data columns for <b>LoadData</b> to read.</li>
<li>Remove data rows corresponding to:
<ul>
<li>Thumbnail images (do not contain imaging data)</li>
<li>Duplicate images (will cause metadata mismatching)</li>
<li>Corrupt files (will cause failure on image loading) </li>
</ul></li>
<li>The image data table may be linked to metadata contained in plate maps. These plate maps should
be stored as flat files, and may be updated periodically via queries to a laboratory information
management system (LIMS) database. </li>
<li>The complete image location and metadata is written to a CSV file where the headers can easily
be formatted to match <b>LoadData</b>'s input requirements (see column descriptions above). Single
plates split across multiple directories (which often occurs in MetaXpress) are written
to separate files and then merged, thereby removing the discontinuity.</li>
</ul>
For a GUI-based approach to performing this task, we suggest using <a href="http://accelrys.com/products/pipeline-pilot/">Pipeline
Pilot</a>.

<p>For more details on configuring CellProfiler
(and LoadData in particular) for a LIMS environment, please see our
<a href="https://github.com/CellProfiler/CellProfiler/wiki/Adapting-CellProfiler-to-a-LIMS-environment">wiki</a> on the subject.</p>

<h4>Available measurements</h4>
<ul>
<li><i>Pathname, Filename:</i> The full path and the filename of each image, if
image loading was requested by the user.</li>
<li>Per-image information obtained from the input file provided by the user.</li>
<li><i>Scaling:</i> The maximum possible intensity value for the image format.</li>
<li><i>Height, Width:</i> The height and width of the current image.</li>
</ul>

See also the <b>Input</b> modules, <b>LoadImages</b> and <b>CalculateStatistics</b>.
'''

import csv
import hashlib
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
import matplotlib.mlab

import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import cellprofiler.preferences as cpprefs
import identify as I
from cellprofiler.modules.loadimages import LoadImagesImageProvider
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME, C_URL
from cellprofiler.modules.loadimages import C_SERIES, C_FRAME
from cellprofiler.modules.loadimages import C_OBJECTS_FILE_NAME
from cellprofiler.modules.loadimages import C_OBJECTS_PATH_NAME
from cellprofiler.modules.loadimages import C_OBJECTS_URL
from cellprofiler.measurement import C_OBJECTS_SERIES, C_OBJECTS_FRAME
from cellprofiler.modules.loadimages import C_MD5_DIGEST, C_SCALING
from cellprofiler.modules.loadimages import C_HEIGHT, C_WIDTH
from cellprofiler.modules.loadimages import bad_sizes_warning
from cellprofiler.modules.loadimages import convert_image_to_objects
from cellprofiler.modules.loadimages import pathname2url, url2pathname
from cellprofiler.preferences import standardize_default_folder_names, \
    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, NO_FOLDER_NAME, \
    ABSOLUTE_FOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT

IMAGE_CATEGORIES = (C_URL, C_FILE_NAME, C_PATH_NAME)
OBJECTS_CATEGORIES = (C_OBJECTS_URL, C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME)
DIR_NONE = 'None'
DIR_OTHER = 'Elsewhere...'
DIR_ALL = [DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME,
           NO_FOLDER_NAME, ABSOLUTE_FOLDER_NAME]

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
    for name in (C_PATH_NAME, C_FILE_NAME, C_URL,
                 C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL):
        if field.startswith(cpmeas.IMAGE + '_' + name + '_'):
            return field[len(cpmeas.IMAGE) + 1:]
    return field


def is_path_name_feature(feature):
    '''Return true if the feature name is a path name'''
    return feature.startswith(C_PATH_NAME + '_')


def is_file_name_feature(feature):
    '''Return true if the feature name is a file name'''
    return feature.startswith(C_FILE_NAME + '_')


def is_url_name_feature(feature):
    return feature.startswith(C_URL + "_")


def is_objects_path_name_feature(feature):
    '''Return true if the feature name is the path to a labels file'''
    return feature.startswith(C_OBJECTS_PATH_NAME + "_")


def is_objects_file_name_feature(feature):
    '''Return true if the feature name is a labels file name'''
    return feature.startswith(C_OBJECTS_FILE_NAME + "_")


def is_objects_url_name_feature(feature):
    return feature.startswith(C_OBJECTS_URL + "_")


def get_image_name(feature):
    '''Extract the image name from a feature name'''
    if is_path_name_feature(feature):
        return feature[len(C_PATH_NAME + '_'):]
    if is_file_name_feature(feature):
        return feature[len(C_FILE_NAME + '_'):]
    if is_url_name_feature(feature):
        return feature[len(C_URL + '_'):]
    raise ValueError('"%s" is not a path feature or file name feature' % feature)


def get_objects_name(feature):
    '''Extract the objects name from a feature name'''
    if is_objects_path_name_feature(feature):
        return feature[len(C_OBJECTS_PATH_NAME + "_"):]
    if is_objects_file_name_feature(feature):
        return feature[len(C_OBJECTS_FILE_NAME + "_"):]
    if is_objects_url_name_feature(feature):
        return feature[len(C_OBJECTS_URL + "_"):]
    raise ValueError('"%s" is not a objects path feature or file name feature' % feature)


def make_path_name_feature(image):
    '''Return the path name feature, given an image name

    The path name feature is the name of the measurement that stores
    the image's path name.
    '''
    return C_PATH_NAME + '_' + image


def make_file_name_feature(image):
    '''Return the file name feature, given an image name

    The file name feature is the name of the measurement that stores
    the image's file name.
    '''
    return C_FILE_NAME + '_' + image


def make_objects_path_name_feature(objects_name):
    '''Return the path name feature, given an object name

    The path name feature is the name of the measurement that stores
    the objects file path name.
    '''
    return C_OBJECTS_PATH_NAME + '_' + objects_name


def make_objects_file_name_feature(objects_name):
    '''Return the file name feature, given an object name

    The file name feature is the name of the measurement that stores
    the objects file name.
    '''
    return C_OBJECTS_FILE_NAME + '_' + objects_name


class LoadData(cpm.Module):
    module_name = "LoadData"
    category = 'File Processing'
    variable_revision_number = 6

    def create_settings(self):
        self.csv_directory = cps.DirectoryPath(
                "Input data file location", allow_metadata=False, support_urls=True,
                doc="""Select the folder containing the CSV file to be loaded.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s
            <p>An additional option is the following:
            <ul>
            <li><i>URL</i>: Use the path part of a URL. For instance, an example .CSV file
            is hosted at <i>http://cellprofiler.org/svnmirror/ExampleImages/ExampleSBSImages/1049_Metadata.csv</i>
            To access this file, you would choose <i>URL</i> and enter
            <i>http://cellprofiler.org/svnmirror/ExampleImages/ExampleSBSImages</i>
            as the path location.</li>
            </ul></p>""" % globals())

        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.csv_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.csv_directory.get_parts_from_path(path)
            self.csv_directory.join_parts(dir_choice, custom_path)

        self.csv_file_name = cps.FilenameText(
                "Name of the file",
                cps.NONE, doc="""
            Provide the file name of the CSV file containing the data.""",
                get_directory_fn=get_directory_fn,
                set_directory_fn=set_directory_fn,
                browse_msg="Choose CSV file",
                exts=[("Data file (*.csv)", "*.csv"), ("All files (*.*)", "*.*")]
        )

        self.browse_csv_button = cps.DoSomething(
                "Press to view CSV file contents", "View...", self.browse_csv)

        self.wants_images = cps.Binary("Load images based on this data?", True, doc="""
            Select <i>%(YES)s</i> to have <b>LoadData</b> load images using the <i>Image_FileName</i> field and the
            <i>Image_PathName</i> fields (the latter is optional).""" % globals())

        self.rescale = cps.Binary(
                "Rescale intensities?", True, doc="""
            This option determines whether image metadata should be
            used to rescale the image's intensities. Some image formats
            save the maximum possible intensity value along with the pixel data.
            For instance, a microscope might acquire images using a 12-bit
            A/D converter which outputs intensity values between zero and 4095,
            but stores the values in a field that can take values up to 65535.
            <p>Select <i>%(YES)s</i> to rescale the image intensity so that
            saturated values are rescaled to 1.0 by dividing all pixels
            in the image by the maximum possible intensity value. </p>
            <p>Select <i>%(NO)s</i> to ignore the image metadata and rescale the image
            to 0 &ndash; 1.0 by dividing by 255 or 65535, depending on the number
            of bits used to store the image.</p>""" % globals())

        self.image_directory = cps.DirectoryPath(
                "Base image location",
                dir_choices=DIR_ALL, allow_metadata=False, doc="""
            The parent (base) folder where images are located. If images are
            contained in subfolders, then the file you load with this module should
            contain a column with path names relative to the base image folder (see
            the general help for this module for more details). You can choose among the following options:
            <ul>
            <li><i>Default Input Folder:</i> Use the Default Input Folder.</li>
            <li><i>Default Output Folder:</i> Use the Default Output Folder.</li>
            <li><i>None:</i> You have an <i>Image_PathName</i> field that supplies an absolute path.</li>
            <li><i>Elsewhere...</i>: Use a particular folder you specify.</li>
            </ul>""")

        self.wants_image_groupings = cps.Binary(
                "Group images by metadata?", False, doc="""
            Select <i>%(YES)s</i> to break the image sets in an experiment into groups
            that can be processed by different nodes on a computing cluster. Each set of
            files that share your selected metadata tags will be processed
            together. See <b>CreateBatchFiles</b> for details on submitting a
            CellProfiler pipeline to a computing cluster for processing.""" % globals())

        self.metadata_fields = cps.MultiChoice(
                "Select metadata tags for grouping", None, doc="""
            <i>(Used only if images are to be grouped by metadata)</i><br>
            Select the tags by which you want to group the image files here. You can select multiple tags. For
            example, if a set of images had metadata for "Run", "Plate", "Well", and
            "Site", selecting <i>Run</i> and <i>Plate</i> will create groups containing
            images that share the same [<i>Run</i>,<i>Plate</i>] pair of tags.""")

        self.wants_rows = cps.Binary(
                "Process just a range of rows?",
                False, doc="""
            Select <i>%(YES)s</i> if you want to process a subset of the rows in the CSV file.
            Rows are numbered starting at 1 (but do not count the header line).
            <b>LoadData</b> will process up to and including the end row.""" % globals())

        self.row_range = cps.IntegerRange(
                "Rows to process",
                (1, 100000), 1, doc="""
            <i>(Used only if a range of rows is to be specified)</i><br>
            Enter the row numbers of the first and last row to be processed.""")

        def do_reload():
            global header_cache
            header_cache = {}
            try:
                self.open_csv()
            except:
                pass

        self.clear_cache_button = cps.DoSomething(
                "Reload cached information", "Reload", do_reload, doc="""
            Press this button to reload header information saved inside
            CellProfiler. <b>LoadData</b> caches information about
            your .csv file in its memory for efficiency.  The
            information is reloaded if a modification is detected.
            <b>LoadData</b> might fail to detect a modification on a
            file accessed over the network and will fail to detect
            modifications on files accessed through HTTP or FTP. In
            this case, you will have to use this button to reload the
            header information after changing the file.
            <p>This button will never destroy any information on
            disk. It is always safe to press it.</p>
            """)

    def settings(self):
        return [self.csv_directory,
                self.csv_file_name, self.wants_images, self.image_directory,
                self.wants_rows,
                self.row_range, self.wants_image_groupings,
                self.metadata_fields, self.rescale]

    def validate_module(self, pipeline):
        csv_path = self.csv_path

        if self.csv_directory.dir_choice != cps.URL_FOLDER_NAME:
            if not os.path.isfile(csv_path):
                raise cps.ValidationError("No such CSV file: %s" % csv_path,
                                          self.csv_file_name)

        try:
            self.open_csv()
        except IOError, e:
            import errno
            if e.errno == errno.EWOULDBLOCK:
                raise cps.ValidationError("Another program (Excel?) is locking the CSV file %s." %
                                          self.csv_path, self.csv_file_name)
            else:
                raise cps.ValidationError("Could not open CSV file %s (error: %s)" %
                                          (self.csv_path, e), self.csv_file_name)

        try:
            self.get_header()
        except Exception, e:
            raise cps.ValidationError(
                    "The CSV file, %s, is not in the proper format. See this module's help for details on CSV format. (error: %s)" %
                    (self.csv_path, e), self.csv_file_name)

    def validate_module_warnings(self, pipeline):
        '''Check for potentially dangerous settings

        The best practice is to have a single LoadImages or LoadData module.
        '''
        from cellprofiler.modules.loadimages import LoadImages

        for module in pipeline.modules():
            if id(module) == id(self):
                return
            if isinstance(module, LoadData):
                raise cps.ValidationError(
                        "Your pipeline has two or more LoadData modules.\n"
                        "The best practice is to have only one LoadData module.\n"
                        "Consider combining the CSV files from all of your\n"
                        "LoadData modules into one and using only a single\n"
                        "LoadData module", self.csv_file_name)
            if isinstance(module, LoadImages):
                raise cps.ValidationError(
                        "Your pipeline has a LoadImages and LoadData module.\n"
                        "The best practice is to have only a single LoadImages\n"
                        "or LoadData module. This LoadData module will match its\n"
                        "metadata against that of the previous LoadImages module\n"
                        "in an attempt to reconcile the two modules' image\n"
                        "set lists and this can result in image sets with\n"
                        "missing images or metadata.", self.csv_file_name)

        # check that user has selected fields for grouping if grouping is turned on
        if self.wants_image_groupings.value and (len(self.metadata_fields.selections) == 0):
            raise cps.ValidationError("Group images by metadata is True, but no metadata "
                                      "tags have been chosen for grouping.",
                                      self.metadata_fields)

    def visible_settings(self):
        result = [self.csv_directory, self.csv_file_name,
                  self.browse_csv_button]
        if self.csv_directory.dir_choice == cps.URL_FOLDER_NAME:
            result += [self.clear_cache_button]
            self.csv_file_name.text = "URL of the file"
            self.csv_file_name.set_browsable(False)
        else:
            self.csv_file_name.text = "Name of the file"
            self.csv_file_name.set_browsable(True)
        result += [self.wants_images]
        if self.wants_images.value:
            result += [self.rescale, self.image_directory,
                       self.wants_image_groupings]
            if self.wants_image_groupings.value:
                result += [self.metadata_fields]
                try:
                    fields = [field[len("Metadata_"):]
                              for field in self.get_header()
                              if field.startswith("Metadata_")]
                    if self.has_synthetic_well_metadata():
                        fields += [cpmeas.FTR_WELL]
                    self.metadata_fields.choices = fields
                except:
                    self.metadata_fields.choices = ["No CSV file"]

        result += [self.wants_rows]
        if self.wants_rows.value:
            result += [self.row_range]
        return result

    def convert(self):
        data = matplotlib.mlab.csv2rec(self.csv_path)
        src_dsc = data['source_description']

        def uniquewaves(seq):
            output = []
            for x in seq:
                if x not in output:
                    output.append(x)
            return output

        waves = uniquewaves(src_dsc)

        pathname = []
        filename = []
        wave_pnames = []
        wave_fnames = []

        for i in range(len(waves)):
            mask = data['source_description'] == waves[i]
            pathname.append(data[mask]['file_path'])
            filename.append(data[mask]['file_name'])
            wave_pnames.append('PathName_%s' % (waves[i].strip('"')))
            wave_fnames.append('FileName_%s' % (waves[i].strip('"')))

        for i in range(len(waves)):
            if len(filename[i]) != len(filename[0]):
                raise RuntimeError("Image %s has %d files, but image %s has %d files" %
                                   (wave_fnames[i], len(filename[i]), wave_fnames[0], len(filename[0])))

        def metadatacols(header):
            output = []
            for h in header:
                if not h.startswith('file_'):
                    if isinstance(h, unicode):
                        output.append(h.encode("utf-8"))
                    else:
                        output.append(h)
            return output

        def data_for_one_wave(data):
            mask = data['source_description'] == waves[0]
            data_onewave = data[mask]
            return data_onewave

        header = data.dtype.names
        metadata_names = metadatacols(header)

        data_onewave = data_for_one_wave(data)
        strdate = []
        for date in data_onewave['date_created']:
            strdate += [str(date)]
        metadata_names.remove('source_description')
        metadata_names.remove('date_created')
        data_onewave_nofilepaths = matplotlib.mlab.rec_keep_fields(data_onewave, metadata_names)
        metadata_names = ['Metadata_' + m for m in metadata_names]
        data_onewave_nofilepaths.dtype.names = metadata_names
        final_data = data_onewave_nofilepaths
        final_data = matplotlib.mlab.rec_append_fields(final_data, 'Metadata_date_created', strdate)
        for i in range(len(waves)):
            final_data = matplotlib.mlab.rec_append_fields(final_data, wave_pnames[i], pathname[i])
            final_data = matplotlib.mlab.rec_append_fields(final_data, wave_fnames[i], filename[i])
        return final_data

    @property
    def csv_path(self):
        '''The path and file name of the CSV file to be loaded'''
        if cpprefs.get_data_file() is not None:
            return cpprefs.get_data_file()
        if self.csv_directory.dir_choice == cps.URL_FOLDER_NAME:
            return self.csv_file_name.value

        path = self.csv_directory.get_absolute_path()
        return os.path.join(path, self.csv_file_name.value)

    @property
    def image_path(self):
        return self.image_directory.get_absolute_path()

    @property
    def legacy_field_key(self):
        '''The key to use to retrieve the metadata from the image set list'''
        return 'LoadTextMetadata_%d' % self.module_num

    def get_cache_info(self):
        '''Get the cached information for the data file'''
        global header_cache
        entry = header_cache.get(self.csv_path, dict(ctime=0))
        if cpprefs.is_url_path(self.csv_path):
            if not header_cache.has_key(self.csv_path):
                header_cache[self.csv_path] = entry
            return entry
        ctime = os.stat(self.csv_path).st_ctime
        if ctime > entry["ctime"]:
            entry = header_cache[self.csv_path] = {}
            entry["ctime"] = ctime
        return entry

    def open_csv(self, do_not_cache=False):
        '''Open the csv file or URL, returning a file descriptor'''
        global header_cache

        if cpprefs.is_url_path(self.csv_path):
            if not header_cache.has_key(self.csv_path):
                header_cache[self.csv_path] = {}
            entry = header_cache[self.csv_path]
            if entry.has_key("URLEXCEPTION"):
                raise entry["URLEXCEPTION"]
            if entry.has_key("URLDATA"):
                fd = StringIO(entry["URLDATA"])
            else:
                if do_not_cache:
                    raise RuntimeError('Need to fetch URL manually.')
                import urllib2
                try:
                    url_fd = urllib2.urlopen(self.csv_path)
                except Exception, e:
                    entry["URLEXCEPTION"] = e
                    raise e
                fd = StringIO()
                while True:
                    text = url_fd.read()
                    if len(text) == 0:
                        break
                    fd.write(text)
                fd.seek(0)
                entry["URLDATA"] = fd.getvalue()
            return fd
        else:
            return open(self.csv_path, 'rb')

    def browse_csv(self):
        import wx
        from cellprofiler.gui import get_cp_icon
        try:
            fd = self.open_csv()
        except:
            wx.MessageBox("Could not read %s" % self.csv_path)
            return
        reader = csv.reader(fd)
        header = reader.next()
        frame = wx.Frame(wx.GetApp().frame, title=self.csv_path)
        sizer = wx.BoxSizer(wx.VERTICAL)
        frame.SetSizer(sizer)
        list_ctl = wx.ListCtrl(frame, style=wx.LC_REPORT)
        sizer.Add(list_ctl, 1, wx.EXPAND)
        for i, field in enumerate(header):
            list_ctl.InsertColumn(i, field)
        for line in reader:
            list_ctl.Append([unicode(s, 'utf8') if isinstance(s, str) else s
                             for s in line[:len(header)]])
        frame.SetMinSize((640, 480))
        frame.SetIcon(get_cp_icon())
        frame.Fit()
        frame.Show()

    def get_header(self, do_not_cache=False):
        '''Read the header fields from the csv file

        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        '''
        entry = self.get_cache_info()
        if entry.has_key("header"):
            return entry["header"]

        fd = self.open_csv(do_not_cache=do_not_cache)
        reader = csv.reader(fd)
        header = reader.next()
        fd.close()
        if header[0].startswith('ELN_RUN_ID'):
            try:
                data = self.convert()
            except Exception, e:
                raise RuntimeError("%s" % e)
            header = data.dtype.names
        entry["header"] = [header_to_column(column) for column in header]
        return entry["header"]

    def get_image_names(self, do_not_cache=False):
        header = self.get_header(do_not_cache=do_not_cache)
        image_names = set([
                              get_image_name(field)
                              for field in header
                              if is_file_name_feature(field) or is_url_name_feature(field)])
        return list(image_names)

    def get_object_names(self, do_not_cache=False):
        header = self.get_header(do_not_cache=do_not_cache)
        object_names = set([get_objects_name(field)
                            for field in header
                            if is_objects_file_name_feature(field) or
                            is_objects_url_name_feature(field)])
        return list(object_names)

    def other_providers(self, group):
        '''Get name providers from the CSV header'''
        if group == 'imagegroup' and self.wants_images.value:
            try:
                # do not load URLs automatically
                return self.get_image_names(do_not_cache=True)
            except Exception, e:
                return []
        elif group == 'objectgroup' and self.wants_images:
            try:
                # do not load URLs automatically
                return self.get_object_names(do_not_cache=True)
            except Exception, e:
                return []

        return []

    def is_image_from_file(self, image_name):
        '''Return True if LoadData provides the given image name'''
        providers = self.other_providers('imagegroup')
        return image_name in providers

    def is_load_module(self):
        '''LoadData can make image sets so it's a load module'''
        return True

    def prepare_run(self, workspace):
        pipeline = workspace.pipeline
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        '''Load the CSV file at the outset and populate the image set list'''
        if pipeline.in_batch_mode():
            return True
        fd = self.open_csv()
        reader = csv.reader(fd)
        header = [header_to_column(column) for column in reader.next()]
        if header[0].startswith('ELN_RUN_ID'):
            reader = self.convert()
            header = list(reader.dtype.names)
        if self.wants_rows.value:
            # skip initial rows
            rows = []
            for idx, row in enumerate(reader):
                if idx + 1 < self.row_range.min:
                    continue
                if idx + 1 > self.row_range.max:
                    break
                if len(row) == 0:
                    continue
                row = [unicode(s, 'utf8') if isinstance(s, str) else s
                       for s in row]
                if len(row) != len(header):
                    raise ValueError("Row # %d has the wrong number of elements: %d. Expected %d" %
                                     (i, len(row), len(header)))
                rows.append(row)
        else:
            rows = [[unicode(s, 'utf8') if isinstance(s, str) else s
                     for s in row] for row in reader
                    if len(row) > 0]
        fd.close()
        #
        # Check for correct # of columns
        #
        n_fields = len(header)
        for i, row in enumerate(rows):
            if len(row) < n_fields:
                text = ('Error on line %d of %s.\n'
                        '\n"%s"\n'
                        '%d rows found, expected %d') % (
                           i + 2, self.csv_file_name.value,
                           ','.join(row),
                           len(row), n_fields)
                raise ValueError(text)
            elif len(row) > n_fields:
                del row[n_fields:]
        #
        # Find the metadata, object_name and image_name columns
        #
        metadata_columns = {}
        object_columns = {}
        image_columns = {}
        well_row_column = well_column_column = well_well_column = None
        for i, column in enumerate(header):
            if column.find("_") == -1:
                category = ""
                feature = column
            else:
                category, feature = column.split("_", 1)
            if category in IMAGE_CATEGORIES:
                if not image_columns.has_key(feature):
                    image_columns[feature] = []
                image_columns[feature].append(i)
            elif category in OBJECTS_CATEGORIES:
                if not object_columns.has_key(feature):
                    object_columns[feature] = []
                object_columns[feature].append(i)
            else:
                metadata_columns[column] = i
                if category == cpmeas.C_METADATA:
                    if feature.lower() == cpmeas.FTR_WELL.lower():
                        well_well_column = i
                    elif cpmeas.is_well_row_token(feature):
                        well_row_column = i
                    elif cpmeas.is_well_column_token(feature):
                        well_column_column = i

        if (well_row_column is not None and well_column_column is not None and
                    well_well_column is None):
            # add a synthetic well column
            metadata_columns[cpmeas.M_WELL] = len(header)
            header.append(cpmeas.M_WELL)
            for row in rows:
                row.append(row[well_row_column] + row[well_column_column])
        if self.wants_images:
            #
            # Add synthetic object and image columns
            #
            if self.image_directory.dir_choice == cps.NO_FOLDER_NAME:
                path_base = ""
            else:
                path_base = self.image_path
            for d, url_category, file_name_category, path_name_category in (
                    (image_columns, C_URL, C_FILE_NAME, C_PATH_NAME),
                    (object_columns, C_OBJECTS_URL, C_OBJECTS_FILE_NAME,
                     C_OBJECTS_PATH_NAME)):
                for name in d.keys():
                    url_column = file_name_column = path_name_column = None
                    for k in d[name]:
                        if header[k].startswith(url_category):
                            url_column = k
                        elif header[k].startswith(file_name_category):
                            file_name_column = k
                        elif header[k].startswith(path_name_category):
                            path_name_column = k
                    if url_column is None:
                        if file_name_column is None:
                            raise ValueError(
                                    ("LoadData needs a %s_%s column to match the "
                                     "%s_%s column") % (file_name_category, name,
                                                        path_name_category, name))
                        #
                        # Add URL column
                        #
                        d[name].append(len(header))
                        url_feature = "_".join((url_category, name))
                        header.append(url_feature)
                        for row in rows:
                            if path_name_column is None:
                                fullname = os.path.join(path_base,
                                                        row[file_name_column])
                            else:
                                row_path_name = os.path.join(
                                        path_base, row[path_name_column])
                                fullname = os.path.join(
                                        row_path_name, row[file_name_column])
                                row[path_name_column] = row_path_name
                            url = pathname2url(fullname)
                            row.append(url)
                        if path_name_column is None:
                            #
                            # Add path column
                            #
                            d[name].append(len(header))
                            path_feature = "_".join((path_name_category, name))
                            header.append(path_feature)
                            for row in rows:
                                row.append(path_base)
                    elif path_name_column is None and file_name_column is None:
                        #
                        # If the .csv just has URLs, break the URL into
                        # path and file names
                        #
                        path_feature = "_".join((path_name_category, name))
                        path_name_column = len(header)
                        header.append(path_feature)

                        file_name_feature = "_".join((file_name_category, name))
                        file_name_column = len(header)
                        header.append(file_name_feature)
                        for row in rows:
                            url = row[url_column]
                            idx = url.rfind("/")
                            if idx == -1:
                                idx = url.rfind(":")
                                if idx == -1:
                                    row += ["", url]
                                else:
                                    row += [url[:(idx + 1)], url[(idx + 1):]]
                            else:
                                row += [url[:idx], url[(idx + 1):]]

        column_type = {}
        for column in self.get_measurement_columns(pipeline):
            column_type[column[1]] = column[2]

        previous_column_types = dict([
                                         (c[1], c[2]) for c in pipeline.get_measurement_columns(self)
                                         if c[0] == cpmeas.IMAGE])
        #
        # Arrange the metadata into columns
        #
        columns = {}
        for index, feature in enumerate(header):
            c = []
            columns[feature] = c
            for row in rows:
                value = row[index]
                if column_type.has_key(feature):
                    datatype = column_type[feature]
                else:
                    datatype = previous_column_types[feature]
                if datatype == cpmeas.COLTYPE_INTEGER:
                    value = int(value)
                elif datatype == cpmeas.COLTYPE_FLOAT:
                    value = float(value)
                c.append(value)

        if len(metadata_columns) > 0:
            # Reorder the rows by matching metadata against previous metadata
            # (for instance, to assign metadata values to images from
            #  loadimages)
            #
            image_numbers = m.match_metadata(
                    metadata_columns.keys(),
                    [columns[k] for k in metadata_columns.keys()])
            image_numbers = np.array(image_numbers, int).flatten()
            max_image_number = np.max(image_numbers)
            new_columns = {}
            for key, values in columns.iteritems():
                new_values = [None] * max_image_number
                for image_number, value in zip(image_numbers, values):
                    new_values[image_number - 1] = value
                new_columns[key] = new_values
            columns = new_columns
        for feature, values in columns.iteritems():
            m.add_all_measurements(cpmeas.IMAGE, feature, values)
        if self.wants_image_groupings and \
                        len(self.metadata_fields.selections) > 0:
            keys = ["_".join((cpmeas.C_METADATA, k))
                    for k in self.metadata_fields.selections]
            m.set_grouping_tags(keys)

        return True

    def prepare_to_create_batch(self, workspace, fn_alter_path):
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

        if self.wants_images:
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            image_numbers = m.get_image_numbers()
            all_image_features = m.get_feature_names(cpmeas.IMAGE)
            for url_category, file_category, path_category, names in (
                    (C_URL, C_FILE_NAME, C_PATH_NAME, self.get_image_names()),
                    (C_OBJECTS_URL, C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME,
                     self.get_object_names())):
                for name in names:
                    url_feature = "_".join((url_category, name))
                    path_feature = "_".join((path_category, name))
                    if path_feature not in all_image_features:
                        path_feature = None
                    file_feature = "_".join((file_category, name))
                    if file_feature not in all_image_features:
                        file_feature = None
                    urls = m.get_measurement(cpmeas.IMAGE,
                                             url_feature,
                                             image_set_number=image_numbers)
                    for image_number, url in zip(image_numbers, urls):
                        url = url.encode("utf-8")
                        if url.lower().startswith("file:"):
                            fullname = url2pathname(url)
                            fullname = fn_alter_path(fullname)
                            path, filename = os.path.split(fullname)
                            url = unicode(pathname2url(fullname), "utf-8")
                            m.add_measurement(cpmeas.IMAGE, url_feature, url,
                                              image_set_number=image_number)
                            if file_feature is not None:
                                m.add_measurement(
                                        cpmeas.IMAGE, file_feature,
                                        filename,
                                        image_set_number=image_number)
                            if path_feature is not None:
                                m.add_measurement(
                                        cpmeas.IMAGE, path_feature,
                                        path, image_set_number=image_number)

        self.csv_directory.alter_for_create_batch_files(fn_alter_path)
        self.image_directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def fetch_provider(self, name, measurements, is_image_name=True):
        path_base = self.image_path
        if is_image_name:
            url_feature = C_URL + "_" + name
            series_feature = C_SERIES + "_" + name
            frame_feature = C_FRAME + "_" + name
        else:
            url_feature = C_OBJECTS_URL + "_" + name
            series_feature = C_OBJECTS_SERIES + "_" + name
            frame_feature = C_OBJECTS_FRAME + "_" + name
        url = measurements.get_measurement(cpmeas.IMAGE, url_feature)
        url = url.encode('utf-8')
        full_filename = url2pathname(url)
        path, filename = os.path.split(full_filename)
        if measurements.has_feature(cpmeas.IMAGE, series_feature):
            series = measurements[cpmeas.IMAGE, series_feature]
        else:
            series = None
        if measurements.has_feature(cpmeas.IMAGE, frame_feature):
            frame = measurements[cpmeas.IMAGE, frame_feature]
        else:
            frame = None
        return LoadImagesImageProvider(
                name, path, filename,
                rescale=self.rescale.value and is_image_name,
                series=series,
                index=frame)

    def run(self, workspace):
        '''Populate the images and objects'''
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        image_set = workspace.image_set
        object_set = workspace.object_set
        statistics = []
        features = [x[1] for x in
                    self.get_measurement_columns(workspace.pipeline)
                    if x[0] == cpmeas.IMAGE]

        if self.wants_images:
            #
            # Load the image. Calculate the MD5 hash of every image
            #
            image_size = None
            for image_name in self.other_providers('imagegroup'):
                provider = self.fetch_provider(image_name, m)
                image_set.providers.append(provider)
                image = image_set.get_image(image_name)
                pixel_data = image.pixel_data
                m.add_image_measurement("_".join((C_MD5_DIGEST, image_name)),
                                        provider.get_md5_hash(m))
                m.add_image_measurement("_".join((C_SCALING, image_name)),
                                        image.scale)
                m.add_image_measurement("_".join((C_HEIGHT, image_name)),
                                        int(pixel_data.shape[0]))
                m.add_image_measurement("_".join((C_WIDTH, image_name)),
                                        int(pixel_data.shape[1]))
                if image_size is None:
                    image_size = tuple(pixel_data.shape[:2])
                    first_filename = image.file_name
                elif tuple(pixel_data.shape[:2]) != image_size:
                    warning = bad_sizes_warning(image_size, first_filename,
                                                pixel_data.shape, image.file_name)
                    if self.show_window:
                        workspace.display_data.warning = warning
                    else:
                        print warning
                        #
                        # Process any object tags
                        #
            objects_names = self.get_object_names()
            for objects_name in objects_names:
                provider = self.fetch_provider(
                        objects_name, m, is_image_name=False)
                image = provider.provide_image(workspace.image_set)
                pixel_data = convert_image_to_objects(image.pixel_data)
                o = cpo.Objects()
                o.segmented = pixel_data
                object_set.add_objects(o, objects_name)
                I.add_object_count_measurements(m, objects_name, o.count)
                I.add_object_location_measurements(m, objects_name, pixel_data)

        for feature_name in sorted(features):
            value = m.get_measurement(cpmeas.IMAGE, feature_name)
            statistics.append((feature_name, value))

        if self.show_window:
            workspace.display_data.statistics = statistics

    def display(self, workspace, figure):
        if hasattr(workspace.display_data, "warning"):
            from cellprofiler.gui.errordialog import show_warning
            show_warning("Images have different sizes",
                         workspace.display_data.warning,
                         cpprefs.get_show_report_bad_sizes_dlg,
                         cpprefs.set_show_report_bad_sizes_dlg)
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, workspace.display_data.statistics)

    def get_groupings(self, workspace):
        '''Return the image groupings of the image sets

        See CPModule for documentation
        '''
        if (self.wants_images.value and
                self.wants_image_groupings.value and
                    len(self.metadata_fields.selections) > 0):
            keys = ["_".join((cpmeas.C_METADATA, k))
                    for k in self.metadata_fields.selections]
            if len(keys) == 0:
                return None
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            return keys, m.get_groupings(keys)
        return None

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements output by this module'''
        entry = None
        try:
            entry = self.get_cache_info()
            if entry.has_key("measurement_columns"):
                return entry["measurement_columns"]
            fd = self.open_csv()
            reader = csv.reader(fd)
            header = [header_to_column(x) for x in reader.next()]
            if header[0].startswith('ELN_RUN_ID'):
                reader = self.convert()
                header = reader.dtype.names
        except:
            if entry is not None:
                entry["measurement_columns"] = []
            return []
        previous_columns = pipeline.get_measurement_columns(self)
        previous_fields = set([x[1] for x in previous_columns
                               if x[0] == cpmeas.IMAGE])
        already_output = [x in previous_fields for x in header]
        coltypes = [cpmeas.COLTYPE_INTEGER] * len(header)
        #
        # Make sure the well_column column type is a string
        #
        for i in range(len(header)):
            if (header[i].startswith(cpmeas.C_METADATA + "_") and
                    cpmeas.is_well_column_token(header[i].split("_")[1])):
                coltypes[i] = cpmeas.COLTYPE_VARCHAR
            if any([header[i].startswith(x)
                    for x in (C_PATH_NAME, C_FILE_NAME, C_OBJECTS_FILE_NAME,
                              C_OBJECTS_PATH_NAME, C_URL, C_OBJECTS_URL)]):
                coltypes[i] = cpmeas.COLTYPE_VARCHAR

        collen = [0] * len(header)
        key_is_path_or_file_name = [
            (key.startswith(C_PATH_NAME) or
             key.startswith(C_FILE_NAME) or
             key.startswith(C_OBJECTS_FILE_NAME) or
             key.startswith(C_OBJECTS_PATH_NAME)) for key in header]
        key_is_path_or_url = [
            (key.startswith(C_PATH_NAME) or
             key.startswith(C_OBJECTS_PATH_NAME) or
             key.startswith(C_URL) or
             key.startswith(C_OBJECTS_URL)) for key in header]

        for row in reader:
            if len(row) > len(header):
                row = row[:len(header)]
            for index, field in enumerate(row):
                if already_output[index]:
                    continue
                if (not self.wants_images) and key_is_path_or_file_name[index]:
                    continue
                try:
                    len_field = len(field)
                except TypeError:
                    field = str(field)
                    len_field = len(field)
                if key_is_path_or_url[index]:
                    # Account for possible rewrite of the pathname
                    # in batch data
                    len_field = max(cpmeas.PATH_NAME_LENGTH,
                                    len_field + PATH_PADDING)
                if coltypes[index] != cpmeas.COLTYPE_VARCHAR:
                    ldtype = get_loaddata_type(field)
                    if coltypes[index] == cpmeas.COLTYPE_INTEGER:
                        coltypes[index] = ldtype
                    elif (coltypes[index] == cpmeas.COLTYPE_FLOAT and
                                  ldtype != cpmeas.COLTYPE_INTEGER):
                        coltypes[index] = ldtype

                if collen[index] < len(field):
                    collen[index] = len(field)

        for index in range(len(header)):
            if coltypes[index] == cpmeas.COLTYPE_VARCHAR:
                coltypes[index] = cpmeas.COLTYPE_VARCHAR_FORMAT % collen[index]

        image_names = self.other_providers('imagegroup')
        result = [(cpmeas.IMAGE, colname, coltype)
                  for colname, coltype in zip(header, coltypes)
                  if colname not in previous_fields]
        if self.wants_images:
            for feature, coltype in (
                    (C_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                    (C_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                    (C_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                    (C_MD5_DIGEST, cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                    (C_SCALING, cpmeas.COLTYPE_FLOAT),
                    (C_HEIGHT, cpmeas.COLTYPE_INTEGER),
                    (C_WIDTH, cpmeas.COLTYPE_INTEGER)):
                for image_name in image_names:
                    measurement = feature + '_' + image_name
                    if not any([measurement == c[1] for c in result]):
                        result.append((cpmeas.IMAGE, measurement, coltype))
            #
            # Add the object features
            #
            for object_name in self.get_object_names():
                result += I.get_object_measurement_columns(object_name)
                for feature, coltype in (
                        (C_OBJECTS_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                        (C_OBJECTS_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                        (C_OBJECTS_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME)):
                    mname = C_OBJECTS_URL + "_" + object_name
                    result.append((cpmeas.IMAGE, mname, coltype))
        #
        # Try to make a well column out of well row and well column
        #
        well_column = None
        well_row_column = None
        well_col_column = None
        for column in result:
            if not column[1].startswith(cpmeas.C_METADATA + "_"):
                continue
            category, feature = column[1].split('_', 1)
            if cpmeas.is_well_column_token(feature):
                well_col_column = column
            elif cpmeas.is_well_row_token(feature):
                well_row_column = column
            elif feature.lower() == cpmeas.FTR_WELL.lower():
                well_column = column
        if (well_column is None and well_row_column is not None and
                    well_col_column is not None):
            length = cpmeas.get_length_from_varchar(well_row_column[2])
            length += cpmeas.get_length_from_varchar(well_col_column[2])
            result += [(cpmeas.IMAGE,
                        '_'.join((cpmeas.C_METADATA, cpmeas.FTR_WELL)),
                        cpmeas.COLTYPE_VARCHAR_FORMAT % length)]
        entry["measurement_columns"] = result
        return result

    def has_synthetic_well_metadata(self):
        '''Determine if we should synthesize a well metadata feature

        '''
        fields = self.get_header()
        has_well_col = False
        has_well_row = False
        for field in fields:
            if not field.startswith(cpmeas.C_METADATA + "_"):
                continue
            category, feature = field.split('_', 1)
            if cpmeas.is_well_column_token(feature):
                has_well_col = True
            elif cpmeas.is_well_row_token(feature):
                has_well_row = True
            elif feature.lower() == cpmeas.FTR_WELL.lower():
                return False
        return has_well_col and has_well_row

    def get_categories(self, pipeline, object_name):
        try:
            columns = self.get_measurement_columns(pipeline)
            result = set([column[1].split('_')[0] for column in columns
                          if column[0] == object_name])
            return list(result)
        except:
            return []

    def get_measurements(self, pipeline, object_name, category):
        columns = self.get_measurement_columns(pipeline)
        return [feature for c, feature in
                [column[1].split('_', 1) for column in columns
                 if column[0] == object_name
                 and column[1].startswith(category + "_")]]

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

        DIR_DEFAULT_IMAGE = 'Default input folder'
        DIR_DEFAULT_OUTPUT = 'Default Output Folder'

        if from_matlab and variable_revision_number == 2:
            logging.warning(
                    "Warning: the format and purpose of LoadText "
                    "has changed substantially.")
            text_file_name = setting_values[0]
            field_name = setting_values[1]
            path_name = setting_values[2]
            if path_name == '.':
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

        if variable_revision_number == 3 and (not from_matlab):
            # directory choice, custom directory merged
            # input_directory_choice, custom_input_directory merged
            csv_directory_choice, csv_custom_directory, \
            csv_file_name, wants_images, image_directory_choice, \
            image_custom_directory, wants_rows, \
            row_range, wants_image_groupings, \
            metadata_fields = setting_values
            csv_directory = cps.DirectoryPath.static_join_string(
                    csv_directory_choice, csv_custom_directory)
            image_directory = cps.DirectoryPath.static_join_string(
                    image_directory_choice, image_custom_directory)
            setting_values = [
                csv_directory, csv_file_name, wants_images,
                image_directory, wants_rows, row_range, wants_image_groupings,
                metadata_fields]
            variable_revision_number = 4

        # Standardize input/output directory name references
        setting_values = list(setting_values)
        for index in (0, 3):
            setting_values[index] = cps.DirectoryPath.upgrade_setting(
                    setting_values[index])

        if variable_revision_number == 4 and (not from_matlab):
            csv_directory, csv_file_name, wants_images, \
            image_directory, wants_rows, row_range, wants_image_groupings, \
            metadata_fields = setting_values
            dir_choice, custom_dir = cps.DirectoryPath.split_string(csv_directory)
            if dir_choice == cps.URL_FOLDER_NAME:
                csv_file_name = custom_dir + '/' + csv_file_name
                csv_directory = cps.DirectoryPath.static_join_string(dir_choice, '')
            setting_values = [
                csv_directory, csv_file_name, wants_images,
                image_directory, wants_rows, row_range, wants_image_groupings,
                metadata_fields]
            variable_revision_number = 5
        if variable_revision_number == 5 and (not from_matlab):
            # Added rescaling option
            setting_values = setting_values + [cps.YES]
            variable_revision_number = 6
        return setting_values, variable_revision_number, from_matlab


LoadText = LoadData


def best_cast(sequence, coltype=None):
    '''Return the best cast (integer, float or string) of the sequence

    sequence - a sequence of strings

    Try casting all elements to integer and float, returning a numpy
    array of values. If all fail, return a numpy array of strings.
    '''
    if (isinstance(coltype, (str, unicode)) and
            coltype.startswith(cpmeas.COLTYPE_VARCHAR)):
        # Cast columns already defined as strings as same
        return np.array(sequence)

    def fn(x, y):
        if cpmeas.COLTYPE_VARCHAR in (x, y):
            return cpmeas.COLTYPE_VARCHAR
        if cpmeas.COLTYPE_FLOAT in (x, y):
            return cpmeas.COLTYPE_FLOAT
        return cpmeas.COLTYPE_INTEGER

    ldtype = reduce(fn, [get_loaddata_type(x) for x in sequence],
                    cpmeas.COLTYPE_INTEGER)
    if ldtype == cpmeas.COLTYPE_VARCHAR:
        return np.array(sequence)
    elif ldtype == cpmeas.COLTYPE_FLOAT:
        return np.array(sequence, np.float64)
    else:
        return np.array(sequence, np.int32)


int32_max = np.iinfo(np.int32).max
int32_min = np.iinfo(np.int32).min


def get_loaddata_type(x):
    '''Return the type to use to represent x

    If x is a 32-bit integer, return cpmeas.COLTYPE_INTEGER.
    If x cannot be represented in 32 bits but is an integer,
    return cpmeas.COLTYPE_VARCHAR
    If x can be represented as a float, return COLTYPE_FLOAT
    '''
    global int32_max, int32_min

    try:
        iv = int(x)
        if iv > int32_max:
            return cpmeas.COLTYPE_VARCHAR
        if iv < int32_min:
            return cpmeas.COLTYPE_VARCHAR
        return cpmeas.COLTYPE_INTEGER
    except:
        try:
            fv = float(x)
            return cpmeas.COLTYPE_FLOAT
        except:
            return cpmeas.COLTYPE_VARCHAR
