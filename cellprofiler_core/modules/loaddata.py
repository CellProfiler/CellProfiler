import csv
import io
import os
import urllib.request

import numpy

from ..constants.image import C_HEIGHT
from ..constants.image import C_MD5_DIGEST
from ..constants.image import C_SCALING
from ..constants.image import C_WIDTH
from ..constants.measurement import COLTYPE_FLOAT
from ..constants.measurement import COLTYPE_INTEGER
from ..constants.measurement import COLTYPE_VARCHAR
from ..constants.measurement import COLTYPE_VARCHAR_FILE_NAME
from ..constants.measurement import COLTYPE_VARCHAR_FORMAT
from ..constants.measurement import COLTYPE_VARCHAR_PATH_NAME
from ..constants.measurement import C_FILE_NAME
from ..constants.measurement import C_FRAME
from ..constants.measurement import C_METADATA
from ..constants.measurement import C_OBJECTS_FILE_NAME
from ..constants.measurement import C_OBJECTS_FRAME
from ..constants.measurement import C_OBJECTS_PATH_NAME
from ..constants.measurement import C_OBJECTS_SERIES
from ..constants.measurement import C_OBJECTS_URL
from ..constants.measurement import C_PATH_NAME
from ..constants.measurement import C_SERIES
from ..constants.measurement import C_URL
from ..constants.measurement import FTR_WELL
from ..constants.measurement import M_WELL
from ..constants.measurement import PATH_NAME_LENGTH
from ..constants.measurement import GROUP_LENGTH
from ..constants.module import IO_FOLDER_CHOICE_HELP_TEXT
from ..constants.modules.load_data import DIR_ALL
from ..constants.modules.load_data import IMAGE_CATEGORIES
from ..constants.modules.load_data import OBJECTS_CATEGORIES
from ..constants.modules.load_data import PATH_PADDING
from ..image import FileImage
from ..measurement import Measurements
from ..module import Module
from ..object import Objects
from ..preferences import NO_FOLDER_NAME
from ..preferences import URL_FOLDER_NAME
from ..preferences import get_data_file
from ..preferences import is_url_path
from ..setting import Binary
from ..setting import ValidationError
from ..setting.do_something import DoSomething
from ..setting.multichoice import MultiChoice
from ..setting.range import IntegerRange
from ..setting.text import Directory
from ..setting.text import Filename
from ..utilities.core.module.identify import add_object_count_measurements
from ..utilities.core.module.identify import add_object_location_measurements
from ..utilities.core.module.identify import get_object_measurement_columns
from ..utilities.core.modules.load_data import bad_sizes_warning
from ..utilities.core.modules.load_data import get_image_name
from ..utilities.core.modules.load_data import get_loaddata_type
from ..utilities.core.modules.load_data import get_objects_name
from ..utilities.core.modules.load_data import header_to_column
from ..utilities.core.modules.load_data import is_file_name_feature
from ..utilities.core.modules.load_data import is_objects_file_name_feature
from ..utilities.core.modules.load_data import is_objects_url_name_feature
from ..utilities.core.modules.load_data import is_url_name_feature
from ..utilities.image import convert_image_to_objects
from ..utilities.image import generate_presigned_url
from ..utilities.measurement import get_length_from_varchar
from ..utilities.measurement import is_well_column_token
from ..utilities.measurement import is_well_row_token
from ..utilities.pathname import pathname2url
from ..utilities.pathname import url2pathname

__doc__ = """\
LoadData
========

**LoadData** loads text or numerical data to be associated with images,
and can also load images specified by file names.

This module loads a file that supplies text or numerical data associated
with the images to be processed, e.g., sample names, plate names, well
identifiers, or even a list of image file names to be processed in the
analysis run. Please note that most researchers will prefer to use the
Input modules (i.e., **Images**,
**Metadata**, **NamesAndTypes** and **Groups**) to load images.

Note that 3D images to be analyzed volumetrically CANNOT be loaded
with LoadData; they must be loaded with the Input modules.

The module reads files in CSV (comma-separated values) format.
These files can be produced by saving a spreadsheet from Excel as
“Windows Comma Separated Values” file format. The lines of the file
represent the rows, and each field in a row is separated by a comma.
Text values may be optionally enclosed by double quotes. The
**LoadData** module uses the first row of the file as a header. The
fields in this row provide the labels for each column of data.
Subsequent rows provide the values for each image cycle.

There are many reasons why you might want to prepare a CSV file and load
it via **LoadData**. Below, we describe how the column nomenclature
allows for special functionality for some downstream modules:

-  *Columns whose name begins with Image\_FileName or Image\_PathName:*
   A column whose name begins with “Image\_FileName” or
   “Image\_PathName” can be used to supply the file name and path name
   (relative to the base folder) of an image that you want to load, which
   will override the settings in the Input modules (**Images**,
   **Metadata**, **NamesAndTypes** and **Groups**). For instance,
   “Image\_FileName\_CY3” would supply the file name for the CY3-stained
   image, and choosing the *Load images based on this data?* option
   allows the CY3 images to be selected later in the pipeline.
   “Image\_PathName\_CY3” would supply the path names for the
   CY3-stained images. The path name column is optional; if all image
   files are in the base folder, this column is not needed.

-  *Columns whose name begins with Image\_ObjectsFileName or
   Image\_ObjectsPathName:* The behavior of these columns is identical
   to that of “Image\_FileName” or “Image\_PathName” except that it is
   used to specify an image that you want to load as objects.

-  *Columns whose name begins with Metadata:* A column whose name begins
   with “Metadata” can be used to group or associate files loaded by
   **LoadData**.

   For instance, an experiment might require that images created on the
   same day use an illumination correction function calculated from all
   images from that day, and furthermore, that the date be captured in
   the file names for the individual image sets and in a CSV file
   specifying the illumination correction functions.

   In this case, if the illumination correction images are loaded with
   the **LoadData** module, the file should have a “Metadata\_Date”
   column which contains the date metadata tags. Similarly, if the
   individual images are loaded using the **LoadImages** module,
   **LoadImages** should be set to extract the metadata tag from the
   file names (see **LoadImages** for more details on how to do so). The
   pipeline will then match the individual image with their
   corresponding illumination correction functions based on matching
   “Metadata\_Date” tags. This is useful if the same data is associated
   with several images (for example, multiple images obtained from a
   single well).

-  *Columns whose name begins with Series or Frame:* A column whose
   name begins with “Series” or “Frame” refers to
   information about image stacks or movies. The name of the image
   within CellProfiler appears after an underscore character. For
   example, “Frame\_DNA” would supply the frame number for the
   movie/image stack file specified by the “Image\_FileName\_DNA” and
   “Image\_PathName\_DNA” columns.

   Using a CSV for loading frames and/or series from an movie/image
   stack allows you more flexibility in assembling image sets for
   operations that would difficult or impossible using the Input modules
   alone. For example, if you wanted to analyze a movie of 1,000 frames
   by computing the difference between frames, you could create two
   image columns in a CSV, one for loading frames 1,2,…,999, and the
   other for loading frames 2,3,…,1000. In this case, CellProfiler would
   load the frame and its predecessor for each cycle and **ImageMath**
   could be used to create the difference image for downstream use.

-  *Columns that contain dose-response or positive/negative control
   information:* The **CalculateStatistics** module can calculate
   metrics of assay quality for an experiment if provided with
   information about which images represent positive and negative
   controls and/or what dose of treatment has been used for which
   images. This information is provided to **CalculateStatistics** via
   the **LoadData** module, using particular formats described in the
   help for **CalculateStatistics**. Again, using **LoadData** is useful
   if the same data is associated with several images (for example,
   multiple images obtained from a single well).

-  *Columns with any other name:* Columns of data beginning with any
   other text will be loaded into CellProfiler and then
   exported as a per-image measurement along with
   CellProfiler-calculated data. This is a convenient way for you to add
   data from your own sources to the files exported by CellProfiler.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also the **Input** modules (i.e., **Images**, **Metadata**,
**NamesAndTypes** and **Groups**), **LoadImages** and
**CalculateStatistics**.

Example CSV file
^^^^^^^^^^^^^^^^

+-------------------------+-------------------------+-------------------+-----------------------+
| Image\_FileName\_FITC   | Image\_PathName\_FITC   | Metadata\_Plate   | Titration\_NaCl\_uM   |
+-------------------------+-------------------------+-------------------+-----------------------+
| “04923\_d1.tif”         | “2009-07-08”            | “P-12345”         | 750                   |
+-------------------------+-------------------------+-------------------+-----------------------+
| “51265\_d1.tif”         | “2009-07-09”            | “P-12345”         | 2750                  |
+-------------------------+-------------------------+-------------------+-----------------------+

After the first row of header information (the column names), the first
image-specific row specifies the file, “2009-07-08/04923\_d1.tif” for
the FITC image (2009-07-08 is the name of the subfolder that contains
the image, relative to the Default Input Folder). The plate metadata is
“P-12345” and the NaCl titration used in the well is 750 uM. The second
image-specific row has the values “2009-07-09/51265\_d1.tif”, “P-12345”
and 2750 uM. The NaCl titration for the image is available for modules
that use numeric metadata, such as **CalculateStatistics**; “Titration”
will be the category and “NaCl\_uM” will be the measurement.

Using metadata in LoadData
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to use the metadata-specific settings, please see
*Help > General help > Using metadata in CellProfiler* for more details
on metadata usage and syntax. Briefly, **LoadData** can use metadata
provided by the input CSV file for grouping similar images together for
the analysis run and for metadata-specfic options in other modules; see
the settings help for *Group images by metadata* and, if that setting is
selected, *Select metadata tags for grouping* for details.

Using MetaXpress-acquired images in CellProfiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To produce a CSV file containing image location and metadata from a
`MetaXpress`_ imaging run, do the following:

-  Collect image locations from all files that match the string *.tif*
   in the desired image folder, one row per image.
-  Split up the image pathname and filename into separate data columns
   for **LoadData** to read.
-  Remove data rows corresponding to:

   -  Thumbnail images (do not contain imaging data)
   -  Duplicate images (will cause metadata mismatching)
   -  Corrupt files (will cause failure on image loading)

-  The image data table may be linked to metadata contained in plate
   maps. These plate maps should be stored as flat files, and may be
   updated periodically via queries to a laboratory information
   management system (LIMS) database.
-  The complete image location and metadata is written to a CSV file
   where the headers can easily be formatted to match **LoadData**\ ’s
   input requirements (see column descriptions above). Single plates
   split across multiple directories (which often occurs in MetaXpress)
   are written to separate files and then merged, thereby removing the
   discontinuity.

More tips on using LoadData
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a GUI-based approach to creating a proper CSV file for use with
**LoadData**, we suggest using `KNIME`_ or `Pipeline Pilot`_.

For more details on configuring CellProfiler (and LoadData in
particular) for a LIMS environment, please see our `wiki`_ on the
subject.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Pathname, Filename:* The full path and the filename of each image,
   if you requested image loading.
-  *Scaling:* The maximum possible intensity value for the image format.
-  *Height, Width:* The height and width of images loaded by this module.
-  Any additional per-image data loaded from the input file you provided.

.. _MetaXpress: http://www.moleculardevices.com/systems/high-content-imaging/metaxpress-high-content-image-acquisition-and-analysis-software
.. _KNIME: https://www.knime.com/about
.. _Pipeline Pilot: http://accelrys.com/products/pipeline-pilot/
.. _wiki: http://github.com/CellProfiler/CellProfiler/wiki/Adapting-CellProfiler-to-a-LIMS-environment
"""

"""Reserve extra space in pathnames for batch processing name rewrites"""

"""Cache of header columns for files"""
header_cache = {}


class LoadData(Module):
    module_name = "LoadData"
    category = "File Processing"
    variable_revision_number = 6

    def create_settings(self):
        self.csv_directory = Directory(
            "Input data file location",
            allow_metadata=False,
            support_urls=True,
            doc="""\
Select the folder containing the CSV file to be loaded. {IO_FOLDER_CHOICE_HELP_TEXT}
""".format(
                **{"IO_FOLDER_CHOICE_HELP_TEXT": IO_FOLDER_CHOICE_HELP_TEXT}
            ),
        )

        def get_directory_fn():
            """Get the directory for the CSV file name"""
            return self.csv_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.csv_directory.get_parts_from_path(path)
            self.csv_directory.join_parts(dir_choice, custom_path)

        self.csv_file_name = Filename(
            "Name of the file",
            "None",
            doc="""Provide the file name of the CSV file containing the data you want to load.""",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            browse_msg="Choose CSV file",
            exts=[("Data file (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
        )

        self.browse_csv_button = DoSomething(
            "Press to view CSV file contents", "View...", self.browse_csv
        )

        self.wants_images = Binary(
            "Load images based on this data?",
            True,
            doc="""\
Select *{YES}* to have **LoadData** load images based on the
*Image\_FileName* column and the *Image\_PathName* column (if specified).
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.rescale = Binary(
            "Rescale intensities?",
            True,
            doc="""\
This option determines whether image metadata should be used to rescale
the image’s intensities. Some image formats save the maximum possible
intensity value along with the pixel data. For instance, a microscope
might acquire images using a 12-bit A/D converter which outputs
intensity values between zero and 4095, but stores the values in a field
that can take values up to 65535.

Select *{YES}* to rescale the image intensity so that the camera's maximum
possible intensity value is rescaled to 1.0 (by dividing all pixels in
the image by the camera's maximum possible intensity value, as indicated by
image metadata).

Select *{NO}* to ignore the image metadata and rescale the image to a
maximum of 1.0 by dividing by 255 or 65535, depending on the maximum possible
intensity value of the image file format.
""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

        self.image_directory = Directory(
            "Base image location",
            dir_choices=DIR_ALL,
            allow_metadata=False,
            doc="""\
The parent (base) folder where images are located. If images are
contained in subfolders, then the file you load with this module should
contain a column with path names relative to the base image folder (see
the general help for this module for more details). You can choose among
the following options:

-  *Default Input Folder:* Use the Default Input Folder.
-  *Default Output Folder:* Use the Default Output Folder.
-  *None:* You have an *Image\_PathName* field that supplies an absolute
   path.
-  *Elsewhere…*: Use a particular folder you specify.""",
        )

        self.wants_image_groupings = Binary(
            "Group images by metadata?",
            False,
            doc="""\
Select *{YES}* to break the image sets in an experiment into groups.
Each set of files that share your selected metadata tags will be processed
together. For example, see **CreateBatchFiles** for details on submitting a
CellProfiler pipeline to a computing cluster for processing groups
separately, and see the **Groups** module for other examples.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.metadata_fields = MultiChoice(
            "Select metadata tags for grouping",
            None,
            doc="""\
*(Used only if images are to be grouped by metadata)*

Select the tags by which you want to group the image files. You can
select multiple tags. For example, if a set of images had metadata for
“Run”, “Plate”, “Well”, and “Site”, selecting *Run* and *Plate* will
create groups containing images that share the same [*Run*,\ *Plate*]
pair of tags.""",
        )

        self.wants_rows = Binary(
            "Process just a range of rows?",
            False,
            doc="""\
Select *{YES}* if you want to process a subset of the rows in the CSV
file. In the boxes below, enter the number of the row you want to begin processing
with in the box on the left. Then, enter the number of the row you want to
end processing with in the box on the right. Rows are numbered starting at 1
(but do not count the header line). **LoadData** will process up to and including the end row.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.row_range = IntegerRange(
            "Rows to process",
            (1, 100000),
            1,
            doc="""\
*(Used only if a range of rows is to be specified)*

Enter the row numbers of the first and last row to be processed.""",
        )

        def do_reload():
            global header_cache
            header_cache = {}
            try:
                self.open_csv()
            except:
                pass

        self.clear_cache_button = DoSomething(
            "Reload cached information",
            "Reload",
            do_reload,
            doc="""\
*(Used only if the CSV is loaded from a URL)*

Press this button to reload header information saved inside
CellProfiler. **LoadData** caches information about your .csv file in
its memory for efficiency. The information is reloaded if a modification
is detected. **LoadData** might fail to detect a modification on a file
accessed over the network and will fail to detect modifications on files
accessed through HTTP or FTP. In this case, you will have to use this
button to reload the header information after changing the file.

This button will never destroy any information on disk. It is always
safe to press it.""",
        )

    def settings(self):
        return [
            self.csv_directory,
            self.csv_file_name,
            self.wants_images,
            self.image_directory,
            self.wants_rows,
            self.row_range,
            self.wants_image_groupings,
            self.metadata_fields,
            self.rescale,
        ]

    def validate_module(self, pipeline):
        csv_path = self.csv_path

        if self.csv_directory.dir_choice != URL_FOLDER_NAME:
            if not os.path.isfile(csv_path):
                raise ValidationError(
                    "No such CSV file: %s" % csv_path, self.csv_file_name
                )

        try:
            self.open_csv()
        except IOError as e:
            import errno

            if e.errno == errno.EWOULDBLOCK:
                raise ValidationError(
                    "Another program (Excel?) is locking the CSV file %s."
                    % self.csv_path,
                    self.csv_file_name,
                )
            else:
                raise ValidationError(
                    "Could not open CSV file %s (error: %s)" % (self.csv_path, e),
                    self.csv_file_name,
                )

        try:
            self.get_header()
        except Exception as e:
            raise ValidationError(
                "The CSV file, %s, is not in the proper format."
                " See this module's help for details on CSV format. (error: %s)"
                % (self.csv_path, e),
                self.csv_file_name,
            )

    def validate_module_warnings(self, pipeline):
        """Check for potentially dangerous settings

        The best practice is to have a single LoadImages or LoadData module.
        """
        for module in pipeline.modules():
            if id(module) == id(self):
                return
            if isinstance(module, LoadData):
                raise ValidationError(
                    "Your pipeline has two or more LoadData modules.\n"
                    "The best practice is to have only one LoadData module.\n"
                    "Consider combining the CSV files from all of your\n"
                    "LoadData modules into one and using only a single\n"
                    "LoadData module",
                    self.csv_file_name,
                )

        # check that user has selected fields for grouping if grouping is turned on
        if self.wants_image_groupings.value and (
            len(self.metadata_fields.selections) == 0
        ):
            raise ValidationError(
                "Group images by metadata is True, but no metadata "
                "tags have been chosen for grouping.",
                self.metadata_fields,
            )

    def visible_settings(self):
        result = [self.csv_directory, self.csv_file_name, self.browse_csv_button]
        if self.csv_directory.dir_choice == URL_FOLDER_NAME:
            result += [self.clear_cache_button]
            self.csv_file_name.text = "URL of the file"
            self.csv_file_name.set_browsable(False)
        else:
            self.csv_file_name.text = "Name of the file"
            self.csv_file_name.set_browsable(True)
        result += [self.wants_images]
        if self.wants_images.value:
            result += [self.rescale, self.image_directory, self.wants_image_groupings]
            if self.wants_image_groupings.value:
                result += [self.metadata_fields]
                try:
                    fields = [
                        field[len("Metadata_") :]
                        for field in self.get_header()
                        if field.startswith("Metadata_")
                    ]
                    if self.has_synthetic_well_metadata():
                        fields += [FTR_WELL]
                    self.metadata_fields.choices = fields
                except:
                    self.metadata_fields.choices = ["No CSV file"]

        result += [self.wants_rows]
        if self.wants_rows.value:
            result += [self.row_range]
        return result

    @property
    def csv_path(self):
        """The path and file name of the CSV file to be loaded"""
        if get_data_file() is not None:
            return get_data_file()
        if self.csv_directory.dir_choice == URL_FOLDER_NAME:
            return self.csv_file_name.value

        path = self.csv_directory.get_absolute_path()
        return os.path.join(path, self.csv_file_name.value)

    @property
    def image_path(self):
        return self.image_directory.get_absolute_path()

    @property
    def legacy_field_key(self):
        """The key to use to retrieve the metadata from the image set list"""
        return "LoadTextMetadata_%d" % self.module_num

    def get_cache_info(self):
        """Get the cached information for the data file"""
        global header_cache
        entry = header_cache.get(self.csv_path, dict(ctime=0))
        if is_url_path(self.csv_path):
            if self.csv_path not in header_cache:
                header_cache[self.csv_path] = entry
            return entry
        ctime = os.stat(self.csv_path).st_ctime
        if ctime > entry["ctime"]:
            entry = header_cache[self.csv_path] = {}
            entry["ctime"] = ctime
        return entry

    def open_csv(self, do_not_cache=False):
        """Open the csv file or URL, returning a file descriptor"""
        global header_cache

        if is_url_path(self.csv_path):
            if self.csv_path not in header_cache:
                header_cache[self.csv_path] = {}
            entry = header_cache[self.csv_path]
            if "URLEXCEPTION" in entry:
                raise entry["URLEXCEPTION"]
            if "URLDATA" in entry:
                fd = io.StringIO(entry["URLDATA"])
            else:
                if do_not_cache:
                    raise RuntimeError("Need to fetch URL manually.")
                try:
                    url = generate_presigned_url(self.csv_path)
                    url_fd = urllib.request.urlopen(url)
                except Exception as e:
                    entry["URLEXCEPTION"] = e
                    raise e
                fd = io.StringIO()
                while True:
                    text = url_fd.read()
                    if isinstance(text, bytes):
                        text = text.decode()
                    if len(text) == 0:
                        break
                    fd.write(text)
                fd.seek(0)
                entry["URLDATA"] = fd.getvalue()
            return fd
        else:
            return open(self.csv_path, "rt")

    def browse_csv(self):
        import wx
        from cellprofiler.gui.utilities.icon import get_cp_icon

        try:
            fd = self.open_csv()
        except:
            wx.MessageBox("Could not read %s" % self.csv_path)
            return
        reader = csv.reader(fd)
        header = next(reader)
        frame = wx.Frame(wx.GetApp().frame, title=self.csv_path)
        sizer = wx.BoxSizer(wx.VERTICAL)
        frame.SetSizer(sizer)
        list_ctl = wx.ListCtrl(frame, style=wx.LC_REPORT)
        sizer.Add(list_ctl, 1, wx.EXPAND)
        for i, field in enumerate(header):
            list_ctl.InsertColumn(i, field)
        for line in reader:
            list_ctl.Append(
                [s if isinstance(s, str) else s for s in line[: len(header)]]
            )
        frame.SetMinSize((640, 480))
        frame.SetIcon(get_cp_icon())
        frame.Fit()
        frame.Show()

    def get_header(self, do_not_cache=False):
        """Read the header fields from the csv file

        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        """
        entry = self.get_cache_info()
        if "header" in entry:
            return entry["header"]

        fd = self.open_csv(do_not_cache=do_not_cache)
        reader = csv.reader(fd)
        header = next(reader)
        fd.close()
        entry["header"] = [header_to_column(column) for column in header]
        return entry["header"]

    def get_image_names(self, do_not_cache=False):
        header = self.get_header(do_not_cache=do_not_cache)
        image_names = set(
            [
                get_image_name(field)
                for field in header
                if is_file_name_feature(field) or is_url_name_feature(field)
            ]
        )
        return list(image_names)

    def get_object_names(self, do_not_cache=False):
        header = self.get_header(do_not_cache=do_not_cache)
        object_names = set(
            [
                get_objects_name(field)
                for field in header
                if is_objects_file_name_feature(field)
                or is_objects_url_name_feature(field)
            ]
        )
        return list(object_names)

    def other_providers(self, group):
        """Get name providers from the CSV header"""
        if group == "imagegroup" and self.wants_images.value:
            try:
                # do not load URLs automatically
                return self.get_image_names(do_not_cache=True)
            except Exception as e:
                return []
        elif group == "objectgroup" and self.wants_images:
            try:
                # do not load URLs automatically
                return self.get_object_names(do_not_cache=True)
            except Exception as e:
                return []

        return []

    def is_image_from_file(self, image_name):
        """Return True if LoadData provides the given image name"""
        providers = self.other_providers("imagegroup")
        return image_name in providers

    def is_load_module(self):
        """LoadData can make image sets so it's a load module"""
        return True

    def prepare_run(self, workspace):
        pipeline = workspace.pipeline
        m = workspace.measurements
        assert isinstance(m, Measurements)
        """Load the CSV file at the outset and populate the image set list"""
        if pipeline.in_batch_mode():
            return True
        fd = self.open_csv()
        reader = csv.reader(fd)
        header = [header_to_column(column) for column in next(reader)]
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
                row = [s if isinstance(s, str) else s for s in row]
                if len(row) != len(header):
                    raise ValueError(
                        "Row # %d has the wrong number of elements: %d. Expected %d"
                        % (idx, len(row), len(header))
                    )
                rows.append(row)
        else:
            rows = [
                [s if isinstance(s, str) else s for s in row]
                for row in reader
                if len(row) > 0
            ]
        fd.close()
        #
        # Check for correct # of columns
        #
        n_fields = len(header)
        for i, row in enumerate(rows):
            if len(row) < n_fields:
                text = (
                    "Error on line %d of %s.\n" '\n"%s"\n' "%d rows found, expected %d"
                ) % (i + 2, self.csv_file_name.value, ",".join(row), len(row), n_fields)
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
                if feature not in image_columns:
                    image_columns[feature] = []
                image_columns[feature].append(i)
            elif category in OBJECTS_CATEGORIES:
                if feature not in object_columns:
                    object_columns[feature] = []
                object_columns[feature].append(i)
            else:
                metadata_columns[column] = i
                if category == C_METADATA:
                    if feature.lower() == FTR_WELL.lower():
                        well_well_column = i
                    elif is_well_row_token(feature):
                        well_row_column = i
                    elif is_well_column_token(feature):
                        well_column_column = i

        if (
            well_row_column is not None
            and well_column_column is not None
            and well_well_column is None
        ):
            # add a synthetic well column
            metadata_columns[M_WELL] = len(header)
            header.append(M_WELL)
            for row in rows:
                row.append(row[well_row_column] + row[well_column_column])
        if self.wants_images:
            #
            # Add synthetic object and image columns
            #
            if self.image_directory.dir_choice == NO_FOLDER_NAME:
                path_base = ""
            else:
                path_base = self.image_path
            for d, url_category, file_name_category, path_name_category in (
                (image_columns, C_URL, C_FILE_NAME, C_PATH_NAME,),
                (
                    object_columns,
                    C_OBJECTS_URL,
                    C_OBJECTS_FILE_NAME,
                    C_OBJECTS_PATH_NAME,
                ),
            ):
                for name in list(d.keys()):
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
                                (
                                    "LoadData needs a %s_%s column to match the "
                                    "%s_%s column"
                                )
                                % (file_name_category, name, path_name_category, name)
                            )
                        #
                        # Add URL column
                        #
                        d[name].append(len(header))
                        url_feature = "_".join((url_category, name))
                        header.append(url_feature)
                        for row in rows:
                            if path_name_column is None:
                                fullname = os.path.join(
                                    path_base, row[file_name_column]
                                )
                            else:
                                row_path_name = os.path.join(
                                    path_base, row[path_name_column]
                                )
                                fullname = os.path.join(
                                    row_path_name, row[file_name_column]
                                )
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
                                    row += [url[: (idx + 1)], url[(idx + 1) :]]
                            else:
                                row += [url[:idx], url[(idx + 1) :]]

        column_type = {}
        for column in self.get_measurement_columns(pipeline):
            column_type[column[1]] = column[2]

        previous_column_types = dict(
            [
                (c[1], c[2])
                for c in pipeline.get_measurement_columns(self)
                if c[0] == "Image"
            ]
        )
        #
        # Arrange the metadata into columns
        #
        columns = {}
        for index, feature in enumerate(header):
            c = []
            columns[feature] = c
            for row in rows:
                value = row[index]
                if feature in column_type:
                    datatype = column_type[feature]
                else:
                    datatype = previous_column_types[feature]
                if datatype == COLTYPE_INTEGER:
                    value = int(value)
                elif datatype == COLTYPE_FLOAT:
                    value = float(value)
                c.append(value)

        if len(metadata_columns) > 0:
            # Reorder the rows by matching metadata against previous metadata
            # (for instance, to assign metadata values to images from
            #  loadimages)
            #
            image_numbers = m.match_metadata(
                list(metadata_columns.keys()),
                [columns[k] for k in list(metadata_columns.keys())],
            )
            image_numbers = numpy.array(image_numbers, int).flatten()
            max_image_number = numpy.max(image_numbers)
            new_columns = {}
            for key, values in list(columns.items()):
                new_values = [None] * max_image_number
                for image_number, value in zip(image_numbers, values):
                    new_values[image_number - 1] = value
                new_columns[key] = new_values
            columns = new_columns
        for feature, values in list(columns.items()):
            m.add_all_measurements("Image", feature, values)
        if self.wants_image_groupings and len(self.metadata_fields.selections) > 0:
            keys = ["_".join((C_METADATA, k)) for k in self.metadata_fields.selections]
            m.set_grouping_tags(keys)
            groupkeys, groupvals = self.get_groupings(workspace)
            group_lengths = []
            for eachval in groupvals:
                group_lengths += [len(eachval[1])] * len(eachval[1])
            m.add_all_measurements(
                "Image", GROUP_LENGTH, group_lengths,
            )
        else:
            group_lengths = [len(rows)] * len(rows)
            m.add_all_measurements(
                "Image", GROUP_LENGTH, group_lengths,
            )

        return True

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Prepare to create a batch file

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
        """

        if self.wants_images:
            m = workspace.measurements
            assert isinstance(m, Measurements)
            image_numbers = m.get_image_numbers()
            all_image_features = m.get_feature_names("Image")
            for url_category, file_category, path_category, names in (
                (C_URL, C_FILE_NAME, C_PATH_NAME, self.get_image_names(),),
                (
                    C_OBJECTS_URL,
                    C_OBJECTS_FILE_NAME,
                    C_OBJECTS_PATH_NAME,
                    self.get_object_names(),
                ),
            ):
                for name in names:
                    url_feature = "_".join((url_category, name))
                    path_feature = "_".join((path_category, name))
                    if path_feature not in all_image_features:
                        path_feature = None
                    file_feature = "_".join((file_category, name))
                    if file_feature not in all_image_features:
                        file_feature = None
                    urls = m.get_measurement(
                        "Image", url_feature, image_set_number=image_numbers,
                    )
                    for image_number, url in zip(image_numbers, urls):
                        url = url
                        if url.lower().startswith("file:"):
                            fullname = url2pathname(url)
                            fullname = fn_alter_path(fullname)
                            path, filename = os.path.split(fullname)
                            url = str(pathname2url(fullname))
                            m.add_measurement(
                                "Image",
                                url_feature,
                                url,
                                image_set_number=image_number,
                            )
                            if file_feature is not None:
                                m.add_measurement(
                                    "Image",
                                    file_feature,
                                    filename,
                                    image_set_number=image_number,
                                )
                            if path_feature is not None:
                                m.add_measurement(
                                    "Image",
                                    path_feature,
                                    path,
                                    image_set_number=image_number,
                                )

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
        url = measurements.get_measurement("Image", url_feature)
        url = url
        full_filename = url2pathname(url)
        path, filename = os.path.split(full_filename)
        if measurements.has_feature("Image", series_feature):
            series = measurements["Image", series_feature]
        else:
            series = None
        if measurements.has_feature("Image", frame_feature):
            frame = measurements["Image", frame_feature]
        else:
            frame = None
        return FileImage(
            name,
            path,
            filename,
            rescale=self.rescale.value and is_image_name,
            series=series,
            index=frame,
        )

    def run(self, workspace):
        """Populate the images and objects"""
        m = workspace.measurements
        assert isinstance(m, Measurements)
        image_set = workspace.image_set
        object_set = workspace.object_set
        statistics = []
        features = [
            x[1]
            for x in self.get_measurement_columns(workspace.pipeline)
            if x[0] == "Image"
        ]

        if self.wants_images:
            #
            # Load the image. Calculate the MD5 hash of every image
            #
            image_size = None
            for image_name in self.other_providers("imagegroup"):
                provider = self.fetch_provider(image_name, m)
                image_set.providers.append(provider)
                image = image_set.get_image(image_name)
                pixel_data = image.pixel_data
                m.add_image_measurement(
                    "_".join((C_MD5_DIGEST, image_name)), provider.get_md5_hash(m),
                )
                m.add_image_measurement(
                    "_".join((C_SCALING, image_name)), image.scale,
                )
                m.add_image_measurement(
                    "_".join((C_HEIGHT, image_name)), int(pixel_data.shape[0]),
                )
                m.add_image_measurement(
                    "_".join((C_WIDTH, image_name)), int(pixel_data.shape[1]),
                )
                if image_size is None:
                    image_size = tuple(pixel_data.shape[:2])
                    first_filename = image.file_name
                elif tuple(pixel_data.shape[:2]) != image_size:
                    warning = bad_sizes_warning(
                        image_size, first_filename, pixel_data.shape, image.file_name
                    )
                    if self.show_window:
                        workspace.display_data.warning = warning
                    else:
                        print(warning)
                        #
                        # Process any object tags
                        #
            objects_names = self.get_object_names()
            for objects_name in objects_names:
                provider = self.fetch_provider(objects_name, m, is_image_name=False)
                image = provider.provide_image(workspace.image_set)
                pixel_data = convert_image_to_objects(image.pixel_data)
                o = Objects()
                o.segmented = pixel_data
                object_set.add_objects(o, objects_name)
                add_object_count_measurements(m, objects_name, o.count)
                add_object_location_measurements(m, objects_name, pixel_data)

        for feature_name in sorted(features):
            value = m.get_measurement("Image", feature_name)
            statistics.append((feature_name, value))

        if self.show_window:
            workspace.display_data.statistics = statistics

    def display(self, workspace, figure):
        # if hasattr(workspace.display_data, "warning"):
        #     from cellprofiler.gui.errordialog import show_warning
        #     show_warning("Images have different sizes",
        #                  workspace.display_data.warning,
        #                  cpprefs.get_show_report_bad_sizes_dlg,
        #                  cpprefs.set_show_report_bad_sizes_dlg)

        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, workspace.display_data.statistics)

    def get_groupings(self, workspace):
        """Return the image groupings of the image sets

        See CPModule for documentation
        """
        if (
            self.wants_images.value
            and self.wants_image_groupings.value
            and len(self.metadata_fields.selections) > 0
        ):
            keys = ["_".join((C_METADATA, k)) for k in self.metadata_fields.selections]
            if len(keys) == 0:
                return None
            m = workspace.measurements
            assert isinstance(m, Measurements)
            return keys, m.get_groupings(keys)
        return None

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements produced by this module"""
        entry = None
        try:
            entry = self.get_cache_info()
            if "measurement_columns" in entry:
                return entry["measurement_columns"]
            fd = self.open_csv()
            reader = csv.reader(fd)
            header = [header_to_column(x) for x in next(reader)]
        except:
            if entry is not None:
                entry["measurement_columns"] = []
            return []
        previous_columns = pipeline.get_measurement_columns(self)
        previous_fields = set([x[1] for x in previous_columns if x[0] == "Image"])
        already_output = [x in previous_fields for x in header]
        coltypes = [COLTYPE_INTEGER] * len(header)
        #
        # Make sure the well_column column type is a string
        #
        for i in range(len(header)):
            if header[i].startswith(C_METADATA + "_") and is_well_column_token(
                header[i].split("_")[1]
            ):
                coltypes[i] = COLTYPE_VARCHAR
            if any(
                [
                    header[i].startswith(x)
                    for x in (
                        C_PATH_NAME,
                        C_FILE_NAME,
                        C_OBJECTS_FILE_NAME,
                        C_OBJECTS_PATH_NAME,
                        C_URL,
                        C_OBJECTS_URL,
                    )
                ]
            ):
                coltypes[i] = COLTYPE_VARCHAR

        collen = [0] * len(header)
        key_is_path_or_file_name = [
            (
                key.startswith(C_PATH_NAME)
                or key.startswith(C_FILE_NAME)
                or key.startswith(C_OBJECTS_FILE_NAME)
                or key.startswith(C_OBJECTS_PATH_NAME)
            )
            for key in header
        ]
        key_is_path_or_url = [
            (
                key.startswith(C_PATH_NAME)
                or key.startswith(C_OBJECTS_PATH_NAME)
                or key.startswith(C_URL)
                or key.startswith(C_OBJECTS_URL)
            )
            for key in header
        ]

        for row in reader:
            if len(row) > len(header):
                row = row[: len(header)]
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
                    len_field = max(PATH_NAME_LENGTH, len_field + PATH_PADDING,)
                if coltypes[index] != COLTYPE_VARCHAR:
                    ldtype = get_loaddata_type(field)
                    if coltypes[index] == COLTYPE_INTEGER:
                        coltypes[index] = ldtype
                    elif coltypes[index] == COLTYPE_FLOAT and ldtype != COLTYPE_INTEGER:
                        coltypes[index] = ldtype

                if collen[index] < len(field):
                    collen[index] = len(field)

        for index in range(len(header)):
            if coltypes[index] == COLTYPE_VARCHAR:
                coltypes[index] = COLTYPE_VARCHAR_FORMAT % collen[index]

        image_names = self.other_providers("imagegroup")
        result = [
            ("Image", colname, coltype)
            for colname, coltype in zip(header, coltypes)
            if colname not in previous_fields
        ]
        if self.wants_images:
            for feature, coltype in (
                (C_URL, COLTYPE_VARCHAR_PATH_NAME,),
                (C_PATH_NAME, COLTYPE_VARCHAR_PATH_NAME,),
                (C_FILE_NAME, COLTYPE_VARCHAR_FILE_NAME,),
                (C_MD5_DIGEST, COLTYPE_VARCHAR_FORMAT % 32,),
                (C_SCALING, COLTYPE_FLOAT,),
                (C_HEIGHT, COLTYPE_INTEGER,),
                (C_WIDTH, COLTYPE_INTEGER,),
            ):
                for image_name in image_names:
                    measurement = feature + "_" + image_name
                    if not any([measurement == c[1] for c in result]):
                        result.append(("Image", measurement, coltype,))
            #
            # Add the object features
            #
            for object_name in self.get_object_names():
                result += get_object_measurement_columns(object_name)
                for feature, coltype in (
                    (C_OBJECTS_URL, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_OBJECTS_PATH_NAME, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_OBJECTS_FILE_NAME, COLTYPE_VARCHAR_FILE_NAME,),
                ):
                    mname = C_OBJECTS_URL + "_" + object_name
                    result.append(("Image", mname, coltype))
        #
        # Try to make a well column out of well row and well column
        #
        well_column = None
        well_row_column = None
        well_col_column = None
        for column in result:
            if not column[1].startswith(C_METADATA + "_"):
                continue
            category, feature = column[1].split("_", 1)
            if is_well_column_token(feature):
                well_col_column = column
            elif is_well_row_token(feature):
                well_row_column = column
            elif feature.lower() == FTR_WELL.lower():
                well_column = column
        if (
            well_column is None
            and well_row_column is not None
            and well_col_column is not None
        ):
            length = get_length_from_varchar(well_row_column[2])
            length += get_length_from_varchar(well_col_column[2])
            result += [
                (
                    "Image",
                    "_".join((C_METADATA, FTR_WELL,)),
                    COLTYPE_VARCHAR_FORMAT % length,
                )
            ]
        entry["measurement_columns"] = result
        return result

    def has_synthetic_well_metadata(self):
        """Determine if we should synthesize a well metadata feature

        """
        fields = self.get_header()
        has_well_col = False
        has_well_row = False
        for field in fields:
            if not field.startswith(C_METADATA + "_"):
                continue
            category, feature = field.split("_", 1)
            if is_well_column_token(feature):
                has_well_col = True
            elif is_well_row_token(feature):
                has_well_row = True
            elif feature.lower() == FTR_WELL.lower():
                return False
        return has_well_col and has_well_row

    def get_categories(self, pipeline, object_name):
        try:
            columns = self.get_measurement_columns(pipeline)
            result = set(
                [
                    column[1].split("_")[0]
                    for column in columns
                    if column[0] == object_name
                ]
            )
            return list(result)
        except:
            return []

    def get_measurements(self, pipeline, object_name, category):
        columns = self.get_measurement_columns(pipeline)
        return [
            feature
            for c, feature in [
                column[1].split("_", 1)
                for column in columns
                if column[0] == object_name and column[1].startswith(category + "_")
            ]
        ]

    def change_causes_prepare_run(self, setting):
        """Check to see if changing the given setting means you have to restart

        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to do "prepare_run".
        """
        if self.wants_images or setting == self.wants_images:
            return True
        return False

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):

        dir_default_image = "Default input folder"
        dir_default_output = "Default Output Folder"

        if variable_revision_number == 1:
            setting_values = setting_values + ["No", ""]
            variable_revision_number = 2

        if variable_revision_number == 2:
            if setting_values[0].startswith("Default Image"):
                setting_values = [dir_default_image] + setting_values[1:]
            elif setting_values[0].startswith("Default Output"):
                setting_values = [dir_default_output] + setting_values[1:]
            if setting_values[4].startswith("Default Image"):
                setting_values = (
                    setting_values[:4] + [dir_default_image] + setting_values[5:]
                )
            elif setting_values[4].startswith("Default Output"):
                setting_values = (
                    setting_values[:4] + [dir_default_output] + setting_values[5:]
                )
            variable_revision_number = 3
        if variable_revision_number == 3:
            module_name = self.module_name

        if variable_revision_number == 3:
            # directory choice, custom directory merged
            # input_directory_choice, custom_input_directory merged
            (
                csv_directory_choice,
                csv_custom_directory,
                csv_file_name,
                wants_images,
                image_directory_choice,
                image_custom_directory,
                wants_rows,
                row_range,
                wants_image_groupings,
                metadata_fields,
            ) = setting_values
            csv_directory = Directory.static_join_string(
                csv_directory_choice, csv_custom_directory
            )
            image_directory = Directory.static_join_string(
                image_directory_choice, image_custom_directory
            )
            setting_values = [
                csv_directory,
                csv_file_name,
                wants_images,
                image_directory,
                wants_rows,
                row_range,
                wants_image_groupings,
                metadata_fields,
            ]
            variable_revision_number = 4

        # Standardize input/output directory name references
        setting_values = list(setting_values)
        for index in (0, 3):
            setting_values[index] = Directory.upgrade_setting(setting_values[index])

        if variable_revision_number == 4:
            (
                csv_directory,
                csv_file_name,
                wants_images,
                image_directory,
                wants_rows,
                row_range,
                wants_image_groupings,
                metadata_fields,
            ) = setting_values
            (dir_choice, custom_dir,) = Directory.split_string(csv_directory)
            if dir_choice == URL_FOLDER_NAME:
                csv_file_name = custom_dir + "/" + csv_file_name
                csv_directory = Directory.static_join_string(dir_choice, "")
            setting_values = [
                csv_directory,
                csv_file_name,
                wants_images,
                image_directory,
                wants_rows,
                row_range,
                wants_image_groupings,
                metadata_fields,
            ]
            variable_revision_number = 5
        if variable_revision_number == 5:
            # Added rescaling option
            setting_values = setting_values + ["Yes"]
            variable_revision_number = 6
        return setting_values, variable_revision_number


LoadText = LoadData
