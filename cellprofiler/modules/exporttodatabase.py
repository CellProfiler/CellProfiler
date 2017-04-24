import cellprofiler.icons
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON

__doc__ = '''<b>Export To Database</b> exports data directly to a database, or in
database readable format, including an imported file
with column names and a CellProfiler Analyst properties file, if desired.
<hr>
This module exports measurements directly to a database or to a SQL-compatible format.
It allows you to create and import MySQL and associated data files into a
database and gives you the option of creating
a properties file for use with CellProfiler Analyst. Optionally, you can create
an SQLite database file if you do not have a server on which to run MySQL itself.

This module must be run at the end of a pipeline, or second to last if
you are using the <b>CreateBatchFiles</b> module. If you forget this module, you
can also run the <i>ExportDatabase</i> data tool after processing is complete;
its functionality is the same.

The database is set up with two primary tables. These tables are the
<i>Per_Image</i> table and the <i>Per_Object</i> table (which may have a prefix if you
specify):
<ul>
<li>The Per_Image table consists of all the per-image measurements made during the pipeline, plus
per-image population statistics (such as mean, median, and standard deviation) of the object measurements. There is one
per_image row for every "cycle" that CellProfiler processes (a cycle is usually a single field of view, and a single cycle
usually contains several image files, each representing a different channel of the same field of view). </li>
<li>The Per_Object table contains all the
measurements for individual objects. There is one row of object
measurements per object identified. The two tables are connected with the
primary key column <i>ImageNumber</i>, which indicates the image to which each object belongs. The Per_Object table has another primary
key called <i>ObjectNumber</i>, which is unique to each image. </li>
</ul>

Typically, if multiple types of objects are identified and measured in a pipeline,
the numbers of those objects are equal to each other. For example, in most pipelines, each nucleus has exactly one cytoplasm, so the first row
of the Per-Object table contains all of the information about object #1, including both nucleus- and cytoplasm-related measurements. If this
one-to-one correspondence is <i>not</i> the case for all objects in the pipeline (for example, if dozens of speckles are identified and
measured for each nucleus), then you must configure <b>ExportToDatabase</b> to export only objects that maintain the one-to-one correspondence
(for example, export only <i>Nucleus</i> and <i>Cytoplasm</i>, but omit <i>Speckles</i>).

If you have extracted "Plate" and "Well" metadata from image filenames or loaded "Plate" and "Well" metadata via the <b>Metadata</b>
or <b>LoadData</b> modules, you can ask CellProfiler to create a "Per_Well" table, which aggregates object measurements across wells.
This option will output a SQL file (regardless of whether you choose to write directly to the database)
that can be used to create the Per_Well table. At the secure shell where you normally log in to MySQL, type
the following, replacing the italics with references to your database and files:

<tt>mysql -h <i>hostname</i> -u <i>username</i> -p <i>databasename</i> &lt <i>pathtoimages/perwellsetupfile.SQL</i></tt>

The commands written by CellProfiler to create the Per_Well table will be executed.

Oracle is not fully supported at present; you can create your own Oracle DB using
the .csv output option and writing a simple script to upload to the database.

<h4>Available measurements</h4>
For details on the nomenclature used by CellProfiler for the exported measurements,
see <i>Help > General Help > How Measurements Are Named</i>.

See also <b>ExportToSpreadsheet</b>.

'''

import csv
import datetime
import hashlib
import logging
import numpy as np
import os
import random
import re
import sys
import traceback

logger = logging.getLogger(__name__)
try:
    import MySQLdb
    from MySQLdb.cursors import SSCursor

    HAS_MYSQL_DB = True
except:
    logger.warning("MySQL could not be loaded.", exc_info=True)
    HAS_MYSQL_DB = False

import cellprofiler
import cellprofiler.module as cpm
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import cellprofiler.preferences as cpprefs
import cellprofiler.measurement as cpmeas
from cellprofiler.pipeline import GROUP_INDEX, M_MODIFICATION_TIMESTAMP
from cellprofiler.measurement import M_NUMBER_OBJECT_NUMBER
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF, USING_METADATA_GROUPING_HELP_REF
from cellprofiler.preferences import \
    standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
    DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME, \
    DEFAULT_OUTPUT_SUBFOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
    IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

##############################################
#
# Keyword for the cached measurement columns
#
##############################################
D_MEASUREMENT_COLUMNS = "MeasurementColumns"

'''The column name for the image number column'''
C_IMAGE_NUMBER = "ImageNumber"

'''The column name for the object number column'''
C_OBJECT_NUMBER = "ObjectNumber"
D_IMAGE_SET_INDEX = "ImageSetIndex"

'''The thumbnail category'''
C_THUMBNAIL = "Thumbnail"

##############################################
#
# Database options for the db_type setting
#
##############################################
DB_MYSQL = "MySQL"
DB_ORACLE = "Oracle"
DB_SQLITE = "SQLite"
DB_MYSQL_CSV = "MySQL / CSV"

##############################################
#
# Choices for which objects to include
#
##############################################

'''Put all objects in the database'''
O_ALL = "All"
'''Don't put any objects in the database'''
O_NONE = "None"
'''Select the objects you want from a list'''
O_SELECT = "Select..."

##############################################
#
# Choices for properties file
#
##############################################
NONE_CHOICE = "None"
PLATE_TYPES = [NONE_CHOICE, "6", "24", "96", "384", "1536", "5600"]
COLOR_ORDER = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray", "none"]
GROUP_COL_DEFAULT = "ImageNumber, Image_Metadata_Plate, Image_Metadata_Well"
CT_IMAGE = "Image"
CT_OBJECT = "Object"
CLASSIFIER_TYPE = [CT_OBJECT, CT_IMAGE]

##############################################
#
# Choices for workspace file
#
##############################################
W_DENSITYPLOT = "DensityPlot"
W_HISTOGRAM = "Histogram"
W_SCATTERPLOT = "ScatterPlot"
W_PLATEVIEWER = "PlateViewer"
W_BOXPLOT = "BoxPlot"
W_DISPLAY_ALL = [W_SCATTERPLOT, W_HISTOGRAM, W_PLATEVIEWER, W_DENSITYPLOT, W_BOXPLOT]
W_INDEX = "Index"
W_TYPE_ALL = [cpmeas.IMAGE, cpmeas.OBJECT, W_INDEX]
W_INDEX_ALL = [C_IMAGE_NUMBER, GROUP_INDEX]

################################################
#
# Choices for overwrite
#
################################################

OVERWRITE_NEVER = "Never"
OVERWRITE_DATA = "Data only"
OVERWRITE_ALL = "Data and schema"

"""Offset of the image group count in the settings"""
SETTING_IMAGE_GROUP_COUNT = 29

"""Offset of the group specification group count in the settings"""
SETTING_GROUP_FIELD_GROUP_COUNT = 30

"""Offset of the filter specification group count in the settings"""
SETTING_FILTER_FIELD_GROUP_COUNT = 31

"""Offset of the workspace specification group count in the settings"""
SETTING_WORKSPACE_GROUP_COUNT = 32

SETTING_OFFSET_PROPERTIES_IMAGE_URL_PREPEND_V26 = 21

SETTING_FIXED_SETTING_COUNT_V21 = 33

SETTING_FIXED_SETTING_COUNT_V22 = 35

SETTING_FIXED_SETTING_COUNT_V23 = 36

SETTING_FIXED_SETTING_COUNT_V24 = 37

SETTING_FIXED_SETTING_COUNT_V25 = 38

SETTING_FIXED_SETTING_COUNT_V26 = 39

SETTING_FIXED_SETTING_COUNT = 39

##############################################
#
# Choices for the output directory
#
##############################################
DIR_CUSTOM = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

##############################################
#
# Choices for object table format
#
##############################################

OT_PER_OBJECT = "One table per object type"
OT_COMBINE = "Single object table"
OT_VIEW = "Single object view"

'''Index of the object table format choice in the settings'''
OT_IDX = 17

'''Use this dictionary to keep track of rewording of above if it happens'''
OT_DICTIONARY = {
    "One table per object type": OT_PER_OBJECT,
    "Single object table": OT_COMBINE,
    "Single object view": OT_VIEW
}
from cellprofiler.measurement import C_PARENT, M_NUMBER_OBJECT_NUMBER

T_EXPERIMENT = "Experiment"
T_EXPERIMENT_PROPERTIES = "Experiment_Properties"

T_RELATIONSHIPS = "Relationships"
T_RELATIONSHIP_TYPES = "RelationshipTypes"
CONSTRAINT_RT_UNIQUE = "RelationshipTypesUnique"
FK_RELATIONSHIP_TYPE_ID = "RRTypeIdFK"
CONSTRAINT_R_UNIQUE = "RelationshipUnique"
V_RELATIONSHIPS = "RelationshipsView"
I_RELATIONSHIPS1 = "IRelationships1"
I_RELATIONSHIPS2 = "IRelationships2"
COL_RELATIONSHIP_TYPE_ID = "relationship_type_id"
COL_MODULE_NUMBER = "module_number"
COL_RELATIONSHIP = "relationship"
COL_OBJECT_NAME1 = "object_name1"
COL_OBJECT_NAME2 = "object_name2"
COL_IMAGE_NUMBER1 = "image_number1"
COL_IMAGE_NUMBER2 = "image_number2"
COL_OBJECT_NUMBER1 = "object_number1"
COL_OBJECT_NUMBER2 = "object_number2"


def execute(cursor, query, bindings=None, return_result=True):
    if bindings is None:
        cursor.execute(query)
    else:
        cursor.execute(query, bindings)
    if return_result:
        return get_results_as_list(cursor)


def get_results_as_list(cursor):
    r = get_next_result(cursor)
    l = []
    while r:
        l.append(r)
        r = get_next_result(cursor)
    return l


def get_next_result(cursor):
    try:
        return cursor.next()
    except MySQLdb.Error, e:
        raise Exception('Error retrieving next result from database: %s' % e)
    except StopIteration, e:
        return None


def connect_mysql(host, user, pw, db):
    '''Creates and returns a db connection and cursor.'''
    connection = MySQLdb.connect(host=host, user=user, passwd=pw, db=db)
    cursor = SSCursor(connection)
    #
    # Use utf-8 encoding for strings
    #
    connection.set_character_set('utf8')
    execute(cursor, "set names 'utf8'")
    execute(cursor, "set character set utf8")
    execute(cursor, "set character_set_connection=utf8")
    return connection, cursor


def connect_sqlite(db_file):
    '''Creates and returns a db connection and cursor.'''
    import sqlite3
    connection = sqlite3.connect(db_file, timeout=30)
    cursor = connection.cursor()
    return connection, cursor


class DBContext(object):
    '''A database context suitable for the "with" statement

    Usage:

    assert isinstance(self, ExportToDatabase)

    with DBContext(self):

       do stuff with self.connection & self.cursor

    # cursor and connection are closed. Changes are either committed
    # or rolled back depending on exception status
    '''

    def __init__(self, module):
        assert isinstance(module, ExportToDatabase)
        self.module = module

    def __enter__(self):
        if self.module.db_type == DB_MYSQL:
            self.connection, self.cursor = connect_mysql(
                    self.module.db_host.value,
                    self.module.db_user.value,
                    self.module.db_passwd.value,
                    self.module.db_name.value)
        elif self.module.db_type == DB_SQLITE:
            db_file = self.module.make_full_filename(
                    self.module.sqlite_file.value)
            self.connection, self.cursor = connect_sqlite(db_file)
        return self.connection, self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()


class ExportToDatabase(cpm.Module):
    module_name = "ExportToDatabase"
    variable_revision_number = 27
    category = ["File Processing", "Data Tools"]

    def create_settings(self):
        db_choices = ([DB_MYSQL, DB_MYSQL_CSV, DB_SQLITE] if HAS_MYSQL_DB
                      else [DB_MYSQL_CSV, DB_SQLITE])
        default_db = DB_MYSQL if HAS_MYSQL_DB else DB_MYSQL_CSV
        self.db_type = cps.Choice(
                "Database type",
                db_choices, default_db, doc="""
            Specify the type of database you want to use:
            <ul>
            <li><i>%(DB_MYSQL)s:</i> Writes the data directly to a MySQL
            database. MySQL is open-source software; you may require help from
            your local Information Technology group to set up a database
            server.</li>
            <li><i>%(DB_MYSQL_CSV)s:</i> Writes a script file that
            contains SQL statements for creating a database and uploading the
            Per_Image and Per_Object tables. This option will write out the Per_Image
            and Per_Object table data to two CSV files; you can use these files can be
            used to import the data directly into an application
            that accepts CSV data.</li>
            <li><i>%(DB_SQLITE)s:</i> Writes SQLite files directly.
            SQLite is simpler to set up than MySQL and
            can more readily be run on your local computer rather than requiring a
            database server. More information about SQLite can be found
            <a href="http://www.sqlite.org/">here</a>. </li>
            </ul>
            <dl>
            <dd><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;
            If running this module on a computing cluster, there are a few
            considerations to note:
            <ul>
            <li>The <i>%(DB_MYSQL)s</i> option is well-suited for cluster use, since
            multiple jobs can write to the database simultaneously.</li>
            <li>The <i>%(DB_SQLITE)s</i> option is not
            as appropriate; a SQLite database only allows access by one job at a time.</li>
            </ul>
            </dd>
            </dl>""" % globals())

        self.test_connection_button = cps.DoSomething(
                "Press this button to test the connection to the remote server using the current settings",
                "Test connection", self.test_connection, doc="""
            This button test the connection to MySQL server specified using
            the settings entered by the user.""")

        self.db_name = cps.Text(
                "Database name", "DefaultDB", doc="""
            Select a name for the database you want to use""")

        self.experiment_name = cps.Text(
                "Experiment name", "MyExpt", doc="""
            Select a name for the experiment. This name will be
            registered in the database and linked to the tables that
            <b>ExportToDatabase</b> creates. You will be able to select the experiment
            by name in CellProfiler Analyst and will be able to find the
            experiment's tables through database queries.""")

        self.want_table_prefix = cps.Binary(
                "Add a prefix to table names?", True, doc="""
            Select whether you want to add a prefix to your table names. The
            default table names are <i>Per_Image</i> for the per-image table and
            <i>Per_Object</i> for the per-object table. Adding a prefix can be useful
            for bookkeeping purposes.
            <ul>
            <li>Select <i>%(YES)s</i> to add a user-specified prefix to the default table names.
            If you want to distinguish multiple sets of data written to the same
            database, you probably want to use a prefix.</li>
            <li>Select <i>%(NO)s</i> to use the default table names. For a one-time export of
            data, this option is fine. </li>
            </ul>
            Whether you chose to use a prefix or not, CellProfiler will warn
            you if your choice entails overwriting an existing table.""" % globals())

        self.table_prefix = cps.Text(
                "Table prefix", "MyExpt_", doc="""
            <i>(Used if Add a prefix to table names?</i> is selected)<br>
            Enter the table prefix you want to use.
            <p>MySQL has a 64 character limit on the full name of the table.
            If the combination of the table name and prefix exceeds this
            limit, you will receive an error associated with this setting.</p>""")

        self.sql_file_prefix = cps.Text(
                "SQL file prefix", "SQL_", doc="""
            <i>(Used if %(DB_MYSQL_CSV)s is selected as the database type)</i><br>
            Enter the prefix to be used to name the SQL file.""" % globals())

        self.directory = cps.DirectoryPath(
                "Output file location",
                dir_choices=[
                    DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME,
                    ABSOLUTE_FOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME,
                    DEFAULT_INPUT_SUBFOLDER_NAME], doc="""
            <i>(Used only when using a CSV or a SQLite database, and/or creating a properties or workspace file)</i><br>
            This setting determines where the CSV files or SQLite database is saved if
            you decide to write measurements to files instead of writing them
            directly to the database. If you request a CellProfiler Analyst properties file
            or workspace file, it will also be saved to this location. %(IO_FOLDER_CHOICE_HELP_TEXT)s

            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have a metadata tag named
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name with the "Plate" metadata tag.
            The module will substitute the metadata values for the last image set
            processed for any metadata tags in the folder name. %(USING_METADATA_HELP_REF)s.</p>""" % globals())
        self.directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        self.save_cpa_properties = cps.Binary(
                "Create a CellProfiler Analyst properties file?",
                False, doc="""
            Select <i>%(YES)s</i> to generate a template properties file that will allow you to use
            your new database with CellProfiler Analyst (a data
            exploration tool which can also be downloaded from
            <a href="http://www.cellprofiler.org/">http://www.cellprofiler.org/</a>).
            The module will attempt to fill in as many as the entries as possible
            based on the pipeline's settings, including the
            server name, username and password if MySQL is used.""" % globals())

        self.location_object = cps.ObjectNameSubscriber(
                "Which objects should be used for locations?", cps.NONE, doc="""
            <i>(Used only if creating a properties file)</i><br>
            CellProfiler Analyst displays cells during classification. This
            setting determines which object centers will be used as the center
            of the cells to be displayed. Choose one of the listed objects
            and CellProfiler will save that object's location columns in
            the properties file so that CellProfiler Analyst centers cells
            using that object's center.
            <p>You can manually change this choice in the properties file by
            editing the <i>cell_x_loc</i> and <i>cell_y_loc</i> properties.
            </p>
            <p>Note that if there are no objects defined in the pipeline (e.g.
            if only using MeasureImageQuality and/or Illumination Correction modules),
            a warning will diplay until you choose <i>'None'</i> for the subsequent setting:
            'Export measurements for all objects to the database?'.</p>
            """ % globals())

        self.wants_properties_image_url_prepend = cps.Binary(
                "Access CPA images via URL?", False,
                doc="""
            <i>(Used only if creating a properties file)</i><br>
            The image paths written to the database will be the absolute
            path the image files on your computer. If you plan to make these
            files accessible via the web, you can have CellProfiler Analyst prepend
            a URL to your file name.
            Eg: If an image is loaded from the path "/cellprofiler/images/" and you use
            a url prepend of "http://mysite.com/", CellProfiler Analyst will look
            for your file at "http://mysite.com/cellprofiler/images/"
            """)
        #
        # Hack: if user is on Broad IP, then plug in the imageweb url prepend
        #
        import socket
        try:
            fqdn = socket.getfqdn()
        except:
            fqdn = '127.0.0.1'
        default_prepend = ""
        if 'broadinstitute' in fqdn.lower():  # Broad
            default_prepend = "http://imageweb/images/CPALinks"

        self.properties_image_url_prepend = cps.Text(
                "Enter an image url prepend if you plan to access your files via http",
                default_prepend, doc="""
            <i>(Used only if accessing CellProfiler Analyst images via URL)</i><br>
            The image paths written to the database will be the absolute
            path the image files on your computer. If you plan to make these
            files accessible via the web, you can enter a url prefix here. Eg:
            If an image is loaded from the path "/cellprofiler/images/" and you use
            a url prepend of "http://mysite.com/", CellProfiler Analyst will look
            for your file at "http://mysite.com/cellprofiler/images/"
            <p>If you are not using the web to access your files (i.e., they are locally
            aceesible by your computer), leave this setting blank.""")

        self.properties_plate_type = cps.Choice(
                "Select the plate type",
                PLATE_TYPES, doc="""
            <i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the plate
            type here. Supported types in CellProfiler Analyst are 96- and 384-well plates,
            as well as 5600-spot microarrays. If you are not using a plate or microarray, select
            <i>None</i>.""")

        self.properties_plate_metadata = cps.Choice(
                "Select the plate metadata",
                ["None"], choices_fn=self.get_metadata_choices, doc="""
            <i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the metadata corresponding
            to the plate here. If there is no plate metadata associated with the image set, select
            <i>None</i>.
            <p>%(USING_METADATA_HELP_REF)s.</p>""" % globals())

        self.properties_well_metadata = cps.Choice(
                "Select the well metadata",
                ["None"], choices_fn=self.get_metadata_choices, doc="""
            <i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the metadata corresponding
            to the well here. If there is no well metadata associated with the image set, select
            <i>None</i>.
            <p>%(USING_METADATA_HELP_REF)s.</p>""" % globals())

        self.properties_export_all_image_defaults = cps.Binary(
                "Include information for all images, using default values?", True, doc="""
            <i>(Used only if creating a properties file)</i><br>
            Select <i>%(YES)s</i> to include information in the properties file for all images. This
            option will do the following:
            <ul>
            <li>All images loaded using the <b>Input</b> modules or saved in <b>SaveImages</b> will be included.</li>
            <li>The CellProfiler image name will be used for the <i>image_name</i> field.</li>
            <li>A channel color listed in the <i>image_channel_colors</i> field will be assigned to the image by default order.</li>
            </ul>
            <p>Select <i>%(NO)s</i> to specify which images should be included or to override the automatic values.</p>""" % globals())

        self.image_groups = []
        self.image_group_count = cps.HiddenCount(self.image_groups, "Properties image group count")
        self.add_image_group(False)
        self.add_image_button = cps.DoSomething("", "Add another image",
                                                self.add_image_group)

        self.properties_wants_groups = cps.Binary(
                "Do you want to add group fields?", False, doc="""
            <i>(Used only if creating a properties file)</i><br>
            <b>Please note that "groups" as defined by CellProfiler Analyst has nothing to do with "grouping" as defined by
            CellProfiler in the Groups module.</b>
            <p>Select <i>%(YES)s</i> to define a "group" for your image data (for example, when several images represent the same experimental
            sample), by providing column(s) that identify unique images (the <i>image key</i>) to another set of columns
            (the <i>group key</i>).</p>
            <p>The format for a group in CPA is:<br>
            <code>group_SQL_&lt;XXX&gt; = &lt;MySQL SELECT statement that returns image-key columns followed by group-key columns&gt;</code>
            For example, if you wanted to be able to group your data by unique plate names, you could define a group called
            <i>SQL_Plate</i> as follows:<br>
            <code>group_SQL_Plate = SELECT ImageNumber, Image_Metadata_Plate FROM Per_Image</code></p>
            <p>Grouping is useful, for example, when you want to aggregate counts for each class of object and their scores
            on a per-group basis (e.g., per-well) instead of on a per-image basis when scoring with Classifier. It will
            also provide new options in the Classifier fetch menu so you can fetch objects from images with specific
            values for the group columns.</p>""" % globals())

        self.group_field_groups = []
        self.group_field_count = cps.HiddenCount(self.group_field_groups, "Properties group field count")
        self.add_group_field_group(False)
        self.add_group_field_button = cps.DoSomething("", "Add another group",
                                                      self.add_group_field_group)

        self.properties_wants_filters = cps.Binary(
                "Do you want to add filter fields?", False, doc=
                """<i>(Used only if creating a properties file)</i><br>
                Select <i>%(YES)s</i> to specify a subset of the images in your experiment by defining a <i>filter</i>.
                Filters are useful, for example, for fetching and scoring objects in Classifier or making graphs using the
                plotting tools that satisfy a specific metadata contraint. """ % globals())

        self.create_filters_for_plates = cps.Binary(
                "Automatically create a filter for each plate?", False, doc="""
            <i>(Used only if creating a properties file and specifiying an image data filter)</i><br>
            If you have specified a plate metadata tag, selecting <i>%(YES)s</i> to create a set of filters
            in the properties file, one for each plate.""" % globals())

        self.filter_field_groups = []
        self.filter_field_count = cps.HiddenCount(self.filter_field_groups, "Properties filter field count")
        self.add_filter_field_button = cps.DoSomething("", "Add another filter",
                                                       self.add_filter_field_group)

        self.properties_class_table_name = cps.Text(
                "Enter a phenotype class table name if using the classifier tool",
                '', doc="""
            <i>(Used only if creating a properties file)</i><br>
            If you are using the machine-learning tool in CellProfiler Analyst,
            you can create an additional table in your database  which contains
            the per-object phenotype labels. This table is produced after scoring
            all the objects in your data set and will be named with the label given here.
            Note that the actual class table will be named by prepending the table prefix
            (if any) to what you enter here.
            <p>You can manually change this choice in the properties file by
            editing the <i>class_table</i> field. Leave this field blank if you are
            not using the classifier or do not need the table written to the database</p>.""")

        self.properties_classification_type = cps.Choice(
                "Select the classification type",
                CLASSIFIER_TYPE, doc="""
            <i>(Used only if creating a properties file)</i><br>
            Choose the type of classification this properties file will be used for. This
            setting will create and set a field called <i>classification_type</i>.
            Note that if you are not using the classifier tool, this setting will be ignored.
            <ul>
            <li><i>%(CT_OBJECT)s:</i> Object-based classification, i.e., set <i>classification_type</i>
            to "object" (or leave it blank).</li>
            <li><i>%(CT_IMAGE)s:</i> Image-based classification, e.g., set <i>classification_type</i>
            to "image".</li>
            </ul>
            You can manually change this choice in the properties file by
            editing the <i>classification_type</i> field.
            """ % globals())

        self.create_workspace_file = cps.Binary(
                "Create a CellProfiler Analyst workspace file?", False, doc="""
            Select <i>%(YES)s</i> to generate a workspace file for use with
            CellProfiler Analyst, a data exploration tool which can
            also be downloaded from <a href="http://www.cellprofiler.org/">
            http://www.cellprofiler.org/</a>. A workspace file allows you
            to open a selected set of measurements with the display tools
            of your choice. This is useful, for example, if you want examine a standard
            set of quality control image measurements for outliers.""" % globals())

        self.divider = cps.Divider(line=True)
        self.divider_props = cps.Divider(line=True)
        self.divider_props_wkspace = cps.Divider(line=True)
        self.divider_wkspace = cps.Divider(line=True)

        self.workspace_measurement_groups = []
        self.workspace_measurement_count = cps.HiddenCount(self.workspace_measurement_groups,
                                                           "Workspace measurement count")

        def add_workspace_measurement_group(can_remove=True):
            self.add_workspace_measurement_group(can_remove)

        add_workspace_measurement_group(False)
        self.add_workspace_measurement_button = cps.DoSomething("", "Add another measurement",
                                                                self.add_workspace_measurement_group)

        self.mysql_not_available = cps.Divider("Cannot write to MySQL directly - CSV file output only", line=False,
                                               doc="""The MySQLdb python module could not be loaded.  MySQLdb is necessary for direct export.""")

        self.db_host = cps.Text("Database host", "")

        self.db_user = cps.Text("Username", "")

        self.db_passwd = cps.Text("Password", "")

        self.sqlite_file = cps.Text(
                "Name the SQLite database file",
                "DefaultDB.db", doc="""
            <i>(Used if SQLite selected as database type)</i><br>
            Enter the name of the SQLite database filename to which you want to write.""")

        self.wants_agg_mean = cps.Binary(
                "Calculate the per-image mean values of object measurements?", True, doc="""
            Select <i>%(YES)s</i> for <b>ExportToDatabase</b> to calculate population statistics over all the objects in each image
            and store the results in the database. For instance, if
            you are measuring the area of the Nuclei objects and you check the box for this option, <b>ExportToDatabase</b> will create a column in the Per_Image
            table called "Mean_Nuclei_AreaShape_Area".
            <p>You may not want to use <b>ExportToDatabase</b> to calculate these population statistics if your pipeline generates
            a large number of per-object measurements; doing so might exceed database
            column limits. These columns can be created manually for selected measurements directly in MySQL.
            For instance, the following SQL command creates the Mean_Nuclei_AreaShape_Area column:

                <p><tt>ALTER TABLE Per_Image ADD (Mean_Nuclei_AreaShape_Area);
                UPDATE Per_Image SET Mean_Nuclei_AreaShape_Area =
                    (SELECT AVG(Nuclei_AreaShape_Area)
                     FROM Per_Object
                     WHERE Per_Image.ImageNumber = Per_Object.ImageNumber);</tt>""" % globals())

        self.wants_agg_median = cps.Binary("Calculate the per-image median values of object measurements?", False)
        self.wants_agg_std_dev = cps.Binary("Calculate the per-image standard deviation values of object measurements?",
                                            False)

        self.wants_agg_mean_well = cps.Binary(
                "Calculate the per-well mean values of object measurements?", False, doc='''
            Select <i>%(YES)s</i> for <b>ExportToDatabase</b> to calculate statistics over all the objects in each well
            and store the results as columns in a "per-well" table in the database. For instance,
            if you are measuring the area of the Nuclei objects and you check the aggregate
            mean box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_avg", with a column called "Mean_Nuclei_AreaShape_Area". Selecting all three
            aggregate measurements will create three per-well tables, one for each of the measurements.

            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be
            run on your Per-Image and Per-Object tables to create the desired per-well table.
            <p><i>Note:</i> this option is only available if you have extracted plate and well metadata
            from the filename using the <b>Metadata</b> or <b>LoadData</b> modules.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %(USING_METADATA_HELP_REF)s''' % globals())

        self.wants_agg_median_well = cps.Binary(
                "Calculate the per-well median values of object measurements?", False, doc='''
            Select <i>%(YES)s</i> for <b>ExportToDatabase</b> to calculate statistics over all the objects in each well
            and store the results as columns in a "per-well" table in the database. For instance,
            if you are measuring the area of the Nuclei objects and you check the aggregate
            median box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_median", with a column called "Median_Nuclei_AreaShape_Area". Selecting all
            three aggregate measurements will create three per-well tables, one for each of the measurements.

            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be run on your
            Per-Image and Per-Object tables to create the desired per-well table.
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename using
            the <b>Metadata</b> or <b>LoadData</b> modules.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %(USING_METADATA_HELP_REF)s''' % globals())

        self.wants_agg_std_dev_well = cps.Binary(
                "Calculate the per-well standard deviation values of object measurements?", False, doc='''
            Select <i>%(YES)s</i> for <b>ExportToDatabase</b> to calculate statistics over all the objects in each well
            and store the results as columns in a "per-well" table in the database. For instance,
            if you are measuring the area of the Nuclei objects and you check the aggregate
            standard deviation box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_std", with a column called "Mean_Nuclei_AreaShape_Area".  Selecting all
            three aggregate measurements will create three per-well tables, one for each of the measurements.
            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be run on your
            Per-Image and Per-Object tables to create the desired per-well table.
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename
            using the <b>Metadata</b> or <b>LoadData</b> modules.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %(USING_METADATA_HELP_REF)s''' % globals())

        self.objects_choice = cps.Choice(
                "Export measurements for all objects to the database?",
                [O_ALL, O_NONE, O_SELECT], doc="""
            This option lets you choose the objects whose measurements will be saved in the Per_Object and
            Per_Well(s) database tables.
            <ul>
            <li><i>%(O_ALL)s:</i> Export measurements from all objects.</li>
            <li><i>%(O_NONE)s:</i> Do not export data to a Per_Object table. Save only Per_Image or Per_Well
            measurements (which nonetheless include population statistics from objects).</li>
            <li><i>%(O_SELECT)s:</i> Select the objects you want to export from a list.</li>
            </ul>""" % globals())

        self.objects_list = cps.ObjectSubscriberMultiChoice(
                "Select the objects", doc="""
            <i>(Used only if Select is chosen for adding objects)</i><br>
            Choose one or more objects from this list (click using shift or command keys to select
            multiple objects). The list includes
            the objects that were created by prior modules. If you choose an
            object, its measurements will be written out to the Per_Object and/or
            Per_Well(s) tables, otherwise, the object's measurements will be skipped.""")

        self.wants_relationship_table_setting = cps.Binary(
                "Export object relationships?", True, doc="""
            <i>(Used only for pipelines which relate objects to each other)</i><br>
            Select <i>%(YES)s</i> to export object relationships to the
            RelationshipsView view. Only certain modules produce
            relationships that can be exported by this setting; see
            the <b>TrackObjects</b>, <b>RelateObjects</b>,
            <b>MeasureObjectNeighbors</b> and the <b>Identify</b> modules
            for more details.
            <p>This view has the following columns:
            <ul><li><i>%(COL_MODULE_NUMBER)s</i>: the module number of the
            module that produced the relationship. The first module in the
            pipeline is module #1, etc.</li>
            <li><i>%(COL_RELATIONSHIP)s</i>: the relationship between the two
            objects, for instance, "Parent".</li>
            <li><i>%(COL_OBJECT_NAME1)s, %(COL_OBJECT_NAME2)s</i>:
            the names of the two objects being related.</li>
            <li><i>%(COL_IMAGE_NUMBER1)s, %(COL_OBJECT_NUMBER1)s</i>:
            the image number and object number of the first object in the
            relationship</li>
            <li><i>%(COL_IMAGE_NUMBER2)s, %(COL_OBJECT_NUMBER2)s</i>:
            the image number and object number of the second object in the
            relationship</li>
            </ul>
            </p>""" % globals())

        self.max_column_size = cps.Integer(
                "Maximum # of characters in a column name", 64,
                minval=10, maxval=64, doc="""
            This setting limits the number of characters that can appear
            in the name of a field in the database. MySQL has a limit of 64
            characters per field, but also has an overall limit on the number of characters
            in all of the columns of a table. <b>ExportToDatabase</b> will
            shorten all of the column names by removing characters, at the
            same time guaranteeing that no two columns have the same name.""")

        self.separate_object_tables = cps.Choice(
                "Create one table per object, a single object table or a single object view?",
                [OT_COMBINE, OT_PER_OBJECT, OT_VIEW], doc="""
            <b>ExportToDatabase</b> can create either one table
            for each type of object exported or a single
            object table.<br><ul>
            <li><i>%(OT_PER_OBJECT)s</i> creates one
            table for each object type you export. The table name will reflect
            the name of your objects. The table will have one row for each
            of your objects. You can write SQL queries that join tables using
            the "Number_ObjectNumber" columns of parent objects (such as those
            created by <b>IdentifyPrimaryObjects</b>) with the corresponding
            "Parent_... column" of the child objects. Choose
            <i>%(OT_PER_OBJECT)s</i> if parent objects can have more than
            one child object, if you want a relational representation of
            your objects in the database,
            or if you need to split columns among different
            tables and shorten column names because of database limitations.</li>
            <li><i>%(OT_COMBINE)s</i> creates a single
            database table that records the object measurements. <b>
            ExportToDatabase</b> will prepend each column name with the
            name of the object associated with that column's measurement.
            Each row of the table will have measurements for all objects
            that have the same image and object number. Choose
            <i>%(OT_COMBINE)s</i> if parent objects have a single child,
            or if you want a simple table structure in your database. You can
            combine the measurements for all or selected objects in this way.</li>
            <li><i>%(OT_VIEW)s</i> creates a single
            database view to contain the object measurements. A <i>view</i> is a
            virtual database table which can be used to package together multiple
            per-object tables into a single structure that is accessed just like a
            regular table. Choose <i>%(OT_VIEW)s</i> if you want to combine multiple
            objects but using <i>%(OT_COMBINE)s</i> would produce a table that hits
            the database size limitations. <br>
            An important note is that only objects that are related as primary, secondary
            or tertiary objects to each other should be combined in a view. This is
            because the view expects a one-to-one relationship between the combined objects. If
            you are selecting objects for the view, the module will warn you if they are
            not related in this way.
            </li>
            </ul>""" % globals())

        self.want_image_thumbnails = cps.Binary(
                "Write image thumbnails directly to the database?", False, doc="""
            <i>(Used only if %(DB_MYSQL)s or %(DB_SQLITE)s are selected as database type)</i><br>
            Select %(YES)s if you'd like to write image thumbnails directly
            into the database. This will slow down the writing step, but will
            enable new functionality in CellProfiler Analyst such as quickly
            viewing images in the Plate Viewer tool by selecting "thumbnail"
            from the "Well display" dropdown.""" % globals())

        self.thumbnail_image_names = cps.ImageNameSubscriberMultiChoice(
                "Select the images for which you want to save thumbnails", doc="""
            <i>(Used only if %(DB_MYSQL)s or %(DB_SQLITE)s are selected as database type
            and writing thumbnails is selected)</i><br>
            Select the images that you wish to save as thumbnails to
            the database. Make multiple selections by using Ctrl-Click (Windows)
            or Command-Click (Mac); """ % globals())

        self.auto_scale_thumbnail_intensities = cps.Binary(
                "Auto-scale thumbnail pixel intensities?", True,
                doc="""
            <i>(Used only if %(DB_MYSQL)s or %(DB_SQLITE)s are selected as database type
            and writing thumbnails is selected)</i><br>
            Select <i>%(YES)s</i> if you'd like to automatically rescale
            the thumbnail pixel intensities to the range 0-1, where 0 is
            black/unsaturated, and 1 is white/saturated.""" % globals())

        self.allow_overwrite = cps.Choice(
                "Overwrite without warning?",
                [OVERWRITE_NEVER, OVERWRITE_DATA, OVERWRITE_ALL],
                doc="""
            <b>ExportToDatabase</b> creates tables and databases at the start
            of a run when writing directly to a MySQL or SQLite database. It
            writes SQL scripts and CSVs when not writing directly. It also
            can write CellProfiler Analysis property files. In some cases,
            it is appropriate to run CellProfiler and append to or overwrite
            the data in existing tables, for instance when running several
            CellProfiler instances which each process a range of the experiment's
            image sets. In other cases, such as when the measurements to be written
            have changed, the data tables must be dropped completely.
            <br>
            You can choose fromm three options to conrtol overwriting behavior:
            <ul><li><i>%(OVERWRITE_NEVER)s:</i> <b>ExportToDatabase</b> will
            ask before dropping and recreating tables unless you are running
            headless. CellProfiler will exit if running headless if the tables
            exist and this option is chosen.</li>
            <li><i>%(OVERWRITE_DATA)s:</i> <b>ExportToDatabase</b> will keep
            the existing tables if present and will overwrite the data. Choose
            <i>%(OVERWRITE_DATA)s</i> if you are breaking your experiment
            into ranges of image sets and running each range on a separate
            instance of CellProfiler.</li>
            <li><i>%(OVERWRITE_ALL)s:</i> <b>ExportToDatabase</b> will
            drop previous versions of tables at the start of a run. This option
            is appropriate if you are using the <b>CreateBatchFiles</b> module;
            your tables will be created by the run that creates the batch
            data file. The actual analysis runs that utilize the <code>Batch_data</code>
            file will use the existing tables without trying to recreate them.</li>
            </ul>
            """ % globals())

    def add_image_group(self, can_remove=True):
        group = cps.SettingsGroup()

        group.can_remove = can_remove

        group.append(
                "image_cols", cps.Choice(
                        "Select an image to include", [cps.NONE],
                        choices_fn=self.get_property_file_image_choices, doc="""
                <i>(Used only if creating a properties file and specifiying the image information)</i><br>
                Choose image name to include it in the properties file of images.
                <p>The images in the drop-down correspond to images that have been:
                <ul>
                <li>Loaded using one of the <b>Load</b> modules.</li>
                <li>Saved with the <b>SaveImages</b> module, with the corresponding file and path information stored.</li>
                </ul>
                If you do not see your desired image listed, check the settings on these modules.</p>"""))

        group.append(
                "wants_automatic_image_name", cps.Binary(
                        "Use the image name for the display?", True, doc=
                        """<i>(Used only if creating a properties file and specifiying the image information)</i><br>
                        Select <i>%(YES)s</i> to use the image name as given above for the displayed name.
                        <p>Select <i>%(NO)s</i> to name the file yourself.</p>""" % globals()))

        group.append(
                "image_name", cps.Text(
                        "Image name", "Channel%d" % (len(self.image_groups) + 1), doc=
                        """<i>(Used only if creating a properties file, specifiying the image information and naming the image)</i><br>
                        Enter a name for the specified image"""))

        default_color = (COLOR_ORDER[len(self.image_groups)]
                         if len(self.image_groups) < len(COLOR_ORDER)
                         else COLOR_ORDER[0])

        group.append(
                "image_channel_colors", cps.Choice(
                        "Channel color", COLOR_ORDER, default_color, doc="""
                <i>(Used only if creating a properties file and specifiying the image information)</i><br>
                Enter a color to display this channel."""))

        group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.image_groups, group))

        group.append("divider", cps.Divider(line=False))

        self.image_groups.append(group)

    def add_group_field_group(self, can_remove=True):
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        group.append(
                "group_name", cps.Text(
                        "Enter the name of the group", '', doc="""
            <i>(Used only if creating a properties file and specifiying an image data group)</i><br>
            Enter a name for the group. Only alphanumeric characters and underscores are permitted."""))
        group.append(
                "group_statement", cps.Text(
                        "Enter the per-image columns which define the group, separated by commas", GROUP_COL_DEFAULT,
                        doc="""
            <i>(Used only if creating a properties file and specifiying an image data group)</i><br>
            To define a group, enter the image key columns followed by group key columns, each separated by commas.
            <p>In CellProfiler, the image key column is always given the name as <i>ImageNumber</i>; group keys
            are typically metadata columns which are always prefixed with <i>Image_Metadata_</i>. For example, if you
            wanted to be able to group your data by unique plate and well metadata tags, you could define a
            group with the following MySQL statement:<br>
            <code>group_SQL_Plate = SELECT ImageNumber, Image_Metadata_Plate, Image_Metadata_Well FROM Per_Image</code><br>
            For this example, the columns to enter in this setting would be:<br>
            <code>ImageNumber, Image_Metadata_Plate, Image_Metadata_Well</code></p>
            <p>Groups are specified as MySQL statements in the properties file, but please note that the full SELECT and
            FROM clauses will be added automatically, so there is no need to enter them here.</p>"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove this group", self.group_field_groups, group))
        group.append("divider", cps.Divider(line=True))

        self.group_field_groups.append(group)

    def add_filter_field_group(self, can_remove=True):
        group = cps.SettingsGroup()

        group.can_remove = can_remove

        group.append(
                "filter_name", cps.Text(
                        "Enter the name of the filter", '', doc="""
            <i>(Used only if creating a properties file and specifiying an image data filter)</i><br>
            Enter a name for the filter. Only alphanumeric characters and underscores are permitted."""))

        group.append(
                "filter_statement", cps.Text(
                        "Enter the MySQL WHERE clause to define a filter", '', doc="""
            <i>(Used only if creating a properties file and specifiying an image data filter)</i><br>
            To define a filter, enter a MySQL <i>WHERE</i> clause that returns image-keys for images you want to include.
            For example, here is a filter that returns only images from plate 1:<br>
            <code>Image_Metadata_Plate = '1'</code><br>
            Here is a filter returns only images from with a gene column that starts with CDK:
            <code>Image_Metadata_Gene REGEXP 'CDK.*'</code><br>
            <p>Filters are specified as MySQL statements in the properties file, but please note that the full SELECT and
            FROM clauses (as well as the WHERE keyword) will be added automatically, so there is no need to enter them here.</p>"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove this filter", self.filter_field_groups, group))
        group.append("divider", cps.Divider(line=True))

        self.filter_field_groups.append(group)

    def add_workspace_measurement_group(self, can_remove=True):
        group = cps.SettingsGroup()
        self.workspace_measurement_groups.append(group)

        group.can_remove = can_remove

        group.append("divider", cps.Divider(line=False))

        group.append(
                "measurement_display", cps.Choice(
                        "Select the measurement display tool",
                        W_DISPLAY_ALL, doc="""
            <i>(Used only if creating a workspace file)</i><br>
            Select what display tool in CPA you want to use to open the
            measurements.
            <ul>
            <li>%(W_SCATTERPLOT)s</li>
            <li>%(W_HISTOGRAM)s</li>
            <li>%(W_DENSITYPLOT)s</li>
            <li>%(W_PLATEVIEWER)s</li>
            <li>%(W_BOXPLOT)s</li>
            </ul>""" % globals()))

        def measurement_type_help():
            return """
                <i>(Used only if creating a workspace file)</i><br>
                You can plot two types of measurements:
                <ul>
                <li><i>Image:</i> For a per-image measurement, one numerical value is
                recorded for each image analyzed.
                Per-image measurements are produced by
                many modules. Many have <b>MeasureImage</b> in the name but others do not
                (e.g., the number of objects in each image is a per-image
                measurement made by <b>IdentifyObject</b>
                modules).</li>
                <li><i>Object:</i> For a per-object measurement, each identified
                object is measured, so there may be none or many
                numerical values recorded for each image analyzed. These are usually produced by
                modules with <b>MeasureObject</b> in the name.</li>
                </ul>""" % globals()

        def object_name_help():
            return """<i>(Used only if creating a workspace file)</i><br>
                Select the object that you want to measure from the list.
                This should be an object created by a previous module such as
                <b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b>, or
                <b>IdentifyTertiaryObjects</b>."""

        def measurement_name_help():
            return """<i>(Used only if creating a workspace file)</i><br>
            Select the measurement to be plotted on the desired axis."""

        def index_name_help():
            return """<i>(Used only if creating a workspace file and an index is plotted)</i><br>
            Select the index to be plot on the selected axis. Two options are available:
            <ul>
            <li><i>%(C_IMAGE_NUMBER)s:</i> In CellProfiler, the unique identifier for each image
            is always given this name. Selecting this option allows you to plot a single measurement
            for each image indexed by the order it was processed.</li>
            <li><i>%(GROUP_INDEX)s:</i> This identifier is used in cases where grouping is applied.
            Each image in a group is given an index indicating the order it was processed. Selecting
            this option allows you to plot a set of measurements grouped by a common index.
            %(USING_METADATA_GROUPING_HELP_REF)s
            </li>
            </ul>""" % globals()

        group.append(
                "x_measurement_type", cps.Choice(
                        "Type of measurement to plot on the X-axis",
                        W_TYPE_ALL, doc=measurement_type_help()))

        group.append(
                "x_object_name", cps.ObjectNameSubscriber(
                        "Enter the object name", cps.NONE,
                        doc=object_name_help()))

        def object_fn_x():
            if group.x_measurement_type.value in (cpmeas.IMAGE, cpmeas.EXPERIMENT):
                return group.x_measurement_type.value
            elif group.x_measurement_type.value == cpmeas.OBJECT:
                return group.x_object_name.value
            else:
                raise NotImplementedError("Measurement type %s is not supported" %
                                          group.x_measurement_type.value)

        group.append(
                "x_measurement_name", cps.Measurement(
                        "Select the X-axis measurement", object_fn_x,
                        doc=measurement_name_help()))

        group.append(
                "x_index_name", cps.Choice(
                        "Select the X-axis index", W_INDEX_ALL,
                        doc=index_name_help()))

        group.append(
                "y_measurement_type", cps.Choice(
                        "Type of measurement to plot on the Y-axis",
                        W_TYPE_ALL, doc=measurement_type_help()))

        group.append(
                "y_object_name", cps.ObjectNameSubscriber(
                        "Enter the object name", cps.NONE,
                        doc=object_name_help()))

        def object_fn_y():
            if group.y_measurement_type.value == cpmeas.IMAGE:
                return cpmeas.IMAGE
            elif group.y_measurement_type.value == cpmeas.OBJECT:
                return group.y_object_name.value
            else:
                raise NotImplementedError("Measurement type %s is not supported" %
                                          group.y_measurement_type.value)

        group.append(
                "y_measurement_name", cps.Measurement(
                        "Select the Y-axis measurement", object_fn_y,
                        doc=measurement_name_help()))

        group.append(
                "y_index_name", cps.Choice(
                        "Select the Y-axis index", W_INDEX_ALL,
                        doc=index_name_help()))

        if can_remove:
            group.append("remove_button", cps.RemoveSettingButton(
                    "", "Remove this measurement", self.workspace_measurement_groups, group))

    def get_metadata_choices(self, pipeline):
        columns = pipeline.get_measurement_columns()
        choices = ["None"]
        for column in columns:
            object_name, feature, coltype = column[:3]
            choice = feature[(len(cpmeas.C_METADATA) + 1):]
            if (object_name == cpmeas.IMAGE and
                    feature.startswith(cpmeas.C_METADATA)):
                choices.append(choice)
        return choices

    def get_property_file_image_choices(self, pipeline):
        columns = pipeline.get_measurement_columns()
        image_names = []
        for column in columns:
            object_name, feature, coltype = column[:3]
            choice = feature[(len(C_FILE_NAME) + 1):]
            if object_name == cpmeas.IMAGE and (feature.startswith(C_FILE_NAME)):
                image_names.append(choice)
        return image_names

    def prepare_settings(self, setting_values):
        # These check the groupings of settings avilable in properties and workspace file creation
        for count, sequence, fn in \
                ((int(setting_values[SETTING_IMAGE_GROUP_COUNT]), self.image_groups, self.add_image_group),
                 (int(setting_values[SETTING_GROUP_FIELD_GROUP_COUNT]), self.group_field_groups,
                  self.add_group_field_group),
                 (int(setting_values[SETTING_FILTER_FIELD_GROUP_COUNT]), self.filter_field_groups,
                  self.add_filter_field_group),
                 (int(setting_values[SETTING_WORKSPACE_GROUP_COUNT]), self.workspace_measurement_groups,
                  self.add_workspace_measurement_group)):
            del sequence[count:]
            while len(sequence) < count:
                fn()

    def visible_settings(self):
        needs_default_output_directory = \
            (self.db_type != DB_MYSQL or
             self.save_cpa_properties.value or
             self.create_workspace_file.value)
        # # # # # # # # # # # # # # # # # #
        #
        # DB type and connection info
        #
        # # # # # # # # # # # # # # # # # #
        result = [self.db_type, self.experiment_name]
        if not HAS_MYSQL_DB:
            result += [self.mysql_not_available]
        if self.db_type == DB_MYSQL:
            result += [self.db_name]
            result += [self.db_host]
            result += [self.db_user]
            result += [self.db_passwd]
            result += [self.test_connection_button]
        elif self.db_type == DB_MYSQL_CSV:
            result += [self.sql_file_prefix]
            result += [self.db_name]
        elif self.db_type == DB_SQLITE:
            result += [self.sqlite_file]
        elif self.db_type == DB_ORACLE:
            result += [self.sql_file_prefix]
        if self.db_type != DB_MYSQL_CSV:
            result += [self.allow_overwrite]
        # # # # # # # # # # # # # # # # # #
        #
        # Table names
        #
        # # # # # # # # # # # # # # # # # #
        result += [self.want_table_prefix]
        if self.want_table_prefix.value:
            result += [self.table_prefix]
        # # # # # # # # # # # # # # # # # #
        #
        # CPA properties file
        #
        # # # # # # # # # # # # # # # # # #
        if self.save_cpa_properties.value:
            result += [self.divider_props]  # Put divider here to make things easier to read
        result += [self.save_cpa_properties]
        if self.save_cpa_properties.value:
            if (self.objects_choice != O_NONE and
                    (self.separate_object_tables == OT_COMBINE or self.separate_object_tables == OT_VIEW)):
                result += [self.location_object]
            result += [self.wants_properties_image_url_prepend]
            if self.wants_properties_image_url_prepend:
                result += [self.properties_image_url_prepend]
            result += [
                self.properties_plate_type, self.properties_plate_metadata,
                self.properties_well_metadata,
                self.properties_export_all_image_defaults]
            if not self.properties_export_all_image_defaults:
                for group in self.image_groups:
                    if group.can_remove:
                        result += [group.divider]
                    result += [group.image_cols, group.wants_automatic_image_name]
                    if not group.wants_automatic_image_name:
                        result += [group.image_name]
                    result += [group.image_channel_colors]
                    if group.can_remove:
                        result += [group.remover]
                result += [self.add_image_button]
            result += [self.properties_wants_groups]
            if self.properties_wants_groups:
                for group in self.group_field_groups:
                    if group.can_remove:
                        result += [group.divider]
                    result += [group.group_name, group.group_statement]
                    if group.can_remove:
                        result += [group.remover]
                result += [self.add_group_field_button]
            result += [self.properties_wants_filters]
            if self.properties_wants_filters:
                result += [self.create_filters_for_plates]
                for group in self.filter_field_groups:
                    result += [group.filter_name, group.filter_statement]
                    if group.can_remove:
                        result += [group.remover]
                    result += [group.divider]
                result += [self.add_filter_field_button]

            result += [self.properties_classification_type]
            result += [self.properties_class_table_name]

        if self.save_cpa_properties.value or self.create_workspace_file.value:  # Put divider here to make things easier to read
            result += [self.divider_props_wkspace]

        result += [self.create_workspace_file]
        if self.create_workspace_file:
            for workspace_group in self.workspace_measurement_groups:
                result += self.workspace_visible_settings(workspace_group)
                if workspace_group.can_remove:
                    result += [workspace_group.remove_button]
            result += [self.add_workspace_measurement_button]

        if self.create_workspace_file.value:  # Put divider here to make things easier to read
            result += [self.divider_wkspace]

        if needs_default_output_directory:
            result += [self.directory]

        # # # # # # # # # # # # # # # # # #
        #
        # Aggregations
        #
        # # # # # # # # # # # # # # # # # #
        result += [self.wants_agg_mean, self.wants_agg_median,
                   self.wants_agg_std_dev]
        if self.db_type != DB_SQLITE:
            # We don't write per-well tables to SQLite yet.
            result += [self.wants_agg_mean_well, self.wants_agg_median_well,
                       self.wants_agg_std_dev_well]
        # # # # # # # # # # # # # # # # # #
        #
        # Table choices (1 / separate object tables, etc)
        #
        # # # # # # # # # # # # # # # # # #
        result += [self.objects_choice]
        if self.objects_choice == O_SELECT:
            result += [self.objects_list]
        result += [self.wants_relationship_table_setting]
        if self.objects_choice != O_NONE:
            result += [self.separate_object_tables]

        # # # # # # # # # # # # # # # # # #
        #
        # Misc (column size + image thumbnails)
        #
        # # # # # # # # # # # # # # # # # #

        result += [self.max_column_size]
        if self.db_type in (DB_MYSQL, DB_MYSQL_CSV, DB_SQLITE):
            result += [self.want_image_thumbnails]
            if self.want_image_thumbnails:
                result += [self.thumbnail_image_names,
                           self.auto_scale_thumbnail_intensities]
        return result

    def workspace_visible_settings(self, workspace_group):
        result = []
        if workspace_group.can_remove:
            result += [workspace_group.divider]
        result += [workspace_group.measurement_display]
        result += [workspace_group.x_measurement_type]
        if workspace_group.x_measurement_type == W_INDEX:
            result += [workspace_group.x_index_name]
        elif workspace_group.x_measurement_type == cpmeas.OBJECT:
            result += [workspace_group.x_object_name, workspace_group.x_measurement_name]
        else:
            result += [workspace_group.x_measurement_name]
        if workspace_group.measurement_display.value in (W_SCATTERPLOT, W_DENSITYPLOT):
            result += [workspace_group.y_measurement_type]
            if workspace_group.y_measurement_type == W_INDEX:
                result += [workspace_group.y_index_name]
            elif workspace_group.y_measurement_type == cpmeas.OBJECT:
                result += [workspace_group.y_object_name, workspace_group.y_measurement_name]
            else:
                result += [workspace_group.y_measurement_name]
        return result

    def settings(self):
        result = [self.db_type, self.db_name, self.want_table_prefix,
                  self.table_prefix, self.sql_file_prefix,
                  self.directory,
                  self.save_cpa_properties,
                  self.db_host, self.db_user, self.db_passwd, self.sqlite_file,
                  self.wants_agg_mean, self.wants_agg_median,
                  self.wants_agg_std_dev, self.wants_agg_mean_well,
                  self.wants_agg_median_well, self.wants_agg_std_dev_well,
                  self.objects_choice, self.objects_list, self.max_column_size,
                  self.separate_object_tables, self.properties_image_url_prepend,
                  self.want_image_thumbnails, self.thumbnail_image_names,
                  self.auto_scale_thumbnail_intensities, self.properties_plate_type,
                  self.properties_plate_metadata, self.properties_well_metadata,
                  self.properties_export_all_image_defaults,
                  self.image_group_count, self.group_field_count, self.filter_field_count,
                  self.workspace_measurement_count, self.experiment_name,
                  self.location_object, self.properties_class_table_name,
                  self.wants_relationship_table_setting, self.allow_overwrite,
                  self.wants_properties_image_url_prepend,
                  self.properties_classification_type]

        # Properties: Image groups
        for group in self.image_groups:
            result += [group.image_cols, group.wants_automatic_image_name, group.image_name,
                       group.image_channel_colors]
        result += [self.properties_wants_groups]

        # Properties: Grouping fields
        for group in self.group_field_groups:
            result += [group.group_name, group.group_statement]

        # Properties: Filter fields
        result += [self.properties_wants_filters, self.create_filters_for_plates]
        for group in self.filter_field_groups:
            result += [group.filter_name, group.filter_statement]

        # Workspace settings
        result += [self.create_workspace_file]
        for group in self.workspace_measurement_groups:
            result += [group.measurement_display,
                       group.x_measurement_type, group.x_object_name, group.x_measurement_name, group.x_index_name,
                       group.y_measurement_type, group.y_object_name, group.y_measurement_name, group.y_index_name]

        return result

    def help_settings(self):
        return [self.db_type, self.experiment_name,
                self.db_name, self.db_host, self.db_user, self.db_passwd,
                self.sql_file_prefix, self.sqlite_file,
                self.allow_overwrite,
                self.want_table_prefix, self.table_prefix,
                self.save_cpa_properties, self.location_object,
                self.wants_properties_image_url_prepend,
                self.properties_image_url_prepend,
                self.properties_plate_type, self.properties_plate_metadata, self.properties_well_metadata,
                self.properties_export_all_image_defaults,
                self.image_groups[0].image_cols, self.image_groups[0].wants_automatic_image_name,
                self.image_groups[0].image_name,
                self.image_groups[0].image_channel_colors,
                self.properties_wants_groups,
                self.group_field_groups[0].group_name, self.group_field_groups[0].group_statement,
                self.properties_wants_filters, self.create_filters_for_plates,
                self.properties_class_table_name,
                self.directory,
                self.create_workspace_file,
                self.workspace_measurement_groups[0].measurement_display,
                self.workspace_measurement_groups[0].x_measurement_type,
                self.workspace_measurement_groups[0].x_object_name,
                self.workspace_measurement_groups[0].x_measurement_name,
                self.workspace_measurement_groups[0].y_measurement_type,
                self.workspace_measurement_groups[0].y_object_name,
                self.workspace_measurement_groups[0].y_measurement_name,
                self.wants_agg_mean, self.wants_agg_median, self.wants_agg_std_dev,
                self.wants_agg_mean_well, self.wants_agg_median_well, self.wants_agg_std_dev_well,
                self.objects_choice, self.objects_list,
                self.separate_object_tables,
                self.max_column_size,
                self.want_image_thumbnails, self.thumbnail_image_names, self.auto_scale_thumbnail_intensities]

    def validate_module(self, pipeline):
        if self.want_table_prefix.value:
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$", self.table_prefix.value):
                raise cps.ValidationError("Invalid table prefix", self.table_prefix)

        if self.db_type == DB_MYSQL:
            if not re.match("^[A-Za-z0-9_]+$", self.db_name.value):
                raise cps.ValidationError("The database name has invalid characters", self.db_name)
        elif self.db_type == DB_SQLITE:
            if not re.match("^[A-Za-z0-9_].*$", self.sqlite_file.value):
                raise cps.ValidationError("The sqlite file name has invalid characters", self.sqlite_file)

        if self.db_type == DB_MYSQL:
            if not re.match("^[A-Za-z0-9_].*$", self.db_host.value):
                raise cps.ValidationError("The database host name has invalid characters", self.db_host)
            if not re.match("^[A-Za-z0-9_]+$", self.db_user.value):
                raise cps.ValidationError("The database user name has invalid characters", self.db_user)
        else:
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$", self.sql_file_prefix.value):
                raise cps.ValidationError("Invalid SQL file prefix", self.sql_file_prefix)

        if self.objects_choice == O_SELECT:
            self.objects_list.load_choices(pipeline)
            if len(self.objects_list.choices) == 0:
                raise cps.ValidationError("Please choose at least one object", self.objects_choice)

        if self.save_cpa_properties:
            if self.properties_plate_metadata == NONE_CHOICE and (
                        self.properties_wants_filters.value and self.create_filters_for_plates.value):
                raise cps.ValidationError("You must specify the plate metadata", self.create_filters_for_plates)

        if self.want_image_thumbnails:
            if not self.thumbnail_image_names.get_selections():
                raise cps.ValidationError("Please choose at least one image", self.thumbnail_image_names)

        if self.want_table_prefix:
            max_char = 64
            table_name_lengths = [len(self.table_prefix.value + "Per_Image")]
            table_name_lengths += [len(
                    self.table_prefix.value + "Per_Object")] if self.objects_choice != O_NONE and self.separate_object_tables.value in (
                OT_COMBINE, OT_VIEW) else []
            table_name_lengths += [len(self.table_prefix.value + "Per_" + x) for x in self.objects_list.value.split(
                    ',')] if self.objects_choice != O_NONE and self.separate_object_tables == OT_PER_OBJECT else []
            if np.any(np.array(table_name_lengths) > max_char):
                msg = "A table name exceeds the %d character allowed by MySQL.\n" % max_char
                msg += "Please shorten the prefix if using a single object table,\n"
                msg += "and/or the object name if using separate tables."
                raise cps.ValidationError(msg, self.table_prefix)

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToDatabase does not produce output in Test Mode",
                                      self.db_type)

        # Warn user if using SQLLite and CreateBatchFiles
        if self.db_type == DB_SQLITE and pipeline.has_create_batch_module():
            raise cps.ValidationError(
                    "Only one process can access a SQLite database at a time.\n"
                    "Database operations will fail if you run more than one copy\n"
                    "of CellProfiler simultaneously. You can run multiple copies\n"
                    "of CellProfiler if you choose to output a MySQL database.\n"
                    "ExportToDatabase will work in multiprocessing mode using a\n"
                    "SQLite database.", self.db_type)

        '''Warn user that they will have to merge tables to use CPA'''
        if self.objects_choice != O_NONE and self.separate_object_tables == OT_PER_OBJECT:
            raise cps.ValidationError(
                    ("You will have to merge the separate object tables in order\n"
                     "to use CellProfiler Analyst fully, or you will be restricted\n"
                     "to only one object's data at a time in CPA. Choose\n"
                     "%s to write a single object table.") % ("'%s' or '%s'" % (OT_COMBINE, OT_VIEW)),
                    self.separate_object_tables)

        '''Warn user re: bad characters in object used for center, filter/group names and class_table name'''
        if self.save_cpa_properties:
            warning_string = "CellProfiler Analyst will not recogize this %s because it contains invalid characters. Allowable characters are letters, numbers and underscores."
            if not re.match("^[\w]*$", self.location_object.value):
                raise cps.ValidationError(warning_string % "object", self.location_object)

            if self.properties_wants_groups:
                for group in self.group_field_groups:
                    if not re.match("^[\w]*$", group.group_name.value) or group.group_name.value == '':
                        raise cps.ValidationError(warning_string % "group name", group.group_name)

            if self.properties_wants_filters:
                for group in self.filter_field_groups:
                    if not re.match("^[\w]*$", group.filter_name.value) or group.filter_name.value == '':
                        raise cps.ValidationError(warning_string % "filter name", group.filter_name)
                    if not re.match("^[\w\s\"\'=]*$",
                                    group.filter_statement.value) or group.filter_statement.value == '':
                        raise cps.ValidationError(warning_string % "filter statement", group.filter_statement)

            if self.properties_class_table_name:
                if not re.match("^[\w]*$", self.properties_class_table_name.value):
                    raise cps.ValidationError(warning_string % "class table name", self.properties_class_table_name)

        '''Warn user re: objects that are not 1:1 (i.e., primary/secondary/tertiary) if creating a view'''
        if self.objects_choice != O_NONE and self.separate_object_tables in (OT_VIEW, OT_COMBINE):
            if self.objects_choice == O_SELECT:
                selected_objs = self.objects_list.value.rsplit(',')
            elif self.objects_choice == O_ALL:
                selected_objs = pipeline.get_provider_dictionary(cps.OBJECT_GROUP).keys()

            if len(selected_objs) > 1:
                # Check whether each selected object comes from an Identify module. If it does, look for its parent.
                d = dict.fromkeys(selected_objs, None)
                for obj in selected_objs:
                    for module in pipeline.modules():
                        if module.is_object_identification_module():  # and module.get_measurements(pipeline,obj,C_PARENT):
                            parent = module.get_measurements(pipeline, obj, C_PARENT)
                            if len(parent) > 0:
                                d[obj] = parent[0]
                # For objects with no parents (primary), use the object itself
                d = dict(zip(d.keys(), [key if value is None else value for (key, value) in d.items()]))

                # Only those objects which have parents in common should be written together
                if len(set(d.values())) > 1:
                    # Pick out the parent with the lowest representation in the selected object list
                    mismatched_parent = \
                        sorted(zip([d.values().count(item) for item in set(d.values())], set(d.values())))[0][1]
                    # Find the objects that this parent goes with
                    mismatched_objs = [key for (key, value) in d.items() if value == mismatched_parent]
                    msg = "%s is not in a 1:1 relationship with the other objects, which may cause downstream problems.\n " % ",".join(
                            mismatched_objs)
                    msg += "You may want to choose another object container"
                    msg += "." if self.objects_choice == O_ALL else " or de-select the object(s)."
                    raise cps.ValidationError(msg, self.separate_object_tables)

    def test_connection(self):
        '''Check to make sure the MySQL server is remotely accessible'''
        import wx

        error = None
        try:
            connection = connect_mysql(self.db_host.value,
                                       self.db_user.value,
                                       self.db_passwd.value,
                                       self.db_name.value)
        except MySQLdb.Error, error:
            if error.args[0] == 1045:
                msg = "Incorrect username or password"
            elif error.args[0] == 1049:
                msg = "The database does not exist."
            else:
                msg = "A connection error to the database host was returned: %s" % error.args[1]

        if not error:
            wx.MessageBox("Connection to database host successful.")
        else:
            wx.MessageBox("%s. Please check your settings." % msg)

    def make_full_filename(self, file_name,
                           workspace=None, image_set_index=None):
        """Convert a file name into an absolute path

        We do a few things here:
        * apply metadata from an image set to the file name if an
          image set is specified
        * change the relative path into an absolute one using the "." and "&"
          convention
        * Create any directories along the path
        """
        if image_set_index is not None and workspace is not None:
            file_name = workspace.measurements.apply_metadata(file_name,
                                                              image_set_index)
        measurements = None if workspace is None else workspace.measurements
        path_name = self.directory.get_absolute_path(measurements,
                                                     image_set_index)
        file_name = os.path.join(path_name, file_name)
        path, file = os.path.split(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, file)

    def prepare_run(self, workspace, as_data_tool=False):
        '''Prepare to run the pipeline
        Establish a connection to the database.'''

        if not as_data_tool:
            self.get_dictionary().clear()
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list

        if pipeline.test_mode:
            return True

        needs_close = False
        try:
            if self.db_type == DB_MYSQL:
                self.connection, self.cursor = connect_mysql(self.db_host.value,
                                                             self.db_user.value,
                                                             self.db_passwd.value,
                                                             self.db_name.value)
                needs_close = True
                if self.wants_well_tables:
                    per_well = self.write_mysql_table_per_well(pipeline, image_set_list)
            elif self.db_type == DB_SQLITE:
                db_file = self.make_full_filename(self.sqlite_file.value)
                self.connection, self.cursor = connect_sqlite(db_file)
                needs_close = True
            #
            # This caches the list of measurement columns for the run,
            # fixing the column order, etc.
            #
            self.get_pipeline_measurement_columns(pipeline, image_set_list)

            if pipeline.in_batch_mode() or not cpprefs.get_allow_schema_write():
                return True
            if self.db_type == DB_ORACLE:
                raise NotImplementedError("Writing to an Oracle database is not yet supported")
            if self.db_type in (DB_MYSQL, DB_SQLITE):
                tables = [self.get_table_name(cpmeas.IMAGE)]
                if self.objects_choice != O_NONE:
                    if self.separate_object_tables == OT_COMBINE:
                        tables.append(self.get_table_name(cpmeas.OBJECT))
                    else:
                        for object_name in self.get_object_names(pipeline, image_set_list):
                            tables.append(self.get_table_name(object_name))
                tables_that_exist = []
                for table in tables:
                    try:
                        r = execute(self.cursor,
                                    'SELECT * FROM %s LIMIT 1' % table)
                        tables_that_exist.append(table)
                    except:
                        pass
                if len(tables_that_exist) > 0:
                    if len(tables_that_exist) == 1:
                        table_msg = "%s table" % tables_that_exist[0]
                    else:
                        table_msg = "%s and %s tables" % (
                            ", ".join(tables_that_exist[:-1]),
                            tables_that_exist[-1])
                    if cpprefs.get_headless():
                        if self.allow_overwrite == OVERWRITE_NEVER:
                            logger.error("%s already in database and overwrite not allowed. Exiting" % table_msg)
                            return False
                        elif self.allow_overwrite == OVERWRITE_DATA:
                            logger.warning("%s already in database, not creating" % table_msg)
                            return True
                    elif self.allow_overwrite in (OVERWRITE_NEVER, OVERWRITE_DATA):
                        import wx
                        message = (
                                      "Do you want ExportToDatabase to drop the %s?\n\n"
                                      'Choose "Yes" to drop and recreate the tables, '
                                      'discarding all existing data.\n'
                                      'Choose "No" to keep the existing tables and '
                                      'overwrite data as necessary.\n'
                                      'Choose "Cancel" to stop and leave the tables intact.') % table_msg

                        with wx.MessageDialog(
                                workspace.frame, message,
                                style=wx.YES | wx.NO | wx.CANCEL | wx.ICON_QUESTION) as dlg:
                            result = dlg.ShowModal()
                            if result == wx.ID_CANCEL:
                                return False
                            elif result != wx.ID_YES:
                                return True

                mappings = self.get_column_name_mappings(pipeline, image_set_list)
                column_defs = self.get_pipeline_measurement_columns(pipeline,
                                                                    image_set_list)
                if self.objects_choice != O_ALL:
                    onames = [cpmeas.EXPERIMENT, cpmeas.IMAGE, cpmeas.NEIGHBORS]
                    if self.objects_choice == O_SELECT:
                        onames += self.objects_list.selections
                    column_defs = [column for column in column_defs
                                   if column[0] in onames]
                self.create_database_tables(self.cursor, workspace)
            return True
        finally:
            if needs_close:
                self.connection.commit()
                self.cursor.close()
                self.connection.close()
                self.connection = None
                self.cursor = None

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Alter the output directory path for the remote batch host'''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def get_measurement_columns(self, pipeline):
        if self.want_image_thumbnails:
            cols = []
            for name in self.thumbnail_image_names.get_selections():
                cols += [(cpmeas.IMAGE, C_THUMBNAIL + "_" + name,
                          cpmeas.COLTYPE_LONGBLOB)]
            return cols
        return []

    def run_as_data_tool(self, workspace):
        '''Run the module as a data tool

        ExportToDatabase has two modes - writing CSVs and writing directly.
        We write CSVs in post_run. We write directly in run.
        '''
        #
        # The measurements may have been created by an old copy of CP. We
        # have to hack our measurement column cache to circumvent this.
        #
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        d = self.get_dictionary()
        columns = m.get_measurement_columns()
        for i, (object_name, feature_name, coltype) in enumerate(columns):
            if (object_name == cpmeas.IMAGE and
                    feature_name.startswith(C_THUMBNAIL)):
                columns[i] = (object_name, feature_name, cpmeas.COLTYPE_LONGBLOB)
        columns = self.filter_measurement_columns(columns)
        d[D_MEASUREMENT_COLUMNS] = columns

        if not self.prepare_run(workspace, as_data_tool=True):
            return
        self.prepare_group(workspace, None, None)
        if self.db_type != DB_MYSQL_CSV:
            workspace.measurements.is_first_image = True

            for i in range(workspace.measurements.image_set_count):
                if i > 0:
                    workspace.measurements.next_image_set()
                self.run(workspace)
        else:
            workspace.measurements.image_set_number = \
                workspace.measurements.image_set_count
        self.post_run(workspace)

    def run(self, workspace):
        if self.want_image_thumbnails:
            import PIL.Image as Image
            from StringIO import StringIO
            measurements = workspace.measurements
            image_set = workspace.image_set
            for name in self.thumbnail_image_names.get_selections():
                # For each desired channel, convert the pixel data into a PIL
                # image and then save it as a PNG into a StringIO buffer.
                # Finally read the raw data out of the buffer and add it as
                # as measurement to be written as a blob.
                pixels = image_set.get_image(name).pixel_data

                if issubclass(pixels.dtype.type, np.floating) or pixels.dtype == np.bool:
                    factor = 255
                    if self.auto_scale_thumbnail_intensities:
                        pixels = (pixels - pixels.min()) / pixels.max()
                else:
                    raise Exception('ExportToDatabase cannot write image thumbnails from images of type "%s".' % (
                        str(pixels.dtype)))
                if pixels.ndim == 2:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'L')
                elif pixels.ndim == 3:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'RGB')
                else:
                    raise Exception(
                            'ExportToDatabase only supports saving thumbnails of grayscale or 3-channel images. "%s" was neither.' % (
                                name))

                # resize the image so the major axis is 200px long
                if im.size[0] == max(im.size):
                    w, h = (200, 200 * min(im.size) / max(im.size))
                else:
                    h, w = (200, 200 * min(im.size) / max(im.size))
                im = im.resize((w, h))

                fd = StringIO()
                im.save(fd, 'PNG')
                blob = fd.getvalue()
                fd.close()
                measurements.add_image_measurement(
                        C_THUMBNAIL + "_" + name, blob.encode('base64'))
        if workspace.pipeline.test_mode:
            return
        if self.db_type == DB_MYSQL and not workspace.pipeline.test_mode:
            try:
                self.connection, self.cursor = connect_mysql(
                        self.db_host.value,
                        self.db_user.value,
                        self.db_passwd.value,
                        self.db_name.value)
                self.write_data_to_db(workspace)
            finally:
                self.connection.commit()
                self.connection.close()
                self.connection = None
                self.cursor = None
        elif self.db_type == DB_SQLITE and not workspace.pipeline.test_mode:
            # For distributed, use the interaction handler to run the
            # database commands on the server
            #
            self.connection = self.cursor = SQLiteCommands()
            try:
                self.write_data_to_db(workspace)
                workspace.interaction_request(
                        self, self.INTERACTION_EXECUTE, self.connection.get_state())
            except workspace.NoInteractionException:
                # Assume that the interaction can be handled directly,
                # for instance, in headless mode with no handler
                #
                self.handle_interaction(self.INTERACTION_EXECUTE,
                                        self.connection.get_state())
            finally:
                self.connection = None
                self.cursor = None

    INTERACTION_EXECUTE = "Execute"
    INTERACTION_GET_RELATIONSHIP_TYPES = "GetRelationshipTypes"
    INTERACTION_ADD_RELATIONSHIP_TYPE = "AddRelationshipType"

    def handle_interaction(self, command, *args, **kwargs):
        '''Handle sqllite interactions from workers'''

        if command == self.INTERACTION_EXECUTE:
            return self.handle_interaction_execute(*args, **kwargs)
        elif command == self.INTERACTION_GET_RELATIONSHIP_TYPES:
            return self.handle_interaction_get_relationship_types(*args, **kwargs)
        elif command == self.INTERACTION_ADD_RELATIONSHIP_TYPE:
            return self.handle_interaction_add_relationship_type(*args, **kwargs)
        else:
            raise ValueError("No %s interaction" % command)

    def handle_interaction_execute(self, state):
        commands = SQLiteCommands()
        commands.set_state(state)
        db_file = self.make_full_filename(self.sqlite_file.value)
        connection, cursor = connect_sqlite(db_file)
        try:
            commands.execute_all(cursor)
            connection.commit()
        except:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def handle_interaction_get_relationship_types(self):
        '''Get the relationship types from the database

        returns a dictionary whose key is
        (module_number, relationship name, object_name1, object_name2) and
        whose value is the relationship type ID for that relationship.
        '''
        db_file = self.make_full_filename(self.sqlite_file.value)
        with DBContext(self) as (connection, cursor):
            return self.get_relationship_types(cursor).items()

    def grt_interaction_to_dict(self, json_struct):
        '''Handle the conversion from json mangled structure to dictionary

        json_struct - the result from handle_interaction_get_relationship_types
                      which has been dumbed-down for json and which json
                      has likely turned tuples to lists
        '''
        return dict([(tuple(k), v) for k, v in json_struct])

    def get_relationship_types(self, cursor):
        '''Get the relationship types from the database

        returns a dictionary whose key is
        (module_number, relationship name, object_name1, object_name2) and
        whose value is the relationship type ID for that relationship.
        '''
        relationship_type_table = self.get_table_name(T_RELATIONSHIP_TYPES)
        statement = "SELECT %s, %s, %s, %s, %s FROM %s" % (
            COL_RELATIONSHIP_TYPE_ID, COL_RELATIONSHIP, COL_MODULE_NUMBER,
            COL_OBJECT_NAME1, COL_OBJECT_NAME2, relationship_type_table)

        return dict(
                [((int(mn), r, o1, o2), int(rt_id))
                 for rt_id, r, mn, o1, o2 in
                 execute(cursor, statement)])

    def handle_interaction_add_relationship_type(
            self, module_num, relationship, object_name1, object_name2):
        '''Add a relationship type to the database

        module_num, relationship, object_name1, object_name2: the key
              to the relationship in the relationship type table

        returns the relationship type ID
        '''
        with DBContext(self) as (connection, cursor):
            return self.add_relationship_type(
                    module_num, relationship, object_name1, object_name2, cursor)

    def add_relationship_type(self, module_num, relationship, object_name1,
                              object_name2, cursor):
        '''Add a relationship type to the database

        module_num, relationship, object_name1, object_name2: the key
              to the relationship in the relationship type table

        returns the relationship type ID
        '''
        logger.info("Adding missing relationship type:")
        logger.info("        module #: %d" % module_num)
        logger.info("    relationship: %s" % relationship)
        logger.info("        object 1: %s" % object_name1)
        logger.info("        object 2: %s" % object_name2)
        #
        # If the code reaches here, it's because:
        # * some module has an absent or mis-coded get_relationship_columns
        # * the user changed the pipeline after prepare_run was called.
        #
        relationship_type_table = self.get_table_name(T_RELATIONSHIP_TYPES)
        #
        # An insert guarantees that a record exists
        #
        # INSERT INTO <t> (...)
        # SELECT * FROM (
        #     SELECT relationship_type_id + 1, <module #>... FROM <t>
        # ) as mytable WHERE NOT EXISTS
        # (SELECT 'x' FROM <t> WHERE MODULE_NUM=<module %>...)
        # ORDER BY relationship_type_id desc LIMIT 1
        #
        statement = \
            "INSERT INTO %s (%s, %s, %s, %s, %s) " % (
                relationship_type_table,
                COL_RELATIONSHIP_TYPE_ID, COL_MODULE_NUMBER,
                COL_RELATIONSHIP, COL_OBJECT_NAME1, COL_OBJECT_NAME2)
        statement += "SELECT * FROM "
        statement += \
            "(SELECT coalesce(max(%s), -1)+1 as %s, %d as %s, '%s' as %s, '%s' as %s, '%s' as %s FROM %s)" % \
            (COL_RELATIONSHIP_TYPE_ID, COL_RELATIONSHIP_TYPE_ID,
             module_num, COL_MODULE_NUMBER,
             relationship, COL_RELATIONSHIP,
             object_name1, COL_OBJECT_NAME1,
             object_name2, COL_OBJECT_NAME2, relationship_type_table)
        statement += " AS mytable WHERE NOT EXISTS "
        statement += "(SELECT 'x' FROM %s WHERE " % relationship_type_table
        statement += "%s = %d AND " % (COL_MODULE_NUMBER, module_num)
        statement += "%s = '%s' AND " % (COL_RELATIONSHIP, relationship)
        statement += "%s = '%s' AND " % (COL_OBJECT_NAME1, object_name1)
        statement += "%s = '%s')" % (COL_OBJECT_NAME2, object_name2)
        cursor.execute(statement)
        #
        # Then we select and find it
        #
        select_statement = \
            "SELECT min(%s) FROM %s WHERE %s = %d" % (
                COL_RELATIONSHIP_TYPE_ID, relationship_type_table,
                COL_MODULE_NUMBER, module_num)
        for col, value in ((COL_RELATIONSHIP, relationship),
                           (COL_OBJECT_NAME1, object_name1),
                           (COL_OBJECT_NAME2, object_name2)):
            select_statement += " AND %s = '%s'" % (col, value)
        cursor.execute(select_statement)
        result = cursor.fetchall()
        if len(result) == 0 or result[0][0] is None:
            raise ValueError(
                    "Failed to retrieve relationship_type_id for "
                    "module # %d, %s %s %s" %
                    (module_num, relationship, object_name1, object_name2))
        return int(result[0][0])

    def post_group(self, workspace, grouping):
        '''Write out any columns that are only available post-group'''
        if workspace.pipeline.test_mode:
            return

        if self.db_type not in (DB_MYSQL, DB_SQLITE):
            return

        try:
            if self.db_type == DB_MYSQL:
                self.connection, self.cursor = connect_mysql(
                        self.db_host.value,
                        self.db_user.value,
                        self.db_passwd.value,
                        self.db_name.value)
            elif self.db_type == DB_SQLITE:
                self.connection = self.cursor = SQLiteCommands()
            #
            # Process the image numbers in the current image's group
            #
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            group_number = m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER,
                             m.image_set_number]
            all_image_numbers = m.get_image_numbers()
            all_group_numbers = m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER,
                                  all_image_numbers]
            group_image_numbers = \
                all_image_numbers[all_group_numbers == group_number]
            for image_number in group_image_numbers:
                self.write_data_to_db(workspace,
                                      post_group=True,
                                      image_number=image_number)
            if self.db_type == DB_SQLITE:
                try:
                    workspace.interaction_request(
                            self, self.INTERACTION_EXECUTE,
                            self.connection.get_state())
                except workspace.NoInteractionException:
                    # Assume that the interaction can be handled directly,
                    # for instance, in headless mode with no handler
                    #
                    self.handle_interaction(self.INTERACTION_EXECUTE,
                                            self.connection.get_state())
        finally:
            self.connection.commit()
            self.connection.close()
            self.connection = None
            self.cursor = None

    def post_run(self, workspace):
        if self.save_cpa_properties.value:
            self.write_properties_file(workspace)
        if self.create_workspace_file.value:
            self.write_workspace_file(workspace)
        if self.db_type == DB_MYSQL_CSV:
            path = self.directory.get_absolute_path(None if workspace is None
                                                    else workspace.measurements)
            if not os.path.isdir(path):
                os.makedirs(path)
            self.write_mysql_table_defs(workspace)
            self.write_csv_data(workspace)
        else:
            self.write_post_run_measurements(workspace)

    @property
    def wants_well_tables(self):
        '''Return true if user wants any well tables'''
        if self.db_type == DB_SQLITE:
            return False
        else:
            return (self.wants_agg_mean_well or self.wants_agg_median_well or
                    self.wants_agg_std_dev_well)

    @property
    def wants_relationship_table(self):
        '''True to write relationships to the database'''
        return self.wants_relationship_table_setting.value

    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True

    def ignore_object(self, object_name, strict=False):
        """Ignore objects (other than 'Image') if this returns true

        If strict is True, then we ignore objects based on the object selection
        """
        if object_name in (cpmeas.EXPERIMENT, cpmeas.NEIGHBORS):
            return True
        if strict and self.objects_choice == O_NONE:
            return True
        if (strict and self.objects_choice == O_SELECT and
                    object_name != cpmeas.IMAGE):
            return object_name not in self.objects_list.selections
        return False

    def ignore_feature(self, object_name, feature_name, measurements=None,
                       strict=False):
        """Return true if we should ignore a feature"""
        if (self.ignore_object(object_name, strict) or
                feature_name.startswith('Description_') or
                feature_name.startswith('ModuleError_') or
                feature_name.startswith('TimeElapsed_') or
                feature_name.startswith('ExecutionTime_') or
                (self.db_type not in (DB_MYSQL, DB_SQLITE) and feature_name.startswith('Thumbnail_'))
            ):
            return True
        return False

    def get_column_name_mappings(self, pipeline, image_set_list):
        """Scan all the feature names in the measurements, creating column names"""
        columns = self.get_pipeline_measurement_columns(pipeline,
                                                        image_set_list)
        mappings = ColumnNameMapping(self.max_column_size.value)
        mappings.add(C_IMAGE_NUMBER)
        mappings.add(C_OBJECT_NUMBER)
        for column in columns:
            object_name, feature_name, coltype = column[:3]
            if self.ignore_feature(object_name, feature_name):
                continue
            mappings.add("%s_%s" % (object_name, feature_name))
            if object_name != cpmeas.IMAGE:
                for agg_name in self.agg_names:
                    mappings.add('%s_%s_%s' % (agg_name, object_name, feature_name))
        return mappings

    def get_aggregate_columns(self, pipeline, image_set_list, post_group=None):
        '''Get object aggregate columns for the PerImage table

        pipeline - the pipeline being run
        image_set_list - for cacheing column data
        post_group - true if only getting aggregates available post-group,
                     false for getting aggregates available after run,
                     None to get all

        returns a tuple:
        result[0] - object_name = name of object generating the aggregate
        result[1] - feature name
        result[2] - aggregation operation
        result[3] - column name in Image database
        '''
        columns = self.get_pipeline_measurement_columns(pipeline,
                                                        image_set_list)
        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        ob_tables = self.get_object_names(pipeline, image_set_list)
        result = []
        for ob_table in ob_tables:
            for column in columns:
                if ((post_group is not None) and
                        not self.should_write(column, post_group)):
                    continue
                obname, feature, ftype = column[:3]
                if (obname == ob_table and
                        (not self.ignore_feature(obname, feature)) and
                        (not cpmeas.agg_ignore_feature(feature))):
                    feature_name = '%s_%s' % (obname, feature)
                    # create per_image aggregate column defs
                    result += [(obname, feature, aggname,
                                '%s_%s' % (aggname, feature_name))
                               for aggname in self.agg_names]
        return result

    def get_object_names(self, pipeline, image_set_list):
        '''Get the names of the objects whose measurements are being taken'''
        column_defs = self.get_pipeline_measurement_columns(pipeline,
                                                            image_set_list)
        obnames = set([c[0] for c in column_defs])
        #
        # In alphabetical order
        #
        obnames = sorted(obnames)
        return [obname for obname in obnames
                if not self.ignore_object(obname, True) and
                obname not in (cpmeas.IMAGE, cpmeas.EXPERIMENT,
                               cpmeas.NEIGHBORS)]

    @property
    def agg_names(self):
        '''The list of selected aggregate names'''
        return [name
                for name, setting
                in ((cpmeas.AGG_MEAN, self.wants_agg_mean),
                    (cpmeas.AGG_MEDIAN, self.wants_agg_median),
                    (cpmeas.AGG_STD_DEV, self.wants_agg_std_dev))
                if setting.value]

    @property
    def agg_well_names(self):
        '''The list of selected aggregate names'''
        return [name
                for name, setting
                in (('avg', self.wants_agg_mean_well),
                    ('median', self.wants_agg_median_well),
                    ('std', self.wants_agg_std_dev_well))
                if setting.value]

    #
    # Create per_image and per_object tables in MySQL
    #
    def create_database_tables(self, cursor, workspace):
        '''Creates empty image and object tables

        Creates the MySQL database (if MySQL), drops existing tables of the
        same name and creates the tables.

        cursor - database cursor for creating the tables
        column_defs - column definitions as returned by get_measurement_columns
        mappings - mappings from measurement feature names to column names
        '''
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        # Create the database
        if self.db_type == DB_MYSQL:
            # result = execute(cursor, "SHOW DATABASES LIKE '%s'" %
            # self.db_name.value)
            # if len(result) == 0:
            execute(cursor, 'CREATE DATABASE IF NOT EXISTS %s' %
                    self.db_name.value, return_result=False)
            execute(cursor, 'USE %s' % self.db_name.value,
                    return_result=False)

        columns = self.get_pipeline_measurement_columns(pipeline,
                                                        image_set_list)

        #
        # Drop either the unified objects table or the view of it
        #
        object_table_name = self.get_table_name(cpmeas.OBJECT)
        try:
            execute(cursor, 'DROP TABLE IF EXISTS %s' %
                    self.get_table_name(cpmeas.OBJECT),
                    return_result=False)
        except:
            # MySQL is fine if the table is a view, but not SQLite
            pass
        try:
            execute(cursor, 'DROP VIEW IF EXISTS %s' %
                    self.get_table_name(cpmeas.OBJECT),
                    return_result=False)
        except:
            pass

        if self.objects_choice != O_NONE:
            # Object table/view
            if self.separate_object_tables == OT_COMBINE:
                statement = self.get_create_object_table_statement(
                        None, pipeline, image_set_list)
                execute(cursor, statement, return_result=False)
            else:
                for object_name in self.get_object_names(pipeline,
                                                         image_set_list):
                    execute(cursor, 'DROP TABLE IF EXISTS %s' %
                            self.get_table_name(object_name),
                            return_result=False)
                    statement = self.get_create_object_table_statement(
                            object_name, pipeline, image_set_list)
                    execute(cursor, statement, return_result=False)
                if self.separate_object_tables == OT_VIEW:
                    statement = self.get_create_object_view_statement(
                            self.get_object_names(pipeline, image_set_list), pipeline, image_set_list)
                    execute(cursor, statement, return_result=False)

        # Image table
        execute(cursor, 'DROP TABLE IF EXISTS %s' %
                self.get_table_name(cpmeas.IMAGE), return_result=False)
        statement = self.get_create_image_table_statement(pipeline,
                                                          image_set_list)
        execute(cursor, statement, return_result=False)

        execute(cursor, 'DROP TABLE IF EXISTS %s' %
                self.get_table_name(cpmeas.EXPERIMENT))
        for statement in self.get_experiment_table_statements(workspace):
            execute(cursor, statement, return_result=False)
        if self.wants_relationship_table:
            for statement in self.get_create_relationships_table_statements(pipeline):
                execute(cursor, statement, return_result=False)
        cursor.connection.commit()

    def get_experiment_table_statements(self, workspace):
        statements = []
        if self.db_type in (DB_MYSQL_CSV, DB_MYSQL):
            autoincrement = "AUTO_INCREMENT"
            need_text_size = True
        else:
            autoincrement = "AUTOINCREMENT"
            need_text_size = False
        create_experiment_table_statement = """
CREATE TABLE IF NOT EXISTS %s (
    experiment_id integer primary key %s,
    name text)""" % (T_EXPERIMENT, autoincrement)
        statements.append(create_experiment_table_statement)
        if need_text_size:
            create_experiment_properties = """
CREATE TABLE IF NOT EXISTS %(T_EXPERIMENT_PROPERTIES)s (
    experiment_id integer not null,
    object_name text not null,
    field text not null,
    value longtext,
    constraint %(T_EXPERIMENT_PROPERTIES)s_pk primary key
    (experiment_id, object_name(200), field(200)))""" % globals()
        else:
            create_experiment_properties = """
CREATE TABLE IF NOT EXISTS %(T_EXPERIMENT_PROPERTIES)s (
    experiment_id integer not null,
    object_name text not null,
    field text not null,
    value longtext,
    constraint %(T_EXPERIMENT_PROPERTIES)s_pk primary key (experiment_id, object_name, field))""" % globals()

        statements.append(create_experiment_properties)
        insert_into_experiment_statement = """
INSERT INTO %s (name) values ('%s')""" % (
            T_EXPERIMENT, MySQLdb.escape_string(self.experiment_name.value))
        statements.append(insert_into_experiment_statement)

        properties = self.get_property_file_text(workspace)
        for p in properties:
            for k, v in p.properties.iteritems():
                if isinstance(v, unicode):
                    v = v.encode('utf-8')
                statement = """
INSERT INTO %s (experiment_id, object_name, field, value)
SELECT MAX(experiment_id), '%s', '%s', '%s' FROM %s""" % (
                    T_EXPERIMENT_PROPERTIES, p.object_name,
                    MySQLdb.escape_string(k),
                    MySQLdb.escape_string(v), T_EXPERIMENT)
                statements.append(statement)

        experiment_columns = filter(
                lambda x: x[0] == cpmeas.EXPERIMENT,
                workspace.pipeline.get_measurement_columns())
        experiment_coldefs = [
            "%s %s" % (x[1],
                       "TEXT" if x[2].startswith(cpmeas.COLTYPE_VARCHAR)
                       else x[2]) for x in experiment_columns]
        create_per_experiment = """
CREATE TABLE %s (
%s)
""" % (self.get_table_name(cpmeas.EXPERIMENT),
       ",\n".join(experiment_coldefs))
        statements.append(create_per_experiment)
        column_names = []
        values = []
        for column in experiment_columns:
            ftr = column[1]
            column_names.append(ftr)
            if ((len(column) > 3 and
                     column[3].get(cpmeas.MCA_AVAILABLE_POST_RUN, False)) or
                    not workspace.measurements.has_feature(cpmeas.EXPERIMENT, ftr)):
                values.append("null")
                continue
            value = workspace.measurements.get_experiment_measurement(ftr)

            if column[2].startswith(cpmeas.COLTYPE_VARCHAR):
                if isinstance(value, unicode):
                    value = value.encode('utf-8')
                if self.db_type != DB_SQLITE:
                    value = MySQLdb.escape_string(value)
                else:
                    value = value.replace("'", "''")
                value = "'" + value + "'"
            else:
                # Both MySQL and SQLite support blob literals of the style:
                # X'0123456789ABCDEF'
                #
                value = "X'" + "".join(["%02X" % ord(x) for x in value]) + "'"
            values.append(value)
        experiment_insert_statement = "INSERT INTO %s (%s) VALUES (%s)" % (
            self.get_table_name(cpmeas.EXPERIMENT),
            ",".join(column_names),
            ",".join(values))
        statements.append(experiment_insert_statement)
        return statements

    def get_create_image_table_statement(self, pipeline, image_set_list):
        '''Return a SQL statement that generates the image table'''
        statement = 'CREATE TABLE ' + self.get_table_name(cpmeas.IMAGE) + ' (\n'
        statement += '%s INTEGER' % C_IMAGE_NUMBER

        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        columns = self.get_pipeline_measurement_columns(
                pipeline, image_set_list)
        for column in columns:
            obname, feature, ftype = column[:3]
            if obname == cpmeas.IMAGE and not self.ignore_feature(obname, feature):
                if ftype.startswith(cpmeas.COLTYPE_VARCHAR):
                    ftype = "TEXT"
                feature_name = '%s_%s' % (obname, feature)
                statement += ',\n%s %s' % (mappings[feature_name], ftype)
        for column in self.get_aggregate_columns(pipeline, image_set_list):
            statement += ',\n%s %s' % (mappings[column[3]],
                                       cpmeas.COLTYPE_FLOAT)
        statement += ',\nPRIMARY KEY (%s) )' % C_IMAGE_NUMBER
        return statement

    def get_create_object_table_statement(self, object_name, pipeline,
                                          image_set_list):
        '''Get the "CREATE TABLE" statement for the given object table

        object_name - None = PerObject, otherwise a specific table
        '''
        if object_name is None:
            object_table = self.get_table_name(cpmeas.OBJECT)
        else:
            object_table = self.get_table_name(object_name)
        statement = 'CREATE TABLE ' + object_table + ' (\n'
        statement += '%s INTEGER\n' % C_IMAGE_NUMBER
        if object_name is None:
            statement += ',%s INTEGER' % C_OBJECT_NUMBER
            object_pk = C_OBJECT_NUMBER
        else:
            object_pk = "_".join((object_name, M_NUMBER_OBJECT_NUMBER))
        column_defs = self.get_pipeline_measurement_columns(pipeline,
                                                            image_set_list)
        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        if object_name is None:
            ob_tables = self.get_object_names(pipeline, image_set_list)
        else:
            ob_tables = [object_name]
        for ob_table in ob_tables:
            for column_def in column_defs:
                obname, feature, ftype = column_def[:3]
                if obname == ob_table and not self.ignore_feature(obname, feature):
                    feature_name = '%s_%s' % (obname, feature)
                    statement += ',\n%s %s' % (mappings[feature_name], ftype)
        statement += ',\nPRIMARY KEY (%s, %s) )' % (C_IMAGE_NUMBER, object_pk)
        return statement

    def get_create_object_view_statement(self, object_names, pipeline,
                                         image_set_list):
        '''Get the "CREATE VIEW" statement for the given object view

        object_names is the list of objects to be included into the view
        '''
        object_table = self.get_table_name(cpmeas.OBJECT)

        # Produce a list of columns from each of the separate tables
        list_of_columns = []
        all_objects = dict(zip(object_names, [self.get_table_name(object_name) for object_name in object_names]))

        column_defs = self.get_pipeline_measurement_columns(pipeline, image_set_list)
        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        for (current_object, current_table) in all_objects.iteritems():
            list_of_columns.append([])
            for column_def in column_defs:
                obname, feature, ftype = column_def[:3]
                if obname == current_object and not self.ignore_feature(obname, feature):
                    feature_name = '%s_%s' % (obname, feature)
                    list_of_columns[-1] += [mappings[feature_name]]
        all_columns = sum(list_of_columns, [])

        selected_object = object_names[0]
        all_columns = ["%s.%s" % (all_objects[selected_object], C_IMAGE_NUMBER),
                       "%s_%s AS %s" % (selected_object, M_NUMBER_OBJECT_NUMBER, C_OBJECT_NUMBER)] + all_columns

        # Create the new view
        statement = "CREATE OR REPLACE VIEW " if self.db_type == DB_MYSQL else "CREATE VIEW "
        statement += "%s AS SELECT %s FROM %s" % (object_table, ",".join(all_columns), all_objects[selected_object])

        object_table_pairs = all_objects.items()
        object_table_pairs = [x for x in object_table_pairs if x[0] != selected_object]
        for (current_object, current_table) in object_table_pairs:
            statement = " ".join((statement, "INNER JOIN %s ON" % current_table, \
                                  " AND ".join(("%s.%s = %s.%s" % (
                                      all_objects[selected_object], C_IMAGE_NUMBER, current_table, C_IMAGE_NUMBER),
                                                "%s.%s_%s = %s.%s_%s" % (
                                                    all_objects[selected_object], selected_object,
                                                    M_NUMBER_OBJECT_NUMBER,
                                                    current_table, current_object, M_NUMBER_OBJECT_NUMBER)))))
        return statement

    def get_create_relationships_table_statements(self, pipeline):
        """Get the statements to create the relationships table

        Returns a list of statements to execute.
        """
        statements = []
        #
        # View name + drop view if appropriate
        #
        relationship_view_name = self.get_table_name(V_RELATIONSHIPS)
        statements.append(
                "DROP VIEW IF EXISTS %s" % relationship_view_name)
        #
        # Table names + drop table if appropriate
        #
        relationship_type_table_name = self.get_table_name(T_RELATIONSHIP_TYPES)
        relationship_table_name = self.get_table_name(T_RELATIONSHIPS)
        statements += [
            "DROP TABLE IF EXISTS %s" % x for x in
            relationship_table_name, relationship_type_table_name]
        #
        # The relationship type table has the module #, relationship name
        # and object names of every relationship reported by
        # pipeline.get_relationship_columns()
        #
        columns = [COL_RELATIONSHIP_TYPE_ID, COL_MODULE_NUMBER,
                   COL_RELATIONSHIP, COL_OBJECT_NAME1, COL_OBJECT_NAME2]
        types = ["integer primary key", "integer", "varchar(255)",
                 "varchar(255)", "varchar(255)"]
        rtt_unique_name = self.get_table_name(CONSTRAINT_RT_UNIQUE)
        statement = "CREATE TABLE %s " % relationship_type_table_name
        statement += "(" + ", ".join(["%s %s" % (c, t)
                                      for c, t in zip(columns, types)])
        statement += ", CONSTRAINT %s UNIQUE ( " % rtt_unique_name
        statement += ", ".join(columns) + " ))"
        statements.append(statement)
        #
        # Create a row in this table for each relationship
        #
        d = self.get_dictionary()
        if T_RELATIONSHIP_TYPES not in d:
            d[T_RELATIONSHIP_TYPES] = {}
        rd = d[T_RELATIONSHIP_TYPES]

        for i, (module_num, relationship, o1, o2, when) in \
                enumerate(pipeline.get_object_relationships()):
            relationship_type_id = i + 1
            statement = "INSERT INTO %s " % relationship_type_table_name
            statement += "( " + ", ".join(columns) + ") "
            statement += "VALUES(%d, %d, '%s', '%s', '%s')" % (
                relationship_type_id, module_num, relationship, o1, o2)
            statements.append(statement)
            rd[module_num, relationship, o1, o2] = relationship_type_id
        #
        # Create the relationships table
        #
        columns = [COL_RELATIONSHIP_TYPE_ID,
                   COL_IMAGE_NUMBER1, COL_OBJECT_NUMBER1,
                   COL_IMAGE_NUMBER2, COL_OBJECT_NUMBER2]
        statement = "CREATE TABLE %s " % relationship_table_name
        statement += "( " + ", ".join(["%s integer" % c for c in columns])
        statement += " ,CONSTRAINT %s FOREIGN KEY ( %s ) " % (
            self.get_table_name(FK_RELATIONSHIP_TYPE_ID),
            COL_RELATIONSHIP_TYPE_ID)
        statement += " REFERENCES %s ( %s )" % (
            relationship_type_table_name, COL_RELATIONSHIP_TYPE_ID)
        statement += " ,CONSTRAINT %s UNIQUE" % self.get_table_name(
                CONSTRAINT_R_UNIQUE)
        statement += " ( " + ", ".join(columns) + " ))"
        statements.append(statement)
        #
        # Create indexes for both the first and second objects
        #
        for index_name, image_column, object_column in (
                (I_RELATIONSHIPS1, COL_IMAGE_NUMBER1, COL_OBJECT_NUMBER1),
                (I_RELATIONSHIPS2, COL_IMAGE_NUMBER2, COL_OBJECT_NUMBER2)):
            statement = "CREATE INDEX %s ON %s ( %s, %s, %s )" % (
                self.get_table_name(index_name),
                relationship_table_name, image_column, object_column,
                COL_RELATIONSHIP_TYPE_ID)
            statements.append(statement)
        #
        # Create the relationship view
        #
        statement = "CREATE VIEW %s AS SELECT " % relationship_view_name
        statement += ", ".join([
                                   "T.%s" % col for col in (
                COL_MODULE_NUMBER, COL_RELATIONSHIP,
                COL_OBJECT_NAME1, COL_OBJECT_NAME2)]) + ", "
        statement += ", ".join([
                                   "R.%s" % col for col in (
                COL_IMAGE_NUMBER1, COL_OBJECT_NUMBER1,
                COL_IMAGE_NUMBER2, COL_OBJECT_NUMBER2)])
        statement += " FROM %s T JOIN %s R ON " % (
            relationship_type_table_name, relationship_table_name)
        statement += " T.%s = R.%s" % (
            COL_RELATIONSHIP_TYPE_ID, COL_RELATIONSHIP_TYPE_ID)
        statements.append(statement)
        return statements

    def get_relationship_type_id(self, workspace, module_num, relationship,
                                 object_name1, object_name2):
        '''Get the relationship_type_id for the given relationship

        workspace - the analysis workspace

        module_num - the module number of the module that generated the
                     record

        relationship - the name of the relationship

        object_name1 - the name of the first object in the relationship

        object_name2 - the name of the second object in the relationship

        Returns the relationship_type_id that joins to the relationship
        type record in the relationship types table.

        NOTE: this should not be called for CSV databases.
        '''
        assert self.db_type != DB_MYSQL_CSV

        d = self.get_dictionary()
        if T_RELATIONSHIP_TYPES not in d:
            if self.db_type == DB_SQLITE:
                try:
                    json_result = workspace.interaction_request(
                            self, self.INTERACTION_GET_RELATIONSHIP_TYPES)
                except workspace.NoInteractionException:
                    # Assume headless and call as if through ZMQ
                    json_result = \
                        self.handle_interaction_get_relationship_types()
                d[T_RELATIONSHIP_TYPES] = \
                    self.grt_interaction_to_dict(json_result)
            else:
                d[T_RELATIONSHIP_TYPES] = \
                    self.get_relationship_types(self.cursor)
        rd = d[T_RELATIONSHIP_TYPES]

        key = (module_num, relationship, object_name1, object_name2)
        if key not in rd:
            if self.db_type == DB_SQLITE:
                try:
                    rd[key] = workspace.interaction_request(
                            self, self.INTERACTION_ADD_RELATIONSHIP_TYPE, *key)
                except workspace.NoInteractionException:
                    rd[key] = \
                        self.handle_interaction_add_relationship_type(*key)
            else:
                rd[key] = self.add_relationship_type(
                        module_num, relationship, object_name1, object_name2,
                        self.cursor)
        return rd[key]

    def write_mysql_table_defs(self, workspace):
        """Write the table definitions to the SETUP.SQL file

        The column order here is the same as in get_pipeline_measurement_columns
        with the aggregates following the regular image columns.
        """

        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        measurements = workspace.measurements

        m_cols = self.get_pipeline_measurement_columns(pipeline,
                                                       image_set_list)
        mappings = self.get_column_name_mappings(pipeline, image_set_list)

        file_name_width, path_name_width = self.get_file_path_width(workspace)
        metadata_name_width = 128
        file_name = "%sSETUP.SQL" % self.sql_file_prefix
        path_name = self.make_full_filename(file_name, workspace)
        fid = open(path_name, "wt")
        fid.write("CREATE DATABASE IF NOT EXISTS %s;\n" % self.db_name.value)
        fid.write("USE %s;\n" % self.db_name.value)
        fid.write(self.get_create_image_table_statement(pipeline,
                                                        image_set_list) + ";\n")
        #
        # Write out the per-object table
        #
        if self.objects_choice != O_NONE:
            if self.separate_object_tables == OT_COMBINE:
                data = [(None, cpmeas.OBJECT)]
            else:
                data = [(x, x) for x in self.get_object_names(
                        pipeline, image_set_list)]

            for gcot_name, object_name in data:
                fid.write(self.get_create_object_table_statement(
                        gcot_name, pipeline, image_set_list) + ";\n")
        else:
            data = []
        for statement in self.get_experiment_table_statements(workspace):
            fid.write(statement + ";\n")
        fid.write("""
LOAD DATA LOCAL INFILE '%s_%s.CSV' REPLACE INTO TABLE %s
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
""" %
                  (self.base_name(workspace), cpmeas.IMAGE, self.get_table_name(cpmeas.IMAGE)))

        for gcot_name, object_name in data:
            fid.write("""
LOAD DATA LOCAL INFILE '%s_%s.CSV' REPLACE INTO TABLE %s
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
""" % (self.base_name(workspace), object_name,
       self.get_table_name(object_name)))

        if self.objects_choice != O_NONE and self.separate_object_tables == OT_VIEW:
            fid.write("\n" + self.get_create_object_view_statement(
                    [object_name for gcot_name, object_name in data], pipeline, image_set_list) + ";\n")

        if self.wants_relationship_table:
            for statement in self.get_create_relationships_table_statements(pipeline):
                fid.write(statement + ";\n")
            fid.write("""
LOAD DATA LOCAL INFILE '%s_%s.CSV' REPLACE INTO TABLE %s
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
""" % (self.base_name(workspace), T_RELATIONSHIPS,
       self.get_table_name(T_RELATIONSHIPS)))
        if self.wants_well_tables:
            self.write_mysql_table_per_well(
                    workspace.pipeline, workspace.image_set_list, fid)
        fid.close()

    def write_mysql_table_per_well(self, pipeline, image_set_list, fid=None):
        '''Write SQL statements to generate a per-well table

        pipeline - the pipeline being run (to get feature names)
        image_set_list -
        fid - file handle of file to write or None if statements
              should be written to a separate file.
        '''
        if fid is None:
            file_name = "%s_Per_Well_SETUP.SQL" % self.sql_file_prefix
            path_name = self.make_full_filename(file_name)
            fid = open(path_name, "wt")
            needs_close = True
        else:
            needs_close = False
        fid.write("USE %s;\n" % self.db_name.value)
        table_prefix = self.get_table_prefix()
        #
        # Do in two passes. Pass # 1 makes the column name mappings for the
        # well table. Pass # 2 writes the SQL
        #
        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        object_names = self.get_object_names(pipeline, image_set_list)
        columns = self.get_pipeline_measurement_columns(pipeline, image_set_list)
        for aggname in self.agg_well_names:
            well_mappings = ColumnNameMapping()
            for do_mapping, do_write in ((True, False), (False, True)):
                if do_write:
                    fid.write("CREATE TABLE %sPer_Well_%s AS SELECT " %
                              (self.get_table_prefix(), aggname))
                for i, object_name in enumerate(object_names + [cpmeas.IMAGE]):
                    if object_name == cpmeas.IMAGE:
                        object_table_name = "IT"
                    elif self.separate_object_tables == OT_COMBINE:
                        object_table_name = "OT"
                    else:
                        object_table_name = "OT%d" % (i + 1)
                    for column in columns:
                        column_object_name, feature, data_type = column[:3]
                        if column_object_name != object_name:
                            continue
                        if self.ignore_feature(object_name, feature):
                            continue
                        #
                        # Don't take an aggregate on a string column
                        #
                        if data_type.startswith(cpmeas.COLTYPE_VARCHAR):
                            continue
                        feature_name = "%s_%s" % (object_name, feature)
                        colname = mappings[feature_name]
                        well_colname = "%s_%s" % (aggname, colname)
                        if do_mapping:
                            well_mappings.add(well_colname)
                        if do_write:
                            fid.write("%s(%s.%s) as %s,\n" %
                                      (aggname, object_table_name, colname,
                                       well_mappings[well_colname]))
            fid.write("IT.Image_Metadata_Plate, IT.Image_Metadata_Well "
                      "FROM %sPer_Image IT\n" % table_prefix)
            if len(object_names) == 0:
                pass
            elif self.separate_object_tables == OT_COMBINE:
                fid.write("JOIN %s OT ON IT.%s = OT.%s\n" %
                          (self.get_table_name(cpmeas.OBJECT), C_IMAGE_NUMBER, C_IMAGE_NUMBER))
            elif len(object_names) == 1:
                fid.write("JOIN %s OT1 ON IT.%s = OT1.%s\n" %
                          (self.get_table_name(object_names[0]), C_IMAGE_NUMBER, C_IMAGE_NUMBER))
            else:
                #
                # We make up a table here that lists all of the possible
                # image and object numbers from any of the object numbers.
                # We need this to do something other than a cartesian join
                # between object tables.
                #
                fid.write(
                        "RIGHT JOIN (SELECT DISTINCT %s, %s FROM\n" % (C_IMAGE_NUMBER, C_OBJECT_NUMBER))
                fid.write("(SELECT %s, %s_%s as %s FROM %s\n" %
                          (C_IMAGE_NUMBER, object_names[0], M_NUMBER_OBJECT_NUMBER, C_OBJECT_NUMBER,
                           self.get_table_name(object_names[0])))
                for object_name in object_names[1:]:
                    fid.write("UNION SELECT %s, %s_%s as %s "
                              "FROM %s\n" %
                              (C_IMAGE_NUMBER, object_name, M_NUMBER_OBJECT_NUMBER, C_OBJECT_NUMBER,
                               self.get_table_name(object_name)))
                fid.write(") N_INNER) N ON IT.%s = N.%s\n" % (C_IMAGE_NUMBER, C_IMAGE_NUMBER))
                for i, object_name in enumerate(object_names):
                    fid.write("LEFT JOIN %s OT%d " %
                              (self.get_table_name(object_name), i + 1))
                    fid.write("ON N.%s = OT%d.%s " % (C_IMAGE_NUMBER, i + 1, C_IMAGE_NUMBER))
                    fid.write("AND N.%s = OT%d.%s_%s\n" %
                              (C_OBJECT_NUMBER, i + 1, object_name, M_NUMBER_OBJECT_NUMBER))
            fid.write("GROUP BY IT.Image_Metadata_Plate, "
                      "IT.Image_Metadata_Well;\n\n""")

        if needs_close:
            fid.close()

    def write_oracle_table_defs(self, workspace):
        raise NotImplementedError("Writing to an Oracle database is not yet supported")

    def base_name(self, workspace):
        """The base for the output file name"""
        m = workspace.measurements
        first = m.image_set_start_number
        last = m.image_set_number
        return '%s%d_%d' % (self.sql_file_prefix, first, last)

    def write_csv_data(self, workspace):
        """Write the data in the measurements out to the csv files
        workspace - contains the measurements
        """
        if self.show_window:
            disp_header = ['Table', 'Filename']
            disp_columns = []

        zeros_for_nan = False
        measurements = workspace.measurements
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        image_filename = self.make_full_filename('%s_%s.CSV' % (self.base_name(workspace), cpmeas.IMAGE), workspace)
        fid_per_image = open(image_filename, "wb")
        columns = self.get_pipeline_measurement_columns(pipeline,
                                                        image_set_list, remove_postgroup_key=True)
        agg_columns = self.get_aggregate_columns(pipeline, image_set_list)
        for image_number in measurements.get_image_numbers():
            image_row = []
            image_row.append(image_number)
            for object_name, feature, coltype in columns:
                if object_name != cpmeas.IMAGE:
                    continue
                if self.ignore_feature(object_name, feature, measurements):
                    continue
                feature_name = "%s_%s" % (object_name, feature)
                if not measurements.has_feature(cpmeas.IMAGE, feature):
                    value = np.NaN
                else:
                    value = measurements.get_measurement(
                            cpmeas.IMAGE, feature, image_number)
                if isinstance(value, np.ndarray):
                    value = value[0]
                if coltype.startswith(cpmeas.COLTYPE_VARCHAR):
                    if isinstance(value, str) or isinstance(value, unicode):
                        value = '"' + MySQLdb.escape_string(value) + '"'
                    elif value is None:
                        value = "NULL"
                    else:
                        value = '"' + MySQLdb.escape_string(value) + '"'
                elif np.isnan(value) or np.isinf(value):
                    value = "NULL"

                image_row.append(value)
            #
            # Add the aggregate measurements
            #
            agg_dict = measurements.compute_aggregate_measurements(
                    image_number, self.agg_names)
            image_row += [agg_dict[col[3]] for col in agg_columns]
            fid_per_image.write(','.join([str(x) for x in image_row]) + "\n")
        fid_per_image.close()
        #
        # Object tables
        #
        object_names = self.get_object_names(pipeline, image_set_list)
        if len(object_names) == 0:
            return

        if self.separate_object_tables == OT_COMBINE:
            data = [(cpmeas.OBJECT, object_names)]
        else:
            data = [(object_name, [object_name])
                    for object_name in object_names]
        for file_object_name, object_list in data:
            file_name = "%s_%s.CSV" % (self.base_name(workspace),
                                       file_object_name)
            file_name = self.make_full_filename(file_name)
            fid = open(file_name, "wb")
            csv_writer = csv.writer(fid, lineterminator='\n')
            for image_number in measurements.get_image_numbers():
                max_count = 0
                for object_name in object_list:
                    count = measurements.get_measurement(
                            cpmeas.IMAGE, "Count_%s" % object_name, image_number)
                    max_count = max(max_count, int(count))
                d = {}
                for j in range(max_count):
                    object_row = [image_number]
                    if file_object_name == cpmeas.OBJECT:
                        # the object number
                        object_row.append(j + 1)
                    #
                    # Write out in same order as in the column definition
                    for object_name in object_list:
                        for object_name_to_check, feature, coltype in columns:
                            if object_name_to_check != object_name:
                                continue
                            key = (object_name, feature)
                            if key not in d:
                                if not measurements.has_feature(
                                        object_name, feature):
                                    values = None
                                else:
                                    values = measurements.get_measurement(
                                            object_name, feature, image_number)
                                d[key] = values
                            else:
                                values = d[key]
                            if (values is None or len(values) <= j or
                                    np.isnan(values[j]) or np.isinf(values[j])):
                                value = "NULL"
                            else:
                                value = values[j]
                            object_row.append(value)
                    csv_writer.writerow(object_row)
            fid.close()
            if self.show_window:
                disp_columns.append((file_object_name, 'Wrote %s' % file_name))
        #
        # Relationships table
        #
        # Note that the code here assumes that pipeline.get_object_relationships
        # returns the rows in the same order every time it's called.
        #
        if self.wants_relationship_table:
            file_name = "%s_%s.CSV" % (
                self.base_name(workspace), T_RELATIONSHIPS)
            file_name = self.make_full_filename(file_name)
            with open(file_name, "wb") as fid:
                csv_writer = csv.writer(fid, lineterminator='\n')
                for i, (module_num, relationship,
                        object_number1, object_number2, when) \
                        in enumerate(pipeline.get_object_relationships()):
                    relationship_type_id = i + 1
                    r = measurements.get_relationships(
                            module_num, relationship,
                            object_number1, object_number2)
                    for i1, o1, i2, o2 in r:
                        csv_writer.writerow((
                            relationship_type_id, i1, o1, i2, o2))
            if self.show_window:
                disp_columns.append((T_RELATIONSHIPS, 'Wrote %s' % file_name))

        if self.show_window:
            workspace.display_data.header = disp_header
            workspace.display_data.columns = disp_columns

    @staticmethod
    def should_write(column, post_group):
        '''Determine if a column should be written in run or post_group

        column - 3 or 4 tuple column from get_measurement_columns
        post_group - True if in post_group, false if in run

        returns True if column should be written
        '''
        if len(column) == 3:
            return not post_group
        if not hasattr(column[3], "has_key"):
            return not post_group
        if not column[3].has_key(cpmeas.MCA_AVAILABLE_POST_GROUP):
            return not post_group
        return (post_group if column[3][cpmeas.MCA_AVAILABLE_POST_GROUP]
                else not post_group)

    def write_data_to_db(self, workspace,
                         post_group=False,
                         image_number=None):
        """Write the data in the measurements out to the database
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        image_number - image number for primary database key. Defaults to current.
        """
        if self.show_window:
            disp_header = ["Table", "Statement"]
            disp_columns = []
        try:
            zeros_for_nan = False
            measurements = workspace.measurements
            assert isinstance(measurements, cpmeas.Measurements)
            pipeline = workspace.pipeline
            image_set_list = workspace.image_set_list
            measurement_cols = self.get_pipeline_measurement_columns(pipeline,
                                                                     image_set_list)
            mapping = self.get_column_name_mappings(pipeline, image_set_list)

            ###########################################
            #
            # The experiment table
            #
            # Update the modification timestamp. This
            # has a side-effect of synchronizing (and blocking
            # writes to the database for transactional DB engines)
            # by taking a lock on the single row of
            # the experiment table
            ###########################################
            stmt = "UPDATE %s SET %s='%s'" % \
                   (self.get_table_name(cpmeas.EXPERIMENT),
                    M_MODIFICATION_TIMESTAMP,
                    datetime.datetime.now().isoformat())
            execute(self.cursor, stmt, return_result=False)
            ###########################################
            #
            # The image table
            #
            ###########################################
            if image_number is None:
                image_number = measurements.image_set_number

            image_row = []
            if not post_group:
                image_row += [(image_number, cpmeas.COLTYPE_INTEGER, C_IMAGE_NUMBER)]
            feature_names = set(measurements.get_feature_names(cpmeas.IMAGE))
            for m_col in measurement_cols:
                if m_col[0] != cpmeas.IMAGE:
                    continue
                if not self.should_write(m_col, post_group):
                    continue
                #
                # Skip if feature name not in measurements. This
                # can happen if image set gets aborted or for some legacy
                # measurement files.
                #
                if m_col[1] not in feature_names:
                    continue
                feature_name = "%s_%s" % (cpmeas.IMAGE, m_col[1])
                value = measurements.get_measurement(
                        cpmeas.IMAGE, m_col[1], image_number)
                if isinstance(value, np.ndarray):
                    value = value[0]
                if isinstance(value, float) and not np.isfinite(value) and zeros_for_nan:
                    value = 0
                image_row.append((value, m_col[2], feature_name))
            #
            # Aggregates for the image table
            #
            agg_dict = measurements.compute_aggregate_measurements(
                    image_number, self.agg_names)
            agg_columns = self.get_aggregate_columns(pipeline, image_set_list,
                                                     post_group)
            image_row += [(agg_dict[agg[3]],
                           cpmeas.COLTYPE_FLOAT,
                           agg[3])
                          for agg in agg_columns]

            #
            # Delete any prior data for this image
            #
            # Useful if you rerun a partially-complete batch
            #
            if not post_group:
                stmt = ('DELETE FROM %s WHERE %s=%d' %
                        (self.get_table_name(cpmeas.IMAGE),
                         C_IMAGE_NUMBER,
                         image_number))
                execute(self.cursor, stmt, return_result=False)
                #
                # Delete relationships as well.
                #
                if self.wants_relationship_table:
                    for col in (COL_IMAGE_NUMBER1, COL_IMAGE_NUMBER2):
                        stmt = 'DELETE FROM %s WHERE %s=%d' % (
                            self.get_table_name(T_RELATIONSHIPS), col,
                            image_number)
                        execute(self.cursor, stmt, return_result=False)

            ########################################
            #
            # Object tables
            #
            ########################################
            object_names = self.get_object_names(pipeline, image_set_list)
            if len(object_names) > 0:
                if self.separate_object_tables == OT_COMBINE:
                    data = [(cpmeas.OBJECT, object_names)]
                else:
                    data = [(object_name, [object_name])
                            for object_name in object_names]
                for table_object_name, object_list in data:
                    table_name = self.get_table_name(table_object_name)
                    columns = [column for column in measurement_cols
                               if column[0] in object_list
                               and self.should_write(column, post_group)]
                    if post_group and len(columns) == 0:
                        continue
                    max_count = 0
                    for object_name in object_list:
                        ftr_count = "Count_%s" % object_name
                        count = measurements.get_measurement(
                                cpmeas.IMAGE, ftr_count, image_number)
                        max_count = max(max_count, int(count))
                    column_values = []
                    for column in columns:
                        object_name, feature, coltype = column[:3]
                        values = measurements.get_measurement(
                                object_name, feature, image_number)

                        if len(values) < max_count:
                            values = list(values) + [None] * (max_count - len(values))
                        values = [
                            None if v is None or np.isnan(v) or np.isinf(v)
                            else str(v)
                            for v in values]
                        column_values.append(values)
                    object_cols = []
                    if not post_group:
                        object_cols += [C_IMAGE_NUMBER]
                    if table_object_name == cpmeas.OBJECT:
                        object_number_column = C_OBJECT_NUMBER
                        if not post_group:
                            object_cols += [object_number_column]
                        object_numbers = np.arange(1, max_count + 1)
                    else:
                        object_number_column = "_".join((object_name, M_NUMBER_OBJECT_NUMBER))
                        object_numbers = measurements.get_measurement(
                                object_name, M_NUMBER_OBJECT_NUMBER, image_number)

                    object_cols += [mapping["%s_%s" % (column[0], column[1])]
                                    for column in columns]
                    object_rows = []
                    for j in range(max_count):
                        if not post_group:
                            object_row = [image_number]
                            if table_object_name == cpmeas.OBJECT:
                                # the object number
                                object_row.append(object_numbers[j])
                        else:
                            object_row = []

                        for column, values in zip(columns, column_values):
                            object_name, feature, coltype = column[:3]
                            object_row.append(values[j])
                        if post_group:
                            object_row.append(object_numbers[j])
                        object_rows.append(object_row)
                    #
                    # Delete any prior data for this image
                    #
                    if not post_group:
                        stmt = ('DELETE FROM %s WHERE %s=%d' %
                                (table_name, C_IMAGE_NUMBER, image_number))

                        execute(self.cursor, stmt, return_result=False)
                        #
                        # Write the object table data
                        #
                        stmt = ('INSERT INTO %s (%s) VALUES (%s)' %
                                (table_name,
                                 ','.join(object_cols),
                                 ','.join(['%s'] * len(object_cols))))
                    else:
                        stmt = (
                            ('UPDATE %s SET\n' % table_name) +
                            (',\n'.join(["  %s=%%s" % c for c in object_cols])) +
                            ('\nWHERE %s = %d' % (C_IMAGE_NUMBER, image_number)) +
                            ('\nAND %s = %%s' % object_number_column))

                    if self.db_type == DB_MYSQL:
                        # Write 25 rows at a time (to get under the max_allowed_packet limit)
                        for i in range(0, len(object_rows), 25):
                            my_rows = object_rows[i:min(i + 25, len(object_rows))]
                            self.cursor.executemany(stmt, my_rows)
                        if self.show_window and len(object_rows) > 0:
                            disp_columns.append(
                                    (table_name, self.truncate_string_for_display(stmt % tuple(my_rows[0]))))
                    else:
                        for row in object_rows:
                            row = ['NULL' if x is None else x for x in row]
                            row_stmt = stmt % tuple(row)
                            execute(self.cursor, row_stmt, return_result=False)
                        if self.show_window and len(object_rows) > 0:
                            disp_columns.append(
                                    (table_name,
                                     self.truncate_string_for_display(row_stmt)))

            image_table = self.get_table_name(cpmeas.IMAGE)
            replacement = '%s' if self.db_type == DB_MYSQL else "?"
            image_row_values = [
                None if field[0] is None
                else None if ((field[1] == cpmeas.COLTYPE_FLOAT) and
                              (np.isnan(field[0]) or np.isinf(field[0])))
                else float(field[0]) if (field[1] == cpmeas.COLTYPE_FLOAT)
                else int(field[0]) if (field[1] == cpmeas.COLTYPE_INTEGER)
                else buffer(field[0])
                if field[1] in (cpmeas.COLTYPE_BLOB, cpmeas.COLTYPE_LONGBLOB,
                                cpmeas.COLTYPE_MEDIUMBLOB)
                else field[0] for field in image_row]
            if len(image_row) > 0:
                if not post_group:
                    stmt = (
                        'INSERT INTO %s (%s) VALUES (%s)' %
                        (image_table,
                         ','.join([mapping[colname] for val, dtype, colname in image_row]),
                         ','.join([replacement] * len(image_row))))
                else:
                    stmt = (
                        ('UPDATE %s SET\n' % image_table) +
                        ',\n'.join(["  %s = %s" % (mapping[colname], replacement)
                                    for val, dtype, colname in image_row]) +
                        ('\nWHERE %s = %d' % (C_IMAGE_NUMBER, image_number)))
                execute(self.cursor, stmt, image_row_values, return_result=False)

            if self.show_window:
                disp_columns.append((image_table, self.truncate_string_for_display(
                        stmt + " VALUES(%s)" % ','.join(map(str, image_row_values))) if len(image_row) > 0 else ''))

            if self.wants_relationship_table:
                #
                # Relationships table - for SQLite, check for previous existence
                # but for MySQL use REPLACE INTO to do the same
                #
                rtbl_name = self.get_table_name(T_RELATIONSHIPS)
                columns = [COL_RELATIONSHIP_TYPE_ID,
                           COL_IMAGE_NUMBER1, COL_OBJECT_NUMBER1,
                           COL_IMAGE_NUMBER2, COL_OBJECT_NUMBER2]
                if self.db_type == DB_SQLITE:
                    stmt = "INSERT INTO %s (%s, %s, %s, %s, %s) " % \
                           tuple([rtbl_name] + columns)
                    stmt += " SELECT %d, %d, %d, %d, %d WHERE NOT EXISTS "
                    stmt += "(SELECT 'x' FROM %s WHERE " % rtbl_name
                    stmt += " AND ".join(["%s = %%d" % col for col in columns]) + ")"
                else:
                    stmt = "REPLACE INTO %s (%s, %s, %s, %s, %s) " % \
                           tuple([rtbl_name] + columns)
                    stmt += "VALUES (%s, %s, %s, %s, %s)"
                for module_num, relationship, object_name1, object_name2, when \
                        in pipeline.get_object_relationships():
                    if post_group != (when == cpmeas.MCA_AVAILABLE_POST_GROUP):
                        continue
                    r = measurements.get_relationships(
                            module_num, relationship, object_name1, object_name2,
                            image_numbers=[image_number])
                    rt_id = self.get_relationship_type_id(
                            workspace, module_num, relationship, object_name1,
                            object_name2)
                    if self.db_type == DB_MYSQL:
                        # max_allowed_packet is 16 MB by default
                        # 8 x 10 = 80/row -> 200K rows
                        row_values = [(rt_id, i1, o1, i2, o2)
                                      for i1, o1, i2, o2 in r]
                        self.cursor.executemany(stmt, row_values)
                        if self.show_window and len(r) > 0:
                            disp_columns.append(
                                    (rtbl_name, self.truncate_string_for_display(stmt % tuple(row_values[0]))))
                    else:
                        for i1, o1, i2, o2 in r:
                            row = (
                                rt_id, i1, o1, i2, o2, rt_id, i1, o1, i2, o2)
                            row_stmt = stmt % tuple(row)
                            execute(self.cursor, row_stmt, return_result=False)
                        if self.show_window and len(r) > 0:
                            disp_columns.append((rtbl_name, self.truncate_string_for_display(row_stmt)))

            if self.show_window:
                workspace.display_data.header = disp_header
                workspace.display_data.columns = disp_columns

            self.connection.commit()
        except:
            logger.error("Failed to write measurements to database", exc_info=True)
            self.connection.rollback()
            raise

    def truncate_string_for_display(self, s, field_size=100):
        ''' Any string with more than this # of characters will
                be truncated using an ellipsis.
        '''
        if len(s) > field_size:
            half = int(field_size - 3) / 2
            s = s[:half] + "..." + s[-half:]
        return s

    def display(self, workspace, figure):
        figure.set_subplots((1, 1,))
        if workspace.pipeline.test_mode:
            figure.subplot_table(
                    0, 0, [["Data not written to database in test mode"]])
        else:
            figure.subplot_table(0, 0,
                                 workspace.display_data.columns,
                                 col_labels=workspace.display_data.header)

    def display_post_run(self, workspace, figure):
        figure.set_subplots((1, 1,))
        figure.subplot_table(0, 0,
                             workspace.display_data.columns,
                             col_labels=workspace.display_data.header)

    def write_post_run_measurements(self, workspace):
        '''Write any experiment measurements marked as post-run'''
        columns = workspace.pipeline.get_measurement_columns()
        columns = filter(
                (lambda c:
                 c[0] == cpmeas.EXPERIMENT and len(c) > 3 and
                 c[3].get(cpmeas.MCA_AVAILABLE_POST_RUN, False)), columns)
        if len(columns) > 0:
            statement = "UPDATE %s SET " % self.get_table_name(cpmeas.EXPERIMENT)
            assignments = []
            for column in columns:
                if workspace.measurements.has_feature(
                        cpmeas.EXPERIMENT, column[1]):
                    value = workspace.measurements[cpmeas.EXPERIMENT, column[1]]
                    if value is not None:
                        assignments.append("%s='%s'" % (column[1], value))
            if len(assignments) > 0:
                statement += ",".join(assignments)
                with DBContext(self) as (connection, cursor):
                    cursor.execute(statement)
                    connection.commit()

    def write_properties_file(self, workspace):
        """Write the CellProfiler Analyst properties file"""
        all_properties = self.get_property_file_text(workspace)
        for properties in all_properties:
            fid = open(properties.file_name, "wt")
            fid.write(properties.text)
            fid.close()

    def get_property_file_text(self, workspace):
        '''Get the text for all property files

        workspace - the workspace from prepare_run

        Returns a list of Property objects which describe each property file

        The Property object has the following attributes:

        * object_name - the name of the object: "Object" if combining all tables,
                        otherwise the name of the relevant object.

        * file_name - save text in this file

        * text - the text to save

        * properties - a key / value dictionary of the properties
        '''

        class Properties(object):
            def __init__(self, object_name, file_name, text):
                self.object_name = object_name
                self.file_name = file_name
                self.text = text
                self.properties = {}
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or line.find("=") == -1:
                        continue
                    k, v = [x.strip() for x in line.split("=", 1)]
                    self.properties[k] = v

        result = []
        #
        # Get appropriate object names
        #
        if self.objects_choice != O_NONE:
            if self.separate_object_tables == OT_COMBINE:
                object_names = [self.location_object.value]
            elif self.separate_object_tables == OT_PER_OBJECT:
                if self.objects_choice == O_SELECT:
                    object_names = self.objects_list.value.split(',')
                else:
                    object_names = [
                        object_name
                        for object_name in workspace.measurements.get_object_names()
                        if (object_name != cpmeas.IMAGE) and
                        (not self.ignore_object(object_name))]
            elif self.separate_object_tables == OT_VIEW:
                object_names = [None]
        else:
            object_names = [None]

        default_image_names = []
        # Find all images that have FileName and PathName
        image_features = [
            c[1] for c in workspace.pipeline.get_measurement_columns()
            if c[0] == cpmeas.IMAGE]
        for feature in image_features:
            match = re.match('^%s_(.+)$' % C_FILE_NAME, feature)
            if match:
                default_image_names.append(match.groups()[0])

        if not self.properties_export_all_image_defaults:
            # Extract the user-specified images
            user_image_names = []
            for group in self.image_groups:
                user_image_names.append(group.image_cols.value)

        if self.db_type == DB_SQLITE:
            name = os.path.splitext(self.sqlite_file.value)[0]
        else:
            name = self.db_name.value
        tbl_prefix = self.get_table_prefix()
        if tbl_prefix is not "":
            if tbl_prefix.endswith('_'): tbl_prefix = tbl_prefix[:-1]
            name = "_".join((name, tbl_prefix))

        tblname = name
        date = datetime.datetime.now().ctime()
        db_type = (self.db_type == DB_MYSQL and 'mysql') or (
            self.db_type == DB_SQLITE and 'sqlite') or 'oracle_not_supported'
        db_port = (self.db_type == DB_MYSQL and 3306) or (self.db_type == DB_ORACLE and 1521) or ''
        db_host = self.db_host
        db_pwd = self.db_passwd
        db_name = self.db_name
        db_user = self.db_user
        db_sqlite_file = (self.db_type == DB_SQLITE and
                          self.make_full_filename(self.sqlite_file.value)) or ''
        if self.db_type == DB_MYSQL or self.db_type == DB_ORACLE:
            db_info = 'db_type      = %(db_type)s\n' % (locals())
            db_info += 'db_port      = %(db_port)d\n' % (locals())
            db_info += 'db_host      = %(db_host)s\n' % (locals())
            db_info += 'db_name      = %(db_name)s\n' % (locals())
            db_info += 'db_user      = %(db_user)s\n' % (locals())
            db_info += 'db_passwd    = %(db_pwd)s' % (locals())
        elif self.db_type == DB_SQLITE:
            db_info = 'db_type         = %(db_type)s\n' % (locals())
            db_info += 'db_sqlite_file  = %(db_sqlite_file)s' % (locals())
        elif self.db_type == DB_MYSQL_CSV:
            db_info = 'db_type      = mysql\n'
            db_info += 'db_port      = \n'
            db_info += 'db_host      = \n'
            db_info += 'db_name      = %(db_name)s\n' % (locals())
            db_info += 'db_user      = \n'
            db_info += 'db_passwd    = '

        spot_tables = '%sPer_Image' % (self.get_table_prefix())
        classification_type = "image" if self.properties_classification_type.value == CT_IMAGE else ""

        for object_name in object_names:
            if object_name:
                if self.objects_choice != O_NONE:
                    if self.separate_object_tables == OT_COMBINE:
                        cell_tables = '%sPer_Object' % (self.get_table_prefix())
                        object_id = C_OBJECT_NUMBER
                        filename = '%s.properties' % tblname
                        properties_object_name = "Object"
                        object_count = 'Image_Count_%s' % self.location_object.value
                        cell_x_loc = '%s_Location_Center_X' % self.location_object.value
                        cell_y_loc = '%s_Location_Center_Y' % self.location_object.value
                    elif self.separate_object_tables == OT_PER_OBJECT:
                        cell_tables = '%sPer_%s' % (self.get_table_prefix(), object_name)
                        object_id = '%s_Number_Object_Number' % object_name
                        filename = '%s_%s.properties' % (tblname, object_name)
                        properties_object_name = object_name
                        object_count = 'Image_Count_%s' % object_name
                        cell_x_loc = '%s_Location_Center_X' % object_name
                        cell_y_loc = '%s_Location_Center_Y' % object_name
            else:
                '''If object_name = None, it's either per_image only or a view '''
                if self.objects_choice == O_NONE:
                    cell_tables = ''
                    object_id = ''
                    filename = '%s.properties' % tblname
                    properties_object_name = object_name
                    object_count = ''
                    cell_x_loc = ''
                    cell_y_loc = ''
                elif self.separate_object_tables == OT_VIEW:
                    cell_tables = '%sPer_Object' % (self.get_table_prefix())
                    object_id = C_OBJECT_NUMBER
                    filename = '%s.properties' % tblname
                    properties_object_name = "Object"
                    object_count = 'Image_Count_%s' % self.location_object.value
                    cell_x_loc = '%s_Location_Center_X' % self.location_object.value
                    cell_y_loc = '%s_Location_Center_Y' % self.location_object.value

            file_name = self.make_full_filename(filename, workspace)
            unique_id = C_IMAGE_NUMBER
            image_thumbnail_cols = ','.join(
                    ['%s_%s_%s' % (cpmeas.IMAGE, C_THUMBNAIL, name)
                     for name in self.thumbnail_image_names.get_selections()]) \
                if self.want_image_thumbnails else ''

            if self.properties_export_all_image_defaults:
                image_file_cols = ','.join(
                        ['%s_%s_%s' % (cpmeas.IMAGE, C_FILE_NAME, name) for name in default_image_names])
                image_path_cols = ','.join(
                        ['%s_%s_%s' % (cpmeas.IMAGE, C_PATH_NAME, name) for name in default_image_names])

                # Provide default colors
                if len(default_image_names) == 1:
                    image_channel_colors = 'gray,'
                else:
                    image_channel_colors = 'red, green, blue, cyan, magenta, yellow, gray, ' + ('none, ' * 10)
                    num_images = len(default_image_names) + len(
                            set([name for name in self.thumbnail_image_names.get_selections()]).difference(
                                    default_image_names)) if self.want_image_thumbnails else 0
                    image_channel_colors = ','.join(image_channel_colors.split(',')[:num_images])
                image_names_csl = ','.join(default_image_names)  # Convert to comma-separated list

                if self.want_image_thumbnails:
                    selected_thumbs = [name for name in self.thumbnail_image_names.get_selections()]
                    thumb_names = [name for name in default_image_names if name in selected_thumbs] + [name for name in
                                                                                                       selected_thumbs
                                                                                                       if
                                                                                                       name not in default_image_names]
                    image_thumbnail_cols = ','.join(
                            ['%s_%s_%s' % (cpmeas.IMAGE, C_THUMBNAIL, name)
                             for name in thumb_names])
                else:
                    image_thumbnail_cols = ''

            else:
                # Extract user-specified image names and colors
                user_image_names = []
                image_channel_colors = []
                selected_image_names = []
                for group in self.image_groups:
                    selected_image_names += [group.image_cols.value]
                    if group.wants_automatic_image_name:
                        user_image_names += [group.image_cols.value]
                    else:
                        user_image_names += [group.image_name.value]
                    image_channel_colors += [group.image_channel_colors.value]

                image_file_cols = ','.join(
                        ['%s_%s_%s' % (cpmeas.IMAGE, C_FILE_NAME, name) for name in selected_image_names])
                image_path_cols = ','.join(
                        ['%s_%s_%s' % (cpmeas.IMAGE, C_PATH_NAME, name) for name in selected_image_names])

                # Try to match thumbnail order to selected image order
                if self.want_image_thumbnails:
                    selected_thumbs = [name for name in self.thumbnail_image_names.get_selections()]
                    thumb_names = [name for name in selected_image_names if name in selected_thumbs] + [name for name in
                                                                                                        selected_thumbs
                                                                                                        if
                                                                                                        name not in selected_image_names]
                    image_thumbnail_cols = ','.join(
                            ['%s_%s_%s' % (cpmeas.IMAGE, C_THUMBNAIL, name)
                             for name in thumb_names])
                else:
                    image_thumbnail_cols = ''
                    selected_thumbs = []

                # Convert to comma-separated list
                image_channel_colors = ','.join(
                        image_channel_colors + ['none'] * len(set(selected_thumbs).difference(selected_image_names)))
                image_names_csl = ','.join(user_image_names)

            group_statements = ''
            if self.properties_wants_groups:
                for group in self.group_field_groups:
                    group_statements += 'group_SQL_' + group.group_name.value + ' = SELECT ' + group.group_statement.value + ' FROM ' + spot_tables + '\n'

            filter_statements = ''
            if self.properties_wants_filters:
                if self.create_filters_for_plates:
                    plate_key = self.properties_plate_metadata.value
                    metadata_groups = workspace.measurements.group_by_metadata([plate_key])
                    for metadata_group in metadata_groups:
                        plate_text = re.sub("[^A-Za-z0-9_]", '_', metadata_group.get(
                                plate_key))  # Replace any odd characters with underscores
                        filter_name = 'Plate_%s' % plate_text
                        filter_statements += 'filter_SQL_' + filter_name + ' = SELECT ImageNumber' \
                                                                           ' FROM ' + spot_tables + \
                                             ' WHERE Image_Metadata_%s' \
                                             ' = "%s"\n' % (plate_key, metadata_group.get(plate_key))

                for group in self.filter_field_groups:
                    filter_statements += 'filter_SQL_' + group.filter_name.value + ' = SELECT ImageNumber' \
                                                                                   ' FROM ' + spot_tables + \
                                         ' WHERE ' + group.filter_statement.value + '\n'

            image_url = self.properties_image_url_prepend.value \
                if self.wants_properties_image_url_prepend else ""
            plate_type = "" if self.properties_plate_type.value == NONE_CHOICE else self.properties_plate_type.value
            plate_id = "" if self.properties_plate_metadata.value == NONE_CHOICE else "%s_%s_%s" % (
                cpmeas.IMAGE, cpmeas.C_METADATA, self.properties_plate_metadata.value)
            well_id = "" if self.properties_well_metadata.value == NONE_CHOICE else "%s_%s_%s" % (
                cpmeas.IMAGE, cpmeas.C_METADATA, self.properties_well_metadata.value)
            class_table = self.get_table_prefix() + self.properties_class_table_name.value

            contents = """#%(date)s
# ==============================================
#
# CellProfiler Analyst 2.0 properties file
#
# ==============================================

# ==== Database Info ====
%(db_info)s

# ==== Database Tables ====
image_table   = %(spot_tables)s
object_table  = %(cell_tables)s

# ==== Database Columns ====
# Specify the database column names that contain unique IDs for images and
# objects (and optionally tables).
#
# table_id (OPTIONAL): This field lets Classifier handle multiple tables if
#          you merge them into one and add a table_number column as a foreign
#          key to your per-image and per-object tables.
# image_id: must be a foreign key column between your per-image and per-object
#           tables
# object_id: the object key column from your per-object table

image_id      = %(unique_id)s
object_id     = %(object_id)s
plate_id      = %(plate_id)s
well_id       = %(well_id)s
series_id     = Image_Group_Number
group_id      = Image_Group_Number
timepoint_id  = Image_Group_Index

# Also specify the column names that contain X and Y coordinates for each
# object within an image.
cell_x_loc    = %(cell_x_loc)s
cell_y_loc    = %(cell_y_loc)s

# ==== Image Path and File Name Columns ====
# Classifier needs to know where to find the images from your experiment.
# Specify the column names from your per-image table that contain the image
# paths and file names here.
#
# Individual image files are expected to be monochromatic and represent a single
# channel. However, any number of images may be combined by adding a new channel
# path and filename column to the per-image table of your database and then
# adding those column names here.
#
# NOTE: These lists must have equal length!
image_path_cols = %(image_path_cols)s
image_file_cols = %(image_file_cols)s

# CPA will now read image thumbnails directly from the database, if chosen in ExportToDatabase.
image_thumbnail_cols = %(image_thumbnail_cols)s

# Give short names for each of the channels (respectively)...
image_names = %(image_names_csl)s

# Specify a default color for each of the channels (respectively)
# Valid colors are: [red, green, blue, magenta, cyan, yellow, gray, none]
image_channel_colors = %(image_channel_colors)s

# ==== Image Accesss Info ====
image_url_prepend = %(image_url)s

# ==== Dynamic Groups ====
# Here you can define groupings to choose from when classifier scores your experiment.  (eg: per-well)
# This is OPTIONAL, you may leave "groups = ".
# FORMAT:
#   group_XXX  =  MySQL select statement that returns image-keys and group-keys.  This will be associated with the group name "XXX" from above.
# EXAMPLE GROUPS:
#   groups               =  Well, Gene, Well+Gene,
#   group_SQL_Well       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Per_Image_Table.well FROM Per_Image_Table
#   group_SQL_Gene       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well
#   group_SQL_Well+Gene  =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.well, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well

%(group_statements)s

# ==== Image Filters ====
# Here you can define image filters to let you select objects from a subset of your experiment when training the classifier.
# FORMAT:
#   filter_SQL_XXX  =  MySQL select statement that returns image keys you wish to filter out.  This will be associated with the filter name "XXX" from above.
# EXAMPLE FILTERS:
#   filters           =  EMPTY, CDKs,
#   filter_SQL_EMPTY  =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene="EMPTY"
#   filter_SQL_CDKs   =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene REGEXP 'CDK.*'

%(filter_statements)s

# ==== Meta data ====
# What are your objects called?
# FORMAT:
#   object_name  =  singular object name, plural object name,
object_name  =  cell, cells,

# What size plates were used?  96, 384 or 5600?  This is for use in the PlateViewer. Leave blank if none
plate_type  = %(plate_type)s

# ==== Excluded Columns ====
# OPTIONAL
# Classifier uses columns in your per_object table to find rules. It will
# automatically ignore ID columns defined in table_id, image_id, and object_id
# as well as any columns that contain non-numeric data.
#
# Here you may list other columns in your per_object table that you wish the
# classifier to ignore when finding rules.
#
# You may also use regular expressions here to match more general column names.
#
# Example: classifier_ignore_columns = WellID, Meta_.*, .*_Position
#   This will ignore any column named "WellID", any columns that start with
#   "Meta_", and any columns that end in "_Position".
#
# A more restrictive example:
# classifier_ignore_columns = ImageNumber, ObjectNumber, .*Parent.*, .*Children.*, .*_Location_Center_.*,.*_Metadata_.*

classifier_ignore_columns  =  table_number_key_column, image_number_key_column, object_number_key_column

# ==== Other ====
# Specify the approximate diameter of your objects in pixels here.
image_tile_size   =  50

# Provides the image width and height. Used for per-image classification.
# If not set, it will be obtained from the Image_Width and Image_Height
# measurements in CellProfiler.

# image_width  = 1000
# image_height = 1000

# OPTIONAL
# Image Gallery can use a different tile size (in pixels) to create thumbnails for images
# If not set, it will be the same as image_tile_size

image_size =

# ======== Classification type ========
# OPTIONAL
# CPA 2.2.0 allows image classification instead of object classification.
# If left blank or set to "object", then Classifier will fetch objects (default).
# If set to "image", then Classifier will fetch whole images instead of objects.

classification_type  = %(classification_type)s

# ======== Auto Load Training Set ========
# OPTIONAL
# You may enter the full path to a training set that you would like Classifier
# to automatically load when started.

training_set  =

# ======== Area Based Scoring ========
# OPTIONAL
# You may specify a column in your per-object table which will be summed and
# reported in place of object-counts when scoring.  The typical use for this
# is to report the areas of objects on a per-image or per-group basis.

area_scoring_column =

# ======== Output Per-Object Classes ========
# OPTIONAL
# Here you can specify a MySQL table in your Database where you would like
# Classifier to write out class information for each object in the
# object_table

class_table  = %(class_table)s

# ======== Check Tables ========
# OPTIONAL
# [yes/no]  You can ask classifier to check your tables for anomalies such
# as orphaned objects or missing column indices.  Default is on.
# This check is run when Classifier starts and may take up to a minute if
# your object_table is extremely large.

check_tables = yes
    """ % (locals())
            result.append(Properties(properties_object_name,
                                     file_name,
                                     contents))
        return result

    def write_workspace_file(self, workspace):
        '''If requested, write a workspace file with selected measurements'''
        if self.db_type == DB_SQLITE:
            name = os.path.splitext(self.sqlite_file.value)[0]
        else:
            name = self.db_name.value
        tbl_prefix = self.get_table_prefix()
        if tbl_prefix is not "":
            if tbl_prefix.endswith('_'): tbl_prefix = tbl_prefix[:-1]
            name = "_".join((name, tbl_prefix))

        filename = '%s.workspace' % name
        file_name = self.make_full_filename(filename, workspace)

        fd = open(file_name, "wb")
        header_text = """CellProfiler Analyst workflow
version: 1
CP version : %d\n""" % int(re.sub(r"\.|rc\d{1}", "", cellprofiler.__version__))
        fd.write(header_text)
        display_tool_text = ""
        for workspace_group in self.workspace_measurement_groups:
            display_tool = workspace_group.measurement_display.value
            # A couple of tools are named a bit differently
            if workspace_group.measurement_display.value == W_SCATTERPLOT:
                display_tool = "Scatter"
            elif workspace_group.measurement_display.value == W_DENSITYPLOT:
                display_tool = "Density"
            display_tool_text += """
%s""" % display_tool

            axis_text = "x-axis" if workspace_group.measurement_display.value != W_PLATEVIEWER else "measurement"
            if workspace_group.x_measurement_type.value == cpmeas.IMAGE:
                axis_meas = "_".join((cpmeas.IMAGE, workspace_group.x_measurement_name.value))
            elif workspace_group.x_measurement_type.value == cpmeas.OBJECT:
                axis_meas = "_".join((workspace_group.x_object_name.value, workspace_group.x_measurement_name.value))
            elif workspace_group.x_measurement_type.value == W_INDEX:
                axis_meas = workspace_group.x_index_name.value
            axis_table = "x-table" if workspace_group.measurement_display.value in (
                W_SCATTERPLOT, W_DENSITYPLOT) else "table"
            table_name = self.get_table_name(
                    cpmeas.OBJECT if workspace_group.x_measurement_type.value == cpmeas.OBJECT else cpmeas.IMAGE)
            display_tool_text += """
\t%s: %s
\t%s: %s""" % (axis_text, axis_meas, axis_table, table_name)

            if workspace_group.measurement_display.value in (W_SCATTERPLOT, W_DENSITYPLOT):
                if workspace_group.y_measurement_type.value == cpmeas.IMAGE:
                    axis_meas = "_".join((cpmeas.IMAGE, workspace_group.y_measurement_name.value))
                elif workspace_group.y_measurement_type.value == cpmeas.OBJECT:
                    axis_meas = "_".join(
                            (workspace_group.y_object_name.value, workspace_group.y_measurement_name.value))
                elif workspace_group.y_measurement_type.value == W_INDEX:
                    axis_meas = workspace_group.y_index_name.value
                table_name = self.get_table_name(
                        cpmeas.OBJECT if workspace_group.y_measurement_type.value == cpmeas.OBJECT else cpmeas.IMAGE)
                display_tool_text += """
\ty-axis: %s
\ty-table: %s""" % (axis_meas, table_name)
            display_tool_text += "\n"

        fd.write(display_tool_text)
        fd.close()

    def get_file_path_width(self, workspace):
        """Compute the file name and path name widths needed in table defs"""
        m = workspace.measurements
        #
        # Find the length for the file name and path name fields
        #
        FileNameWidth = 128
        PathNameWidth = 128
        image_features = m.get_feature_names(cpmeas.IMAGE)
        for feature in image_features:
            if feature.startswith(C_FILE_NAME):
                names = [name
                         for name in m.get_all_measurements(cpmeas.IMAGE, feature)
                         if name is not None]
                if len(names) > 0:
                    FileNameWidth = max(FileNameWidth, np.max(map(len, names)))
            elif feature.startswith(C_PATH_NAME):
                names = [name
                         for name in m.get_all_measurements(cpmeas.IMAGE, feature)
                         if name is not None]
                if len(names) > 0:
                    PathNameWidth = max(PathNameWidth, np.max(map(len, names)))
        return FileNameWidth, PathNameWidth

    def get_table_prefix(self):
        if self.want_table_prefix.value:
            return self.table_prefix.value
        return ""

    def get_table_name(self, object_name):
        '''Return the table name associated with a given object

        object_name - name of object or "Image", "Object" or "Well"
        '''
        return self.get_table_prefix() + 'Per_' + object_name

    def get_pipeline_measurement_columns(self, pipeline, image_set_list, remove_postgroup_key=False):
        '''Get the measurement columns for this pipeline, possibly cached'''
        d = self.get_dictionary(image_set_list)
        if not d.has_key(D_MEASUREMENT_COLUMNS):
            d[D_MEASUREMENT_COLUMNS] = pipeline.get_measurement_columns()
            d[D_MEASUREMENT_COLUMNS] = self.filter_measurement_columns(
                    d[D_MEASUREMENT_COLUMNS])

        if remove_postgroup_key:
            d[D_MEASUREMENT_COLUMNS] = [x[:3] for x in d[D_MEASUREMENT_COLUMNS]]
        return d[D_MEASUREMENT_COLUMNS]

    def filter_measurement_columns(self, columns):
        '''Filter out and properly sort measurement columns'''
        columns = [x for x in columns
                   if not self.ignore_feature(x[0], x[1], True)]

        #
        # put Image ahead of any other object
        # put Number_ObjectNumber ahead of any other column
        #
        def cmpfn(x, y):
            if x[0] != y[0]:
                if x[0] == cpmeas.IMAGE:
                    return -1
                elif y[0] == cpmeas.IMAGE:
                    return 1
                else:
                    return cmp(x[0], y[0])
            if x[1] == M_NUMBER_OBJECT_NUMBER:
                return -1
            if y[1] == M_NUMBER_OBJECT_NUMBER:
                return 1
            return cmp(x[1], y[1])

        columns.sort(cmp=cmpfn)
        #
        # Remove all but the last duplicate
        #
        duplicate = [
                        c0[0] == c1[0] and c0[1] == c1[1]
                        for c0, c1 in zip(columns[:-1],
                                          columns[1:])] + [False]
        columns = [x for x, y in zip(columns, duplicate) if not y]
        return columns

    def obfuscate(self):
        '''Erase sensitive information about the database

        This is run on a copy of the pipeline, so it's ok to erase info.
        '''
        self.db_host.value = ''.join(['*'] * len(self.db_host.value))
        self.db_user.value = ''.join(['*'] * len(self.db_user.value))
        self.db_name.value = ''.join(['*'] * len(self.db_name.value))
        self.db_passwd.value = ''.join(['*'] * len(self.db_passwd.value))

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):

        DIR_DEFAULT_OUTPUT = "Default output folder"
        DIR_DEFAULT_IMAGE = "Default input folder"

        if from_matlab and variable_revision_number == 4:
            setting_values = setting_values + [cps.NO]
            variable_revision_number = 5
        if from_matlab and variable_revision_number == 5:
            if setting_values[-1] == cps.YES:
                setting_values = setting_values[:-1] + ["Yes - V1.0 format"]
            variable_revision_number = 6
        if from_matlab and variable_revision_number == 6:
            new_setting_values = [setting_values[0], setting_values[1]]
            if setting_values[2] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
                new_setting_values.append("MyExpt_")
            else:
                new_setting_values.append(cps.YES)
                new_setting_values.append(setting_values[2])
            new_setting_values.append(setting_values[3])
            if setting_values[4] == '.':
                new_setting_values.append(cps.YES)
                new_setting_values.append(setting_values[4])
            else:
                new_setting_values.append(cps.NO)
                new_setting_values.append(setting_values[4])
            if setting_values[5][:3] == cps.YES:
                new_setting_values.append(cps.YES)
            else:
                new_setting_values.append(cps.NO)
            from_matlab = False
            variable_revision_number = 6
            setting_values = new_setting_values
        if from_matlab and variable_revision_number == 7:
            #
            # Added object names
            #
            setting_values = (setting_values[:-1] + [cpmeas.IMAGE] +
                              [cps.DO_NOT_USE] * 3 + setting_values[-1:])
            variable_revision_number = 8

        if from_matlab and variable_revision_number == 8:
            #
            # Added more object names
            #
            setting_values = (setting_values[:-1] +
                              [cps.DO_NOT_USE] * 3 + setting_values[-1:])
            variable_revision_number = 9
        if from_matlab and variable_revision_number == 9:
            #
            # Per-well export
            #
            setting_values = (setting_values[:-1] +
                              [cps.NO, cps.DO_NOT_USE, cps.DO_NOT_USE] +
                              setting_values[-1:])
            variable_revision_number = 10
        if from_matlab and variable_revision_number == 10:
            new_setting_values = setting_values[0:2]
            if setting_values[2] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
                new_setting_values.append("MyExpt_")
            else:
                new_setting_values.append(cps.YES)
                new_setting_values.append(setting_values[2])
            new_setting_values.append(setting_values[3])
            if setting_values[4] == '.':
                new_setting_values.append(cps.YES)
                new_setting_values.append(setting_values[4])
            else:
                new_setting_values.append(cps.NO)
                new_setting_values.append(setting_values[4])
            if setting_values[18][:3] == cps.YES:
                new_setting_values.append(cps.YES)
            else:
                new_setting_values.append(cps.NO)
            #
            # store_csvs
            #
            new_setting_values.append(cps.YES)
            #
            # DB host / user / password
            #
            new_setting_values += ['imgdb01', 'cpuser', 'password']
            #
            # SQLite file name
            #
            new_setting_values += ['DefaultDB.db']
            #
            # Aggregate mean, median & std dev
            wants_mean = cps.NO
            wants_std_dev = cps.NO
            wants_median = cps.NO
            for setting in setting_values[5:8]:
                if setting == "Median":
                    wants_median = cps.YES
                elif setting == "Mean":
                    wants_mean = cps.YES
                elif setting == "Standard deviation":
                    wants_std_dev = cps.YES
            new_setting_values += [wants_mean, wants_median, wants_std_dev]
            #
            # Object export
            #
            if setting_values[8] == "All objects":
                new_setting_values += [O_ALL, ""]
            else:
                objects_list = []
                for setting in setting_values[8:15]:
                    if setting not in (cpmeas.IMAGE, cps.DO_NOT_USE):
                        objects_list.append(setting)
                if len(objects_list) > 0:
                    new_setting_values += [O_SELECT, ",".join(objects_list)]
                else:
                    new_setting_values += [O_NONE, ""]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 9

        if (not from_matlab) and variable_revision_number == 6:
            # Append default values for store_csvs, db_host, db_user,
            #  db_passwd, and sqlite_file to update to revision 7
            setting_values += [False, 'imgdb01', 'cpuser', '', 'DefaultDB.db']
            variable_revision_number = 7

        if (not from_matlab) and variable_revision_number == 7:
            # Added ability to selectively turn on aggregate measurements
            # which were all automatically calculated in version 7
            setting_values = setting_values + [True, True, True]
            variable_revision_number = 8

        if (not from_matlab) and variable_revision_number == 8:
            # Made it possible to choose objects to save
            #
            setting_values += [O_ALL, ""]
            variable_revision_number = 9

        if (not from_matlab) and variable_revision_number == 9:
            # Added aggregate per well choices
            #
            setting_values = (setting_values[:-2] +
                              [False, False, False] +
                              setting_values[-2:])
            variable_revision_number = 10

        if (not from_matlab) and variable_revision_number == 10:
            #
            # Added a directory choice instead of a checkbox
            #
            if setting_values[5] == cps.NO or setting_values[6] == '.':
                directory_choice = DIR_DEFAULT_OUTPUT
            elif setting_values[6] == '&':
                directory_choice = DIR_DEFAULT_IMAGE
            else:
                directory_choice = DIR_CUSTOM
            setting_values = (setting_values[:5] + [directory_choice] +
                              setting_values[6:])
            variable_revision_number = 11

        if (not from_matlab) and variable_revision_number == 11:
            #
            # Added separate "database type" of CSV files and removed
            # "store_csvs" setting
            #
            db_type = setting_values[0]
            store_csvs = (setting_values[8] == cps.YES)
            if db_type == DB_MYSQL and store_csvs:
                db_type = DB_MYSQL_CSV
            setting_values = ([db_type] + setting_values[1:8] +
                              setting_values[9:])
            variable_revision_number = 12

        if (not from_matlab) and variable_revision_number == 12:
            #
            # Added maximum column size
            #
            setting_values = setting_values + ["64"]
            variable_revision_number = 13

        if (not from_matlab) and variable_revision_number == 13:
            #
            # Added single/multiple table choice
            #
            setting_values = setting_values + [OT_COMBINE]
            variable_revision_number = 14

        if (not from_matlab) and variable_revision_number == 14:
            #
            # Combined directory_choice and output_folder into directory
            #
            dir_choice, custom_directory = setting_values[5:7]
            if dir_choice in (DIR_CUSTOM, DIR_CUSTOM_WITH_METADATA):
                if custom_directory.startswith('.'):
                    dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                elif custom_directory.startswith('&'):
                    dir_choice = DEFAULT_INPUT_SUBFOLDER_NAME
                    custom_directory = '.' + custom_directory[1:]
                else:
                    dir_choice = ABSOLUTE_FOLDER_NAME
            directory = cps.DirectoryPath.static_join_string(dir_choice,
                                                             custom_directory)
            setting_values = (setting_values[:5] + [directory] +
                              setting_values[7:])
            variable_revision_number = 15

        setting_values = list(setting_values)
        setting_values[OT_IDX] = OT_DICTIONARY.get(setting_values[OT_IDX],
                                                   setting_values[OT_IDX])

        # Standardize input/output directory name references
        SLOT_DIRCHOICE = 5
        directory = setting_values[SLOT_DIRCHOICE]
        directory = cps.DirectoryPath.upgrade_setting(directory)
        setting_values[SLOT_DIRCHOICE] = directory

        if (not from_matlab) and variable_revision_number == 15:
            #
            # Added 3 new args: url_prepend and thumbnail options
            #
            setting_values = setting_values + ["", cps.NO, ""]
            variable_revision_number = 16

        if (not from_matlab) and variable_revision_number == 16:
            #
            # Added binary choice for auto-scaling thumbnail intensities
            #
            setting_values = setting_values + [cps.NO]
            variable_revision_number = 17

        if (not from_matlab) and variable_revision_number == 17:
            #
            # Added choice for plate type in properties file
            #
            setting_values = setting_values + [NONE_CHOICE]
            variable_revision_number = 18

        if (not from_matlab) and variable_revision_number == 18:
            #
            # Added choices for plate and well metadata in properties file
            #
            setting_values = setting_values + [NONE_CHOICE, NONE_CHOICE]
            variable_revision_number = 19

        if (not from_matlab) and variable_revision_number == 19:
            #
            # Added configuration of image information, groups, filters in properties file
            #
            setting_values = setting_values + [cps.YES, "1", "1", "0"]  # Hidden counts
            setting_values = setting_values + ["None", cps.YES, "None", "gray"]  # Image info
            setting_values = setting_values + [cps.NO, "",
                                               "ImageNumber, Image_Metadata_Plate, Image_Metadata_Well"]  # Group specifications
            setting_values = setting_values + [cps.NO, cps.NO]  # Filter specifications
            variable_revision_number = 20

        if (not from_matlab) and variable_revision_number == 20:
            #
            # Added configuration of workspace file
            #
            setting_values = setting_values[:SETTING_WORKSPACE_GROUP_COUNT] + \
                             ["1"] + \
                             setting_values[SETTING_WORKSPACE_GROUP_COUNT:]  # workspace_measurement_count
            setting_values += [cps.NO]  # create_workspace_file
            setting_values += [W_SCATTERPLOT,  # measurement_display
                               cpmeas.IMAGE, cpmeas.IMAGE, "", C_IMAGE_NUMBER,
                               # x_measurement_type, x_object_name, x_measurement_name, x_index_name
                               cpmeas.IMAGE, cpmeas.IMAGE, "",
                               C_IMAGE_NUMBER]  # y_measurement_type, y_object_name, y_measurement_name, y_index_name
            variable_revision_number = 21

        if (not from_matlab) and variable_revision_number == 21:
            #
            # Added experiment name and location object
            #
            setting_values = (
                setting_values[:SETTING_FIXED_SETTING_COUNT_V21] +
                ["MyExpt", cps.NONE] +
                setting_values[SETTING_FIXED_SETTING_COUNT_V21:])
            variable_revision_number = 22

        if (not from_matlab) and variable_revision_number == 22:
            #
            # Added class table properties field
            #
            setting_values = (
                setting_values[:SETTING_FIXED_SETTING_COUNT_V22] +
                [""] +
                setting_values[SETTING_FIXED_SETTING_COUNT_V22:])
            variable_revision_number = 23

        if (not from_matlab) and variable_revision_number == 23:
            #
            # Added wants_relationships_table
            #
            setting_values = (
                setting_values[:SETTING_FIXED_SETTING_COUNT_V23] +
                [cps.NO] +
                setting_values[SETTING_FIXED_SETTING_COUNT_V23:])
            variable_revision_number = 24

        if (not from_matlab) and variable_revision_number == 24:
            #
            # Added allow_overwrite
            #
            setting_values = (
                setting_values[:SETTING_FIXED_SETTING_COUNT_V24] +
                [OVERWRITE_DATA] +
                setting_values[SETTING_FIXED_SETTING_COUNT_V24:])
            variable_revision_number = 25

        if (not from_matlab) and variable_revision_number == 25:
            #
            # added wants_properties_image_url_prepend setting
            #
            wants_urls = len(setting_values[SETTING_OFFSET_PROPERTIES_IMAGE_URL_PREPEND_V26]) > 0
            setting_values = \
                setting_values[:SETTING_FIXED_SETTING_COUNT_V25] + \
                [cps.YES if wants_urls else cps.NO] + \
                setting_values[SETTING_FIXED_SETTING_COUNT_V25:]
            variable_revision_number = 26

        # Added view creation to object table settings
        setting_values[OT_IDX] = OT_DICTIONARY.get(setting_values[OT_IDX],
                                                   setting_values[OT_IDX])

        if (not from_matlab) and variable_revision_number == 26:
            #
            # added classification_type setting
            #
            setting_values = (
                setting_values[:SETTING_FIXED_SETTING_COUNT_V26] +
                [CT_OBJECT] +
                setting_values[SETTING_FIXED_SETTING_COUNT_V26:])
            variable_revision_number = 27

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


class ColumnNameMapping:
    """Represents a mapping of feature name to column name"""

    def __init__(self, max_len=64):
        self.__dictionary = {}
        self.__mapped = False
        self.__max_len = max_len

    def add(self, feature_name):
        """Add a feature name to the collection"""

        self.__dictionary[feature_name] = feature_name
        self.__mapped = False

    def __getitem__(self, feature_name):
        """Return the column name for a feature"""
        if not self.__mapped:
            self.do_mapping()
        return self.__dictionary[feature_name]

    def keys(self):
        return self.__dictionary.keys()

    def values(self):
        if not self.__mapped:
            self.do_mapping()
        return self.__dictionary.values()

    def do_mapping(self):
        """Scan the dictionary for feature names > max_len and shorten"""
        reverse_dictionary = {}
        problem_names = []
        seeded_random = False
        valid_name_regexp = "^[0-9a-zA-Z_$]+$"
        for key in sorted(self.__dictionary.keys()):
            value = self.__dictionary[key]
            reverse_dictionary[value] = key
            if len(value) > self.__max_len:
                problem_names.append(value)
            elif not re.match(valid_name_regexp, value):
                problem_names.append(value)

        for name in problem_names:
            key = reverse_dictionary[name]
            orig_name = name
            if not re.match(valid_name_regexp, name):
                name = re.sub("[^0-9a-zA-Z_$]", "_", name)
                if reverse_dictionary.has_key(name):
                    i = 1
                    while reverse_dictionary.has_key(name + str(i)):
                        i += 1
                    name = name + str(i)
            starting_name = name
            starting_positions = [x for x in [name.find("_"), 0]
                                  if x != -1]
            for pos in starting_positions:
                # remove vowels
                to_remove = len(name) - self.__max_len
                if to_remove > 0:
                    remove_count = 0
                    for to_drop in (('a', 'e', 'i', 'o', 'u'),
                                    ('b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
                                     'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'),
                                    ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                     'W', 'X', 'Y', 'Z')):
                        for index in range(len(name) - 1, pos - 1, -1):
                            if name[index] in to_drop:
                                name = name[:index] + name[index + 1:]
                                remove_count += 1
                                if remove_count == to_remove:
                                    break
                        if remove_count == to_remove:
                            break

                rng = None
                while name in reverse_dictionary.keys():
                    # if, improbably, removing the vowels hit an existing name
                    # try deleting "random" characters. This has to be
                    # done in a very repeatable fashion, so I use a message
                    # digest to initialize a random # generator and then
                    # rehash the message digest to get the next
                    if rng is None:
                        rng = random_number_generator(starting_name)
                    name = starting_name
                    while len(name) > self.__max_len:
                        index = rng.next() % len(name)
                        name = name[:index] + name[index + 1:]
            reverse_dictionary.pop(orig_name)
            reverse_dictionary[name] = key
            self.__dictionary[key] = name
        self.__mapped = True


def random_number_generator(seed):
    '''This is a very repeatable pseudorandom number generator

    seed - a string to seed the generator

    yields integers in the range 0-65535 on iteration
    '''
    m = hashlib.md5()
    m.update(seed)
    while True:
        digest = m.digest()
        m.update(digest)
        yield ord(digest[0]) + 256 * ord(digest[1])


class SQLiteCommands(object):
    '''This class ducktypes a connection and cursor to aggregate and bulk execute SQL'''

    def __init__(self):
        self.commands_and_bindings = []

    def execute(self, query, bindings=None):
        self.commands_and_bindings.append((query, bindings))

    def commit(self):
        pass

    def close(self):
        del self.commands_and_bindings

    def rollback(self):
        self.commands_and_bindings = []

    def next(self):
        raise NotImplementedError(
                "The SQLite interaction handler can only write to the database")

    def get_state(self):
        return self.commands_and_bindings

    def set_state(self, state):
        self.commands_and_bindings = state

    def execute_all(self, cursor):
        for query, binding in self.commands_and_bindings:
            execute(cursor, query, binding)
