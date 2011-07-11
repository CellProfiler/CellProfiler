'''<b>Export To Database</b> exports data directly to a database, or in 
database readable format, including an imported file
with column names and a CellProfiler Analyst properties file, if desired
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
specify). The Per_Image table consists of all the per-image measurements made during the pipeline, plus
per-image population statistics (such as mean, median, and standard deviation) of the object measurements. There is one
per_image row for every "cycle" that CellProfiler processes (a cycle is usually a single field of view, and a single cycle 
usually contains several image files, each representing a different channel of the same field of view). The Per_Object table contains all the
measurements for individual objects. There is one row of object
measurements per object identified. The two tables are connected with the
primary key column <i>ImageNumber</i>, which indicates the image to which each object belongs. The Per_Object table has another primary
key called <i>ObjectNumber</i>, which is unique to each image. Typically, if multiple types of objects are identified and measured in a pipeline, 
the numbers of those objects are equal to each other. For example, in most pipelines, each nucleus has exactly one cytoplasm, so the first row 
of the Per-Object table contains all of the information about object #1, including both nucleus- and cytoplasm-related measurements. If this 
one-to-one correspondence is <i>not</i> the case for all objects in the pipeline (for example, if dozens of speckles are identified and 
measured for each nucleus), then you must configure <b>ExportToDatabase</b> to export only objects that maintain the one-to-one correspondence 
(for example, export only <i>Nucleus</i> and <i>Cytoplasm</i>, but omit <i>Speckles</i>).

If you have extracted "Plate" and "Well" metadata from image filenames or loaded "Plate" and "Well" metadata via <b>LoadData</b>, 
you can ask CellProfiler to create a "Per_Well" table, which aggregates object measurements across wells.  
This option will output a SQL file (regardless of whether you choose to write directly to the database)
that can be used to create the Per_Well table. At the secure shell where you normally log in to MySQL, type
the following, replacing the italics with references to your database and files:

<tt>mysql -h <i>hostname</i> -u <i>username</i> -p <i>databasename</i> &lt<i>pathtoimages/perwellsetupfile.SQL</i></tt>

The commands written by CellProfiler to create the Per_Well table will be executed.

Oracle is not fully supported at present; you can create your own Oracle DB using
the .csv output option and writing a simple script to upload to the database.

<h4>Available measurements</h4>
For details on the nomenclature used by CellProfiler for the exported measurements,
see <i>Help > General Help > How Measurements Are Named</i>.

See also <b>ExportToSpreadsheet</b>.

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

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
    HAS_MYSQL_DB=True
except:
    logger.warning("MySQL could not be loaded.", exc_info=True)
    HAS_MYSQL_DB=False

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
import cellprofiler.measurements as cpmeas
from cellprofiler.pipeline import GROUP_INDEX
from identify import M_NUMBER_OBJECT_NUMBER
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
PLATE_TYPES = [NONE_CHOICE,"96","384","5600"]
COLOR_ORDER = ["red", "green", "blue", "cyan", "magenta", "yellow", "gray", "none"]
GROUP_COL_DEFAULT = "ImageNumber, Image_Metadata_Plate, Image_Metadata_Well"

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

"""Offset of the image group count in the settings"""
SETTING_IMAGE_GROUP_COUNT = 29

"""Offset of the group specification group count in the settings"""
SETTING_GROUP_FIELD_GROUP_COUNT = 30

"""Offset of the filter specification group count in the settings"""
SETTING_FILTER_FIELD_GROUP_COUNT = 31

"""Offset of the workspace specification group count in the settings"""
SETTING_WORKSPACE_GROUP_COUNT = 32

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

'''Index of the object table format choice in the settings'''
OT_IDX = 17

'''Use this dictionary to keep track of rewording of above if it happens'''
OT_DICTIONARY = {
    "One table per object type": OT_PER_OBJECT,
    "Single object table": OT_COMBINE
    }

def execute(cursor, query, bindings = None, return_result=True):
    if bindings == None:
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
        raise DBException, 'Error retrieving next result from database: %s'%(e)
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
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    return connection, cursor

    
class ExportToDatabase(cpm.CPModule):
 
    module_name = "ExportToDatabase"
    variable_revision_number = 21
    category = ["File Processing","Data Tools"]

    def create_settings(self):
        db_choices = ([DB_MYSQL, DB_MYSQL_CSV, DB_SQLITE] if HAS_MYSQL_DB
                      else [DB_MYSQL_CSV, DB_SQLITE])
        default_db = DB_MYSQL if HAS_MYSQL_DB else DB_MYSQL_CSV
        self.db_type = cps.Choice(
            "Database type",
            db_choices, default_db, doc = """
            What type of database do you want to use? 
            <ul>
            <li><i>MySQL</i> allows the data to be written directly to a MySQL 
            database. MySQL is open-source software; you may require help from 
            your local Information Technology group to set up a database 
            server.</li>
            <li><i>MySQL / CSV</i> writes a script file that
            contains SQL statements for creating a database and uploading the
            Per_Image and Per_Object tables. This option will write out the Per_Image
            and Per_Object table data to two CSV files; you can use these files can be
            used to import the data directly into an application
            that accepts CSV data.</li>
            <li><i>SQLite</i> writes 
            SQLite files directly. SQLite is simpler to set up than MySQL and 
            can more readily be run on your local computer rather than requiring a 
            database server. More information about SQLite can be found at 
            <a href="http://www.sqlite.org/">here</a>. </li>
            </ul>""")
        
        self.db_name = cps.Text(
            "Database name", "DefaultDB",doc = """
            Select a name for the database you want to use""")
        
        self.want_table_prefix = cps.Binary(
            "Add a prefix to table names?", False, doc = """
            Do you want to add a prefix to your table names?
            This option enables you to prepend text to your table names
            (Per_Image and Per_Object).  CellProfiler will warn you before overwriting an existing table.""")
        
        self.table_prefix = cps.Text(
            "Table prefix", "Expt_" , doc = """
            <i>(Used if Add a prefix to table names?</i> is selected)<br>
            What is the table prefix you want to use?""")
        
        self.sql_file_prefix = cps.Text(
            "SQL file prefix", "SQL_", doc = """
            <i>(Used if SQL is selected as the database type and if CSV files are to be written)</i><br>
            What prefix do you want to use to name the SQL file?""")
        
        self.directory = cps.DirectoryPath(
            "Output file location",
            dir_choices = [
                DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME, 
                ABSOLUTE_FOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME,
                DEFAULT_INPUT_SUBFOLDER_NAME],
            doc="""<i>(Used only when using .csv's or a SQLite database, and/or creating a properties or workspace file)</i><br>
            This setting determines where the .csv files or SQLite database is saved if
            you decide to write measurements to files instead of writing them
            directly to the database. If you request a CellProfiler Analyst properties file
            or workspace file, it will also be saved to this location. %(IO_FOLDER_CHOICE_HELP_TEXT)s 
            
            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have a metadata tag named 
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name with the "Plate" metadata tag. 
            The module will substitute the metadata values for the last image set 
            processed for any metadata tags in the folder name. %(USING_METADATA_HELP_REF)s.</p>"""% globals())
        
        self.save_cpa_properties = cps.Binary(
            "Create a CellProfiler Analyst properties file?", 
            False, doc = """
            You can generate a template properties file that will allow you to use your new database with CellProfiler Analyst (a data
            exploration tool which can also be downloaded from
            <a href="http://www.cellprofiler.org/"> http://www.cellprofiler.org/ </a>). 
            The module will attempt to fill in as many as the entries as possible 
            based on the pipeline's settings, including the 
            server name, username and password if MySQL or Oracle is used.""")
        
        #
        # Hack: if user is on Broad IP, then plug in the imageweb url prepend
        #
        import socket
        try:
            ip = socket.gethostbyaddr(socket.gethostname())[-1][0]
        except:
            ip = '127.0.0.1'
        default_prepend = ""
        if ip.startswith('69.173'): # Broad
            default_prepend = "http://imageweb/images/CPALinks"
        self.properties_image_url_prepend = cps.Text(
            "Enter an image url prepend if you plan to access your files via http",
            default_prepend, 
            doc = """<i>(Used only if creating a properties file)</i><br>
            The image paths written to the database will be the absolute
            path the the image files on your computer. If you plan to make these 
            files accessible via the web, you can enter a url prefix here. Eg: 
            If an image is loaded from the path "/cellprofiler/images/" and you use
            a url prepend of "http://mysite.com/", CellProfiler Analyst will look
            for your file at "http://mysite.com/cellprofiler/images/" 
            <p>If you are not using the web to access your files (i.e., they are locally
            aceesible by your computer), leave this setting blank.""")
        
        self.properties_plate_type = cps.Choice("Select the plate type",
            PLATE_TYPES,
            doc="""<i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the plate 
            type here. Supported types in CellProfiler Analyst are 96- and 384-well plates,
            as well as 5600-spot microarrays. If you are not using a plate or microarray, select
            <i>None</i>.""")
        
        self.properties_plate_metadata = cps.Choice("Select the plate metadata",
            ["None"],choices_fn = self.get_metadata_choices,
            doc="""<i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the metadata corresponding
            to the plate here. If there is no plate metadata associated with the image set, select
            <i>None</i>. 
            <p>%(USING_METADATA_HELP_REF)s.</p>"""% globals())
        
        self.properties_well_metadata = cps.Choice("Select the well metadata",
            ["None"],choices_fn = self.get_metadata_choices,
            doc="""<i>(Used only if creating a properties file)</i><br>
            If you are using a multi-well plate or microarray, you can select the metadata corresponding
            to the well here. If there is no well metadata associated with the image set, select
            <i>None</i>. 
            <p>%(USING_METADATA_HELP_REF)s.</p>"""% globals())
        
        self.properties_export_all_image_defaults = cps.Binary(
            "Include information for all images, using default values?", True,
            doc="""<i>(Used only if creating a properties file)</i><br>
            Check this setting to include information in the properties file for all images.
            Leaving this box checked will do the following:
            <ul>
            <li>All images loaded using <b>LoadImages</b>, <b>LoadData</b> or saved in <b>SaveImages</b> will be included.</li>
            <li>The CellProfiler image name will be used for the <i>image_name</i> field.</li>
            <li>A channel color listed in the <i>image_channel_colors</i> field will be assigned to the image by default order.</li>
            </ul>
            Leave this box unchecked to specify which images should be included or to override the automatic values.""")
        
        self.image_groups = []
        self.image_group_count = cps.HiddenCount(self.image_groups,"Properties image group count")
        self.add_image_group(False)
        self.add_image_button = cps.DoSomething("", "Add another image",
                                           self.add_image_group)
        
        self.properties_wants_groups = cps.Binary(
            "Do you want to add group fields?", False,
            doc = """<i>(Used only if creating a properties file)</i><br>
            You can define ways of grouping your image data (for example, when several images represent the same experimental 
            sample), by providing column(s) that identify unique images (the <i>image key</i>) to another set of columns (the <i>group key</i>).
            <p>Grouping is useful, for example, when you want to aggregate counts for each class of object and their scores 
            on a per-group basis (e.g.: per-well) instead of on a per-image basis when scoring with Classifier. It will 
            also provide new options in the Classifier fetch menu so you can fetch objects from images with specific 
            values for the group columns.</p>""")
        
        self.group_field_groups = []
        self.group_field_count = cps.HiddenCount(self.group_field_groups,"Properties group field count")
        self.add_group_field_group(False)
        self.add_group_field_button = cps.DoSomething("", "Add another group",
                                           self.add_group_field_group)
        
        self.properties_wants_filters = cps.Binary(
            "Do you want to add filter fields?", False,doc = 
            """<i>(Used only if creating a properties file)</i><br>
            You can specify a subset of the images in your experiment by defining a <i>filter</i>. Filters are useful, for 
            example, for fetching and scoring objects in Classifier or making graphs using the 
            plotting tools that satisfy a specific metadata contraint. """)
        
        self.create_filters_for_plates = cps.Binary(
            "Automatically create a filter for each plate?",False, doc= """
            <i>(Used only if creating a properties file and specifiying an image data filter)</i><br>
            If you have specified a plate metadata tag, checking this setting will create a set of filters
            in the properties file, one for each plate.""")
        
        self.filter_field_groups = []
        self.filter_field_count = cps.HiddenCount(self.filter_field_groups,"Properties filter field count")
        self.add_filter_field_button = cps.DoSomething("", "Add another filter",
                                           self.add_filter_field_group)
        
        self.create_workspace_file = cps.Binary(
            "Create a CellProfiler Analyst workspace file?", False, doc = """
            You can generate a workspace file for use with 
            CellProfiler Analyst, a data exploration tool which can 
            also be downloaded from <a href="http://www.cellprofiler.org/">
            http://www.cellprofiler.org/</a>. A workspace file allows you 
            to open a selected set of measurements with the display tools
            of your choice. This is useful, for example, if you want examine a standard
            set of quality control image measurements for outliers.""")
        
        self.divider = cps.Divider(line=True)
        self.divider_props = cps.Divider(line=True)
        self.divider_props_wkspace = cps.Divider(line=True)
        self.divider_wkspace = cps.Divider(line=True)
        
        self.workspace_measurement_groups = []
        self.workspace_measurement_count = cps.HiddenCount(self.workspace_measurement_groups, "Workspace measurement count")
        
        def add_workspace_measurement_group(can_remove = True):
            self.add_workspace_measurement_group(can_remove)
            
        add_workspace_measurement_group(False)
        self.add_workspace_measurement_button = cps.DoSomething("", "Add another measurement", self.add_workspace_measurement_group)
        
        self.mysql_not_available = cps.Divider("Cannot write to MySQL directly - CSV file output only", line=False, 
            doc= """The MySQLdb python module could not be loaded.  MySQLdb is necessary for direct export.""")
        
        self.db_host = cps.Text("Database host", "")
        self.db_user = cps.Text("Username", "")
        self.db_passwd = cps.Text("Password", "")
        self.sqlite_file = cps.Text("Name the SQLite database file", 
            "DefaultDB.db", doc = """
            <i>(Used if SQLite selected as database type)</i><br>
            What is the SQLite database filename to which you want to write?""")
        
        self.wants_agg_mean = cps.Binary("Calculate the per-image mean values of object measurements?", True, doc = """
            <b>ExportToDatabase</b> can calculate population statistics over all the objects in each image
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
                     WHERE Per_Image.ImageNumber = Per_Object.ImageNumber);</tt>""")
        self.wants_agg_median = cps.Binary("Calculate the per-image median values of object measurements?", False)
        self.wants_agg_std_dev = cps.Binary("Calculate the per-image standard deviation values of object measurements?", False)
        
        self.wants_agg_mean_well = cps.Binary(
            "Calculate the per-well mean values of object measurements?", False, doc = '''
            <b>ExportToDatabase</b> can calculate statistics over all the objects in each well 
            and store the results as columns in a "per-well" table in the database. For instance, 
            if you are measuring the area of the Nuclei objects and you check the aggregate
            mean box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_avg", with a column called "Mean_Nuclei_AreaShape_Area". Selecting all three aggregate measurements will create three per-well tables, one for each of the measurements.

            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be run on your Per-Image and Per-Object tables to create the desired per-well table. 
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename or via a <b>LoadData</b> module.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %s'''% USING_METADATA_HELP_REF)
        
        self.wants_agg_median_well = cps.Binary(
            "Calculate the per-well median values of object measurements?", False, doc = '''
            <b>ExportToDatabase</b> can calculate statistics over all the objects in each well 
            and store the results as columns in a "per-well" table in the database. For instance, 
            if you are measuring the area of the Nuclei objects and you check the aggregate
            median box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_median", with a column called "Median_Nuclei_AreaShape_Area". Selecting all three aggregate measurements will create three per-well tables, one for each of the measurements.

            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be run on your Per-Image and Per-Object tables to create the desired per-well table. 
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename or via a <b>LoadData</b> module.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %s'''% USING_METADATA_HELP_REF)
        
        self.wants_agg_std_dev_well = cps.Binary(
            "Calculate the per-well standard deviation values of object measurements?", False, doc = '''
            <b>ExportToDatabase</b> can calculate statistics over all the objects in each well 
            and store the results as columns in a "per-well" table in the database. For instance, 
            if you are measuring the area of the Nuclei objects and you check the aggregate
            standard deviation box in this module, <b>ExportToDatabase</b> will create a table in the database called
            "Per_Well_std", with a column called "Mean_Nuclei_AreaShape_Area".  Selecting all three aggregate measurements will create three per-well tables, one for each of the measurements.
            <p>The per-well functionality will create the appropriate lines in a .SQL file, which can be run on your Per-Image and Per-Object tables to create the desired per-well table. 
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename or via a <b>LoadData</b> module.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %s'''% USING_METADATA_HELP_REF)
        
        self.objects_choice = cps.Choice(
            "Export measurements for all objects to the database?",
            [O_ALL, O_NONE, O_SELECT], doc = """
            This option lets you choose the objects whose measurements will be saved in the Per_Object and Per_Well(s) database tables.
            <ul>
            <li><i>All:</i> Export measurements from all objects.</li>
            <li><i>None:</i> Do not export data to a Per_Object table. Save only Per_Image or Per_Well measurements (which nonetheless include population statistics from objects).</li>
            <li><i>Select:</i> Select the objects you want to export from a list.</li>
            </ul>""")
        
        self.objects_list = cps.ObjectSubscriberMultiChoice(
            "Select the objects", doc = """
            <i>(Used only if Select is chosen for adding objects)</i><br>
            Choose one or more objects from this list (click using shift or command keys to select multiple objects). The list includes
            the objects that were created by prior modules. If you choose an
            object, its measurements will be written out to the Per_Object and/or
            Per_Well(s) tables, otherwise, the object's measurements will be skipped.""")
        
        self.max_column_size = cps.Integer(
            "Maximum # of characters in a column name", 64, 
            minval = 10, maxval = 64,
            doc="""This setting limits the number of characters that can appear
            in the name of a field in the database. MySQL has a limit of 64
            characters per field, but also has an overall limit on the number of characters
            in all of the columns of a table. <b>ExportToDatabase</b> will
            shorten all of the column names by removing characters, at the
            same time guaranteeing that no two columns have the same name.""")
        
        self.separate_object_tables = cps.Choice(
            "Create one table per object or a single object table?",
            [OT_COMBINE, OT_PER_OBJECT],
            doc = """<b>ExportToDatabase</b> can create either one table
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
            database table that records all object measurements. <b>
            ExportToDatabase</b> will prepend each column name with the
            name of the object associated with that column's measurement.
            Each row of the table will have measurements for all objects
            that have the same image and object number. Choose
            <i>%(OT_COMBINE)s</i> if parent objects have a single child,
            or if you want a simple table structure in your database.</li>
            </ul>""" % globals())
        
        self.want_image_thumbnails = cps.Binary(
            "Write image thumbnails directly to the database?", False, doc = """
            <i>(Used only if MySQL is selected as database type)</i><br>
            Check this option if you'd like to write image thumbnails directly
            into the database. This will slow down the writing step, but will
            enable new functionality in CellProfiler Analyst such as quickly
            viewing images in the Plate Viewer tool by selecting "thumbnail"
            from the "Well display" dropdown.""")
        
        self.thumbnail_image_names = cps.ImageNameSubscriberMultiChoice(
            "Select the images you want to save thumbnails of",
            doc = """
            <i>(Used only if MySQL is selected as database type and writing thumbnails is selected)</i><br>
            Select the images that you wish to save as thumbnails to 
            the database.""")
        
        self.auto_scale_thumbnail_intensities = cps.Binary(
            "Auto-scale thumbnail pixel intensities?", True,
            doc = """
            <i>(Used only if MySQL is selected as database type and writing thumbnails is selected)</i><br>
            Check this option if you'd like to automatically rescale 
            the thumbnail pixel intensities to the range 0-1, where 0 is 
            black/unsaturated, and 1 is white/saturated.""")
        
    def add_image_group(self,can_remove = True):
        group = cps.SettingsGroup()
        
        group.can_remove = can_remove
        
        group.append(
            "image_cols", cps.Choice("Select an image to include",["None"],choices_fn = self.get_property_file_image_choices, doc="""
            <i>(Used only if creating a properties file and specifiying the image information)</i><br>
            Choose image name to include it in the properties file of images."""))
        
        group.append(
            "wants_automatic_image_name", cps.Binary(
            "Use the image name for the display?", True, doc=
            """<i>(Used only if creating a properties file and specifiying the image information)</i><br>
            Use the image name as given above for the displayed name. You can name
            the file yourself if you leave this box unchecked."""))

        group.append(
            "image_name", cps.Text(
            "Image name", "Channel%d"%(len(self.image_groups)+1), doc=
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
                             
    def add_group_field_group(self,can_remove = True):
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        group.append(
            "group_name",cps.Text(
            "Enter the name of the group",'',doc="""
            <i>(Used only if creating a properties file and specifiying an image data group)</i><br>
            Enter a name for the group. Only alphanumeric characters and underscores are permitted."""))
        group.append(
            "group_statement", cps.Text(
            "Enter the per-image columns which define the group, separated by commas",GROUP_COL_DEFAULT, doc="""
            <i>(Used only if creating a properties file and specifiying an image data group)</i><br>
            To define a group, enter the image key columns followed by group key columns, each separated by commas.
            <p>In CellProfiler, the image key column is always given the name as <i>ImageNumber</i>; group keys
            are typically metadata columns which are always prefixed with <i>Image_Metadata_</i>. For example, if you 
            wanted to be able to group your data by unique plate names and well identifiers, you could define a 
            group as follows:<br>
            <code>ImageNumber, Image_Metadata_Plate, Image_Metadata_Well</code><br>
            <p>Groups are specified as MySQL statements in the properties file, but please note that the full SELECT and  
            FROM clauses will be added automatically, so there is no need to enter them here.</p>"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove this group", self.group_field_groups, group))
        group.append("divider", cps.Divider(line=True))
        
        self.group_field_groups.append(group)
        
    def add_filter_field_group(self,can_remove = True):
        group = cps.SettingsGroup()
        
        group.can_remove = can_remove
        
        group.append(
            "filter_name",cps.Text(
            "Enter the name of the filter",'',doc="""
            <i>(Used only if creating a properties file and specifiying an image data filter)</i><br>
            Enter a name for the filter. Only alphanumeric characters and underscores are permitted."""))
        group.append(
            "filter_statement", cps.Text(
            "Enter the MySQL WHERE clause to define a filter",'',doc="""
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
        
    def add_workspace_measurement_group(self, can_remove = True):
        group = cps.SettingsGroup()
        self.workspace_measurement_groups.append(group)
        
        group.can_remove = can_remove
        
        group.append("divider", cps.Divider(line=False))
        
        group.append(
            "measurement_display", cps.Choice(
            "Select the measurement display tool",
            W_DISPLAY_ALL, doc="""
            <i>(Used only when Create workspace file is checked)</i><br>
            Select what display tool in CPA you want to use to open the 
            measurements.
            <ul>
            <li>%(W_SCATTERPLOT)s</li>
            <li>%(W_HISTOGRAM)s</li>
            <li>%(W_DENSITYPLOT)s</li>
            <li>%(W_PLATEVIEWER)s</li>
            <li>%(W_BOXPLOT)s</li>
            </ul>"""%globals()))
        
        def measurement_type_help():
            return """
                <i>(Used only when Create workspace file is checked)</i><br>
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
                </ul>"""%globals()
        
        def object_name_help():
            return """<i>(Used only when Create workspace file is checked)</i><br>
                Select the object that you want to measure from the list.
                This should be an object created by a previous module such as
                <b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b>, or
                <b>IdentifyTertiaryObjects</b>."""
            
        def measurement_name_help():
            return """<i>(Used only when Create workspace file is checked)</i><br>
            Select the measurement to be plotted on the desired axis."""
        
        def index_name_help():
            return """<i>(Used only when Create workspace file is checked and an index is plotted)</i><br>
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
            </ul>"""%globals()
            
        group.append(
            "x_measurement_type", cps.Choice(
            "Type of measurement to plot on the x-axis",
            W_TYPE_ALL, doc = measurement_type_help()))

        group.append(
            "x_object_name", cps.ObjectNameSubscriber(
            "Enter the object name","None",
            doc = object_name_help()))
        
        def object_fn_x():
            if group.x_measurement_type.value in ( cpmeas.IMAGE, cpmeas.EXPERIMENT ):
                return group.x_measurement_type.value
            elif group.x_measurement_type.value == cpmeas.OBJECT:
                return group.x_object_name.value
            else:
                raise NotImplementedError("Measurement type %s is not supported"%
                                              group.x_measurement_type.value)
                
        group.append(
            "x_measurement_name", cps.Measurement(
            "Select the x-axis measurement", object_fn_x,
            doc = measurement_name_help()))
        
        group.append(
            "x_index_name", cps.Choice(
            "Select the x-axis index", W_INDEX_ALL,
            doc = index_name_help()))
    
        group.append(
            "y_measurement_type", cps.Choice(
            "Type of measurement to plot on the y-axis",
            W_TYPE_ALL, doc = measurement_type_help()))

        group.append(
            "y_object_name", cps.ObjectNameSubscriber(
            "Enter the object name","None",
            doc=object_name_help()))
        
        def object_fn_y():
            if group.y_measurement_type.value == cpmeas.IMAGE:
                return cpmeas.IMAGE
            elif group.y_measurement_type.value == cpmeas.OBJECT:
                return group.y_object_name.value
            else:
                raise NotImplementedError("Measurement type %s is not supported"%
                                              group.y_measurement_type.value)
            
        group.append(
            "y_measurement_name", cps.Measurement(
            "Select the y-axis measurement", object_fn_y, 
            doc = measurement_name_help()))
        
        group.append(
            "y_index_name", cps.Choice(
            "Select the x-axis index", W_INDEX_ALL,
            doc = index_name_help()))
        
        if can_remove:
            group.append("remove_button", cps.RemoveSettingButton(
                "", "Remove this measurement", self.workspace_measurement_groups, group))
            
    def get_metadata_choices(self,pipeline):
        columns = pipeline.get_measurement_columns()
        choices = ["None"]
        for column in columns:
            object_name, feature, coltype = column[:3]
            choice = feature[(len(cpmeas.C_METADATA)+1):]
            if (object_name == cpmeas.IMAGE and
                feature.startswith(cpmeas.C_METADATA)):
                choices.append(choice)
        return choices
    
    def get_property_file_image_choices(self,pipeline):
        columns = pipeline.get_measurement_columns()
        image_names = []
        for column in columns:
            object_name, feature, coltype = column[:3]
            choice = feature[(len(C_FILE_NAME)+1):]
            if (object_name == cpmeas.IMAGE and (feature.startswith(C_FILE_NAME))):
                image_names.append(choice)
        return image_names
    
    def prepare_settings(self, setting_values):
        # These check the groupings of settings avilable in properties and workspace file creation
        for count, sequence, fn in\
            ((int(setting_values[SETTING_IMAGE_GROUP_COUNT]), self.image_groups, self.add_image_group),
             (int(setting_values[SETTING_GROUP_FIELD_GROUP_COUNT]), self.group_field_groups, self.add_group_field_group),
             (int(setting_values[SETTING_FILTER_FIELD_GROUP_COUNT]), self.filter_field_groups, self.add_filter_field_group),
             (int(setting_values[SETTING_WORKSPACE_GROUP_COUNT]), self.workspace_measurement_groups, self.add_workspace_measurement_group)):
            del sequence[count:]
            while len(sequence) < count:
                fn()
            
    def visible_settings(self):
        needs_default_output_directory =\
            (self.db_type != DB_MYSQL or
             self.save_cpa_properties.value or
             self.create_workspace_file.value)
        result = [self.db_type]
        if not HAS_MYSQL_DB:
            result += [self.mysql_not_available]
        if self.db_type == DB_MYSQL:
                result += [self.db_name]
                result += [self.db_host]
                result += [self.db_user]
                result += [self.db_passwd]
        elif self.db_type == DB_MYSQL_CSV:
            result += [self.sql_file_prefix]
            result += [self.db_name]
        elif self.db_type == DB_SQLITE:
            result += [self.sqlite_file]
        elif self.db_type == DB_ORACLE:
            result += [self.sql_file_prefix]
        result += [self.want_table_prefix]
        if self.want_table_prefix.value:
            result += [self.table_prefix]
        if self.save_cpa_properties.value:
            result += [self.divider_props] # Put divider here to make things easier to read
        result += [self.save_cpa_properties]
        if self.save_cpa_properties.value:
            result += [self.properties_image_url_prepend, self.properties_plate_type, 
                       self.properties_plate_metadata, self.properties_well_metadata,
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
                result += [ self.add_image_button]
            result += [self.properties_wants_groups]
            if self.properties_wants_groups:
                for group in self.group_field_groups:
                    if group.can_remove:
                        result += [group.divider]
                    result += [group.group_name, group.group_statement]
                    if group.can_remove:
                        result += [group.remover]
                result += [ self.add_group_field_button ]
            result += [self.properties_wants_filters]
            if self.properties_wants_filters:
                result += [self.create_filters_for_plates]
                for group in self.filter_field_groups:
                    result += [group.filter_name, group.filter_statement]
                    if group.can_remove:
                        result += [group.remover]
                    result += [group.divider]
                result += [ self.add_filter_field_button ]
        
        if self.save_cpa_properties.value or self.create_workspace_file.value : # Put divider here to make things easier to read
            result += [self.divider_props_wkspace] 
            
        result += [self.create_workspace_file]
        if self.create_workspace_file:
            for workspace_group in self.workspace_measurement_groups:
                result += self.workspace_visible_settings(workspace_group)
                if workspace_group.can_remove:
                    result += [workspace_group.remove_button]
            result += [self.add_workspace_measurement_button]

        if self.create_workspace_file.value: # Put divider here to make things easier to read
            result += [self.divider_wkspace] 
            
        if needs_default_output_directory:
            result += [self.directory]
            
        result += [self.wants_agg_mean, self.wants_agg_median,
                   self.wants_agg_std_dev]
        if self.db_type != DB_SQLITE:
            # We don't write per-well tables to SQLite yet.
            result += [self.wants_agg_mean_well, self.wants_agg_median_well, 
                       self.wants_agg_std_dev_well]
        result += [self.objects_choice]
        if self.objects_choice == O_SELECT:
            result += [self.objects_list]
        if self.objects_choice != O_NONE:
            result += [self.separate_object_tables]
        result += [self.max_column_size]
        if self.db_type in (DB_MYSQL, DB_SQLITE):
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
                self.want_image_thumbnails,self.thumbnail_image_names, 
                self.auto_scale_thumbnail_intensities,self.properties_plate_type,
                self.properties_plate_metadata, self.properties_well_metadata, 
                self.properties_export_all_image_defaults,
                self.image_group_count, self.group_field_count, self.filter_field_count,
                self.workspace_measurement_count]
        
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
        result += [ self.create_workspace_file ]
        for group in self.workspace_measurement_groups:
            result += [ group.measurement_display, 
                        group.x_measurement_type, group.x_object_name, group.x_measurement_name, group.x_index_name,
                        group.y_measurement_type, group.y_object_name, group.y_measurement_name, group.y_index_name]
        
        return result
    
    def help_settings(self):
        return [self.db_type, self.db_name, self.db_host, self.db_user, self.db_passwd, self.sql_file_prefix, self.sqlite_file, 
                self.want_table_prefix, self.table_prefix,  
                self.save_cpa_properties, self.properties_image_url_prepend, 
                self.properties_plate_type, self.properties_plate_metadata, self.properties_well_metadata,
                self.properties_export_all_image_defaults,
                self.image_groups[0].image_cols, self.image_groups[0].wants_automatic_image_name, self.image_groups[0].image_name,
                self.image_groups[0].image_channel_colors,
                self.properties_wants_groups, 
                self.group_field_groups[0].group_name, self.group_field_groups[0].group_statement,
                self.properties_wants_filters, self.create_filters_for_plates,
                self.directory,
                self.create_workspace_file, 
                self.workspace_measurement_groups[0].measurement_display, 
                self.workspace_measurement_groups[0].x_measurement_type, self.workspace_measurement_groups[0].x_object_name, self.workspace_measurement_groups[0].x_measurement_name, 
                self.workspace_measurement_groups[0].y_measurement_type, self.workspace_measurement_groups[0].y_object_name, self.workspace_measurement_groups[0].y_measurement_name,
                self.wants_agg_mean, self.wants_agg_median, self.wants_agg_std_dev, 
                self.wants_agg_mean_well, self.wants_agg_median_well, self.wants_agg_std_dev_well,
                self.objects_choice, self.objects_list,
                self.separate_object_tables,
                self.max_column_size,
                self.want_image_thumbnails,self.thumbnail_image_names, self.auto_scale_thumbnail_intensities]
    
    def validate_module(self,pipeline):
        if self.want_table_prefix.value:
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$",self.table_prefix.value):
                raise cps.ValidationError("Invalid table prefix",self.table_prefix)

        if self.db_type==DB_MYSQL:
            if not re.match("^[A-Za-z0-9_]+$",self.db_name.value):
                raise cps.ValidationError("The database name has invalid characters",self.db_name)
        elif self.db_type==DB_SQLITE:
            if not re.match("^[A-Za-z0-9_].*$",self.sqlite_file.value):
                raise cps.ValidationError("The sqlite file name has invalid characters",self.sqlite_file)

        if self.db_type == DB_MYSQL:
            if not re.match("^[A-Za-z0-9_].*$",self.db_host.value):
                raise cps.ValidationError("The database host name has invalid characters",self.db_host)
            if not re.match("^[A-Za-z0-9_]+$",self.db_user.value):
                raise cps.ValidationError("The database user name has invalid characters",self.db_user)
        else:
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$", self.sql_file_prefix.value):
                raise cps.ValidationError("Invalid SQL file prefix", self.sql_file_prefix)
        
        if self.objects_choice == O_SELECT:
            self.objects_list.load_choices(pipeline)
            if len(self.objects_list.choices) == 0:
                raise cps.ValidationError("Please choose at least one object", self.objects_choice)
            
        if self.save_cpa_properties:
            if self.properties_plate_metadata == NONE_CHOICE and (self.properties_wants_filters.value and self.create_filters_for_plates.value):
                raise cps.ValidationError("You must specify the plate metadata",self.create_filters_for_plates)

        if self.want_image_thumbnails:
            if not self.thumbnail_image_names.get_selections():
                raise cps.ValidationError("Please choose at least one image", self.thumbnail_image_names)
            
    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToDatabase does not produce output in Test Mode",
                                      self.db_type)
        
        '''Warn user that they will have to merge tables to use CPA'''
        if self.objects_choice != O_NONE and self.separate_object_tables == OT_PER_OBJECT:
            raise cps.ValidationError(
                ("You will have to merge the separate object tables in order\n"
                 "to use CellProfiler Analyst fully, or you will be restricted \n"
                 "to only one object's data at a time in CPA. Choose %s to write a single\n"
                 "object table.") % OT_COMBINE, self.separate_object_tables)
                
        '''Warn user re: bad characters in filter/group names'''
        if self.save_cpa_properties and self.properties_wants_groups:
            for group in self.group_field_groups:
                if not re.match("^[A-Za-z0-9_]*$",group.group_name.value) or group.group_name.value == '':
                    raise cps.ValidationError("CellProfiler Analyst will not recognize this group name because it has invalid characters.",group.group_name)
        if self.save_cpa_properties and self.properties_wants_filters:
            for group in self.filter_field_groups:
                if not re.match("^[A-Za-z0-9_]*$",group.filter_name.value) or group.filter_name.value == '':
                    raise cps.ValidationError("CellProfiler Analyst will not recognize this filter name because it has invalid characters.",group.filter_name)
                if not re.match("^[A-Za-z0-9_]*$",group.filter_statement.value) or group.filter_statement.value == '':
                    raise cps.ValidationError("CellProfiler Analyst will not recognize this filter statement because it has invalid characters.",group.filter_statement)

    def make_full_filename(self, file_name, 
                           workspace = None, image_set_index = None):
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
        return os.path.join(path,file)
    
    def prepare_run(self, workspace):
        '''Prepare to run the pipeline
        Establish a connection to the database.'''

        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        
        if pipeline.test_mode:
            return True

        if self.db_type==DB_MYSQL:
            self.connection, self.cursor = connect_mysql(self.db_host.value, 
                                                         self.db_user.value, 
                                                         self.db_passwd.value,
                                                         self.db_name.value)
            if self.wants_well_tables:
                per_well = self.write_mysql_table_per_well(pipeline, image_set_list)
        elif self.db_type==DB_SQLITE:
            db_file = self.make_full_filename(self.sqlite_file.value)
            self.connection, self.cursor = connect_sqlite(db_file)
        #
        # This caches the list of measurement columns for the run,
        # fixing the column order, etc.
        #
        self.get_pipeline_measurement_columns(pipeline, image_set_list)
        
        if pipeline.in_batch_mode():
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
                                'SELECT * FROM %s LIMIT 1'%(table))
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
                    logger.warning("%s already in database, not creating" , table_msg)
                    return True
                import wx
                dlg = wx.MessageDialog(
                    frame, 
                    'ExportToDatabase will overwrite the %s. OK?' % table_msg,
                                    'Overwrite tables?', 
                                    style=wx.OK|wx.CANCEL|wx.ICON_QUESTION)
                if dlg.ShowModal() != wx.ID_OK:
                    dlg.Destroy()
                    return False
                dlg.Destroy()

            mappings = self.get_column_name_mappings(pipeline, image_set_list)
            column_defs = self.get_pipeline_measurement_columns(pipeline, 
                                                                image_set_list)
            if self.objects_choice != O_ALL:
                onames = [cpmeas.EXPERIMENT, cpmeas.IMAGE, cpmeas.NEIGHBORS]
                if self.objects_choice == O_SELECT:
                    onames += self.objects_list.selections
                column_defs = [column for column in column_defs
                               if column[0] in onames]
            self.create_database_tables(self.cursor, pipeline, image_set_list)
        return True
    
    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Initialize for writing post-group measurements'''
        d = self.get_dictionary(image_set_list)
        d[D_IMAGE_SET_INDEX] = []
        d[C_IMAGE_NUMBER] = []

    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''Alter the output directory path for the remote batch host'''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def get_measurement_columns(self, pipeline):
        if self.want_image_thumbnails:
            cols = []
            for name in self.thumbnail_image_names.get_selections():
                cols += [(cpmeas.IMAGE, "Thumbnail_%s"%(name), cpmeas.COLTYPE_LONGBLOB)]
            return cols
        return []
            
    def run_as_data_tool(self, workspace):
        '''Run the module as a data tool
        
        ExportToDatabase has two modes - writing CSVs and writing directly.
        We write CSVs in post_run. We write directly in run.
        '''
        if not self.prepare_run(workspace):
            return
        self.prepare_group(workspace.pipeline,
                           workspace.image_set_list, None, None)
        if self.db_type != DB_MYSQL_CSV:
            workspace.measurements.is_first_image = True
            #
            # Modify the columns so that none get written post_group
            #
            columns = self.get_pipeline_measurement_columns(
                workspace.pipeline, workspace.image_set_list)
            for i in range(len(columns)):
                column = columns[i]
                if self.should_write(column, True):
                    column[3][cpmeas.MCA_AVAILABLE_POST_GROUP] = False
                    
            for i in range(workspace.measurements.image_set_count):
                if i > 0:
                    workspace.measurements.next_image_set()
                self.run(workspace)
        else:
            workspace.measurements.image_set_number = \
                     workspace.measurements.image_set_count
        self.post_run(workspace)
    
    def run(self, workspace):
        if self.db_type in (DB_MYSQL, DB_SQLITE) and self.want_image_thumbnails:
            import Image
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
                    raise Exception('ExportToDatabase cannot write image thumbnails from images of type "%s".'%(str(pixels.dtype)))
                if pixels.ndim == 2:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'L')
                elif pixels.ndim == 3:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'RGB')
                else:
                    raise Exception('ExportToDatabase only supports saving thumbnails of grayscale or 3-channel images. "%s" was neither.'%(name))

                # resize the image so the major axis is 200px long
                if im.size[0] == max(im.size):
                    w, h = (200, 200 * min(im.size) / max(im.size))
                else:
                    h, w = (200, 200 * min(im.size) / max(im.size))
                im = im.resize((w,h))

                fd = StringIO()
                im.save(fd, 'PNG')
                blob = fd.getvalue()
                fd.close()
                measurements.add_image_measurement('Thumbnail_%s'%(name), blob.encode('base64'))
        if workspace.pipeline.test_mode:
            return
        if (self.db_type == DB_MYSQL or self.db_type == DB_SQLITE):
            if not workspace.pipeline.test_mode:
                d = self.get_dictionary(workspace.image_set_list)
                d[D_IMAGE_SET_INDEX].append(workspace.measurements.image_set_number)
                d[C_IMAGE_NUMBER].append(workspace.measurements.image_set_number)
                self.write_data_to_db(workspace)

    def post_group(self, workspace, grouping):
        '''Write out any columns that are only available post-group'''
        if workspace.pipeline.test_mode:
            return
        
        if self.db_type not in (DB_MYSQL, DB_SQLITE):
            return
        
        d = self.get_dictionary(workspace.image_set_list)
        for image_set_index, image_number in zip(d[D_IMAGE_SET_INDEX], 
                                                 d[C_IMAGE_NUMBER]):
            self.write_data_to_db(workspace,
                                  post_group = True,
                                  index = image_set_index,
                                  image_number = image_number)
        
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
        if self.db_type in (DB_MYSQL, DB_SQLITE):
            self.connection.commit()

    
    @property
    def wants_well_tables(self):
        '''Return true if user wants any well tables'''
        if self.db_type == DB_SQLITE:
            return False
        else:
            return (self.wants_agg_mean_well or self.wants_agg_median_well or
                    self.wants_agg_std_dev_well)

    
    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True

    
    def ignore_object(self,object_name, strict = False):
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
                       strict = False):
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
            mappings.add("%s_%s"%(object_name,feature_name))
            if object_name != cpmeas.IMAGE:
                for agg_name in self.agg_names:
                    mappings.add('%s_%s_%s'%(agg_name, object_name, feature_name))
        return mappings
    
    def get_aggregate_columns(self, pipeline, image_set_list, post_group = None):
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
                if (obname==ob_table and 
                    (not self.ignore_feature(obname, feature)) and
                    (not cpmeas.agg_ignore_feature(feature))):
                    feature_name = '%s_%s'%(obname, feature)
                    # create per_image aggregate column defs 
                    result += [(obname, feature, aggname,
                                '%s_%s' % (aggname, feature_name))
                               for aggname in self.agg_names ]
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
        return [ obname for obname in obnames
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
    def create_database_tables(self, cursor, pipeline, image_set_list):
        '''Creates empty image and object tables
        
        Creates the MySQL database (if MySQL), drops existing tables of the
        same name and creates the tables.
        
        cursor - database cursor for creating the tables
        column_defs - column definitions as returned by get_measurement_columns
        mappings - mappings from measurement feature names to column names
        '''
        # Create the database
        if self.db_type==DB_MYSQL:
            #result = execute(cursor, "SHOW DATABASES LIKE '%s'" % 
                             #self.db_name.value)
            #if len(result) == 0:
            execute(cursor, 'CREATE DATABASE IF NOT EXISTS %s' % 
                    (self.db_name.value), return_result = False)
            execute(cursor, 'USE %s'% self.db_name.value, 
                    return_result = False)

        columns = self.get_pipeline_measurement_columns(pipeline, 
                                                        image_set_list)
            
        if self.objects_choice != O_NONE:
            # Object table
            if self.separate_object_tables == OT_COMBINE:
                execute(cursor, 'DROP TABLE IF EXISTS %s' %
                        self.get_table_name(cpmeas.OBJECT), 
                        return_result = False)
                statement = self.get_create_object_table_statement(
                    None, pipeline, image_set_list)
                execute(cursor, statement)
            else:
                for object_name in self.get_object_names(pipeline, 
                                                         image_set_list):
                    execute(cursor, 'DROP TABLE IF EXISTS %s' %
                            self.get_table_name(object_name), 
                            return_result = False)
                    statement = self.get_create_object_table_statement(
                        object_name, pipeline, image_set_list)
                    execute(cursor, statement)
        # Image table

        execute(cursor, 'DROP TABLE IF EXISTS %s' % 
                self.get_table_name(cpmeas.IMAGE), return_result = False)
        statement = self.get_create_image_table_statement(pipeline, 
                                                          image_set_list)
        execute(cursor, statement)
        cursor.connection.commit()
    
    def get_create_image_table_statement(self, pipeline, image_set_list):
        '''Return a SQL statement that generates the image table'''
        statement = 'CREATE TABLE '+ self.get_table_name(cpmeas.IMAGE) +' (\n'
        statement += '%s INTEGER'%C_IMAGE_NUMBER

        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        columns = self.get_pipeline_measurement_columns(
            pipeline, image_set_list)
        for column in columns:
            obname, feature, ftype = column[:3]
            if obname == cpmeas.IMAGE and not self.ignore_feature(obname, feature):
                feature_name = '%s_%s' % (obname, feature)
                statement += ',\n%s %s'%(mappings[feature_name], ftype)
        for column in self.get_aggregate_columns(pipeline, image_set_list):
            statement += ',\n%s %s' % (mappings[column[3]], 
                                       cpmeas.COLTYPE_FLOAT)
        statement += ',\nPRIMARY KEY (%s) )'%C_IMAGE_NUMBER
        return statement
        
    def get_create_object_table_statement(self, object_name, pipeline, 
                                          image_set_list):
        '''Get the "CREATE TABLE" statement for the given object table
        
        object_name - None = PerObject, otherwise a specific table
        '''
        if object_name == None:
            object_table = self.get_table_name(cpmeas.OBJECT)
        else:
            object_table = self.get_table_name(object_name)
        statement = 'CREATE TABLE '+object_table+' (\n'
        statement += '%s INTEGER\n'%C_IMAGE_NUMBER
        if object_name == None:
            statement += ',%s INTEGER'%C_OBJECT_NUMBER
            object_pk = C_OBJECT_NUMBER
        else:
            object_pk = "_".join((object_name,M_NUMBER_OBJECT_NUMBER))
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
                if obname==ob_table and not self.ignore_feature(obname, feature):
                    feature_name = '%s_%s'%(obname, feature)
                    statement += ',\n%s %s'%(mappings[feature_name], ftype)
        statement += ',\nPRIMARY KEY (%s, %s) )' %(C_IMAGE_NUMBER, object_pk)
        return statement

        
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
        file_name = "%sSETUP.SQL"%(self.sql_file_prefix)
        path_name = self.make_full_filename(file_name,workspace)
        fid = open(path_name,"wt")
        fid.write("CREATE DATABASE IF NOT EXISTS %s;\n"%(self.db_name.value))
        fid.write("USE %s;\n"%(self.db_name.value))
        fid.write(self.get_create_image_table_statement(pipeline, 
                                                        image_set_list) + ";\n")
        #
        # Write out the per-object table
        #
        if self.objects_choice != O_NONE:
            if self.separate_object_tables == OT_COMBINE:
                data = [(None, cpmeas.OBJECT)]
            else:
                data = [ (x, x) for x in self.get_object_names(
                    pipeline, image_set_list)]
            
            for gcot_name, object_name in data:
                fid.write(self.get_create_object_table_statement(
                    gcot_name, pipeline, image_set_list) + ";\n")
        else:
            data = []
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
            file_name = "%s_Per_Well_SETUP.SQL"%(self.sql_file_prefix)
            path_name = self.make_full_filename(file_name)
            fid = open(path_name,"wt")
            needs_close = True
        else:
            needs_close = False
        fid.write("USE %s;\n"%(self.db_name.value))
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
            for do_mapping, do_write in ((True, False),(False, True)):
                if do_write:
                    fid.write("CREATE TABLE %sPer_Well_%s AS SELECT " %
                              (self.get_table_prefix(), aggname))
                for i, object_name in enumerate(object_names + [cpmeas.IMAGE]):
                    if object_name == cpmeas.IMAGE:
                        object_table_name = "IT"
                    elif self.separate_object_tables == OT_COMBINE:
                        object_table_name = "OT"
                    else:
                        object_table_name = "OT%d" % (i+1)
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
                        feature_name = "%s_%s"%(object_name,feature)
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
                          (self.get_table_name(cpmeas.OBJECT),C_IMAGE_NUMBER,C_IMAGE_NUMBER))
            elif len(object_names) == 1:
                fid.write("JOIN %s OT1 ON IT.%s = OT1.%s\n" %
                          (self.get_table_name(object_names[0]),C_IMAGE_NUMBER,C_IMAGE_NUMBER))
            else:
                #
                # We make up a table here that lists all of the possible
                # image and object numbers from any of the object numbers.
                # We need this to do something other than a cartesian join
                # between object tables.
                #
                fid.write(
                    "RIGHT JOIN (SELECT DISTINCT %s, %s FROM\n"%(C_IMAGE_NUMBER, C_OBJECT_NUMBER))
                fid.write("(SELECT %s, %s_%s as %s FROM %s\n" %
                          (C_IMAGE_NUMBER, object_names[0], M_NUMBER_OBJECT_NUMBER, C_OBJECT_NUMBER,
                           self.get_table_name(object_names[0])))
                for object_name in object_names[1:]:
                    fid.write("UNION SELECT %s, %s_%s as %s "
                              "FROM %s\n" % 
                              (C_IMAGE_NUMBER, object_name, M_NUMBER_OBJECT_NUMBER, C_OBJECT_NUMBER,
                               self.get_table_name(object_name)))
                fid.write(") N_INNER) N ON IT.%s = N.%s\n"%(C_IMAGE_NUMBER, C_IMAGE_NUMBER))
                for i, object_name in enumerate(object_names):
                    fid.write("LEFT JOIN %s OT%d " % 
                              (self.get_table_name(object_name), i+1))
                    fid.write("ON N.%s = OT%d.%s " % (C_IMAGE_NUMBER, i+1, C_IMAGE_NUMBER))
                    fid.write("AND N.%s = OT%d.%s_%s\n" %
                              (C_OBJECT_NUMBER, i+1, object_name, M_NUMBER_OBJECT_NUMBER))
            fid.write("GROUP BY IT.Image_Metadata_Plate, "
                      "IT.Image_Metadata_Well;\n\n""")
                
        if needs_close:
            fid.close()

    
    def write_oracle_table_defs(self, workspace):
        raise NotImplementedError("Writing to an Oracle database is not yet supported")

    
    def base_name(self,workspace):
        """The base for the output file name"""
        m = workspace.measurements
        first = m.image_set_start_number
        last = m.image_set_number
        return '%s%d_%d'%(self.sql_file_prefix, first, last)

    
        
    def write_csv_data(self, workspace):
        """Write the data in the measurements out to the csv files
        workspace - contains the measurements
        """
        zeros_for_nan = False
        measurements = workspace.measurements
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        image_filename = self.make_full_filename('%s_%s.CSV'%(self.base_name(workspace),cpmeas.IMAGE),workspace)
        fid_per_image = open(image_filename,"wb")
        columns = self.get_pipeline_measurement_columns(pipeline, 
                                                        image_set_list, remove_postgroup_key = True)
        agg_columns = self.get_aggregate_columns(pipeline, image_set_list)
        for image_number in measurements.get_image_numbers():
            image_row = []
            image_row.append(image_number)
            for object_name, feature, coltype in columns:
                if object_name != cpmeas.IMAGE:
                    continue
                if self.ignore_feature(object_name, feature, measurements):
                    continue
                feature_name = "%s_%s" % (object_name,feature)
                value = measurements.get_measurement(
                    cpmeas.IMAGE, feature, image_number)
                if isinstance(value, np.ndarray):
                    value = value[0]
                if coltype.startswith(cpmeas.COLTYPE_VARCHAR):
                    if isinstance(value, str) or isinstance(value, unicode):
                        value = '"'+MySQLdb.escape_string(value)+'"'
                    elif value is None:
                        value = "NULL"
                    else:
                        value = '"'+MySQLdb.escape_string(value)+'"'
                elif np.isnan(value) or np.isinf(value):
                    value = "NULL"
                    
                image_row.append(value)
            #
            # Add the aggregate measurements
            #
            agg_dict = measurements.compute_aggregate_measurements(
                image_number, self.agg_names)
            image_row += [agg_dict[col[3]] for col in agg_columns]
            fid_per_image.write(','.join([str(x) for x in image_row])+"\n")
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
                for j in range(max_count):
                    object_row = [image_number]
                    if file_object_name == cpmeas.OBJECT:
                        # the object number
                        object_row.append(j+1)
                    #
                    # Write out in same order as in the column definition
                    for object_name in object_names:
                        for object_name_to_check, feature, coltype in columns:
                            if object_name_to_check != object_name:
                                continue
                            values = measurements.get_measurement(
                                object_name, feature, image_number)
                            if (values is None or len(values) <= j or
                                np.isnan(values[j]) or np.isinf(values[j])):
                                value = "NULL"
                            else:
                                value = values[j]
                            object_row.append(value)
                    csv_writer.writerow(object_row)
            fid.close()

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
                         post_group = False, 
                         index = None,
                         image_number = None):
        """Write the data in the measurements out to the database
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        index - index of image set's measurements. Defaults to current.
        image_number - image number for primary database key. Defaults to current.
        """
        try:            
            zeros_for_nan = False
            measurements = workspace.measurements
            assert isinstance(measurements, cpmeas.Measurements)
            pipeline = workspace.pipeline
            image_set_list = workspace.image_set_list
            measurement_cols = self.get_pipeline_measurement_columns(pipeline,
                                                                     image_set_list)
            mapping = self.get_column_name_mappings(pipeline, image_set_list)
            if index is None:
                index = measurements.image_set_number - 1
            
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
                feature_name = "%s_%s"%(cpmeas.IMAGE, m_col[1])
                value = measurements.get_measurement(
                    cpmeas.IMAGE, m_col[1], image_number)
                if isinstance(value, np.ndarray):
                    value=value[0]
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
                stmt = ('DELETE FROM %s WHERE %s=%d'%
                        (self.get_table_name(cpmeas.IMAGE), 
                         C_IMAGE_NUMBER,
                         image_number))
                execute(self.cursor, stmt)
            
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
                    object_cols = []
                    if not post_group:
                        object_cols += [C_IMAGE_NUMBER]
                    if table_object_name == cpmeas.OBJECT:
                        object_number_column = C_OBJECT_NUMBER
                        if not post_group:
                            object_cols += [object_number_column]
                        object_numbers = np.arange(1, max_count+1)
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
                            
                        for column in columns:
                            object_name, feature, coltype = column[:3]
                            values = measurements.get_measurement(
                                object_name, feature, image_number)
                            if (values is None or len(values) <= j or
                                np.isnan(values[j]) or
                                np.isinf(values[j])):
                                value = None
                            else:
                                value = str(values[j])
                            object_row.append(value)
                        if post_group:
                            object_row.append(object_numbers[j])
                        object_rows.append(object_row)
                    #
                    # Delete any prior data for this image
                    #
                    if not post_group:
                        stmt = ('DELETE FROM %s WHERE %s=%d'%
                                (table_name, C_IMAGE_NUMBER, image_number))
                        execute(self.cursor, stmt)
                        #
                        # Write the object table data
                        #
                        stmt = ('INSERT INTO %s (%s) VALUES (%s)'%
                                (table_name, 
                                 ','.join(object_cols),
                                 ','.join(['%s']*len(object_cols))))
                    else:
                        stmt = (
                            ('UPDATE %s SET\n' % table_name) +
                            (',\n'.join(["  %s=%%s" % c for c in object_cols])) +
                            ('\nWHERE %s = %d' % (C_IMAGE_NUMBER, image_number)) +
                            ('\nAND %s = %%s' % object_number_column))
            
                    if self.db_type == DB_MYSQL:
                        # Write 25 rows at a time (to get under the max_allowed_packet limit)
                        for i in range(0,len(object_rows), 25):
                            my_rows = object_rows[i:min(i+25, len(object_rows))]
                            self.cursor.executemany(stmt, my_rows)
                    else:
                        for row in object_rows:
                            row = [ 'NULL' if x is None else x for x in row]
                            row_stmt = stmt % tuple(row)
                            self.cursor.execute(row_stmt)
            
            image_table = self.get_table_name(cpmeas.IMAGE)
            replacement = '%s' if self.db_type == DB_MYSQL else "?"
            image_row_values = [
                None 
                if ((field[1] == cpmeas.COLTYPE_FLOAT) and 
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
                execute(self.cursor, stmt, image_row_values)
            self.connection.commit()
        except:
            logger.error("Failed to write measurements to database", exc_info=True)
            self.connection.rollback()
            raise

    def write_properties_file(self, workspace):
        """Write the CellProfiler Analyst properties file"""
        #
        # Get appropriate object names
        #
        if self.objects_choice == O_SELECT:
            object_names = (self.objects_list.value).split(',')
        elif self.objects_choice == O_NONE:
            object_names = ""
        else:
            object_names = [object_name for object_name in workspace.measurements.get_object_names() 
                           if (object_name is not cpmeas.IMAGE and not self.ignore_object(object_name))]
            ## Defaults to the first object in the list, which is the last one defined in the pipeline
##            object_names = [object_names[0]] if len(object_names) > 0 else ""
                
        image_names = []
        if self.properties_export_all_image_defaults:
            # Find all images that have FileName and PathName
            for feature in workspace.measurements.get_feature_names(cpmeas.IMAGE):
                match = re.match('^%s_(.+)$'%C_FILE_NAME,feature)
                if match:
                    image_names.append(match.groups()[0])
        else:
            # Extract the user-specified images
            for group in self.image_groups:
                image_names.append(group.image_cols.value)
        
        if self.db_type==DB_SQLITE:
            name = os.path.splitext(self.sqlite_file.value)[0]
        else:
            name = self.db_name.value
        tbl_prefix = self.get_table_prefix()
        if tbl_prefix is not "":
            if tbl_prefix.endswith('_'): tbl_prefix = tbl_prefix[:-1]
            name = "_".join((name, tbl_prefix))

        tblname = name
        date = datetime.datetime.now().ctime()
        db_type = (self.db_type == DB_MYSQL and 'mysql') or (self.db_type == DB_SQLITE and 'sqlite') or 'oracle_not_supported'
        db_port = (self.db_type == DB_MYSQL and 3306) or (self.db_type == DB_ORACLE and 1521) or ''
        db_host = self.db_host
        db_pwd  = self.db_passwd
        db_name = self.db_name
        db_user = self.db_user
        db_sqlite_file = (self.db_type == DB_SQLITE and 
                          self.make_full_filename(self.sqlite_file.value) ) or ''
        if self.db_type == DB_MYSQL or self.db_type == DB_ORACLE:
            db_info =  'db_type      = %(db_type)s\n'%(locals())
            db_info += 'db_port      = %(db_port)d\n'%(locals())
            db_info += 'db_host      = %(db_host)s\n'%(locals())
            db_info += 'db_name      = %(db_name)s\n'%(locals())
            db_info += 'db_user      = %(db_user)s\n'%(locals())
            db_info += 'db_passwd    = %(db_pwd)s'%(locals())
        elif self.db_type == DB_SQLITE:
            db_info =  'db_type         = %(db_type)s\n'%(locals())
            db_info += 'db_sqlite_file  = %(db_sqlite_file)s'%(locals())
        elif self.db_type == DB_MYSQL_CSV:
            db_info =  'db_type      = mysql\n'
            db_info += 'db_port      = \n'
            db_info += 'db_host      = \n'
            db_info += 'db_name      = %(db_name)s\n'%(locals())
            db_info += 'db_user      = \n'
            db_info += 'db_passwd    = '
        
        spot_tables = '%sPer_Image'%(self.get_table_prefix())
        
        for object_name in object_names:
        
            if self.objects_choice != O_NONE and self.separate_object_tables == OT_COMBINE:
                cell_tables = '%sPer_Object'%(self.get_table_prefix())
                object_id = C_OBJECT_NUMBER
                filename = '%s.properties'%(tblname)
                ## Defaults to the first object in the list, which is the last one defined in the pipeline
##                object_names = [object_names[0]] if len(object_names) > 0 else ""
                if len(object_names) > 1 and object_name == object_names[1]:
                    break  ## Stop on second iteration
            else:
                cell_tables = '%sPer_%s'%(self.get_table_prefix(),object_name) if object_name else ''
                object_id = '%s_Number_Object_Number'%(object_name) if object_name else ''
                filename = '%s_%s.properties'%(tblname,object_name)
                
            file_name = self.make_full_filename(filename,workspace)
            fid = open(file_name,'wt')            
            unique_id = C_IMAGE_NUMBER
            object_count = 'Image_Count_%s'%(object_name) if object_name else ''
            cell_x_loc = '%s_Location_Center_X'%(object_name) if object_name else ''
            cell_y_loc = '%s_Location_Center_Y'%(object_name) if object_name else ''
            image_file_cols = ','.join(['%s_%s_%s'%(cpmeas.IMAGE,C_FILE_NAME,name) for name in image_names])
            image_path_cols = ','.join(['%s_%s_%s'%(cpmeas.IMAGE,C_PATH_NAME,name) for name in image_names])
            image_thumbnail_cols = ','.join(['%s_Thumbnail_%s'%(cpmeas.IMAGE,name) for name in self.thumbnail_image_names.get_selections()])
            
            if self.properties_export_all_image_defaults:
                # Provide default colors
                if len(image_names) == 1:
                    image_channel_colors = 'gray,'
                else:
                    image_channel_colors = 'red, green, blue, cyan, magenta, yellow, gray, '+('none, ' * 10)
                    image_channel_colors = ','.join(image_channel_colors.split(',')[:len(image_names)])
            else:
                # Extract user-specified image names
                image_names = [];
                for group in self.image_groups:
                    if group.wants_automatic_image_name:
                        image_names += [group.image_cols.value]
                    else:
                        image_names += [group.image_name.value]
                        
                # Extract user-specified colors
                image_channel_colors = []
                for group in self.image_groups:
                    image_channel_colors += [group.image_channel_colors.value]
                image_channel_colors = ','.join(image_channel_colors)
            
            image_names_csl = ','.join(image_names) # Convert to comma-separated list
                
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
                        plate_text = re.sub("[^A-Za-z0-9_]",'_',metadata_group.get(plate_key)) # Replace any odd characters with underscores
                        filter_name = 'Plate_%s'%plate_text
                        filter_statements += 'filter_SQL_' + filter_name + ' = SELECT ImageNumber'\
                                            ' FROM ' + spot_tables + \
                                            ' WHERE Image_Metadata_%s' \
                                            ' = "%s"\n'%(plate_key, metadata_group.get(plate_key))
                    
                for group in self.filter_field_groups:
                    filter_statements += 'filter_SQL_' + group.filter_name.value + ' = SELECT ImageNumber'\
                                            ' FROM ' + spot_tables + \
                                            ' WHERE ' + group.filter_statement.value + '\n'
            
            image_url = self.properties_image_url_prepend.value
            plate_type = "" if self.properties_plate_type.value == NONE_CHOICE else self.properties_plate_type.value
            plate_id = "" if self.properties_plate_metadata.value == NONE_CHOICE else "%s_%s_%s"%(cpmeas.IMAGE, cpmeas.C_METADATA, self.properties_plate_metadata.value)
            well_id = "" if self.properties_well_metadata.value == NONE_CHOICE else "%s_%s_%s"%(cpmeas.IMAGE, cpmeas.C_METADATA, self.properties_well_metadata.value)
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

class_table  =

# ======== Check Tables ========
# OPTIONAL
# [yes/no]  You can ask classifier to check your tables for anomalies such
# as orphaned objects or missing column indices.  Default is on.
# This check is run when Classifier starts and may take up to a minute if
# your object_table is extremely large.

check_tables = yes
    """%(locals())
            fid.write(contents)
            fid.close()
        
    def write_workspace_file(self, workspace):
        from cellprofiler.utilities.get_revision import get_revision
        '''If requested, write a workspace file with selected measurements'''
        if self.db_type==DB_SQLITE:
            name = os.path.splitext(self.sqlite_file.value)[0]
        else:
            name = self.db_name.value
        tbl_prefix = self.get_table_prefix()
        if tbl_prefix is not "":
            if tbl_prefix.endswith('_'): tbl_prefix = tbl_prefix[:-1]
            name = "_".join((name, tbl_prefix))
            
        filename = '%s.workspace'%(name)
        file_name = self.make_full_filename(filename,workspace)
            
        fd = open(file_name,"wb")
        header_text = """CellProfiler Analyst workflow
version: 1
svn revision: %d\n"""%get_revision()
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
%s"""%display_tool
            
            axis_text = "x-axis" if workspace_group.measurement_display.value != W_PLATEVIEWER else "measurement"
            if workspace_group.x_measurement_type.value == cpmeas.IMAGE:
                axis_meas = "_".join((cpmeas.IMAGE, workspace_group.x_measurement_name.value))
            elif workspace_group.x_measurement_type.value == cpmeas.OBJECT:
                axis_meas = "_".join((workspace_group.x_object_name.value, workspace_group.x_measurement_name.value))
            elif workspace_group.x_measurement_type.value == W_INDEX:
                axis_meas = workspace_group.x_index_name.value
            axis_table = "x-table" if workspace_group.measurement_display.value in (W_SCATTERPLOT, W_DENSITYPLOT) else "table"
            table_name = self.get_table_name(cpmeas.OBJECT if workspace_group.x_measurement_type.value == cpmeas.OBJECT else cpmeas.IMAGE)
            display_tool_text += """
\t%s: %s
\t%s: %s"""%(axis_text, axis_meas, axis_table, table_name)
            
            if workspace_group.measurement_display.value in (W_SCATTERPLOT, W_DENSITYPLOT):
                if workspace_group.y_measurement_type.value == cpmeas.IMAGE:
                    axis_meas = "_".join((cpmeas.IMAGE, workspace_group.y_measurement_name.value))
                elif workspace_group.y_measurement_type.value == cpmeas.OBJECT:
                    axis_meas = "_".join((workspace_group.y_object_name.value, workspace_group.y_measurement_name.value))
                elif workspace_group.y_measurement_type.value == W_INDEX:
                    axis_meas = workspace_group.y_index_name.value
                table_name = self.get_table_name(cpmeas.OBJECT if workspace_group.y_measurement_type.value == cpmeas.OBJECT else cpmeas.IMAGE)
                display_tool_text += """ 
\ty-axis: %s
\ty-table: %s"""%(axis_meas, table_name)
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
                         for name in m.get_all_measurements(cpmeas.IMAGE,feature)
                         if name is not None]
                if len(names) > 0:
                    FileNameWidth = max(FileNameWidth, np.max(map(len,names)))
            elif feature.startswith(C_PATH_NAME):
                names = [name
                         for name in m.get_all_measurements(cpmeas.IMAGE,feature)
                         if name is not None]
                if len(names) > 0:
                    PathNameWidth = max(PathNameWidth, np.max(map(len,names)))
        return FileNameWidth, PathNameWidth
    
    def get_table_prefix(self):
        if self.want_table_prefix.value:
            return self.table_prefix.value
        return ""

    
    def get_table_name(self, object_name):
        '''Return the table name associated with a given object
        
        object_name - name of object or "Image", "Object" or "Well"
        '''
        return self.get_table_prefix()+'Per_'+object_name

    
    def get_pipeline_measurement_columns(self, pipeline, image_set_list, remove_postgroup_key = False):
        '''Get the measurement columns for this pipeline, possibly cached'''
        d = self.get_dictionary(image_set_list)
        if not d.has_key(D_MEASUREMENT_COLUMNS):
            d[D_MEASUREMENT_COLUMNS] = pipeline.get_measurement_columns()
            d[D_MEASUREMENT_COLUMNS] = \
             [x for x in d[D_MEASUREMENT_COLUMNS]
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
            d[D_MEASUREMENT_COLUMNS].sort(cmp=cmpfn)
        if remove_postgroup_key:
            d[D_MEASUREMENT_COLUMNS] = [x[:3] for x in d[D_MEASUREMENT_COLUMNS]]
        return d[D_MEASUREMENT_COLUMNS]

    def obfuscate(self):
        '''Erase sensitive information about the database
        
        This is run on a copy of the pipeline, so it's ok to erase info.
        '''
        self.db_host.value = ''.join(['*'] * len(self.db_host.value))
        self.db_user.value = ''.join(['*'] * len(self.db_user.value))
        self.db_name.value = ''.join(['*'] * len(self.db_name.value))
        self.db_passwd.value = ''.join(['*'] * len(self.db_passwd.value))
    
    def upgrade_settings(self,setting_values,variable_revision_number,
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
            new_setting_values = [setting_values[0],setting_values[1]]
            if setting_values[2] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
                new_setting_values.append("Expt_")
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
            if setting_values[5][:3]==cps.YES:
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
                new_setting_values.append("Expt_")
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
            if setting_values[18][:3]==cps.YES:
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
            new_setting_values += [ 'imgdb01','cpuser','password']
            #
            # SQLite file name
            #
            new_setting_values += [ 'DefaultDB.db' ]
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
                new_setting_values += [ O_ALL, ""]
            else:
                objects_list = []
                for setting in setting_values[8:15]:
                    if setting not in (cpmeas.IMAGE, cps.DO_NOT_USE):
                        objects_list.append(setting)
                if len(objects_list) > 0:
                    new_setting_values += [ O_SELECT, ",".join(objects_list)]
                else:
                    new_setting_values += [ O_NONE, ""]
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
            setting_values += [ O_ALL, ""]
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
            setting_values = ([ db_type ] + setting_values[1:8] +
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
                    custom_directory = '.'+custom_directory[1:]
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
            setting_values = setting_values + [cps.YES, "1", "1", "0"] # Hidden counts
            setting_values = setting_values + ["None", cps.YES, "None", "gray"] # Image info
            setting_values = setting_values + [cps.NO, "", "ImageNumber, Image_Metadata_Plate, Image_Metadata_Well"] # Group specifications
            setting_values = setting_values + [cps.NO, cps.NO] # Filter specifications
            variable_revision_number = 20
            
        if (not from_matlab) and variable_revision_number == 20:
            #
            # Added configuration of workspace file
            #
            setting_values = setting_values[:SETTING_WORKSPACE_GROUP_COUNT] + \
                            ["1"] + \
                            setting_values[SETTING_WORKSPACE_GROUP_COUNT:]      # workspace_measurement_count
            setting_values += [ cps.NO]                                         # create_workspace_file
            setting_values += [ W_SCATTERPLOT,                                  # measurement_display
                                cpmeas.IMAGE, cpmeas.IMAGE, "", C_IMAGE_NUMBER, # x_measurement_type, x_object_name, x_measurement_name, x_index_name
                                cpmeas.IMAGE, cpmeas.IMAGE, "", C_IMAGE_NUMBER] # y_measurement_type, y_object_name, y_measurement_name, y_index_name
            variable_revision_number == 21
            
        return setting_values, variable_revision_number, from_matlab
    
class ColumnNameMapping:
    """Represents a mapping of feature name to column name"""
    
    def __init__(self,max_len=64):
        self.__dictionary = {}
        self.__mapped = False
        self.__max_len = max_len
    
    def add(self,feature_name):
        """Add a feature name to the collection"""
        
        self.__dictionary[feature_name] = feature_name
        self.__mapped = False
    
    def __getitem__(self,feature_name):
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
                name = re.sub("[^0-9a-zA-Z_$]","_",name)
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
                to_remove = len(name)-self.__max_len
                if to_remove > 0:
                    remove_count = 0
                    for to_drop in (('a','e','i','o','u'),
                                    ('b','c','d','f','g','h','j','k','l','m','n',
                                     'p','q','r','s','t','v','w','x','y','z'),
                                    ('A','B','C','D','E','F','G','H','I','J','K',
                                     'L','M','N','O','P','Q','R','S','T','U','V',
                                     'W','X','Y','Z')):
                        for index in range(len(name)-1,pos-1,-1):
                            if name[index] in to_drop:
                                name = name[:index]+name[index+1:]
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
                        name = name[:index]+name[index+1:]
            reverse_dictionary.pop(orig_name)
            reverse_dictionary[name] = key
            self.__dictionary[key] = name


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
    
