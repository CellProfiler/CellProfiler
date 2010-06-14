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

See also <b>ExportToSpreadsheet</b>.

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

__version__="$Revision$"

import csv
import datetime
import hashlib
import numpy as np
import os
import random
import re
import sys
import traceback
try:
    import MySQLdb
    from MySQLdb.cursors import SSCursor
    HAS_MYSQL_DB=True
except:
    traceback.print_exc()
    sys.stderr.write("WARNING: MySQL could not be loaded.\n")
    HAS_MYSQL_DB=False

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
import cellprofiler.measurements as cpmeas
from identify import M_NUMBER_OBJECT_NUMBER
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import \
     standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
     DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME, \
     DEFAULT_OUTPUT_SUBFOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
     IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

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

##############################################
#
# Keyword for the cached measurement columns
#
##############################################
D_MEASUREMENT_COLUMNS = "MeasurementColumns"

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
    return connection, cursor


def connect_sqlite(db_file):
    '''Creates and returns a db connection and cursor.'''
    import sqlite3 
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    return connection, cursor

    
class ExportToDatabase(cpm.CPModule):
 
    module_name = "ExportToDatabase"
    variable_revision_number = 16
    category = "Data Tools"

    def create_settings(self):
        db_choices = ([DB_MYSQL, DB_MYSQL_CSV, DB_SQLITE] if HAS_MYSQL_DB
                      else [DB_MYSQL_CSV, DB_SQLITE])
        default_db = DB_MYSQL if HAS_MYSQL_DB else DB_MYSQL_CSV
        self.db_type = cps.Choice(
            "Database type",
            db_choices, default_db, doc = """
            What type of database do you want to use? <ul><li><i>MySQL</i>
            allows the data to be written directly to a MySQL 
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
            <a href="http://www.sqlite.org/">here</a>. </li></ul>""")
        
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
            doc="""<i>(Used only when saving csvs, or creating a properties file)</i><br>This setting determines where the .csv files are saved if
            you decide to save measurements to files instead of writing them
            directly to the database. %(IO_FOLDER_CHOICE_HELP_TEXT)s 
            
            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s. 
            For instance, if you have a metadata tag named 
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name as "\g&lt;Plate&gt;". 
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
        ip = socket.gethostbyaddr(socket.gethostname())[-1][0]
        default_prepend = ""
        if ip.startswith('69.173'): # Broad
            default_prepend = "http://imageweb/images/CPALinks"
        self.properties_image_url_prepend = cps.Text(
            "Enter an image url prepend if you plan to access your files via http (leave blank if local)",
            default_prepend, 
            doc = """The image paths written to the database will be the absolute
            path the the image files on your computer. If you plan to make these 
            files accessible via the web, you can enter a url prefix here. Eg: 
            If an image is loaded from the path "/cellprofiler/images/" and you use
            a url prepend of "http://mysite.com/", CellProfiler Analyst will look
            for your file at "http://mysite.com/cellprofiler/images/" """)

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
            "Per_Well_Mean", with a column called "Mean_Nuclei_AreaShape_Area". 
            <p><i>Note:</i> this option is only
            available if you have extracted plate and well metadata from the filename or via a <b>LoadData</b> module.
            It will write out a .sql file with the statements necessary to create the Per_Well
            table, regardless of the option chosen above. %s'''% USING_METADATA_HELP_REF)
        
        self.wants_agg_median_well = cps.Binary(
            "Calculate the per-well median values of object measurements?", False)
        
        self.wants_agg_std_dev_well = cps.Binary(
            "Calculate the per-well standard deviation values of object measurements?", False)
        
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
            <i>(Used if Select is chosen for adding objects)</i><br>
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
            Check this option if you'd like to write image thumbnails directly
            into the database. This will slow down the writing step, but will
            enable new functionality in CellProfiler Analyst such as quickly
            viewing images in the Plate Viewer tool by selecting "thumbnail"
            from the "Well display" dropdown.""")
        self.thumbnail_image_names = cps.ImageNameSubscriberMultiChoice(
            "Select the images you want to save thumbnails of",
            doc = """Select the images that you wish to save as thumbnails to 
            the database.""")
                                                
    def visible_settings(self):
        needs_default_output_directory =\
            (self.db_type != DB_MYSQL or
             self.save_cpa_properties.value)
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
        result += [self.save_cpa_properties]
        if self.save_cpa_properties.value:
            result += [self.properties_image_url_prepend]
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
        if self.db_type == DB_MYSQL:
            result += [self.want_image_thumbnails]
            if self.want_image_thumbnails:
                result += [self.thumbnail_image_names]
        return result
    
    def settings(self):
        return [self.db_type, self.db_name, self.want_table_prefix,
                self.table_prefix, self.sql_file_prefix, 
                self.directory,
                self.save_cpa_properties, 
                self.db_host, self.db_user, self.db_passwd, self.sqlite_file,
                self.wants_agg_mean, self.wants_agg_median,
                self.wants_agg_std_dev, self.wants_agg_mean_well, 
                self.wants_agg_median_well, self.wants_agg_std_dev_well,
                self.objects_choice, self.objects_list, self.max_column_size,
                self.separate_object_tables, self.properties_image_url_prepend, self.want_image_thumbnails,
                self.thumbnail_image_names]
    
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
                raise cps.ValidationError('Invalid SQL file prefix', self.sql_file_prefix)
        
        if self.objects_choice == O_SELECT:
            self.objects_list.load_choices(pipeline)
            if len(self.objects_list.choices) == 0:
                raise cps.ValidationError("Please choose at least one object",
                                          self.objects_choice)
            

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToDatabase will not produce output in Test Mode",
                                      self.db_type)
        
        '''Warn user that they will have to merge tables to use CPA'''
        if self.objects_choice != O_NONE and self.separate_object_tables == OT_PER_OBJECT:
            raise cps.ValidationError(
                ("You will have to merge the separate object tables in order\n"
                 "to use CellProfiler Analyst. Choose %s to write a single\n"
                 "object table.") % OT_COMBINE, self.separate_object_tables)
                

    def prepare_run(self, pipeline, image_set_list, frame):
        '''Prepare to run the pipeline
        Establish a connection to the database.'''

        if pipeline.test_mode:
            return True

        if self.db_type==DB_MYSQL:
            self.connection, self.cursor = connect_mysql(self.db_host.value, 
                                                         self.db_user.value, 
                                                         self.db_passwd.value,
                                                         self.db_name.value)
        elif self.db_type==DB_SQLITE:
            db_file = self.get_output_directory()+'/'+self.sqlite_file.value
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
                    for object_name in self.objects_choice.choices:
                        if not self.ignore_object(object_name, True):
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
                    sys.stderr.write("Warning: %s already in database, not creating" %table_msg)
                    return True
                import wx
                dlg = wx.MessageDialog(
                    frame, 
                    'ExportToDatabase will overwrite the %s. OK?' % table_msg,
                                    'Overwrite tables?', 
                                    style=wx.OK|wx.CANCEL|wx.ICON_QUESTION)
                if dlg.ShowModal() != wx.ID_OK:
                    return False

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
    

    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''Alter the output directory path for the remote batch host'''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def get_measurement_columns(self, pipeline):
        if self.want_image_thumbnails:
            cols = []
            for name in self.thumbnail_image_names.get_selections():
                # NOTE: We currently use type BLOB which can only store 64K
                #   This is sufficient for images up to 256 x 256 px
                #   If larger thumbnails are to be stored, this may have to be
                #   bumped to a MEDIUMBLOB.
                cols += [(cpmeas.IMAGE, "Thumbnail_%s"%(name), cpmeas.COLTYPE_BLOB)]
            return cols
        return []
            
    def run_as_data_tool(self, workspace):
        '''Run the module as a data tool
        
        ExportToDatabase has two modes - writing CSVs and writing directly.
        We write CSVs in post_run. We write directly in run.
        '''
        if not self.prepare_run(workspace.pipeline, workspace.image_set_list,
                                workspace.frame):
            return
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
                image_set.get_image(name)
                fd = StringIO()
                
                if issubclass(pixels.dtype.type, np.floating):
                    factor = 255
                elif pixels.dtype == np.bool:
                    factor = 255
                else:
                    raise Exception('ExportToDatabase cannot write image thumbnails from images of type "%s".'%(str(pixels.dtype)))

                if pixels.ndim == 2:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'L')
                elif pixels.ndim == 3:
                    im = Image.fromarray((pixels * factor).astype('uint8'), 'RGB')
                else:
                    raise Exception('ExportToDatabase only supports saving thumbnails of grayscale or 3-channel images. "%s" was neither.'%(name))
                # rescale major axis to 200
                if im.size[0] == max(im.size):
                    w, h = (200, 200 * max(im.size) / min(im.size))
                else:
                    h, w = (200, 200 * max(im.size) / min(im.size))
                im = im.resize((w,h))
                im.save(fd, 'PNG')
                blob = fd.getvalue()
                fd.close()
                measurements.add_image_measurement('Thumbnail_%s'%(name), blob)
        if workspace.pipeline.test_mode:
            return
        if (self.db_type == DB_MYSQL or self.db_type == DB_SQLITE):
            if not workspace.pipeline.test_mode:
                self.write_data_to_db(workspace)

    def post_run(self, workspace):
        if self.save_cpa_properties.value:
            self.write_properties(workspace)
        if self.db_type == DB_MYSQL_CSV:
            path = self.get_output_directory(workspace)
            if not os.path.isdir(path):
                os.makedirs(path)
            self.write_mysql_table_defs(workspace)
            self.write_csv_data(workspace)
        elif self.wants_well_tables:
            if self.db_type != DB_SQLITE:
                per_well = self.write_mysql_table_per_well(workspace)
        if self.db_type in (DB_MYSQL, DB_SQLITE):
            # commit changes to db here or in run?
            print 'Commit'
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
            feature_name.startswith('ExecutionTime_')
            ):
            return True
        return False

    
    def get_column_name_mappings(self, pipeline, image_set_list):
        """Scan all the feature names in the measurements, creating column names"""
        columns = self.get_pipeline_measurement_columns(pipeline, 
                                                        image_set_list)
        mappings = ColumnNameMapping(self.max_column_size.value)
        mappings.add("ImageNumber")
        mappings.add("ObjectNumber")
        for object_name, feature_name, coltype in columns:
            if self.ignore_feature(object_name, feature_name):
                    continue
            mappings.add("%s_%s"%(object_name,feature_name))
            if object_name != 'Image':
                for agg_name in self.agg_names:
                    mappings.add('%s_%s_%s'%(agg_name, object_name, feature_name))
        return mappings
    
    def get_aggregate_columns(self, pipeline, image_set_list):
        '''Get object aggregate columns for the PerImage table
        
        pipeline - the pipeline being run
        image_set_list - for cacheing column data
        
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
            for obname, feature, ftype in columns:
                if obname==ob_table and not self.ignore_feature(obname, feature):
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
        statement += 'ImageNumber INTEGER'

        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        for obname, feature, ftype in self.get_pipeline_measurement_columns(
            pipeline, image_set_list):
            if obname==cpmeas.IMAGE and not self.ignore_feature(obname, feature):
                feature_name = '%s_%s' % (obname, feature)
                statement += ',\n%s %s'%(mappings[feature_name], ftype)
        for column in self.get_aggregate_columns(pipeline, image_set_list):
            statement += ',\n%s %s' % (mappings[column[3]], 
                                       cpmeas.COLTYPE_FLOAT)
        statement += ',\nPRIMARY KEY (ImageNumber) )'
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
        statement += 'ImageNumber INTEGER\n'
        if object_name == None:
            statement += ',ObjectNumber INTEGER'
            object_pk = 'ObjectNumber'
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
            for obname, feature, ftype in column_defs:
                if obname==ob_table and not self.ignore_feature(obname, feature):
                    feature_name = '%s_%s'%(obname, feature)
                    statement += ',\n%s %s'%(mappings[feature_name], ftype)
        statement += ',\nPRIMARY KEY (ImageNumber, %s) )' % object_pk
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
        path_name = os.path.join(self.get_output_directory(workspace), file_name)
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
LOAD DATA LOCAL INFILE '%s_image.CSV' REPLACE INTO TABLE %s
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
""" %
                  (self.base_name(workspace), self.get_table_name(cpmeas.IMAGE)))

        for gcot_name, object_name in data:
            fid.write("""
LOAD DATA LOCAL INFILE '%s_%s.CSV' REPLACE INTO TABLE %s 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
""" % (self.base_name(workspace), object_name,
       self.get_table_name(object_name)))
        if self.wants_well_tables:
            self.write_mysql_table_per_well(workspace, fid)
        fid.close()
    
    def write_mysql_table_per_well(self, workspace, fid=None):
        '''Write SQL statements to generate a per-well table
        
        workspace - workspace at the end of the run
        fid - file handle of file to write or None if statements
              should be written to a separate file.
        '''
        if fid is None:
            file_name = "%s_Per_Well_SETUP.SQL"%(self.sql_file_prefix)
            path_name = os.path.join(self.get_output_directory(workspace), 
                                     file_name)
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
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        mappings = self.get_column_name_mappings(pipeline, image_set_list)
        object_names = self.get_object_names(pipeline, image_set_list)
        columns = self.get_pipeline_measurement_columns(pipeline, image_set_list)
        for aggname in self.agg_well_names:
            measurements = workspace.measurements
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
                    for feature in measurements.get_feature_names(object_name):
                        if self.ignore_feature(object_name, feature, measurements):
                            continue
                        #
                        # Don't take an aggregate on a string column
                        #
                        if not any([True for column in columns
                                    if column[0] == object_name and
                                    column[1] == feature and
                                    not column[2].startswith(cpmeas.COLTYPE_VARCHAR)]):
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
                fid.write("JOIN %s OT ON IT.ImageNumber = OT.ImageNumber\n" %
                          self.get_table_name(cpmeas.OBJECT))
            elif len(object_names) == 1:
                fid.write("JOIN %s OT1 ON IT.ImageNumber = OT1.ImageNumber\n" %
                          self.get_table_name(object_names[0]))
            else:
                #
                # We make up a table here that lists all of the possible
                # image and object numbers from any of the object numbers.
                # We need this to do something other than a cartesian join
                # between object tables.
                #
                fid.write(
                    "RIGHT JOIN (SELECT DISTINCT ImageNumber, ObjectNumber FROM\n")
                fid.write("(SELECT ImageNumber, %s_%s as ObjectNumber FROM %s\n" %
                          (object_names[0], M_NUMBER_OBJECT_NUMBER, 
                           self.get_table_name(object_names[0])))
                for object_name in object_names[1:]:
                    fid.write("UNION SELECT ImageNumber, %s_%s as ObjectNumber "
                              "FROM %s\n" % 
                              (object_name, M_NUMBER_OBJECT_NUMBER, 
                               self.get_table_name(object_name)))
                fid.write(") N_INNER) N ON IT.ImageNumber = N.ImageNumber\n")
                for i, object_name in enumerate(object_names):
                    fid.write("LEFT JOIN %s OT%d " % 
                              (self.get_table_name(object_name), i+1))
                    fid.write("ON N.ImageNumber = OT%d.ImageNumber " % (i+1))
                    fid.write("AND N.ObjectNumber = OT%d.%s_%s\n" %
                              (i+1, object_name, M_NUMBER_OBJECT_NUMBER))
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
        image_filename = os.path.join(self.get_output_directory(workspace),
                                      '%s_image.CSV'%(self.base_name(workspace)))
        fid_per_image = open(image_filename,"wb")
        columns = self.get_pipeline_measurement_columns(pipeline, 
                                                        image_set_list)
        agg_columns = self.get_aggregate_columns(pipeline, image_set_list)
        for i in range(measurements.image_set_index+1):
            image_row = []
            image_number = i+measurements.image_set_start_number
            image_row.append(image_number)
            for object_name, feature, coltype in columns:
                if object_name != cpmeas.IMAGE:
                    continue
                if self.ignore_feature(object_name, feature, measurements):
                    continue
                feature_name = "%s_%s" % (object_name,feature)
                value = measurements.get_measurement(cpmeas.IMAGE, feature, i)
                if isinstance(value, np.ndarray):
                    value = value[0]
                if coltype.startswith(cpmeas.COLTYPE_VARCHAR):
                    if isinstance(value, str) or isinstance(value, unicode):
                        value = '"'+MySQLdb.escape_string(value)+'"'
                    elif value is None:
                        value = "NULL"
                    else:
                        value = '"'+MySQLdb.escape_string(value)+'"'
                elif np.isnan(value):
                    value = "NULL"
                    
                image_row.append(value)
            #
            # Add the aggregate measurements
            #
            agg_dict = measurements.compute_aggregate_measurements(
                i, self.agg_names)
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
            file_name = os.path.join(self.get_output_directory(), file_name)
            fid = open(file_name, "wb")
            csv_writer = csv.writer(fid, lineterminator='\n')
            for i in range(measurements.image_set_index+1):
                image_number = i+measurements.image_set_start_number
                max_count = 0
                for object_name in object_list:
                    count = measurements.get_measurement(cpmeas.IMAGE,
                                                         "Count_%s" % 
                                                         object_name, i)
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
                            values = measurements.get_measurement(object_name,
                                                                  feature, i)
                            if (values is None or len(values) <= j or
                                np.isnan(values[j])):
                                value = "NULL"
                            else:
                                value = values[j]
                            object_row.append(value)
                    csv_writer.writerow(object_row)
            fid.close()

            
    def write_data_to_db(self, workspace):
        """Write the data in the measurements out to the database
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        """
        try:            
            zeros_for_nan = False
            measurements = workspace.measurements
            pipeline = workspace.pipeline
            image_set_list = workspace.image_set_list
            measurement_cols = self.get_pipeline_measurement_columns(pipeline,
                                                                     image_set_list)
            mapping = self.get_column_name_mappings(pipeline, image_set_list)
            index = measurements.image_set_index
            
            ###########################################
            #
            # The image table
            #
            ###########################################
            image_number = measurements.image_set_number
            image_row = [(image_number, cpmeas.COLTYPE_INTEGER, "ImageNumber")]
            feature_names = set(measurements.get_feature_names(cpmeas.IMAGE))
            for m_col in measurement_cols:
                if m_col[0] != cpmeas.IMAGE:
                    continue
                #
                # Skip if feature name not in measurements. This
                # can happen if image set gets aborted or for some legacy
                # measurement files.
                #
                if m_col[1] not in feature_names:
                    continue
                feature_name = "%s_%s"%(cpmeas.IMAGE, m_col[1])
                value = measurements.get_current_image_measurement(m_col[1])
                if isinstance(value, np.ndarray):
                    value=value[0]
                if isinstance(value, float) and not np.isfinite(value) and zeros_for_nan:
                    value = 0
                image_row.append((value, m_col[2], feature_name))
            #
            # Aggregates for the image table
            #
            agg_dict = measurements.compute_aggregate_measurements(index,
                                                                   self.agg_names)
            agg_columns = self.get_aggregate_columns(pipeline, image_set_list)
            image_row += [(agg_dict[agg[3]], 
                           cpmeas.COLTYPE_FLOAT, 
                           agg[3])
                          for agg in agg_columns]
            
            #
            # Delete any prior data for this image
            #
            # XXX: This shouldn't be neccessary since the table is dropped 
            #      before writing.
            stmt = ('DELETE FROM %s WHERE ImageNumber=%d'%
                    (self.get_table_name(cpmeas.IMAGE), image_number))
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
                               if column[0] in object_list]
                    max_count = 0
                    for object_name in object_list:
                        count = measurements.get_current_image_measurement(
                            "Count_%s" % object_name)
                        max_count = max(max_count, int(count))
                    object_cols = ["ImageNumber"]
                    if table_object_name == cpmeas.OBJECT:
                        object_cols += ["ObjectNumber"]
                
                    object_cols += [mapping["%s_%s" % (column[0], column[1])]
                                    for column in columns]
                    object_rows = []
                    for j in range(max_count):
                        object_row = [image_number]
                        if table_object_name == cpmeas.OBJECT:
                            # the object number
                            object_row.append(j+1)
                        for object_name, feature, coltype in columns:
                            values = measurements.get_current_measurement(object_name,
                                                                          feature)
                            if (values is None or len(values) <= j or
                                np.isnan(values[j])):
                                value = None
                            else:
                                value = str(values[j])
                            object_row.append(value)
                        object_rows.append(object_row)
                    #
                    # Delete any prior data for this image
                    #
                    stmt = ('DELETE FROM %s WHERE ImageNumber=%d'%
                            (table_name, image_number))
                    execute(self.cursor, stmt)
                    #
                    # Write the object table data
                    #
                    stmt = ('INSERT INTO %s (%s) VALUES (%s)'%
                            (table_name, 
                             ','.join(object_cols),
                             ','.join(['%s']*len(object_cols))))
            
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
                if ((field[1] == cpmeas.COLTYPE_FLOAT) and (np.isnan(field[0])))
                else float(field[0]) if (field[1] == cpmeas.COLTYPE_FLOAT)
                else int(field[0]) if (field[1] == cpmeas.COLTYPE_INTEGER)
                else field[0] for field in image_row]
            stmt = ('INSERT INTO %s (%s) VALUES (%s)' % 
                    (image_table, 
                     ','.join([mapping[colname] for val, dtype, colname in image_row]),
                     ','.join([replacement] * len(image_row))))
            execute(self.cursor, stmt, image_row_values)
            self.connection.commit()
        except:
            traceback.print_exc()
            self.connection.rollback()
            raise

        
    
    def write_properties(self, workspace):
        """Write the CellProfiler Analyst properties file"""
        #
        # Find the primary object
        #
        for object_name in workspace.measurements.get_object_names():
            if object_name == 'Image':
                continue
            if self.ignore_object(object_name):
                continue
        supposed_primary_object = object_name
        #
        # Find all images that have FileName and PathName
        #
        image_names = []
        for feature in workspace.measurements.get_feature_names('Image'):
            match = re.match('^FileName_(.+)$',feature)
            if match:
                image_names.append(match.groups()[0])
        
        if self.db_type==DB_SQLITE:
            name = os.path.splitext(self.sqlite_file.value)[0]
        else:
            name = self.db_name
        filename = '%s.properties'%(name)
        path = os.path.join(self.get_output_directory(workspace), filename)
        fid = open(path,'wt')
        date = datetime.datetime.now().ctime()
        db_type = (self.db_type == DB_MYSQL and 'mysql') or (self.db_type == DB_SQLITE and 'sqlite') or 'oracle_not_supported'
        db_port = (self.db_type == DB_MYSQL and 3306) or (self.db_type == DB_ORACLE and 1521) or ''
        db_host = self.db_host
        db_pwd  = self.db_passwd
        db_name = self.db_name
        db_user = self.db_user
        db_sqlite_file = (self.db_type == DB_SQLITE and 
                          self.get_output_directory(workspace)+
                          '/'+self.sqlite_file.value) or ''
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
        if self.separate_object_tables == OT_COMBINE:
            cell_tables = '%sPer_Object'%(self.get_table_prefix())
        else:
            cell_tables = '%sPer_%s'%(self.get_table_prefix(),supposed_primary_object)               
        unique_id = 'ImageNumber'
        object_count = 'Image_Count_%s'%(supposed_primary_object)
        if self.separate_object_tables == OT_COMBINE:
            object_id = 'ObjectNumber'
        else:
            object_id = '%s_Number_Object_Number'%(supposed_primary_object)
        cell_x_loc = '%s_Location_Center_X'%(supposed_primary_object)
        cell_y_loc = '%s_Location_Center_Y'%(supposed_primary_object)
        image_file_cols = ','.join(['Image_FileName_%s'%(name) for name in image_names])
        image_path_cols = ','.join(['Image_PathName_%s'%(name) for name in image_names])
        image_thumbnail_cols = ','.join(['Image_Thumbnail_%s'%(name) for name in self.thumbnail_image_names.get_selections()])
        image_names = ','.join(image_names)
        if len(image_names) == 1:
            image_channel_colors = 'gray,'
        else:
            image_channel_colors = 'red, green, blue, cyan, magenta, yellow, gray, '+('none, ' * 10)[:len(image_names)]
        image_url = self.properties_image_url_prepend.value
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
plate_id      = 
well_id       = 

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
image_names = %(image_names)s

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

# 

# ==== Image Filters ====
# Here you can define image filters to let you select objects from a subset of your experiment when training the classifier.
# This is OPTIONAL, you may leave "filters = ".
# FORMAT:
#   filter_SQL_XXX  =  MySQL select statement that returns image keys you wish to filter out.  This will be associated with the filter name "XXX" from above.
# EXAMPLE FILTERS:
#   filters           =  EMPTY, CDKs,
#   filter_SQL_EMPTY  =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene="EMPTY"
#   filter_SQL_CDKs   =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene REGEXP 'CDK.*'

#

# ==== Meta data ====
# What are your objects called?
# FORMAT:
#   object_name  =  singular object name, plural object name,
object_name  =  cell, cells,

# What size plates were used?  384 or 96?  This is for use in the PlateViewer
plate_type  = 

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
        
    def get_file_path_width(self, workspace):
        """Compute the file name and path name widths needed in table defs"""
        m = workspace.measurements
        #
        # Find the length for the file name and path name fields
        #
        FileNameWidth = 128
        PathNameWidth = 128
        image_features = m.get_feature_names('Image')
        for feature in image_features:
            if feature.startswith('FileName'):
                names = [name 
                         for name in m.get_all_measurements('Image',feature)
                         if name is not None]
                if len(names) > 0:
                    FileNameWidth = max(FileNameWidth, np.max(map(len,names)))
            elif feature.startswith('PathName'):
                names = [name
                         for name in m.get_all_measurements('Image',feature)
                         if name is not None]
                if len(names) > 0:
                    PathNameWidth = max(PathNameWidth, np.max(map(len,names)))
        return FileNameWidth, PathNameWidth

    
    def get_output_directory(self, workspace=None):
        return self.directory.get_absolute_path(None if workspace is None
                                                else workspace.measurements)

    
    def get_table_prefix(self):
        if self.want_table_prefix.value:
            return self.table_prefix.value
        return ""

    
    def get_table_name(self, object_name):
        '''Return the table name associated with a given object
        
        object_name - name of object or "Image", "Object" or "Well"
        '''
        return self.get_table_prefix()+'Per_'+object_name

    
    def get_pipeline_measurement_columns(self, pipeline, image_set_list):
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
        return d[D_MEASUREMENT_COLUMNS]

    
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
            new_setting_values = setting_values
            new_setting_values += [False, 'imgdb01', 'cpuser', '', 'DefaultDB.db']
            variable_revision_number = 7
        
        if (not from_matlab) and variable_revision_number == 7:
            # Added ability to selectively turn on aggregate measurements
            # which were all automatically calculated in version 7
            new_setting_values = setting_values + [True, True, True]
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
    
