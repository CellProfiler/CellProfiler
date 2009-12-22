'''<b>Export To Database</b> exports data directly to a database, or in 
database readable format, including an importing file
with column names and a CellProfiler Analyst properties file, if desired.
<hr>

This module exports measurements directly to a database, or to a SQL compatible format. 
It allows you to create MySQL and associated data files which will create a
database and import the data into it and gives you the option of creating
a properties file for use with CellProfiler Analyst. Optionally, you can create
an SQLite DB file if you do not have a server on which to run MySQL itself.

This module must be run at the end of a pipeline, or second to last if 
you are using the CreateBatchFiles module. If you forget this module, you
can also run the ExportDatabase data tool (note: under construction) after processing is complete; 
its functionality is the same.

The database is set up with two primary tables. These tables are the
Per_Image table and the Per_Object table (which may have a prefix if you
specify). The Per_Image table consists of all the per-image measurements made during the pipeline, plus
per-image population statistics (such as mean, median, and standard deviation) of the object measurements. There is one
Per_Image row for every "cycle" that CellProfiler processes (usually, a cycle is a single field of view, and a single cycle usually contains several image files, each representing a different channel of the same field of view). The Per_Object table contains all the
measurements for individual objects. There is one row of object
measurements per object identified. The two tables are connected with the
primary key column ImageNumber, which indicates to which image each object belongs. The Per_Object table has another primary
key called ObjectNumber, which is unique per image. In the most typical use, if multiple types of objects are identified and measured in a pipeline, the number of those objects are equal to each other. For example, in most pipelines, each nucleus has exactly one cytoplasm, so the first row of the Per-Object table contains all of the information about object #1, including both nucleus- and cytoplasm-related measurements. If this one-to-one correspondence is <em>not</em> the case for all objects in the pipeline (for example, if dozens of speckles are identified and measured for each nucleus), then the ExportToDatabase module must be configured to export only objects that maintain the one-to-one correspondence (for example, export only Nucleus and Cytoplasm, but omit Speckles).

If metadata has been used to group images (for example, grouping several image cycles into a single well), then the database can also contain a Per_Group table (for example, a Per_Well table).  

Oracle is not currently fully supported; you can create your own Oracle DB using
the .csv output option, and writing a simple script to upload to the DB.

See also <b>ExportToExcel</b>.

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

import csv
import datetime
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
import cellprofiler.preferences as cpp
import cellprofiler.measurements as cpmeas

##############################################
#
# Database options for the db_type setting
#
##############################################
DB_MYSQL = "MySQL"
DB_ORACLE = "Oracle"
DB_SQLITE = "SQLite"

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
# Keyword for the cached measurement columns
#
##############################################
D_MEASUREMENT_COLUMNS = "MeasurementColumns"

def execute(cursor, query, return_result=True):
    print query
    cursor.execute(query)
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
        return None
    except StopIteration, e:
        return None
    
def connect_mysql(host, user, pw, db):
    '''Creates and returns a db connection and cursor.'''
    connection = MySQLdb.connect(host=host, user=user, passwd=pw, db=db)
    cursor = SSCursor(connection)
    return connection, cursor

def connect_sqlite(db_file):
    '''Creates and returns a db connection and cursor.'''
    from pysqlite2 import dbapi2 as sqlite
    connection = sqlite.connect(db_file)
    cursor = connection.cursor()
    return connection, cursor

    
    
class ExportToDatabase(cpm.CPModule):
 
    module_name = "ExportToDatabase"
    variable_revision_number = 10
    category = "File Processing"

    def create_settings(self):
        self.db_type = cps.Choice("Database type",
                                  [DB_MYSQL,DB_ORACLE,DB_SQLITE], DB_MYSQL, doc = """
                                  What type of database do you want to use? <ul><li><i>MySQL:</i>
                                  This option will allow the data to be written directly to a MySQL database. MySQL is open-source software and may require help from your local Information Technology group to set up a database server.</li>  <li><i>Oracle:</i>This option  is currently
                                  not fully supported, but your data will be written to .csv files. You can then upload your
                                  data to an Oracle database by writing a simple script.</li> <li><i>SQLite:</i> This option will write 
                                  sqlite files directly. SQLite is simpler to set up than MySQL and can more readily be run on your local computer rather than a database server. More information about sqlite can be found at 
                                  <a href="http://www.sqlite.org/"> http://www.sqlite.org/</a> </li></ul>""")
        
        self.db_name = cps.Text(
            "Database name", "DefaultDB",doc = """
            Select a name for the database you want to use?""")
        
        self.want_table_prefix = cps.Binary(
            "Add a prefix to table names?", False, doc = """
            Do you want to add a prefix to your table names?
            This gives you the option to prepend text to your table names
            (Per_Image and Per_Object).  CellProfiler will warn you before overwriting an existing table.""")
        
        self.table_prefix = cps.Text(
            "Table prefix", "Expt_" , doc = """
            <i>(Used if Add Table Prefix is selected)</i><br>
            What is the table prefix you want to use?""")
        
        self.sql_file_prefix = cps.Text(
            "SQL file prefix", "SQL_", doc = """
            <i>(Used if SQL is selected as the database type and if CSV files are to be written)</i><br>
            What prefix do you want to use to name the SQL file?""")
        
        self.use_default_output_directory = cps.Binary(
            "Save files in the default output folder?", True)
        
        self.output_directory = cps.Text(
            "Enter the output folder", ".", doc = """
            <i>(Used if SQL is selected as the database type and if CSV files are to be written)</i><br>
            What folder should be used to save files? Use a "." to indicate the default
            output folder.""")
        
        self.save_cpa_properties = cps.Binary(
            "Create a CellProfiler Analyst properties file?", 
            False, doc = """
            You can generate a template properties file that will allow you to use your new database with CellProfiler Analyst (a data
            exploration tool which can also be downloaded from
            <a href="http://www.cellprofiler.org/"> http://www.cellprofiler.org/ </a>). 
            The module will attempt to fill in as many as the entries as possible 
            based on the pipeline's settings. However, entries such as the 
            server name, username and password are omitted and will need to be edited within CellProfiler Analyst. Opening the 
            properties file in CPA without editing those fields will produce an error since it won't be able to
            connect to the server.""")
        
        self.store_csvs = cps.Binary(
            "Store the database in CSV files? ", False, doc = """
            This will write per_image and per_object tables as a series of CSV files along with an SQL file 
            that can be used with those files to create the database.  You can also look at the csv
            files in a spreadsheet program, such as Excel. The typical usage of the module omits the creation of CSV files and instead data is written directly to the MySQL database.""")
        
        self.mysql_not_available = cps.Divider("Cannot write to MySQL directly - CSV file output only", line=False, 
            doc= """The MySQLdb python module could not be loaded.  MySQLdb is necessary for direct export.""")
        
        self.db_host = cps.Text("Database host", "imgdb01")
        
        self.db_user = cps.Text("Username", "cpuser")
        
        self.db_passwd = cps.Text("Password", "cPus3r")
        
        self.sqlite_file = cps.Text("Name the SQLite database file", 
            "DefaultDB.db", doc = """
            <i>(Used if SQLite selected as database type)</i><br>
            What is the SQLite database filename to which you want to write?""")
        
        self.wants_agg_mean = cps.Binary("Calculate the per-image mean values of object measurements?", True, doc = """
            ExportToDatabase can calculate population statistics over all the objects in each image
            and store the results in the database. For instance, if
            you are measuring the area of the Nuclei objects and you check the box for this option, ExportToDatabase will create a column in the Per_Image
            table called Mean_Nuclei_AreaShape_Area. Check this setting to add 
            these columns to your image file; uncheck it to remove these columns from your image file.
            <p>You may not want to use ExportToDatabase to calculate these measurements if your pipeline generates
            a large number of per-object measurements; doing so might exceed database
            column limits. These columns can be created manually for selected measurements.
            For instance, the following SQL creates the Mean_Nuclei_AreaShape_Area column:
            
                ALTER TABLE Per_Image ADD (Mean_Nuclei_AreaShape_Area);
                UPDATE Per_Image SET Mean_Nuclei_AreaShape_Area = 
                    (SELECT AVG(Nuclei_AreaShape_Area)
                     FROM Per_Object
                     WHERE Per_Image.ImageNumber = Per_Object.ImageNumber);""")
        
        self.wants_agg_median = cps.Binary("Calculate the per-image median values of object measurements?", False)
        
        self.wants_agg_std_dev = cps.Binary("Calculate the per-image standard deviation values of object measurements?", False)
        
        self.wants_agg_mean_well = cps.Binary(
            "Calculate the per-well mean values of object measurements?", False, doc = '''
            ExportToDatabase can calculate statistics over all the objects in each well 
            and store the results as columns in a Per_Well table in the database. For instance, 
            if you are measuring the area of the Nuclei objects and you check the aggregate
            mean box in this module, ExportToDatabase will create a table in database called
            Per_Well_Mean, with a column called Mean_Nuclei_AreaShape_Area. NOTE: this option is only
            available if you have extracted plate and well metadata from the filename or via a LoadText module.
            This option will write out a .SQL file with the statements necessary to create the per_well
            table, regardless of the option chosen above.''')
        
        self.wants_agg_median_well = cps.Binary(
            "Calculate the per-well median values of object measurements?", False)
        
        self.wants_agg_std_dev_well = cps.Binary(
            "Calculate the per-well standard deviation values of object measurements?", False)
        
        self.objects_choice = cps.Choice(
            "Export measurements for all objects to the database?",
            [O_ALL, O_NONE, O_SELECT], doc = """
            This option lets you choose the objects that will have
            their measurements saved in the Per_Object and Per_Well(s) database tables.
            <ul>
            <li><i>All:</i> Export measurements from all objects</li>
            <li><i>None:</i> Do not export data to a Per_Object table. Save only Per_Image or Per_Well measurements (which nonetheless include population statistics from objects).</li>
            <li><i>Select:</i> Select the objects you want to export from a list</li>
            </ul>""")
        
        self.objects_list = cps.ObjectSubscriberMultiChoice(
            "Select the objects", doc = """
            <i>(Used if Select is chosen for adding objects)</i><br>
            Choose one or more objects from this list (click using shift or command keys to select multiple objects from the list). The list includes
            the objects that were created by prior modules. If you choose an
            object, its measurements will be written out to the Per_Object and/or
            Per_Well(s) tables, otherwise, the object's measurements will be skipped.""")
                                                            
    def visible_settings(self):
        needs_default_output_directory =\
            (self.db_type != DB_MYSQL or self.store_csvs.value or
             self.save_cpa_properties.value)
        result = [self.db_type]
        if self.db_type==DB_MYSQL:
            if HAS_MYSQL_DB:
                result += [self.store_csvs]
            else:
                result += [self.mysql_not_available]
            if self.store_csvs.value or not HAS_MYSQL_DB:
                result += [self.sql_file_prefix]
                result += [self.db_name]
            else:
                result += [self.db_name]
                result += [self.db_host]
                result += [self.db_user]
                result += [self.db_passwd]
        elif self.db_type==DB_SQLITE:
            result += [self.sqlite_file]
        elif self.db_type==DB_ORACLE:
            result += [self.sql_file_prefix]
        result += [self.want_table_prefix]
        if self.want_table_prefix.value:
            result += [self.table_prefix]
        result += [self.save_cpa_properties]
        if needs_default_output_directory:
            result += [self.use_default_output_directory]
            if not self.use_default_output_directory.value:
                result += [self.output_directory]
        result += [self.wants_agg_mean, self.wants_agg_median,
                   self.wants_agg_std_dev, self.wants_agg_mean_well, 
                   self.wants_agg_median_well, self.wants_agg_std_dev_well,
                   self.objects_choice]
        if self.objects_choice == O_SELECT:
            result += [self.objects_list]
        return result
    
    def settings(self):
        return [self.db_type, self.db_name, self.want_table_prefix,
                self.table_prefix, self.sql_file_prefix, 
                self.use_default_output_directory, self.output_directory,
                self.save_cpa_properties, self.store_csvs, self.db_host, 
                self.db_user, self.db_passwd, self.sqlite_file,
                self.wants_agg_mean, self.wants_agg_median,
                self.wants_agg_std_dev, self.wants_agg_mean_well, 
                self.wants_agg_median_well, self.wants_agg_std_dev_well,
                self.objects_choice, self.objects_list]
    
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

        if not self.store_csvs.value:
            if not re.match("^[A-Za-z0-9_]+$",self.db_user.value):
                raise cps.ValidationError("The database user name has invalid characters",self.db_user)
            if not re.match("^[A-Za-z0-9_].*$",self.db_host.value):
                raise cps.ValidationError("The database host name has invalid characters",self.db_host)
        else:
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$", self.sql_file_prefix.value):
                raise cps.ValidationError('Invalid SQL file prefix', self.sql_file_prefix)
        
        if self.objects_choice == O_SELECT:
            self.objects_list.load_choices(pipeline)
            if len(self.objects_list.choices) == 0:
                raise cps.ValidationError("Please choose at least one object",
                                          self.objects_choice)
            
    def prepare_run(self, pipeline, image_set_list, frame):
        if self.db_type == DB_ORACLE:
            raise NotImplementedError("Writing to an Oracle database is not yet supported")
        if not self.store_csvs.value:
            if self.db_type==DB_MYSQL:
                self.connection, self.cursor = connect_mysql(self.db_host.value, 
                                                             self.db_user.value, 
                                                             self.db_passwd.value,
                                                             self.db_name.value)
            elif self.db_type==DB_SQLITE:
                db_file = self.get_output_directory()+'/'+self.sqlite_file.value
                self.connection, self.cursor = connect_sqlite(db_file)
            try:
                object_table = self.get_table_prefix()+'Per_Object'
                image_table = self.get_table_prefix()+'Per_Image'
                r = execute(self.cursor, 'SELECT * FROM %s LIMIT 1'%(image_table))
                if self.objects_choice != O_NONE:
                    r = execute(self.cursor, 'SELECT * FROM %s LIMIT 1'%(object_table))
                import wx
                dlg = wx.MessageDialog(frame, 'ExportToDatabase will overwrite your tables "%s" and "%s". OK?'%(image_table, object_table),
                                    'Overwrite per_image and per_object table?', style=wx.OK|wx.CANCEL|wx.ICON_QUESTION)
                if dlg.ShowModal() != wx.ID_OK:
                    return False
            except:
                pass
            mappings = self.get_column_name_mappings(pipeline)
            column_defs = self.get_pipeline_measurement_columns(pipeline, 
                                                                image_set_list)
            if self.objects_choice != O_ALL:
                onames = [cpmeas.EXPERIMENT, cpmeas.IMAGE, cpmeas.NEIGHBORS]
                if self.objects_choice == O_SELECT:
                    onames += self.objects_list.selections
                column_defs = [column for column in column_defs
                               if column[0] in onames]
            self.create_database_tables(self.cursor, 
                                        column_defs,
                                        mappings,
                                        not pipeline.in_batch_mode())
        return True
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''Alter the output directory path for the remote batch host'''
        self.output_directory.value = fn_alter_path(self.output_directory.value)
            
    def run(self, workspace):
        if ((self.db_type == DB_MYSQL and not self.store_csvs.value) or
            self.db_type == DB_SQLITE):
            mappings = self.get_column_name_mappings(workspace.pipeline)
            self.write_data_to_db(workspace, mappings)
            
    def post_run(self, workspace):
        if self.save_cpa_properties.value:
            self.write_properties(workspace)
        mappings = self.get_column_name_mappings(workspace.pipeline)
        if self.db_type == DB_MYSQL:
            per_image, per_object = self.write_mysql_table_defs(workspace, mappings)
        else:
            per_image, per_object = self.write_oracle_table_defs(workspace, mappings)
        if self.wants_agg_mean_well.value:
            per_well = self.write_mysql_table_per_well(workspace, mappings)
        if not self.store_csvs.value:
            # commit changes to db here or in run?
            print 'Commit'
            self.connection.commit()
            return
        self.write_data(workspace, mappings, per_image, per_object)
        if self.wants_agg_mean_well.value:
            self.write_data(workspace, mappings, per_well)
    
    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True
    
    def ignore_object(self,object_name, strict = False):
        """Ignore objects (other than 'Image') if this returns true
        
        If strict is True, then we ignore objects based on the object selection
        """
        if object_name in ('Experiment','Neighbors'):
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
    
    def get_column_name_mappings(self, pipeline):
        """Scan all the feature names in the measurements, creating column names"""
        columns = pipeline.get_measurement_columns()
        mappings = ColumnNameMapping()
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
    def create_database_tables(self, cursor, column_defs, mappings, 
                               create_database):
        '''Creates empty image and object tables
        
        Creates the MySQL database (if MySQL), drops existing tables of the
        same name and creates the tables. Also initializes image_col_order,
        object_col_order and col_dict.
        
        cursor - database cursor for creating the tables
        column_defs - column definitions as returned by get_measurement_columns
        mappings - mappings from measurement feature names to column names
        create_database - True to actually create the database, False if
                          we have to call this just to do the initialization
                          of image_col_order etc. during batch mode
        '''
        self.image_col_order = {}
        self.object_col_order = {}
        
        object_table = self.get_table_prefix()+'Per_Object'
        image_table = self.get_table_prefix()+'Per_Image'
        
        # Build a dictionary keyed by object type of measurement cols
        self.col_dict = {}
        for c in column_defs:
            if c[0]!=cpmeas.EXPERIMENT:
                if c[0] in self.col_dict.keys():
                    self.col_dict[c[0]] += [c]
                else:
                    self.col_dict[c[0]] = [c]
        
        # Create the database
        if self.db_type==DB_MYSQL and create_database:
            execute(cursor, 'CREATE DATABASE IF NOT EXISTS %s'%(self.db_name.value))
            execute(cursor, 'USE %s'% self.db_name.value)
        
        agg_column_defs = []
        if self.objects_choice != O_NONE:
            # Object table
            ob_tables = set([obname for obname, _, _ in column_defs 
                             if obname!=cpmeas.IMAGE and obname!=cpmeas.EXPERIMENT])
            statement = 'CREATE TABLE '+object_table+' (\n'
            statement += 'ImageNumber INTEGER,\n'
            statement += 'ObjectNumber INTEGER'
            c = 2
            for ob_table in ob_tables:
                for obname, feature, ftype in column_defs:
                    if obname==ob_table and not self.ignore_feature(obname, feature):
                        feature_name = '%s_%s'%(obname, feature)
                        # create per_image aggregate column defs 
                        for aggname in self.agg_names:
                            agg_column_defs += [(cpmeas.IMAGE,
                                                 '%s_%s'%(aggname,feature_name),
                                                 cpmeas.COLTYPE_FLOAT)]
                        self.object_col_order[feature_name] = c
                        c+=1
                        statement += ',\n%s %s'%(mappings[feature_name], ftype)
            statement += ',\nPRIMARY KEY (ImageNumber, ObjectNumber) )'
            
            if create_database:
                execute(cursor, 'DROP TABLE IF EXISTS %s'%(object_table))
                execute(cursor, statement)
        
        # Image table
        statement = 'CREATE TABLE '+image_table+' (\n'
        statement += 'ImageNumber INTEGER'
        c = 1
        for obname, feature, ftype in column_defs+agg_column_defs:
            if obname==cpmeas.IMAGE and not self.ignore_feature(obname, feature):
                if feature not in [d[1] for d in agg_column_defs]:
                    feature_name = '%s_%s'%(obname, feature)
                else:
                    feature_name = feature
                self.image_col_order[feature_name] = c
                statement += ',\n%s %s'%(mappings[feature_name], ftype)
                c+=1
        statement += ',\nPRIMARY KEY (ImageNumber) )'

        if create_database:
            execute(cursor, 'DROP TABLE IF EXISTS %s'%(image_table))
            execute(cursor, statement)
            print 'Commit'
            cursor.connection.commit()
    
    def write_mysql_table_defs(self, workspace, mappings):
        """Returns dictionaries mapping per-image and per-object column names to column #s"""
        
        m_cols = self.get_pipeline_measurement_columns(workspace.pipeline, 
                                                       workspace.image_set_list)
        
        per_image = {"ImageNumber":0}
        per_object = {"ImageNumber":0,"ObjectNumber":1}
        per_image_idx = 1
        per_object_idx = 2
        measurements = workspace.measurements
        file_name_width, path_name_width = self.get_file_path_width(workspace)
        metadata_name_width = 128
        file_name = "%s_SETUP.SQL"%(self.sql_file_prefix)
        path_name = os.path.join(self.get_output_directory(), file_name)
        fid = open(path_name,"wt")
        fid.write("CREATE DATABASE IF NOT EXISTS %s;\n"%(self.db_name.value))
        fid.write("USE %s;\n"%(self.db_name.value))
        fid.write("CREATE TABLE %sPer_Image (ImageNumber INTEGER PRIMARY KEY"%
                  (self.get_table_prefix()))
        for object_name, feature, coltype in m_cols:
            if object_name != cpmeas.IMAGE:
                continue
            if self.ignore_feature(object_name, feature, measurements):
                continue
            feature_name = "%s_%s"%(object_name,feature)
            colname = mappings[feature_name]
            if coltype.upper() == 'FLOAT':
                coltype = 'FLOAT'
            fid.write(",\n%s %s"%(colname, coltype))
            per_image[feature_name] = per_image_idx
            per_image_idx += 1
        #
        # Put mean and std dev measurements for objects in the per_image table
        #
        for aggname in self.agg_names:
            for object_name in workspace.measurements.get_object_names():
                if object_name == 'Image':
                    continue
                for feature in measurements.get_feature_names(object_name):
                    if self.ignore_feature(object_name, feature, measurements):
                        continue
                    feature_name = "%s_%s_%s"%(aggname,object_name,feature)
                    colname = mappings[feature_name]
                    fid.write(",\n%s FLOAT"%(colname))
                    per_image[feature_name] = per_image_idx
                    per_image_idx += 1
        fid.write(");\n\n")
        
        #
        # Write out the per-object table
        #
        if self.objects_choice != O_NONE:
            fid.write("""CREATE TABLE %sPer_Object(
ImageNumber INTEGER,
ObjectNumber INTEGER"""%(self.get_table_prefix()))
            for object_name in workspace.measurements.get_object_names():
                if object_name == 'Image':
                    continue
                for feature in measurements.get_feature_names(object_name):
                    if self.ignore_feature(object_name, feature, measurements, True):
                        continue
                    feature_name = '%s_%s'%(object_name,feature)
                    fid.write(",\n%s FLOAT"%(mappings[feature_name]))
                    per_object[feature_name]=per_object_idx
                    per_object_idx += 1
            fid.write(""",
PRIMARY KEY (ImageNumber, ObjectNumber));

LOAD DATA LOCAL INFILE '%s_image.CSV' REPLACE INTO TABLE %sPer_Image 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';

LOAD DATA LOCAL INFILE '%s_object.CSV' REPLACE INTO TABLE %sPer_Object 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\\\';
"""%(self.base_name(workspace),self.get_table_prefix(),
     self.base_name(workspace),self.get_table_prefix()))
        return per_image, per_object
    
    def write_mysql_table_per_well(self, workspace, mappings):
        file_name = "%s_Per_Well_SETUP.SQL"%(self.sql_file_prefix)
        path_name = os.path.join(self.get_output_directory(), file_name)
        fid = open(path_name,"wt")
        fid.write("CREATE DATABASE IF NOT EXISTS %s;\n"%(self.db_name.value))
        fid.write("USE %s;\n"%(self.db_name.value))
        table_prefix = self.get_table_prefix()
        #
        # Do in two passes. Pass # 1 makes the column name mappings for the
        # well table. Pass # 2 writes the SQL
        #
        for aggname in self.agg_well_names:
            measurements = workspace.measurements
            well_mappings = ColumnNameMapping()
            for do_mapping, do_write in ((True, False),(False, True)):
                if do_write:
                    fid.write("CREATE TABLE %sPer_Well_%s AS SELECT " %
                              (self.get_table_prefix(), aggname))
                for object_name in workspace.measurements.get_object_names():
                        if object_name == 'Image':
                            continue
                        for feature in measurements.get_feature_names(object_name):
                            if self.ignore_feature(object_name, feature, measurements):
                                continue
                            feature_name = "%s_%s"%(object_name,feature)
                            colname = mappings[feature_name]
                            well_colname = "%s_%s" % (aggname, colname)
                            if do_mapping:
                                well_mappings.add(well_colname)
                            if do_write:
                                fid.write("%s(OT.%s) as %s,\n" %
                                          (aggname, colname, 
                                           well_mappings[well_colname]))
            fid.write("""IT.Image_Metadata_Plate, IT.Image_Metadata_Well
            FROM %sPer_Image IT JOIN %sPer_Object OT 
            ON IT.ImageNumber = OT.ImageNumber
            GROUP BY IT.Image_Metadata_Plate, IT.Image_Metadata_Well;\n\n""" %
                      (table_prefix, table_prefix))
        fid.close()
    
    def write_oracle_table_defs(self, workspace, mappings):
        raise NotImplementedError("Writing to an Oracle database is not yet supported")
    
    def base_name(self,workspace):
        """The base for the output file name"""
        m = workspace.measurements
        first = m.image_set_start_number
        last = m.image_set_number
        return '%s%d_%d'%(self.sql_file_prefix, first, last)
    
    
#    def write_data(self, workspace, mappings, per_image, per_object):
#        """Write the data in the measurements out to the csv files
#        workspace - contains the measurements
#        mappings  - map a feature name to a column name
#        per_image - map a feature name to its column index in the per_image table
#        per_object - map a feature name to its column index in the per_object table
#        """
#        measurements = workspace.measurements
#        image_filename = os.path.join(self.get_output_directory(),
#                                      '%s_image.CSV'%(self.base_name(workspace)))
#        object_filename = os.path.join(self.get_output_directory(),
#                                       '%s_object.CSV'%(self.base_name(workspace)))
#        fid_per_image = open(image_filename,"wt")
#        csv_per_image = csv.writer(fid_per_image)
#        fid_per_object = open(object_filename,"wt")
#        csv_per_object = csv.writer(fid_per_object)
#        
#        per_image_cols = max(per_image.values())+1
#        per_object_cols = max(per_object.values())+1
#        
#        image_rows, object_rows = self.get_measurement_rows(measurements, per_image, per_object)
#        
#        print 'write data'
#        print image_rows
#        print object_rows
#        
#        for row in image_rows:
#            csv_per_image.writerow(row)
#        for row in object_rows:
#            csv_per_object.writerow(row)
#        
#        fid_per_image.close()
#        fid_per_object.close()
        
        
        
    def write_data(self, workspace, mappings, per_image, per_object):
        """Write the data in the measurements out to the csv files
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        per_image - map a feature name to its column index in the per_image table
        per_object - map a feature name to its column index in the per_object table
        """
        zeros_for_nan = False
        measurements = workspace.measurements
        image_filename = os.path.join(self.get_output_directory(),
                                      '%s_image.CSV'%(self.base_name(workspace)))
        object_filename = os.path.join(self.get_output_directory(),
                                       '%s_object.CSV'%(self.base_name(workspace)))
        fid_per_image = open(image_filename,"wb")
        if self.objects_choice != O_NONE:
            fid_per_object = open(object_filename,"wb")
            csv_per_object = csv.writer(fid_per_object, lineterminator='\n')
        
        per_image_cols = max(per_image.values())+1
        per_object_cols = max(per_object.values())+1
        for i in range(measurements.image_set_index+1):
            # Loop once per image set
            image_row = [None for k in range(per_image_cols)]
            image_number = i+measurements.image_set_start_number
            image_row[per_image['ImageNumber']] = image_number
            #
            # Fill in the image table
            #
            #
            # The individual feature measurements
            #
            max_count = 0
            max_rows = 0
            for feature in measurements.get_feature_names('Image'):
                if self.ignore_feature('Image', feature, measurements):
                    continue
                feature_name = "%s_%s"%('Image',feature)
                value = measurements.get_measurement('Image',feature, i)
                if isinstance(value, np.ndarray):
                    if value.dtype.kind in ('O','S','U'):
                        if value[0] is None:
                            value = "NULL"
                        else:
                            value = '"'+MySQLdb.escape_string(value[0])+'"'
                    elif np.isnan(value[0]):
                        value = "NULL"
                    else:
                        value = value[0]
                elif isinstance(value, str) or isinstance(value, unicode):
                    value = '"'+MySQLdb.escape_string(value)+'"'
                elif np.isnan(value):
                    value = "NULL"
                    
                image_row[per_image[feature_name]] = value
                if feature_name.startswith('Image_Count_'):
                    max_count = max(max_count,int(value))
                    object_name = feature_name[len('Image_Count_'):]
                    if not self.ignore_object(object_name, True):
                        max_rows = max(max_rows, int(value))
            if max_count == 0:
                for object_name in measurements.get_object_names():
                    if object_name == 'Image':
                        continue
                    for feature in measurements.get_feature_names(object_name):
                        if self.ignore_feature(object_name, feature, measurements):
                            continue
                        for agg_name in self.agg_names:
                            feature_name = "%s_%s_%s"%(agg_name,object_name, feature)
                            image_row[per_image[feature_name]] = 0
            else:
                #
                # The aggregate measurements
                #
                agg_dict = measurements.compute_aggregate_measurements(
                    i, self.agg_names)
                for feature_name in agg_dict.keys():
                    value = agg_dict[feature_name]
                    if np.isnan(value):
                        value = "NULL"
                    image_row[per_image[feature_name]] = value
                #
                # Allocate an array for the per_object values
                #
                object_rows = np.zeros((max_count,per_object_cols))
                object_rows[:,per_object['ImageNumber']] = image_number
                object_rows[:,per_object['ObjectNumber']] = np.array(range(max_count))+1
                #
                # Loop through the objects, collecting their values
                #
                if self.objects_choice != O_NONE and max_rows > 0:
                    for object_name in measurements.get_object_names():
                        if (object_name == 'Image' or
                            self.ignore_object(object_name, True)):
                            continue
                        for feature in measurements.get_feature_names(object_name):
                            if self.ignore_feature(object_name, feature, measurements):
                                continue
                            feature_name = "%s_%s"%(object_name, feature)
                            values = measurements.get_measurement(object_name, feature, i)
                            if zeros_for_nan:
                                values[np.logical_not(np.isfinite(values))] = 0
                            nvalues = np.product(values.shape)
                            if (nvalues < max_rows):
                                sys.stderr.write("Warning: too few measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_rows))
                            elif nvalues > max_count:
                                sys.stderr.write("Warning: too many measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_rows))
                                values = values[:max_rows]
                            object_rows[:nvalues,per_object[feature_name]] = values
                    for row in range(max_rows):
                        row_values = ["NULL" if np.isnan(value) else value
                                      for value in object_rows[row,:]]
                        csv_per_object.writerow(row_values)
            fid_per_image.write(','.join([str(x) for x in image_row])+"\n")
        fid_per_image.close()
        if self.objects_choice != O_NONE:
            fid_per_object.close()
        
    def write_data_to_db(self, workspace, mappings):
        """Write the data in the measurements out to the database
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        """
        zeros_for_nan = False
        measurements = workspace.measurements
        measurement_cols = self.get_pipeline_measurement_columns(workspace.pipeline,
                                                                 workspace.image_set_list)
        index = measurements.image_set_index
        
        # TODO:
        # Check that all image and object columns reported by 
        #  get_measurement_columns agree with measurements.get_feature_names
#        for obname, col in self.col_dict.items():
#            f1 = measurements.get_feature_names(obname)
#            f2 = [c[1] for c in self.col_dict[obname]]
#            diff = set(f1).difference(set(f2))
#            assert not diff, 'The following columns were returned by measurements.get_feature_names and not pipeline.get_measurements: \n %s'%(diff)
#            diff = set(f2).difference(set(f1))
#            assert not diff, 'The following columns were returned by pipeline.get_measurements and not measurements.get_feature_names: \n %s'%(diff)
        
        # Fill image row with non-aggregate cols    
        max_count = 0
        max_rows = 0
        image_number = index + measurements.image_set_start_number
        image_row = [None for k in range(len(self.image_col_order)+1)]
        image_row[0] = (image_number, cpmeas.COLTYPE_INTEGER, 'ImageNumber')
        if cpmeas.IMAGE in self.col_dict.keys():
            for m_col in self.col_dict[cpmeas.IMAGE]:
                feature_name = "%s_%s"%(cpmeas.IMAGE, m_col[1])
                value = measurements.get_measurement(cpmeas.IMAGE, m_col[1], index)
                if isinstance(value, np.ndarray):
                    value=value[0]
                if isinstance(value, float) and not np.isfinite(value) and zeros_for_nan:
                    value = 0
                if feature_name in self.image_col_order.keys():
                    image_row[self.image_col_order[feature_name]] =\
                             (value, m_col[2], feature_name)
                    if feature_name.startswith('Image_Count_'):
                        max_count = max(max_count,int(value))
                        object_name = feature_name[len('Image_Count_'):]
                        if not self.ignore_object(object_name, True):
                            max_rows = max(max_rows, int(value))
        
        # The object columns in order
        object_cols = (['ImageNumber','ObjectNumber'] + 
                       [ None] * len(self.object_col_order))
        for key in self.object_col_order.keys():
            object_cols[self.object_col_order[key]] = key

        if max_count == 0:
            for obname, cols in self.col_dict.items():
                if obname==cpmeas.IMAGE:
                    continue
                for col in cols:
                    for agg_name in self.agg_names:
                        feature_name = "%s_%s_%s"%(agg_name, obname, col[1])
                        if feature_name in self.image_col_order.keys():
                            image_row[self.image_col_order[feature_name]] =\
                                     (0, cpmeas.COLTYPE_FLOAT, feature_name)
            object_rows = []
        else:    
            # Compute and insert the aggregate measurements
            agg_dict = measurements.compute_aggregate_measurements(
                index, self.agg_names)
            for feature_name, value in agg_dict.items():
                if feature_name in self.image_col_order.keys():
                    image_row[self.image_col_order[feature_name]] =\
                             (value, cpmeas.COLTYPE_FLOAT, feature_name)
            
            object_rows = np.zeros((max_rows, len(self.object_col_order)+2), 
                                   dtype=object)
            
            for i in xrange(max_rows):
                object_rows[i,0] = (image_number, cpmeas.COLTYPE_INTEGER)
                object_rows[i,1] = (i+1, cpmeas.COLTYPE_INTEGER)

            # Loop through the object columns, setting all object values for each column
            for obname, cols in self.col_dict.items():
                if obname==cpmeas.IMAGE or obname==cpmeas.EXPERIMENT:
                    continue
                for _, feature, ftype in cols:
                    feature_name = "%s_%s"%(obname, feature)
                    values = measurements.get_measurement(obname, feature, index)
                    if zeros_for_nan:
                        values[np.logical_not(np.isfinite(values))] = 0
                    nvalues = np.product(values.shape)
                    if (nvalues < max_rows):
                        sys.stderr.write("Warning: too few measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_rows))
                        new_values = np.zeros(max_rows, dtype=values.dtype)
                        new_values[:nvalues] = values.flatten()
                        values = new_values
                    elif nvalues > max_rows:
                        sys.stderr.write("Warning: too many measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_count))
                        values = values[:max_rows]
                    for i in xrange(max_rows):
                        object_rows[i,self.object_col_order[feature_name]] = (values[i], cpmeas.COLTYPE_FLOAT)
        
        # wrap non-numeric types in quotes
        image_row_formatted = [("NULL" if np.isnan(val)
                                else str(val))
                               if dtype in [cpmeas.COLTYPE_FLOAT, cpmeas.COLTYPE_INTEGER]
                               else "NULL" if val is None
                               else "'%s'"%MySQLdb.escape_string(str(val)) 
                               for val, dtype, colname in image_row]
        
        image_table = self.get_table_prefix()+'Per_Image'
        object_table = self.get_table_prefix()+'Per_Object'
        #
        # Delete any prior data for this image
        #
        for table_name in (image_table, object_table):
            stmt = ('DELETE FROM %s WHERE ImageNumber=%d'%
                    (table_name, image_number))
            execute(self.cursor, stmt)
        
        stmt = ('INSERT INTO %s (%s) VALUES (%s)' % 
                (image_table, 
                 ','.join([mappings[colname] for val, dtype, colname in image_row]),
                 ','.join([str(v) for v in image_row_formatted])))
        execute(self.cursor, stmt)
        stmt = ('INSERT INTO %s (%s) VALUES (%s)'%
                (object_table,
                 ','.join([mappings[col] for col in object_cols]),
                 ','.join(['%s']*len(object_cols))))

        if self.db_type == DB_MYSQL:
            # Write 25 rows at a time (to get under the max_allowed_packet limit)
            for i in range(0,len(object_rows), 25):
                my_rows = object_rows[i:min(i+25, len(object_rows))]
                self.cursor.executemany(stmt,[ [ None if np.isnan(v)
                                                 else str(v) for v,t in ob_row] 
                                               for ob_row in my_rows])
        else:
            for row in object_rows:
                row_stmt = stmt % tuple([None if np.isnan(v) else str(v) 
                                         for v,t in row])
                self.cursor.execute(row_stmt)
        self.connection.commit()
        
    
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
        path = os.path.join(self.get_output_directory(), filename)
        fid = open(path,'wt')
        date = datetime.datetime.now().ctime()
        db_type = (self.db_type == DB_MYSQL and 'mysql') or (self.db_type == DB_SQLITE and 'sqlite') or 'oracle_not_supported'
        db_port = (self.db_type == DB_MYSQL and 3306) or (self.db_type == DB_ORACLE and 1521) or ''
        db_host = 'imgdb01'
        db_pwd  = ''
        db_name = self.db_name
        db_user = 'cpuser'
        db_sqlite_file = (self.db_type == DB_SQLITE and self.get_output_directory()+'/'+self.sqlite_file.value) or ''
        if self.db_type != DB_SQLITE:
            db_info =  'db_type      = %(db_type)s\n'%(locals())
            db_info += 'db_port      = %(db_port)d\n'%(locals())
            db_info += 'db_host      = %(db_host)s\n'%(locals())
            db_info += 'db_name      = %(db_name)s\n'%(locals())
            db_info += 'db_user      = %(db_user)s\n'%(locals())
            db_info += 'db_passwd    = %(db_pwd)s'%(locals())
        else:
            db_info =  'db_type         = %(db_type)s\n'%(locals())
            db_info += 'db_sqlite_file  = %(db_sqlite_file)s'%(locals())
        
        
        spot_tables = '%sPer_Image'%(self.get_table_prefix())
        cell_tables = '%sPer_Object'%(self.get_table_prefix())
        unique_id = 'ImageNumber'
        object_count = 'Image_Count_%s'%(supposed_primary_object)
        object_id = 'ObjectNumber'
        cell_x_loc = '%s_Location_Center_X'%(supposed_primary_object)
        cell_y_loc = '%s_Location_Center_Y'%(supposed_primary_object)
        image_channel_file_names = ','.join(['Image_FileName_%s'%(name) for name in image_names])+','
        image_channel_file_paths = ','.join(['Image_PathName_%s'%(name) for name in image_names])+','
        image_channel_names = ','.join(image_names)+','
        if len(image_names) == 1:
            image_channel_colors = 'gray,'
        else:
            image_channel_colors = 'red,green,blue,cyan,magenta,yellow,gray,none,none,none,'
        # TODO: leave blank if image files are local  
        image_url = 'http://imageweb/images/CPALinks'
        contents = """#%(date)s
# ==============================================
#
# Classifier 2.0 properties file
#
# ==============================================

# ==== Database Info ====
%(db_info)s

# ==== Database Tables ====
image_table   = %(spot_tables)s
object_table  = %(cell_tables)s

# ==== Database Columns ====
image_id      = %(unique_id)s
object_id     = %(object_id)s
cell_x_loc    = %(cell_x_loc)s
cell_y_loc    = %(cell_y_loc)s

# ==== Image Path and File Name Columns ====
# Here you specify the DB columns from your "image_table" that specify the image paths and file names.
# NOTE: These lists must have equal length!
image_channel_paths = %(image_channel_file_paths)s
image_channel_files = %(image_channel_file_names)s

# Give short names for each of the channels (respectively)...
image_channel_names = %(image_channel_names)s

# ==== Image Accesss Info ====
image_url_prepend = %(image_url)s

# ==== Dynamic Groups ====
# Here you can define groupings to choose from when classifier scores your experiment.  (eg: per-well)
# This is OPTIONAL, you may leave "groups = ".
# FORMAT:
#   groups     =  comma separated list of group names (MUST END IN A COMMA IF THERE IS ONLY ONE GROUP)
#   group_XXX  =  MySQL select statement that returns image-keys and group-keys.  This will be associated with the group name "XXX" from above.
# EXAMPLE GROUPS:
#   groups               =  Well, Gene, Well+Gene,
#   group_SQL_Well       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Per_Image_Table.well FROM Per_Image_Table
#   group_SQL_Gene       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well
#   group_SQL_Well+Gene  =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.well, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well

groups  =  

# ==== Image Filters ====
# Here you can define image filters to let you select objects from a subset of your experiment when training the classifier.
# This is OPTIONAL, you may leave "filters = ".
# FORMAT:
#   filters         =  comma separated list of filter names (MUST END IN A COMMA IF THERE IS ONLY ONE FILTER)
#   filter_SQL_XXX  =  MySQL select statement that returns image keys you wish to filter out.  This will be associated with the filter name "XXX" from above.
# EXAMPLE FILTERS:
#   filters           =  EMPTY, CDKs,
#   filter_SQL_EMPTY  =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene="EMPTY"
#   filter_SQL_CDKs   =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene REGEXP 'CDK.*'

filters  =  

# ==== Meta data ====
# What are your objects called?
# FORMAT:
#   object_name  =  singular object name, plural object name,
object_name  =  cell, cells,

# ==== Excluded Columns ====
# DB Columns the classifier should exclude:
classifier_ignore_substrings  =  table_number_key_column, image_number_key_column, object_number_key_column

# ==== Other ====
# Specify the approximate diameter of your objects in pixels here.
image_tile_size   =  50

# ==== Internal Cache ====
# It shouldn't be necessary to cache your images in the application, but the cache sizes can be set here.
# (Units = 1 image. ie: "image_buffer_size = 100", will cache 100 images before it starts replacing old ones.
image_buffer_size = 1
tile_buffer_size  = 1
image_channel_colors = %(image_channel_colors)s
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
    
    def get_output_directory(self):
        if (self.use_default_output_directory.value or
            self.output_directory == cps.DO_NOT_USE):
            return cpp.get_default_output_directory()
        elif self.output_directory.value.startswith("."+os.path.sep):
            return os.path.join(cpp.get_default_output_directory(),
                                self.output_directory.value[2:])
        else:
            return self.output_directory.value
    
    def get_table_prefix(self):
        if self.want_table_prefix.value:
            return self.table_prefix.value
        return ""
    
    def get_pipeline_measurement_columns(self, pipeline, image_set_list):
        '''Get the measurement columns for this pipeline, possibly cached'''
        d = self.get_dictionary(image_set_list)
        if not d.has_key(D_MEASUREMENT_COLUMNS):
            d[D_MEASUREMENT_COLUMNS] = pipeline.get_measurement_columns()
        return d[D_MEASUREMENT_COLUMNS]
    
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name, from_matlab):
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
        elif from_matlab and variable_revision_number == 10:
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
            new_setting_values += [ 'imgdb01','cpuser','cPus3r']
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
            new_setting_values = settings_values
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
        valid_name_regexp = "^[0-9a-zA-Z_$]+$"
        for key,value in self.__dictionary.iteritems():
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
                    for index in range(len(name)-1,-1,-1):
                        if name[index] in to_drop:
                            name = name[:index]+name[index+1:]
                            remove_count += 1
                            if remove_count == to_remove:
                                break
                    if remove_count == to_remove:
                        break
    
                while name in reverse_dictionary.keys():
                    # if, improbably, removing the vowels hit an existing name
                    # try deleting random characters
                    name = starting_name
                    while len(name) > self.__max_len:
                        index = int(random.uniform(0,len(name)))
                        name = name[:index]+name[index+1:]
            reverse_dictionary.pop(orig_name)
            reverse_dictionary[name] = key
            self.__dictionary[key] = name
