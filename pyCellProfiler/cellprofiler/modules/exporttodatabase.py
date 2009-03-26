"""ExportToDatabase -  export measurements to database

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision$"

import csv
import datetime
import numpy as np
import os
import random
import re
import sys

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp

DB_MYSQL = "MySQL"
DB_ORACLE = "Oracle"

AGG_MEAN = "Mean"
AGG_STD_DEV= "StDev"
AGG_NAMES = [AGG_MEAN,AGG_STD_DEV]

class ExportToDatabase(cpm.CPModule):
    """% SHORT DESCRIPTION:
Exports data in database readable format, including an importing file
with column names and a CellProfiler Analyst properties file, if desired.
*************************************************************************

This module exports measurements to a SQL compatible format. It creates
MySQL or Oracle scripts and associated data files which will create a
database and import the data into it and gives you the option of creating
a properties file for use with CellProfiler Analyst. 
 
This module must be run at the end of a pipeline, or second to last if 
you are using the CreateBatchFiles module. If you forget this module, you
can also run the ExportDatabase data tool after processing is complete; 
its functionality is the same.

The database is set up with two primary tables. These tables are the
Per_Image table and the Per_Object table (which may have a prefix if you
specify). The Per_Image table consists of all the Image measurements and
the Mean and Standard Deviation of the object measurements. There is one
Per_Image row for every image. The Per_Object table contains all the
measurements for individual objects. There is one row of object
measurements per object identified. The two tables are connected with the
primary key column ImageNumber. The Per_Object table has another primary
key called ObjectNumber, which is unique per image.

The Oracle database has an extra table called Column_Names. This table is
necessary because Oracle has the unfortunate limitation of not being able
to handle column names longer than 32 characters. Since we must
distinguish many different objects and measurements, our column names are
very long. This required us to create a separate table which contains a
short name and corresponding long name. The short name is simply "col"
with an attached number, such as "col1" "col2" "col3" etc. The short name
has a corresponding long name such as "Nuclei_AreaShape_Area". Each of
the Per_Image and Per_Object columnnames are loaded as their "short name"
but the long name can be determined from the Column_Names table.

Settings:

Database Type: 
You can choose to export MySQL or Oracle database scripts. The exported
data is the same for each type, but the setup files for MySQL and Oracle
are different.

Database Name: 
  In MySQL, you can enter the name of a database to create or the name of
an existing database. When using the script, if the database already
exists, the database creation step will be skipped so the existing
database will not be overwritten but new tables will be added. Do be
careful, however, in choosing the Table Prefix. If you use an existing
table name, you might unintentionally overwrite the data in that table.
  In Oracle, when you log in you must choose a database to work with, so
there is no need to specify the database name in this module. This also
means it is impossible to create/destroy a database with these
CellProfiler scripts.

Table Prefix: 
Here you can choose what to append to the table names Per_Image and
Per_Object. If you choose "Do not use", no prefix will be appended. If you choose
a prefix, the tables will become PREFIX_Per_Image and PREFIX_Per_Object
in the database. If you are using the same database for all of your
experiments, the table prefix is necessary and will be the only way to
distinguish different experiments. If you are creating a new database for
every experiment, then it may be easier to keep the generic Per_Image and
Per_Object table names. Be careful when choosing the table prefix, since
you may unintentionally overwrite existing tables.

SQL File Prefix: All the CSV files will start with this prefix.

Create a CellProfiler Analyst properties file: Generate a template
properties for using your new database in CellProfiler Analyst (a data
exploration tool which can also be downloaded from
http://www.cellprofiler.org/)
 
If creating a properties file for use with CellProfiler Analyst (CPA): 
The module will attempt to fill in as many as the entries as possible 
based on the current handles structure. However, entries such as the 
server name, username and password are omitted. Hence, opening the 
properties file in CPA will produce an error since it won't be able to
connect to the server. However, you can still edit the file in CPA and
then fill in the required information.

********************* How To Import MySQL *******************************
Step 1: Log onto the server where the database will be located.

Step 2: From within a terminal logged into that server, navigate to folder 
where the CSV output files and the SETUP script is located.

Step 3: Type the following within the terminal to log into MySQL on the 
server where the database will be located:
   mysql -uUsername -pPassword -hHost

Step 4: Type the following within the terminal to run SETUP script: 
     \. DefaultDB_SETUP.SQL

The SETUP file will do everything necessary to load the database.

********************* How To Import Oracle ******************************
Step 1: Using a terminal, navigate to folder where the CSV output files
and the SETUP script is located.

Step 2: Log into SQLPlus: "sqlplus USERNAME/PASSWORD@DATABASESCRIPT"
You may need to ask your IT department the name of DATABASESCRIPT.

Step 3: Run SETUP script: "@DefaultDB_SETUP.SQL"

Step 4: Exit SQLPlus: "exit"

Step 5: Load data files (for columnames, images, and objects):

sqlldr USERNAME/PASSWORD@DATABASESCRIPT control=DefaultDB_LOADCOLUMNS.CTL
sqlldr USERNAME/PASSWORD@DATABASESCRIPT control=DefaultDB_LOADIMAGE.CTL
sqlldr USERNAME/PASSWORD@DATABASESCRIPT control=DefaultDB_LOADOBJECT.CTL

Step 6: Log into SQLPlus: "sqlplus USERNAME/PASSWORD@DATABASESCRIPT"

Step 7: Run FINISH script: "@DefaultDB_FINISH.SQL"
"""
    variable_revision_number = 6
    category = "File Processing"
    
    def create_settings(self):
        self.module_name = "ExportToDatabase"
        self.database_type = cps.Choice("What type of database do you want to use?",
                                        [DB_MYSQL,DB_ORACLE],DB_MYSQL)
        self.database_name = cps.Text("What is the name of the database you want to use?","DefaultDB")
        self.want_table_prefix = cps.Binary("Do you want to add a prefix to your table names? (Use the default table names unless checked)",False)
        self.table_prefix = cps.Text("What is the table prefix you want to use?","Expt")
        self.file_prefix = cps.Text("What prefix do you want to use to name the SQL files?","SQL_")
        self.use_default_output_directory = cps.Binary("Do you want to save files in the default output directory?",True)
        self.output_directory = cps.Text("What directory should be used to save files?",".")
        self.save_cpa_properties = cps.Binary("Do you want to create a CellProfilerAnalyst properties file?",False)
    
    def visible_settings(self):
        result = [self.database_type]
        if self.database_type == DB_MYSQL:
            result.append(self.database_name)
        result.append(self.want_table_prefix)
        if self.want_table_prefix.value:
            result.append(self.table_prefix)
        result.append(self.file_prefix)
        result.append(self.use_default_output_directory)
        if not self.use_default_output_directory.value:
            result.append(self.output_directory)
        result.append(self.save_cpa_properties)
        return result
    
    def settings(self):
        return [self.database_type, self.database_name,self.want_table_prefix,
                self.table_prefix, self.file_prefix, 
                self.use_default_output_directory, self.output_directory,
                self.save_cpa_properties]
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
                                module_name,from_matlab):
        if from_matlab and variable_revision_number == 6:
            new_setting_values = [setting_values[0],setting_values[1]]
            if setting_values[2] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
                new_setting_values.append("Expt")
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
            setting_values = new_setting_values
        return setting_values, variable_revision_number, from_matlab
    
    def test_valid(self,pipeline):
        if self.want_table_prefix.value:
            # Test the table prefix
            if not self.table_prefix.value[0].isalpha():
                raise ValidationError("The table prefix must begin with a letter", self.table_prefix)
            if not re.match("^[A-Za-z][A-Za-z0-9_]+$",self.table_prefix):
                raise ValidationError("The table prefix has invalid characters",self.table_prefix)
            
    def run(self, workspace):
        if (workspace.measurements.image_set_number+1 !=
            workspace.image_set_list.count()):
            return
        
        mappings = self.get_column_name_mappings(workspace)
        if self.database_type == DB_MYSQL:
            per_image, per_object = self.write_mysql_table_defs(workspace, mappings)
        else:
            per_image, per_object = self.write_oracle_table_defs(workspace, mappings)
        self.write_data(workspace,mappings, per_image, per_object)
        if self.save_cpa_properties.value:
            self.write_properties(workspace)
    
    def ignore_object(self,object_name):
        """Ignore objects (other than 'Image') if this returns true"""
        if object_name in ('Experiment','Neighbors'):
            return True
    def ignore_feature(self, measurements, object_name, feature_name):
        """Return true if we should ignore a feature"""
        
        if self.ignore_object(object_name):
            return True
        if measurements.has_feature(object_name, "SubObjectFlag"):
            return True
        if feature_name.startswith('Description_'):
            return True
        if feature_name.startswith('ModuleError_'):
            return True
        if feature_name.startswith('TimeElapsed_'):
            return True
        return False
    
    def get_column_name_mappings(self,workspace):
        """Scan all the feature names in the measurements, creating column names"""
        measurements = workspace.measurements
        mappings = ColumnNameMapping()
        for object_name in measurements.get_object_names():
            for feature_name in measurements.get_feature_names(object_name):
                if self.ignore_feature(measurements, object_name, feature_name):
                    continue
                mappings.add("%s_%s"%(object_name,feature_name))
                if object_name != 'Image':
                    for agg_name in AGG_NAMES:
                        mappings.add('%s_%s_%s'%(agg_name, object_name, feature_name))
        return mappings
    
    def write_mysql_table_defs(self, workspace, mappings):
        """Returns dictionaries mapping per-image and per-object column names to column #s"""
        per_image = {"ImageNumber":0}
        per_object = {"ImageNumber":0,"ObjectNumber":1}
        per_image_idx = 1
        per_object_idx = 2
        measurements = workspace.measurements
        file_name_width, path_name_width = self.get_file_path_width(workspace)
        metadata_name_width = 128
        file_name = "%s_SETUP.SQL"%(self.file_prefix)
        path_name = os.path.join(self.get_output_directory(), file_name)
        fid = open(path_name,"wt")
        fid.write("CREATE DATABASE IF NOT EXISTS %s;\n"%(self.database_name.value))
        fid.write("USE %s;\n"%(self.database_name.value))
        fid.write("CREATE TABLE %sPer_Image (ImageNumber INTEGER PRIMARY KEY"%
                  (self.get_table_prefix()))
        for feature in measurements.get_feature_names('Image'):
            if self.ignore_feature(measurements, 'Image', feature):
                continue
            feature_name = "%s_%s"%('Image',feature)
            colname = mappings[feature_name]
            if feature.startswith('FileName'):
                fid.write(",\n%s VARCHAR(%d)"%(colname,file_name_width))
            elif feature.find('Path')!=-1:
                fid.write(",\n%s VARCHAR(%d)"%(colname,path_name_width))
            elif feature.startswith('MetaData'):
                fid.write(",\n%s VARCHAR(%d)"%(colname,metadata_name_width))
            else:
                fid.write(",\n%s FLOAT NOT NULL"%(colname))
            per_image[feature_name] = per_image_idx
            per_image_idx += 1
        #
        # Put mean and std dev measurements for objects in the per_image table
        #
        for aggname in AGG_NAMES:
            for object_name in workspace.measurements.get_object_names():
                if object_name == 'Image':
                    continue
                for feature in measurements.get_feature_names(object_name):
                    if self.ignore_feature(measurements,object_name, feature):
                        continue
                    feature_name = "%s_%s_%s"%(aggname,object_name,feature)
                    colname = mappings[feature_name]
                    fid.write(",\n%s FLOAT NOT NULL"%(colname))
                    per_image[feature_name] = per_image_idx
                    per_image_idx += 1
        fid.write(");\n\n")
        #
        # Write out the per-object table
        #
        fid.write("""CREATE TABLE %sPer_Object(
ImageNumber INTEGER,
ObjectNumber INTEGER"""%(self.get_table_prefix()))
        for object_name in workspace.measurements.get_object_names():
            if object_name == 'Image':
                continue
            for feature in measurements.get_feature_names(object_name):
                if self.ignore_feature(measurements,object_name, feature):
                    continue
                feature_name = '%s_%s'%(object_name,feature)
                fid.write(",\n%s FLOAT NOT NULL"%(mappings[feature_name]))
                per_object[feature_name]=per_object_idx
                per_object_idx += 1
        fid.write(""",
PRIMARY KEY (ImageNumber, ObjectNumber));

LOAD DATA LOCAL INFILE '%s_image.CSV' REPLACE INTO TABLE %sPer_Image 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '';

LOAD DATA LOCAL INFILE '%s_object.CSV' REPLACE INTO TABLE %sPer_Object 
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"' ESCAPED BY '';
"""%(self.base_name(workspace),self.get_table_prefix(),
     self.base_name(workspace),self.get_table_prefix()))
        fid.close()
        return per_image, per_object
    
    def write_oracle_table_defs(self, workspace, mappings):
        raise NotImplementedError("Writing to an Oracle database is not yet supported")
    
    def base_name(self,workspace):
        """The base for the output file name"""
        return '%s%d_%d'%(self.file_prefix,1,workspace.image_set_list.count())
    
    def write_data(self, workspace, mappings, per_image, per_object):
        """Write the data in the measurements out to the csv files
        workspace - contains the measurements
        mappings  - map a feature name to a column name
        per_image - map a feature name to its column index in the per_image table
        per_object - map a feature name to its column index in the per_object table
        """
        image_filename = os.path.join(self.get_output_directory(),
                                      '%s_image.CSV'%(self.base_name(workspace)))
        object_filename = os.path.join(self.get_output_directory(),
                                       '%s_object.CSV'%(self.base_name(workspace)))
        fid_per_image = open(image_filename,"wt")
        csv_per_image = csv.writer(fid_per_image)
        fid_per_object = open(object_filename,"wt")
        csv_per_object = csv.writer(fid_per_object)
        
        per_image_cols = max(per_image.values())+1
        per_object_cols = max(per_object.values())+1
        measurements = workspace.measurements
        for i in range(workspace.image_set_list.count()):
            # Loop once per image set
            image_row = [None for k in range(per_image_cols)]
            image_number = i+1
            image_row[per_image['ImageNumber']] = image_number
            #
            # Fill in the image table
            max_count = 0
            for feature in measurements.get_feature_names('Image'):
                if self.ignore_feature(measurements, 'Image', feature):
                    continue
                feature_name = "%s_%s"%('Image',feature)
                value = measurements.get_measurement('Image',feature, i)
                if isinstance(value, np.ndarray):
                    value=value[0]
                image_row[per_image[feature_name]] = value
                if feature_name.find('Count') != -1:
                    max_count = max(max_count,int(value))
            if max_count == 0:
                for object_name in measurements.get_object_names():
                    if object_name == 'Image':
                        continue
                    for feature in measurements.get_feature_names(object_name):
                        if self.ignore_feature(measurements, object_name, feature):
                            continue
                        for agg_name in AGG_NAMES:
                            feature_name = "%s_%s_%s"%(agg_name,object_name, feature)
                            image_row[per_image[feature_name]] = 0
            else:
                #
                # Allocate an array for the per_object values
                #
                object_rows = np.zeros((max_count,per_object_cols))
                object_rows[:,per_object['ImageNumber']] = image_number
                object_rows[:,per_object['ObjectNumber']] = np.array(range(max_count))+1
                for object_name in measurements.get_object_names():
                    if object_name == 'Image':
                        continue
                    for feature in measurements.get_feature_names(object_name):
                        if self.ignore_feature(measurements, object_name, feature):
                            continue
                        feature_name = "%s_%s"%(object_name, feature)
                        values = measurements.get_measurement(object_name, feature, i)
                        values[np.logical_not(np.isfinite(values))] = 0
                        nvalues = np.product(values.shape)
                        if (nvalues < max_count):
                            sys.stderr.write("Warning: too few measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_count))
                        elif nvalues > max_count:
                            sys.stderr.write("Warning: too many measurements for %s in image set #%d, got %d, expected %d\n"%(feature_name,image_number,nvalues,max_count))
                            values = values[:max_count]
                        object_rows[:nvalues,per_object[feature_name]] = values
                        #
                        # Compute the mean and standard deviation
                        #
                        mean_feature_name = '%s_%s_%s'%(AGG_MEAN,object_name, feature)
                        mean = values.mean()
                        image_row[per_image[mean_feature_name]] = mean
                        stdev_feature_name = '%s_%s_%s'%(AGG_STD_DEV,object_name, feature)
                        stdev = values.std()
                        image_row[per_image[stdev_feature_name]] = stdev
                for row in range(max_count):
                    csv_per_object.writerow(object_rows[row,:])
            csv_per_image.writerow(image_row)
        fid_per_image.close()
        fid_per_object.close()
    
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
        
        filename = '%s_v2.properties'%(self.database_name)
        path = os.path.join(self.get_output_directory(), filename)
        fid = open(path,'wt')
        date = datetime.datetime.now().ctime()
        db_type = (self.database_type == DB_MYSQL and 'mysql') or 'oracle'
        db_port = (self.database_type == DB_MYSQL and 3306) or 1521
        db_host = 'imgdb01'
        db_pwd  = ''
        db_name = self.database_name
        db_user = 'cpuser'
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
        image_url = 'http://imageweb/images/CPALinks'
        contents = """#%(date)s
# ==============================================
#
# Classifier 2.0 properties file
#
# ==============================================

# ==== Database Info ====
db_type      = %(db_type)s
db_port      = %(db_port)d
db_host      = %(db_host)s
db_name      = %(db_name)s
db_user      = %(db_user)s
db_passwd    = %(db_pwd)s

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
                names = m.get_all_measurements('Image',feature)
                FileNameWidth = max(FileNameWidth, np.max(map(len,names)))
            elif feature.startswith('PathName'):
                names = m.get_all_measurements('Image',feature)
                PathNameWidth = max(PathNameWidth, np.max(map(len,names)))
        return FileNameWidth, PathNameWidth
    
    def get_output_directory(self):
        if self.use_default_output_directory.value:
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
        for key,value in self.__dictionary.iteritems():
            reverse_dictionary[value] = key
            if len(value) > self.__max_len:
                problem_names.append(value)
        
        for name in problem_names:
            key = reverse_dictionary[name]
            orig_name = name
            # remove vowels 
            to_remove = len(name)-self.__max_len
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
                name = orig_name
                while len(name) > self.__max_len:
                    index = int(random.uniform(0,len(name)))
                    name = name[:index]+name[index+1:]
            reverse_dictionary.pop(orig_name)
            reverse_dictionary[name] = key
            self.__dictionary[key] = name