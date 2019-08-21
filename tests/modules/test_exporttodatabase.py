"""test_exporttodatabase.py - test the ExportToDatabase module
"""

import base64
import io
import os
import socket
import tempfile
import traceback
import unittest
import uuid

import PIL.Image as PILImage
import numpy as np
from six.moves import StringIO

import cellprofiler.measurement

if hasattr(unittest, "SkipTest"):
    SkipTestException = unittest.SkipTest
else:
    SkipTestException = None

from cellprofiler.preferences import (
    set_headless,
    ABSOLUTE_FOLDER_NAME,
    DEFAULT_OUTPUT_SUBFOLDER_NAME,
)
from cellprofiler.measurement import C_FILE_NAME, C_PATH_NAME

set_headless()

import cellprofiler.module as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.image as cpi
import cellprofiler.workspace as cpw
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.utilities.legacy

import cellprofiler.modules.exporttodatabase as E
import pytest

np.random.seed(9804)

M_CATEGORY = "my"
OBJ_FEATURE = "objmeasurement"
INT_IMG_FEATURE = "int_imagemeasurement"
FLOAT_IMG_FEATURE = "float_imagemeasurement"
STRING_IMG_FEATURE = "string_imagemeasurement"
LONG_IMG_FEATURE = (
    "image_measurement_with_a_column_name_that_exceeds_64_characters_in_width"
)
LONG_OBJ_FEATURE = (
    "obj_measurement_with_a_column_name_that_exceeds_64_characters_in_width"
)
WIERD_IMG_FEATURE = (
    'image_measurement_with_"!@%*\n~!\t\ra\+=' "and other &*^% in it.........."
)
WIERD_OBJ_FEATURE = 'measurement w"!@%*\n~!\t\ra\+=' "and other &*^% in it"
GROUP_IMG_FEATURE = "group_imagemeasurement"
GROUP_OBJ_FEATURE = "group_objmeasurement"
MISSING_FROM_MEASUREMENTS = "Missing from measurements"
MISSING_FROM_MODULE = "Missing from module"

OBJ_MEASUREMENT, INT_IMG_MEASUREMENT, FLOAT_IMG_MEASUREMENT, STRING_IMG_MEASUREMENT, LONG_IMG_MEASUREMENT, LONG_OBJ_MEASUREMENT, WIERD_IMG_MEASUREMENT, WIERD_OBJ_MEASUREMENT, GROUP_IMG_MEASUREMENT, GROUP_OBJ_MEASUREMENT = [
    "_".join((M_CATEGORY, x))
    for x in (
        OBJ_FEATURE,
        INT_IMG_FEATURE,
        FLOAT_IMG_FEATURE,
        STRING_IMG_FEATURE,
        LONG_IMG_FEATURE,
        LONG_OBJ_FEATURE,
        WIERD_IMG_FEATURE,
        WIERD_OBJ_FEATURE,
        GROUP_IMG_FEATURE,
        GROUP_OBJ_FEATURE,
    )
]
OBJECT_NAME = "myobject"
IMAGE_NAME = "myimage"
OBJECT_COUNT_MEASUREMENT = "Count_%s" % OBJECT_NAME

ALTOBJECT_NAME = "altobject"
ALTOBJECT_COUNT_MEASUREMENT = "Count_%s" % ALTOBJECT_NAME

RELATIONSHIP_NAME = "cousin"

INT_VALUE = 10
FLOAT_VALUE = 15.5
STRING_VALUE = "Hello, world"
OBJ_VALUE = np.array([1.5, 3.67, 2.8])
ALTOBJ_VALUE = np.random.uniform(size=100)
PLATE = "P-12345"
WELL = "A01"
DB_NAME = "MyDatabaseName"
DB_HOST = "MyHost"
DB_USER = "MyUser"
DB_PASSWORD = "MyPassword"

MYSQL_HOST = os.environ.get("CP_MYSQL_TEST_HOST", "imgdb02.broadinstitute.org")
MYSQL_DATABASE = os.environ.get("CP_MYSQL_TEST_DB", "CPUnitTest")
MYSQL_PASSWORD = os.environ.get("CP_MYSQL_TEST_PASSWORD", "cPus3r")
if MYSQL_PASSWORD == "None":
    MYSQL_PASSWORD = ""
MYSQL_USER = os.environ.get("CP_MYSQL_TEST_USER", "cpuser")


class TestExportToDatabase:
    def setUp():
        __cursor = None
        __connection = None
        __has_median = None
        try:
            if MYSQL_HOST.endswith("broadinstitute.org"):
                fqdn = socket.getfqdn().lower()
                if (
                    ("broadinstitute" in fqdn)
                    or fqdn.endswith("broad.mit.edu")
                    or fqdn.endswith("broad")
                ):
                    __test_mysql = True
                elif socket.gethostbyaddr(socket.gethostname())[-1][0].startswith(
                    "69.173"
                ):
                    __test_mysql = True
                else:
                    __test_mysql = False
            __test_mysql = True
        except:
            __test_mysql = False

    @property
    def connection():
        if not __test_mysql:
            skipTest("Skipping actual DB work, no DB configured.")
        if __connection is None:
            import MySQLdb

            __connection = MySQLdb.connect(
                host=MYSQL_HOST, user=MYSQL_USER, passwd=MYSQL_PASSWORD, local_infile=1
            )
        return __connection

    def close_connection():
        if __test_mysql and __connection is not None:
            if __cursor is not None:
                __cursor.close()
            __connection.close()
            __connection = None
            __cursor = None

    @property
    def cursor():
        if not __test_mysql:
            skipTest("Skipping actual DB work, database not configured.")
        if __cursor is None:
            import MySQLdb
            from MySQLdb.cursors import SSCursor

            __cursor = SSCursor(connection)
            try:
                __cursor.execute("use " + MYSQL_DATABASE)
            except:
                __cursor.execute("create database " + MYSQL_DATABASE)
                __cursor.execute("use " + MYSQL_DATABASE)
        return __cursor

    @property
    def mysql_has_median():
        """True if MySQL database has a median function"""
        if __has_median is None:
            try:
                cursor = connection.cursor()
                cursor.execute("select median(1)")
                cursor.close()
                __has_median = True
            except:
                __has_median = False
        return __has_median

    def get_sqlite_cursor(module):
        import sqlite3

        assert isinstance(module, E.ExportToDatabase)
        file_name = os.path.join(
            module.directory.get_absolute_path(), module.sqlite_file.value
        )
        connection = sqlite3.connect(file_name)
        cursor = connection.cursor()
        return cursor, connection

    def test_00_write_load_test():
        #
        # If this fails, you need to write a test for your variable revision
        # number change.
        #
        assert E.ExportToDatabase.variable_revision_number == 27

    def test_load_v11():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8952

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    CSV file location:Default Input Folder
    Path to the CSV file:.
    Name of the CSV file:1049.csv
    Load images from CSV data?:Yes
    Image folder location:Default Input Folder
    Path to the images:.
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:No
    Select metadata fields for grouping:

ExportToDatabase:[module_num:2|svn_version:\'8947\'|variable_revision_number:11|show_window:False|notes:\x5B\x5D]
    Database type:MySQL
    Database name:DefaultDB
    Add a prefix to table names?:No
    Table prefix:Expt_
    SQL file prefix:SQL_
    Where do you want to save files?:Custom folder with metadata
    Enter the output folder:./\\g<Plate>
    Create a CellProfiler Analyst properties file?:No
    Store the database in CSV files? :Yes
    Database host:
    Username:
    Password:
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL_CSV
        assert module.directory.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME
        assert module.directory.custom_path == r"./\g<Plate>"
        assert module.sql_file_prefix == "SQL_"
        assert module.db_name == "DefaultDB"

    def test_load_v12():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8952

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    CSV file location:Default Input Folder
    Path to the CSV file:.
    Name of the CSV file:1049.csv
    Load images from CSV data?:Yes
    Image folder location:Default Input Folder
    Path to the images:.
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:No
    Select metadata fields for grouping:

ExportToDatabase:[module_num:2|svn_version:\'8947\'|variable_revision_number:12|show_window:False|notes:\x5B\x5D]
    Database type:MySQL
    Database name:DefaultDB
    Add a prefix to table names?:No
    Table prefix:Expt_
    SQL file prefix:SQL_
    Where do you want to save files?:Custom folder with metadata
    Enter the output folder:./\\g<Plate>
    Create a CellProfiler Analyst properties file?:No
    Database host:
    Username:
    Password:
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.directory.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME
        assert module.directory.custom_path == r"./\g<Plate>"
        assert module.sql_file_prefix == "SQL_"
        assert module.db_name == "DefaultDB"
        assert module.max_column_size == 64

    def test_load_v13():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8952

LoadData:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    CSV file location:Default Input Folder
    Path to the CSV file:.
    Name of the CSV file:1049.csv
    Load images from CSV data?:Yes
    Image folder location:Default Input Folder
    Path to the images:.
    Process just a range of rows?:No
    Rows to process:1,100000
    Group images by metadata?:No
    Select metadata fields for grouping:

ExportToDatabase:[module_num:2|svn_version:\'8947\'|variable_revision_number:13|show_window:False|notes:\x5B\x5D]
    Database type:MySQL
    Database name:DefaultDB
    Add a prefix to table names?:No
    Table prefix:Expt_
    SQL file prefix:SQL_
    Where do you want to save files?:Custom folder with metadata
    Enter the output folder:./\\g<Plate>
    Create a CellProfiler Analyst properties file?:No
    Database host:
    Username:
    Password:
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:61
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.directory.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME
        assert module.directory.custom_path == r"./\g<Plate>"
        assert module.sql_file_prefix == "SQL_"
        assert module.db_name == "DefaultDB"
        assert module.max_column_size == 61

    def test_load_v15():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9503

ExportToDatabase:[module_num:1|svn_version:\'9461\'|variable_revision_number:15|show_window:True|notes:\x5B\x5D]
    Database type:MySQL / CSV
    Database name:Heel
    Add a prefix to table names?:No
    Table prefix:Ouch
    SQL file prefix:LQS_
    Output file location:Elsewhere...\x7C//achilles/red/shoes
    Create a CellProfiler Analyst properties file?:No
    Database host:Zeus
    Username:Hera
    Password:Athena
    Name the SQLite database file:Poseidon
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:Yes
    Calculate the per-well median values of object measurements?:Yes
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:62
    Create one table per object or a single object table?:One table per object type
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL_CSV
        assert module.db_name == "Heel"
        assert not module.want_table_prefix
        assert module.table_prefix == "Ouch"
        assert module.sql_file_prefix == "LQS_"
        assert module.directory.dir_choice == ABSOLUTE_FOLDER_NAME
        assert module.directory.custom_path == "//achilles/red/shoes"
        assert not module.save_cpa_properties
        assert module.db_host == "Zeus"
        assert module.db_user == "Hera"
        assert module.db_passwd == "Athena"
        assert module.sqlite_file == "Poseidon"
        assert module.wants_agg_mean
        assert not module.wants_agg_median
        assert not module.wants_agg_std_dev
        assert module.wants_agg_mean_well
        assert module.wants_agg_median_well
        assert not module.wants_agg_std_dev_well
        assert module.objects_choice == E.O_ALL
        assert module.max_column_size == 62
        assert module.separate_object_tables == E.OT_PER_OBJECT
        assert not module.wants_properties_image_url_prepend

    def test_load_v22():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
SVNRevision:11412

ExportToDatabase:[module_num:1|svn_version:\'11377\'|variable_revision_number:22|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Database type:MySQL
    Database name:Gamma
    Add a prefix to table names?:Yes
    Table prefix:Delta_
    SQL file prefix:Iota_
    Output file location:Default Output Folder\x7CNone
    Create a CellProfiler Analyst properties file?:Yes
    Database host:Alpha
    Username:Beta
    Password:Gamma
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object or a single object table?:Single object table
    Enter an image url prepend if you plan to access your files via http:http\x3A//server.university.edu
    Write image thumbnails directly to the database?:Yes
    Select the images you want to save thumbnails of:Actin,DNA
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:384
    Select the plate metadata:Plate
    Select the well metadata:Well
    Include information for all images, using default values?:No
    Properties image group count:2
    Properties group field count:1
    Properties filter field count:1
    Workspace measurement count:2
    Experiment name:Sigma
    Which objects should be used for locations?:Cells
    Select an image to include:DNA
    Use the image name for the display?:No
    Image name:NucleicAcid
    Channel color:green
    Select an image to include:Actin
    Use the image name for the display?:No
    Image name:Protein
    Channel color:blue
    Do you want to add group fields?:Yes
    Enter the name of the group:WellGroup
    Enter the per-image columns which define the group, separated by commas:Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:Yes
    Automatically create a filter for each plate?:Yes
    Enter the name of the filter:Site1Filter
    Enter the MySQL WHERE clause to define a filter:Image_Metadata_Plate = \'1\'
    Create a CellProfiler Analyst workspace file?:Yes
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Mitochondria
    Select the x-axis measurement:Width_DNA
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Nuclei
    Select the y-axis measurement:Height_DNA
    Select the x-axis index:ImageNumber
    Select the measurement display tool:PlateViewer
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Cells
    Select the x-axis measurement:Height_Actin
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Speckles
    Select the y-axis measurement:Width_Actin
    Select the x-axis index:ImageNumber
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.sql_file_prefix == "Iota_"
        assert module.experiment_name == "Sigma"
        assert module.directory.dir_choice == cps.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.save_cpa_properties
        assert module.location_object == "Cells"
        assert module.properties_image_url_prepend == "http://server.university.edu"
        assert module.properties_plate_type == "384"
        assert module.properties_plate_metadata == "Plate"
        assert module.properties_well_metadata == "Well"
        assert not module.properties_export_all_image_defaults
        assert module.properties_wants_groups
        assert module.properties_wants_filters
        assert module.create_filters_for_plates
        assert module.create_workspace_file
        assert len(module.image_groups) == 2
        for image_group, input_image_name, output_image_name, color in (
            (module.image_groups[0], "DNA", "NucleicAcid", "green"),
            (module.image_groups[1], "Actin", "Protein", "blue"),
        ):
            assert not image_group.wants_automatic_image_name
            assert image_group.image_cols == input_image_name
            assert image_group.image_name == output_image_name
            assert image_group.image_channel_colors == color

        assert len(module.group_field_groups) == 1
        g = module.group_field_groups[0]
        assert g.group_name == "WellGroup"
        assert g.group_statement == "Image_Metadata_Plate, Image_Metadata_Well"

        assert len(module.workspace_measurement_groups) == 2
        for (
            g,
            measurement_display,
            x_measurement_type,
            x_object_name,
            x_measurement_name,
            x_index_name,
            y_measurement_type,
            y_object_name,
            y_measurement_name,
            y_index_name,
        ) in (
            (
                module.workspace_measurement_groups[0],
                "ScatterPlot",
                cpmeas.IMAGE,
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                cpmeas.IMAGE,
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                cpmeas.IMAGE,
                "Cells",
                "Height_Actin",
                "ImageNumber",
                cpmeas.IMAGE,
                "Speckles",
                "Width_Actin",
                "ImageNumber",
            ),
        ):
            assert g.measurement_display == measurement_display
            assert g.x_measurement_type == x_measurement_type
            assert g.x_object_name == x_object_name
            assert g.x_measurement_name == x_measurement_name
            assert g.x_index_name == x_index_name
            assert g.y_measurement_type == y_measurement_type
            assert g.y_object_name == y_object_name
            assert g.y_measurement_name == y_measurement_name
            assert g.y_index_name == y_index_name

        assert len(module.filter_field_groups) == 1
        g = module.filter_field_groups[0]
        assert g.filter_name == "Site1Filter"
        assert g.filter_statement == "Image_Metadata_Plate = '1'"

    def test_load_v23():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
SVNRevision:11412

ExportToDatabase:[module_num:1|svn_version:\'11377\'|variable_revision_number:23|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Database type:MySQL
    Database name:Gamma
    Add a prefix to table names?:Yes
    Table prefix:Delta_
    SQL file prefix:Iota_
    Output file location:Default Output Folder\x7CNone
    Create a CellProfiler Analyst properties file?:Yes
    Database host:Alpha
    Username:Beta
    Password:Gamma
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object or a single object table?:Single object table
    Enter an image url prepend if you plan to access your files via http:http\x3A//server.university.edu
    Write image thumbnails directly to the database?:Yes
    Select the images you want to save thumbnails of:Actin,DNA
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:384
    Select the plate metadata:Plate
    Select the well metadata:Well
    Include information for all images, using default values?:No
    Properties image group count:2
    Properties group field count:1
    Properties filter field count:1
    Workspace measurement count:2
    Experiment name:Sigma
    Which objects should be used for locations?:Cells
    Enter a phenotype class table name if using the classifier tool:Hoopla
    Select an image to include:DNA
    Use the image name for the display?:No
    Image name:NucleicAcid
    Channel color:green
    Select an image to include:Actin
    Use the image name for the display?:No
    Image name:Protein
    Channel color:blue
    Do you want to add group fields?:Yes
    Enter the name of the group:WellGroup
    Enter the per-image columns which define the group, separated by commas:Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:Yes
    Automatically create a filter for each plate?:Yes
    Enter the name of the filter:Site1Filter
    Enter the MySQL WHERE clause to define a filter:Image_Metadata_Plate = \'1\'
    Create a CellProfiler Analyst workspace file?:Yes
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Mitochondria
    Select the x-axis measurement:Width_DNA
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Nuclei
    Select the y-axis measurement:Height_DNA
    Select the x-axis index:ImageNumber
    Select the measurement display tool:PlateViewer
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Cells
    Select the x-axis measurement:Height_Actin
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Speckles
    Select the y-axis measurement:Width_Actin
    Select the x-axis index:ImageNumber
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.sql_file_prefix == "Iota_"
        assert module.experiment_name == "Sigma"
        assert module.directory.dir_choice == cps.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.save_cpa_properties
        assert module.location_object == "Cells"
        assert module.properties_image_url_prepend == "http://server.university.edu"
        assert module.properties_plate_type == "384"
        assert module.properties_plate_metadata == "Plate"
        assert module.properties_well_metadata == "Well"
        assert not module.properties_export_all_image_defaults
        assert module.properties_wants_groups
        assert module.properties_wants_filters
        assert module.create_filters_for_plates
        assert module.create_workspace_file
        assert module.properties_class_table_name == "Hoopla"
        assert not module.wants_relationship_table
        assert len(module.image_groups) == 2
        for image_group, input_image_name, output_image_name, color in (
            (module.image_groups[0], "DNA", "NucleicAcid", "green"),
            (module.image_groups[1], "Actin", "Protein", "blue"),
        ):
            assert not image_group.wants_automatic_image_name
            assert image_group.image_cols == input_image_name
            assert image_group.image_name == output_image_name
            assert image_group.image_channel_colors == color

        assert len(module.group_field_groups) == 1
        g = module.group_field_groups[0]
        assert g.group_name == "WellGroup"
        assert g.group_statement == "Image_Metadata_Plate, Image_Metadata_Well"

        assert len(module.workspace_measurement_groups) == 2
        for (
            g,
            measurement_display,
            x_measurement_type,
            x_object_name,
            x_measurement_name,
            x_index_name,
            y_measurement_type,
            y_object_name,
            y_measurement_name,
            y_index_name,
        ) in (
            (
                module.workspace_measurement_groups[0],
                "ScatterPlot",
                cpmeas.IMAGE,
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                cpmeas.IMAGE,
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                cpmeas.IMAGE,
                "Cells",
                "Height_Actin",
                "ImageNumber",
                cpmeas.IMAGE,
                "Speckles",
                "Width_Actin",
                "ImageNumber",
            ),
        ):
            assert g.measurement_display == measurement_display
            assert g.x_measurement_type == x_measurement_type
            assert g.x_object_name == x_object_name
            assert g.x_measurement_name == x_measurement_name
            assert g.x_index_name == x_index_name
            assert g.y_measurement_type == y_measurement_type
            assert g.y_object_name == y_object_name
            assert g.y_measurement_name == y_measurement_name
            assert g.y_index_name == y_index_name

        assert len(module.filter_field_groups) == 1
        g = module.filter_field_groups[0]
        assert g.filter_name == "Site1Filter"
        assert g.filter_statement == "Image_Metadata_Plate = '1'"

    def test_load_v24():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
SVNRevision:11412

ExportToDatabase:[module_num:1|svn_version:\'11377\'|variable_revision_number:24|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Database type:MySQL
    Database name:Gamma
    Add a prefix to table names?:Yes
    Table prefix:Delta_
    SQL file prefix:Iota_
    Output file location:Default Output Folder\x7CNone
    Create a CellProfiler Analyst properties file?:Yes
    Database host:Alpha
    Username:Beta
    Password:Gamma
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object or a single object table?:Single object table
    Enter an image url prepend if you plan to access your files via http:http\x3A//server.university.edu
    Write image thumbnails directly to the database?:Yes
    Select the images you want to save thumbnails of:Actin,DNA
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:384
    Select the plate metadata:Plate
    Select the well metadata:Well
    Include information for all images, using default values?:No
    Properties image group count:2
    Properties group field count:1
    Properties filter field count:1
    Workspace measurement count:2
    Experiment name:Sigma
    Which objects should be used for locations?:Cells
    Enter a phenotype class table name if using the classifier tool:Hoopla
    Export object relationships?:Yes
    Select an image to include:DNA
    Use the image name for the display?:No
    Image name:NucleicAcid
    Channel color:green
    Select an image to include:Actin
    Use the image name for the display?:No
    Image name:Protein
    Channel color:blue
    Do you want to add group fields?:Yes
    Enter the name of the group:WellGroup
    Enter the per-image columns which define the group, separated by commas:Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:Yes
    Automatically create a filter for each plate?:Yes
    Enter the name of the filter:Site1Filter
    Enter the MySQL WHERE clause to define a filter:Image_Metadata_Plate = \'1\'
    Create a CellProfiler Analyst workspace file?:Yes
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Mitochondria
    Select the x-axis measurement:Width_DNA
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Nuclei
    Select the y-axis measurement:Height_DNA
    Select the x-axis index:ImageNumber
    Select the measurement display tool:PlateViewer
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Cells
    Select the x-axis measurement:Height_Actin
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Speckles
    Select the y-axis measurement:Width_Actin
    Select the x-axis index:ImageNumber
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.sql_file_prefix == "Iota_"
        assert module.experiment_name == "Sigma"
        assert module.directory.dir_choice == cps.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.save_cpa_properties
        assert module.location_object == "Cells"
        assert module.properties_image_url_prepend == "http://server.university.edu"
        assert module.properties_plate_type == "384"
        assert module.properties_plate_metadata == "Plate"
        assert module.properties_well_metadata == "Well"
        assert not module.properties_export_all_image_defaults
        assert module.properties_wants_groups
        assert module.properties_wants_filters
        assert module.create_filters_for_plates
        assert module.create_workspace_file
        assert module.properties_class_table_name == "Hoopla"
        assert module.wants_relationship_table
        assert len(module.image_groups) == 2
        for image_group, input_image_name, output_image_name, color in (
            (module.image_groups[0], "DNA", "NucleicAcid", "green"),
            (module.image_groups[1], "Actin", "Protein", "blue"),
        ):
            assert not image_group.wants_automatic_image_name
            assert image_group.image_cols == input_image_name
            assert image_group.image_name == output_image_name
            assert image_group.image_channel_colors == color

        assert len(module.group_field_groups) == 1
        g = module.group_field_groups[0]
        assert g.group_name == "WellGroup"
        assert g.group_statement == "Image_Metadata_Plate, Image_Metadata_Well"

        assert len(module.workspace_measurement_groups) == 2
        for (
            g,
            measurement_display,
            x_measurement_type,
            x_object_name,
            x_measurement_name,
            x_index_name,
            y_measurement_type,
            y_object_name,
            y_measurement_name,
            y_index_name,
        ) in (
            (
                module.workspace_measurement_groups[0],
                "ScatterPlot",
                cpmeas.IMAGE,
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                cpmeas.IMAGE,
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                cpmeas.IMAGE,
                "Cells",
                "Height_Actin",
                "ImageNumber",
                cpmeas.IMAGE,
                "Speckles",
                "Width_Actin",
                "ImageNumber",
            ),
        ):
            assert g.measurement_display == measurement_display
            assert g.x_measurement_type == x_measurement_type
            assert g.x_object_name == x_object_name
            assert g.x_measurement_name == x_measurement_name
            assert g.x_index_name == x_index_name
            assert g.y_measurement_type == y_measurement_type
            assert g.y_object_name == y_object_name
            assert g.y_measurement_name == y_measurement_name
            assert g.y_index_name == y_index_name

        assert len(module.filter_field_groups) == 1
        g = module.filter_field_groups[0]
        assert g.filter_name == "Site1Filter"
        assert g.filter_statement == "Image_Metadata_Plate = '1'"
        assert module.allow_overwrite == E.OVERWRITE_DATA

    def test_load_v25():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
SVNRevision:11412

ExportToDatabase:[module_num:1|svn_version:\'11377\'|variable_revision_number:25|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Database type:MySQL
    Database name:Gamma
    Add a prefix to table names?:Yes
    Table prefix:Delta_
    SQL file prefix:Iota_
    Output file location:Default Output Folder\x7CNone
    Create a CellProfiler Analyst properties file?:Yes
    Database host:Alpha
    Username:Beta
    Password:Gamma
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object or a single object table?:Single object table
    Enter an image url prepend if you plan to access your files via http:http\x3A//server.university.edu
    Write image thumbnails directly to the database?:Yes
    Select the images you want to save thumbnails of:Actin,DNA
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:384
    Select the plate metadata:Plate
    Select the well metadata:Well
    Include information for all images, using default values?:No
    Properties image group count:2
    Properties group field count:1
    Properties filter field count:1
    Workspace measurement count:2
    Experiment name:Sigma
    Which objects should be used for locations?:Cells
    Enter a phenotype class table name if using the classifier tool:Hoopla
    Export object relationships?:Yes
    Allow overwrite?:Never
    Select an image to include:DNA
    Use the image name for the display?:No
    Image name:NucleicAcid
    Channel color:green
    Select an image to include:Actin
    Use the image name for the display?:No
    Image name:Protein
    Channel color:blue
    Do you want to add group fields?:Yes
    Enter the name of the group:WellGroup
    Enter the per-image columns which define the group, separated by commas:Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:Yes
    Automatically create a filter for each plate?:Yes
    Enter the name of the filter:Site1Filter
    Enter the MySQL WHERE clause to define a filter:Image_Metadata_Plate = \'1\'
    Create a CellProfiler Analyst workspace file?:Yes
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Mitochondria
    Select the x-axis measurement:Width_DNA
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Nuclei
    Select the y-axis measurement:Height_DNA
    Select the x-axis index:ImageNumber
    Select the measurement display tool:PlateViewer
    Type of measurement to plot on the x-axis:Image
    Enter the object name:Cells
    Select the x-axis measurement:Height_Actin
    Select the x-axis index:ImageNumber
    Type of measurement to plot on the y-axis:Image
    Enter the object name:Speckles
    Select the y-axis measurement:Width_Actin
    Select the x-axis index:ImageNumber
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.sql_file_prefix == "Iota_"
        assert module.experiment_name == "Sigma"
        assert module.directory.dir_choice == cps.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.save_cpa_properties
        assert module.location_object == "Cells"
        assert module.wants_properties_image_url_prepend
        assert module.properties_image_url_prepend == "http://server.university.edu"
        assert module.properties_plate_type == "384"
        assert module.properties_plate_metadata == "Plate"
        assert module.properties_well_metadata == "Well"
        assert not module.properties_export_all_image_defaults
        assert module.properties_wants_groups
        assert module.properties_wants_filters
        assert module.create_filters_for_plates
        assert module.create_workspace_file
        assert module.properties_class_table_name == "Hoopla"
        assert module.wants_relationship_table
        assert len(module.image_groups) == 2
        for image_group, input_image_name, output_image_name, color in (
            (module.image_groups[0], "DNA", "NucleicAcid", "green"),
            (module.image_groups[1], "Actin", "Protein", "blue"),
        ):
            assert not image_group.wants_automatic_image_name
            assert image_group.image_cols == input_image_name
            assert image_group.image_name == output_image_name
            assert image_group.image_channel_colors == color

        assert len(module.group_field_groups) == 1
        g = module.group_field_groups[0]
        assert g.group_name == "WellGroup"
        assert g.group_statement == "Image_Metadata_Plate, Image_Metadata_Well"

        assert len(module.workspace_measurement_groups) == 2
        for (
            g,
            measurement_display,
            x_measurement_type,
            x_object_name,
            x_measurement_name,
            x_index_name,
            y_measurement_type,
            y_object_name,
            y_measurement_name,
            y_index_name,
        ) in (
            (
                module.workspace_measurement_groups[0],
                "ScatterPlot",
                cpmeas.IMAGE,
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                cpmeas.IMAGE,
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                cpmeas.IMAGE,
                "Cells",
                "Height_Actin",
                "ImageNumber",
                cpmeas.IMAGE,
                "Speckles",
                "Width_Actin",
                "ImageNumber",
            ),
        ):
            assert g.measurement_display == measurement_display
            assert g.x_measurement_type == x_measurement_type
            assert g.x_object_name == x_object_name
            assert g.x_measurement_name == x_measurement_name
            assert g.x_index_name == x_index_name
            assert g.y_measurement_type == y_measurement_type
            assert g.y_object_name == y_object_name
            assert g.y_measurement_name == y_measurement_name
            assert g.y_index_name == y_index_name

        assert len(module.filter_field_groups) == 1
        g = module.filter_field_groups[0]
        assert g.filter_name == "Site1Filter"
        assert g.filter_statement == "Image_Metadata_Plate = '1'"
        assert module.allow_overwrite == E.OVERWRITE_NEVER

    def test_load_v26():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140130135727
GitHash:d666db0
ModuleCount:1
HasImagePlaneDetails:False

ExportToDatabase:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:26|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Database type:MySQL
    Database name:Gamma
    Add a prefix to table names?:Yes
    Table prefix:Delta_
    SQL file prefix:Iota_
    Output file location:Default Output Folder\x7CNone
    Create a CellProfiler Analyst properties file?:Yes
    Database host:Alpha
    Username:Beta
    Password:Gamma
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object, a single object table or a single object view?:Single object table
    Enter an image url prepend if you plan to access your files via http:http\x3A//server.university.edu
    Write image thumbnails directly to the database?:Yes
    Select the images for which you want to save thumbnails:Actin,DNA
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:384
    Select the plate metadata:Plate
    Select the well metadata:Well
    Include information for all images, using default values?:No
    Properties image group count:2
    Properties group field count:1
    Properties filter field count:1
    Workspace measurement count:2
    Experiment name:Sigma
    Which objects should be used for locations?:Cells
    Enter a phenotype class table name if using the classifier tool:Hoopla
    Export object relationships?:Yes
    Overwrite without warning?:Never
    Access CPA images via URL?:No
    Select an image to include:DNA
    Use the image name for the display?:No
    Image name:NucleicAcid
    Channel color:green
    Select an image to include:Actin
    Use the image name for the display?:No
    Image name:Protein
    Channel color:blue
    Do you want to add group fields?:Yes
    Enter the name of the group:WellGroup
    Enter the per-image columns which define the group, separated by commas:Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:Yes
    Automatically create a filter for each plate?:Yes
    Enter the name of the filter:Site1Filter
    Enter the MySQL WHERE clause to define a filter:Image_Metadata_Plate = \'1\'
    Create a CellProfiler Analyst workspace file?:Yes
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the X-axis:Image
    Enter the object name:Mitochondria
    Select the X-axis measurement:Width_DNA
    Select the X-axis index:ImageNumber
    Type of measurement to plot on the Y-axis:Image
    Enter the object name:Nuclei
    Select the Y-axis measurement:Height_DNA
    Select the Y-axis index:ImageNumber
    Select the measurement display tool:PlateViewer
    Type of measurement to plot on the X-axis:Image
    Enter the object name:Cells
    Select the X-axis measurement:Height_Actin
    Select the X-axis index:ImageNumber
    Type of measurement to plot on the Y-axis:Image
    Enter the object name:Speckles
    Select the Y-axis measurement:Width_Actin
    Select the Y-axis index:ImageNumber

"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(module, E.ExportToDatabase)
        assert module.db_type == E.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.sql_file_prefix == "Iota_"
        assert module.experiment_name == "Sigma"
        assert module.directory.dir_choice == cps.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.save_cpa_properties
        assert module.location_object == "Cells"
        assert not module.wants_properties_image_url_prepend
        assert module.properties_image_url_prepend == "http://server.university.edu"
        assert module.properties_plate_type == "384"
        assert module.properties_plate_metadata == "Plate"
        assert module.properties_well_metadata == "Well"
        assert not module.properties_export_all_image_defaults
        assert module.properties_wants_groups
        assert module.properties_wants_filters
        assert module.create_filters_for_plates
        assert module.create_workspace_file
        assert module.properties_class_table_name == "Hoopla"
        assert module.wants_relationship_table
        assert module.properties_classification_type == E.CT_OBJECT
        assert len(module.image_groups) == 2
        for image_group, input_image_name, output_image_name, color in (
            (module.image_groups[0], "DNA", "NucleicAcid", "green"),
            (module.image_groups[1], "Actin", "Protein", "blue"),
        ):
            assert not image_group.wants_automatic_image_name
            assert image_group.image_cols == input_image_name
            assert image_group.image_name == output_image_name
            assert image_group.image_channel_colors == color

        assert len(module.group_field_groups) == 1
        g = module.group_field_groups[0]
        assert g.group_name == "WellGroup"
        assert g.group_statement == "Image_Metadata_Plate, Image_Metadata_Well"

        assert len(module.workspace_measurement_groups) == 2
        for (
            g,
            measurement_display,
            x_measurement_type,
            x_object_name,
            x_measurement_name,
            x_index_name,
            y_measurement_type,
            y_object_name,
            y_measurement_name,
            y_index_name,
        ) in (
            (
                module.workspace_measurement_groups[0],
                "ScatterPlot",
                cpmeas.IMAGE,
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                cpmeas.IMAGE,
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                cpmeas.IMAGE,
                "Cells",
                "Height_Actin",
                "ImageNumber",
                cpmeas.IMAGE,
                "Speckles",
                "Width_Actin",
                "ImageNumber",
            ),
        ):
            assert g.measurement_display == measurement_display
            assert g.x_measurement_type == x_measurement_type
            assert g.x_object_name == x_object_name
            assert g.x_measurement_name == x_measurement_name
            assert g.x_index_name == x_index_name
            assert g.y_measurement_type == y_measurement_type
            assert g.y_object_name == y_object_name
            assert g.y_measurement_name == y_measurement_name
            assert g.y_index_name == y_index_name

        assert len(module.filter_field_groups) == 1
        g = module.filter_field_groups[0]
        assert g.filter_name == "Site1Filter"
        assert g.filter_statement == "Image_Metadata_Plate = '1'"
        assert module.allow_overwrite == E.OVERWRITE_NEVER

    def test_load_v27():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160129211738
GitHash:cd1cb4e
ModuleCount:1
HasImagePlaneDetails:False

ExportToDatabase:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:27|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Database type:MySQL
    Database name:DefaultDB
    Add a prefix to table names?:Yes
    Table prefix:MyExpt_
    SQL file prefix:SQL_
    Output file location:Default Output Folder\x7C
    Create a CellProfiler Analyst properties file?:No
    Database host:
    Username:
    Password:
    Name the SQLite database file:DefaultDB.db
    Calculate the per-image mean values of object measurements?:Yes
    Calculate the per-image median values of object measurements?:No
    Calculate the per-image standard deviation values of object measurements?:No
    Calculate the per-well mean values of object measurements?:No
    Calculate the per-well median values of object measurements?:No
    Calculate the per-well standard deviation values of object measurements?:No
    Export measurements for all objects to the database?:All
    Select the objects:
    Maximum # of characters in a column name:64
    Create one table per object, a single object table or a single object view?:Single object table
    Enter an image url prepend if you plan to access your files via http:
    Write image thumbnails directly to the database?:No
    Select the images for which you want to save thumbnails:
    Auto-scale thumbnail pixel intensities?:Yes
    Select the plate type:None
    Select the plate metadata:None
    Select the well metadata:None
    Include information for all images, using default values?:Yes
    Properties image group count:1
    Properties group field count:1
    Properties filter field count:0
    Workspace measurement count:1
    Experiment name:MyExpt
    Which objects should be used for locations?:None
    Enter a phenotype class table name if using the classifier tool:
    Export object relationships?:Yes
    Overwrite without warning?:Never
    Access CPA images via URL?:No
    Select the classification type:Image
    Select an image to include:None
    Use the image name for the display?:Yes
    Image name:Channel1
    Channel color:red
    Do you want to add group fields?:No
    Enter the name of the group:
    Enter the per-image columns which define the group, separated by commas:ImageNumber, Image_Metadata_Plate, Image_Metadata_Well
    Do you want to add filter fields?:No
    Automatically create a filter for each plate?:No
    Create a CellProfiler Analyst workspace file?:No
    Select the measurement display tool:ScatterPlot
    Type of measurement to plot on the X-axis:Image
    Enter the object name:None
    Select the X-axis measurement:None
    Select the X-axis index:ImageNumber
    Type of measurement to plot on the Y-axis:Image
    Enter the object name:MyObjects
    Select the Y-axis measurement:None
    Select the Y-axis index:ImageNumber
"""
        pipeline = cpp.Pipeline()
        pipeline.load(StringIO(data))
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToDatabase)
        assert module.properties_classification_type == E.CT_IMAGE
        assert len(module.workspace_measurement_groups) == 1
        g = module.workspace_measurement_groups[0]
        assert g.y_object_name == "MyObjects"

    RTEST_NONE = 0
    RTEST_SOME = 1
    RTEST_DUPLICATE = 2

    def make_workspace(
        wants_files,
        alt_object=False,
        long_measurement=False,
        wierd_measurement=False,
        well_metadata=False,
        image_set_count=1,
        group_measurement=False,
        relationship_type=None,
        relationship_test_type=None,
        post_run_test=False,
    ):
        """Make a measurements structure with image and object measurements"""

        class TestModule(cpm.Module):
            module_name = "TestModule"
            module_num = 1
            variable_revision_number = 1

            def create_settings():
                image_name = cps.ImageNameProvider("Foo", IMAGE_NAME)
                objects_name = cps.ObjectNameProvider("Bar", OBJECT_NAME)
                if alt_object:
                    altobjects_name = cps.ObjectNameProvider("Baz", ALTOBJECT_NAME)

            def settings():
                return [image_name, objects_name] + (
                    [altobjects_name] if alt_object else []
                )

            @staticmethod
            def in_module(flag):
                """Return True to add the measurement to module's get_measurement_columns"""
                return flag and flag != MISSING_FROM_MODULE

            @staticmethod
            def in_measurements(flag):
                return flag and flag != MISSING_FROM_MEASUREMENTS

            def get_measurement_columns(pipeline):
                columns = [
                    (cpmeas.IMAGE, INT_IMG_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                    (cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                    (
                        cpmeas.IMAGE,
                        STRING_IMG_MEASUREMENT,
                        cpmeas.COLTYPE_VARCHAR_FORMAT % 40,
                    ),
                    (cpmeas.IMAGE, OBJECT_COUNT_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                    (
                        OBJECT_NAME,
                        cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                        cpmeas.COLTYPE_INTEGER,
                    ),
                    (OBJECT_NAME, OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                ]
                if in_module(alt_object):
                    columns += [
                        (
                            cpmeas.IMAGE,
                            ALTOBJECT_COUNT_MEASUREMENT,
                            cpmeas.COLTYPE_INTEGER,
                        ),
                        (
                            ALTOBJECT_NAME,
                            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                            cpmeas.COLTYPE_INTEGER,
                        ),
                        (ALTOBJECT_NAME, OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                    ]
                if in_module(long_measurement):
                    columns += [
                        (cpmeas.IMAGE, LONG_IMG_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                        (OBJECT_NAME, LONG_OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                    ]
                if in_module(wierd_measurement):
                    columns += [
                        (cpmeas.IMAGE, WIERD_IMG_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                        (OBJECT_NAME, WIERD_OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                    ]
                if in_module(well_metadata):
                    columns += [
                        (
                            cpmeas.IMAGE,
                            "Metadata_Plate",
                            cpmeas.COLTYPE_VARCHAR_FORMAT % 20,
                        ),
                        (
                            cpmeas.IMAGE,
                            "Metadata_Well",
                            cpmeas.COLTYPE_VARCHAR_FORMAT % 3,
                        ),
                    ]
                if in_module(group_measurement):
                    d = {cpmeas.MCA_AVAILABLE_POST_GROUP: True}
                    columns += [
                        (
                            cpmeas.IMAGE,
                            GROUP_IMG_MEASUREMENT,
                            cpmeas.COLTYPE_INTEGER,
                            d,
                        ),
                        (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT, d),
                    ]
                if post_run_test:
                    columns += [
                        (
                            cpmeas.EXPERIMENT,
                            STRING_IMG_MEASUREMENT,
                            cpmeas.COLTYPE_VARCHAR,
                            {cpmeas.MCA_AVAILABLE_POST_RUN: True},
                        )
                    ]
                return columns

            def get_object_relationships(pipeline):
                if relationship_type is not None:
                    return [
                        (
                            RELATIONSHIP_NAME,
                            OBJECT_NAME,
                            ALTOBJECT_NAME,
                            relationship_type,
                        ),
                        (
                            RELATIONSHIP_NAME,
                            ALTOBJECT_NAME,
                            OBJECT_NAME,
                            relationship_type,
                        ),
                    ]
                return []

            def get_categories(pipeline, object_name):
                return (
                    [M_CATEGORY, cellprofiler.measurement.C_NUMBER]
                    if (
                        object_name == OBJECT_NAME
                        or ((object_name == ALTOBJECT_NAME) and in_module(alt_object))
                    )
                    else [M_CATEGORY, "Count", "Metadata"]
                    if object_name == cpmeas.IMAGE
                    else []
                )

            def get_measurements(pipeline, object_name, category):
                if category == M_CATEGORY:
                    if object_name == OBJECT_NAME:
                        if in_module(long_measurement):
                            return [OBJ_FEATURE, LONG_OBJ_FEATURE]
                        else:
                            return [OBJ_FEATURE]
                    elif (object_name == ALTOBJECT_NAME) and in_module(alt_object):
                        return [OBJ_FEATURE]
                    else:
                        return (
                            [INT_IMG_FEATURE, FLOAT_IMG_FEATURE, STRING_IMG_FEATURE]
                            + [LONG_IMG_FEATURE]
                            if in_module(long_measurement)
                            else [WIERD_IMG_FEATURE]
                            if in_module(wierd_measurement)
                            else []
                        )
                elif category == cellprofiler.measurement.C_NUMBER and object_name in (
                    OBJECT_NAME,
                    ALTOBJECT_NAME,
                ):
                    return cellprofiler.measurement.FTR_OBJECT_NUMBER
                elif category == "Count" and object_name == cpmeas.IMAGE:
                    result = [OBJECT_NAME]
                    if in_module(alt_object):
                        result += [ALTOBJECT_NAME]
                    return result
                elif category == "Metadata" and object_name == cpmeas.IMAGE:
                    return ["Plate", "Well"]
                return []

        m = cpmeas.Measurements(can_overwrite=True)
        for i in range(image_set_count):
            if i > 0:
                m.next_image_set()
            m.add_image_measurement(cpp.GROUP_NUMBER, 1)
            m.add_image_measurement(cpp.GROUP_INDEX, i + 1)
            m.add_image_measurement(INT_IMG_MEASUREMENT, INT_VALUE)
            m.add_image_measurement(FLOAT_IMG_MEASUREMENT, FLOAT_VALUE)
            m.add_image_measurement(STRING_IMG_MEASUREMENT, STRING_VALUE)
            m.add_image_measurement(OBJECT_COUNT_MEASUREMENT, len(OBJ_VALUE))
            m.add_measurement(
                OBJECT_NAME,
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                np.arange(len(OBJ_VALUE)) + 1,
            )
            m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, OBJ_VALUE.copy())
            if TestModule.in_measurements(alt_object):
                m.add_measurement(
                    ALTOBJECT_NAME,
                    cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                    np.arange(100) + 1,
                )
                m.add_image_measurement(ALTOBJECT_COUNT_MEASUREMENT, 100)
                m.add_measurement(ALTOBJECT_NAME, OBJ_MEASUREMENT, ALTOBJ_VALUE)
            if TestModule.in_measurements(long_measurement):
                m.add_image_measurement(LONG_IMG_MEASUREMENT, 100)
                m.add_measurement(OBJECT_NAME, LONG_OBJ_MEASUREMENT, OBJ_VALUE.copy())
            if TestModule.in_measurements(wierd_measurement):
                m.add_image_measurement(WIERD_IMG_MEASUREMENT, 100)
                m.add_measurement(OBJECT_NAME, WIERD_OBJ_MEASUREMENT, OBJ_VALUE.copy())
            if TestModule.in_measurements(well_metadata):
                m.add_image_measurement("Metadata_Plate", PLATE)
                m.add_image_measurement("Metadata_Well", WELL)
            if TestModule.in_measurements(group_measurement):
                m.add_image_measurement(GROUP_IMG_MEASUREMENT, INT_VALUE)
                m.add_measurement(OBJECT_NAME, GROUP_OBJ_MEASUREMENT, OBJ_VALUE.copy())
        r = np.random.RandomState()
        r.seed(image_set_count)
        if relationship_test_type == RTEST_SOME:
            n = 10
            i1, o1 = [
                x.flatten() for x in np.mgrid[1 : (image_set_count + 1), 1 : (n + 1)]
            ]
            for o1name, o2name in (
                (OBJECT_NAME, ALTOBJECT_NAME),
                (ALTOBJECT_NAME, OBJECT_NAME),
            ):
                i2 = r.permutation(i1)
                o2 = r.permutation(o1)

                m.add_relate_measurement(
                    1, RELATIONSHIP_NAME, o1name, o2name, i1, o1, i2, o2
                )

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cpi.Image(r.uniform(size=(512, 512))))
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        object_set.add_objects(objects, OBJECT_NAME)
        if alt_object:
            objects = cpo.Objects()
            objects.segmented = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
            object_set.add_objects(objects, ALTOBJECT_NAME)
        test_module = TestModule()
        pipeline = cpp.Pipeline()

        def callback_handler(caller, event):
            assert not isinstance(event, cpp.RunExceptionEvent)

        pipeline.add_listener(callback_handler)
        pipeline.add_module(test_module)
        module = E.ExportToDatabase()
        module.set_module_num(2)
        table_prefix = "T_%s" % str(uuid.uuid4()).replace("-", "")
        module.table_prefix.value = table_prefix
        module.want_table_prefix.value = True
        module.db_host.value = MYSQL_HOST
        module.db_user.value = MYSQL_USER
        module.db_passwd.value = MYSQL_PASSWORD
        module.db_name.value = MYSQL_DATABASE
        module.wants_relationship_table_setting.value = relationship_type is not None
        pipeline.add_module(module)
        pipeline.write_experiment_measurements(m)
        workspace = cpw.Workspace(
            pipeline, module, image_set, object_set, m, image_set_list
        )
        for column in pipeline.get_measurement_columns():
            if column[1].startswith("ModuleError_") or column[1].startswith(
                "ExecutionTime_"
            ):
                m.add_image_measurement(column[1], 0)
        m.next_image_set(image_set_count)
        if wants_files or well_metadata:
            output_dir = tempfile.mkdtemp()
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir

            def finally_fn():
                for filename in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, filename))

            return workspace, module, output_dir, finally_fn
        else:
            return workspace, module

    def load_database(output_dir, module, image_set_count=1):
        """Load a database written by DB_MYSQL_CSV"""
        assert isinstance(module, E.ExportToDatabase)
        curdir = os.path.abspath(os.curdir)
        os.chdir(output_dir)
        try:
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_%d" % image_set_count
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            if module.separate_object_tables == E.OT_PER_OBJECT:
                object_file = os.path.join(
                    output_dir, base_name + "_" + OBJECT_NAME + ".CSV"
                )
            else:
                object_file = "%s_%s.CSV" % (base_name, cpmeas.OBJECT)
                object_file = os.path.join(output_dir, object_file)
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename)
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
        finally:
            connection.commit()
            os.chdir(curdir)

    def tteesstt_no_relationships(module, cursor):
        for t in (E.T_RELATIONSHIPS, E.V_RELATIONSHIPS):
            statement = "select count('x') from %s" % module.get_table_name(t)
            cursor.execute(statement)
            assert cursor.fetchall()[0][0] == 0

    def tteesstt_relate(measurements, module, cursor):
        v_name = module.get_table_name(E.V_RELATIONSHIPS)
        statement = (
            "select count('x') from %s "
            "where %s=%d and %s='%s' and %s='%%s' and %s='%%s' "
            "and %s = %%d and %s = %%d and %s = %%d and %s = %%d"
        ) % (
            v_name,
            E.COL_MODULE_NUMBER,
            1,
            E.COL_RELATIONSHIP,
            RELATIONSHIP_NAME,
            E.COL_OBJECT_NAME1,
            E.COL_OBJECT_NAME2,
            E.COL_IMAGE_NUMBER1,
            E.COL_OBJECT_NUMBER1,
            E.COL_IMAGE_NUMBER2,
            E.COL_OBJECT_NUMBER2,
        )
        for rk in measurements.get_relationship_groups():
            module_num = rk.module_number
            relationship = rk.relationship
            object_name1 = rk.object_name1
            object_name2 = rk.object_name2
            for i1, o1, i2, o2 in measurements.get_relationships(
                module_num, relationship, object_name1, object_name2
            ):
                cursor.execute(statement % (object_name1, object_name2, i1, o1, i2, o2))
                assert cursor.fetchall()[0][0] == 1

    def drop_tables(module, table_suffixes=None):
        """Drop all tables and views  that match the prefix"""
        cursor = connection.cursor()
        try:
            for info_table, thing in (("VIEWS", "view"), ("TABLES", "table")):
                statement = (
                    "select table_name from INFORMATION_SCHEMA.%s "
                    "where table_schema='%s' "
                    "and table_name like '%s%%'"
                ) % (info_table, module.db_name.value, module.table_prefix.value)
                cursor.execute(statement)
                for (table_name,) in cursor.fetchall():
                    assert table_name.startswith(module.table_prefix.value)
                    statement = "drop %s %s.%s" % (
                        thing,
                        module.db_name.value,
                        table_name,
                    )
                    try:
                        cursor.execute(statement)
                    except Exception:
                        traceback.print_exc()
                        print(("Failed to drop table %s" % table_name))
        except:
            traceback.print_exc()
            print("Failed to drop all tables")
        finally:
            try:
                cursor.nextset()
                cursor.close()
            except:
                pass

    def drop_views(module, table_suffixes=None):
        drop_tables(module)

    def test_write_mysql_db():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            load_database(output_dir, module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Make sure no relationships tables were created.
            #
            assert not module.wants_relationship_table
            for t in (E.T_RELATIONSHIPS, E.T_RELATIONSHIP_TYPES):
                statement = "select count('x') from INFORMATION_SCHEMA.TABLES "
                statement += "where table_schema=%s and table_name=%s"
                cursor.execute(
                    statement, (module.db_name.value, module.get_table_name(t))
                )
                assert cursor.fetchall()[0][0] == 0
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_mysql_db_filter_objs():
        workspace, module, output_dir, finally_fn = make_workspace(True, True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_SELECT
            module.objects_list.choices = [OBJECT_NAME, ALTOBJECT_NAME]
            module.objects_list.value = OBJECT_NAME
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = "%s_%s.CSV" % (base_name, cpmeas.OBJECT)
            object_file = os.path.join(output_dir, object_file)
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename), "Can't find %s" % filename
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    OBJECT_NAME,
                    cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert row[3] == i + 1
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_mysql_db_dont_filter_objs():
        workspace, module, output_dir, finally_fn = make_workspace(True, True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = "%s_%s.CSV" % (base_name, cpmeas.OBJECT)
            object_file = os.path.join(output_dir, object_file)
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename), "No such file: " + filename
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    ALTOBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == len(ALTOBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s, %s_%s, "
                "%s_%s, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    OBJECT_NAME,
                    cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                    ALTOBJECT_NAME,
                    OBJ_MEASUREMENT,
                    ALTOBJECT_NAME,
                    cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 6
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 4) == 0
                assert row[3] == i + 1
                assert round(abs(row[4] - ALTOBJ_VALUE[i]), 4) == 0
                assert row[5] == i + 1
            for i in range(len(OBJ_VALUE), len(ALTOBJ_VALUE)):
                row = cursor.fetchone()
                assert len(row) == 6
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[4] - ALTOBJ_VALUE[i]), 4) == 0
                assert row[5] == i + 1
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_mysql_direct():
        """Write directly to the mysql DB, not to a file"""
        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            if not __test_mysql:
                skipTest("Skipping actual DB work, not at the Broad.")
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_Group_Number, Image_Group_Index, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 7
            assert row[0] == 1
            assert row[1] == 1
            assert row[2] == 1
            assert round(abs(row[3] - INT_VALUE), 7) == 0
            assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
            assert row[5] == STRING_VALUE
            assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_00_write_direct_long_colname():
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = make_workspace(False, long_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_img_column = mappings["Image_%s" % LONG_IMG_MEASUREMENT]
            long_obj_column = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            long_aggregate_obj_column = mappings[
                "Mean_%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)
            ]

            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    long_img_column,
                    long_aggregate_obj_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 7
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            assert round(abs(row[6] - np.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s,%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    long_obj_column,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_01_write_csv_long_colname():
        """Write to MySQL, ensuring some columns have long names

        This is a regression test of IMG-786
        """
        workspace, module, output_dir, finally_fn = make_workspace(
            True, long_measurement=True
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.post_run(workspace)
            load_database(output_dir, module)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_img_column = mappings["Image_%s" % LONG_IMG_MEASUREMENT]
            long_obj_column = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            long_aggregate_obj_column = mappings[
                "Mean_%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)
            ]

            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    long_img_column,
                    long_aggregate_obj_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 7
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 4) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            assert round(abs(row[6] - np.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s,%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    long_obj_column,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_01_write_nulls():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.NaN, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = np.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = "%s_%s.CSV" % (base_name, cpmeas.OBJECT)
            object_file = os.path.join(output_dir, object_file)
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename)
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - np.mean(om[~np.isnan(om)])), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_02_write_inf():
        """regression test of img-1149"""
        workspace, module, output_dir, finally_fn = make_workspace(True)
        #
        # Insert inf into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.inf, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = np.inf
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = "%s_%s.CSV" % (base_name, cpmeas.OBJECT)
            object_file = os.path.join(output_dir, object_file)
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename)
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            mask = ~(np.isnan(om) | np.isinf(om))
            assert round(abs(row[5] - np.mean(om[mask])), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_mysql_direct_null():
        """Write directly to the mysql DB, not to a file and write nulls"""
        workspace, module = make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.NaN, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = np.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - np.mean(om[np.isfinite(om)])), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None or i != 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_wierd_colname():
        """Write to MySQL, even if illegal characters are in the column name"""
        workspace, module = make_workspace(False, wierd_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            wierd_img_column = mappings["Image_%s" % WIERD_IMG_MEASUREMENT]
            wierd_obj_column = mappings["%s_%s" % (OBJECT_NAME, WIERD_OBJ_MEASUREMENT)]

            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    wierd_img_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s,%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    wierd_obj_column,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_50_char_colname():
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = make_workspace(False, long_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_img_column = mappings["Image_%s" % LONG_IMG_MEASUREMENT]
            long_obj_column = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            assert len(long_img_column) <= 50
            assert len(long_obj_column) <= 50
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    long_img_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s,%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    long_obj_column,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_backslash():
        """Regression test for IMG-898

        Make sure CP can write string data containing a backslash character
        to the database in direct-mode.
        """
        backslash_string = "\\Why worry?"
        workspace, module = make_workspace(False)
        assert isinstance(module, E.ExportToDatabase)
        module.objects_choice.value = E.O_NONE
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_image_measurement(STRING_IMG_MEASUREMENT, backslash_string)
        try:
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            statement = "select Image_%s from %sPer_Image" % (
                STRING_IMG_MEASUREMENT,
                module.table_prefix.value,
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 1
            assert row[0] == backslash_string
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Experiment"))

    def test_mysql_as_data_tool():
        """Write directly to the mysql DB, not to a file"""
        workspace, module = make_workspace(False, image_set_count=2)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.allow_overwrite.value = E.OVERWRITE_DATA
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run_as_data_tool(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_Group_Number, Image_Group_Index, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            for i in range(2):
                row = cursor.fetchone()
                assert len(row) == 7
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def get_interaction_handler(ran_interaction_handler):
        """Return an interaction handler for testing

        return an interaction handler function that sets
        ran_interaction_handler[0] to True when run.
        """

        def interaction_handler(*args, **vargs):
            assert len(args) > 0
            result = args[0].handle_interaction(*args[1:], **vargs)
            ran_interaction_handler[0] = True
            return result

        return interaction_handler

    def test_write_sqlite_direct():
        """Write directly to a SQLite database"""
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = make_workspace(True)
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = get_interaction_handler(
                    ran_interaction_handler
                )
            cursor = None
            connection = None
            try:
                assert isinstance(module, E.ExportToDatabase)
                module.db_type.value = E.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = E.O_ALL
                module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = E.OT_COMBINE
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.run(workspace)
                assert with_interaction_handler == ran_interaction_handler[0]
                cursor, connection = get_sqlite_cursor(module)
                check_experiment_table(cursor, module, workspace.measurements)
                #
                # Now read the image file from the database
                #
                image_table = module.table_prefix.value + "Per_Image"
                statement = (
                    "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                    "from %s"
                    % (
                        INT_IMG_MEASUREMENT,
                        FLOAT_IMG_MEASUREMENT,
                        STRING_IMG_MEASUREMENT,
                        OBJECT_NAME,
                        image_table,
                    )
                )
                cursor.execute(statement)
                row = cursor.fetchone()
                assert len(row) == 5
                assert row[0] == 1
                assert round(abs(row[1] - INT_VALUE), 7) == 0
                assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
                assert row[3] == STRING_VALUE
                assert row[4] == len(OBJ_VALUE)
                with pytest.raises(StopIteration):
                    cursor.__next__()
                statement = (
                    "select ImageNumber, ObjectNumber, %s_%s "
                    "from %sPer_Object order by ImageNumber, ObjectNumber"
                    % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
                )
                cursor.execute(statement)
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
                with pytest.raises(StopIteration):
                    cursor.__next__()
            finally:
                if cursor is not None:
                    cursor.close()
                if connection is not None:
                    connection.close()
                if hasattr(module, "cursor") and module.cursor is not None:
                    module.cursor.close()
                if hasattr(module, "connection") and module.connection is not None:
                    module.connection.close()
                finally_fn()

    def test_write_sqlite_backslash():
        """Regression test of IMG-898 sqlite with backslash in string"""
        workspace, module, output_dir, finally_fn = make_workspace(True)
        backslash_string = "\\Why doesn't he worry?"
        m = workspace.measurements
        m.add_image_measurement(STRING_IMG_MEASUREMENT, backslash_string)
        cursor = None
        connection = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_NONE
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = get_sqlite_cursor(module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select Image_%s from %s" % (
                STRING_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 1
            assert row[0] == backslash_string
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_numpy_float32():
        """Regression test of img-915

        This error occurred when the sqlite3 driver was unable to convert
        a numpy.float32 to a float.
        """
        workspace, module, output_dir, finally_fn = make_workspace(True)
        fim = workspace.measurements.get_all_measurements(
            cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT
        )
        for i in range(len(fim)):
            fim[i] = np.float32(fim[i])
        iim = workspace.measurements.get_all_measurements(
            cpmeas.IMAGE, INT_IMG_MEASUREMENT
        )
        for i in range(len(iim)):
            iim[i] = np.int32(iim[i])
        cursor = None
        connection = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = get_sqlite_cursor(module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_sqlite_data_tool():
        workspace, module, output_dir, finally_fn = make_workspace(
            True, image_set_count=2
        )
        cursor = None
        connection = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.allow_overwrite.value = E.OVERWRITE_DATA
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run_as_data_tool(workspace)
            cursor, connection = get_sqlite_cursor(module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            for i in range(2):
                row = cursor.fetchone()
                assert len(row) == 5
                assert row[0] == i + 1
                assert round(abs(row[1] - INT_VALUE), 7) == 0
                assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
                assert row[3] == STRING_VALUE
                assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_stable_column_mapper():
        """Make sure the column mapper always yields the same output"""
        mapping = E.ColumnNameMapping()
        k1 = (
            "abcdefghijkABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABC"
        )
        k2 = (
            "ebcdefghijkABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABC"
        )
        mapping.add(k1)
        mapping.add(k2)
        mapping.do_mapping()
        assert (
            mapping[k1]
            == "bABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABC"
        )
        assert (
            mapping[k2]
            == "ebcdefghijkABCDFHIJABCDEFGHIJABCFGHIACDEFGHJABCDEFHIJABCDEFIJABC"
        )

    def test_leave_start_intact():
        """The column mapper should leave stuff before the first _ alone"""
        mapping = E.ColumnNameMapping(25)
        k1 = "leaveme_EVEN_THOUGH_WE_LIKE_REMOVING_LOWER_CASE_VOWELS"
        k2 = "keepmee_EVEN_THOUGH_WE_LIKE_REMOVING_LOWER_CASE_VOWELS"
        mapping.add(k1)
        mapping.add(k2)
        mapping.do_mapping()
        assert mapping[k1].startswith("leaveme_")
        assert mapping[k2].startswith("keepmee_")

    def per_object_statement(module, object_name, fields):
        """Return a statement that will select the given fields from the table"""
        field_string = ", ".join(
            [
                field
                if field.startswith(object_name)
                else "%s_%s" % (object_name, field)
                for field in fields
            ]
        )
        statement = (
            "select ImageNumber, %s_%s, %s "
            "from %sPer_%s order by ImageNumber, %s_%s"
            % (
                object_name,
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                field_string,
                module.table_prefix.value,
                object_name,
                object_name,
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            )
        )
        return statement

    def check_experiment_table(cursor, module, m):
        """Check the per_experiment table values against measurements"""
        statement = "select %s, %s, %s from %s" % (
            cpp.M_PIPELINE,
            cpp.M_VERSION,
            cpp.M_TIMESTAMP,
            module.get_table_name(cpmeas.EXPERIMENT),
        )
        cursor.execute(statement)
        row = cursor.fetchone()
        with pytest.raises(StopIteration):
            cursor.__next__()
        assert len(row) == 3
        for feature, value in zip(
            (cpp.M_PIPELINE, cpp.M_VERSION, cpp.M_TIMESTAMP), row
        ):
            assert cellprofiler.utilities.legacy.equals(
                value, m.get_experiment_measurement(feature)
            )

    def test_write_mysql_db():
        """Multiple objects / write - per-object tables"""
        workspace, module, output_dir, finally_fn = make_workspace(True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.post_run(workspace)
            load_database(output_dir, module)
            check_experiment_table(cursor, module, workspace.measurements)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_mysql_db_filter_objs():
        workspace, module, output_dir, finally_fn = make_workspace(True, True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_SELECT
            module.objects_list.choices = [OBJECT_NAME, ALTOBJECT_NAME]
            module.objects_list.value = OBJECT_NAME
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = os.path.join(
                output_dir, base_name + "_" + OBJECT_NAME + ".CSV"
            )
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename)
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_mysql_direct():
        """Write directly to the mysql DB, not to a file"""
        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            check_experiment_table(cursor, module, workspace.measurements)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_long_colname():
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = make_workspace(False, long_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_img_column = mappings["Image_%s" % LONG_IMG_MEASUREMENT]
            long_obj_column = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]

            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    long_img_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, long_obj_column]
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_nulls():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.NaN, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = np.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir, base_name + "_" + cpmeas.IMAGE + ".CSV"
            )
            object_file = os.path.join(
                output_dir, "%s_%s.CSV" % (base_name, OBJECT_NAME)
            )
            for filename in (sql_file, image_file, object_file):
                assert os.path.isfile(filename)
            fd = open(sql_file, "rt")
            sql_text = fd.read()
            fd.close()
            os.chdir(output_dir)
            for statement in sql_text.split(";"):
                if len(statement.strip()) == 0:
                    continue
                cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - np.mean(om[~np.isnan(om)])), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_01_mysql_direct_null():
        """Write directly to the mysql DB, not to a file and write nulls"""
        workspace, module = make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.NaN, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[:] = np.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_02_mysql_direct_inf():
        """regression test of img-1149: infinite values"""
        workspace, module = make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, np.NaN, True, 1)
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[:] = np.inf
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, "
                "Image_Count_%s, Mean_%s_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_wierd_colname():
        """Write to MySQL, even if illegal characters are in the column name"""
        workspace, module = make_workspace(False, wierd_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            wierd_img_column = mappings["Image_%s" % WIERD_IMG_MEASUREMENT]
            wierd_obj_column = mappings["%s_%s" % (OBJECT_NAME, WIERD_OBJ_MEASUREMENT)]

            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    wierd_img_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, wierd_obj_column]
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_50_char_colname():
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = make_workspace(False, long_measurement=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_img_column = mappings["Image_%s" % LONG_IMG_MEASUREMENT]
            long_obj_column = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            assert len(long_img_column) <= 50
            assert len(long_obj_column) <= 50
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s,"
                "Image_Count_%s, %s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    long_img_column,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, long_obj_column]
            )
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_01_write_two_object_tables_direct():
        """Write two object tables using OT_PER_OBJECT"""
        workspace, module = make_workspace(False, alt_object=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Read from one object table
            #
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read from the other table
            #
            statement = per_object_statement(module, ALTOBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i in range(len(ALTOBJ_VALUE)):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - ALTOBJ_VALUE[i]), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    def test_02_write_two_object_tables_csv():
        """Write two object tables using OT_PER_OBJECT"""
        workspace, module, output_dir, finally_fn = make_workspace(
            True, alt_object=True
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            load_database(output_dir, module)
            #
            # Read from one object table
            #

            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read from the other table
            #
            statement = per_object_statement(module, ALTOBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i in range(len(ALTOBJ_VALUE)):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - ALTOBJ_VALUE[i]), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    def test_write_mysql_db_as_data_tool():
        """Multiple objects / write - per-object tables"""
        workspace, module, output_dir, finally_fn = make_workspace(
            True, image_set_count=2
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.run_as_data_tool(workspace)
            load_database(output_dir, module, image_set_count=2)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s order by ImageNumber"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            for j in range(2):
                row = cursor.fetchone()
                assert len(row) == 5
                assert row[0] == j + 1
                assert round(abs(row[1] - INT_VALUE), 7) == 0
                assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
                assert row[3] == STRING_VALUE
                assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_data_tool_and_get_measurement_columns():
        # Regression test of issue #444
        #
        # Old measurements might not conform to get_measurement_columns
        # if a new measurement has been added.
        #
        workspace, module = make_workspace(
            False, image_set_count=2, long_measurement=MISSING_FROM_MEASUREMENTS
        )
        try:
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.run_as_data_tool(workspace)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = """
            select ImageNumber, Image_Group_Number, Image_Group_Index,
                   Image_%s, Image_%s, Image_%s, Image_Count_%s
                   from %s""" % (
                INT_IMG_MEASUREMENT,
                FLOAT_IMG_MEASUREMENT,
                STRING_IMG_MEASUREMENT,
                OBJECT_NAME,
                image_table,
            )
            cursor.execute(statement)
            for i in range(2):
                row = cursor.fetchone()
                assert len(row) == 7
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_data_tool_and_get_measurement_columns():
        # Regression test of issue #444
        #
        # Old measurements might not conform to get_measurement_columns
        # if an old measurement has been removed
        #
        workspace, module = make_workspace(
            False, image_set_count=2, long_measurement=MISSING_FROM_MODULE
        )
        try:
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.run_as_data_tool(workspace)
            mappings = module.get_column_name_mappings(
                workspace.pipeline, workspace.image_set_list
            )
            long_mean_col = mappings["Mean_%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            long_obj_col = mappings["%s_%s" % (OBJECT_NAME, LONG_OBJ_MEASUREMENT)]
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = """
            select ImageNumber, Image_Group_Number, Image_Group_Index,
                   Image_%s, Image_%s, Image_%s, Image_Count_%s, %s
                   from %s""" % (
                INT_IMG_MEASUREMENT,
                FLOAT_IMG_MEASUREMENT,
                STRING_IMG_MEASUREMENT,
                OBJECT_NAME,
                long_mean_col,
                image_table,
            )
            cursor.execute(statement)
            for i in range(2):
                row = cursor.fetchone()
                assert len(row) == 8
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
                assert abs(row[7] - np.mean(OBJ_VALUE)) < 0.0001
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s, %s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    long_obj_col,
                    module.table_prefix.value,
                )
            )
            cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = cursor.fetchone()
                    assert len(row) == 4
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
                    assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_write_sqlite_direct():
        """Write directly to a SQLite database"""
        workspace, module, output_dir, finally_fn = make_workspace(True)
        cursor = None
        connection = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = get_sqlite_cursor(module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                "from %s"
                % (
                    INT_IMG_MEASUREMENT,
                    FLOAT_IMG_MEASUREMENT,
                    STRING_IMG_MEASUREMENT,
                    OBJECT_NAME,
                    image_table,
                )
            )
            cursor.execute(statement)
            row = cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                cursor.__next__()
            statement = per_object_statement(module, OBJECT_NAME, [OBJ_MEASUREMENT])
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def execute_well_sql(output_dir, module):
        file_name = "SQL__Per_Well_SETUP.SQL"
        sql_file = os.path.join(output_dir, file_name)
        fd = open(sql_file, "rt")
        sql_text = fd.read()
        fd.close()
        print(sql_text)
        for statement in sql_text.split(";"):
            if len(statement.strip()) == 0:
                continue
            cursor.execute(statement)

    def select_well_agg(module, aggname, fields):
        field_string = ", ".join(["%s_%s" % (aggname, field) for field in fields])
        statement = (
            "select Image_Metadata_Plate, Image_Metadata_Well, %s "
            "from %sPer_Well_%s" % (field_string, module.table_prefix.value, aggname)
        )
        return statement

    def test_well_single_objtable():
        workspace, module, output_dir, finally_fn = make_workspace(
            False, well_metadata=True, image_set_count=3
        )
        aggs = [("avg", np.mean), ("std", np.std)]
        if mysql_has_median:
            aggs.append(("median", np.median))
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_COMBINE
            module.wants_agg_mean_well.value = True
            module.wants_agg_median_well.value = mysql_has_median
            module.wants_agg_std_dev_well.value = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            execute_well_sql(output_dir, module)
            meas = (
                (cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT),
                (cpmeas.IMAGE, INT_IMG_MEASUREMENT),
                (OBJECT_NAME, OBJ_MEASUREMENT),
            )
            m = workspace.measurements
            image_numbers = m.get_image_numbers()

            for aggname, aggfn in aggs:
                fields = [
                    "%s_%s" % (object_name, feature) for object_name, feature in meas
                ]
                statement = select_well_agg(module, aggname, fields)
                cursor.execute(statement)
                rows = cursor.fetchall()
                assert len(rows) == 1
                row = rows[0]
                assert row[0] == PLATE
                assert row[1] == WELL
                for i, (object_name, feature) in enumerate(meas):
                    value = row[i + 2]
                    values = m[object_name, feature, image_numbers]
                    expected = aggfn(values)
                    if np.isnan(expected):
                        assert value is None
                    else:
                        assert round(abs(float(value) - expected), 7) == 0
        finally:
            drop_tables(
                module, ["Per_Image", "Per_Object"] + ["Per_Well_" + x for x, _ in aggs]
            )
            finally_fn()

    def test_well_two_objtables():
        workspace, module, output_dir, finally_fn = make_workspace(
            False, well_metadata=True, alt_object=True, image_set_count=3
        )
        aggs = [("avg", np.mean), ("std", np.std)]
        if mysql_has_median:
            aggs.append(("median", np.median))
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.wants_agg_mean_well.value = True
            module.wants_agg_median_well.value = mysql_has_median
            module.wants_agg_std_dev_well.value = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1, 2, 3])
            module.run(workspace)
            module.post_run(workspace)
            execute_well_sql(output_dir, module)
            meas = (
                (cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT),
                (cpmeas.IMAGE, INT_IMG_MEASUREMENT),
                (OBJECT_NAME, OBJ_MEASUREMENT),
                (ALTOBJECT_NAME, OBJ_MEASUREMENT),
            )
            m = workspace.measurements
            image_numbers = m.get_image_numbers()
            for aggname, aggfn in aggs:
                fields = [
                    "%s_%s" % (object_name, feature) for object_name, feature in meas
                ]
                statement = select_well_agg(module, aggname, fields)
                cursor.execute(statement)
                rows = cursor.fetchall()
                assert len(rows) == 1
                row = rows[0]
                assert row[0] == PLATE
                assert row[1] == WELL
                for i, (object_name, feature) in enumerate(meas):
                    value = row[i + 2]
                    values = m[object_name, feature, image_numbers]
                    expected = aggfn(values)
                    if np.isnan(expected):
                        assert value is None
                    else:
                        assert round(abs(float(value) - expected), 7) == 0
        finally:
            drop_tables(
                module,
                ["Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME]
                + ["Per_Well_" + x for x, _ in aggs],
            )
            finally_fn()

    def test_image_thumbnails():
        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_NONE
            module.max_column_size.value = 50
            module.separate_object_tables.value = E.OT_COMBINE
            module.wants_agg_mean_well.value = False
            module.wants_agg_median_well.value = False
            module.wants_agg_std_dev_well.value = False
            module.want_image_thumbnails.value = True
            module.thumbnail_image_names.load_choices(workspace.pipeline)
            module.thumbnail_image_names.value = module.thumbnail_image_names.choices[0]
            columns = module.get_measurement_columns(workspace.pipeline)
            assert len(columns) == 1
            expected_thumbnail_column = "Thumbnail_" + IMAGE_NAME
            assert columns[0][1] == expected_thumbnail_column
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            image_table = module.table_prefix.value + "Per_Image"
            stmt = "select Image_%s from %s" % (expected_thumbnail_column, image_table)
            cursor.execute(stmt)
            result = cursor.fetchall()
            im = PILImage.open(StringIO(result[0][0].decode("base64")))
            assert tuple(im.size) == (200, 200)

        finally:
            drop_tables(module, ["Per_Image"])

    def test_image_thumbnails_sqlite():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        cursor = None
        connection = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_NONE
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.want_image_thumbnails.value = True
            module.thumbnail_image_names.load_choices(workspace.pipeline)
            module.thumbnail_image_names.value = module.thumbnail_image_names.choices[0]
            columns = module.get_measurement_columns(workspace.pipeline)
            assert len(columns) == 1
            expected_thumbnail_column = "Thumbnail_" + IMAGE_NAME
            assert columns[0][1] == expected_thumbnail_column
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            image_table = module.table_prefix.value + "Per_Image"
            stmt = "select Image_%s from %s" % (expected_thumbnail_column, image_table)
            cursor, connection = get_sqlite_cursor(module)
            cursor.execute(stmt)
            result = cursor.fetchall()
            im = PILImage.open(io.BytesIO(base64.b64decode(result[0][0])))
            assert tuple(im.size) == (200, 200)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_post_group_single_object_table():
        """Write out measurements that are only available post-group"""
        count = 5
        workspace, module = make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(module, E.ExportToDatabase)
        assert isinstance(workspace, cpw.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, np.arange(count) + 1)
            for i in range(count):
                workspace.set_image_set_for_testing_only(i + 1)
                measurements.next_image_set(i + 1)
                module.run(workspace)
            #
            # Read the image data after the run but before group.
            # It should be null.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s order by ImageNumber" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_single_object_table_agg():
        """Test single object table, post_group aggregation"""
        count = 5
        workspace, module = make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(module, E.ExportToDatabase)
        assert isinstance(workspace, cpw.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        module.wants_agg_mean.value = True
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, np.arange(count) + 1)
            for i in range(count):
                workspace.set_image_set_for_testing_only(i + 1)
                measurements.next_image_set(i + 1)
                module.run(workspace)
            #
            # Read the image data after the run but before group.
            # It should be null.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] is None
                assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
                assert round(abs(row[2] - np.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_separate_object_tables():
        """Write out measurements post_group to separate object tables"""
        count = 5
        workspace, module = make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(module, E.ExportToDatabase)
        assert isinstance(workspace, cpw.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, np.arange(count) + 1)
            for i in range(count):
                workspace.set_image_set_for_testing_only(i + 1)
                measurements.next_image_set(i + 1)
                module.run(workspace)
            #
            # Read the image data after the run but before group.
            # It should be null.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s order by ImageNumber" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data too
            #
            statement = per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data
            #
            statement = per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_post_group_separate_table_agg():
        """Test single object table, post_group aggregation"""
        count = 5
        workspace, module = make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(module, E.ExportToDatabase)
        assert isinstance(workspace, cpw.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        module.wants_agg_mean.value = True
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = E.OT_PER_OBJECT
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, np.arange(count) + 1)
            for i in range(count):
                workspace.set_image_set_for_testing_only(i + 1)
                measurements.next_image_set(i + 1)
                module.run(workspace)
            #
            # Read the image data after the run but before group.
            # It should be null.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] is None
                assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data too
            #
            statement = per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
                assert round(abs(row[2] - np.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data
            #
            statement = per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_sqlite():
        for with_interaction_handler in (False, True):
            count = 5
            workspace, module, output_dir, finally_fn = make_workspace(
                True, image_set_count=count, group_measurement=True
            )
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = get_interaction_handler(
                    ran_interaction_handler
                )
            assert isinstance(module, E.ExportToDatabase)
            assert isinstance(workspace, cpw.Workspace)
            measurements = workspace.measurements
            assert isinstance(measurements, cpmeas.Measurements)
            module.db_type.value = E.DB_SQLITE
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            cursor, connection = get_sqlite_cursor(module)
            try:
                module.separate_object_tables.value = E.OT_COMBINE
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, np.arange(count) + 1)
                for i in range(count):
                    workspace.set_image_set_for_testing_only(i + 1)
                    measurements.next_image_set(i + 1)
                    module.run(workspace)
                    assert with_interaction_handler == ran_interaction_handler[0]
                    ran_interaction_handler[0] = False
                #
                # Read the image data after the run but before group.
                # It should be null.
                #
                image_table = module.table_prefix.value + "Per_Image"
                statement = (
                    "select ImageNumber, Image_%s from %s order by ImageNumber"
                    % (GROUP_IMG_MEASUREMENT, image_table)
                )
                cursor.execute(statement)
                for i in range(count):
                    row = cursor.fetchone()
                    assert len(row) == 2
                    assert row[0] == i + 1
                    assert row[1] is None
                with pytest.raises(StopIteration):
                    cursor.__next__()
                #
                # Read the object data too
                #
                object_table = module.table_prefix.value + "Per_Object"
                statement = (
                    "select ImageNumber, ObjectNumber, %s_%s "
                    "from %sPer_Object order by ImageNumber, ObjectNumber"
                    % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
                )
                cursor.execute(statement)
                for i in range(count):
                    for j in range(len(OBJ_VALUE)):
                        row = cursor.fetchone()
                        assert len(row) == 3
                        assert row[0] == i + 1
                        assert row[1] == j + 1
                        assert row[2] is None
                with pytest.raises(StopIteration):
                    cursor.__next__()
                #
                # Run post_group and see that the values do show up
                #
                module.post_group(workspace, {})
                assert with_interaction_handler == ran_interaction_handler[0]
                image_table = module.table_prefix.value + "Per_Image"
                statement = "select ImageNumber, Image_%s from %s" % (
                    GROUP_IMG_MEASUREMENT,
                    image_table,
                )
                cursor.execute(statement)
                for i in range(count):
                    row = cursor.fetchone()
                    assert len(row) == 2
                    assert row[0] == i + 1
                    assert row[1] == INT_VALUE
                with pytest.raises(StopIteration):
                    cursor.__next__()
                #
                # Read the object data
                #
                object_table = module.table_prefix.value + "Per_Object"
                statement = (
                    "select ImageNumber, ObjectNumber, %s_%s "
                    "from %sPer_Object order by ImageNumber, ObjectNumber"
                    % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
                )
                cursor.execute(statement)
                for i in range(count):
                    for j in range(len(OBJ_VALUE)):
                        row = cursor.fetchone()
                        assert len(row) == 3
                        assert row[0] == i + 1
                        assert row[1] == j + 1
                        assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
                with pytest.raises(StopIteration):
                    cursor.__next__()
            finally:
                cursor.close()
                connection.close()
                finally_fn()

    def test_post_group_object_view():
        """Write out measurements post_group to single object view"""
        count = 5
        workspace, module = make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(module, E.ExportToDatabase)
        assert isinstance(workspace, cpw.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = E.OT_VIEW
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, np.arange(count) + 1)
            for i in range(count):
                workspace.set_image_set_for_testing_only(i + 1)
                measurements.next_image_set(i + 1)
                module.run(workspace)
            #
            # Read the image data after the run but before group.
            # It should be null.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s order by ImageNumber" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                cursor.__next__()
            close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            cursor.execute(statement)
            for i in range(count):
                row = cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                cursor.__next__()
            #
            # Finally, confirm that the Per_Object item is a view
            #
            statement = (
                "SELECT * FROM information_schema.views WHERE TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'"
                % (module.db_name.value, object_table)
            )
            cursor.execute(statement)
            assert len(cursor.fetchall()) != 0
        finally:
            drop_tables(module, ["Per_Image"])
            drop_views(module, ["Per_Object"])

    def test_properties_file():
        old_get_measurement_columns = E.ExportToDatabase.get_measurement_columns

        def get_measurement_columns(
            pipeline, old_get_measurement_columns=old_get_measurement_columns
        ):
            result = [
                (cpmeas.IMAGE, C_FILE_NAME + "_" + IMAGE_NAME, cpmeas.COLTYPE_VARCHAR),
                (cpmeas.IMAGE, C_PATH_NAME + "_" + IMAGE_NAME, cpmeas.COLTYPE_VARCHAR),
            ] + old_get_measurement_columns(pipeline)
            return result

        E.ExportToDatabase.get_measurement_columns = get_measurement_columns
        workspace, module, output_dir, finally_fn = make_workspace(
            True, alt_object=True
        )
        assert isinstance(module, E.ExportToDatabase)
        file_name = "%s_%s.properties" % (DB_NAME, module.get_table_prefix())
        path = os.path.join(output_dir, file_name)
        #
        # Do a monkey-patch of ExportToDatabase.get_measurement_columns
        #
        try:
            m = workspace.measurements
            for image_number in m.get_image_numbers():
                m.add_measurement(
                    cpmeas.IMAGE,
                    C_FILE_NAME + "_" + IMAGE_NAME,
                    os.path.join(path, "img%d.tif" % image_number),
                    image_set_number=image_number,
                )
                m.add_measurement(
                    cpmeas.IMAGE,
                    C_PATH_NAME + "_" + IMAGE_NAME,
                    os.path.join(path, "img%d.tif" % image_number),
                    image_set_number=image_number,
                )
            module.db_type.value = E.DB_MYSQL_CSV
            module.db_name.value = DB_NAME
            module.db_host.value = DB_HOST
            module.db_user.value = DB_USER
            module.db_passwd.value = DB_PASSWORD
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.save_cpa_properties.value = True
            module.location_object.value = OBJECT_NAME
            module.post_run(workspace)
            fd = open(path, "rt")
            text = fd.read()
            fd.close()
            #
            # Parse the file
            #
            dictionary = {}
            for line in text.split("\n"):
                line = line.strip()
                if (not line.startswith("#")) and line.find("=") != -1:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    dictionary[k] = v
            for k, v in (
                ("db_type", "mysql"),
                ("db_port", ""),  # The CSV file has nulls in lots of places
                ("db_host", ""),
                ("db_name", DB_NAME),
                ("db_user", ""),
                ("db_passwd", ""),
                ("image_table", "%sPer_Image" % module.get_table_prefix()),
                ("object_table", "%sPer_Object" % module.get_table_prefix()),
                ("image_id", "ImageNumber"),
                ("object_id", "ObjectNumber"),
                ("cell_x_loc", "%s_Location_Center_X" % OBJECT_NAME),
                ("cell_y_loc", "%s_Location_Center_Y" % OBJECT_NAME),
                (
                    "image_path_cols",
                    "%s_%s_%s" % (cpmeas.IMAGE, C_PATH_NAME, IMAGE_NAME),
                ),
                (
                    "image_file_cols",
                    "%s_%s_%s" % (cpmeas.IMAGE, C_FILE_NAME, IMAGE_NAME),
                ),
            ):
                assert k in dictionary
                assert dictionary[k] == v
        finally:
            E.ExportToDatabase.get_measurement_columns = old_get_measurement_columns
            os.chdir(output_dir)
            if os.path.exists(path):
                os.unlink(path)
            finally_fn()

    def test_experiment_table_combine():
        workspace, module = make_workspace(False, True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.location_object.value = OBJECT_NAME
            if not __test_mysql:
                skipTest("Skipping actual DB work, not at the Broad.")
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Find the experiment ID by looking for the image table in
            # the properties.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                """
                select max(experiment_id) from Experiment_Properties
                where field = 'image_table' and value = '%s'"""
                % image_table
            )
            cursor.execute(statement)
            experiment_id = int(cursor.fetchone()[0])
            with pytest.raises(StopIteration):
                cursor.__next__()
            properties = module.get_property_file_text(workspace)
            assert len(properties) == 1
            for k, v in list(properties[0].properties.items()):
                statement = """
                select max(value) from Experiment_Properties where
                field = '%s' and experiment_id = %d and object_name = '%s'
                """ % (
                    k,
                    experiment_id,
                    properties[0].object_name,
                )
                cursor.execute(statement)
                dbvalue = cursor.fetchone()[0]
                with pytest.raises(StopIteration):
                    cursor.__next__()
                assert dbvalue == v
        finally:
            drop_tables(module, ("Per_Image", "Per_Object"))

    def test_experiment_table_separate():
        workspace, module = make_workspace(False, True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_PER_OBJECT
            if not __test_mysql:
                skipTest("Skipping actual DB work, not at the Broad.")
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Find the experiment ID by looking for the image table in
            # the properties.
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                """
                select max(experiment_id) from Experiment_Properties
                where field = 'image_table' and value = '%s'"""
                % image_table
            )
            cursor.execute(statement)
            experiment_id = int(cursor.fetchone()[0])
            with pytest.raises(StopIteration):
                cursor.__next__()
            properties = module.get_property_file_text(workspace)
            assert len(properties) == 2
            for k, v in list(properties[0].properties.items()):
                statement = """
                select max(value) from Experiment_Properties where
                field = '%s' and experiment_id = %d and object_name = '%s'
                """ % (
                    k,
                    experiment_id,
                    properties[0].object_name,
                )
                cursor.execute(statement)
                dbvalue = cursor.fetchone()[0]
                with pytest.raises(StopIteration):
                    cursor.__next__()
                assert dbvalue == v
        finally:
            drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    def test_write_no_mysql_relationships():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")
        workspace, module, output_dir, finally_fn = make_workspace(
            True, relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            load_database(output_dir, module)
            tteesstt_no_relationships(module, cursor)
        finally:
            drop_tables(module)
            os.chdir(output_dir)
            finally_fn()

    def test_write_no_mysql_direct_relationships():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(
            False, relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            tteesstt_no_relationships(module, cursor)

        finally:
            drop_tables(module)

    def test_write_sqlite_no_relationships():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module, output_dir, finally_fn = make_workspace(
            True, relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE
        )
        cursor = None
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = get_sqlite_cursor(module)
            tteesstt_no_relationships(module, cursor)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_write_mysql_relationships():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")
        workspace, module, output_dir, finally_fn = make_workspace(
            True,
            relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.post_run(workspace)
            load_database(output_dir, module)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            drop_tables(module)
            os.chdir(output_dir)
            finally_fn()

    def test_write_mysql_direct_relationships():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(
            False,
            relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            drop_tables(module)

    def test_write_sqlite_relationships():
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = make_workspace(
                True,
                relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = get_interaction_handler(
                    ran_interaction_handler
                )
            try:
                assert isinstance(module, E.ExportToDatabase)
                module.db_type.value = E.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = E.O_ALL
                module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = E.OT_COMBINE
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.run(workspace)
                cursor, connection = get_sqlite_cursor(module)
                tteesstt_relate(workspace.measurements, module, cursor)
            finally:
                if cursor is not None:
                    cursor.close()
                if connection is not None:
                    connection.close()
                if hasattr(module, "cursor") and module.cursor is not None:
                    module.cursor.close()
                if hasattr(module, "connection") and module.connection is not None:
                    module.connection.close()
                finally_fn()

    def test_write_sqlite_duplicates():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module, output_dir, finally_fn = make_workspace(
            True,
            relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_DUPLICATE,
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = get_sqlite_cursor(module)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor") and module.cursor is not None:
                module.cursor.close()
            if hasattr(module, "connection") and module.connection is not None:
                module.connection.close()
            finally_fn()

    def test_add_relationship_id_mysql():
        #
        # Add a missing relationship ID
        #
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(
            False,
            relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            #
            # Get rid of the module dictionary entry and the table row
            #
            module.get_dictionary()[E.T_RELATIONSHIP_TYPES] = {}
            cursor.execute(
                "delete from %s" % module.get_table_name(E.T_RELATIONSHIP_TYPES)
            )
            close_connection()
            module.run(workspace)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            drop_tables(module)

    def test_get_relationship_id_mysql():
        #
        # Get a missing relationship ID (e.g., worker # 2 gets worker # 1's row)
        #
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(
            False,
            relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            #
            # Get rid of the module dictionary entry and the table row
            #
            module.get_dictionary()[E.T_RELATIONSHIP_TYPES] = {}
            module.run(workspace)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            drop_tables(module)

    def test_add_relationship_id_sqlite():
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = make_workspace(
                True,
                relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            if with_interaction_handler:
                ran_interaction_handler = [False]
                workspace.interaction_handler = get_interaction_handler(
                    ran_interaction_handler
                )
            try:
                assert isinstance(module, E.ExportToDatabase)
                module.db_type.value = E.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = E.O_ALL
                module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = E.OT_COMBINE
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                with E.DBContext(module) as (connection, cursor):
                    cursor.execute(
                        "delete from %s" % module.get_table_name(E.T_RELATIONSHIP_TYPES)
                    )
                module.get_dictionary()[E.T_RELATIONSHIP_TYPES] = {}
                module.run(workspace)
                with E.DBContext(module) as (connection, cursor):
                    tteesstt_relate(workspace.measurements, module, cursor)
            finally:
                finally_fn()

    def test_get_relationship_id_sqlite():
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = make_workspace(
                True,
                relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            if with_interaction_handler:
                ran_interaction_handler = [False]
                workspace.interaction_handler = get_interaction_handler(
                    ran_interaction_handler
                )
            cursor = None
            connection = None
            try:
                assert isinstance(module, E.ExportToDatabase)
                module.db_type.value = E.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = E.O_ALL
                module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = E.OT_COMBINE
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.get_dictionary()[E.T_RELATIONSHIP_TYPES] = {}
                module.run(workspace)
                cursor, connection = get_sqlite_cursor(module)
                tteesstt_relate(workspace.measurements, module, cursor)
            finally:
                if cursor is not None:
                    cursor.close()
                if connection is not None:
                    connection.close()
                if hasattr(module, "cursor") and module.cursor is not None:
                    module.cursor.close()
                if hasattr(module, "connection") and module.connection is not None:
                    module.connection.close()
                finally_fn()

    def test_write_mysql_direct_relationships():
        # Regression test of #1757
        #
        # No relationships in relationships table and ExportToDatabase
        # is configured to display its window
        #
        if not __test_mysql:
            skipTest("Skipping actual DB work, no database configured.")

        workspace, module = make_workspace(
            False, relationship_type=cpmeas.MCA_AVAILABLE_EACH_CYCLE
        )
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            module.show_window = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            tteesstt_relate(workspace.measurements, module, cursor)
        finally:
            drop_tables(module)

    def test_mysql_no_overwrite():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.allow_overwrite.value = E.OVERWRITE_NEVER
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            assert module.prepare_run(workspace)
            assert not module.prepare_run(workspace)
        finally:
            drop_tables(module)

    def test_mysql_keep_schema():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.allow_overwrite.value = E.OVERWRITE_DATA
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(cpmeas.IMAGE)
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
            close_connection()
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
            assert module.prepare_run(workspace)
            #
            # The row should still be there after the second prepare_run
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
        finally:
            drop_tables(module)

    def test_mysql_drop_schema():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")

        workspace, module = make_workspace(False)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.allow_overwrite.value = E.OVERWRITE_ALL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(cpmeas.IMAGE)
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
            close_connection()
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
            close_connection()
            assert module.prepare_run(workspace)
            #
            # The row should not be there after the second prepare_run
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
        finally:
            drop_tables(module)

    def test_sqlite_no_overwrite():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        assert isinstance(module, E.ExportToDatabase)
        module.db_type.value = E.DB_SQLITE
        module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.allow_overwrite.value = E.OVERWRITE_NEVER
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = E.O_ALL
        module.separate_object_tables.value = E.OT_COMBINE
        try:
            assert module.prepare_run(workspace)
            assert not module.prepare_run(workspace)
        finally:
            finally_fn()

    def test_sqlite_keep_schema():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        assert isinstance(module, E.ExportToDatabase)
        module.db_type.value = E.DB_SQLITE
        module.allow_overwrite.value = E.OVERWRITE_DATA
        module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = E.O_ALL
        module.separate_object_tables.value = E.OT_COMBINE
        cursor, connection = get_sqlite_cursor(module)
        try:
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(cpmeas.IMAGE)
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
            assert module.prepare_run(workspace)
            #
            # The row should still be there after the second prepare_run
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
        finally:
            cursor.close()
            connection.close()
            finally_fn()

    def test_sqlite_drop_schema():
        workspace, module, output_dir, finally_fn = make_workspace(True)
        assert isinstance(module, E.ExportToDatabase)
        module.db_type.value = E.DB_SQLITE
        module.allow_overwrite.value = E.OVERWRITE_ALL
        module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = E.O_ALL
        module.separate_object_tables.value = E.OT_COMBINE
        cursor, connection = get_sqlite_cursor(module)
        try:
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(cpmeas.IMAGE)
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 1
            assert module.prepare_run(workspace)
            #
            # The row should not be there after the second prepare_run
            #
            cursor.execute(how_many)
            assert cursor.fetchall()[0][0] == 0
        finally:
            drop_tables(module)

    def test_dbcontext_mysql():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")
        module = E.ExportToDatabase()
        module.db_type.value = E.DB_MYSQL
        module.db_host.value = MYSQL_HOST
        module.db_user.value = MYSQL_USER
        module.db_passwd.value = MYSQL_PASSWORD
        module.db_name.value = MYSQL_DATABASE
        with E.DBContext(module) as (connection, cursor):
            cursor.execute("select 1")
            result = cursor.fetchall()
            assert len(result) == 1
            assert result[0][0] == 1

    def test_dbcontext_sqlite():
        output_dir = tempfile.mkdtemp()
        try:
            module = E.ExportToDatabase()
            module.db_type.value = E.DB_SQLITE
            module.directory.dir_choice = ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = output_dir
            with E.DBContext(module) as (connection, cursor):
                cursor.execute("select 1")
                result = cursor.fetchall()
                assert len(result) == 1
                assert result[0][0] == 1
        finally:
            try:
                for filename in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, filename))
                os.rmdir(output_dir)
            except:
                print(("Failed to remove %s" % output_dir))

    def test_post_run_experiment_measurement_mysql():
        if not __test_mysql:
            skipTest("Skipping actual DB work, not at the Broad.")
        workspace, module = make_workspace(False, post_run_test=True)
        try:
            assert isinstance(module, E.ExportToDatabase)
            module.db_type.value = E.DB_MYSQL
            module.allow_overwrite.value = E.OVERWRITE_ALL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.separate_object_tables.value = E.OT_COMBINE
            workspace.measurements[
                cpmeas.EXPERIMENT, STRING_IMG_MEASUREMENT
            ] = STRING_VALUE
            assert module.prepare_run(workspace)
            cursor.execute(
                "select %s from %s"
                % (STRING_IMG_MEASUREMENT, module.get_table_name(cpmeas.EXPERIMENT))
            )
            result = cursor.fetchall()[0][0]
            assert result is None
            close_connection()
            module.post_run(workspace)
            cursor.execute(
                "select %s from %s"
                % (STRING_IMG_MEASUREMENT, module.get_table_name(cpmeas.EXPERIMENT))
            )
            assert cursor.fetchall()[0][0] == STRING_VALUE
        finally:
            drop_tables(module)
