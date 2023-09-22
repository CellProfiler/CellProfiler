import base64
import io
import os
import socket
import tempfile
import traceback
import unittest
import uuid

import PIL.Image
import numpy
import pytest
import six.moves
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER, COLTYPE_FLOAT, COLTYPE_VARCHAR_FORMAT, \
    M_NUMBER_OBJECT_NUMBER, C_NUMBER, FTR_OBJECT_NUMBER, GROUP_NUMBER, GROUP_INDEX, OBJECT, EXPERIMENT, COLTYPE_VARCHAR, \
    MCA_AVAILABLE_POST_RUN, MCA_AVAILABLE_POST_GROUP, MCA_AVAILABLE_EACH_CYCLE, C_FILE_NAME, C_PATH_NAME
from cellprofiler_core.constants.pipeline import M_PIPELINE, M_VERSION, M_TIMESTAMP
from cellprofiler_core.setting.text import ImageName, LabelName
from cellprofiler_core.utilities.legacy import equals

import tests.frontend.modules
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.modules
import cellprofiler.modules.exporttodatabase
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.setting
import cellprofiler_core.workspace

ogmc = cellprofiler.modules.exporttodatabase.ExportToDatabase.get_measurement_columns

if hasattr(unittest, "pytest.skip"):
    pytest.skipException = unittest.pytest.skip
else:
    pytest.skipException = None

numpy.random.seed(9804)

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

(
    OBJ_MEASUREMENT,
    INT_IMG_MEASUREMENT,
    FLOAT_IMG_MEASUREMENT,
    STRING_IMG_MEASUREMENT,
    LONG_IMG_MEASUREMENT,
    LONG_OBJ_MEASUREMENT,
    WIERD_IMG_MEASUREMENT,
    WIERD_OBJ_MEASUREMENT,
    GROUP_IMG_MEASUREMENT,
    GROUP_OBJ_MEASUREMENT,
) = [
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
OBJ_VALUE = numpy.array([1.5, 3.67, 2.8])
ALTOBJ_VALUE = numpy.random.uniform(size=100)
PLATE = "P-12345"
WELL = "A01"
DB_NAME = "MyDatabaseName"
DB_HOST = "MyHost"
DB_USER = "MyUser"
DB_PASSWORD = "MyPassword"

MYSQL_HOST = os.environ.get("CP_MYSQL_TEST_HOST", "localhost")
MYSQL_DATABASE = os.environ.get("CP_MYSQL_TEST_DB", "cellprofiler_test")
MYSQL_PASSWORD = os.environ.get("CP_MYSQL_TEST_PASSWORD", "password")
if MYSQL_PASSWORD == "None":
    MYSQL_PASSWORD = ""
MYSQL_USER = os.environ.get("CP_MYSQL_TEST_USER", "cellprofiler")

RTEST_NONE = 0
RTEST_SOME = 1
RTEST_DUPLICATE = 2


class TestExportToDatabase(unittest.TestCase):
    def setUp(self):
        self.__cursor = None
        self.__connection = None
        self.has_median = None
        try:
            if MYSQL_HOST.endswith("broadinstitute.org"):
                fqdn = socket.getfqdn().lower()
                if (
                    ("broadinstitute" in fqdn)
                    or fqdn.endswith("broad.mit.edu")
                    or fqdn.endswith("broad")
                ):
                    self.test_mysql = True
                elif socket.gethostbyaddr(socket.gethostname())[-1][0].startswith(
                    "69.173"
                ):
                    self.test_mysql = True
                else:
                    self.test_mysql = False
            self.test_mysql = True
        except:
            self.test_mysql = False

    @property
    def connection(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, no DB configured.")
        if self.__connection is None:
            import MySQLdb

            self.__connection = MySQLdb.connect(
                host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, local_infile=1
            )
        return self.__connection

    def close_connection(self):
        if self.test_mysql and self.connection is not None:
            if self.__cursor is not None:
                self.__cursor.close()
            self.__connection.close()
            self.__connection = None
            self.__cursor = None

    @property
    def cursor(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, database not configured.")
        if self.__cursor is None:
            import MySQLdb
            from MySQLdb.cursors import SSCursor

            self.__cursor = SSCursor(self.connection)
            try:
                self.__cursor.execute("use " + MYSQL_DATABASE)
            except:
                self.__cursor.execute("create database " + MYSQL_DATABASE)
                self.__cursor.execute("use " + MYSQL_DATABASE)
        return self.__cursor

    @property
    def mysql_has_median(self):
        """True if MySQL database has a median function"""
        if self.has_median is None:
            try:
                cursor = self.connection.cursor()
                cursor.execute("select median(1)")
                cursor.close()
                self.has_median = True
            except:
                self.has_median = False
        return self.has_median

    def get_sqlite_cursor(self, module):
        import sqlite3

        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        file_name = os.path.join(
            module.directory.get_absolute_path(), module.sqlite_file.value
        )
        connection = sqlite3.connect(file_name)
        cursor = connection.cursor()
        return cursor, connection

    def test_00_write_load_test(self):
        #
        # If this fails, you need to write a test for your variable revision
        # number change.
        #
        assert (
            cellprofiler.modules.exporttodatabase.ExportToDatabase.variable_revision_number
            == 28
        )

    def test_load_v11(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v11.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_SQLITE
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
        )
        assert module.directory.custom_path == r"./\\g<Plate>"
        assert module.db_name == "DefaultDB"

    def test_load_v12(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v12.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
        )
        assert module.directory.custom_path == r"./\\g<Plate>"
        assert module.db_name == "DefaultDB"
        assert module.max_column_size == 64

    def test_load_v13(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v13.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 2
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
        )
        assert module.directory.custom_path == r"./\\g<Plate>"
        assert module.db_name == "DefaultDB"
        assert module.max_column_size == 61

    def test_load_v15(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v15.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_SQLITE
        assert module.db_name == "Heel"
        assert not module.want_table_prefix
        assert module.table_prefix == "Ouch"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        )
        assert module.directory.custom_path == "//achilles/red/shoes"
        assert not module.save_cpa_properties
        assert module.db_host == "Zeus"
        assert module.db_user == "Hera"
        assert module.db_password == "Athena"
        assert module.sqlite_file == "Poseidon"
        assert module.wants_agg_mean
        assert not module.wants_agg_median
        assert not module.wants_agg_std_dev
        assert module.wants_agg_mean_well
        assert module.wants_agg_median_well
        assert not module.wants_agg_std_dev_well
        assert module.objects_choice == cellprofiler.modules.exporttodatabase.O_ALL
        assert module.max_column_size == 62
        assert (
            module.separate_object_tables
            == cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
        )
        assert not module.wants_properties_image_url_prepend

    def test_load_v22(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v22.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.experiment_name == "Sigma"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        )
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
                "Image",
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                "Image",
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                "Image",
                "Cells",
                "Height_Actin",
                "ImageNumber",
                "Image",
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
        assert g.filter_statement.value == r"Image_Metadata_Plate = \'1\'"

    def test_load_v23(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v23.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.experiment_name == "Sigma"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        )
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
                "Image",
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                "Image",
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                "Image",
                "Cells",
                "Height_Actin",
                "ImageNumber",
                "Image",
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
        assert g.filter_statement.value == r"Image_Metadata_Plate = \'1\'"

    def test_load_v24(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v24.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.experiment_name == "Sigma"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        )
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
                "Image",
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                "Image",
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                "Image",
                "Cells",
                "Height_Actin",
                "ImageNumber",
                "Image",
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
        assert g.filter_statement.value == r"Image_Metadata_Plate = \'1\'"
        assert (
            module.allow_overwrite
            == cellprofiler.modules.exporttodatabase.OVERWRITE_DATA
        )

    def test_load_v25(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v25.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.experiment_name == "Sigma"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        )
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
                "Image",
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                "Image",
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                "Image",
                "Cells",
                "Height_Actin",
                "ImageNumber",
                "Image",
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
        assert g.filter_statement.value == r"Image_Metadata_Plate = \'1\'"
        assert (
            module.allow_overwrite
            == cellprofiler.modules.exporttodatabase.OVERWRITE_NEVER
        )

    def test_load_v26(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v26.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(six.moves.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[-1]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_MYSQL
        assert module.db_name == "Gamma"
        assert module.want_table_prefix
        assert module.table_prefix == "Delta_"
        assert module.experiment_name == "Sigma"
        assert (
            module.directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        )
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
        assert (
            module.properties_classification_type
            == cellprofiler.modules.exporttodatabase.CT_OBJECT
        )
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
                "Image",
                "Mitochondria",
                "Width_DNA",
                "ImageNumber",
                "Image",
                "Nuclei",
                "Height_DNA",
                "ImageNumber",
            ),
            (
                module.workspace_measurement_groups[1],
                "PlateViewer",
                "Image",
                "Cells",
                "Height_Actin",
                "ImageNumber",
                "Image",
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
        assert g.filter_statement.value == r"Image_Metadata_Plate = \'1\'"
        assert (
            module.allow_overwrite
            == cellprofiler.modules.exporttodatabase.OVERWRITE_NEVER
        )

    def test_load_v27(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v27.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.load(six.moves.StringIO(data))
        module = pipeline.modules()[0]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert (
            module.properties_classification_type
            == cellprofiler.modules.exporttodatabase.CT_IMAGE
        )
        assert len(module.workspace_measurement_groups) == 1
        g = module.workspace_measurement_groups[0]
        assert g.y_object_name == "MyObjects"

    def test_load_v28(self):
        file = tests.frontend.modules.get_test_resources_directory("exporttodatabase/v28.pipeline")
        with open(file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.load(six.moves.StringIO(data))
        module = pipeline.modules()[0]
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert module.db_type.value == cellprofiler.modules.exporttodatabase.DB_SQLITE
        assert len(module.workspace_measurement_groups) == 1
        g = module.workspace_measurement_groups[0]
        assert g.y_object_name == "MyObjects"

    def make_workspace(
        self,
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

        class TestModule(cellprofiler_core.module.Module):
            module_name = "TestModule"
            module_num = 1
            variable_revision_number = 1

            def create_settings(self):
                self.image_name = ImageName(
                    "Foo", IMAGE_NAME
                )
                self.objects_name = LabelName(
                    "Bar", OBJECT_NAME
                )
                if alt_object:
                    self.altobjects_name = LabelName(
                        "Baz", ALTOBJECT_NAME
                    )

            def settings(self):
                return [self.image_name, self.objects_name] + (
                    [self.altobjects_name] if alt_object else []
                )

            @staticmethod
            def in_module(flag):
                """Return True to add the measurement to module's get_measurement_columns"""
                return flag and flag != MISSING_FROM_MODULE

            @staticmethod
            def in_measurements(flag):
                return flag and flag != MISSING_FROM_MEASUREMENTS

            def get_measurement_columns(self, pipeline):
                columns = [
                    (
                        "Image",
                        INT_IMG_MEASUREMENT,
                        COLTYPE_INTEGER,
                    ),
                    (
                        "Image",
                        FLOAT_IMG_MEASUREMENT,
                        COLTYPE_FLOAT,
                    ),
                    (
                        "Image",
                        STRING_IMG_MEASUREMENT,
                        COLTYPE_VARCHAR_FORMAT % 40,
                    ),
                    (
                        "Image",
                        OBJECT_COUNT_MEASUREMENT,
                        COLTYPE_INTEGER,
                    ),
                    (
                        OBJECT_NAME,
                        M_NUMBER_OBJECT_NUMBER,
                        COLTYPE_INTEGER,
                    ),
                    (
                        OBJECT_NAME,
                        OBJ_MEASUREMENT,
                        COLTYPE_FLOAT,
                    ),
                ]
                if self.in_module(alt_object):
                    columns += [
                        (
                            "Image",
                            ALTOBJECT_COUNT_MEASUREMENT,
                            COLTYPE_INTEGER,
                        ),
                        (
                            ALTOBJECT_NAME,
                            M_NUMBER_OBJECT_NUMBER,
                            COLTYPE_INTEGER,
                        ),
                        (
                            ALTOBJECT_NAME,
                            OBJ_MEASUREMENT,
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.in_module(long_measurement):
                    columns += [
                        (
                            "Image",
                            LONG_IMG_MEASUREMENT,
                            COLTYPE_INTEGER,
                        ),
                        (
                            OBJECT_NAME,
                            LONG_OBJ_MEASUREMENT,
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.in_module(wierd_measurement):
                    columns += [
                        (
                            "Image",
                            WIERD_IMG_MEASUREMENT,
                            COLTYPE_INTEGER,
                        ),
                        (
                            OBJECT_NAME,
                            WIERD_OBJ_MEASUREMENT,
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.in_module(well_metadata):
                    columns += [
                        (
                            "Image",
                            "Metadata_Plate",
                            COLTYPE_VARCHAR_FORMAT % 20,
                        ),
                        (
                            "Image",
                            "Metadata_Well",
                            COLTYPE_VARCHAR_FORMAT % 3,
                        ),
                    ]
                if self.in_module(group_measurement):
                    d = {MCA_AVAILABLE_POST_GROUP: True}
                    columns += [
                        (
                            "Image",
                            GROUP_IMG_MEASUREMENT,
                            COLTYPE_INTEGER,
                            d,
                        ),
                        (
                            OBJECT_NAME,
                            GROUP_OBJ_MEASUREMENT,
                            COLTYPE_FLOAT,
                            d,
                        ),
                    ]
                if post_run_test:
                    columns += [
                        (
                            EXPERIMENT,
                            STRING_IMG_MEASUREMENT,
                            COLTYPE_VARCHAR,
                            {
                                MCA_AVAILABLE_POST_RUN: True
                            },
                        )
                    ]
                return columns

            def get_object_relationships(self, pipeline):
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

            def get_categories(self, pipeline, object_name):
                return (
                    [M_CATEGORY, C_NUMBER]
                    if (
                        object_name == OBJECT_NAME
                        or ((object_name == ALTOBJECT_NAME) and self.in_module(alt_object))
                    )
                    else [M_CATEGORY, "Count", "Metadata"]
                    if object_name == "Image"
                    else []
                )

            def get_measurements(self, pipeline, object_name, category):
                if category == M_CATEGORY:
                    if object_name == OBJECT_NAME:
                        if self.in_module(long_measurement):
                            return [OBJ_FEATURE, LONG_OBJ_FEATURE]
                        else:
                            return [OBJ_FEATURE]
                    elif (object_name == ALTOBJECT_NAME) and self.in_module(alt_object):
                        return [OBJ_FEATURE]
                    else:
                        return (
                            [INT_IMG_FEATURE, FLOAT_IMG_FEATURE, STRING_IMG_FEATURE]
                            + [LONG_IMG_FEATURE]
                            if self.in_module(long_measurement)
                            else [WIERD_IMG_FEATURE]
                            if self.in_module(wierd_measurement)
                            else []
                        )
                elif (
                    category == C_NUMBER
                    and object_name in (OBJECT_NAME, ALTOBJECT_NAME,)
                ):
                    return FTR_OBJECT_NUMBER
                elif (
                    category == "Count"
                    and object_name == "Image"
                ):
                    result = [OBJECT_NAME]
                    if self.in_module(alt_object):
                        result += [ALTOBJECT_NAME]
                    return result
                elif (
                    category == "Metadata"
                    and object_name == "Image"
                ):
                    return ["Plate", "Well"]
                return []

        m = cellprofiler_core.measurement.Measurements()
        for i in range(image_set_count):
            if i > 0:
                m.next_image_set()
            m.add_image_measurement(GROUP_NUMBER, 1)
            m.add_image_measurement(GROUP_INDEX, i + 1)
            m.add_image_measurement(INT_IMG_MEASUREMENT, INT_VALUE)
            m.add_image_measurement(FLOAT_IMG_MEASUREMENT, FLOAT_VALUE)
            m.add_image_measurement(STRING_IMG_MEASUREMENT, STRING_VALUE)
            m.add_image_measurement(OBJECT_COUNT_MEASUREMENT, len(OBJ_VALUE))
            m.add_measurement(
                OBJECT_NAME,
                M_NUMBER_OBJECT_NUMBER,
                numpy.arange(len(OBJ_VALUE)) + 1,
            )
            m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, OBJ_VALUE.copy())
            if TestModule.in_measurements(alt_object):
                m.add_measurement(
                    ALTOBJECT_NAME,
                    M_NUMBER_OBJECT_NUMBER,
                    numpy.arange(100) + 1,
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
        r = numpy.random.RandomState()
        r.seed(image_set_count)
        if relationship_test_type == RTEST_SOME:
            n = 10
            i1, o1 = [
                x.flatten() for x in numpy.mgrid[1 : (image_set_count + 1), 1 : (n + 1)]
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

        image_set_list = cellprofiler_core.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(
            IMAGE_NAME, cellprofiler_core.image.Image(r.uniform(size=(512, 512)))
        )
        object_set = cellprofiler_core.object.ObjectSet()
        objects = cellprofiler_core.object.Objects()
        objects.segmented = numpy.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        object_set.add_objects(objects, OBJECT_NAME)
        if alt_object:
            objects = cellprofiler_core.object.Objects()
            objects.segmented = numpy.array([[0, 1, 2, 3], [0, 1, 2, 3]])
            object_set.add_objects(objects, ALTOBJECT_NAME)
        test_module = TestModule()
        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback_handler(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

        pipeline.add_listener(callback_handler)
        pipeline.add_module(test_module)
        module = cellprofiler.modules.exporttodatabase.ExportToDatabase()
        module.set_module_num(2)
        table_prefix = "T_%s" % str(uuid.uuid4()).replace("-", "")
        module.table_prefix.value = table_prefix
        module.want_table_prefix.value = True
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
        module.db_host.value = MYSQL_HOST
        module.db_user.value = MYSQL_USER
        module.db_password.value = MYSQL_PASSWORD
        module.db_name.value = MYSQL_DATABASE
        module.wants_relationship_table_setting.value = relationship_type is not None
        pipeline.add_module(module)
        pipeline.write_experiment_measurements(m)
        workspace = cellprofiler_core.workspace.Workspace(
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
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir

            def finally_fn():
                for filename in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, filename))

            return workspace, module, output_dir, finally_fn
        else:
            return workspace, module

    def load_database(self, output_dir, module, image_set_count=1):
        """Load a database written by DB_MYSQL_CSV"""
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        curdir = os.path.abspath(os.curdir)
        os.chdir(output_dir)
        #try:
        sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
        base_name = "SQL_1_%d" % image_set_count
        image_file = os.path.join(
            output_dir,
            base_name + "_" + "Image" + ".CSV",
        )
        if (
            module.separate_object_tables
            == cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
        ):
            object_file = os.path.join(
                output_dir, base_name + "_" + OBJECT_NAME + ".CSV"
            )
        else:
            object_file = "%s_%s.CSV" % (
                base_name,
                OBJECT,
            )
            object_file = os.path.join(output_dir, object_file)
        for filename in (sql_file, image_file, object_file):
            assert os.path.isfile(filename)
        fd = open(sql_file, "rt")
        sql_text = fd.read()
        fd.close()
        for statement in sql_text.split(";"):
            if len(statement.strip()) == 0:
                continue
            self.cursor.execute(statement)
        #finally:
        self.connection.commit()
        os.chdir(curdir)

    def tteesstt_no_relationships(self, module, cursor):
        for t in (
            cellprofiler.modules.exporttodatabase.T_RELATIONSHIPS,
            cellprofiler.modules.exporttodatabase.V_RELATIONSHIPS,
        ):
            statement = "select count('x') from %s" % module.get_table_name(t)
            cursor.execute(statement)
            assert cursor.fetchall()[0][0] == 0

    def tteesstt_relate(self, measurements, module, cursor):
        v_name = module.get_table_name(
            cellprofiler.modules.exporttodatabase.V_RELATIONSHIPS
        )
        statement = (
            "select count('x') from %s "
            "where %s=%d and %s='%s' and %s='%%s' and %s='%%s' "
            "and %s = %%d and %s = %%d and %s = %%d and %s = %%d"
        ) % (
            v_name,
            cellprofiler.modules.exporttodatabase.COL_MODULE_NUMBER,
            1,
            cellprofiler.modules.exporttodatabase.COL_RELATIONSHIP,
            RELATIONSHIP_NAME,
            cellprofiler.modules.exporttodatabase.COL_OBJECT_NAME1,
            cellprofiler.modules.exporttodatabase.COL_OBJECT_NAME2,
            cellprofiler.modules.exporttodatabase.COL_IMAGE_NUMBER1,
            cellprofiler.modules.exporttodatabase.COL_OBJECT_NUMBER1,
            cellprofiler.modules.exporttodatabase.COL_IMAGE_NUMBER2,
            cellprofiler.modules.exporttodatabase.COL_OBJECT_NUMBER2,
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

    def drop_tables(self, module, table_suffixes=None):
        """Drop all tables and views  that match the prefix"""
        connection = self.connection
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

    def drop_views(self, module, table_suffixes=None):
        self.drop_tables(module)

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            self.load_database(output_dir, module)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Make sure no relationships tables were created.
            #
            assert not module.wants_relationship_table
            for t in (
                cellprofiler.modules.exporttodatabase.T_RELATIONSHIPS,
                cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES,
            ):
                statement = "select count('x') from INFORMATION_SCHEMA.TABLES "
                statement += "where table_schema=%s and table_name=%s"
                self.cursor.execute(
                    statement, (module.db_name.value, module.get_table_name(t))
                )
                assert self.cursor.fetchall()[0][0] == 0
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db_filter_objs(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True, True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_SELECT
            module.objects_list.choices = [OBJECT_NAME, ALTOBJECT_NAME]
            module.objects_list.value = OBJECT_NAME
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
            )
            object_file = "%s_%s.CSV" % (
                base_name,
                OBJECT,
            )
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    OBJECT_NAME,
                    M_NUMBER_OBJECT_NUMBER,
                    module.table_prefix.value,
                )
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert row[3] == i + 1
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db_dont_filter_objs(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True, True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
            )
            object_file = "%s_%s.CSV" % (
                base_name,
                OBJECT,
            )
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == len(ALTOBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s, %s_%s, "
                "%s_%s, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (
                    OBJECT_NAME,
                    OBJ_MEASUREMENT,
                    OBJECT_NAME,
                    M_NUMBER_OBJECT_NUMBER,
                    ALTOBJECT_NAME,
                    OBJ_MEASUREMENT,
                    ALTOBJECT_NAME,
                    M_NUMBER_OBJECT_NUMBER,
                    module.table_prefix.value,
                )
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 6
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 4) == 0
                assert row[3] == i + 1
                assert round(abs(row[4] - ALTOBJ_VALUE[i]), 4) == 0
                assert row[5] == i + 1
            for i in range(len(OBJ_VALUE), len(ALTOBJ_VALUE)):
                row = self.cursor.fetchone()
                assert len(row) == 6
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[4] - ALTOBJ_VALUE[i]), 4) == 0
                assert row[5] == i + 1
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_mysql_direct(self):
        """Write directly to the mysql DB, not to a file"""
        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            if not self.test_mysql:
                pytest.skip("Skipping actual DB work, not at the Broad.")
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 7
            assert row[0] == 1
            assert row[1] == 1
            assert row[2] == 1
            assert round(abs(row[3] - INT_VALUE), 7) == 0
            assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
            assert row[5] == STRING_VALUE
            assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_00_write_direct_long_colname(self):
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = self.make_workspace(False, long_measurement=True)
        #try:
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
        module.wants_agg_mean.value = True
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
        module.separate_object_tables.value = (
            cellprofiler.modules.exporttodatabase.OT_COMBINE
        )
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
        self.cursor.execute(statement)
        row = self.cursor.fetchone()
        assert len(row) == 7
        assert row[0] == 1
        assert round(abs(row[1] - INT_VALUE), 7) == 0
        assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
        assert row[3] == STRING_VALUE
        assert row[4] == len(OBJ_VALUE)
        assert row[5] == 100
        assert round(abs(row[6] - numpy.mean(OBJ_VALUE)), 4) == 0
        with pytest.raises(StopIteration):
            self.cursor.__next__()
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
        self.cursor.execute(statement)
        for i, value in enumerate(OBJ_VALUE):
            row = self.cursor.fetchone()
            assert len(row) == 4
            assert row[0] == 1
            assert row[1] == i + 1
            assert round(abs(row[2] - value), 7) == 0
            assert round(abs(row[3] - value), 7) == 0
        with pytest.raises(StopIteration):
            self.cursor.__next__()
        #finally:
        self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    @pytest.mark.skip("MySQL/CSV mode removed")
    def test_01_write_csv_long_colname(self):
        """Write to MySQL, ensuring some columns have long names
    
        This is a regression test of IMG-786
        """
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True, long_measurement=True
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.post_run(workspace)
            self.load_database(output_dir, module)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 7
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 4) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            assert round(abs(row[6] - numpy.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    @pytest.mark.skip("MySQL/CSV mode removed")
    def test_01_write_nulls(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.NaN,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = numpy.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
            )
            object_file = "%s_%s.CSV" % (
                base_name,
                OBJECT,
            )
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - numpy.mean(om[~numpy.isnan(om)])), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    @pytest.mark.skip("MySQL/CSV mode removed")
    def test_02_write_inf(self):
        """regression test of img-1149"""
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        #
        # Insert inf into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.inf,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = numpy.inf
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
            )
            object_file = "%s_%s.CSV" % (
                base_name,
                OBJECT,
            )
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            mask = ~(numpy.isnan(om) | numpy.isinf(om))
            assert round(abs(row[5] - numpy.mean(om[mask])), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_mysql_direct_null(self):
        """Write directly to the mysql DB, not to a file and write nulls"""
        workspace, module = self.make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.NaN,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = numpy.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - numpy.mean(om[numpy.isfinite(om)])), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None or i != 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_wierd_colname(self):
        """Write to MySQL, even if illegal characters are in the column name"""
        workspace, module = self.make_workspace(False, wierd_measurement=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_50_char_colname(self):
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = self.make_workspace(False, long_measurement=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object", "Per_Experiment"))

    def test_write_direct_backslash(self):
        """Regression test for IMG-898
    
        Make sure CP can write string data containing a backslash character
        to the database in direct-mode.
        """
        backslash_string = "\\Why worry?"
        workspace, module = self.make_workspace(False)
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_NONE
        m = workspace.measurements
        assert isinstance(m,cellprofiler_core.measurement.Measurements)
        m.add_image_measurement(STRING_IMG_MEASUREMENT, backslash_string)
        try:
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            statement = "select Image_%s from %sPer_Image" % (
                STRING_IMG_MEASUREMENT,
                module.table_prefix.value,
            )
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 1
            assert row[0] == backslash_string
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Experiment"))

    def test_mysql_as_data_tool(self):
        """Write directly to the mysql DB, not to a file"""
        workspace, module = self.make_workspace(False, image_set_count=2)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_DATA
            )
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
            
            self.cursor.execute(statement)
            for i in range(2):
                row = self.cursor.fetchone()
                assert len(row) == 7
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def get_interaction_handler(self, ran_interaction_handler):
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

    def test_write_sqlite_direct(self):
        """Write directly to a SQLite database"""
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = self.make_workspace(True)
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = self.get_interaction_handler(
                    ran_interaction_handler
                )
            cursor = None
            connection = None
            try:
                assert isinstance(
                    module, cellprofiler.modules.exporttodatabase.ExportToDatabase
                )
                module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = (
                    cellprofiler.modules.exporttodatabase.O_ALL
                )
                module.directory.dir_choice = (
                    cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
                )
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = (
                    cellprofiler.modules.exporttodatabase.OT_COMBINE
                )
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.run(workspace)
                assert with_interaction_handler == ran_interaction_handler[0]
                cursor, connection = self.get_sqlite_cursor(module)
                self.check_experiment_table(cursor, module, workspace.measurements)
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
                self.cursor.execute(statement)
                row = self.cursor.fetchone()
                assert len(row) == 5
                assert row[0] == 1
                assert round(abs(row[1] - INT_VALUE), 7) == 0
                assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
                assert row[3] == STRING_VALUE
                assert row[4] == len(OBJ_VALUE)
                with pytest.raises(StopIteration):
                    self.cursor.__next__()
                statement = (
                    "select ImageNumber, ObjectNumber, %s_%s "
                    "from %sPer_Object order by ImageNumber, ObjectNumber"
                    % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
                )
                self.cursor.execute(statement)
                for i, value in enumerate(OBJ_VALUE):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
                with pytest.raises(StopIteration):
                    self.cursor.__next__()
            finally:
                if self.cursor is not None:
                    self.cursor.close()
                if connection is not None:
                    connection.close()
                if hasattr(module, "cursor") and module.cursor is not None:
                    module.cursor.close()
                if hasattr(module, "connection") and module.connection is not None:
                    module.connection.close()
                finally_fn()

    def test_write_sqlite_backslash(self):
        """Regression test of IMG-898 sqlite with backslash in string"""
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        backslash_string = "\\Why doesn't he worry?"
        m = workspace.measurements
        m.add_image_measurement(STRING_IMG_MEASUREMENT, backslash_string)
        cursor = None
        connection = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_NONE
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
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

    def test_numpy_float32(self):
        """Regression test of img-915
    
        This error occurred when the sqlite3 driver was unable to convert
        a numpy.float32 to a float.
        """
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        fim = workspace.measurements.get_all_measurements(
            "Image", FLOAT_IMG_MEASUREMENT
        )
        for i in range(len(fim)):
            fim[i] = numpy.float32(fim[i])
        iim = workspace.measurements.get_all_measurements(
            "Image", INT_IMG_MEASUREMENT
        )
        for i in range(len(iim)):
            iim[i] = numpy.int32(iim[i])
        cursor = None
        connection = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
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

    def test_sqlite_data_tool(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True, image_set_count=2
        )
        cursor = None
        connection = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_DATA
            )
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run_as_data_tool(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
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

    def test_stable_column_mapper(self):
        """Make sure the column mapper always yields the same output"""
        mapping = cellprofiler.modules.exporttodatabase.ColumnNameMapping()
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

    def test_leave_start_intact(self):
        """The column mapper should leave stuff before the first _ alone"""
        mapping = cellprofiler.modules.exporttodatabase.ColumnNameMapping(25)
        k1 = "leaveme_EVEN_THOUGH_WE_LIKE_REMOVING_LOWER_CASE_VOWELS"
        k2 = "keepmee_EVEN_THOUGH_WE_LIKE_REMOVING_LOWER_CASE_VOWELS"
        mapping.add(k1)
        mapping.add(k2)
        mapping.do_mapping()
        assert mapping[k1].startswith("leaveme_")
        assert mapping[k2].startswith("keepmee_")

    def per_object_statement(self, module, object_name, fields):
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
                M_NUMBER_OBJECT_NUMBER,
                field_string,
                module.table_prefix.value,
                object_name,
                object_name,
                M_NUMBER_OBJECT_NUMBER,
            )
        )
        return statement

    def check_experiment_table(self, cursor, module, m):
        """Check the per_experiment table values against measurements"""
        statement = "select %s, %s, %s from %s" % (
            M_PIPELINE,
            M_VERSION,
            M_TIMESTAMP,
            module.get_table_name(EXPERIMENT),
        )
        cursor.execute(statement)
        row = cursor.fetchone()
        with pytest.raises(StopIteration):
            cursor.__next__()
        assert len(row) == 3
        for feature, value in zip(
            (
                M_PIPELINE,
                M_VERSION,
                M_TIMESTAMP,
            ),
            row,
        ):
            assert equals(
                value, m.get_experiment_measurement(feature)
            )

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db_2(self):
        """Multiple objects / write - per-object tables"""
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.post_run(workspace)
            self.load_database(output_dir, module)
            
            self.check_experiment_table(self.cursor, module, workspace.measurements)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db_filter_objs_2(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True, True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_SELECT
            module.objects_list.choices = [OBJECT_NAME, ALTOBJECT_NAME]
            module.objects_list.value = OBJECT_NAME
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_mysql_direct(self):
        """Write directly to the mysql DB, not to a file"""
        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            
            self.check_experiment_table(self.cursor, module, workspace.measurements)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 5
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_long_colname(self):
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = self.make_workspace(False, long_measurement=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, long_obj_column]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_nulls(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.NaN,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[0] = numpy.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL_SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(
                output_dir,
                base_name + "_" + "Image" + ".CSV",
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
                self.cursor.execute(statement)
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
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert round(abs(row[5] - numpy.mean(om[~numpy.isnan(om)])), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                if i == 0:
                    assert row[2] is None
                else:
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_01_mysql_direct_null(self):
        """Write directly to the mysql DB, not to a file and write nulls"""
        workspace, module = self.make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.NaN,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[:] = numpy.NaN
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_02_mysql_direct_inf(self):
        """regression test of img-1149: infinite values"""
        workspace, module = self.make_workspace(False)
        #
        # Insert a NaN into the float image measurement and one of the
        # object measurements
        #
        m = workspace.measurements
        m.add_measurement(
            "Image",
            FLOAT_IMG_MEASUREMENT,
            numpy.NaN,
            True,
            1,
        )
        om = m.get_measurement(OBJECT_NAME, OBJ_MEASUREMENT, 1)
        om[:] = numpy.inf
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, om, True, 1)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert row[2] is None
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_wierd_colname(self):
        """Write to MySQL, even if illegal characters are in the column name"""
        workspace, module = self.make_workspace(False, wierd_measurement=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, wierd_obj_column]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_write_direct_50_char_colname(self):
        """Write to MySQL, ensuring some columns have long names"""
        workspace, module = self.make_workspace(False, long_measurement=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
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
            
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            assert len(row) == 6
            assert row[0] == 1
            assert round(abs(row[1] - INT_VALUE), 7) == 0
            assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
            assert row[3] == STRING_VALUE
            assert row[4] == len(OBJ_VALUE)
            assert row[5] == 100
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT, long_obj_column]
            )
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 4
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
                assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_01_write_two_object_tables_direct(self):
        """Write two object tables using OT_PER_OBJECT"""
        workspace, module = self.make_workspace(False, alt_object=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # Read from one object table
            #
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read from the other table
            #
            statement = self.per_object_statement(
                module, ALTOBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(len(ALTOBJ_VALUE)):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - ALTOBJ_VALUE[i]), 4) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    @pytest.mark.skip("MySQL/CSV mode removed")
    def test_02_write_two_object_tables_csv(self):
        """Write two object tables using OT_PER_OBJECT"""
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True, alt_object=True
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            self.load_database(output_dir, module)
            #
            # Read from one object table
            #

            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read from the other table
            #
            statement = self.per_object_statement(
                module, ALTOBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(len(ALTOBJ_VALUE)):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == 1
                assert row[1] == i + 1
                assert round(abs(row[2] - ALTOBJ_VALUE[i]), 4) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_db_as_data_tool(self):
        """Multiple objects / write - per-object tables"""
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True, image_set_count=2
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.run_as_data_tool(workspace)
            self.load_database(output_dir, module, image_set_count=2)
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
            
            self.cursor.execute(statement)
            for j in range(2):
                row = self.cursor.fetchone()
                assert len(row) == 5
                assert row[0] == j + 1
                assert round(abs(row[1] - INT_VALUE), 7) == 0
                assert round(abs(row[2] - FLOAT_VALUE), 7) == 0
                assert row[3] == STRING_VALUE
                assert row[4] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            os.chdir(output_dir)
            finally_fn()
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_data_tool_and_get_measurement_columns(self):
        # Regression test of issue #444
        #
        # Old measurements might not conform to get_measurement_columns
        # if a new measurement has been added.
        #
        workspace, module = self.make_workspace(
            False, image_set_count=2, long_measurement=MISSING_FROM_MEASUREMENTS
        )
        try:
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            self.cursor.execute(statement)
            for i in range(2):
                row = self.cursor.fetchone()
                assert len(row) == 7
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_data_tool_and_get_measurement_columns(self):
        # Regression test of issue #444
        #
        # Old measurements might not conform to get_measurement_columns
        # if an old measurement has been removed
        #
        workspace, module = self.make_workspace(
            False, image_set_count=2, long_measurement=MISSING_FROM_MODULE
        )
        try:
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = True
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            self.cursor.execute(statement)
            for i in range(2):
                row = self.cursor.fetchone()
                assert len(row) == 8
                assert row[0] == i + 1
                assert row[1] == 1
                assert row[2] == i + 1
                assert round(abs(row[3] - INT_VALUE), 7) == 0
                assert round(abs(row[4] - FLOAT_VALUE), 7) == 0
                assert row[5] == STRING_VALUE
                assert row[6] == len(OBJ_VALUE)
                assert abs(row[7] - numpy.mean(OBJ_VALUE)) < 0.0001
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
            self.cursor.execute(statement)
            for j in range(2):
                for i, value in enumerate(OBJ_VALUE):
                    row = self.cursor.fetchone()
                    assert len(row) == 4
                    assert row[0] == j + 1
                    assert row[1] == i + 1
                    assert round(abs(row[2] - value), 7) == 0
                    assert round(abs(row[3] - value), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_write_sqlite_direct(self):
        """Write directly to a SQLite database"""
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        cursor = None
        connection = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
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
            statement = self.per_object_statement(
                module, OBJECT_NAME, [OBJ_MEASUREMENT]
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

    def execute_well_sql(self, output_dir, module):
        file_name = "SQL__Per_Well_SETUP.SQL"
        sql_file = os.path.join(output_dir, file_name)
        fd = open(sql_file, "rt")
        sql_text = fd.read()
        fd.close()
        print(sql_text)
        for statement in sql_text.split(";"):
            if len(statement.strip()) == 0:
                continue
            self.cursor.execute(statement)

    def select_well_agg(self, module, aggname, fields):
        field_string = ", ".join(["%s_%s" % (aggname, field) for field in fields])
        statement = (
            "select Image_Metadata_Plate, Image_Metadata_Well, %s "
            "from %sPer_Well_%s" % (field_string, module.table_prefix.value, aggname)
        )
        return statement

    def test_well_single_objtable(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(
            False, well_metadata=True, image_set_count=3
        )
        aggs = [("avg", numpy.mean), ("std", numpy.std)]
        if self.mysql_has_median:
            aggs.append(("median", numpy.median))
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.wants_agg_mean_well.value = True
            module.wants_agg_median_well.value = self.mysql_has_median
            module.wants_agg_std_dev_well.value = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            module.post_run(workspace)
            self.execute_well_sql(output_dir, module)
            meas = (
                ("Image", FLOAT_IMG_MEASUREMENT),
                ("Image", INT_IMG_MEASUREMENT),
                (OBJECT_NAME, OBJ_MEASUREMENT),
            )
            m = workspace.measurements
            image_numbers = m.get_image_numbers()

            for aggname, aggfn in aggs:
                fields = [
                    "%s_%s" % (object_name, feature) for object_name, feature in meas
                ]
                statement = self.select_well_agg(module, aggname, fields)
                self.cursor.execute(statement)
                rows = self.cursor.fetchall()
                assert len(rows) == 1
                row = rows[0]
                assert row[0] == PLATE
                assert row[1] == WELL
                for i, (object_name, feature) in enumerate(meas):
                    value = row[i + 2]
                    values = m[object_name, feature, image_numbers]
                    expected = aggfn(values)
                    if numpy.isnan(expected):
                        assert value is None
                    else:
                        assert round(abs(float(value) - expected), 7) == 0
        finally:
            self.drop_tables(
                module, ["Per_Image", "Per_Object"] + ["Per_Well_" + x for x, _ in aggs]
            )
            finally_fn()

    def test_well_two_objtables(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(
            False, well_metadata=True, alt_object=True, image_set_count=3
        )
        aggs = [("avg", numpy.mean), ("std", numpy.std)]
        if self.mysql_has_median:
            aggs.append(("median", numpy.median))
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.wants_agg_mean_well.value = True
            module.wants_agg_median_well.value = self.mysql_has_median
            module.wants_agg_std_dev_well.value = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1, 2, 3])
            module.run(workspace)
            module.post_run(workspace)
            self.execute_well_sql(output_dir, module)
            meas = (
                ("Image", FLOAT_IMG_MEASUREMENT),
                ("Image", INT_IMG_MEASUREMENT),
                (OBJECT_NAME, OBJ_MEASUREMENT),
                (ALTOBJECT_NAME, OBJ_MEASUREMENT),
            )
            m = workspace.measurements
            image_numbers = m.get_image_numbers()
            for aggname, aggfn in aggs:
                fields = [
                    "%s_%s" % (object_name, feature) for object_name, feature in meas
                ]
                statement = self.select_well_agg(module, aggname, fields)
                self.cursor.execute(statement)
                rows = self.cursor.fetchall()
                assert len(rows) == 1
                row = rows[0]
                assert row[0] == PLATE
                assert row[1] == WELL
                for i, (object_name, feature) in enumerate(meas):
                    value = row[i + 2]
                    values = m[object_name, feature, image_numbers]
                    expected = aggfn(values)
                    if numpy.isnan(expected):
                        assert value is None
                    else:
                        assert round(abs(float(value) - expected), 7) == 0
        finally:
            self.drop_tables(
                module,
                ["Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME]
                + ["Per_Well_" + x for x, _ in aggs],
            )
            finally_fn()

    def test_image_thumbnails(self):
        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_NONE
            module.max_column_size.value = 50
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            self.cursor.execute(stmt)
            result = self.cursor.fetchall()
            print(result[0][0])
            im = PIL.Image.open(io.BytesIO(base64.b64decode(result[0][0])))
            assert tuple(im.size) == (200, 200)

        finally:
            self.drop_tables(module, ["Per_Image"])

    def test_image_thumbnails_sqlite(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        cursor = None
        connection = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_NONE
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
            cursor, connection = self.get_sqlite_cursor(module)
            cursor.execute(stmt)
            result = cursor.fetchall()
            print(result[0][0])
            im = PIL.Image.open(io.BytesIO(base64.b64decode(result[0][0])))
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

    def test_post_group_single_object_table(self):
        """Write out measurements that are only available post-group"""
        count = 5
        workspace, module = self.make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            self.close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_single_object_table_agg(self):
        """Test single object table, post_group aggregation"""
        count = 5
        workspace, module = self.make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        module.wants_agg_mean.value = True
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] is None
                assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            self.close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
                assert round(abs(row[2] - numpy.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_separate_object_tables(self):
        """Write out measurements post_group to separate object tables"""
        count = 5
        workspace, module = self.make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data too
            #
            statement = self.per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            self.close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data
            #
            statement = self.per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_%s" % OBJECT_NAME))

    def test_post_group_separate_table_agg(self):
        """Test single object table, post_group aggregation"""
        count = 5
        workspace, module = self.make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        module.wants_agg_mean.value = True
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] is None
                assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data too
            #
            statement = self.per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            self.close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = (
                "select ImageNumber, Image_%s, Mean_%s_%s "
                "from %s order by ImageNumber"
            ) % (GROUP_IMG_MEASUREMENT, OBJECT_NAME, GROUP_OBJ_MEASUREMENT, image_table)
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 3
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
                assert round(abs(row[2] - numpy.mean(OBJ_VALUE)), 4) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data
            #
            statement = self.per_object_statement(
                module, OBJECT_NAME, [GROUP_OBJ_MEASUREMENT]
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_post_group_sqlite(self):
        for with_interaction_handler in (False, True):
            count = 5
            workspace, module, output_dir, finally_fn = self.make_workspace(
                True, image_set_count=count, group_measurement=True
            )
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = self.get_interaction_handler(
                    ran_interaction_handler
                )
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
            measurements = workspace.measurements
            assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            cursor, connection = self.get_sqlite_cursor(module)
            try:
                module.separate_object_tables.value = (
                    cellprofiler.modules.exporttodatabase.OT_COMBINE
                )
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
                    % (GROUP_IMG_MEASUREMENT, image_table,)
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

    def test_post_group_object_view(self):
        """Write out measurements post_group to single object view"""
        count = 5
        workspace, module = self.make_workspace(
            False, image_set_count=count, group_measurement=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        measurements = workspace.measurements
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        try:
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_VIEW
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, numpy.arange(count) + 1)
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
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data too
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert row[2] is None
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            self.close_connection()
            #
            # Run post_group and see that the values do show up
            #
            module.post_group(workspace, {})
            image_table = module.table_prefix.value + "Per_Image"
            statement = "select ImageNumber, Image_%s from %s" % (
                GROUP_IMG_MEASUREMENT,
                image_table,
            )
            self.cursor.execute(statement)
            for i in range(count):
                row = self.cursor.fetchone()
                assert len(row) == 2
                assert row[0] == i + 1
                assert row[1] == INT_VALUE
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Read the object data
            #
            object_table = module.table_prefix.value + "Per_Object"
            statement = (
                "select ImageNumber, ObjectNumber, %s_%s "
                "from %sPer_Object order by ImageNumber, ObjectNumber"
                % (OBJECT_NAME, GROUP_OBJ_MEASUREMENT, module.table_prefix.value)
            )
            self.cursor.execute(statement)
            for i in range(count):
                for j in range(len(OBJ_VALUE)):
                    row = self.cursor.fetchone()
                    assert len(row) == 3
                    assert row[0] == i + 1
                    assert row[1] == j + 1
                    assert round(abs(row[2] - OBJ_VALUE[j]), 7) == 0
            with pytest.raises(StopIteration):
                self.cursor.__next__()
            #
            # Finally, confirm that the Per_Object item is a view
            #
            statement = (
                "SELECT * FROM information_schema.views WHERE TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s'"
                % (module.db_name.value, object_table)
            )
            self.cursor.execute(statement)
            assert len(self.cursor.fetchall()) != 0
        finally:
            self.drop_tables(module, ["Per_Image"])
            self.drop_views(module, ["Per_Object"])

    def test_properties_file(self):
        def patched_get_measurement_columns(
            module, pipeline, old_get_measurement_columns=ogmc
        ):
            result = [
                (
                    "Image",
                    C_FILE_NAME + "_" + IMAGE_NAME,
                    COLTYPE_VARCHAR,
                ),
                (
                    "Image",
                    C_PATH_NAME + "_" + IMAGE_NAME,
                    COLTYPE_VARCHAR,
                ),
            ] + old_get_measurement_columns(module, pipeline)
            return result

        cellprofiler.modules.exporttodatabase.ExportToDatabase.get_measurement_columns = (
            patched_get_measurement_columns
        )

        workspace, module, output_dir, finally_fn = self.make_workspace(
            True, alt_object=True
        )
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        file_name = "%s_%s.properties" % (DB_NAME, module.get_table_prefix())
        path = os.path.join(output_dir, file_name)
        #
        # Do a monkey-patch of ExportToDatabase.get_measurement_columns
        #
        try:
            m = workspace.measurements
            for image_number in m.get_image_numbers():
                m.add_measurement(
                    "Image",
                    C_FILE_NAME + "_" + IMAGE_NAME,
                    os.path.join(path, "img%d.tif" % image_number),
                    image_set_number=image_number,
                )
                m.add_measurement(
                    "Image",
                    C_PATH_NAME + "_" + IMAGE_NAME,
                    os.path.join(path, "img%d.tif" % image_number),
                    image_set_number=image_number,
                )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.db_name.value = DB_NAME
            module.db_host.value = DB_HOST
            module.db_user.value = DB_USER
            module.db_password.value = DB_PASSWORD
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
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
                ("db_port", "3306"),  # The CSV file has nulls in lots of places
                ("db_host", DB_HOST),
                ("db_name", DB_NAME),
                ("db_user", DB_USER),
                ("db_passwd", DB_PASSWORD),
                ("image_table", "%sPer_Image" % module.get_table_prefix()),
                ("object_table", "%sPer_Object" % module.get_table_prefix()),
                ("image_id", "ImageNumber"),
                ("object_id", "ObjectNumber"),
                ("cell_x_loc", "%s_Location_Center_X" % OBJECT_NAME),
                ("cell_y_loc", "%s_Location_Center_Y" % OBJECT_NAME),
                (
                    "image_path_cols",
                    "%s_%s_%s"
                    % (
                        "Image",
                        C_PATH_NAME,
                        IMAGE_NAME,
                    ),
                ),
                (
                    "image_file_cols",
                    "%s_%s_%s"
                    % (
                        "Image",
                        C_FILE_NAME,
                        IMAGE_NAME,
                    ),
                ),
            ):
                assert k in dictionary
                assert dictionary[k] == v
        finally:
            cellprofiler.modules.exporttodatabase.ExportToDatabase.get_measurement_columns = (
                ogmc
            )
            os.chdir(output_dir)
            if os.path.exists(path):
                os.unlink(path)
            finally_fn()

    def test_experiment_table_combine(self):
        workspace, module = self.make_workspace(False, True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.location_object.value = OBJECT_NAME
            if not self.test_mysql:
                pytest.skip("Skipping actual DB work, not at the Broad.")
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
            self.cursor.execute(statement)
            experiment_id = int(self.cursor.fetchone()[0])
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
                self.cursor.execute(statement)
                dbvalue = self.cursor.fetchone()[0]
                with pytest.raises(StopIteration):
                    self.cursor.__next__()
                assert dbvalue == v
        finally:
            self.drop_tables(module, ("Per_Image", "Per_Object"))

    def test_experiment_table_separate(self):
        workspace, module = self.make_workspace(False, True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_PER_OBJECT
            )
            if not self.test_mysql:
                pytest.skip("Skipping actual DB work, not at the Broad.")
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
            self.cursor.execute(statement)
            experiment_id = int(self.cursor.fetchone()[0])
            with pytest.raises(StopIteration):
                self.cursor.__next__()
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
                self.cursor.execute(statement)
                dbvalue = self.cursor.fetchone()[0]
                with pytest.raises(StopIteration):
                    self.cursor.__next__()
                assert dbvalue == v
        finally:
            self.drop_tables(
                module, ("Per_Image", "Per_%s" % OBJECT_NAME, "Per_%s" % ALTOBJECT_NAME)
            )

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_no_mysql_relationships(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            self.load_database(output_dir, module)
            self.tteesstt_no_relationships(module, self.cursor)
        finally:
            self.drop_tables(module)
            os.chdir(output_dir)
            finally_fn()

    def test_write_no_mysql_direct_relationships(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module = self.make_workspace(
            False,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            self.tteesstt_no_relationships(module, self.cursor)

        finally:
            self.drop_tables(module)

    def test_write_sqlite_no_relationships(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module, output_dir, finally_fn = self.make_workspace(
            True,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
        )
        cursor = None
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
            self.tteesstt_no_relationships(module, cursor)
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

    @pytest.mark.skip("MySQL/CSV Mode Removed")
    def test_write_mysql_relationships(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        workspace, module, output_dir, finally_fn = self.make_workspace(
            True,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL_CSV
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.post_run(workspace)
            self.load_database(output_dir, module)
            self.tteesstt_relate(workspace.measurements, module, self.cursor)
        finally:
            self.drop_tables(module)
            os.chdir(output_dir)
            finally_fn()

    def test_write_mysql_direct_relationships(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module = self.make_workspace(
            False,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            self.tteesstt_relate(workspace.measurements, module, self.cursor)
        finally:
            self.drop_tables(module)

    def test_write_sqlite_relationships(self):
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = self.make_workspace(
                True,
                relationship_type=MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            ran_interaction_handler = [False]
            if with_interaction_handler:
                workspace.interaction_handler = self.get_interaction_handler(
                    ran_interaction_handler
                )
            try:
                assert isinstance(
                    module, cellprofiler.modules.exporttodatabase.ExportToDatabase
                )
                module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = (
                    cellprofiler.modules.exporttodatabase.O_ALL
                )
                module.directory.dir_choice = (
                    cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
                )
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = (
                    cellprofiler.modules.exporttodatabase.OT_COMBINE
                )
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.run(workspace)
                cursor, connection = self.get_sqlite_cursor(module)
                self.tteesstt_relate(workspace.measurements, module, cursor)
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

    def test_write_sqlite_duplicates(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module, output_dir, finally_fn = self.make_workspace(
            True,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_DUPLICATE,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
            self.tteesstt_relate(workspace.measurements, module, cursor)
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

    def test_add_relationship_id_mysql(self):
        #
        # Add a missing relationship ID
        #
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module = self.make_workspace(
            False,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            #
            # Get rid of the module dictionary entry and the table row
            #
            module.get_dictionary()[
                cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
            ] = {}
            self.cursor.execute(
                "delete from %s"
                % module.get_table_name(
                    cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
                )
            )
            self.close_connection()
            module.run(workspace)
            self.tteesstt_relate(workspace.measurements, module, self.cursor)
        finally:
            self.drop_tables(module)

    def test_get_relationship_id_mysql(self):
        #
        # Get a missing relationship ID (e.g., worker # 2 gets worker # 1's row)
        #
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module = self.make_workspace(
            False,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
            relationship_test_type=RTEST_SOME,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            #
            # Get rid of the module dictionary entry and the table row
            #
            module.get_dictionary()[
                cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
            ] = {}
            module.run(workspace)
            self.tteesstt_relate(workspace.measurements, module, self.cursor)
        finally:
            self.drop_tables(module)

    def test_add_relationship_id_sqlite(self):
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = self.make_workspace(
                True,
                relationship_type=MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            if with_interaction_handler:
                ran_interaction_handler = [False]
                workspace.interaction_handler = self.get_interaction_handler(
                    ran_interaction_handler
                )
            try:
                assert isinstance(
                    module, cellprofiler.modules.exporttodatabase.ExportToDatabase
                )
                module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = (
                    cellprofiler.modules.exporttodatabase.O_ALL
                )
                module.directory.dir_choice = (
                    cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
                )
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = (
                    cellprofiler.modules.exporttodatabase.OT_COMBINE
                )
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                with cellprofiler.modules.exporttodatabase.DBContext(module) as (
                    connection,
                    cursor,
                ):
                    cursor.execute(
                        "delete from %s"
                        % module.get_table_name(
                            cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
                        )
                    )
                module.get_dictionary()[
                    cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
                ] = {}
                module.run(workspace)
                with cellprofiler.modules.exporttodatabase.DBContext(module) as (
                    connection,
                    cursor,
                ):
                    self.tteesstt_relate(workspace.measurements, module, cursor)
            finally:
                finally_fn()

    def test_get_relationship_id_sqlite(self):
        for with_interaction_handler in (False, True):
            workspace, module, output_dir, finally_fn = self.make_workspace(
                True,
                relationship_type=MCA_AVAILABLE_EACH_CYCLE,
                relationship_test_type=RTEST_SOME,
            )
            if with_interaction_handler:
                ran_interaction_handler = [False]
                workspace.interaction_handler = self.get_interaction_handler(
                    ran_interaction_handler
                )
            cursor = None
            connection = None
            try:
                assert isinstance(
                    module, cellprofiler.modules.exporttodatabase.ExportToDatabase
                )
                module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
                module.wants_agg_mean.value = False
                module.wants_agg_median.value = False
                module.wants_agg_std_dev.value = False
                module.objects_choice.value = (
                    cellprofiler.modules.exporttodatabase.O_ALL
                )
                module.directory.dir_choice = (
                    cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
                )
                module.directory.custom_path = output_dir
                module.separate_object_tables.value = (
                    cellprofiler.modules.exporttodatabase.OT_COMBINE
                )
                module.prepare_run(workspace)
                module.prepare_group(workspace, {}, [1])
                module.get_dictionary()[
                    cellprofiler.modules.exporttodatabase.T_RELATIONSHIP_TYPES
                ] = {}
                module.run(workspace)
                cursor, connection = self.get_sqlite_cursor(module)
                self.tteesstt_relate(workspace.measurements, module, cursor)
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

    def test_write_mysql_direct_relationships_2(self):
        # Regression test of #1757
        #
        # No relationships in relationships table and ExportToDatabase
        # is configured to display its window
        #
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, no database configured.")

        workspace, module = self.make_workspace(
            False,
            relationship_type=MCA_AVAILABLE_EACH_CYCLE,
        )
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            module.show_window = True
            module.prepare_run(workspace)
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            self.tteesstt_relate(workspace.measurements, module, self.cursor)
        finally:
            self.drop_tables(module)

    def test_mysql_no_overwrite(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")

        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_NEVER
            )
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            assert module.prepare_run(workspace)
            assert not module.prepare_run(workspace)
        finally:
            self.drop_tables(module)

    def test_mysql_keep_schema(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        
        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_DATA
            )
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(
                "Image"
            )
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 0
            self.close_connection()
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 1
            assert module.prepare_run(workspace)
            #
            # The row should still be there after the second prepare_run
            #
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 1
        finally:
            self.drop_tables(module)

    def test_mysql_drop_schema(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        workspace, module = self.make_workspace(False)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_ALL
            )
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(
                "Image"
            )
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 0
            self.close_connection()
            module.prepare_group(workspace, {}, [1])
            module.run(workspace)
            #
            # There should be one row after "run"
            #
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 1
            self.close_connection()
            assert module.prepare_run(workspace)
            #
            # The row should not be there after the second prepare_run
            #
            self.cursor.execute(how_many)
            assert self.cursor.fetchall()[0][0] == 0
        finally:
            self.drop_tables(module)

    def test_sqlite_no_overwrite(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
        module.directory.dir_choice = cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.allow_overwrite.value = (
            cellprofiler.modules.exporttodatabase.OVERWRITE_NEVER
        )
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
        module.separate_object_tables.value = (
            cellprofiler.modules.exporttodatabase.OT_COMBINE
        )
        try:
            assert module.prepare_run(workspace)
            assert not module.prepare_run(workspace)
        finally:
            finally_fn()

    def test_sqlite_keep_schema(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
        module.allow_overwrite.value = (
            cellprofiler.modules.exporttodatabase.OVERWRITE_DATA
        )
        module.directory.dir_choice = cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
        module.separate_object_tables.value = (
            cellprofiler.modules.exporttodatabase.OT_COMBINE
        )
        cursor, connection = self.get_sqlite_cursor(module)
        try:
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(
                "Image"
            )
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

    def test_sqlite_drop_schema(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        assert isinstance(
            module, cellprofiler.modules.exporttodatabase.ExportToDatabase
        )
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
        module.allow_overwrite.value = (
            cellprofiler.modules.exporttodatabase.OVERWRITE_ALL
        )
        module.directory.dir_choice = cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_agg_mean.value = False
        module.wants_agg_median.value = False
        module.wants_agg_std_dev.value = False
        module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
        module.separate_object_tables.value = (
            cellprofiler.modules.exporttodatabase.OT_COMBINE
        )
        cursor, connection = self.get_sqlite_cursor(module)
        try:
            assert module.prepare_run(workspace)
            #
            # There should be no rows in the image table after prepare_run
            #
            how_many = "select count('x') from %s" % module.get_table_name(
                "Image"
            )
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
            cursor.close()
            connection.close()
            finally_fn()

    def test_dbcontext_mysql(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        module = cellprofiler.modules.exporttodatabase.ExportToDatabase()
        module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
        module.db_host.value = MYSQL_HOST
        module.db_user.value = MYSQL_USER
        module.db_password.value = MYSQL_PASSWORD
        module.db_name.value = MYSQL_DATABASE
        with cellprofiler.modules.exporttodatabase.DBContext(module) as (
            connection,
            cursor,
        ):
            cursor.execute("select 1")
            result = cursor.fetchall()
            assert len(result) == 1
            assert result[0][0] == 1

    def test_dbcontext_sqlite(self):
        output_dir = tempfile.mkdtemp()
        try:
            module = cellprofiler.modules.exporttodatabase.ExportToDatabase()
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_SQLITE
            module.directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.directory.custom_path = output_dir
            with cellprofiler.modules.exporttodatabase.DBContext(module) as (
                connection,
                cursor,
            ):
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

    def test_post_run_experiment_measurement_mysql(self):
        if not self.test_mysql:
            pytest.skip("Skipping actual DB work, not at the Broad.")
        workspace, module = self.make_workspace(False, post_run_test=True)
        try:
            assert isinstance(
                module, cellprofiler.modules.exporttodatabase.ExportToDatabase
            )
            module.db_type.value = cellprofiler.modules.exporttodatabase.DB_MYSQL
            module.allow_overwrite.value = (
                cellprofiler.modules.exporttodatabase.OVERWRITE_ALL
            )
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = cellprofiler.modules.exporttodatabase.O_ALL
            module.separate_object_tables.value = (
                cellprofiler.modules.exporttodatabase.OT_COMBINE
            )
            workspace.measurements[
                EXPERIMENT, STRING_IMG_MEASUREMENT
            ] = STRING_VALUE
            assert module.prepare_run(workspace)
            self.cursor.execute(
                "select %s from %s"
                % (
                    STRING_IMG_MEASUREMENT,
                    module.get_table_name(EXPERIMENT),
                )
            )
            result = self.cursor.fetchall()[0][0]
            assert result is None
            self.close_connection()
            module.post_run(workspace)
            self.cursor.execute(
                "select %s from %s"
                % (
                    STRING_IMG_MEASUREMENT,
                    module.get_table_name(EXPERIMENT),
                )
            )
            assert self.cursor.fetchall()[0][0] == STRING_VALUE
        finally:
            self.drop_tables(module)
