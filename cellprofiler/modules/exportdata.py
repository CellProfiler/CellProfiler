"""
ExportData
==========

**ExportData** exports measurements into one or more files.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============


Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For details on the nomenclature used by CellProfiler for the exported
measurements, see *Help > General Help > How Measurements Are Named*.
^^^^^^^^
"""


import base64
import csv
import datetime
import functools
import hashlib
import io
import logging
import os
import re

import numpy

import cellprofiler_core.pipeline
import cellprofiler_core.utilities.legacy

from cellprofiler_core.constants.image import C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH
from cellprofiler_core.constants.measurement import (
    EXPERIMENT,
    IMAGE,
    OBJECT,
    AGG_MEAN,
    AGG_MEDIAN,
    AGG_STD_DEV,
    C_URL,
    C_PATH_NAME,
    C_FILE_NAME,
    C_PATH_NAME,
    C_META,
    NEIGHBORS,
    R_FIRST_IMAGE_NUMBER,
    R_SECOND_IMAGE_NUMBER,
    R_FIRST_OBJECT_NUMBER,
    R_SECOND_OBJECT_NUMBER,
    COLTYPE_BLOB,
    COLTYPE_FLOAT,
    COLTYPE_LONGBLOB,
    COLTYPE_MEDIUMBLOB,
    COLTYPE_VARCHAR,
    GROUP_INDEX,
    GROUP_NUMBER,
    MCA_AVAILABLE_POST_GROUP,
    MCA_AVAILABLE_POST_RUN,
    M_NUMBER_OBJECT_NUMBER
)
from cellprofiler_core.constants.module import (
    IO_FOLDER_CHOICE_HELP_TEXT,
    IO_WITH_METADATA_HELP_TEXT,
    USING_METADATA_HELP_REF,
    USING_METADATA_TAGS_REF,
)
from cellprofiler_core.constants.pipeline import EXIT_STATUS, M_MODIFICATION_TIMESTAMP
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import get_headless
from cellprofiler_core.preferences import get_allow_schema_write
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import HiddenCount
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import CustomChoice, Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.multichoice import (
    MeasurementMultiChoice,
    ObjectSubscriberMultiChoice,
    ImageNameSubscriberMultiChoice,
)
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.text import Directory, Integer, Text
from cellprofiler_core.utilities.core.modules.load_data import (
    is_file_name_feature,
    is_path_name_feature,
)
from cellprofiler_core.utilities.measurement import (
    find_metadata_tokens,
    get_agg_measurement_name,
    agg_ignore_feature
)

import cellprofiler
import cellprofiler.icons

from cellprofiler.gui.help.content import MEASUREMENT_NAMING_HELP
from cellprofiler.modules import _help
from cellprofiler.modules._help import IO_FOLDER_CHOICE_HELP_TEXT

LOGGER = logging.getLogger(__name__)

buffer = memoryview

try:
    import MySQLdb
    from MySQLdb.cursors import SSCursor
    import sqlite3

    HAS_MYSQL_DB = True
except Exception:
    LOGGER.warning("MySQL could not be loaded.", exc_info=True)
    HAS_MYSQL_DB = False

OBJECT_RELATIONSHIPS = "Object relationships"
RELATIONSHIPS = "Relationships"

D_MEASUREMENT_COLUMNS = "MeasurementColumns"

"""The column name for the image number column"""
C_IMAGE_NUMBER = "ImageNumber"

"""The column name for the object number column"""
C_OBJECT_NUMBER = "ObjectNumber"
D_IMAGE_SET_INDEX = "ImageSetIndex"

"""The thumbnail category"""
C_THUMBNAIL = "Thumbnail"

# """As far as I can tell, this is deprecated, but let's leave here but commented for now"""
# DIR_CUSTOM = "Custom folder"
# DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

##############################################
#
# Choices from ExportToSpreadsheet
#
##############################################
DELIMITER_TAB = "Tab"
DELIMITER_COMMA = 'Comma (",")'
DELIMITERS = (DELIMITER_COMMA, DELIMITER_TAB)

# SETTING_OG_OFFSET_V7 = 15
# SETTING_OG_OFFSET_V8 = 16
# SETTING_OG_OFFSET_V9 = 15
# SETTING_OG_OFFSET_V10 = 17
# SETTING_OG_OFFSET_V11 = 18
"""Offset of the first object group in the settings"""
SETTING_OG_OFFSET = 18

"""Offset of the object name setting within an object group"""
SETTING_OBJECT_NAME_IDX = 0

"""Offset of the previous file flag setting within an object group"""
SETTING_PREVIOUS_FILE_IDX = 1

"""Offset of the file name setting within an object group"""
SETTING_FILE_NAME_IDX = 2

SETTING_AUTOMATIC_FILE_NAME_IDX = 3

"""# of settings within an object group"""
SETTING_OBJECT_GROUP_CT = 4

"""The caption for the image set number"""
IMAGE_NUMBER = "ImageNumber"

"""The caption for the object # within an image set"""
OBJECT_NUMBER = "ObjectNumber"

"""The heading for the "Key" column in the experiment CSV"""
EH_KEY = "Key"

"""The heading for the "Value" column in the experiment CSV"""
EH_VALUE = "Value"

"""Options for GenePattern GCT file export"""
GP_NAME_FILENAME = "Image filename"
GP_NAME_METADATA = "Metadata"
GP_NAME_OPTIONS = [GP_NAME_METADATA, GP_NAME_FILENAME]

NANS_AS_NULLS = "Null"
NANS_AS_NANS = "NaN"

##############################################
#
# Choices from ExportToDatabase
#
##############################################

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

"""Put all objects in the database"""
O_ALL = "All"
"""Don't put any objects in the database"""
O_NONE = "None"
"""Select the objects you want from a list"""
O_SELECT = "Select..."


################################################
#
# Choices for overwrite
#
################################################

OVERWRITE_NEVER = "Never"
OVERWRITE_DATA = "Data only"
OVERWRITE_ALL = "Data and schema"

"""Offset of the image group count in the settings"""
SETTING_IMAGE_GROUP_COUNT = 28

"""Offset of the group specification group count in the settings"""
SETTING_GROUP_FIELD_GROUP_COUNT = 29

"""Offset of the filter specification group count in the settings"""
SETTING_FILTER_FIELD_GROUP_COUNT = 30

"""Offset of the workspace specification group count in the settings"""
SETTING_WORKSPACE_GROUP_COUNT = 31

# SETTING_WORKSPACE_GROUP_COUNT_PRE_V28 = 32

# SETTING_FIXED_SETTING_COUNT_V21 = 33

# SETTING_FIXED_SETTING_COUNT_V22 = 35

# SETTING_FIXED_SETTING_COUNT_V23 = 36

# SETTING_FIXED_SETTING_COUNT_V24 = 37

# SETTING_FIXED_SETTING_COUNT_V25 = 38

# SETTING_FIXED_SETTING_COUNT_V26 = 39

SETTING_FIXED_SETTING_COUNT = 38



##############################################
#
# Choices for object table format
#
##############################################

OT_PER_OBJECT = "One table per object type"
OT_COMBINE = "Single object table"
OT_VIEW = "Single object view"

"""Index of the object table format choice in the settings"""
OT_IDX = 17

"""Use this dictionary to keep track of rewording of above if it happens"""
OT_DICTIONARY = {
    "One table per object type": OT_PER_OBJECT,
    "Single object table": OT_COMBINE,
    "Single object view": OT_VIEW,
}

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