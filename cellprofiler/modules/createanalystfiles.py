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

"""The column name for the image number column"""
C_IMAGE_NUMBER = "ImageNumber"

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
W_TYPE_ALL = [
    "Image",
    OBJECT,
    W_INDEX,
]
W_INDEX_ALL = [C_IMAGE_NUMBER, GROUP_INDEX]

SETTING_OFFSET_PROPERTIES_IMAGE_URL_PREPEND_V26 = 21