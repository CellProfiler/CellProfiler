import numpy

IMAGE_NUMBER = "ImageNumber"
GROUP_NUMBER = "Group_Number"
GROUP_INDEX = "Group_Index"
GROUP_LENGTH = "Group_Length"
CURRENT = "Current"
NUMBER_OF_IMAGE_SETS = "NumberOfImageSets"
NUMBER_OF_MODULES = "NumberOfModules"
SET_BEING_ANALYZED = "SetBeingAnalyzed"
SAVE_OUTPUT_HOW_OFTEN = "SaveOutputHowOften"
TIME_STARTED = "TimeStarted"
STARTING_IMAGE_SET = "StartingImageSet"
STARTUP_DIRECTORY = "StartupDirectory"
DEFAULT_MODULE_DIRECTORY = "DefaultModuleDirectory"
DEFAULT_IMAGE_DIRECTORY = "DefaultImageDirectory"
DEFAULT_OUTPUT_DIRECTORY = "DefaultOutputDirectory"
IMAGE_TOOLS_FILENAMES = "ImageToolsFilenames"
IMAGE_TOOL_HELP = "ImageToolHelp"
PREFERENCES = "Preferences"
PIXEL_SIZE = "PixelSize"
SKIP_ERRORS = "SkipErrors"
INTENSITY_COLOR_MAP = "IntensityColorMap"
LABEL_COLOR_MAP = "LabelColorMap"
STRIP_PIPELINE = "StripPipeline"
DISPLAY_MODE_VALUE = "DisplayModeValue"
DISPLAY_WINDOWS = "DisplayWindows"
FONT_SIZE = "FontSize"
IMAGES = "Images"
MEASUREMENTS = "Measurements"
PIPELINE = "Pipeline"
SETTINGS = "Settings"
VARIABLE_VALUES = "VariableValues"
VARIABLE_INFO_TYPES = "VariableInfoTypes"
MODULE_NAMES = "ModuleNames"
NUMBERS_OF_VARIABLES = "NumbersOfVariables"
VARIABLE_REVISION_NUMBERS = "VariableRevisionNumbers"
MODULE_REVISION_NUMBERS = "ModuleRevisionNumbers"
MODULE_NOTES = "ModuleNotes"
CURRENT_MODULE_NUMBER = "CurrentModuleNumber"
SHOW_WINDOW = "ShowFrame"
BATCH_STATE = "BatchState"
EXIT_STATUS = "Exit_Status"
SETTINGS_DTYPE = numpy.dtype(
    [
        (VARIABLE_VALUES, "|O4"),
        (VARIABLE_INFO_TYPES, "|O4"),
        (MODULE_NAMES, "|O4"),
        (NUMBERS_OF_VARIABLES, "|O4"),
        (PIXEL_SIZE, "|O4"),
        (VARIABLE_REVISION_NUMBERS, "|O4"),
        (MODULE_REVISION_NUMBERS, "|O4"),
        (MODULE_NOTES, "|O4"),
        (SHOW_WINDOW, "|O4"),
        (BATCH_STATE, "|O4"),
    ]
)


def make_cell_struct_dtype(fields):
    """Makes the dtype of a struct composed of cells

    fields - the names of the fields in the struct
    """
    return numpy.dtype([(str(x), "|O4") for x in fields])


CURRENT_DTYPE = make_cell_struct_dtype(
    [
        NUMBER_OF_IMAGE_SETS,
        SET_BEING_ANALYZED,
        NUMBER_OF_MODULES,
        SAVE_OUTPUT_HOW_OFTEN,
        TIME_STARTED,
        STARTING_IMAGE_SET,
        STARTUP_DIRECTORY,
        DEFAULT_OUTPUT_DIRECTORY,
        DEFAULT_IMAGE_DIRECTORY,
        IMAGE_TOOLS_FILENAMES,
        IMAGE_TOOL_HELP,
    ]
)
PREFERENCES_DTYPE = make_cell_struct_dtype(
    [
        PIXEL_SIZE,
        DEFAULT_MODULE_DIRECTORY,
        DEFAULT_OUTPUT_DIRECTORY,
        DEFAULT_IMAGE_DIRECTORY,
        INTENSITY_COLOR_MAP,
        LABEL_COLOR_MAP,
        STRIP_PIPELINE,
        SKIP_ERRORS,
        DISPLAY_MODE_VALUE,
        FONT_SIZE,
        DISPLAY_WINDOWS,
    ]
)
NATIVE_VERSION = 5
IMAGE_PLANE_DESCRIPTOR_VERSION = 1
H_SVN_REVISION = "SVNRevision"
H_DATE_REVISION = "DateRevision"
H_GIT_HASH = "GitHash"
H_PLANE_COUNT = "PlaneCount"
H_URL = "URL"
H_SERIES = "Series"
H_INDEX = "Index"
H_CHANNEL = "Channel"
H_MODULE_COUNT = "ModuleCount"
H_HAS_IMAGE_PLANE_DETAILS = "HasImagePlaneDetails"
H_MESSAGE_FOR_USER = "MessageForUser"
COOKIE = "CellProfiler Pipeline: http://www.nucleus.org"
SAD_PROOFPOINT_COOKIE = r"CellProfiler Pipeline: https?://\S+.proofpoint.com.+http-3A__www.cellprofiler\.org"
HDF5_HEADER = (
    chr(137) + chr(72) + chr(68) + chr(70) + chr(13) + chr(10) + chr(26) + chr(10)
)
C_PIPELINE = "Pipeline"
C_CELLPROFILER = "CellProfiler"
F_PIPELINE = "Pipeline"
F_USER_PIPELINE = "UserPipeline"
M_PIPELINE = "_".join((C_PIPELINE, F_PIPELINE))
M_USER_PIPELINE = "_".join((C_PIPELINE, F_USER_PIPELINE))
F_VERSION = "Version"
M_VERSION = "_".join((C_CELLPROFILER, F_VERSION))
C_RUN = "Run"
C_MODIFICATION = "Modification"
F_TIMESTAMP = "Timestamp"
M_TIMESTAMP = "_".join((C_RUN, F_TIMESTAMP))
M_MODIFICATION_TIMESTAMP = "_".join((C_MODIFICATION, F_TIMESTAMP))
M_DEFAULT_INPUT_FOLDER = "Default_InputFolder"
M_DEFAULT_OUTPUT_FOLDER = "Default_OutputFolder"
RF_STATE_PREQUOTE = 0
RF_STATE_FIELD = 1
RF_STATE_BACKSLASH_ESCAPE = 2
RF_STATE_SEPARATOR = 3
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
