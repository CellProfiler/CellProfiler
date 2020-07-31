import wx

HELP_ON_FILE_LIST = """\
The *File List* panel displays the image files that are managed by the
**Images**, **Metadata**, **NamesAndTypes** and **Groups** modules.
You can drop files and directories into this window or use the
*Browse…* button to add files to the list. The context menu for the
window lets you display or remove files and lets you remove folders.

The buttons and checkbox along the bottom have the following
functions:

-  *Browse…*: Browse for files and folders to add.
-  *Clear*: Clear all entries from the File list
-  *Show files excluded by filters*: *(Only shown if filtered based on
   rules is selected)* Check this to see all files in the list. Uncheck
   it to see only the files that pass the rules criteria in the
   **Images** module.
-  *Expand tree*: Expand all of the folders in the tree
-  *Collapse tree*: Collapse the folders in the tree
"""
HELP_ON_MODULE_BUT_NONE_SELECTED = """\
The help button can be used to obtain help for the currently selected
module in the pipeline panel on the left side of the CellProfiler
interface.

You do not have any modules in the pipeline, yet. Add a
module to the pipeline using the “+” button or by using File > Load
Pipeline.\
"""
ID_FILE_NEW_WORKSPACE = wx.ID_NEW
ID_FILE_LOAD = wx.ID_OPEN
ID_FILE_LOAD_PIPELINE = wx.NewId()
ID_FILE_URL_LOAD_PIPELINE = wx.NewId()
ID_FILE_OPEN_IMAGE = wx.NewId()
ID_FILE_EXIT = wx.NewId()
ID_FILE_WIDGET_INSPECTOR = wx.NewId()
ID_FILE_SAVE_PIPELINE = wx.NewId()
ID_FILE_SAVE = wx.ID_SAVE
ID_FILE_SAVE_AS = wx.ID_SAVEAS
ID_FILE_REVERT_TO_SAVED = wx.NewId()
ID_FILE_CLEAR_PIPELINE = wx.NewId()
ID_FILE_EXPORT_IMAGE_SETS = wx.NewId()
ID_FILE_EXPORT_PIPELINE_NOTES = wx.NewId()
ID_FILE_IMPORT_FILE_LIST = wx.NewId()
ID_FILE_ANALYZE_IMAGES = wx.NewId()
ID_FILE_STOP_ANALYSIS = wx.NewId()
ID_FILE_PRINT = wx.NewId()
ID_FILE_PLATEVIEWER = wx.NewId()
ID_FILE_NEW_CP = wx.NewId()
ID_EDIT_SELECT_ALL = wx.NewId()
ID_EDIT_COPY = wx.NewId()
ID_EDIT_DUPLICATE = wx.NewId()
ID_EDIT_UNDO = wx.ID_UNDO
ID_EDIT_MOVE_UP = wx.NewId()
ID_EDIT_MOVE_DOWN = wx.NewId()
ID_EDIT_DELETE = wx.NewId()
ID_EDIT_EXPAND_ALL = wx.NewId()
ID_EDIT_COLLAPSE_ALL = wx.NewId()
ID_EDIT_BROWSE_FOR_FILES = wx.NewId()
ID_EDIT_BROWSE_FOR_FOLDER = wx.NewId()
ID_EDIT_CLEAR_FILE_LIST = wx.NewId()
ID_EDIT_REMOVE_FROM_FILE_LIST = wx.NewId()
ID_EDIT_SHOW_FILE_LIST_IMAGE = wx.NewId()
ID_EDIT_ENABLE_MODULE = wx.NewId()
ID_EDIT_GO_TO_MODULE = wx.NewId()
ID_OPTIONS_PREFERENCES = wx.ID_PREFERENCES
ID_CHECK_NEW_VERSION = wx.NewId()
ID_DEBUG_TOGGLE = wx.NewId()
ID_DEBUG_STEP = wx.NewId()
ID_DEBUG_NEXT_IMAGE_SET = wx.NewId()
ID_DEBUG_NEXT_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_IMAGE_SET = wx.NewId()
ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET = wx.NewId()
ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP = wx.NewId()
ID_DEBUG_RELOAD = wx.NewId()
ID_DEBUG_PDB = wx.NewId()
ID_DEBUG_RUN_FROM_THIS_MODULE = wx.NewId()
ID_DEBUG_STEP_FROM_THIS_MODULE = wx.NewId()
ID_DEBUG_HELP = wx.NewId()
ID_SAMPLE_INIT = wx.NewId()
ID_WINDOW = wx.NewId()
ID_WINDOW_CLOSE_ALL = wx.NewId()
ID_WINDOW_SHOW_ALL_WINDOWS = wx.NewId()
ID_WINDOW_HIDE_ALL_WINDOWS = wx.NewId()
ID_WINDOW_ALL = (
    ID_WINDOW_CLOSE_ALL,
    ID_WINDOW_SHOW_ALL_WINDOWS,
    ID_WINDOW_HIDE_ALL_WINDOWS,
)
window_ids = []
ID_HELP_MODULE = wx.NewId()
ID_HELP_SOURCE_CODE = wx.NewId()
ID_HELP_ABOUT = wx.ID_ABOUT
