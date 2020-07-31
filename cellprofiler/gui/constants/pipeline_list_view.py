import wx

from cellprofiler.icons import get_builtin_image

IMG_OK = get_builtin_image("check")
IMG_ERROR = get_builtin_image("remove-sign")
IMG_EYE = get_builtin_image("eye-open")
IMG_CLOSED_EYE = get_builtin_image("eye-close")
IMG_STEP = get_builtin_image("IMG_ANALYZE_16")
IMG_STEPPED = get_builtin_image("IMG_ANALYZED")
IMG_PAUSE = get_builtin_image("IMG_PAUSE")
IMG_GO = get_builtin_image("IMG_GO_DIM")
IMG_DISABLED = get_builtin_image("unchecked")
IMG_UNAVAILABLE = get_builtin_image("IMG_UNAVAILABLE")
IMG_SLIDER = get_builtin_image("IMG_SLIDER")
IMG_SLIDER_ACTIVE = get_builtin_image("IMG_SLIDER_ACTIVE")
IMG_DOWNARROW = get_builtin_image("downarrow")
BMP_WARNING = wx.ArtProvider.GetBitmap(wx.ART_WARNING, size=(16, 16))
NO_PIPELINE_LOADED = "No pipeline loaded"
PADDING = 1
STEP_COLUMN = 0
PAUSE_COLUMN = 1
EYE_COLUMN = 2
ERROR_COLUMN = 3
MODULE_NAME_COLUMN = 4
NUM_COLUMNS = 5
INPUT_ERROR_COLUMN = 0
INPUT_MODULE_NAME_COLUMN = 1
NUM_INPUT_COLUMNS = 2
ERROR = "error"
WARNING = "warning"
OK = "ok"
DISABLED = "disabled"
EYE = "eye-open"
CLOSED_EYE = "eye-close"
PAUSE = "pause"
GO = "go"
NOTDEBUG = "notdebug"
UNAVAILABLE = "unavailable"
PLV_HITTEST_SLIDER = 4096
PLV_STATE_WARNING = 1024
PLV_STATE_ERROR = 2048
PLV_STATE_ERROR_MASK = PLV_STATE_ERROR + PLV_STATE_WARNING
PLV_STATE_UNAVAILABLE = 4096
PLV_STATE_PROCEED = 8192
EVT_PLV_SLIDER_MOTION = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_STEP_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_PAUSE_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_EYE_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_ERROR_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_VALID_STEP_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
image_index_dictionary = {}
CHECK_TIMEOUT_SEC = 2
CHECK_FAIL_SEC = 20
PIPELINE_DATA_FORMAT = "cellprofiler_core.pipeline"
