import wx

from cellprofiler.icons import get_builtin_image

COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White"]
COLOR_VALS = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
]
CPLDM_ALPHA = "alpha"
CPLDM_LINES = "lines"
CPLDM_NONE = "none"
CPLDM_OUTLINES = "outlines"
CPLD_ALPHA_COLORMAP = "alpha_colormap"
CPLD_ALPHA_VALUE = "alpha_value"
CPLD_LABELS = "labels"
CPLD_LINE_WIDTH = "line_width"
CPLD_MODE = "mode"
CPLD_NAME = "name"
CPLD_OUTLINE_COLOR = "outline_color"
CPLD_SHOW = "show"
EVT_NAV_MODE_CHANGE = wx.PyEventBinder(wx.NewEventType())
MATPLOTLIB_FILETYPES = ["png", "pdf"]
MATPLOTLIB_UNSUPPORTED_FILETYPES = []
MENU_CLOSE_ALL = wx.NewId()
MENU_CLOSE_WINDOW = wx.NewId()
MENU_FILE_SAVE = wx.NewId()
MENU_FILE_SAVE_TABLE = wx.NewId()
MENU_INTERPOLATION_NEAREST = wx.NewId()
MENU_INTERPOLATION_BILINEAR = wx.NewId()
MENU_INTERPOLATION_BICUBIC = wx.NewId()
MENU_LABELS_ALPHA = {}
MENU_LABELS_LINES = {}
MENU_LABELS_OFF = {}
MENU_LABELS_OUTLINE = {}
MENU_LABELS_OVERLAY = {}
MENU_RGB_CHANNELS = {}
MENU_SAVE_SUBPLOT = {}
MENU_TOOLS_MEASURE_LENGTH = wx.NewId()
MODE_MEASURE_LENGTH = 2
MODE_NONE = 0
NAV_MODE_NONE = ""
NAV_MODE_PAN = "pan/zoom"
NAV_MODE_ZOOM = "zoom rect"
NAV_MODE_MEASURE = "measure"
try:
    BMP_MEASURE = wx.Bitmap(get_builtin_image("IMG_MEASURE"))
except wx.PyNoAppError:
    BMP_MEASURE = None
WINDOW_IDS = []
