import wx
from cellprofiler_core.preferences import RECENT_FILE_COUNT

RECENT_PIPELINE_FILE_MENU_ID = [wx.NewId() for i in range(RECENT_FILE_COUNT)]
RECENT_WORKSPACE_FILE_MENU_ID = [wx.NewId() for i in range(RECENT_FILE_COUNT)]
ED_STOP = "Stop"
ED_CONTINUE = "Continue"
ED_SKIP = "Skip"
