import wx.adv
from cellprofiler import __version__


class AboutDialogInfo(wx.adv.AboutDialogInfo):
    def __init__(self):
        super(AboutDialogInfo, self).__init__()

        self.SetCopyright(
            "Copyright © 2003 - 2020 Broad Institute, Inc.\nAll rights reserved."
        )

        self.SetName("CellProfiler")

        self.SetVersion(__version__)
