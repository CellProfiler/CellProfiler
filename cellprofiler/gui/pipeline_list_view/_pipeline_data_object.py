import wx

from ..constants.pipeline_list_view import PIPELINE_DATA_FORMAT


class PipelineDataObject(wx.CustomDataObject):
    def __init__(self):
        super(PipelineDataObject, self).__init__(wx.DataFormat(PIPELINE_DATA_FORMAT))
