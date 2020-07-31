import wx

from ._pipeline_data_object import PipelineDataObject


class PipelineDropTarget(wx.DropTarget):
    def __init__(self, window):
        super(PipelineDropTarget, self).__init__()
        self.window = window
        self.data_object = wx.DataObjectComposite()
        self.pipeline_data_object = PipelineDataObject()
        self.file_data_object = wx.FileDataObject()
        self.data_object.Add(self.pipeline_data_object)
        self.data_object.Add(self.file_data_object)
        self.SetDataObject(self.data_object)

    def OnDragOver(self, x, y, data):
        if not self.window.provide_drag_feedback(x, y, data):
            return wx.DragNone
        if wx.GetKeyState(wx.WXK_CONTROL) == 0:
            return wx.DragMove
        return wx.DragCopy

    def OnDrop(self, x, y):
        return self.window.on_drop(x, y)

    def OnData(self, x, y, action):
        if self.GetData():
            if (
                self.data_object.GetReceivedFormat().GetType()
                == self.pipeline_data_object.GetFormat().GetType()
            ):
                pipeline_data = self.pipeline_data_object.GetData().tobytes().decode()
                if pipeline_data is not None:
                    self.window.on_data(x, y, action, pipeline_data)
            elif self.data_object.GetReceivedFormat().GetType() == wx.DF_FILENAME:
                self.window.on_filelist_data(
                    x, y, action, self.file_data_object.GetFilenames()
                )
        return action
