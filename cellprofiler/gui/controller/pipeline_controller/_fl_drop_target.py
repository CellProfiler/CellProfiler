import wx


# TODO: wx 3.0 seems to have broken the composite drop target functionality
#       so I am reverting to only allowing files, hence the commented-out code
class FLDropTarget(wx.FileDropTarget):
    """A generic drop target (for the path list)"""

    def __init__(self, file_callback_fn, text_callback_fn):
        super(self.__class__, self).__init__()
        self.file_callback_fn = file_callback_fn
        self.text_callback_fn = text_callback_fn
        self.file_data_object = wx.FileDataObject()
        self.text_data_object = wx.TextDataObject()
        self.composite_data_object = wx.DataObjectComposite()
        self.composite_data_object.Add(self.file_data_object, True)
        self.composite_data_object.Add(self.text_data_object)
        # self.SetDataObject(self.composite_data_object)

    def OnDropFiles(self, x, y, filenames):
        self.file_callback_fn(x, y, filenames)
        return True

    def OnDropText(self, x, y, text):
        self.text_callback_fn(x, y, text)
        return True

    def OnEnter(self, x, y, d):
        return wx.DragCopy

    def OnDragOver(self, x, y, d):
        return wx.DragCopy

    # def OnData(self, x, y, d):
    #    if self.GetData():
    #        df = self.composite_data_object.GetReceivedFormat().GetType()
    #        if df in (wx.DF_TEXT, wx.DF_UNICODETEXT):
    #            self.OnDropText(x, y, self.text_data_object.GetText())
    #        elif df == wx.DF_FILENAME:
    #            self.OnDropFiles(x, y,
    #                             self.file_data_object.GetFilenames())
    #    return wx.DragCopy
    #
    # @staticmethod
    # def OnDrop(x, y):
    #    return True
