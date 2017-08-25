import cellprofiler.pipeline
import wx

FMT_NATIVE = "Native"


class Pipeline(cellprofiler.pipeline.Pipeline):
    def create_progress_dialog(self, message, pipeline, title):
        return wx.ProgressDialog(title, message, style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

    def respond_to_version_mismatch_error(self, message):
        if wx.GetApp():
            dialog = wx.MessageDialog(
                parent=None,
                message=message + " Continue?",
                caption='Pipeline version mismatch',
                style=wx.OK | wx.CANCEL | wx.ICON_QUESTION
            )

            if dialog.ShowModal() != wx.ID_OK:
                dialog.Destroy()

                raise cellprofiler.pipeline.PipelineLoadCancelledException(message)

            dialog.Destroy()
        else:
            super(Pipeline, self).respond_to_version_mismatch_error(message)
    
    def save(self, fd_or_filename, format=FMT_NATIVE, save_image_plane_details=True):
        super(Pipeline, self).save(fd_or_filename, format, save_image_plane_details)
