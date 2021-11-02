import wx
from cellprofiler_core import pipeline
from cellprofiler_core.pipeline.io._v6 import dump
from cellprofiler_core.pipeline import PipelineLoadCancelledException


class Pipeline(pipeline.Pipeline):
    def create_progress_dialog(self, message, pipeline, title):
        return wx.ProgressDialog(
            title, message, style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
        )

    def respond_to_version_mismatch_error(self, message):
        if wx.GetApp():
            dialog = wx.MessageDialog(
                parent=None,
                message=message + " Continue?",
                caption="Pipeline version mismatch",
                style=wx.OK | wx.CANCEL | wx.ICON_QUESTION,
            )

            if dialog.ShowModal() != wx.ID_OK:
                dialog.Destroy()

                raise PipelineLoadCancelledException(message)

            dialog.Destroy()
        else:
            super(Pipeline, self).respond_to_version_mismatch_error(message)

    def save(self, fd_or_filename, save_image_plane_details=True):
        with open(fd_or_filename, "wt") as fd:
            if fd_or_filename.endswith(".json"):
                dump(self, fd, save_image_plane_details=False)
            else:
                super(Pipeline, self).dump(fd, save_image_plane_details)

