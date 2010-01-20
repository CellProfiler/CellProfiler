'''errordialog - dialog box for reporting error.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import httplib
from StringIO import StringIO
import urllib
import wx
import traceback

ED_STOP = "Stop"
ED_CONTINUE = "Continue"

ERROR_HOST = 'imageweb'
ERROR_URL = '/batchprofiler/cgi-bin/development/CellProfiler_2.0/reporterror.py'

def display_error_dialog(frame, exc, pipeline, message=None, tb = None):
    '''Display an error dialog, returning an indication of whether to continue
    
    frame - parent frame for application
    exc - exception that caused the error
    pipeline - currently executing pipeline
    message - message to display
    
    Returns either ED_STOP or ED_CONTINUE indicating how to handle.
    '''
    if message is None:
        message = exc.message
    
    if tb is None:
        traceback_text = traceback.format_exc()
    else:
        traceback_text = reduce(lambda x,y: x+y, traceback.format_tb(tb))
    dialog = wx.Dialog(frame, title="Pipeline error")
    sizer = wx.BoxSizer(wx.VERTICAL)
    dialog.SetSizer(sizer)
    question_control = wx.StaticText(dialog,-1, 
                                     "Encountered error while processing. "
                                     "Do you want to stop processing?")
    question_control.Font = wx.Font(int(dialog.Font.GetPointSize()*5/4),
                                    dialog.Font.GetFamily(),
                                    dialog.Font.GetStyle(),
                                    wx.FONTWEIGHT_BOLD)
    sizer.Add(question_control,0,
              wx.EXPAND | wx.ALL, 5)
    error_control = wx.StaticBox(dialog, -1, "Error:")
    error_box = wx.StaticBoxSizer(error_control, wx.HORIZONTAL)
    message_control = wx.StaticText(dialog, -1, message)
    error_box.Add(message_control, 1, wx.EXPAND | wx.RIGHT, 5)
    sizer.Add(error_box, 1, wx.EXPAND | wx.ALL, 5)
    aux_button_box = wx.BoxSizer(wx.VERTICAL)
    error_box.Add(aux_button_box, 0, wx.EXPAND)
    
    #####################################################
    #
    # Handle show details button
    #
    #####################################################
    
    details_button = wx.Button(dialog, -1, "Details...")
    details_button.SetToolTipString("Show error details")
    aux_button_box.Add(details_button,0,
                       wx.EXPAND | wx.BOTTOM,
                       5)
    details_on = [False]
    def on_details(event):
        if not details_on[0]:
            message_control.Label = "%s\n%s" % (message, traceback_text)
            message_control.Refresh()
            details_button.Label = "Hide..."
            details_button.Refresh()
            dialog.Fit()
            details_on[0] = True
        else:
            message_control.Label = message
            message_control.Refresh()
            details_button.Label = "Details..."
            details_button.Refresh()
            dialog.Fit()
            details_on[0] = False
    dialog.Bind(wx.EVT_BUTTON, on_details, details_button)

    #######################################################
    #
    # Handle copy button
    #
    #######################################################
    
    copy_button = wx.Button(dialog, -1, "Copy...")
    copy_button.SetToolTipString("Copy error to clipboard")
    aux_button_box.Add(copy_button, 0,
                       wx.EXPAND | wx.BOTTOM, 5)
    def on_copy(event):
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.Clear()
                wx.TheClipboard.SetData(wx.TextDataObject(traceback_text))
                wx.TheClipboard.Flush()
            finally:
                wx.TheClipboard.Close()
    dialog.Bind(wx.EVT_BUTTON, on_copy, copy_button)

    ############################################################
    #
    # Handle report button
    #
    ############################################################
    
    report_button = wx.Button(dialog, -1, "Report...")
    report_button.SetToolTipString("Upload error report to imaging group at the Broad Institute")
    aux_button_box.Add(report_button, 0, wx.EXPAND | wx.Bottom, 5)
    def handle_report(event):
        on_report(event, dialog, traceback_text, pipeline)
    dialog.Bind(wx.EVT_BUTTON, handle_report, report_button)

    button_sizer = dialog.CreateStdDialogButtonSizer(wx.YES | wx.NO)
    sizer.Add(button_sizer,0,wx.EXPAND)
    #
    # Handle the "No" button being pressed
    #
    no_button = button_sizer.GetNegativeButton()
    def on_no(event):
        dialog.SetReturnCode(wx.NO)
        dialog.Close()
        event.Skip()
    dialog.Bind(wx.EVT_BUTTON, on_no, no_button)
    
    dialog.Fit()
    result = dialog.ShowModal()
    if result == wx.ID_YES:
        return ED_STOP
    return ED_CONTINUE

def on_report(event, dialog, traceback_text, pipeline):
    '''Report an error to us'''
    from cellprofiler.utilities.get_revision import get_revision
    params = { "traceback":traceback_text,
               "revision":str(get_revision())
               }
    try:
        fd = StringIO()
        pipeline.savetxt(fd)
        fd.seek(0)
        pipeline_text = fd.read()
        params["pipeline"] = pipeline_text
    except:
        pass
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain"}
    conn = httplib.HTTPConnection(ERROR_HOST)
    conn.request("POST", ERROR_URL, 
                 urllib.urlencode(params), headers)
    response = conn.getresponse()
    if response.status == 200:
        wx.MessageBox("Error successfully uploaded. Thank you for reporting.",
                      parent = dialog)
    else:
        wx.MessageBox("Failed to upload. Server reported %d, %s" %
                      (response.status, response.reason))
    
if __name__ == "__main__":
    import cellprofiler.pipeline
    import cellprofiler.modules.loadimages
    try:
        float("my boat")
    except Exception, e:
        app = wx.PySimpleApp()
        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.loadimages.LoadImages()
        module.module_num = 1
        pipeline.add_module(module)
        display_error_dialog(None, e, pipeline)
