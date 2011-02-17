'''errordialog - dialog box for reporting error.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

from StringIO import StringIO
import urllib
import urllib2
import traceback
import sys
import platform

ED_STOP = "Stop"
ED_CONTINUE = "Continue"
ED_SKIP = "Skip"

ERROR_URL = 'http://www.cellprofiler.org/cgi-bin/reporterror.cgi'

def display_error_dialog(frame, exc, pipeline, message=None, tb = None):
    '''Display an error dialog, returning an indication of whether to continue
    
    frame - parent frame for application
    exc - exception that caused the error
    pipeline - currently executing pipeline
    message - message to display
    
    Returns either ED_STOP or ED_CONTINUE indicating how to handle.
    '''
    import wx
    if message is None:
        message = str(exc)
    
    if tb is None:
        traceback_text = traceback.format_exc()
    else:
        traceback_text = "".join(traceback.format_exception(type(exc), exc.message, tb))
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
    
    copy_button = wx.Button(dialog, -1, "Copy to clipboard")
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
    
    def handle_report(event):
        on_report(event, dialog, traceback_text, pipeline)

    report_button = wx.Button(dialog, wx.ID_APPLY, "Send report...")
    report_button.SetToolTipString("Upload error report to the CellProfiler Project")
    dialog.Bind(wx.EVT_BUTTON, handle_report, report_button)

    ############################################################
    #
    # Handle pdb button
    #
    ############################################################
    
    if (tb is not None) and (not hasattr(sys, 'frozen')):
        pdb_button = wx.Button(dialog, -1, "Debug in pdb...")
        pdb_button.SetToolTipString("Debug in python's pdb on the console")
        aux_button_box.Add(pdb_button, 0, wx.EXPAND | wx.BOTTOM, 5)
        def handle_pdb(event):
            import pdb
            pdb.post_mortem(tb)
        dialog.Bind(wx.EVT_BUTTON, handle_pdb, pdb_button)

    button_sizer = wx.StdDialogButtonSizer()
    yes_button = wx.Button(dialog, wx.ID_YES, "Stop processing...")
    no_button = wx.Button(dialog, wx.ID_NO, "Continue processing...")
    skip_button = wx.Button(dialog,wx.ID_HELP,'Skip Image, Continue Pipeline')
    button_sizer.AddButton(yes_button)
    button_sizer.AddButton(no_button)
    button_sizer.AddButton(report_button)
    button_sizer.AddButton(skip_button)
    button_sizer.Realize()
    sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 4)
    result = [None]
    #
    # Handle the "No" button being pressed
    #
    no_button = button_sizer.GetNegativeButton()
    def on_no(event):
        result[0] = ED_CONTINUE
        dialog.SetReturnCode(wx.NO)
        dialog.Close()
        event.Skip()
    dialog.Bind(wx.EVT_BUTTON, on_no, no_button)
    
    def on_yes(event):
        dialog.SetReturnCode(wx.YES)
        result[0] = ED_STOP
        dialog.Close()
        event.Skip()
    dialog.Bind(wx.EVT_BUTTON, on_yes, yes_button)
    
    #
    # Handle "Skip Image" button being pressed
    #
    def on_skip(event):
        result[0] = ED_SKIP
        dialog.Close()
        event.Skip()
    dialog.Bind(wx.EVT_BUTTON,on_skip,skip_button)
    
    dialog.Fit()
    dialog.ShowModal()
    return result[0]

def on_report(event, dialog, traceback_text, pipeline):
    '''Report an error to us'''
    from cellprofiler.utilities.get_revision import get_revision
    params = { "traceback":traceback_text,
               "revision":str(get_revision()),
               "platform":str(platform.platform())
               }
    try:
        obfuscated_pipeline = pipeline.copy()
        obfuscated_pipeline.obfuscate()
        fd = StringIO()
        obfuscated_pipeline.savetxt(fd)
        fd.seek(0)
        pipeline_text = fd.read()
        params["pipeline"] = pipeline_text
    except:
        pass
    headers = {"Accept": "text/plain"}
    data =  urllib.urlencode(params)
    req = urllib2.Request(ERROR_URL, data, headers)
    try:
        conn = urllib2.urlopen(req)
        response = conn.read()
        wx.MessageBox("Report successfully sent to CellProfiler.org. Thank you.",
                      parent = dialog)
    except urllib2.HTTPError, e:
        wx.MessageBox("Failed to upload, server reported code %d"%(e.code))
    except urllib2.URLError, e:
        wx.MessageBox("Failed to upload: %s"%(e.reason))

def show_warning(title, message, get_preference, set_preference):
    '''Show a silenceable warning message to the user
    
    title - title for the dialog box
    
    message - message to be displayed
    
    get_preference - function that gets a user preference: do you want to
                     show this warning?
    
    set_preference - function that sets the user preference if they choose
                     not to see the warning again.
                     
    The message is printed to the console if headless.
    '''
    from cellprofiler.preferences import get_headless
    
    if get_headless():
        print message
        return
    
    if not get_preference():
        return
    
    import wx
    if wx.GetApp() is None:
        print message
        return
    
    dlg = wx.Dialog(wx.GetApp().GetTopWindow(), title = title)
    dlg.Sizer = sizer = wx.BoxSizer(wx.VERTICAL)
    subsizer = wx.BoxSizer(wx.HORIZONTAL)
    sizer.Add(subsizer, 0, wx.EXPAND | wx.ALL, 5)
    subsizer.Add(wx.StaticBitmap(dlg, wx.ID_ANY,
                                 wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
                                                          wx.ART_CMN_DIALOG)),
                 0, wx.ALIGN_LEFT | wx.ALIGN_TOP | wx.RIGHT, 5)
    text = wx.StaticText(dlg, wx.ID_ANY, message)
    subsizer.Add(text, 0, wx.ALIGN_LEFT | wx.ALIGN_TOP | wx.ALL, 5)
    dont_show = wx.CheckBox(dlg, 
                            label = "Don't show this message again.")
    sizer.Add(dont_show, 0, wx.ALIGN_LEFT | wx.ALL, 5)
    buttons_sizer = wx.StdDialogButtonSizer()
    buttons_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
    buttons_sizer.Realize()
    sizer.Add(buttons_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
    dlg.Fit()
    dlg.ShowModal()
    if dont_show.Value:
        set_preference(False)

    
    
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
