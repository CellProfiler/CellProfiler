# coding=utf-8
"""errordialog - dialog box for reporting error.
"""


import functools
import io
import logging
import os
import platform
import sys
import traceback
import urllib
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from cellprofiler import __version__ as cellprofiler_version
from cellprofiler_core.preferences import get_headless

ED_STOP = "Stop"
ED_CONTINUE = "Continue"
ED_SKIP = "Skip"

ERROR_URL = "http://www.cellprofiler.org/cgi-bin/reporterror.cgi"

__inside_display_error_dialog = False

# keep track of errors that have already been reported this session,
# and just log them to the console, rather than putting up the dialog.
previously_seen_error_locations = set()


def clear_old_errors():
    global previously_seen_error_locations
    previously_seen_error_locations = set()


def display_error_dialog(*args, **kwargs):
    """Display an error dialog, returning an indication of whether to continue

    frame - parent frame for application
    exc - exception that caused the error
    pipeline - currently executing pipeline
    message - message to display
    tb - traceback
    continue_only - show "continue" option, only
    remote_exc_info - None (the default) for exceptions in the current process.
        For remote processes:
            (exc_name, exc_message, traceback_text, filename, line_number, remote_debug_callback)

    Returns either ED_STOP or ED_CONTINUE indicating how to handle.
    """
    global __inside_display_error_dialog
    if __inside_display_error_dialog:
        return
    __inside_display_error_dialog = True
    try:
        return _display_error_dialog(*args, **kwargs)
    except Exception:
        # raising exceptions in our exception handler is bad.
        try:
            logging.root.error("Exception in display_error_dialog()!", exc_info=True)
        except Exception:
            sys.stderr.write(
                "Exception logging exception in display_error_dialog().  Everything probably broken.\n"
            )
            pass
    finally:
        __inside_display_error_dialog = False


def _display_error_dialog(
    frame,
    exc,
    pipeline,
    message=None,
    tb=None,
    continue_only=False,
    remote_exc_info=None,
):
    """Display an error dialog, returning an indication of whether to continue

    frame - parent frame for application
    exc - exception that caused the error
    pipeline - currently executing pipeline
    message - message to display
    tb - traceback
    continue_only - show "continue" option, only
    remote_exc_info - None (the default) for exceptions in the current process.
        For remote processes:
            (exc_name, exc_message, traceback_text, filename,
             line_number, remote_event_queue)

    Returns either ED_STOP or ED_CONTINUE indicating how to handle.
    """

    import wx

    assert wx.IsMainThread(), "Can only display errors from WX thread."

    if remote_exc_info:
        from_subprocess = True
        (
            exc_name,
            exc_message,
            traceback_text,
            filename,
            line_number,
            remote_debug_callback,
        ) = remote_exc_info
        if message is None:
            message = exc_message
    else:
        from_subprocess = False
        if message is None:
            message = str(exc)
        if tb is None:
            traceback_text = traceback.format_exc()
            tb = sys.exc_info()[2]
        else:
            traceback_text = "".join(traceback.format_exception(type(exc), exc, tb))

        # find the place where this error occurred, and if we've already
        # reported it, don't do so again (instead, just log it to the
        # console), to prevent the UI from becoming unusable.
        filename, line_number, _, _ = traceback.extract_tb(tb)[-1]

    if (filename, line_number) in previously_seen_error_locations:
        if from_subprocess:
            logging.root.error(
                "Previously displayed remote exception:\n%s\n%s",
                exc_name,
                traceback_text,
            )
        else:
            logging.root.error(
                "Previously displayed uncaught exception:",
                exc_info=(type(exc), exc, tb),
            )
        return ED_CONTINUE

    dialog = wx.Dialog(
        frame, title="Pipeline error", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
    )
    sizer = wx.BoxSizer(wx.VERTICAL)
    dialog.SetSizer(sizer)
    if continue_only:
        qc_msg = "Encountered error while processing."
    else:
        qc_msg = (
            "Encountered error while processing. " "Do you want to stop processing?"
        )
    question_control = wx.StaticText(dialog, -1, qc_msg)
    question_control.Font = wx.Font(
        int(dialog.GetFont().GetPointSize() * 5 / 4),
        dialog.GetFont().GetFamily(),
        dialog.GetFont().GetStyle(),
        wx.FONTWEIGHT_BOLD,
    )
    sizer.Add(question_control, 0, wx.EXPAND | wx.ALL, 5)
    error_box = wx.BoxSizer(wx.HORIZONTAL)
    message_control = wx.StaticText(dialog, -1, message)
    error_box.Add(message_control, 1, wx.EXPAND | wx.RIGHT, 5)
    sizer.Add(error_box, 1, wx.EXPAND | wx.ALL, 5)
    aux_button_box = wx.BoxSizer(wx.VERTICAL)
    error_box.Add(aux_button_box, 0, wx.EXPAND)

    #
    # Handle show details button
    #
    details_button = wx.Button(dialog, -1, "Details...")
    details_button.SetToolTip("Show error details")
    aux_button_box.Add(details_button, 0, wx.EXPAND | wx.BOTTOM, 5)
    details_on = [False]

    def on_details(event):
        if not details_on[0]:
            message_control.SetLabel("%s\n%s" % (message, traceback_text))
            message_control.Refresh()
            details_button.SetLabel("Hide details...")
            details_button.Refresh()
            dialog.Fit()
            details_on[0] = True
        else:
            message_control.SetLabel(message)
            message_control.Refresh()
            details_button.SetLabel("Details...")
            details_button.Refresh()
            dialog.Fit()
            details_on[0] = False

    dialog.Bind(wx.EVT_BUTTON, on_details, details_button)

    #
    # Handle copy button
    #
    copy_button = wx.Button(dialog, -1, "Copy to clipboard")
    copy_button.SetToolTip("Copy error to clipboard")
    aux_button_box.Add(copy_button, 0, wx.EXPAND | wx.BOTTOM, 5)

    def on_copy(event):
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.Clear()
                wx.TheClipboard.SetData(wx.TextDataObject(traceback_text))
                wx.TheClipboard.Flush()
            finally:
                wx.TheClipboard.Close()

    dialog.Bind(wx.EVT_BUTTON, on_copy, copy_button)

    #
    # Handle pdb button
    #
    if ((tb or remote_exc_info) is not None) and (
        not hasattr(sys, "frozen") or os.getenv("CELLPROFILER_DEBUG")
    ):
        if not from_subprocess:
            pdb_button = wx.Button(dialog, -1, "Debug in pdb...")
            pdb_button.SetToolTip("Debug in python's pdb on the console")
            aux_button_box.Add(pdb_button, 0, wx.EXPAND | wx.BOTTOM, 5)

            def handle_pdb(event):
                import pdb

                pdb.post_mortem(tb)
                # This level of interest seems to indicate the user might
                # want to debug this error if it occurs again.
                if (filename, line_number) in previously_seen_error_locations:
                    previously_seen_error_locations.remove((filename, line_number))

        else:
            pdb_button = wx.Button(dialog, -1, "Debug remotely...")
            pdb_button.SetToolTip("Debug remotely in pdb via telnet")
            aux_button_box.Add(pdb_button, 0, wx.EXPAND | wx.BOTTOM, 5)

            def handle_pdb(event):
                if not remote_debug_callback():
                    # The user has told us that remote debugging has gone wonky
                    pdb_button.Enable(False)
                # This level of interest seems to indicate the user might
                # want to debug this error if it occurs again.
                if (filename, line_number) in previously_seen_error_locations:
                    previously_seen_error_locations.remove((filename, line_number))

        dialog.Bind(wx.EVT_BUTTON, handle_pdb, pdb_button)
    dont_show_exception_checkbox = wx.CheckBox(
        dialog, label="Don't show this error again"
    )
    dont_show_exception_checkbox.SetValue(False)
    sizer.Add(dont_show_exception_checkbox, 0, wx.ALIGN_LEFT | wx.ALL, 5)
    #
    # Handle the "stop" button being pressed
    #
    result = [None]

    def on_stop(event):
        dialog.SetReturnCode(wx.YES)
        result[0] = ED_STOP
        dialog.Close()
        event.Skip()

    stop_button = wx.Button(dialog, label="Stop processing...")
    dialog.Bind(wx.EVT_BUTTON, on_stop, stop_button)

    #
    # Handle the "continue" button being pressed
    #
    def on_continue(event):
        result[0] = ED_CONTINUE
        dialog.SetReturnCode(wx.NO)
        dialog.Close()
        event.Skip()

    continue_button = wx.Button(dialog, label="Continue processing...")
    dialog.Bind(wx.EVT_BUTTON, on_continue, continue_button)

    #
    # Handle report button
    #
    def handle_report(event):
        on_report(event, dialog, traceback_text, pipeline)

    report_button = wx.Button(dialog, label="Send report...")
    report_button.SetToolTip("Upload error report to the CellProfiler Project")
    dialog.Bind(wx.EVT_BUTTON, handle_report, report_button)

    #
    # Handle "Skip Image" button being pressed
    #
    def on_skip(event):
        result[0] = ED_SKIP
        dialog.Close()
        event.Skip()

    skip_button = wx.Button(dialog, label="Skip Image, Continue Pipeline")
    dialog.Bind(wx.EVT_BUTTON, on_skip, skip_button)

    button_sizer = wx.BoxSizer(wx.HORIZONTAL)
    button_sizer.Add((2, 2))
    button_sizer.Add(stop_button)
    button_sizer.Add((5, 5), proportion=1)
    button_sizer.Add(continue_button)
    button_sizer.Add((5, 5), proportion=1)
    button_sizer.Add(report_button)
    button_sizer.Add((5, 5), proportion=1)
    button_sizer.Add(skip_button)
    button_sizer.Add((2, 2))
    if continue_only:
        button_sizer.Hide(stop_button)
        button_sizer.Hide(skip_button)

    sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 4)

    dialog.Fit()
    dialog.ShowModal()
    if dont_show_exception_checkbox.IsChecked():
        previously_seen_error_locations.add((filename, line_number))
    return result[0]


def on_report(event, dialog, traceback_text, pipeline):
    """Report an error to us"""
    params = {
        "traceback": traceback_text,
        "revision": cellprofiler_version,
        "platform": str(platform.platform()),
    }
    try:
        obfuscated_pipeline = pipeline.copy()
        obfuscated_pipeline.obfuscate()
        fd = io.StringIO()
        obfuscated_pipeline.dump(fd)
        fd.seek(0)
        pipeline_text = fd.read()
        params["pipeline"] = pipeline_text
    except:
        pass
    headers = {"Accept": "text/plain"}
    data = urllib.parse.urlencode(params)
    req = urllib.request.Request(ERROR_URL, data, headers)
    import wx

    try:
        conn = urlopen(req)
        response = conn.read()
        wx.MessageBox(
            "Report successfully sent to CellProfiler.org. Thank you.", parent=dialog
        )
    except HTTPError as e:
        wx.MessageBox("Failed to upload, server reported code %d" % e.code)
    except URLError as e:
        wx.MessageBox("Failed to upload: %s" % e.reason)


def show_warning(title, message, get_preference, set_preference):
    """Show a silenceable warning message to the user

    title - title for the dialog box

    message - message to be displayed

    get_preference - function that gets a user preference: do you want to
                     show this warning?

    set_preference - function that sets the user preference if they choose
                     not to see the warning again.

    The message is printed to the console if headless.
    """

    if get_headless():
        print(message)
        return

    if not get_preference():
        return

    import wx

    if wx.GetApp() is None:
        print(message)
        return

    with wx.Dialog(None, title=title) as dlg:
        dlg.Sizer = sizer = wx.BoxSizer(wx.VERTICAL)
        subsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(subsizer, 0, wx.EXPAND | wx.ALL, 5)
        subsizer.Add(
            wx.StaticBitmap(
                dlg,
                wx.ID_ANY,
                wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_CMN_DIALOG),
            ),
            0,
            wx.ALIGN_LEFT | wx.ALIGN_TOP | wx.RIGHT,
            5,
        )
        text = wx.StaticText(dlg, wx.ID_ANY, message)
        subsizer.Add(text, 0, wx.ALIGN_LEFT | wx.ALIGN_TOP | wx.ALL, 5)
        dont_show = wx.CheckBox(dlg, label="Don't show this message again.")
        sizer.Add(dont_show, 0, wx.ALIGN_LEFT | wx.ALL, 5)
        buttons_sizer = wx.StdDialogButtonSizer()
        buttons_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        buttons_sizer.Realize()
        sizer.Add(buttons_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        dlg.Fit()
        dlg.ShowModal()
        if dont_show.GetValue():
            set_preference(False)


def display_error_message(parent, message, title, buttons=None, size=(300, 200)):
    """Display an error in a scrolling message box

    parent - parent window to the error message
    message - message to display in scrolling box
    title - title to display in frame
    buttons - a list of buttons to put at bottom of dialog. For instance,
              [wx.ID_YES, wx.ID_NO]. Defaults to OK button
    size - size of frame. Defaults to 300 x 200 but will fit.

    returns the code from ShowModal.
    """
    import wx

    if buttons is None:
        buttons = [wx.ID_OK]
    else:
        assert len(buttons) > 0

    with wx.Dialog(
        parent, title=title, size=size, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
    ) as dlg:
        assert isinstance(dlg, wx.Dialog)
        dlg.SetSizer(wx.BoxSizer(wx.VERTICAL))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        dlg.GetSizer().AddSpacer(20)
        dlg.GetSizer().Add(sizer, 1, wx.EXPAND)

        sizer.AddSpacer(10)
        icon = wx.ArtProvider.GetBitmap(wx.ART_ERROR)
        sizer.Add(
            wx.StaticBitmap(dlg, bitmap=icon),
            0,
            wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP | wx.ALL,
            10,
        )
        sizer.AddSpacer(10)
        message_ctrl = wx.TextCtrl(
            dlg, value=message, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.NO_BORDER
        )
        line_sizes = [
            message_ctrl.GetFullTextExtent(line) for line in message.split("\n")
        ]
        width = functools.reduce(max, [x[0] for x in line_sizes])
        width += wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_X)
        width += wx.SystemSettings.GetMetric(wx.SYS_BORDER_X) * 2
        height = sum([x[1] for x in line_sizes])
        message_ctrl.SetMinSize((width, min(height, size[1])))
        sizer.Add(message_ctrl, 1, wx.EXPAND)
        sizer.AddSpacer(10)

        dlg.GetSizer().AddSpacer(10)
        button_sizer = wx.StdDialogButtonSizer()
        dlg.GetSizer().Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)

        def on_button(event):
            id2code = {
                wx.ID_YES: wx.YES,
                wx.ID_NO: wx.NO,
                wx.ID_CANCEL: wx.CANCEL,
                wx.ID_OK: wx.OK,
            }
            assert isinstance(event, wx.Event)
            dlg.EndModal(id2code[event.GetId()])

        for button in buttons:
            button_ctl = wx.Button(dlg, button)
            button_sizer.AddButton(button_ctl)
            button_ctl.Bind(wx.EVT_BUTTON, on_button)
        button_ctl.SetFocus()
        button_sizer.Realize()
        dlg.Fit()
        return dlg.ShowModal()
