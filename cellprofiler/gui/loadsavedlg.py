'''loadsavedlg.py - dialog boxes for loading and saving CellProfiler files

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import wx
from cellprofiler.gui.htmldialog import HTMLDialog

'''The value returned if the user wants to load a workspace and its pipeline'''
LOAD_PIPELINE_AND_WORKSPACE = wx.NewId()

'''The value returned if the user wants to load a pipeline into the current ws'''
LOAD_PIPELINE_ONLY = wx.NewId()

'''The value returned if the user wants to load a workspace, but keep pipeline'''
LOAD_WORKSPACE_ONLY = wx.NewId()

'''The value returned if the user wants to view an image'''
LOAD_VIEW_IMAGE = wx.NewId()

LOAD_HELP = """
<h1>Help for <i>Load files</i></h1>
Choose the file you want to load into CellProfiler. You can load workspaces,
pipelines and images with this dialog. Pipeline files contain the modules
that will be run when processing your image sets. Workspace files contain
both a pipeline and the list of files that make up your image set. You have
the choice of loading just the list of files or the files and pipeline
when you load a workspace. If you load just the list of files, CellProfiler
will overwrite the pipeline in your workspace with your current pipeline.<b>
In addition, you can view images using this dialog by choosing 
<i>View image</i>
"""

"""User wants to save the workspace as a .cpi"""
SAVE_PIPELINE_AND_WORKSPACE = wx.NewId()

"""User wants to save the pipeline as a .cp"""
SAVE_PIPELINE_ONLY = wx.NewId()

"""User wants to export the image sets as a .csv"""
SAVE_EXPORT_IMAGE_SET_LIST = wx.NewId()

SAVE_HELP = """
<h1>Help for <i>Save files</i></h1>
Choose the kind of file you want to save.
"""
def show_load_dlg(frame):
    '''Show the Load dialog
    
    Show the Load dialog and return the user response, one of
    LOAD_PIPELINE_AND_WORKSPACE, LOAD_PIPELINE_ONLY, LOAD_WORKSPACE_ONLY,
    LOAD_VIEW_IMAGE or None if user cancelled.
    '''
    
    return show_a_dialog(
        frame, "Load files", LOAD_HELP, (
        (LOAD_PIPELINE_AND_WORKSPACE, "Load pipeline and workspace",
         "Loads a pipeline, along with the workspace (image locations and\n"
         "associated information) created during the session."),
        (LOAD_PIPELINE_ONLY, "Load pipeline only",
         "Loads the pipeline only, allowing you to load a workspace separately\n"
         "or create a new one from scratch."),
        (LOAD_WORKSPACE_ONLY, "Load workspace only",
         "Loads only the workspace, allowing you to load pipeline modules\n"
         "separately or build one from scratch."),
        (LOAD_VIEW_IMAGE, "View image",
         "Opens images for viewing. Note that images opened here are not\n"
         "placed in the workspace.")))
    
def show_save_dlg(frame):
    '''Show the save dialog
    
    frame - parent frame
    
    Show the save dialog and return the user choice:
    SAVE_PIPELINE_AND_WORKSPACE - user wants to save the workspace as a .cpi
    SAVE_PIPELINE_ONLY - user wants to save the pipeline as a .cp
    SAVE_EXPORT_IMAGE_SET_LIST - user wants to export the image sets as a .csv
    '''
    return show_a_dialog(frame, "Save files", SAVE_HELP, (
        (SAVE_PIPELINE_AND_WORKSPACE, "Save pipeline and workspace (.cpi)",
         "Save pipeline settings and current workspace (image locations\n"
         "and associated information). Recommended for saving your current\n"
         "work across sessions with the same workspace."),
        (SAVE_PIPELINE_ONLY, "Save pipeline only (.cp)",
         "Save a text file of module settings. Recommended for runing on\n"
         "analysis on a different workspace."),
        (SAVE_EXPORT_IMAGE_SET_LIST, "Export image set listing (.csv)",
         "Save a comma-delimited file (.csv) of the image filenames,\n"
         "paths and associated workspace information")))
         
def show_a_dialog(frame, title, help, contents):
    '''Show a dialog with buttons on the left and help on the right
    
    frame - parent frame to dialog
    
    title - dialog's title
    
    help - HTML help for the dialog
    
    contents - a sequence of 3 tuples for each choice
       contents[i][0] is the ID to be assigned to the button
       contents[i][1] is the label that appears on the button
       contents[i][2] is the text (with line breaks please) that appears
       to the right of the button
       
    returns either the ID of the button that the user pressed or None for cancel
    '''
    with wx.Dialog(frame, -1, title) as dlg:
        #
        # Two sizers used here so that the FlexGridSizer can have some
        # decent borders between it and the side of the window
        #
        dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.FlexGridSizer(4, 2, vgap = 10, hgap = 10)
        dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        result = [ None ]
        def on_button(event):
            result[0] = event.Id
            dlg.EndModal(event.Id)
        
        for button_id, button_text, text in contents:
            button = wx.Button(dlg, button_id, button_text)
            button.Bind(wx.EVT_BUTTON, on_button)
            sizer.Add(button, 0, wx.EXPAND)
            static_text = wx.StaticText(dlg, label = text)
            sizer.Add(static_text, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP)
        
        button_sizer = dlg.CreateStdDialogButtonSizer(wx.CANCEL | wx.HELP)
        dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        def on_help(event):
            with HTMLDialog(dlg, "Help for %s" % title, help) as help_dlg:
                help_dlg.ShowModal()
        dlg.Bind(wx.EVT_BUTTON, on_help, id=wx.ID_HELP)
        dlg.Bind(wx.EVT_HELP, on_help)
        dlg.Fit()
        dlg.ShowModal()
        return result[0]
                   
if __name__ == "__main__":

    app = wx.PySimpleApp()
    result = show_load_dlg(None)
    print result
    
    result = show_save_dlg(None)
    print result
    