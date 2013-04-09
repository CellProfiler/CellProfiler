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
from cellprofiler.gui.help import WORKSPACE_INTRO_HELP

'''The value returned if the user wants to load a workspace and its pipeline'''
LOAD_PIPELINE_AND_WORKSPACE = wx.NewId()
S_LOAD_PIPELINE_AND_WORKSPACE = "Load pipeline and workspace"

'''The value returned if the user wants to load a pipeline into the current ws'''
LOAD_PIPELINE_ONLY = wx.NewId()
S_LOAD_PIPELINE_ONLY = "Load pipeline only"

'''The value returned if the user wants to load a workspace, but keep pipeline'''
LOAD_WORKSPACE_ONLY = wx.NewId()
S_LOAD_WORKSPACE_ONLY = "Load workspace only"

'''The value returned if the user wants to view an image'''
LOAD_VIEW_IMAGE = wx.NewId()
S_VIEW_IMAGE = "View image"

LOAD_HELP = """
<p>From this dialog, you can choose the file you want to load into CellProfiler. 
You can load workspaces, pipelines and images with this dialog.</p>

<p><i>Pipelines</i> contain the analysis 
modules that will be run when processing your image sets. <i>Workspaces</i> 
contain the list of files that make up your image set.<br>
You can select from the following options:
<ul>
<li><i>%(S_LOAD_PIPELINE_AND_WORKSPACE)s</i>: This option loads a file with the extension 
.cpi. These files contain the saved pipeline and the saved list of files that make up 
your image set, along with any associated 
metadata. Choosing this option will overwrite both the settings in the workspace panel 
and the pipeline panel.<br>
If you are new to workspaces, please see <i>%(WORKSPACE_INTRO_HELP)s</i>
for more details.
</li>
<li><i>%(S_LOAD_PIPELINE_ONLY)s</i>: This option load a .cp file containing the series of 
modules comprising your analysis pipeline. Only the modules in 
the pipeline panel will be overwritten; your current workspace will be preserved.</li> 
<li><i>%(S_LOAD_WORKSPACE_ONLY)s</i>: This option loads the workspace contained in
the .cpi file. Only the settings in the workspace panel will be overwritten; your
current pipeline settings will be preserved.</li>
<li><i>%(S_VIEW_IMAGE)s</i>: Selecting this option will allow you to view
a selected image.</li> 
</ul>
"""%globals()

"""User wants to save the workspace as a .cpi"""
SAVE_PIPELINE_AND_WORKSPACE = wx.NewId()
S_SAVE_PIPELINE_AND_WORKSPACE = "Save pipeline and workspace (.cpi)"

"""User wants to save the pipeline as a .cp"""
SAVE_PIPELINE_ONLY = wx.NewId()
S_SAVE_PIPELINE_ONLY = "Save pipeline only (.cp)"

"""User wants to export the image sets as a .csv"""
SAVE_EXPORT_IMAGE_SET_LIST = wx.NewId()
S_SAVE_EXPORT_IMAGE_SET_LIST = "Export image set listing (.csv)"

SAVE_HELP = """
<p>Chose the file format that you want to save your work as.
You can select from the following three format options:
<ul>
<li><i>%(S_SAVE_PIPELINE_AND_WORKSPACE)s:</i> Save your current pipeline 
settings as well as your current workspace to a .cpi file. The file 
contains the image information, such as the list of files 
that make up your image set and associated metadata, along with the pipeline.<br>
This option is recommended if 
you want run the same pipeline on the same selection of images.<br>
If you are new to workspaces, please see <i>%(WORKSPACE_INTRO_HELP)s</i>
for more details.</li>
<li><i>%(S_SAVE_PIPELINE_ONLY)s:</i> Save the current analysis pipeline
to a .cp file. <br>
This option is recomended if you want to apply the same analysis to a 
<i>different</i> set of images. Also, since the image file list is not
included, the pipeline file is smaller.</li>
<li><i>%(S_SAVE_EXPORT_IMAGE_SET_LIST)s:</i> Save a comma-delimited file 
(.csv) of the image filenames, locations and associated workspace 
information. <br>
This option is recommended if you want to use <b>LoadData</b> for loading
image files, which may be optimal for some laboratory information management 
systems.</li>
</ul></p>
"""%globals()

def show_load_dlg(frame):
    '''Show the Load dialog
    
    Show the Load dialog and return the user response, one of
    LOAD_PIPELINE_AND_WORKSPACE, LOAD_PIPELINE_ONLY, LOAD_WORKSPACE_ONLY,
    LOAD_VIEW_IMAGE or None if user cancelled.
    '''
    
    return show_a_dialog(
        frame, "Load files", LOAD_HELP, (
        (LOAD_PIPELINE_AND_WORKSPACE, S_LOAD_PIPELINE_AND_WORKSPACE,
         "Loads a pipeline, along with the workspace (image locations and\n"
         "associated information) created during the session."),
        (LOAD_PIPELINE_ONLY, S_LOAD_PIPELINE_ONLY,
         "Loads the pipeline only, allowing you to load a workspace separately\n"
         "or create a new one from scratch."),
        (LOAD_WORKSPACE_ONLY, S_LOAD_WORKSPACE_ONLY,
         "Loads only the workspace, allowing you to load pipeline modules\n"
         "separately or build one from scratch."),
        (LOAD_VIEW_IMAGE, S_VIEW_IMAGE,
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
        (SAVE_PIPELINE_AND_WORKSPACE, S_SAVE_PIPELINE_AND_WORKSPACE,
         "Save pipeline settings and current workspace (image locations\n"
         "and associated information). Recommended for saving your current\n"
         "work across sessions with the same workspace."),
        (SAVE_PIPELINE_ONLY, S_SAVE_PIPELINE_ONLY,
         "Save a text file of module settings. Recommended for runing on\n"
         "analysis on a different workspace."),
        (SAVE_EXPORT_IMAGE_SET_LIST, S_SAVE_EXPORT_IMAGE_SET_LIST,
         "Save a comma-delimited file (.csv) of the image filenames,\n"
         "locations and associated workspace information")))
         
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
    