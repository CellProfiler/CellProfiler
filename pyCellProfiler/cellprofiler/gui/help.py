""" help.py - contains menu structures for help menus in CP

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

#######################################################
#
# There are different windows in CP and many of them
# have help categories that need their text populated.
# This file holds that help. First, there are lists
# of tuples where the first item in the tuple
# is whatever goes into the menu and the second item
# is either another list or it is
# HTML text to be displayed.
#
# At the bottom of this file is the uber-dictionary which
# has all of the help and that one is used when we generate
# the HTML manual.
#
########################################################

import os
import wx
import htmldialog

##################################################
#
# Help for the main window
#
##################################################

DEFAULT_IMAGE_FOLDER_HELP = """
The default image folder is the folder that holds your images
unless you do something to customize it."""

DEFAULT_OUTPUT_FOLDER_HELP = """
The default output folder is the folder that CellProfiler uses to
store its output unless you do something to customize it."""

OUTPUT_FILENAME_HELP = """
The measurements for your run are stored in the output file name"""

'''The help menu for CP's main window'''
MAIN_HELP = (
    ( "Getting started", (
        ("Installation",
         """Here's how to install <b>CellProfiler</b>"""),
        ("Making a pipeline",
         """So you want to make a <i>Pipeline</i>?"""))),
    ( "General help", (
        ( "Memory",
          """Memory is important"""),
        ("Speed",
         """Speed is nice""") ) ),
    ( "Folders and files", (
        ("Default image folder", DEFAULT_IMAGE_FOLDER_HELP),
        ("Default output folder", DEFAULT_OUTPUT_FOLDER_HELP),
        ("Output file name", OUTPUT_FILENAME_HELP) ) )
)

####################################################
#
# Help for the module figure windows
#
####################################################
'''The help menu for the figure window'''
FIGURE_HELP = (
    ( "Zoom", 
      """You can zoom in and zoom out""" ),
    ( "Show pixel data", 
      """You can show pixel intensity and position. You can measure the
      distance between two things"""))

###################################################
#
# Help for the preferences dialog
#
###################################################

TITLE_FONT_HELP = """The title font preference sets the font that's used
in titles above plots in module figure windows"""
TABLE_FONT_HELP = """The table font preference sets the font that's used
in tables in module figure windows"""
DEFAULT_COLORMAP_HELP = """The default colormap preference chooses the
color map that's used to get the colors for labels and other elements"""
WINDOW_BACKGROUND_HELP = """The window background preference sets the
window background color"""
CHECK_FOR_UPDATES_HELP = """The check for updates preference controls how
CellProfiler looks for updates on startup."""

PREFERENCES_HELP = (
    ( "Default image folder", DEFAULT_IMAGE_FOLDER_HELP),
    ( "Default output folder", DEFAULT_OUTPUT_FOLDER_HELP),
    ( "Title font", TITLE_FONT_HELP ),
    ( "Table font", TABLE_FONT_HELP ),
    ( "Default colormap", DEFAULT_COLORMAP_HELP ),
    ( "Window background", WINDOW_BACKGROUND_HELP ),
    ( "Check for updates", CHECK_FOR_UPDATES_HELP ))

HELP = ( ("User guide", MAIN_HELP ), 
         ("Module figures", FIGURE_HELP ),
         ("Preferences", PREFERENCES_HELP))

def make_help_menu(h, window):
    menu = wx.Menu()
    for key, value in h:
        my_id = wx.NewId()
        if hasattr(value, "__iter__") and not isinstance(value, (str, unicode)):
            menu.AppendMenu(my_id, key, make_help_menu(value, window))
        else:
            def show_dialog(event, key=key, value=value):
                dlg = htmldialog.HTMLDialog(window, key, value)
                dlg.Show()
                
            menu.Append(my_id, key)
            window.Bind(wx.EVT_MENU, show_dialog, id=my_id)
    return menu

def output_gui_html():
    root = os.path.split(__file__)[0]
    if len(root) == 0:
        root = os.curdir
    root = os.path.abspath(root)
    gui_html = os.path.join(root, 'html')
    if not (os.path.exists(gui_html) and os.path.isdir(gui_html)):
        try:
            os.mkdir(gui_html)
        except IOError:
            gui_html = root
    index_fd = open(os.path.join(gui_html,'index.html'),'w')
        
    index_fd.write("""<html>
<head>
    <title>CellProfiler: User guide table of contents</title>
</head>
<body>
<h1>CellProfiler: User guide table of contents</h1>""")
    def write_menu(prefix, h):
        index_fd.write("<ul>\n")
        for key, value in h:
            index_fd.write("<li>")
            if hasattr(value, "__iter__") and not isinstance(value, (str, unicode)):
                index_fd.write(key)
                write_menu(prefix+"_"+key, value)
            else:
                file_name = "%s_%s.html" % (prefix, key)
                path = os.path.join(gui_html, file_name)
                fd = open(path, "w")
                fd.write("<html><head><title>%s</title></head>\n" % key)
                fd.write("<body><h1>%s</h1>\n<div>\n" % key)
                fd.write(value)
                fd.write("</div></body>\n")
                fd.close()
                index_fd.write("<a href='%s'>%s</a>\n" % 
                               (file_name, key) )
            index_fd.write("</li>\n")
        index_fd.write("</ul>\n")
    write_menu("help", HELP)
    index_fd.write("</body>\n")
    index_fd.close()
    