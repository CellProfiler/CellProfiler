# coding: utf-8

import os
import re

import pkg_resources

import cellprofiler


def __image_resource(filename):
    return pkg_resources.resource_filename(
        "cellprofiler",
        os.path.join("data", "images", filename)
    )


def read_content(filename):
    resource_filename = pkg_resources.resource_filename(
        "cellprofiler",
        os.path.join("data", "help", filename)
    )

    with open(resource_filename, "r") as f:
        content = f.read()

    return re.sub(
        r"image:: (.*\.png)",
        lambda md: "image:: {}".format(
            __image_resource(os.path.basename(md.group(0)))
        ),
        content
    )

MANUAL_URL = "http://cellprofiler-manual.s3.amazonaws.com/CellProfiler-{}/index.html".format(cellprofiler.__version__)

X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
VIEW_OUTPUT_SETTINGS_BUTTON_NAME = "View output settings"


####################
#
# ICONS
#
####################
MODULE_HELP_BUTTON = __image_resource('module_help.png')
MODULE_MOVEUP_BUTTON = __image_resource('module_moveup.png')
MODULE_MOVEDOWN_BUTTON = __image_resource('module_movedown.png')
MODULE_ADD_BUTTON = __image_resource('module_add.png')
MODULE_REMOVE_BUTTON = __image_resource('module_remove.png')
TESTMODE_PAUSE_ICON = __image_resource('IMG_PAUSE.png')
TESTMODE_GO_ICON = __image_resource('IMG_GO.png')
DISPLAYMODE_SHOW_ICON = __image_resource('eye-open.png')
DISPLAYMODE_HIDE_ICON = __image_resource('eye-close.png')
SETTINGS_OK_ICON = __image_resource('check.png')
SETTINGS_ERROR_ICON = __image_resource('remove-sign.png')
SETTINGS_WARNING_ICON = __image_resource('IMG_WARN.png')
RUNSTATUS_PAUSE_BUTTON = __image_resource('status_pause.png')
RUNSTATUS_STOP_BUTTON = __image_resource('status_stop.png')
RUNSTATUS_SAVE_BUTTON = __image_resource('status_save.png')
WINDOW_HOME_BUTTON = __image_resource('window_home.png')
WINDOW_BACK_BUTTON = __image_resource('window_back.png')
WINDOW_FORWARD_BUTTON = __image_resource('window_forward.png')
WINDOW_PAN_BUTTON = __image_resource('window_pan.png')
WINDOW_ZOOMTORECT_BUTTON = __image_resource('window_zoom_to_rect.png')
WINDOW_SAVE_BUTTON = __image_resource('window_filesave.png')
ANALYZE_IMAGE_BUTTON = __image_resource('IMG_ANALYZE_16.png')
STOP_ANALYSIS_BUTTON = __image_resource('stop.png')
PAUSE_ANALYSIS_BUTTON = __image_resource('IMG_PAUSE.png')


####################
#
# MENU HELP PATHS
#
####################
BATCH_PROCESSING_HELP_REF = """Help > Batch Processing"""
TEST_MODE_HELP_REF = """Help > Testing Your Pipeline"""
IMAGE_TOOLS_HELP_REF = """Help > Using Module Display Windows > How To Use The Image Tools"""
DATA_TOOL_HELP_REF = """Data Tools > Help"""
USING_YOUR_OUTPUT_REF = """Help > Using Your Output"""
MEASUREMENT_NAMING_HELP = """Help > Using Your Output > How Measurements are Named"""

####################
#
# MENU HELP CONTENT
#
####################
FIGURE_HELP = (
    ("Using The Display Window Menu Bar", read_content("display_menu_bar.rst")),
    ("Using The Interactive Navigation Toolbar", read_content("display_interactive_navigation.rst")),
    ("How To Use The Image Tools", read_content("display_image_tools.rst"))
)

CREATING_A_PROJECT_CAPTION = "Creating A Project"
