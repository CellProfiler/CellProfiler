# coding: utf-8

import os
import os.path
import re

import pkg_resources

import cellprofiler


def read_content(filename):
    resource_filename = pkg_resources.resource_filename(
        "cellprofiler", os.path.join("data", "help", filename)
    )

    with open(resource_filename, "r", encoding="utf-8") as f:
        content = f.read()

    return re.sub(
        r"image:: (.*\.png)",
        lambda md: "image:: {}".format(image_resource(os.path.basename(md.group(0)))),
        content,
    )


def image_resource(filename):
    relpath = os.path.relpath(
        pkg_resources.resource_filename(
            "cellprofiler", os.path.join("data", "images", filename)
        )
    )

    # With this specific relative path we are probably building the documentation
    # in sphinx The path separator used by sphinx is "/" on all platforms.
    if relpath == os.path.join("..", "cellprofiler", "data", "images", filename):
        return "../images/{}".format(filename)

    # Otherwise, if you're rendering in the GUI, relative paths are fine
    # Note: the HTML renderer requires to paths to use '/' so we replace
    # the windows default '\\' here
    return relpath.replace("\\", "/")


MANUAL_URL = "http://cellprofiler-manual.s3.amazonaws.com/CellProfiler-{}/index.html".format(
    cellprofiler.__version__
)

X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
VIEW_OUTPUT_SETTINGS_BUTTON_NAME = "View output settings"


####################
#
# ICONS
#
####################
MODULE_HELP_BUTTON = image_resource("module_help.png")
MODULE_MOVEUP_BUTTON = image_resource("module_moveup.png")
MODULE_MOVEDOWN_BUTTON = image_resource("module_movedown.png")
MODULE_ADD_BUTTON = image_resource("module_add.png")
MODULE_REMOVE_BUTTON = image_resource("module_remove.png")
TESTMODE_PAUSE_ICON = image_resource("IMG_PAUSE.png")
TESTMODE_GO_ICON = image_resource("IMG_GO.png")
DISPLAYMODE_SHOW_ICON = image_resource("eye-open.png")
DISPLAYMODE_HIDE_ICON = image_resource("eye-close.png")
SETTINGS_OK_ICON = image_resource("check.png")
SETTINGS_ERROR_ICON = image_resource("remove-sign.png")
SETTINGS_WARNING_ICON = image_resource("IMG_WARN.png")
RUNSTATUS_PAUSE_BUTTON = image_resource("status_pause.png")
RUNSTATUS_STOP_BUTTON = image_resource("status_stop.png")
RUNSTATUS_SAVE_BUTTON = image_resource("status_save.png")
WINDOW_HOME_BUTTON = image_resource("window_home.png")
WINDOW_BACK_BUTTON = image_resource("window_back.png")
WINDOW_FORWARD_BUTTON = image_resource("window_forward.png")
WINDOW_PAN_BUTTON = image_resource("window_pan.png")
WINDOW_ZOOMTORECT_BUTTON = image_resource("window_zoom_to_rect.png")
WINDOW_SAVE_BUTTON = image_resource("window_filesave.png")
ANALYZE_IMAGE_BUTTON = image_resource("IMG_ANALYZE_16.png")
INACTIVE_STEP_BUTTON = image_resource("IMG_ANALYZED.png")
STOP_ANALYSIS_BUTTON = image_resource("IMG_STOP.png")
PAUSE_ANALYSIS_BUTTON = image_resource("IMG_PAUSE.png")
INACTIVE_PAUSE_BUTTON = image_resource("IMG_GO_DIM.png")


####################
#
# MENU HELP PATHS
#
####################
BATCH_PROCESSING_HELP_REF = """Help > Batch Processing"""
TEST_MODE_HELP_REF = """Help > Testing Your Pipeline"""
IMAGE_TOOLS_HELP_REF = (
    """Help > Using Module Display Windows > How To Use The Image Tools"""
)
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
    (
        "Using The Interactive Navigation Toolbar",
        read_content("display_interactive_navigation.rst"),
    ),
    ("How To Use The Image Tools", read_content("display_image_tools.rst")),
)

CREATING_A_PROJECT_CAPTION = "Creating A Project"
