# coding: utf-8

import os
import os.path
import re
import base64
import importlib.resources

from packaging.version import Version

from cellprofiler import __version__ as cellprofiler_version


def read_content(filename):
    resource_filename = importlib.resources.files("cellprofiler").joinpath(
        "data", "help", filename
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
        importlib.resources.files("cellprofiler").joinpath(
            "data", "images", filename
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

# returns a base64 encoded dataURL of the image pointed to by filename
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs
def image_resource_dataUrl(filename):
    ext = os.path.splitext(filename)[1][1:]
    if ext in ["svg", "xml"]:
        ext = "svg+xml"

    # acceptable image MIME types:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types#image_types
    if ext not in ["apng", "avif", "gif", "jpeg", "png", "svg+xml", "webp"]:
        return None

    media_type = f"image/{ext}"

    with open(image_resource(filename), "rb") as img:
        b64_string = base64.b64encode(img.read())

    data_url = f"data:{media_type};base64,{b64_string.decode('utf-8')}"

    return data_url


MANUAL_URL = "http://cellprofiler-manual.s3.amazonaws.com/CellProfiler-{}/index.html".format(
    Version(cellprofiler_version).base_version
)


####################
#
# ICONS
#
####################
MODULE_HELP_BUTTON = "module_help.png"
MODULE_MOVEUP_BUTTON = "module_moveup.png"
MODULE_MOVEDOWN_BUTTON = "module_movedown.png"
MODULE_ADD_BUTTON = "module_add.png"
MODULE_REMOVE_BUTTON = "module_remove.png"
TESTMODE_PAUSE_ICON = "IMG_PAUSE.png"
TESTMODE_GO_ICON = "IMG_GO.png"
DISPLAYMODE_SHOW_ICON = "eye-open.png"
DISPLAYMODE_HIDE_ICON = "eye-close.png"
SETTINGS_OK_ICON = "check.png"
SETTINGS_ERROR_ICON = "remove-sign.png"
SETTINGS_WARNING_ICON = "IMG_WARN.png"
RUNSTATUS_PAUSE_BUTTON = "status_pause.png"
RUNSTATUS_STOP_BUTTON = "status_stop.png"
RUNSTATUS_SAVE_BUTTON = "status_save.png"
WINDOW_HOME_BUTTON = "window_home.png"
WINDOW_BACK_BUTTON = "window_back.png"
WINDOW_FORWARD_BUTTON = "window_forward.png"
WINDOW_PAN_BUTTON = "window_pan.png"
WINDOW_ZOOMTORECT_BUTTON = "window_zoom_to_rect.png"
WINDOW_SAVE_BUTTON = "window_filesave.png"
ANALYZE_IMAGE_BUTTON = "IMG_ANALYZE_16.png"
INACTIVE_STEP_BUTTON = "IMG_ANALYZED.png"
STOP_ANALYSIS_BUTTON = "IMG_STOP.png"
PAUSE_ANALYSIS_BUTTON = "IMG_PAUSE.png"
INACTIVE_PAUSE_BUTTON = "IMG_GO_DIM.png"


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
