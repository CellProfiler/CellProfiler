# coding: utf-8

import os
import re

import pkg_resources


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
LEGACY_LOAD_MODULES_HELP = u"""\
The image loading modules **LoadImages** and **LoadSingleImage** are deprecated
and will be removed in a future version of CellProfiler. It is recommended you
choose to convert these modules as soon as possible. CellProfiler can do this
automatically for you when you import a pipeline using either of these legacy
modules.

Historically, these modules served the same functionality as the current
project structure (via **Images**, **Metadata**, **NamesAndTypes**, and **Groups**).
Pipelines loaded into CellProfiler that contain these modules will provide the option
of preserving them; these pipelines will operate exactly as before.

The section details information relevant for those who would like
to continue using these modules. Please note, however, that these
modules are deprecated and will be removed in a future version of CellProfiler.

Associating metadata with images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata (i.e., additional data about image data) is sometimes available
for input images. This information can be:

#. Used by CellProfiler to group images with common metadata identifiers
   (or “tags”) together for particular steps in a pipeline;
#. Stored in the output file along with CellProfiler-measured features
   for annotation or sample-tracking purposes;
#. Used to name additional input/output files.

Metadata is provided in the image filename or location (pathname). For
example, images produced by an automated microscope can be given
names such as “Experiment1\_A01\_w1\_s1.tif” in which the metadata
about the plate (“Experiment1”), the well (“A01”), the wavelength
number (“w1”) and the imaging site (“s1”) are captured. The name
of the folder in which the images are saved may be meaningful and may
also be considered metadata as well. If this is the case for your
data, use **LoadImages** to extract this information for use in the
pipeline and storage in the output file.

Details for the metadata-specific help is given next to the appropriate
settings in **LoadImages**, as well the specific
settings in other modules which can make use of metadata. However, here
is an overview of how metadata is obtained and used.

In **LoadImages**, metadata can be extracted from the filename and/or
folder location using regular expression, a specialized syntax used for
text pattern-matching. These regular expressions can be used to identify
different parts of the filename / folder. The syntax
*(?P<fieldname>expr)* will extract whatever matches *expr* and assign it
to the image’s *fieldname* measurement. A regular expression tool is
available which will allow you to check the accuracy of your regular
expression.

For instance, say a researcher has folder names with the date and
subfolders containing the images with the run ID (e.g.,
*./2009\_10\_02/1234/*). The following regular expression will capture
the plate, well and site in the fields *Date* and *Run*:
``.\*[\\\\\\/](?P<Date>.\\*)[\\\\\\\\/](?P<Run>.\\*)$``

=============   ============
Subexpression   Explanation
=============   ============
.\\*[\\\\\\\\/]      Skip characters at the beginning of the pathname until either a slash (/) or backslash (\\\\) is encountered (depending on the OS). The extra slash for the backslash is used as an escape sequence.
(?P<Date>       Name the captured field *Date*
.\\*             Capture as many characters that follow
[\\\\\\\\/]         Discard the slash/backslash character
(?P<Run>        Name the captured field *Run*
$               The *Run* field must be at the end of the path string, i.e., the last folder on the path. This also means that the *Date* field contains the parent folder of the *Date* folder.
=============   ============

In **LoadImages**, metadata is extracted from the image *File name*,
*Path* or *Both*. File names or paths containing “Metadata” can be used
to group files loaded by **LoadImages** that are associated with a common
metadata value. The files thus grouped together are then processed as a
distinct image set.

For instance, an experiment might require that images created on the
same day use an illumination correction function calculated from all
images from that day, and furthermore, that the date be captured in the
file names for the individual image sets specifying the illumination
correction functions.

In this case, if the illumination correction images are loaded with the
**LoadImages** module, **LoadImages** should be set to extract the metadata
tag from the file names. The pipeline will then match the individual images
with their corresponding illumination correction functions based on matching
“Metadata\_Date” fields.

Using image grouping
~~~~~~~~~~~~~~~~~~~~

To use grouping, you must define the relevant metadata for each image.
This can be done using regular expressions in **LoadImages**.

To use image grouping in **LoadImages**, please note the following:

-  *Metadata tags must be specified for all images listed.* You cannot
   use grouping unless an appropriate regular expression is defined for
   all the images listed in the module.
-  *Shared metadata tags must be specified with the same name for each
   image listed.* For example, if you are grouping on the basis of a
   metadata tag “Plate” in one image channel, you must also specify the
   “Plate” metadata tag in the regular expression for the other channels
   that you want grouped together.
"""

USING_THE_OUTPUT_FILE_HELP = u"""\
Please note that the output file will be deprecated in the future. This
setting is temporarily present for those needing HDF5 or MATLAB formats,
and will be moved to Export modules in future versions of CellProfiler.

The *output file* is a file where all information about the analysis as
well as any measurements will be stored to the hard drive. **Important
note:** This file does *not* provide the same functionality as the
Export modules. If you want to produce a spreadsheet of measurements
easily readable by Excel or a database viewer (or similar programs),
please refer to the **ExportToSpreadsheet** or **ExportToDatabase**
modules and the associated help.

The options associated with the output file are accessible by pressing
the “View output settings” button at the bottom of the pipeline panel.
In the settings panel to the left, in the *Output Filename* box, you can
specify the name of the output file.

The output file can be written in one of two formats:

-  A *.mat file* which is readable by CellProfiler and by `MATLAB`_
   (Mathworks).
-  An *.h5 file* which is readable by CellProfiler, MATLAB and any other
   program capable of reading the HDF5 data format. Documentation on how
   measurements are stored and handled in CellProfiler using this format
   can be found `here`_.

Results in the output file can also be accessed or exported using **Data
Tools** from the main menu of CellProfiler. The pipeline with its
settings can be be loaded from an output file using *File > Load
Pipeline…*

The output file will be saved in the Default Output Folder unless you
type a full path and file name into the file name box. The path must not
have spaces or characters disallowed by your computer’s platform.

If the output filename ends in *OUT.mat* (the typical text appended to
an output filename), CellProfiler will prevent you from overwriting this
file on a subsequent run by generating a new file name and asking if you
want to use it instead. You can override this behavior by checking the
*Allow overwrite?* box to the right.

For analysis runs that generate a large number of measurements, you may
notice that even though the analysis completes, CellProfiler continues
to use an inordinate amount of your CPU and RAM. This is because the
output file is written after the analysis is completed and can take a
very long time for a lot of measurements. If you do not need this file,
select "*Do not write measurements*" from
the “Measurements file format” drop-down box.

.. _MATLAB: http://www.mathworks.com/products/matlab/
.. _here: http://github.com/CellProfiler/CellProfiler/wiki/Module-structure-and-data-storage-retrieval#HDF5
"""

MATLAB_FORMAT_IMAGES_HELP = u"""\
Previous versions of CellProfiler supported reading and writing of MATLAB
format (.mat) images. MATLAB format images were useful for exporting
illumination correction functions generated by **CorrectIlluminationCalculate**.
These images could be loaded and applied to other pipelines using
**CorrectIlluminationApply**.

This version of CellProfiler no longer supports exporting MATLAB format
images. Instead, the recommended image format for illumination correction
functions is NumPy (.npy). Loading MATLAB format images is deprecated and
will be removed in a future version of CellProfiler. To ensure compatibility
with future versions of CellProfiler you can convert your .mat files to .npy
files via **SaveImages** using this version of CellProfiler.

See **SaveImages** for more details on saving NumPy format images.
"""

FIGURE_HELP = (
    ("Using The Display Window Menu Bar", read_content("display_menu_bar.rst")),
    ("Using The Interactive Navigation Toolbar", read_content("display_interactive_navigation.rst")),
    ("How To Use The Image Tools", read_content("display_image_tools.rst"))
)

CREATING_A_PROJECT_CAPTION = "Creating A Project"
