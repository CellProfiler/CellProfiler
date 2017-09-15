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

MEMORY_AND_SPEED_HELP = u"""\
If you find that you are running into out-of-memory errors and/or speed
issues associated with your analysis run, check out a number of
solutions on our forum `FAQ`_ .

.. _FAQ: http://forum.cellprofiler.org
"""

BATCHPROCESSING_HELP = u"""\
CellProfiler is designed to analyze images in a high-throughput manner.
Once a pipeline has been established for a set of images, CellProfiler
can export files that enable batches of images to be analyzed on a
computing cluster with the pipeline.

It is possible to process tens or even hundreds of thousands of images
for one analysis in this manner. We do this by breaking the entire set
of images into separate batches, then submitting each of these batches
as individual jobs to a cluster. Each individual batch can be separately
analyzed from the rest.

The following describes the workflow for running your pipeline on a cluster
that's physically located at your local institution; for running in a cloud-based
cluster using Amazon Web Services, please see our `blog post`_ on Distributed
CellProfiler, a tool designed to streamline that process.

Submitting files for batch processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a basic workflow for submitting your image batches to a
cluster.

#. *Create a folder for your project on your cluster.* For
   high-throughput analysis, it is recommended to create a separate
   project folder for each run.
#. Within this project folder, create the following folders (both of
   which must be connected to the cluster computing network):

   -  Create an input folder, then transfer all of your images to this
      folder as the input folder. The input folder must be readable by
      everyone (or at least your cluster) because each of the separate
      cluster computers will read input files from this folder.
   -  Create an output folder where all your output data will be stored.
      The output folder must be writeable by everyone (or at least your
      cluster) because each of the separate cluster computers will write
      output files to this folder.

   If you cannot create folders and set read/write permissions to these
   folders (or do not know how), ask your Information Technology (IT)
   department for help.
#. Press the “{VIEW_OUTPUT_SETTINGS_BUTTON_NAME}” button. In the
   panel that appears, set the Default Input and Default Output Folders
   to the *images* and *output* folders created above, respectively. The
   Default Input Folder setting will only appear if a legacy pipeline is
   being run.
#. *Create a pipeline for your image set.* You should test it on a few
   example images from your image set (if you are unfamilar with the
   concept of an image set, please see the help for the **Input**
   modules). The module settings selected for your pipeline will be
   applied to *all* your images, but the results may vary depending on
   the image quality, so it is critical to ensure your settings are
   robust against your “worst-case” images.
   For instance, some images may contain no cells. If this happens, the
   automatic thresholding algorithms will incorrectly choose a very low
   threshold, and therefore “find” spurious objects. This can be
   overcome by setting a lower limit on the threshold in the
   **IdentifyPrimaryObjects** module.
   The Test mode in CellProfiler may be used for previewing the results
   of your settings on images of your choice. Please refer to
   *{TEST_MODE_HELP_REF}* for more details on how to use this
   utility.
#. *Add the CreateBatchFiles module to the end of your pipeline.*
   This module is needed to resolve the pathnames to your files with
   respect to your local machine and the cluster computers. If you are
   processing large batches of images, you may also consider adding
   **ExportToDatabase** to your pipeline, after your measurement modules
   but before the CreateBatchFiles module. This module will export your
   data either directly to a MySQL/SQLite database or into a set of
   comma-separated files (CSV) along with a script to import your data
   into a MySQL database. Please refer to the help for these modules in
   order learn more about which settings are appropriate.
#. *Run the pipeline to create a batch file.* Click the *Analyze images*
   button and the analysis will begin processing locally. Do not be
   surprised if this initial step takes a while: CellProfiler must
   first create the entire image set list based on your settings in the
   **Input** modules (this process can be sped up by creating your list
   of images as a CSV and using the **LoadData** module to load it).
   With the **CreateBatchFiles** module in place, the pipeline will not
   process all the images, but instead will create a batch file (a file
   called *Batch\_data.h5*) and save it in the Default Output Folder
   (Step 1). The advantage of using **CreateBatchFiles** from the
   researcher’s perspective is that the Batch\_data.h5 file generated by
   the module captures all of the data needed to run the analysis. You
   are now ready to submit this batch file to the cluster to run each of
   the batches of images on different computers on the cluster.
#. *Submit your batches to the cluster.* Log on to your cluster, and
   navigate to the directory where you have installed CellProfiler on
   the cluster.
   A single batch can be submitted with the following command:

   .. code-block::

      ./python -m cellprofiler -p <Default_Output_Folder_path>/Batch_data.h5 \\
      -c -r -b \\
      -f <first_image_set_number> \\
      -l <last_image_set_number>

   This command submits the batch file to CellProfiler and specifies
   that CellProfiler run in a batch mode without its user interface to
   process the pipeline. This run can be modified by using additional
   options to CellProfiler that specify the following:

   -  ``-p <Default_Output_Folder_path>/Batch_data.h5``: The
      location of the batch file, where ``<Default\_Output\_Folder\_path>``
      is the output folder path as seen by the cluster computer.
   -  ``-c``: Run “headless”, i.e., without the GUI
   -  ``-r``: Run the pipeline specified on startup, which is contained
      in the batch file.
   -  ``-b``: Do not build extensions, since by this point, they should
      already be built.
   -  ``-f <first_image_set_number>``: Start processing with the image
      set specified, <first\_image\_set\_number>
   -  ``-l <last_image_set_number>``: Finish processing with the image
      set specified, <last\_image\_set\_number>

   Typically, a user will break a long image set list into pieces and
   execute each of these pieces using the command line switches, ``-f``
   and ``-l`` to specify the first and last image sets in each job. A
   full image set would then need a script that calls CellProfiler with
   these options with sequential image set numbers, e.g, 1-50, 51-100,
   etc to submit each as an individual job.

   If you need help in producing the batch commands for submitting your
   jobs, use the ``--get-batch-commands`` along with the ``-p`` switch to
   specify the Batch\_data.h5 file output by the CreateBatchFiles module.
   When specified, CellProfiler will output one line to the terminal per
   job to be run. This output should be further processed to generate a
   script that can invoke the jobs in a cluster-computing context.

   The above notes assume that you are running CellProfiler using our
   source code (see “Developer’s Guide” under Help for more details). If
   you are using the compiled version, you would replace
   ``./python -m cellprofiler`` with the CellProfiler executable
   file itself and run it from the installation folder.

Once all the jobs are submitted, the cluster will run each batch
individually and output any measurements or images specified in the
pipeline. Specifying the output filename using the ``-o`` switch when
calling CellProfiler will also produce an output file containing the
measurements for that batch of images in the output folder. Check the
output from the batch processes to make sure all batches complete.
Batches that fail for transient reasons can be resubmitted.

To see a listing and documentation for all available arguments to
CellProfiler, type``cellprofiler --help``.

For additional help on batch processing, refer to our `wiki`_ if
installing CellProfiler on a Unix system, our
`wiki <http://github.com/CellProfiler/CellProfiler/wiki/Adapting-CellProfiler-to-a-LIMS-environment>`__ on adapting CellProfiler to a LIMS
environment, or post your questions on the CellProfiler `CPCluster
forum`_.

.. _wiki: http://github.com/CellProfiler/CellProfiler/wiki/Source-installation-%28Linux%29
.. _CPCluster forum: http://forum.cellprofiler.org/c/cellprofiler/cpcluster-help
.. _blog post: http://blog.cellprofiler.org/2016/12/28/making-it-easier-to-run-image-analysis-in-the-cloud-announcing-distributed-cellprofiler/
""".format(**{
    "TEST_MODE_HELP_REF": TEST_MODE_HELP_REF,
    "VIEW_OUTPUT_SETTINGS_BUTTON_NAME": VIEW_OUTPUT_SETTINGS_BUTTON_NAME
})

RUN_MULTIPLE_PIPELINES_HELP = u"""\
The **Run multiple pipelines** dialog lets you select several
pipelines which will be run consecutively. Please note the following:

-  Pipeline files (.cppipe) are supported.
-  Project files (.cpproj) from CellProfiler 2.1 or newer are not supported.
   To convert your project to a pipeline (.cppipe), select *File > Export > Pipeline…*
   and, under the “Save as type” dropdown, select “CellProfiler pipeline and file list”
   to export the project file list with the pipeline.

You can invoke **Run multiple pipelines** by selecting it from the file menu. The dialog has three parts to it:

-  *File chooser*: The file chooser lets you select the pipeline files
   to be run. The *Select all* and *Deselect all* buttons to the right
   will select or deselect all pipeline files in the list. The *Add*
   button will add the pipelines to the pipeline list. You can add a
   pipeline file multiple times, for instance if you want to run that
   pipeline on more than one input folder.
-  *Directory chooser*: The directory chooser lets you navigate to
   different directories. The file chooser displays all pipeline files
   in the directory chooser’s current directory.
-  *Pipeline list*: The pipeline list has the pipelines to be run in the
   order that they will be run. Each pipeline has a default input and
   output folder and a measurements file. You can change any of these by
   clicking on the file name - an appropriate dialog will then be
   displayed. You can click the remove button to remove a pipeline from
   the list.

CellProfiler will run all of the pipelines on the list when you hit
the “OK” button.
"""

CONFIGURING_LOGGING_HELP = u"""\
CellProfiler prints diagnostic messages to the console by default. You
can change this behavior for most messages by configuring logging. The
simplest way to do this is to use the command-line switch, “-L”, to
set the log level. For instance, to show error messages or more
critical events, start CellProfiler like this:
``CellProfiler -L ERROR``
The following is a list of log levels that can be used:

-  **DEBUG:** Detailed diagnostic information
-  **INFO:** Informational messages that confirm normal progress
-  **WARNING:** Messages that report problems that might need attention
-  **ERROR:** Messages that report unrecoverable errors that result in
   data loss or termination of the current operation.
-  **CRITICAL:** Messages indicating that CellProfiler should be
   restarted or is incapable of running.

You can tailor CellProfiler’s logging with much more control using a
logging configuration file. You specify the file name in place of the
log level on the command line, like this:

``CellProfiler -L ~/CellProfiler/my_log_config.cfg``

Files are in the Microsoft .ini format which is grouped into
categories enclosed in square brackets and the key/value pairs for
each category. Here is an example file:

::

    [loggers]
    keys=root,pipelinestatistics

    [handlers]
    keys=console,logfile

    [formatters]
    keys=detailed

    [logger_root]
    level=WARNING
    handlers=console

    [logger_pipelinestatistics]
    level=INFO
    handlers=logfile
    qualname=pipelineStatistics
    propagate=0

    [handler_console]
    class=StreamHandler
    formatter=detailed
    level=WARNING
    args=(sys.stderr)

    [handler_logfile]
    class=FileHandler
    level=INFO
    args=('~/CellProfiler/logfile.log','w')

    [formatter_detailed]
    format=[%(asctime)s] %(name)s %(levelname)s %(message)s
    datefmt=

The above file would print warnings and errors to the console for all
messages but “pipeline statistics” which are configured using the
*pipelineStatistics* logger are written to a file instead. The
pipelineStatistics logger is the logger that is used to print progress
messages when the pipeline is run. You can find out which loggers are
being used to write particular messages by printing all messages with a
formatter that prints the logger name (“%(name)s”).
The format of the file is described in greater detail `here`_.

.. _here: http://docs.python.org/2.7/howto/logging.html#configuring-logging
"""

ACCESSING_OMERO_IMAGES = u"""\
CellProfiler can load images from `OMERO`_. Please see CellProfiler's
`developer wiki`_ for instructions.

.. _OMERO: http://www.openmicroscopy.org/site/products/omero
.. _developer wiki: http://github.com/CellProfiler/CellProfiler/wiki/OMERO:-Accessing-images-from-CellProfiler

"""

FIGURE_HELP = (
    ("Using The Display Window Menu Bar", read_content("display_menu_bar.rst")),
    ("Using The Interactive Navigation Toolbar", read_content("display_interactive_navigation.rst")),
    ("How To Use The Image Tools", read_content("display_image_tools.rst"))
)

CREATING_A_PROJECT_CAPTION = "Creating A Project"
