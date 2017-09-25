# coding: utf-8

import os

import pkg_resources

X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
VIEW_OUTPUT_SETTINGS_BUTTON_NAME = "View output settings"


####################
#
# ICONS
#
####################
def __image_resource(filename):
    #If you're rendering in the GUI, relative paths are fine
    if os.path.relpath(pkg_resources.resource_filename(
        "cellprofiler",
        os.path.join("data", "images", filename)
    )) == os.path.join("cellprofiler","data", "images", filename):
        return os.path.relpath(pkg_resources.resource_filename(
            "cellprofiler",
            os.path.join("data", "images", filename)
        ))
    else:
    #If you're rendering in sphinx, the relative path of the rst file is one below the make file so compensate accordingly
        return os.path.join('..',os.path.relpath(pkg_resources.resource_filename(
            "cellprofiler",
            os.path.join("data", "images", filename)
        )))

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

WHEN_CAN_I_USE_CELLPROFILER_HELP = u"""\
Most laboratories studying biological processes and human disease use
light/fluorescence microscopes to image cells and other biological
samples. There is strong and growing demand for software to analyze
these images, as automated microscopes collect images faster than can be
examined by eye and the information sought from images is increasingly
quantitative and complex.

CellProfiler is a versatile, open-source software tool for quantifying
data from biological images, particularly in high-throughput
experiments. CellProfiler is designed for modular, flexible,
high-throughput analysis of images, measuring size, shape, intensity,
and texture of every cell (or other object) in every image. Using the
point-and-click graphical user interface (GUI), users construct an image
analysis “pipeline”, a sequential series of modules that each perform an
image processing function such as illumination correction, object
identification (segmentation), and object measurement. Users mix and
match modules and adjust their settings to measure the phenotype of
interest. While originally designed for high-throughput images, it is
equally appropriate for low-throughput assays as well (i.e., assays of <
100 images).

CellProfiler can extract valuable biological information from images
quickly while increasing the objectivity and statistical power of
assays. It helps researchers approach a variety of biological questions
quantitatively, including standard assays (e.g., cell count, size,
per-cell protein levels) as well as complex morphological assays (e.g.,
cell/organelle shape or subcellular patterns of DNA or protein
staining).

The wide variety of measurements produced by CellProfiler serves as
useful “raw material” for machine learning algorithms. CellProfiler’s
companion software, CellProfiler Analyst, has an interactive machine
learning tool called Classifier which can learn to recognize a phenotype
of interest based on your guidance. Once you complete the training
phase, CellProfiler Analyst will score every object in your images based
on CellProfiler’s measurements. CellProfiler Analyst also contains tools
for the interactive visualization of the data produced by CellProfiler.

In summary, CellProfiler contains:

-  Advanced algorithms for image analysis that are able to accurately
   identify crowded cells and non-mammalian cell types.
-  A modular, flexible design allowing analysis of new assays and
   phenotypes.
-  Open-source code so the underlying methodology is known and can be
   modified or improved by others.
-  A user-friendly interface.
-  The ability to use high-throughput computing (clusters, cloud).
-  A design that eliminates the tedium of the many steps typically
   involved in image analysis, many of which are not easily transferable
   from one project to another (for example, image formatting, combining
   several image analysis steps, or repeating the analysis with slightly
   different parameters).

References
^^^^^^^^^^

For a full list of references, visit our `citation`_ page.

-  Carpenter AE, Jones TR, Lamprecht MR, Clarke C, Kang IH, Friman O,
   Guertin DA, Chang JH, Lindquist RA, Moffat J, Golland P, Sabatini DM
   (2006) “CellProfiler: image analysis software for identifying and
   quantifying cell phenotypes” *Genome Biology* 7:R100 (`link`_)
-  Kamentsky L, Jones TR, Fraser A, Bray MA, Logan D, Madden K, Ljosa V,
   Rueden C, Harris GB, Eliceiri K, Carpenter AE (2011) “Improved
   structure, function, and compatibility for CellProfiler: modular
   high-throughput image analysis software” *Bioinformatics*
   27(8):1179-1180
   (`link <http://dx.doi.org/10.1093/bioinformatics/btr095>`__)
-  Lamprecht MR, Sabatini DM, Carpenter AE (2007) “CellProfiler: free,
   versatile software for automated biological image analysis”
   *Biotechniques* 42(1):71-75.
   (`link <http://dx.doi.org/10.2144/000112257>`__)
-  Jones TR, Carpenter AE, Lamprecht MR, Moffat J, Silver S, Grenier J,
   Root D, Golland P, Sabatini DM (2009) “Scoring diverse cellular
   morphologies in image-based screens with iterative feedback and
   machine learning” *PNAS* 106(6):1826-1831
   (`link <http://dx.doi.org/10.1073/pnas.0808843106>`__)
-  Jones TR, Kang IH, Wheeler DB, Lindquist RA, Papallo A, Sabatini DM,
   Golland P, Carpenter AE (2008) “CellProfiler Analyst: data
   exploration and analysis software for complex image-based screens”
   *BMC Bioinformatics* 9(1):482
   (`link <http://dx.doi.org/10.1186/1471-2105-9-482>`__)

.. _citation: http://cellprofiler.org/citations/
.. _link: http://dx.doi.org/10.1186/gb-2006-7-10-r100
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

BUILDING_A_PIPELINE_HELP = u"""\
A *pipeline* is a sequential set of image analysis modules. The best way
to learn how to use CellProfiler is to load an example pipeline from the
CellProfiler website’s Examples page and try it with its included images,
then adapt it for
your own images. You can also build a pipeline from scratch. Click the
*Help* |HelpContent_BuildPipeline_image0|  button in the main window to get help for a specific
module.

Loading an existing pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Put the images and pipeline into a folder on your computer.
#. Set the Default Output Folder (press the “View output settings”) to
   the folder where you want to place your output (preferably a
   different location than in the input images).
#. Load the pipeline using *File > Import Pipeline > From File…* in the
   main menu of CellProfiler, or drag and drop it to the pipeline window.
#. Click the *Analyze Images* button to start processing.
#. Examine the measurements using *Data tools*. The *Data tools* options
   are accessible in the main menu of CellProfiler and allow you to
   plot, view, or export your measurements (e.g., to Excel).
#. Alternately, you can load data into CellProfiler Analyst for more
   complex analysis. Please refer to its help for instructions.
#. If you modify the modules or settings in the pipeline, you can save
   the pipeline using *File > Export > Pipeline…*. Alternately, you can
   save the project as a whole using *File > Save Project* or *Save
   Project As…* which also saves the file list, i.e., the list of images.
#. To learn how to use a cluster of computers to process large batches
   of images, see *{BATCH_PROCESSING_HELP_REF}*.

Building a pipeline from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructing a pipeline involves placing individual modules into a
pipeline. The list of modules in the pipeline is shown in the *pipeline
panel* (located on the left-hand side of the CellProfiler window).

#. *Place analysis modules in a new pipeline.*

   Choose image analysis modules to add to your pipeline by clicking the
   *Add* |HelpContent_BuildPipeline_image1| button (located underneath the pipeline panel) or
   right-clicking in the pipeline panel itself and selecting a module
   from the pop-up box that appears.

   You can learn more about each module by clicking *Module Help* in the
   “Add modules” window or the *?* button after the module has been
   placed and selected in the pipeline. Modules are added to the end of
   the pipeline or after the currently selected module, but you can
   adjust their order in the main window by dragging and dropping them,
   or by selecting a module (or modules, using the *Shift* key) and
   using the *Move Module Up* |HelpContent_BuildPipeline_image2| and *Move Module Down*
   |HelpContent_BuildPipeline_image3| buttons. The *Remove Module* |HelpContent_BuildPipeline_image4| button will delete the
   selected module(s) from the pipeline.

   Most pipelines depend on one major step: identifying the objects,
   (otherwise known as “segmentation”). In
   CellProfiler, the objects you identify are called *primary*,
   *secondary*, or *tertiary*:

   -  **IdentifyPrimary** modules identify objects without relying on
      any information other than a single grayscale input image (e.g.,
      nuclei are typically primary objects).
   -  **IdentifySecondaryObjects** modules require a grayscale image
      plus an image where primary objects have already been identified,
      because the secondary objects are determined based on the primary
      objects (e.g., cells can be secondary objects when their
      identification is based on the location of nuclei).
   -  **IdentifyTertiary** modules require images in which two sets of
      objects have already been identified (e.g., nuclei and cell
      regions are used to define cytoplasm objects, which are
      tertiary objects).

#. *Adjust the settings in each module.*

   In the CellProfiler main window, click a module in the pipeline to
   see its settings in the settings panel. To learn more about the
   settings for each module, select the module in the pipeline and
   click the *Help* button to the right of each setting, or at the
   bottom of the pipeline panel for the help for all the settings for
   that module.

   If there is an error with the settings (e.g., a setting refers to an
   image that doesn’t exist yet), a |HelpContent_BuildPipeline_image5| icon will appear next to the
   module name. If there is a warning (e.g., a special notification
   attached to a choice of setting), a |HelpContent_BuildPipeline_image6| icon will appear. Errors
   will cause the pipeline to fail upon running, whereas a warning will
   not. Once the errors/warnings have been resolved, a |HelpContent_BuildPipeline_image7|  icon will
   appear indicating that the module is ready to run.

#. *Set your Default Output Folder and, if necessary, your Default Input Folder*

   Both of these can be set via *File > Preferences…*.  Default Output Folder can
   be additionally changed by clicking the *View output settings* button directly
   below the list of modules in the pipeline; if any modules in your pipeline have
   referenced the Default Input Folder it will also appear in *View output settings*.

#. *Click *Analyze images* to start processing.*

   All of the images in your selected folder(s) will be analyzed using
   the modules and settings you have specified. The bottom of the
   CellProfiler window will show:

   -  A *pause button* |HelpContent_BuildPipeline_image8|  which pauses execution and allows you
      to subsequently resume the analysis.
   -  A *stop button* |HelpContent_BuildPipeline_image9|  which cancels execution after prompting
      you for a place to save the measurements collected to that point.
   -  A *progress bar* which gives the elapsed time and estimates the
      time remaining to process the full image set.

   At the end of each cycle:

   -  If you are creating a MATLAB or HDF5 output file, CellProfiler saves the measurements in the output file.
   -  If you are using the **ExportToDatabase** module, CellProfiler saves the measurements in the
      output database.
   -  If you are using the **ExportToSpreadsheet** module, CellProfiler saves the measurements *into a
      temporary file*; spreadsheets are not written until all modules have been processed.

#. *Click *Start Test Mode* to preview results.*

   You can optimize your pipeline by selecting the *Test* option from
   the main menu. Test mode allows you to run the pipeline on a
   selected image, preview the results, and adjust the module settings
   on the fly. See *{TEST_MODE_HELP_REF}* for more details.

#. Save your project (which includes your pipeline) via *File > Save
   Project*.

*Saving images in your pipeline:* Due to the typically high number of
intermediate images produced during processing, images produced during
processing are not saved to the hard drive unless you specifically
request it, using a **SaveImages** module.

*Saving data in your pipeline:* You can include an **Export** module to
automatically export data in a format you prefer. See
*{USING_YOUR_OUTPUT_REF}* for more details.

.. |HelpContent_BuildPipeline_image0| image:: {MODULE_HELP_BUTTON}
.. |HelpContent_BuildPipeline_image1| image:: {MODULE_ADD_BUTTON}
.. |HelpContent_BuildPipeline_image2| image:: {MODULE_MOVEUP_BUTTON}
.. |HelpContent_BuildPipeline_image3| image:: {MODULE_MOVEDOWN_BUTTON}
.. |HelpContent_BuildPipeline_image4| image:: {MODULE_REMOVE_BUTTON}
.. |HelpContent_BuildPipeline_image5| image:: {SETTINGS_ERROR_ICON}
.. |HelpContent_BuildPipeline_image6| image:: {SETTINGS_WARNING_ICON}
.. |HelpContent_BuildPipeline_image7| image:: {SETTINGS_OK_ICON}
.. |HelpContent_BuildPipeline_image8| image:: {RUNSTATUS_PAUSE_BUTTON}
.. |HelpContent_BuildPipeline_image9| image:: {RUNSTATUS_STOP_BUTTON}
""".format(**{
    "BATCH_PROCESSING_HELP_REF": BATCH_PROCESSING_HELP_REF,
    "MODULE_ADD_BUTTON": MODULE_ADD_BUTTON,
    "MODULE_HELP_BUTTON": MODULE_HELP_BUTTON,
    "MODULE_MOVEDOWN_BUTTON": MODULE_MOVEDOWN_BUTTON,
    "MODULE_MOVEUP_BUTTON": MODULE_MOVEUP_BUTTON,
    "MODULE_REMOVE_BUTTON": MODULE_REMOVE_BUTTON,
    "RUNSTATUS_PAUSE_BUTTON": RUNSTATUS_PAUSE_BUTTON,
    "RUNSTATUS_SAVE_BUTTON": RUNSTATUS_SAVE_BUTTON,
    "RUNSTATUS_STOP_BUTTON": RUNSTATUS_STOP_BUTTON,
    "SETTINGS_ERROR_ICON": SETTINGS_ERROR_ICON,
    "SETTINGS_OK_ICON": SETTINGS_OK_ICON,
    "SETTINGS_WARNING_ICON": SETTINGS_WARNING_ICON,
    "TEST_MODE_HELP_REF": TEST_MODE_HELP_REF,
    "USING_YOUR_OUTPUT_REF": USING_YOUR_OUTPUT_REF
})

SPREADSHEETS_DATABASE_HELP = u"""\
CellProfiler can save measurements as a *spreadsheet* or as a *database*.
Which format you use will depend on some of the considerations below:

-  *Learning curve:* Applications that handle spreadsheets (e.g., Excel,
   `Calc`_ or `Google Docs`_) are easy for beginners to use. Databases
   are more sophisticated and require knowledge of specialized languages
   (e.g., MySQL, Oracle, etc); a popular freeware access tool is
   `SQLyog`_.
-  *Capacity and speed:* Databases are designed to hold larger amounts
   of data than spreadsheets. Spreadsheets may contain a few
   thousand rows of data, whereas databases can hold many millions of
   rows of data. Accessing a particular portion of data in a database
   is optimized for speed.
-  *Downstream application:* If you wish to use Excel or another simple
   tool to analyze your data, a spreadsheet is likely the best choice.  If you
   intend to use CellProfiler Analyst, you must create a database.  If you
   plan to use a scripting language, most languages have ways to import
   data from either format.

.. _Calc: http://www.libreoffice.org/discover/calc/
.. _Google Docs: http://docs.google.com
.. _SQLyog: http://www.webyog.com/
"""

MEMORY_AND_SPEED_HELP = u"""\
If you find that you are running into out-of-memory errors and/or speed
issues associated with your analysis run, check out a number of
solutions on our forum `FAQ`_ .

.. _FAQ: http://forum.cellprofiler.org
"""

TEST_MODE_HELP = u"""\
Before starting an analysis run, you can test the pipeline settings on a
selected image cycle using the *Test* mode option on the main menu. Test
mode allows you to run the pipeline on a selected image, preview the
results and adjust the module settings on the fly.

To enter Test mode once you have built a pipeline, choose *Test > Start
Test Mode* from the menu bar in the main window. At this point, you will
see the following features appear:

-  A Pause icon |HelpContent_TestMode_image0|  will appear to the left of each module.
-  The buttons available at the bottom of the pipeline panel change.

You can run your pipeline in Test mode by selecting *Test > Step to Next
Module* or clicking the *Run* or *Step* buttons at the bottom of the
pipeline panel. The pipeline will execute normally, but you will be able
to back up to a previous module or jump to a downstream module, change
module settings to see the results, or execute the pipeline on the image
of your choice. The additional controls allow you to do the following:

-  *Run from module N:* Start or resume execution of the pipeline at any
   time from a selected module. Right-click the module
   and select "Run from module N", where "N" is the module number.
   This menu option is only available from modules which have already been
   run in test mode, or from the current module. Test mode will run until
   it reaches the end of the pipeline or it encounters a pause.
-  *Pause:* Clicking the pause icon will cause the pipeline test run to
   halt execution when that module is reached (the paused module itself
   is not executed). The icon changes from |HelpContent_TestMode_image1| to |HelpContent_TestMode_image2| to
   indicate that a pause has been inserted at that point.
-  *Run:* Execution of the pipeline will be started/resumed until the
   next module pause is reached. When all modules have been executed for
   a given image cycle, execution will stop.
-  *Step:* Execute the next module (as indicated by the slider
   location).
-  *Next Image:* Skip ahead to the next image cycle as determined by the
   image order in the Input modules. The slider will automatically
   return to the first module in the pipeline.

From the *Test* menu, you can choose additional options:

-  *Exit Test Mode:* Exit *Test* mode. Loading a new pipeline or
   adding/subtracting modules will also automatically exit test mode.
-  *Step to Next Module:* Execute the next module (as indicated by the
   slider location)
-  *Next Image Set:* Step to the next image set in the current image
   group.
-  *Next Image Group:* Step to the next group in the image set. The
   slider will then automatically return to the first module in the
   pipeline.
-  *Random Image Set:* Randomly select and jump to an image set in the
   current image group.
-  *Choose Image Set:* Choose the image set to jump to. The slider will
   then automatically return to the first module in the pipeline.
-  *Choose Image Group:* Choose an image group to jump to. The slider
   will then automatically return to the first module in the pipeline.
-  *Reload Modules Source (enabled only if running from source code):*
   This option will reload the module source code, so any changes to the
   code will be reflected immediately.
-  *Break into debugger (enabled only if running from source code):*
   This option will allow you to open a debugger in the terimal window.

Note that if movies are being loaded, the individual movie is defined as
a group automatically. Selecting *Choose Image Group* will allow you to
choose the movie file, and *Choose Image Set* will let you choose the
individual movie frame from that file.

Please see the **Groups** module for more details on the proper use of
metadata for grouping.

.. |HelpContent_TestMode_image0| image:: {TESTMODE_GO_ICON}
.. |HelpContent_TestMode_image1| image:: {TESTMODE_GO_ICON}
.. |HelpContent_TestMode_image2| image:: {TESTMODE_PAUSE_ICON}
""".format(**{
    "TESTMODE_GO_ICON": TESTMODE_GO_ICON,
    "TESTMODE_PAUSE_ICON": TESTMODE_PAUSE_ICON
})

RUNNING_YOUR_PIPELINE_HELP = u"""\
Once you have tested your pipeline using Test mode and you are satisfied
with the module settings, you are ready to run the pipeline on your
entire set of images. To do this:

-  Exit Test mode by clicking the “Exit Test Mode” button or selecting
   *Test > Exit Test Mode*.
-  Click the "|HelpContent_RunningPipeline_image0| Analyze Images" button and begin processing your
   data sets.

During the analysis run, the progress will appear in the status bar at
the bottom of CellProfiler. It will show you the total number of image
sets, the number of image sets completed, the time elapsed and the
approximate time remaining in the run.

If you need to pause analysis, click the "|HelpContent_RunningPipeline_image1| Pause" button, then
click the “Resume” button to continue. If you want to terminate
analysis, click the "|HelpContent_RunningPipeline_image2| Stop Analysis" button.

If your computer has multiple processors, CellProfiler will take
advantage of them by starting multiple copies of itself to process the
image sets in parallel. You can set the number of *workers* (i.e., copies
of CellProfiler activated) under *File > Preferences…*

.. |HelpContent_RunningPipeline_image0| image:: {ANALYZE_IMAGE_BUTTON}
.. |HelpContent_RunningPipeline_image1| image:: {PAUSE_ANALYSIS_BUTTON}
.. |HelpContent_RunningPipeline_image2| image:: {STOP_ANALYSIS_BUTTON}
""".format(**{
    "ANALYZE_IMAGE_BUTTON": ANALYZE_IMAGE_BUTTON,
    "PAUSE_ANALYSIS_BUTTON": PAUSE_ANALYSIS_BUTTON,
    "STOP_ANALYSIS_BUTTON": STOP_ANALYSIS_BUTTON
})

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

MEASUREMENT_NOMENCLATURE_HELP = u"""\
In CellProfiler, measurements are exported as well as stored internally
using the following general nomenclature:
``MeasurementType_Category_SpecificFeatureName_Parameters``

Below is the description for each of the terms:

-  ``MeasurementType``: The type of data contained in the measurement,
   which can be one of three forms:

   -  *Per-image:* These measurements are image-based (e.g., thresholds,
      counts) and are specified with the name “Image” or with the
      measurement (e.g., “Mean”) for per-object measurements aggregated
      over an image.
   -  *Per-object:* These measurements are per-object and are specified
      as the name given by the user to the identified objects (e.g.,
      “Nuclei” or “Cells”).
   -  *Experiment:* These measurements are produced for a particular
      measurement across the entire analysis run (e.g., Z’ factors), and
      are specified with the name “Experiment”. See
      **CalculateStatistics** for an example.

-  ``Category:`` Typically, this information is specified in one of two
   ways:

   -  A descriptive name indicative of the type of measurement taken
      (e.g., “Intensity”)
   -  No name if there is no appropriate ``Category`` (e.g., if the
      *SpecificFeatureName* is “Count”, no ``Category`` is specfied).

-  ``SpecificFeatureName:`` The specific feature recorded by a module
   (e.g., “Perimeter”). Usually the module recording the measurement
   assigns this name, but a few modules allow the user to type in the
   name of the feature (e.g., the **CalculateMath** module allows the
   user to name the arithmetic measurement).
-  ``Parameters:`` This specifier is to distinguish measurements
   obtained from the same objects but in different ways. For example,
   **MeasureObjectIntensity** can measure intensities for “Nuclei” in
   two different images. This specifier is used primarily for data
   obtained from an individual image channel specified by the **Images**
   module or a legacy **Load** module (e.g., “OrigBlue” and “OrigGreen”)
   or a particular spatial scale (e.g., under the category “Texture” or
   “Neighbors”). Multiple parameters are separated by underscores.

   Below are additional details specific to various modules:

   -  Measurements from the *AreaShape* and *Math* categories do not
      have a ``Parameter`` specifier.
   -  Measurements from *Intensity*, *Granularity*, *Children*,
      *RadialDistribution*, *Parent* and *AreaOccupied* categories will
      have an associated image as the Parameter.
   -  *Measurements from the *Neighbors* and *Texture* category will
      have a spatial scale ``Parameter``.*
   -  Measurements from the *Texture* and *RadialDistribution*
      categories will have both a spatial scale and an image
      ``Parameter``.

As an example, consider a measurement specified as
``Nuclei_Texture_DifferenceVariance_ER_3``:

-  ``MeasurementType`` is “Nuclei,” the name given to the detected
   objects by the user.
-  ``Category`` is “Texture,” indicating that the module
   **MeasureTexture** produced the measurements.
-  ``SpecificFeatureName`` is “DifferenceVariance,” which is one of the
   many texture measurements made by the **MeasureTexture** module.
-  There are two ``Parameters``, the first of which is “ER”. “ER” is the
   user-provided name of the image in which this texture measurement was
   made.
-  The second ``Parameter`` is “3”, which is the spatial scale at which
   this texture measurement was made, according to the user-provided
   settings for the module.

See also the *Available measurements* heading under the main help for
many of the modules, as well as **ExportToSpreadsheet** and
**ExportToDatabase** modules.
"""

MENU_BAR_FILE_HELP = u"""\
The *File* menu provides options for loading and saving your pipelines
and performing an analysis run.

-  **New project:** Clears the current project by removing all the
   analysis modules and resetting the input modules.
-  **Open Project…:** Open a previously saved CellProfiler project
   (*.cpproj* file) from your hard drive.
-  **Open Recent:** Displays a list of the most recent projects used.
   Select any one of these projects to load it.
-  **Save Project:** Save the current project to your hard drive as a
   *.cpproj* file. If it has not been saved previously, you will be
   asked for a file name to give the project. Thereafter, any changes to
   the project will be automatically saved to that filename unless you
   choose **Save as…**.
-  **Save Project As…:** Save the project to a new file name.
-  **Revert to Saved:** Restore the currently open project to the
   settings it had when it was first opened.
-  **Import…:** Gives you the choice of importing a CellProfiler
   pipeline file from your hard drive (*From file…*) or from a web
   address (*From URL…*). In either case, you can import a pipeline
   from a pipeline (*.cppipe*) file or a project (*.cpproj*) file.
   Alternately, you can import an image file list.
-  **Export…:** You have the choice of exporting the pipeline you are
   currently working on as a CellProfiler *.cppipe* pipeline file
   (*Pipeline*), or the image set list as a CSV (*Image set listing*),
   or the pipeline notes as a text file (*Pipeline notes*).
-  **Clear Pipeline:** Removes all modules from the current pipeline,
   while keeping the image file list intact.
-  **View Image:** Opens a dialog box prompting you to select an image
   file for display. Images listed in the File list panel in the
   **Images** module can be also be displayed by double-clicking on the
   filename.
-  **Analyze Images:** Executes the current pipeline using the current
   pipeline and Default Input and Output folder settings.
-  **Stop Analysis:** Stop the current analysis run.
-  **Run Multiple Pipelines:** Execute multiple pipelines in sequential
   order. This option opens a dialog box allowing you to select the
   pipelines you would like to run, as well as the associated input and
   output folders. See the help in the *Run Multiple Pipelines* dialog
   for more details.
-  **Open a New CP Window:** You can run multiple instances of
   CellProfiler simultaneously; this is how you launch a new instance.
-  **Resume Pipeline:** Resume a partially completed analysis run from
   where it left off. You will be prompted to choose the output
   *.h5/.mat* file containing the partially complete measurements and
   the analysis run will pick up starting with the last cycle that was
   processed.
-  **Preferences…:** Displays the Preferences window, where you can
   change many options in CellProfiler.
-  **Quit:** End the current CellProfiler session. You will be given the
   option of saving your current pipeline if you have not done so.
"""

MENU_BAR_EDIT_HELP = u"""\
The *Edit* menu provides options for modifying modules in your current
pipeline.

-  **Undo:** Undo the last module modification. You can undo multiple
   actions by using *Undo* repeatedly.
-  **Cut:** If a module text setting is currently active, remove the
   selected text.
-  **Copy:** Copy the currently selected text to the clipboard.
-  **Paste:** Paste clipboard text to the cursor location, if a text
   setting is active.
-  **Select All:** If a text setting is active, select all the text in
   the setting. If the module list is active, select all the modules in
   the module list.
-  **Move Module Up:** Move the currently selected module(s) up. You can
   also use the |image0| button located below the Pipeline panel.
-  **Move Module Down:** Move the currently selected module(s) down. You
   can also use the |image1| button located below the Pipeline panel.
-  **Delete Module:** Remove the currently selected module(s). Pressing
   the Delete key also removes the module(s). You can also use the
   |image2| button located under the Pipeline panel.
-  **Duplicate Module:** Duplicate the currently selected module(s) in
   the pipeline. The current settings of the selected module(s) are
   retained in the duplicate.
-  **Add Module:** Select a module from the pop-up list to insert into
   the current pipeline. You can also use the |image3| button located
   under the Pipeline panel.
-  **Go to Module:** Select a module from the pop-up list to view the
   settings for that module in the current pipeline. You can also click
   the module in the Pipeline panel.

You can select multiple modules at once for moving, deletion and
duplication by selecting the first module and using Shift-click on the
last module to select all the modules in between.

.. |image0| image:: {MODULE_MOVEUP_BUTTON}
.. |image1| image:: {MODULE_MOVEDOWN_BUTTON}
.. |image2| image:: {MODULE_REMOVE_BUTTON}
.. |image3| image:: {MODULE_ADD_BUTTON}
""".format(**{
    "MODULE_ADD_BUTTON": MODULE_ADD_BUTTON,
    "MODULE_MOVEDOWN_BUTTON": MODULE_MOVEDOWN_BUTTON,
    "MODULE_MOVEUP_BUTTON": MODULE_MOVEUP_BUTTON,
    "MODULE_REMOVE_BUTTON": MODULE_REMOVE_BUTTON
})

MENU_BAR_WINDOW_HELP = u"""\
The *Windows* menu provides options for showing and hiding the module
display windows.

-  **Close All Open Windows:** Closes all display windows that are
   currently open.
-  **Show All Windows On Run:** Select to show all display windows
   during the current test run or next analysis run. The display mode
   icons next to each module in the pipeline panel will switch to
   |image0|.
-  **Hide All Windows On Run:** Select to show no display windows during
   the current test run or next analysis run. The display mode icons
   next to each module in the pipeline panel will switch to |image1|.

If there are any open windows, the window titles are listed underneath
these options. Select any of these window titles to bring that window to
the front.

.. |image0| image:: {DISPLAYMODE_SHOW_ICON}
.. |image1| image:: {DISPLAYMODE_HIDE_ICON}
""".format(**{
    "DISPLAYMODE_HIDE_ICON": DISPLAYMODE_HIDE_ICON,
    "DISPLAYMODE_SHOW_ICON": DISPLAYMODE_SHOW_ICON
})

PARAMETER_SAMPLING_MENU_HELP = u"""\
The *Sampling* menu is an interface for Paramorama, a plugin for an
interactive visualization program for exploring the parameter space of
image analysis algorithms.

This menu option is only shown if specified in the Preferences. Note
that if this preference setting is changed, CellProfiler must be
restarted.

Using this plugin will allow you sample a range of setting values in
**IdentifyPrimaryObjects** and save the object identification results
for later inspection. Upon completion, the plug-in will generate a text
file, which specifies: (1) all unique combinations of the sampled
parameter values; (2) the mapping from each combination of parameter
values to one or more output images; and (3) the actual output images.

**References**

-  Pretorius AJ, Bray MA, Carpenter AE and Ruddle RA. (2011)
   “Visualization of parameter space for image analysis” *IEEE
   Transactions on Visualization and Computer Graphics* 17(12),
   2402-2411.

"""

MENU_BAR_DATATOOLS_HELP = u"""
The *Data Tools* menu provides tools to allow you to plot, view, export
or perform specialized analyses on your measurements.

Each data tool has a corresponding module with the same name and
functionality. The difference between the data tool and the module is
that the data tool takes a CellProfiler output file (i.e., a *.mat or
.h5* file) as input, which contains measurements from a previously
completed analysis run. In contrast, a module uses measurements received
from the upstream modules during an in-progress analysis run.

Opening a data tool will present a prompt in which the user is asked to
provide the location of the output file. Once specified, the user is
then prompted to enter the desired settings. The settings behave
identically as those from the corresponding module.

Please note that with the exception of *PlateViewer* and *Export* functions the 
*Data Tools*, like most CellProfiler modules, are designed to operate on only one image 
set at a time. If you want to use data tool modules to examine and/or 
graph data on the whole experiment level, you should instead consider using 
CellProfiler Analyst; see the *ExportToDatabase* help to learn more about exporting 
your data into a database that CellProfiler Analyst can access and about creating a 
CellProfiler Analyst properties file.  

Help for each *Data Tool* is available under *{DATA_TOOL_HELP_REF}*
or the corresponding module help.
""".format(**{
    "DATA_TOOL_HELP_REF": DATA_TOOL_HELP_REF
})

MODULE_DISPLAY_MENU_BAR_HELP = u"""\
From the menu bar of each module display window, you have the following
options:

-  **File**

   -  *Save:* You can save the figure window to an image file. Note that
      this will save the entire contents of the window, not just the
      individual subplot(s) or images.
   -  *Save table:* This option is only enabled on windows which are
      displaying tabular output, such as that from a **Measure** module.
      This allows you to save the tabular data to a comma-delimited file
      (CSV).

-  **Tools**

   -  *Measure length:* Select this option to measure distances within
      an image window. If you click on an image and drag, a line will
      appear between the two endpoints, and the distance between them
      will be shown at the right-most portion of the bottom panel. This is
      useful for measuring distances in order to obtain estimates of
      typical object diameters for use in **IdentifyPrimaryObjects**.

-  **Subplots:** If the module display window has multiple subplots
   (such as **IdentifyPrimaryObjects**), the Image Tool options for the
   individual subplots are displayed here. See
   *{IMAGE_TOOLS_HELP_REF}* for more details.
""".format(**{
    "IMAGE_TOOLS_HELP_REF": IMAGE_TOOLS_HELP_REF
})

MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP = u"""\
All figure windows come with a navigation toolbar, which can be used to
navigate through the data set.

-  **Home, Forward, Back buttons:** *Home* |image0| always takes you to
   the initial, default view of your data. The *Forward* |image1|  and
   *Back* |image2| buttons are akin to the web browser forward and back
   buttons in that they are used to navigate back and forth between
   previously defined views, one step at a time. They will not be
   enabled unless you have already navigated within an image else using
   the **Pan** and **Zoom** buttons, which are used to define new views.
-  **Pan/Zoom button:** This button has two modes: pan and zoom. Click
   the toolbar button |image3| to activate panning and zooming, then put
   your mouse somewhere over an axes, where it will turn into a hand
   icon.

   -  *Pan:* Press the left mouse button and hold it to pan the figure,
      dragging it to a new position. Press Ctrl+Shift with the pan tool
      to move in one axis only, which one you have moved farther on.
      Keep in mind that that this button will allow you pan outside the
      bounds of the image; if you get lost, you can always use the
      **Home** to back you back to the initial view.
   -  *Zoom:* You can zoom in and out of a plot by pressing Ctrl (Mac)
      or holding down the right mouse button (Windows) while panning.
      Once you’re done, the right click menu will pop up when you’re
      done with the action; dismiss it by clicking off the plot. This is
      a known bug to be corrected in the next release.

-  **Zoom-to-rectangle button:** Click this toolbar button |image4|  to
   activate this mode. To zoom in, press the left mouse button and drag
   in the window to draw a box around the area you want to zoom in on.
   When you release the mouse button, the image is re-drawn to display
   the specified area. Remember that you can always use *Backward*
   button to go back to the previous zoom level, or use the *Home*
   button to reset the window to the initial view.
-  **Save:** Click this button |image5|  to launch a file save dialog.
   You can save the figure window to an image file. Note that this will
   save the entire contents of the window, not just the individual
   subplot(s) or images.

.. |image0| image:: {WINDOW_HOME_BUTTON}
.. |image1| image:: {WINDOW_FORWARD_BUTTON}
.. |image2| image:: {WINDOW_BACK_BUTTON}
.. |image3| image:: {WINDOW_PAN_BUTTON}
.. |image4| image:: {WINDOW_ZOOMTORECT_BUTTON}
.. |image5| image:: {WINDOW_SAVE_BUTTON}
""".format(**{
    "WINDOW_BACK_BUTTON": WINDOW_BACK_BUTTON,
    "WINDOW_FORWARD_BUTTON": WINDOW_FORWARD_BUTTON,
    "WINDOW_HOME_BUTTON": WINDOW_HOME_BUTTON,
    "WINDOW_PAN_BUTTON": WINDOW_PAN_BUTTON,
    "WINDOW_SAVE_BUTTON": WINDOW_SAVE_BUTTON,
    "WINDOW_ZOOMTORECT_BUTTON": WINDOW_ZOOMTORECT_BUTTON
})

MODULE_DISPLAY_IMAGE_TOOLS_HELP = u"""\
Right-clicking in an image displayed in a window will bring up a pop-up
menu with the following options:

-  *Open image in new window:* Displays the image in a new display
   window. This is useful for getting a closer look at a window subplot
   that has a small image.
-  *Show image histogram:* Produces a new window containing a histogram
   of the pixel intensities in the image. This is useful for
   qualitatively examining whether a threshold value determined by
   **IdentifyPrimaryObjects** seems reasonable, for example. Image
   intensities in CellProfiler typically range from zero (dark) to one
   (bright). If you have an RGB image, the histogram shows the intensity
   values for all three channels combined, even if one or more channels
   is turned off for viewing.
-  *Image contrast:* Presents three options for displaying the
   color/intensity values in the images:

      -  *Raw:* Shows the image using the full colormap range permissible for
         the image type. For example, for a 16-bit image, the pixel data will
         be shown using 0 as black and 65535 as white. However, if the actual
         pixel intensities span only a portion of the image intensity range,
         this may render the image unviewable. For example, if a 16-bit image
         only contains 12 bits of data, the resulting image will be entirely
         black.
      -  *Normalized (default):* Shows the image with the colormap
         “autoscaled” to the maximum and minimum pixel intensity values; the
         minimum value is black and the maximum value is white.
      -  *Log normalized:* Same as *Normalized* except that the color values
         are then log transformed. This is useful for when the pixel intensity
         spans a wide range of values but the standard deviation is small
         (e.g., the majority of the interesting information is located at the
         dim values). Using this option increases the effective contrast.

-  *Interpolation:* Presents three options for displaying the resolution
   in the images. This is useful for specifying the amount of detail
   that you want to be visible if you zoom in:

      -  *Nearest neighbor:* Use the intensity of the nearest image pixel when
         displaying screen pixels at sub-pixel resolution. This produces a
         blocky image, but the image accurately reflects the data.
      -  *Linear:* Use the weighted average of the four nearest image pixels
         when displaying screen pixels at sub-pixel resolution. This produces
         a smoother, more visually-appealing image, but makes it more
         difficult to find pixel borders.
      -  *Cubic:* Perform a bicubic interpolation of the nearby image pixels
         when displaying screen pixels at sub-pixel resolution. This produces
         the most visually-appealing image but is the least faithful to the
         image pixel values.

-  *Save subplot:* Save the clicked subplot as an image file. If there
   is only one p lot in the figure, this option will save that one.
-  *Channels:* For color images only. You can show any combination of
   the red, green, and blue color channels.
"""

FIGURE_HELP = (
    ("Using The Display Window Menu Bar", MODULE_DISPLAY_MENU_BAR_HELP),
    ("Using The Interactive Navigation Toolbar", MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP),
    ("How To Use The Image Tools", MODULE_DISPLAY_IMAGE_TOOLS_HELP))

CREATING_A_PROJECT_CAPTION = "Creating A Project"

INTRODUCTION_TO_PROJECTS_HELP = u"""\
What is a project?
~~~~~~~~~~~~~~~~~~

In CellProfiler, a *project* is comprised of two elements:

-  An *image file list* which is the list of files and their locations
   that you select as candidates for analysis.
-  The *pipeline*, which is a series of modules put together to
   analyze a set of images.
-  Optionally, the associated information about the images (*metadata*).
   This information may be part of the images themselves, or you may
   import them from an external file.

The project is the container for image information associated with a
CellProfiler analysis. It stores such details as:

-  What type of image(s) are the input files?
-  Where are the input images located?
-  What distinguishes multiple image channels from each other? How are
   these relationships represented?
-  What information about the images and/or experiment is linked to the
   images, and how?
-  Are certain groups of images to be processed differently from other
   groups?


Working with projects
~~~~~~~~~~~~~~~~~~~~~

Creating a project
^^^^^^^^^^^^^^^^^^

Upon starting CellProfiler, you will be presented with a new, blank
project. At this point, you may start building your project by using the
modules located in the pipeline window on the upper-left. The
modules are:

-  **Images**: Assemble the relevant images for analysis (required).
-  **Metadata**: Associate metadata with the images (optional).
-  **NamesAndTypes**: Assign names to channels and define their
   relationship (required).
-  **Groups**: Define sub-divisions between groups of images for
   processing (optional).

Detailed help for each module is provided by selecting the module and
clicking the “?” button on the bottom of CellProfiler.

Saving a project
^^^^^^^^^^^^^^^^

As you work in CellProfiler, the project is updated automatically, so
there is no need to save it unless you are saving the project to a new
name or location. You can always save your current work to a new project
file by selecting *File > Save Project As…*, which will save your
project, complete with the current image file list and pipeline, to a
file with the extension *.cpproj*.

You also have the option of automatically saving the associated pipeline
file and the file list in addition to the project file. See *File >
Preferences…* for more details.

For those interested, some technical details:

-  The *.cpproj* file stores collected information using the HDF5
   format. Documentation on how measurements are stored and handled in
   CellProfiler using this format can be found `here`_.
-  All information is cached in the project file after it is computed.
   It is either re-computed or retrieved from the cache when an analysis
   run is started, when entering Test mode, or when the user requests a
   refreshed view of the information (e.g., when a setting has been
   changed).

.. _here: http://github.com/CellProfiler/CellProfiler/wiki/Module-structure-and-data-storage-retrieval#HDF5
"""

SELECTING_IMAGES_HELP = u"""\
Any image analysis project using CellProfiler begins with providing the
program with a set of image files to be analyzed. You can do this by
clicking on the **Images** module to select it (located in the Input
modules panel on the left); this module is responsible for collecting
the names and locations of the files to be processed.

The most straightforward way to provide files to the **Images** module
is to simply drag-and-drop them from your file manager tool (e.g.,
Windows Explorer, Mac Finder) onto the file list panel (the blank space
indicated by the text “Drop files and folders here”). Both individual
files and entire folders can be dragged onto this panel, and as many
folders and files can be placed onto this panel as needed. As you add
files, you will see a listing of the files appear in the panel.

CellProfiler supports a wide variety of image formats, including most of
those used in imaging, by using a library called Bio-Formats; see
`here`_ for the formats available. Some image formats are better than
others for image analysis. Some are `“lossy”`_ (information is lost in
the conversion to the format) like most JPG/JPEG files; others are
`lossless`_ (no image information is lost). For image analysis purposes,
a lossless format like TIF or PNG is recommended.

If you have a subset of files that you want to analyze from the full
list shown in the panel, you can also filter the files according to a
set of rules that you specify. This is useful when, for example, you
have dragged a folder of images onto the file list panel, but the folder
contains the images from one experiment that you want to process along
with images from another experiment that you want to ignore for now. You
may specify as many rules as necessary to define the desired list of
images.

For more information on this module and how to configure it for the best
performance, please see the detailed help by selecting the module and
clicking the |image0| button at the bottom of the pipeline panel, or
check out the Input module tutorials on our `Tutorials`_ page.

.. _here: http://docs.openmicroscopy.org/bio-formats/5.7.0/supported-formats.html
.. _“lossy”: http://www.techterms.com/definition/lossy
.. _lossless: http://www.techterms.com/definition/lossless
.. _Tutorials: http://cellprofiler.org/tutorials/

.. |image0| image:: {MODULE_HELP_BUTTON}
""".format(**{
    "MODULE_HELP_BUTTON": MODULE_HELP_BUTTON
})

CONFIGURE_IMAGES_HELP = u"""\
Once you have used the **Images** module to produce a list of images to
be analyzed, you can use the other Input modules to define how images
are related to one another, give them a memorable name for future
reference, attach additional image information about the experiment,
among other things.

After **Images**, you can use the following Input modules:

+---------------------+-------------------------------------------------------------------------+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| \ **Module**\       | \ **Description**\                                                      | \ **Use required?**\    | \ **Usage notes**\                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
+=====================+=========================================================================+=========================+===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| **Metadata**        | Associate image information (metadata) with the images                  | No                      | With this module, you can extract metadata from various sources and append it to the measurements that your pipeline will collect, or use it to define how the images are related to each other. The metadata can come from the image filename or location, or from a spreadsheet that you provide. If your assay does not require or have such information, this module can be skipped.                                                                                                  |
+---------------------+-------------------------------------------------------------------------+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **NamesAndTypes**   | Assign names to images and/or channels and define their relationship.   | Yes                     | This module gives each image a meaningful name by which modules in the analysis pipeline will refer to it. The most common usage for this module is to define a collection of channels that represent a single field of view. By using this module, each of these channels will be loaded and processed together for each field of view.                                                                                                                                                  |
+---------------------+-------------------------------------------------------------------------+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Groups**          | Define sub-divisions between groups of images for processing.           | No                      | For some assays, you will need the option of further sub-dividing an image set into *groups* that share a common feature. An example of this is a time-lapse movie that consists of individual files; each group of files that define a single movie needs to be processed independently of the others. This module allows you to specify what distinguishes one group of images from another. If your assay does not require this sort of behavior, this module can be skipped.          |
+---------------------+-------------------------------------------------------------------------+-------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

For more information on these modules and how to configure them for the
best performance, please see the detailed help by selecting the module
and clicking the |image0| button at the bottom of the pipeline panel, or
check out the Input module tutorials on our `Tutorials`_ page.

.. _Tutorials: http://cellprofiler.org/tutorials.html

.. |image0| image:: {MODULE_HELP_BUTTON}
""".format(**{
    "MODULE_HELP_BUTTON": MODULE_HELP_BUTTON
})

LOADING_IMAGE_SEQUENCES_HELP = u"""\
Introduction
~~~~~~~~~~~~

In this context, the term *image sequence* is used to refer to a
collection of images from a time-lapse assay (movie), a
three-dimensional (3-D) Z-stack assay, or both. This section will teach
you how to load these collections in order to properly represent your
data for processing.

Sequences of individual files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some microscopes, the simplest method of capturing image sequences
is to simply acquire them as a series of individual image files, where
each image file represents a single timepoint and/or Z-slice. Typically,
the image filename reflects the timepoint or Z-slice, such that the
alphabetical image listing corresponds to the proper sequence, e.g.,
*img000.png*, *img001.png*, *img002.png*, etc.

It is also not uncommon to store the movie such that one movie’s worth
of files is stored in a single folder.

*Example:* You have a time-lapse movie of individual files set up as
follows:

-  Three folders, one for each image channel, named *DNA*, *actin* and
   *phase*.
-  In each folder, the files are named as follows:

   -  *DNA*: calibrate2-P01.001.TIF, calibrate2-P01.002.TIF,…,
      calibrate2-P01.287.TIF
   -  *actin*: calibrated-P01.001.TIF, calibrated-P01.002.TIF,…,
      calibrated-P01.287.TIF
   -  *phase*: phase-P01.001.TIF, phase-P01.002.TIF,…, phase-P01.287.TIF

   where the file names are in the format
   *<Stain>-<Well>.<Timepoint>.TIF*.
-  There are 287 timepoints per movie, and a movie of the 3 channels
   above is acquired from each well in a multi-well plate.

In this case, the procedure to set up the input modules to handle these
files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not part of a movie sequence.

   In the above example, you would drag-and-drop the *DNA*, *actin* and
   *phase* folders into the File list panel.

-  In the **Metadata** module, check the box to enable metadata
   extraction. The key step here is to obtain the metadata tags
   necessary to do two things:

   -  Distinguish the movies from each other. This information is
      typically encapsulated in the filename and/or the folder name.
   -  For each movie, distinguish the timepoints from each other and
      ensure their proper ordering. This information is usually
      contained in the filename.

   To accomplish this, do the following:

   -  Select “{X_MANUAL_EXTRACTION}” or “{X_IMPORTED_EXTRACTION}” as
      the metadata extraction method. You will use these to extract the
      movie and timepoint tags from the images.
   -  Use “{X_MANUAL_EXTRACTION}” to create a regular expression to
      extract the metadata from the filename and/or path name.
   -  Or, use “{X_IMPORTED_EXTRACTION}” if you have a comma-delimited
      file (CSV) of the necessary metadata columns (including the movie
      and timepoint tags) for each image. Note that microscopes rarely
      produce such a file, but it might be worthwhile to write scripts
      to create them if you do this frequently.

   If there are multiple channels for each movie, this step may need to
   be performed for each channel.

   In this example, you could do the following:

   -  Select “{X_MANUAL_EXTRACTION}” as the method, “From file name”
      as the source, and
      ``.*-(?P<Well>[A-P][0-9]{{2}})\.(?P<Timepoint>[0-9]{{3}})`` as the
      regular expression. This step will extract the well ID and
      timepoint from each filename.
   -  Click the “Add” button to add another extraction method.
   -  In the new group of extraction settings, select
      “{X_MANUAL_EXTRACTION}” as the method, “From folder name” as the
      source, and ``.*[\\/](?P<Stain>.*)[\\/].*$`` as the regular
      expression. This step will extract the stain name from each folder
      name.
   -  Click the “Update” button below the divider and check the output
      in the table to confirm that the proper metadata values are being
      collected from each image.

-  In the **NamesAndTypes** module, assign the channel(s) to a name of
   your choice. If there are multiple channels, you will need to do this
   for each channel.

   For this example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][actin]`` and
      name it *OrigFluor*.
   -  Click the “Add” button to define another image with a rule.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][DNA]`` and
      name it *OrigFluo2*.
   -  Click the “Add” button to define another image with a rule.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][phase]`` and
      name it *OrigPhase*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “Well” for the *OrigFluor*, *OrigFluo2*, and *OrigPhase*
      channels.
   -  Click the |image0| button to the right to add another row, and
      select “Timepoint” for each channel.
   -  Click the “Update” button below the divider to view the resulting
      table and confirm that the proper files are listed and matched
      across the channels. The corresponding well and frame for each
      channel should now be matched to each other.

-  In the **Groups** module, enable image grouping for these images in
   order to select the metadata that defines a distinct movie of data.

   For the example above, do the following:

   -  Select “Well” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each well is defined as a
      group, each with 287 frames’ worth of images.

   Without this step, CellProfiler would not know where one movie ends
   and the next one begins, and would process the images in all movies
   together as if they were a single movie. This would result in, for
   example, the TrackObjects module attempting to track cells from the
   end of one movie to the start of the next movie.

If your images represent a 3D image, you can follow the above example to
process your data. It is important to note, however, that CellProfiler
will analyze each Z-slice individually and sequentially. Whole volume
(3D image) processing is supported for single-channel .TIF stacks.
Splitting image channels and converting image sets into .TIF stacks can
be done using another software application, like FIJI.

Basic image sequences consisting of a single file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another common means of storing time-lapse or Z-stack data is as a
single file containing frames. Examples of this approach include image
formats such as:

-  Multi-frame TIF
-  Metamorph stack: STK
-  Evotec/PerkinElmer Opera Flex
-  Zeiss ZVI, LSM
-  Standard movie formats: AVI, Quicktime MOV, etc

CellProfiler uses the Bio-Formats library for reading various image
formats. For more details on supported files, see this `webpage`_. In
general, we recommend saving stacks and movies in .TIF format.

*Example:* You have several image stacks representing 3D structures in
the following format:

-  The stacks are saved in .TIF format.
-  Each stack is a single-channel grayscale image.
-  Your files have names like IMG01\_CH01.TIF, IMG01\_CH02.TIF, …
   IMG01\_CH04.TIF and IMG02\_CH01.TIF, IMG02\_CH02.TIF, etc, where
   IMG01\_CH01.TIF designates channel 1 from image 1, IMG01\_CH02.TIF
   designates channel 2 from image 1, and IMG02\_CH01.TIF designates
   channel 1 from image 2.

You would like to process each stack as a single image, not as a series
of 2D images. In this case, the procedure to set up the input modules to
handle these files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the .TIF files into the
   File list panel.
-  In the **NamesAndTypes** module, select “Yes” for “Data is 3D”. You
   should also provide the relative X, Y, and Z pixel sizes of your
   images. X and Y will be determined by the camera and objective you
   used to capture your images. Your Z size represents the spacing of
   your Z-series. In most cases, the X and Y pixel size will be the
   same. You can divide the Z size by X or Y to get a relative value,
   with X = Y = 1. CellProfiler will use this information to correctly
   compute filter sizes and shape features, for example.
   Additionally assign each channel to a name of your choice. You will
   need to do this for each channel. For this example, you could do the
   following:

   -  Select “Assign images matching rules”.
   -  Make a new rule ``[File][Does][Contain][CH01]``
   -  Provide a descriptive name for the channel, e.g., *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule ``[File][Does][Contain][CH02]``
   -  Provide a descriptive name for the channel *GFP*.
   -  Click the “Update” button below the divider to confirm that the
      proper images are listed and matched across the channels. All file
      names ending in CH01.TIF should be matched together.

*Example:* You have two image stacks in the following format:

-  The stacks are Opera’s FLEX format.
-  Each FLEX file contains 8 fields of view, with 3 channels at each
   site (DAPI, GFP, Texas Red).
-  Each channel is in grayscale format.

In this case, the procedure to set up the input modules to handle these
files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the FLEX files into the
   File list panel.
-  In the **Metadata** module, enable metadata extraction in order to
   obtain metadata from these files. The key step here is to obtain the
   necessary metadata tags to do two things:

   -  Distinguish the stacks from each other. This information is
      contained as the file itself, that is, each file represents a
      different stack.
   -  For each stack, distinguish the frames from each other. This
      information is usually contained in the image’s internal metadata,
      in contrast to the image sequence described above.

   To accomplish this, do the following:

   -  Select “{X_AUTOMATIC_EXTRACTION}” as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the “Update metadata” button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while to complete.
   -  Click the “Update” button below the divider.
   -  The resulting table should show the various metadata contained in
      the file. In this case, the relevant information is contained in
      the *C* and *Series* columns. In the figure shown, the *C* column
      shows three unique values for the channels represented, numbered
      from 0 to 2. The *Series* column shows 8 values for the slices
      collected in each stack, numbered from 0 to 7, followed by the
      slices for other stacks.

-  In the **NamesAndTypes** module, assign the channel to a name of your
   choice. If there are multiple channels, you will need to do this for
   each channel. For this example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][0]``
   -  Click the |image1| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``. This combination tells
      CellProfiler not to treat the image as a single file, but rather
      as a series of frames.
   -  Name the image *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][1]``
   -  Click the |image2| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *GFP*.
   -  Click the “Add another image” button to define a third image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][2]``
   -  Click the |image3| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *TxRed*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “FileLocation” for the DAPI, GFP and TxRed channels. The
      FileLocation metadata tag identifies the individual stack, and
      selecting this parameter ensures that the channels are first
      matched within each stack, rather than across stacks.
   -  Click the |image4|  button to the right to add another row, and
      select *Series* for each channel.
   -  Click the “Update” button below the divider to confirm that the
      proper image slices are listed and matched across the channels.
      The corresponding *FileLocation* and *Series* for each channel
      should now be matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select “FileLocation” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the two image
      stacks are defined as a group, each with 8 slices’ worth of
      images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

*Example:* You have four Z-stacks in the following format:

-  The stacks are in Zeiss’ CZI format.
-  Each stack consists of a number of slices with 4 channels (DAPI, GFP,
   Texas Red and Cy3) at each slice.
-  One stack has 9 slices, two stacks have 7 slices and the fourth has
   12 slices. Even though the stacks were collected with differing
   numbers of slices, the pipeline to be constructed is intended to
   analyze all stacks in the same manner.
-  Each slice is in grayscale format.

In this case, the procedure to set up the input modules to handle these
this file is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the CZI files into the
   File list panel. In this case, the default “Images only” filter is
   sufficient to capture the necessary files.
-  In the **Metadata** module, enable metadata extraction in order to
   obtain metadata from these files. The key step here is to obtain the
   metadata tags necessary to do two things:

   -  Distinguish the stacks from each other. This information is
      contained as the file itself, that is, each file represents a
      different stack.
   -  For each stack, distinguish the z-planes from each other, ensuring
      proper ordering. This information is usually contained in the
      image file’s internal metadata.

   To accomplish this, do the following:

   -  Select “{X_AUTOMATIC_EXTRACTION}” as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the “Update metadata” button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while.
   -  Click the “Update” button below the divider.
   -  The resulting table should show the various metadata contained in
      the file. In this case, the relevant information is contained in
      the C and Z columns. The *C* column shows four unique values for
      the channels represented, numbered from 0 to 3. The *Z* column
      shows nine values for the slices represented from the first stack,
      numbered from 0 to 8.
   -  Of note in this case, for each file there is a single row
      summarizing this information. The *sizeC* column reports a value
      of 4 and *sizeZ* column shows a value of 9. You may need to scroll
      down the table to see this summary for the other stacks.

-  In the **NamesAndTypes** module, assign the channel(s) to a name of
   your choice. If there are multiple channels, you will need to do this
   for each channel.

   For the above example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][0]``
   -  Click the |image5| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][1]``
   -  Click the |image6| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the second image *GFP*.
   -  Click the “Add another image” button to define a third image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][2]``.
   -  Click the |image7| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the third image *TxRed*.
   -  Click the “Add another image” button to define a fourth image with
      set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][3]``.
   -  Click the |image8| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the fourth image *Cy3*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “FileLocation” for the *DAPI*,\ *GFP*,\ *TxRed*, and
      *Cy3*\ channels. The *FileLocation* identifies the individual
      stack, and selecting this parameter insures that the channels are
      matched within each stack, rather than across stacks.
   -  Click the |image9| button to the right to add another row, and
      select “Z” for each channel.
   -  Click “Update table” to confirm the channel matching. The
      corresponding *FileLocation* and *Z* for each channel should be
      matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select “FileLocation” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the four image
      stacks are defined as a group, with 9, 7, 7 and 12 slices’ worth
      of images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

.. _webpage: http://docs.openmicroscopy.org/bio-formats/5.6.0/supported-formats.html

.. |image0| image:: {MODULE_ADD_BUTTON}
.. |image1| image:: {MODULE_ADD_BUTTON}
.. |image2| image:: {MODULE_ADD_BUTTON}
.. |image3| image:: {MODULE_ADD_BUTTON}
.. |image4| image:: {MODULE_ADD_BUTTON}
.. |image5| image:: {MODULE_ADD_BUTTON}
.. |image6| image:: {MODULE_ADD_BUTTON}
.. |image7| image:: {MODULE_ADD_BUTTON}
.. |image8| image:: {MODULE_ADD_BUTTON}
.. |image9| image:: {MODULE_ADD_BUTTON}
""".format(**{
    "MODULE_ADD_BUTTON": MODULE_ADD_BUTTON,
    "X_AUTOMATIC_EXTRACTION": X_AUTOMATIC_EXTRACTION,
    "X_IMPORTED_EXTRACTION": X_IMPORTED_EXTRACTION,
    "X_MANUAL_EXTRACTION": X_MANUAL_EXTRACTION
})

PLATEVIEWER_HELP = u"""\
The plate viewer is a data tool that displays the images in your
experiment in plate format. Your project must define an image set list
with metadata annotations for the image’s well and, optionally its plate
and site. The plate viewer will then group your images by well and
display a plate map for you. If you have defined a plate metadata tag
(with the name, “Plate”), the plate viewer will group your images by
plate and display a choice box that lets you pick the plate to display.

Click on a well to see the images for that well. If you have more than
one site per well and have site metadata (with the name, “Site”), the
plate viewer will tile the sites when displaying, and the values under
“X” and “Y” determine the position of each site in the tiled grid.

The values for “Red”, “Green”, and “Blue” in each row are brightness
multipliers- changing the values will determine the color and scaling
used to display each channel. “Alpha” determines the weight each channel
contributes to the summed image.
"""
