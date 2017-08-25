# coding=utf-8
""" help.py - contains menu structures for help menus in CP
"""

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

import cellprofiler.icons
import logging
import os
import sys
import cellprofiler.icons
from cellprofiler.setting import YES, NO
import os.path

# from cellprofiler.modules.metadata import X_AUTOMATIC_EXTRACTION, X_MANUAL_EXTRACTION, X_IMPORTED_EXTRACTION
X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
DO_NOT_WRITE_MEASUREMENTS = "Do not write measurements"
VIEW_OUTPUT_SETTINGS_BUTTON_NAME = "View output settings"

logger = logging.getLogger(__name__)

# For some reason, Adobe doesn't like using absolute paths to assemble the PDF.
# Also, Firefox doesn't like displaying the HTML image links using abs paths either.
# So I have use relative ones. Should check this to see if works on the
# compiled version
try:
    path = os.path.relpath(cellprofiler.icons.get_builtin_images_path())
except:
    if any([x == "--html" for x in sys.argv]) and sys.platform.startswith("win"):
        if hasattr(sys, "frozen"):
            drive = sys.argv[0][0]
        else:
            drive = __file__[0][0]
        logger.warning(
                ("Warning: HTML being written with absolute paths. You must\n"
                 "change the current drive to %s: to get image links with\n"
                 "relative paths.\n") % drive)
    path = os.path.abspath(cellprofiler.icons.get_builtin_images_path())

####################################################
#
# Module icon references
#
####################################################

# General help references

REFRESH_BUTTON = 'folder_refresh.png'
BROWSE_BUTTON = 'folder_browse.png'
CREATE_BUTTON = 'folder_create.png'

MODULE_HELP_BUTTON = 'module_help.png'
MODULE_MOVEUP_BUTTON = 'module_moveup.png'
MODULE_MOVEDOWN_BUTTON = 'module_movedown.png'
MODULE_ADD_BUTTON = 'module_add.png'
MODULE_REMOVE_BUTTON = 'module_remove.png'

TESTMODE_PAUSE_ICON = 'IMG_PAUSE.png'
TESTMODE_GO_ICON = 'IMG_GO.png'

DISPLAYMODE_SHOW_ICON = 'IMG_EYE.png'
DISPLAYMODE_HIDE_ICON = 'IMG_CLOSED_EYE.png'

SETTINGS_OK_ICON = 'IMG_OK.png'
SETTINGS_ERROR_ICON = 'IMG_ERROR.png'
SETTINGS_WARNING_ICON = 'IMG_WARN.png'

RUNSTATUS_PAUSE_BUTTON = 'status_pause.png'
RUNSTATUS_STOP_BUTTON = 'status_stop.png'
RUNSTATUS_SAVE_BUTTON = 'status_save.png'

WINDOW_HOME_BUTTON = 'window_home.png'
WINDOW_BACK_BUTTON = 'window_back.png'
WINDOW_FORWARD_BUTTON = 'window_forward.png'
WINDOW_PAN_BUTTON = 'window_pan.png'
WINDOW_ZOOMTORECT_BUTTON = 'window_zoom_to_rect.png'
WINDOW_SAVE_BUTTON = 'window_filesave.png'

ANALYZE_IMAGE_BUTTON = 'IMG_ANALYZE_16.png'
STOP_ANALYSIS_BUTTON = 'IMG_STOP.png'
PAUSE_ANALYSIS_BUTTON = 'IMG_PAUSE.png'

OMERO_IMAGEID_PIC = 'OMERO_imageID_screenshot.png'
OMERO_LOGIN_PIC = 'OMERO_login_screenshot.png'

# Module specific help
PROTIP_RECOMEND_ICON = "thumb-up.png"
PROTIP_AVOID_ICON = "thumb-down.png"
TECH_NOTE_ICON = "gear.png"
IMAGES_FILELIST_BLANK = "Images_FilelistPanel_Blank.png"
IMAGES_FILELIST_FILLED = "Images_FilelistPanel_Filled.png"
MEASUREOBJSIZESHAPE_ECCENTRICITY = 'MeasureObjectSizeShape_Eccentricity.png'
IMAGES_USING_RULES_ICON = 'Images_UsingRules.png'
METADATA_DISPLAY_TABLE = 'Metadata_ExampleDisplayTable.png'
NAMESANDTYPES_DISPLAY_TABLE = 'NamesAndTypes_ExampleDisplayTable.png'
GROUPS_DISPLAY_TABLE = 'Groups_ExampleDisplayTable.png'
EXAMPLE_DAPI_PIC = "dapi.png"
EXAMPLE_GFP_PIC = "gfp.png"

####################################################
#
# Module help specifics for repeated use
#
####################################################

BATCH_PROCESSING_HELP_REF = """Help > Batch Processing"""
TEST_MODE_HELP_REF = """Help > Testing Your Pipeline"""
IMAGE_TOOLS_HELP_REF = """Help > Using Module Display Windows > How To Use The Image Tools"""
DATA_TOOL_HELP_REF = """Help > Data Tool Help"""
PROJECT_INTRO_HELP = """Help > Creating a Project > Introduction to Projects"""
USING_YOUR_OUTPUT_REF = """Help > Using Your Output"""
MEASUREMENT_NAMING_HELP = """Help > Using Your Output >How Measurements are Named"""
USING_METADATA_HELP_REF = """Please see the **Metadata** module for more details on metadata collection and usage"""
LOADING_IMAGE_SEQ_HELP_REF = """Help > Creating a Project > Loading Image Stacks and Movies"""
USING_METADATA_TAGS_REF = """ You can insert a previously defined metadata tag by either using:

-  The insert key
-  A right mouse button click inside the control
-  In Windows, the Context menu key, which is between the Windows key
   and Ctrl key

The inserted metadata tag will appear in green. To change a previously
inserted metadata tag, navigate the cursor to just before the tag and
either:

-  Use the up and down arrows to cycle through possible values.
-  Right-click on the tag to display and select the available values.

| """

USING_METADATA_GROUPING_HELP_REF = """Please see the   **Groups** module for more details on the proper use of metadata for
  grouping"""

from cellprofiler.setting import YES, NO

RETAINING_OUTLINES_HELP = """Select *%(YES)s* to retain the outlines
  of the new objects for later use in the pipeline. For example, a
  common use is for quality control purposes by overlaying them on your
  image of choice using the **OverlayOutlines** module and then saving
  the overlay image with the **SaveImages** module.""" % locals()

NAMING_OUTLINES_HELP = """ *(Used only if the outline image is to be
  retained for later use in the pipeline)*
| Enter a name for the outlines of the identified objects. The outlined
  image can be selected in downstream modules by selecting them from any
  drop-down image list."""

################################################## # # Help for the main window # ##################################################

LEGACY_LOAD_MODULES_HELP = """

Historically, two modules served the same functionality as the current
project structure: **LoadImages** and **LoadData**. While the approach
described above supersedes these modules in part, old pipelines loaded
into CellProfiler that contain these modules will provide the option of
preserving them; these pipelines will operate exactly as before.

Alternately, the user can choose to convert these modules into the
project equivalent as closely as possible. Both modules remain accesible
via the "Add module" and |image0|  button at the bottom of the pipeline
panel. The section details information relevant for users who would like
to continue using these modules. Please note, however, that these
modules are deprecated and may be removed in the future.

Associating metadata with images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata (i.e., additional data about image data) is sometimes available
for input images. This information can be:

#. Used by CellProfiler to group images with common metadata identifiers
   (or "tags") together for particular steps in a pipeline;
#. Stored in the output file along with CellProfiler-measured features
   for annotation or sample-tracking purposes;
#. Used to name additional input/output files.

Two sources of metadata are:

-  *Metadata provided in the image filename or location (pathname).* For
   example, images produced by an automated microscope can be given
   names similar to "Experiment1\_A01\_w1\_s1.tif" in which the metadata
   about the plate ("Experiment1"), the well ("A01"), the wavelength
   number ("w1") and the imaging site ("s1") are encapsulated. The name
   of the folder in which the images are saved may be meaningful and may
   also be considered metadata as well. If this is the case for your
   data, use **LoadImages** to extract this information for use in the
   pipeline and storage in the output file.
-  *Metadata provided as a table of information*. Often, information
   associated with each image (such as treatment, plate, well, etc) is
   available as a separate spreadsheet. If this is the case for your
   data, use **LoadData** to load this information.

Details for the metadata-specific help is given next to the appropriate
settings in **LoadImages** and **LoadData**, as well the specific
settings in other modules which can make use of metadata. However, here
is an overview of how metadata is obtained and used.

In **LoadImages**, metadata can be extracted from the filename and/or
folder location using regular expression, a specialized syntax used for
text pattern-matching. These regular expressions can be used to identify
different parts of the filename / folder. The syntax
*(?P<fieldname>expr)* will extract whatever matches *expr* and assign it
to the image's *fieldname* measurement. A regular expression tool is
available which will allow you to check the accuracy of your regular
expression.

| For instance, say a researcher has folder names with the date and
  subfolders containing the images with the run ID (e.g.,
  *./2009\_10\_02/1234/*). The following regular expression will capture
  the plate, well and site in the fields *Date* and *Run*:

.\*[\\\\\\/](?P<Date>.\*)[\\\\\\\\/](?P<Run>.\*)$

.\*[\\\\\\\\/]

Skip characters at the beginning of the pathname until either a slash
(/) or backslash (\\\\) is encountered (depending on the OS). The extra
slash for the backslash is used as an escape sequence.

(?P<Date>

Name the captured field *Date*

.\*

Capture as many characters that follow

[\\\\\\\\/]

Discard the slash/backslash character

(?P<Run>

Name the captured field *Run*

.\*

Capture as many characters as follow

$

The *Run* field must be at the end of the path string, i.e. the last
folder on the path. This also means that the *Date* field contains the
parent folder of the *Date* folder.

In **LoadData**, metadata is extracted from a CSV (comma-separated) file
(a spreadsheet). Columns whose name begins with "Metadata" can be used
to group files loaded by **LoadData** that are associated with a common
metadata value. The files thus grouped together are then processed as a
distinct image set.

For instance, an experiment might require that images created on the
same day use an illumination correction function calculated from all
images from that day, and furthermore, that the date be captured in the
file names for the individual image sets and in a CSV file specifying
the illumination correction functions.

In this case, if the illumination correction images are loaded with the
**LoadData** module, the file should have a "Metadata\_Date" column
which contains the date metadata tags. Similarly, if the individual
images are loaded using the **LoadImages** module, **LoadImages** should
be set to extract the metadata tag from the file names. The pipeline
will then match the individual images with their corresponding
illumination correction functions based on matching "Metadata\_Date"
fields.

Using image grouping
~~~~~~~~~~~~~~~~~~~~

To use grouping, you must define the relevant metadata for each image.
This can be done using regular expressions in **LoadImages** or having
them pre-defined in a CSV file for use in **LoadData**.

To use image grouping in **LoadImages**, please note the following:

-  *Metadata tags must be specified for all images listed.* You cannot
   use grouping unless an appropriate regular expression is defined for
   all the images listed in the module.
-  *Shared metadata tags must be specified with the same name for each
   image listed.* For example, if you are grouping on the basis of a
   metadata tag "Plate" in one image channel, you must also specify the
   "Plate" metadata tag in the regular expression for the other channels
   that you want grouped together.

.. |image0| image:: memory:%(MODULE_ADD_BUTTON)s

""" % globals()

DEFAULT_IMAGE_FOLDER_HELP = """

Please note that the Default Input Folder will be deprecated in the
future. The location of non-image files needed by some modules will be
set to an absolute path in future versions of CellProfiler. For
specifying the location of image files, please use the *Input modules*
panel starting with the **Images** module.

The *Default Input Folder* is enabled only if a legacy pipeline is
loaded into CellProfiler and is accessible by pressing the "View output
settings" button at the bottom of the pipeline panel. The folder
designated as the *Default Input Folder* contains the input image or
data files that you want to analyze. Several File Processing modules
(e.g., **LoadImages** or **LoadData**) provide the option of retrieving
images from this folder on a default basis unless you specify, within
the module, an alternate, specific folder on your computer. Within
modules, we recommend selecting the Default Input Folder as much as
possible, so that your pipeline will work even if you transfer your
images and pipeline to a different computer. If, instead, you type
specific folder path names into a module's settings, your pipeline will
not work on someone else's computer until you adjust those pathnames
within each module.

Use the *Browse* button |image1| to specify the folder you would like to
use as the Default Input Folder, or type the full folder path in the
edit box. If you type a folder path that cannot be found, the message
box below will indicate this fact until you correct the problem. If you
want to specify a folder that does not yet exist, type the desired name
and click on the *New folder* button |image2|. The folder will be
created according to the pathname you have typed.

.. |image1| image:: memory:%(BROWSE_BUTTON)s
.. |image2| image:: memory:%(CREATE_BUTTON)s
""" % globals()

DEFAULT_OUTPUT_FOLDER_HELP = """

Please note that the Default Output Folder will be deprecated in the
future. The location of files written by the various output modules will
be set to an absolute path in future versions of CellProfiler.

The *Default Output Folder* is accessible by pressing the "View output
settings" button at the bottom of the pipeline panel. The Default Output
Folder is the folder that CellProfiler uses to store the output file it
creates. Also, several File Processing modules (e.g., **SaveImages** or
**ExportToSpreadsheet**) provide the option of saving analysis results
to this folder on a default basis unless you specify, within the module,
an alternate, specific folder on your computer. Within modules, we
recommend selecting the Default Output Folder as much as possible, so
that your pipeline will work even if you transfer your images and
pipeline to a different computer. If, instead, you type specific folder
path names into a module's settings, your pipeline will not work on
someone else's computer until you adjust those pathnames within each
module.

Use the *Browse* button (to the right of the text box) to specify the
folder you would like to use as the Default Output Folder, or type the
full folder path in the edit box. If you type a folder path that cannot
be found, the message box below will indicate this fact until you
correct the problem. If you want to specify a folder that does not yet
exist, type the desired name and click on the *New folder* icon to the
right of the *Browse folder* icon. The folder will be created according
to the pathname you have typed.

"""

USING_THE_OUTPUT_FILE_HELP = """

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
the "View output settings" button at the bottom of the pipeline panel.
In the settings panel to the left, in the *Output Filename* box, you can
specify the name of the output file.

The output file can be written in one of two formats:

-  A *.mat file* which is readable by CellProfiler and by
   `MATLAB <http://www.mathworks.com/products/matlab/>`__ (Mathworks).
-  An *.h5 file* which is readable by CellProfiler, MATLAB and any other
   program capable of reading the HDF5 data format. Documentation on how
   measurements are stored and handled in CellProfiler using this format
   can be found
   `here <https://github.com/CellProfiler/CellProfiler/wiki/Module-Structure-and-Data-Storage-Retrieval#hdf5-measurement-and-workspace-format>`__.

Results in the output file can also be accessed or exported using **Data
Tools** from the main menu of CellProfiler. The pipeline with its
settings can be be loaded from an output file using *File > Load
Pipeline...*

The output file will be saved in the Default Output Folder unless you
type a full path and file name into the file name box. The path must not
have spaces or characters disallowed by your computer's platform.

If the output filename ends in *OUT.mat* (the typical text appended to
an output filename), CellProfiler will prevent you from overwriting this
file on a subsequent run by generating a new file name and asking if you
want to use it instead. You can override this behavior by checking the
*Allow overwrite?* box to the right.

For analysis runs that generate a large number of measurements, you may
notice that even though the analysis completes, CellProfiler continues
to use an inordinate amount of your CPU and RAM. This is because the
output file is written after the analysis is completed and can take a
very long time for a lot of measurements. If you do not need this file
and/or notice this behavior, select "*{DO_NOT_WRITE_MEASUREMENTS}*"
from the "Measurements file format" drop-down box.

""".format(**{
    "DO_NOT_WRITE_MEASUREMENTS": DO_NOT_WRITE_MEASUREMENTS
})

WHEN_CAN_I_USE_CELLPROFILER_HELP = """

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
analysis "pipeline", a sequential series of modules that each perform an
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
useful "raw material" for machine learning algorithms. CellProfiler's
companion software, CellProfiler Analyst, has an interactive machine
learning tool called Classifier which can learn to recognize a phenotype
of interest based on your guidance. Once you complete the training
phase, CellProfiler Analyst will score every object in your images based
on CellProfiler's measurements. CellProfiler Analyst also contains tools
for the interactive visualization of the data produced by CellProfiler.

In summary, CellProfiler contains:

-  Advanced algorithms for image analysis that are able to accurately
   identify crowded cells and non-mammalian cell types.
-  A modular, flexible design allowing analysis of new assays and
   phenotypes.
-  Open-source code so the underlying methodology is known and can be
   modified or improved by others.
-  A user-friendly interface.
-  The capability to make use of clusters of computers when available.
-  A design that eliminates the tedium of the many steps typically
   involved in image analysis, many of which are not easily transferable
   from one project to another (for example, image formatting, combining
   several image analysis steps, or repeating the analysis with slightly
   different parameters).

References
''''''''''

For a full list of references, visit our
`citation <http://www.cellprofiler.org/citations.html>`__ page.

-  Carpenter AE, Jones TR, Lamprecht MR, Clarke C, Kang IH, Friman O,
   Guertin DA, Chang JH, Lindquist RA, Moffat J, Golland P, Sabatini DM
   (2006) "CellProfiler: image analysis software for identifying and
   quantifying cell phenotypes" *Genome Biology* 7:R100
   (`link <http://dx.doi.org/10.1186/gb-2006-7-10-r100>`__)
-  Kamentsky L, Jones TR, Fraser A, Bray MA, Logan D, Madden K, Ljosa V,
   Rueden C, Harris GB, Eliceiri K, Carpenter AE (2011) "Improved
   structure, function, and compatibility for CellProfiler: modular
   high-throughput image analysis software" *Bioinformatics*
   27(8):1179-1180
   (`link <http://dx.doi.org/10.1093/bioinformatics/btr095>`__)
-  Lamprecht MR, Sabatini DM, Carpenter AE (2007) "CellProfiler: free,
   versatile software for automated biological image analysis"
   *Biotechniques* 42(1):71-75.
   [`link <http://dx.doi.org/10.2144/000112257>`__)
-  Jones TR, Carpenter AE, Lamprecht MR, Moffat J, Silver S, Grenier J,
   Root D, Golland P, Sabatini DM (2009) "Scoring diverse cellular
   morphologies in image-based screens with iterative feedback and
   machine learning" *PNAS* 106(6):1826-1831
   (`link <http://dx.doi.org/10.1073/pnas.0808843106>`__)
-  Jones TR, Kang IH, Wheeler DB, Lindquist RA, Papallo A, Sabatini DM,
   Golland P, Carpenter AE (2008) "CellProfiler Analyst: data
   exploration and analysis software for complex image-based screens"
   *BMC Bioinformatics* 9(1):482
   (`link <http://dx.doi.org/10.1186/1471-2105-9-482>`__)

"""

BUILDING_A_PIPELINE_HELP = """

A *pipeline* is a sequential set of image analysis modules. The best way
to learn how to use CellProfiler is to load an example pipeline from the
CellProfiler website's Examples page and try it out, then adapt it for
your own images. You can also build a pipeline from scratch. Click the
*Help* |image3|  button in the main window to get help for a specific
module.

Loading an existing pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Put the images and pipeline into a folder on your computer.
#. Set the Default Output Folder (press the "View output settings") to
   the folder where you want to place your output (preferably a
   different location than in the input images).
#. Load the pipeline using *File > Import Pipeline > From File...* in
   the main menu of CellProfiler.
#. Click the *Analyze Images* button to start processing.
#. Examine the measurements using *Data tools*. The *Data tools* options
   are accessible in the main menu of CellProfiler and allow you to
   plot, view, or export your measurements (e.g., to Excel).
#. If you modify the modules or settings in the pipeline, you can save
   the pipeline using *File > Export > Pipeline...*. Alternately, you
   can save the project as a whole using *File > Save Project* or *Save
   Project As...* which also saves the file list.
#. To learn how to use a cluster of computers to process large batches
   of images, see *{BATCH_PROCESSING_HELP_REF}*.

Building a pipeline from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructing a pipeline involves placing individual modules into a
pipeline. The list of modules in the pipeline is shown in the *pipeline
panel* (located on the left-hand side of the CellProfiler window).

#. | *Place analysis modules in a new pipeline.*

   Choose image analysis modules to add to your pipeline by clicking the
   *Add* |image4| button (located underneath the pipeline panel) or
   right-clicking in the pipeline panel itself and selecting a module
   from the pop-up box that appears.

   You can learn more about each module by clicking *Module Help* in the
   "Add modules" window or the *?* button after the module has been
   placed and selected in the pipeline. Modules are added to the end of
   the pipeline or after the currently selected module, but you can
   adjust their order in the main window by dragging and dropping them,
   or by selecting a module (or modules, using the *Shift* key) and
   using the *Move Module Up* |image5| and *Move Module Down*
   |image6| buttons. The *Remove Module* |image7| button will delete the
   selected module(s) from the pipeline.

   Most pipelines depend on one major step: identifying the objects. In
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
      regions are used to define the cytoplasm objects, which are
      tertiary objects).

#. | *Adjust the settings in each module.*
   | In the CellProfiler main window, click a module in the pipeline to
     see its settings in the settings panel. To learn more about the
     settings for each module, select the module in the pipeline and
     click the *Help* button to the right of each setting, or at the
     bottom of the pipeline panel for the help for all the settings for
     that module.

   If there is an error with the settings (e.g., a setting refers to an
   image that doesn't exist yet), a |image8| icon will appear next to
   the module name. If there is a warning (e.g., a special notification
   attached to a choice of setting), a |image9| icon will appear. Errors
   will cause the pipeline to fail upon running, whereas a warning will
   not. Once the errors/warnings have been resolved, a |image10|  icon
   will appear indicating that the module is ready to run.

#. | *Set your Default Input Folder, Default Output Folder and output
     filename.*
   | For more help, click their nearby *Help* buttons in the main
     window.

#. | *Click *Analyze images* to start processing.*
   | All of the images in your selected folder(s) will be analyzed using
     the modules and settings you have specified. A status window will
     appear which has the following:

   -  A *progress bar* which gives the elapsed time and estimates the
      time remaining to process the full image set.
   -  A *pause button* |image11|  which pauses execution and allows you
      to subsequently resume the analysis.
   -  A *stop button* |image12|  which cancels execution after prompting
      you for a place to save the measurements collected to that point.
   -  A *save measurements* button |image13|  which will prompt you for
      a place to save the measurements collected to that point while
      continuing the analysis run.

   At the end of each cycle, CellProfiler saves the measurements in the
   output file.

#. | *Click *Start Test Mode* to preview results.*
   | You can optimize your pipeline by selecting the *Test* option from
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

.. |image3| image:: memory:%(MODULE_HELP_BUTTON)s
.. |image4| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image5| image:: memory:%(MODULE_MOVEUP_BUTTON)s
.. |image6| image:: memory:%(MODULE_MOVEDOWN_BUTTON)s
.. |image7| image:: memory:%(MODULE_REMOVE_BUTTON)s
.. |image8| image:: memory:%(SETTINGS_ERROR_ICON)s
.. |image9| image:: memory:%(SETTINGS_WARNING_ICON)s
.. |image10| image:: memory:%(SETTINGS_OK_ICON)s
.. |image11| image:: memory:%(RUNSTATUS_PAUSE_BUTTON)s
.. |image12| image:: memory:%(RUNSTATUS_STOP_BUTTON)s
.. |image13| image:: memory:%(RUNSTATUS_SAVE_BUTTON)s
""".format(**{
    "BATCH_PROCESSING_HELP_REF": BATCH_PROCESSING_HELP_REF,
    "TEST_MODE_HELP_REF": TEST_MODE_HELP_REF,
    "USING_YOUR_OUTPUT_REF": USING_YOUR_OUTPUT_REF
})

REGEXP_HELP_REF = """ Patterns are specified using
combinations of metacharacters and literal characters. There are a few
classes of metacharacters, partially listed below. Some helpful links
follow:

-  A more extensive explanation of regular expressions can be found
   `here <http://www.python.org/doc/2.3/lib/re-syntax.html>`__
-  A helpful quick reference can be found
   `here <http://www.addedbytes.com/cheat-sheets/regular-expressions-cheat-sheet/>`__
-  `Pythex <http://pythex.org/>`__ provides quick way to test your
   regular expressions. Here is an
   `example <http://pythex.org/?regex=Channel%5B1-2%5D-%5B0-9%5D(2)-(%3FP%3CWellRow%3E%5BA-H%5D)-(%3FP%3CWellColumn%3E%5B0-9%5D(2))%5C.tif&test_string=Channel1-01-A-01.tif&ignorecase=0&multiline=0&dotall=0&verbose=0>`__
   to capture information from a common microscope nomenclature.

| The following metacharacters match exactly one character from its
  respective set of characters:

+---------------------+---------------------------------------------------+
| **Metacharacter**   | **Meaning**                                       |
+=====================+===================================================+
| .                   | Any character                                     |
+---------------------+---------------------------------------------------+
| []                  | Any character contained within the brackets       |
+---------------------+---------------------------------------------------+
| [^]                 | Any character not contained within the brackets   |
+---------------------+---------------------------------------------------+
| \\w                 | A word character [a-z\_A-Z0-9]                    |
+---------------------+---------------------------------------------------+
| \\W                 | Not a word character [^a-z\_A-Z0-9]               |
+---------------------+---------------------------------------------------+
| \\d                 | A digit [0-9]                                     |
+---------------------+---------------------------------------------------+
| \\D                 | Not a digit [^0-9]                                |
+---------------------+---------------------------------------------------+
| \\s                 | Whitespace [ \\\\t\\\\r\\\\n\\\\f\\\\v]           |
+---------------------+---------------------------------------------------+
| \\S                 | Not whitespace [^ \\\\t\\\\r\\\\n\\\\f\\\\v]      |
+---------------------+---------------------------------------------------+

| The following metacharacters are used to logically group
  subexpressions or to specify context for a position in the match.
  These metacharacters do not match any characters in the string:

+---------------------+----------------------------------------------+
| **Metacharacter**   | **Meaning**                                  |
+=====================+==============================================+
| ( )                 | Group subexpression                          |
+---------------------+----------------------------------------------+
| \|                  | Match subexpression before or after the \|   |
+---------------------+----------------------------------------------+
| ^                   | Match expression at the start of string      |
+---------------------+----------------------------------------------+
| $                   | Match expression at the end of string        |
+---------------------+----------------------------------------------+
| \\<                 | Match expression at the start of a word      |
+---------------------+----------------------------------------------+
| \\>                 | Match expression at the end of a word        |
+---------------------+----------------------------------------------+

| The following metacharacters specify the number of times the previous
  metacharacter or grouped subexpression may be matched:

+---------------------+-------------------------------------+
| **Metacharacter**   | **Meaning**                         |
+=====================+=====================================+
| \*                  | Match zero or more occurrences      |
+---------------------+-------------------------------------+
| +                   | Match one or more occurrences       |
+---------------------+-------------------------------------+
| ?                   | Match zero or one occurrence        |
+---------------------+-------------------------------------+
| {n,m}               | Match between n and m occurrences   |
+---------------------+-------------------------------------+

Characters that are not special metacharacters are all treated literally
in a match. To match a character that is a special metacharacter, escape
that character with a '\\\\'. For example '.' matches any character, so
to match a '.' specifically, use '\\.' in your pattern. Examples:

-  ``[trm]ail`` matches 'tail' or 'rail' or 'mail'.
-  ``[0-9]`` matches any digit between 0 to 9.
-  ``[^Q-S]`` matches any character other than 'Q' or 'R' or 'S'.
-  ``[[]A-Z]`` matches any upper case alphabet along with square
   brackets.
-  ``[ag-i-9]`` matches characters 'a' or 'g' or 'h' or 'i' or '-' or
   '9'.
-  ``[a-p]*`` matches '' or 'a' or 'aab' or 'p' etc.
-  ``[a-p]+`` matches 'a' or 'abc' or 'p' etc.
-  ``[^0-9]`` matches any string that is not a number.
-  ``^[0-9]*$`` matches either a blank string or a natural number.
-  ``^-[0-9]+$|^\+?[0-9]+$`` matches any integer.

"""

SPREADSHEETS_DATABASE_HELP = """

The most common form of output for cellular analysis is a *spreadsheet*,
which is an application which tabulates data values. CellProfiler can
also output data into a *database*, which is a collection of data that
is stored for retrieval by users. Which format you use will depend on
some of the considerations below:

-  *Assessibility:* Spreadsheet applications are typically designed to
   allow easy user interaction with the data, to edit values, make
   graphs and the like. In contrast, the values in databases are
   typically not modified after the fact. Instead, database applications
   typically allow for viewing a specific data range.
-  *Capacity and speed:* Databases are designed to hold larger amounts
   of data than spreadsheets. Spreadsheets may contain hundreds to a few
   thousand rows of data, whereas databases can hold mnay millions of
   rows of data. Due to the high capacity, accessing a particular
   portion of data in a database is optimized for speed.
-  *Learning curve:* The applications that access spreadsheets are
   usually made for ease-of-use to allow for user edits. Databases are
   more sophisticated and are not typically edited or modified; to do so
   require knowledge of specialized languages made for this purpose
   (e.g., MySQL, Oracle, etc).

For spreadsheets, the most widely used program to open these files is
Microsoft's Excel program. Since the file is plain text, other editors
can also be used, such as
`Calc <http://www.libreoffice.org/features/calc/>`__ or `Google
Docs <https://docs.google.com>`__. For databases, a popular freeware
access tool is `SQLyog <https://www.webyog.com/>`__.

"""

MEMORY_AND_SPEED_HELP = """

If you find that you are running into out-of-memory errors and/or speed
issues associated with your analysis run, we have detailed a number of
solutions on our forum
`FAQ <http://cellprofiler.org/forum/viewtopic.php?f=14&t=806&p=4490#p4490>`__
on this issue. We will continue to add more tips and tricks to this page
over time.

""" % globals()

TEST_MODE_HELP = """

Before starting an analysis run, you can test the pipeline settings on a
selected image cycle using the *Test* mode option on the main menu. Test
mode allows you to run the pipeline on a selected image, preview the
results and adjust the module settings on the fly.

To enter Test mode once you have built a pipeline, choose *Test > Start
Test Mode* from the menu bar in the main window. At this point, you will
see the following features appear:

-  The module view will have a slider bar appearing on the far left.
-  A Pause icon |image14|  will appear to the left of each module.
-  A series of buttons will appear at the bottom of the pipeline panel
   above the module adjustment buttons.
-  The grayed-out items in the *Test* menu will become active, and the
   *Analyze Images* button will become inactive.

You can run your pipeline in Test mode by selecting *Test > Step to Next
Module* or clicking the *Run* or *Step* buttons at the bottom of the
pipeline panel. The pipeline will execute normally, but you will be able
to back up to a previous module or jump to a downstream module, change
module settings to see the results, or execute the pipeline on the image
of your choice. The additional controls allow you to do the following:

-  *Slider:* Start/resume execution of the pipeline at any time by
   moving the slider. However, if the selected module depends on objects
   and/or images generated by prior modules, you will see an error
   message indicating that the data has not been produced yet. To avoid
   this, it is best to actually run the pipeline up to the module of
   interest, and move the slider to modules already executed.
-  *Pause:* Clicking the pause icon will cause the pipeline test run to
   halt execution when that module is reached (the paused module itself
   is not executed). The icon changes from |image15| to |image16| to
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

Note that if movies are being loaded, the individual movie is defined as
a group automatically. Selecting *Choose Image Group* will allow you to
choose the movie file, and *Choose Image Set* will let you choose the
individual movie frame from that file.

{USING_METADATA_GROUPING_HELP_REF}

.. |image14| image:: memory:%(TESTMODE_GO_ICON)s
.. |image15| image:: memory:%(TESTMODE_GO_ICON)s
.. |image16| image:: memory:%(TESTMODE_PAUSE_ICON)s
""".format(**{
    "USING_METADATA_GROUPING_HELP_REF": USING_METADATA_GROUPING_HELP_REF
})


RUNNING_YOUR_PIPELINE_HELP = """ Once you have tested
your pipeline using Test mode and you are satisfied with the module
settings, you are ready to run the pipeline on your entire set of
images. To do this:

-  Exit Test mode by clicking the "Exit Test Mode" button or selecting
   *Test > Exit Test Mode*.
-  Click the "|image17| Analyze Images" button and begin processing your
   data sets.

During the analysis run, the progress will appear in the status bar at
the bottom of CellProfiler. It will show you the total number of image
sets, the number of image sets completed, the time elapsed and the
approximate time remaining in the run.

If you need to pause analysis, click the "|image18| Pause" button, then
click the "Resume" button to continue. If you want to terminate
analysis, click the "|image19| Stop Analysis" button.

If your computer has multiple processors, CellProfiler will take
advantage of them by starting multiple copies of itself to process the
image sets in parallel. You can set the number of *workers* (i.e.,copies
of CellProfiler activated) under *File > Preferences...*

.. |image17| image:: memory:%(ANALYZE_IMAGE_BUTTON)s
.. |image18| image:: memory:%(PAUSE_ANALYSIS_BUTTON)s
.. |image19| image:: memory:%(STOP_ANALYSIS_BUTTON)s
""" % globals()


# The help below contains a Google URL shortener since the URL has a control character that the URL reader doesn't interpretcorrectly
BATCHPROCESSING_HELP = """

CellProfiler is designed to analyze images in a high-throughput manner.
Once a pipeline has been established for a set of images, CellProfiler
can export batches of images to be analyzed on a computing cluster with
the pipeline.

It is possible to process tens or even hundreds of thousands of images
for one analysis in this manner. We do this by breaking the entire set
of images into separate batches, then submitting each of these batches
as individual jobs to a cluster. Each individual batch can be separately
analyzed from the rest.

Submitting files for batch processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a basic workflow for submitting your image batches to the
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
   folders (or don't know how), ask your Information Technology (IT)
   department for help.
#. Press the "{VIEW_OUTPUT_SETTINGS_BUTTON_NAME}" button. In the
   panel that appears, set the Default Input and Default Output Folders
   to the *images* and *output* folders created above, respectively. The
   Default Input Folder setting will only appear if a legacy pipeline is
   being run.
#. *Create a pipeline for your image set.* You should test it on a few
   example images from your image set (if you are unfamilar with the
   concept of an image set, please see the help for the **Input**
   modules). The module settings selected for your pipeline will be
   applied to *all* your images, but the results may vary depending on
   the image quality, so it is critical to insure that your settings be
   robust against your "worst-case" images.
   For instance, some images may contain no cells. If this happens, the
   automatic thresholding algorithms will incorrectly choose a very low
   threshold, and therefore "find" spurious objects. This can be
   overcome by setting a lower limit on the threshold in the
   **IdentifyPrimaryObjects** module.
   The Test mode in CellProfiler may be used for previewing the results
   of your settings on images of your choice. Please refer to
   *%(TEST\_MODE\_HELP\_REF)s* for more details on how to use this
   utility.
#. *Add the **CreateBatchFiles** module to the end of your pipeline.*
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
   surprised if this initial step takes a while since CellProfiler must
   first create the entire image set list based on your settings in the
   **Input** modules (this process can be sped up by creating your list
   of images as a CSV and using the **LoadData** module to load it).
   With the **CreateBatchFiles** module in place, the pipeline will not
   process all the images, but instead will creates a batch file (a file
   called *Batch\_data.h5*) and save it in the Default Output Folder
   (Step 1). The advantage of using **CreateBatchFiles** from the
   researcher's perspective is that the Batch\_data.h5 file generated by
   the module captures all of the data needed to run the analysis. You
   are now ready to submit this batch file to the cluster to run each of
   the batches of images on different computers on the cluster.
#. | *Submit your batches to the cluster.* Log on to your cluster, and
     navigate to the directory where you have installed CellProfiler on
     the cluster.
   | A single batch can be submitted with the following command:
   | ``./python CellProfiler.py -p <Default_Output_Folder_path>/Batch_data.h5 -c -r -b -f <first_image_set_number> -l <last_image_set_number>``
     This command submits the batch file to CellProfiler and specifies
     that CellProfiler run in a batch mode without its user interface to
     process the pipeline. This run can be modified by using additional
     options to CellProfiler that specify the following:

   -  ``-p <Default_Output_Folder_path>/Batch_data.h5``: The location of
      the batch file, where <Default\_Output\_Folder\_path> is the
      output folder path as seen by the cluster computer.
   -  ``-c``: Run "headless", i.e., without the GUI
   -  ``-r``: Run the pipeline specified on startup, which is contained
      in the batch file.
   -  ``-b``: Do not build extensions, since by this point, they should
      already be built.
   -  ``-f <first_image_set_number>``: Start processing with the image
      set specified, <first\_image\_set\_number>
   -  ``-l <last_image_set_number>`` : Finish processing with the image
      set specified, <last\_image\_set\_number>

   | Typically, a user will break a long image set list into pieces and
     execute each of these pieces using the command line switches,
     ``-f`` and ``-l`` to specify the first and last image sets in each
     job. A full image set would then need a script that calls
     CellProfiler with these options with sequential image set numbers,
     e.g, 1-50, 51-100, etc to submit each as an individual job.

   | If you need help in producing the batch commands for submitting
     your jobs, use the ``--get-batch-commands`` along with the ``-p``
     switch to specify the Batch\_data.h5 file output by the
     CreateBatchFiles module. When specified, CellProfiler will output
     one line to the terminal per job to be run. This output should be
     further processed to generate a script that can invoke the jobs in
     a cluster-computing context.
   | The above notes assume that you are running CellProfiler using our
     source code (see "Developer's Guide" under Help for more details).
     If you are using the compiled version, you would replace
     ``./python CellProfiler.py`` with the CellProfiler executable file
     itself and run it from the installation folder.

Once all the jobs are submitted, the cluster will run each batch
individually and output any measurements or images specified in the
pipeline. Specifying the output filename using the ``-o`` switch when
calling CellProfiler will also produce an output file containing the
measurements for that batch of images in the output folder. Check the
output from the batch processes to make sure all batches complete.
Batches that fail for transient reasons can be resubmitted.

To see documentation for all available arguments to CellProfiler, type
``CellProfiler.py --help`` to see a listing.

For additional help on batch processing, refer to our
`wiki <http://goo.gl/HtTzD>`__ if installing CellProfiler on a Unix
system, our `wiki <http://goo.gl/WG9doZ>`__ on adapting CellProfiler to
a LIMS environment, or post your questions on the CellProfiler
`CPCluster forum <http://cellprofiler.org/forum/viewforum.php?f=18>`__.
""".format(**{
    "VIEW_OUTPUT_SETTINGS_BUTTON_NAME": VIEW_OUTPUT_SETTINGS_BUTTON_NAME
})

RUN_MULTIPLE_PIPELINES_HELP = """
| The **Run multiple pipelines** dialog lets you select several
  pipelines which will be run consecutively. Please note the following:

-  CellProfiler 2.1 project files are not currently supported.
-  Pipelines from CellProfiler 2.0 and lower are supported.
-  If you want to use a pipeline made using CellProfiler 2.1, then you
   need to include the project file list with the pipeline, by selecting
   *Export > Pipeline...*, and under the "Save as type" dropdown, select
   "CellProfiler pipeline and file list".

| You can invoke **Run multiple pipelines** by selecting it from the
  file menu. The dialog has three parts to it:

-  *File chooser*: The file chooser lets you select the pipeline files
   to be run. The *Select all* and *Deselect all* buttons to the right
   will select or deselect all pipeline files in the list. The *Add*
   button will add the pipelines to the pipeline list. You can add a
   pipeline file multiple times, for instance if you want to run that
   pipeline on more than one input folder.
-  *Directory chooser*: The directory chooser lets you navigate to
   different directories. The file chooser displays all pipeline files
   in the directory chooser's current directory.
-  *Pipeline list*: The pipeline list has the pipelines to be run in the
   order that they will be run. Each pipeline has a default input and
   output folder and a measurements file. You can change any of these by
   clicking on the file name - an appropriate dialog will then be
   displayed. You can click the remove button to remove a pipeline from
   the list

|
| CellProfiler will run all of the pipelines on the list when you hit
  the "OK" button."""

CONFIGURING_LOGGING_HELP = """CellProfiler
  prints diagnostic messages to the console by default. You can change
  this behavior for most messages by configuring logging. The simplest
  way to do this is to use the command-line switch, "-L", to set the log
  level. For instance, to show error messages or more critical events,
  start CellProfiler like this:
| ``CellProfiler -L ERROR``
| The following is a list of log levels that can be used:

-  **DEBUG:** Detailed diagnostic information
-  **INFO:** Informational messages that confirm normal progress
-  **WARNING:** Messages that report problems that might need attention
-  **ERROR:** Messages that report unrecoverable errors that result in
   data loss or termination of the current operation.
-  **CRITICAL:** Messages indicating that CellProfiler should be
   restarted or is incapable of running.

|
| You can tailor CellProfiler's logging with much more control using a
  logging configuration file. You specify the file name in place of the
  log level on the command line, like this:
| ``CellProfiler -L ~/CellProfiler/my_log_config.cfg``
| Files are in the Microsoft .ini format which is grouped into
  categories enclosed in square brackets and the key/value pairs for
  each category. Here is an example file:

.. raw:: html

   <div>

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

.. raw:: html

   </div>

| The above file would print warnings and errors to the console for all
  messages but "pipeline statistics" which are configured using the
  *pipelineStatistics* logger are written to a file instead.. The
  pipelineStatistics logger is the logger that is used to print progress
  messages when the pipeline is run. You can find out which loggers are
  being used to write particular messages by printing all messages with
  a formatter that prints the logger name ("%(name)s").
| The format of the file is described in greater detail
  `here <http://docs.python.org/howto/logging.html#configuring-logging>`__.
  """

ACCESSING_OMERO_IMAGES = """ CellProfiler has first-class
  support for loading images from
  `OMERO <http://www.openmicroscopy.org/site/products/omero>`__. The
  input modules and the LoadData module can refer to images by URL, for
  instance, the example pipeline on the welcome page loads its images
  from ``http://cellprofiler.org/ExampleFlyImages``. The first part of a
  URL (the part before the colon) is the schema. CellProfiler decides
  which communication protocol to use, depending on the schema; in the
  case of the example on the welcome page, the schema is HTTP and
  CellProfiler uses the HTTP protocol to get the image data. For OMERO,
  the schema that should be used is "omero" and we use the OMERO client
  library to fetch and load the data.

| OMERO URLs have the form, "omero:iid=". You can find the image IDs
  using either the OMERO web client or the `Insight
  software <http://www.openmicroscopy.org/site/support/omero4/downloads>`__.
  As an example, the screen capture below indicates that the image,
  "Channel1-01-A-01.tif", has an IID of 58038:
| |image20|

At present, manually curating the URL list can be somewhat
time-consuming, but we are planning to develop plug-ins for Insight that
will automatically generate these lists for CellProfiler from within the
Insight user interface. The plugin will allow you to select a screen or
plate and export an image set list that can be used with CellProfiler's
LoadData module.

OMERO login credentials
~~~~~~~~~~~~~~~~~~~~~~~

| CellProfiler will ask you for your OMERO login credentials when you
  first access an OMERO URL, either by viewing it from the file list or
  by loading it in a pipeline. CellProfiler will create and maintain a
  session for you based on these credentials until you close the
  application. You should only need to enter your credentials once. To
  use the "Log into Omero" dialog, enter your server's name or IP
  address, the port (usually 4064), your user name and password and
  press the "Connect" button. The "Connect" button should turn green and
  the OK button of the dialog should become enabled (see below). Press
  OK to complete the login.
| |image21|

Currently, CellProfiler cannot establish a connection to OMERO when
running headless - to do that, we would need to store the user password
where it might be otherwise visible. We would like to provide a secure
mechanism for establishing a session when headless and would like to
work with you to make this work in your environment; please contact us
for further information on how to modify CellProfiler yourself to do
this or with suggestions for us to implement.

Using OMERO URLs with the Input modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Images** module has a file list panel of all of the image files in
a project. This file list supports URLs including OMERO URLs. You can
drag URLs from a text document and drop them into the file list. The
URLs do not end with image file extensions (like .TIF), so you need to
change the "Filter images?" setting to "No filtering" to allow the OMERO
URLs to be processed further. You should be able to view the image by
double-clicking on it and you should be able to extract plate, well,
site and channel name metadata from each image using the "Extract from
image file headers" method in the **Metadata** module (press the "Update
Metadata" button after selecting the "Extract from image file headers"
method). If your experiment has more than one image channel, you can use
the "ChannelName" metadata extracted from the OMERO image to create
image sets containing all of your image channels. In the
**NamesAndTypes** module, change the "Assign a name to" setting to
"Images matching rules". For the rule criteria, select "Metadata does
have ChannelName matching" and enter the name that appears under
"Channels" in the OMERO Insight browser. Add additional channels to
**NamesAndTypes** using the "Add another image" button.

OMERO URLs and LoadData
~~~~~~~~~~~~~~~~~~~~~~~

The LoadData module reads image sets from a .CSV file. The CSV file has
a one-line header that tells LoadData how to use each of the columns of
the file. You can load channels from a URL by adding a "URL" tag to this
header. The OMERO URLs themselves appear in rows below. For instance,
here is a .CSV that loads a DNA and GFP channel:

::

    URL_DNA,URL_GFP
    omero:iid=58134,omero:iid=58038
    omero:iid=58135,omero:iid=58039
    omero:iid=58136,omero:iid=58040

.. |image20| image:: memory:%(OMERO_IMAGEID_PIC)s
.. |image21| image:: memory:%(OMERO_LOGIN_PIC)s
""" % globals()

MEASUREMENT_NOMENCLATURE_HELP = """ In CellProfiler,
measurements are exported as well as stored internally using the
following general nomenclature:
``MeasurementType_Category_SpecificFeatureName_Parameters``

Below is the description for each of the terms:

-  ``MeasurementType``: The type of data contained in the measurement,
   which can be one of three forms:

   -  *Per-image:* These measurements are image-based (e.g., thresholds,
      counts) and are specified with the name "Image" or with the
      measurement (e.g., "Mean") for per-object measurements aggregated
      over an image.
   -  *Per-object:* These measurements are per-object and are specified
      as the name given by the user to the identified objects (e.g.,
      "Nuclei" or "Cells")
   -  *Experiment:* These measurements are produced for a particular
      measurement across the entire analysis run (e.g., Z'-factors), and
      are specified with the name "Experiment". See
      **CalculateStatistics** for an example.

-  ``Category:`` Typically, this information is specified in one of two
   ways

   -  A descriptive name indicative of the type of measurement taken
      (e.g., "Intensity")
   -  No name if there is no appropriate ``Category`` (e.g., if the
      *SpecificFeatureName* is "Count", no ``Category`` is specfied).

-  ``SpecificFeatureName:`` The specific feature recorded by a module
   (e.g., "Perimeter"). Usually the module recording the measurement
   assigns this name, but a few modules allow the user to type in the
   name of the feature (e.g., the **CalculateMath** module allows the
   user to name the arithmetic measurement).
-  ``Parameters:`` This specifier is to distinguish measurements
   obtained from the same objects but in different ways. For example,
   **MeasureObjectIntensity** can measure intensities for "Nuclei" in
   two different images. This specifier is used primarily for data
   obtained from an individual image channel specified by the **Images**
   module or a legacy **Load** module (e.g., "OrigBlue" and "OrigGreen")
   or a particular spatial scale (e.g., under the category "Texture" or
   "Neighbors"). Multiple parameters are separated by underscores.

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

-  ``MeasurementType`` is "Nuclei," the name given to the detected
   objects by the user.
-  ``Category`` is "Texture," indicating that the module
   **MeasureTexture** produced the measurements.
-  ``SpecificFeatureName`` is "DifferenceVariance," which is one of the
   many texture measurements made by the **MeasureTexture** module.
-  There are two ``Parameters``, the first of which is "ER". "ER" is the
   user-provided name of the image in which this texture measurement was
   made.
-  The second ``Parameter`` is "3", which is the spatial scale at which
   this texture measurement was made.

See also the *Available measurements* heading under the main help for
many of the modules, as well as **ExportToSpreadsheet** and
**ExportToDatabase** modules. """

MENU_BAR_FILE_HELP = """ The *File*
menu provides options for loading and saving your pipelines and
performing an analysis run.

-  **New project:** Clears the current project by removing all the
   analysis modules and resetting the input modules.
-  **Open Project...:** Open a previously saved CellProfiler project
   (*.cpproj* file) from your hard drive.
-  **Open Recent:** Displays a list of the most recent projects used.
   Select any one of these projects to load it.
-  **Save Project:** Save the current project to your hard drive as a
   *.cpproj* file. If it has not been saved previously, you will be
   asked for a file name to give the project. Thereafter, any changes to
   the project will be automatically saved to that filename unless you
   choose **Save as...**.
-  **Save Project As...:** Save the project to a new file name.
-  **Revert to Saved:** Restore the currently open project to the
   settings it had when it was first opened.
-  **Import Pipeline:** Gives you the choice of importing a CellProfiler
   pipeline file from your hard drive (*From file...*) or from a web
   address (*From URL...*). If importing from a file, you can point it
   to a pipeline (*.cppipe*) file or have it extract the pipeline from a
   project (*.cpproj*) file.
-  **Export:** You have the choice of exporting the pipeline you are
   currently working on as a CellProfiler *.cppipe* pipeline file
   (*Pipeline*), or the image set list as a CSV (*Image set listing*).
-  **Clear Pipeline:** Removes all modules from the current pipeline.
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
-  **Resume Pipeline:** Resume a partially completed analysis run from
   where it left off. You will be prompted to choose the output
   *.h5/.mat* file containing the partially complete measurements and
   the analysis run will pick up starting with the last cycle that was
   processed.
-  **Preferences...:** Displays the Preferences window, where you can
   change many options in CellProfiler.
-  **Exit:** End the current CellProfiler session. You will be given the
   option of saving your current pipeline if you have not done so.

"""

MENU_BAR_EDIT_HELP = """ The *Edit* menu provides options for
modifying modules in your current pipeline.

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
   also use the |image22| button located below the Pipeline panel.
-  **Move Module Down:** Move the currently selected module(s) down. You
   can also use the |image23| button located below the Pipeline panel.
-  **Delete Module:** Remove the currently selected module(s). Pressing
   the Delete key also removes the module(s). You can also use the
   |image24| button located under the Pipeline panel.
-  **Duplicate Module:** Duplicate the currently selected module(s) in
   the pipeline. The current settings of the selected module(s) are
   retained in the duplicate.
-  **Add Module:** Select a module from the pop-up list to inster into
   the current pipeline. You can also use the |image25| button located
   under the Pipeline panel.

You can select multiple modules at once for moving, deletion and
duplication by selecting the first module and using Shift-click on the
last module to select all the modules in between.

.. |image22| image:: memory:%(MODULE_MOVEUP_BUTTON)s
.. |image23| image:: memory:%(MODULE_MOVEDOWN_BUTTON)s
.. |image24| image:: memory:%(MODULE_REMOVE_BUTTON)s
.. |image25| image:: memory:%(MODULE_ADD_BUTTON)s
""" % globals()

MENU_BAR_WINDOW_HELP = """ The *Windows* menu provides options for
showing and hiding the module display windows.

-  **Close All Open Windows:** Closes all display windows that are
   currently open.
-  **Show All Windows On Run:** Select to show all display windows
   during the current test run or next analysis run. The display mode
   icons next to each module in the pipeline panel will switch to
   |image26|.
-  **Hide All Windows On Run:** Select to show no display windows during
   the current test run or next analysis run. The display mode icons
   next to each module in the pipeline panel will switch to |image27|.

If there are any open windows, the window titles are listed underneath
these options. Select any of these window titles to bring that window to
the front.

.. |image26| image:: memory:%(DISPLAYMODE_SHOW_ICON)s
.. |image27| image:: memory:%(DISPLAYMODE_HIDE_ICON)s
""" % globals()

PARAMETER_SAMPLING_MENU_HELP = """ The
*Sampling* menu is an interface for Paramorama, a plugin for an
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

More information on how to use the plugin can be found
`here <http://www.comp.leeds.ac.uk/scsajp/applications/paramorama2/>`__.

**References**

-  Pretorius AJ, Bray MA, Carpenter AE and Ruddle RA. (2011)
   "Visualization of parameter space for image analysis" *IEEE
   Transactions on Visualization and Computer Graphics* 17(12),
   2402-2411.

""" #consider deprecating

MENU_BAR_DATATOOLS_HELP = """ The *Data
Tools* menu provides tools to allow you to plot, view, export or perform
specialized analyses on your measurements.

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

Help for each *Data Tool* is available under *{DATA_TOOL_HELP_REF}*
or the corresponding module help.
""".format(**{
    "DATA_TOOL_HELP_REF": DATA_TOOL_HELP_REF
})


#################################################### # #Help for the module figure windows#####################################################
'''The help menu for the figure window'''

MODULE_DISPLAY_MENU_BAR_HELP = """ From the
menu bar of each module display window, you have the following options:

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
      shown at the right-most portion of the bottom panel. This is
      useful for measuring distances in order to obtain estimates of
      typical object diameters for use in **IdentifyPrimaryObjects**.

-  **Subplots:** If the module display window has multiple subplots
   (such as **IdentifyPrimaryObjects**), the Image Tool options for the
   individual subplots are displayed here. See
   *{IMAGE_TOOLS_HELP_REF}* for more details.
""".format(**{
    "IMAGE_TOOLS_HELP_REF": IMAGE_TOOLS_HELP_REF
})

MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP = """ All
figure windows come with a navigation toolbar, which can be used to
navigate through the data set.

-  **Home, Forward, Back buttons:** *Home* |image28| always takes you to
   the initial, default view of your data. The *Forward* |image29|  and
   *Back* |image30| buttons are akin to the web browser forward and back
   buttons in that they are used to navigate back and forth between
   previously defined views, one step at a time. They will not be
   enabled unless you have already navigated within an image else using
   the **Pan** and **Zoom** buttons, which are used to define new views.
-  **Pan/Zoom button:** This button has two modes: pan and zoom. Click
   the toolbar button |image31| to activate panning and zooming, then
   put your mouse somewhere over an axes, where it will turn into a hand
   icon.

   -  *Pan:* Press the left mouse button and hold it to pan the figure,
      dragging it to a new position. Press Ctrl+Shift with the pan tool
      to move in one axis only, which one you have moved farther on.
      Keep in mind that that this button will allow you pan outside the
      bounds of the image; if you get lost, you can always use the
      **Home** to back you back to the initial view.
   -  *Zoom:* You can zoom in and out of a plot by pressing Ctrl (Mac)
      or holding down the right mouse button (Windows) while panning.
      Once you're done, the right click menu will pop up when you're
      done with the action; dismiss it by clicking off the plot. This is
      a known bug to be corrected in the next release.

-  **Zoom-to-rectangle button:** Click this toolbar button |image32|  to
   activate this mode. To zoom in, press the left mouse button and drag
   in the window to draw a box around the area you want to zoom in on.
   When you release the mouse button, the image is re-drawn to display
   the specified area. Remember that you can always use *Backward*
   button to go back to the previous zoom level, or use the *Home*
   button to reset the window to the initial view.
-  **Save:** Click this button |image33|  to launch a file save dialog.
   You can save the figure window to an image file. Note that this will
   save the entire contents of the window, not just the individual
   subplot(s) or images.

.. |image28| image:: memory:%(WINDOW_HOME_BUTTON)s
.. |image29| image:: memory:%(WINDOW_FORWARD_BUTTON)s
.. |image30| image:: memory:%(WINDOW_BACK_BUTTON)s
.. |image31| image:: memory:%(WINDOW_PAN_BUTTON)s
.. |image32| image:: memory:%(WINDOW_ZOOMTORECT_BUTTON)s
.. |image33| image:: memory:%(WINDOW_SAVE_BUTTON)s
""" % globals()

INTENSITY_MODE_HELP_LIST = """

-  *Raw:* Shows the image using the full colormap range permissible for
   the image type. For example, for a 16-bit image, the pixel data will
   be shown using 0 as black and 65535 as white. However, if the actual
   pixel intensities span only a portion of the image intensity range,
   this may render the image unviewable. For example, if a 16-bit image
   only contains 12 bits of data, the resulting image will be entirely
   black.
-  *Normalized (default):* Shows the image with the colormap
   "autoscaled" to the maximum and minimum pixel intensity values; the
   minimum value is black and the maximum value is white.
-  *Log normalized:* Same as *Normalized* except that the color values
   are then log transformed. This is useful for when the pixel intensity
   spans a wide range of values but the standard deviation is small
   (e.g., the majority of the interesting information is located at the
   dim values). Using this option increases the effective contrast.

"""

INTERPOLATION_MODE_HELP_LIST = """

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

"""

MODULE_DISPLAY_IMAGE_TOOLS_HELP = """ Right-clicking in an image
displayed in a window will bring up a pop-up menu with the following
options:

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
   color/intensity values in the images: {INTENSITY_MODE_HELP_LIST}
-  *Interpolation:* Presents three options for displaying the resolution
   in the images. This is useful for specifying the amount of detail
   that you want to be visible if you zoom in:
   {INTERPOLATION_MODE_HELP_LIST}
-  *Save subplot:* Save the clicked subplot as an image file. If there
   is only one p lot in the figure, this option will save that one.
-  *Channels:* For color images only. You can show any combination of
   the red, green, and blue color channels.

""".format(**{
    "INTENSITY_MODE_HELP_LIST": INTENSITY_MODE_HELP_LIST,
    "INTERPOLATION_MODE_HELP_LIST": INTERPOLATION_MODE_HELP_LIST
})

FIGURE_HELP = (
        ("Using The Display Window Menu Bar",MODULE_DISPLAY_MENU_BAR_HELP),
        ("Using The Interactive Navigation Toolbar", MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP),
        ("How To Use The Image Tools", MODULE_DISPLAY_IMAGE_TOOLS_HELP))

WORKSPACE_VIEWER_HELP = """ The workspace viewer is a flexible tool
that you can use to explore your images, objects and measurements in
test mode. To use the viewer, select *View Workspace* from the *Test*
menu after starting test mode. This will display the CellProfiler
Workspace, a window with an image pane to the left and a panel of
controls to the right.

Key concepts
~~~~~~~~~~~~

The workspace viewer lets you examine the CellProfiler workspace as you
progress through your pipeline's execution. A pipeline's *workspace* is
the collection of images, objects and measurements that are produced by
the modules. At the start of the pipeline, the only things that are
available are the images and objects loaded by the input modules. New
images, objects and measurements are added to the workspace as you step
through modules. If you modify a module's setting and re-execute the
module, the images, objects and measurements produced by that module
will be overwritten.

The selected views are persistent across image cycles. That is, you can
set up the viewer to view the workspace at the end of a pipeline cycle,
then start a new cycle and CellProfiler will fill in the images, objects
and measurements that you have chosen to display as they become
available. You can also zoom in on a particular region and change
settings and the viewer will remain focused on that region with the same
settings across modules or image cycles.

All elements of the display are configurable, either through the
Subplots menu on the viewer, or through the context menu available by
right-clicking on the figure window.

Available displays
~~~~~~~~~~~~~~~~~~

A number of displays are available on the right-side of the workspace
viewer. You can add, remove and modify displays of *images*, *objects*,
*masks* and *measurements*,

Images
^^^^^^

| The workspace viewer can display any image that is available from the
  input modules or from modules previously executed. To display a single
  image, select it from the *Images* drop down and check the *Show*
  checkbox. Initially, the image will be displayed in color, using the
  color shown in the "Color" box. This color can be changed by clicking
  on the color box.
| You can add images to the display by clicking the *Add Image* button.
  You can remove images other than the first by hitting the button in
  the *Remove* column. You can toggle the image display using the
  checkbox in the *Show* column. You can also set the interpolation mode
  by selecting *Interpolation* from the *Subplots* menu.

Objects
^^^^^^^

You can display the objects that have been created or loaded by all
modules that have been executed. To display a set of objects, select
them from the *Objects* drop-down and check the *Show* checkbox. You can
add additional objects by pressing the *Add Objects* button.

Masks
^^^^^

You can display the mask for any image produced by any of the modules
that have been executed. Most images are not masked. In these cases, you
can display the mask, but the display will show that the whole image is
unmasked. You can mask an image with the **MaskImage** or **Crop**
modules.

To display the mask of an image, select it from the *Masks* dropdown and
check the *Show* checkbox. You can add additional masks by pressing the
*Add Mask* button. The options for masks are the same as that for
objects with the addition that you can invert and overlay the mask by
choosing *Inverted* from the mask's menu; the masked portion will be
displayed in color.

Measurements
^^^^^^^^^^^^

You can display any measurement produced by any of the modules that have
been executed. Image measurements will be displayed in the title bar
above the image. Object measurements will be displayed centered over the
measurement's object. To display a measurement, select it from the
*Measurements* drop-down and check the *Show* checkbox next to the
measurement. You can add a measurement by pressing the *Add Measurement*
button or remove it by checking the button in the *Remove* column.

You can configure the font used to display an object measurement, the
color of the text, and the color, transparency and shape of the
background behind the text. To configure the measurement's appearance,
press the *Font* button to the right of the measurement. Press the
*Font* button in the *Measurement appearance* dialog to choose the font
and its size, press the *Text color* and *Background color* to change
the color used to display the text and background. Use the *Alpha*
slider to control the transparency of the background behind the
measurement text. The *Box shape* drop-down controls the shape of the
background box. The *Precision* control determines the number of digits
displayed to the right of the decimal point.

Using the Subplot menu to configure the display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following items modify how the display is rendered. You can
configure them through the Subplots menu on the viewer, or through the
context menu available by right-clicking on the figure window.

Interpolation
^^^^^^^^^^^^^

The interpolation mode used to render images, objects and masks is a
configuration option that applies to the entire workspace. Interpolation
controls how the intensities of pixels are blended together. You can set
the interpolation mode by selecting *Interpolation* from the *Subplots*
menu. The available modes are: {INTERPOLATION_MODE_HELP_LIST}

Images
^^^^^^

You can change the way an image is scaled, you can change its display
mode and you can change its color and transparency from the menus. To do
this, select the image from the *Subplots* menu. The images that are
shown will appear in the menu under the *--- Images ---* heading. Select
the image you want to configure from the menu to display the options
that are available for that image. There are three categories in the
menu, one for intensity normalization, one for the display mode and one
to adjust color and transparency.

The intensity normalization mode controls how the pixel's intensity
value is translated into the brightness of a pixel on the screen. The
modes that are available are: {INTENSITY_MODE_HELP_LIST}

The *Mode* controls how pixel intensities are mapped to colors in the
image. You can display each image using the following modes:

-  *Color:* Pixels will have a uniform color which can be selected by
   either clicking on the *Color* button next to the image name or by
   choosing the image's *Color* menu entry.
-  *Grayscale:* The image will be rendered in shades of gray. The color
   choice will have no effect and the image's *Color* menu entry will be
   unavailable.
-  *Color map:* The image will be rendered using a palette. Your default
   color map will be used initially. To change the color map, select the
   image's *Color* menu entry from its menu and choose one of the color
   maps from the drop-down. The display will change interactively as you
   change the selection, allowing you to see the image as rendered by
   your choice. Hit *OK* to accept the new color map or hit *Cancel* to
   use the color map that was originally selected.

The image's *Alpha* menu entry lets you control the image's
transparency. This will let you blend colors when the palettes overlap
and choose which image's intensity has the highest priority. To change
the transparency, select *Alpha* from the image's menu. Adjust the
transparency interactively using the slider bar and hit *OK* to accept
the new value or *Cancel* to restore the value that was originally
selected.

Objects
^^^^^^^

You can configure the appearance of objects using the context or
*Subplots* menu. Choose the objects you wish to configure from the *---
Objects ---* list in the menu. You will see configuration menu items for
the objects' display mode, color and alpha value. You can display
objects using one of the following modes:

-  *Lines:* This mode draws a line through the center of each pixel that
   borders the background of the object or another object. It does not
   display holes in the object. The line is drawn using the color shown
   in the *Color* button next to the objects' name. This option does not
   obscure the border pixels, but can take longer to render, especially
   if there are a large number of objects.
-  *Outlines:* This mode displays each pixel in the object's border
   using the color shown in the *Color* button next to the objects'
   name. This option will display holes in unfilled objects, but the
   display obscures the image underneath the border pixels.
-  *Overlay:* This mode displays a different color overlay over each
   object's pixels. Each object is assigned a color using the default
   color map initially. You can choose the color map by selecting
   *Color* from the objects' menu and choosing one of the available
   color maps. You can change the transparency of the overlay by
   choosing *Alpha* from the objects' menu.

""".format(**{
    "INTERPOLATION_MODE_HELP_LIST": INTERPOLATION_MODE_HELP_LIST,
    "INTENSITY_MODE_HELP_LIST": INTENSITY_MODE_HELP_LIST
})

WV_FIGURE_HELP = tuple(list(FIGURE_HELP) +
                       [( "How To Use The Workspace Viewer", WORKSPACE_VIEWER_HELP)])

################################################### # # Help for the preferences dialog ####################################################

TITLE_FONT_HELP = """ Sets the font used in titles above plots displayed in module figure windows."""

TABLE_FONT_HELP = """ Sets the font used in tables displayed in module figure windows."""

DEFAULT_COLORMAP_HELP = """ Specifies the color map that sets the
colors for labels and other elements. See this
`page <http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps>`__ for
pictures of available colormaps."""

WINDOW_BACKGROUND_HELP = """ Sets the window background color of the CellProfiler main window."""

ERROR_COLOR_HELP =""" Sets the color used for the error alerts associated with misconfigured settings and other errors."""

PLUGINS_DIRECTORY_HELP = """ Chooses the directory that holds
dynamically-loaded CellProfiler modules. You can write your own module
and place it in this directory and CellProfiler will make it available
for your pipeline. You must restart CellProfiler after modifying this
setting."""

IJ_PLUGINS_DIRECTORY_HELP = """ Sets the directory that
holds ImageJ plugins (for the **RunImageJ** module). You can download or
write your own ImageJ plugin and place it in this directory and
CellProfiler will make it available for your pipeline. You must restart
CellProfiler after modifying this setting.""" #consider deprecating

IJ_VERSION_HELP = """ Chooses which version of ImageJ to use in the
**RunImageJ** module. You must restart CellProfiler after changing this
preference for the preference to take effect.

-  **ImageJ 1.x:** This is a version of ImageJ 1.44 with added support
   for ``&parameter`` plugin decorations. You should use this if you
   only have ImageJ 1.0 plugins.
-  **ImageJ 2.0:** This is an alpha release of ImageJ 2.0. ImageJ 2.0
   has better interoperability with CellProfiler. CellProfiler will
   display ImageJ 2.0 plugin settings as part of the RunImageJ module
   interface and will let you use regular and masked images in ImageJ
   2.0 plugins. ImageJ 2.0 can run ImageJ 1.0 plugins, but there may be
   incompatibilities.

""" #consider deprecating

CHECK_FOR_UPDATES_HELP = """ Controls whether CellProfiler looks for updates on startup."""

SHOW_TELEMETRY_HELP = """ Allow limited and anonymous usage statistics
and exception reports to be sent to the CellProfiler team to help
improve CellProfiler. """

SHOW_STARTUP_BLURB_HELP = """ Controls whether CellProfiler displays an orientation message on startup."""

SHOW_ANALYSIS_COMPLETE_HELP = """ Determines whether CellProfiler
displays a message box at the end of a run. Check this preference to
show the message box or uncheck it to stop display."""

SHOW_EXITING_TEST_MODE_HELP = """ Determines whether CellProfiler
displays a message box to inform you that a change made to the pipeline
will cause test mode to end. Check this preference to show the message
box or uncheck it to stop display."""

SHOW_REPORT_BAD_SIZES_DLG_HELP = """ Determines whether
CellProfiler will display a warning dialog if images of different sizes
are loaded together in an image set. Check this preference to show the
message box or uncheck it to stop display."""

PRIMARY_OUTLINE_COLOR_HELP = """ Sets the color used for the outline
of the object of interest in the **IdentifyPrimaryObjects**,
**IdentifySecondaryObjects** and **IdentifyTertiaryObjects**
displays."""

SECONDARY_OUTLINE_COLOR_HELP = """ Sets the color used
for objects other than the ones of interest. In
**IdentifyPrimaryObjects**, these are the objects that are too small or
too large. In **IdentifySecondaryObjects** and
**IdentifyTertiaryObjects**, this is the color of the secondary objects'
outline."""

TERTIARY_OUTLINE_COLOR_HELP = """ Sets the color used for
the objects touching the image border or image mask in
**IdentifyPrimaryObjects**."""

INTERPOLATION_MODE_HELP = """ Sets the
way CellProfiler displays image pixels. If you choose *Nearest*,
CellProfiler will display each pixel as a square block of uniform
intensity. This is truest to the data, but the resulting images look
blocky and pixelated. You can choose either *Bilinear* or *Bicubic* to
see images where the a bilinear or bicubic spline model has been used to
interpolate the screen pixel value for screen pixels that do not fall
exactly in the center of the image pixel. The result, for bilinear or
bicubic interpolation is an image that is more visually appealing and
easier to interpret, but obscures the true pixel nature of the real
data. """

INTENSITY_MODE_HELP = """ Sets the way CellProfiler
normalizes pixel intensities when displaying. If you choose "raw",
CellProfiler will display a pixel with a value of "1" or above with the
maximum brightness and a pixel with a value of "0" or below as black. If
you choose "normalize", CellProfiler will find the minimum and maximum
intensities in the display image and show pixels at maximum intensity
with the maximum brightness and pixels at the minimum intensity as
black. This can be used to view dim images. If you choose "log",
CellProfiler will use the full brightness range and will use a log scale
to scale the intensities. This can be used to view the image background
in more detail. """

REPORT_JVM_ERROR_HELP = """ Determines whether
CellProfiler will display a warning on startup if CellProfiler can't
locate the Java installation on your computer. Check this box if you
want to be warned. Uncheck this box to hide warnings."""

MAX_WORKERS_HELP = """ Controls the maximum number of *workers* (i.e.,
copies of CellProfiler) that will be started at the outset of an
analysis run. CellProfiler uses these copies to process multiple image
sets in parallel, utilizing the computer's CPUs and memory fully. The
default value is the number of CPUs detected on your computer. Use fewer
workers for pipelines that require a large amount of memory. Use more
workers for pipelines that are accessing image data over a slow
connection.

If using the **Groups** module, only one worker will be allocated to
handle each group. This means that you may have multiple workers
created, but only a subset of them may actually be active, depending on
the number of groups you have.

"""

TEMP_DIR_HELP = """ Sets the folder that CellProfiler uses when
storing temporary files. CellProfiler will create a temporary
measurements file for analyses when the user specifies that a MATLAB
measurements file should be created or when the user asks that no
measurements file should be permanently saved. CellProfiler will also
save images accessed by http URL temporarily to disk (but will
efficiently access OMERO image planes directly from the server). """

JVM_HEAP_HELP = """ Sets the maximum amount of memory that can be used
by the Java virtual machine. CellProfiler uses Java for loading images,
for running ImageJ and for processing image sets. If you load extremely
large images, use the RunImageJ module extensively or process large
image set lists, you can use this option to start Java with a larger
amount of memory. By default, CellProfiler starts Java with 512 MB, but
you can override this by specifying the number of megabytes to load. You
can also start CellProfiler from the command-line with the
--jvm-heap-size switch to get the same effect. """

SAVE_PIPELINE_WITH_PROJECT_HELP = """ Controls whether a pipeline
and/or file list file is saved whenever the user saves the project file.
Users may find it handy to have the pipeline and/or file list saved in a
readable format, for instance, for version control whenever the project
file is saved. Your project can be restored by importing both the
pipeline and file list, and your pipeline can be run using a different
file list, and your file list can be reused by importing it into a
different project. Note: When using LoadData, it is not recommended to
auto-save the file list, as this feature only saves the file list
existing in the Input Modules, not LoadData input files.

-  *Neither:* Refrain from saving either file.
-  *Pipeline:* Save the pipeline, using the project's file name and path
   and a .cppipe extension.
-  *File list:* Save the file list, using the project's file name and
   path and a .txt extension.
-  *Pipeline and file list:* Save both files.

"""

BATCHPROFILER_URL_HELP = """ The base URL for BatchProfiler.
BatchProfiler is a set of CGI scripts for running CellProfiler on a
GridEngine cluster or compatible. If BatchProfiler is available, the
CreateBatchFiles module can optionally launch a browser to display the
appropriate batch configuration page."""

EACH_PREFERENCE_HELP = (
        ("Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
        ("Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
        ("Title font", TITLE_FONT_HELP),
        ("Table font", TABLE_FONT_HELP),
        ("Default colormap", DEFAULT_COLORMAP_HELP),
        ("Window background",WINDOW_BACKGROUND_HELP),
        ("Error color", ERROR_COLOR_HELP),
        ("Primary outline color", PRIMARY_OUTLINE_COLOR_HELP),
        ("Secondary outline color", SECONDARY_OUTLINE_COLOR_HELP),
        ("Tertiary outline color", TERTIARY_OUTLINE_COLOR_HELP),
        ("Interpolation mode", INTERPOLATION_MODE_HELP),
        ("Intensity mode", INTENSITY_MODE_HELP),
        ("CellProfiler plugins directory", PLUGINS_DIRECTORY_HELP),
        ("ImageJ plugins directory", IJ_PLUGINS_DIRECTORY_HELP),
        # ( "ImageJ version",IJ_VERSION_HELP),
        ("Check for updates", CHECK_FOR_UPDATES_HELP),
        ("Display welcome text on startup", SHOW_STARTUP_BLURB_HELP),
        ("Warn if Java runtime environment not present", REPORT_JVM_ERROR_HELP),
        ('Show the "Analysis complete" message at the end of a run', SHOW_ANALYSIS_COMPLETE_HELP),
        ('Show the "Exiting test mode" message', SHOW_EXITING_TEST_MODE_HELP),
        ("Warn if images are different sizes", SHOW_REPORT_BAD_SIZES_DLG_HELP),
        ("Show the parameter sampling menu", PARAMETER_SAMPLING_MENU_HELP),
        ("Maximum number of workers", MAX_WORKERS_HELP),
        ("Temporary folder", TEMP_DIR_HELP),
        ("Save pipeline and/or file list in addition to project", SAVE_PIPELINE_WITH_PROJECT_HELP),
        ("BatchProfiler URL", BATCHPROFILER_URL_HELP)
        )

PREFERENCES_HELP = """The Preferences allow you to change many options in CellProfiler-  """

for key, value in enumerate(EACH_PREFERENCE_HELP):
   PREFERENCES_HELP += """-  **""" + value[0] + """:**""" + value[1] + """"""

######################################################## # # Help re: projects ##########################################################

CREATING_A_PROJECT_CAPTION = "Creating A Project"

INTRODUCTION_TO_PROJECTS_HELP = """

What is a project?
~~~~~~~~~~~~~~~~~~

In CellProfiler, a *project* is comprised of two elements:

-  An *image file list* which is the list of files and their locations
   that are selected by the user as candidates for analysis.
-  The *pipeline*, which is a series of modules put together used to
   analyze a set of images.
-  Optionally, the associated information about the images (*metadata*).
   This information may be part of the images themselves, or imported
   externally by the user.

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

By using projects, the above information is stored along with the
analysis pipeline and is available on demand.

Working with projects
~~~~~~~~~~~~~~~~~~~~~

Creating a project
^^^^^^^^^^^^^^^^^^

Upon starting CellProfiler, you will be presented with a new, blank
project. At this point, you may start building your project by using the
modules located in the "Input modules" panel on the upper-left. The
modules are:

-  **Images**: Assemble the relevant images for analysis (required).
-  **Metadata**: Associate metadata with the images (optional).
-  **NamesAndTypes**: Assign names to channels and define their
   relationship (required).
-  **Groups**: Define sub-divisions between groups of images for
   processing (optional).

Detailed help for each module is provided by selecting the module and
clicking the "?" button on the bottom of CellProfiler.

Saving a project
^^^^^^^^^^^^^^^^

As you work in CellProfiler, the project is updated automatically, so
there is no need to save it unless you are saving the project to a new
name or location. You can always save your current work to a new project
file by selecting *File > Save Project As...*, which will save your
project, complete with the current image file list and pipeline, to a
file with with the extension *.cpproj*.

You also have the option of automatically saving the associated pipeline
file and the file list in addition to the project file. See *File >
Preferences...* for more details.

For those interested, some technical details:

-  The *.cpproj* file stores collected information using the HDF5
   format. Documentation on how measurements are stored and handled in
   CellProfiler using this format can be found
   `here <https://github.com/CellProfiler/CellProfiler/wiki/Module-Structure-and-Data-Storage-Retrieval#hdf5-measurement-and-workspace-format>`__.
-  All information is cached in the project file after it is computed.
   It is either re-computed or retrieved from the cache when an analysis
   run is started, when entering Test mode, or when the user requests a
   refreshed view of the information (e.g., when a setting has been
   changed).

Legacy modules: LoadImages and LoadData
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Historically, two modules were used for project creation: **LoadImages**
and **LoadData**. While the approach described above partly supersedes
these modules, you have the option of preserving these modules if you
load old pipelines into CellProfiler that contain them; these pipelines
will operate exactly as before.

Alternately, the user can choose to convert these modules into the
project equivalent as closely as possible. Both **LoadImages** and
**LoadData** remain accessible via the "Add module" and |image34|
buttons at the bottom of the pipeline panel.

.. |image34| image:: memory:%(MODULE_ADD_BUTTON)s
""" % globals()

SELECTING_IMAGES_HELP = """

Any image analysis project using CellProfiler begins with providing the
program with a set of image files to be analyzed. You can do this by
clicking on the **Images** module to select it (located in the Input
modules panel on the left); this module is responsible for collecting
the names and locations of the files to be processed.

The most straightforward way to provide files to the **Images** module
is to simply drag-and-drop them from your file manager tool (e.g.,
Windows Explorer, Finder) onto the file list panel (the blank space
indicated by the text "Drop files and folders here"). Both individual
files and entire folders can be dragged onto this panel, and as many
folders and files can be placed onto this panel as needed. As you add
files, you will see a listing of the files appear in the panel.

CellProfiler supports a wide variety of image formats, including most of
those used in imaging, by using a library called Bio-Formats; see
`here <http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html>`__
for the formats available. Some image formats are better than others for
image analysis. Some are
`"lossy" <http://www.techterms.com/definition/lossy>`__ (information is
lost in the conversion to the format) like most JPG/JPEG files; others
are `lossless <http://www.techterms.com/definition/lossless>`__ (no
image information is lost). For image analysis purposes, a lossless
format like TIF or PNG is recommended.

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
clicking the |image35| button at the bottom of the pipeline panel, or
check out the Input module tutorials on our
`Tutorials <http://cellprofiler.org/tutorials.html>`__ page.

.. |image35| image:: memory:%(MODULE_HELP_BUTTON)s
""" % globals()

CONFIGURE_IMAGES_HELP = """

Once you have used the **Images** module to produce a list of images to
be analyzed, you can use the other Input modules to define how images
are related to one another, give them a memorable name for future
reference, attach additional image information about the experiment,
among other things.

After **Images**, you can use the following Input modules:

+---------------------+-------------------------------------------------------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Module**          | **Description**                                                         | **Use required?**   | **Usage notes**                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
+=====================+=========================================================================+=====================+===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| **Metadata**        | Associate image information (metadata) with the images                  | No                  | With this module, you can extract metadata from various sources and append it to the measurements that your pipeline will collect, or use it to define how the images are related to each other. The metadata can come from the image filename or location, or from a spreadsheet that you provide. If your assay does not require or have such information, this module can be safely skipped.                                                                                           |
+---------------------+-------------------------------------------------------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **NamesAndTypes**   | Assign names to images and/or channels and define their relationship.   | Yes                 | This module gives each image a meaningful name by which modules in the analysis pipeline will refer to it. The most common usage for this module is to define a collection of channels that represent a single field of view. By using this module, each of these channels will be loaded and processed together for each field of view.                                                                                                                                                  |
+---------------------+-------------------------------------------------------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Groups**          | Define sub-divisions between groups of images for processing.           | No                  | For some assays, you will need the option of further sub-dividing an image set into *groups* that share a common feature. An example of this is a time-lapse movie that consists of individual files; each group of files that define a single movie needs to be processed independently of the others. This module allows you to specify what distinguishes one group of images from another. If your assay does not require this sort of behavior, this module can be safely skipped.   |
+---------------------+-------------------------------------------------------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

For more information on these modules and how to configure them for the
best performance, please see the detailed help by selecting the module
and clicking the |image36| button at the bottom of the pipeline panel,
or check out the Input module tutorials on our
`Tutorials <http://cellprofiler.org/tutorials.html>`__ page.

.. |image36| image:: memory:%(MODULE_HELP_BUTTON)s
""" % globals()


LOADING_IMAGE_SEQUENCES_HELP = """

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
*img000.png*, *img001.png*, *img002.png*, etc

. It is also not uncommon to store the movie such that one movie's worth
of files is stored in a single folder.

*Example:* You have a time-lapse movie of individual files set up as
follows:

-  Three folders, one for each image channel, named *DNA*, *actin* and
   *phase*.
-  In each folder, the files are named as follows:

   -  *DNA*: calibrate2-P01.001.TIF, calibrate2-P01.002.TIF,...,
      calibrate2-P01.287.TIF
   -  *actin*: calibrated-P01.001.TIF, calibrated-P01.002.TIF,...,
      calibrated-P01.287.TIF
   -  *phase*: phase-P01.001.TIF, phase-P01.002.TIF,...,
      phase-P01.287.TIF

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

   -  Select "{X_MANUAL_EXTRACTION}" or "{X_IMPORTED_EXTRACTION}" as
      the metadata extraction method. You will use these to extract the
      movie and timepoint tags from the images.
   -  Use "{X_MANUAL_EXTRACTION}" to create a regular expression to
      extract the metadata from the filename and/or path name.
   -  Or, use "{X_IMPORTED_EXTRACTION}" if you have a comma-delimited
      file (CSV) of the necessary metadata columns (including the movie
      and timepoint tags) for each image. Note that microscopes rarely
      produce such a file, but it might be worthwhile to write scripts
      to create them if you do this frequently.

   If there are multiple channels for each movie, this step may need to
   be performed for each channel.

   In this example, you could do the following:

   -  Select "{X_MANUAL_EXTRACTION}" as the method, "From file name"
      as the source, and
      ``.*-(?P<Well>[A-P][0-9]{{2}})\.(?P<Timepoint>[0-9]{{3}})`` as the
      regular expression. This step will extract the well ID and
      timepoint from each filename.
   -  Click the "Add" button to add another extraction method.
   -  In the new group of extraction settings, select
      "{X_MANUAL_EXTRACTION}" as the method, "From folder name" as the
      source, and ``.*[\\/](?P<Stain>.*)[\\/].*$`` as the regular
      expression. This step will extract the stain name from each folder
      name.
   -  Click the "Update" button below the divider and check the output
      in the table to confirm that the proper metadata values are being
      collected from each image.

-  In the **NamesAndTypes** module, assign the channel(s) to a name of
   your choice. If there are multiple channels, you will need to do this
   for each channel.
   For this example, you could do the following:

   -  Select "Assign images matching rules".
   -  Make a new rule ``[Metadata][Does][Have Stain matching][actin]``
      and name it *OrigFluor*.
   -  Click the "Add" button to define another image with a rule.
   -  Make a new rule ``[Metadata][Does][Have Stain matching][DNA]`` and
      name it *OrigFluo2*.
   -  Click the "Add" button to define another image with a rule.
   -  Make a new rule ``[Metadata][Does][Have Stain matching][phase]``
      and name it *OrigPhase*.
   -  In the "Image set matching method" setting, select "Metadata".
   -  Select "Well" for the *OrigFluor*, *OrigFluo2*, and *OrigPhase*
      channels.
   -  Click the |image37| button to the right to add another row, and
      select "Timepoint" for each channel.
   -  Click the "Update" button below the divider to view the resulting
      table and confirm that the proper files are listed and matched
      across the channels. The corresponding well and frame for each
      channel should now be matched to each other.

-  In the **Groups** module, enable image grouping for these images in
   order to select the metadata that defines a distinct movie of data.
   For the example above, do the following:

   -  Select "Well" as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each well is defined as a
      group, each with 287 frames' worth of images.

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
formats. For more details on supported files, see this
`webpage <http://www.openmicroscopy.org/site/support/bio-formats4/supported-formats.html>`__.
In general, we recommend saving stacks and movies in .TIF format.

*Example:* You have several image stacks representing 3D structures in
the following format:

-  The stacks are saved in .TIF format.
-  Each stack is a single-channel grayscale image.
-  Your files have names like IMG01\_CH01.TIF, IMG01\_CH02.TIF, ...
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
-  In the **NamesAndTypes** module, select "Yes" for "Data is 3D". You
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

   -  Select "Assign images matching rules".
   -  Make a new rule ``[File][Does][Contain][CH01]``
   -  Provide a descriptive name for the channel, e.g., *DAPI*.
   -  Click the "Add another image" button to define a second image with
      a set of rules.
   -  Make a new rule ``[File][Does][Contain][CH02]``
   -  Provide a descriptive name for the channel *GFP*.
   -  Click the "Update" button below the divider to confirm that the
      proper images are listed and matched across the channels. All file
      names ending in CH01.TIF should be matched together.

*Example:* You have two image stacks in the following format:

-  The stacks are Opera's FLEX format.
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
      information is usually contained in the image's internal metadata,
      in contrast to the image sequence described above.

   To accomplish this, do the following:

   -  Select "{X_AUTOMATIC_EXTRACTION}" as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the "Update metadata" button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while to complete.
   -  Click the "Update" button below the divider.
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

   -  Select "Assign images matching rules".
   -  Make a new rule ``[Metadata][Does][Have C matching][0]``
   -  Click the |image38| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``. This combination tells
      CellProfiler not to treat the image as a single file, but rather
      as a series of frames.
   -  Name the image *DAPI*.
   -  Click the "Add another image" button to define a second image with
      a set of rules.
   -  Make a new rule ``[Metadata][Does][Have C matching][1]``
   -  Click the |image39| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *GFP*.
   -  Click the "Add another image" button to define a third image with
      a set of rules.
   -  Make a new rule ``[Metadata][Does][Have C matching][2]``
   -  Click the |image40| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *TxRed*.
   -  In the "Image set matching method" setting, select "Metadata".
   -  Select "FileLocation" for the DAPI, GFP and TxRed channels. The
      FileLocation metadata tag identifies the individual stack, and
      selecting this parameter ensures that the channels are first
      matched within each stack, rather than across stacks.
   -  Click the |image41|  button to the right to add another row, and
      select *Series* for each channel.
   -  Click the "Update" button below the divider to confirm that the
      proper image slices are listed and matched across the channels.
      The corresponding *FileLocation* and *Series* for each channel
      should now be matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select "FileLocation" as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the two image
      stacks are defined as a group, each with 8 slices' worth of
      images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

*Example:* You have four Z-stacks in the following format:

-  The stacks are in Zeiss' CZI format.
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
   File list panel. In this case, the default "Images only" filter is
   sufficient to capture the necessary files.
-  In the **Metadata** module, enable metadata extraction in order to
   obtain metadata from these files. The key step here is to obtain the
   metadata tags necessary to do two things:

   -  Distinguish the stacks from each other. This information is
      contained as the file itself, that is, each file represents a
      different stack.
   -  For each stack, distinguish the z-planes from each other, ensuring
      proper ordering. This information is usually contained in the
      image file's internal metadata.

   To accomplish this, do the following:

   -  Select "{X_AUTOMATIC_EXTRACTION}" as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the "Update metadata" button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while.
   -  Click the "Update" button below the divider.
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

   -  Select "Assign images matching rules".
   -  Make a new rule ``[Metadata][Does][Have C matching][0]``
   -  Click the |image42| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *DAPI*.
   -  Click the "Add another image" button to define a second image with
      a set of rules.
   -  Make a new rule ``[Metadata][Does][Have C matching][1]``
   -  Click the |image43| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the second image *GFP*.
   -  Click the "Add another image" button to define a third image with
      a set of rules.
   -  Make a new rule ``[Metadata][Does][Have C matching][2]``.
   -  Click the |image44| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the third image *TxRed*.
   -  Click the "Add another image" button to define a fourth image with
      set of rules.
   -  Make a new rule ``[Metadata][Does][Have C matching][3]``.
   -  Click the |image45| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the fourth image *Cy3*.
   -  In the "Image set matching method" setting, select "Metadata".
   -  Select "FileLocation" for the *DAPI*,\ *GFP*,\ *TxRed*, and
      *Cy3*\ channels. The *FileLocation* identifies the individual
      stack, and selecting this parameter insures that the channels are
      matched within each stack, rather than across stacks.
   -  Click the |image46| button to the right to add another row, and
      select "Z" for each channel.
   -  Click "Update table" to confirm the channel matching. The
      corresponding *FileLocation* and *Z* for each channel should be
      matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select "FileLocation" as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the four image
      stacks are defined as a group, with 9, 7, 7 and 12 slices' worth
      of images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

.. |image37| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image38| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image39| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image40| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image41| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image42| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image43| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image44| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image45| image:: memory:%(MODULE_ADD_BUTTON)s
.. |image46| image:: memory:%(MODULE_ADD_BUTTON)s

""".format(**{
    "MODULE_ADD_BUTTON": MODULE_ADD_BUTTON,
    "X_AUTOMATIC_EXTRACTION": X_AUTOMATIC_EXTRACTION,
    "X_IMPORTED_EXTRACTION": X_IMPORTED_EXTRACTION,
    "X_MANUAL_EXTRACTION": X_MANUAL_EXTRACTION
})


######################################################### # # Misc.  help ##########################################################
'''The help to be displayed if someone asks for help on a module but none is
  selected'''

HELP_ON_MODULE_BUT_NONE_SELECTED = """The help button
  can be used to obtain help for the currently selected module in the
  pipeline panel on the left side of the CellProfiler interface. You do
  not have any modules in the pipeline, yet. Add a module to the
  pipeline using the "+" button or by using File > Load Pipeline."""

HELP_ON_MEASURING_DISTANCES = """To measure distances in an open
  image, use the "Measure length" tool under *Tools* in the display
  window menu bar. If you click on an image and drag, a line will appear
  between the two endpoints, and the distance between them shown at the
  right-most portion of the bottom panel."""

HELP_ON_PIXEL_INTENSITIES = """To view pixel intensities in an open
  image, use the pixel intensity tool which is available in any open
  display window. When you move your mouse over the image, the pixel
  intensities will appear in the bottom bar of the display window."""

HELP_ON_FILE_LIST = """The *File List* panel displays the image
  files that are managed by the **Images**, **Metadata**,
  **NamesAndTypes** and **Groups** modules. You can drop files and
  directories into this window or use the *Browse...* button to add
  files to the list. The context menu for the window lets you display or
  remove files and lets you remove folders.
| The buttons and checkbox along the bottom have the following
  functions:

-  *Browse...*: Browse for files and folders to add.
-  *Clear*: Clear all entries from the File list
-  *Show files excluded by filters*: *(Only shown if filtered based on
   rules is selected)* Check this to see all files in the list. Uncheck
   it to see only the files that pass the rules criteria in the
   **Images** module.
-  *Expand tree*: Expand all of the folders in the tree
-  *Collapse tree*: Collapse the folders in the tree

"""

FILTER_RULES_BUTTONS_HELP = """ Clicking the rule menus shows you
all the file *attributes*, *operators* and *conditions* you can specify
to narrow down the image list.

#. For each rule, first select the *attribute* that the rule is to be
   based on. For example, you can select "File" to define a rule that
   will filter files on the basis of their filename.
#. The *operator* drop-down is then updated with operators applicable to
   the attribute you selected. For example, if you select "File" as the
   attribute, the operator menu includes text operators such as
   *Contain* or *Starts with*. On the other hand, if you select
   "Extension" as the attribute, you can choose the logical operators
   "Is" or "Is not" from the menu.
#. In the operator drop-down menu, select the operator you want to use.
   For example, if you want to match data exactly, you may want the
   "Exactly match" or the "Is" operator. If you want the condition to be
   more loose, select an operator such as "Contains".
#. Use the *condition* box to type the condition you want to match. The
   more you type, the more specific the condition is.

   -  As an example, if you create a new filter and select *File* as the
      attribute, then select "Does" and "Contain" as the operators, and
      type "Channel" as the condition, the filter finds all files that
      include the text "Channel", such as "Channel1.tif" "Channel2.jpg",
      "1-Channel-A01.BMP" and so on.
   -  If you select "Does" and "Start with" as the operators and
      "Channel1" in the Condition box, the rule will includes such files
      as "Channel1.tif" "Channel1-A01.png", and so on.

   +-------------+
   | |image47|   |
   +-------------+

   You can also create regular expressions (an advanced syntax for
   pattern matching; see `below <#regexp>`__) in order to select
   particular files.

To add another rule, click the plus buttons to the right of each rule.
Subtract an existing rule by clicking the minus button.

You can also link a set of rules by choosing the logical expression
*All* or *Any*. If you use *All* logical expression, all the rules must
be true for a file to be included in the File list. If you use the *Any*
option, only one of the conditions has to be met for a file to be
included.

If you want to create more complex rules (e.g, some criteria matching
all rules and others matching any), you can create sets of rules, by
clicking the ellipsis button (to the right of the plus button). Repeat
the above steps to add more rules to the filter until you have all the
conditions you want to include.

Details on regular expressions
''''''''''''''''''''''''''''''

A *regular expression* is a general term refering to a method of
searching for pattern matches in text. There is a high learning curve to
using them, but are quite powerful once you understand the basics.

{REGEXP_HELP_REF}
.. |image47| image:: memory:%(IMAGES_USING_RULES_ICON)s
""".format(**{
    "REGEXP_HELP_REF": REGEXP_HELP_REF
})


######################################################### # # Plate viewer help ##########################################################
PLATEVIEWER_HELP = """

Plate Viewer help
=================

The plate viewer is a data tool that displays the images in your
experiment in plate format. Your project must define an image set list
with metadata annotations for the image's well and, optionally its plate
and site. The plate viewer will then group your images by well and
display a plate map for you. If you have defined a plate metadata tag
(with the name, "Plate"), the plate viewer will group your images by
plate and display a choice box that lets you pick the plate to display.

Click on a well to see the images for that well. If you have more than
one site per well and have site metadata (with the name, "Site"), the
plate viewer will tile the sites when displaying, and the values under
"X" and "Y" determine the position of each site in the tiled grid.

The values for "Red", "Green", and "Blue" in each row are brightness
multipliers- changing the values will determine the color and scaling
used to display each channel. "Alpha" determines the weight each channel
contributes to the summed image. """

"""
"""

#########################################################
#
# The top-level of help - used when building the HTML manual
#
#########################################################

__doc__ = """
Why Use CellProfiler?
=====================
{WHEN_CAN_I_USE_CELLPROFILER_HELP}

Navigating The Menu Bar
=======================
Using the File Menu
-------------------
{MENU_BAR_FILE_HELP}

Using the Edit Menu
-------------------
{MENU_BAR_EDIT_HELP}

Using the Test Menu
-------------------
{TEST_MODE_HELP}

Using the Window Menu
---------------------
{MENU_BAR_WINDOW_HELP}

Using the Parameter Sampling Menu
---------------------------------
{PARAMETER_SAMPLING_MENU_HELP}

Using the Data Tools Menu
-------------------------
{MENU_BAR_DATATOOLS_HELP}

Using Module Display Windows
============================
Using The Display Window Menu Bar
---------------------------------
{MODULE_DISPLAY_MENU_BAR_HELP}

Using The Interactive Navigation Toolbar
----------------------------------------
{MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP}

How To Use The Image Tools
--------------------------
{MODULE_DISPLAY_IMAGE_TOOLS_HELP}

{CREATING_A_PROJECT_CAPTION}
============================
Introduction to Projects
------------------------
{INTRODUCTION_TO_PROJECTS_HELP}

Selecting Images for Input
--------------------------
{SELECTING_IMAGES_HELP}

Configuring Images for Analysis
-------------------------------
{CONFIGURE_IMAGES_HELP}

Loading Image Stacks and Movies
-------------------------------
{LOADING_IMAGE_SEQUENCES_HELP}

How To Build A Pipeline
=======================
{BUILDING_A_PIPELINE_HELP}

Testing Your Pipeline
=====================
{TEST_MODE_HELP}

Running Your Pipeline
=====================
{RUNNING_YOUR_PIPELINE_HELP}

Using Your Output
=================
How Measurements are Named
--------------------------
{MEASUREMENT_NOMENCLATURE_HELP}

Using Spreadsheets and Databases
--------------------------------
{SPREADSHEETS_DATABASE_HELP}

Using the Output File
---------------------
{USING_THE_OUTPUT_FILE_HELP}

Troubleshooting Memory and Speed Issues
=======================================
{MEMORY_AND_SPEED_HELP}

Batch Processing
================
{BATCHPROCESSING_HELP}

Legacy Modules and Features
===========================
Load Modules
------------
{LEGACY_LOAD_MODULES_HELP}

Setting the Default Input Folder
--------------------------------
{DEFAULT_IMAGE_FOLDER_HELP}

Setting the Default Output Folder
---------------------------------
{DEFAULT_OUTPUT_FOLDER_HELP}

Setting the Output Filename
---------------------------
{USING_THE_OUTPUT_FILE_HELP}

Other Features
==============
Running Multiple Pipelines
--------------------------
{RUN_MULTIPLE_PIPELINES_HELP}

Configuring Logging
-------------------
{CONFIGURING_LOGGING_HELP}

Accessing Images From OMERO
---------------------------
{ACCESSING_OMERO_IMAGES}

Plate Viewer
------------
{PLATEVIEWER_HELP}
""".format(**{
    "ACCESSING_OMERO_IMAGES": ACCESSING_OMERO_IMAGES,
    "BATCHPROCESSING_HELP": BATCHPROCESSING_HELP,
    "BUILDING_A_PIPELINE_HELP": BUILDING_A_PIPELINE_HELP,
    "CONFIGURE_IMAGES_HELP": CONFIGURE_IMAGES_HELP,
    "CONFIGURING_LOGGING_HELP": CONFIGURING_LOGGING_HELP,
    "CREATING_A_PROJECT_CAPTION": CREATING_A_PROJECT_CAPTION,
    "DEFAULT_IMAGE_FOLDER_HELP": DEFAULT_IMAGE_FOLDER_HELP,
    "DEFAULT_OUTPUT_FOLDER_HELP": DEFAULT_OUTPUT_FOLDER_HELP,
    "INTRODUCTION_TO_PROJECTS_HELP": INTRODUCTION_TO_PROJECTS_HELP,
    "LEGACY_LOAD_MODULES_HELP": LEGACY_LOAD_MODULES_HELP,
    "LOADING_IMAGE_SEQUENCES_HELP": LOADING_IMAGE_SEQUENCES_HELP,
    "MEASUREMENT_NOMENCLATURE_HELP": MEASUREMENT_NOMENCLATURE_HELP,
    "MEMORY_AND_SPEED_HELP": MEMORY_AND_SPEED_HELP,
    "MENU_BAR_DATATOOLS_HELP": MENU_BAR_DATATOOLS_HELP,
    "MENU_BAR_EDIT_HELP": MENU_BAR_EDIT_HELP,
    "MENU_BAR_FILE_HELP": MENU_BAR_FILE_HELP,
    "MENU_BAR_WINDOW_HELP": MENU_BAR_WINDOW_HELP,
    "MODULE_DISPLAY_IMAGE_TOOLS_HELP": MODULE_DISPLAY_IMAGE_TOOLS_HELP,
    "MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP": MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP,
    "MODULE_DISPLAY_MENU_BAR_HELP": MODULE_DISPLAY_MENU_BAR_HELP,
    "PARAMETER_SAMPLING_MENU_HELP": PARAMETER_SAMPLING_MENU_HELP,
    "PLATEVIEWER_HELP": PLATEVIEWER_HELP,
    "RUN_MULTIPLE_PIPELINES_HELP": RUN_MULTIPLE_PIPELINES_HELP,
    "RUNNING_YOUR_PIPELINE_HELP": RUNNING_YOUR_PIPELINE_HELP,
    "SELECTING_IMAGES_HELP": SELECTING_IMAGES_HELP,
    "SPREADSHEETS_DATABASE_HELP": SPREADSHEETS_DATABASE_HELP,
    "TEST_MODE_HELP": TEST_MODE_HELP,
    "USING_THE_OUTPUT_FILE_HELP": USING_THE_OUTPUT_FILE_HELP,
    "WHEN_CAN_I_USE_CELLPROFILER_HELP": WHEN_CAN_I_USE_CELLPROFILER_HELP
})


'''The help menu for CP's main window'''
MAIN_HELP = (
    ("Why Use CellProfiler?", WHEN_CAN_I_USE_CELLPROFILER_HELP),
    ("Navigating The Menu Bar", (
        ("Using the File Menu", MENU_BAR_FILE_HELP),
        ("Using the Edit Menu", MENU_BAR_EDIT_HELP),
        ("Using the Test Menu", TEST_MODE_HELP),
        ("Using the Window Menu", MENU_BAR_WINDOW_HELP),
        ("Using the Parameter Sampling Menu", PARAMETER_SAMPLING_MENU_HELP),
        ("Using the Data Tools Menu", MENU_BAR_DATATOOLS_HELP))),
    ("Using Module Display Windows", FIGURE_HELP),
    # ("Setting the Preferences", PREFERENCES_HELP),
    (CREATING_A_PROJECT_CAPTION, (
        ("Introduction to Projects", INTRODUCTION_TO_PROJECTS_HELP),
        ("Selecting Images for Input", SELECTING_IMAGES_HELP),
        ("Configuring Images for Analysis", CONFIGURE_IMAGES_HELP),
        ("Loading Image Stacks and Movies", LOADING_IMAGE_SEQUENCES_HELP))),
    ("How To Build A Pipeline", BUILDING_A_PIPELINE_HELP),
    ("Testing Your Pipeline", TEST_MODE_HELP),
    ("Running Your Pipeline", RUNNING_YOUR_PIPELINE_HELP),
    ("Using Your Output", (
        ("How Measurements are Named", MEASUREMENT_NOMENCLATURE_HELP),
        ("Using Spreadsheets and Databases", SPREADSHEETS_DATABASE_HELP),
        ("Using the Output File", USING_THE_OUTPUT_FILE_HELP))),
    ("Troubleshooting Memory and Speed Issues", MEMORY_AND_SPEED_HELP),
    ("Batch Processing", BATCHPROCESSING_HELP),
    ("Legacy Modules and Features", (
        ("Load Modules", LEGACY_LOAD_MODULES_HELP),
        ("Setting the Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
        ("Setting the Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
        ("Setting the Output Filename", USING_THE_OUTPUT_FILE_HELP))),
    ("Other Features", (
        ("Running Multiple Pipelines", RUN_MULTIPLE_PIPELINES_HELP),
        ("Configuring Logging", CONFIGURING_LOGGING_HELP),
        ("Accessing Images From OMERO", ACCESSING_OMERO_IMAGES),
        ("Plate Viewer", PLATEVIEWER_HELP)))
)


def make_help_menu(h, window, menu=None):
    import wx
    import htmldialog
    if menu is None:
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