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
MEASUREMENT_NAMING_HELP = """Help > Using Your Output > How Measurements are Named"""
USING_METADATA_HELP_REF = """Please see the <b>Metadata</b> module for more details on metadata collection and usage"""
LOADING_IMAGE_SEQ_HELP_REF = """Help > Creating a Project > Loading Image Stacks and Movies"""

USING_METADATA_TAGS_REF = """
You can insert a previously defined metadata tag by either using:
<ul><li>The insert key</li>
<li>A right mouse button click inside the control</li>
<li>In Windows, the Context menu key, which is between the Windows key and Ctrl key </li></ul>
The inserted metadata tag will appear in green. To change a previously inserted metadata tag,
navigate the cursor to just before the tag and either:
<ul><li>Use the up and down arrows to cycle through possible values.</li>
<li>Right-click on the tag to display and select the available values.</li></ul>"""

USING_METADATA_GROUPING_HELP_REF = """Please see the <b>Groups</b> module for more details on the
proper use of metadata for grouping"""

from cellprofiler.setting import YES, NO

RETAINING_OUTLINES_HELP = """Select <i>%(YES)s</i> to retain the outlines of the new objects
for later use in the pipeline. For example, a common use is for quality control purposes by
overlaying them on your image of choice using the <b>OverlayOutlines</b> module and then saving
the overlay image with the <b>SaveImages</b> module.""" % locals()

NAMING_OUTLINES_HELP = """
<i>(Used only if the outline image is to be retained for later use in the pipeline)</i> <br>
Enter a name for the outlines of the identified
objects. The outlined image can be selected in downstream modules by selecting
them from any drop-down image list."""

##################################################
#
# Help for the main window
#
##################################################

LEGACY_LOAD_MODULES_HELP = """
<p>Historically, two modules served the same functionality as the current project structure:
<b>LoadImages</b> and <b>LoadData</b>.
While the approach described above supersedes these modules in part, old pipelines
loaded into CellProfiler that contain these modules will provide the option of preserving them;
these pipelines will operate exactly as before.</p>
<p>Alternately, the user can choose to convert these
modules into the project equivalent as closely as possible. Both modules remain accesible
via the "Add module" and <img src="memory:%(MODULE_ADD_BUTTON)s">&nbsp;
button at the bottom of the pipeline panel. The section details
information relevant for users who would like to continue using these modules. Please note,
however, that these modules are deprecated and may be removed in the future.</p>

<h3>Associating metadata with images</h3>
<p>Metadata (i.e., additional data about image data) is sometimes available for input images.
This information can be:
<ol>
<li>Used by CellProfiler to group images with common metadata identifiers (or "tags")
together for particular steps in a pipeline;</li>
<li>Stored in the output file along with CellProfiler-measured features for
annotation or sample-tracking purposes;
<li>Used to name additional input/output files.</li>
</ol></p>

<p>Two sources of metadata are:
<ul>
<li><i>Metadata provided in the image filename or location (pathname).</i> For example, images produced by an automated
microscope can be given names similar to "Experiment1_A01_w1_s1.tif" in which the metadata about the
plate ("Experiment1"), the well ("A01"), the wavelength number ("w1") and the imaging site ("s1") are encapsulated. The
name of the folder in which the images are saved may be meaningful and may also be considered metadata as well.
If this is the case for your data, use <b>LoadImages</b> to extract this information for
use in the pipeline and storage in the output file.</li>
<li><i>Metadata provided as a table of information</i>. Often, information associated with each image (such as
treatment, plate, well, etc) is available as a separate spreadsheet. If this is the case for your data, use
<b>LoadData</b> to load this information.</li>
</ul>
Details for the metadata-specific help is given next to the appropriate settings in
<b>LoadImages</b> and <b>LoadData</b>, as well the specific settings in other modules which
can make use of metadata. However, here is an overview of how metadata is obtained and used.</p>

<p>In <b>LoadImages</b>, metadata can be extracted from the filename and/or folder
location using regular expression, a specialized syntax used for text pattern-matching.
These regular expressions can be used to identify different parts of the filename / folder.
The syntax <i>(?P&lt;fieldname&gt;expr)</i> will extract whatever matches <i>expr</i> and
assign it to the image's <i>fieldname</i> measurement. A regular expression tool is available
which will allow you to check the accuracy of your regular expression.</p>

<p>For instance, say a researcher has folder names with the date and subfolders containing the
images with the run ID (e.g., <i>./2009_10_02/1234/</i>).
The following regular expression will capture the plate, well and site in the fields
<i>Date</i> and <i>Run</i>:<br>
<table border = "1">
<tr><td colspan = "2">.*[\\\/](?P&lt;Date&gt;.*)[\\\\/](?P&lt;Run&gt;.*)$ </td></tr>
<tr><td>.*[\\\\/]</td><td>Skip characters at the beginning of the pathname until either a slash (/) or
backslash (\\) is encountered (depending on the OS). The extra slash for the backslash is used as
an escape sequence.</td></tr>
<tr><td>(?P&lt;Date&gt;</td><td>Name the captured field <i>Date</i></td></tr>
<tr><td>.*</td><td>Capture as many characters that follow</td></tr>
<tr><td>[\\\\/]</td><td>Discard the slash/backslash character</td></tr>
<tr><td>(?P&lt;Run&gt;</td><td>Name the captured field <i>Run</i></td></tr>
<tr><td>.*</td><td>Capture as many characters as follow</td></tr>
<tr><td>$</td><td>The <i>Run</i> field must be at the end of the path string, i.e. the
last folder on the path. This also means that the <i>Date</i> field contains the parent
folder of the <i>Date</i> folder.</td></tr>
</table>

<p>In <b>LoadData</b>, metadata is extracted from a CSV (comma-separated) file
(a spreadsheet). Columns whose name begins with "Metadata" can be used to group
files loaded by <b>LoadData</b> that are associated with a common metadata value.
The files thus grouped together are then processed as a distinct image set.</p>

<p>For instance, an experiment might require that images created on the same day
use an illumination correction function calculated from all images from that day,
and furthermore, that the date be captured in the file names for the individual image
sets and in a CSV file specifying the illumination correction functions. </p>

<p>In this case, if the illumination correction images are loaded with the
<b>LoadData</b> module, the file should have a "Metadata_Date"
column which contains the date metadata tags. Similarly, if the individual images
are loaded using the <b>LoadImages</b> module, <b>LoadImages</b> should be set to extract the
<Date> metadata tag from the file names. The pipeline will then match the individual
images with their corresponding illumination correction functions based on matching
"Metadata_Date" fields.</p>

<h3>Using image grouping</h3>
<p>To use grouping, you must define the relevant metadata for each image. This can be done using regular
expressions in <b>LoadImages</b> or having them pre-defined in a CSV file for use in <b>LoadData</b>.</p>

<p>To use image grouping in <b>LoadImages</b>, please note the following:
<ul>
<li><i>Metadata tags must be specified for all images listed.</i> You cannot use
grouping unless an appropriate regular expression is defined for all the images listed
in the module.</li>
<li><i>Shared metadata tags must be specified with the same name for each image listed.</i> For example, if you
are grouping on the basis of a metadata tag "Plate" in one image channel, you
must also specify the "Plate" metadata tag in the regular expression for the other channels that you
want grouped together.</li>
</ul>
</p>
""" % globals()

DEFAULT_IMAGE_FOLDER_HELP = """
<p>Please note that the Default Input Folder will be deprecated in the future. The location
of non-image files needed by some modules will be set to an absolute path
in future versions of CellProfiler. For specifying the location of image files, please
use the <i>Input modules</i> panel starting with the <b>Images</b> module.</p>

<p>The <i>Default Input Folder</i> is enabled only if a legacy pipeline is loaded into
CellProfiler and is accessible by pressing the "View output settings"
button at the bottom of the pipeline panel. The folder designated as the
<i>Default Input Folder</i> contains the input image or data files
that you want to analyze. Several File Processing modules (e.g.,
<b>LoadImages</b> or <b>LoadData</b>) provide the option of retrieving images
from this folder on a default basis unless you specify, within the module, an alternate,
specific folder on your computer. Within modules, we recommend selecting the
Default Input Folder as much as possible, so that your pipeline will
work even if you transfer your images and pipeline to a different
computer. If, instead, you type specific folder path names into a module's settings,
your pipeline will not work on someone else's computer until you adjust those
pathnames within each module.</p>

<p>Use the <i>Browse</i> button <img src="memory:%(BROWSE_BUTTON)s">&nbsp;to specify
the folder you would like to use as the Default Input Folder, or
type the full folder path in the edit box. If you type a folder path that
cannot be found, the message box below will indicate this fact until you correct the problem.
If you want to specify a folder that does not yet exist, type the desired name and
click on the <i>New folder</i> button <img src="memory:%(CREATE_BUTTON)s">.
The folder will be created according to the pathname you have typed.</p>""" % globals()

DEFAULT_OUTPUT_FOLDER_HELP = """
<p>Please note that the Default Output Folder will be deprecated in the future. The location
of files written by the various output modules will be set to an absolute path
in future versions of CellProfiler.</p>

<p>The <i>Default Output Folder</i> is accessible by pressing the "View output settings"
button at the bottom of the pipeline panel. The Default Output Folder is the folder that CellProfiler uses to
store the output file it creates. Also, several File Processing modules (e.g., <b>SaveImages</b> or
<b>ExportToSpreadsheet</b>) provide the option of saving analysis results to
this folder on a default basis unless you specify, within the module, an alternate,
specific folder on your computer. Within modules, we recommend selecting the
Default Output Folder as much as possible, so that your pipeline will
work even if you transfer your images and pipeline to a different
computer. If, instead, you type specific folder path names into a module's settings,
your pipeline will not work on someone else's computer until you adjust those
pathnames within each module.</p>

<p>Use the <i>Browse</i> button (to the right of the text box) to specify
the folder you would like to use as the Default Output Folder, or
type the full folder path in the edit box. If you type a folder path that
cannot be found, the message box below will indicate this fact until you correct the
problem. If you want to specify a folder that does not yet exist, type the desired name and
click on the <i>New folder</i> icon to the right of the <i>Browse folder</i> icon.
The folder will be created according to the pathname you have typed.</p>"""

USING_THE_OUTPUT_FILE_HELP = """
<p>Please note that the output file will be deprecated in the future. This setting
is temporarily present for those needing HDF5 or MATLAB formats, and will be moved to
Export modules in future versions of CellProfiler.</p>

<p>The <i>output file</i> is a file where all information about the analysis as well
as any measurements will be stored to the hard drive. <b>Important note:</b> This file
does <i>not</i> provide the same functionality as the Export modules. If you want to
produce a spreadsheet of measurements easily readable by Excel or a database viewer
(or similar programs), please refer to the <b>ExportToSpreadsheet</b> or
<b>ExportToDatabase</b> modules and the associated help.</p>

<p>The options associated with the output file are accessible by pressing the
"View output settings" button at the bottom of the pipeline panel. In the
settings panel to the left, in the <i>Output Filename</i> box, you can specify the
name of the output file.</p>

<p>The output file can be written in one of two formats:
<ul>
<li>A <i>.mat file</i> which is readable by CellProfiler and by
<a href="http://www.mathworks.com/products/matlab/">MATLAB</a> (Mathworks). </li>
<li>An <i>.h5 file</i> which is readable by CellProfiler, MATLAB and any other program
capable of reading the HDF5 data format. Documentation on
how measurements are stored and handled in CellProfiler using this format can be found
<a href="https://github.com/CellProfiler/CellProfiler/wiki/Module-Structure-and-Data-Storage-Retrieval#hdf5-measurement-and-workspace-format">here</a>.</li>
</li>
</ul>
Results in the output file can also be accessed or exported
using <b>Data Tools</b> from the main menu of CellProfiler.
The pipeline with its settings can be be loaded from an output file using
<i>File > Load Pipeline...</i></p>

<p>The output file will be saved in the Default Output Folder unless you type a
full path and file name into the file name box. The path must not have
spaces or characters disallowed by your computer's platform.</p>

<p>If the output filename ends in <i>OUT.mat</i> (the typical text appended to
an output filename), CellProfiler will prevent you from overwriting this file
on a subsequent run by generating a new file name and asking if you want to
use it instead. You can override this behavior by checking the <i>Allow
overwrite?</i> box to the right.</p>

<p>For analysis runs that generate a large number of measurements, you may notice
that even though the analysis completes, CellProfiler continues to use
an inordinate amount of your CPU and RAM. This is because the output file is written
after the analysis is completed and can take a very long time for a lot of measurements.
If you do not need this file and/or notice this behavior, select "<i>%(DO_NOT_WRITE_MEASUREMENTS)s</i>"
from the "Measurements file format" drop-down box.</p>""" % globals()

NEW_FEATURES_HELP = """
A number of new features have been incorporated into this re-engineered Python
version of CellProfiler:
<h3>Interface</h3>
<ul>
<li><i>Resizable user interface: </i>The main CellProfiler interface can now be resized
by dragging the window corner.</li>
<li><i>Help for individual module settings:</i> Every setting in every module now has
a help button that you can click to display information and advice for that setting.</li>
<li><i>Settings verification:</i> CellProfiler constantly checks for setting values
that are not allowed, and immediately flags them for you.</li>
<li><i>Context-dependent module settings</i>: Prior versions of CellProfiler
displayed all settings for each module, whether or not the values were necessary,
given existing choices for other settings. Now, only those settings
you require are displayed, simplifying the interface.</li>
<li><i>Test mode for assay development:</i> This feature allows you to
preview the effect of a module setting on your data. You can step backward or forward
in the pipeline as you modify settings, optimizing your results prior to running an
actual analysis.</li>
<li><i>Unlimited number of images/objects as module input:</i> Some modules can accept an arbitrary number
of images or objects as input, and you can dynamically add or remove any of these inputs as needed.
For example, you can specify any number of single images in LoadSingleImage; previously,
the module could accept only three input images at a time. For example, in OverlayOutlines, you can specify that any number of outlines be
overlaid on an image; previously, you would have had to string multiple OverlayOutline modules
together.</li>
<li><i>Image grouping:</i> Images which share common metadata tags, whether
provided in the filename or in an accompanying text data file, can be processed together.
This is useful for any situation in which the images are organized
in groups  and each group needs to be analyzed as an individual set, such as
illumination correction for multiple plates.</li>
<li><i>Module drag and drop:</i> You can drag and drop selected modules
within a pipeline or into another instance of CellProfiler, keeping their associated
settings intact.</li>
<li><i>Listing of recent pipelines:</i> A selectable list of recently used pipelines
is available from the menu bar, for easy access.</li>
<li><i>Figure display choice:</i> Easier access to which module figure display windows are shown.
This functionality is now controlled within the pipeline, and is saved as part of the pipeline.
<li><i>Context menus:</i>  The pipeline panel responds to right-clicks, providing
easy access to module manipulation or help.</li>
<li><i>Error handling:</i> This feature sends bug reports (stack traces) to our developers.
<li><i>Better access for developers:</i> We are providing a developer's guide
as a practical introduction for programming in the CellProfiler environment, an
email list, and wiki, in addition to the available user <a href="http://cellprofiler.org/forum/viewforum.php?f=15">forum</a>.
</ul>

<h3>Module improvements</h3>
<ul>
<li><i>Improved Otsu thresholding:</i> Choose two- or three-class thresholding to handle
images where there might be an intermediate intensity level between foreground and
background.</li>
<li>Secondary object identification now permits discarding of objects touching
the image border, along with the associated primary objects.</li>
<li>Filtering objects by measurements now permits a set of objects to be filtered
with any number of measurements. </li>
<li><i>Masking of images/objects:</i> You can create masks for use with both
images and objects such that image/object measurements will include only those
regions within the masked area.</li>
<li><i>Improved loading of text information:</i> Previously, you could load only a
limited amount of annotation relevant to your images, via a text file. Now you can use
comma-delimited files to load tables of metadata, in addition to lists of input
images for analysis.</li>
<li><i>Convex hull</i> has been included as an image morphological operation.</li>
<li>A new module, MeasureNeurons, has been added, which measures the number
of trunks and branches for each neuron in an image.</li>
<li><i>Detection of new features:</i> Neurites can be extracted from images of neurons.
Branching points of line segments can be found as an image morphological operation.
Also, "dark holes" (dark spots surrounded bright rings) can be detected. </li>
<li><i>Improvements to object tracking:</i> A new tracking algorithm has been added
to the TrackObjects module which is capable of bridging temporal gaps in trajectories
and accounting for splitting/merging events.</li>
<li><i>Per-object data exporting:<i> Object data can be exported to a database as a single table containing
all user-defined object measurements, or as separate tables, one for each object.
<li><i>SQLite support:</i> Data can be exported in SQLite, a
self-contained database format. Users can create their own local databases and
no longer need access to a separate database server. Because CellProfiler
Analyst also supports SQLite, any user can access CellProfiler Analyst's
suite of data exploration and machine-leaning tools without installing a complicated database server.</li>
</ul>
"""

WHEN_CAN_I_USE_CELLPROFILER_HELP = """

<p>Most laboratories studying biological processes and human disease use
light/fluorescence microscopes to image cells and other biological samples. There
is strong and growing demand for software to analyze these images, as automated
microscopes collect images faster than can be examined by eye and the information
sought from images is increasingly quantitative and complex.</p>

<p>CellProfiler is a versatile, open-source software tool for quantifying data
from biological images, particularly in high-throughput experiments. CellProfiler
is designed for modular, flexible, high-throughput analysis of images, measuring
size, shape, intensity, and texture of every cell (or other object) in every image.
Using the point-and-click graphical user interface (GUI), users construct an image
analysis "pipeline", a sequential series of modules that each perform
an image processing function such as illumination correction, object identification
(segmentation), and object measurement. Users mix and match modules and adjust
their settings to measure the phenotype of interest. While originally designed for
high-throughput images, it is equally appropriate for low-throughput assays as
well (i.e., assays of &lt; 100 images).</p>

<p>CellProfiler can extract valuable biological information from images quickly
while increasing the objectivity and statistical power of assays. It helps researchers
approach a variety of biological questions quantitatively, including standard
assays (e.g., cell count, size, per-cell protein levels) as well as complex
morphological assays (e.g., cell/organelle shape or subcellular patterns of DNA
or protein staining).</p>

<p>The wide variety of measurements produced by CellProfiler serves as useful "raw material"
for machine learning algorithms. CellProfiler's companion software, CellProfiler Analyst,
has an interactive machine learning tool called Classifier which can learn to recognize a
phenotype of interest based on your guidance. Once you complete the training phase,
CellProfiler Analyst will score every object in your images based on CellProfiler's
measurements.  CellProfiler Analyst also contains tools for the interactive visualization
of the data produced by CellProfiler.</p>

<p>In summary, CellProfiler contains:
<ul>
<li>Advanced algorithms for image analysis that are able to accurately identify
crowded cells and non-mammalian cell types.</li>
<li>A modular, flexible design allowing analysis of new assays and phenotypes.</li>
<li>Open-source code so the underlying methodology is known and can be modified
or improved by others.</li>
<li>A user-friendly interface.</li>
<li>The capability to make use of clusters of computers when available.</li>
<li>A design that eliminates the tedium of the many steps typically involved in
image analysis, many of which are not easily transferable from one project to
another (for example, image formatting, combining several image analysis steps,
or repeating the analysis with slightly different parameters).</li>
</ul>
</p>


<h5>References</h5>
<p>For a full list of references, visit our
<a href="http://www.cellprofiler.org/citations.html">citation</a> page.</p>
<ul>
<li>Carpenter AE, Jones TR, Lamprecht MR, Clarke C, Kang IH, Friman O,
Guertin DA, Chang JH, Lindquist RA, Moffat J, Golland P, Sabatini DM (2006)
"CellProfiler: image analysis software for identifying and quantifying cell
phenotypes" <i>Genome Biology</i> 7:R100
(<a href="http://dx.doi.org/10.1186/gb-2006-7-10-r100">link</a>)</li>
<li>Kamentsky L, Jones TR, Fraser A, Bray MA, Logan D, Madden K, Ljosa V,
Rueden C, Harris GB, Eliceiri K, Carpenter AE (2011) "Improved structure,
function, and compatibility for CellProfiler: modular high-throughput image
analysis software" <i>Bioinformatics</i> 27(8):1179-1180
(<a href="http://dx.doi.org/10.1093/bioinformatics/btr095">link</a>)</li>
<li>Lamprecht MR, Sabatini DM, Carpenter AE (2007) "CellProfiler: free, versatile
software for automated biological image analysis" <i>Biotechniques</i>
42(1):71-75. [<a href="http://dx.doi.org/10.2144/000112257">link</a>)</li>
<li>Jones TR, Carpenter AE, Lamprecht MR, Moffat J, Silver S, Grenier J, Root D,
Golland P, Sabatini DM (2009) "Scoring diverse cellular morphologies in image-based
screens with iterative feedback and machine learning" <i>PNAS</i> 106(6):1826-1831
(<a href="http://dx.doi.org/10.1073/pnas.0808843106">link</a>)</li>
<li>Jones TR, Kang IH, Wheeler DB, Lindquist RA, Papallo A, Sabatini DM, Golland P,
Carpenter AE (2008) "CellProfiler Analyst: data exploration and analysis software for
complex image-based screens" <i>BMC Bioinformatics</i> 9(1):482
(<a href="http://dx.doi.org/10.1186/1471-2105-9-482">link</a>)</li>
</ul>
"""

BUILDING_A_PIPELINE_HELP = """
<p>A <i>pipeline</i> is a sequential set of image analysis modules. The
best way to learn how to use CellProfiler is to load an example pipeline
from the CellProfiler website's Examples page and try it out, then adapt it for
your own images. You can also build a
pipeline from scratch. Click the <i>Help</i> <img src="memory:%(MODULE_HELP_BUTTON)s">
&nbsp;button in the main window to get
help for a specific module.</p>

<h3>Loading an existing pipeline</h3>
<ol>
<li>Put the images and pipeline into a folder on your computer.</li>
<li>Set the Default Output Folder (press the "View output settings") to the folder where you
want to place your output (preferably a different location than in the input images).</li>
<li>Load the pipeline using <i>File > Import Pipeline > From File...</i> in the main menu of
CellProfiler.</li>
<li>Click the <i>Analyze Images</i> button to start processing.</li>
<li>Examine the measurements using <i>Data tools</i>. The <i>Data tools</i> options are accessible in
the main menu of CellProfiler and allow you to plot, view, or export your
measurements (e.g., to Excel).</li>
<li>If you modify the modules or settings in the pipeline, you can save the
pipeline using <i>File > Export > Pipeline...</i>. Alternately, you can save the project as a whole
using <i>File > Save Project</i> or <i>Save Project As...</i> which also saves the
file list.</li>
<li>To learn how to use a cluster of computers to process
large batches of images, see <i>%(BATCH_PROCESSING_HELP_REF)s</i>.</li>
</ol>

<h3>Building a pipeline from scratch</h3>
<p>Constructing a pipeline involves placing individual modules into a pipeline. The list
of modules in the pipeline is shown in the <i>pipeline panel</i> (located on the
left-hand side of the CellProfiler window).</p>
<ol>
<li><p><i>Place analysis modules in a new pipeline.</i><br>
<p>Choose image analysis modules to add to your pipeline by clicking the <i>Add</i>
<img src="memory:%(MODULE_ADD_BUTTON)s">&nbsp;button
(located underneath the pipeline panel) or right-clicking in the pipeline panel
itself and selecting a module from the
pop-up box that appears.</p>
<p>You can learn more about each module by clicking
<i>Module Help</i> in the "Add modules" window or the <i>?</i> button after the module
has been placed and selected in the pipeline. Modules are added to the end of the
pipeline or after the currently selected module, but you can
adjust their order in the main window by dragging and dropping them, or by selecting a module (or
modules, using the <i>Shift</i> key) and using the <i>Move Module Up</i>
<img src="memory:%(MODULE_MOVEUP_BUTTON)s">&nbsp;and <i>Move Module Down</i>
<img src="memory:%(MODULE_MOVEDOWN_BUTTON)s">&nbsp;buttons.
The <i>Remove Module</i> <img src="memory:%(MODULE_REMOVE_BUTTON)s">&nbsp;button will
delete the selected module(s) from the pipeline.</p>
<p>Most pipelines depend on one major step: identifying the objects. In
CellProfiler, the objects you identify are called <i>primary</i>,
<i>secondary</i>, or <i>tertiary</i>:
<ul>
<li><b>IdentifyPrimary</b> modules identify objects without relying on any
information other than a single grayscale input image (e.g., nuclei are
typically primary objects).</li>
<li><b>IdentifySecondaryObjects</b> modules require a grayscale image plus an image
where primary objects have already been identified, because the secondary
objects are determined based on the primary objects (e.g., cells can be
secondary objects when their identification is based on the location of nuclei). </li>
<li><b>IdentifyTertiary</b> modules require images in which two sets of objects have
already been identified (e.g., nuclei and cell regions are used to define the
cytoplasm objects, which are tertiary objects).</li>
</ul></p>
</li>

<li><p><i>Adjust the settings in each module.</i><br>
In the CellProfiler main window, click a module in the pipeline to see its
settings in the settings panel. To learn more about the settings for each
module, select the module in the pipeline and click the <i>Help</i> button to the
right of each setting, or at the bottom of the pipeline panel
for the help for all the settings for that module.</p>
<p>If there is an error with the settings (e.g., a setting refers to an image
that doesn't exist yet),
a <img src="memory:%(SETTINGS_ERROR_ICON)s">&nbsp;icon will appear next to the
module name. If there is a warning (e.g., a special notification attached to a choice of setting),
a <img src="memory:%(SETTINGS_WARNING_ICON)s">&nbsp;icon will appear. Errors
will cause the pipeline to fail upon running, whereas a warning will not. Once
the errors/warnings have been resolved, a <img src="memory:%(SETTINGS_OK_ICON)s">
&nbsp;icon will appear indicating that the module is ready to run.</p>
</li>
<li><p><i>Set your Default Input Folder, Default Output Folder and output filename.</i><br>
For more help, click their nearby <i>Help</i> buttons in the main window. </p></li>

<li><p><i>Click <i>Analyze images</i> to start processing.</i><br>
All of the images in your selected folder(s) will be analyzed using the modules
and settings you have specified. A status window will appear which has the following:
<ul>
<li>A <i>progress bar</i> which gives the elapsed time and estimates the time remaining to
process the full image set.</li>
<li>A <i>pause button</i> <img src="memory:%(RUNSTATUS_PAUSE_BUTTON)s">&nbsp;
which pauses execution and allows you to subsequently
resume the analysis.
<li>A <i>stop button</i> <img src="memory:%(RUNSTATUS_STOP_BUTTON)s">&nbsp;
which cancels execution after prompting you for a place to
save the measurements collected to that point.</li>
<li>A <i>save measurements</i> button <img src="memory:%(RUNSTATUS_SAVE_BUTTON)s">&nbsp;
which will prompt you for a place to
save the measurements collected to that point while continuing the analysis run.</li>
</ul>
At the end of each cycle, CellProfiler saves the measurements in the output file.</p></li>

<li><p><i>Click <i>Start Test Mode</i> to preview results.</i><br>
You can optimize your pipeline by selecting the <i>Test</i> option from
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results, and adjust the module settings on the fly. See
<i>%(TEST_MODE_HELP_REF)s</i> for more details.</p>
</li>
<li><p>Save your project (which includes your pipeline) via <i>File > Save Project</i>.</p>
</li>
</ol>
<p><i>Saving images in your pipeline:</i> Due to the typically high number
of intermediate images produced during processing, images produced during
processing are not saved to the hard drive unless you specifically request it,
using a <b>SaveImages</b> module.</p>
<p><i>Saving data in your pipeline:</i> You can include an <b>Export</b> module to automatically export
data in a format you prefer. See <i>%(USING_YOUR_OUTPUT_REF)s</i> for more details.</p>
""" % globals()

REGEXP_HELP_REF = """
Patterns are specified using
combinations of metacharacters and literal characters. There are a few
classes of metacharacters, partially listed below. Some helpful links follow:
<ul>
<li>A more extensive explanation of regular expressions can be found
<a href="http://www.python.org/doc/2.3/lib/re-syntax.html">here</a></li>
<li>A helpful quick reference can be found <a href="http://www.addedbytes.com/cheat-sheets/regular-expressions-cheat-sheet/">here</a></li>
<li><a href="http://pythex.org/">Pythex</a> provides quick way to test your regular expressions. Here is an
<a href="http://pythex.org/?regex=Channel%5B1-2%5D-%5B0-9%5D%7B2%7D-(%3FP%3CWellRow%3E%5BA-H%5D)-(%3FP%3CWellColumn%3E%5B0-9%5D%7B2%7D)%5C.tif&test_string=Channel1-01-A-01.tif&ignorecase=0&multiline=0&dotall=0&verbose=0">example</a>
to capture information from a common microscope nomenclature.</li>
</ul>

<p>The following metacharacters match exactly one character from its respective set of characters:<br><br>
<table border="1">
<tr bgcolor="#555555"><th><font color="#FFFFFF"><b>Metacharacter</b></font></th><th><font color="#FFFFFF"><b>Meaning</b></font></th></tr>
<tr><td>.</td><td>Any character</td></tr>
<tr><td>[]</td><td>Any character contained within the brackets</td></tr>
<tr><td>[^]</td><td>Any character not contained within the brackets</td></tr>
<tr><td>\w</td><td>A word character [a-z_A-Z0-9]</td></tr>
<tr><td>\W</td><td>Not a word character [^a-z_A-Z0-9]</td></tr>
<tr><td>\d</td><td>A digit [0-9]</td></tr>
<tr><td>\D</td><td>Not a digit [^0-9]</td></tr>
<tr><td>\s</td><td>Whitespace [ \\t\\r\\n\\f\\v]</td></tr>
<tr><td>\S</td><td>Not whitespace [^ \\t\\r\\n\\f\\v]</td></tr>
</table>

<p>The following metacharacters are used to logically group subexpressions
or to specify context for a position in the match. These metacharacters
do not match any characters in the string:<br><br>
<table border="1">
<tr bgcolor="#555555"><th><font color="#FFFFFF"><b>Metacharacter</b></font></th><th><font color="#FFFFFF"><b>Meaning</b></font></th></tr>
<tr><td>( )</td><td>Group subexpression</td></tr>
<tr><td>|</td><td>Match subexpression before or after the |</td></tr>
<tr><td>^</td><td>Match expression at the start of string</td></tr>
<tr><td>$</td><td>Match expression at the end of string</td></tr>
<tr><td>\&lt;</td><td>Match expression at the start of a word</td></tr>
<tr><td>\&gt;</td><td>Match expression at the end of a word</td></tr>
</table>

<p>The following metacharacters specify the number of times the previous
metacharacter or grouped subexpression may be matched:<br><br>
<table border="1">
<tr bgcolor="#555555"><th><font color="#FFFFFF"><b>Metacharacter</b></font></th><th><font color="#FFFFFF"><b>Meaning</b></font></th></tr>
<tr><td>*</td><td>Match zero or more occurrences</td></tr>
<tr><td>+</td><td>Match one or more occurrences</td></tr>
<tr><td>?</td><td>Match zero or one occurrence</td></tr>
<tr><td>{n,m}</td><td>Match between n and m occurrences</td></tr>
</table>

<p>Characters that are not special metacharacters are all treated literally
in a match. To match a character that is a special metacharacter, escape
that character with a '\\'. For example '.' matches any character, so to
match a '.' specifically, use '\.' in your pattern.

Examples:
<ul>
<li><code>[trm]ail</code> matches 'tail' or 'rail' or 'mail'.</li>
<li><code>[0-9]</code> matches any digit between 0 to 9.</li>
<li><code>[^Q-S]</code> matches any character other than 'Q' or 'R' or 'S'.</li>
<li><code>[[]A-Z]</code> matches any upper case alphabet along with square brackets.</li>
<li><code>[ag-i-9]</code> matches characters 'a' or 'g' or 'h' or 'i' or '-' or '9'.</li>
<li><code>[a-p]*</code> matches '' or 'a' or 'aab' or 'p' etc.</li>
<li><code>[a-p]+</code> matches  'a' or 'abc' or 'p' etc.</li>
<li><code>[^0-9]</code> matches any string that is not a number.</li>
<li><code>^[0-9]*$</code> matches either a blank string or a natural number.</li>
<li><code>^-[0-9]+$|^\+?[0-9]+$</code> matches any integer.</li>
</ul>
"""

SPREADSHEETS_DATABASE_HELP = """
<p>The most common form of output for cellular analysis is a <i>spreadsheet<i>, which is an application which
tabulates data values.
CellProfiler can also output data into a <i>database</i>, which is a collection of
data that is stored for retrieval by users. Which format you use will depend on
some of the considerations below:
<ul>
<li><i>Assessibility:</i> Spreadsheet applications are typically designed to allow easy
user interaction with the data, to edit values, make graphs and the like. In contrast, the values in databases are
typically not modified after the fact. Instead, database applications typically allow for viewing a specific data range.</li>
<li><i>Capacity and speed:</i> Databases are designed to hold larger amounts of data than spreadsheets. Spreadsheets may contain
hundreds to a few thousand rows of data, whereas databases can hold mnay millions of rows of data. Due to the high
capacity, accessing a particular portion of data in a database is optimized for speed.</li>
<li><i>Learning curve:</i> The applications that access spreadsheets are usually made for ease-of-use to allow for user edits.
Databases are more sophisticated and are not typically edited or modified; to do so
require knowledge of specialized languages made for this purpose (e.g., MySQL, Oracle, etc).</li>
</ul>
For spreadsheets, the most widely used program to open these files is Microsoft's Excel program.
Since the file is plain text, other editors can also be used, such as
<a href="http://www.libreoffice.org/features/calc/">Calc</a> or
<a href="https://docs.google.com">Google Docs</a>.
For databases, a popular freeware access tool is <a href="https://www.webyog.com/">SQLyog</a>.
</p>
"""

MEMORY_AND_SPEED_HELP = """
<p>If you find that you are running into out-of-memory
errors and/or speed issues associated with your analysis run, we
have detailed a number of solutions on our forum
<a href="http://cellprofiler.org/forum/viewtopic.php?f=14&t=806&p=4490#p4490">FAQ</a>
on this issue. We will continue to add more tips and tricks to this page
over time.</p>
""" % globals()

TEST_MODE_HELP = """
<p>Before starting an analysis run, you can test the pipeline settings on a selected image
cycle using the <i>Test</i> mode option on
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results and adjust the module settings on the fly.</p>

<p>To enter Test mode once you have built a pipeline, choose <i>Test > Start Test Mode</i> from the
menu bar in the main window. At this point, you will see the following features appear:
<ul>
<li>The module view will have a slider bar appearing on the far left.</li>
<li>A Pause icon <img src="memory:%(TESTMODE_GO_ICON)s">&nbsp;
will appear to the left of each module.</li>
<li>A series of buttons will appear at the bottom of the pipeline panel above the
module adjustment buttons.</li>
<li>The grayed-out items in the <i>Test</i> menu will become active, and the
<i>Analyze Images</i> button will become inactive.
</ul>
</p>

<p>You can run your pipeline in Test mode by selecting <i>Test > Step to Next Module</i>
or clicking the <i>Run</i> or <i>Step</i> buttons at the bottom of the pipeline panel.
The pipeline will execute normally, but you will
be able to back up to a previous module or jump to a downstream module, change
module settings to see the results, or execute the pipeline on the image of your choice.
The additional controls allow you to do the following:
<ul>
<li><i>Slider:</i> Start/resume execution of the pipeline at any time by moving the slider. However,
if the selected module depends on objects and/or images
generated by prior modules, you will see an error message indicating that the data has not
been produced yet. To avoid this, it is best to actually run the pipeline up to the module
of interest, and move the slider to modules already executed.
<li><i>Pause:</i> Clicking the pause icon will cause the pipeline test run to halt
execution when that module is reached (the paused module itself is not executed). The icon
changes from <img src="memory:%(TESTMODE_GO_ICON)s">&nbsp;to
<img src="memory:%(TESTMODE_PAUSE_ICON)s">&nbsp;to indicate that a pause has
been inserted at that point.</li>
<li><i>Run:</i> Execution of the pipeline will be started/resumed until
the next module pause is reached. When all modules have been executed for a given image cycle,
execution will stop.</li>
<li><i>Step:</i> Execute the next module (as indicated by the slider location).</li>
<li><i>Next Image:</i> Skip ahead to the next image cycle as determined by the image
order in the Input modules. The slider will automatically return to the
first module in the pipeline.</li>
</ul>
</p>
<p>From the <i>Test</i> menu, you can choose additional options:
<ul>
<li><i>Exit Test Mode:</i> Exit <i>Test</i> mode. Loading a new pipeline or adding/subtracting
modules will also automatically exit test mode.</li>
<li><i>Step to Next Module:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Next Image Set:</i> Step to the next image set in the current image group.</li>
<li><i>Next Image Group:</i> Step to the next group in the image set.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Random Image Set:</i> Randomly select and jump to an image set in the current image group.</li>
<li><i>Choose Image Set:</i> Choose the image set to jump to.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Choose Image Group:</i> Choose an image group to jump to.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Reload Modules Source (enabled only if running from source code):</i> This option will reload
the module source code, so any changes to the code will be reflected immediately.</li>
</ul>
Note that if movies are being loaded, the individual movie is defined as a group automatically.
Selecting <i>Choose Image Group</i> will allow you to choose the movie file, and <i>Choose Image Set</i>
will let you choose the individual movie frame from that file.
<p>%(USING_METADATA_GROUPING_HELP_REF)s</p>
</p>
""" % globals()

RUNNING_YOUR_PIPELINE_HELP = """
Once you have tested your pipeline using Test mode and you are satisfied with the
module settings, you are ready to run the pipeline on your entire set of images. To
do this:
<ul>
<li>Exit Test mode by clicking the "Exit Test Mode" button or selecting <i>Test > Exit Test Mode</i>.</li>
<li>Click the "<img src="memory:%(ANALYZE_IMAGE_BUTTON)s">&nbsp;Analyze Images" button and begin processing your data sets.</li>
</ul>
During the analysis run, the progress will appear in the status bar at the bottom of CellProfiler. It will
show you the total number of image sets, the number of image sets completed, the time elapsed and the approximate
time remaining in the run.
<p>If you need to pause analysis, click the "<img src="memory:%(PAUSE_ANALYSIS_BUTTON)s">&nbsp;Pause" button, then click the
"Resume" button to continue. If you
want to terminate analysis, click the "<img src="memory:%(STOP_ANALYSIS_BUTTON)s">&nbsp;Stop Analysis" button.</p>
<p>If your computer has multiple processors, CellProfiler will take advantage of them by starting multiple copies
of itself to process the image sets in parallel. You can set the number of <i>workers</i> (i.e.,copies of
CellProfiler activated) under <i>File > Preferences...</i></p>
""" % globals()

# The help below contains a Google URL shortener since the URL has a control character that the URL reader doesn't interpret correctly
BATCHPROCESSING_HELP = """
<p>CellProfiler is designed to analyze images in a high-throughput manner.
Once a pipeline has been established for a set of images, CellProfiler
can export batches of images to be analyzed on a computing cluster with the
pipeline. </p>

<p>It is possible to process tens or even hundreds of thousands of
images for one analysis in this manner. We do this by breaking the entire
set of images into separate batches, then submitting each of these batches
as individual jobs to a cluster. Each individual batch can be separately
analyzed from the rest.</p>

<h3>Submitting files for batch processing</h3>

Below is a basic workflow for submitting your image batches to the cluster.
<ol>
<li><i>Create a folder for your project on your cluster.</i> For high-throughput
analysis, it is recommended to create a separate project folder for each run. </li>
<li>Within this project folder, create the following folders (both of which must
be connected to the cluster computing network):
<ul>
<li>Create an input folder, then transfer all of your images to this folder
as the input folder. The input folder must be readable by everyone (or at least your
cluster) because each of the separate cluster computers will read input files from
this folder.
<li>Create an output folder where all your output data will be stored. The
output folder must be writeable by everyone (or at least your cluster) because
each of the separate cluster computers will write output files to this folder.
</ul>
If you cannot create folders and set read/write permissions to these folders (or don't know
how), ask your Information Technology (IT) department for help. </li>

<li>Press the "%(VIEW_OUTPUT_SETTINGS_BUTTON_NAME)s" button. In the panel that appears,
set the Default Input and Default Output Folders
to the <i>images</i> and <i>output</i> folders created above, respectively. The Default Input
Folder setting will only appear if a legacy pipeline is being run.</li>

<li><i>Create a pipeline for your image set.</i> You should test it on a few example
images from your image set (if you are unfamilar with the concept of an image set, please
see the help for the <b>Input</b> modules). The module settings selected for your pipeline will be
applied to <i>all</i> your images, but the results may vary
depending on the image quality, so it is critical to insure that your settings be
robust against your "worst-case" images.<br>
For instance, some images may contain no cells. If this happens, the automatic thresholding
algorithms will incorrectly choose a very low threshold, and therefore "find"
spurious objects. This can be overcome by setting a lower limit on the threshold in
the <b>IdentifyPrimaryObjects</b> module.<br>
The Test mode in CellProfiler may be used for previewing the results of your settings
on images of your choice. Please refer to <i>%(TEST_MODE_HELP_REF)s</i>
for more details on how to use this utility.</li>""" % globals() + \
                       """<li><i>Add the <b>CreateBatchFiles</b> module to the end of your pipeline.</i>
                       This module is needed to resolve the pathnames to your files with respect to
                       your local machine and the cluster computers. If you are processing large batches
                       of images, you may also consider adding <b>ExportToDatabase</b> to your pipeline,
                       after your measurement modules but before the CreateBatchFiles module. This module
                       will export your data either directly to a MySQL/SQLite database or into a set of
                       comma-separated files (CSV) along with a script to import your data into a
                       MySQL database. Please refer to the help for these modules in order learn more
                       about which settings are appropriate.</li>

                       <li><i>Run the pipeline to create a batch file.</i> Click the <i>Analyze images</i>
                       button and the analysis will begin processing locally. Do not be surprised if this initial step
                       takes a while since CellProfiler must first create the entire image set list based
                       on your settings in the <b>Input</b> modules (this process can be sped
                       up by creating your list of images as a CSV and using the <b>LoadData</b> module to load it).
                       With the <b>CreateBatchFiles</b> module in place, the pipeline will not process all
                       the images, but instead will creates a batch file (a file called
                       <i>Batch_data.h5</i>) and save it in the Default Output Folder (Step 1). The advantage of
                       using <b>CreateBatchFiles</b> from the researcher's perspective is that the Batch_data.h5
                       file generated by the module captures all of the data needed to run the analysis. You
                       are now ready to submit this batch file to the cluster to run each of the batches
                       of images on different computers on the cluster.</li>

                       <li><i>Submit your batches to the cluster.</i> Log on to your cluster, and navigate
                       to the directory where you have installed CellProfiler on the cluster. <br>
                       A single batch can be submitted with the following command:<br>
                       <code>
                       ./python CellProfiler.py -p &lt;Default_Output_Folder_path&gt;/Batch_data.h5 -c -r -b -f &lt;first_image_set_number&gt; -l &lt;last_image_set_number&gt;
                       </code>
                       This command submits the batch file to CellProfiler and specifies that CellProfiler run in a
                       batch mode without its user interface to process the pipeline.
                       This run can be modified by using additional options to CellProfiler that
                       specify the following:
                       <ul>
                       <li><code>-p &lt;Default_Output_Folder_path&gt;/Batch_data.h5</code>: The
                       location of the batch file, where &lt;Default_Output_Folder_path&gt; is the
                       output folder path as seen by the cluster computer.</li>
                       <li><code>-c</code>: Run "headless", i.e., without the GUI</li>
                       <li><code>-r</code>: Run the pipeline specified on startup, which is contained in
                       the batch file.
                       <li><code>-b</code>: Do not build extensions, since by this point, they should
                       already be built.</li>
                       <li><code>-f &lt;first_image_set_number&gt;</code>: Start processing with the image
                       set specified, &lt;first_image_set_number&gt;</li>
                       <li><code>-l &lt;last_image_set_number&gt; </code>: Finish processing with the image
                       set specified, &lt;last_image_set_number&gt;</li>
                       </ul>
                       Typically, a user will break a long image set list into pieces and execute each of
                       these pieces using the command line switches, <code>-f</code> and <code>-l</code> to
                       specify the first and last image sets in each job. A full image set would then need
                       a script that calls CellProfiler with these options with sequential image set numbers,
                       e.g, 1-50, 51-100, etc to submit each as an individual job.<br>

                       <p>If you need help in producing the batch commands for submitting your jobs, use the
                       <code>--get-batch-commands</code> along with the <code>-p</code> switch to specify the
                       Batch_data.h5 file output by the CreateBatchFiles module. When specified, CellProfiler
                       will output one line to the terminal per job to be run. This output should be further
                       processed to generate a script that can invoke the jobs in a cluster-computing context.<br>
                       The above notes assume that you are running CellProfiler using our source code (see
                       "Developer's Guide" under Help for more details). If you are using the compiled version,
                       you would replace <code>./python CellProfiler.py</code> with the CellProfiler
                       executable file itself and run it from the installation folder.</p></li>
                       </ol>

                       <p>Once all the jobs are submitted, the cluster will run each batch individually
                       and output any measurements or images specified in the pipeline. Specifying the output filename
                       using the <code>-o</code> switch when
                       calling CellProfiler will also produce an output file containing the measurements
                       for that batch of images in the output folder. Check the output from the batch
                       processes to make sure all batches complete. Batches that fail for transient reasons
                       can be resubmitted.</p>

                       <p>To see documentation for all available arguments to CellProfiler, type <code>CellProfiler.py
                       --help</code> to see a listing.</p>

                       <p>For additional help on batch processing, refer to our
                       <a href = "http://goo.gl/HtTzD">wiki</a> if installing CellProfiler on a Unix system,
                       our <a href="http://goo.gl/WG9doZ">wiki</a> on
                       adapting CellProfiler to a LIMS environment, or post your questions on
                       the CellProfiler <a href = "http://cellprofiler.org/forum/viewforum.php?f=18">CPCluster forum</a>.</p>
                       """ % globals()

RUN_MULTIPLE_PIPELINES_HELP = """
<br>The <b>Run multiple pipelines</b> dialog lets you select several pipelines
which will be run consecutively. Please note the following:
<ul>
<li>CellProfiler 2.1 project files are not currently supported.</li>
<li>Pipelines from CellProfiler 2.0 and lower are supported.</li>
<li>If you want to use a pipeline made using CellProfiler 2.1, then you
need to include the project file list with the pipeline, by selecting <i>Export &gt;
Pipeline...</i>, and under the "Save as type" dropdown, select "CellProfiler
pipeline and file list".</li>
</ul>

<p>You can invoke <b>Run multiple pipelines</b> by selecting it from the file menu.
The dialog has three parts to it:
<br><ul><li><i>File chooser</i>: The file chooser lets you select the pipeline
files to be run. The <i>Select all</i> and <i>Deselect all</i> buttons to
the right will select or deselect all pipeline files in the list. The
<i>Add</i> button will add the pipelines to the pipeline list. You can add
a pipeline file multiple times, for instance if you want to run that pipeline
on more than one input folder.</li>
<li><i>Directory chooser</i>: The directory chooser lets you navigate to
different directories. The file chooser displays all pipeline files in the
directory chooser's current directory.</li>
<li><i>Pipeline list</i>: The pipeline list has the pipelines to be run in
the order that they will be run. Each pipeline has a default input and
output folder and a measurements file. You can change any of these by clicking
on the file name - an appropriate dialog will then be displayed. You can
click the remove button to remove a pipeline from the list</li></ul>
<br>
CellProfiler will run all of the pipelines on the list when you hit the
"OK" button."""

CONFIGURING_LOGGING_HELP = """
<br>CellProfiler prints diagnostic messages to the console by default. You
can change this behavior for most messages by configuring logging. The simplest
way to do this is to use the command-line switch, "-L", to set the log level.
For instance, to show error messages or more critical events, start CellProfiler
like this:
<br>
<code>CellProfiler -L ERROR</code>
<br>
The following is a list of log levels that can be used:
<ul>
<li><b>DEBUG:</b> Detailed diagnostic information</li>
<li><b>INFO:</b> Informational messages that confirm normal progress</li>
<li><b>WARNING:</b> Messages that report problems that might need attention</li>
<li><b>ERROR:</b> Messages that report unrecoverable errors that result in data loss or termination of the current operation.</li>
<li><b>CRITICAL:</b> Messages indicating that CellProfiler should be restarted or is incapable of running.</li></ul>
<br>
You can tailor CellProfiler's logging with much more control using a logging
configuration file. You specify the file name in place of the log level on
the command line, like this:
<br>
<code>CellProfiler -L ~/CellProfiler/my_log_config.cfg</code>
</br>
Files are in the Microsoft .ini format which is grouped into categories enclosed
in square brackets and the key/value pairs for each category.
Here is an example file:
<div><pre>
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
</pre></div>
The above file would print warnings and errors to the console for all messages
but "pipeline statistics" which are configured using the <i>pipelineStatistics</i>
logger are written to a file instead.. The pipelineStatistics logger is the
logger that is used to print progress messages when the pipeline is run. You
can find out which loggers are being used to write particular messages by
printing all messages with a formatter that prints the logger name ("%(name)s").
<br>
The format of the file is described in greater detail
<a href="http://docs.python.org/howto/logging.html#configuring-logging">here</a>.
"""

ACCESSING_OMERO_IMAGES = """
CellProfiler has first-class support for loading images from
<a href="http://www.openmicroscopy.org/site/products/omero">OMERO</a>. The input modules
and the LoadData module can refer to images by URL, for instance, the example pipeline
on the welcome page loads its images from <code>http://cellprofiler.org/ExampleFlyImages</code>.
The first part of a URL (the part before the colon) is the schema. CellProfiler decides
which communication protocol to use, depending on the schema; in the case of the
example on the welcome page, the schema is HTTP and CellProfiler uses the HTTP
protocol to get the image data. For OMERO, the schema that should be used is "omero"
and we use the OMERO client library to fetch and load the data.

<p>OMERO URLs have the form, "omero:iid=". You can find the image IDs using either the
OMERO web client or the
<a href="http://www.openmicroscopy.org/site/support/omero4/downloads">Insight software</a>.
As an example, the screen capture below
indicates that the image, "Channel1-01-A-01.tif", has an IID of 58038:<br>
<img src="memory:%(OMERO_IMAGEID_PIC)s"></p>
<p>At present, manually curating the URL list can be somewhat time-consuming, but we
are planning to develop plug-ins for Insight that will automatically generate these
lists for CellProfiler from within the Insight user interface. The plugin will allow
you to select a screen or plate and export an image set list that can be used with
CellProfiler's LoadData module.</p>

<h3>OMERO login credentials</h3>

CellProfiler will ask you for your OMERO login credentials when you first access an
OMERO URL, either by viewing it from the file list or by loading it in a pipeline.
CellProfiler will create and maintain a session for you based on these credentials
until you close the application. You should only need to enter your credentials once.
To use the "Log into Omero" dialog, enter your server's name or IP address, the port
(usually 4064), your user name and password and press the "Connect" button. The
"Connect" button should turn green and the OK button of the dialog should become
enabled (see below). Press OK to complete the login.<br>
<img src="memory:%(OMERO_LOGIN_PIC)s">
<p>Currently, CellProfiler cannot establish a connection to OMERO when running
headless - to do that, we would need to store the user password where it might be
otherwise visible. We would like to provide a secure mechanism for establishing a
session when headless and would like to work with you to make this work in your
environment; please contact us for further information on how to modify CellProfiler
yourself to do this or with suggestions for us to implement.</p>

<h3>Using OMERO URLs with the Input modules</h3>

The <b>Images</b> module has a file list panel of all of the image files in a project.
This file list supports URLs including OMERO URLs. You can drag URLs from a text document
and drop them into the file list. The URLs do not end with image file extensions (like .TIF),
so you need to change the "Filter images?" setting to "No filtering" to allow the OMERO URLs
to be processed further. You should be able to view the image by double-clicking on it and
you should be able to extract plate, well, site and channel name metadata from each image
using the "Extract from image file headers" method in the <b>Metadata</b> module (press the
"Update Metadata" button after selecting the "Extract from image file headers" method).

If your experiment has more than one image channel, you can use the "ChannelName" metadata
extracted from the OMERO image to create image sets containing all of your image channels.
In the <b>NamesAndTypes</b> module, change the "Assign a name to" setting to "Images
matching rules". For the rule criteria, select "Metadata does have ChannelName matching"
and enter the name that appears under "Channels" in the OMERO Insight browser. Add additional
channels to <b>NamesAndTypes</b> using the "Add another image" button.

<h3>OMERO URLs and LoadData</h3>

The LoadData module reads image sets from a .CSV file. The CSV file has a one-line header
that tells LoadData how to use each of the columns of the file. You can load channels from
a URL by adding a "URL" tag to this header. The OMERO URLs themselves appear in rows below.
For instance, here is a .CSV that loads a DNA and GFP channel:
<pre>
URL_DNA,URL_GFP
omero:iid=58134,omero:iid=58038
omero:iid=58135,omero:iid=58039
omero:iid=58136,omero:iid=58040
</pre>
""" % globals()

MEASUREMENT_NOMENCLATURE_HELP = """
In CellProfiler, measurements are exported as well as stored internally using the
following general nomenclature:
<code><i>MeasurementType_Category_SpecificFeatureName_Parameters</i></code>

<p>Below is the description for each of the terms:
<ul>
<li><code>MeasurementType</code>: The type of data contained in the measurement, which can be
one of three forms:
<ul>
<li><i>Per-image:</i> These measurements are image-based (e.g., thresholds, counts) and
are specified with the name "Image" or with the measurement (e.g.,
"Mean") for per-object measurements aggregated over an image.</li>
<li><i>Per-object:</i> These measurements are per-object and are specified as the name given
by the user to the identified objects (e.g., "Nuclei" or "Cells")</li>
<li><i>Experiment:</i> These measurements are produced for a particular measurement
across the entire analysis run (e.g., Z'-factors), and are specified with the
name "Experiment". See <b>CalculateStatistics</b> for an example.
</ul></li>

<li><code>Category:</code> Typically, this information is specified in one of two ways
<ul>
<li>A descriptive name indicative of the type of measurement taken (e.g., "Intensity")</li>
<li>No name if there is no appropriate <code>Category</code> (e.g., if the <i>SpecificFeatureName</i> is
"Count", no <code>Category</code> is specfied).</li>
</ul></li>

<li><code>SpecificFeatureName:</code> The specific feature recorded by a module (e.g.,
"Perimeter"). Usually the module recording the measurement assigns this name, but
a few modules allow the user to type in the name of the feature (e.g., the
<b>CalculateMath</b> module allows the user to name the arithmetic measurement).</li>

<li><code>Parameters:</code> This specifier is to distinguish measurements
obtained from the same objects but in different ways. For example,
<b>MeasureObjectIntensity</b> can measure intensities for "Nuclei" in two different
images. This specifier is used primarily for data obtained from an individual image
channel specified by the <b>Images</b> module or a legacy <b>Load</b> module
(e.g.,  "OrigBlue" and "OrigGreen") or a
particular spatial scale (e.g., under the category "Texture" or "Neighbors"). Multiple
parameters are separated by underscores.

<p>Below are additional details specific to various modules:
<ul>
<li>Measurements from the <i>AreaShape</i> and <i>Math</i> categories do not have a
<code>Parameter</code> specifier.</li>
<li>Measurements from <i>Intensity</i>, <i>Granularity</i>, <i>Children</i>,
<i>RadialDistribution</i>, <i>Parent</i> and
<i>AreaOccupied</i> categories will have an associated image as the Parameter.</li>
<li><i>Measurements from the <i>Neighbors</i> and <i>Texture</i> category will have a
spatial scale <code>Parameter</code>.</li>
<li>Measurements from the <i>Texture</i> and <i>RadialDistribution</i> categories will
have both a spatial scale and an image <code>Parameter</code>.</li>
</ul>
</li>
</ul>

<p>As an example, consider a measurement specified as <code>Nuclei_Texture_DifferenceVariance_ER_3</code>:
<ul>
<li><code>MeasurementType</code> is "Nuclei," the name given to the detected objects by the user.</li>
<li><code>Category</code> is "Texture," indicating that the module <b>MeasureTexture</b>
produced the measurements.</li>
<li><code>SpecificFeatureName</code> is "DifferenceVariance," which is one of the many
texture measurements made by the <b>MeasureTexture</b> module.
<li>There are two <code>Parameters</code>, the first of which is "ER". "ER" is the user-provided
name of the image in which this texture measurement was made.</li>
<li>The second <code>Parameter</code> is "3", which is the spatial scale at which this texture
measurement was made.</li>
</ul>

<p>See also the <i>Available measurements</i> heading under the main help for many
of the modules, as well as <b>ExportToSpreadsheet</b> and <b>ExportToDatabase</b> modules.
"""

MENU_BAR_FILE_HELP = """
The <i>File</i> menu provides options for loading and saving your pipelines and
performing an analysis run.
<ul>
<li><b>New project:</b> Clears the current project by removing all the analysis
modules and resetting the input modules.</li>
<li><b>Open Project...:</b> Open a previously saved CellProfiler project (<i>.cpproj</i> file)
from your hard drive.</li>
<li><b>Open Recent:</b> Displays a list of the most recent projects used. Select any
one of these projects to load it.</li>
<li><b>Save Project:</b> Save the current project to your hard drive as a <i>.cpproj</i> file.
If it has not been saved previously, you will be asked for a file name to give the
project. Thereafter, any changes to the project will be automatically saved to that filename unless
you choose <b>Save as...</b>.</li>
<li><b>Save Project As...:</b> Save the project to a new file name.</li>
<li><b>Revert to Saved:</b> Restore the currently open project to the settings it had when
it was first opened.</li>
<li><b>Import Pipeline:</b> Gives you the choice of importing a CellProfiler pipeline file from
your hard drive (<i>From file...</i>) or from a web address (<i>From URL...</i>). If importing from
a file, you can point it to a pipeline (<i>.cppipe</i>) file or have it extract the pipeline from
a project (<i>.cpproj</i>) file.</li>
<li><b>Export:</b> You have the choice of exporting the pipeline you are currently working on as a
CellProfiler <i>.cppipe</i> pipeline file (<i>Pipeline</i>), or the image set list as a CSV (<i>Image set listing</i>).</li>
<li><b>Clear Pipeline:</b> Removes all modules from the current pipeline.</li>
<li><b>View Image:</b> Opens a dialog box prompting you to select an image file for
display. Images listed in the File list panel in the <b>Images</b> module can be also be displayed by double-clicking
on the filename.</li>
<li><b>Analyze Images:</b> Executes the current pipeline using the current pipeline
and Default Input and Output folder settings.</li>
<li><b>Stop Analysis:</b> Stop the current analysis run.</li>
<li><b>Run Multiple Pipelines:</b> Execute multiple pipelines in sequential order.
This option opens a dialog box allowing you to select the pipelines you would like
to run, as well as the associated input and output folders. See
the help in the <i>Run Multiple Pipelines</i> dialog for more details.</li>
<li><b>Resume Pipeline:</b> Resume a partially completed analysis run from where it left off.
You will be prompted to choose the output <i>.h5/.mat</i> file
containing the partially complete measurements and the analysis run will pick up
starting with the last cycle that was processed. </li>
<li><b>Preferences...:</b> Displays the Preferences window, where you can change
many options in CellProfiler.</li>
<li><b>Exit:</b></b> End the current CellProfiler session. You will be given the option
of saving your current pipeline if you have not done so.</li>
</ul>"""

MENU_BAR_EDIT_HELP = """
The <i>Edit</i> menu provides options for modifying modules in your current pipeline.
<ul>
<li><b>Undo:</b> Undo the last module modification. You can undo multiple actions by using <i>Undo</i>
repeatedly.</li>
<li><b>Cut:</b> If a module text setting is currently active, remove the selected text.</li>
<li><b>Copy:</b> Copy the currently selected text to the clipboard.</li>
<li><b>Paste:</b> Paste clipboard text to the cursor location, if a text setting is active.</li>
<li><b>Select All:</b> If a text setting is active, select all the text in the setting. If the module
list is active, select all the modules in the module list.</li>
<li><b>Move Module Up:</b> Move the currently selected module(s) up. You
can also use the <img src="memory:%(MODULE_MOVEUP_BUTTON)s">&nbsp;button located
below the Pipeline panel.</li>
<li><b>Move Module Down:</b> Move the currently selected module(s) down. You
can also use the <img src="memory:%(MODULE_MOVEDOWN_BUTTON)s">&nbsp;button located
below the Pipeline panel.</li>
<li><b>Delete Module:</b> Remove the currently selected module(s).
Pressing the Delete key also removes the module(s). You
can also use the <img src="memory:%(MODULE_REMOVE_BUTTON)s">&nbsp;button located
under the Pipeline panel.</li>
<li><b>Duplicate Module:</b> Duplicate the currently selected module(s) in the pipeline.
The current settings of the selected module(s) are retained in the duplicate.</li>
<li><b>Add Module:</b> Select a module from the pop-up list to inster into the current
pipeline. You can also use the <img src="memory:%(MODULE_ADD_BUTTON)s">&nbsp;button located
under the Pipeline panel.</li>
</ul>
You can select multiple modules at once for moving, deletion and duplication
by selecting the first module and using Shift-click on the last module to select
all the modules in between.
""" % globals()

MENU_BAR_WINDOW_HELP = """
The <i>Windows</i> menu provides options for showing and hiding the module display windows.
<ul>
<li><b>Close All Open Windows:</b> Closes all display windows that are currently open.</li>
<li><b>Show All Windows On Run:</b> Select to show all display windows during the
current test run or next analysis run. The display mode icons next to each module
in the pipeline panel will switch to <img src="memory:%(DISPLAYMODE_SHOW_ICON)s">.</li>
<li><b>Hide All Windows On Run:</b> Select to show no display windows during the
current test run or next analysis run. The display mode icons next to each module
in the pipeline panel will switch to <img src="memory:%(DISPLAYMODE_HIDE_ICON)s">.</li>
</ul>
If there are any open windows, the window titles are listed underneath these options. Select any
of these window titles to bring that window to the front.""" % globals()

PARAMETER_SAMPLING_MENU_HELP = """
The <i>Sampling</i> menu is an interface for Paramorama, a plugin for an interactive visualization
program for exploring the parameter space of image analysis algorithms.<p>

<p>This menu option is only shown if specified in the Preferences. Note that if this preference setting
is changed, CellProfiler must be restarted.</p>

<p>Using this plugin will allow you sample a range of setting values in <b>IdentifyPrimaryObjects</b> and
save the object identification results for later inspection. Upon completion, the plug-in will
generate a text file, which specifies: (1) all unique combinations of
the sampled parameter values; (2) the mapping from each combination of parameter values to
one or more output images; and (3) the actual output images.</p>

<p>More information on how to use the plugin can be found
<a href="http://www.comp.leeds.ac.uk/scsajp/applications/paramorama2/">here</a>.</p>

<p><b>References</b>
<ul>
<li>Pretorius AJ, Bray MA, Carpenter AE and Ruddle RA. (2011) "Visualization of
parameter space for image analysis" <i>IEEE Transactions on Visualization and Computer Graphics</i>
17(12), 2402-2411.</li>
</ul></p>
"""

MENU_BAR_DATATOOLS_HELP = """
The <i>Data Tools</i> menu provides tools to allow you
to plot, view, export or perform specialized analyses on your measurements.

<p>Each data tool has a corresponding module with the same name and
functionality. The difference between the data tool and the module is that the
data tool takes a CellProfiler output file (i.e., a <i>.mat or .h5</i> file)
as input, which contains measurements from a previously completed analysis run.
In contrast, a module uses measurements
received from the upstream modules during an in-progress analysis run.</p>

<p>Opening a data tool will present a prompt in which the user is asked to provide
the location of the output file. Once specified, the user is then prompted to
enter the desired settings. The settings behave identically as those from the
corresponding module.</p>

<p>Help for each <i>Data Tool</i> is available under <i>%(DATA_TOOL_HELP_REF)s</i> or the corresponding
module help.</p>""" % globals()

####################################################
#
# Help for the module figure windows
#
####################################################
'''The help menu for the figure window'''

MODULE_DISPLAY_MENU_BAR_HELP = """
From the menu bar of each module display window, you have the following options:
<ul>
<li><b>File</b>
<ul>
<li><i>Save:</i> You can save the figure window to an image file. Note that this will save the entire
contents of the window, not just the individual subplot(s) or images.</li>
<li><i>Save table:</i> This option is only enabled on windows which are displaying
tabular output, such as that from a <b>Measure</b> module. This allows you to
save the tabular data to a comma-delimited file (CSV).</li>
</ul>

<li><b>Tools</b>
<ul>
<li><i>Measure length: </i> Select this option to measure distances within an image window.
If you click on an image and drag, a line will appear
between the two endpoints, and the distance between them shown at the right-most
portion of the bottom panel. This is useful for measuring distances in order to obtain
estimates of typical object diameters for use in <b>IdentifyPrimaryObjects</b>.</li>
</ul>

<li><b>Subplots:</b> If the module display window has multiple subplots (such as
<b>IdentifyPrimaryObjects</b>), the Image Tool options for the individual subplots
are displayed here. See <i>%(IMAGE_TOOLS_HELP_REF)s</i> for more details.
</li>
</ul>
""" % globals()

MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP = """
All figure windows come with a navigation toolbar, which can be used to navigate through the data set.
<ul>
<li><b>Home, Forward, Back buttons:</b>
<i>Home</i> <img src="memory:%(WINDOW_HOME_BUTTON)s"> always takes you to
the initial, default view of your data. The <i>Forward</i> <img src="memory:%(WINDOW_FORWARD_BUTTON)s">&nbsp;
and <i>Back</i> <img src="memory:%(WINDOW_BACK_BUTTON)s">&nbsp;buttons are akin
to the web browser forward and back buttons in that they are used to navigate back
and forth between previously defined views, one step at a time. They will not be
enabled unless you have already navigated within an image else using
the <b>Pan</b> and <b>Zoom</b> buttons, which are used to define new views. </li>

<li><b>Pan/Zoom button:</b>
This button has two modes: pan and zoom. Click the toolbar button
<img src="memory:%(WINDOW_PAN_BUTTON)s"> to activate panning
and zooming, then put your mouse somewhere over an axes, where it will turn into a hand
icon.
<ul>
<li><i>Pan:</i> Press the left mouse button and hold it to pan the figure, dragging it to a new
position. Press Ctrl+Shift with the pan tool to move in one axis only, which one you
have moved farther on.<br>
Keep in mind that that this button will allow you pan outside the bounds of
the image; if you get lost, you can always use the <b>Home</b> to back you back to the
initial view.</li>
<li><i>Zoom:</i> You can zoom in and out of a plot by pressing Ctrl (Mac)
or holding down the right mouse button (Windows) while panning. <br>
Once you're done, the right click menu will pop up when you're done with the action; dismiss it by
clicking off the plot. This is a known bug to be corrected in the next release.</li>
</ul>
</li>

<li><b>Zoom-to-rectangle button:</b> Click this toolbar button <img src="memory:%(WINDOW_ZOOMTORECT_BUTTON)s">&nbsp;
to activate this mode. To zoom in, press the left mouse button and drag in the window
to draw a box around the area you want to zoom in on. When you release the mouse button,
the image is re-drawn to display the specified area. Remember that you can always use
<i>Backward</i> button to go back to the previous zoom level, or use the <i>Home</i>
button to reset the window to the initial view.</li>

<li><b>Save:</b> Click this button <img src="memory:%(WINDOW_SAVE_BUTTON)s">&nbsp;
to launch a file save dialog. You can save the figure window to an image file.
Note that this will save the entire contents of the window, not just the individual
subplot(s) or images.</li>
</ul>
""" % globals()

INTENSITY_MODE_HELP_LIST = """
<ul>
<li><i>Raw:</i> Shows the image using the full colormap range permissible for the
image type. For example, for a 16-bit image, the pixel data will be shown using 0 as black
and 65535 as white. However, if the actual pixel intensities span only a portion of the
image intensity range, this may render the image unviewable. For example, if a 16-bit image
only contains 12 bits of data, the resulting image will be entirely black.</li>
<li><i>Normalized (default):</i> Shows the image with the colormap "autoscaled" to
the maximum and minimum pixel intensity values; the minimum value is black and the
maximum value is white. </li>
<li><i>Log normalized:</i> Same as <i>Normalized</i> except that the color values
are then log transformed. This is useful for when the pixel intensity spans a wide
range of values but the standard deviation is small (e.g., the majority of the
interesting information is located at the dim values). Using this option
increases the effective contrast.</li>
</ul>
"""

INTERPOLATION_MODE_HELP_LIST = """
<ul>
<li><i>Nearest neighbor:</i> Use the intensity of the nearest image pixel when
displaying screen pixels at sub-pixel resolution. This produces a blocky image,
but the image accurately reflects the data.</li>
<li><i>Linear:</i> Use the weighted average of the four nearest image pixels when
displaying screen pixels at sub-pixel resolution. This produces a smoother, more
visually-appealing image, but makes it more difficult to find pixel borders.</li>
<li><i>Cubic: </i> Perform a bicubic interpolation of the nearby image pixels when
displaying screen pixels at sub-pixel resolution. This produces the most
visually-appealing image but is the least faithful to the image pixel values.</li>
</ul>
"""

MODULE_DISPLAY_IMAGE_TOOLS_HELP = """
Right-clicking in an image displayed in a window will bring up a pop-up menu with
the following options:
<ul>
<li><i>Open image in new window:</i> Displays the image in a new display window. This is useful
for getting a closer look at a window subplot that has a small image.</li>
<li><i>Show image histogram:</i> Produces a new window containing a histogram
of the pixel intensities in the image. This is useful for qualitatively examining
whether a threshold value determined by <b>IdentifyPrimaryObjects</b> seems
reasonable, for example. Image intensities in CellProfiler typically range from
zero (dark) to one (bright). If you have an RGB image, the histogram shows the intensity values
for all three channels combined, even if one or more channels is turned off for viewing.</li>
<li><i>Image contrast:</i> Presents three options for displaying the color/intensity values in
the images:
%(INTENSITY_MODE_HELP_LIST)s
</li>
<li><i>Interpolation:</i> Presents three options for displaying the resolution in
the images. This is useful for specifying the amount of detail that you want to be
visible if you zoom in:
%(INTERPOLATION_MODE_HELP_LIST)s
</li>
<li><i>Save subplot:</i> Save the clicked subplot as an image file. If there is only one p
lot in the figure, this option will save that one.</li>
<li><i>Channels:</i> For color images only. You can show any combination of the red,
green, and blue color channels.</li>
</ul>
""" % globals()

FIGURE_HELP = (
    ("Using The Display Window Menu Bar", MODULE_DISPLAY_MENU_BAR_HELP),
    ("Using The Interactive Navigation Toolbar", MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP),
    ("How To Use The Image Tools", MODULE_DISPLAY_IMAGE_TOOLS_HELP))

WORKSPACE_VIEWER_HELP = """
The workspace viewer is a flexible tool that you can use to explore your
images, objects and measurements in test mode. To use the viewer, select
<i>View Workspace</i> from the <i>Test</i> menu after starting test mode. This
will display the CellProfiler Workspace, a window with an image pane to the
left and a panel of controls to the right.

<h3>Key concepts</h3>
<p>
The workspace viewer lets you examine the CellProfiler workspace as you
progress through your pipeline's execution. A pipeline's <i>workspace</i>
is the collection of images, objects and measurements that are produced
by the modules. At the start of the pipeline,
the only things that are available are the images and objects loaded by the
input modules. New images, objects and measurements are added to the workspace
as you step through modules. If you modify a module's setting and re-execute
the module, the images, objects and measurements produced by that module
will be overwritten.</p>

<p>The selected views are persistent across image cycles. That is, you can set up the viewer to view
the workspace at the end of a pipeline cycle, then start a new cycle and
CellProfiler will fill in the images, objects and measurements that you have chosen to
display as they become available. You can also zoom in on a particular region
and change settings and the viewer will remain focused on that region with
the same settings across modules or image cycles.</p>

<p>All elements of the display are configurable, either through the Subplots menu
on the viewer, or through the context menu available by right-clicking on the figure
window.</p>

<h3>Available displays</h3>

<p>A number of displays are available on the right-side of the workspace viewer. You
can add, remove and modify displays of <i>images</i>, <i>objects</i>, <i>masks</i> and
<i>measurements</i>,

<h4>Images</h4>

<p>The workspace viewer can display any image that is available from the input
modules or from modules previously executed. To display a single image, select
it from the <i>Images</i> drop down and check the <i>Show</i> checkbox. Initially,
the image will be displayed in color, using the color shown in the "Color" box.
This color can be changed by clicking on the color box.
<br>
You can add images to the display by clicking the <i>Add Image</i> button. You
can remove images other than the first by hitting the button in the
<i>Remove</i> column. You can toggle the image display using the checkbox in the
<i>Show</i> column. You can also set the interpolation mode by selecting
<i>Interpolation</i> from the <i>Subplots</i> menu. </p>

<h4>Objects</h4>

<p>You can display the objects that have been created or loaded by all modules
that have been executed. To display a set of objects, select them from the
<i>Objects</i> drop-down and check the <i>Show</i> checkbox. You can add
additional objects by pressing the <i>Add Objects</i> button.</p>

<h4>Masks</h4>

<p>You can display the mask for any image produced by any of the modules that
have been executed. Most images are not masked. In these cases, you can
display the mask, but the display will show that the whole image is unmasked.
You can mask an image with the <b>MaskImage</b> or <b>Crop</b> modules.</p>

<p>To display the mask of an image, select it from the <i>Masks</i> dropdown and
check the <i>Show</i> checkbox. You can add additional masks by pressing the
<i>Add Mask</i> button. The options for masks are the same as that for
objects with the addition that you can invert and overlay the mask by choosing
<i>Inverted</i> from the mask's menu; the masked portion will be displayed
in color.</p>

<h4>Measurements</h4>

<p>You can display any measurement produced by any of the modules that have
been executed. Image measurements will be displayed in the title bar above
the image. Object measurements will be displayed centered over the measurement's
object. To display a measurement, select it from the <i>Measurements</i>
drop-down and check the <i>Show</i> checkbox next to the measurement. You can
add a measurement by pressing the <i>Add Measurement</i> button or remove it
by checking the button in the <i>Remove</i> column.</p>

<p>You can configure the font used to display an object measurement, the color
of the text, and the color, transparency and shape of the background behind
the text. To configure the measurement's appearance, press the <i>Font</i>
button to the right of the measurement. Press the <i>Font</i> button in the
<i>Measurement appearance</i> dialog to choose the font and its size, press
the <i>Text color</i> and <i>Background color</i> to change the color used
to display the text and background. Use the <i>Alpha</i> slider to control the
transparency of the background behind the measurement text. The <i>Box shape</i>
drop-down controls the shape of the background box. The <i>Precision</i>
control determines the number of digits displayed to the right of the
decimal point.</p>

<h3>Using the Subplot menu to configure the display</h3>

<p>The following items modify how the display is rendered. You can configure
them through the Subplots menu on the viewer, or through the context menu
available by right-clicking on the figure window.</p>

<h4>Interpolation</h4>

The interpolation mode used to render images, objects and masks is a
configuration option that applies to the entire workspace. Interpolation
controls how the intensities of pixels are blended together. You can set
the interpolation mode by selecting <i>Interpolation</i> from the
<i>Subplots</i> menu. The available modes are:
%(INTERPOLATION_MODE_HELP_LIST)s

<h4>Images</h4>

<p>You can change the way an image is scaled, you can change its display mode
and you can change its color and transparency from the menus. To do this,
select the image from the <i>Subplots</i> menu. The images that are shown
will appear in the menu under the <i>--- Images ---</i> heading. Select the
image you want to configure from the menu to display the options that are
available for that image. There are three categories in the menu, one for
intensity normalization, one for the display mode and one to adjust color
and transparency.</p>

<p>The intensity normalization mode controls how the pixel's intensity value is
translated into the brightness of a pixel on the screen. The modes that are
available are:
%(INTENSITY_MODE_HELP_LIST)s</p>

<p>The <i>Mode</i> controls how pixel intensities are mapped to colors in the
image. You can display each image using the following modes:
<ul>
<li><i>Color:</i> Pixels will have a uniform color which can be selected
by either clicking on the <i>Color</i> button next to the image name or
by choosing the image's <i>Color</i> menu entry.</li>
<li><i>Grayscale:</i> The image will be rendered in shades of gray. The
color choice will have no effect and the image's <i>Color</i> menu entry will
be unavailable.</li>
<li><i>Color map:</i> The image will be rendered using a palette. Your default
color map will be used initially. To change the color map, select
the image's <i>Color</i> menu entry from its menu and choose one of the
color maps from the drop-down. The display will change interactively as you
change the selection, allowing you to see the image as rendered by your choice.
Hit <i>OK</i> to accept the new color map or hit <i>Cancel</i> to use the
color map that was originally selected.</li>
</ul></p>

<p>The image's <i>Alpha</i> menu entry lets you control the image's transparency.
This will let you blend colors when the palettes overlap and choose which
image's intensity has the highest priority. To change the transparency, select
<i>Alpha</i> from the image's menu. Adjust the transparency interactively
using the slider bar and hit <i>OK</i> to accept the new value or <i>Cancel</i>
to restore the value that was originally selected.</p>

<h4>Objects</h4>

<p>You can configure the appearance of objects using the context or <i>Subplots</i>
menu. Choose the objects you wish to configure from the <i>--- Objects ---</i>
list in the menu. You will see configuration menu items for the objects'
display mode, color and alpha value. You can display objects using one of the
following modes:
<ul>
<li><i>Lines:</i> This mode draws a line through the center of each pixel
that borders the background of the object or another object. It does not
display holes in the object. The line is drawn using the color shown in
the <i>Color</i> button next to the objects' name. This option does not
obscure the border pixels, but can take longer to render, especially if there
are a large number of objects.</li>
<li><i>Outlines:</i> This mode displays each pixel in the object's border using
the color shown in the <i>Color</i> button next to the objects' name. This
option will display holes in unfilled objects, but the display obscures the
image underneath the border pixels.</li>
<li><i>Overlay:</i> This mode displays a different color overlay over each
object's pixels. Each object is assigned a color using the default color map
initially. You can choose the color map by
selecting <i>Color</i> from the objects' menu and choosing one of the available
color maps. You can change the transparency of the overlay by choosing
<i>Alpha</i> from the objects' menu.</li>
</ul></p>

""" % globals()

WV_FIGURE_HELP = tuple(list(FIGURE_HELP) + [(
    "How To Use The Workspace Viewer", WORKSPACE_VIEWER_HELP)])

###################################################
#
# Help for the preferences dialog
#
###################################################

TITLE_FONT_HELP = """
Sets the font used in titles above plots displayed in module figure windows."""

TABLE_FONT_HELP = """
Sets the font used in tables displayed in module figure windows."""

DEFAULT_COLORMAP_HELP = """
Specifies the color map that sets the colors for labels and other elements. See this
<a href ="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">
page</a> for pictures of available colormaps."""

WINDOW_BACKGROUND_HELP = """
Sets the window background color of the CellProfiler main window."""

ERROR_COLOR_HELP = """
Sets the color used for the error alerts associated with misconfigured settings and other
errors."""

PLUGINS_DIRECTORY_HELP = """
Chooses the directory that holds dynamically-loaded CellProfiler modules. You
can write your own module and place it in this directory and CellProfiler
will make it available for your pipeline. You must restart CellProfiler
after modifying this setting."""

IJ_PLUGINS_DIRECTORY_HELP = """
Sets the directory that holds ImageJ plugins (for the <b>RunImageJ</b> module).
You can download or write your own ImageJ plugin and place it in this directory
and CellProfiler will make it available for your pipeline. You must restart
CellProfiler after modifying this setting."""

IJ_VERSION_HELP = """
Chooses which version of ImageJ to use in the <b>RunImageJ</b> module. You must restart CellProfiler
after changing this preference for the preference to take effect.<p>
<ul><li><b>ImageJ 1.x:</b> This is a version of ImageJ 1.44 with added support
for <code>&amp;parameter</code> plugin decorations. You should use this if you only have
ImageJ 1.0 plugins.</li>
<li><b>ImageJ 2.0:</b> This is an alpha release of ImageJ 2.0. ImageJ 2.0
has better interoperability with CellProfiler. CellProfiler will display
ImageJ 2.0 plugin settings as part of the RunImageJ module interface and
will let you use regular and masked images in ImageJ 2.0 plugins. ImageJ 2.0
can run ImageJ 1.0 plugins, but there may be incompatibilities.</li></ul>"""

CHECK_FOR_UPDATES_HELP = """
Controls whether CellProfiler looks for updates on startup."""

SHOW_TELEMETRY_HELP = """
Allow limited and anonymous usage statistics and exception reports to be sent
to the CellProfiler team to help improve CellProfiler.
"""

SHOW_STARTUP_BLURB_HELP = """
Controls whether CellProfiler displays an orientation message on startup."""

SHOW_ANALYSIS_COMPLETE_HELP = """
Determines whether CellProfiler displays a message box at the
end of a run. Check this preference to show the message box or uncheck it
to stop display."""

SHOW_EXITING_TEST_MODE_HELP = """
Determines whether CellProfiler displays a message box to inform you
that a change made to the pipeline will cause test mode to end. Check this preference
to show the message box or uncheck it to stop display."""

SHOW_REPORT_BAD_SIZES_DLG_HELP = """
Determines whether CellProfiler will display a warning dialog
if images of different sizes are loaded together in an image set.
Check this preference to show the message box or uncheck it to stop display."""

PRIMARY_OUTLINE_COLOR_HELP = """
Sets the color used for the outline of the object of interest in the
<b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b> and
<b>IdentifyTertiaryObjects</b> displays."""

SECONDARY_OUTLINE_COLOR_HELP = """
Sets the color used for objects other than the ones of interest. In
<b>IdentifyPrimaryObjects</b>, these are the objects that are too small or
too large. In <b>IdentifySecondaryObjects</b> and <b>IdentifyTertiaryObjects</b>,
this is the color of the secondary objects' outline."""

TERTIARY_OUTLINE_COLOR_HELP = """
Sets the color used for the objects touching the image border or image mask
in <b>IdentifyPrimaryObjects</b>."""

INTERPOLATION_MODE_HELP = """
Sets the way CellProfiler displays image pixels. If you choose <i>Nearest</i>,
CellProfiler will display each pixel as a square block of uniform intensity.
This is truest to the data, but the resulting images look blocky and
pixelated. You can choose either <i>Bilinear</i> or <i>Bicubic</i> to see
images where the a bilinear or bicubic spline model has been used to interpolate
the screen pixel value for screen pixels that do not fall exactly in the
center of the image pixel. The result, for bilinear or bicubic interpolation is
an image that is more visually appealing and easier to interpret, but obscures
the true pixel nature of the real data.
"""

INTENSITY_MODE_HELP = """
Sets the way CellProfiler normalizes pixel intensities when displaying.
If you choose "raw", CellProfiler will display a pixel with a value of "1" or
above with the maximum brightness and a pixel with a value of "0" or below
as black. If you choose "normalize", CellProfiler will find the minimum and
maximum intensities in the display image and show pixels at maximum intensity
with the maximum brightness and pixels at the minimum intensity as black. This
can be used to view dim images. If you choose "log", CellProfiler will use
the full brightness range and will use a log scale to scale the intensities.
This can be used to view the image background in more detail.
"""

REPORT_JVM_ERROR_HELP = """
Determines whether CellProfiler will display a warning on startup
if CellProfiler can't locate the Java installation on your computer. Check
this box if you want to be warned. Uncheck this box to hide warnings."""

MAX_WORKERS_HELP = """
Controls the maximum number of <i>workers</i> (i.e., copies of CellProfiler)
that will be started at the outset of an analysis run. CellProfiler uses these
copies to process multiple image
sets in parallel, utilizing the computer's CPUs and memory fully. The default
value is the number of CPUs detected on your computer. Use fewer workers for
pipelines that require a large amount of memory. Use more workers for
pipelines that are accessing image data over a slow connection.

<p>If using the <b>Groups</b> module, only one worker will be allocated to
handle each group. This means that you may have multiple workers created,
but only a subset of them may actually be active, depending on the number of
groups you have.</p>
"""

TEMP_DIR_HELP = """
Sets the folder that CellProfiler uses when storing temporary files. CellProfiler will
create a temporary measurements file for analyses when the user specifies that
a MATLAB measurements file should be created or when the user asks that no
measurements file should be permanently saved. CellProfiler will also save
images accessed by http URL temporarily to disk (but will efficiently access
OMERO image planes directly from the server).
"""

JVM_HEAP_HELP = """
Sets the maximum amount of memory that can be used by the Java virtual machine.
CellProfiler uses Java for loading images, for running ImageJ and for
processing image sets. If you load extremely large images, use the RunImageJ
module extensively or process large image set lists, you can use this option
to start Java with a larger amount of memory. By default, CellProfiler starts
Java with 512 MB, but you can override this by specifying the number of megabytes
to load. You can also start CellProfiler from the command-line with the
--jvm-heap-size switch to get the same effect.
"""

SAVE_PIPELINE_WITH_PROJECT_HELP = """
Controls whether a pipeline and/or file list file is saved whenever the user
saves the project file. Users may find it handy to have the pipeline and/or file
list saved in a readable format, for instance, for version control whenever
the project file is saved. Your project can be restored by importing both
the pipeline and file list, and your pipeline can be run using a different
file list, and your file list can be reused by importing it into a different
project.
Note: When using LoadData, it is not recommended to auto-save the file list,
as this feature only saves the file list existing in the Input Modules, not
LoadData input files.
<ul>
<li><i>Neither:</i> Refrain from saving either file. </li>
<li><i>Pipeline:</i> Save the pipeline, using the project's file name and
path and a .cppipe extension. </li>
<li><i>File list:</i> Save the file list, using the project's file name and
path and a .txt extension.</li>
<li><i>Pipeline and file list:</i> Save both files.</li>
</ul>
"""

BATCHPROFILER_URL_HELP = """
The base URL for BatchProfiler. BatchProfiler is a set of CGI scripts for
running CellProfiler on a GridEngine cluster or compatible. If BatchProfiler
is available, the CreateBatchFiles module can optionally launch a browser
to display the appropriate batch configuration page."""

EACH_PREFERENCE_HELP = (
    ("Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
    ("Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
    ("Title font", TITLE_FONT_HELP),
    ("Table font", TABLE_FONT_HELP),
    ("Default colormap", DEFAULT_COLORMAP_HELP),
    ("Window background", WINDOW_BACKGROUND_HELP),
    ("Error color", ERROR_COLOR_HELP),
    ("Primary outline color", PRIMARY_OUTLINE_COLOR_HELP),
    ("Secondary outline color", SECONDARY_OUTLINE_COLOR_HELP),
    ("Tertiary outline color", TERTIARY_OUTLINE_COLOR_HELP),
    ("Interpolation mode", INTERPOLATION_MODE_HELP),
    ("Intensity mode", INTENSITY_MODE_HELP),
    ("CellProfiler plugins directory", PLUGINS_DIRECTORY_HELP),
    ("ImageJ plugins directory", IJ_PLUGINS_DIRECTORY_HELP),
    # ( "ImageJ version", IJ_VERSION_HELP),
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
PREFERENCES_HELP = """The Preferences allow you to change many options in CellProfiler
<ul>"""
for key, value in enumerate(EACH_PREFERENCE_HELP):
    PREFERENCES_HELP += """<li><b>""" + value[0] + """:</b>""" + value[1] + """</li>"""
PREFERENCES_HELP += """</ul>"""

#########################################################
#
# Help re: projects
#
#########################################################
CREATING_A_PROJECT_CAPTION = "Creating A Project"

INTRODUCTION_TO_PROJECTS_HELP = """
<h3>What is a project?</h3>
<p>In CellProfiler, a <i>project</i> is comprised of two elements:
<ul>
<li>An <i>image file list</i> which is the list of files and their locations that are selected by the user as
candidates for analysis.</li>
<li>The <i>pipeline</i>, which is a series of modules put together used to analyze a set of images.</li>
<li>Optionally, the associated information about the images (<i>metadata</i>). This
information may be part of the images themselves, or imported externally by the user.</li>
</ul>
</p>

<p>The project is the container for image information associated with a CellProfiler analysis. It stores
such details as:
<ul>
<li>What type of image(s) are the input files?</li>
<li>Where are the input images located?</li>
<li>What distinguishes multiple image channels from each other? How are these relationships
represented?</li>
<li>What information about the images and/or experiment is linked to the images, and how?</li>
<li>Are certain groups of images to be processed differently from other groups?</li>
</ul>
By using projects, the above information is stored along with the analysis pipeline and is
available on demand. </p>

<h3>Working with projects</h3>
<h4>Creating a project</h4>
<p>Upon starting CellProfiler, you will be presented with a new, blank project.
At this point, you may start building your project by using the modules located in the "Input
modules" panel on the upper-left. The modules are:
<ul>
<li><b>Images</b>: Assemble the relevant images for analysis (required).</li>
<li><b>Metadata</b>: Associate metadata with the images (optional).</li>
<li><b>NamesAndTypes</b>: Assign names to channels and define their relationship (required).</li>
<li><b>Groups</b>: Define sub-divisions between groups of images for processing (optional).</li>
</ul>
Detailed help for each module is provided by selecting the module and clicking the "?" button on
the bottom of CellProfiler.</p>

<h4>Saving a project</h4>
<p>As you work in CellProfiler, the project is updated automatically, so there is no need to
save it unless you are saving the project to a new name or location. You can always save your
current work to a new project file by selecting <i>File > Save Project As...</i>, which will
save your project, complete with the current image file list and pipeline, to a file with
with the extension <i>.cpproj</i>.</p>

<p>You also have the option of automatically saving the associated pipeline file and the file list
in addition to the project file. See <i>File &gt; Preferences...</i> for more details.</p>

<p>For those interested, some technical details:
<ul>
<li>The <i>.cpproj</i> file stores collected information using the HDF5 format. Documentation on
how measurements are stored and handled in CellProfiler using this format can be found
<a href="https://github.com/CellProfiler/CellProfiler/wiki/Module-Structure-and-Data-Storage-Retrieval#hdf5-measurement-and-workspace-format">here</a>.</li>
<li>All information is cached in the project file after it is computed. It is either
re-computed or retrieved from the cache when an analysis run is started, when
entering Test mode, or when the user requests a refreshed view of the information
(e.g., when a setting has been changed).</li>
</ul>
</p>

<h4>Legacy modules: LoadImages and LoadData</h4>
<p>Historically, two modules were used for project creation: <b>LoadImages</b> and <b>LoadData</b>.
While the approach described above partly supersedes these modules, you have the option
of preserving these modules if you load old pipelines into CellProfiler that contain them;
these pipelines will operate exactly as before.</p>
<p>Alternately, the user can choose to convert these
modules into the project equivalent as closely as possible. Both <b>LoadImages</b> and <b>LoadData</b>
remain accessible via the "Add module" and <img src="memory:%(MODULE_ADD_BUTTON)s">&nbsp;
buttons at the bottom of the pipeline panel.</p>
""" % globals()

SELECTING_IMAGES_HELP = """
<p>Any image analysis project using CellProfiler begins with providing the program with a set of image files
to be analyzed. You can do this by clicking on
the <b>Images</b> module to select it (located in the Input modules panel on the left); this module is responsible for collecting
the names and locations of the files to be processed.</p>

<p>The most straightforward way to provide files to the <b>Images</b> module is
to simply drag-and-drop them from your file manager tool (e.g., Windows Explorer, Finder) onto the file list panel
(the blank space indicated by the text "Drop files and folders here").
Both individual files and entire folders can be dragged onto this panel, and as many folders and files can be
placed onto this panel as needed. As you add files, you will see a listing of the files appear in the panel.</p>

<p>CellProfiler supports a wide variety of image formats, including most of those used in imaging, by using a library called
Bio-Formats; see <a href="http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html">here</a> for the formats available. Some image formats are better
than others for image analysis. Some are <a href="http://www.techterms.com/definition/lossy">"lossy"</a>
(information is lost in the conversion to the format) like most JPG/JPEG files; others are
<a href="http://www.techterms.com/definition/lossless">lossless</a> (no image information is lost). For image analysis purposes, a
lossless format like TIF or PNG is recommended.</p>

<p>If you have a subset of files that you want to analyze from the full list shown in the
panel, you can also filter the files according to a set of rules that you specify. This is useful when, for example, you
have dragged a folder of images onto the file list panel, but the folder contains the images
from one experiment that you want to process along with images from another experiment that you
want to ignore for now. You may specify as many rules as necessary to define the desired
list of images.</p>

<p>For more information on this module and how to configure it for the best performance, please see the detailed help by selecting the
module and clicking the <img src="memory:%(MODULE_HELP_BUTTON)s">&nbsp;button at the bottom of the pipeline panel, or check out
the Input module tutorials on our <a href="http://cellprofiler.org/tutorials.html">Tutorials</a> page.</p>
""" % globals()

CONFIGURE_IMAGES_HELP = """
<p>Once you have used the <b>Images</b> module to produce a list of images to be analyzed, you can use the other
Input modules to define how images are related to one another, give them a memorable name for future reference,
attach additional image information about the experiment, among other things.</p>

<p>After <b>Images</b>, you can use the following Input modules:
<table border="1" cellpadding="10">
    <tr bgcolor="#555555" align="center">
    <th><font color="#FFFFFF"><b>Module</b></font></th>
    <th><font color="#FFFFFF"><b>Description</b></font></th>
    <th><font color="#FFFFFF"><b>Use required?</b></font></th>
    <th><font color="#FFFFFF"><b>Usage notes</b></font></th></tr>
    <tr align="center"><td><b>Metadata</b></td></td><td>Associate image information (metadata) with the images</td><td>No</td>
    <td>With this module, you can extract metadata from various sources and append it to the measurements that your pipeline
    will collect, or use it to define how the images are related to each other. The metadata can come from the image
    filename or location, or from a spreadsheet that you provide. If your assay does not require or have such
    information, this module can be safely skipped.</td></tr>
    <tr align="center"><td><b>NamesAndTypes</b></td><td>Assign names to images and/or channels and define their relationship.</td><td>Yes</td>
    <td>This module gives each image a meaningful name by which modules in the analysis pipeline will refer to it.
    The most common usage for this module is to define a collection of channels that represent a single
    field of view. By using this module, each of these channels will be loaded and processed together for each field of view.</td></tr>
    <tr align="center"><td><b>Groups</b></td><td>Define sub-divisions between groups of images for processing.</td><td>No</td>
    <td>For some assays, you will need the option of further sub-dividing an image set into <i>groups</i> that share a
    common feature. An example of this is a time-lapse movie that consists of individual files; each group of files that
    define a single movie needs to be processed independently of the others. This module allows you to specify what
    distinguishes one group of images from another. If your assay does not require this sort of behavior, this module
    can be safely skipped.</td></tr>
</table>
</p>
<p>For more information on these modules and how to configure them for the best performance, please see the detailed help by selecting the
module and clicking the <img src="memory:%(MODULE_HELP_BUTTON)s">&nbsp;button at the bottom of the pipeline panel, or check out
the Input module tutorials on our <a href="http://cellprofiler.org/tutorials.html">Tutorials</a> page.</p>
""" % globals()

LOADING_IMAGE_SEQUENCES_HELP = """
<h3>Introduction</h3>
In this context, the term <i>image sequence</i> is used to refer to a collection of images
from a time-lapse assay (movie), a three-dimensional (3-D) Z-stack assay, or both.
This section will teach you how to
load these collections in order to properly represent your data for processing.
<h3>Sequences of individual files</h3>
<p>For some microscopes, the simplest method of capturing image sequences is to simply acquire them as a series of
individual image files, where each image file represents a single timepoint and/or Z-slice.
Typically, the image filename reflects the timepoint or Z-slice,
such that the alphabetical image listing corresponds to the proper sequence, e.g., <i>img000.png</i>, <i>img001.png</i>,
<i>img002.png</i>, etc</p>. It is also not uncommon to store the movie such that one movie's worth of files is stored
in a single folder.
<p><i>Example:</i> You have a time-lapse movie of individual files set up as follows:</p>
<ul>
    <li>Three folders, one for each image channel, named <i>DNA</i>, <i>actin</i> and <i>phase</i>.</li>
    <li>In each folder, the files are named as follows:
        <ul>
            <li><i>DNA</i>: calibrate2-P01.001.TIF, calibrate2-P01.002.TIF,..., calibrate2-P01.287.TIF</li>
            <li><i>actin</i>: calibrated-P01.001.TIF, calibrated-P01.002.TIF,..., calibrated-P01.287.TIF</li>
            <li><i>phase</i>: phase-P01.001.TIF, phase-P01.002.TIF,..., phase-P01.287.TIF</li>
        </ul>where the file names are in the format <i>&lt;Stain&gt;-&lt;Well&gt;.&lt;Timepoint&gt;.TIF</i>.
    </li>
    <li>There are 287 timepoints per movie, and a movie of the 3 channels above is acquired from each well
    in a multi-well plate.</li>
</ul>
<p>In this case, the procedure to set up the input modules to handle these files is as follows:</p>
<ul>
    <li>In the <b>Images</b> module, drag-and-drop your folders of images into the File list panel. If necessary, set
    your rules accordingly in order to filter out any files that are not part of a movie sequence.
        <p>In the above example, you would drag-and-drop the <i>DNA</i>, <i>actin</i> and <i>phase</i> folders into
        the File list panel.</p>
    </li>
    <li>In the <b>Metadata</b> module, check the box to enable metadata extraction. The key step here is to obtain the
    metadata tags necessary to do two things:
        <ul>
            <li>Distinguish the movies from each other. This information is typically encapsulated in the filename
            and/or the folder name.</li>
            <li>For each movie, distinguish the timepoints from each other and ensure their proper ordering. This
            information is usually contained in the filename.</li>
        </ul>To accomplish this, do the following:
        <ul>
            <li>Select "{X_MANUAL_EXTRACTION}" or "{X_IMPORTED_EXTRACTION}" as the metadata extraction method. You will
            use these to extract the movie and timepoint tags from the images.</li>
            <li>Use "{X_MANUAL_EXTRACTION}" to create a regular expression to extract the metadata from the filename
            and/or path name.</li>
            <li>Or, use "{X_IMPORTED_EXTRACTION}" if you have a comma-delimited file (CSV) of the necessary metadata
            columns (including the movie and timepoint tags) for each image. Note that microscopes rarely produce
            such a file, but it might be worthwhile to write scripts to create them if you do this frequently.</li>
        </ul>If there are multiple channels for each movie, this step may need to be performed for each channel.
        <p>In this example, you could do the following:</p>
        <ul>
            <li>Select "{X_MANUAL_EXTRACTION}" as the method, "From file name" as the source, and
            <code>.*-(?P&lt;Well&gt;[A-P][0-9]{{2}})\.(?P&lt;Timepoint&gt;[0-9]{{3}})</code> as the regular expression.
            This step will extract the well ID and timepoint from each filename.</li>
            <li>Click the "Add" button to add another extraction method.</li>
            <li>In the new group of extraction settings, select "{X_MANUAL_EXTRACTION}" as the method, "From folder
            name" as the source, and <code>.*[\\/](?P&lt;Stain&gt;.*)[\\/].*$</code> as the regular expression. This
            step will extract the stain name from each folder name.</li>
            <li>Click the "Update" button below the divider and check the output in the table to confirm that the
            proper metadata values are being collected from each image.</li>
        </ul>
    </li>
    <li>In the <b>NamesAndTypes</b> module, assign the channel(s) to a name of your choice. If there are multiple
    channels, you will need to do this for each channel.<br>
        For this example, you could do the following:
        <ul>
            <li>Select "Assign images matching rules".</li>
            <li>Make a new rule <code>[Metadata][Does][Have Stain matching][actin]</code> and name it
            <i>OrigFluor</i>.</li>
            <li>Click the "Add" button to define another image with a rule.</li>
            <li>Make a new rule <code>[Metadata][Does][Have Stain matching][DNA]</code> and name it
            <i>OrigFluo2</i>.</li>
            <li>Click the "Add" button to define another image with a rule.</li>
            <li>Make a new rule <code>[Metadata][Does][Have Stain matching][phase]</code> and name it
            <i>OrigPhase</i>.</li>
            <li>In the "Image set matching method" setting, select "Metadata".</li>
            <li>Select "Well" for the <i>OrigFluor</i>, <i>OrigFluo2</i>, and <i>OrigPhase</i> channels.</li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right to add another row, and
            select "Timepoint" for each channel.</li>
            <li>Click the "Update" button below the divider to view the resulting table and confirm that the proper
            files are listed and matched across the channels. The corresponding well and frame for each channel should
            now be matched to each other.</li>
        </ul>
    </li>
    <li>In the <b>Groups</b> module, enable image grouping for these images in order to select the metadata that
    defines a distinct movie of data.<br>
        For the example above, do the following:
        <ul>
            <li>Select "Well" as the metadata category.</li>
            <li>The tables below this setting will update themselves, and you should be able to visually confirm that
            each well is defined as a group, each with 287 frames' worth of images.</li>
        </ul>Without this step, CellProfiler would not know where one movie ends and the next one begins, and would
        process the images in all movies together as if they were a single movie. This would result in, for example,
        the TrackObjects module attempting to track cells from the end of one movie to the start of the next movie.
    </li>
</ul>
<p>If your images represent a 3D image, you can follow the above example to process your data. It is important to note,
however, that CellProfiler will analyze each Z-slice individually and sequentially. Whole volume (3D image) processing
is supported for single-channel .TIF stacks. Splitting image channels and converting image sets into .TIF stacks can be
done using another software application, like FIJI.</p>
<h3>Basic image sequences consisting of a single file</h3>
<p>Another common means of storing time-lapse or Z-stack data is as a single file containing frames. Examples of this
approach include image formats such as:</p>
<ul>
    <li>Multi-frame TIF</li>
    <li>Metamorph stack: STK</li>
    <li>Evotec/PerkinElmer Opera Flex</li>
    <li>Zeiss ZVI, LSM</li>
    <li>Standard movie formats: AVI, Quicktime MOV, etc</li>
</ul>CellProfiler uses the Bio-Formats library for reading various image formats. For more details on supported files,
see this <a href="http://www.openmicroscopy.org/site/support/bio-formats4/supported-formats.html">webpage</a>. In
general, we recommend saving stacks and movies in .TIF format.
<p><i>Example:</i> You have several image stacks representing 3D structures in the following format:</p>
<ul>
    <li>The stacks are saved in .TIF format.</li>
    <li>Each stack is a single-channel grayscale image.</li>
    <li>Your files have names like IMG01_CH01.TIF, IMG01_CH02.TIF, ... IMG01_CH04.TIF and IMG02_CH01.TIF,
    IMG02_CH02.TIF, etc, where IMG01_CH01.TIF designates channel 1 from image 1, IMG01_CH02.TIF designates channel 2
    from image 1, and IMG02_CH01.TIF designates channel 1 from image 2.</li>
</ul>
<p>You would like to process each stack as a single image, not as a series of 2D images. In this case, the procedure
to set up the input modules to handle these files is as follows:</p>
<ul>
    <li>In the <b>Images</b> module, drag-and-drop your folders of images into the File list panel. If necessary, set
    your rules accordingly in order to filter out any files that are not images to be processed.<br>
    In the above example, you would drag-and-drop the .TIF files into the File list panel.</li>
    <li>In the <b>NamesAndTypes</b> module, select "Yes" for "Data is 3D". You should also provide the relative X, Y,
    and Z pixel sizes of your images. X and Y will be determined by the camera and objective you used to capture your
    images. Your Z size represents the spacing of your Z-series. In most cases, the X and Y pixel size will be the same.
    You can divide the Z size by X or Y to get a relative value, with X = Y = 1. CellProfiler will use this
    information to correctly compute filter sizes and shape features, for example.<br>
    Additionally assign each channel to a name of your choice. You will need to do this for
    each channel. For this example, you could do the following:
        <ul>
            <li>Select "Assign images matching rules".</li>
            <li>Make a new rule <code>[File][Does][Contain][CH01]</code></li>
            <li>Provide a descriptive name for the channel, e.g., <i>DAPI</i>.</li>
            <li>Click the "Add another image" button to define a second image with a set of rules.</li>
            <li>Make a new rule <code>[File][Does][Contain][CH02]</code></li>
            <li>Provide a descriptive name for the channel <i>GFP</i>.</li>
            <li>Click the "Update" button below the divider to confirm that the proper images are listed and
            matched across the channels. All file names ending in CH01.TIF should be matched together.</li>
        </ul>
    </li>
</ul>
<p><i>Example:</i> You have two image stacks in the following format:</p>
<ul>
    <li>The stacks are Opera's FLEX format.</li>
    <li>Each FLEX file contains 8 fields of view, with 3 channels at each site (DAPI, GFP, Texas Red).</li>
    <li>Each channel is in grayscale format.</li>
</ul>
<p>In this case, the procedure to set up the input modules to handle these files is as follows:</p>
<ul>
    <li>In the <b>Images</b> module, drag-and-drop your folders of images into the File list panel. If necessary, set
    your rules accordingly in order to filter out any files that are not images to be processed.<br>
    In the above example, you would drag-and-drop the FLEX files into the File list panel.</li>
    <li>In the <b>Metadata</b> module, enable metadata extraction in order to obtain metadata from these files. The key
    step here is to obtain the necessary metadata tags to do two things:
        <ul>
            <li>Distinguish the stacks from each other. This information is contained as the file itself, that is, each
            file represents a different stack.</li>
            <li>For each stack, distinguish the frames from each other. This information is usually contained in the
            image's internal metadata, in contrast to the image sequence described above.</li>
        </ul>To accomplish this, do the following:
        <ul>
            <li>Select "{X_AUTOMATIC_EXTRACTION}" as the metadata extraction method. In this case, CellProfiler will
            extract the requisite information from the metadata stored in the image headers.</li>
            <li>Click the "Update metadata" button. A progress bar will appear showing the time elapsed; depending on
            the number of files present, this step may take a while to complete.</li>
            <li>Click the "Update" button below the divider.</li>
            <li>The resulting table should show the various metadata contained in the file. In this case, the relevant
            information is contained in the <i>C</i> and <i>Series</i> columns. In the figure shown, the <i>C</i>
            column shows three unique values for the channels represented, numbered from 0 to 2. The <i>Series</i>
            column shows 8 values for the slices collected in each stack, numbered from 0 to 7, followed by the slices
            for other stacks.</li>
        </ul>
    </li>
    <li>In the <b>NamesAndTypes</b> module, assign the channel to a name of your choice. If there are multiple
    channels, you will need to do this for each channel. For this example, you could do the following:
        <ul>
            <li>Select "Assign images matching rules".</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][0]</code></li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rules underneath.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>. This combination tells CellProfiler not to treat
            the image as a single file, but rather as a series of frames.</li>
            <li>Name the image <i>DAPI</i>.</li>
            <li>Click the "Add another image" button to define a second image with a set of rules.</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][1]</code></li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rules underneath.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the image <i>GFP</i>.</li>
            <li>Click the "Add another image" button to define a third image with a set of rules.</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][2]</code></li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rules underneath.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the image <i>TxRed</i>.</li>
            <li>In the "Image set matching method" setting, select "Metadata".</li>
            <li>Select "FileLocation" for the DAPI, GFP and TxRed channels. The FileLocation metadata tag identifies
            the individual stack, and selecting this parameter ensures that the channels are first matched within each
            stack, rather than across stacks.</li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp; button to the right to add another row, and
            select <i>Series</i> for each channel.</li>
            <li>Click the "Update" button below the divider to confirm that the proper image slices are listed and
            matched across the channels. The corresponding <i>FileLocation</i> and <i>Series</i> for each channel
            should now be matched to each other.</li>
        </ul>
    </li>
    <li>In the <b>Groups</b> module, select the metadata that defines a distinct image stack. For the example above, do
    the following:
        <ul>
            <li>Select "FileLocation" as the metadata category.</li>
            <li>The tables below this setting will update themselves, and you should be able to visually confirm that
            each of the two image stacks are defined as a group, each with 8 slices' worth of images.</li>
        </ul>Without this step, CellProfiler would not know where one stack ends and the next one begins, and would
        process the slices in all stacks together as if they were constituents of only one stack.
    </li>
</ul>
<p><i>Example:</i> You have four Z-stacks in the following format:</p>
<ul>
    <li>The stacks are in Zeiss' CZI format.</li>
    <li>Each stack consists of a number of slices with 4 channels (DAPI, GFP, Texas Red and Cy3) at each slice.</li>
    <li>One stack has 9 slices, two stacks have 7 slices and the fourth has 12 slices. Even though the stacks were
    collected with differing numbers of slices, the pipeline to be constructed is intended to analyze all stacks in the
    same manner.</li>
    <li>Each slice is in grayscale format.</li>
</ul>
<p>In this case, the procedure to set up the input modules to handle these this file is as follows:</p>
<ul>
    <li>In the <b>Images</b> module, drag-and-drop your folders of images into the File list panel. If necessary, set
    your rules accordingly in order to filter out any files that are not images to be processed.<br>
    In the above example, you would drag-and-drop the CZI files into the File list panel. In this case, the default
    "Images only" filter is sufficient to capture the necessary files.</li>
    <li>In the <b>Metadata</b> module, enable metadata extraction in order to obtain metadata from these files. The key
    step here is to obtain the metadata tags necessary to do two things:
        <ul>
            <li>Distinguish the stacks from each other. This information is contained as the file itself, that is, each
            file represents a different stack.</li>
            <li>For each stack, distinguish the z-planes from each other, ensuring proper ordering. This information
            is usually contained in the image file's internal metadata.</li>
        </ul>To accomplish this, do the following:
        <ul>
            <li>Select "{X_AUTOMATIC_EXTRACTION}" as the metadata extraction method. In this case, CellProfiler will
            extract the requisite information from the metadata stored in the image headers.</li>
            <li>Click the "Update metadata" button. A progress bar will appear showing the time elapsed; depending on
            the number of files present, this step may take a while.</li>
            <li>Click the "Update" button below the divider.</li>
            <li>The resulting table should show the various metadata contained in the file. In this case, the relevant
            information is contained in the C and Z columns. The <i>C</i> column shows four unique values for the
            channels represented, numbered from 0 to 3. The <i>Z</i> column shows nine values for the slices
            represented from the first stack, numbered from 0 to 8.</li>
            <li>Of note in this case, for each file there is a single row summarizing this information. The
            <i>sizeC</i> column reports a value of 4 and <i>sizeZ</i> column shows a value of 9. You may need to scroll
            down the table to see this summary for the other stacks.</li>
        </ul>
    </li>
    <li>In the <b>NamesAndTypes</b> module, assign the channel(s) to a name of your choice. If there are multiple
    channels, you will need to do this for each channel.
        <p>For the above example, you could do the following:</p>
        <ul>
            <li>Select "Assign images matching rules".</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][0]</code></li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rule options.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the image <i>DAPI</i>.</li>
            <li>Click the "Add another image" button to define a second image with a set of rules.</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][1]</code></li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rule options.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the second image <i>GFP</i>.</li>
            <li>Click the "Add another image" button to define a third image with a set of rules.</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][2]</code>.</li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rule options.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the third image <i>TxRed</i>.</li>
            <li>Click the "Add another image" button to define a fourth image with set of rules.</li>
            <li>Make a new rule <code>[Metadata][Does][Have C matching][3]</code>.</li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right of the rule to add another
            set of rule options.</li>
            <li>Add the rule <code>[Image][Is][Stack frame]</code>.</li>
            <li>Name the fourth image <i>Cy3</i>.</li>
            <li>In the "Image set matching method" setting, select "Metadata".</li>
            <li>Select "FileLocation" for the <i>DAPI</i>,<i>GFP</i>,<i>TxRed</i>, and <i>Cy3</i>channels. The
            <i>FileLocation</i> identifies the individual stack, and selecting this parameter insures that the channels
            are matched within each stack, rather than across stacks.</li>
            <li>Click the <img src="memory:{MODULE_ADD_BUTTON}">&nbsp;button to the right to add another row, and
            select "Z" for each channel.</li>
            <li>Click "Update table" to confirm the channel matching. The corresponding <i>FileLocation</i> and
            <i>Z</i> for each channel should be matched to each other.</li>
        </ul>
        <p></p>
    </li>
    <li>In the <b>Groups</b> module, select the metadata that defines a distinct image stack. For the example above, do
    the following:
        <ul>
            <li>Select "FileLocation" as the metadata category.</li>
            <li>The tables below this setting will update themselves, and you should be able to visually confirm that
            each of the four image stacks are defined as a group, with 9, 7, 7 and 12 slices' worth of images.</li>
        </ul>Without this step, CellProfiler would not know where one stack ends and the next one begins, and would
        process the slices in all stacks together as if they were constituents of only one stack.
    </li>
</ul>
""".format(**{
    "MODULE_ADD_BUTTON": MODULE_ADD_BUTTON,
    "X_AUTOMATIC_EXTRACTION": X_AUTOMATIC_EXTRACTION,
    "X_IMPORTED_EXTRACTION": X_IMPORTED_EXTRACTION,
    "X_MANUAL_EXTRACTION": X_MANUAL_EXTRACTION
})

#########################################################
#
# Misc. help
#
#########################################################

'''The help to be displayed if someone asks for help on a module but none is selected'''
HELP_ON_MODULE_BUT_NONE_SELECTED = (
    "The help button can be used to obtain help for the currently selected module "
    "in the pipeline panel on the left side of the CellProfiler interface.\n\n"
    "You do not have any modules in the pipeline, yet. Add a module to the "
    'pipeline using the "+" button or by using File > Load Pipeline.')

HELP_ON_MEASURING_DISTANCES = """To measure distances in an open image, use the "Measure
length" tool under <i>Tools</i> in the display window menu bar. If you click on an image
and drag, a line will appear between the two endpoints, and the distance between them shown at the right-most
portion of the bottom panel."""

HELP_ON_PIXEL_INTENSITIES = """To view pixel intensities in an open image, use the
pixel intensity tool which is available in any open display window. When you move
your mouse over the image, the pixel intensities will appear in the bottom bar of the display window."""

HELP_ON_FILE_LIST = """The <i>File List</i> panel displays the image files
that are managed by the <b>Images</b>, <b>Metadata</b>, <b>NamesAndTypes</b>
and <b>Groups</b> modules. You can drop files and directories into this window
or use the <i>Browse...</i> button to add files to the list. The context menu
for the window lets you display or remove files and lets you remove folders.
<br>
The buttons and checkbox along the bottom have the following functions:<br>
<ul><li><i>Browse...</i>: Browse for files and folders to add.</li>
<li><i>Clear</i>: Clear all entries from the File list</li>
<li><i>Show files excluded by filters</i>: <i>(Only shown if filtered based on rules is selected)</i>
Check this to see all files in the
list. Uncheck it to see only the files that pass the rules criteria in the
<b>Images</b> module.</li>
<li><i>Expand tree</i>: Expand all of the folders in the tree</li>
<li><i>Collapse tree</i>: Collapse the folders in the tree</li></ul>
"""

FILTER_RULES_BUTTONS_HELP = """
Clicking the rule menus shows you all the file <i>attributes</i>, <i>operators</i> and <i>conditions</i> you can specify to narrow down
the image list.
<ol>
<li>For each rule, first select the <i>attribute</i> that the rule is to be based on. For example, you can select "File"
to define a rule that will filter files on the basis of their filename. </li>
<li>The <i>operator</i> drop-down is then updated with operators applicable to the attribute you selected. For example, if you select
"File" as the attribute, the operator
menu includes text operators such as <i>Contain</i> or <i>Starts with</i>. On the other hand, if you
select "Extension" as the attribute, you can choose the logical operators "Is" or "Is not" from the menu.</li>
<li>In the operator drop-down menu, select the operator you want to use. For example,
if you want to match data exactly, you may want the "Exactly match" or the "Is" operator. If you want the
condition to be more loose, select an operator such as "Contains".</li>
<li>Use the <i>condition</i> box to type the condition you want to match. The more
you type, the more specific the condition is.
<ul>
<li>As an example, if you create a new filter and select
<i>File</i> as the attribute, then select "Does" and "Contain" as the operators, and type "Channel" as the condition,
the filter finds all files that include the text "Channel", such as "Channel1.tif" "Channel2.jpg", "1-Channel-A01.BMP" and so on.</li>
<li>If you select "Does" and "Start with" as the operators and "Channel1" in the Condition box,
the rule will includes such files as "Channel1.tif" "Channel1-A01.png", and so on.</li></ul>
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:%(IMAGES_USING_RULES_ICON)s"></td></tr>
</table>
</li>
You can also create regular expressions (an advanced syntax for pattern matching; see <a href="#regexp">below</a>) in order to select particular files.
</ol>
<p>To add another rule, click the plus  buttons to the right of each rule. Subtract an existing rule by clicking the
minus button.</p>
<p>You can also link a set of rules by choosing the logical expression <i>All</i> or <i>Any</i>. If you use
<i>All</i> logical expression, all the rules must be true for a file to be included in the File list. If
you use the <i>Any</i> option, only one of the conditions has to be met for a file to be included.</p>
<p>If you want to create more complex rules (e.g, some criteria matching all rules and others matching any),
you can create sets of rules, by clicking the ellipsis button (to the right of the plus button).
Repeat the above steps to add more rules to the filter until you have
all the conditions you want to include.</p>

<a name="regexp"><h5>Details on regular expressions</h5></a>
<p>A <i>regular expression</i> is a general term refering to a method of searching for pattern matches in text. There is a high
learning curve to using them, but are quite powerful once you understand the basics.</p>
<p>%(REGEXP_HELP_REF)s</p>
""" % globals()

#########################################################
#
# Plate viewer help
#
#########################################################
PLATEVIEWER_HELP = """<h1>Plate Viewer help</h1>
<p>The plate viewer is a data tool that displays the images in your
experiment in plate format. Your project must define an image set list with
metadata annotations for the image's well and, optionally its plate and site.
The plate viewer will then group your images by well and display a plate map
for you. If you have defined a plate metadata tag (with the name, "Plate"),
the plate viewer will group your images by plate and display a choice box
that lets you pick the plate to display.
<p>
Click on a well to see the images for that well. If you have more than one site
per well and have site metadata (with the name, "Site"), the plate viewer will
tile the sites when displaying, and the values under "X" and "Y" determine the
position of each site in the tiled grid.
<p>
The values for "Red", "Green", and "Blue" in each row are brightness multipliers-
changing the values will determine the color and scaling used to display each
channel.  "Alpha" determines the weight each channel contributes to the summed image.
"""

#########################################################
#
# The top-level of help - used when building the HTML manual
#
#########################################################

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
