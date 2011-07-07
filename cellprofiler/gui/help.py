""" help.py - contains menu structures for help menus in CP

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

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

import logging
import os
import sys
import cellprofiler.icons
from cellprofiler.utilities.relpath import relpath

logger = logging.getLogger(__name__)

#For some reason, Adobe doesn't like using absolute paths to assemble the PDF.
#Also, Firefox doesn't like displaying the HTML image links using abs paths either.
#So I have use relative ones. Should check this to see if works on the 
#compiled version
try:
    path = relpath(cellprofiler.icons.get_builtin_images_path())
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

LOCATION_REFRESH_BUTTON = os.path.join(path,'folder_refresh.png')
LOCATION_BROWSE_BUTTON = os.path.join(path,'folder_browse.png')
LOCATION_CREATE_BUTTON = os.path.join(path,'folder_create.png')

LOCATION_MODULE_HELP_BUTTON = os.path.join(path,'module_help.png')
LOCATION_MODULE_MOVEUP_BUTTON = os.path.join(path,'module_moveup.png')
LOCATION_MODULE_MOVEDOWN_BUTTON = os.path.join(path,'module_movedown.png')
LOCATION_MODULE_ADD_BUTTON = os.path.join(path,'module_add.png')
LOCATION_MODULE_REMOVE_BUTTON = os.path.join(path,'module_remove.png')

LOCATION_TESTMODE_PAUSE_ICON = os.path.join(path,'IMG_PAUSE.png')
LOCATION_TESTMODE_GO_ICON = os.path.join(path,'IMG_GO.png')

LOCATION_DISPLAYMODE_SHOW_ICON = os.path.join(path,'IMG_EYE.png')
LOCATION_DISPLAYMODE_HIDE_ICON = os.path.join(path,'IMG_CLOSED_EYE.png')

LOCATION_SETTINGS_OK_ICON = os.path.join(path,'IMG_OK.png')
LOCATION_SETTINGS_ERROR_ICON = os.path.join(path,'IMG_ERROR.png')
LOCATION_SETTINGS_WARNING_ICON = os.path.join(path,'IMG_WARN.png')

LOCATION_RUNSTATUS_PAUSE_BUTTON = os.path.join(path,'status_pause.png')
LOCATION_RUNSTATUS_STOP_BUTTON  = os.path.join(path,'status_stop.png')
LOCATION_RUNSTATUS_SAVE_BUTTON  = os.path.join(path,'status_save.png')

LOCATION_WINDOW_HOME_BUTTON = os.path.join(path,'window_home.png')
LOCATION_WINDOW_BACK_BUTTON  = os.path.join(path,'window_back.png')
LOCATION_WINDOW_FORWARD_BUTTON  = os.path.join(path,'window_forward.png')
LOCATION_WINDOW_PAN_BUTTON  = os.path.join(path,'window_pan.png')
LOCATION_WINDOW_ZOOMTORECT_BUTTON  = os.path.join(path,'window_zoom_to_rect.png')
LOCATION_WINDOW_SAVE_BUTTON  = os.path.join(path,'window_filesave.png')


####################################################
#
# Module help specifics for repeated use
#
####################################################
BATCH_PROCESSING_HELP_REF = """Help > Using CellProfiler > Other Features > Batch Processing"""
TEST_MODE_HELP_REF = """Help > Using CellProfiler > Testing Your Pipeline"""
METADATA_HELP_REF = """Help > Using CellProfiler > How Data Is Handled > Using Metadata In CellProfiler"""
IMAGE_TOOLS_HELP_REF = """"Help > How To Use The Image Tools"""
METADATA_GROUPING_HELP_REF = """Help > Using CellProfiler > How Data Is Handled > Image Grouping """
DATA_TOOL_HELP_REF = """Help > Data Tool Help """

USING_METADATA_HELP_REF = """ 
Please see <b>LoadImages</b>, <b>LoadData</b>, or <i>%(METADATA_HELP_REF)s</i> 
for more details on obtaining, extracting, and using metadata tags from your images"""%globals()

USING_METADATA_TAGS_REF = """
You can insert a previously defined metadata tag by either using:
<ul><li>The insert key</li>
<li>A right mouse button click inside the control</li>
<li>In Windows, the Context menu key, which is between the Windows key and Ctrl key </li></ul>
The inserted metadata tag will appear in green. To change a previously inserted metadata tag, navigate the cursor to just before the tag and either:
<ul><li>Use the up and down arrows to cycle through possible values.</li>
<li>Right-click on the tag to display and select the available values.</li></ul>"""

USING_METADATA_GROUPING_HELP_REF = """Please see <i>%(METADATA_GROUPING_HELP_REF)s</i> for more details on the 
proper use of metadata for grouping"""%globals()

##################################################
#
# Help for the main window
#
##################################################

DEFAULT_IMAGE_FOLDER_HELP = """
<p>In the Folder panel, the <i>Default Input Folder</i> contains the input image or data files
that you want to analyze. Several File Processing modules (e.g., 
<b>LoadImages</b> or <b>LoadData</b>) provide the option of retrieving images 
from this folder on a default basis unless you specify, within the module, an alternate, 
specific folder on your computer. Within modules, we recommend selecting the 
Default Input Folder as much as possible, so that your pipeline will 
work even if you transfer your images and pipeline to a different 
computer. If, instead, you type specific folder path names into a module's settings, 
your pipeline will not work on someone else's computer until you adjust those
pathnames within each module.</p>

<p>Use the <i>Browse</i> button <img src="%(LOCATION_BROWSE_BUTTON)s"></img> to specify 
the folder you would like to use as the Default Input Folder, or 
type the full folder path in the edit box. If you type a folder path that  
cannot be found, the message box below will indicate this fact until you correct the problem. 
If you want to specify a folder that does not yet exist, type the desired name and 
click on the <i>New folder</i> button <img src="%(LOCATION_CREATE_BUTTON)s"></img>.
The folder will be created according to the pathname you have typed.</p>

<p>The contents of the Default Input Folder are shown in the file panel to the left.
Double-clicking image file names in this panel opens them in a figure window.             
If you double-click on .mat pipeline or output files (CellProfiler 1.0) or .cp 
pipeline files (CellProfiler 2.0), you will be asked if you want to load a       
pipeline from the file. To refresh the contents of this panel, click the <i>Refresh</i>
button <img src="%(LOCATION_REFRESH_BUTTON)s"></img>.</p>"""%globals()

DEFAULT_OUTPUT_FOLDER_HELP = """
<p>In the Folder panel, the <i>Default Output Folder</i> is the folder that CellProfiler uses to
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

OUTPUT_FILENAME_HELP = """
<p>In the <i>Output Filename</i> box in the Folder panel, you can specify the name of the output file 
where all information about the analysis as well as any measurements will be 
stored to the hard drive. The output file is a 
.mat file, which is readable by CellProfiler and by MATLAB. Results in the 
output file can be accessed or exported
using <b>Data Tools</b> from the main menu of CellProfiler.
The pipeline with its settings can be be loaded from an output file using 
<i>File > Load Pipeline...</i>, or by double-clicking the output file in the file
list panel (located in the lower left corner of the CellProfiler main window).</p>

<p>The output file will be saved in the Default Output Folder unless you type a 
full path and file name into the output file name box. The path must not have 
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
If you do not need this file and/or notice this behavior, uncheck the <i>Write output
file?</i> box to the right.</p>"""

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
email list, and wiki, in addition to the available user forum.
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
comma-delimited files to load tables of metadata, in addition to file lists of input 
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
well (i.e., assays of < 100 images).</p>

<p>CellProfiler can extract valuable biological information from images quickly 
while increasing the objectivity and statistical power of assays. It helps researchers 
approach a variety of biological questions quantitatively, including standard 
assays (e.g., cell count, size, per-cell protein levels) as well as complex 
morphological assays (e.g., cell/organelle shape or subcellular patterns of DNA 
or protein staining).</p>

<p>The wide variety of measurements produced by CellProfiler serves as useful "raw material" for machine learning algorithms. CellProfiler's companion software, CellProfiler Analyst, has an interactive machine learning tool called Classifier which can learn to recognize a phenotype of interest based on your guidance. Once you complete the training phase, CellProfiler Analyst will score every object in your images based on CellProfiler's measurements.  CellProfiler Analyst also contains tools for the interactive visualization of the data produced by CellProfiler.</p>

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
<ul>
<li>Carpenter AE, Jones TR, Lamprecht MR, Clarke C, Kang IH, Friman O, 
Guertin DA, Chang JH, Lindquist RA, Moffat J, Golland P, Sabatini DM (2006) 
CellProfiler: image analysis software for identifying and quantifying cell 
phenotypes. <i>Genome Biology</i> 7:R100. PMID: 17076895</li>
<li>Lamprecht MR, Sabatini DM, Carpenter AE (2007) CellProfiler: free, versatile 
software for automated biological image analysis. <i>Biotechniques</i> 
42(1):71-75. PMID: 17269487</li>
<li>Jones TR, Carpenter AE, Lamprecht MR, Moffat J, Silver S, Grenier J, Root D, Golland P, Sabatini DM (2009) Scoring diverse cellular morphologies in image-based screens with iterative feedback and machine learning. PNAS 106(6):1826-1831/doi: 10.1073/pnas.0808843106. PMID: 19188593 PMCID: PMC2634799</li>
<li>Jones TR, Kang IH, Wheeler DB, Lindquist RA, Papallo A, Sabatini DM, Golland P, Carpenter AE (2008) CellProfiler Analyst: data exploration and analysis software for complex image-based screens. BMC Bioinformatics 9(1):482/doi: 10.1186/1471-2105-9-482. PMID: 19014601 PMCID: PMC2614436</li>
</ul>
"""

BUILDING_A_PIPELINE_HELP = """
<p>A <i>pipeline</i> is a sequential set of image analysis modules. The 
best way to learn how to use CellProfiler is to load an example pipeline 
from the CellProfiler website's Examples page and try it out, then adapt it for your own images. You can also build a 
pipeline from scratch. Click the <i>Help</i> <img src="%(LOCATION_MODULE_HELP_BUTTON)s"></img> button in the main window to get
help for a specific module.</p>

<p>To adjust the CellProfiler source code, see <i>Help > Developer's Guide</i>. 
</p>

<h3>Loading an existing pipeline</h3>
<ol>
<li>Put the images and pipeline into a folder on your computer.</li>
<li> Set the Default Input and Output Folders (lower right of the main 
window) to be the folder where you put the images.</li> 
<li>Load the pipeline using <i>File > Load Pipeline</i> in the main menu of 
CellProfiler.</li> 
<li>Click <i>Analyze images</i> to start processing.</li> 
<li>Examine the measurements using <i>Data tools</i>. The <i>Data tools</i> options are accessible in 
the main menu of CellProfiler and allow you to plot, view, or export your 
measurements (e.g., to Excel).</li>   
<li>If you modify the modules or settings in the pipeline, you can save the 
pipeline using <i>File > Save Pipeline.</i></li>
<li>To learn how to use a cluster of computers to process 
large batches of images, see <i>%(BATCH_PROCESSING_HELP_REF)s</i>.</li>
</ol>

<h3>Building a pipeline from scratch</h3>
<p>Constructing a pipeline involves placing individual modules into a pipeline. The list
of modules in the pipeline is shown in the <i>pipeline panel</i> (located on the 
left-hand side of the CellProfiler window).</p>
<ol>
<li><p><i>Place modules in a new pipeline.</i><br>
Choose image analysis modules to add to your pipeline by clicking the <i>Add</i> 
<img src="%(LOCATION_MODULE_ADD_BUTTON)s"></img> button
(located underneath the pipeline panel) or right-clicking in the pipeline panel
itself and selecting a module from the 
pop-up box that appears. You can learn more about each module by clicking
<i>Module Help</i> in the "Add modules" window or the <i>?</i> button after the module has been placed and selected
in the pipeline. Modules are added to the end of the pipeline, but you can
adjust their order in the main window by dragging and dropping them, or by selecting a module (or
modules, using the <i>Shift</i> key) and using the <i>Move up</i> 
<img src="%(LOCATION_MODULE_MOVEUP_BUTTON)s"></img> and <i>Move down</i> 
<img src="%(LOCATION_MODULE_MOVEDOWN_BUTTON)s"></img> buttons. 
The <i>Remove</i> <img src="%(LOCATION_MODULE_REMOVE_BUTTON)s"></img> button will delete the selected 
module(s) from the pipeline.</p> 
<p>Typically, the first module you must run is 
 <b>LoadImages</b>, in which you specify the identity of the images 
you want to analyze. </p>
<p>Most pipelines depend on one major step: identifying the objects. In 
CellProfiler, the objects you identify are called <i>primary</i>, 
<i>secondary</i>, or <i>tertiary</i>:
<ul>
<li><b>IdentifyPrimary</b> modules identify objects without relying on any 
information other than a single grayscale input image (e.g., nuclei are 
typically primary objects).</li>
<li><b>IdentifySecondary</b> modules require a grayscale image plus an image 
where primary objects have already been identified, because the secondary 
objects are determined based on the primary objects (e.g., cells can be 
secondary objects when their identification is based on the location of nuclei). </li>
<li><b>IdentifyTertiary</b> modules require images in which two sets of objects have 
already been identified (e.g., nuclei and cell regions are used to define the 
cytoplasm objects, which are tertiary objects).</li>
</ul></p>
<p><i>Saving images in your pipeline:</i> Due to the typically high number 
of intermediate images produced during processing, images produced during 
processing are not saved to the hard drive unless you specifically request it, 
using a <b>SaveImages</b> module.</p>
<p><i>Saving data in your pipeline:</i> All measurements will be stored in the CellProfiler-formatted
output file, but you can include an <b>Export</b> module to automatically export
data in a format you prefer.</p></li> 

<li><p><i>Adjust the settings in each module.</i><br>
In the CellProfiler main window, click a module in the pipeline to see its 
settings in the main workspace. To learn more about the settings for each 
module, select the module in the pipeline and click the <i>Help</i> button to the 
right of each setting, or at the bottom of the pipeline panel
for the help for all the settings for that module.</p>
<p>If there is an error with the settings (e.g., a setting refers to an image 
that doesn't exist yet), 
a <img src="%(LOCATION_SETTINGS_ERROR_ICON)s"></img>  icon will appear next to the 
module name. If there is a warning (e.g., a special notification attached to a choice of setting), 
a <img src="%(LOCATION_SETTINGS_WARNING_ICON)s"></img>  icon will appear. Errors
will cause the pipeline to fail upon running, whereas a warning will not. Once 
the errors/warnings have been resolved, a <img src="%(LOCATION_SETTINGS_OK_ICON)s">
</img>  icon will appear indicating that the module is ready to run.</p>
</li>
<li><p><i>Set your Default Input Folder, Default Output Folder and output filename.</i><br>
For more help, click their nearby <i>Help</i> buttons in the main window. </p></li>

<li><p><i>Click <i>Analyze images</i> to start processing.</i><br> 
All of the images in your selected folder(s) will be analyzed using the modules 
and settings you have specified. A status window will appear which has the following:
<ul>
<li>A <i>progress bar</i> which gives the elapsed time and estimates the time remaining to 
process the full image set.</li>
<li>A <i>pause button</i> <img src="%(LOCATION_RUNSTATUS_PAUSE_BUTTON)s"></img> 
which pauses execution and allows you to subsequently 
resume the analysis.
<li>A <i>stop button</i> <img src="%(LOCATION_RUNSTATUS_STOP_BUTTON)s"></img> 
which cancels execution after prompting you for a place to
save the measurements collected to that point.</li>
<li>A <i>save measurements</i> button <img src="%(LOCATION_RUNSTATUS_SAVE_BUTTON)s"></img> 
which will prompt you for a place to
save the measurements collected to that point while continuing the analysis run.</li>
</ul> 
At the end of each cycle, CellProfiler saves the measurements in the output file.</p></li>

<li><p><i>Use Test mode to preview results.</i><br>
You can optimize your pipeline by selecting the <i>Test</i> option from 
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results, and adjust the module settings on the fly. See 
<i>%(TEST_MODE_HELP_REF)s</i> for more details.</p>
</li>
<li><p>Save your pipeline via <i>File > Save Pipeline</i>.</p>
</li>
</ol>
"""% globals()


USING_METADATA_HELP = """
Metadata (i.e., additional data about image data) is sometimes available for input images.
This information can be:
<ol>
<li>Used by CellProfiler to group images with common metadata identifiers (or "tags") 
together for particular steps in a pipeline;</li>
<li>Stored in the output file along with CellProfiler-measured features for
annotation or sample-tracking purposes;
<li>Used to name additional input/output files.</li></ol></p>
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
can make use of metadata. However, here is an overview of how metadata is obtained and used.

<h3>Associating images with metadata</h3>

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
sets and in a .csv file specifying the illumination correction functions. </p>

<p>In this case, if the illumination correction images are loaded with the 
<b>LoadData</b> module, the file should have a "Metadata_Date" 
column which contains the date identifiers. Similarly, if the individual images 
are loaded using the <b>LoadImages</b> module, <b>LoadImages</b> should be set to extract the 
<Date> metadata field from the file names. The pipeline will then match the individual 
images with their corresponding illumination correction functions based on matching 
"Metadata_Date" fields.</p>
""" + \
"""<h3>Use of metadata-specific module settings</h3>

<p>Once the metadata has been obtained, you can use <i>metadata tags</i> to reference them
in later modules. %(USING_METADATA_TAGS_REF)s Several modules are capable of 
using metadata tags for various purposes. Examples include:
<ul>
<li>You would like to create and apply an illumination correction function to all images from a particular
plate. You can use metadata tags to save each illumination correction function with a plate-specific
name in <b>SaveImages</b>, and then use <b>LoadSingleImage</b> to get files
with the name associated with your image's plate to be applied to your original images.</li>
<li>You have a set of experiments for which you would like to produce and save results
individually for each experiment but using only one analysis run. You can use metadata tags
in <b>ExportToSpreadsheet</b> or <b>ExportToDatabase</b> to save a spreadsheet for each experiment in 
a folder named according to the experiment.</li>
</ul>
<p>In each case, the pre-defined metadata tag is used to name a file or folder. Tags are case-sensitive; 
the name must match the metadata field defined by <b>LoadImages</b> or <b>LoadData</b>. The options
for the setting will specify whether tags are applicable; see the module setting help for additional
information on how to use them in the context of the specific module.</p>
"""%globals()

USING_METADATA_GROUPING_HELP = """
<p>In some instances, you may want to subdivide an image set into groups that share a common feature.
For example, if you are performing illumination correction, we usually recommend that the illumination
function be calculated on a per-plate basis, and hence each plate should be processed independently.
Since running the same pipeline multiple times (one for each grop) would be time-consuming, CellProfiler
is able to perform image grouping for this purpose. If the image subdivisions are distinguishable by some metadata
(usually defined within the filename or folder location), grouping will enable you to
process those images that have the metadata together.</p>

<p>To use grouping, you must define the relevant metadata for each image. This can be done using regular
expressions in <b>LoadImages</b> or having them pre-defined in a .csv for use in <b>LoadData</b>.</p>

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
<p>%(USING_METADATA_HELP_REF)s</p>
"""%globals()

MEMORY_AND_SPEED_HELP = """
<p>CellProfiler includes several options for dealing with out-of-memory
errors associated with image analysis: </p>
<ul>
<li><p><i>Resize the input images.</i><br>
If the image is high-resolution, it may be helpful to determine whether the 
features of interest can be processed (and accurate data obtained) by using a 
lower-resolution image. If this is the  case, use the <b>Resize</b> module (in the
<i>Image Processing</i> category) to scale down the image to a more manageable size
and perform the desired operations on the smaller image.</p></li>

<li><p><i>Use the <b>ConserveMemory</b> module.</i><br>                                 
The <b>ConserveMemory</b> module lets you clear the images stored in memory, 
with the exception of any you specify. Please see the 
<b>ConserveMemory</b> module help for more details.</p></li>
</ul>

<p>In addition, there are several options in CellProfiler for speeding up processing: </p>

<ul>
<li><p><i>Run without display windows.</i><br>
Each module is associated with a display window that takes time to render and/or
update. Closing these windows improves speed somewhat. 
To the left of each module listed in your pipeline an icon 
<img src="%(LOCATION_DISPLAYMODE_SHOW_ICON)s"></img> indicates whether
the module window will be displayed during the analysis run. You can turn off individual module windows by
clicking on the icon; this icon <img src="%(LOCATION_DISPLAYMODE_HIDE_ICON)s"></img> indicates that the window 
will not be shown. Select <i>Window > Hide all windows on run</i> to prevent display
of all module windows.</p></li>           
                                                                            
<li><p><i>Use care in object identification </i><br>                                   
If you have a large image which contains many small        
objects, a good deal of computer time will be spent processing each   
individual object, many of which you might not need. To avoid this, make 
sure that you adjust the diameter options in <b>IdentifyPrimaryObjects</b> to   
exclude small objects in which you are not interested, or use a <b>FilterObjects</b> 
module to eliminate such objects.</p></li>               
</ul>
"""%globals()

TEST_MODE_HELP = """ 
<p>You can test an analysis on a selected image cycle using the <i>Test</i> mode option on 
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results and adjust the module settings on the fly.</p>

<p>To enter Test mode once you have built a pipeline, choose <i>Test > Start test run</i> in the
menu bar in the main window. At this point, you will see the following features appear:
<ul>
<li>The module view will have a slider bar appearing on the far left.</li>
<li>A Pause icon <img src="%(LOCATION_TESTMODE_GO_ICON)s"></img> will appear to the left of each module.</li>
<li>A series of buttons will appear at the bottom of the pipeline panel above the 
module adjustment buttons.</li>
<li>The grayed-out items in the <i>Test</i> menu will become active, and the 
<i>Analyze Images</i> button will become inactive.
</ul>
</p>

<p>You can run your pipeline in Test mode by selecting <i>Test > Step to next module</i>
or clicking the <i>Run</i> button. The pipeline will execute normally, but you will
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
changes from <img src="%(LOCATION_TESTMODE_GO_ICON)s"></img> to 
<img src="%(LOCATION_TESTMODE_PAUSE_ICON)s"></img> to indicate that a pause has 
been inserted at that point.</li>
<li><i>Run:</i> Execution of the pipeline will be started/resumed until
the next module pause is reached. When all modules have been executed for a given image cycle,
execution will stop.</li>
<li><i>Step:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Next image:</i> Skip ahead to the next image cycle as determined by the image 
order in <b>LoadImages</b>/<b>LoadData</b>. The slider will automatically return to the 
first module in the pipeline.</li>
</ul>
</p>
<p>From the <i>Test</i> menu, you can choose additional options:
<ul>
<li><i>Stop test run:</i> Exit <i>Test</i> mode. Loading a new pipeline or adding/subtracting
modules will also automatically exit test mode.</li>
<li><i>Step to next module:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Next image cycle:</i> Step to the next image cycle in the current group or the .</li>
<li><i>Next image group:</i> Step to the next group in the image set</li>
<li><i>Random image cycle:</i> Randomly select an image cycle in the current group to jump to.</li>
<li><i>Choose image cycle:</i> Choose the image cycle to jump to.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Choose group:</i> Choose the group to jump to.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Reload modules source:</i> For developers only. This option will reload the module source 
code, so any changes to the code will be reflected immediately.</li>
</ul>
Note that if movies are being loaded, the individual movie is defined as a group automatically.
Selecting <i>Choose group</i> will allow you to choose the movie file, and <i>Choose image cycle</i> 
will let you choose the individual movie frame from that file.
<p>%(USING_METADATA_GROUPING_HELP_REF)s</p>
</p>
"""%globals()

BATCHPROCESSING_HELP = """ 
CellProfiler is designed to analyze images in a high-throughput manner.   
Once a pipeline has been established for a set of images, CellProfiler    
can export batches of images to be analyzed on a computing cluster with the         
pipeline. We often process tens or even hundreds of thousands of images for one analysis in this 
manner. We do this by breaking the entire set of images into    
separate batches, then submitting each of these batches as individual 
jobs to a cluster. Each individual batch can be separately analyzed from  
the rest.

<h3>Submitting files for batch processing</h3>

Below is a basic workflow for submitting your image batches to the cluster.
<ol>
<li><i>Create a folder for your project on your cluster.</i> For high throughput 
analysis, it is recommended to create a separate project folder for each run. </li>
<li>Within this project folder, create the following folders (both of which must be connected to 
the cluster computing network):
<ul>
<li>Create an <i>images</i> folder, then transfer all of our images to this folder
as the input folder. The input folder must be readable by everyone (or at least your 
cluster) because each of the separate cluster computers will read input files from 
this folder.
<li>Create an <i>output</i> folder where all your output data will be stored. The
output folder must be writeable by everyone (or at least your cluster) because 
each of the separate cluster computers will write output files to this folder.
</ul>
If you cannot create folders and set read/write permissions to these folders (or don't know 
how), ask your Information Technology (IT) department for help. </li>

<li>In the CellProfiler folder panel, set the Default Input and Default Output Folders
to the <i>images</i> and <i>output</i> folders created above, respectively.</li>

<li><i>Create a pipeline for your image set.</i> You should test it on a few example
images from your image set. The module settings selected for your pipeline will be 
applied to <i>all</i> your images, but the results may vary 
depending on the image quality, so it is critical to insure that your settings be
robust against your "worst-case" images.
<p>For instance, some images may contain no cells. If this happens, the automatic thresholding
algorithms will incorrectly choose a very low threshold, and therefore "find" 
spurious objects. This can be overcome by setting a lower limit on the threshold in 
the <b>IdentifyPrimaryObjects</b> module.</p>
<p>The Test mode in CellProfiler may be used for previewing the results of your settings
on images of your choice. Please refer to <i>%(TEST_MODE_HELP_REF)s</i>
for more details on how to use this utility.</li>

<li><i>Add the <b>CreateBatchFiles</b> module to the end of your pipeline.</i>
This module is needed to resolve the pathnames to your files with respect to 
your local machine and the cluster computers. If you are processing large batches 
of images, you may also consider adding <b>ExportToDatabase</b> to your pipeline, 
after your measurement modules but before the CreateBatchFiles module. This module 
will export your data either directly to a MySQL database or into a set of 
comma-separated files (CSVs) along with a script to import your data into a 
MySQL database. Please refer to the help for these modules in order learn more 
about which settings are appropriate.</li>

<li><i>Analyze your images to create a batch file.</i> Click the <i>Analyze images</i>
button and the analysis will begin locally processing the first image set only. 
Do not be surprised if processing the first image set takes much longer than usual
if using <b>LoadImages</b> since this module creates a list of all images to be 
processed which can take a while if there are many of them (this process can be sped
up by creating your list of images as a CSV and using the <b>LoadData</b> module to load it).
<p>At the end of processing the first cycle locally, the <b>CreateBatchFiles</b>
module halts execution, creates the proper batch file (a file called 
<i>Batch_data.mat</i>) and saves it in the Default Output Folder (Step 1). You 
are now ready to submit this batch file to the cluster to run each of the batches 
of images on different computers on the cluster.</p></li>

<li><i>Submit your batches to the cluster.</i> Log on to your cluster, and navigate 
to the directory where you have installed CellProfiler on the cluster. A single
batch can be submitted with the following command:<br>
<code>
./python-2.6.sh CellProfiler.py -p &lt;Default_Output_Folder_path&gt;/Batch_data.mat -c -r -b -f &lt;first_image_set_number&gt; -l &lt;last_image_set_number&gt;
</code>
This command runs the batch by using additional options to CellProfiler that 
specify the following (type "CellProfiler.py -h" to see a list of available options):
<ul>
<li><code>-p &lt;Default_Output_Folder_path&gt;/Batch_data.mat</code>: The 
location of the batch file, where &lt;Default_Output_Folder_path%gt; is the 
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
To submit all the batches for a full image set, you will need a script that calls
CellProfiler with these options with sequential image set numbers, e.g, 1-50, 51-100, 
etc and submit each as an individual job.
<p>The above notes assume that you are running CellProfiler using our source code (see 
"Developer's Guide" under Help for more details). If you are using the compiled version,
you would replace <code>./python-2.6.sh CellProfiler.py</code> with the CellProfiler 
executable file itself and run it from the installation folder.</li>
</ol>

<p>Once all the jobs are submitted, the cluster will run each batch individually 
and output any measurements or images specified in the pipeline. Specifying the output filename when
calling CellProfiler will also produce an output file containing the measurements  
for that batch of images in the output folder. Check the output from the batch 
processes to make sure all batches complete. Batches that fail for transient reasons
can be resubmitted.</p>

<p>For additional help on batch processing, please post your questions on 
the CellProfiler <a href = "http://cellprofiler.org/forum/viewforum.php?f=14">forum</a>.</p>
"""

RUN_MULTIPLE_PIPELINES_HELP = """
<br>The <b>Run multiple pipelines</b> dialog lets you select several pipelines
which will be run consecutively. You can invoke <b>Run multiple pipelines</b>
by selecting it from the file menu. The dialog has three parts to it:
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
CellProfiler -L ERROR
<br>
The following is a list of log levels that can be used:
<br><ul><li><b>DEBUG</b> - detailed diagnostic information</li>
<li><b>INFO</b> - informational messages that confirm normal progress</li>
<li><b>WARNING</b> - messages that report problems that might need attention</li>
<li><b>ERROR</b> - messages that report unrecoverable errors that result in data loss or termination of the current operation.</li>
<li><b>CRITICAL</b> - messages indicating that CellProfiler should be restarted or is incapable of running.</li></ul>
<br>
You can tailor CellProfiler's logging with much more control using a logging
configuration file. You specify the file name in place of the log level on
the command line, like this:
<br>
CellProfiler -L ~/CellProfiler/my_log_config.cfg
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
channel specified by a <b>Load</b> module (e.g.,  "OrigBlue" and "OrigGreen") or a
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
<li><b>Load Pipeline...:</b> Load a CellProfiler pipeline (.cp or .mat) file from 
your hard drive.</li>
<li><b>Load Pipeline from URL:</b> Load a CellProfiler pipeline from a web address 
(i.e., http://...)</li>
<li><b>Save Pipeline:</b> Save the pipeline you are currently working on as a 
CellProfiler .cp pipeline file.</li>
<li><b>Save Pipeline as...:</b> Prompts the user for a filename to save the 
pipeline as.</li>
<li><b>Clear Pipeline:</b> Removes all modules from the current pipeline.</li>
<li><b>Open Image:</b> Opens a dialog box prompting you to select an image file for
display. Images listed in the file panel can be also be displayed by double-clicking 
on the filename.</li>
<li><b>Analyze Images:</b> Executes the current pipeline using the current module
and Default Input and Output folder settings.</li>
<li><b>Stop Analysis:</b> Stop the current analysis run.</li>
<li><b>Run Multiple Pipelines:</b> Execute multiple pipelines in sequential order. 
This option opens a dialog box allowing you to select the pipelines you would like 
to run as well as the associated input and output folders. See
the help in the Run Multiple Pipelines dialog for more details.</li>
<li><b>Restart:</b> Resume a partially completed analysis run from here it left off.
To restart an analysis run, go to <i>File > Restart</i> choose the output .mat file 
containing the partially complete measurements and the analysis run will pick up 
starting with the last cycle that was processed. </li>
<li><b>Preferences...:</b> Displays the Preferences window, where you can change 
many options in CellProfiler.</li>
<li><b>Check for updates...:</b> Displays a dialog which lets you know whether your
installation of CellProfiler is the latest version and if not, notifies you
as what version is available from CellProfiler website.</li>
<li><b>Recent:</b> Displays a list of the most recent pipelines loaded. Select any
one of these pipelines to load it.</li>
<li><b>Exit:</b></b> End the current CellProfiler session. You will be given the option
of saving your current pipeline if you have not done so.</li>
</ul>"""

MENU_BAR_EDIT_HELP = """
The <i>Edit</i> menu provides options for modifying modules in your current pipeline.
<ul>
<li><b>Undo:</b> Undo your previous module modification.</li>
<li><b>Move Up:</b> Move the currently selected module(s) up in the module list. You
can also use the <img src="%(LOCATION_MODULE_MOVEUP_BUTTON)s"></img> button located
below the Pipeline panel.</li>
<li><b>Move Down:</b> Move the currently selected module(s) down in the module list. You
can also use the <img src="%(LOCATION_MODULE_MOVEDOWN_BUTTON)s"></img> button located
below the Pipeline panel.</li>
<li><b>Delete:</b> Remove the currently selected module(s) from the module list. 
Pressing the Delete key also removes the module(s). You
can also use the <img src="%(LOCATION_MODULE_REMOVE_BUTTON)s"></img> button located
under the Pipeline panel.</li>
<li><b>Duplicate:</b> Duplicate the currently selected module(s) in the pipeline.
The current settings of the selected module(s) are retained in the duplicate.</li>
<li><b>Add Module:</b> Select a module from the pop-up list to inster into the current
pipeline. You can also use the <img src="%(LOCATION_MODULE_ADD_BUTTON)s"></img> button located
under the Pipeline panel.</li>
</ul>
You can select multiple modules at once for moving, deletion and duplication 
by selecting the first module and using Shift-click on the last module to select 
all the modules in between.
"""%globals()

MENU_BAR_WINDOW_HELP = """
The <i>Windows</i> menu provides options for showing and hiding the module display windows.
<ul>
<li><b>Close All Open Windows:</b> Closes all display windows that are currently open.</li>
<li><b>Show All Windows On Run:</b> Select to show all display windows during the
current test run or next analysis run. The display mode icons next to each module
in the pipeline panel will switch to <img src="%(LOCATION_DISPLAYMODE_SHOW_ICON)s"></img>.</li>
<li><b>Hide All Windows On Run:</b> Select to show no display windows during the
current test run or next analysis run. The display mode icons next to each module
in the pipeline panel will switch to <img src="%(LOCATION_DISPLAYMODE_HIDE_ICON)s"></img>.</li>
</ul>"""%globals()

MENU_BAR_DATATOOLS_HELP = """
The <i>Data Tools</i> menu provides tools to allow you
to plot, view, export or perform specialized analyses on your measurements.

<p>Each data tool has a corresponding module with the same name and 
functionality. The difference between the data tool and the module is that the
data tool takes a CellProfiler output file as input, which contains measurements
from a previously completed analysis run. In contrast, a module uses measurements
received from the upstream modules during an in-progress analysis run.</p>

<p>Opening a data tool will present a prompt in which the user is asked to provide
the location of the output file. Once specified, the user is then prompted to
enter the desired settings. The settings behave identically as those from the 
corresponding module.</p>

<p>Help for each <i>Data Tool</i> is available under <i>%(DATA_TOOL_HELP_REF)s</i> or the corresponding
module help.</p>"""%globals()

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
<li><i>Save</i>: You can save the figure window to a file (currently,
Postscript (.PS), PNGs and PDFs are supported). Note that this will save the entire
contents of the window, not just the individual subplot(s) or images.</li>
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
"""%globals()

MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP = """
All figure windows come with a navigation toolbar, which can be used to navigate through the data set.
<ul>
<li><b>Home, Forward, Back buttons:</b>
<i>Home</i> <img src="%(LOCATION_WINDOW_HOME_BUTTON)s"></img> always takes you to 
the initial, default view of your data. The <i>Forward</i> <img src="%(LOCATION_WINDOW_FORWARD_BUTTON)s"></img>
and <i>Back</i> <img src="%(LOCATION_WINDOW_BACK_BUTTON)s"></img> buttons are akin 
to the web browser forward and back buttons in that they are used to navigate back 
and forth between previously defined views, one step at a time. They will not be 
enabled unless you have already navigated within an image else using 
the <b>Pan</b> and <b>Zoom</b> buttons, which are used to define new views. </li>

<li><b>Pan/Zoom button:</b>
This button has two modes: pan and zoom. Click the toolbar button 
<img src="%(LOCATION_WINDOW_PAN_BUTTON)s"></img> to activate panning 
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

<li><b>Zoom-to-rectangle button:</b> Click this toolbar button <img src="%(LOCATION_WINDOW_ZOOMTORECT_BUTTON)s"></img> 
to activate this mode. To zoom in, press the left mouse button and drag in the window 
to draw a box around the area you want to zoom in on. When you release the mouse button, 
the image is re-drawn to display the specified area. Remember that you can always use 
<i>Backward</i> button to go back to the previous zoom level, or use the <i>Home</i> 
button to reset the window to the initial view.</li>

<li><b>Save:</b> Click this button <img src="%(LOCATION_WINDOW_SAVE_BUTTON)s"></img>to launch a file save dialog. You can save the 
figure window to a file (currently, Postscript (.PS), PNGs and PDFs are supported). 
Note that this will save the entire contents of the window, not just the individual 
subplot(s) or images.</li>
</ul>
"""%globals()

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
zero (dark) to one (bright).</li>
<li><i>Image contrast:</i> Presents three options for displaying the color/intensity values in 
the images:
<ul>
<li><i>Raw:</i> Shows the image using the full colormap range permissible for the
image type. For example, for a 16-bit image, the pixel data will be shown using 0 as black
and 65535 as white. However, if the actual pixel intensities span only a portion of the
image intensity range, this may render the image unviewable. For example, if a 16-bit image
only contains 12 bits of data, the resultant image will be entirely black.</li>
<li><i>Normalized (default):</i> Shows the image with the colormap "autoscaled" to
the maximum and minimum pixel intensity values; the minimum value is black and the
maximum value is white. </li>
<li><i>Log normalized:</i> Same as <i>Normalized</i> except that the color values
are then log transformed. This is useful for when the pixel intensity spans a wide
range of values but the standard deviation is small (e.g., the majority of the 
interesting information is located at the dim values). Using this option 
increases the effective contrast.</li>
</ul>
</li>
<li><i>Channels:</i> For color images only. You can show any combination of the red, 
green, and blue color channels.</li>
</ul>
"""

FIGURE_HELP = (
    ("Using The Display Window Menu Bar", MODULE_DISPLAY_MENU_BAR_HELP ),
    ("Using The Interactive Navigation Toolbar", MODULE_DISPLAY_INTERACTIVE_NAVIGATION_HELP),
    ("How To Use The Image Tools",MODULE_DISPLAY_IMAGE_TOOLS_HELP))

###################################################
#
# Help for the preferences dialog
#
###################################################

TITLE_FONT_HELP = """The <i>Title Font </i>preference sets the font used
in titles above plots displayed in module figure windows."""

TABLE_FONT_HELP = """The <i>Table Font</i> preference sets the font used
in tables displayed in module figure windows."""

DEFAULT_COLORMAP_HELP = """The <i>Default Colormap</i> preference specifies the
color map that sets the colors for labels and other elements. See this
<a href ="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">
page</a> for pictures of available colormaps."""

WINDOW_BACKGROUND_HELP = """The <i>Window Background</i> preference sets the
window background color of the CellProfiler main window."""

PLUGINS_DIRECTORY_HELP = """The <i>Plugins directory</i> preference chooses
the directory that holds dynamically-loaded CellProfiler modules. You
can write your own module and place it in this directory and CellProfiler
will make it available for your pipeline. You must restart CellProfiler
after modifying this setting."""

IJ_PLUGINS_DIRECTORY_HELP = """The <i>ImageJ plugins directory</i> preference
sets the directory that holds ImageJ plugins (for the <b>RunImageJ</b> module).
You can download or write your own ImageJ plugin and place it in this directory
and CellProfiler will make it available for your pipeline. You must restart
CellProfiler after modifying this setting."""

CHECK_FOR_UPDATES_HELP = """The <i>Check for Updates</i> preference controls how
CellProfiler looks for updates on startup."""

SHOW_STARTUP_BLURB_HELP = """The <i>Display welcome text on startup</i> preference 
controls whether CellProfiler displays an orientation message on startup."""

SHOW_ANALYSIS_COMPLETE_HELP = """The <i>Show "Analysis complete"</i>
preference determines whether CellProfiler displays a message box at the
end of a run. Check this preference to show the message box or uncheck it
to stop display."""

SHOW_EXITING_TEST_MODE_HELP = """The <i>Show "Exiting test mode"</i>
preference determines whether CellProfiler displays a message box to inform you
that a change made to the pipeline will cause test mode to end. Check this preference 
to show the message box or uncheck it to stop display."""

SHOW_REPORT_BAD_SIZES_DLG_HELP = """The <i>Show Report Bad Sizes dialog</i>
preference determines whether CellProfiler will display a warning dialog
if images of different sizes are loaded together in an image set.
Check this preference to show the message box or uncheck it to stop display."""

WARN_ABOUT_OLD_PIPELINES_HELP = """
The <i>Warn if a pipeline was saved in an old version of CellProfiler</i>
preference determines whether CellProfiler displays a warning dialog
if you open a pipeline that was saved using an old version of CellProfiler.
The purpose of this warning is to remind you that, if you save the pipeline
using the newer version of CellProfiler, people using the old version of
CellProfiler might not be able to use the new pipeline. Pipelines saved
by old CellProfiler versions can always be loaded and used by new
CellProfiler versions."""

USE_MORE_FIGURE_SPACE_HELP = """
The <i>Use more figure space</i> preference reduces the padding space
and font sizes in module display figures.  It is suggested that you
also set the <i>Title font</i> preference to a smaller value."""

PRIMARY_OUTLINE_COLOR_HELP = """The <i>Primary Outline Color</i> preference
sets the color used for the outline of the object of interest in the
<i>IdentifyPrimaryObjects</i>, <i>IdentifySecondaryObjects</i> and
<i>IdentifyTertiaryObjects</i> displays."""

SECONDARY_OUTLINE_COLOR_HELP = """The <i>Secondary Outline Color</i> preference
sets the color used for objects other than the ones of interest. In
<i>IdentifyPrimaryObjects</i>, these are the objects that are too small or
too large. In <i>IdentifySecondaryObjects</i> and <i>IdentifyTertiaryObjects</i>,
this is the color of the secondary objects' outline."""

TERTIARY_OUTLINE_COLOR_HELP = """The <i>Tertiary Outline Color</i> preference
sets the color used for the objects touching the image border or image mask
in <i>IdentifyPrimaryObjects</i>."""

REPORT_JVM_ERROR_HELP = """The <i>Warn if Java runtime environment not present</i>
preference determines whether CellProfiler will display a warning on startup
if CellProfiler can't locate the Java installation on your computer. Check
this box if you want to be warned. Uncheck this box to hide warnings."""

EACH_PREFERENCE_HELP = (
    ( "Title font", TITLE_FONT_HELP ),
    ( "Table font", TABLE_FONT_HELP ),
    ( "Default colormap", DEFAULT_COLORMAP_HELP ),
    ( "Window background", WINDOW_BACKGROUND_HELP ),
    ( "Plugins directory", PLUGINS_DIRECTORY_HELP ),
    ( "ImageJ plugins directory", IJ_PLUGINS_DIRECTORY_HELP),
    ( "Check for updates", CHECK_FOR_UPDATES_HELP ),
    ( "Primary outline color", PRIMARY_OUTLINE_COLOR_HELP),
    ( "Secondary outline color", SECONDARY_OUTLINE_COLOR_HELP),
    ( "Tertiary outline color", TERTIARY_OUTLINE_COLOR_HELP),
    ( "Warn if Java runtime not present", REPORT_JVM_ERROR_HELP)
    )
PREFERENCES_HELP = """The Preferences allow you to change many options in CellProfiler
<ul>"""
for key, value in enumerate(EACH_PREFERENCE_HELP):
    PREFERENCES_HELP += """<li><b>""" + value[0] + """:</b>""" + value[1] + """</li>"""
PREFERENCES_HELP += """</ul>"""
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

#########################################################
#
# The top-level of help - used when building the HTML manual
#
#########################################################

'''The help menu for CP's main window'''
MAIN_HELP = (
    ("Introduction",(
        ("Why Use CellProfiler", WHEN_CAN_I_USE_CELLPROFILER_HELP),
        ("New Features in 2.0", NEW_FEATURES_HELP))),
    ("Using CellProfiler",(
        ( "Navigating The Menu Bar", (
            ("Using The File Menu",MENU_BAR_FILE_HELP),
            ("Using The Edit Menu",MENU_BAR_EDIT_HELP),
            ("Using The Test Menu",TEST_MODE_HELP),
            ("Using The Window Menu",MENU_BAR_WINDOW_HELP),
            ("Using The Data Tools Menu",MENU_BAR_DATATOOLS_HELP)) ),
        ("Using Module Display Windows", FIGURE_HELP ),
        ("Setting the Preferences", PREFERENCES_HELP),
        ("How Data Is Handled",(
            ("Using Metadata In CellProfiler",USING_METADATA_HELP),
            ("How To Use Image Grouping",USING_METADATA_GROUPING_HELP),
            ("How Measurements Are Named", MEASUREMENT_NOMENCLATURE_HELP))),
        ("How To Build A Pipeline", BUILDING_A_PIPELINE_HELP),
        ("Before The Analysis Run", (
            ("Setting The Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
            ("Setting The Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
            ("Setting The Output Filename", OUTPUT_FILENAME_HELP))),
        ("Testing Your Pipeline",TEST_MODE_HELP),
        ("Troubleshooting Memory and Speed Issues",MEMORY_AND_SPEED_HELP),
        ("Other Features",(
            ("Batch Processing", BATCHPROCESSING_HELP),
            ("Running Multiple Pipelines", RUN_MULTIPLE_PIPELINES_HELP),
            ("Configuring logging", CONFIGURING_LOGGING_HELP)))
    ))
)

def make_help_menu(h, window):
    import wx
    import htmldialog
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
