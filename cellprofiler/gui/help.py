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
<p>The <i>Default Input Folder</i> contains the input image or data files
that you want to analyze. Several File Processing modules (e.g., 
<b>LoadImages</b> or <b>LoadData</b>) provide the option of retrieving images 
from this folder on a default basis unless you specify, within the module, an alternate, 
specific folder on your computer. Within modules, we recommend selecting the 
Default Input Folder as much as possible, so that your pipeline will 
work even if you transfer your images and pipeline to a different 
computer. If, instead, you type specific folder path names into a module's settings, 
your pipeline will not work on someone else's computer until you adjust those
pathnames within each module.</p>

<p>Use the <i>Browse</i> button (to the right of the text box) to specify 
the folder you would like to use as the Default Input Folder, or 
type the full folder path in the edit box. If you type a folder path that  
cannot be found, the message box below will indicate this fact until you correct the problem. 
If you want to specify a folder that does not yet exist, type the desired name and 
click on the <i>New folder</i> icon to the right of the <i>Browse folder</i> icon.
The folder will be created according to the pathname you have typed.
You can change which folder will appear as the Default Input Folder upon CellProfiler startup
within the <i>File > Preferences...</i> option in the main window.</p>

<p>The contents of the Default Input Folder are shown in the box to the left.
Double-clicking image file names in this list opens them in a figure window.             
If you double-click on .mat pipeline or output files (CellProfiler 1.0) or .cp 
pipeline files (CellProfiler 2.0), you will be asked if you want to load a       
pipeline from the file. To refresh the contents of this window, press    
<i>Enter</i> in the Default Input Folder edit box.</p>"""

DEFAULT_OUTPUT_FOLDER_HELP = """
<p>The <i>Default Output Folder</i> is the folder that CellProfiler uses to
store its output. Several File Processing modules (e.g., <b>SaveImages</b> or 
<b>SaveToSpreadsheet</b>) provide the option of saving analysis results to 
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
The folder will be created according to the pathname you have typed.
You can change which folder will appear as the Default Output Folder upon CellProfiler startup
within the <i>File > Preferences...</i> option in the main window.</p>"""

OUTPUT_FILENAME_HELP = """
<p>Specify the name of the output file where all information 
about the analysis as well as any measurements will be stored to the hard drive. 
The output file is a 
.mat file, which is readable by CellProfiler and by MATLAB. Results in the 
output file can be accessed or exported
using <b>Data Tools</b> from the main menu of CellProfiler.
The pipeline with its settings can be be loaded from an output file using 
<i>File > Load Pipeline...</i>, or by double-clicking the output file in the lower
left window of CellProfiler.</p>

<p>The output file will be saved in the Default Output Folder unless you type a 
full path and file name into the output file name box. The path must not have 
spaces or characters disallowed by your computer's platform.</p>
                                                                           
<p>If the output filename ends in <i>OUT.mat</i> (the typical text appended to 
an output filename), CellProfiler will prevent you from overwriting this file 
on a subsequent run by generating a new file name and asking if you want to 
use it instead.</p>"""

NEW_FEATURES_HELP = """ 
<h2>New Features in CellProfiler 2.0</h2>
A number of new features have been incorporated into this re-engineered Python 
version of CellProfiler:
<h3>Interface</h3>
<ul>
<li><i>Resizable user interface:</i>The main CellProfiler interface can now be resized
by dragging the window corner.</li>
<li><i>Help for individual module settings:</i> Every setting in every module now has 
a help button that you can click to display information and advice for that setting.</li>
<li><i>Settings verification:</i> CellProfiler constantly checks for setting values  
that are not allowed, and immediately flags them for you.</li>
<li><i>Context-dependent module settings</i>: Prior versions of CellProfiler 
displayed all settings for each module, whether or not the values were necessary, 
given existing choices for other settings. Now, only those settings 
you require are displayed, simplifying the interface</li>
<li><i>Test mode for assay development:</i> This feature allows you to 
preview the effect of a module setting on your data. You can step backward or forward 
in the pipeline as you modify settings, optimizing your results prior to running an
actual analysis.</li>
<li><i>Unlimited number of images/objects as module input</i> Some modules can accept an arbitrary number 
of images or objects as input, and you can dynamically add or remove any of these inputs as needed.
For example, you can specify any number of single images in LoadSingleImage; previously,
the module could accept only three input images at a time.<br>
For example, in OverlayOutlines, you can specify that any number of outlines be  
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
<li><i>Figure display choice:</i> Easier access to which windows are displayed is 
now controlled within the pipeline, and is saved as part of the pipeline.
<li><i>Context menus:</i>  The module list responds to right-clicks, providing 
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
and accouting for splitting/merging events.</li>
<li>Object data can be exported to a database as a single table containing
all user-defined object measurements, or as separate tables, one for each object.
<li><i>SQLite support:</i> Data can be exported in SQLite, a 
self-contained database format. Users can create their own local databases and 
no longer need access to a separate database server. Because CellProfiler 
Analyst also supports SQLite, any user can access CellProfiler Analyst's
suite of data exploration and machine-leaning tools.</li>
</ul>

<h3>Speed and Memory Performance</h3>
[TO BE INSERTED]
"""

WHEN_CAN_I_USE_CELLPROFILER_HELP = """ 
<h2>When can I use CellProfiler?</h2>

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
analysis "pipeline", a sequential series of individual modules that each perform 
an image processing function such as illumination correction, object identification 
(segmentation), and object measurement. Users mix and match modules and adjust 
their settings to measure the phenotype of interest. While originally designed for 
high-throughput images, it is equally appropriate for low-throughput assays as 
well(i.e., assays of < 100 images).</p>

<p>CellProfiler can extract valuable biological information from images quickly 
while increasing the objectivity and statistical power of assays. It helps researchers 
approach a variety of biological questions quantitatively, including standard 
assays (e.g., cell count, size, per-cell protein levels) as well as complex 
morphological assays (e.g., cell/organelle shape or subcellular patterns of DNA 
or protein staining).</p>

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
</ul>
"""

BUILDING_A_PIPELINE_HELP = """
<h2>Making a pipeline</h2>
<p>A <i>pipeline</i> is a sequential set of individual image analysis modules. The 
best way to learn how to use CellProfiler is to load an example pipeline 
from the CellProfiler website Examples page and try it out. You can also build a 
pipeline from scratch. Click the <i>?</i> button in the main window to get
help for a specific module.</p>

<p>To learn how to program in CellProfiler, see <i>Help > Developer's Guide</i>. 
To learn how to use a cluster of computers to process 
large batches of images, see <i>Help > General Help > Batch Processing</i>.</p>

<h3>Loading a pipeline</h3>
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
pipeline using <i>File > Save Pipeline.</i> See the end of this document for more 
information on pipeline files.</li> 
</ol>

<h3>Building a pipeline from scratch</h3>
<ol>
<li><p><i>Place modules in a new pipeline.</i><br>
Choose image analysis modules to add to your pipeline by clicking "+" or 
right-clicking in the module list window and selecting a module from the 
pop-up box that appears. Typically, the first module you must run is 
the <b>LoadImages</b> module, in which you specify the identity of the images 
you want to analyze. Modules are added to the end of the pipeline, but you can
adjust their order in the main window by selecting a module (or
modules, using the shift key) and using the <i>Move up</i> ("^") and 
<i>Move down</i> ("v") buttons. The "-" button will delete the selected 
module(s) from the pipeline.</p> 
<p>Most pipelines depend on one major step: identifying the objects. In 
CellProfiler, the objects you identify are called <i>primary</i>, 
<i>secondary</i>, or <i>tertiary</i>:
<ul>
<li><i>Identify Primary modules </i> identify objects without relying on any 
information other than a single grayscale input image (e.g., nuclei are 
typically primary objects).</li>
<li><i>Identify Secondary</i> modules require a grayscale image plus an image 
where primary objects have already been identified, because the secondary 
objects are determined based on the primary objects (e.g., cells can be 
secondary objects). </li>
<li><i>Identify Tertiary modules</i> require images in which two sets of objects have 
already been identified (e.g., nuclei and cell regions are used to define the 
cytoplasm objects, which are tertiary objects).</li>
</ul></p>
<p><i>A note on saving images in your pipeline:</i> Due to the typically high number 
of intermediate images produced during processing, images produced during 
processing are not saved to the hard drive unless you specifically request it, 
using a <b>SaveImages</b> module.</p></li> 

<li><p><i>Adjust the settings in each module.</i><br>
In the CellProfiler main window, click a module in the pipeline to see its 
settings in the main workspace. To learn more about the settings for each 
module, select the module in the pipeline and click the "?" button to the 
right of each setting, or click the "?" button at the bottom of the module
list window for the help for all the settings for that module.</p>
</li>
<li><p><i>Set your Default Input Folder, Default Output Folder and output filename.</i><br>
For more help, click their nearby "?" buttons in the main window. </p></li>

<li><p><i>Click <i>Analyze images</i> to start processing.</i><br> 
All of the images in your selected folder(s) will be analyzed using the modules 
and settings you have specified.You have the option to cancel at any time. 
At the end of each cycle, CellProfiler savees the measurements in the output file.</p></li>

<li><p><i>Use Test mode to preview results.</i><br>
You can test the analysis of a selected image cycle by selecting the <i>Test</i> mode option from 
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results, and adjust the module settings on the fly. See 
<i>Help > General Help > Test Mode </i> for more details.</p>
</li>
<li><p>Save your pipeline via <i>File > Save Pipeline</i>.</p>
</li>
</ol>
"""

NEW_MODULE_NAMES_HELP = """
<h2>Changes to module names in CellProfiler</h2>

<p>Some of the modules have changed their names between CellProfiler 1.0 and 2.0. 
In some cases, the change was made to make the nomenclature more consistent;  
in others, to make the module name suitably generic. </p>

<p>A pipeline created in CellProfiler 1.0 and loaded into CellProfiler 2.0 will have the appropriate modules 
converted to their new names automatically. However, if you are looking for a 
module in CellProfiler 2.0 and can't find it using the familiar CellProfiler 1.0 name, consult this  
list of modules that have changed names (former name in parentheses):
<ul>
<li><b>ConserveMemory</b> (from SpeedUpCellProfiler)
<li><b>ConvertObjectsToImage</b> (from ConvertToImage)</li>
<li><b>FilterObjects</b> (from FilterByObjectMeasurement)</li>
<li><b>EnhanceEdges</b> (from FindEdges)</li>
<li><b>EnhanceOrSuppressFeatures</b> (from EnhanceOrSuppressSpeckles)</li>
<li><b>ExpandOrShrinkObjects</b> (from ExpandOrShrink)</li>
<li><b>ExportToSpreadsheet</b> (from ExportToExcel)</li>
<li><b>FlagImage</b> (from FlagImageForQC)</li>
<li><b>IdentifyObjectsManually</b> (from IdentifyPrimManual)</li>
<li><b>IdentifyPrimaryObjects</b> (from IdentifyPrimAutomatic)</li>
<li><b>IdentifySecondaryObjects</b> (from IdentifySecondary)</li>
<li><b>IdentifyTertiaryObjects</b> (from IdentifyTertiarySubregion)</li>
<li><b>LoadData</b> (from LoadText)
<li><b>MaskObjects</b> (from Exclude)
<li><b>MeasureObjectSizeShape</b> (from MeasureObjectAreaShape)</li>
<li><b>ReassignObjectNumbers</b> (from RelabelObjects)</li>
<li><b>RelateObjects</b> (from Relate)</li>
<li><b>Smooth</b> (from SmoothOrEnhance)</li>
</ul>
</p>

<p>The functionality of some modules has been superseded by others. The modules listed
below have been deprecated and are no longer present (names of the modules to be used in their place in 
parentheses. Where possible, deprecated modules will automatically be imported as equivalent
modules with the appropriate settings:
<ul>
<li><b>LoadImageDirectory</b> (use MakeProjection, in conjunction with LoadImages 
with metadata extracted from the path)</li>
<li><b>KeepLargestObject</b> (imported as FilterObjects with filtering method as 'Maximal per object';
use in conjunction with MeasureObjectSizeShape)</li>
<li><b>Combine</b> (imported as ImageMath with weighted 'Add' operation)</li>
<li><b>PlaceAdjacent</b> (imported as Tile with tiling enabled 'Within cycles')</li>
<li><b>FilenameMetadata</b> (use LoadImages with metadata)</li>
<li><b>SubtractBackground</b> (use ApplyThreshold with grayscale setting)</li>
<li><b>CalculateRatios</b> (use CalculateMath with 'Divide' operation)</li>
</ul>
</p>

<p>A new module category called <i>Data tools</i> has been created, and some 
modules that have been moved from their original categories can be found this menu option. These modules are listed 
below (former category in parentheses):
<ul>
<li><b>CalculateMath</b> (from Measurement)</li>
<li><b>CalculateStatistics</b> (from Measurement)</li>
<li><b>ExportToDatabase</b>(from File Processing)</li>
<li><b>ExportToSpreadsheet</b> (from File Processing)</li>
<li><b>FlagImage</b>(from Image Processing)</li>
</ul>
</p>

<p>Some modules have yet to be implemented in CellProfiler 2.0. Loading these
modules will currently produce an error message with the option of skipping
the module and loading the rest of the pipeline.
<ul>
<li><b>CreateWebPage</b></li>
<li><b>DICTransform</b></li>
<li><b>DifferentiateStains</b></li>
<li><b>DisplayGridInfo</b></li>
<li><b>DisplayMeasurement</b></li>
<li><b>GroupMovieFrames</b></li>
<li><b>LabelImages</b></li>
<li><b>Restart</b></li>
<li><b>SplitOrSpliceMovie</b></li>
</ul>
</p>
"""

USING_METADATA_HELP = """
<h2>Using Metadata in CellProfiler</h2>

It is not uncommon for metadata (i.e, additional data about the data) to be included with the input images.
This information can be used by CellProfiler to group images with common metadata identifiers (or "tags") 
together for a processing run, output to a spreadsheet as annotated information or used to name 
additional input/output files. Oftentimes, this is encountered in two forms:
<ul>
<li><i>Metadata provided in the image filename or location.</i> For example, images produced by an automated
microscope may be given names similar to "Experiment1_A01_w1_s1.tif" in which the metadata about the
plate ("Experiment1"), the well ("A01"), the wavelength number ("w1") and the site ("s1") are encapsulated. The
name of the folder in which the images are saved may be meaningful and may also be considered metadata as well.
If this is the case for your data, use <b>LoadImages</b> to extract this information.</li>
<li><i>Metadata provided as a table of information</i> Often, information associated with each image (such as
treatment, plate, well, etc) is given as a separate spreadsheet. If this is the case for your data, use 
<b>LoadData</b> to load this information.</li>
</ul>
Details for the metadata-specific help is given next to the appropriate setting. However, here is an overview
of how metadata is obtained and used.

<h3>Associating images with metadata</h3>

<p>In <b>LoadImages</b>, metadata is obtained from the filename and/or folder location using regular expression, 
a specialized syntax used for text pattern-matching. These regular expressions can be used to name different 
parts of the filename / folder. The syntax <i>(?&lt;fieldname&gt;expr)</i> will extract whatever matches 
<i>expr</i> and assign it to the image's <i>fieldname</i> measurement. A regular expression tool is available 
which will allow you to check the accuracy of your regular expression.</p>

<p>For instance, a researcher uses folder names with the date and subfolders containing the
images with the run ID (e.g., <i>./2009_10_02/1234/</i>)
The following regular expression will capture the plate, well and site in the fields 
<i>Date</i> and <i>Run</i>:<br>
<table border = "1">
<tr><td colspan = "2">.*[\\\/](?P&lt;Date&gt;.*)[\\\\/](?P&lt;Run&gt;.*)$</td></tr>
<tr><td>.*[\\\\/]</td><td>Skip characters at the beginning of the pathname until either a slash (/) or
backslash (\\) is encountered (depending on the OS)</td></tr>
<tr><td>(?P&lt;Date&gt;</td><td>Name the captured field <i>Date</i></td></tr>
<tr><td>.*</td><td>Capture as many characters that follow</td></tr>
<tr><td>[\\\\/]</td><td>Discard the slash/backslash character</td></tr>
<tr><td>(?P&lt;Run&gt;</td><td>Name the captured field <i>Run</i></td></tr>
<tr><td>.*</td><td>Capture as many characters as follow</td></tr>
<tr><td>$</td><td>The <i>Run</i> field must be at the end of the path string, i.e. the
last folder on the path. This also means that the <i>Date</i> field contains the parent
folder of the <i>Date</i> folder.</td></tr>
</table>

<p>In <b>LoadData</b>, it is assumed that the metadata has already been gathered and is in the form
of a CSV (comma-separated) file. Columns whose name begins with 
"Metadata" can be used to group or associate files loaded by <b>LoadData</b>.</p>

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

<h3>Use of metadata-specific module settings</h3>

<p>Once the metadata has been obtained, you can use <i>metadata tags</i> to reference them
in later modules. A metadata tag has the syntax <i>\g&lt;metadata-tag&gt;</i> where 
<i>&lt;metadata-tag&gt</i> is the name of the previously defined metadata field. Several modules are capable of 
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
"""

MEMORY_AND_SPEED_HELP = """
<h2>Help for memory and speed issues in CellProfiler</h2>

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
To the left of each module listed in your pipeline there is an icon (an eye) that indicates whether
the module window will be displayed during the analysis run. You can turn off individual module windows by
clicking on the icon (when the eye is closed, the window 
will not be shown); select <i>Window > Hide all windows</i> to prevent display
of all module windows.</p></li>           
                                                                            
<li><p><i>Use care in object identification </i><br>                                   
If you have a large image which contains many small        
objects, a good deal of computer time will be spent processing each   
individual object, many of which you might not need. To avoid this, make 
sure that you adjust the diameter options in <b>IdentifyPrimaryObjects</b> to   
exclude small objects in which you are not interested, or use a <b>FilterObjects</b> 
module to eliminate such objects.</p></li>               
</ul>
"""

TEST_MODE_HELP = """ 
<h2>Test mode for pipeline development</h2>

<p>You can test an analysis on a selected image cycle using the <i>Test</i> mode option on 
the main menu. Test mode allows you to run the pipeline on a selected
image, preview the results and adjust the module settings on the fly.</p>

<p>To enter Test mode once you have built a pipeline, choose <i>Test > Start test run</i> in the
menu bar in the main window. At this point, you will see the following features appear:
<ul>
<li>The module view will have a slider bar appearing on the far left.</li>
<li>A Pause icon ("||") will appear to the left of each module.</li>
<li>A series of buttons will appear at the bottom of the module list above the 
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
<li><i>Pause ("||" icon):</i> Clicking the pause icon will cause the pipeline test run to halt
execution when that module is reached (the paused module itself is not executed). The icon 
changes from black to yellow to indicate that a pause has been inserted at that point.</li>
<li><i>Run:</i> Execution of the pipeline will be started/resumed until
the next module pause is reached. When all modules have been executed for a given image cycle,
execution will stop.</li>
<li><i>Step:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Next image cycle:</i> Skip ahead to the next image cycle as determined by the image 
order in <b>LoadImages</b>/<b>LoadData</b>. The slider will automatically return to the 
first module in the pipeline.</li>
</ul>
</p>
<p>From the <i>Test</i> menu, you can choose additional options:
<ul>
<li><i>Stop test run:</i> Exit <i>Test</i> mode. Loading a new pipeline or adding/subtracting
modules will also automatically exit test mode.</li>
<li><i>Step to next module:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Choose image / group:</i>Choose the image or group to jump to.
The slider will then automatically return to the first module in the pipeline.</li>
<li><i>Reload modules source:</i> For developers only. This option will reload the module source 
code, so any changes to the code will be reflected immediately.</li>
</ul>
</p>
"""

BATCHPROCESSING_HELP = """ 
<h2>Batch processing in CellProfiler</h2>

CellProfiler is designed to analyze images in a high-throughput manner.   
Once a pipeline has been established for a set of images, CellProfiler    
can export batches of images to be analyzed on a computing cluster with the         
pipeline. We often process 40,000-130,000 images for one analysis in this 
manner. We do this by breaking the entire set of images into    
separate batches, then submitting each of these batches as individual 
jobs to a cluster. Each individual batch can be separately analyzed from  
the rest.                                                                 
"""

'''The help menu for CP's main window'''
MAIN_HELP = (
    ( "Getting started", (
        ("When To Use CellProfiler",WHEN_CAN_I_USE_CELLPROFILER_HELP),
        ("New Features",NEW_FEATURES_HELP),
        ("How To Build A Pipeline", BUILDING_A_PIPELINE_HELP) ) ),
    ( "General help", (
        ("Updates To Module Names",NEW_MODULE_NAMES_HELP),
        ("Using Metadata In CellProfiler",USING_METADATA_HELP),
        ("Memory And Speed", MEMORY_AND_SPEED_HELP),
        ("Test Mode",TEST_MODE_HELP),
        ("Batch Processing", BATCHPROCESSING_HELP) ) ),
    ( "Folders and files", (
        ("Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
        ("Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
        ("Output Filename", OUTPUT_FILENAME_HELP) ) )
)

'''A couple of strings for generic insertion into module help'''
USING_METADATA_HELP_REF = ''' 
Please see <b>LoadImages</b>, <b>LoadData</b>, or <i>Help > General help > Using metadata in CellProfiler</i> 
for more details on obtaining, extracting and using metadata tags from your images'''

USING_METADATA_TAGS_REF = '''
Tags have the form <i>\g&lt;metadata-tag&gt;</i> where <i>&lt;metadata-tag&gt</i> is the name of the previously defined metadata field.'''

####################################################
#
# Help for the module figure windows
#
####################################################
'''The help menu for the figure window'''

FILE_HELP = """
You can save the figure window to a file (currently,
Postscript (.ps), PNGs and PDFs are supported). Note that this will save the entire
contents of the window, not just the individual subplot(s) or images.
"""

ZOOM_HELP = """ 
From any image or plot axes, you can zoom in or out of the field of view. 
<ul>
<li>To zoom in, click the area of the axes where you want to zoom in, or drag 
the cursor to draw a box around the area you want to zoom in on. The axes are 
redrawn, changing the limits to display the specified area.</li>
<li>Zoom out is active only when you have zoomed into the field of view. Click any
point within the current axes to zoom out to the previous zoom level; that is, each
zoom out undoes the previous zoom in.
</ul>
"""

SHOW_PIXEL_DATA_HELP = """
Select <i>Show pixel data</i> to view pixel intensity and position. 
The tool can display pixel information for all the images in a figure window.
<ul>
<li>The (x,y) coordinates are shown for the current cursor position in the figure,
at the bottom left panel of the window</li>
<li>The pixel intensity is shown in the bottom right panel of the window. For 
intensity images, the information is shown as a single intensity value. For a color
image, the red/green/blue (RGB) values are shown.</li>
<li>If you click on an image and drag, a line will appear 
between the two endpoints, and the distance between them shown at the right-most
portion of the bottom panel. This is useful for measuring distances in order to obtain
estimates of typical object diameters for use in <b>IdentifyPrimaryObjects</b>.</li>
</ul>
"""

IMAGE_TOOLS_HELP = """
Right-clicking in an image displayed in a window will bring up a pop-up menu with
the following options:
<ul>
<li><i>Open image in new window:</i> Displays the image in a new window. This is useful for getting a closer look at a window subplot that has
a small image.</li>
<li><i>Show image histogram:</i> Produces a new window containing a histogram 
of the pixel intensities in the image. This is useful for qualitatively examining
whether a threshold value determined by <b>IdentifyPrimaryObjects</b> seems 
reasonable, for example.</li>
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
    ("File", FILE_HELP),
    ("Zoom",ZOOM_HELP ),
    ("Show pixel data", SHOW_PIXEL_DATA_HELP),
    ("Image tools",IMAGE_TOOLS_HELP))

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

CHECK_FOR_UPDATES_HELP = """The <i>Check for Updates</i> preference controls how
CellProfiler looks for updates on startup."""

PREFERENCES_HELP = (
    ( "Default Input Folder", DEFAULT_IMAGE_FOLDER_HELP),
    ( "Default Output Folder", DEFAULT_OUTPUT_FOLDER_HELP),
    ( "Title font", TITLE_FONT_HELP ),
    ( "Table font", TABLE_FONT_HELP ),
    ( "Default colormap", DEFAULT_COLORMAP_HELP ),
    ( "Window background", WINDOW_BACKGROUND_HELP ),
    ( "Check for updates", CHECK_FOR_UPDATES_HELP ))

#########################################################
#
# Misc. help
#
#########################################################

'''The help to be displayed if someone asks for help on a module but none is selected'''
HELP_ON_MODULE_BUT_NONE_SELECTED = (
    "The help button can be used to obtain help for a particular module\n"
    "selected in the pipeline window at the middle left of CellProfiler.\n"
    "You currently have no modules in the pipeline. Add a module to the\n"
    'pipeline using the "+" button or by using File > Load Pipeline.')

#########################################################
#
# The top-level of help - used when building the HTML manual
#
#########################################################
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
    root = os.path.split(os.path.abspath(root))[0] # Back up one level
    webpage_path = os.path.join(root, 'help')
    if not (os.path.exists(webpage_path) and os.path.isdir(webpage_path)):
        try:
            os.mkdir(webpage_path)
        except IOError:
            webpage_path = root
    index_fd = open(os.path.join(webpage_path,'gui_index.html'),'w')
        
    index_fd.write("""
<html style="font-family:arial">
<head>
    <title>User guide</title>
</head>
<body>
<h1><a name = "user_guide">User guide</a></h1>""")
    def write_menu(prefix, h):
        index_fd.write("<ul>\n")
        for key, value in h:
            index_fd.write("<li>")
            if hasattr(value, "__iter__") and not isinstance(value, (str, unicode)):
                index_fd.write("<b>%s</b>"%key)
                write_menu(prefix+"_"+key, value)
            else:
                file_name = "%s_%s.html" % (prefix, key)
                fd = open(os.path.join(gui_path, file_name),"w")
                fd.write("<html style=""font-family:arial""><head><title>%s</title></head>\n" % key)
                fd.write("<body><h1>%s</h1>\n<div>\n" % key)
                fd.write(value)
                fd.write("</div></body>\n")
                fd.close()
                index_fd.write("<a href='%s'>%s</a>\n" % 
                               (os.path.join(gui_dir,file_name), key) )
            index_fd.write("</li>\n")
        index_fd.write("</ul>\n")
        
    gui_dir = 'gui'
    gui_path = os.path.join(webpage_path,gui_dir)
    if not (os.path.exists(gui_path) and os.path.isdir(gui_path)):
        try:
            os.mkdir(gui_path)
        except IOError:
            raise ValueError("Could not create directory %s" % gui_path)
        
    write_menu("help", HELP)
    index_fd.write("</body>\n")
    index_fd.close()
    
