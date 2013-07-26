"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
# Icon attributions for welcome screen
# Help icon: Aha-Soft - http://www.softicons.com/free-icons/toolbar-icons/free-3d-glossy-interface-icons-by-aha-soft/help-icon
# Manual icon: Double-J Design - http://www.doublejdesign.co.uk (found at http://www.softicons.com/free-icons/toolbar-icons/ravenna-3d-icons-by-double-j-design/book-icon)
# Tutorial icon: Everaldo Coelho - http://www.softicons.com/free-icons/system-icons/crystal-project-icons-by-everaldo-coelho/apps-tutorials-icon
# Forum icon: - Aha-Soft - http://www.softicons.com/free-icons/web-icons/free-3d-glossy-icons-by-aha-soft/forum-icon

import urllib
from cellprofiler.gui.help import LOCATION_MODULE_HELP_BUTTON, LOCATION_MODULE_ADD_BUTTON, MEASUREMENT_NAMING_HELP, USING_YOUR_OUTPUT_REF, TEST_MODE_HELP, RUNNING_YOUR_PIPELINE_HELP

SELECTING_IMAGES_REF = urllib.quote("Selecting images")
SELECTING_IMAGES_HELP = '''
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
Bio-Formats; see <a href="http://loci.wisc.edu/bio-formats/formats">here</a> for the formats available. Some image formats are better 
than others for image analysis. Some are <a href="http://www.techterms.com/definition/lossy">"lossy"</a> 
(information is lost in the conversion to the format) like most JPG/JPEG files; others are 
<a href="http://www.techterms.com/definition/lossless">lossless</a> (no image information is lost). For image analysis purposes, a 
lossless format like TIF or PNG is recommended.</p>

<p>If you have a subset of files that you want to analyze from the full list shown in the 
panel, you can also filter the files according to a set of rules that you specify. This is useful when, for example, you
have dragged a folder of images onto the file list panel, but the folder contains the images
from one experiment that you want to process along with images from another experiment that you
want to ignore for now. You may specify as many rules as necessary to define the desired 
list of images.</p>'''

CONFIGURE_IMAGES_REF = urllib.quote("Configure images")
CONFIGURE_IMAGES_HELP = '''
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
<p>For more information on these modules and how to configure for best performance, please see the detailed help by selecting the
module and clicking the <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img>&nbsp;button at the bottom of the pipeline panel.</p>
'''%globals()

IDENTIFY_FEATUREES_REF = urllib.quote("Identifying features")
IDENTIFY_FEATUREES_HELP = '''
<p>A hallmark of most CellProfiler pipelines is the identification of cellular features in your images, whether they are 
nuclei, organelles or something else. </p>

<table width="75%%"><tr>
<td width="50%%" valign="top">
<p>A number of modules are dedicated to the purpose of detecting these features; the <b>IdentifyPrimaryObjects</b> module is the
one that is most commonly used. The result of this module is a set of labeled <i>objects</i>; we define an object as a collection of connected pixels 
in an image which share the same label. The challenge here is to find a combination of settings that best identify the objects from 
the image, a task called <i>segmentation</i>. The typical expectation is to end up with one object for each cellular feature of interest 
(for example, each nucleus is assigned to a single object in a DNA stained image). If this is not the
case, the module settings can be adjusted to make it so (or as close as possible). In some cases, image processing modules must be used beforehand
to transform the image so it is more amenable to object detection.</p></td>
<td align="center"><img src="memory:image_to_object_dataflow.png" width="254" height="225"></td>
</tr></table>

<p>In brief, the workflow of finding objects using this module is to do the following:
<ul>
<li><i>Distinguish the foreground from background:</i> The foreground is defined as that part of the image which contains
the features of interest, as opposed to the <i>background</i> which does not. The module 
assumes that the foreground is brighter than the background, which is the case for fluorescence images; for other
types of images, other modules can be used to first invert the image, turning dark regions into bright regions 
and vice versa. </li>
<li><i>Identify the objects in each foreground region:</i> Each foreground region may contain multiple objects
of interest (for example, touching nuclei in a DNA stained image). Recognizing the presence of these objects is the
objective of this step.</li>
<li><i>Splitting clusters of objects:</i> If objects are touching each other, the final step is to separate them in
a way that reflects the actual boundaries as much as possible. This process is referred to as "declumping."</li>
</ul>
The module also contains additional settings for filtering the results of this process on the basis of size, location, etc. 
to get the final object set. At this point, the objects are ready for measurements to be made, or for further manipulations 
as a means of extracting other features.</p>

<p>Other modules are able to take the results of this module and use them in combination with additional images 
(like <b>IdentifySecondaryObjects</b>) or other objects (like <b>IdentifyTertiaryObjects</b>) to define yet more objects.
</p>

<p>For more information on these identification modules work and how to configure them for best performance, please see 
the detailed help by selecting the <b>IdentifyPrimaryObjects</b> module and clicking the <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img>&nbsp;
button at the bottom of the pipeline panel.</p>
'''%globals()

MAKING_MEASUREMENTS_REF = urllib.quote("Making measurements")
MAKING_MEASUREMENTS_HELP = '''
<p>In most cases, the reason for identifying image features is to make measurements on them. CellProfiler has a number
of modules dedicated to calculating measurements of various types, on both images and objects; these
are accessible by clicking the <img src="memory:%(LOCATION_MODULE_ADD_BUTTON)s"></img>&nbsp;button
(located underneath the pipeline panel) </p>

<p>Below is a list of measurement categories; these is not meant to be comprehensive, but are sufficient for most assays:
<table border="1" cellpadding="10">
    <tr bgcolor="#555555" align="center">
    <th><font color="#FFFFFF"><b>Measurement</b></font></th>
    <th><font color="#FFFFFF"><b>Description</b></font></th>
    <th><font color="#FFFFFF"><b>Relevant modules</b></font></th></tr>
    <tr align="center"><td><i>Count</i></td><td>The number of objects in an image.</td><td>All modules which produce a new set of objects, such as <b>IdentifyPrimaryObjects</b></td></tr>
    <tr align="center"><td><i>Location</i></td><td> The (x,y) coordinates of each object, which can be of interest in time-lapse imaging.</td>
    <td>All modules which produce a new set of objects</td></tr>
    <tr align="center"><td><i>Morphology</i></td><td> Quantities defining the geometry of the object, as defined by its external boundary.
    This includes quantities like area, perimeter, etc.</td><td><b>MeasureImageAreaOccupied</b>,<b>MeasureObjectSizeShape</b></td></tr>
    <tr align="center"><td><i>Intensity</i></td><td> In fluorescence assays, the intensity of a pixel is related to the substance labeled with
    a fluorescence marker at that location. The maximal, minimal, mean, and integrated (total) intensity of each marker
    can be measured as well as correlations in intensity between channels.</td>
    <td><b>MeasureObjectIntensity</b>, <b>MeasureImageIntensity</b>, <b>MeasureObjectRadialDistribution</b>, <b>MeasureCorrelation</b></td></tr>
    <tr align="center"><td><i>Texture</i></td><td> These quantities characterize spatial smoothness and regularity across an object, and are often useful 
    for characterizing the fine patterns of localization.</td><td><b>MeasureTexture</b></td></tr>
    <tr align="center"><td><i>Clustering</i></td><td> Spatial relationships can be characterized by adjacency descriptors, such as the number of neighboring 
    objects, the percent of the perimeter touching neighbor objects, etc.</td><td><b>MeasureObjectNeighbors</b></td></tr>
</table>
</p>

<p>For more information on these modules and how to configure them for best performance, please see 
the detailed help by selecting the module and clicking the <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img>&nbsp;
button at the bottom of the pipeline panel. You can also find details on measurement nomenclature when exporting under 
<i>%(MEASUREMENT_NAMING_HELP)s</i></p>
'''%globals()

EXPORTING_RESULTS_REF = urllib.quote("Exporting results")
EXPORTING_RESULTS_HELP = '''
<p>Writing the measurements generated by CellProfiler is necessary for downstream statistical analysis. The most
common format for export is the <i>spreadsheet</i> which is a table of values. The module <b>ExportToSpreadsheet</b>
handles the task of writing the measurements (for images, objects or both) to a file readable by Excel, or the 
spreadsheet program of your choice.</p>

<p>For larger assays, involving tens of thousands of images or more, a spreadsheet is usually insufficient to handle the massive
amounts of data generated. A <i>database</i> is a better solution in this case, although this requires more sophistication
by the user; the <b>ExportToDatabase</b> module is to be used for this task. If this avenue is needed, it is best to consult 
with your information technology department.</p>

<p>CellProfiler will not save images produce by analysis modules unless told to do so. It is often desirable to save 
the outlines of the objects identified; this can is useful as a sanity check of the object identification results or for quality control
purposes. The <b>SaveImages</b> module is used for saving images to a variety of output formats, with the 
nomenclature specified by the user.</p>

<p>For more information on these modules and how to configure them for best performance, please see 
the detailed help by selecting the module and clicking the <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img>
button at the bottom of the pipeline panel. You can also find details on various exporting options under 
<i>%(USING_YOUR_OUTPUT_REF)s</i></p>
'''%globals()
    
TEST_MODE_REF = urllib.quote("Using test mode")
RUNNING_YOUR_PIPELINE_REF = urllib.quote("Analyzing your images")

IN_APP_HELP_REF = urllib.quote("Using the help")
IN_APP_HELP_HELP = '''
In addition to the Help menu in the main CellProfiler window, there are <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img> 
buttons containing more specific documentation for using 
CellProfiler. Clicking the "?" button near the pipeline window will show information about the selected module within the pipeline, 
whereas clicking the <img src="memory:%(LOCATION_MODULE_HELP_BUTTON)s"></img> button to the right of each of the module setting 
displays help for that particular setting. 
'''%globals()

WELCOME_HELP = {
    SELECTING_IMAGES_REF: SELECTING_IMAGES_HELP,
    CONFIGURE_IMAGES_REF: CONFIGURE_IMAGES_HELP,
    IDENTIFY_FEATUREES_REF: IDENTIFY_FEATUREES_HELP,
    MAKING_MEASUREMENTS_REF: MAKING_MEASUREMENTS_HELP,
    EXPORTING_RESULTS_REF: EXPORTING_RESULTS_HELP,
    TEST_MODE_REF: TEST_MODE_HELP,
    RUNNING_YOUR_PIPELINE_REF: RUNNING_YOUR_PIPELINE_HELP,
    IN_APP_HELP_REF: IN_APP_HELP_HELP
    }

startup_main = '''<html>
<body>
<center><h1>Welcome to CellProfiler!</h1></center>
<br>
<p>CellProfiler is automated image analysis software to measure biological phenotypes in images.</p>
<br>
<br>
<table border="0" cellpadding="5" width="100%%">
<tr>
    <td colspan="3"><b><font size="+2">See how it works</font></b></td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td colspan="2"><a href="loadexample:http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cp">Load a simple pipeline</a> from our website, then click on the "Analyze images" button.</td>
</tr>
<tr>
    <td colspan="3"><b><font size="+2">Build your own pipeline</font></b></td>
</tr>
<tr>
    <td width="10">&nbsp;</td>
    <td width="110"><font size="+2">1: Start</font></td>
    <td >Download an <a href="http://www.cellprofiler.org/examples.shtml">example pipeline</a> that suits your application and load it with <i>File &gt; Open Project</i>.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><i><font size="+2">2: Adjust</font></b></td>
    <td>Use the Input modules to <a href="help://%(SELECTING_IMAGES_REF)s">select</a> and <a href="help://%(CONFIGURE_IMAGES_REF)s">configure</a> your images for analysis. 
    Add Analysis modules to <a href="help://%(IDENTIFY_FEATUREES_REF)s">identify</a> image features, make <a href="help://%(MAKING_MEASUREMENTS_REF)s">measurements</a> and 
    <a href="help://%(EXPORTING_RESULTS_REF)s">export</a> results.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><i><font size="+2">3: Test</font></b></td>
    <td>Click the "Start Test Mode" button to step through the pipeline and <a href="help://%(TEST_MODE_REF)s">check</a> the module settings on a few images.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><i><font size="+2">4: Analyze</font></b></td>
    <td>Click the "Analyze Images" button to <a href="help://%(RUNNING_YOUR_PIPELINE_REF)s">process</a> all of your images with your pipeline.</td>
</tr>
</table>
<br>
<table>
<tr>
    <td colspan="3"><b><font size="+2">Need more help?</font></b></td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td colspan="2">
        <table border="5" cellspacing="5" cellpadding="5">
        <tr>
            <td align="center" width="100"><b><font size="+1">In-App Help</font></b><br><br>
            <a href="help://%(IN_APP_HELP_REF)s"><img src="memory:welcome_screen_help.png"></a><br><br>
            Click <b>?</b> buttons for detailed help
            </td>
            <td align="center" width="100"><b><font size="+1">Manual</font></b><br><br>
            <a href="http://www.cellprofiler.org/CPmanual#table_of_contents" ><img src="memory:welcomescreen_manual.png"></a><br><br>
            Online version of In-App help
            </td>
            <td align="center" width="100"><b><font size="+1">Tutorials/Demos</font></b><br><br>
            <a href="http://www.cellprofiler.org/tutorials.shtml"><img src="memory:welcomescreen_tutorials.png"></a><br><br>
            For written and video guidance to image analysis
            </td>
            <td align="center" width="100"><b><font size="+1">Q&A Forum</font></b><br><br>
            <a href="http://www.cellprofiler.org/forum/"><img src="memory:welcomescreen_forum.png"></a><br><br>
            Post a question online
            </td>
        </tr>
        </table>
    </td>
</tr>
</table>
<p>Click <a href="pref:no_display">here</a> to stop displaying this page when CellProfiler starts. This page can be accessed from <i>Help > Show Welcome Screen</i> at any time.</p>
</body>
</html>'''%globals()

startup_interface = '''<html>
<body>
<h2>Summary of the Interface</h2>
The CellProfiler interface has tools for managing images, pipelines and modules. The interface is divided into four main parts, as shown in the following illustration:
<p>
<center>
<img src="memory:cp_panel_schematic.png"></img>
</center>
<p>
<table cellspacing="0" class="body" cellpadding="4" border="2">
<colgroup><col width="200"><col width="300%"></colgroup>
<thead><tr valign="top"><th bgcolor="#B2B2B2">Element</th><th bgcolor="#B2B2B2">Description</th></tr></thead>
<tbody>
<tr><td><i>Pipeline</i></td><td>Lists the modules in the pipeline, with controls for display and testing. Below this panel 
are tools for adding, removing, and reordering modules and getting help.</td></tr>
<tr><td><i>Files</i></td><td>Lists images and pipeline files in the current input folder.</td></tr>
<tr><td><i>Module Settings</i></td><td>Contains the options for the currently selected module.</td></tr>
<tr><td><i>Folders</i></td><td>Dialogs for controlling default input and output folders and output filename.</td></tr>
</tbody></table>
<p>Go <a href="startup_main">back</a> to the welcome screen.</p>
</body>
</html>'''

def find_link(name):
    return globals().get(name, None)
