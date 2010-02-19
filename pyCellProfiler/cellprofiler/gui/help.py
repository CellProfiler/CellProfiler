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
<p>The <i>Default Input folder</i> is the folder that contains the input images
that you want to analyze. Several File Processing modules (e.g., <b>Load 
Images</b> or <b>Load Data</b>) provide the option of retrieveing images from 
this folder as the default unless specified otherwise. We recommend using the 
Default Input folder for this purpose since these locations are set relative to 
the local machine, and will not cause your pipeline to fail if you transfer it
to another machine with a different drive mapping.</p>

<p>Use the Browse button (to the right of the text box) to select the folder, or 
type the full folder path in the edit box. If you type the folder path and it 
cannot be found, the message box below will indicate this fact until corrected. 
If you specify a folder which does not exist but you want to create it, you can 
click on the <i>New folder</i> icon to the right of the <i>Browse folder</i> icon 
to create it. You can change the folder which is the default image folder upon 
CellProfiler startup by using <i>File > Preferences...</i> in the main window.</p>

<p>The contents of the folder are shown to the left, which allows you to confirm 
recognized file names or view recognized images from within CellProfiler.
Double-clicking image file names in this list will open them in a figure window.             
Double-clicking on .mat pipeline or output files (CellProfiler 1.0) or .cp 
pipeline files (CellProfiler 2.0) will ask if you want to load a       
pipeline from the file. To refresh the contents of this window, press    
enter in the default image directory edit box.</p>

<p>You will have the option within the <b>Load Images</b> or <b>Load Data</b>
module to retrieve images from other folders, but selecting a folder here allows
for pipeline portability (as opposed to explicitly specifying a folder in those 
modules, which may not exist on someone else's computer).</p>"""

DEFAULT_OUTPUT_FOLDER_HELP = """
<p>The <i>Default Output folder</i> is the folder that CellProfiler uses to
store its output. Several File Processing modules (e.g., <b>Save Images</b> or 
<b>Save To Spreadsheet </b>) provide the option of saving analysis results to 
this folder as the default unless specified otherwise. We recommend using the 
Default Output folder for this purpose since these locations are set relative to 
the local machine, and will not cause your pipeline to fail if you transfer it
to another machine with a different drive mapping.</p>

<p>Use the Browse button (to the right of the text box) to select the folder, or 
type the full folder path in the edit box. If you type the folder path and it 
cannot be found, the message box below will indicate this fact until corrected. 
If you specify a folder which does not exist but you want to create it, you can 
click on the <i>New folder</i> icon to the right of the <i>Browse folder</i> icon 
to create it. You can change the folder which is the default image folder upon 
CellProfiler startup by using <i>File > Preferences...</i> in the main window.</p>"""

OUTPUT_FILENAME_HELP = """
<p>Specify the name of the output file, which is where all of the information 
about the analysis as well as any measurements are stored. The output file is a 
.mat file which is readable by MATLAB.</p>

<p>The output file will be saved in the Default Output folder unless you type a 
full path and file name into the output file name box. The path must not have 
spaces or characters disallowed by your platform.</p>
                                                                           
<p>If the output filename ends in <i>OUT.mat</i> (the typical text appended to 
an output filename), CellProfiler will prevent you from overwritting this file 
on a subsequent run by generating a new file name and prompting if you want to 
use it.</p>"""

WHEN_CAN_I_USE_CELLPROFILER_HELP = """ """

BUILDING_A_PIPELINE_HELP = """
<h2>Making a pipeline</h2>
<p>A <i>pipeline</i> is a sequential set of individual image analysis modules. The 
best way to learn how to use CellProfiler is to load an example pipeline 
from the Examples page on the CellProfiler website and try it out. Or, you can build a 
pipeline from scratch. See also the "?" button in the main window to get
help for a specific module.</p>

<p>To learn how to program in CellProfiler, see <i>Help > Developer's Guide</i>. 
To learn how to use a cluster of computers to process 
large batches of images, see <i>Help > General Help > Batch Processing</i>.</p>

<h3>Loading a pipeline</h3>
<ol>
<li>Put the images and pipeline into a folder on your computer.</li>
<li> Set the default image and output folders (lower right of the main 
window) to be the folder where you put the images.</li> 
<li>Load the pipeline using File > Load Pipeline in the main menu of 
CellProfiler.</li> 
<li>Click "Analyze images" to start processing.</li> 
<li>Examine the measurements using Data Tools. Data Tools are accessible in 
the main menu of CellProfiler and allow you to plot, view, or export your 
measurements (e.g. to Excel).</li>   
<li>If you modify the modules or settings in the pipeline, you can save the 
pipeline using File > Save Pipeline. See the end of this document for more 
information on pipeline files.</li> 
</ol>

<h3>Building a pipeline from scratch</h3>
<ol>
<li><p><i>Place modules in a new pipeline.</i><br>
Choose image analysis modules to add to your pipeline by clicking '+' or 
right-clicking in the module list window and selecting a module from the 
pop-up box that appears. Typically, the first module which must be run is 
the <b>Load Images</b> module, where you specify the identity of the images 
that you want to analyze. Modules are added to the end of the pipeline, but 
their order can be adjusted in the main window by selecting module (or
modules by using the shift key) and using the <i>Move up</i> ('^') and 
<i>Move down</i> ('v') buttons. The '-' button will delete the selected 
module(s) from the pipeline.</p> 
<p>Most pipelines depend on a major step: identifying the objects. In 
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
<li>Identify Tertiary modules require images where two sets of objects have 
already been identified (e.g., nuclei and cell regions are used to define the 
cytoplasm objects, which are tertiary objects).</li>
</ul></p>
<p>A note on saving images in your pipeline: Due to the typically high number 
of intermediate images produced during processing, images produced during 
processing are not saved to the hard drive unless you specifically request it, 
using a <b>Save Images</b> module.</p></li> 

<li><p><i>Adjust the settings in each module.</i><br>
In the CellProfiler main window, click a module in the pipeline to see its 
settings in the main workspace. To learn more about the settings for each 
module, select the module in the pipeline and click the "?" button to the 
right of each setting, or click the "?" button at the bottom of the module
list window for the help for all the settings for that module.</p>
</li>
<li><p>Set the default image folder, default output folder and output filename.
For more help, click their nearby "?" buttons in the main window. </p></li>

<li><p><i>Click Analyze images to start processing.</i><br> 
All of the images in the selected folder(s) will be analyzed using the modules 
and settings you have specified.  You will have the option to cancel at any time. 
At the end of each cycle, the measurements are saved in the output file.</p></li>

<li><p><i>Using test mode to preview results</i><br>
You can test an analysis on a selected image cycle using the <i>Test</i> mode from 
the main menu. Using this setting will allow you to run the pipeline on a selected
image, preview the results and adjust the module settings on the fly. See 
<i>Help > General Help > Test Mode </i> for more details.</p>
</li>
<li><p>Save your pipeline via <i>File > Save Pipeline</i>.</p>
</li>
</ol>
"""

MEMORY_AND_SPEED_HELP = """
<h2>Help for memory and speed issues in CellProfiler</h2>

<p>There are several options in CellProfiler for dealing with out-of-memory
errors associated with analyzing images: </p>
<ul>
<li><p><i>Resize the input images.</i><br>
If the image is high-resolution, it may be helpful to determine whether the 
features of interest can be processed (and accurate data obtained) by using a 
lower-resolution image. If this is the  case, use the <b>Resize</b> module (under 
<i>Image Processing</i> category) to scale down the image to a more manageable size, and
perform the desired operations on the smaller image.</p></li>

<li><p><i>Use the <b>ConserveMemory</b> module.</i><br>                                 
The ConserveMemory module permits the user to clear the images stored in memory 
with the exception of those specified by the user. Please see the help for the 
ConserveMemory module for more details.</p></li>
</ul>

<p>In addition, there are several options in CellProfiler for dealing with 
out-of-memory errors associated with analyzing images: </p>

<ul>
<li><p><i>Running without display windows</i><br>
Each module is associated with a display window which takes time to render and/or
update. CLosing these windows gives some amount of gain in speed. Do do this,
to the left of each module, there is an icon (an eye) which indicates the whether
the module window will be displayed during the analysis run. You can turn off any
of the module windows by either clicking on the icon (when the eye is closed, the window 
will not be shown) or selecting <i>Window > Hide all windows</i> to prevent display
of all module windows. Once your pipeline is properly set up, we recommend    
running the entire cycle without any windows displayed.</p></li>           
                                                                            
<li><p><i>Use care in object identification </i><br>                                   
If you have a large image which contains a large number of small        
objects, a good deal of computer time will be used in processing each   
individual object, many of which you might not need. In this case, make 
sure that you adjust the diameter options in <b>IdentifyPrimaryObjects</b> to   
exclude small objects you are not interested in, or use a <b>FilterObjects</b> 
module to eliminate objects that are not of interest.</p></li>               
</ul>
"""

TEST_MODE_HELP = """ 
<h2>Test mode for pipeline development</h2>

<p>You can test an analysis on a selected image cycle using the <i>Test</i> mode from 
the main menu. Using this setting will allow you to run the pipeline on a selected
image, preview the results and adjust the module settings on the fly.</p>

<p>You can enter into Test mode at any time via <i>Test > Start test run</i> in the
menu bar in the main GUI. At this point, you will see the following features appear
<ul>
<li>The module view will have a slider bar appearing on the far left.</li>
<li>A Pause icon ("||") appearing to the left of each module.</li>
<li>A series of buttons will appear at the bottom of the module list above the 
module adjustment buttons.</li>
<li>The grayed out items in the <i>Test</i> menu will become active, and the 
<i>Analyze Images</i> button will become inactive.
</ul>
</p>

<p>You can run your pipeline in Test mode by selecting <i>Test > Step to next module</i>
or click the <i>Run</i> button. The pipeline will execute normally but you will
be able to back up to a previous module or jump to a downstream module, change
module settings to see the results, or execute the pipeline on the image of your choice.
The additional controls will allow you to do the following:
<ul>
<li><i>Slider:</i> Execution of the pipeline can be started/resumed at any module in the 
pipeline by moving the slider. However, if the selected module depends on objects and/or images
generated by prior modules, an error will be produced indicating that this data has not 
been produced yet. To solve this, it is best to actually run the pipeline up to the module
of interest, and move the slider to modules already executed.
<li><i>Pause ("||" icon):</i> Clicking the pause icon will cause the pipeline test run to halt
execution when that module is reached (the paused module itself is not executed). The icon 
changes from black to yellow to indicate that a pause has been inserted at that point.</li>
<li><i>Run:</i> Execution of the pipeline will be started/resumed until
the next module pause is reached. When all modules have been executed for a given image set,
execution will stop.</li>
<li><i>Step:</i></li> Execute the next module (as indicated by the slider location)</li>
<li><i>Next image set:</i> Skip ahead to the next image set as determined by the image 
order in <b>LoadImages</b>/<b>LoadData</b>. The slider will automatically return to the 
first module in the pipeline.</li>
</ul>
</p>
<p>From the <i>Test</i> menu, you can also choose additional options:
<ul>
<li><i>Stop test run:</i>This exits <i>Test</i> mode. Loading a new pipeline or adding/subtracting
modules will also automatically exit test mode.</li>
<li><i>Step to next module:</i> Execute the next module (as indicated by the slider location)</li>
<li><i>Choose image set / group:</i> This allows you to choose the image set or group to jump to.
Upon choosing, the slider will automatically return to the first module in the pipeline.</li>
<li><i>Reload modules source:</i> For developers only. This option will reload the module source 
code so any changes to the code will be reflected immediately.</li>
</ul>
</p>
"""

BATCHPROCESSING_HELP = """ 
<h2>Batch processing in CellProfiler</h2>

CellProfiler is designed to analyze images in a high-throughput manner.   
Once a pipeline has been established for a set of images, CellProfiler    
can export batches of images to be analyzed on a computing cluster with the         
pipeline. We often analyze 40,000-130,000 images for one analysis in this 
manner. This is accomplished by breaking the entire set of images into    
separate batches, and then submitting each of these batches as individual 
jobs to a cluster. Each individual batch can be separately analyzed from  
the rest.                                                                 
"""

'''The help menu for CP's main window'''
MAIN_HELP = (
    ( "Getting started", (
        ("When to use CellProfiler",WHEN_CAN_I_USE_CELLPROFILER_HELP),
        ("How to build a pipeline", BUILDING_A_PIPELINE_HELP) ) ),
    ( "General help", (
        ("Memory and Speed", MEMORY_AND_SPEED_HELP),
        ("Test Mode",TEST_MODE_HELP),
        ("Batch Processing", BATCHPROCESSING_HELP) ) ),
    ( "Folders and files", (
        ("Default image folder", DEFAULT_IMAGE_FOLDER_HELP),
        ("Default output folder", DEFAULT_OUTPUT_FOLDER_HELP),
        ("Output file name", OUTPUT_FILENAME_HELP) ) )
)

####################################################
#
# Help for the module figure windows
#
####################################################
'''The help menu for the figure window'''

FILE_HELP = """
Under this item, you have the option of saving the figure window to a file (currently,
Postscript (.ps), PNGs and PDFs are supported). Note that this will save the entire
contents of the window, not just the individual subplot(s) or images.
"""

ZOOM_HELP = """ 
From any image or plot axes, you can zoom in or out of the field of view. 
<ul>
<li>To zoom in, click the area of the axes where you want to zoom in, or drag 
the cursor to draw a box around the area you want to zoom in on. The axes is 
redrawn, changing the limits to display the specified area.</li>
<li>Zoom out is active only when you have zoomed into the field of view. Click any
point within the current axes to zoom out to the previous zoom level; that is, each
zoom out undoes the previous zoom in.
</ul>
"""

SHOW_PIXEL_DATA_HELP = """
Selecting <i>Show pixel data</i> allows the user to view pixel intensity and position. 
The tool can display pixel information for all the images in a figure window.
<ul>
<li>The (x,y) coordinates are shown for the current cursor position in the figure,
at the bottom left panel of the window</li>
<li>The pixel intensty is shown in the bottom right panel of the window. For 
intensity images, the information is shown as a single intensity value. For a color
image, the red/green/blue (RGB) values are shown.</li>
<li>Additionally, by clicking on an image and dragging, you will see a line drawn
between the two endpoints, and the distance between them shown at right-most
portion of the bottom panel. This is useful for measuring distances to obtains
estimates of typical object diameters for use in <b>IdentifyPrimaryObjects</b>.</li>
</ul>
"""

IMAGE_TOOLS_HELP = """
Right-clicking in an image displayed in a window will bring up a pop-up menu with
the following options:
<ul>
<li><i>Open image in new window:</i> This will create a new window with the image
displayed. This is useful for getting a closer look at a window subplot that has
a small image.</li>
<li><i>Show image histogram:</i> This will produce a new window with a histogram 
of the pixel intensities in the image. This is useful for qualitiatively examining
whether a threshold value determined by <b>IdentifyPrimaryObjects</b> seems 
reasonable, for example.</li>
<li><i>Image contrast:</i>You have three options for how the color/intensity values in 
the images are displayed:
<ul>
<li><i>Raw:</i> Shows the image using the full colormap range permissible for the
image type. For example, for a 16-bit image, the pixel data will be shown using 0 as black
and 65535 as white. However, if the actual pixel intensities span only a portion of the
image intensity range, this may render the image unviewable. For example, if a 16-bit image
only contains 12 bits of data, the resultant image will be entirely black.</li>
<li><i>Normalized (default):</i>Shows the image with the colormap "autoscaled" to
the maximum and minimum pixel intensity values; the minimum value is black and the
maximum value is white. </li>
<li><i>Log normalized:</i> Same as <i>Normalized</i> except that the color values
are then log transformed. This is useful for when the pixel intensity span a wide
range of values but the standard deviation is small (e.g., the majority of the 
interesting information is located at the dimm values). Using this option 
increases the effective contrast.</li>
</ul>
</li>
<li><i>Channels:</i> For color images only. You can show any combination of the red, 
green and blue color channels.</li>
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
in titles above plots in module figure windows."""

TABLE_FONT_HELP = """The <i>Table Font</i> preference sets the font that's used
in tables in module figure windows."""

DEFAULT_COLORMAP_HELP = """The <i>Default Colormap</i> preference chooses the
color map that's used to get the colors for labels and other elements. See this
<a href ="http://www.astro.princeton.edu/~msshin/science/code/matplotlib_cm/">
page</a> for pictures of available colormaps."""

WINDOW_BACKGROUND_HELP = """The <i>Window Background</i> preference sets the
window background color of the CellProfiler main window."""

CHECK_FOR_UPDATES_HELP = """The <i>Check for Updates</i> preference controls how
CellProfiler looks for updates on startup."""

PREFERENCES_HELP = (
    ( "Default image folder", DEFAULT_IMAGE_FOLDER_HELP),
    ( "Default output folder", DEFAULT_OUTPUT_FOLDER_HELP),
    ( "Title font", TITLE_FONT_HELP ),
    ( "Table font", TABLE_FONT_HELP ),
    ( "Default colormap", DEFAULT_COLORMAP_HELP ),
    ( "Window background", WINDOW_BACKGROUND_HELP ),
    ( "Check for updates", CHECK_FOR_UPDATES_HELP ))

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
    