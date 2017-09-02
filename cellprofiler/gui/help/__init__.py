# # coding=utf-8
#
# import urllib
#
# def make_help_menu(h, window, menu=None):
#     import wx
#     import htmldialog
#     if menu is None:
#         menu = wx.Menu()
#     for key, value in h:
#         my_id = wx.NewId()
#         if hasattr(value, "__iter__") and not isinstance(value, (str, unicode)):
#             menu.AppendMenu(my_id, key, make_help_menu(value, window))
#         else:
#             def show_dialog(event, key=key, value=value):
#                 dlg = htmldialog.HTMLDialog(window, key, value)
#                 dlg.Show()
#
#             menu.Append(my_id, key)
#             window.Bind(wx.EVT_MENU, show_dialog, id=my_id)
#
#     return menu
#
#
# SELECTING_IMAGES_REF = urllib.quote("Selecting images")
# CONFIGURE_IMAGES_REF = urllib.quote("Configure images")
# IDENTIFY_FEATUREES_REF = urllib.quote("Identifying features")
# IDENTIFY_FEATUREES_HELP = '''
# <p>A hallmark of most CellProfiler pipelines is the identification of cellular features in your images, whether they are
# nuclei, organelles or something else. </p>
#
# <table width="75%%" cellpadding="0">
# <tr>
# <td width="75%%" valign="top">
# A number of modules are dedicated to the purpose of detecting these features; the <b>IdentifyPrimaryObjects</b> module is the
# one that is most commonly used. The result of this module is a set of labeled <i>objects</i>; we define an object as a collection of connected pixels
# in an image which share the same label. The challenge here is to find a combination of settings that best identify the objects from
# the image, a task called <i>segmentation</i>. The typical expectation is to end up with one object for each cellular feature of interest
# (for example, each nucleus is assigned to a single object in a DNA stained image). If this is not the
# case, the module settings can be adjusted to make it so (or as close as possible). In some cases, image processing modules must be used beforehand
# to transform the image so it is more amenable to object detection.</td>
# <td width="25%%" align="center"><img src="memory:image_to_object_dataflow.png" width="254" height="225"></td>
# </tr>
# </table>
#
# <p>In brief, the workflow of finding objects using this module is to do the following:
# <ul>
# <li><i>Distinguish the foreground from background:</i> The foreground is defined as that part of the image which contains
# the features of interest, as opposed to the <i>background</i> which does not. The module
# assumes that the foreground is brighter than the background, which is the case for fluorescence images; for other
# types of images, other modules can be used to first invert the image, turning dark regions into bright regions
# and vice versa. </li>
# <li><i>Identify the objects in each foreground region:</i> Each foreground region may contain multiple objects
# of interest (for example, touching nuclei in a DNA stained image). Recognizing the presence of these objects is the
# objective of this step.</li>
# <li><i>Splitting clusters of objects:</i> If objects are touching each other, the final step is to separate them in
# a way that reflects the actual boundaries as much as possible. This process is referred to as "declumping."</li>
# </ul>
# The module also contains additional settings for filtering the results of this process on the basis of size, location, etc.
# to get the final object set. At this point, the objects are ready for measurements to be made, or for further manipulations
# as a means of extracting other features.</p>
#
# <p>Other modules are able to take the results of this module and use them in combination with additional images
# (like <b>IdentifySecondaryObjects</b>) or other objects (like <b>IdentifyTertiaryObjects</b>) to define yet more objects.
# </p>
#
# <p>For more information on these identification modules work and how to configure them for best performance, please see
# the detailed help by selecting the <b>IdentifyPrimaryObjects</b> module and clicking the <img src="memory:%(MODULE_HELP_BUTTON)s">&nbsp;
# button at the bottom of the pipeline panel.</p>
# ''' % globals()
# MAKING_MEASUREMENTS_REF = urllib.quote("Making measurements")
# MAKING_MEASUREMENTS_HELP = '''
# <p>In most cases, the reason for identifying image features is to make measurements on them. CellProfiler has a number
# of modules dedicated to calculating measurements of various types, on both images and objects; these
# are accessible by clicking the <img src="memory:%(MODULE_ADD_BUTTON)s">&nbsp;button
# (located underneath the pipeline panel) </p>
#
# <p>Below is a list of measurement categories; these is not meant to be comprehensive, but are sufficient for most assays:
# <table border="1" cellpadding="10">
#     <tr bgcolor="#555555" align="center">
#     <th><font color="#FFFFFF"><b>Measurement</b></font></th>
#     <th><font color="#FFFFFF"><b>Description</b></font></th>
#     <th><font color="#FFFFFF"><b>Relevant modules</b></font></th></tr>
#     <tr align="center"><td><i>Count</i></td><td>The number of objects in an image.</td><td>All modules which produce a new set of objects, such as <b>IdentifyPrimaryObjects</b></td></tr>
#     <tr align="center"><td><i>Location</i></td><td> The (x,y) coordinates of each object, which can be of interest in time-lapse imaging.</td>
#     <td>All modules which produce a new set of objects</td></tr>
#     <tr align="center"><td><i>Morphology</i></td><td> Quantities defining the geometry of the object, as defined by its external boundary.
#     This includes quantities like area, perimeter, etc.</td><td><b>MeasureImageAreaOccupied</b>,<b>MeasureObjectSizeShape</b></td></tr>
#     <tr align="center"><td><i>Intensity</i></td><td> In fluorescence assays, the intensity of a pixel is related to the substance labeled with
#     a fluorescence marker at that location. The maximal, minimal, mean, and integrated (total) intensity of each marker
#     can be measured as well as correlations in intensity between channels.</td>
#     <td><b>MeasureObjectIntensity</b>, <b>MeasureImageIntensity</b>, <b>MeasureObjectRadialDistribution</b>, <b>MeasureColocalization</b></td></tr>
#     <tr align="center"><td><i>Texture</i></td><td> These quantities characterize spatial smoothness and regularity across an object, and are often useful
#     for characterizing the fine patterns of localization.</td><td><b>MeasureTexture</b></td></tr>
#     <tr align="center"><td><i>Clustering</i></td><td> Spatial relationships can be characterized by adjacency descriptors, such as the number of neighboring
#     objects, the percent of the perimeter touching neighbor objects, etc.</td><td><b>MeasureObjectNeighbors</b></td></tr>
# </table>
# </p>
#
# <p>For more information on these modules and how to configure them for best performance, please see
# the detailed help by selecting the module and clicking the <img src="memory:%(MODULE_HELP_BUTTON)s">&nbsp;
# button at the bottom of the pipeline panel. You can also find details on measurement nomenclature when exporting under
# <i>%(MEASUREMENT_NAMING_HELP)s</i></p>
# ''' % globals()
# EXPORTING_RESULTS_REF = urllib.quote("Exporting results")
# EXPORTING_RESULTS_HELP = '''
# <p>Writing the measurements generated by CellProfiler is necessary for downstream statistical analysis. The most
# common format for export is the <i>spreadsheet</i> which is a table of values. The module <b>ExportToSpreadsheet</b>
# handles the task of writing the measurements (for images, objects or both) to a file readable by Excel, or the
# spreadsheet program of your choice.</p>
#
# <p>For larger assays, involving tens of thousands of images or more, a spreadsheet is usually insufficient to handle the massive
# amounts of data generated. A <i>database</i> is a better solution in this case, although this requires more sophistication
# by the user; the <b>ExportToDatabase</b> module is to be used for this task. If this avenue is needed, it is best to consult
# with your information technology department.</p>
#
# <p>CellProfiler will not save images produce by analysis modules unless told to do so. It is often desirable to save
# the outlines of the objects identified; this can is useful as a sanity check of the object identification results or for quality control
# purposes. The <b>SaveImages</b> module is used for saving images to a variety of output formats, with the
# nomenclature specified by the user.</p>
#
# <p>For more information on these modules and how to configure them for best performance, please see
# the detailed help by selecting the module and clicking the <img src="memory:%(MODULE_HELP_BUTTON)s">
# button at the bottom of the pipeline panel. You can also find details on various exporting options under
# <i>%(USING_YOUR_OUTPUT_REF)s</i></p>
# ''' % globals()
# TEST_MODE_REF = urllib.quote("Using test mode")
# RUNNING_YOUR_PIPELINE_REF = urllib.quote("Analyzing your images")
# IN_APP_HELP_REF = urllib.quote("Using the help")
# IN_APP_HELP_HELP = '''
# In addition to the Help menu in the main CellProfiler window, there are <img src="memory:%(MODULE_HELP_BUTTON)s">
# buttons containing more specific documentation for using
# CellProfiler. Clicking the "?" button near the pipeline window will show information about the selected module within the pipeline,
# whereas clicking the <img src="memory:%(MODULE_HELP_BUTTON)s"> button to the right of each of the module setting
# displays help for that particular setting.
# ''' % globals()
# WELCOME_HELP = {
#     SELECTING_IMAGES_REF: cellprofiler.gui.help.content.SELECTING_IMAGES_HELP,
#     CONFIGURE_IMAGES_REF: cellprofiler.gui.help.content.CONFIGURE_IMAGES_HELP,
#     IDENTIFY_FEATUREES_REF: IDENTIFY_FEATUREES_HELP,
#     MAKING_MEASUREMENTS_REF: MAKING_MEASUREMENTS_HELP,
#     EXPORTING_RESULTS_REF: EXPORTING_RESULTS_HELP,
#     TEST_MODE_REF: cellprofiler.gui.help.content.TEST_MODE_HELP,
#     RUNNING_YOUR_PIPELINE_REF: cellprofiler.gui.help.content.RUNNING_YOUR_PIPELINE_HELP,
#     IN_APP_HELP_REF: IN_APP_HELP_HELP
# }
# startup_main = '''<html>
# <body>
# <table border="0" cellpadding="4" width="100%%">
# <tr>
# <td colspan="3" align="center"><b><font size="+3">Welcome to CellProfiler!</font></b></td>
# </tr>
# <tr>
# <td colspan="3">CellProfiler is automated image analysis software to measure biological phenotypes in images.</td>
# </tr>
# <tr>
#     <td colspan="3"><b><font size="+2">See a pipeline in action</font></b></td>
# </tr>
# <tr>
#     <td width="1">&nbsp;</td>
#     <td colspan="2"><a href="loadexample:https://raw.githubusercontent.com/CellProfiler/examples/{}/ExampleFly/ExampleFlyURL.cppipe">Load</a> an example pipeline, then click on the "Analyze Images" button.</td>
# </tr>
# <tr>
#     <td colspan="3"><b><font size="+2">Build your own pipeline</font></b></td>
# </tr>
# <tr>
#     <td>&nbsp;</td>
#     <td width="100"><font size="+2">1: Start</font></td>
#     <td><a href="http://www.cellprofiler.org/examples.html">Download</a> a pipeline template from our website of examples. Load it with <i>File &gt; Import &gt; Pipeline from File...</i>. Run it, then modify it to suit your assay.</td>
# </tr>
# <tr>
#     <td>&nbsp;</td>
#     <td><i><font size="+2">2: Adjust</font></b></td>
#     <td>Use the Input modules to <a href="help://%(SELECTING_IMAGES_REF)s">select</a> and <a href="help://%(CONFIGURE_IMAGES_REF)s">configure</a> your images for analysis.
#     Add Analysis modules to <a href="help://%(IDENTIFY_FEATUREES_REF)s">identify</a> image features, make <a href="help://%(MAKING_MEASUREMENTS_REF)s">measurements</a> and
#     <a href="help://%(EXPORTING_RESULTS_REF)s">export</a> results.</td>
# </tr>
# <tr>
#     <td>&nbsp;</td>
#     <td><i><font size="+2">3: Test</font></b></td>
#     <td>Click the "Start Test Mode" button to step through the pipeline and <a href="help://%(TEST_MODE_REF)s">check</a> the module settings on a few images.</td>
# </tr>
# <tr>
#     <td>&nbsp;</td>
#     <td><i><font size="+2">4: Analyze</font></b></td>
#     <td>Click the "Analyze Images" button to <a href="help://%(RUNNING_YOUR_PIPELINE_REF)s">process</a> all of your images with your pipeline.</td>
# </tr>
# </table>
# <br>
# <table>
# <tr>
#     <td colspan="3"><b><font size="+2">Need more help?</font></b></td>
# </tr>
# <tr>
#     <td>&nbsp;</td>
#     <td colspan="2">
#         <table border="5" cellspacing="5" cellpadding="5">
#         <tr>
#             <td align="center" width="100"><b><font size="+1">In-App Help</font></b><br><br>
#             <a href="help://%(IN_APP_HELP_REF)s"><img src="memory:welcome_screen_help.png"></a><br><br>
#             Click <b>?</b> buttons for detailed help
#             </td>
#             <td align="center" width="100"><b><font size="+1">Manual</font></b><br><br>
#             <a href="http://d1zymp9ayga15t.cloudfront.net/CPmanual/index.html" ><img src="memory:welcomescreen_manual.png"></a><br><br>
#             Online version of In-App help
#             </td>
#             <td align="center" width="100"><b><font size="+1">Tutorials/Demos</font></b><br><br>
#             <a href="http://cellprofiler.org/tutorials.html"><img src="memory:welcomescreen_tutorials.png"></a><br><br>
#             For written and video guidance to image analysis
#             </td>
#             <td align="center" width="100"><b><font size="+1">Q&A Forum</font></b><br><br>
#             <a href="http://forum.cellprofiler.org/"><img src="memory:welcomescreen_forum.png"></a><br><br>
#             Post a question online
#             </td>
#         </tr>
#         </table>
#     </td>
# </tr>
# </table>
# <p>Click <a href="pref:no_display">here</a> to stop displaying this page when CellProfiler starts. This page can be accessed from <i>Help > Show Welcome Screen</i> at any time.</p>
# </body>
# </html>''' % globals()
# startup_interface = '''<html>
# <body>
# <h2>Summary of the Interface</h2>
# The CellProfiler interface has tools for managing images, pipelines and modules. The interface is divided into four main parts, as shown in the following illustration:
# <p>
# <center>
# <img src="memory:cp_panel_schematic.png">
# </center>
# <p>
# <table cellspacing="0" class="body" cellpadding="4" border="2">
# <colgroup><col width="200"><col width="300%"></colgroup>
# <thead><tr valign="top"><th bgcolor="#B2B2B2">Element</th><th bgcolor="#B2B2B2">Description</th></tr></thead>
# <tbody>
# <tr><td><i>Pipeline</i></td><td>Lists the modules in the pipeline, with controls for display and testing. Below this panel
# are tools for adding, removing, and reordering modules and getting help.</td></tr>
# <tr><td><i>Files</i></td><td>Lists images and pipeline files in the current input folder.</td></tr>
# <tr><td><i>Module Settings</i></td><td>Contains the options for the currently selected module.</td></tr>
# <tr><td><i>Folders</i></td><td>Dialogs for controlling default input and output folders and output filename.</td></tr>
# </tbody></table>
# <p>Go <a href="startup_main">back</a> to the welcome screen.</p>
# </body>
# </html>'''
