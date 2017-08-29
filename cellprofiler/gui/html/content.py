# coding=utf-8

import urllib

import cellprofiler.gui.help

SELECTING_IMAGES_REF = urllib.quote("Selecting images")

CONFIGURE_IMAGES_REF = urllib.quote("Configure images")

IDENTIFY_FEATUREES_REF = urllib.quote("Identifying features")

IDENTIFY_FEATUREES_HELP = """\
A hallmark of most CellProfiler pipelines is the identification of
cellular features in your images, whether they are nuclei, organelles or
something else.

A number of modules are dedicated to the purpose of detecting these features;
the **IdentifyPrimaryObjects** module is the one that is most commonly used.
The result of this module is a set of labeled *objects*; we define an object
as a collection of connected pixels in an image which share the same label.
The challenge here is to find a combination of settings that best identify
the objects from the image, a task called *segmentation*. The typical
expectation is to end up with one object for each cellular feature of
interest (for example, each nucleus is assigned to a single object in a DNA
stained image). If this is not the case, the module settings can be adjusted
to make it so (or as close as possible). In some cases, image processing
modules must be used beforehand to transform the image so it is more amenable
to object detection.

|image0|

In brief, the workflow of finding objects using this module is to do the
following:

-  *Distinguish the foreground from background:* The foreground is
   defined as that part of the image which contains the features of
   interest, as opposed to the *background* which does not. The module
   assumes that the foreground is brighter than the background, which is
   the case for fluorescence images; for other types of images, other
   modules can be used to first invert the image, turning dark regions
   into bright regions and vice versa.
-  *Identify the objects in each foreground region:* Each foreground
   region may contain multiple objects of interest (for example,
   touching nuclei in a DNA stained image). Recognizing the presence of
   these objects is the objective of this step.
-  *Splitting clusters of objects:* If objects are touching each other,
   the final step is to separate them in a way that reflects the actual
   boundaries as much as possible. This process is referred to as
   “declumping.”

The module also contains additional settings for filtering the results
of this process on the basis of size, location, etc. to get the final
object set. At this point, the objects are ready for measurements to be
made, or for further manipulations as a means of extracting other
features.

Other modules are able to take the results of this module and use them
in combination with additional images (like
**IdentifySecondaryObjects**) or other objects (like
**IdentifyTertiaryObjects**) to define yet more objects.

For more information on these identification modules work and how to con

.. |image0| image:: memory:image_to_object_dataflow.png
   :width: 254px
   :height: 225px
""".format(**{
    "MODULE_HELP_BUTTON": cellprofiler.gui.help.MODULE_HELP_BUTTON
})

MAKING_MEASUREMENTS_REF = urllib.quote("Making measurements")
MAKING_MEASUREMENTS_HELP = """\
In most cases, the reason for identifying image features is to make
measurements on them. CellProfiler has a number of modules dedicated to
calculating measurements of various types, on both images and objects;
these are accessible by clicking the |image0| button (located underneath
the pipeline panel)

Below is a list of measurement categories; these is not meant to be
comprehensive, but are sufficient for most assays:

+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| \ **Measurement**\    | \ **Description**\                                                                                                                                                                                                                                                                     | \ **Relevant modules**\                                                                                              |
+=======================+========================================================================================================================================================================================================================================================================================+======================================================================================================================+
| *Count*               | The number of objects in an image.                                                                                                                                                                                                                                                     | All modules which produce a new set of objects, such as **IdentifyPrimaryObjects**                                   |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| *Location*            | The (x,y) coordinates of each object, which can be of interest in time-lapse imaging.                                                                                                                                                                                                  | All modules which produce a new set of objects                                                                       |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| *Morphology*          | Quantities defining the geometry of the object, as defined by its external boundary. This includes quantities like area, perimeter, etc.                                                                                                                                               | **MeasureImageAreaOccupied**,\ **MeasureObjectSizeShape**                                                            |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| *Intensity*           | In fluorescence assays, the intensity of a pixel is related to the substance labeled with a fluorescence marker at that location. The maximal, minimal, mean, and integrated (total) intensity of each marker can be measured as well as correlations in intensity between channels.   | **MeasureObjectIntensity**, **MeasureImageIntensity**, **MeasureObjectRadialDistribution**, **MeasureCorrelation**   |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| *Texture*             | These quantities characterize spatial smoothness and regularity across an object, and are often useful for characterizing the fine patterns of localization.                                                                                                                           | **MeasureTexture**                                                                                                   |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| *Clustering*          | Spatial relationships can be characterized by adjacency descriptors, such as the number of neighboring objects, the percent of the perimeter touching neighbor objects, etc.                                                                                                           | **MeasureObjectNeighbors**                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+

For more information on these modules and how to configure them for best
performance, please see the detailed help by selecting the module and
clicking the |image1|  button at the bottom of the pipeline panel. You
can also find details on measurement nomenclature when exporting under
*{MEASUREMENT_NAMING_HELP}*

.. |image0| image:: {MODULE_ADD_BUTTON}
.. |image1| image:: {MODULE_HELP_BUTTON}
""".format(**{
    "MEASUREMENT_NAMING_HELP": cellprofiler.gui.help.MEASUREMENT_NAMING_HELP,
    "MODULE_ADD_BUTTON": cellprofiler.gui.help.MODULE_ADD_BUTTON,
    "MODULE_HELP_BUTTON": cellprofiler.gui.help.MODULE_HELP_BUTTON
})

EXPORTING_RESULTS_REF = urllib.quote("Exporting results")
EXPORTING_RESULTS_HELP = """\
Writing the measurements generated by CellProfiler is necessary for
downstream statistical analysis. The most common format for export is
the *spreadsheet* which is a table of values. The module
**ExportToSpreadsheet** handles the task of writing the measurements
(for images, objects or both) to a file readable by Excel, or the
spreadsheet program of your choice.

For larger assays, involving tens of thousands of images or more, a
spreadsheet is usually insufficient to handle the massive amounts of
data generated. A *database* is a better solution in this case, although
this requires more sophistication by the user; the **ExportToDatabase**
module is to be used for this task. If this avenue is needed, it is best
to consult with your information technology department.

CellProfiler will not save images produce by analysis modules unless
told to do so. It is often desirable to save the outlines of the objects
identified; this can is useful as a sanity check of the object
identification results or for quality control purposes. The
**SaveImages** module is used for saving images to a variety of output
formats, with the nomenclature specified by the user.

For more information on these modules and how to configure them for best
performance, please see the detailed help by selecting the module and
clicking the |image0| button at the bottom of the pipeline panel. You
can also find details on various exporting options under
*{USING_YOUR_OUTPUT_REF}*

.. |image0| image:: {MODULE_HELP_BUTTON}
""".format(**{
    "MODULE_HELP_BUTTON": cellprofiler.gui.help.MODULE_HELP_BUTTON,
    "USING_YOUR_OUTPUT_REF": cellprofiler.gui.help.USING_YOUR_OUTPUT_REF
})

TEST_MODE_REF = urllib.quote("Using test mode")
RUNNING_YOUR_PIPELINE_REF = urllib.quote("Analyzing your images")

IN_APP_HELP_REF = urllib.quote("Using the help")
IN_APP_HELP_HELP = """\
In addition to the Help menu in the main CellProfiler window, there are
|image0| buttons containing more specific documentation for using
CellProfiler. Clicking the “?” button near the pipeline window will show
information about the selected module within the pipeline, whereas
clicking the |image1| button to the right of each of the module setting
displays help for that particular setting.

.. |image0| image:: {MODULE_HELP_BUTTON}
.. |image1| image:: {MODULE_HELP_BUTTON}
""".format(**{
    "MODULE_HELP_BUTTON": cellprofiler.gui.help.MODULE_HELP_BUTTON
})

WELCOME_HELP = {
    SELECTING_IMAGES_REF: cellprofiler.gui.help.SELECTING_IMAGES_HELP,
    CONFIGURE_IMAGES_REF: cellprofiler.gui.help.CONFIGURE_IMAGES_HELP,
    IDENTIFY_FEATUREES_REF: IDENTIFY_FEATUREES_HELP,
    MAKING_MEASUREMENTS_REF: MAKING_MEASUREMENTS_HELP,
    EXPORTING_RESULTS_REF: EXPORTING_RESULTS_HELP,
    TEST_MODE_REF: cellprofiler.gui.help.TEST_MODE_HELP,
    RUNNING_YOUR_PIPELINE_REF: cellprofiler.gui.help.RUNNING_YOUR_PIPELINE_HELP,
    IN_APP_HELP_REF: IN_APP_HELP_HELP
}

startup_main = """\
<table border="0" cellpadding="4" width="100%">
    <tr>
        <td align="center" colspan="3"><b><font size="+3">Welcome to CellProfiler!</font></b></td>
    </tr>
    <tr>
        <td colspan="3">CellProfiler is automated image analysis software to measure biological phenotypes in
        images.</td>
    </tr>
    <tr>
        <td colspan="3"><b><font size="+2">See a pipeline in action</font></b></td>
    </tr>
    <tr>
        <td width="1">&nbsp;</td>
        <td colspan="2">
            <a href="loadexample:http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cppipe">Load</a> an example
            pipeline, then click on the "Analyze Images" button.
        </td>
    </tr>
    <tr>
        <td colspan="3"><b><font size="+2">Build your own pipeline</font></b></td>
    </tr>
    <tr>
        <td>&nbsp;</td>
        <td width="100"><font size="+2">1: Start</font></td>
        <td>
            <a href="http://www.cellprofiler.org/examples.html">Download</a> a pipeline template from our website of
            examples. Load it with <i>File &gt; Import &gt; Pipeline from File...</i>. Run it, then modify it to suit
            your assay.
        </td>
    </tr>
    <tr>
        <td>&nbsp;</td>
        <td><i><font size="+2">2: Adjust</font></i></td>
        <td>
            Use the Input modules to <a href="help://{SELECTING_IMAGES_REF}">select</a> and <a href=
            "help://{CONFIGURE_IMAGES_REF}">configure</a> your images for analysis. Add Analysis modules to <a href=
            "help://{IDENTIFY_FEATUREES_REF}">identify</a> image features, make <a href=
            "help://{MAKING_MEASUREMENTS_REF}">measurements</a> and <a href="help://{EXPORTING_RESULTS_REF}">export</a>
            results.
        </td>
    </tr>
    <tr>
        <td>&nbsp;</td>
        <td><i><font size="+2">3: Test</font></i></td>
        <td>
            Click the "Start Test Mode" button to step through the pipeline and <a href=
            "help://{TEST_MODE_REF}">check</a> the module settings on a few images.
        </td>
    </tr>
    <tr>
        <td>&nbsp;</td>
        <td><i><font size="+2">4: Analyze</font></i></td>
        <td>
            Click the "Analyze Images" button to <a href="help://{RUNNING_YOUR_PIPELINE_REF}">process</a> all of your
            images with your pipeline.
        </td>
    </tr>
</table><br>
<table>
    <tr>
        <td colspan="3"><b><font size="+2">Need more help?</font></b></td>
    </tr>
    <tr>
        <td>&nbsp;</td>
        <td colspan="2">
            <table border="5" cellpadding="5" cellspacing="5">
                <tr>
                    <td align="center" width="100">
                        <b><font size="+1">In-App Help</font></b><br>
                        <br>
                        <a href="help://{IN_APP_HELP_REF}"><img src="memory:welcome_screen_help.png"></a><br>
                        <br>
                        Click <b>?</b> buttons for detailed help
                    </td>
                    <td align="center" width="100">
                        <b><font size="+1">Manual</font></b><br>
                        <br>
                        <a href="http://d1zymp9ayga15t.cloudfront.net/CPmanual/index.html"><img src=
                        "memory:welcomescreen_manual.png"></a><br>
                        <br>
                        Online version of In-App help
                    </td>
                    <td align="center" width="100">
                        <b><font size="+1">Tutorials/Demos</font></b><br>
                        <br>
                        <a href="http://cellprofiler.org/tutorials.html"><img src=
                        "memory:welcomescreen_tutorials.png"></a><br>
                        <br>
                        For written and video guidance to image analysis
                    </td>
                    <td align="center" width="100">
                        <b><font size="+1">Q&A Forum</font></b><br>
                        <br>
                        <a href="http://forum.cellprofiler.org/"><img src="memory:welcomescreen_forum.png"></a><br>
                        <br>
                        Post a question online
                    </td>
                </tr>
            </table>
        </td>
    </tr>
</table>
<p>Click <a href="pref:no_display">here</a> to stop displaying this page when CellProfiler starts. This page can be
accessed from <i>Help &gt; Show Welcome Screen</i> at any time.</p>
""".format(**{
    "CONFIGURE_IMAGES_REF": CONFIGURE_IMAGES_REF,
    "EXPORTING_RESULTS_REF": EXPORTING_RESULTS_REF,
    "IDENTIFY_FEATUREES_REF": IDENTIFY_FEATUREES_REF,
    "IN_APP_HELP_REF": IN_APP_HELP_REF,
    "MAKING_MEASUREMENTS_REF": MAKING_MEASUREMENTS_REF,
    "RUNNING_YOUR_PIPELINE_REF": RUNNING_YOUR_PIPELINE_REF,
    "SELECTING_IMAGES_REF": SELECTING_IMAGES_REF,
    "TEST_MODE_REF": TEST_MODE_REF
})

startup_interface = """\
<h2>Summary of the Interface</h2>The CellProfiler interface has tools for managing images, pipelines and modules. The
interface is divided into four main parts, as shown in the following illustration:
<p></p>
<center>
    <img src="memory:cp_panel_schematic.png">
</center>
<p></p>
<table border="2" cellpadding="4" cellspacing="0" class="body">
    <colgroup>
        <col width="200">
        <col width="300%">
    </colgroup>
    <thead>
        <tr valign="top">
            <th bgcolor="#B2B2B2">Element</th>
            <th bgcolor="#B2B2B2">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><i>Pipeline</i></td>
            <td>Lists the modules in the pipeline, with controls for display and testing. Below this panel are tools
            for adding, removing, and reordering modules and getting help.</td>
        </tr>
        <tr>
            <td><i>Files</i></td>
            <td>Lists images and pipeline files in the current input folder.</td>
        </tr>
        <tr>
            <td><i>Module Settings</i></td>
            <td>Contains the options for the currently selected module.</td>
        </tr>
        <tr>
            <td><i>Folders</i></td>
            <td>Dialogs for controlling default input and output folders and output filename.</td>
        </tr>
    </tbody>
</table>
<p>Go <a href="startup_main">back</a> to the welcome screen.</p>
"""


def find_link(name):
    return globals().get(name, None)
