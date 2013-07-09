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

startup_main = '''<html>
<body>
<center><h1>Welcome to CellProfiler!</h1></center>

<p>CellProfiler is automated image analysis software designed to measure biological phenotypes in images.</p>
<br>
<br>
<table border="0" cellpadding="1" width="100%">
<tr>
    <td colspan="3"><h2>See how it works</h2></td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td colspan="2"><a href="loadexample:http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cp">Load a simple pipeline</a> from our website, then click on the "Analyze images" button.</td>
</tr>
<tr>
    <td colspan="3"><h2>Build your own pipeline</h2></td>
</tr>
<tr>
    <td width="50">&nbsp;</td>
    <td width="150"><h4>1: Start</h4></td>
    <td >Download an <a href="http://www.cellprofiler.org/examples.shtml">example pipeline</a> that suits your application and load it with <i>File &lt; Open Project</i>.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><h4>2: Adjust</h4></td>
    <td>Use the input module to select and configure your images for analysis. Add analysis module to identify image features, make measurements and export results.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><h4>3: Test</h4></td>
    <td>Click the "Test mode" button to step through the pipeline and check the module settings on a few images.</td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td><h4>4: Analyze</h4></td>
    <td>Click the "Analyze images" button to process all of your images with your pipeline.</td>
</tr>
</table>
<table>
<tr>
    <td colspan="3"><h2>Need more help?</h2></td>
</tr>
<tr>
    <td>&nbsp;</td>
    <td colspan="2">
        <table border="5" cellspacing="10" cellpadding="10">
        <tr>
            <td align="center" width="150"><p><h5>In-App Help</h5></p>
            <p><button type="button"><a href=""><img src="memory:welcome_screen_help.png"></a></button></p>
            <p>Click <b>?</b> buttons<br>for detailed help</p>
            </td>
            <td align="center" width="150"><p><h5>Manual</h5></p>
            <p><a href="http://www.cellprofiler.org/CPmanual#table_of_contents" ><img src="memory:welcomescreen_manual.png"></a></p>
            <p>Online version of<br>detailed help</p></a>
            </td>
            <td align="center" width="150"><p><h5>Tutorials/Demos</h5></p>
            <p><a href="http://www.cellprofiler.org/tutorials.shtml"><img src="memory:welcomescreen_tutorials.png"></a></p>
            <p>For written and<br>video guidance</p></a>
            </td>
            <td align="center" width="150"><p><h5>Q&A Forum</h5></p>
            <p><a href="http://www.cellprofiler.org/forum/"><img src="memory:welcomescreen_forum.png"></a></p>
            <p>Post a question<br>online</p></a>
            </td>
        </tr>
        </table>
    </td>
</tr>
</table>
<p>Click <a href="pref:no_display">here</a> to stop displaying this page when CellProfiler starts. This page can be accessed from <i>Help > Show Welcome Screen</i>.</p>
</body>
</html>'''

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
<tr><td><i>Pipeline</i></td><td>Lists the modules in the pipeline, with controls for display and testing. Below this panel are tools for adding, removing, and reordering modules and getting help.</td></tr>
<tr><td><i>Files</i></td><td>Lists images and pipeline files in the current input folder.</td></tr>
<tr><td><i>Module Settings</i></td><td>Contains the options for the currently selected module.</td></tr>
<tr><td><i>Folders</i></td><td>Dialogs for controlling default input and output folders and output filename.</td></tr>
</tbody></table>
<p>Go <a href="startup_main">back</a> to the main startup page.</p>
</body>
</html>'''

def find_link(name):
    return globals().get(name, None)
