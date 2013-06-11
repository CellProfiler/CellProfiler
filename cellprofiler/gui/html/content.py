"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

startup_main = '''<html>
<body>
<center><h2>Welcome to CellProfiler!</h2></center>

<p>CellProfiler is automated image analysis software designed to measure biological phenotypes in images.</p>
<br>
<br>
From here, you can...
<table border="0" cellpadding="5" width="100%">
<tr>
    <td width="200"><i>Get oriented</i></td>
    <td>See this <a href="startup_interface">summary</a> for a quick overview of CellProfiler's interface.</td>
</tr>
<tr>
    <td width="200"><i>Read the documentation</i></td>
    <td>There is an <a href="http://www.cellprofiler.org/CPmanual#table_of_contents">online manual</a>. Also, detailed help is available for any module by clicking the "<b>?</b>" button, or using the <i>Help</i> menu in the toolbar.</td>
 </tr>
<tr>
    <td><i>Try an example pipeline</i></td>
    <td>You can <a href="loadexample:http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cp">load a simple pipeline</a> from our website.</td>
</tr>
<tr>
    <td><i>Find other examples, read tutorials</i></td>
    <td>We have <a href="http://www.cellprofiler.org/examples.shtml">examples</a> of pipelines and images for various biological assays. You can pick one that most resembles your phenotypes of interest and begin adjusting its settings.  There are also text and video <a href="http://www.cellprofiler.org/tutorials.shtml">tutorials</a>.</td>
</tr>
<tr>
    <td><i>Get user support</i></td>
    <td>If you need help or advice, you can post a question in our online <a href="http://www.cellprofiler.org/forum/">forum.</a></td>
</tr>
</table>
<p>Click <a href="pref:no_display">here</a> to stop displaying this page when CellProfiler starts.</p>
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
