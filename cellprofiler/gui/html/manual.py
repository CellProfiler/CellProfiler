"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import sys
import os
import cellprofiler.icons
from glob import glob
from shutil import copy
from cellprofiler.modules import get_module_names, instantiate_module
from cellprofiler.gui.help import MAIN_HELP
from cellprofiler.utilities.relpath import relpath
import cellprofiler.utilities.version as version
    
LOCATION_COVERPAGE = '/'.join(['images','CPCoverPage.png'])
LOCATION_WHITEHEADLOGO = '/'.join(['images','WhiteheadInstituteLogo.png'])
LOCATION_CSAILLOGO = '/'.join(['images','CSAIL_Logo.png'])
LOCATION_IMAGINGPLATFORMBANNER  = '/'.join(['images','BroadPlusImagingPlusBanner.png'])    
VERSION = version.version_string
VERSION_NUMBER = version.version_number

def generate_html(webpage_path = None):
    if webpage_path is None:
        webpage_path = os.path.join('.', 'CellProfiler_Manual_' + str(VERSION_NUMBER))
        
    if not (os.path.exists(webpage_path) and os.path.isdir(webpage_path)):
        os.mkdir(webpage_path)
                
    # Copy the png images to a new 'images' directory under the html folder
    webpage_images_path = os.path.join(webpage_path, 'images')
    if not (os.path.exists(webpage_images_path) and os.path.isdir(webpage_images_path)):
            os.mkdir(webpage_images_path)

    # Write the individual topic files
    module_help_text  = output_module_html(webpage_path)
    nonmodule_help_text = output_gui_html(webpage_path)
    
    index_fd = open(os.path.join(webpage_path,'index.html'), 'w')

    icons_path = cellprofiler.icons.__path__[0]
    all_pngs = glob(os.path.join(icons_path, "*.png"))
    for f in all_pngs:
        copy(f,webpage_images_path)
    
    intro_text = """
<html style="font-family:arial">
<head>
<title>CellProfiler: Table of contents</title>
</head>
<body>
<div style="page-break-after:always"> 
<table width="100%%">
<tr><td align="center">
<img src="%(LOCATION_COVERPAGE)s" align="middle" style="border-style: none"></img>
</tr></td>
</table>
</div>
<div style="page-break-after:always"> 
<table width="100%%" cellpadding="10">
<tr><td align="middle"><b>CellProfiler</b> cell image analysis software</td></tr>
<tr><td align="middle"><b>Created by</b><br>Anne E. Carpenter and Thouis R. Jones</td></tr>
<tr><td align="middle"><b>In the laboratories of</b><br>David M. Sabatini and Polina Golland at</td></tr>
<tr><td align="middle"><img src="%(LOCATION_WHITEHEADLOGO)s" style="border-style: none"></img>
<img src="%(LOCATION_CSAILLOGO)s" style="border-style: none"></img></td></tr>
<tr><td align="middle">And now based at</td></tr>
<tr><td align="middle"><img src="%(LOCATION_IMAGINGPLATFORMBANNER)s" style="border-style: none"></img></td></tr>
<tr><td align="middle">
<b>CellProfiler is free and open-source!</b>

<p>If you find it useful, please credit CellProfiler in publications
<ol>
<li>Cite the website (www.cellprofiler.org)</li>
<li>Cite the publication (check the website for the citation).</li>
<li>Post the reference for your publication on the CellProfiler Forum (accessible 
from the website) so that we are aware of it.</li>
</ol></p>

<p>These steps will help us to maintain funding for the project and continue to 
improve and support it.</p>
</td></tr>
</table>
</div>

<b>This manual accompanies version %(VERSION)s of CellProfiler. The most 
recent manual is available <a href="http://www.cellprofiler.org/CPmanual/">here</a>.</b>

<h1><a name="table_of_contents">Table of contents</a></h1>"""%globals()
            
    index_fd.write(intro_text)
    index_fd.write(nonmodule_help_text)
    index_fd.write(module_help_text)
    index_fd.write("""</body></html>\n""")
    
    index_fd.close()

    print "Wrote CellProfiler Manual to", webpage_path

def output_gui_html(webpage_path):
    '''Output an HTML page for each non-module help item'''
    icons_relpath = relpath(cellprofiler.icons.__path__[0])
    
    help_text = """
<h2>Using CellProfiler</a></h2>"""
    
    def write_menu(prefix, h, help_text):
        help_text += "<ul>\n"
        for key, value in h:
            help_text += "<li>"
            if hasattr(value, "__iter__") and not isinstance(value, (str, unicode)):
                help_text += "<b>%s</b>"%key
                help_text = write_menu(prefix+"_"+key, value, help_text)
            else:
                file_name = "%s_%s.html" % (prefix, key)
                fd = open(os.path.join(webpage_path, file_name),"w")
                fd.write("<html style=""font-family:arial""><head><title>%s</title></head>\n" % key)
                fd.write("<body><h1>%s</h1>\n<div>\n" % key)
                # Replace the relative paths to the icons with the relative path to the image dir
                value = value.replace(icons_relpath,'images')
                fd.write(value)
                fd.write("</div></body>\n")
                fd.close()
                help_text += "<a href='%s'>%s</a>\n" % (file_name, key)
            help_text += "</li>\n"
        help_text += "</ul>\n"
        return help_text
        
    help_text = write_menu("Help", MAIN_HELP, help_text)
    help_text += "\n"
    
    return help_text
    
def output_module_html(webpage_path):
    '''Output an HTML page for each module'''
        
    icons_relpath = relpath(cellprofiler.icons.__path__[0])
    all_png_icons = glob(os.path.join(icons_relpath, "*.png"))
    icon_names = [os.path.basename(f)[:-4] for f in all_png_icons]
    
    help_text = """
<h2>Help for CellProfiler Modules</a></h2>
<ul>\n"""
    d = {}
    module_path = webpage_path
    if not (os.path.exists(module_path) and os.path.isdir(module_path)):
        try:
            os.mkdir(module_path)
        except IOError:
            raise ValueError("Could not create directory %s" % module_path)
        
    for module_name in sorted(get_module_names()):
        module = instantiate_module(module_name)
        if isinstance(module.category, (str,unicode)):
            module.category = [module.category]
        for category in module.category:
            if not d.has_key(category):
                d[category] = {}
            d[category][module_name] = module
        result = module.get_help()
        if result is None:
            continue
        result = result.replace('<body><h1>','<body><h1>Module: ')
        
        # Check if a corresponding image exists for the module
        if module_name in icon_names:
            # Strip out end html tags so I can add more stuff
            result = result.replace('</body>','').replace('</html>','')
            # Include images specific to the module, relative to html files ('images' dir)
            LOCATION_MODULE_IMAGES = '/'.join(['images','%s.png'%(module_name)])
            result += '\n\n<div><p><img src="%s", width="50%%"></img></p></div>\n'%LOCATION_MODULE_IMAGES
            # Now end the help text
            result += '</body></html>'
        fd = open(os.path.join(module_path,"%s.html" % module_name), "w")
        fd.write(result)
        fd.close()
    for category in sorted(d.keys()):
        sub_d = d[category]
        help_text += "<li><b>%s</b><br><ul>\n"%category
        for module_name in sorted(sub_d.keys()):
            help_text += "<li><a href='%s.html'>%s</a></li>\n" % (module_name, module_name)
        help_text += "</ul></li>\n"
    help_text += "</ul>\n"
    return help_text
        
