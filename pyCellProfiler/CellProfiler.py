"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import sys
import os
root = os.path.split(__file__)[0]
if len(root) == 0:
    root = os.curdir
root = os.path.abspath(root)
site_packages = os.path.join(root, 'site-packages')
if os.path.exists(site_packages) and os.path.isdir(site_packages):
    import site
    site.addsitedir(site_packages)
import optparse
import wx
import subprocess
# necessary to prevent matplotlib trying to use Tkinter as its backend
from matplotlib import use as mpluse
mpluse('WXAgg')

if not hasattr(sys, 'frozen'):
    import cellprofiler.cpmath.setup
    import cellprofiler.ffmpeg.setup
    from distutils.dep_util import newer_group
    #
    # Check for dependencies and compile if necessary
    #
    compile_scripts = [(os.path.join('cellprofiler','cpmath','setup.py'),
                        cellprofiler.cpmath.setup)]
    if sys.platform == 'win32':
        compile_scripts += [(os.path.join('cellprofiler','ffmpeg','setup.py'),
                             cellprofiler.ffmpeg.setup)]
    current_directory = os.curdir
    for compile_script,my_module in compile_scripts:
        script_path, script_file = os.path.split(compile_script)
        os.chdir(os.path.join(root,script_path))
        configuration = my_module.configuration()
        needs_build = False
        for extension in configuration['ext_modules']:
            target = extension.name+'.pyd'
            if newer_group(extension.sources,target):
                needs_build = True
        if not needs_build:
            continue 
        if sys.platform == 'win32':
            p = subprocess.Popen(["python",
                                  script_file,
                                  "build_ext","-i",
                                  "--compiler=mingw32"])
        else:            
            p = subprocess.Popen(["python",
                                  script_file,
                                  "build_ext","-i"])
        p.communicate()
    os.chdir(current_directory)

from cellprofiler.cellprofilerapp import CellProfilerApp
from cellprofiler.pipeline import Pipeline
import cellprofiler.preferences as cpprefs
import cellprofiler.gui.cpframe as cpgframe

usage = """usage: %prog [options] [<measurement-file>])
     where <measurement-file> is the optional filename for measurement output
           when running headless"""

parser = optparse.OptionParser(usage=usage)
parser.add_option("-p", "--pipeline", 
                  dest="pipeline_filename",
                  help="Load this pipeline file on startup",
                  default=None)
parser.add_option("-c", "--run-headless",
                  action="store_false",
                  dest="show_gui",
                  default=True,
                  help="Run headless (without the GUI)")
parser.add_option("-r", "--run",
                  action="store_true",
                  dest="run_pipeline",
                  default=False,
                  help="Run the given pipeline on startup")
parser.add_option("-o", "--output-directory",
                  dest="output_directory",
                  default=None,
                  help="Make this directory the default output directory")
parser.add_option("-i", "--image-directory",
                  dest="image_directory",
                  default=None,
                  help="Make this directory the default image directory")
parser.add_option("-f","--first-image-set",
                  dest="first_image_set",
                  default=None,
                  help="The one-based index of the first image set to process")
parser.add_option("-l","--last-image-set",
                  dest="last_image_set",
                  default=None,
                  help="The one-based index of the last image set to process")

options, args = parser.parse_args()

if not options.show_gui:
    cpprefs.set_headless()
    # What's there to do but run if you're running headless?
    # Might want to change later if there's some headless setup 
    options.run_pipeline = True
if options.run_pipeline and not options.pipeline_filename:
    raise ValueError("You must specify a pipeline filename to run")

if not options.first_image_set is None:
    if not options.first_image_set.isdigit():
        raise ValueError("The --first-image-set option takes a numeric argument")
    else:
        image_set_start = int(options.first_image_set) - 1
else:
    image_set_start = None

if not options.last_image_set is None:
    if not options.last_image_set.isdigit():
        raise ValueError("The --last-image-set option takes a numeric argument")
    else:
        image_set_end = int(options.last_image_set)
else:
    image_set_end = None

if options.output_directory:
    cpprefs.set_default_output_directory(options.output_directory)

if options.image_directory:
    cpprefs.set_default_image_directory(options.image_directory)

if options.show_gui:
    App = CellProfilerApp(0)
    if options.pipeline_filename:
        App.frame.pipeline.load(options.pipeline_filename)
    if options.run_pipeline:
        App.frame.Command(cpgframe.ID_FILE_ANALYZE_IMAGES)
    App.MainLoop()
else:
    pipeline = Pipeline()
    pipeline.load(options.pipeline_filename)
    measurements = pipeline.run(image_set_start = image_set_start, 
                                image_set_end = image_set_end)
    if len(args) > 0:
        pipeline.save_measurements(args[0], measurements)
    
