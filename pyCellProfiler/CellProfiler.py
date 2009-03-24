"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import optparse
import wx

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

options, args = parser.parse_args()

if not options.show_gui:
    # What's there to do but run if you're running headless?
    # Might want to change later if there's some headless setup 
    options.run_pipeline = True
if options.run_pipeline and not options.pipeline_filename:
    raise ValueError("You must specify a pipeline filename to run")

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
    measurements = pipeline.run()
    if len(args) > 0:
        pipeline.save_measurements(args[0], measurements)
    