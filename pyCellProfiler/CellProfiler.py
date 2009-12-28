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


# Mark's machine
if sys.platform.startswith('win'):
    try:
        import cellprofiler.cpmath.propagate
    except:
        print "Propagate module doesn't exist yet."
        print "CellProfiler will compile it, but may crash soon after."
        print "Restart CellProfiler and it will probably work."


root = os.path.split(__file__)[0]
if len(root) == 0:
    root = os.curdir
root = os.path.abspath(root)
site_packages = os.path.join(root, 'site-packages')
if os.path.exists(site_packages) and os.path.isdir(site_packages):
    import site
    site.addsitedir(site_packages)

import optparse
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
parser.add_option("-f", "--first-image-set",
                  dest="first_image_set",
                  default=None,
                  help="The one-based index of the first image set to process")
parser.add_option("-l", "--last-image-set",
                  dest="last_image_set",
                  default=None,
                  help="The one-based index of the last image set to process")
parser.add_option("-g", "--group",
                  dest="groups",
                  default=None,
                  help=('Restrict processing to one grouping in a grouped '
                        'pipeline. For instance, "-g ROW=H,COL=01", will '
                        'process only the group of image sets that match '
                        'the keys.'))
parser.add_option("-b", "--do-not_build",
                  dest="build_extensions",
                  default=True,
                  action="store_false",
                  help="Do not build C and Cython extensions")
parser.add_option("-d", "--done-file",
                  dest="done_file",
                  default=None,
                  help=('The path to the "Done" file, written by CellProfiler'
                        ' shortly before exiting'))
options, args = parser.parse_args()

# necessary to prevent matplotlib trying to use Tkinter as its backend.
# has to be done before CellProfilerApp is imported
from matplotlib import use as mpluse
mpluse('WXAgg')

if (not hasattr(sys, 'frozen')) and options.build_extensions:
    import subprocess
    import cellprofiler.cpmath.setup
    if sys.platform == 'win32':
        import cellprofiler.ffmpeg.setup
    import cellprofiler.utilities.setup
    import contrib.setup
    from distutils.dep_util import newer_group
    #
    # Check for dependencies and compile if necessary
    #
    compile_scripts = [(os.path.join('cellprofiler', 'cpmath', 'setup.py'),
                        cellprofiler.cpmath.setup),
                       (os.path.join('cellprofiler', 'utilities', 'setup.py'),
                        cellprofiler.utilities.setup),
                       (os.path.join('contrib', 'setup.py'),
                        contrib.setup)]
    if sys.platform == 'win32':
        compile_scripts += [(os.path.join('cellprofiler','ffmpeg','setup.py'),
                             cellprofiler.ffmpeg.setup)]
    current_directory = os.path.abspath(os.curdir)
    for compile_script, my_module in compile_scripts:
        script_path, script_file = os.path.split(compile_script)
        os.chdir(os.path.join(root, script_path))
        configuration = my_module.configuration()
        needs_build = False
        for extension in configuration['ext_modules']:
            target = extension.name + '.pyd'
            if newer_group(extension.sources, target):
                needs_build = True
        if not needs_build:
            continue
        if sys.platform == 'win32':
            p = subprocess.Popen(["python",
                                  script_file,
                                  "build_ext", "-i",
                                  "--compiler=mingw32"])
        else:
            p = subprocess.Popen(["python",
                                  script_file,
                                  "build_ext", "-i"])
        p.communicate()
    os.chdir(current_directory)
    # build icon files
    if options.show_gui:
        import glob
        zippo_script = os.path.join(root, 'zippo.py')
        zippo_outfile = os.path.join(root, "cellprofiler", "icons", "__init__.py")
        icon_files = glob.glob(os.path.join(root, "cellprofiler", "icons", "*.png"))
        print "packing icons", " ".join([os.path.basename(f) for f in icon_files])
        p = subprocess.Popen(["python", zippo_script, zippo_outfile] + icon_files)
        p.communicate()

if options.show_gui:
    from cellprofiler.cellprofilerapp import CellProfilerApp
    App = CellProfilerApp(0)

#
# Important to go headless ASAP
#
import cellprofiler.preferences as cpprefs
if not options.show_gui:
    cpprefs.set_headless()
    # What's there to do but run if you're running headless?
    # Might want to change later if there's some headless setup 
    options.run_pipeline = True

from cellprofiler.utilities.get_revision import version
print "Subversion revision: %d"%version
if options.run_pipeline and not options.pipeline_filename:
    raise ValueError("You must specify a pipeline filename to run")

if not options.first_image_set is None:
    if not options.first_image_set.isdigit():
        raise ValueError("The --first-image-set option takes a numeric argument")
    else:
        image_set_start = int(options.first_image_set)
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
    import cellprofiler.gui.cpframe as cpgframe
    if options.pipeline_filename:
        App.frame.pipeline.load(options.pipeline_filename)
    if options.run_pipeline:
        App.frame.Command(cpgframe.ID_FILE_ANALYZE_IMAGES)
    App.MainLoop()
else:
    from cellprofiler.pipeline import Pipeline, EXIT_STATUS
    import cellprofiler.measurements as cpmeas
    pipeline = Pipeline()
    pipeline.load(options.pipeline_filename)
    if options.groups is not None:
        kvs = [x.split('=') for x in options.groups.split(',')]
        groups = dict(kvs)
    else:
        groups = None
    measurements = pipeline.run(image_set_start=image_set_start, 
                                image_set_end=image_set_end,
                                grouping=groups)
    if len(args) > 0:
        pipeline.save_measurements(args[0], measurements)
    if options.done_file is not None:
        if (measurements is not None and 
            measurements.has_feature(cpmeas.EXPERIMENT, EXIT_STATUS)):
            done_text = measurements.get_experiment_measurement(EXIT_STATUS)
        else:
            done_text = "Failure"
        fd = open(options.done_file, "wt")
        fd.write("%s\n"%done_text)
        fd.close()
try:
    import cellprofiler.utilities.jutil as jutil
    jutil.kill_vm()
except:
    import traceback
    traceback.print_exc()
    print "Caught exception while killing VM"
