"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import sys
import os
import numpy as np
#
# CellProfiler expects NaN as a result during calculation
#
np.seterr(all='ignore')

# Mark's machine
if sys.platform.startswith('win'):
    try:
        import cellprofiler.cpmath.propagate
    except:
        print "Propagate module doesn't exist yet."
        print "CellProfiler will compile it, but may crash soon after."
        print "Restart CellProfiler and it will probably work."

if not hasattr(sys, 'frozen'):
    root = os.path.split(__file__)[0]
else:
    root = os.path.split(sys.argv[0])[0]
if len(root) == 0:
    root = os.curdir
root = os.path.abspath(root)
site_packages = os.path.join(root, 'site-packages')
if os.path.exists(site_packages) and os.path.isdir(site_packages):
    import site
    site.addsitedir(site_packages)

import optparse
usage = """usage: %prog [options] [<output-file>])
     where <output-file> is the optional filename for the output file of measurements
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
                  help="Make this directory the default output folder")
parser.add_option("-i", "--image-directory",
                  dest="image_directory",
                  default=None,
                  help="Make this directory the default input folder")
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
parser.add_option("--html",
                  action="store_true",
                  dest="output_html",
                  default = False,
                  help = ('Output HTML help for all modules. Use with the -o '
                          'option to specify the output directory for the '
                          'files. Assumes -b.'))

parser.add_option("--plugins-directory",
                  dest="plugins_directory",
                  help=("CellProfiler will look for plugin modules in this "
                        "directory."))

parser.add_option("--ij-plugins-directory",
                  dest="ij_plugins_directory",
                  help=("CellProfiler will look for ImageJ plugin modules "
                        "in this directory."))

parser.add_option("--jvm-heap-size",
                  dest="jvm_heap_size",
                  help=("This is the amount of memory reserved for the "
                        "Java Virtual Machine (similar to the java -Xmx switch)."
                        "Example formats: 512000k, 512m, 1g"))

if not hasattr(sys, 'frozen'):
    parser.add_option("-b", "--do-not_build",
                      dest="build_extensions",
                      default=True,
                      action="store_false",
                      help="Do not build C and Cython extensions")
    parser.add_option("--build-and-exit",
                      dest="build_and_exit",
                      default=False,
                      action="store_true",
                      help="Build extensions, then exit CellProfiler")
parser.add_option("--ilastik",
                  dest = "run_ilastik",
                  default=False,
                  action="store_true",
                  help = ("Run Ilastik instead of CellProfiler. "
                          "Ilastik is a pixel-based classifier. See "
                          "www.ilastik.org for more details."))
parser.add_option("-d", "--done-file",
                  dest="done_file",
                  default=None,
                  help=('The path to the "Done" file, written by CellProfiler'
                        ' shortly before exiting'))
parser.add_option("--measurements",
                  dest="print_measurements",
                  default=False,
                  action="store_true",
                  help="Open the pipeline file specified by the -p switch "
                  "and print the measurements made by that pipeline")
parser.add_option("--data-file",
                  dest="data_file",
                  default = None,
                  help = "Specify a data file for LoadData modules that "
                  'use the "From command-line" option')
options, args = parser.parse_args()

if options.run_ilastik:
    #
    # Fake ilastik into thinking it is __main__
    #
    import ilastik
    import imp
    sys.argv.remove("--ilastik")
    il_path = ilastik.__path__
    il_file, il_path, il_description = imp.find_module('ilastikMain', il_path)
    imp.load_module('__main__', il_file, il_path, il_description)
    sys.exit()
    
# necessary to prevent matplotlib trying to use Tkinter as its backend.
# has to be done before CellProfilerApp is imported
from matplotlib import use as mpluse
mpluse('WXAgg')

if (not hasattr(sys, 'frozen')) and options.build_extensions:
    import subprocess
    import cellprofiler.cpmath.setup
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
    current_directory = os.path.abspath(os.curdir)
    old_pythonpath = os.getenv('PYTHONPATH', None)

    # if we're using a local site_packages, the subprocesses will need
    # to be able to find it.
    if old_pythonpath:
        os.environ['PYTHONPATH'] = site_packages + ':' + old_pythonpath
    else:
        os.environ['PYTHONPATH'] = site_packages

    use_mingw = (sys.platform == 'win32' and sys.version_info[0] <= 2 and
                 sys.version_info[1] <= 5)
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
        if use_mingw:
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
    if old_pythonpath:
        os.environ['PYTHONPATH'] = old_pythonpath
    else:
        del os.environ['PYTHONPATH']
    if options.build_and_exit:
        exit()

if options.show_gui and not options.output_html:
    from cellprofiler.cellprofilerapp import CellProfilerApp
    App = CellProfilerApp(0, 
                          check_for_new_version = True,
                          show_splashbox = True)

try:
    #
    # Important to go headless ASAP
    #
    import cellprofiler.preferences as cpprefs
    if (not options.show_gui) or options.output_html:
        cpprefs.set_headless()
        # What's there to do but run if you're running headless?
        # Might want to change later if there's some headless setup 
        if (not options.output_html) and (not options.print_measurements):
            options.run_pipeline = True
    
    if options.output_html:
        from cellprofiler.gui.html.manual import generate_html
        webpage_path = options.output_directory if options.output_directory else None
        generate_html(webpage_path)
        
    if options.print_measurements:
        if options.pipeline_filename is None:
            raise ValueError("Can't print measurements, no pipeline file")
        import cellprofiler.pipeline as cpp
        pipeline = cpp.Pipeline()
        def callback(pipeline, event):
            if isinstance(event, cpp.LoadExceptionEvent):
                raise ValueError("Failed to load %s" % options.pipeline_filename)
        pipeline.add_listener(callback)
        pipeline.load(os.path.expanduser(options.pipeline_filename))
        columns = pipeline.get_measurement_columns()
        print "--- begin measurements ---"
        print "Object,Feature,Type"
        for column in columns:
            object_name, feature, data_type = column[:3]
            print "%s,%s,%s" % (object_name, feature, data_type)
        print "--- end measurements ---"
    
    if options.plugins_directory is not None:
        cpprefs.set_plugin_directory(options.plugins_directory)
    if options.ij_plugins_directory is not None:
        cpprefs.set_ij_plugin_directory(options.ij_plugins_directory)
    if options.data_file is not None:
        cpprefs.set_data_file(os.path.abspath(options.data_file))
        
    from cellprofiler.utilities.get_revision import version
    print "Subversion revision: %d"%version
    if options.output_html:
        sys.exit(0) 
    
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
            try:
                App.frame.pipeline.load(os.path.expanduser(options.pipeline_filename))
                if options.run_pipeline:
                    App.frame.Command(cpgframe.ID_FILE_ANALYZE_IMAGES)
            except:
                import wx
                wx.MessageBox(
                    'CellProfiler was unable to load the pipeline file, "%s"' %
                    options.pipeline_filename, "Error loading pipeline",
                    style = wx.OK | wx.ICON_ERROR)
        App.MainLoop()
    elif options.run_pipeline:
        from cellprofiler.pipeline import Pipeline, EXIT_STATUS
        import cellprofiler.measurements as cpmeas
        pipeline = Pipeline()
        pipeline.load(os.path.expanduser(options.pipeline_filename))
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
finally:
    try:
        import cellprofiler.utilities.jutil as jutil
        jutil.kill_vm()
    except:
        import traceback
        traceback.print_exc()
        print "Caught exception while killing VM"
