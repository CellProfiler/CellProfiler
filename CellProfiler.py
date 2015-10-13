"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import h5py
import logging
import logging.config
import re
import sys
import os
import numpy as np
import tempfile
from cStringIO import StringIO

OMERO_CK_HOST = "host"
OMERO_CK_PORT = "port"
OMERO_CK_USER = "user"
OMERO_CK_PASSWORD = "password"
OMERO_CK_SESSION_ID = "session-id"
OMERO_CK_CONFIG_FILE = "config-file"

if sys.platform.startswith('win'):
    # This recipe is largely from zmq which seems to need this magic
    # in order to import in frozen mode - a topic the developers never
    # dealt with.
    if hasattr(sys, 'frozen'):
        here = os.path.split(sys.argv[0])[0]

        import ctypes
        libzmq = os.path.join(here, 'libzmq.dll')
        if os.path.exists(libzmq):
            ctypes.cdll.LoadLibrary(libzmq)
import zmq
#
# CellProfiler expects NaN as a result during calculation
#
np.seterr(all='ignore')
#
# Defeat pyreadline which graciously sets its logging to DEBUG and it
# appears when CP is frozen
#
try:
    from pyreadline.logger import stop_logging, pyreadline_logger
    pyreadline_logger.setLevel(logging.INFO)
    stop_logging()
except:
    pass

if not hasattr(sys, 'frozen'):
    root = os.path.split(__file__)[0]
else:
    root = os.path.split(sys.argv[0])[0]
if len(root) == 0:
    root = os.curdir
root = os.path.abspath(root)
site_packages = os.path.join(root, 'site-packages').encode('utf-8')
if os.path.exists(site_packages) and os.path.isdir(site_packages):
    import site
    site.addsitedir(site_packages)
    
def main(args):
    '''Run CellProfiler

    args - command-line arguments, e.g. sys.argv
    '''
    import cellprofiler.preferences as cpprefs
    cpprefs.set_awt_headless(True)
    switches = ('--work-announce', '--knime-bridge-address')
    if any([any([arg.startswith(switch) for switch in switches])
            for arg in args]):
        #
        # Go headless ASAP
        #
        cpprefs.set_headless()
        for i, arg in enumerate(args):
            if arg == "--ij-plugins-directory" and len(args) > i+1:
                cpprefs.set_ij_plugin_directory(args[i+1])
                break
        import cellprofiler.analysis_worker
        cellprofiler.analysis_worker.aw_parse_args()
        cellprofiler.analysis_worker.main()
        sys.exit(0)
        
    if any([arg.startswith('--xml-test-file=') for arg in sys.argv]):
        import cpnose
        cpnose.main(*sys.argv)
        return
    options, args = parse_args(args)
    if options.print_version:
        from cellprofiler.utilities.version import \
             dotted_version, version_string, git_hash, version_number
        print "CellProfiler %s" % dotted_version
        print "Git %s" % git_hash
        print "Version %s" % version_number
        print "Built %s" % version_string.split(" ")[0]
        sys.exit(0)
    #
    # Important to go headless ASAP
    #
    if (not options.show_gui) or options.write_schema_and_exit:
        import cellprofiler.preferences as cpprefs
        cpprefs.set_headless()
        # What's there to do but run if you're running headless?
        # Might want to change later if there's some headless setup 
        options.run_pipeline = True

    if options.jvm_heap_size is not None:
        from cellprofiler.preferences import set_jvm_heap_mb
        set_jvm_heap_mb(options.jvm_heap_size, False)
    set_log_level(options)
    
    if options.print_groups_file is not None:
        print_groups(options.print_groups_file)
        return
    
    if options.batch_commands_file is not None:
        get_batch_commands(options.batch_commands_file)
        return
        
    if options.run_ilastik:
        run_ilastik()
        return
    
    if options.add_message_for_user:
        if len(args) != 3:
            sys.stderr.write("Usage: (for add_message-for-user)\n")
            sys.stderr.write("CellProfiler --add-message-for-user <caption> <message> <pipeline-or-project>\n")
            sys.stderr.write("where:\n")
            sys.stderr.write("    <caption> - the message box caption\n")
            sys.stderr.write("    <message> - the message displayed inside the message box\n")
            sys.stderr.write("    <pipeline-or-project> - the path to the pipeline or project file to modify\n")
            return
        caption = args[0]
        message = args[1]
        path = args[2]
        
        import h5py
        using_hdf5 = h5py.is_hdf5(path)
        if using_hdf5:
            import cellprofiler.measurements as cpmeas
            m = cpmeas.Measurements(
                filename = path, mode="r+")
            pipeline_text = m[cpmeas.EXPERIMENT, "Pipeline_Pipeline"]
        else:
            with open(path, "r") as fd:
                pipeline_text = fd.read()
        header, body = pipeline_text.split("\n\n", 1)
        pipeline_text = header + \
            ("\nMessageForUser:%s|%s\n\n" % (caption, message)) + body
        if using_hdf5:
            m[cpmeas.EXPERIMENT, "Pipeline_Pipeline"] = pipeline_text
            m.close()
        else:
            with open(path, "w") as fd:
                fd.write(pipeline_text)
        print "Message added to %s" % path
        return
    
    # necessary to prevent matplotlib trying to use Tkinter as its backend.
    # has to be done before CellProfilerApp is imported
    from matplotlib import use as mpluse
    mpluse('WXAgg')
    
    if (not hasattr(sys, 'frozen')) and options.build_extensions:
        build_extensions()
        if options.build_and_exit:
            return
    
    if options.omero_credentials is not None:
        set_omero_credentials_from_string(options.omero_credentials)
    if options.plugins_directory is not None:
        cpprefs.set_plugin_directory(options.plugins_directory,
                                     globally=False)
    if options.ij_plugins_directory is not None:
        cpprefs.set_ij_plugin_directory(options.ij_plugins_directory,
                                        globally=False)
    if options.temp_dir is not None:
        if not os.path.exists(options.temp_dir):
            os.makedirs(options.temp_dir)
        cpprefs.set_temporary_directory(options.temp_dir, globally=False)
    if not options.allow_schema_write:
        cpprefs.set_allow_schema_write(False)
    #
    # After the crucial preferences are established, we can start the VM
    #
    from cellprofiler.utilities.cpjvm import cp_start_vm, cp_stop_vm
    cp_start_vm()
    #
    # Not so crucial preferences...
    #
    if options.image_set_file is not None:
        cpprefs.set_image_set_file(options.image_set_file)
    try:
        #---------------------------------------
        #
        # Handle command-line tasks that that need to load the modules to run
        # 
        if options.output_html:
            from cellprofiler.gui.html.manual import generate_html
            webpage_path = options.output_directory if options.output_directory else None
            generate_html(webpage_path)
            return
        if options.print_measurements:
            print_measurements(options)
            return
        if not hasattr(sys, "frozen") and options.code_statistics:
            print_code_statistics()
            return
        if options.write_schema_and_exit:
            write_schema(options.pipeline_filename)
            return
        #
        #------------------------------------------
        if options.show_gui:
            import wx
            wx.Log.EnableLogging(False)
            from cellprofiler.cellprofilerapp import CellProfilerApp
            from cellprofiler.workspace import is_workspace_file
            show_splashbox = (options.pipeline_filename is None and
                              (not options.new_project) and
                              options.show_splashbox)
            
            if options.pipeline_filename:
                if is_workspace_file(options.pipeline_filename):
                    workspace_path = os.path.expanduser(options.pipeline_filename)
                    pipeline_path = None
                else:
                    pipeline_path = os.path.expanduser(options.pipeline_filename)
                    workspace_path = None
            elif options.new_project:
                workspace_path = False
                pipeline_path = None
            else:
                workspace_path = None
                pipeline_path = None
            App = CellProfilerApp(
                0, 
                check_for_new_version = (options.pipeline_filename is None),
                show_splashbox = show_splashbox,
                workspace_path = workspace_path,
                pipeline_path = pipeline_path)

        if options.data_file is not None:
            cpprefs.set_data_file(os.path.abspath(options.data_file))
            
        from cellprofiler.utilities.version import version_string, version_number
        logging.root.info("Version: %s / %d" % (version_string, version_number))
    
        if options.run_pipeline and not options.pipeline_filename:
            raise ValueError("You must specify a pipeline filename to run")
    
        if options.output_directory:
            if not os.path.exists(options.output_directory):
                os.makedirs(options.output_directory)
            cpprefs.set_default_output_directory(options.output_directory)
        
        if options.image_directory:
            cpprefs.set_default_image_directory(options.image_directory)
    
        if options.show_gui:
            if options.run_pipeline:
                App.frame.pipeline_controller.do_analyze_images()
            App.MainLoop()
            return
        
        elif options.run_pipeline:
            run_pipeline_headless(options, args)
    except Exception, e:
        logging.root.fatal("Uncaught exception in CellProfiler.py", exc_info=True)
        raise
    
    finally:
        if __name__ == "__main__":
            try:
                from ilastik.core.jobMachine import GLOBAL_WM
                GLOBAL_WM.stopWorkers()
            except:
                logging.root.warn("Failed to stop Ilastik")
            try:
                from cellprofiler.utilities.zmqrequest import join_to_the_boundary
                join_to_the_boundary()
            except:
                logging.root.warn("Failed to stop zmq boundary", exc_info=1)
            try:
                cp_stop_vm()
            except:
                logging.root.warn("Failed to stop the JVM", exc_info=1)
            os._exit(0)

def parse_args(args):
    '''Parse the CellProfiler command-line arguments'''
    import optparse
    usage = """usage: %prog [options] [<output-file>])
         where <output-file> is the optional filename for the output file of 
               measurements when running headless. 
         The flags -p, -r and -c are required for a headless run."""
    
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-p", "--pipeline", "--project",
                      dest="pipeline_filename",
                      help=('Load this pipeline file or project on startup. '
                            'If specifying a pipeline file rather than a '
                            'project, the -i flag is also needed unless the '
                            'pipeline is saved with the file list.'),
                      default=None)
    parser.add_option("-n", "--new-project",
                      dest="new_project",
                      help="Open a new project, prompting for its name using a file dialog",
                      action="store_true",
                      default=False)
    
    default_show_gui = True
    if sys.platform.startswith('linux'):
        if not os.getenv('DISPLAY'):
            default_show_gui = False

    parser.add_option("-c", "--run-headless",
                      action="store_false",
                      dest="show_gui",
                      default=default_show_gui,
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
                            "directory (headless-only)."))
    
    parser.add_option("--ij-plugins-directory",
                      dest="ij_plugins_directory",
                      help=("CellProfiler will look for ImageJ plugin modules "
                            "in this directory (headless-only)."))
    
    parser.add_option("-t", "--temporary-directory",
                      dest="temp_dir",
                      default = None,
                      help=("The temporary directory if in headless mode. "
                            "CellProfiler uses this for downloaded image files "
                            "and for the measurements file, if not specified. "
                            "The default is " + tempfile.gettempdir()))
    
    parser.add_option("--jvm-heap-size",
                      dest="jvm_heap_size",
                      default=None,
                      help=("This is the amount of memory reserved for the "
                            "Java Virtual Machine (similar to the java -Xmx switch)."
                            "Example formats: 512000k, 512m, 1g"))
    
    parser.add_option("--add-message-for-user",
                      dest="add_message_for_user",
                      default=False,
                      action="store_true",
                      help = ("This option lets you add a message to a pipeline "
                              "or project file which will appear in a message box"
                              "when that pipeline or project file is opened. "))
    parser.add_option("--version",
                      dest = "print_version",
                      default = False,
                      action = "store_true",
                      help = "Print the version and exit")
    
    if not hasattr(sys, 'frozen'):
        parser.add_option("-b", "--do-not-build", "--do-not_build",
                          dest="build_extensions",
                          default=True,
                          action="store_false",
                          help="Do not build C and Cython extensions")
        parser.add_option("--build-and-exit",
                          dest="build_and_exit",
                          default=False,
                          action="store_true",
                          help="Build extensions, then exit CellProfiler")
    
    parser.add_option("--no-splash-screen",
                      dest="show_splashbox",
                      action="store_false",
                      default=True,
                      help="Do not show the splash screen when starting CellProfiler")
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
    
    parser.add_option("--print-groups",
                      dest="print_groups_file",
                      default=None,
                      help = "Open the measurements file following the "
                      "--print-groups switch and print the groups in its image "
                      "sets. The measurements file should be generated using "
                      "CreateBatchFiles. The output is a JSON-encoded data "
                      "structure containing the group keys and values and the "
                      "image sets in each group.")
    parser.add_option("--get-batch-commands",
                      dest = "batch_commands_file",
                      default = None,
                      help = "Open the measurements file following the "
                      "--get-batch-commands switch and print one line to the "
                      "console per group. The measurements file should be "
                      "generated using CreateBatchFiles and the image sets "
                      "should be grouped into the units to be run. Each line "
                      "is a command to invoke CellProfiler. You can use this "
                      "option to generate a shell script that will invoke "
                      'CellProfiler on a cluster by substituting "CellProfiler" '
                      "with your invocation command in the script's text, for "
                      "instance: CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_jobs.sh")
    
    parser.add_option("--data-file",
                      dest="data_file",
                      default = None,
                      help = "Specify the location of a .csv file for LoadData. "
                      "If this switch is present, this file is used instead of "
                      "the one specified in the LoadData module.")
    parser.add_option("--file-list",
                      dest = "image_set_file",
                      default = None,
                      help = "Specify a file list of one file or URL per line "
                      "to be used to initially populate the Images module's "
                      "file list.")
    parser.add_option("--do-not-write-schema",
                      dest = 'allow_schema_write',
                      default = True,
                      action = "store_false",
                      help = "Do not execute the schema definition and other "
                      "per-experiment SQL commands during initialization "
                      "when running a pipeline in batch mode.")
    parser.add_option("--write-schema-and-exit",
                      dest = 'write_schema_and_exit',
                      default = False,
                      action = 'store_true',
                      help = "Create the experiment database schema and exit")
    parser.add_option("--omero-credentials",
                      dest="omero_credentials",
                      default= None,
                      help = (
                          "Enter login credentials for OMERO. The credentials"
                          " are entered as comma-separated key/value pairs with"
                          " keys, \"%(OMERO_CK_HOST)s\" - the DNS host name for the OMERO server"
                          ", \"%(OMERO_CK_PORT)s\" - the server's port # (typically 4064)"
                          ", \"%(OMERO_CK_USER)s\" - the name of the connecting user"
                          ", \"%(OMERO_CK_PASSWORD)s\" - the connecting user's password"
                          ", \"%(OMERO_CK_SESSION_ID)s\" - the session ID for an OMERO client session."
                          ", \"%(OMERO_CK_CONFIG_FILE)s\" - the path to the OMERO credentials config file."
                          " A typical set of credentials might be:"
                          " --omero-credentials host=demo.openmicroscopy.org,port=4064,session-id=atrvomvjcjfe7t01e8eu59amixmqqkfp"
                          ) % globals())
                      
    parser.add_option("-L", "--log-level",
                      dest = "log_level",
                      default = str(logging.INFO),
                      help = ("Set the verbosity for logging messages: " +
                              ("%d or %s for debugging, " % (logging.DEBUG, "DEBUG")) +
                              ("%d or %s for informational, " % (logging.INFO, "INFO")) +
                              ("%d or %s for warning, " % (logging.WARNING, "WARNING")) +
                              ("%d or %s for error, " % (logging.ERROR, "ERROR")) +
                              ("%d or %s for critical, " % (logging.CRITICAL, "CRITICAL")) +
                              ("%d or %s for fatal." % (logging.FATAL, "FATAL")) +
                              " Otherwise, the argument is interpreted as the file name of a log configuration file (see http://docs.python.org/library/logging.config.html for file format)"))
    
    parser.add_option(
        "--xml-test-file",
        dest = "tests",
        default = None,
        metavar = "XML-FILE",
        help = ("Run unit tests. Tests can be collected by nose using a "
                "command line such as \"python cpnose.py --collect-only "
                "--with-xunit --xunit-file=XML-FILE"))
    
    parser.add_option("--code-statistics", 
                          dest = "code_statistics",
                          action = "store_true",
                          default = False,
                          help = "Print the number of modules, settings and lines of code")
    
    if sys.platform == 'darwin':
        parser.add_option("--use-awt",
                          dest = "start_awt",
                          action = "store_true",
                          default = False,
                          help = "Initialize the Java AWT UI so it can be used when running headless on OS/X")
    options, result_args = parser.parse_args(args[1:])
    if sys.platform == 'darwin' and len(args) == 2:
        if args[1].lower().endswith(".cpproj"):
            # Assume fakey open of .cpproj and OS can't be configured to
            # add the switch as it can in Windows.
            options.project_filename = args[1]
            result_args = []
        elif args[1].lower().endswith(".cpproj"):
            options.pipeline_filename = args[1]
            result_args = []
    return options, result_args

def set_log_level(options):
    '''Set the logging package's log level based on command-line options'''
    try:
        if options.log_level.isdigit():
            logging.root.setLevel(int(options.log_level))
        else:
            logging.root.setLevel(options.log_level)
        if len(logging.root.handlers) == 0:
            logging.root.addHandler(logging.StreamHandler())
    except ValueError:
        logging.config.fileConfig(options.log_level)
        
def set_omero_credentials_from_string(credentials_string):
    '''Set the OMERO server / port / session ID
    
    credentials_string: a comma-separated key/value pair string (key=value)
                        that gives the credentials. Keys are
                        host - the DNS name or IP address of the OMERO server
                        port - the TCP port to use to connect
                        user - the user name
                        session-id - the session ID used for authentication
    '''
    import cellprofiler.preferences as cpprefs
    from bioformats.formatreader import use_omero_credentials
    from bioformats.formatreader import \
         K_OMERO_SERVER, K_OMERO_PORT, K_OMERO_USER, K_OMERO_SESSION_ID,\
         K_OMERO_PASSWORD, K_OMERO_CONFIG_FILE
    
    if re.match("([^=^,]+=[^=^,]+,)*([^=^,]+=[^=^,]+)", credentials_string) is None:
        logging.root.error(
            'The OMERO credentials string, "%s", is badly-formatted.' %
            credentials_string)
        logging.root.error(
            'It should have the form: '
            '"host=hostname.org,port=####,user=<user>,session-id=<session-id>\n')
        raise ValueError("Invalid format for --omero-credentials")
        
    for k, v in [kv.split("=", 1) for kv in credentials_string.split(",")]:
        k = k.lower()
        credentials = {
            K_OMERO_SERVER: cpprefs.get_omero_server(),
            K_OMERO_PORT: cpprefs.get_omero_port(),
            K_OMERO_USER: cpprefs.get_omero_user(),
            K_OMERO_SESSION_ID: cpprefs.get_omero_session_id()
        }
        if k == OMERO_CK_HOST:
            cpprefs.set_omero_server(v, globally=False)
            credentials[K_OMERO_SERVER] = v
        elif k == OMERO_CK_PORT:
            cpprefs.set_omero_port(v, globally=False)
            credentials[K_OMERO_PORT] = v
        elif k == OMERO_CK_SESSION_ID:
            credentials[K_OMERO_SESSION_ID] = v
        elif k == OMERO_CK_USER:
            cpprefs.set_omero_user(v, globally=False)
            credentials[K_OMERO_USER] = v
        elif k == OMERO_CK_PASSWORD:
            credentials[K_OMERO_PASSWORD] = v
        elif k == OMERO_CK_CONFIG_FILE:
            credentials[K_OMERO_CONFIG_FILE] = v
            if not os.path.isfile(v):
                msg = "Cannot find OMERO config file, %s" % v
                logging.root.error(msg)
                raise ValueError(msg)
        else:
            logging.root.error(
                'Unknown --omero-credentials keyword: "%s"' % k)
            logging.root.error(
                'Acceptable keywords are: "%s"' %
            '","'.join([OMERO_CK_HOST, OMERO_CK_PORT, OMERO_CK_SESSION_ID]))
            raise ValueError("Invalid format for --omero-credentials")
    use_omero_credentials(credentials)

def print_code_statistics():
    '''Print # lines of code, # modules, etc to console
    
    This is the official source of code statistics for things like grants.
    '''
    from cellprofiler.modules import builtin_modules, all_modules, instantiate_module
    import subprocess
    print "\n\n\n**** CellProfiler code statistics ****"
    print "# of built-in modules: %d" % len(builtin_modules)
    setting_count = 0
    for module in all_modules.values():
        if module.__module__.find(".") < 0:
            continue
        mn = module.__module__.rsplit(".", 1)[1]
        if mn not in builtin_modules:
            continue
        module_instance = instantiate_module(module.module_name)
        setting_count += len(module_instance.help_settings())
    directory = os.path.abspath(os.path.split(sys.argv[0])[0])
    try:
        filelist = subprocess.Popen(
            ["git", "ls-files"], 
            stdout=subprocess.PIPE,
            cwd = directory).communicate()[0].split("\n")
    except:
        filelist = []
        for root, dirs, files in os.walk(directory):
            filelist += [os.path.join(root, f) for f in files]
    linecount = 0
    for filename in filelist:
        if (os.path.exists(filename) and 
            any([filename.endswith(x) for x in ".py", ".c", ".pyx", ".java"])):
            if filename.endswith(".c") and os.path.exists(filename[:-1]+"pyx"):
                continue
            with open(filename, "r") as fd:
                linecount += len(fd.readlines())
    print "# of settings: %d" % setting_count
    print "# of lines of code: %d" % linecount

def print_measurements(options):
    '''Print the measurements that would be output by a pipeline
    
    This function calls Pipeline.get_measurement_columns() to get the
    measurements that would be output by a pipeline. This can be used in
    a workflow tool or LIMS to find the outputs of a pipeline without
    running it. For instance, someone might want to integrate CellProfiler
    with Knime and write a Knime node that let the user specify a pipeline
    file. The node could then execute CellProfiler with the --measurements
    switch and display the measurements as node outputs.
    '''
    
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
    
def print_groups(filename):
    '''Print the image set groups for this pipeline
    
    This function outputs a JSON string to the console composed of a list
    of the groups in the pipeline image set. Each element of the list is
    a two-tuple whose first element is a key/value dictionary of the
    group's key and the second is a tuple of the image numbers in the group.
    '''
    import json
    
    import cellprofiler.measurements as cpmeas
    
    path = os.path.expanduser(filename)
    m = cpmeas.Measurements(filename = path, mode="r")
    metadata_tags = m.get_grouping_tags()
    groupings = m.get_groupings(metadata_tags)
    json.dump(groupings, sys.stdout)
    
def get_batch_commands(filename):
    '''Print the commands needed to run the given batch data file headless
    
    filename - the name of a Batch_data.h5 file. The file should group image sets.
    
    The output assumes that the executable, "CellProfiler", can be used
    to run the command from the shell. Alternatively, the output could be
    run through a utility such as "sed":
    
    CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_job.sh/
    '''

    import cellprofiler.measurements as cpmeas
    
    path = os.path.expanduser(filename)
    m = cpmeas.Measurements(filename = path, mode="r")

    image_numbers = m.get_image_numbers()
    if m.has_feature(cpmeas.IMAGE, cpmeas.GROUP_NUMBER):
        group_numbers = m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_numbers]
        group_indexes = m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_numbers]
        if np.any(group_numbers != 1) and np.all(
            (group_indexes[1:] == group_indexes[:-1] + 1) |
            ((group_indexes[1:] == 1) & 
             (group_numbers[1:] == group_numbers[:-1]+1))):
            #
            # Do -f and -l if more than one group and group numbers
            # and indices are properly constructed
            #
            bins = np.bincount(group_numbers)
            cumsums = np.cumsum(bins)
            prev = 0
            for i, off in enumerate(cumsums):
                if off == prev:
                    continue
                print "CellProfiler -c -r -b -p %s -f %d -l %d" % (
                    filename, prev+1, off)
                prev = off
            return
    
    metadata_tags = m.get_grouping_tags()
    groupings = m.get_groupings(metadata_tags)
    for grouping in groupings:
        group_string = ",".join(
            ["%s=%s" % (k,v) for k, v in grouping[0].iteritems()])
        print "CellProfiler -c -r -b -p %s -g %s" % (
            filename, group_string)
        
def write_schema(pipeline_filename):
    if pipeline_filename is None:
        raise ValueError(
            "The --write-schema-and-exit switch must be used in conjunction\n"
            "with the -p or --pipeline switch to load a pipeline with an\n"
            "ExportToDatabase module.")
        
    import cellprofiler.pipeline as cpp
    import cellprofiler.measurements as cpmeas
    import cellprofiler.objects as cpo
    import cellprofiler.workspace as cpw
    pipeline = cpp.Pipeline()
    pipeline.load(pipeline_filename)
    pipeline.turn_off_batch_mode()
    for module in pipeline.modules():
        if module.module_name == "ExportToDatabase":
            break
    else:
        raise ValueError(
        "The pipeline, \"%s\", does not have an ExportToDatabase module" %
        pipeline_filename)
    m = cpmeas.Measurements()
    workspace = cpw.Workspace(
        pipeline, module, m, cpo.ObjectSet, m, None)
    module.prepare_run(workspace)
    
def run_ilastik():
    #
    # Fake ilastik into thinking it is __main__
    #
    import ilastik
    import imp
    sys.argv.remove("--ilastik")
    il_path = ilastik.__path__
    il_file, il_path, il_description = imp.find_module('ilastikMain', il_path)
    imp.load_module('__main__', il_file, il_path, il_description)

def build_extensions():
    '''Compile C and Cython files as needed'''
    import subprocess
    import cellprofiler.utilities.setup
    from distutils.dep_util import newer_group
    #
    # Check for dependencies and compile if necessary
    #
    compile_scripts = [(os.path.join('cellprofiler', 'utilities', 'setup.py'), cellprofiler.utilities.setup)]
    env = os.environ.copy()
    old_pythonpath = os.getenv('PYTHONPATH', None)

    # if we're using a local site_packages, the subprocesses will need
    # to be able to find it.
    
    if old_pythonpath:
        env['PYTHONPATH'] = site_packages + os.pathsep + old_pythonpath
    else:
        env['PYTHONPATH'] = site_packages

    use_mingw = (sys.platform == 'win32' and sys.version_info[0] <= 2 and
                 sys.version_info[1] <= 5)
    for key in list(env.keys()):
        value = env[key]
        if isinstance(key, unicode):
            key = key.encode("utf-8")
        if isinstance(value, unicode):
            value = value.encode("utf-8")
        env[key] = value
    for compile_script, my_module in compile_scripts:
        script_path, script_file = os.path.split(compile_script)
        script_path = os.path.join(root, script_path)
        configuration = my_module.configuration()
        needs_build = False
        for extension in configuration['ext_modules']:
            target = extension.name + '.pyd'
            if newer_group(extension.sources, target):
                needs_build = True
        if not needs_build:
            continue
        if use_mingw:
            p = subprocess.Popen([sys.executable,
                                  script_file,
                                  "build_ext", "-i",
                                  "--compiler=mingw32"],
                                 cwd=script_path,
                                 env=env)
        else:
            p = subprocess.Popen([sys.executable,
                                  script_file,
                                  "build_ext", "-i"],
                                 cwd=script_path,
                                 env=env)
        p.communicate()

def run_pipeline_headless(options, args):
    '''Run a CellProfiler pipeline in headless mode'''
    #
    # Start Ilastik's workers
    #
    try:
        from ilastik.core.jobMachine import GLOBAL_WM
        GLOBAL_WM.set_thread_count(1)
    except:
        logging.root.warn("Failed to stop Ilastik")
    
    if sys.platform == 'darwin':
        if options.start_awt:
            import bioformats
            from javabridge import activate_awt
            activate_awt()
        
    if not options.first_image_set is None:
        if not options.first_image_set.isdigit():
            raise ValueError("The --first-image-set option takes a numeric argument")
        else:
            image_set_start = int(options.first_image_set)
    else:
        image_set_start = None
    
    image_set_numbers = None
    if not options.last_image_set is None:
        if not options.last_image_set.isdigit():
            raise ValueError("The --last-image-set option takes a numeric argument")
        else:
            image_set_end = int(options.last_image_set)
            if image_set_start is None:
                image_set_numbers = np.arange(1, image_set_end+1)
            else:
                image_set_numbers = np.arange(image_set_start, image_set_end+1)
    else:
        image_set_end = None
    
    if ((options.pipeline_filename is not None) and 
        (not options.pipeline_filename.lower().startswith('http'))):
        options.pipeline_filename = os.path.expanduser(options.pipeline_filename)
    from cellprofiler.pipeline import Pipeline, EXIT_STATUS, M_PIPELINE
    import cellprofiler.measurements as cpmeas
    import cellprofiler.preferences as cpprefs
    pipeline = Pipeline()
    initial_measurements = None
    try:
        if h5py.is_hdf5(options.pipeline_filename):
            initial_measurements = cpmeas.load_measurements(
                options.pipeline_filename,
                image_numbers=image_set_numbers)
    except:
        logging.root.info("Failed to load measurements from pipeline")
    if initial_measurements is not None:
        pipeline_text = \
            initial_measurements.get_experiment_measurement(
                M_PIPELINE)
        pipeline_text = pipeline_text.encode('us-ascii')
        pipeline.load(StringIO(pipeline_text))
        if not pipeline.in_batch_mode():
            #
            # Need file list in order to call prepare_run
            #
            from cellprofiler.utilities.hdf5_dict import HDF5FileList
            with h5py.File(options.pipeline_filename, "r") as src:
                if HDF5FileList.has_file_list(src):
                    HDF5FileList.copy(
                        src, initial_measurements.hdf5_dict.hdf5_file)
    else:
        pipeline.load(options.pipeline_filename)
    if options.groups is not None:
        kvs = [x.split('=') for x in options.groups.split(',')]
        groups = dict(kvs)
    else:
        groups = None
    file_list = cpprefs.get_image_set_file()
    if file_list is not None:
        pipeline.read_file_list(file_list)
    #
    # Fixup CreateBatchFiles with any command-line input or output directories
    #
    if pipeline.in_batch_mode():
        create_batch_files = [
            m for m in pipeline.modules()
            if m.is_create_batch_module()]
        if len(create_batch_files) > 0:
            create_batch_files = create_batch_files[0]
            if options.output_directory is not None:
                create_batch_files.custom_output_directory.value = \
                    options.output_directory
            if options.image_directory is not None:
                create_batch_files.default_image_directory.value = \
                    options.image_directory
        
    use_hdf5 = len(args) > 0 and not args[0].lower().endswith(".mat")
    measurements = pipeline.run(
        image_set_start=image_set_start, 
        image_set_end=image_set_end,
        grouping=groups,
        measurements_filename = None if not use_hdf5 else args[0],
        initial_measurements = initial_measurements)
    if len(args) > 0 and not use_hdf5:
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
    if measurements is not None:
        measurements.close()
    
if __name__ == "__main__":
    main(sys.argv)
    os._exit(0)
