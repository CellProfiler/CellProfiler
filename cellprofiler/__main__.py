import bioformats.formatreader
import cellprofiler.gui.app
import cellprofiler.gui.html.manual
import cellprofiler.measurement
import cellprofiler.modules
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preference
import cellprofiler.utilities
import cellprofiler.utilities.cpjvm
import cellprofiler.utilities.hdf5_dict
import cellprofiler.utilities.version
import cellprofiler.utilities.zmqrequest
import cellprofiler.worker
import cellprofiler.workspace
import cStringIO
# import ctypes
import h5py
# import ilastik
# import ilastik.core.jobMachine
import imp
import javabridge
import json
import logging
import logging.config
import matplotlib
import numpy
import optparse
import os
# import pyreadline.logger
import re
# import site
import subprocess
import sys
import tempfile
import wx

OMERO_CK_HOST = "host"
OMERO_CK_PORT = "port"
OMERO_CK_USER = "user"
OMERO_CK_PASSWORD = "password"
OMERO_CK_SESSION_ID = "session-id"
OMERO_CK_CONFIG_FILE = "config-file"

# if sys.platform.startswith('win'):
#     # This recipe is largely from zmq which seems to need this magic
#     # in order to import in frozen mode - a topic the developers never
#     # dealt with.
#     if hasattr(sys, 'frozen'):
#         here = os.path.split(sys.argv[0])[0]
#
#         libzmq = os.path.join(here, 'libzmq.dll')
#         if os.path.exists(libzmq):
#             ctypes.cdll.LoadLibrary(libzmq)
#
# Defeat pyreadline which graciously sets its logging to DEBUG and it
# appears when CP is frozen
#
# try:
#     pyreadline_logger.setLevel(logging.INFO)
#     stop_logging()
# except:
#     pass
#
# if not hasattr(sys, 'frozen'):
#     root = os.path.split(__file__)[0]
# else:
#     root = os.path.split(sys.argv[0])[0]
#
# if len(root) == 0:
#     root = os.curdir
#
# root = os.path.abspath(root)
#
# site_packages = os.path.join(root, 'site-packages').encode('utf-8')
#
# if os.path.exists(site_packages) and os.path.isdir(site_packages):
#     site.addsitedir(site_packages)

# CellProfiler expects NaN as a result during calculation
numpy.seterr(all='ignore')


def main(args=None):
    """
    Run CellProfiler

    :param args: command-line arguments, e.g. sys.argv

    :return:
    """
    if args is None:
        args = sys.argv

    cellprofiler.preference.set_awt_headless(True)

    switches = ('--work-announce', '--knime-bridge-address')

    if any([any([arg.startswith(switch) for switch in switches]) for arg in args]):
        #
        # Go headless ASAP
        #
        cellprofiler.preference.set_headless()

        for i, arg in enumerate(args):
            if arg == "--ij-plugins-directory" and len(args) > i + 1:
                cellprofiler.preference.set_ij_plugin_directory(args[i + 1])

                break

        cellprofiler.worker.aw_parse_args()

        cellprofiler.worker.main()

        sys.exit(0)

    options, args = parse_args(args)

    if options.print_version:
        print u"CellProfiler {0:s}".format(cellprofiler.utilities.version.dotted_version)
        print "Git {0:s}".format(cellprofiler.utilities.version.git_hash)
        print "Version {0:s}".format(cellprofiler.utilities.version.version_number)
        print "Built {0:s}".format(cellprofiler.utilities.version.version_string.split(" ")[0])
        sys.exit(0)

    #
    # Important to go headless ASAP
    #
    if (not options.show_gui) or options.write_schema_and_exit:
        cellprofiler.preference.set_headless()
        # What's there to do but run if you're running headless?
        # Might want to change later if there's some headless setup
        options.run_pipeline = True

    if options.jvm_heap_size is not None:
        cellprofiler.preference.set_jvm_heap_mb(options.jvm_heap_size, False)

    set_log_level(options)

    if options.print_groups_file is not None:
        print_groups(options.print_groups_file)
        return

    if options.batch_commands_file is not None:
        get_batch_commands(options.batch_commands_file)
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

        using_hdf5 = h5py.is_hdf5(path)

        if using_hdf5:
            m = cellprofiler.measurement.Measurement(filename=path, mode="r+")

            pipeline_text = m[cellprofiler.measurement.EXPERIMENT, "Pipeline_Pipeline"]
        else:
            with open(path, "r") as fd:
                pipeline_text = fd.read()

        header, body = pipeline_text.split("\n\n", 1)

        pipeline_text = header + ("\nMessageForUser:{0:s}|{1:s}\n\n".format(caption, message)) + body

        if using_hdf5:
            m[cellprofiler.measurement.EXPERIMENT, "Pipeline_Pipeline"] = pipeline_text

            m.close()
        else:
            with open(path, "w") as fd:
                fd.write(pipeline_text)

        print u"Message added to {0:s}".format(path)

        return

    # necessary to prevent matplotlib trying to use Tkinter as its backend.
    # has to be done before CellProfilerApp is imported
    matplotlib.use('WXAgg')

    if options.omero_credentials is not None:
        set_omero_credentials_from_string(options.omero_credentials)

    if options.plugins_directory is not None:
        cellprofiler.preference.set_plugin_directory(options.plugins_directory, globally=False)

    if options.ij_plugins_directory is not None:
        cellprofiler.preference.set_ij_plugin_directory(options.ij_plugins_directory, globally=False)

    if options.temp_dir is not None:
        if not os.path.exists(options.temp_dir):
            os.makedirs(options.temp_dir)

        cellprofiler.preference.set_temporary_directory(options.temp_dir, globally=False)

    if not options.allow_schema_write:
        cellprofiler.preference.set_allow_schema_write(False)

    #
    # After the crucial preferences are established, we can start the VM
    #
    cellprofiler.utilities.cpjvm.cp_start_vm()

    #
    # Not so crucial preferences...
    #
    if options.image_set_file is not None:
        cellprofiler.preference.set_image_set_file(options.image_set_file)

    try:
        #
        # Handle command-line tasks that that need to load the modules to run
        #
        if options.output_html:
            webpage_path = options.output_directory if options.output_directory else None

            cellprofiler.gui.html.manual.generate_html(webpage_path)

            return

        if options.print_measurements:
            print_measurements(options)
            return

        if not hasattr(sys, "frozen") and options.code_statistics:
            return

        if options.write_schema_and_exit:
            write_schema(options.pipeline_filename)
            return

        if options.show_gui:
            wx.Log.EnableLogging(False)

            if options.pipeline_filename:
                if cellprofiler.workspace.is_workspace_file(options.pipeline_filename):
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

            app = cellprofiler.gui.app.App(0, workspace_path=workspace_path, pipeline_path=pipeline_path)

        if options.data_file is not None:
            cellprofiler.preference.set_data_file(os.path.abspath(options.data_file))

        logging.root.info("Version: {0:s} / {1:d}".format(cellprofiler.utilities.version.version_string, cellprofiler.utilities.version.version_number))

        if options.run_pipeline and not options.pipeline_filename:
            raise ValueError("You must specify a pipeline filename to run")

        if options.output_directory:
            if not os.path.exists(options.output_directory):
                os.makedirs(options.output_directory)

            cellprofiler.preference.set_default_output_directory(options.output_directory)

        if options.image_directory:
            cellprofiler.preference.set_default_image_directory(options.image_directory)

        if options.show_gui:
            if options.run_pipeline:
                app.frame.pipeline_controller.do_analyze_images()
            app.MainLoop()
            return

        elif options.run_pipeline:
            run_pipeline_headless(options, args)
    except Exception, e:
        logging.root.fatal("Uncaught exception in CellProfiler.py", exc_info=True)
        raise

    finally:
        stop_cellprofiler()


def stop_cellprofiler():
    """

    :return:
    """
    try:
        GLOBAL_WM.stopWorkers()
    except:
        logging.root.warn("Failed to stop Ilastik")
    try:
        cellprofiler.utilities.zmqrequest.join_to_the_boundary()
    except:
        logging.root.warn("Failed to stop zmq boundary", exc_info=1)
    try:
        cellprofiler.utilities.cpjvm.cp_stop_vm()
    except:
        logging.root.warn("Failed to stop the JVM", exc_info=1)


def parse_args(args):
    """
    Parse the CellProfiler command-line arguments

    :param args:

    :return:
    """

    usage = """usage: %prog [options] [<output-file>])
         where <output-file> is the optional filename for the output file of
               measurements when running headless.
         The flags -p, -r and -c are required for a headless run."""

    if "--do-not-fetch" in args:
        args = list(args)
        args.remove("--do-not-fetch")
        # Would be nice to log here, but logging is not yet set up.

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

    parser.add_option("--plugins-directory",
                      dest="plugins_directory",
                      help=("CellProfiler will look for plugin modules in this "
                            "directory (headless-only)."))

    parser.add_option("--ij-plugins-directory",
                      dest="ij_plugins_directory",
                      help=("CellProfiler will look for ImageJ plugin modules "
                            "in this directory (headless-only)."))

    parser.add_option("--add-message-for-user",
                      dest="add_message_for_user",
                      default=False,
                      action="store_true",
                      help=("This option lets you add a message to a pipeline "
                            "or project file which will appear in a message box"
                            "when that pipeline or project file is opened. "))

    parser.add_option("--version",
                      dest="print_version",
                      default=False,
                      action="store_true",
                      help="Print the version and exit")

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
                      help="Open the measurements file following the "
                           "--print-groups switch and print the groups in its image "
                           "sets. The measurements file should be generated using "
                           "CreateBatchFiles. The output is a JSON-encoded data "
                           "structure containing the group keys and values and the "
                           "image sets in each group.")

    parser.add_option("--get-batch-commands",
                      dest="batch_commands_file",
                      default=None,
                      help="Open the measurements file following the "
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
                      default=None,
                      help="Specify the location of a .csv file for LoadData. "
                           "If this switch is present, this file is used instead of "
                           "the one specified in the LoadData module.")

    parser.add_option("--file-list",
                      dest="image_set_file",
                      default=None,
                      help="Specify a file list of one file or URL per line "
                           "to be used to initially populate the Images module's "
                           "file list.")
    parser.add_option("--do-not-write-schema",
                      dest='allow_schema_write',
                      default=True,
                      action="store_false",
                      help="Do not execute the schema definition and other "
                           "per-experiment SQL commands during initialization "
                           "when running a pipeline in batch mode.")

    parser.add_option("--write-schema-and-exit",
                      dest='write_schema_and_exit',
                      default=False,
                      action='store_true',
                      help="Create the experiment database schema and exit")

    parser.add_option("--omero-credentials",
                      dest="omero_credentials",
                      default=None,
                      help=(
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
                      dest="log_level",
                      default=str(logging.INFO),
                      help=("Set the verbosity for logging messages: " +
                            ("{0:d} or {1:s} for debugging, ".format(logging.DEBUG, "DEBUG")) +
                            ("{0:d} or {1:s} for informational, ".format(logging.INFO, "INFO")) +
                            ("{0:d} or {1:s} for warning, ".format(logging.WARNING, "WARNING")) +
                            ("{0:d} or {1:s} for error, ".format(logging.ERROR, "ERROR")) +
                            ("{0:d} or {1:s} for critical, ".format(logging.CRITICAL, "CRITICAL")) +
                            ("{0:d} or {1:s} for fatal.".format(logging.FATAL, "FATAL")) +
                            " Otherwise, the argument is interpreted as the file name of a log configuration file (see http://docs.python.org/library/logging.config.html for file format)"))

    if sys.platform == 'darwin':
        parser.add_option("--use-awt",
                          dest="start_awt",
                          action="store_true",
                          default=False,
                          help="Initialize the Java AWT UI so it can be used when running headless on OS/X")

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
    """
    Set the logging package's log level based on command-line options

    :param options:

    :return:
    """
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
    """
    Set the OMERO server / port / session ID

    :param credentials_string:  a comma-separated key/value pair string (key=value)
                                that gives the credentials. Keys are
                                host - the DNS name or IP address of the OMERO server
                                port - the TCP port to use to connect
                                user - the user name
                                session-id - the session ID used for authentication

    :return:
    """

    if re.match("([^=^,]+=[^=^,]+,)*([^=^,]+=[^=^,]+)", credentials_string) is None:
        logging.root.error(u'The OMERO credentials string, "{0:s}", is badly-formatted.'.format(credentials_string))

        logging.root.error('It should have the form: ' '"host=hostname.org,port=####,user=<user>,session-id=<session-id>\n')

        raise ValueError("Invalid format for --omero-credentials")

    for k, v in [kv.split("=", 1) for kv in credentials_string.split(",")]:
        k = k.lower()

        credentials = {
            bioformats.formatreader.K_OMERO_SERVER: cellprofiler.preference.get_omero_server(),
            bioformats.formatreader.K_OMERO_PORT: cellprofiler.preference.get_omero_port(),
            bioformats.formatreader.K_OMERO_USER: cellprofiler.preference.get_omero_user(),
            bioformats.formatreader.K_OMERO_SESSION_ID: cellprofiler.preference.get_omero_session_id()
        }

        if k == OMERO_CK_HOST:
            cellprofiler.preference.set_omero_server(v, globally=False)
            credentials[bioformats.formatreader.K_OMERO_SERVER] = v
        elif k == OMERO_CK_PORT:
            cellprofiler.preference.set_omero_port(v, globally=False)
            credentials[bioformats.formatreader.K_OMERO_PORT] = v
        elif k == OMERO_CK_SESSION_ID:
            credentials[bioformats.formatreader.K_OMERO_SESSION_ID] = v
        elif k == OMERO_CK_USER:
            cellprofiler.preference.set_omero_user(v, globally=False)
            credentials[bioformats.formatreader.K_OMERO_USER] = v
        elif k == OMERO_CK_PASSWORD:
            credentials[bioformats.formatreader.K_OMERO_PASSWORD] = v
        elif k == OMERO_CK_CONFIG_FILE:
            credentials[bioformats.formatreader.K_OMERO_CONFIG_FILE] = v

            if not os.path.isfile(v):
                msg = u"Cannot find OMERO config file, {0:s}".format(v)

                logging.root.error(msg)

                raise ValueError(msg)
        else:
            logging.root.error(u'Unknown --omero-credentials keyword: "{0:s}"'.format(k))
            logging.root.error(u'Acceptable keywords are: "{0:s}"'.format('","'.join([OMERO_CK_HOST, OMERO_CK_PORT, OMERO_CK_SESSION_ID])))

            raise ValueError("Invalid format for --omero-credentials")

    bioformats.formatreader.use_omero_credentials(credentials)


def print_measurements(options):
    """
    Print the measurements that would be output by a pipeline

    This function calls Pipeline.get_measurement_columns() to get the
    measurements that would be output by a pipeline. This can be used in
    a workflow tool or LIMS to find the outputs of a pipeline without
    running it. For instance, someone might want to integrate CellProfiler
    with Knime and write a Knime node that let the user specify a pipeline
    file. The node could then execute CellProfiler with the --measurements
    switch and display the measurements as node outputs.

    :param options:

    :return:
    """
    if options.pipeline_filename is None:
        raise ValueError("Can't print measurements, no pipeline file")

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(pipeline, event):
        if isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            raise ValueError(u"Failed to load {0:s}".format(options.pipeline_filename))

    pipeline.add_listener(callback)

    pipeline.load(os.path.expanduser(options.pipeline_filename))

    columns = pipeline.get_measurement_columns()

    print "--- begin measurements ---"

    print "Object,Feature,Type"

    for column in columns:
        object_name, feature, data_type = column[:3]

        print "{0:s},{1:s},{2:s}".format(object_name, feature, data_type)

    print "--- end measurements ---"


def print_groups(filename):
    """
    Print the image set groups for this pipeline

    This function outputs a JSON string to the console composed of a list
    of the groups in the pipeline image set. Each element of the list is
    a two-tuple whose first element is a key/value dictionary of the
    group's key and the second is a tuple of the image numbers in the group.

    :param filename:

    :return:
    """

    path = os.path.expanduser(filename)

    m = cellprofiler.measurement.Measurement(filename=path, mode="r")

    metadata_tags = m.get_grouping_tags()

    groupings = m.get_groupings(metadata_tags)

    json.dump(groupings, sys.stdout)


def get_batch_commands(filename):
    """
    Print the commands needed to run the given batch data file headless

    The output assumes that the executable, "CellProfiler", can be used
    to run the command from the shell. Alternatively, the output could be
    run through a utility such as "sed":

    CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_job.sh/

    :param filename: the name of a Batch_data.h5 file. The file should group image sets.

    :return:
    """
    path = os.path.expanduser(filename)

    m = cellprofiler.measurement.Measurement(filename=path, mode="r")

    image_numbers = m.get_image_numbers()

    if m.has_feature(cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_NUMBER):
        group_numbers = m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_NUMBER, image_numbers]

        group_indexes = m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_INDEX, image_numbers]

        if numpy.any(group_numbers != 1) and numpy.all((group_indexes[1:] == group_indexes[:-1] + 1) | ((group_indexes[1:] == 1) & (group_numbers[1:] == group_numbers[:-1] + 1))):
            #
            # Do -f and -l if more than one group and group numbers
            # and indices are properly constructed
            #
            bins = numpy.bincount(group_numbers)

            cumsums = numpy.cumsum(bins)

            prev = 0

            for i, off in enumerate(cumsums):
                if off == prev:
                    continue

                print "CellProfiler -c -r -p {0:s} -f {1:d} -l {2:d}".format(filename, prev + 1, off)

                prev = off

            return

    metadata_tags = m.get_grouping_tags()

    groupings = m.get_groupings(metadata_tags)

    for grouping in groupings:
        group_string = ",".join(["{0:s}={1:s}".format(k, v) for k, v in grouping[0].iteritems()])

        print "CellProfiler -c -r -p {0:s} -g {1:s}".format(filename, group_string)


def write_schema(pipeline_filename):
    """

    :param pipeline_filename:

    :return:
    """
    if pipeline_filename is None:
        raise ValueError("The --write-schema-and-exit switch must be used in conjunction\n with the -p or --pipeline switch to load a pipeline with an\n ExportToDatabase module.")

    pipeline = cellprofiler.pipeline.Pipeline()

    pipeline.load(pipeline_filename)

    pipeline.turn_off_batch_mode()

    for module in pipeline.modules():
        if module.module_name == "ExportToDatabase":
            break
    else:
        raise ValueError(u"The pipeline, \"{0:s}\", does not have an ExportToDatabase module".format(pipeline_filename))

    m = cellprofiler.measurement.Measurement()

    workspace = cellprofiler.workspace.Workspace(pipeline, module, m, cellprofiler.object.ObjectSet, m, None)

    module.prepare_run(workspace)


def run_pipeline_headless(options, args):
    """
    Run a CellProfiler pipeline in headless mode

    :param options:
    :param args:

    :return:
    """
    #
    # Start Ilastik's workers
    #
    try:
        GLOBAL_WM.set_thread_count(1)
    except:
        logging.root.warn("Failed to stop Ilastik")

    if sys.platform == 'darwin':
        if options.start_awt:
            javabridge.activate_awt()

    if options.first_image_set is not None:
        if not options.first_image_set.isdigit():
            raise ValueError("The --first-image-set option takes a numeric argument")
        else:
            image_set_start = int(options.first_image_set)
    else:
        image_set_start = None

    image_set_numbers = None

    if options.last_image_set is not None:
        if not options.last_image_set.isdigit():
            raise ValueError("The --last-image-set option takes a numeric argument")
        else:
            image_set_end = int(options.last_image_set)

            if image_set_start is None:
                image_set_numbers = numpy.arange(1, image_set_end + 1)
            else:
                image_set_numbers = numpy.arange(image_set_start, image_set_end + 1)
    else:
        image_set_end = None

    if (options.pipeline_filename is not None) and (not options.pipeline_filename.lower().startswith('http')):
        options.pipeline_filename = os.path.expanduser(options.pipeline_filename)

    pipeline = cellprofiler.pipeline.Pipeline()

    initial_measurements = None

    try:
        if h5py.is_hdf5(options.pipeline_filename):
            initial_measurements = cellprofiler.measurement.load_measurements(options.pipeline_filename, image_numbers=image_set_numbers)
    except:
        logging.root.info("Failed to load measurements from pipeline")

    if initial_measurements is not None:
        pipeline_text = initial_measurements.get_experiment_measurement(cellprofiler.pipeline.M_PIPELINE)

        pipeline_text = pipeline_text.encode('us-ascii')

        pipeline.load(cStringIO.StringIO(pipeline_text))

        if not pipeline.in_batch_mode():
            #
            # Need file list in order to call prepare_run
            #
            with h5py.File(options.pipeline_filename, "r") as src:
                if cellprofiler.utilities.hdf5_dict.HDF5FileList.has_file_list(src):
                    cellprofiler.utilities.hdf5_dict.HDF5FileList.copy(src, initial_measurements.hdf5_dict.hdf5_file)
    else:
        pipeline.load(options.pipeline_filename)

    if options.groups is not None:
        kvs = [x.split('=') for x in options.groups.split(',')]

        groups = dict(kvs)
    else:
        groups = None

    file_list = cellprofiler.preference.get_image_set_file()

    if file_list is not None:
        pipeline.read_file_list(file_list)

    #
    # Fixup CreateBatchFiles with any command-line input or output directories
    #
    if pipeline.in_batch_mode():
        create_batch_files = [m for m in pipeline.modules() if m.is_create_batch_module()]

        if len(create_batch_files) > 0:
            create_batch_files = create_batch_files[0]

            if options.output_directory is not None:
                create_batch_files.custom_output_directory.value = options.output_directory

            if options.image_directory is not None:
                create_batch_files.default_image_directory.value = options.image_directory

    use_hdf5 = len(args) > 0 and not args[0].lower().endswith(".mat")
    measurements = pipeline.run(
            image_set_start=image_set_start,
            image_set_end=image_set_end,
            grouping=groups,
            measurements_filename=None if not use_hdf5 else args[0],
            initial_measurements=initial_measurements)

    if len(args) > 0 and not use_hdf5:
        pipeline.save_measurements(args[0], measurements)

    if options.done_file is not None:
        if measurements is not None and measurements.has_feature(cellprofiler.measurement.EXPERIMENT, cellprofiler.pipeline.EXIT_STATUS):
            done_text = measurements.get_experiment_measurement(cellprofiler.pipeline.EXIT_STATUS)
        else:
            done_text = "Failure"

        fd = open(options.done_file, "wt")

        fd.write(u"{0:s}\n".format(done_text))

        fd.close()

    if measurements is not None:
        measurements.close()


if __name__ == "__main__":
    main()
