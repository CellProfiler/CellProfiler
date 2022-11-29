import io
import json
import logging
import logging.config
import optparse
import os
import os.path
import re
import sys
import tempfile
import urllib.parse

import h5py
import matplotlib
import numpy
import pkg_resources
from cellprofiler_core.constants.measurement import EXPERIMENT
from cellprofiler_core.constants.measurement import GROUP_INDEX
from cellprofiler_core.constants.measurement import GROUP_NUMBER
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.constants.pipeline import M_PIPELINE, EXIT_STATUS
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import LoadException
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.preferences import get_image_set_file
from cellprofiler_core.preferences import get_temporary_directory
from cellprofiler_core.preferences import set_conserve_memory
from cellprofiler_core.preferences import set_force_bioformats
from cellprofiler_core.preferences import get_omero_port
from cellprofiler_core.preferences import get_omero_server
from cellprofiler_core.preferences import get_omero_session_id
from cellprofiler_core.preferences import get_omero_user
from cellprofiler_core.preferences import set_allow_schema_write
from cellprofiler_core.preferences import set_always_continue
from cellprofiler_core.preferences import set_awt_headless
from cellprofiler_core.preferences import set_data_file
from cellprofiler_core.preferences import set_default_image_directory
from cellprofiler_core.preferences import set_default_output_directory
from cellprofiler_core.preferences import set_headless
from cellprofiler_core.preferences import set_image_set_file
from cellprofiler_core.preferences import set_omero_port
from cellprofiler_core.preferences import set_omero_server
from cellprofiler_core.preferences import set_omero_user
from cellprofiler_core.preferences import set_plugin_directory
from cellprofiler_core.preferences import set_temporary_directory
from cellprofiler_core.preferences import set_widget_inspector
from cellprofiler_core.utilities.core.workspace import is_workspace_file
from cellprofiler_core.utilities.hdf5_dict import HDF5FileList
from cellprofiler_core.utilities.java import start_java, stop_java
from cellprofiler_core.utilities.measurement import load_measurements
from cellprofiler_core.utilities.zmq import join_to_the_boundary
from cellprofiler_core.worker import aw_parse_args
from cellprofiler_core.worker import main as worker_main
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.reader import activate_readers

if hasattr(sys, "frozen"):
    if sys.platform == "darwin":
        # Some versions of Macos like to put CP in a sandbox. If we're frozen Java should be packed in,
        # so let's just figure out the directory at run time.
        try:
            os.environ["CP_JAVA_HOME"] = os.path.abspath(os.path.join(sys.prefix, "..", "Resources/Home"))
        except:
            print("Unable to set JAVA directory to inbuilt java environment")
    elif sys.platform.startswith("win"):
        # Clear out deprecation warnings from PyInstaller
        os.system('cls')
        # For Windows builds use built-in Java for CellProfiler, otherwise try to use Java from elsewhere on the system.
        # Users can use a custom java installation by removing CP_JAVA_HOME.
        # JAVA_HOME must be set before bioformats import.
        try:
            if "CP_JAVA_HOME" in os.environ:
                # Use user-provided Java
                os.environ["JAVA_HOME"] = os.environ["CP_JAVA_HOME"]
            elif "JAVA_HOME" not in os.environ:
                # Use built-in java
                test_dir = os.path.abspath(os.path.join(sys.prefix, "java"))
                if os.path.exists(test_dir):
                    os.environ["JAVA_HOME"] = test_dir
                else:
                    print(f"Failed to detect java automatically. Searched in: {test_dir}.")
            assert "JAVA_HOME" in os.environ and os.path.exists(os.environ['JAVA_HOME'])
            # Ensure we start in the correct directory when launching a build.
            # Opening a file directly may end up with us starting on the wrong drive.
            os.chdir(sys.prefix)
        except AssertionError:
            print(
                "CellProfiler Startup ERROR: Could not find path to Java environment directory.\n"
                "Please set the CP_JAVA_HOME system environment variable.\n"
                "Visit http://broad.io/cpjava for instructions."
            )
            os.system("pause")  # Keep console window open until keypress.
            sys.exit(1)
        except Exception as e:
            print(f"Encountered unknown error during startup: {e}")
    else:
        # Clear out deprecation warnings from PyInstaller
        os.system('clear')
    from cellprofiler import __version__ as ver
    print(f"Starting CellProfiler {ver}")


OMERO_CK_HOST = "host"
OMERO_CK_PORT = "port"
OMERO_CK_USER = "user"
OMERO_CK_PASSWORD = "password"
OMERO_CK_SESSION_ID = "session-id"
OMERO_CK_CONFIG_FILE = "config-file"

numpy.seterr(all="ignore")


def main(args=None):
    """Run CellProfiler

    args - command-line arguments, e.g., sys.argv
    """
    if args is None:
        args = sys.argv

    set_awt_headless(True)

    exit_code = 0

    switches = ("--work-announce", "--knime-bridge-address")

    if any([any([arg.startswith(switch) for switch in switches]) for arg in args]):
        set_headless()
        aw_parse_args()
        activate_readers()
        worker_main()
        return exit_code

    options, args = parse_args(args)
    
    if options.print_version:
        set_headless()
        __version__(exit_code)

    if (not options.show_gui) or options.write_schema_and_exit:
        set_headless()

        options.run_pipeline = True

    if options.batch_commands_file or options.new_batch_commands_file:
        set_headless()
        options.run_pipeline = False
        options.show_gui = False

    # must be run after last possible invocation of set_headless()
    activate_readers()

    if options.temp_dir is not None:
        if not os.path.exists(options.temp_dir):
            os.makedirs(options.temp_dir)
        set_temporary_directory(options.temp_dir, globally=False)

    temp_dir = get_temporary_directory()

    to_clean = []

    if options.pipeline_filename:
        o = urllib.parse.urlparse(options.pipeline_filename)
        if o[0] in ("ftp", "http", "https"):
            from urllib.request import urlopen

            temp_pipe_file = tempfile.NamedTemporaryFile(
                mode="w+b", suffix=".cppipe", dir=temp_dir, delete=False
            )
            downloaded_pipeline = urlopen(options.pipeline_filename)
            for line in downloaded_pipeline:
                temp_pipe_file.write(line)
            options.pipeline_filename = temp_pipe_file.name
            to_clean.append(os.path.join(temp_dir, temp_pipe_file.name))

    if options.image_set_file:
        o = urllib.parse.urlparse(options.image_set_file)
        if o[0] in ("ftp", "http", "https"):
            from urllib.request import urlopen

            temp_set_file = tempfile.NamedTemporaryFile(
                mode="w+b", suffix=".csv", dir=temp_dir, delete=False
            )
            downloaded_set_csv = urlopen(options.image_set_file)
            for line in downloaded_set_csv:
                temp_set_file.write(line)
            options.image_set_file = temp_set_file.name
            to_clean.append(os.path.join(temp_dir, temp_set_file.name))

    if options.data_file:
        o = urllib.parse.urlparse(options.data_file)
        if o[0] in ("ftp", "http", "https"):
            from urllib.request import urlopen

            temp_data_file = tempfile.NamedTemporaryFile(
                mode="w+b", suffix=".csv", dir=temp_dir, delete=False
            )
            downloaded_data_csv = urlopen(options.data_file)
            for line in downloaded_data_csv:
                temp_data_file.write(line)
            options.data_file = temp_data_file.name
            to_clean.append(os.path.join(temp_dir, temp_data_file.name))

    set_log_level(options)

    if options.print_groups_file is not None:
        print_groups(options.print_groups_file)

    if options.batch_commands_file is not None:
        try:
            nr_per_batch = int(options.images_per_batch)
        except ValueError:
            logging.warning(
                "non-integer argument to --images-per-batch. Defaulting to 1."
            )
            nr_per_batch = 1
        get_batch_commands(options.batch_commands_file, nr_per_batch)
    
    if options.new_batch_commands_file is not None:
        try:
            nr_per_batch = int(options.images_per_batch)
        except ValueError:
            logging.warning(
                "non-integer argument to --images-per-batch. Defaulting to 1."
            )
            nr_per_batch = 1
        get_batch_commands_new(options.new_batch_commands_file, nr_per_batch)

    if options.omero_credentials is not None:
        set_omero_credentials_from_string(options.omero_credentials)

    if options.plugins_directory is not None:
        set_plugin_directory(options.plugins_directory, globally=False)

    if options.conserve_memory is not None:
        set_conserve_memory(options.conserve_memory, globally=False)

    if options.force_bioformats is not None:
        set_force_bioformats(options.force_bioformats, globally=False)

    if options.always_continue is not None:
        set_always_continue(options.always_continue, globally=False)

    if not options.allow_schema_write:
        set_allow_schema_write(False)

    if options.output_directory:
        if not os.path.exists(options.output_directory):
            os.makedirs(options.output_directory)

        set_default_output_directory(options.output_directory)

    if options.image_directory:
        set_default_image_directory(options.image_directory)

    if options.run_pipeline and not options.pipeline_filename:
        raise ValueError("You must specify a pipeline filename to run")

    if options.data_file is not None:
        set_data_file(os.path.abspath(options.data_file))

    if options.widget_inspector:
        set_widget_inspector(True, globally=False)

    try:

        if options.image_set_file is not None:
            set_image_set_file(options.image_set_file)

        #
        # Handle command-line tasks that that need to load the modules to run
        #
        if options.print_measurements:
            print_measurements(options)

        if options.write_schema_and_exit:
            write_schema(options.pipeline_filename)

        if options.show_gui:
            matplotlib.use("WXAgg")

            import cellprofiler.gui.app

            if options.pipeline_filename:
                if is_workspace_file(options.pipeline_filename):
                    workspace_path = os.path.expanduser(options.pipeline_filename)

                    pipeline_path = None
                else:
                    pipeline_path = os.path.expanduser(options.pipeline_filename)

                    workspace_path = None
            else:
                workspace_path = None

                pipeline_path = None

            app = cellprofiler.gui.app.App(
                0, workspace_path=workspace_path, pipeline_path=pipeline_path
            )

            if options.run_pipeline:
                app.frame.pipeline_controller.do_analyze_images()

            app.MainLoop()

            return
        elif options.run_pipeline:
            exit_code = run_pipeline_headless(options, args)

    finally:
        # Cleanup the temp files we made, if any
        if len(to_clean) > 0:
            for each_temp in to_clean:
                os.remove(each_temp)
        # If anything goes wrong during the startup sequence headlessly, the JVM needs
        # to be explicitly closed
        if not options.show_gui:
            stop_cellprofiler()

    return exit_code


def __version__(exit_code):
    print(pkg_resources.get_distribution("CellProfiler").version)

    sys.exit(exit_code)


def stop_cellprofiler():

    # Bioformats readers have to be properly closed.
    # This is especially important when using OmeroReaders as leaving the
    # readers open leaves the OMERO.server services open which in turn leads to
    # high memory consumption.
    from cellprofiler_core.constants.reader import ALL_READERS
    for reader in ALL_READERS.values():
        reader.clear_cached_readers()
    stop_java()


def parse_args(args):
    """Parse the CellProfiler command-line arguments"""
    usage = """usage: %prog [options] [<output-file>])
         where <output-file> is the optional filename for the output file of
               measurements when running headless.
         The flags -p, -r and -c are required for a headless run."""

    if "--do-not-fetch" in args:
        args = list(args)

        args.remove("--do-not-fetch")

    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "-p",
        "--pipeline",
        "--project",
        dest="pipeline_filename",
        help="Load this pipeline file or project on startup. If specifying a pipeline file rather than a project, the -i flag is also needed unless the pipeline is saved with the file list.",
        default=None,
    )

    default_show_gui = True

    if sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
        default_show_gui = False

    parser.add_option(
        "-c",
        "--run-headless",
        action="store_false",
        dest="show_gui",
        default=default_show_gui,
        help="Run headless (without the GUI)",
    )

    parser.add_option(
        "-r",
        "--run",
        action="store_true",
        dest="run_pipeline",
        default=False,
        help="Run the given pipeline on startup",
    )

    parser.add_option(
        "-o",
        "--output-directory",
        dest="output_directory",
        default=None,
        help="Make this directory the default output folder",
    )

    parser.add_option(
        "-i",
        "--image-directory",
        dest="image_directory",
        default=None,
        help="Make this directory the default input folder",
    )

    parser.add_option(
        "-f",
        "--first-image-set",
        dest="first_image_set",
        default=None,
        help="The one-based index of the first image set to process",
    )

    parser.add_option(
        "-l",
        "--last-image-set",
        dest="last_image_set",
        default=None,
        help="The one-based index of the last image set to process",
    )

    parser.add_option(
        "-g",
        "--group",
        dest="groups",
        default=None,
        help='Restrict processing to one grouping in a grouped pipeline. For instance, "-g ROW=H,COL=01", will process only the group of image sets that match the keys.',
    )

    parser.add_option(
        "--plugins-directory",
        dest="plugins_directory",
        help="CellProfiler will look for plugin modules in this directory (headless-only).",
    )

    parser.add_option(
        "--conserve-memory",
        dest="conserve_memory",
        default=None,
        help="CellProfiler will attempt to release unused memory after each image set.",
    )

    parser.add_option(
        "--force-bioformats",
        dest="force_bioformats",
        default=None,
        help="CellProfiler will always use BioFormats for reading images.",
    )

    parser.add_option(
        "--version",
        dest="print_version",
        default=False,
        action="store_true",
        help="Print the version and exit",
    )

    parser.add_option(
        "-t",
        "--temporary-directory",
        dest="temp_dir",
        default=None,
        help=(
            "Temporary directory. "
            "CellProfiler uses this for downloaded image files "
            "and for the measurements file, if not specified. "
            "The default is " + tempfile.gettempdir()
        ),
    )

    parser.add_option(
        "-d",
        "--done-file",
        dest="done_file",
        default=None,
        help='The path to the "Done" file, written by CellProfiler shortly before exiting',
    )

    parser.add_option(
        "--measurements",
        dest="print_measurements",
        default=False,
        action="store_true",
        help="Open the pipeline file specified by the -p switch and print the measurements made by that pipeline",
    )

    parser.add_option(
        "--print-groups",
        dest="print_groups_file",
        default=None,
        help="Open the measurements file following the --print-groups switch and print the groups in its image sets. The measurements file should be generated using CreateBatchFiles. The output is a JSON-encoded data structure containing the group keys and values and the image sets in each group.",
    )

    parser.add_option(
        "--get-batch-commands",
        dest="batch_commands_file",
        default=None,
        help='Open the measurements file following the --get-batch-commands switch and print one line to the console per group. The measurements file should be generated using CreateBatchFiles and the image sets should be grouped into the units to be run. Each line is a command to invoke CellProfiler. You can use this option to generate a shell script that will invoke CellProfiler on a cluster by substituting "CellProfiler" '
        "with your invocation command in the script's text, for instance: CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_jobs.sh. Note that CellProfiler will always run in headless mode when --get-batch-commands is present and will exit after generating the batch commands without processing any pipeline. Note that this exact version is deprecated and will be removed in CellProfiler 5; you may use the new version now with --get-batch-commands-new",
    )

    parser.add_option(
        "--get-batch-commands-new",
        dest="new_batch_commands_file",
        default=None,
        help='Open the batch file following the --get-batch-commands-new switch and print one line to the console per group. Each line is a command to invoke CellProfiler. You can use this option to generate a shell script that will invoke CellProfiler on a cluster by substituting "CellProfiler". This new version (which will be the only version in CellProfiler 5) will return groups if CellProfiler has more than one group and --images-per-batch is NOT passed (or is passed as 1), otherwise it will always return -f and -l commands. '
        "with your invocation command in the script's text, for instance: CellProfiler --get-batch-commands-new Batch_data.h5 | sed s/CellProfiler/farm_jobs.sh. Note that CellProfiler will always run in headless mode when --get-batch-commands is present and will exit after generating the batch commands without processing any pipeline.",
    )

    parser.add_option(
        "--images-per-batch",
        dest="images_per_batch",
        default="1",
        help="For pipelines that do not use image grouping this option specifies the number of images that should be processed in each batch if --get-batch-commands is used. Defaults to 1.",
    )

    parser.add_option(
        "--data-file",
        dest="data_file",
        default=None,
        help="Specify the location of a .csv file for LoadData. If this switch is present, this file is used instead of the one specified in the LoadData module.",
    )

    parser.add_option(
        "--file-list",
        dest="image_set_file",
        default=None,
        help="Specify a file list of one file or URL per line to be used to initially populate the Images module's file list.",
    )

    parser.add_option(
        "--do-not-write-schema",
        dest="allow_schema_write",
        default=True,
        action="store_false",
        help="Do not execute the schema definition and other per-experiment SQL commands during initialization when running a pipeline in batch mode.",
    )

    parser.add_option(
        "--write-schema-and-exit",
        dest="write_schema_and_exit",
        default=False,
        action="store_true",
        help="Create the experiment database schema and exit",
    )

    parser.add_option(
        "--omero-credentials",
        dest="omero_credentials",
        default=None,
        help=(
            "Enter login credentials for OMERO. The credentials"
            " are entered as comma-separated key/value pairs with"
            ' keys, "%(OMERO_CK_HOST)s" - the DNS host name for the OMERO server'
            ', "%(OMERO_CK_PORT)s" - the server\'s port # (typically 4064)'
            ', "%(OMERO_CK_USER)s" - the name of the connecting user'
            ', "%(OMERO_CK_PASSWORD)s" - the connecting user\'s password'
            ', "%(OMERO_CK_SESSION_ID)s" - the session ID for an OMERO client session.'
            ', "%(OMERO_CK_CONFIG_FILE)s" - the path to the OMERO credentials config file.'
            " A typical set of credentials might be:"
            " --omero-credentials host=demo.openmicroscopy.org,port=4064,session-id=atrvomvjcjfe7t01e8eu59amixmqqkfp"
        )
        % globals(),
    )

    parser.add_option(
        "-L",
        "--log-level",
        dest="log_level",
        default=str(logging.INFO),
        help=(
            "Set the verbosity for logging messages: "
            + ("%d or %s for debugging, " % (logging.DEBUG, "DEBUG"))
            + ("%d or %s for informational, " % (logging.INFO, "INFO"))
            + ("%d or %s for warning, " % (logging.WARNING, "WARNING"))
            + ("%d or %s for error, " % (logging.ERROR, "ERROR"))
            + ("%d or %s for critical, " % (logging.CRITICAL, "CRITICAL"))
            + ("%d or %s for fatal." % (logging.FATAL, "FATAL"))
            + " Otherwise, the argument is interpreted as the file name of a log configuration file (see http://docs.python.org/library/logging.config.html for file format)"
        ),
    )

    parser.add_option(
        "--always-continue",
        dest="always_continue",
        default=None,
        action="store_true",
        help="Keep running after an image set throws an error"
    )

    parser.add_option(
        "--widget-inspector",
        dest="widget_inspector",
        default=False,
        action="store_true",
        help="Enable the widget inspector menu item under \"Test\""
    )

    options, result_args = parser.parse_args(args[1:])
    if len(args) == 2:
        if args[1].lower().endswith((".cpproj", ".cppipe")):
            # Opening a file with CellProfiler will supply the file as an argument
            options.pipeline_filename = args[1]
            result_args = []
    return options, result_args


def set_log_level(options):
    """Set the logging package's log level based on command-line options"""
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
    """Set the OMERO server / port / session ID

    credentials_string: a comma-separated key/value pair string (key=value)
                        that gives the credentials. Keys are
                        host - the DNS name or IP address of the OMERO server
                        port - the TCP port to use to connect
                        user - the user name
                        session-id - the session ID used for authentication
    """
    import bioformats.formatreader

    if re.match("([^=^,]+=[^=^,]+,)*([^=^,]+=[^=^,]+)", credentials_string) is None:
        logging.root.error(
            'The OMERO credentials string, "%s", is badly-formatted.'
            % credentials_string
        )

        logging.root.error(
            'It should have the form: "host=hostname.org,port=####,user=<user>,session-id=<session-id>\n'
        )

        raise ValueError("Invalid format for --omero-credentials")

    credentials = {}

    for k, v in [kv.split("=", 1) for kv in credentials_string.split(",")]:
        k = k.lower()

        credentials = {
            bioformats.formatreader.K_OMERO_SERVER: get_omero_server(),
            bioformats.formatreader.K_OMERO_PORT: get_omero_port(),
            bioformats.formatreader.K_OMERO_USER: get_omero_user(),
            bioformats.formatreader.K_OMERO_SESSION_ID: get_omero_session_id(),
        }

        if k == OMERO_CK_HOST:
            set_omero_server(v, globally=False)

            credentials[bioformats.formatreader.K_OMERO_SERVER] = v
        elif k == OMERO_CK_PORT:
            set_omero_port(v, globally=False)

            credentials[bioformats.formatreader.K_OMERO_PORT] = v
        elif k == OMERO_CK_SESSION_ID:
            credentials[bioformats.formatreader.K_OMERO_SESSION_ID] = v
        elif k == OMERO_CK_USER:
            set_omero_user(v, globally=False)

            credentials[bioformats.formatreader.K_OMERO_USER] = v
        elif k == OMERO_CK_PASSWORD:
            credentials[bioformats.formatreader.K_OMERO_PASSWORD] = v
        elif k == OMERO_CK_CONFIG_FILE:
            credentials[bioformats.formatreader.K_OMERO_CONFIG_FILE] = v

            if not os.path.isfile(v):
                msg = "Cannot find OMERO config file, %s" % v

                logging.root.error(msg)

                raise ValueError(msg)
        else:
            logging.root.error('Unknown --omero-credentials keyword: "%s"' % k)

            logging.root.error(
                'Acceptable keywords are: "%s"'
                % '","'.join([OMERO_CK_HOST, OMERO_CK_PORT, OMERO_CK_SESSION_ID])
            )

            raise ValueError("Invalid format for --omero-credentials")

    bioformats.formatreader.use_omero_credentials(credentials)


def print_measurements(options):
    """Print the measurements that would be output by a pipeline

    This function calls Pipeline.get_measurement_columns() to get the
    measurements that would be output by a pipeline. This can be used in
    a workflow tool or LIMS to find the outputs of a pipeline without
    running it. For instance, someone might want to integrate CellProfiler
    with Knime and write a Knime node that let the user specify a pipeline
    file. The node could then execute CellProfiler with the --measurements
    switch and display the measurements as node outputs.
    """

    if options.pipeline_filename is None:
        raise ValueError("Can't print measurements, no pipeline file")

    pipeline = Pipeline()

    def callback(pipeline, event):
        if isinstance(event, LoadException):
            raise ValueError("Failed to load %s" % options.pipeline_filename)

    pipeline.add_listener(callback)

    pipeline.load(os.path.expanduser(options.pipeline_filename))

    columns = pipeline.get_measurement_columns()

    print("--- begin measurements ---")

    print("Object,Feature,Type")

    for column in columns:
        object_name, feature, data_type = column[:3]

        print("%s,%s,%s" % (object_name, feature, data_type))

    print("--- end measurements ---")


def print_groups(filename):
    """
    Print the image set groups for this pipeline

    This function outputs a JSON string to the console composed of a list
    of the groups in the pipeline image set. Each element of the list is
    a two-tuple whose first element is a key/value dictionary of the
    group's key and the second is a tuple of the image numbers in the group.
    """
    path = os.path.expanduser(filename)

    m = Measurements(filename=path, mode="r")

    metadata_tags = m.get_grouping_tags_or_metadata()

    groupings = m.get_groupings(metadata_tags)

    # Groupings are np.int64 which cannot be dumped to json
    groupings_export = []
    for g in groupings:
        groupings_export.append((g[0], [int(imgnr) for imgnr in g[1]]))

    json.dump(groupings_export, sys.stdout)


def get_batch_commands(filename, n_per_job=1):
    """Print the commands needed to run the given batch data file headless

    filename - the name of a Batch_data.h5 file. The file should group image sets.

    The output assumes that the executable, "CellProfiler", can be used
    to run the command from the shell. Alternatively, the output could be
    run through a utility such as "sed":

    CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_job.sh/
    """
    path = os.path.expanduser(filename)

    m = Measurements(filename=path, mode="r")

    image_numbers = m.get_image_numbers()

    if m.has_feature(IMAGE, GROUP_NUMBER):
        group_numbers = m[
            IMAGE, GROUP_NUMBER, image_numbers,
        ]

        group_indexes = m[
            IMAGE, GROUP_INDEX, image_numbers,
        ]

        if numpy.any(group_numbers != 1) and numpy.all(
            (group_indexes[1:] == group_indexes[:-1] + 1)
            | ((group_indexes[1:] == 1) & (group_numbers[1:] == group_numbers[:-1] + 1))
        ):
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

                print(
                    "CellProfiler -c -r -p %s -f %d -l %d" % (filename, prev + 1, off)
                )

                prev = off
    else:
        metadata_tags = m.get_grouping_tags_or_metadata()

        if len(metadata_tags) == 1 and metadata_tags[0] == "ImageNumber":
            for i in range(0, len(image_numbers), n_per_job):
                first = image_numbers[i]
                last = image_numbers[min(i + n_per_job - 1, len(image_numbers) - 1)]
                print("CellProfiler -c -r -p %s -f %d -l %d" % (filename, first, last))
        else:
            # LoadData w/ images grouped by metadata tags
            groupings = m.get_groupings(metadata_tags)

            for grouping in groupings:
                group_string = ",".join(
                    ["%s=%s" % (k, v) for k, v in list(grouping[0].items())]
                )

                print("CellProfiler -c -r -p %s -g %s" % (filename, group_string))
    return

def get_batch_commands_new(filename, n_per_job=1):
    """Print the commands needed to run the given batch data file headless

    filename - the name of a Batch_data.h5 file. The file may (but need not) group image sets.

    You can explicitly set the batch size with --images-per-batch, but note that
    it will override existing groupings, so use with caution

    The output assumes that the executable, "CellProfiler", can be used
    to run the command from the shell. Alternatively, the output could be
    run through a utility such as "sed":

    CellProfiler --get-batch-commands Batch_data.h5 | sed s/CellProfiler/farm_job.sh/
    """
    path = os.path.expanduser(filename)

    m = Measurements(filename=path, mode="r")

    image_numbers = m.get_image_numbers()

    grouping_tags = m.get_grouping_tags_only()

    if n_per_job != 1 or grouping_tags == []:
        # One of two things is happening:
        # 1) We've manually set a batch size, and we should always obey it, even if there was grouping
        # 2) There was no grouping so our only choice is to use -f -l
        for i in range(0, len(image_numbers), n_per_job):
            first = image_numbers[i]
            last = image_numbers[min(i + n_per_job - 1, len(image_numbers) - 1)]
            print("CellProfiler -c -r -p %s -f %d -l %d" % (filename, first, last))
    
    else: #We have grouping enabled and haven't overriden it
        groupings = m.get_groupings(grouping_tags)
        for grouping in groupings:
            group_string = ",".join(
                ["%s=%s" % (k, v) for k, v in list(grouping[0].items())]
            )

            print("CellProfiler -c -r -p %s -g %s" % (filename, group_string))

    return


def write_schema(pipeline_filename):
    if pipeline_filename is None:
        raise ValueError(
            "The --write-schema-and-exit switch must be used in conjunction\nwith the -p or --pipeline switch to load a pipeline with an\n"
            "ExportToDatabase module."
        )

    pipeline = Pipeline()

    pipeline.load(pipeline_filename)

    pipeline.turn_off_batch_mode()

    for module in pipeline.modules():
        if module.module_name == "ExportToDatabase":
            break
    else:
        raise ValueError(
            'The pipeline, "%s", does not have an ExportToDatabase module'
            % pipeline_filename
        )

    m = Measurements()

    workspace = Workspace(pipeline, module, m, ObjectSet, m, None)

    module.prepare_run(workspace)


def run_pipeline_headless(options, args):
    """
    Run a CellProfiler pipeline in headless mode
    """
    if options.first_image_set is not None:
        if not options.first_image_set.isdigit():
            raise ValueError("The --first-image-set option takes a numeric argument")
        else:
            image_set_start = int(options.first_image_set)
    else:
        image_set_start = 1

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

    if (options.pipeline_filename is not None) and (
        not options.pipeline_filename.lower().startswith("http")
    ):
        options.pipeline_filename = os.path.expanduser(options.pipeline_filename)

    pipeline = Pipeline()

    initial_measurements = None

    try:
        if h5py.is_hdf5(options.pipeline_filename):
            initial_measurements = load_measurements(
                options.pipeline_filename, image_numbers=image_set_numbers
            )
    except:
        logging.root.info("Failed to load measurements from pipeline")

    if initial_measurements is not None:
        pipeline_text = initial_measurements.get_experiment_measurement(M_PIPELINE)

        pipeline_text = pipeline_text

        pipeline.load(io.StringIO(pipeline_text))

        if not pipeline.in_batch_mode():
            #
            # Need file list in order to call prepare_run
            #

            with h5py.File(options.pipeline_filename, "r") as src:
                if HDF5FileList.has_file_list(src):
                    HDF5FileList.copy(src, initial_measurements.hdf5_dict.hdf5_file)
    else:
        pipeline.load(options.pipeline_filename)

    if options.groups is not None:
        kvs = [x.split("=") for x in options.groups.split(",")]

        groups = dict(kvs)
    else:
        groups = None

    file_list = get_image_set_file()

    if file_list is not None:
        pipeline.read_file_list(file_list)
    elif options.image_directory is not None:
        pathnames = []

        for dirname, _, fnames in os.walk(os.path.abspath(options.image_directory)):
            pathnames.append(
                [
                    os.path.join(dirname, fname)
                    for fname in fnames
                    if os.path.isfile(os.path.join(dirname, fname))
                ]
            )

        pathnames = sum(pathnames, [])

        pipeline.add_pathnames_to_file_list(pathnames)

    #
    # Fixup CreateBatchFiles with any command-line input or output directories
    #
    if pipeline.in_batch_mode():
        create_batch_files = [
            m for m in pipeline.modules() if m.is_create_batch_module()
        ]

        if len(create_batch_files) > 0:
            create_batch_files = create_batch_files[0]

            if options.output_directory is not None:
                create_batch_files.custom_output_directory.value = (
                    options.output_directory
                )

            if options.image_directory is not None:
                create_batch_files.default_image_directory.value = (
                    options.image_directory
                )

    measurements = pipeline.run(
        image_set_start=image_set_start,
        image_set_end=image_set_end,
        grouping=groups,
        measurements_filename=None,
        initial_measurements=initial_measurements,
    )

    if options.done_file is not None:
        if measurements is not None and measurements.has_feature(
            EXPERIMENT, EXIT_STATUS,
        ):
            done_text = measurements.get_experiment_measurement(EXIT_STATUS)

            exit_code = 0 if done_text == "Complete" else -1
        else:
            done_text = "Failure"

            exit_code = -1

        fd = open(options.done_file, "wt")
        fd.write("%s\n" % done_text)
        fd.close()
    elif not measurements.has_feature(EXPERIMENT, EXIT_STATUS):
        # The pipeline probably failed
        exit_code = 1
    else:
        exit_code = 0

    if measurements is not None:
        measurements.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
