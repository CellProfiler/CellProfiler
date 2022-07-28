"""analysis_worker.py - Run pipelines on imagesets to produce measurements.

The analysis worker listens on a ZMQ port for work announcements. It then
requests jobs from the announcer and executes them. As an application,
the analysis worker runs three threads:

* Main thread - spawns the worker and monitor threads and enters a run loop.
                The run loop is needed on OS/X in order to process the UI.
                The UI is needed by ImageJ 1.0 which starts AWT. The main thread
                issues a stop notification to the worker thread after exiting
                the run loop.

* Worker thread - listens for jobs and processes them until it receives a stop
                  notification from the main thread.

* Monitor thread - reads from STDIN. If the parent process closes STDIN,
                   the read call throws an exception and the monitor thread
                   stops the main thread's run loop.
"""

import logging
import os
import sys

import pkg_resources

from cellprofiler_core.constants.worker import (
    DEADMAN_START_ADDR,
    DEADMAN_START_MSG,
    NOTIFY_ADDR,
    NOTIFY_STOP,
    the_zmq_context,
    all_measurements,
)
from cellprofiler_core.preferences import set_always_continue, set_conserve_memory
from cellprofiler_core.worker._worker import Worker

"""Set the log level through the environment by specifying AW_LOG_LEVEL"""
AW_LOG_LEVEL = "AW_LOG_LEVEL"

work_announce_address = None
knime_bridge_address = None


def aw_parse_args():
    """Parse the application arguments into setup parameters"""
    from cellprofiler_core.preferences import (
        set_headless,
        set_awt_headless,
        set_plugin_directory,
    )
    import optparse

    global work_announce_address
    global knime_bridge_address
    set_headless()
    set_awt_headless(True)
    parser = optparse.OptionParser()
    parser.add_option(
        "--work-announce",
        dest="work_announce_address",
        help="ZMQ port where work announcements are published",
        default=None,
    )
    parser.add_option(
        "--log-level",
        dest="log_level",
        help="Logging level for logger: DEBUG, INFO, WARNING, ERROR",
        default=os.environ.get(AW_LOG_LEVEL, logging.INFO),
    )
    parser.add_option(
        "--plugins-directory",
        dest="plugins_directory",
        help="Folder containing the CellProfiler plugin modules needed by client pipelines",
        default=None,
    )
    parser.add_option(
        "--conserve-memory",
        dest="conserve_memory",
        default=None,
        help="CellProfiler will attempt to release unused memory after each image set.",
    )
    parser.add_option(
        "--jvm-heap-size",
        dest="jvm_heap_size",
        default=None,
        help=(
            "This is the amount of memory reserved for the "
            "Java Virtual Machine (similar to the java -Xmx switch)."
            "Example formats: 512000k, 512m, 1g"
        ),
    )
    parser.add_option(
        "--knime-bridge-address",
        dest="knime_bridge_address",
        help="Open up a port to handle the Knime bridge protocol",
        default=None,
    )
    parser.add_option(
        "--always-continue",
        dest="always_continue",
        help="Don't stop the analysis when an image set raises an error",
        default=None
    )

    options, args = parser.parse_args()

    logging.root.setLevel(options.log_level)
    if len(logging.root.handlers) == 0:
        logging.root.addHandler(logging.StreamHandler())

    if not (options.work_announce_address or options.knime_bridge_address):
        parser.print_help()
        sys.exit(1)
    work_announce_address = options.work_announce_address
    knime_bridge_address = options.knime_bridge_address
    #
    # Set up the headless plugins directories before doing
    # anything so loading will get them
    #
    if options.plugins_directory is not None:
        set_plugin_directory(options.plugins_directory, globally=False)
    if options.conserve_memory is not None:
        set_conserve_memory(options.conserve_memory, globally=False)
    if options.always_continue is not None:
        set_always_continue(options.always_continue, globally=False)
    else:
        logging.warning("Plugins directory not set")


if __name__ == "__main__":
    if "CP_DEBUG_WORKER" not in os.environ:
        #
        # Sorry to put ugliness so early:
        #     The process inherits file descriptors from the parent. Windows doesn't
        #     let you selectively inherit file descriptors, so we close them here.
        #
        try:
            maxfd = os.sysconf("SC_OPEN_MAX")
        except:
            maxfd = 256
        os.closerange(3, maxfd)
    if not hasattr(sys, "frozen"):
        # In the development version, maybe the bioformats package is installed?
        # Add the root to the pythonpath
        root = os.path.split(os.path.split(__file__)[0])[0]
        sys.path.insert(0, root)

    aw_parse_args()

import time
import threading
import zmq

#
# CellProfiler expects NaN as a result during calculation
#
import numpy as np

np.seterr(all="ignore")


# to guarantee closing of measurements, we store all of them in a WeakSet, and
# close them on exit.


def main():
    #
    # For OS/X set up the UI elements that users expect from
    # an app.
    #
    if sys.platform == "darwin":
        import os.path

        icon_path = pkg_resources.resource_filename(
            "cellprofiler", os.path.join("data", "icons", "CellProfiler.png")
        )
        os.environ["APP_NAME_%d" % os.getpid()] = "CellProfilerWorker"
        os.environ["APP_ICON_%d" % os.getpid()] = icon_path

    # Start the JVM
    from cellprofiler_core.utilities.java import start_java, stop_java

    start_java()

    deadman_start_socket = the_zmq_context.socket(zmq.PAIR)
    deadman_start_socket.bind(DEADMAN_START_ADDR)

    # Start the deadman switch thread.
    start_daemon_thread(target=exit_on_stdin_close, name="exit_on_stdin_close")
    deadman_start_socket.recv()
    deadman_start_socket.close()

    from cellprofiler.knime_bridge import KnimeBridgeServer

    with Worker(work_announce_address) as worker:
        worker_thread = threading.Thread(target=worker.run, name="WorkerThread")
        worker_thread.setDaemon(True)
        worker_thread.start()
        with KnimeBridgeServer(
            the_zmq_context, knime_bridge_address, NOTIFY_ADDR, NOTIFY_STOP
        ):
            worker_thread.join()

    #
    # Shutdown - need to handle some global cleanup here
    #
    try:
        stop_java()
    except:
        logging.warning("Failed to stop the JVM", exc_info=True)


__the_notify_pub_socket = None


def get_the_notify_pub_socket():
    """Get the socket used to publish the worker stop message"""
    global __the_notify_pub_socket
    if __the_notify_pub_socket is None or __the_notify_pub_socket.closed:
        __the_notify_pub_socket = the_zmq_context.socket(zmq.PUB)
        __the_notify_pub_socket.bind(NOTIFY_ADDR)
    return __the_notify_pub_socket


def exit_on_stdin_close():
    """Read until EOF, then exit, possibly without cleanup."""
    notify_pub_socket = get_the_notify_pub_socket()
    deadman_socket = the_zmq_context.socket(zmq.PAIR)
    deadman_socket.connect(DEADMAN_START_ADDR)
    deadman_socket.send(DEADMAN_START_MSG)
    deadman_socket.close()

    # If sys.stdin closes, either our parent has closed it (indicating we
    # should exit), or our parent has died.  Attempt to exit cleanly via main
    # thread, but if that takes too long (hung filesystem or socket, perhaps),
    # use a hard os._exit() instead.
    stdin = sys.stdin
    try:
        while stdin.read():
            pass
    except:
        pass
    finally:
        print("Cancelling worker")
        notify_pub_socket.send(NOTIFY_STOP)
        notify_pub_socket.close()
        # hard exit after 10 seconds unless app exits
        time.sleep(10)
        for m in all_measurements:
            try:
                m.close()
            except:
                pass
        os._exit(0)


def start_daemon_thread(target=None, args=(), name=None):
    thread = threading.Thread(target=target, args=args, name=name)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    main()
